import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

import exchange_calendars as xc
import numpy as np
import pandas as pd

from intraday.checks import external_fills, independent_votes
from intraday.data import NY, load
from intraday.experiment import FEATURES, score, session_rows, walk_forward
from intraday.paper import advance, open_ledger
from intraday.review import opinion, packet


def fixture():
    cal = xc.get_calendar("XNYS")
    sessions = cal.sessions_in_range("2024-02-01", "2024-07-08")
    segments = []
    for n, date in enumerate(sessions):
        op = cal.session_open(date).tz_convert(NY)
        end = cal.session_close(date).tz_convert(NY)
        index = pd.date_range(op, periods=30, freq="min").append(
            pd.DatetimeIndex(
                [
                    op + pd.Timedelta(minutes=34),
                    op + pd.Timedelta(minutes=35),
                    end - pd.Timedelta(minutes=11),
                    end - pd.Timedelta(minutes=10),
                ]
            )
        )
        price = 100 + np.arange(len(index)) * (0.02 if n % 3 else -0.03)
        price[-1] += 1 if n % 2 else -1
        segments.append(
            pd.DataFrame(
                dict(
                    open=price,
                    high=price + 0.01,
                    low=price - 0.01,
                    close=price + 0.005,
                    volume=np.full(len(price), 500 + n),
                ),
                index=index,
            )
        )
    return pd.concat(segments)


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = fixture()
        cls.rows = session_rows(cls.bars)
        cls.test, cls.folds = walk_forward(cls.rows)

    def test_independent_votes_and_prefix_invariance(self):
        self.assertEqual(
            independent_votes(self.bars, self.rows), self.rows.eligible.sum()
        )
        cutoff = pd.Timestamp("2024-07-02 10:00", tz=NY)
        mutated = self.bars.astype(float).copy()
        mutated.loc[
            mutated.index >= cutoff, ["open", "high", "low", "close", "volume"]
        ] *= 1.3
        newer = session_rows(mutated)
        pd.testing.assert_frame_equal(
            self.rows.loc[:"2024-07-02", FEATURES + ["breakout", "band"]],
            newer.loc[:"2024-07-02", FEATURES + ["breakout", "band"]],
        )

    def test_monthly_fit_precedes_test_and_rejects_future_labels(self):
        for row in self.folds.itertuples():
            self.assertLess(row.last, row.month + "-01")
        changed = self.rows.copy()
        changed.loc["2024-07-01":, "exit"] *= 3
        rerun, _ = walk_forward(changed)
        np.testing.assert_allclose(rerun.probability, self.test.probability)
        self.assertFalse((self.test.agreement_ml & ~self.test.agreement).any())

    def test_clock_cost_and_early_close(self):
        row = self.test.loc["2024-07-03"]
        self.assertEqual(str(row.planned_exit.time()), "12:50:00")
        daily, fills, _ = score(self.test, 5)
        expected = row.exit - row.entry - (row.exit + row.entry) * 0.0005
        self.assertAlmostEqual(daily.at["2024-07-03", "clock"], expected)
        self.assertTrue((fills.exit_at.dt.date == fills.entry_at.dt.date).all())
        self.assertTrue((daily.cash == 0).all())

    def test_gap_behaviour(self):
        entry = pd.Timestamp("2024-07-03 10:05", tz=NY)
        early = pd.Timestamp("2024-07-02 09:34", tz=NY)
        rows = session_rows(self.bars.drop([entry, early]))
        test, _ = walk_forward(rows)
        _, fills, _ = score(test, 5)
        self.assertFalse(rows.at["2024-07-02", "eligible"])
        self.assertFalse(
            ((fills.index == "2024-07-03") & (fills.version == "clock")).any()
        )
        exit_at = self.test.at["2024-07-03", "planned_exit"]
        with self.assertRaisesRegex(ValueError, "same-session exit"):
            session_rows(self.bars.drop(exit_at))

    def test_independent_broker_detects_wrong_ledger(self):
        _, fills, _ = score(self.test, 5)
        count = external_fills(self.bars, self.test, fills)
        self.assertEqual(count["clock"], len(self.test))
        damaged = fills.copy()
        damaged.loc[damaged.version == "clock", "entry"] += 1
        with self.assertRaises(AssertionError):
            external_fills(self.bars, self.test, damaged)

    def test_streaming_replay_and_late_start(self):
        date = "2024-07-03"
        with tempfile.TemporaryDirectory() as temp:
            db = open_ledger(Path(temp) / "paper.db")
            try:

                def tick(at, mode="replay"):
                    return advance(
                        db, self.bars, mode, date, pd.Timestamp(f"{date} {at}", tz=NY)
                    )

                self.assertEqual(tick("09:55"), {})
                morning = tick("10:00")
                self.assertEqual(set(morning), {"decision"})
                self.assertAlmostEqual(
                    morning["decision"]["probability"],
                    self.test.at[date, "probability"],
                )
                self.assertNotIn("exit", tick("10:06"))
                complete = tick("13:00")
                self.assertEqual(complete, tick("13:15"))
                self.assertAlmostEqual(
                    complete["exit"]["net_usd_per_participating_variant"],
                    score(self.test, 5)[0].at[date, "clock"],
                )
                self.assertEqual(
                    tick("13:00", "forward")["decision"]["status"],
                    "decision window missed",
                )
                self.assertEqual(
                    db.execute("SELECT count(*) FROM events").fetchone()[0], 4
                )
            finally:
                db.close()

    def test_review_packet_and_role_contract(self):
        evidence = packet(self.test.iloc[0])
        self.assertEqual(set(evidence["features"]), set(FEATURES))
        self.assertTrue(
            set(evidence).isdisjoint({"entry", "exit", "date", "symbol", "net_usd"})
        )
        result = {
            "model": "example",
            "message": {
                "content": json.dumps(
                    {"vote": "abstain", "reason": "Relative volume is 1.01."}
                )
            },
        }
        with patch(
            "intraday.review.urlopen", return_value=BytesIO(json.dumps(result).encode())
        ) as called:
            self.assertEqual(opinion("example", "sceptic", evidence)["vote"], "abstain")
            self.assertEqual(
                called.call_args.args[0].full_url, "http://127.0.0.1:11434/api/chat"
            )

    def test_duplicate_or_invalid_bar_is_rejected(self):
        first = self.bars.iloc[:3].copy()
        first.index.name = "caldt"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            pd.concat([first, first.iloc[-1:]]).to_csv(path)
            with self.assertRaisesRegex(ValueError, "unique"):
                load(path)
            first.iloc[0, first.columns.get_loc("low")] = 110
            first.to_csv(path)
            with self.assertRaisesRegex(ValueError, "OHLC"):
                load(path)
