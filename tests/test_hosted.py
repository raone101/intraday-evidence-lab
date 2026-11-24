import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from intraday import hosted
from intraday.experiment import P, score, session_rows, historical_decisions
from test_lab import fixture


class HostedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = fixture()
        cls.date = "2024-07-03"
        _, cls.cut, cls.entry, cls.close = hosted.due_times(cls.date)
        cls.plan = dict(
            after="2024-07-02", horizon="15m", recipe_sha256=hosted.recipe_digest()
        )

    def data(self, start, end):
        return self.bars.loc[start:end]

    def test_morning_and_early_close_match_standalone_ledger(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(hosted, "bars_between", side_effect=self.data),
        ):
            root = Path(temp)
            day = root / self.date
            with patch.object(hosted, "now", return_value=self.cut):
                hosted.collect_open(self.date, self.plan, root)
            decision, forecast = [
                json.loads((day / name).read_text())
                for name in ["decision.json", "forecast.json"]
            ]
            self.assertEqual(decision["status"], "ready")
            self.assertEqual(
                set(forecast["probabilities"]), {"price_neural", "profile_neural"}
            )
            self.assertLess(forecast["training_through"], "2024-07-01")
            with patch.object(hosted, "now", return_value=self.close):
                hosted.collect_close(self.date, root)
                original = (day / "close.json").read_bytes()
                hosted.collect_close(self.date, root)
                self.assertEqual((day / "close.json").read_bytes(), original)
            result = json.loads(original)
            daily = score(historical_decisions(session_rows(self.bars)), 5)[0]
            for name in P["variants"]:
                self.assertAlmostEqual(
                    result["net_usd_by_variant"][name], daily.at[self.date, name]
                )
            self.assertIn("12:50", result["exit_time"])
            self.assertEqual(result["review"]["mismatches"], {})
            self.assertEqual(
                set(result["forecast_outcome"]["brier"]), set(forecast["probabilities"])
            )

    def test_late_start_has_no_backdated_trade_or_forecast(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(hosted, "now", return_value=self.entry),
            patch.object(hosted, "bars_between") as fetch,
        ):
            root = Path(temp)
            hosted.collect_open(self.date, self.plan, root)
            day = root / self.date
            vote = json.loads((day / "decision.json").read_text())
            self.assertEqual(vote["orders"], [])
            self.assertFalse((day / "forecast.json").exists())
            self.assertEqual(
                hosted.settlement(self.bars, self.date, vote, None)["status"],
                "no timely decision",
            )
            fetch.assert_not_called()

    def test_forecast_failure_preserves_decision_and_price_gap_is_recoverable(self):
        with (
            TemporaryDirectory() as temp,
            patch.object(hosted, "bars_between", side_effect=self.data),
        ):
            root = Path(temp)
            with (
                patch.object(hosted, "now", return_value=self.cut),
                patch.object(
                    hosted, "forecasts", side_effect=ValueError("model offline")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "model offline"):
                    hosted.collect_open(self.date, self.plan, root)
            self.assertTrue((root / self.date / "decision.json").exists())
            with patch.object(hosted, "now", return_value=self.close):
                with patch.object(hosted, "bars_between", return_value=pd.DataFrame()):
                    with self.assertRaisesRegex(ValueError, "Settlement bars missing"):
                        hosted.collect_close(self.date, root)
                self.assertTrue((root / self.date / "close-gap.json").exists())
                hosted.collect_close(self.date, root)
                final = json.loads((root / self.date / "close.json").read_text())
                self.assertEqual(final["status"], "settled")
                self.assertIsNone(final["forecast_outcome"])
