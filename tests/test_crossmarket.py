import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd

from crossmarket import forward
from crossmarket.run import choose, returns, run_asset


class MarketTests(unittest.TestCase):
    def test_bid_ask_and_signed_tie(self):
        x = pd.Series(dict(open=100, bid=99, ask=101))
        y = pd.Series(dict(open=102, bid=101, ask=103))
        self.assertEqual(returns(x, y, 0), {1: 0, -1: -400, 0: 0})
        self.assertIsNone(returns(x, pd.Series(dict(open=100, bid=101, ask=99)), 1))
        self.assertEqual(choose({"long": 1, "bear": -1}, {"long": 1, "bear": 1}), 0)

    def test_freeze_before_execution_and_score_dissent(self):
        date = "2024-07-02"
        index = pd.date_range(
            date + " 09:30", date + " 11:00", freq="min", tz="America/New_York"
        )
        bars = pd.DataFrame(index=index)
        bars["open"] = [100 + i * 0.1 for i in range(len(index))]
        bars["close"] = bars.open + 0.07
        bars["low"] = bars.open - 0.02
        bars["high"] = bars.close + 0.02
        bars["volume"] = 100.0
        with TemporaryDirectory() as tmp:
            packet = forward.morning(bars, date, Path(tmp))
            changed = bars.copy()
            changed.loc[index[-1], "open"] *= 0.99
            self.assertEqual(packet, forward.morning(changed, date, Path(tmp)))
        packet["observed_at"] = date + "T10:00:10-04:00"
        result = forward.close(bars, date, packet)
        votes, days = run_asset("ETF", "SPY", bars.tz_localize(None), date)
        self.assertEqual(len(votes), 23)
        self.assertAlmostEqual(result["weighted_bps"], days[0]["weighted_bps"])
        for row in votes:
            rival = result["bots"][row["strategy"]]
            self.assertAlmostEqual(rival["net_bps"], row["net_bps"])
            self.assertAlmostEqual(rival["advantage_bps"], row["advantage_bps"])
            self.assertAlmostEqual(rival["next_weight"], row["next_weight"])
        self.assertIsNone(packet["bots"]["Trend with historical context"]["forecast"])
        packet["observed_at"] = date + "T10:01:00-04:00"
        self.assertEqual(forward.close(bars, date, packet)["status"], "untimely ballot")
