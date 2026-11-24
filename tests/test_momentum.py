import unittest
import pandas as pd
from intraday.momentum import signals, confirmed, carry_step, qlib_signal
from test_lab import fixture


class MomentumTests(unittest.TestCase):
    def test_exact_exchange_window_and_missing_marks(self):
        bars = fixture()
        marks = bars.at_time("09:59").close
        data = signals(bars)
        self.assertAlmostEqual(
            data.iloc[60]["mean"], 100 * (marks.iloc[60] / marks.iloc[:60].mean() - 1)
        )
        self.assertAlmostEqual(
            data.iloc[60].endpoint, 100 * (marks.iloc[60] / marks.iloc[0] - 1)
        )
        self.assertTrue(pd.isna(signals(bars.drop(marks.index[20])).iloc[60]["mean"]))
        self.assertEqual(qlib_signal(data).index.names, ["datetime", "instrument"])

    def test_reversal_overnight_and_costs(self):
        self.assertIsNone(confirmed(3, -2))
        self.assertEqual(confirmed(-3, -2), -1)
        self.assertEqual(carry_step(-1, -1, 95, 90, 100), (10, 0))
        gross, cost = carry_step(-1, 1, 95, 101, 100, True)
        self.assertEqual(gross, 11)
        self.assertAlmostEqual(cost, (2 * 95 + 101) / 10000)
