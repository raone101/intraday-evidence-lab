import unittest
import pandas as pd
from intraday.arena import morning_ballot, votes, combinations_book
from test_lab import fixture


class ArenaTests(unittest.TestCase):
    def test_conflict_keeps_both_books(self):
        sample = pd.DataFrame(
            [
                dict(
                    inside=False,
                    above=True,
                    below=False,
                    swept_low=True,
                    swept_high=True,
                    opening_relative_volume=1.08,
                    vwap_z=2.55,
                    forecast_bps=-2.56,
                )
            ]
        )
        result = combinations_book(votes(sample)).iloc[0]
        self.assertEqual(result["Breakout"], 1)
        self.assertEqual(result["VWAP fade"], -1)
        self.assertEqual(result["Two net votes"], 0)
        self.assertEqual(result["Breakout + VWAP fade"], 0)

    def test_morning_packet_excludes_outcome_and_matches_replay(self):
        bars = fixture()
        stamp = pd.Timestamp("2024-07-03 10:00", tz="America/New_York")
        full = morning_ballot(bars, "2024-07-03")
        early = morning_ballot(bars.loc[bars.index < stamp], "2024-07-03")
        self.assertEqual(full, early)
        self.assertNotIn("exit", early["evidence"])
        self.assertNotIn("entry", early["evidence"])
