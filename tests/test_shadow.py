from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from intraday.activity import PROFILE, sample_sessions
from intraday.data import NY
from intraday.shadow import FORECASTS, evaluate, forecasts, write_plan
from test_lab import fixture


class ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = fixture()
        cls.data = sample_sessions(cls.bars, "15m")

    def test_profile_and_probabilities_ignore_current_month_outcomes(self):
        end = pd.Timestamp("2024-07-03 10:00", tz=NY)
        mutated = self.bars.astype(float).copy()
        mutated.loc[
            mutated.index >= end, ["open", "high", "low", "close", "volume"]
        ] *= 1.25
        day = "2024-07-03"
        changed = sample_sessions(mutated, "15m")
        np.testing.assert_allclose(
            self.data.loc[day, list(PROFILE)].astype(float),
            changed.loc[day, list(PROFILE)].astype(float),
        )
        prediction, folds = forecasts(self.data, day, day)
        _, second_folds = forecasts(changed, day, day)
        flipped = self.data.copy()
        flipped.loc["2024-07-01":, "won"] = 1 - flipped.loc["2024-07-01":, "won"]
        second, _ = forecasts(flipped, day, day)
        np.testing.assert_allclose(prediction[list(FORECASTS)], second[list(FORECASTS)])
        self.assertLess(folds.training_through.iat[0], "2024-07-01")
        pd.testing.assert_frame_equal(folds, second_folds)

    def test_transfer_only_learns_from_source_before_recent_target_window(self):
        source = self.data.copy()
        source.index = (
            pd.DatetimeIndex(source.index) - pd.DateOffset(years=1)
        ).strftime("%Y-%m-%d")
        preds, folds = forecasts(self.data, "2024-07-03", "2024-07-03", source)
        self.assertTrue(preds.transfer_neural.notna().all())
        self.assertLess(folds.source_through.iat[0], self.data.index[0])
        self.assertEqual(set(evaluate(preds).result), {"insufficient"})
        self.assertEqual(
            set(evaluate(preds, observation_complete=False).result), {"collecting"}
        )

    def test_future_trial_is_exclusive_and_records_its_observation_boundary(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trial.json"
            today = pd.Timestamp.now(tz=NY).date()
            future = str(today + pd.Timedelta(days=140))
            plan = write_plan(
                path, future, "15m", "The price profile reduces Brier error"
            )
            self.assertEqual(plan["after"], str(today))
            with self.assertRaises(FileExistsError):
                write_plan(path, future, "15m", "An accidental overwrite")
            with self.assertRaises(ValueError):
                write_plan(path, str(today), "15m", "A backdated study")
