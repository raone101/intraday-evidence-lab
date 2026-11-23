"""Opening time-at-price features and resolved hypothetical labels for offline research."""

import numpy as np
import pandas as pd

from .experiment import FEATURES, P, session_rows

PROFILE = (
    "profile_close_to_mode_bps",
    "profile_mode_share",
    "profile_half_shift_bps",
    "opening_range_bps",
)
RANGE_INPUTS = ["opening_range_bps", "opening_relative_volume", "profile_mode_share"]


def time_at_price(opening):
    """Approximate a time-price opportunity profile from 30 one-minute high-low spans."""
    step = P["shadow"]["profile_bps"]
    open_price = opening.open.iat[0]
    high = 10_000 * (opening.high.to_numpy() / open_price - 1)
    low = 10_000 * (opening.low.to_numpy() / open_price - 1)
    lo = np.floor(low / step).astype(int)
    hi = np.floor(high / step).astype(int)
    if hi.max() - lo.min() > 1_200:
        return dict.fromkeys(PROFILE, np.nan)
    levels = np.arange(lo.min(), hi.max() + 1)
    touches = (lo[:, None] <= levels) & (levels <= hi[:, None])
    weights = touches.sum(axis=0)
    mid = (levels + 0.5) * step
    centre = np.average(mid, weights=weights)
    modes = np.flatnonzero(weights == weights.max())
    mode = mid[modes[np.argmin(np.abs(mid[modes] - centre))]]
    last = 10_000 * (opening.close.iat[-1] / open_price - 1)
    first_half = np.average(mid, weights=touches[:15].sum(axis=0))
    last_half = np.average(mid, weights=touches[15:].sum(axis=0))
    return dict(
        zip(
            PROFILE,
            (
                last - mode,
                weights.max() / weights.sum(),
                last_half - first_half,
                high.max() - low.min(),
            ),
        )
    )


def sample_sessions(bars, horizon, rows=None):
    """Decisions use completed opening bars. Trade outcomes are separate, nullable columns."""
    if rows is None:
        rows = session_rows(bars, require_scored_exit=False)
    sample = rows.loc[
        rows.eligible, FEATURES + ["breakout", "band", "entry_at", "entry", "exit"]
    ].copy()
    opening = bars.between_time("09:30", "09:59")
    profiles = pd.DataFrame.from_dict(
        {
            d: time_at_price(g)
            for d, g in opening.groupby(opening.index.strftime("%Y-%m-%d"))
            if d in sample.index
        },
        orient="index",
        columns=PROFILE,
    )
    sample = sample.join(profiles).dropna(subset=list(PROFILE))
    if horizon == "15m":
        close_at = pd.DatetimeIndex(sample.entry_at + pd.Timedelta(minutes=15))
        sample["exit"] = bars.open.reindex(close_at).to_numpy()
    sample["agreement"] = sample.breakout & sample.band
    fee = P["primary_cost_bps_each_side"] / 10_000
    sample["net_bps"] = 10_000 * (sample.exit * (1 - fee) / sample.entry - (1 + fee))
    sample["won"] = sample.net_bps.gt(0).where(sample.net_bps.notna()).astype("Int64")
    return sample
