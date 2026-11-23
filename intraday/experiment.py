"""Build auditable ballots, then model fills separately from the morning information."""

from collections import deque
from datetime import time
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

from .data import NY

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "protocol.json").read_text())
FEATURES = P["features"]


def sessions_for(bars):
    start, end = bars.index[0].date(), bars.index[-1].date()
    return xc.get_calendar("XNYS").schedule.loc[
        str(start) : str(end), ["open", "close"]
    ]


def session_rows(bars, *, require_scored_exit=True):
    """Flat days remain in the denominator, including days with missing early prints."""
    groups = {k: v for k, v in bars.groupby(bars.index.date, sort=False)}
    history = deque(maxlen=P["lookback_sessions"])
    rows = []
    for d, session in sessions_for(bars).iterrows():
        op = session.open.tz_convert(NY)
        cut = op + pd.Timedelta(minutes=30)
        if cut.time() != time.fromisoformat(P["decision_et"]):
            raise ValueError("Unexpected exchange open for the registered clock.")
        entry_clock = time.fromisoformat(P["entry_et"])
        start = op.replace(hour=entry_clock.hour, minute=entry_clock.minute)
        end = session.close.tz_convert(NY) - pd.Timedelta(
            minutes=P["exit_minutes_before_close"]
        )
        day = groups.get(d.date(), bars.iloc[:0])
        day = day.loc[(day.index >= op) & (day.index < session.close)]
        early = day.loc[(day.index >= op) & (day.index < cut)]
        row = dict(
            date=d.strftime("%Y-%m-%d"),
            observed_at=cut,
            entry_at=start,
            planned_exit=end,
            breakout=False,
            band=False,
            quality="incomplete opening",
            entry=np.nan,
            exit=np.nan,
            exit_at=pd.NaT,
            eligible=False,
        )
        row.update(dict.fromkeys(FEATURES, np.nan))
        if (
            len(early) == 30
            and early.volume.sum() > 0
            and early.volume.iloc[:5].sum() > 0
        ):
            five = early.iloc[:5]
            opening, last = early.open.iloc[0], early.close.iloc[-1]
            ret = last / opening - 1
            volume = five.volume.sum()
            row["quality"] = "warm-up"
            if len(history) == history.maxlen:
                previous_volume, movement = np.mean(history, axis=0)
                rvol = volume / previous_volume
                typical = early[["high", "low", "close"]].mean(axis=1)
                vwap = np.average(typical, weights=early.volume)
                row.update(
                    opening5_return=five.close.iloc[-1] / opening - 1,
                    first30_return=ret,
                    opening_relative_volume=rvol,
                    above_range=last / five.high.max() - 1,
                    above_band=last / (opening * (1 + movement)) - 1,
                    above_vwap=last / vwap - 1,
                    quality="ready",
                    eligible=True,
                )
                row["breakout"] = bool(
                    five.close.iloc[-1] > opening
                    and rvol >= P["rvol_min"]
                    and row["above_range"] > 0
                )
                row["band"] = bool(row["above_band"] > 0 and row["above_vwap"] > 0)
            history.append((volume, abs(ret)))
        # Execution/outcome columns are calculated only after the morning ballot.
        if start in day.index:
            row["entry"] = day.at[start, "open"]
            exits = day.loc[day.index >= end]
            if not exits.empty:
                row["exit"], row["exit_at"] = exits.open.iloc[0], exits.index[0]
            elif (
                require_scored_exit
                and row["eligible"]
                and row["date"] >= P["historical_test_start"]
            ):
                raise ValueError(
                    f"Entry exists without a same-session exit on {d.date()}"
                )
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def rule_variants(eligible, breakout, band):
    """The complete order policy. Inputs are bools or aligned boolean Series."""
    return dict(
        cash=eligible & False,
        clock=eligible,
        breakout=eligible & breakout,
        band=eligible & band,
        either=eligible & (breakout | band),
        agreement=eligible & breakout & band,
    )


def historical_decisions(rows):
    test = rows.loc[P["historical_test_start"] :].copy()
    if test.empty:
        raise ValueError("No sessions in the registered historical test period.")
    return test.assign(**rule_variants(test.eligible, test.breakout, test.band))


def score(test, bps):
    variants = P["variants"]
    rate = bps / 10000
    outcomes = test.exit - test.entry - rate * (test.entry + test.exit)
    daily = pd.DataFrame(index=test.index)
    trades = []
    summaries = []
    for version in variants:
        order = test[version] & test.entry.notna()
        fills = test.loc[order, ["entry_at", "entry", "exit_at", "exit"]].copy()
        fills["version"] = version
        fills["cost_usd"] = rate * (fills.entry + fills.exit) * P["shares"]
        fills["net_usd"] = (fills.exit - fills.entry) * P["shares"] - fills.cost_usd
        fills["net_bps"] = 10000 * fills.net_usd / (fills.entry * P["shares"])
        trades.append(fills)
        daily[version] = (outcomes * P["shares"]).where(order, 0)
        equity = P["initial_usd"] + daily[version].cumsum()
        peak = equity.cummax().clip(lower=P["initial_usd"])
        summaries.append(
            dict(
                version=version,
                bps_each_side=bps,
                days=len(test),
                trades=len(fills),
                net_usd=float(daily[version].sum()),
                cost_usd=float(fills.cost_usd.sum()),
                mean_bps_traded=float(fills.net_bps.mean()) if len(fills) else None,
                return_pct=100 * daily[version].sum() / P["initial_usd"],
                closing_dd_pct=100 * (equity / peak - 1).min(),
            )
        )
    return daily, pd.concat(trades), pd.DataFrame(summaries)


def paired_interval(differences):
    rng = np.random.default_rng(719)
    n, length = len(differences), min(10, len(differences))
    starts = rng.integers(0, n - length + 1, (5000, (n + length - 1) // length))
    indices = (starts[:, :, None] + np.arange(length)).reshape(5000, -1)[:, :n]
    return np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])


def contrasts(daily):
    output = []
    for left, right in [
        ("agreement", "breakout"),
        ("agreement", "band"),
        ("either", "agreement"),
    ]:
        difference = (daily[left] - daily[right]).to_numpy()
        lo, hi = paired_interval(difference)
        output.append(
            dict(
                left=left,
                right=right,
                delta_usd_per_session=difference.mean(),
                lower95=lo,
                upper95=hi,
                differing_sessions=int((difference != 0).sum()),
            )
        )
    return pd.DataFrame(output)
