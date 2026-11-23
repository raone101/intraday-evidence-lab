"""Build auditable ballots, then model fills separately from the morning information."""

from collections import deque
from datetime import time
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def fit_month(rows, month):
    train = rows.loc[(rows.index < month + "-01") & rows.eligible].dropna(
        subset=["entry", "exit"]
    )
    rate = P["primary_cost_bps_each_side"] / 10000
    y = (train.exit * (1 - rate) > train.entry * (1 + rate)).astype(int)
    fold = dict(month=month, rows=len(train), status="no model")
    if len(train) < P["minimum_training_sessions"] or y.nunique() < 2:
        return None, fold
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=P["logit_C"], max_iter=1000)
    )
    model.fit(train[FEATURES], y)
    fold.update(
        first=train.index[0],
        last=train.index[-1],
        positive_rate=float(y.mean()),
        status="fitted",
        **dict(zip(FEATURES, model[-1].coef_[0])),
    )
    return model, fold


def walk_forward(rows):
    test = rows.loc[P["historical_test_start"] :].copy()
    if test.empty:
        raise ValueError("No sessions in the registered historical test period.")
    test["probability"] = np.nan
    test["training_through"] = ""
    folds = []
    for month, block in test.groupby(test.index.str[:7], sort=True):
        model, fold = fit_month(rows, month)
        folds.append(fold)
        if model is None:
            continue
        valid = block.index[block.eligible]
        if len(valid):
            test.loc[valid, "probability"] = model.predict_proba(
                block.loc[valid, FEATURES]
            )[:, 1]
        test.loc[block.index, "training_through"] = fold["last"]
    test["cash"] = False
    test["clock"] = test.eligible
    test["either"] = test.breakout | test.band
    test["agreement"] = test.breakout & test.band
    test["agreement_ml"] = test.agreement & test.probability.ge(P["veto_probability"])
    return test, pd.DataFrame(folds)


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


def contrasts(daily):
    rng = np.random.default_rng(719)
    n, length = len(daily), min(10, len(daily))
    starts = rng.integers(0, n - length + 1, (5000, (n + length - 1) // length))
    indices = (starts[:, :, None] + np.arange(length)).reshape(5000, -1)[:, :n]
    output = []
    for left, right in [
        ("agreement_ml", "agreement"),
        ("agreement", "breakout"),
        ("agreement", "band"),
    ]:
        difference = (daily[left] - daily[right]).to_numpy()
        lo, hi = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
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
