"""A separate one-share intraday experiment using a 60-session price comparison."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc
from .data import load
from .experiment import P, session_rows


def signals(bars):
    marks = bars.at_time("09:59").close.copy()
    marks.index = marks.index.strftime("%Y-%m-%d")
    sessions = (
        xc.get_calendar("XNYS")
        .sessions_in_range(str(bars.index[0].date()), str(bars.index[-1].date()))
        .strftime("%Y-%m-%d")
    )
    price = marks.reindex(sessions)
    return pd.DataFrame(
        {
            "endpoint": 100 * (price / price.shift(60) - 1),
            "mean": 100
            * (price / price.shift(1).rolling(60, min_periods=60).mean() - 1),
        }
    ).rename_axis("date")


def side(change):
    if pd.isna(change):
        return None
    return int(np.sign(change))


def paper_net(position, buy, sell, bps):
    return position * (sell - buy) - abs(position) * (buy + sell) * bps / 10000


def confirmed(change, previous):
    direction = side(change)
    return direction if direction is not None and direction == side(previous) else None


def carry_step(previous, target, entry, mark, previous_mark, liquidate=False):
    """Mark previous overnight exposure, then charge actual share turnover."""
    gap = previous * (entry - previous_mark) if previous else 0.0
    gross = gap + target * (mark - entry)
    costs = (
        abs(target - previous) * entry + (abs(target) * mark if liquidate else 0)
    ) / 10000
    return gross, costs


def report(bars):
    rows = session_rows(bars, require_scored_exit=False)
    frame = signals(bars).join(rows[["entry", "exit", "breakout", "band", "eligible"]])
    frame["prior_mean"] = frame["mean"].shift(1)
    frame = frame.loc[P["historical_test_start"] : P["source_end"]].copy()
    frame["move_usd"] = frame.exit - frame.entry
    frame["one_bp_cost"] = (frame.entry + frame.exit) / 10000
    # A missing valuation must never silently shorten the holding period.
    if frame[["entry", "exit"]].isna().any().any():
        raise ValueError(
            "Continuous historical P&L requires an entry and closing mark on every test session."
        )
    for variant in ("carry", "carry_cash", "hold"):
        previous, previous_mark = 0, 0.0
        ledger = []
        for date, row in frame.iterrows():
            target = confirmed(row["mean"], row.prior_mean)
            target = previous if target is None else target
            if variant == "hold":
                target = 1
            elif variant == "carry_cash":
                target = max(0, target)
            gross, cost = carry_step(
                previous,
                target,
                row.entry,
                row.exit,
                previous_mark,
                date == frame.index[-1],
            )
            ledger.append((target, gross, cost))
            previous, previous_mark = target, row.exit
        frame[[variant + "_pos", variant + "_gross", variant + "_fee"]] = ledger
    return frame.drop(columns=["entry", "exit", "prior_mean"])


def qlib_signal(table, reference="mean"):
    score = table[reference].dropna().rename("score").to_frame()
    score.index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(score.index + " 10:00"), ["SPY"] * len(score)],
        names=["datetime", "instrument"],
    )
    return score


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument(
        "--qlib", type=Path, help="Optional pandas pickle for your own Qlib environment"
    )
    cli.add_argument("--reference", choices=["endpoint", "mean"], default="mean")
    args = cli.parse_args()
    table = report(load(args.data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, float_format="%.12g")
    if args.qlib:
        args.qlib.parent.mkdir(parents=True, exist_ok=True)
        qlib_signal(table, args.reference).to_pickle(args.qlib)
