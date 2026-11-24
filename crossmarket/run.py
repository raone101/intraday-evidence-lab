"""Replay the cross-market paper arena: python -m crossmarket.run --cache data/markets"""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
from intraday.voting import weight
from .feeds import streams
from .strategies import ballot, sign

FEES = {"Equity": 5, "ETF": 5, "Futures": 1, "Forex": 0.2, "Crypto": 10, "Options": 5}


def choose(calls, weights):
    active = [b for b, p in calls.items() if p]
    total = sum(calls[b] * weights[b] for b in active)
    return sign(total) if len(active) >= 2 and abs(total) > 1e-10 else 0


def returns(entry, exit_bar, fee):
    bid0, ask0 = entry.get("bid", entry.open), entry.get("ask", entry.open)
    bid1, ask1 = exit_bar.get("bid", exit_bar.open), exit_bar.get("ask", exit_bar.open)
    if not (0 < bid0 <= ask0 and 0 < bid1 <= ask1):
        return None

    def result(buy, sell):
        return (10000 * (sell - buy) - fee * (sell + buy)) / entry.open

    return {1: result(ask0, bid1), -1: result(ask1, bid0), 0: 0.0}


def run_asset(asset, symbol, frame, score_from=None, initial=None):
    rows, days = [], []
    prices, moves, training = deque(maxlen=60), deque(maxlen=14), deque(maxlen=120)
    history = defaultdict(lambda: [0, 0])
    if initial:
        history.update({bot: list(counts) for bot, counts in initial.items()})
    f = frame.between_time("09:30", "15:59").sort_index()
    if f.index.has_duplicates:
        raise ValueError(f"Duplicate minute in {symbol}")
    window = {day: g for day, g in f.groupby(f.index.normalize())}
    if asset in ("Equity", "ETF", "Options"):
        sessions = (
            xc.get_calendar("XNYS")
            .sessions_in_range(f.index.min().date(), f.index.max().date())
            .tz_localize(None)
        )
    elif asset == "Crypto":
        sessions = pd.date_range(f.index.min().date(), f.index.max().date())
    else:
        sessions = pd.bdate_range(f.index.min().date(), f.index.max().date())
    recent_gap_fades = deque(maxlen=5)
    prev = None
    for day in sessions:
        g = window.get(day)
        morning = g.between_time("09:30", "09:59") if g is not None else None
        equity_clock = asset in ("ETF", "Equity")
        prev_close = prev
        prev = None

        def at(clock):
            key = day + pd.Timedelta(clock)
            return g.loc[key] if g is not None and key in g.index else None

        prior_fades = tuple(recent_gap_fades)
        close = at("15:59:00")
        if equity_clock and close is not None:
            prev = close.close
        first = at("09:30:00")
        if equity_clock:
            if prev_close is None or first is None or close is None:
                recent_gap_fades.append(np.nan)
            else:
                overnight = first.open - prev_close
                daytime = close.close - first.open
                recent_gap_fades.append(float(overnight * daytime < 0))
        if morning is None or len(morning) != 30:
            prices.append(np.nan)
            moves.append(np.nan)
            continue
        ohlc = morning[["open", "high", "low", "close"]]
        if (
            not np.isfinite(ohlc).all().all()
            or (ohlc <= 0).any().any()
            or (ohlc.high < ohlc[["open", "close"]].max(axis=1)).any()
            or (ohlc.low > ohlc[["open", "close"]].min(axis=1)).any()
        ):
            prices.append(np.nan)
            moves.append(np.nan)
            continue
        raw, vector, pct60 = ballot(
            morning, prices, moves, prev_close, training, prior_fades
        )
        p = morning.close.iloc[-1]
        prices.append(p)
        moves.append(abs(p / morning.open.iloc[0] - 1))
        # Voluntary abstentions and unavailable votes both have zero exposure.
        calls = {
            b: max(0, v or 0) if asset in ("Crypto", "Options") else (v or 0)
            for b, v in raw.items()
        }
        weights = {b: weight(*history[b]) for b in calls}
        forecasts = {b: v or 0 for b, v in raw.items()}
        selected = choose(forecasts, weights)
        equal = choose(forecasts, dict.fromkeys(calls, 1))
        if asset in ("Crypto", "Options"):
            selected, equal = max(0, selected), max(0, equal)
        entry, end = at("10:01:00"), at("11:00:00")
        net = (
            returns(entry, end, FEES[asset])
            if entry is not None and end is not None
            else None
        )
        scored = (
            day >= pd.Timestamp(score_from)
            if score_from
            else (symbol != "SPY" or day >= pd.Timestamp("2024-07-01"))
        )
        if scored:
            days.append(
                dict(
                    asset=asset,
                    symbol=symbol,
                    date=day.date(),
                    weighted=selected,
                    equal=equal,
                    weighted_bps=net[selected] if net else None,
                    equal_bps=net[equal] if net else None,
                    always_long_bps=net[1] if net else None,
                    pct_vs_60=pct60,
                )
            )
        for bot, pos in calls.items():
            if net and scored:
                grade = sign(net[pos])
                if grade:
                    history[bot][0 if grade > 0 else 1] += 1
            if scored:
                rows.append(
                    dict(
                        asset=asset,
                        symbol=symbol,
                        date=day.date(),
                        strategy=bot,
                        forecast=raw[bot],
                        position=pos,
                        weight=weights[bot],
                        selected=selected,
                        net_bps=net[pos] if net else None,
                        advantage_bps=net[pos] - net[selected] if net else None,
                        next_weight=weight(*history[bot]),
                    )
                )
        if net:
            training.append((vector, int(end.open > entry.open)))
    return rows, days


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/markets"))
    parser.add_argument("--out", type=Path, default=Path("runs/crossmarket"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    votes, days = [], []
    for asset, symbol, frame in streams(args.cache):
        v, d = run_asset(asset, symbol, frame)
        print(symbol, len(d), flush=True)
        votes.extend(v)
        days.extend(d)
    ledger = pd.DataFrame(votes)
    ledger.to_csv(
        args.out / "votes.csv.gz",
        index=False,
        float_format="%.8f",
        compression={"method": "gzip", "mtime": 0},
    )
    sessions = pd.DataFrame(days)
    sessions.to_csv(args.out / "sessions.csv", index=False, float_format="%.8f")
    summary = []
    for (asset, symbol), g in sessions.groupby(["asset", "symbol"]):
        resolved = g.dropna(subset=["weighted_bps"])
        rec = dict(
            asset=asset,
            symbol=symbol,
            start=g.date.min(),
            end=g.date.max(),
            ballots=len(g),
            resolved=len(resolved),
        )
        for variant in ["weighted", "equal"]:
            active = resolved[resolved[variant] != 0]
            rec[variant + "_trades"] = len(active)
            rec[variant + "_mean_bps_per_session"] = resolved[variant + "_bps"].mean()
        rec["long_mean_bps_per_session"] = resolved.always_long_bps.mean()
        summary.append(rec)
    pd.DataFrame(summary).to_csv(
        args.out / "coverage.csv", index=False, float_format="%.4f"
    )
    ledger["traded"] = ledger.position.ne(0) & ledger.net_bps.notna()
    ledger.groupby(["asset", "symbol", "strategy"]).agg(
        available=("forecast", "count"),
        resolved=("net_bps", "count"),
        trades=("traded", "sum"),
        mean_bps_per_session=("net_bps", "mean"),
        mean_advantage_bps=("advantage_bps", "mean"),
    ).to_csv(args.out / "strategies.csv", float_format="%.4f")


if __name__ == "__main__":
    main()
