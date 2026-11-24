"""Replay four independently scored intraday hypotheses and their intersections."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .activity import PROFILE, sample_sessions
from .data import load
from .experiment import FEATURES, P, session_rows

from .voting import BOTS, fresh_record, freeze, settle

START, END = P["historical_test_start"], P["source_end"]


def opening_evidence(bars):
    result = {}
    opening = bars.between_time("09:30", "09:59")
    for date, g in opening.groupby(opening.index.strftime("%Y-%m-%d")):
        if len(g) != 30:
            continue
        first, rest = g.iloc[:5], g.iloc[5:]
        hi, lo = first.high.max(), first.low.min()
        price = g.close.iloc[-1]
        typical = g[["high", "low", "close"]].mean(axis=1)
        if g.volume.sum() <= 0:
            continue
        vwap = np.average(typical, weights=g.volume)
        spread = np.sqrt(np.average((typical - vwap) ** 2, weights=g.volume))
        result[date] = dict(
            above=price > hi,
            below=price < lo,
            swept_high=rest.high.max() > hi,
            swept_low=rest.low.min() < lo,
            inside=lo < price < hi,
            vwap_z=(price - vwap) / spread if spread else 0,
        )
    return pd.DataFrame.from_dict(result, orient="index")


def votes(frame):
    # A failed high and failed low on the same morning is ambiguous.
    failed = frame.inside * (frame.swept_low.astype(int) - frame.swept_high.astype(int))
    breakout = (frame.opening_relative_volume >= 1) * (
        frame.above.astype(int) - frame.below.astype(int)
    )
    fade = np.select([frame.vwap_z >= 1.5, frame.vwap_z <= -1.5], [-1, 1], default=0)
    prediction = frame.forecast_bps
    model = np.select([prediction > 10, prediction < -10], [1, -1], default=0)
    return pd.DataFrame(
        dict(zip(BOTS, [breakout, failed, fade, model])), index=frame.index
    ).astype(int)


def forecast(frame, start=START, end=END):
    pred = pd.Series(np.nan, index=frame.index)
    cutoff = pd.Series("", index=frame.index)
    inputs = FEATURES + list(PROFILE)
    test = frame.loc[start:end]
    for month, block in test.groupby(test.index.str[:7]):
        train = frame.loc[frame.index < month + "-01"].dropna(subset=["gross_bps"])
        if len(train) < 126:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=100))
        model.fit(train[inputs], train.gross_bps)
        pred.loc[block.index] = model.predict(block[inputs])
        cutoff.loc[block.index] = train.index[-1]
    return pred, cutoff


def combinations_book(v):
    books = {name: v[name] for name in BOTS}
    for a, b in combinations(BOTS, 2):
        books[a + " + " + b] = v[a].where(v[a] == v[b], 0)
    total = v.sum(axis=1)
    books["Two net votes"] = np.sign(total).where(abs(total) >= 2, 0)
    books["Clock"] = pd.Series(1, index=v.index)
    books["Cash"] = pd.Series(0, index=v.index)
    return pd.DataFrame(books)


def build_frame(bars):
    raw = session_rows(bars, require_scored_exit=False)
    frame = sample_sessions(bars, "session_exit", raw).join(opening_evidence(bars))
    frame["gross_bps"] = 10000 * (frame.exit / frame.entry - 1)
    return frame


def morning_ballot(bars, date):
    frame = build_frame(bars)
    if date not in frame.index:
        return None
    frame["forecast_bps"], frame["trained_through"] = forecast(frame, date, date)
    today = frame.loc[[date]]
    books = combinations_book(votes(today))
    evidence = today[
        [
            "above",
            "below",
            "inside",
            "swept_high",
            "swept_low",
            "vwap_z",
            "opening_relative_volume",
            "forecast_bps",
            "trained_through",
        ]
    ]
    return dict(
        positions=json.loads(books.to_json(orient="records"))[0],
        evidence=json.loads(evidence.to_json(orient="records"))[0],
        model_available=bool(today.forecast_bps.notna().iloc[0]),
    )


def run(bars):
    frame = build_frame(bars)
    frame["forecast_bps"], frame["trained_through"] = forecast(frame)
    test = frame.loc[START:END].copy()
    if test.empty or test[["entry", "exit", "forecast_bps"]].isna().any().any():
        raise ValueError(
            "Expected complete execution and forecast data in the evaluation period."
        )
    v = votes(test)
    books = combinations_book(v)
    history = fresh_record()
    ledger = {}
    for date, row in test.iterrows():
        ballot = freeze(v.loc[date], history)
        ledger[date] = settle(
            ballot, row.entry, row.exit, P["primary_cost_bps_each_side"], history
        )
    books["Weighted vote"] = [r["selected"] for r in ledger.values()]
    books["Equal active vote"] = [r["equal"] for r in ledger.values()]
    gross = books.mul(test.exit - test.entry, axis=0)
    costs = books.abs().mul(
        (test.entry + test.exit) * P["primary_cost_bps_each_side"] / 10000, axis=0
    )
    pnl = gross - costs
    summary = pd.DataFrame(
        {
            "trades": books.abs().sum(),
            "gross_usd": gross.sum(),
            "cost_usd": costs.sum(),
            "net_usd": pnl.sum(),
        }
    )
    journal = test[
        [
            "above",
            "below",
            "inside",
            "swept_high",
            "swept_low",
            "vwap_z",
            "opening_relative_volume",
            "forecast_bps",
            "trained_through",
            "entry",
            "exit",
        ]
    ].join(books)
    journal["dispute"] = (v.min(axis=1) < 0) & (v.max(axis=1) > 0)
    journal.index.name = "date"
    return journal, summary.rename_axis("strategy"), ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    journal, summary, ledger = run(load(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    journal.to_csv(args.out / "sessions.csv")
    summary.to_csv(args.out / "summary.csv")
    rows = [
        dict(
            date=date,
            bot=b,
            selected=d["selected"],
            selected_net_usd=d["selected_net_usd"],
            **audit,
        )
        for date, d in ledger.items()
        for b, audit in d["bots"].items()
    ]
    pd.DataFrame(rows).to_csv(args.out / "vote-audit.csv", index=False)
    page = Path(__file__).with_name("arena.html").read_text()
    payload = {
        "sessions": json.loads(journal.reset_index().to_json(orient="records")),
        "voting": ledger,
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
    }
    (args.out / "arena.html").write_text(
        page.replace("/*REPLAY*/{}", json.dumps(payload, allow_nan=False)),
        encoding="utf-8",
    )
    print(summary.round(2))
    print("Sessions with opposing votes:", journal.dispute.sum())


if __name__ == "__main__":
    main()
