"""Collect SPY forward ballots with the same clocks and rules as the replay."""

import json
from collections import defaultdict

import pandas as pd

from intraday.voting import weight
from .run import FEES, returns, run_asset
from .strategies import sign


def morning(bars, date, root):
    history = defaultdict(lambda: [0, 0])
    for path in sorted(root.glob("*/close.json")):
        if path.parent.name < date:
            for bot, row in (
                json.loads(path.read_text())
                .get("crossmarket", {})
                .get("bots", {})
                .items()
            ):
                grade = sign(row["net_bps"])
                if grade:
                    history[bot][0 if grade > 0 else 1] += 1
    local = bars.tz_convert("America/New_York").tz_localize(None)
    cut = pd.Timestamp(f"{date} 09:59")
    rows, days = run_asset(
        "ETF", "SPY", local.loc[cut - pd.Timedelta(days=220) : cut], date, history
    )
    if not rows:
        return None
    day = days[-1]
    return dict(
        symbol="SPY",
        weighted=day["weighted"],
        equal=day["equal"],
        bots={
            r["strategy"]: dict(
                forecast=r["forecast"],
                position=r["position"],
                weight=r["weight"],
                wins=history[r["strategy"]][0],
                losses=history[r["strategy"]][1],
            )
            for r in rows
        },
    )


def close(bars, date, packet):
    tz = "America/New_York"
    entry, exit_bar = [pd.Timestamp(f"{date} {t}", tz=tz) for t in ("10:01", "11:00")]
    observed = pd.Timestamp(packet["observed_at"])
    if not entry - pd.Timedelta(minutes=1) <= observed < entry:
        return {"status": "untimely ballot"}
    if entry not in bars.index or exit_bar not in bars.index:
        return {"status": "missing settlement prices"}
    net = returns(bars.loc[entry], bars.loc[exit_bar], FEES["ETF"])
    if net is None:
        return {"status": "missing settlement prices"}
    selected = net[packet["weighted"]]
    bots = {}
    for bot, vote in packet["bots"].items():
        value = net[vote["position"]]
        bots[bot] = dict(
            **vote,
            net_bps=value,
            advantage_bps=value - selected,
            next_weight=weight(
                vote["wins"] + int(value > 0), vote["losses"] + int(value < 0)
            ),
        )
    return dict(
        status="settled",
        weighted_bps=selected,
        equal_bps=net[packet["equal"]],
        bots=bots,
    )
