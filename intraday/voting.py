"""Frozen morning ballots and after-cost feedback from independently scored calls."""

import json
from math import isclose

BOTS = ("Breakout", "Failed break", "VWAP fade", "Profile model")


def fresh_record():
    return {bot: [0, 0] for bot in BOTS}


def weight(wins, losses):
    return 1 + (wins - losses) / (2 * (20 + wins + losses))


def choose(positions, weights):
    active = [bot for bot in BOTS if positions[bot]]
    score = sum(positions[b] * weights[b] for b in active)
    if len(active) < 2 or isclose(score, 0, abs_tol=1e-10):
        return 0
    return 1 if score > 0 else -1


def freeze(positions, history):
    calls = {bot: int(positions[bot]) for bot in BOTS}
    weights = {bot: weight(*history[bot]) for bot in BOTS}
    return dict(
        calls=calls,
        weights=weights,
        selected=choose(calls, weights),
        equal=choose(calls, dict.fromkeys(BOTS, 1)),
    )


def settle(ballot, entry, exit_price, bps, history):
    def net(position):
        return (
            position * (exit_price - entry)
            - abs(position) * (entry + exit_price) * bps / 10000
        )

    selected_net = net(ballot["selected"])
    audit = {}
    for bot, position in ballot["calls"].items():
        result = net(position)
        grade = int(result > 0) - int(result < 0)
        if grade:
            history[bot][0 if grade > 0 else 1] += 1
        audit[bot] = dict(
            call=position,
            before=ballot["weights"][bot],
            net_usd=result,
            delta_usd=result - selected_net,
            grade=grade,
            after=weight(*history[bot]),
        )
    return dict(
        selected=ballot["selected"],
        selected_net_usd=selected_net,
        equal=ballot["equal"],
        equal_net_usd=net(ballot["equal"]),
        bots=audit,
    )


def settled_record(root):
    history = fresh_record()
    for path in sorted(root.glob("*/close.json")):
        data = json.loads(path.read_text()).get("voting")
        if data:
            for bot, row in data["bots"].items():
                if row["grade"]:
                    history[bot][0 if row["grade"] > 0 else 1] += 1
    return history
