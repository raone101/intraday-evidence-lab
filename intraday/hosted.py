"""Capture time-stamped SPY paper decisions and settle them after the session."""

import argparse
from datetime import time
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time as clock

import exchange_calendars as xc
import pandas as pd

from crossmarket import forward as wide

from .arena import morning_ballot
from .voting import freeze, settle, settled_record
from .activity import sample_sessions
from .data import NY, load
from .experiment import P, ROOT
from .feed import FIELDS, fetch_bars
from .momentum import signals, side, paper_net, confirmed, carry_step
from .paper import ballot
from .review import check_packet, packet
from .shadow import forecasts, recipe_digest

PLAN = ROOT / "research" / "profile-follow-up.json"
JOURNAL = ROOT / "forward"
CAL = xc.get_calendar("XNYS")
FEED = "iex"


def now():
    return pd.Timestamp.now(tz=NY)


def bars_between(start, end):
    """Fetch from the stock-data endpoint only. Raw bars stay on the runner."""
    key, secret = (os.getenv(k) for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"))
    if not (key and secret):
        raise ValueError(
            "Set both Alpaca paper credentials in repository Actions secrets."
        )
    with TemporaryDirectory() as temp:
        path = Path(temp) / "bars.csv"
        with path.open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(FIELDS)
            rows = iter(fetch_bars(FEED, start, end, key, secret))
            first = next(rows, None)
            if first is None:
                return pd.DataFrame()
            writer.writerow(first)
            writer.writerows(rows)
        data = load(path)
    return data.loc[start:end].between_time("09:30", "16:00", inclusive="left")


def record(path, payload):
    """Keep the first observation of each stage across retries."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with path.open("x") as out:
        out.write(content)


def stamp(date, stage, observed, **data):
    return dict(
        session=date,
        stage=stage,
        observed_at=observed.isoformat(),
        feed=FEED,
        run_id=os.getenv("GITHUB_RUN_ID"),
        **data,
    )


def due_times(date):
    opening = CAL.session_open(date).tz_convert(NY)
    cut = opening + pd.Timedelta(minutes=30)
    entry = opening.replace(
        hour=time.fromisoformat(P["entry_et"]).hour,
        minute=time.fromisoformat(P["entry_et"]).minute,
    )
    return opening, cut, entry, CAL.session_close(date).tz_convert(NY)


def collect_open(date, plan, root):
    opening, cut, entry, _ = due_times(date)
    day = root / date
    decision = day / "decision.json"
    forecast = day / "forecast.json"
    if decision.exists() and (forecast.exists() or now() >= entry):
        return
    started = now()
    if started >= entry:
        record(
            decision,
            stamp(date, "decision", started, status="window missed", orders=[]),
        )
        return
    if started < cut - pd.Timedelta(minutes=45):
        raise ValueError("Start the open run within 45 minutes of the decision window.")
    previous = CAL.previous_session(date).strftime("%Y-%m-%d")
    history_start = pd.Timestamp(plan["after"], tz=NY) - pd.Timedelta(days=400)
    history = bars_between(
        history_start.tz_convert("UTC"),
        CAL.session_close(previous) - pd.Timedelta(minutes=1),
    )
    if history.empty:
        raise ValueError("The provider returned no training history.")
    clock.sleep(max(0, (cut - now()).total_seconds()))
    while now() < entry:
        today = bars_between(opening, cut - pd.Timedelta(minutes=1))
        if not today.empty:
            bars = pd.concat([history, today])
            wide_path = day / "crossmarket.json"
            if not wide_path.exists() and now() < cut + pd.Timedelta(minutes=1):
                wide_packet = wide.morning(bars, date, root)
                seen = now()
                if wide_packet and seen < cut + pd.Timedelta(minutes=1):
                    record(wide_path, stamp(date, "crossmarket", seen, **wide_packet))
            capture_momentum(bars, date, day, entry)
            vote = ballot(bars, date, previous)
            observed = now()
            if observed < entry and vote["status"] == "ready":
                if not decision.exists():
                    record(decision, stamp(date, "decision", observed, **vote))
                saved = json.loads(decision.read_text())
                if saved["status"] != "ready":
                    return
                # A failed or slow forecast cannot remove the persisted rule decision.
                if recipe_digest() != plan["recipe_sha256"]:
                    raise ValueError(
                        "Research code differs from the registered recipe."
                    )
                if saved["features"] != vote["features"]:
                    raise ValueError(
                        "The feed revised the opening since the decision was captured."
                    )
                sample = sample_sessions(bars, plan["horizon"])
                if date not in sample.index:
                    raise ValueError(
                        "The opening profile is unavailable for this session."
                    )
                preds, folds = forecasts(sample, date, date)
                row = preds.loc[date]
                models = ["price_neural", "profile_neural"]
                observed = now()
                timely = observed < entry
                scores = {
                    m: float(row[m]) for m in models if timely and pd.notna(row[m])
                }
                status = (
                    "ready" if len(scores) == len(models) else "forecast unavailable"
                )
                record(
                    forecast,
                    stamp(
                        date,
                        "forecast",
                        observed,
                        status=status,
                        probabilities=scores,
                        training_through=str(folds.training_through.iloc[0]),
                        recipe_sha256=plan["recipe_sha256"],
                    ),
                )
                if now() < entry and not (day / "arena.json").exists():
                    ballot_packet = morning_ballot(bars, date)
                    if ballot_packet:
                        ballot_packet["voting"] = freeze(
                            ballot_packet["positions"], settled_record(root)
                        )
                    captured = now()
                    if ballot_packet and captured < entry:
                        record(
                            day / "arena.json",
                            stamp(date, "arena", captured, **ballot_packet),
                        )
                return
        clock.sleep(min(10, max(0, (entry - now()).total_seconds())))
    record(
        decision,
        stamp(date, "decision", now(), status="no timely complete opening", orders=[]),
    )


def capture_momentum(bars, date, day, deadline):
    path = day / "momentum60.json"
    if path.exists():
        return
    changes = signals(bars)
    if date not in changes.index or pd.isna(changes.at[date, "mean"]):
        return
    change = float(changes.at[date, "mean"])
    observed = now()
    if observed < deadline:
        record(
            path,
            stamp(
                date,
                "momentum60",
                observed,
                reference="mean",
                confirmed_target=confirmed(change, changes["mean"].shift(1).at[date]),
                change_pct=change,
                position=side(change),
            ),
        )


def settle_momentum(bars, date, signal):
    _, cut, entry, close = due_times(date)
    if not signal or not cut <= pd.Timestamp(signal["observed_at"]) < entry:
        return {"status": "no timely signal"}
    position = signal["position"]
    if position == 0:
        return {"status": "flat", "net_usd": 0}
    if bars.empty or entry not in bars.index:
        return {"status": "entry bar missing"}
    exits = bars.loc[
        (bars.index >= close - pd.Timedelta(minutes=P["exit_minutes_before_close"]))
        & (bars.index < close)
    ]
    if exits.empty:
        return {"status": "exit bar missing"}
    return {
        "status": "settled",
        "net_usd": paper_net(
            position,
            float(bars.at[entry, "open"]),
            float(exits.open.iloc[0]),
            P["primary_cost_bps_each_side"],
        ),
    }


def settlement(bars, date, decision, forecast):
    """Score a frozen paper decision and the separately frozen probability pair."""
    _, cut, entry, close = due_times(date)
    if (
        not decision
        or decision["status"] != "ready"
        or not cut <= pd.Timestamp(decision["observed_at"]) < entry
    ):
        return {"status": "no timely decision", "orders": []}
    orders = decision["orders"]
    result = dict(
        status="entry bar missing",
        orders=orders,
        review=check_packet(packet(decision)),
        net_usd_by_variant=None,
        forecast_outcome=None,
    )
    if bars.empty or entry not in bars.index:
        return result
    buy = float(bars.at[entry, "open"])
    fee = P["primary_cost_bps_each_side"] / 10_000
    end_time = close - pd.Timedelta(minutes=P["exit_minutes_before_close"])
    exits = bars.loc[(bars.index >= end_time) & (bars.index < close)]
    result["status"] = "exit bar missing"
    if not exits.empty:
        sell = float(exits.open.iloc[0])
        net = ((sell - buy) - (buy + sell) * fee) * P["shares"]
        result.update(
            status="settled",
            exit_time=exits.index[0].isoformat(),
            net_usd_by_variant={
                v: round(net, 8) if v in orders else 0.0 for v in P["variants"]
            },
        )
    horizon = entry + pd.Timedelta(minutes=15)
    if (
        forecast
        and forecast["status"] == "ready"
        and cut <= pd.Timestamp(forecast["observed_at"]) < entry
        and horizon in bars.index
    ):
        won = int(bars.at[horizon, "open"] * (1 - fee) > buy * (1 + fee))
        result["forecast_outcome"] = dict(
            won=won,
            brier={m: (p - won) ** 2 for m, p in forecast["probabilities"].items()},
        )
    return result


def collect_close(date, root):
    day = root / date
    final = day / "close.json"
    if final.exists():
        return
    opening, _, _, close = due_times(date)
    if now() < close:
        raise ValueError("Settlement requires a completed exchange session.")

    def read(name):
        path = day / name
        return json.loads(path.read_text()) if path.exists() else None

    decision = read("decision.json")
    momentum = read("momentum60.json")
    earlier = sorted(p for p in root.glob("*/close.json") if p.parent.name < date)
    prior = json.loads(earlier[-1].read_text()).get("carry60", {}) if earlier else {}
    previous = prior.get("position", 0)
    # Never jump over a captured instruction whose execution is still unresolved.
    since = earlier[-1].parent.name if earlier else ""
    if any(since < p.parent.name < date for p in root.glob("*/momentum60.json")):
        raise ValueError("A prior captured session still requires settlement.")
    wide_packet = read("crossmarket.json")
    data = (
        bars_between(opening, close - pd.Timedelta(minutes=1))
        if previous
        or momentum
        or wide_packet
        or (decision and decision["status"] == "ready")
        else pd.DataFrame()
    )
    result = settlement(data, date, decision, read("forecast.json"))
    if wide_packet:
        result["crossmarket"] = wide.close(data, date, wide_packet)
        if result["crossmarket"]["status"] == "missing settlement prices":
            record(
                day / "crossmarket-gap.json",
                stamp(date, "close", now(), **result["crossmarket"]),
            )
    result["momentum60"] = settle_momentum(data, date, momentum)
    gaps = {"entry bar missing", "exit bar missing"}
    if result["status"] in gaps or result["momentum60"]["status"] in gaps:
        record(day / "close-gap.json", stamp(date, "close", now(), **result))
        raise ValueError(
            "Settlement bars missing. A later close run may resolve the gap."
        )
    target = momentum.get("confirmed_target") if momentum else None
    target = previous if target is None else target
    if previous or target:
        _, _, entry, close = due_times(date)
        marks = (
            data.loc[
                (
                    data.index
                    >= close - pd.Timedelta(minutes=P["exit_minutes_before_close"])
                )
                & (data.index < close)
            ]
            if not data.empty
            else data
        )
        if marks.empty or entry not in data.index:
            raise ValueError("Missing multi-day ledger valuation.")
        mark = float(marks.open.iloc[0])
        liquidate = date == json.loads(PLAN.read_text())["through"]
        gross, cost = carry_step(
            previous,
            target,
            float(data.at[entry, "open"]),
            mark,
            prior.get("mark", 0),
            liquidate,
        )
        result["carry60"] = dict(
            position=0 if liquidate else target,
            mark=mark,
            net_usd=gross - cost * P["primary_cost_bps_each_side"],
            previous_position=previous,
        )
    else:
        result["carry60"] = dict(position=0, net_usd=0)
    arena = read("arena.json")
    if arena and result["status"] == "settled":
        _, cutoff, entry, close = due_times(date)
        if cutoff <= pd.Timestamp(arena["observed_at"]) < entry:
            buy = float(data.at[entry, "open"])
            sell = float(data.at[pd.Timestamp(result["exit_time"]), "open"])
            result["arena_net_usd"] = {
                name: paper_net(pos, buy, sell, P["primary_cost_bps_each_side"])
                for name, pos in arena["positions"].items()
            }
            if "voting" in arena:
                result["voting"] = settle(
                    arena["voting"],
                    buy,
                    sell,
                    P["primary_cost_bps_each_side"],
                    settled_record(root),
                )
    record(final, stamp(date, "close", now(), **result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "open", "close"])
    args = parser.parse_args()
    date = str(now().date())
    plan = json.loads(PLAN.read_text())
    if args.stage == "check":
        if recipe_digest() != plan["recipe_sha256"]:
            parser.error("The registered recipe fingerprint differs.")
        day = CAL.date_to_session(date, direction="previous")
        if CAL.session_close(day) > now():
            day = CAL.previous_session(day)
        end = CAL.session_close(day) - pd.Timedelta(minutes=1)
        data = bars_between(end - pd.Timedelta(minutes=10), end)
        if data.empty:
            parser.error("The feed returned no bars for the last completed session.")
        print(f"Alpaca IEX returned {len(data)} SPY bars for {day.date()}.")
        return
    if not plan["after"] < date <= plan["through"] or not CAL.is_session(date):
        print("Outside an eligible research session.")
        return
    if args.stage == "open":
        collect_open(date, plan, JOURNAL)
    else:
        for pending in sorted(
            {*JOURNAL.glob("*/decision.json"), *JOURNAL.glob("*/momentum60.json")}
        ):
            session = pending.parent.name
            if (
                plan["after"] < session < date
                and not (pending.parent / "close.json").exists()
            ):
                try:
                    collect_close(session, JOURNAL)
                except ValueError as error:
                    print(f"Pending close for {session}: {error}")
        collect_close(date, JOURNAL)


if __name__ == "__main__":
    main()
