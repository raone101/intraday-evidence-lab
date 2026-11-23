"""Timestamped, idempotent paper decisions against an arriving one-minute CSV."""

import argparse
import hashlib
from contextlib import closing
from datetime import time
import json
from pathlib import Path
import sqlite3
import time as clock
from urllib.error import HTTPError, URLError

import exchange_calendars as xc
import pandas as pd

from .data import NY, load
from .experiment import FEATURES, P, fit_month, session_rows
from .feed import refresh


def open_ledger(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("""CREATE TABLE IF NOT EXISTS events (
        mode TEXT, session TEXT, event TEXT, recorded_at TEXT, payload TEXT,
        PRIMARY KEY (mode, session, event))""")
    return db


def save(db, mode, date, event, now, payload):
    with db:
        db.execute(
            "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?)",
            (mode, date, event, now.isoformat(), json.dumps(payload, allow_nan=False)),
        )


def events(db, mode, date):
    return {
        e: json.loads(p)
        for e, p in db.execute(
            "SELECT event, payload FROM events WHERE mode=? AND session=? ORDER BY rowid",
            (mode, date),
        )
    }


def ballot(visible, date, previous):
    if visible.empty or previous not in set(visible.index.strftime("%Y-%m-%d")):
        return {"status": "incomplete history", "orders": []}
    rows = session_rows(visible, require_scored_exit=False)
    if date not in rows.index or not rows.at[date, "eligible"]:
        return {"status": "opening bars or warm-up incomplete", "orders": []}
    row = rows.loc[date]
    model, fold = fit_month(rows, date[:7])
    p = (
        float(model.predict_proba(rows.loc[[date], FEATURES])[0, 1])
        if model is not None
        else None
    )
    a, b = bool(row.breakout), bool(row.band)
    flags = dict(
        clock=True,
        breakout=a,
        band=b,
        either=a or b,
        agreement=a and b,
        agreement_ml=a and b and p is not None and p >= P["veto_probability"],
    )
    return dict(
        status="ready",
        orders=[k for k, v in flags.items() if v],
        breakout=a,
        band=b,
        probability=p,
        trained_through=fold.get("last"),
        training_rows=fold["rows"],
        features={k: float(row[k]) for k in FEATURES},
    )


def advance(db, bars, mode, date, now):
    """Only closed bars are visible. A late first observation cannot backdate an order."""
    cal = xc.get_calendar("XNYS")
    if not cal.is_session(date):
        return {}
    opened = cal.session_open(date).tz_convert(NY)
    closed = cal.session_close(date).tz_convert(NY)
    decide = opened + pd.Timedelta(minutes=30)
    at = time.fromisoformat(P["entry_et"])
    entry = opened.replace(hour=at.hour, minute=at.minute)
    exit_time = closed - pd.Timedelta(minutes=P["exit_minutes_before_close"])
    if now < decide:
        return {}
    log = events(db, mode, date)
    if "decision" not in log:
        if now >= entry:
            packet = {"status": "decision window missed", "orders": []}
        else:
            visible = bars.loc[bars.index < decide]
            previous = cal.previous_session(date).strftime("%Y-%m-%d")
            packet = ballot(visible, date, previous)
            if packet["status"] != "ready":
                return {}
            if mode.startswith("forward"):
                now = pd.Timestamp.now(tz=NY)
                if now >= entry:
                    packet = {
                        "status": "decision window missed before record",
                        "orders": [],
                    }
        save(db, mode, date, "decision", now, packet)
        log = events(db, mode, date)
    if not log["decision"]["orders"] or "exit" in log or "expired" in log:
        return log
    # A 10:05 bar is available only when its minute has closed.
    day = bars.loc[
        (bars.index >= opened)
        & (bars.index < closed)
        & (bars.index + pd.Timedelta(minutes=1) <= now)
    ]
    if "entry" not in log and now >= entry + pd.Timedelta(minutes=1):
        if entry in day.index:
            save(
                db,
                mode,
                date,
                "entry",
                now,
                dict(time=entry.isoformat(), price=float(day.at[entry, "open"])),
            )
        elif (day.index > entry).any() or now >= closed:
            save(db, mode, date, "expired", now, {"reason": "entry bar unavailable"})
        log = events(db, mode, date)
    if "entry" in log:
        exits = day.loc[day.index >= exit_time]
        if not exits.empty:
            px = float(exits.open.iloc[0])
            buy = log["entry"]["price"]
            fee = P["shares"] * (buy + px) * P["primary_cost_bps_each_side"] / 10000
            save(
                db,
                mode,
                date,
                "exit",
                now,
                dict(
                    time=exits.index[0].isoformat(),
                    price=px,
                    cost_usd=fee,
                    net_usd_per_participating_variant=(px - buy) * P["shares"] - fee,
                ),
            )
        elif now >= closed:
            save(
                db,
                mode,
                date,
                "data_gap",
                now,
                {"reason": "exit unresolved; awaiting a same-session bar"},
            )
    return events(db, mode, date)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--replay", metavar="YYYY-MM-DD", help="advance an archived day in three stages"
    )
    group.add_argument(
        "--watch", action="store_true", help="follow a locally updated SPY minute CSV"
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data", type=Path, help="locally refreshed, chronological SPY minute CSV"
    )
    source.add_argument(
        "--alpaca", choices=["iex", "sip"], help="read-only market data for --watch"
    )
    p.add_argument("--db", type=Path, default=Path("runs/paper.sqlite"))
    p.add_argument(
        "--poll", type=int, default=20, help="seconds between forward observations"
    )
    args = p.parse_args()
    if args.poll < 5:
        p.error("--poll must be at least five seconds")
    if args.replay and args.alpaca:
        p.error("Replay uses archived --data. The market feed is for --watch.")
    cal = xc.get_calendar("XNYS")
    with closing(open_ledger(args.db)) as db:
        if args.replay:
            date = args.replay
            if not cal.is_session(date):
                p.error("Replay requires an XNYS session")
            data = load(args.data)
            with args.data.open("rb") as opened_file:
                replay_id = (
                    "replay:"
                    + hashlib.file_digest(opened_file, "sha256").hexdigest()[:16]
                )
            if date not in set(data.index.strftime("%Y-%m-%d")):
                p.error("Session absent from supplied data")
            start, end = (
                cal.session_open(date).tz_convert(NY),
                cal.session_close(date).tz_convert(NY),
            )
            for at in [
                start + pd.Timedelta(minutes=30),
                start + pd.Timedelta(minutes=36),
                end,
            ]:
                log = advance(db, data, replay_id, date, at)
            print(json.dumps(log, indent=2))
            return
        last_mtime = None
        last_status = {}
        path = args.data
        source_id = (
            "alpaca-" + args.alpaca
            if args.alpaca
            else hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
        )
        mode = "forward:" + source_id
        while True:
            now = pd.Timestamp.now(tz=NY)
            date = str(now.date())
            if args.alpaca:
                # Stop polling the provider outside the session, except to bootstrap on startup.
                active = cal.is_session(date) and cal.session_open(
                    date
                ) <= now <= cal.session_close(date) + pd.Timedelta(minutes=10)
                if active or path is None:
                    try:
                        path = refresh(args.alpaca, now)
                    except (URLError, TimeoutError) as error:
                        if isinstance(error, HTTPError) and error.code in (401, 403):
                            p.error(
                                "Data provider rejected the supplied credentials or feed entitlement."
                            )
                        print(
                            "Data request failed. Retrying after the poll interval.",
                            flush=True,
                        )
                        clock.sleep(args.poll)
                        continue
                    now = pd.Timestamp.now(tz=NY)
                    date = str(now.date())
            stamp = path.stat().st_mtime_ns
            if stamp != last_mtime:
                data = load(path)
                last_mtime = stamp
            pending = [
                r[0]
                for r in db.execute(
                    """SELECT DISTINCT a.session FROM events a
                WHERE a.mode=? AND a.event='decision' AND a.session < ?
                AND json_array_length(a.payload,'$.orders') > 0 AND NOT EXISTS
                (SELECT 1 FROM events b WHERE b.mode=a.mode AND b.session=a.session
                 AND b.event IN ('exit','expired'))""",
                    (mode, date),
                )
            ]
            for session in pending + [date]:
                log = advance(db, data, mode, session, now)
                if log and log != last_status.get(session):
                    print(session, json.dumps(log), flush=True)
                    last_status[session] = log
            clock.sleep(args.poll)


if __name__ == "__main__":
    main()
