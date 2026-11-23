"""Read-only Alpaca stock-bars adapter. No account or order endpoints are used."""

import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import NY

FIELDS = ["caldt", "open", "high", "low", "close", "volume"]


def fetch_bars(feed, start, end, key, secret, opener=urlopen):
    query = {
        "symbols": "SPY",
        "timeframe": "1Min",
        "feed": feed,
        "adjustment": "raw",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 10000,
        "sort": "asc",
    }
    seen = set()
    while True:
        request = Request(
            "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(query),
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )
        with opener(request, timeout=25) as response:
            page = json.load(response)
        for bar in (page.get("bars") or {}).get("SPY", []):
            yield [bar["t"], bar["o"], bar["h"], bar["l"], bar["c"], bar["v"]]
        token = page.get("next_page_token")
        if not token:
            break
        if token in seen:
            raise ValueError("Provider repeated its pagination token.")
        seen.add(token)
        query["page_token"] = token


def refresh(feed, now, root=Path("data"), opener=urlopen):
    key, secret = (
        os.environ.get("APCA_API_KEY_ID"),
        os.environ.get("APCA_API_SECRET_KEY"),
    )
    if not key or not secret:
        raise ValueError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY locally for read-only data access."
        )
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"alpaca-{feed}-SPY.csv"
    old = {}
    if path.exists():
        with path.open(newline="") as source:
            old = {row[0]: row for row in list(csv.reader(source))[1:]}
    end = now.tz_convert("UTC").floor("min") - pd.Timedelta(minutes=1)
    start = (
        pd.Timestamp(max(old)).tz_convert("UTC") if old else end - timedelta(days=400)
    )
    if start > end:
        return path
    changed = False
    for row in fetch_bars(feed, start, end, key, secret, opener):
        stamp = pd.Timestamp(row[0]).tz_convert("UTC")
        local = stamp.tz_convert(NY)
        if stamp > end or not ("09:30" <= local.strftime("%H:%M") < "16:00"):
            continue
        row = [stamp.isoformat(), *(str(v) for v in row[1:])]
        if old.get(row[0]) != row:
            changed = True
            old[row[0]] = row
    if not old:
        raise ValueError("No regular-session SPY bars returned for the selected feed.")
    if changed:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=root, delete=False, newline=""
        ) as tmp:
            temp = Path(tmp.name)
            try:
                writer = csv.writer(tmp)
                writer.writerow(FIELDS)
                writer.writerows(old[k] for k in sorted(old))
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        os.replace(temp, path)
    return path
