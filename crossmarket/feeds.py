"""Normalise vendor minute bars. Raw licensed samples stay outside Git."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

NY = "America/New_York"
COLS = ["open", "high", "low", "close"]
HERE = Path(__file__).parent


def get_file(cache, name):
    spec = json.loads((HERE / "sources.json").read_text())[name]
    p = cache / name
    if not p.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urlopen(spec["url"], timeout=90) as response:
            body = response.read()
        if hashlib.sha256(body).hexdigest() != spec["sha256"]:
            raise ValueError(f"{name}: source has rotated, supply the pinned sample")
        p.write_bytes(body)
    if hashlib.sha256(p.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"{name}: checksum mismatch")
    return p


def quotes(bid, ask):
    mid = (bid + ask) / 2
    mid["bid"], mid["ask"] = bid.open, ask.open
    mid["volume"] = np.nan
    return mid


def streams(cache):
    def file(name):
        return get_file(cache, name)

    for asset, symbols in [
        ("Equity", "TSLA NVDA AAPL AMZN META MSFT NFLX F"),
        ("ETF", "QQQ"),
        ("Futures", "ES NQ GC"),
    ]:
        for symbol in symbols.split():
            future = asset == "Futures"
            with ZipFile(file(("futures_" if future else "") + symbol + ".zip")) as z:
                member = symbol + ("_1min_continuous.csv" if future else "_1min.csv")
                with z.open(member) as f:
                    bars = pd.read_csv(f, names=["time", *COLS, "volume"], index_col=0)
            bars.index = pd.to_datetime(bars.index)
            yield asset, symbol, bars
    spy = pd.read_csv(file("primary.csv"), parse_dates=["caldt"]).set_index("caldt")
    yield "ETF", "SPY", spy.loc["2023":]
    for symbol in ("EURUSD", "GBPUSD", "USDJPY"):
        frames = []
        for w in range(24, 28):
            raw = pd.read_csv(file(f"{symbol}-{w}.gz"))
            raw.index = (
                pd.to_datetime(raw.DateTime, format="%m/%d/%Y %H:%M:%S.%f", utc=True)
                .dt.tz_convert(NY)
                .dt.tz_localize(None)
            )
            sides = [
                raw[[s + c.title() for c in COLS]].set_axis(COLS, axis=1)
                for s in ("Bid", "Ask")
            ]
            frames.append(quotes(*sides))
        yield "Forex", symbol, pd.concat(frames)
    for symbol in ("BTCUSDT", "ETHUSDT"):
        frames = []
        for month in ("04", "05", "06"):
            raw = pd.read_csv(
                file(f"{symbol}-2025-{month}.zip"), header=None, usecols=range(6)
            )
            raw.columns = ["time", *COLS, "volume"]
            raw.index = (
                pd.to_datetime(raw.pop("time"), unit="us", utc=True)
                .dt.tz_convert(NY)
                .dt.tz_localize(None)
            )
            frames.append(raw)
        yield "Crypto", symbol, pd.concat(frames)
    with ZipFile(file("option.zip")) as z:
        for right in ("call", "put"):
            member = f"20140606_aapl_minute_quote_american_{right}_6500000_20140621.csv"
            with z.open(member) as f:
                raw = pd.read_csv(f, header=None)
            raw.index = pd.Timestamp("2014-06-06") + pd.to_timedelta(raw[0], unit="ms")
            sides = [
                raw.iloc[:, i : i + 4].set_axis(COLS, axis=1) / 10000 for i in (1, 6)
            ]
            yield "Options", f"AAPL-20140621-650-{right}", quotes(*sides)
