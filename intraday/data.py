"""Retrieve pinned archives on demand and validate the one-minute OHLCV contract."""

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SOURCES = {
    "primary": (
        "https://media.githubusercontent.com/media/Ascensao/Intraday-momentum-strategy/"
        "5a9e956994d1ca9ef16e5e0ae6af4ce4e04e87eb/spy_intra_data.csv",
        "d4671c48d18bdaab21f4b44e02992d2fc91070b01a357189f67b2305a75f9ee6",
    ),
    "comparison": (
        "https://raw.githubusercontent.com/noiiiise/SPY-Intraday-Momentum/"
        "d5af5981fd66477b5791a445b97077bcd8d07a01/data/spy_intra_data.csv",
        "ab9b2790a7af73a16d9776ad81eefdbd7eb4f456e4b4419a95cd1ba0f0bed8eb",
    ),
}
NY = "America/New_York"


def fetch(name, folder):
    url, expected = SOURCES[name]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (name + ".csv")
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Local {name} hash differs from the pinned archive.")
        return dest
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
        temp_path = Path(tmp.name)
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "IntradayEvidenceLab/1"}),
                timeout=90,
            ) as r:
                while chunk := r.read(262144):
                    tmp.write(chunk)
            tmp.flush()
            if hashlib.sha256(temp_path.read_bytes()).hexdigest() != expected:
                raise ValueError("Source bytes differ from the registered download.")
        except BaseException:
            tmp.close()
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, dest)
    return dest


def load(path):
    df = pd.read_csv(path, usecols=["caldt", "open", "high", "low", "close", "volume"])
    stamps = df.pop("caldt")
    if stamps.str.contains(r"(?:Z|[+-]\d\d:\d\d)$").all():
        index = pd.to_datetime(stamps, utc=True).dt.tz_convert(NY)
    elif not stamps.str.contains(r"(?:Z|[+-]\d\d:\d\d)$").any():
        index = pd.to_datetime(stamps).dt.tz_localize(
            NY, ambiguous="raise", nonexistent="raise"
        )
    else:
        raise ValueError("Mixed zoned and unzoned timestamps.")
    df.index = pd.DatetimeIndex(index, name="time")
    if df.empty or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError(
            "Expected a non-empty chronological archive with unique bar starts."
        )
    if not (df.index == df.index.floor("min")).all():
        raise ValueError("Expected minute-aligned timestamps.")
    price = df[["open", "high", "low", "close"]]
    if (
        not np.isfinite(df.to_numpy()).all()
        or (price <= 0).any().any()
        or (df.volume < 0).any()
        or (df.high < price.max(axis=1) - 1e-7).any()
        or (df.low > price.min(axis=1) + 1e-7).any()
    ):
        raise ValueError("Invalid prices, volume or OHLC bounds.")
    return df
