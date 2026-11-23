import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data import fetch, load
from .experiment import P, ROOT, session_rows, historical_decisions, score, contrasts
from .checks import independent_votes, external_fills


def run(path, dest, other=None):
    bars = load(path).loc[P["source_start"] : P["source_end"]]
    if bars.empty:
        raise ValueError("No bars within the frozen study dates.")
    rows = session_rows(bars)
    votes_checked = independent_votes(bars, rows)
    test = historical_decisions(rows)
    daily, trades, main = score(test, P["primary_cost_bps_each_side"])
    counts = external_fills(bars, test, trades)
    summaries = pd.concat(
        [score(test, b)[2] for b in P["sensitivity_bps_each_side"]], ignore_index=True
    )
    test["decision_reason"] = np.select(
        [~test.eligible, test.agreement, test.either],
        [
            test.quality,
            "both price rules agree",
            "rules disagree",
        ],
        default="neither proposes a trade",
    )
    test["execution_state"] = np.select(
        [~test.agreement, test.entry.isna()],
        ["flat", "entry print missing"],
        default="round trip",
    )
    metadata = dict(
        protocol_version=P["version"],
        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(
            (ROOT / "protocol.json").read_bytes()
        ).hexdigest(),
        bars=len(bars),
        source_start=str(bars.index[0]),
        source_end=str(bars.index[-1]),
        sessions=len(rows),
        eligible_ballots_checked=votes_checked,
        unresolved_training_outcomes=rows.index[
            (rows.index < P["historical_test_start"])
            & rows.eligible
            & rows.entry.notna()
            & rows.exit.isna()
        ].tolist(),
        test_sessions=len(test),
        first_test=test.index[0],
        last_test=test.index[-1],
        test_quality=test.quality.value_counts().to_dict(),
        external_round_trips=counts,
        fills_match_usd_tolerance=1e-8,
    )
    tables = {
        "summary": summaries,
        "trades": trades.reset_index(),
        "daily_net_usd": daily.reset_index(),
        "journal": test.reset_index(),
        "contrasts": contrasts(daily),
    }
    if other is not None:
        second_bars = load(other)
        a, b = rows, session_rows(second_bars)
        keys = a.index.intersection(b.index)
        keys = keys[(a.loc[keys].eligible & b.loc[keys].eligible).values]
        if keys.empty:
            raise ValueError(
                "No overlapping eligible sessions between the two archives."
            )
        both = pd.DataFrame(
            {
                "breakout_primary": a.loc[keys].breakout,
                "breakout_comparison": b.loc[keys].breakout,
                "band_primary": a.loc[keys].band,
                "band_comparison": b.loc[keys].band,
                "clock_return_gap_bps": 10000
                * (
                    (a.loc[keys].exit / a.loc[keys].entry)
                    - (b.loc[keys].exit / b.loc[keys].entry)
                ),
            }
        )
        metadata["cross_archive"] = dict(
            sessions=len(both),
            first=keys[0],
            last=keys[-1],
            breakout_disagreements=int(
                (both.breakout_primary != both.breakout_comparison).sum()
            ),
            band_disagreements=int((both.band_primary != both.band_comparison).sum()),
            return_gap_pairs=int(both.clock_return_gap_bps.notna().sum()),
            median_absolute_return_gap_bps=both.clock_return_gap_bps.abs().median(),
            p95_absolute_return_gap_bps=both.clock_return_gap_bps.abs().quantile(0.95),
            comparison_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise ValueError(
            f"Use a fresh output directory to preserve previous runs: {dest}"
        )
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / "run"
        staged.mkdir()
        for name, table in tables.items():
            table.to_csv(staged / (name + ".csv"), index=False, float_format="%.10g")
        (staged / "audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
        os.rename(staged, dest)
    print(main.to_string(index=False))
    print(
        f"Verified {sum(counts.values())} round trips across variants. Results in {dest}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Historical SPY research. All fills are simulated."
    )
    p.add_argument("--data", type=Path, help="existing one-minute SPY CSV")
    p.add_argument("--compare", type=Path, help="optional second SPY archive")
    p.add_argument(
        "--download",
        action="store_true",
        help="download both registered archives and verify their hashes",
    )
    p.add_argument("--out", type=Path, default=Path("runs/rules-v2"))
    args = p.parse_args()
    if args.download and (args.data or args.compare):
        p.error("Choose downloaded archives or explicit paths.")
    if args.download:
        args.data = fetch("primary", Path("data"))
        args.compare = fetch("comparison", Path("data"))
    if not args.data:
        p.error("Supply --download or --data PATH.")
    run(args.data, args.out, args.compare)


if __name__ == "__main__":
    main()
