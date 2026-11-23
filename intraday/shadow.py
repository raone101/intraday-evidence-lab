"""Run or register read-only intraday forecasting experiments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import exchange_calendars as xc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_limits

from .activity import PROFILE, RANGE_INPUTS, sample_sessions
from .data import NY, load
from .experiment import FEATURES, P, ROOT, paired_interval, session_rows

S = P["shadow"]
FULL = FEATURES + list(PROFILE)
COMPARISONS = S["experiments"]
FORECASTS = (
    "base_rate",
    "price_logit",
    "price_neural",
    "profile_neural",
    "recent_neural",
    "transfer_neural",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recipe_digest():
    files = [
        ROOT / name
        for name in (
            "protocol.json",
            "intraday/activity.py",
            "intraday/shadow.py",
            "intraday/experiment.py",
            "requirements.txt",
        )
    ]
    return hashlib.sha256("".join(digest(path) for path in files).encode()).hexdigest()


def can_fit(sample):
    return len(sample) >= S["minimum_training_sessions"] and sample.won.nunique() == 2


def network():
    return MLPClassifier(
        hidden_layer_sizes=(S["hidden_units"],),
        activation="logistic",
        solver="adam",
        random_state=S["seed"],
        alpha=0.05,
    )


def train_network(model, x, y):
    for _ in range(S["epochs_each_stage"]):
        model.partial_fit(x, y.to_numpy(dtype=int), classes=np.array([0, 1]))
    return model


def neural_forecast(train, test, features):
    scale = MinMaxScaler(clip=True).fit(train[features])
    model = train_network(network(), scale.transform(train[features]), train.won)
    return model.predict_proba(scale.transform(test[features]))[:, 1]


def range_clusters(train, test):
    scale = StandardScaler().fit(train[RANGE_INPUTS])
    count = min(3, len(train[RANGE_INPUTS].drop_duplicates()))
    model = KMeans(n_clusters=count, n_init=10, random_state=S["seed"])
    model.fit(scale.transform(train[RANGE_INPUTS]))
    ranks = np.argsort(np.argsort(model.cluster_centers_[:, 0])) + 1
    return [
        f"range_{rank}"
        for rank in ranks[model.predict(scale.transform(test[RANGE_INPUTS]))]
    ]


def forecasts(target, start, end, source=None):
    predictions = target.loc[start:end, ["agreement", "won", "net_bps"]].copy()
    folds = []
    for name in FORECASTS:
        predictions[name] = np.nan
    predictions["regime"] = "unavailable"
    predictions["training_through"] = ""
    history = target.dropna(subset=["won"])
    origin = history if source is None else source.dropna(subset=["won"])
    with threadpool_limits(limits=1):
        for month, block in predictions.groupby(predictions.index.str[:7]):
            train = history.loc[history.index < month + "-01"]
            recent = train.tail(S["recent_sessions"])
            prior = (
                origin.loc[origin.index < recent.index[0]]
                if len(recent)
                else origin.iloc[:0]
            )
            fold = dict(
                month=month,
                training_sessions=len(train),
                recent_sessions=len(recent),
                older_sessions=len(prior),
                training_through=train.index.max() if len(train) else "",
                source_through=prior.index.max() if len(prior) else "",
            )
            folds.append(fold)
            if not can_fit(train):
                continue
            x = target.loc[block.index]
            predictions.loc[block.index, "training_through"] = fold["training_through"]
            predictions.loc[block.index, "base_rate"] = float(train.won.mean())
            logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            logit.fit(train[FEATURES], train.won.to_numpy(dtype=int))
            predictions.loc[block.index, "price_logit"] = logit.predict_proba(
                x[FEATURES]
            )[:, 1]
            predictions.loc[block.index, "price_neural"] = neural_forecast(
                train, x, FEATURES
            )
            predictions.loc[block.index, "profile_neural"] = neural_forecast(
                train, x, FULL
            )
            predictions.loc[block.index, "regime"] = range_clusters(train, x)
            if can_fit(recent):
                scale = MinMaxScaler(clip=True).fit(recent[FULL])
                rx, tx = scale.transform(recent[FULL]), scale.transform(x[FULL])
                adapted = train_network(network(), rx, recent.won)
                predictions.loc[block.index, "recent_neural"] = adapted.predict_proba(
                    tx
                )[:, 1]
                if can_fit(prior):
                    pretrained = train_network(
                        network(), scale.transform(prior[FULL]), prior.won
                    )
                    adapted = train_network(pretrained, rx, recent.won)
                    predictions.loc[block.index, "transfer_neural"] = (
                        adapted.predict_proba(tx)[:, 1]
                    )
    predictions.index.name = "date"
    return predictions, pd.DataFrame(folds)


def evaluate(predictions, observation_complete=True):
    """Compare on matched days. Lower Brier error means better probability estimates."""
    usable = predictions.dropna(subset=["won"])
    slices = [("all", usable), ("agreement", usable.loc[usable.agreement])]
    slices.extend(
        (name, x) for name, x in usable.groupby("regime") if name != "unavailable"
    )
    rows = []
    for group, sample in slices:
        for experiment, (candidate, baseline) in COMPARISONS.items():
            pair = sample[[candidate, baseline, "won"]].dropna()
            y = pair.won.to_numpy(dtype=float)
            a, b = (
                (pair[candidate].to_numpy() - y) ** 2,
                (pair[baseline].to_numpy() - y) ** 2,
            )
            n = len(pair)
            low, high = (
                paired_interval(a - b)
                if n >= S["minimum_paired_sessions"]
                else (np.nan, np.nan)
            )
            if not observation_complete:
                result = "collecting"
            elif n < S["minimum_paired_sessions"]:
                result = "insufficient"
            elif high < 0:
                result = "lower error"
            elif low > 0:
                result = "higher error"
            else:
                result = "inconclusive"
            rows.append(
                dict(
                    group=group,
                    experiment=experiment,
                    candidate=candidate,
                    baseline=baseline,
                    matched=n,
                    candidate_brier=a.mean() if n else np.nan,
                    baseline_brier=b.mean() if n else np.nan,
                    delta=(a - b).mean() if n else np.nan,
                    lower95=low,
                    upper95=high,
                    result=result,
                )
            )
    return pd.DataFrame(rows)


def fill_stress(bars, rows):
    """Use common rows while perturbing the agreement entry clock and cost assumptions."""
    agreement = rows.loc[rows.breakout & rows.band & rows.exit.notna()]
    delays = S["stress_entry_delay_minutes"]
    entries = pd.DataFrame(
        {
            m: bars.open.reindex(
                pd.DatetimeIndex(agreement.entry_at + pd.Timedelta(minutes=m))
            ).to_numpy()
            for m in delays
        },
        index=agreement.index,
    ).dropna()
    exit_price = agreement.loc[entries.index, "exit"]
    return pd.DataFrame(
        [
            dict(
                entry_delay_minutes=m,
                cost_bps_each_side=bps,
                matched_trades=len(entries),
                net_usd=(
                    (exit_price - entries[m] - (exit_price + entries[m]) * bps / 10_000)
                    * P["shares"]
                ).sum()
                if len(entries)
                else np.nan,
            )
            for m in delays
            for bps in P["sensitivity_bps_each_side"]
        ]
    )


def day(value):
    try:
        date = pd.Timestamp(value)
        if date.strftime("%Y-%m-%d") != value:
            raise ValueError
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Use date format YYYY-MM-DD.") from None
    return value


def write_plan(
    path,
    through,
    horizon,
    hypothesis,
    transfer_symbol=None,
    *,
    experiment="profile",
    cohort="all",
):
    if (
        experiment not in COMPARISONS
        or cohort not in ("all", "agreement")
        or horizon not in S["horizons"]
    ):
        raise ValueError("Unknown experiment, cohort or horizon.")
    now = pd.Timestamp.now(tz="UTC")
    after = str(now.tz_convert(NY).date())
    if not hypothesis.strip():
        raise ValueError("Provide a falsifiable hypothesis.")
    if transfer_symbol and transfer_symbol.upper() == P["symbol"]:
        raise ValueError("Use a non-SPY instrument for cross-instrument transfer.")
    if through <= after:
        raise ValueError("The trial end must be after today's New York date.")
    payload = dict(
        version=S["version"],
        registered_at=now.isoformat(),
        after=after,
        through=through,
        horizon=horizon,
        hypothesis=hypothesis,
        experiment=experiment,
        cohort=cohort,
        transfer_symbol=transfer_symbol,
        recipe_sha256=recipe_digest(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return payload


def run(args):
    plan = json.loads(args.trial.read_text()) if args.trial else None
    if plan:
        registered = pd.Timestamp(plan["registered_at"])
        if (
            plan["version"] != S["version"]
            or plan["recipe_sha256"] != recipe_digest()
            or registered > pd.Timestamp.now(tz="UTC")
            or plan["after"] != str(registered.tz_convert(NY).date())
            or plan["through"] <= plan["after"]
            or plan["experiment"] not in COMPARISONS
            or plan["cohort"] not in ("all", "agreement")
        ):
            raise ValueError(
                "The trial manifest no longer matches this protocol or registration date."
            )
        start = str((pd.Timestamp(plan["after"]) + pd.Timedelta(days=1)).date())
        end, horizon, symbol = plan["through"], plan["horizon"], plan["transfer_symbol"]
    else:
        start = args.start or P["historical_test_start"]
        end = args.end or P["source_end"]
        horizon, symbol = args.horizon or "15m", args.transfer_symbol
    if start > end or horizon not in S["horizons"]:
        raise ValueError("Check the date range and forecast horizon.")
    if bool(args.transfer_data) != bool(symbol) or (
        symbol and symbol.upper() == P["symbol"]
    ):
        raise ValueError(
            "Supply one non-SPY source symbol together with its own minute CSV."
        )
    target_hash = digest(args.data)
    bars = load(args.data)
    rows = session_rows(bars, require_scored_exit=False)
    target = sample_sessions(bars, horizon, rows)
    sample = target.loc[start:end]
    if sample.empty:
        raise ValueError(f"No eligible completed openings between {start} and {end}.")
    source = None
    source_hash = None
    if args.transfer_data:
        source_hash = digest(args.transfer_data)
        if source_hash == target_hash:
            raise ValueError("Source and target CSVs have identical contents.")
        source = sample_sessions(load(args.transfer_data), horizon)
    preds, folds = forecasts(target, start, end, source)
    cal = xc.get_calendar("XNYS")
    last_session = cal.date_to_session(end, direction="previous")
    complete = bars.index[-1] >= cal.session_close(last_session) - pd.Timedelta(
        minutes=1
    )
    metrics = evaluate(preds, complete)
    costs = fill_stress(bars, rows.loc[start:end])
    votes = rows.loc[start:end].query("eligible")
    overlap = {
        f"breakout_{int(a)}_band_{int(b)}": int(
            ((votes.breakout == a) & (votes.band == b)).sum()
        )
        for a in (False, True)
        for b in (False, True)
    }
    phi = (
        votes.breakout.corr(votes.band)
        if votes.breakout.nunique() == votes.band.nunique() == 2
        else np.nan
    )
    meta = dict(
        mode="registered follow-up" if plan else "retrospective",
        requested_start=start,
        requested_end=end,
        first_prediction_session=preds.index[0],
        last_prediction_session=preds.index[-1],
        horizon=horizon,
        observation_end_reached=bool(complete),
        target_sha256=target_hash,
        source_symbol=symbol or "SPY earlier sessions",
        source_sha256=source_hash,
        trial_sha256=digest(args.trial) if plan else None,
        recipe_sha256=recipe_digest(),
        protocol_sha256=digest(ROOT / "protocol.json"),
        eligible_vote_overlap=overlap,
        vote_phi=float(phi) if np.isfinite(phi) else None,
    )
    if plan:
        selected = metrics.loc[
            (metrics.experiment == plan["experiment"])
            & (metrics.group == plan["cohort"])
        ]
        meta["registered_comparison"] = json.loads(selected.to_json(orient="records"))[
            0
        ]
    dest = args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"Use a new output directory: {dest}")
    with TemporaryDirectory(dir=dest.parent) as tmp:
        path = Path(tmp) / "run"
        path.mkdir()
        for name, table in [
            ("predictions", preds.reset_index()),
            ("folds", folds),
            ("comparisons", metrics),
            ("fill_stress", costs),
        ]:
            table.to_csv(path / f"{name}.csv", index=False, float_format="%.10g")
        (path / "audit.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n"
        )
        os.rename(path, dest)
    print(metrics.query("group in ['all', 'agreement']").to_string(index=False))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("register")
    draft.add_argument("--out", type=Path, required=True)
    draft.add_argument("--through", type=day, required=True)
    draft.add_argument("--horizon", choices=S["horizons"], default="15m")
    draft.add_argument("--hypothesis", required=True)
    draft.add_argument("--experiment", choices=COMPARISONS, default="profile")
    draft.add_argument("--cohort", choices=["all", "agreement"], default="all")
    draft.add_argument("--transfer-symbol")
    score = commands.add_parser("run")
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--start", type=day)
    score.add_argument("--end", type=day)
    score.add_argument("--horizon", choices=S["horizons"])
    score.add_argument("--trial", type=Path)
    score.add_argument("--transfer-data", type=Path)
    score.add_argument("--transfer-symbol")
    args = cli.parse_args()
    try:
        if args.command == "register":
            write_plan(
                args.out,
                args.through,
                args.horizon,
                args.hypothesis,
                args.transfer_symbol,
                experiment=args.experiment,
                cohort=args.cohort,
            )
            print(f"Registered strictly later sessions in {args.out}")
        else:
            if args.trial and any(
                [args.start, args.end, args.horizon, args.transfer_symbol]
            ):
                cli.error("The trial controls its dates, horizon and transfer source.")
            run(args)
    except (ValueError, OSError, KeyError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
