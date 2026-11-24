"""Attach an independent TradingAgents opinion to an archived research-desk session."""

import argparse
import json
from pathlib import Path
import pandas as pd
from .desk import TABLE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True)
    p.add_argument("--model", required=True, help="Installed local Ollama model tag")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    table = pd.read_csv(TABLE, index_col="date")
    if args.date not in table.index or pd.isna(table.at[args.date, "mean"]):
        p.error("Choose a date with a 60-session signal from the historical journal.")
    if args.out.exists():
        p.error("Choose a new report path.")
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = {
        **DEFAULT_CONFIG,
        "llm_provider": "ollama",
        "backend_url": "http://localhost:11434/v1",
        "deep_think_llm": args.model,
        "quick_think_llm": args.model,
        "max_debate_rounds": 1,
    }
    graph = TradingAgentsGraph(selected_analysts=["market"], config=config)
    _, opinion = graph.propagate("SPY", args.date)
    note = dict(
        session=args.date,
        ticker="SPY",
        retrospective=True,
        change_pct=float(table.at[args.date, "mean"]),
        agent_opinion=str(opinion),
        model=args.model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as out:
        json.dump(note, out, indent=2)


if __name__ == "__main__":
    main()
