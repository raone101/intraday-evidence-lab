"""Give two local reviewers identical, outcome-blind observations from a historical journal."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .experiment import FEATURES, P

SCHEMA = {
    "type": "object",
    "properties": {
        "vote": {"type": "string", "enum": ["participate", "abstain"]},
        "reason": {"type": "string"},
    },
    "required": ["vote", "reason"],
    "additionalProperties": False,
}


def packet(row):
    return {
        "features": {k: float(row[k]) if pd.notna(row[k]) else None for k in FEATURES},
        "breakout": bool(row.breakout),
        "band": bool(row.band),
        "model_probability": float(row.probability)
        if pd.notna(row.probability)
        else None,
        "veto_threshold": P["veto_probability"],
        "cost_bps_each_side": P["primary_cost_bps_each_side"],
    }


def opinion(model, role, evidence):
    instruction = (
        "Make the strongest case for participation."
        if role == "advocate"
        else "Examine whether the evidence warrants abstaining."
    )
    message = (
        "Review one anonymous long-or-cash intraday paper decision. "
        + instruction
        + " Use only the supplied numbers. Returns and distances are decimal fractions. "
        "Probability is an estimate of a positive net clock trade. Return a JSON vote and "
        "a short reason citing at least one number. The role does not compel either vote."
    )
    body = dict(
        model=model,
        stream=False,
        format=SCHEMA,
        options={"temperature": 0, "seed": 719},
        messages=[
            {"role": "system", "content": message},
            {"role": "user", "content": json.dumps(evidence)},
        ],
    )
    req = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=120) as response:
        answer = json.load(response)
    reply = json.loads(answer["message"]["content"])
    if (
        set(reply) != {"vote", "reason"}
        or reply["vote"] not in ("participate", "abstain")
        or not isinstance(reply["reason"], str)
        or not reply["reason"].strip()
    ):
        raise ValueError("Reviewer returned an invalid vote.")
    return {"role": role, "model": answer.get("model", model), **reply}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", type=Path, default=Path("results/journal.csv"))
    p.add_argument("--date", required=True)
    p.add_argument(
        "--model", help="local Ollama model tag; omit to print only the sealed packet"
    )
    args = p.parse_args()
    rows = pd.read_csv(args.journal).set_index("date")
    if args.date not in rows.index:
        p.error("Date absent from journal")
    evidence = packet(rows.loc[args.date])
    result = {"packet": evidence}
    if args.model:
        result["reviews"] = [
            opinion(args.model, r, evidence) for r in ("advocate", "sceptic")
        ]
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
