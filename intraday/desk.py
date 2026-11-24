"""Serve the research desk and OpenBB widgets, or export an offline HTML copy."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd
from .experiment import ROOT

TABLE = ROOT / "results" / "momentum60.csv"


def rows():
    return json.loads(pd.read_csv(TABLE).to_json(orient="records", double_precision=12))


def html():
    return (
        (ROOT / "intraday" / "desk.html")
        .read_text()
        .replace("/*DATA*/[]", json.dumps(rows(), allow_nan=False))
    )


def forward_rows():
    records = []
    for name in sorted((ROOT / "forward").glob("*/momentum60.json"), reverse=True):
        signal = json.loads(name.read_text())
        close = name.with_name("close.json")
        result = (
            json.loads(close.read_text()).get("momentum60", {})
            if close.exists()
            else {}
        )
        carry = (
            json.loads(close.read_text()).get("carry60", {}) if close.exists() else {}
        )
        records.append(
            dict(
                date=signal["session"],
                change_pct=signal["change_pct"],
                confirmed_target=signal.get("confirmed_target"),
                carry_position=carry.get("position"),
                carry_net_usd=carry.get("net_usd"),
                position=signal["position"],
                status=result.get("status", "awaiting close"),
                net_usd=result.get("net_usd"),
            )
        )
    return records


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/":
            body, mime = html(), "text/html"
        elif route in ("/sessions", "/forward", "/widgets.json"):
            widgets = {
                key: dict(
                    name=name,
                    description=desc,
                    type="table",
                    endpoint=key,
                    gridData=dict(w=24, h=12),
                )
                for key, name, desc in [
                    (
                        "sessions",
                        "SPY historical sessions",
                        "Historical signals and one-share price moves for the 60-session comparison.",
                    ),
                    (
                        "forward",
                        "SPY forward journal",
                        "Timestamped 60-session positions and outcomes from the paper runner.",
                    ),
                ]
            }
            payload = (
                widgets
                if route == "/widgets.json"
                else (rows() if route == "/sessions" else forward_rows())
            )
            body, mime = json.dumps(payload, allow_nan=False), "application/json"
        else:
            self.send_error(404)
            return
        value = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", mime + "; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(value)))
        self.end_headers()
        self.wfile.write(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        args.export.write_text(html(), encoding="utf-8")
    else:
        print(
            f"Research desk and OpenBB backend: http://127.0.0.1:{args.port}",
            flush=True,
        )
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
