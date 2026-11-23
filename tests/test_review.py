from contextlib import closing
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from intraday.data import NY
from intraday.paper import open_ledger, save
from intraday.review import check_packet, packet, peer_review, read_ledger


class ReviewerTests(unittest.TestCase):
    def test_arithmetic_correction_and_two_rounds_use_read_only_packet(self):
        recorded = dict(
            status="ready",
            breakout=False,
            band=True,
            orders=["clock", "band", "either"],
            features=dict(
                opening5_return=0.001,
                first30_return=0.003,
                opening_relative_volume=1.5,
                above_range=0.001,
                above_band=0.002,
                above_vwap=0.001,
            ),
        )
        evidence = packet(recorded)
        checked = check_packet(evidence)
        self.assertEqual(
            checked["mismatches"]["breakout"], {"recorded": False, "recomputed": True}
        )
        self.assertEqual(
            checked["mismatches"]["agreement"], {"recorded": False, "recomputed": True}
        )
        self.assertEqual(recorded["orders"], ["clock", "band", "either"])
        self.assertTrue(
            set(evidence).isdisjoint({"entry", "exit", "date", "orders", "net_usd"})
        )
        calls = []

        def reply(request, timeout):
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
            request_data = json.loads(request.data)
            arguments = json.loads(request_data["messages"][1]["content"])
            calls.append(arguments)
            self.assertNotIn("vote", request_data["format"]["properties"])
            first = {
                "evidence_field": "breakout",
                "suggested_experiment": "custom",
                "finding": "Recorded flag disagrees.",
                "future_test": "Replay more complete openings.",
            }
            second = {
                "evidence_field": "breakout",
                "suggested_experiment": "custom",
                "assessment": "correction proposed",
                "correction": "Log the mismatched flag for investigation.",
                "future_test": "Test close-to-high comparison.",
            }
            body = {
                "message": {
                    "content": json.dumps(
                        second if "other_review" in arguments else first
                    )
                }
            }
            return BytesIO(json.dumps(body).encode())

        with patch("intraday.review.urlopen", side_effect=reply):
            result = peer_review("local-test-model", evidence, checked)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(result["cross_checks"]), 2)
        self.assertNotIn("other_review", calls[0])
        self.assertEqual(calls[2]["other_review"]["role"], "evidence_reviewer")
        self.assertEqual(calls[3]["other_review"]["role"], "rule_reviewer")

    def test_read_ledger_pulls_only_decision_without_a_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.db"
            now = pd.Timestamp("2024-07-17 16:00", tz=NY)
            with closing(open_ledger(path)) as db:
                save(
                    db,
                    "replay:v2",
                    "2024-07-17",
                    "decision",
                    now,
                    {"orders": ["agreement"], "status": "ready"},
                )
                save(db, "replay:v2", "2024-07-17", "exit", now, {"net_usd": 1.0})
            before = hashlib.sha256(path.read_bytes()).digest()
            row, mode = read_ledger(path, "2024-07-17", None)
            self.assertEqual(mode, "replay:v2")
            self.assertNotIn("net_usd", row)
            self.assertTrue(packet(row)["recorded"]["agreement"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)
