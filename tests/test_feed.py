from io import BytesIO
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from intraday.data import load
from intraday.feed import refresh


class FeedTests(unittest.TestCase):
    def test_pagination_complete_minutes_and_retry(self):
        pages = []

        def bar(at):
            return dict(t=at, o=100, h=101, l=99, c=100, v=40)

        def get(request, timeout):
            self.assertTrue(
                request.full_url.startswith(
                    "https://data.alpaca.markets/v2/stocks/bars?"
                )
            )
            self.assertNotIn("test-secret", request.full_url)
            pages.append(request.full_url)
            if "page_token=" not in request.full_url:
                body = {
                    "bars": {"SPY": [bar("2024-07-03T13:30:00Z")]},
                    "next_page_token": "second",
                }
            else:
                body = {
                    "bars": {
                        "SPY": [
                            bar("2024-07-03T13:31:00Z"),
                            bar("2024-07-03T13:32:00Z"),
                        ]
                    }
                }
            return BytesIO(json.dumps(body).encode())

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {"APCA_API_KEY_ID": "test-id", "APCA_API_SECRET_KEY": "test-secret"},
            ),
        ):
            root = Path(tmp)
            at = pd.Timestamp("2024-07-03 09:32", tz="America/New_York")
            path = refresh("iex", at, root, get)
            first = path.read_bytes()
            self.assertEqual(len(load(path)), 2)
            refresh("iex", at, root, get)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(len(pages), 4)
