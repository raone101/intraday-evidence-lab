"""Build an offline, single-file ballot viewer from the replay outputs."""

import base64
import sys
from pathlib import Path

folder = Path(sys.argv[1])
template = Path(__file__).with_name("viewer.html").read_text()
for token, name in [
    ("VOTES_PAYLOAD", "votes.csv.gz"),
    ("SESSION_PAYLOAD", "sessions.csv"),
]:
    template = template.replace(
        token, base64.b64encode((folder / name).read_bytes()).decode()
    )
(folder / "replay.html").write_text(template)
