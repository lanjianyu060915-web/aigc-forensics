from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True)
parser.add_argument("--output", required=True)
args, _unknown = parser.parse_known_args()

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(
        {
            "status": "success",
            "query_image": args.query,
            "candidates": [
                {
                    "rank": 1,
                    "source_id": "fixture-source",
                    "final_score": 1.0,
                }
            ],
            "runtime_seconds": 0.001,
            "error_message": None,
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print("SOURCE_FIXTURE_RAN", flush=True)
