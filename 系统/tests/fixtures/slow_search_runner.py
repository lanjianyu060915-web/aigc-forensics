from __future__ import annotations

import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--query", required=True)
parser.add_argument("--output", required=True)
parser.parse_known_args()
time.sleep(5)
