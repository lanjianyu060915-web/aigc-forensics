"""Execute only a G2-bound, independently admitted, fixed acceptance manifest."""
import argparse
import json
from pathlib import Path
import sys
try:
    from tools.run_v4_dev_batch import run
except ModuleNotFoundError:
    from run_v4_dev_batch import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--g2-release', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--minimum-free-gib', type=float, default=20)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.max_items = None
    try:
        return run(args, acceptance=True)
    except Exception as exc:
        print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__, 'error_message': str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
