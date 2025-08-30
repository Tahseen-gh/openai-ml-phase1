from __future__ import annotations

import argparse
import json
import sys

from fastapi_app.app.config import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()

    with open(args.current) as f:
        current = json.load(f)
    with open(args.baseline) as f:
        baseline = json.load(f)

    delta = settings.allowed_regression_delta
    failed = False
    for k, base_v in baseline["metrics"].items():
        cur_v = current["metrics"].get(k, 0.0)
        if cur_v + delta < base_v:
            print(f"Metric {k} regressed: {cur_v} < {base_v - delta}")
            failed = True
    if failed:
        sys.exit(1)
    print("All metrics within allowed delta")


if __name__ == "__main__":
    main()
