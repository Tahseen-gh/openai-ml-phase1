"""Compare evaluation metrics against baseline and enforce regression gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval metrics")
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument(
        "--gate-metrics",
        default="recall@3,MRR,NDCG@10",
        help="Comma-separated metrics to guard",
    )
    args = parser.parse_args()

    cur = load(args.current)
    base = load(args.baseline)
    metrics = [m.strip() for m in args.gate_metrics.split(",")]

    cur_metrics = cur["metrics"]
    base_metrics = base["metrics"]

    rows: list[str] = []
    failed = False
    for m in metrics:
        cm = cur_metrics.get(m)
        bm = base_metrics.get(m)
        diff = None
        if cm is not None and bm is not None:
            diff = cm - bm
            if diff < -args.delta:
                failed = True
        rows.append(f"| {m} | {bm:.4f} | {cm:.4f} | {diff:.4f} |")

    header = "| metric | baseline | current | diff |\n|---|---|---|---|"
    table = "\n".join([header] + rows)
    print(table)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
