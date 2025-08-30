"""Update evaluation baseline and append changelog entry."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Update baseline report")
    parser.add_argument("--from", dest="src", type=Path, required=True)
    args = parser.parse_args()

    src: Path = args.src
    baseline = Path("evals/baseline.json")
    shutil.copyfile(src, baseline)

    report = json.loads(src.read_text())
    changelog = Path("evals/CHANGELOG.md")
    changelog.parent.mkdir(exist_ok=True)
    line = f"- {datetime.datetime.utcnow().isoformat()} baseline from {src.name} (sha {report.get('git_sha')})\n"
    with changelog.open("a") as f:
        f.write(line)
    print(f"Baseline updated from {src}")


if __name__ == "__main__":
    main()
