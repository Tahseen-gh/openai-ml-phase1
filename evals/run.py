import json
from pathlib import Path

DATASET = Path(__file__).parent / "dataset" / "qas.jsonl"


def load_dataset():
    rows = []
    with DATASET.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def run():
    data = load_dataset()
    # Placeholder: we just echo the dataset to demonstrate structure.
    results = [
        {"question": r["question"], "pred": "TBD", "gold": r["answer"]} for r in data
    ]
    out = Path("evals/results_run01.jsonl")
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {out} ({len(results)} rows)")


if __name__ == "__main__":
    run()
