import json
import tempfile
from pathlib import Path
from typing import Any

from scripts import eval_retrieval


def _make_tiny_dataset(tmp: Path) -> Path:
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "doc_0001.txt").write_text("alpha bravo")
    (corpus / "doc_0002.txt").write_text("charlie delta")
    queries = [
        {"id": "q1", "q": "alpha", "relevant_ids": ["doc_0001"]},
        {"id": "q2", "q": "charlie", "relevant_ids": ["doc_0002"]},
    ]
    qpath = tmp / "queries.jsonl"
    with qpath.open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    import datetime
    import hashlib

    docs_dict: dict[str, str] = {
        "doc_0001.txt": hashlib.sha256(b"alpha bravo").hexdigest(),
        "doc_0002.txt": hashlib.sha256(b"charlie delta").hexdigest(),
    }
    manifest: dict[str, Any] = {
        "docs": docs_dict,
        "docs_count": 2,
        "queries_count": len(queries),
        "created_at": datetime.datetime.utcnow().isoformat(),
    }
    hash_input = "".join(sorted(docs_dict.values())) + str(len(queries))
    manifest["version"] = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    mpath = tmp / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    return mpath


def test_eval_outputs_schema(monkeypatch):
    monkeypatch.setenv("USE_DUMMY_EMBEDDINGS", "true")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        manifest = _make_tiny_dataset(tmp)
        report = eval_retrieval.evaluate("bm25", 10, 1337, manifest)
        assert {"recall@1", "MRR", "NDCG@10"}.issubset(report["metrics"].keys())


def test_compare_eval_regression_guard(tmp_path):
    current = tmp_path / "cur.json"
    baseline = tmp_path / "base.json"
    cur = {"metrics": {"MRR": 0.4, "recall@3": 0.5, "NDCG@10": 0.6}}
    base = {"metrics": {"MRR": 0.5, "recall@3": 0.5, "NDCG@10": 0.7}}
    current.write_text(json.dumps(cur))
    baseline.write_text(json.dumps(base))
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_eval.py",
            "--current",
            str(current),
            "--baseline",
            str(baseline),
            "--delta",
            "0.01",
        ],
    )
    assert result.returncode == 1


def test_manifest_hashes_valid():
    import hashlib

    manifest = json.loads(Path("data/manifest.json").read_text())
    for fname, sha in manifest["docs"].items():
        data = Path("data/corpus") / fname
        actual = hashlib.sha256(data.read_bytes()).hexdigest()
        assert actual == sha
