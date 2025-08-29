# Model Card — openai-ml-phase1 (RAG service)

**Purpose:** Retrieval-Augmented Generation backend for answering questions over small corpora.
**Intended use:** Local/dev demo; educational; small internal KBs.
**Not for:** Safety-critical decisions, PHI/PII processing without review.

## System
- Retriever: BM25 over sentence/heading-aware chunks.
- Orchestration: FastAPI service + Python libs.
- Data: user-supplied text/markdown; provenance recorded in datasheet.

## Risks & mitigations
- Hallucinations → keep answers grounded in retrieved chunks; show sources.
- Prompt injection → strip/escape user-provided markup; avoid tool invocation.
- Privacy → don’t log raw queries/contexts by default; allow opt-out logging.

## Evaluation
- Lite: recall@k, MRR, nDCG on a tiny fixture set (CI).
- Optional: RAGAS (faithfulness, answer relevance) when API key is present.

## Limitations
- BM25 only (no dense/rerank yet).
- Small corpora; no sharding/TTL/ACLs.
