# SECURITY.md

**Reporting a vulnerability:** Please open a private advisory on GitHub or email SECURITY_CONTACT (replace).

## Threat model (LLM/RAG-focused)
- Prompt injection & data exfil → sanitize/strip HTML, never execute user content.
- Poisoned corpus → record provenance; allow block/rollback of specific docs.
- Secrets leakage → `.env` is gitignored; use GitHub Actions secrets; run detect-secrets.
- Supply chain → SBOM uploaded from CI; pin/lock deps when possible.

## Mitigations implemented
- Pre-commit: ruff, mypy (type errors), trailing whitespace/EOF fixers, detect-secrets.
- CI: tests + coverage ≥80%, eval-lite, SBOM artifact.
- Service: health check; optional OTEL traces; no write-side endpoints by default.
