# OWASP GenAI/LLM Risks — What we address

- Prompt Injection: treat retrieved text as untrusted; strip scripts/links in UI; avoid tool use.
- Sensitive Data Exposure: keep logs minimal; redact obvious PII before storing.
- Supply Chain: SBOM via CycloneDX; dependabot recommended.
- Model Behavior: RAGAS evals (optional) to monitor faithfulness/relevance.
- Abuse Handling: small rate-limit & size limit recommended at gateway (not included here).
