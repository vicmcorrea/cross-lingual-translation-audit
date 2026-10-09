# Cross-lingual translation audit

Code for [When Short Responses Are Hard to Distinguish in Translation Audits](https://openreview.net/forum?id=Kq2qqKPOUE), accepted at MultiPsyche 2026.

Includes frozen configurations, sensitivity analyses, and synthetic tests. Survey text is confidential and excluded. Reproducing results requires authorized data access.

## Verify

Python 3.14 and uv are required.

```bash
cd experiments
uv sync --frozen
uv run pytest
```

Preview the configuration:

```bash
uv run translation-audit --cfg job
```
