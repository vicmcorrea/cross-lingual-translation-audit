# Cross-Lingual Translation Audit

This repository contains the analysis code for a reference-free audit of paired Portuguese and English survey translations. It includes the Hydra pipeline, frozen model configurations, sensitivity-analysis implementation, and synthetic tests.

No survey responses, source identifiers, credentials, or author metadata are included.

## Verify

```bash
cd experiments
uv sync --frozen
uv run pytest
```

The default Hydra configuration is non-executing and can be inspected with `uv run translation-audit --cfg job`.
