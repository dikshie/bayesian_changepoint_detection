---
name: "changepoint-testing"
version: "1.0.0"
platforms: [ANTIGRAVITY, GEMINI, CLAUDE_CODE]
description: "Procedures and rules for running and authoring tests in the bayesian_changepoint_detection repository. Use when running pytest, adding new tests, debugging test failures, or validating the mandatory math vs behavior marker contract."
---

# Testing Bayesian Changepoint Detection

This skill outlines the strict testing rules, marker contracts, device conventions, and execution workflows for the `bayesian_changepoint_detection` repository.

## 1. Test Markers Contract (Mandatory)

The test suite enforces a strict collection-time check defined in [`tests/conftest.py`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/tests/conftest.py):

Every test **must** declare what kind of evidence it represents:
- **`@pytest.mark.math`**: Mathematically proves the implementation against an independent reference (e.g., `scipy.stats`, exhaustive enumeration, closed form from literature, or exact rational arithmetic in `mpmath`).
- **`@pytest.mark.behavior`**: Pins software behavior, API contracts, input validation, edge cases, device transfers, synthetic data detection, or regression goldens.

> [!CAUTION]
> A test with **neither** marker, or with **both** markers, will trigger a fatal `pytest.UsageError` during test collection.
> You can set markers per function (`@pytest.mark.math`), per class, or module-wide (`pytestmark = pytest.mark.behavior`).

## 2. Device Conventions

- **Default Device is CPU**: Always pass `device="cpu"` explicitly in tests unless testing device-specific mechanics.
- **Accelerator Warning**: Small PyTorch tensors on MPS or CUDA incur high kernel launch latency and run dramatically slower than CPU (minutes vs seconds).
- **GPU Markers**: Any test requiring an accelerator or CUDA must carry `@pytest.mark.gpu` and include an automatic skip if no GPU is available:
  ```python
  @pytest.mark.gpu
  def test_gpu_feature():
      if not torch.cuda.is_available():
          pytest.skip("CUDA not available")
      ...
  ```

## 3. Running Tests

Always run tests through the Python virtual environment (`.venv` via `uv`):

```bash
# Run all tests (CPU)
.venv/bin/pytest

# Run only mathematical proofs
.venv/bin/pytest -m math

# Run only behavior and contract tests
.venv/bin/pytest -m behavior

# Run a specific test module
.venv/bin/pytest tests/test_studentt_exactness.py

# Run with coverage
.venv/bin/pytest --cov=bayesian_changepoint_detection
```

## 4. Linting and Formatting

All changes must pass Ruff checks before submitting:

```bash
# Check linting
.venv/bin/ruff check

# Check formatting
.venv/bin/ruff format --check

# Format files automatically
.venv/bin/ruff format
```
