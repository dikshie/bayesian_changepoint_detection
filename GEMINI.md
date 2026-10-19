# Bayesian Changepoint Detection (Gemini-Antigravity)

Project guidelines, architecture invariants, and workflow procedures for AI coding agents operating within Google Antigravity (Antigravity CLI `agy`, Antigravity IDE, Antigravity 2.0 desktop app) and Google Gemini environments.

> [!NOTE]
> This file is synchronized with [`AGENTS.md`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/AGENTS.md) (the repository's [agents.md](https://agents.md) specification).

---

## 1. Project Overview

A PyTorch implementation of two Bayesian changepoint detection algorithms:

- **Offline** (`offline_changepoint_detection`): Fearnhead (2006), computes exact posterior distributions over changepoint locations via dynamic programming over segments.
- **Online** (`online_changepoint_detection`): Adams & MacKay (2007), recursively updates the posterior distribution over the *run length* (elapsed timesteps since the last changepoint).

This is a research-grade mathematical library. Prioritize mathematical correctness, numerical precision, and API stability.

---

## 2. Environment & Tooling

The repository uses a local Python virtual environment managed via `uv`:

- **Virtual Environment Path**: `.venv/`
- **Python Executable**: `.venv/bin/python`
- **Testing**: `.venv/bin/pytest`
- **Linter & Formatter**: `.venv/bin/ruff`

Always use the virtual environment binaries or `uv run <cmd>` to ensure dependencies are properly resolved.

---

## 3. Mandatory Development Rules

### Language & Spelling
- All identifiers, docstrings, comments, test markers, and agent documentation **must** be written in **American English**:
  `initialize`, `normalize`, `behavior`, `color`, `neighbor`, `modeling`.
  Renaming public symbols requires a deprecation alias that raises a warning.

### Strict Test Marker Contract
Every test **must** be marked with exactly one classification marker (`tests/conftest.py` will reject tests with neither or both during collection):
- **`@pytest.mark.math`**: Mathematical proofs compared against independent references (e.g., `scipy.stats`, exact marginals in `mpmath`, analytical closed forms). Run via `.venv/bin/pytest -m math`.
- **`@pytest.mark.behavior`**: Software contracts, input validation, edge cases, device transfers, synthetic data detection, or goldens. Run via `.venv/bin/pytest -m behavior`.

### Device Selection & Acceleration
- **Default device is CPU**: Small PyTorch tensors on MPS or CUDA suffer from high launch overhead and run significantly slower than on CPU.
- **Apple Silicon & Metal Support**: On macOS Apple Silicon (M-series), PyTorch uses Apple Metal via the **MPS** backend (`device="mps"`). `get_device()` supports `"metal"`, `"apple_metal"`, `"gpu"`, and `"auto"` aliases.
- In tests and function calls, default to or explicitly specify `device="cpu"` unless specifically testing device mechanics.
- Any test requiring an accelerator must carry `@pytest.mark.gpu` and include an automatic skip when neither CUDA nor MPS is available.
- Always use `device.get_device()` and `device.ensure_tensor()` rather than calling `.to()` or `torch.device` ad hoc.

---

## 4. Critical Mathematical & Implementation Invariants

### 1. Likelihood Class Namespace Separation
Two distinct class hierarchies share identical names (`StudentT`, `NormalKnownVariance`, `Poisson`, `MultivariateT`):
- `bayesian_changepoint_detection.offline_likelihoods`
- `bayesian_changepoint_detection.online_likelihoods`

These interfaces are completely incompatible. **Never import them bare into the same namespace; always import them module-qualified.**

### 2. Differing Likelihood Interfaces
- **Offline**: `pdf(data, t, s)` returns a scalar log marginal likelihood of the segment `data[t:s]` under the prior. Index `s` is **exclusive**.
- **Online**: `pdf(data)` takes a single observation and returns a **vector** of log predictive densities (one per possible run length). Calling `update_theta(data)` advances posterior parameters.

### 3. Online Likelihoods are Stateful and Single-Use
Online likelihood instances accumulate state across timesteps (`update_theta` grows parameter vectors; `pdf` increments an internal counter). Re-instantiate likelihood objects between detection runs. Likelihoods declaring `_run_length_state` support pruning via `OnlineChangepointDetector(max_run_length=...)`.

### 4. Online Detection Output Interpretation
`R[0, t]` (run length 0) equals the hazard rate under a constant hazard function and is not a detection probability. True changepoints appear as probability mass at run length $k$ at time $t + k$. Always use [`get_map_changepoints(R)`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/bayesian_models.py) or [`changepoint_probabilities(R, lag)`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/bayesian_models.py). Never threshold `R[0, :]`.

### 5. Log Space vs. Probability Space
- The **offline** algorithm operates entirely in **log space** (`torch.logaddexp` / `logsumexp`).
- The **online** algorithm updates $R$ in **probability space**, renormalizing each column. (Note: `viterbi_changepoints` maintains its score table $V$ in log space).
- Hazard functions return true probabilities in $[0, 1]$ and must raise `ValueError` for invalid parameters.

### 6. Numerical Precision & Large Offset Stability
- The offline recursion accumulates across $O(n^2)$ terms and runs in **float64** (falling back to CPU on MPS, which lacks float64 support).
- Likelihood models must be translation-equivariant and withstand large offsets (tested up to $10^{10}$ in `tests/test_offline_large_offsets.py` and `tests/test_online_large_offsets.py`).
- Determinant updates with rank-one terms must use `_logdet_plus_rank_one`.

---

## 5. Antigravity Skills & Customizations

Workspace skills are located under [`.agents/skills/`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/):

- [`changepoint-testing`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/changepoint-testing/SKILL.md): Testing workflows, running `pytest`, marker enforcement, and coverage.
- [`changepoint-math`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/changepoint-math/SKILL.md): Conjugate prior derivations, independent proof bar (~1e-9 tolerance), and numerical precision.
- [`changepoint-benchmarks`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/changepoint-benchmarks/SKILL.md): Running runtime benchmarks (`benchmarks/performance.py`) and TCPD evaluation (`benchmarks/tcpd.py`).
- [`python-code-style`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/python-code-style/SKILL.md): Ruff formatting, linting rules, and docstring standards.
- [`python-error-handling`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/python-error-handling/SKILL.md): Narrow exception design and error boundaries.
- [`python-language`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/python-language/SKILL.md): Modern Python typing, dataclasses, and standard library conventions.
- [`python-uv`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/.agents/skills/python-uv/SKILL.md): Dependency management and virtual environment maintenance using `uv`.

### Recommended Slash Commands
- `/plan`: Use when designing new likelihood models, algorithm vectorizations, or refactoring numerical paths.
- `/boost`: Use when performing complex mathematical derivations or deep debugging across conjugate priors.
- `/schedule`: Use to schedule periodic benchmark runs or automated evaluation against TCPD.

---

## 6. Pre-Commit / Pre-PR Verification Routine

Before completing any task or proposing changes, execute the following verification steps:

```bash
# 1. Lint and style checks
.venv/bin/ruff check
.venv/bin/ruff format --check

# 2. Mathematical proof validation
.venv/bin/pytest -m math

# 3. Behavior and contract test suite
.venv/bin/pytest -m behavior

# 4. Full test suite execution
.venv/bin/pytest
```
