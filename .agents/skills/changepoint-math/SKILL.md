---
name: "changepoint-math"
version: "1.0.0"
platforms: [ANTIGRAVITY, GEMINI, CLAUDE_CODE]
description: "Mathematical standards and verification procedures for Bayesian changepoint likelihoods and priors. Use when modifying, adding, or refactoring conjugate-prior likelihood models, priors, or hazard functions."
---

# Mathematical Standards for Likelihoods and Priors

This skill details mathematical derivations, numerical precision requirements, and validation standards for likelihood models and priors in `bayesian_changepoint_detection`.

## 1. Grounding in Literature

Every likelihood model is a conjugate-prior derivation from published literature:
- **Offline detector**: Fearnhead, P. (2006). *Exact and efficient Bayesian inference for multiple changepoint problems*. Statistics and Computing 16(2), 203–213.
- **Online detector**: Adams, R. P., & MacKay, D. J. (2007). *Bayesian online changepoint detection*. arXiv:0710.3742.
- **Normal-Normal & Normal-Gamma**: Murphy, K. (2007). *Conjugate Bayesian analysis of the Gaussian distribution*.
- **Normal-Wishart**: Xuan, X., & Murphy, K. (2007). *Modeling changing dependency structure in multivariate time series*. ICML.
- **Gamma-Poisson**: Gelman et al. (2013). *Bayesian Data Analysis*, 3rd ed., section 2.6.

When modifying or introducing a model:
1. Cite the exact paper and equation in the docstring.
2. Maintain mathematical equivalence to the documented formulation.

## 2. Independent Proof Standard (~1e-9 in float64)

Changes to mathematical formulas cannot be tested solely against self-generated values. They must be validated against an independent implementation:
- Check marginal log-likelihoods against chain-rule products of predictive densities evaluated in `scipy.stats`.
- Validate rational representations using `mpmath` under high precision.
- Agreement must achieve ~1e-9 tolerance in `float64`.

## 3. Offline vs. Online Likelihood Contracts

> [!WARNING]
> Two distinct classes share the same names (`StudentT`, `NormalKnownVariance`, `Poisson`, `MultivariateT`) across `offline_likelihoods` and `online_likelihoods`. Always import them qualified!

### Offline Likelihoods (`offline_likelihoods.py`)
- `pdf(data, t, s)` returns a scalar log marginal likelihood for segment `data[t:s]`, where `s` is **exclusive**.
- Evaluates exact marginal likelihoods via prefix sums.
- Recursion operates strictly in **float64** in log space (`torch.logaddexp` / `logsumexp`).
- MPS does not support float64 and automatically falls back to CPU.
- Must preserve numerical stability under large data offsets (tested up to 1e10 in [`tests/test_offline_large_offsets.py`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/tests/test_offline_large_offsets.py)).

### Online Likelihoods (`online_likelihoods.py`)
- `pdf(data)` accepts a single observation and returns a **vector** of log predictive densities, one per candidate run length.
- `update_theta(data)` advances the state by appending updated posterior hyperparameters.
- Objects are **stateful and single-use**. Reinstantiate between detection runs.
- Models supporting pruning define `_run_length_state` for `OnlineChangepointDetector(max_run_length=...)`.
- Evaluated in probability space during `online_changepoint_detection`.
- Large offset stability is checked up to 1e10 in [`tests/test_online_large_offsets.py`](file:///Users/dikshie/VIRTUAL/bayesian_changepoint_detection/tests/test_online_large_offsets.py).

## 4. Hazard Functions (`hazard_functions.py`)

- Must return true probabilities in `[0, 1]`.
- Must validate inputs and raise `ValueError` for out-of-range parameters (e.g., `lam < 1` in `constant_hazard`).
