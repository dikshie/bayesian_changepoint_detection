---
name: "changepoint-benchmarks"
version: "1.0.0"
platforms: [ANTIGRAVITY, GEMINI, CLAUDE_CODE]
description: "Procedures for running performance benchmarks and TCPD quality evaluations in the bayesian_changepoint_detection repository. Use when benchmarking speedups, evaluating changepoint detection quality, or profiling."
---

# Benchmarking Bayesian Changepoint Detection

This skill describes how to run and interpret performance benchmarks and changepoint detection quality benchmarks on the Turing Change Point Dataset (TCPD).

## 1. Runtime Performance Benchmarking

Benchmark timing across algorithm versions, matrix vectorization, and data sizes:

```bash
# Run runtime performance suite
.venv/bin/python benchmarks/performance.py

# Inspect results in benchmarks/results/
ls -la benchmarks/results/
```

Key considerations:
- The offline recursion is $O(n^2)$ over segments.
- Avoid Python-level inner loops over segment points.
- Always benchmark before and after algorithmic or vectorization changes, and document numbers in the PR description.

## 2. Detection Quality on TCPD (`tcpd.py`)

Evaluate changepoint detection accuracy and F1 metrics against the Turing Change Point Dataset benchmark:

```bash
# Run TCPD evaluation
.venv/bin/python benchmarks/tcpd.py
```

Notes:
- Benchmark datasets cache in `benchmarks/.cache/` (which is `.gitignore`d).
- Compare precision, recall, and covering metrics before and after statistical tuning.
