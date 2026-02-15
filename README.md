# Pension ALM under Joint Longevity–Market Regimes

Research-oriented Python code for defined-benefit pension asset-liability management (ALM) under:

- stochastic interest rates (Nelson-Siegel factor dynamics),
- stochastic longevity (mortality improvement index `kappa`),
- regime switching (two-state Markov process), and
- Monte Carlo funding-ratio analysis of static, dynamic, and ML-duration hedges.

This repository is a stylized research framework intended for experimentation and insight generation.

## What Changed

The codebase now includes:

- stronger input/model validation (transition matrices, covariance matrices, regime-state maps),
- shared simulation configuration and path-generation utilities (`src/sim/`),
- vectorized liability/survival valuation paths for better runtime,
- standardized CLI flags for key scripts (`--n-years`, `--n-paths`, `--seed`, `--no-plots`),
- run metadata output for reproducibility (`outputs/*_metadata.json`),
- improved headless plotting behavior via a central plotting utility.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

1. Build toy mortality data:

```bash
.venv/bin/python -m src.build_toy_data
```

2. Run a single-path regime demo:

```bash
.venv/bin/python -m src.run_regime_joint_demo
```

3. Run a short Monte Carlo smoke test:

```bash
.venv/bin/python -m src.run_monte_carlo_regime_model --n-paths 200 --n-years 10
```

4. Run tests:

```bash
.venv/bin/python -m pytest -q
```

## Key Scripts

All commands are run from project root.

- Joint regime single-path demo:

```bash
.venv/bin/python -m src.run_regime_joint_demo --n-years 10 --seed 42
```

- Monte Carlo joint regime model:

```bash
.venv/bin/python -m src.run_monte_carlo_regime_model --n-paths 1000 --n-years 10
```

- Static vs regime-based dynamic hedge:

```bash
.venv/bin/python -m src.run_monte_carlo_dynamic_vs_static --n-paths 1000 --h-normal 1.0 --h-stress 1.1
```

- Static vs ML-duration hedge:

```bash
.venv/bin/python -m src.run_monte_carlo_ml_hedge --n-paths 1000 --n-train-paths 400
```

Useful flags for most simulation scripts:

- `--seed <int>`: reproducible runs
- `--no-plots`: skip plot generation (faster, useful for CI/headless)
- `--metadata-tag <str>`: suffix for metadata filename

## Reproducibility Output

Most scripts write a metadata JSON file in `outputs/`, containing:

- run name,
- UTC timestamp,
- current git commit hash (if available),
- CLI arguments,
- summary statistics.

Example:

- `outputs/mc_regime_joint_metadata.json`
- `outputs/mc_dynamic_vs_static_metadata.json`
- `outputs/mc_ml_hedge_metadata.json`

## Project Structure

```text
pension_alm/
├── src/
│   ├── datasets/                  # toy mortality generation
│   ├── finance/                   # curve models, assets, duration
│   ├── liabilities/               # cashflow build + valuation + survival
│   ├── mortality/                 # kappa processes
│   ├── regimes/                   # Markov regime process
│   ├── sim/                       # shared scenario defaults + path simulation
│   ├── plotting.py                # safe matplotlib setup
│   ├── runtime_utils.py           # CLI metadata helpers
│   └── run_*.py                   # demos and Monte Carlo drivers
├── data/processed/toy_mortality_uk.csv
├── outputs/
├── tests/
├── requirements.txt
└── README.md
```

## Testing

The test suite covers:

- stochastic process reproducibility,
- Markov and model input validation,
- survival and cashflow logic,
- valuation correctness checks,
- shared path-simulation shape/reproducibility guarantees.

Run:

```bash
.venv/bin/python -m pytest -q
```

## Modeling Notes

- Mortality shift uses `log m_x -> log m_x + kappa`.
- Negative `kappa` implies improving mortality and higher expected payouts.
- Regime 0 and 1 use different rate and mortality dynamics.
- Strategies are intentionally stylized to highlight structural hedge limitations.

## Limitations

- Not calibrated to real pension or market data.
- No transaction costs/liquidity constraints.
- Single representative member structure.
- Designed for conceptual research, not production investment advice.
