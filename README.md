# Pension Asset–Liability Management under Joint Longevity–Market Regimes
### Rates · Longevity · Regimes · Monte Carlo · Machine Learning

This repository contains a **research-oriented Python project** studying **joint longevity and financial risk** in defined benefit (DB) pension schemes.

The project develops a unified **stochastic Asset–Liability Management (ALM)** framework in which:

- Interest rates evolve stochastically via a factor-based yield curve model  
- Longevity evolves stochastically via a mortality improvement index κ (kappa)  
- A latent macroeconomic regime process jointly drives both rates and longevity  
- Pension funding risk is evaluated via Monte Carlo simulation  
- Static, dynamic, and machine-learning-based hedging strategies are compared  

The central finding is that **standard duration-based hedging techniques can fail structurally** when longevity risk affects the *level* of liabilities rather than only their timing, even when machine learning is used to estimate duration accurately.

---

## Motivation

Defined benefit pension schemes are exposed to two dominant long-horizon risks:

1. **Interest rate risk**, typically managed using liability-driven investment (LDI)  
2. **Longevity risk**, driven by sustained mortality improvements  

In most practical and academic settings, these risks are modelled separately. However, low interest-rate environments often coincide with accelerated longevity improvements, creating **joint and state-dependent risk**.

This project investigates how pension funding risk behaves when interest rates and longevity are modelled jointly, and whether dynamic or machine-learning-based hedging strategies can mitigate downside risk.

---

## Overview of the Framework

The project proceeds through the following stages:

1. Construction of a toy mortality surface and visualisation of log-mortality dynamics  
2. Definition of DB pension cashflows and survival-adjusted expected payments  
3. Valuation of liabilities under flat and term-structure discounting  
4. Introduction of stochastic interest rates via a factor-based yield curve model  
5. Introduction of stochastic longevity improvements via a mortality index κ  
6. Introduction of regime switching that jointly governs:
   - interest-rate dynamics  
   - longevity improvement dynamics  
7. Monte Carlo simulation of assets, liabilities, and funding ratios  
8. Comparison of hedging strategies:
   - Static LDI-style hedge  
   - Naive regime-based dynamic hedge  
   - Machine-learning-based duration hedge  

---

## Key Findings

Monte Carlo results produced by this repository show that:

- A correctly sized static hedge does **not** eliminate downside funding risk  
- Regime persistence creates **state-dependent tail risk**  
- Naive dynamic hedging rules can **increase** downside risk  
- Machine-learning-based duration hedging can also **worsen tail outcomes**  
- Longevity risk primarily affects **cashflow magnitude**, not just timing  

These findings indicate that duration matching is not a sufficient control variable for managing pension risk under joint longevity–market dynamics.

---

## Project Structure

```text
pension_alm/
├── src/
│   ├── finance/
│   ├── liabilities/
│   ├── mortality/
│   ├── regimes/
│   ├── run_*.py
│   └── config.py
├── data/
│   └── processed/
│       └── toy_mortality_uk.csv
├── outputs/
├── tests/
│── research/
│    ├── research_note.md
│    └── figures/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation


```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run

All commands are run from the project root.

### Regime joint demo

```bash
python -m src.run_regime_joint_demo
```

### Monte Carlo: joint regime model

```bash
python -m src.run_monte_carlo_regime_model
```

### Static vs naive dynamic hedge

```bash
python -m src.run_monte_carlo_dynamic_vs_static
```

### Static vs ML-duration hedge

```bash
python -m src.run_monte_carlo_ml_hedge
```

---

## Mortality Improvement Index κ

The mortality model applies:

```
log m_{x,t} = log m_{x,base} + κ_t
m_{x,t} = m_{x,base} · exp(κ_t)
```

Negative κ corresponds to improving mortality and increasing expected pension payouts.

---

## Regime Switching

A two-state Markov chain governs regimes:

- Regime 0: normal conditions  
- Regime 1: stress conditions  

Regimes jointly affect interest-rate dynamics and longevity improvement dynamics.

---

## Testing and Validation

Tests verify:
- Reproducibility of stochastic processes
- Validity of regime paths
- Logical behaviour of survival probabilities
- Correct structure of pension cashflows

Run tests with:

```bash
pytest
```

---

## Limitations

This is a stylised research framework:

- No calibration to real data  
- No transaction costs or liquidity constraints  
- Single representative DB member  
- Simplified leverage representation  

The focus is conceptual insight, not production realism.

---

## Disclaimer

This project is for **educational and research purposes only**.

