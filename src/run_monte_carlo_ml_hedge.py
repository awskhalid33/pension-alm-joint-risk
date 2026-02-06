from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import settings
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.io_utils import ensure_dirs_exist
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows
from src.liabilities.valuation import (
    pv_remaining_cashflows_at_time_k,
    duration_remaining_cashflows_at_time_k,
)
from src.mortality.regime_improvement import RegimeMortalityModel
from src.regimes.markov import MarkovRegime


def simulate_joint_paths(
    *,
    n_years: int,
    P: np.ndarray,
    rate_models: dict[int, StochasticNelsonSiegelModel],
    mort_model: RegimeMortalityModel,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    regime_proc = MarkovRegime(P)
    regimes = regime_proc.simulate(n_steps=n_years, s0=0, seed=seed + 10)

    rng_rates = np.random.default_rng(seed + 2)
    x_path = np.zeros((n_years + 1, 3), dtype=float)
    x_path[0] = np.array([0.030, -0.010, 0.010], dtype=float)

    for t in range(n_years):
        s = int(regimes[t])
        m = rate_models[s]
        eps = rng_rates.multivariate_normal(mean=np.zeros(3), cov=m.Sigma)
        x_path[t + 1] = m.A @ x_path[t] + eps

    kappas = mort_model.simulate(regimes=regimes, kappa0=0.0, seed=seed + 20)
    return regimes, x_path, kappas


def fit_linear_regression(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Closed-form ridge regression:
      beta = (X'X + ridge*I)^(-1) X'y
    Includes intercept if you add a column of ones to X.
    """
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])
    beta = np.linalg.solve(XtX + ridge * I, X.T @ y)
    return beta


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ beta


def simulate_static_fr(
    *,
    df_mort: pd.DataFrame,
    spec: DBLiabilitySpec,
    n_years: int,
    hedge_maturity: float,
    regimes: np.ndarray,
    x_path: np.ndarray,
    kappas: np.ndarray,
    rate_models: dict[int, StochasticNelsonSiegelModel],
) -> np.ndarray:
    s0 = int(regimes[0])
    m0 = rate_models[s0]
    f0 = m0.factors_from_array(x_path[0])

    cf0 = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[0]))
    L0 = pv_remaining_cashflows_at_time_k(cf=cf0, k=0, curve_model=m0, curve_factors=f0)

    df0 = m0.discount_factor(f0, hedge_maturity)
    notional = L0 / df0

    fr = np.zeros(n_years + 1, dtype=float)
    for k in range(n_years + 1):
        s = int(regimes[k])
        mk = rate_models[s]
        fk = mk.factors_from_array(x_path[k])

        cfk = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[k]))
        Lk = pv_remaining_cashflows_at_time_k(cf=cfk, k=k, curve_model=mk, curve_factors=fk)

        ttm = hedge_maturity - float(k)
        bond_val = float(notional) if ttm <= 0 else float(notional) * mk.discount_factor(fk, ttm)

        fr[k] = bond_val / Lk if Lk > 0 else np.nan

    return fr


def simulate_ml_duration_hedge_fr(
    *,
    df_mort: pd.DataFrame,
    spec: DBLiabilitySpec,
    n_years: int,
    hedge_maturity: float,
    regimes: np.ndarray,
    x_path: np.ndarray,
    kappas: np.ndarray,
    rate_models: dict[int, StochasticNelsonSiegelModel],
    beta: np.ndarray,
) -> np.ndarray:
    """
    Self-financing bond + cash strategy:
    - Each year we predict liability duration from observables.
    - Set hedge ratio h_k = D_hat / D_bond, so bond value target = h_k * Lk.
    - Cash is residual (can be negative -> leverage).
    - Cash accrues at short rate approx (level+slope).
    """
    # time 0
    s0 = int(regimes[0])
    m0 = rate_models[s0]
    f0 = m0.factors_from_array(x_path[0])

    cf0 = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[0]))
    L0 = pv_remaining_cashflows_at_time_k(cf=cf0, k=0, curve_model=m0, curve_factors=f0)

    total_assets = float(L0)

    # features at k=0
    kappa0 = float(kappas[0])
    dk0 = 0.0
    X0 = np.array([1.0, f0.level, f0.slope, f0.curvature, kappa0, dk0], dtype=float)
    D_hat0 = float(predict(beta, X0.reshape(1, -1))[0])

    ttm0 = hedge_maturity - 0.0
    D_bond0 = max(ttm0, 0.5)
    h0 = float(np.clip(D_hat0 / D_bond0, 0.5, 1.5))

    price0 = m0.discount_factor(f0, ttm0)
    bond_value_target0 = h0 * L0
    notional = bond_value_target0 / price0
    cash = total_assets - bond_value_target0

    fr = np.zeros(n_years + 1, dtype=float)
    fr[0] = total_assets / L0

    for k in range(1, n_years + 1):
        # accrue cash at short rate from k-1 to k
        s_prev = int(regimes[k - 1])
        m_prev = rate_models[s_prev]
        f_prev = m_prev.factors_from_array(x_path[k - 1])
        r_short = float(m_prev.zero_rate(f_prev, 0.0))
        cash *= float(np.exp(r_short * 1.0))

        # current state
        s = int(regimes[k])
        mk = rate_models[s]
        fk = mk.factors_from_array(x_path[k])

        # revalue bond before rebalancing
        ttm = hedge_maturity - float(k)
        price_k = 1.0 if ttm <= 0 else float(mk.discount_factor(fk, ttm))
        bond_val = float(notional) if ttm <= 0 else float(notional) * price_k
        total_assets = bond_val + cash

        # liabilities at time k
        cfk = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[k]))
        Lk = pv_remaining_cashflows_at_time_k(cf=cfk, k=k, curve_model=mk, curve_factors=fk)

        # ML-predicted duration
        kappa_k = float(kappas[k])
        dk = float(kappas[k] - kappas[k - 1])
        Xk = np.array([1.0, fk.level, fk.slope, fk.curvature, kappa_k, dk], dtype=float)
        D_hat = float(predict(beta, Xk.reshape(1, -1))[0])

        D_bond = max(ttm, 0.5)
        h = float(np.clip(D_hat / D_bond, 0.5, 1.5))

        # rebalance: target bond value = h * Lk
        bond_value_target = h * Lk
        notional_target = bond_value_target / price_k if price_k > 0 else notional

        delta_notional = notional_target - notional
        trade_cost = float(delta_notional) * float(price_k)

        cash -= trade_cost
        notional = notional_target

        total_assets = bond_value_target + cash
        fr[k] = total_assets / Lk if Lk > 0 else np.nan

    return fr


def main() -> None:
    ensure_dirs_exist([settings.output_dir])

    df_mort = pd.read_csv(settings.processed_dir / "toy_mortality_uk.csv")

    spec = DBLiabilitySpec(
        base_year=2023,
        x0=55,
        retirement_age=65,
        max_age=110,
        annual_pension=10_000.0,
        flat_discount_rate=0.03,
    )

    n_years = 10
    hedge_maturity = 25.0

    P = np.array([[0.90, 0.10], [0.20, 0.80]], dtype=float)

    A0 = np.array([[0.995, 0.000, 0.000], [0.000, 0.990, 0.000], [0.000, 0.000, 0.985]], dtype=float)
    Sigma0 = np.diag([0.00003, 0.00005, 0.00005])

    A1 = np.array([[0.990, 0.000, 0.000], [0.000, 0.985, 0.000], [0.000, 0.000, 0.975]], dtype=float)
    Sigma1 = np.diag([0.00010, 0.00012, 0.00012])

    rate_models = {
        0: StochasticNelsonSiegelModel(A=A0, Sigma=Sigma0, tau=2.5, dt_years=1.0),
        1: StochasticNelsonSiegelModel(A=A1, Sigma=Sigma1, tau=2.5, dt_years=1.0),
    }

 


    mort_model = RegimeMortalityModel(
        mu={0: -0.012, 1: -0.004},
        sigma={0: 0.008, 1: 0.012},
    )

    # -----------------------------
    # 1) Build training set: predict true liability duration from observables
    # Features: [1, level, slope, curvature, kappa, delta_kappa]
    # Label: true duration of remaining liability cashflows at time k
    # -----------------------------
    n_train_paths = 400
    X_rows = []
    y_rows = []

    for i in range(n_train_paths):
        regimes, x_path, kappas = simulate_joint_paths(
            n_years=n_years,
            P=P,
            rate_models=rate_models,
            mort_model=mort_model,
            seed=30_000 + i,
        )

        for k in range(n_years + 1):
            s = int(regimes[k])
            mk = rate_models[s]
            fk = mk.factors_from_array(x_path[k])

            cfk = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[k]))
            D_true = duration_remaining_cashflows_at_time_k(
                cf=cfk, k=k, curve_model=mk, curve_factors=fk
            )

            kappa_k = float(kappas[k])
            dk = float(kappas[k] - kappas[k - 1]) if k > 0 else 0.0

            X_rows.append([1.0, fk.level, fk.slope, fk.curvature, kappa_k, dk])
            y_rows.append(D_true)

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=float)

    beta = fit_linear_regression(X, y, ridge=1e-6)
    print("Trained linear model coefficients (beta):")
    print(beta)

    # -----------------------------
    # 2) Test Monte Carlo: static vs ML-duration hedge
    # -----------------------------
    n_paths = 1000
    frT_static = np.zeros(n_paths, dtype=float)
    frmin_static = np.zeros(n_paths, dtype=float)

    frT_ml = np.zeros(n_paths, dtype=float)
    frmin_ml = np.zeros(n_paths, dtype=float)

    for i in range(n_paths):
        regimes, x_path, kappas = simulate_joint_paths(
            n_years=n_years,
            P=P,
            rate_models=rate_models,
            mort_model=mort_model,
            seed=40_000 + i,
        )

        fr_s = simulate_static_fr(
            df_mort=df_mort,
            spec=spec,
            n_years=n_years,
            hedge_maturity=hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            rate_models=rate_models,
        )

        fr_m = simulate_ml_duration_hedge_fr(
            df_mort=df_mort,
            spec=spec,
            n_years=n_years,
            hedge_maturity=hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            rate_models=rate_models,
            beta=beta,
        )

        frT_static[i] = fr_s[-1]
        frmin_static[i] = np.nanmin(fr_s)

        frT_ml[i] = fr_m[-1]
        frmin_ml[i] = np.nanmin(fr_m)

    def summarize(x: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(x)),
            "p5": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "Pr(<0.95)": float(np.mean(x < 0.95)),
            "Pr(<0.90)": float(np.mean(x < 0.90)),
        }

    print("\n=== Static hedge: FR at year 10 ===")
    for k, v in summarize(frT_static).items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== ML-duration hedge: FR at year 10 ===")
    for k, v in summarize(frT_ml).items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Static hedge: min FR over 10y ===")
    for k, v in summarize(frmin_static).items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== ML-duration hedge: min FR over 10y ===")
    for k, v in summarize(frmin_ml).items():
        print(f"{k:>10}: {v:.4f}")

    # Plots
    plt.hist(frmin_static, bins=40, alpha=0.6, label="static")
    plt.hist(frmin_ml, bins=40, alpha=0.6, label="ML-duration")
    plt.title("Minimum funding ratio over 10y: static vs ML-duration hedge")
    plt.xlabel("min FR")
    plt.ylabel("Count")
    plt.legend()
    p1 = settings.output_dir / "compare_hist_frmin_static_vs_ml_duration.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved plot:", p1)


if __name__ == "__main__":
    main()
