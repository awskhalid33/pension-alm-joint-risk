from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import settings
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.io_utils import ensure_dirs_exist
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k
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
    """
    Simulate:
      regimes (n_years+1,)
      x_path (n_years+1,3)
      kappas (n_years+1,)
    """
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


def short_rate(m: StochasticNelsonSiegelModel, factors) -> float:
    """
    Approximate short rate using ttm -> 0 limit of NS:
      z(0+) = level + slope
    Our zero_rate handles ttm<=0 by returning level+slope.
    """
    return float(m.zero_rate(factors, 0.0))


def simulate_static_hedge_path(
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
    """
    Static hedge: bond notional fixed from time 0 (plus no cash account).
    """
    s0 = int(regimes[0])
    m0 = rate_models[s0]
    f0 = m0.factors_from_array(x_path[0])

    cf0 = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[0]))
    L0 = pv_remaining_cashflows_at_time_k(cf=cf0, k=0, curve_model=m0, curve_factors=f0)

    df0 = m0.discount_factor(f0, hedge_maturity)
    notional = L0 / df0  # so A0 = L0

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


def simulate_dynamic_hedge_path(
    *,
    df_mort: pd.DataFrame,
    spec: DBLiabilitySpec,
    n_years: int,
    hedge_maturity: float,
    regimes: np.ndarray,
    x_path: np.ndarray,
    kappas: np.ndarray,
    rate_models: dict[int, StochasticNelsonSiegelModel],
    h_normal: float = 1.00,
    h_stress: float = 1.10,
) -> np.ndarray:
    """
    Dynamic hedge with cash account (can be negative):
    - Total assets evolve through market moves and cash accrual.
    - Each year, choose bond exposure so that bond_value = h(regime) * Lk.
      The residual goes to cash (possibly negative).
    This changes interest-rate sensitivity conditionally on regime.
    """
    # time 0: fully funded
    s0 = int(regimes[0])
    m0 = rate_models[s0]
    f0 = m0.factors_from_array(x_path[0])

    cf0 = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[0]))
    L0 = pv_remaining_cashflows_at_time_k(cf=cf0, k=0, curve_model=m0, curve_factors=f0)

    total_assets = float(L0)

    # initial hedge ratio
    h0 = h_stress if regimes[0] == 1 else h_normal

    price0 = m0.discount_factor(f0, hedge_maturity)
    bond_value_target0 = h0 * L0
    notional = bond_value_target0 / price0
    cash = total_assets - bond_value_target0  # can be negative (leverage)

    fr = np.zeros(n_years + 1, dtype=float)
    fr[0] = total_assets / L0

    for k in range(1, n_years + 1):
        # accrue cash over the year k-1 -> k using previous year's short rate
        s_prev = int(regimes[k - 1])
        m_prev = rate_models[s_prev]
        f_prev = m_prev.factors_from_array(x_path[k - 1])
        r = short_rate(m_prev, f_prev)
        cash *= float(np.exp(r * 1.0))

        # revalue bond at time k under current curve
        s = int(regimes[k])
        mk = rate_models[s]
        fk = mk.factors_from_array(x_path[k])

        ttm = hedge_maturity - float(k)
        price_k = 1.0 if ttm <= 0 else mk.discount_factor(fk, ttm)
        bond_val = float(notional) if ttm <= 0 else float(notional) * float(price_k)

        total_assets = bond_val + cash

        # compute liability at time k
        cfk = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[k]))
        Lk = pv_remaining_cashflows_at_time_k(cf=cfk, k=k, curve_model=mk, curve_factors=fk)

        # choose hedge ratio based on regime
        hk = h_stress if regimes[k] == 1 else h_normal

        # rebalance (self-financing): change bond notional, offsetting cash
        bond_value_target = hk * Lk
        notional_target = bond_value_target / float(price_k) if price_k > 0 else notional

        # trading cost at current prices
        delta_notional = notional_target - notional
        trade_cost = float(delta_notional) * float(price_k)

        cash -= trade_cost
        notional = notional_target

        # after rebalance
        bond_val = bond_value_target
        total_assets = bond_val + cash

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

    n_paths = 1000

    frT_static = np.zeros(n_paths, dtype=float)
    frmin_static = np.zeros(n_paths, dtype=float)

    frT_dyn = np.zeros(n_paths, dtype=float)
    frmin_dyn = np.zeros(n_paths, dtype=float)

    for i in range(n_paths):
        regimes, x_path, kappas = simulate_joint_paths(
            n_years=n_years,
            P=P,
            rate_models=rate_models,
            mort_model=mort_model,
            seed=20_000 + i,
        )

        fr_s = simulate_static_hedge_path(
            df_mort=df_mort,
            spec=spec,
            n_years=n_years,
            hedge_maturity=hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            rate_models=rate_models,
        )
        fr_d = simulate_dynamic_hedge_path(
            df_mort=df_mort,
            spec=spec,
            n_years=n_years,
            hedge_maturity=hedge_maturity,
            regimes=regimes,
            x_path=x_path,
            kappas=kappas,
            rate_models=rate_models,
            h_normal=1.00,
            h_stress=1.10,
        )

        frT_static[i] = fr_s[-1]
        frmin_static[i] = np.nanmin(fr_s)

        frT_dyn[i] = fr_d[-1]
        frmin_dyn[i] = np.nanmin(fr_d)

    def summarize(x: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(x)),
            "p5": float(np.percentile(x, 5)),
            "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "Pr(<0.95)": float(np.mean(x < 0.95)),
            "Pr(<0.90)": float(np.mean(x < 0.90)),
        }

    print("=== Static hedge: FR at year 10 ===")
    for k, v in summarize(frT_static).items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Dynamic hedge: FR at year 10 ===")
    for k, v in summarize(frT_dyn).items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Static hedge: min FR over 10y ===")
    for k, v in summarize(frmin_static).items():
        print(f"{k:>10}: {v:.4f}")

    print("\n=== Dynamic hedge: min FR over 10y ===")
    for k, v in summarize(frmin_dyn).items():
        print(f"{k:>10}: {v:.4f}")

    # Plots: compare histograms
    plt.hist(frT_static, bins=40, alpha=0.6, label="static")
    plt.hist(frT_dyn, bins=40, alpha=0.6, label="dynamic")
    plt.title("Funding ratio at year 10: static vs dynamic hedge")
    plt.xlabel("FR_T")
    plt.ylabel("Count")
    plt.legend()
    p1 = settings.output_dir / "compare_hist_frT_static_vs_dynamic.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved plot:", p1)

    plt.hist(frmin_static, bins=40, alpha=0.6, label="static")
    plt.hist(frmin_dyn, bins=40, alpha=0.6, label="dynamic")
    plt.title("Minimum funding ratio over 10y: static vs dynamic hedge")
    plt.xlabel("min FR")
    plt.ylabel("Count")
    plt.legend()
    p2 = settings.output_dir / "compare_hist_frmin_static_vs_dynamic.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", p2)


if __name__ == "__main__":
    main()
