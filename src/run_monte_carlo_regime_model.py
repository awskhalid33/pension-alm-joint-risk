from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import settings
from src.finance.asset_valuation import ZeroCouponAsset
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.io_utils import ensure_dirs_exist
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k
from src.mortality.regime_improvement import RegimeMortalityModel
from src.regimes.markov import MarkovRegime


def simulate_one_path(
    *,
    df_mort: pd.DataFrame,
    spec: DBLiabilitySpec,
    n_years: int,
    hedge_maturity: float,
    P: np.ndarray,
    rate_models: dict[int, StochasticNelsonSiegelModel],
    mort_model: RegimeMortalityModel,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      funding_ratio_path (n_years+1,)
      regimes (n_years+1,)
      kappas (n_years+1,)
    """
    # separate RNG streams for reproducibility and to avoid accidental coupling
    rng_reg = np.random.default_rng(seed + 1)
    rng_rates = np.random.default_rng(seed + 2)

    # 1) regimes
    regime_proc = MarkovRegime(P)
    regimes = regime_proc.simulate(n_steps=n_years, s0=0, seed=seed + 10)

    # 2) rate factors path x_t
    x_path = np.zeros((n_years + 1, 3), dtype=float)
    x_path[0] = np.array([0.030, -0.010, 0.010], dtype=float)

    for t in range(n_years):
        s = int(regimes[t])
        m = rate_models[s]
        eps = rng_rates.multivariate_normal(mean=np.zeros(3), cov=m.Sigma)
        x_path[t + 1] = m.A @ x_path[t] + eps

    # 3) kappa path conditional on regimes
    kappas = mort_model.simulate(regimes=regimes, kappa0=0.0, seed=seed + 20)

    # 4) size hedge at t=0 so A0 = L0
    s0 = int(regimes[0])
    m0 = rate_models[s0]
    factors0 = m0.factors_from_array(x_path[0])

    cf0 = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[0]))
    L0 = pv_remaining_cashflows_at_time_k(cf=cf0, k=0, curve_model=m0, curve_factors=factors0)

    df_hedge0 = m0.discount_factor(factors0, hedge_maturity)
    notional = L0 / df_hedge0
    asset = ZeroCouponAsset(maturity=hedge_maturity, notional=float(notional))

    # 5) revalue through time
    fr = np.zeros(n_years + 1, dtype=float)
    for k in range(n_years + 1):
        s = int(regimes[k])
        mk = rate_models[s]
        factors_k = mk.factors_from_array(x_path[k])

        cf_k = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[k]))
        Lk = pv_remaining_cashflows_at_time_k(cf=cf_k, k=k, curve_model=mk, curve_factors=factors_k)

        Ak = asset.value(curve_model=mk, curve_factors=factors_k, time_elapsed=float(k))
        fr[k] = Ak / Lk if Lk > 0 else np.nan

    return fr, regimes, kappas


def main() -> None:
    ensure_dirs_exist([settings.output_dir])

    # -----------------------------
    # Core setup
    # -----------------------------
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

    # Regime transition matrix
    P = np.array([[0.90, 0.10], [0.20, 0.80]], dtype=float)

    # Rate models by regime
    A0 = np.array([[0.995, 0.000, 0.000], [0.000, 0.990, 0.000], [0.000, 0.000, 0.985]], dtype=float)
    Sigma0 = np.diag([0.00003, 0.00005, 0.00005])

    A1 = np.array([[0.990, 0.000, 0.000], [0.000, 0.985, 0.000], [0.000, 0.000, 0.975]], dtype=float)
    Sigma1 = np.diag([0.00010, 0.00012, 0.00012])

    rate_models = {
        0: StochasticNelsonSiegelModel(A=A0, Sigma=Sigma0, tau=2.5, dt_years=1.0),
        1: StochasticNelsonSiegelModel(A=A1, Sigma=Sigma1, tau=2.5, dt_years=1.0),
    }

    # Mortality model by regime
    mort_model = RegimeMortalityModel(
        mu={0: -0.012, 1: -0.004},
        sigma={0: 0.008, 1: 0.012},
    )

    # -----------------------------
    # Monte Carlo
    # -----------------------------
    n_paths = 1000  # start with 1000; later we can do 5000+
    fr_T = np.zeros(n_paths, dtype=float)
    fr_min = np.zeros(n_paths, dtype=float)
    stress_frac = np.zeros(n_paths, dtype=float)

    for i in range(n_paths):
        fr, regimes, kappas = simulate_one_path(
            df_mort=df_mort,
            spec=spec,
            n_years=n_years,
            hedge_maturity=hedge_maturity,
            P=P,
            rate_models=rate_models,
            mort_model=mort_model,
            seed=10_000 + i,
        )
        fr_T[i] = fr[-1]
        fr_min[i] = np.nanmin(fr)
        stress_frac[i] = float(np.mean(regimes == 1))

    # -----------------------------
    # Summary statistics
    # -----------------------------
    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    summary = {
        "n_paths": n_paths,
        "FR_T_mean": float(np.mean(fr_T)),
        "FR_T_p5": pct(fr_T, 5),
        "FR_T_p50": pct(fr_T, 50),
        "FR_T_p95": pct(fr_T, 95),
        "FR_min_mean": float(np.mean(fr_min)),
        "Pr(FR_T < 0.95)": float(np.mean(fr_T < 0.95)),
        "Pr(FR_T < 0.90)": float(np.mean(fr_T < 0.90)),
        "Pr(min FR < 0.95)": float(np.mean(fr_min < 0.95)),
        "Pr(min FR < 0.90)": float(np.mean(fr_min < 0.90)),
        "stress_frac_mean": float(np.mean(stress_frac)),
    }

    print("=== Monte Carlo Summary (Regime joint model) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:>18}: {v:.4f}")
        else:
            print(f"{k:>18}: {v}")

    # Save results
    out_df = pd.DataFrame(
        {
            "fr_T": fr_T,
            "fr_min": fr_min,
            "stress_frac": stress_frac,
        }
    )
    csv_path = settings.output_dir / "mc_regime_joint_results.csv"
    out_df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    # -----------------------------
    # Plots
    # -----------------------------
    # Histogram of FR at horizon
    plt.hist(fr_T, bins=40)
    plt.title("Funding ratio at year 10 (Regime joint model)")
    plt.xlabel("FR_T")
    plt.ylabel("Count")
    p1 = settings.output_dir / "hist_fr_T_regime_joint.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", p1)

    # Histogram of min FR
    plt.hist(fr_min, bins=40)
    plt.title("Minimum funding ratio over 10y (Regime joint model)")
    plt.xlabel("min FR")
    plt.ylabel("Count")
    p2 = settings.output_dir / "hist_fr_min_regime_joint.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", p2)


if __name__ == "__main__":
    main()
