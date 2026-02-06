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


def main() -> None:
    ensure_dirs_exist([settings.output_dir])

    # -----------------------------
    # 1) Inputs: mortality + liability spec
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

    # -----------------------------
    # 2) Regime process (2-state Markov chain)
    # State 0 = "normal"
    # State 1 = "stress"
    # -----------------------------
    P = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
        ],
        dtype=float,
    )
    regime_proc = MarkovRegime(P)

    n_years = 10
    regimes = regime_proc.simulate(n_steps=n_years, s0=0, seed=42)

    # -----------------------------
    # 3) Regime-dependent interest rate dynamics (NS factors VAR)
    # We'll keep NS tau constant, but make volatility different by regime.
    # -----------------------------
    # Stable (normal) regime
    A0 = np.array(
        [
            [0.995, 0.000, 0.000],
            [0.000, 0.990, 0.000],
            [0.000, 0.000, 0.985],
        ],
        dtype=float,
    )
    Sigma0 = np.diag([0.00003, 0.00005, 0.00005])

    # Stress regime: bigger shocks, slightly less persistence
    A1 = np.array(
        [
            [0.990, 0.000, 0.000],
            [0.000, 0.985, 0.000],
            [0.000, 0.000, 0.975],
        ],
        dtype=float,
    )
    Sigma1 = np.diag([0.00010, 0.00012, 0.00012])

    model0 = StochasticNelsonSiegelModel(A=A0, Sigma=Sigma0, tau=2.5, dt_years=1.0)
    model1 = StochasticNelsonSiegelModel(A=A1, Sigma=Sigma1, tau=2.5, dt_years=1.0)
    models = {0: model0, 1: model1}

    # Simulate NS factors given regimes
    rng = np.random.default_rng(123)
    x_path = np.zeros((n_years + 1, 3), dtype=float)
    x_path[0] = np.array([0.030, -0.010, 0.010], dtype=float)

    for t in range(n_years):
        s = int(regimes[t])
        m = models[s]
        eps = rng.multivariate_normal(mean=np.zeros(3), cov=m.Sigma)
        x_path[t + 1] = m.A @ x_path[t] + eps

    # -----------------------------
    # 4) Regime-dependent mortality improvement (kappa)
    # Negative drift = improvement (longevity increases)
    # Make improvement faster (more negative) in "normal" than "stress"
    # (You can swap this later; this is just a demo design choice.)
    # -----------------------------
    mort_model = RegimeMortalityModel(
        mu={0: -0.012, 1: -0.004},
        sigma={0: 0.008, 1: 0.012},
    )
    kappas = mort_model.simulate(regimes=regimes, kappa0=0.0, seed=999)

    # -----------------------------
    # 5) Build hedge asset (zero-coupon) sized so A0 = L0 at time 0
    # -----------------------------
    hedge_maturity = 25.0

    s0 = int(regimes[0])
    m0 = models[s0]
    factors0 = m0.factors_from_array(x_path[0])

    cf0 = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[0]))
    L0 = pv_remaining_cashflows_at_time_k(cf=cf0, k=0, curve_model=m0, curve_factors=factors0)

    df_hedge0 = m0.discount_factor(factors0, hedge_maturity)
    notional = L0 / df_hedge0

    asset = ZeroCouponAsset(maturity=hedge_maturity, notional=float(notional))

    # -----------------------------
    # 6) Revalue over time
    # -----------------------------
    records = []
    for k in range(0, n_years + 1):
        s = int(regimes[k])
        mk = models[s]
        factors_k = mk.factors_from_array(x_path[k])

        cf_k = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[k]))
        Lk = pv_remaining_cashflows_at_time_k(cf=cf_k, k=k, curve_model=mk, curve_factors=factors_k)

        Ak = asset.value(curve_model=mk, curve_factors=factors_k, time_elapsed=float(k))
        FR = Ak / Lk if Lk > 0 else float("nan")

        records.append(
            (
                k,
                s,
                kappas[k],
                factors_k.level,
                Ak,
                Lk,
                FR,
            )
        )

    out = pd.DataFrame(
        records,
        columns=["year", "regime", "kappa", "level", "asset", "liability", "funding_ratio"],
    )
    print(out.to_string(index=False))

    # -----------------------------
    # 7) Plots
    # -----------------------------
    # Funding ratio
    plt.plot(out["year"], out["funding_ratio"])
    plt.axhline(1.0)
    plt.title("Funding ratio under regimes (rates + longevity, static hedge)")
    plt.xlabel("Year")
    plt.ylabel("Funding ratio")
    p1 = settings.output_dir / "funding_ratio_regime_joint.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # Regime path
    plt.step(out["year"], out["regime"], where="post")
    plt.title("Regime path (0=normal, 1=stress)")
    plt.xlabel("Year")
    plt.ylabel("Regime")
    p2 = settings.output_dir / "regime_path.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()

    # Kappa path
    plt.plot(out["year"], out["kappa"])
    plt.axhline(0.0)
    plt.title("Kappa path (negative = improving mortality)")
    plt.xlabel("Year")
    plt.ylabel("kappa")
    p3 = settings.output_dir / "kappa_path_regime.png"
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", p1)
    print("Saved:", p2)
    print("Saved:", p3)


if __name__ == "__main__":
    main()
