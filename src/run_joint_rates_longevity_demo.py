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
from src.mortality.improvement import MortalityImprovementRW


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

    # --- stochastic rates (same as before) ---
    dt = 0.25
    n_years = 10
    n_steps = int(n_years / dt)

    A = np.array(
        [
            [0.995, 0.000, 0.000],
            [0.000, 0.990, 0.000],
            [0.000, 0.000, 0.985],
        ],
        dtype=float,
    )
    Sigma = np.diag([0.00005, 0.00008, 0.00008])
    model = StochasticNelsonSiegelModel(A=A, Sigma=Sigma, tau=2.5, dt_years=dt)

    x0 = np.array([0.030, -0.010, 0.010], dtype=float)
    x_path = model.simulate_factors(x0=x0, n_steps=n_steps, seed=123)

    # --- stochastic mortality improvement ---
    mort_model = MortalityImprovementRW(mu=-0.01, sigma=0.01)
    kappas = mort_model.simulate(n_years=n_years, seed=999, kappa0=0.0)

    # Build initial CF with kappa0
    cf0 = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[0]))

    # Liability PV at time 0
    factors0 = model.factors_from_array(x_path[0])
    L0 = pv_remaining_cashflows_at_time_k(cf=cf0, k=0, curve_model=model, curve_factors=factors0)

    # Hedge asset (correct notional so A0 = L0)
    hedge_maturity = 25.0
    df_hedge0 = model.discount_factor(factors0, hedge_maturity)
    notional = L0 / df_hedge0

    asset = ZeroCouponAsset(maturity=hedge_maturity, notional=notional)

    years = list(range(0, n_years + 1))
    records = []

    for k in years:
        # curve factors at time k
        idx = int(k / dt)
        factors = model.factors_from_array(x_path[idx])

        # rebuild expected CF using updated kappa at year k
        cf_k = build_expected_cashflows(df_mort, spec, kappa_shift=float(kappas[k]))

        # value liabilities and assets at time k
        L = pv_remaining_cashflows_at_time_k(cf=cf_k, k=k, curve_model=model, curve_factors=factors)
        A_val = asset.value(curve_model=model, curve_factors=factors, time_elapsed=k)

        FR = A_val / L if L > 0 else float("nan")
        records.append((k, kappas[k], A_val, L, FR))

    df = pd.DataFrame(records, columns=["year", "kappa", "asset", "liability", "funding_ratio"])
    print(df.to_string(index=False))

    # plot funding ratio and kappa
    plt.plot(df["year"], df["funding_ratio"])
    plt.axhline(1.0)
    plt.title("Funding ratio with stochastic rates + longevity (static LDI)")
    plt.xlabel("Year")
    plt.ylabel("Funding ratio")
    out1 = settings.output_dir / "funding_ratio_joint_rates_longevity.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", out1)

    plt.plot(df["year"], df["kappa"])
    plt.axhline(0.0)
    plt.title("Mortality improvement index kappa (negative = improving)")
    plt.xlabel("Year")
    plt.ylabel("kappa")
    out2 = settings.output_dir / "kappa_path.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", out2)


if __name__ == "__main__":
    main()
