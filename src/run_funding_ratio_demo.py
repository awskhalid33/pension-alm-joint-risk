from __future__ import annotations
from xml.parsers.expat import model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import settings
from src.finance.asset_valuation import ZeroCouponAsset
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.io_utils import ensure_dirs_exist
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k


def main() -> None:
    ensure_dirs_exist([settings.output_dir])

    # Load mortality
    df_mort = pd.read_csv(settings.processed_dir / "toy_mortality_uk.csv")

    # Liability
    spec = DBLiabilitySpec(
        base_year=2023,
        x0=55,
        retirement_age=65,
        max_age=110,
        annual_pension=10_000.0,
        flat_discount_rate=0.03,
    )
    cf = build_expected_cashflows(df_mort, spec)

    # Stochastic curve model
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

    # Asset: duration-matched zero-coupon hedge
    hedge_maturity = 25.0
    initial_liability_pv = pv_remaining_cashflows_at_time_k(
        cf=cf,
        k=0,
        curve_model=model,
        curve_factors=model.factors_from_array(x_path[0]),
    )

# Correct notional so asset PV = liability PV at t=0
    factors0 = model.factors_from_array(x_path[0])
    df_hedge = model.discount_factor(factors0, hedge_maturity)

    correct_notional = initial_liability_pv / df_hedge

    asset = ZeroCouponAsset(
        maturity=hedge_maturity,
        notional=correct_notional,
)


    years = list(range(0, n_years + 1))
    records = []

    for k in years:
        idx = int(k / dt)
        factors = model.factors_from_array(x_path[idx])

        L = pv_remaining_cashflows_at_time_k(
            cf=cf,
            k=k,
            curve_model=model,
            curve_factors=factors,
        )

        A_val = asset.value(
            curve_model=model,
            curve_factors=factors,
            time_elapsed=k,
        )

        FR = A_val / L if L > 0 else float("nan")
        records.append((k, A_val, L, FR))

    df = pd.DataFrame(records, columns=["year", "asset", "liability", "funding_ratio"])
    print(df.to_string(index=False))

    # Plot funding ratio
    plt.plot(df["year"], df["funding_ratio"])
    plt.axhline(1.0)
    plt.title("Funding ratio under stochastic rates (LDI hedge)")
    plt.xlabel("Year")
    plt.ylabel("Funding ratio")
    out_path = settings.output_dir / "funding_ratio_stochastic_rates.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
