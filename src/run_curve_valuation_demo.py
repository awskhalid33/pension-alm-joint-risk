from __future__ import annotations

import pandas as pd

from src.config import settings
from src.finance.yield_curve import NelsonSiegelCurve
from src.liabilities.db_liability import (
    DBLiabilitySpec,
    build_expected_cashflows,
    present_value_from_curve,
    present_value_from_expected_cashflows,
)


def main() -> None:
    mort_path = settings.processed_dir / "toy_mortality_uk.csv"
    df_mort = pd.read_csv(mort_path)

    spec = DBLiabilitySpec(
        base_year=2023,
        x0=55,
        retirement_age=65,
        max_age=110,
        annual_pension=10_000.0,
        flat_discount_rate=0.03,
    )
    cf = build_expected_cashflows(df_mort, spec)

    # Flat rate PV (old way)
    pv_flat = present_value_from_expected_cashflows(cf, spec.flat_discount_rate)

    # Curve PV (new way)
    curve = NelsonSiegelCurve(
        beta0=0.03,   # long-run level
        beta1=-0.01,  # slope (short end lower/higher depending sign)
        beta2=0.01,   # curvature
        tau=2.5,
    )
    pv_curve = present_value_from_curve(cf, curve)

    print("PV (flat 3%):", round(pv_flat, 2))
    print("PV (Nelson–Siegel curve):", round(pv_curve, 2))

    # Print a few discount factors for intuition
    for t in [1, 5, 10, 20, 30, 40, 50]:
        print(f"t={t:>2}  z(t)={curve.zero_rate(t):.4f}  P(0,t)={curve.discount_factor(t):.6f}")


if __name__ == "__main__":
    main()
