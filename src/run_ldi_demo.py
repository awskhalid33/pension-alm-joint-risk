from __future__ import annotations

import pandas as pd

from src.config import settings
from src.finance.bonds import ZeroCouponBond
from src.finance.duration import macaulay_duration_from_curve
from src.finance.yield_curve import NelsonSiegelCurve
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows


def main() -> None:
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

    curve = NelsonSiegelCurve()
    D_L = macaulay_duration_from_curve(cf, curve)

    # Asset: single zero-coupon bond
    bond = ZeroCouponBond(maturity=25.0)
    D_A = bond.duration()

    print("Liability duration:", round(D_L, 2))
    print("Asset duration (zero):", round(D_A, 2))
    print("Duration mismatch:", round(D_A - D_L, 2))


if __name__ == "__main__":
    main()
