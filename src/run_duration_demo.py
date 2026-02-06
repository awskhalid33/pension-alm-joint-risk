from __future__ import annotations

import pandas as pd

from src.config import settings
from src.finance.duration import (
    macaulay_duration_from_curve,
    modified_duration_from_curve,
)
from src.finance.yield_curve import NelsonSiegelCurve
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows


def main() -> None:
    df_mort = pd.read_csv(settings.processed_dir / "toy_mortality_uk.csv")

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

    D_mac = macaulay_duration_from_curve(cf, curve)
    D_mod = modified_duration_from_curve(cf, curve)

    print("Macaulay duration (years):", round(D_mac, 2))
    print("Modified duration:", round(D_mod, 2))


if __name__ == "__main__":
    main()
