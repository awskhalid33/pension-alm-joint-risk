from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings
from src.io_utils import ensure_dirs_exist
from src.liabilities.db_liability import (
    DBLiabilitySpec,
    build_expected_cashflows,
    present_value_from_expected_cashflows,
)


def main() -> None:
    ensure_dirs_exist([settings.output_dir])

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
    pv = present_value_from_expected_cashflows(cf, spec.flat_discount_rate)

    print("DB Liability Spec:", spec)
    print("Present Value (flat 3%):", round(pv, 2))
    print()
    print("Cashflow preview:")
    print(cf.head(15).to_string(index=False))
    print()
    print("Cashflow tail:")
    print(cf.tail(10).to_string(index=False))

    # Plot expected cashflows over time
    plt.plot(cf["t"], cf["expected_cashflow"])
    plt.title("Expected DB pension cashflow (toy mortality)")
    plt.xlabel("t (years from valuation)")
    plt.ylabel("Expected cashflow (£)")
    out_path = settings.output_dir / "expected_db_cashflows_toy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
