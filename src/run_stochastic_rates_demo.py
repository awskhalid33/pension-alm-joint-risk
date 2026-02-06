from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import settings
from src.finance.stochastic_ns import StochasticNelsonSiegelModel
from src.io_utils import ensure_dirs_exist
from src.liabilities.db_liability import DBLiabilitySpec, build_expected_cashflows
from src.liabilities.valuation import pv_remaining_cashflows_at_time_k


def main() -> None:
    ensure_dirs_exist([settings.output_dir])

    # Load toy mortality and build expected CFs (deterministic for now)
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

    # Quarterly steps for 10 years
    dt = 0.25
    n_years = 10
    n_steps = int(n_years / dt)

    # VAR(1) parameters (toy but stable)
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

    # Start factors near a reasonable curve
    x0 = np.array([0.030, -0.010, 0.010], dtype=float)
    x_path = model.simulate_factors(x0=x0, n_steps=n_steps, seed=123)

    # Revalue liability each year (k = 0..10) for clarity
    years = list(range(0, n_years + 1))
    pv_series = []
    lvl_series = []
    for k in years:
        idx = int(k / dt)
        factors = model.factors_from_array(x_path[idx])

        pv_k = pv_remaining_cashflows_at_time_k(
            cf=cf,
            k=k,
            curve_model=model,
            curve_factors=factors,
        )

        pv_series.append(pv_k)
        lvl_series.append(factors.level)

    out = pd.DataFrame({"year": years, "pv_liability": pv_series, "level_factor": lvl_series})
    print(out.to_string(index=False))

    # Plot PV path
    plt.plot(out["year"], out["pv_liability"])
    plt.title("Liability PV over time under stochastic interest rates (toy)")
    plt.xlabel("Year")
    plt.ylabel("PV (£)")
    out_path = settings.output_dir / "liability_pv_stochastic_rates.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
