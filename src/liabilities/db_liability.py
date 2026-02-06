from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.liabilities.survival import t_year_survival


@dataclass(frozen=True)
class DBLiabilitySpec:
    """
    Stylised single-cohort DB pension:

    - Member is age x0 at valuation date (base_year)
    - Retires at retirement_age (e.g. 65)
    - After retirement, pays an annual pension amount (in nominal terms for now)
      at the END of each year while alive (annuity-immediate).
    - We cap at max_age for practicality.

    Later we add:
    - inflation indexation
    - spouse benefits
    - improvements year-by-year
    """
    base_year: int = 2023
    x0: int = 55
    retirement_age: int = 65
    max_age: int = 110

    annual_pension: float = 10_000.0  # £ per year
    flat_discount_rate: float = 0.03  # flat yield for now


def build_expected_cashflows(
    df_mort: pd.DataFrame,
    spec: DBLiabilitySpec,
    kappa_shift: float = 0.0,
) -> pd.DataFrame:

    """
    Build expected pension cashflows by year.

    For payment at time t (in years from valuation):
      Expected CF_t = benefit * Pr(alive at payment time)
    where Pr(alive) uses period survival from base_year.

    Payment starts at retirement, end of each year.
    If x0=55 and retirement=65, first payment at t=10.
    """
    start_t = spec.retirement_age - spec.x0
    if start_t < 0:
        raise ValueError("retirement_age must be >= x0")

    # last payment time (cap by max_age)
    last_t = spec.max_age - spec.x0
    times = np.arange(0, last_t + 1)

    rows = []
    for t in times:
        year = spec.base_year + t

        if t < start_t:
            benefit = 0.0
        else:
            benefit = spec.annual_pension

        surv_prob = t_year_survival(
            df_mort=df_mort,
             base_year=spec.base_year,
             base_age=spec.x0,
            t=int(t),
            kappa_shift=kappa_shift,
        )


        expected_cf = benefit * surv_prob

        rows.append((t, year, benefit, surv_prob, expected_cf))

    return pd.DataFrame(
        rows,
        columns=["t", "year", "benefit", "survival_prob", "expected_cashflow"],
    )


def present_value_from_expected_cashflows(
    cf: pd.DataFrame,
    flat_discount_rate: float,
) -> float:
    """
    PV = sum_t expected_cashflow_t * v^t, where v = (1+i)^(-1)
    """
    i = float(flat_discount_rate)
    v = 1.0 / (1.0 + i)

    pv = 0.0
    for _, row in cf.iterrows():
        t = float(row["t"])
        pv += float(row["expected_cashflow"]) * (v ** t)

    return float(pv)
def present_value_from_curve(
    cf: pd.DataFrame,
    curve,
) -> float:
    """
    PV = sum_t expected_cashflow_t * P(0,t)

    curve must have method: discount_factor(t: float) -> float
    """
    pv = 0.0
    for _, row in cf.iterrows():
        t = float(row["t"])
        df_t = float(curve.discount_factor(t))
        pv += float(row["expected_cashflow"]) * df_t

    return float(pv)
