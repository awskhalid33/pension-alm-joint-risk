from __future__ import annotations

import pandas as pd


def pv_remaining_cashflows_at_time_k(
    cf: pd.DataFrame,
    k: int,
    curve_model,
    curve_factors,
) -> float:
    """
    Value at time step k (i.e., at t=k years from original valuation),
    of remaining expected cashflows.

    We treat cf['t'] as years from original valuation.
    At time k, a cashflow at time t has time-to-maturity = (t - k).

    curve_model must provide discount_factor(factors, ttm).
    """
    pv = 0.0
    for _, row in cf.iterrows():
        t = float(row["t"])
        if t < k:
            continue

        ttm = t - float(k)
        if ttm <= 0.0:
            # cashflow at "now" (rare here)
            df = 1.0
        else:
            df = float(curve_model.discount_factor(curve_factors, ttm))

        pv += float(row["expected_cashflow"]) * df

    return float(pv)
def duration_remaining_cashflows_at_time_k(
    cf: pd.DataFrame,
    k: int,
    curve_model,
    curve_factors,
) -> float:
    """
    Macaulay duration (in years from time k) of remaining expected cashflows.

    D(k) = sum_{t>=k} (ttm * CF_t * P_k(ttm)) / PV_k
    where ttm = t - k.
    """
    pv = 0.0
    weighted = 0.0

    for _, row in cf.iterrows():
        t = float(row["t"])
        if t < k:
            continue

        ttm = t - float(k)
        if ttm <= 0.0:
            df = 1.0
        else:
            df = float(curve_model.discount_factor(curve_factors, ttm))

        cf_t = float(row["expected_cashflow"])
        pv += cf_t * df
        weighted += ttm * cf_t * df

    if pv <= 0.0:
        return float("nan")

    return float(weighted / pv)
