from __future__ import annotations

import numpy as np
import pandas as pd


def macaulay_duration_from_curve(
    cf: pd.DataFrame,
    curve,
) -> float:
    """
    Macaulay duration of expected cashflows under a given curve.

    D = sum_t [ t * CF_t * P(0,t) ] / PV
    """
    pv = 0.0
    weighted_sum = 0.0

    for _, row in cf.iterrows():
        t = float(row["t"])
        cf_t = float(row["expected_cashflow"])
        df_t = float(curve.discount_factor(t))

        pv += cf_t * df_t
        weighted_sum += t * cf_t * df_t

    if pv <= 0.0:
        raise ValueError("PV must be positive to compute duration")

    return float(weighted_sum / pv)


def modified_duration_from_curve(
    cf: pd.DataFrame,
    curve,
    shift: float = 1e-4,
) -> float:
    """
    Approximate modified duration via finite difference:
      D_mod ≈ - (PV_up - PV_down) / (2 * PV * shift)

    where rates are shifted in parallel by 'shift'.
    """
    # Base PV
    pv0 = 0.0
    for _, row in cf.iterrows():
        t = float(row["t"])
        pv0 += float(row["expected_cashflow"]) * curve.discount_factor(t)

    # Shift curve up
    class ShiftedCurve:
        def discount_factor(self, t):
            z = curve.zero_rate(t) + shift
            return float(np.exp(-z * t))

    pv_up = 0.0
    pv_down = 0.0
    for _, row in cf.iterrows():
        t = float(row["t"])
        cf_t = float(row["expected_cashflow"])
        pv_up += cf_t * ShiftedCurve().discount_factor(t)
        pv_down += cf_t * np.exp(-(curve.zero_rate(t) - shift) * t)

    return float(- (pv_up - pv_down) / (2.0 * pv0 * shift))
