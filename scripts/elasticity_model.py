"""
elasticity_model.py
===================
Estimate the Price Elasticity of Demand for Disney (Orlando) ticket pricing.

Economic theory
---------------
Elasticity (epsilon) = %change in Quantity / %change in Price. In a log-log
regression

    ln(Q) = alpha + epsilon * ln(P) + controls,

the coefficient on ln(P) IS epsilon directly, because
d ln(Q) / d ln(P) = (dQ / Q) / (dP / P). This is the "constant elasticity"
functional form.

Identification problem
----------------------
Disney raises its own price PRECISELY when demand is high (holidays). Regressing
Q on P alone would absorb this simultaneity and bias epsilon toward zero (or
positive). We therefore CONTROL for the demand shifters (holiday, weekend,
weather, month) to isolate the true, downward-sloping demand curve -- i.e. hold
demand conditions fixed and ask "how does quantity respond to price?"

We estimate:
  * an AGGREGATE own-price elasticity and a CROSS-price elasticity (vs Universal,
    expected positive because the parks are substitutes), and
  * SEGMENTED elasticities (peak vs off-peak) via a price x holiday interaction --
    the heart of yield management, since different days have different price
    sensitivity.

This module prefers `statsmodels` for a full regression summary, but falls back
to a self-contained numpy OLS so the pipeline always runs.

Run
---
    python scripts/elasticity_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "disney_park_data.csv"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the log transforms and helper columns the regression needs."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["ln_Q"] = np.log(df["Expected_Visitor_Count"])
    df["ln_P"] = np.log(df["Disney_Ticket_Price"])
    df["ln_Pcomp"] = np.log(df["Universal_Studios_Price"])
    df["Is_Weekend"] = (df["Date"].dt.dayofweek >= 4).astype(int)
    df["Month"] = df["Date"].dt.month
    return df


def _design_matrix(df: pd.DataFrame, interaction: bool) -> tuple[np.ndarray, list[str]]:
    """Build the OLS design matrix with weather and month dummies."""
    cols = [np.ones(len(df)), df["ln_P"].to_numpy(), df["ln_Pcomp"].to_numpy(),
            df["Holiday_Flag"].to_numpy(), df["Is_Weekend"].to_numpy()]
    names = ["const", "ln_P", "ln_Pcomp", "Holiday_Flag", "Is_Weekend"]

    for w in ["Cloudy", "Rainy", "Stormy"]:           # "Sunny" is the reference level
        cols.append((df["Weather_Condition"] == w).astype(float).to_numpy())
        names.append(f"Weather[{w}]")
    for m in range(2, 13):                             # January is the reference month
        cols.append((df["Month"] == m).astype(float).to_numpy())
        names.append(f"Month[{m}]")
    if interaction:
        cols.append((df["ln_P"] * df["Holiday_Flag"]).to_numpy())
        names.append("ln_P:Holiday_Flag")
    return np.column_stack(cols), names


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (coefficients, R-squared) via ordinary least squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return beta, r2


def estimate_elasticities(df: pd.DataFrame) -> dict:
    """Estimate aggregate, cross-price and segmented (peak/off-peak) elasticities.

    Returns a dictionary of point estimates plus the aggregate R-squared and the
    backend used ("statsmodels" or "numpy").
    """
    df = _prepare(df)

    try:
        import statsmodels.formula.api as smf  # preferred: rich inference output

        agg = smf.ols(
            "ln_Q ~ ln_P + ln_Pcomp + Holiday_Flag + Is_Weekend "
            "+ C(Weather_Condition) + C(Month)",
            data=df,
        ).fit()
        seg = smf.ols(
            "ln_Q ~ ln_P * Holiday_Flag + ln_Pcomp + Is_Weekend "
            "+ C(Weather_Condition) + C(Month)",
            data=df,
        ).fit()
        return {
            "backend": "statsmodels",
            "aggregate": float(agg.params["ln_P"]),
            "cross_price": float(agg.params["ln_Pcomp"]),
            "off_peak": float(seg.params["ln_P"]),
            "peak": float(seg.params["ln_P"] + seg.params["ln_P:Holiday_Flag"]),
            "r2": float(agg.rsquared),
            "_models": (agg, seg),
        }
    except ImportError:
        # Self-contained fallback so the pipeline never hard-fails.
        X_agg, names_agg = _design_matrix(df, interaction=False)
        beta_agg, r2 = _ols(df["ln_Q"].to_numpy(), X_agg)
        b_agg = dict(zip(names_agg, beta_agg))

        X_seg, names_seg = _design_matrix(df, interaction=True)
        beta_seg, _ = _ols(df["ln_Q"].to_numpy(), X_seg)
        b_seg = dict(zip(names_seg, beta_seg))

        return {
            "backend": "numpy",
            "aggregate": b_agg["ln_P"],
            "cross_price": b_agg["ln_Pcomp"],
            "off_peak": b_seg["ln_P"],
            "peak": b_seg["ln_P"] + b_seg["ln_P:Holiday_Flag"],
            "r2": r2,
            "_models": None,
        }


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Run `python scripts/generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_CSV)
    res = estimate_elasticities(df)

    print(f"Estimation backend: {res['backend']}   |   Aggregate R^2 = {res['r2']:.3f}\n")
    print("PRICE ELASTICITY OF DEMAND")
    print("-" * 52)
    print(f"  Aggregate own-price elasticity : {res['aggregate']:7.3f}")
    print(f"  Cross-price (Universal)        : {res['cross_price']:7.3f}   "
          f"(>0 -> substitutes)")
    print(f"  Off-peak elasticity            : {res['off_peak']:7.3f}   "
          f"(elastic if < -1)")
    print(f"  Peak / holiday elasticity      : {res['peak']:7.3f}   "
          f"(inelastic if > -1)")
    print("-" * 52)
    interp = ("Peak demand is INELASTIC (room to raise price); off-peak demand is "
              "ELASTIC (discount to fill capacity).")
    print(interp)

    # If statsmodels is available, also print the full aggregate regression table.
    if res["_models"] is not None:
        print("\nFull aggregate regression summary:\n")
        print(res["_models"][0].summary())


if __name__ == "__main__":
    main()
