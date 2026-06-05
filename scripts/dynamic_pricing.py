"""
dynamic_pricing.py
==================
Turn the estimated elasticities into an actionable, segment-specific price
schedule under a crowd-yield constraint, and quantify the profit uplift over the
best single flat price.

Economic theory
---------------
1. Lerner / inverse-elasticity rule:  (P - MC) / P = -1 / epsilon.
   Less elastic demand justifies a larger markup, so peak (inelastic) days carry
   a higher price than off-peak (elastic) days.

2. As long as demand is INELASTIC (|epsilon| < 1), raising price raises revenue.
   On peak days we therefore raise price toward the brand-goodwill ceiling -- we
   stay LEFT of the unit-elastic "demand drop-off" rather than crossing it.

3. Yield management adds a CAPACITY constraint: beyond the crowd cap, extra
   guests degrade the experience. The optimizer maximizes profit subject to
   Q <= crowd_cap.

Why measure uplift against the BEST single flat price?
------------------------------------------------------
A naive comparison against an arbitrary flat price overstates the gain. The
honest measure of the value of *segmentation* is the gain over the best possible
single price -- and we annualize it, weighting each segment by its true number of
days, because a year has far more off-peak days than peak days.

Run
---
    python scripts/dynamic_pricing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make the pricing_engine package importable

from pricing_engine import PricingConfig, demand_at_price, optimize_price  # noqa: E402
from elasticity_model import DATA_CSV, estimate_elasticities  # noqa: E402

# Reference operating points (typical price / volume) per segment.
REF_PEAK = (170.0, 58_000)
REF_OFFPEAK = (115.0, 33_000)


def _served(price: float, elasticity: float, ref, cfg: PricingConfig) -> float:
    return min(demand_at_price(price, elasticity, ref[0], ref[1]), cfg.crowd_cap)


def best_single_flat_price(
    peak_e: float, off_e: float, peak_days: int, off_days: int, cfg: PricingConfig
) -> tuple[float, float]:
    """Find the single annual flat price that maximizes total profit."""
    grid = np.arange(cfg.price_floor, cfg.price_ceiling + cfg.price_step, cfg.price_step)
    best_price, best_profit = cfg.price_floor, -np.inf
    for price in grid:
        profit = (price - cfg.marginal_cost) * (
            peak_days * _served(price, peak_e, REF_PEAK, cfg)
            + off_days * _served(price, off_e, REF_OFFPEAK, cfg)
        )
        if profit > best_profit:
            best_price, best_profit = float(price), float(profit)
    return best_price, best_profit


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Run `python scripts/generate_dataset.py` first."
        )

    df = pd.read_csv(DATA_CSV)
    elast = estimate_elasticities(df)
    cfg = PricingConfig()

    peak_days = int(df["Holiday_Flag"].sum())
    off_days = len(df) - peak_days

    # Segment-specific optimal prices.
    rec_peak = optimize_price(elast["peak"], *REF_PEAK, cfg)
    rec_off = optimize_price(elast["off_peak"], *REF_OFFPEAK, cfg)

    # Annualized profit: dynamic schedule vs the best single flat price.
    ann_dynamic = peak_days * rec_peak["profit"] + off_days * rec_off["profit"]
    flat_price, ann_flat = best_single_flat_price(
        elast["peak"], elast["off_peak"], peak_days, off_days, cfg
    )
    uplift = (ann_dynamic / ann_flat - 1) * 100

    bar = "=" * 60
    print(bar)
    print("DYNAMIC PRICE SCHEDULE  (crowd-yield constrained)")
    print(bar)
    print(f"Calendar mix              : {peak_days} peak days / {off_days} off-peak days")
    print(f"Constraints               : MC ${cfg.marginal_cost:.0f} | "
          f"crowd cap {cfg.crowd_cap:,} | "
          f"price ${cfg.price_floor:.0f}-${cfg.price_ceiling:.0f}")
    print("-" * 60)
    peak_bind = "BRAND-CEILING-bound" if not rec_peak["cap_binding"] else "CROWD-CAP-bound"
    print(f"OFF-PEAK  ->  ${rec_off['price']:6.0f}   "
          f"(elasticity {elast['off_peak']:5.2f}, served {rec_off['demand']:,.0f})")
    print(f"PEAK      ->  ${rec_peak['price']:6.0f}   "
          f"(elasticity {elast['peak']:5.2f}, served {rec_peak['demand']:,.0f}, {peak_bind})")
    print("-" * 60)
    print(f"Best single flat price    : ${flat_price:.0f}")
    print(f"Annual profit (flat)      : ${ann_flat / 1e6:,.1f}M")
    print(f"Annual profit (dynamic)   : ${ann_dynamic / 1e6:,.1f}M")
    print(f"Profit uplift from pricing : +{uplift:.1f}%  (value of segmentation)")
    print(bar)


if __name__ == "__main__":
    main()
