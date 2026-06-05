"""
optimizer.py
============
Constrained profit optimizer for theme-park ticket pricing.

Economic theory
---------------
1. Lerner / inverse-elasticity rule:  (P - MC) / P = -1 / epsilon.
   The less elastic demand is (|epsilon| small), the larger the optimal markup.

2. Revenue is maximized at unit elasticity (epsilon = -1). While demand is
   INELASTIC (|epsilon| < 1), raising price raises revenue, because quantity
   falls more slowly than price rises. We therefore stay to the LEFT of the
   unit-elastic point on peak days rather than chasing the "demand drop-off."

3. Yield management adds a CAPACITY constraint that textbook monopoly pricing
   ignores. Beyond an experience cap, additional guests degrade the product
   (wait times explode, satisfaction and long-run brand equity erode). We
   therefore maximize profit SUBJECT TO Q <= crowd_cap, and also respect a
   brand-goodwill PRICE CEILING (the political/PR limit on peak pricing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .demand import demand_at_price


@dataclass(frozen=True)
class PricingConfig:
    """Business constraints for the optimizer.

    Attributes
    ----------
    marginal_cost : float
        Variable cost to serve one additional guest (staffing, consumables, ops).
    crowd_cap : int
        Yield cap on daily guests. Deliberately set BELOW physical capacity to
        protect the guest experience.
    price_floor : float
        Brand-positioning floor; we never discount below this.
    price_ceiling : float
        Brand/PR goodwill ceiling; the political limit on peak pricing.
    price_step : float
        Granularity of the price grid search.
    """

    marginal_cost: float = 38.0
    crowd_cap: int = 68_000
    price_floor: float = 99.0
    price_ceiling: float = 249.0
    price_step: float = 1.0


def optimize_price(
    elasticity: float,
    ref_price: float,
    ref_quantity: float,
    config: PricingConfig | None = None,
) -> dict:
    """Find the profit-maximizing price subject to the crowd-yield cap.

    Parameters
    ----------
    elasticity : float
        Price elasticity of demand for this segment.
    ref_price, ref_quantity : float
        A reference point the demand curve passes through.
    config : PricingConfig, optional
        Business constraints. Defaults to ``PricingConfig()``.

    Returns
    -------
    dict
        Keys: price, demand, revenue, profit, cap_binding (bool).
    """
    cfg = config or PricingConfig()
    grid = np.arange(cfg.price_floor, cfg.price_ceiling + cfg.price_step, cfg.price_step)

    best: dict | None = None
    for price in grid:
        raw_demand = demand_at_price(price, elasticity, ref_price, ref_quantity)
        # Yield constraint: serving more than the cap is not allowed.
        served = min(raw_demand, cfg.crowd_cap)
        revenue = price * served
        profit = (price - cfg.marginal_cost) * served  # contribution margin x volume
        if best is None or profit > best["profit"]:
            best = {
                "price": float(price),
                "demand": float(served),
                "revenue": float(revenue),
                "profit": float(profit),
                "cap_binding": bool(raw_demand > cfg.crowd_cap),
            }
    assert best is not None  # grid is always non-empty
    return best
