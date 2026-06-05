"""
Unit tests for the pricing engine.

Run from the repository root:
    python -m pytest -q
or, without pytest installed:
    python tests/test_pricing_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pricing_engine import PricingConfig, demand_at_price, optimize_price


def test_demand_is_downward_sloping_for_negative_elasticity():
    """Higher price -> lower quantity when elasticity is negative."""
    q_low = demand_at_price(100, -1.2, ref_price=100, ref_quantity=50_000)
    q_high = demand_at_price(150, -1.2, ref_price=100, ref_quantity=50_000)
    assert q_high < q_low


def test_demand_passes_through_reference_point():
    """At the reference price, quantity equals the reference quantity."""
    q = demand_at_price(120, -0.8, ref_price=120, ref_quantity=42_000)
    assert abs(q - 42_000) < 1e-6


def test_demand_rejects_nonpositive_price():
    for bad in (0, -10):
        try:
            demand_at_price(bad, -1.0, ref_price=100, ref_quantity=10_000)
        except ValueError:
            continue
        raise AssertionError("Expected ValueError for non-positive price.")


def test_optimizer_respects_price_bounds():
    cfg = PricingConfig()
    rec = optimize_price(-0.3, ref_price=170, ref_quantity=58_000, config=cfg)
    assert cfg.price_floor <= rec["price"] <= cfg.price_ceiling


def test_optimizer_never_serves_above_crowd_cap():
    cfg = PricingConfig(crowd_cap=40_000)
    rec = optimize_price(-0.2, ref_price=170, ref_quantity=58_000, config=cfg)
    assert rec["demand"] <= cfg.crowd_cap + 1e-6


def test_inelastic_demand_prices_higher_than_elastic_demand():
    """The inverse-elasticity rule: inelastic segments command a higher price."""
    cfg = PricingConfig()
    inelastic = optimize_price(-0.2, 170, 58_000, cfg)["price"]
    elastic = optimize_price(-1.5, 115, 33_000, cfg)["price"]
    assert inelastic > elastic


def test_profit_is_nonnegative_at_optimum():
    cfg = PricingConfig()
    rec = optimize_price(-1.4, 115, 33_000, cfg)
    assert rec["profit"] > 0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
