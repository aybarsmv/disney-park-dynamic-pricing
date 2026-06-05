"""
demand.py
=========
Constant-elasticity demand-curve helpers.

Economic theory
---------------
A constant-elasticity demand curve has the form:

    Q(P) = Q_ref * (P / P_ref) ** epsilon

where `epsilon` (the price elasticity of demand) is constant at every price
point. This is the functional form implied by a log-log regression
ln(Q) = alpha + epsilon * ln(P), which is why the estimation step and the
optimization step share the same model: whatever elasticity we *estimate* can be
fed directly back into this curve to *simulate* counterfactual prices.
"""

from __future__ import annotations


def demand_at_price(
    price: float,
    elasticity: float,
    ref_price: float,
    ref_quantity: float,
) -> float:
    """Return quantity demanded at `price` on a constant-elasticity curve.

    Parameters
    ----------
    price : float
        The candidate price to evaluate.
    elasticity : float
        Price elasticity of demand (expected to be negative for a normal good).
    ref_price : float
        A known reference price (the curve passes through this point).
    ref_quantity : float
        The known quantity demanded at `ref_price`.

    Returns
    -------
    float
        Predicted quantity demanded at `price`.
    """
    if ref_price <= 0 or price <= 0:
        raise ValueError("Prices must be strictly positive.")
    if ref_quantity < 0:
        raise ValueError("Reference quantity cannot be negative.")
    return ref_quantity * (price / ref_price) ** elasticity
