"""
pricing_engine
==============
Reusable components for the Experience-Driven Dynamic Pricing project.

Modules
-------
demand     : constant-elasticity demand curve helpers.
optimizer  : constrained profit optimizer with a crowd-yield cap.
"""

from .demand import demand_at_price
from .optimizer import PricingConfig, optimize_price

__all__ = ["demand_at_price", "PricingConfig", "optimize_price"]
__version__ = "1.0.0"
