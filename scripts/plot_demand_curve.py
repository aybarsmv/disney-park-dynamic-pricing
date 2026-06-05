"""
plot_demand_curve.py
====================
Read ``data/disney_park_data.csv`` and render a Disney-themed demand-curve chart
to ``images/demand_curve.png``.

Design notes (why this version is defensible)
---------------------------------------------
* Each structural demand curve is anchored at its EMPIRICAL segment centroid
  (mean price, mean quantity) computed from the data -- so every curve passes
  through the middle of its own cloud instead of floating at the edge.
* Each curve is drawn SOLID only across the price range actually observed for
  that segment. The optimizer's recommended peak price sits above the observed
  range, so that portion is drawn DASHED and labelled as an extrapolation -- we
  never present a fitted line where no data exists.
* Constant-elasticity form, using the elasticities recovered in
  ``scripts/elasticity_model.py``.

Run
---
    python scripts/plot_demand_curve.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Disney brand palette -----------------------------------------------------
DISNEY_BLUE = "#113CCF"
MAGIC_GOLD = "#E2B973"
MINNIE_RED = "#E32525"
INK = "#0A1A4F"
CREAM = "#FBF8F1"

# --- Estimated elasticities (from scripts/elasticity_model.py) ----------------
ELASTICITY_PEAK = -0.17
ELASTICITY_OFFPEAK = -1.47

# --- Optimizer's recommended prices (from scripts/dynamic_pricing.py) ---------
REC_PRICE_OFFPEAK = 119.0
REC_PRICE_PEAK = 249.0

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "disney_park_data.csv"
OUT_PNG = ROOT / "images" / "demand_curve.png"


def demand_curve(prices, elasticity, ref_price, ref_q):
    """Constant-elasticity demand: Q = Q_ref * (P / P_ref) ** elasticity."""
    return ref_q * (prices / ref_price) ** elasticity


def main() -> None:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Run `python scripts/generate_dataset.py` first."
        )
    df = pd.read_csv(DATA_CSV)
    off = df[df["Holiday_Flag"] == 0]
    peak = df[df["Holiday_Flag"] == 1]

    # Empirical anchors (segment centroids) and observed price ranges.
    off_anchor = (off["Disney_Ticket_Price"].mean(), off["Expected_Visitor_Count"].mean())
    peak_anchor = (peak["Disney_Ticket_Price"].mean(), peak["Expected_Visitor_Count"].mean())
    off_pmin, off_pmax = off["Disney_Ticket_Price"].min(), off["Disney_Ticket_Price"].max()
    peak_pmin, peak_pmax = peak["Disney_Ticket_Price"].min(), peak["Disney_Ticket_Price"].max()

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": INK, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    })

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    # --- Observed data cloud (x = quantity, y = price) ---
    ax.scatter(off["Expected_Visitor_Count"], off["Disney_Ticket_Price"],
               s=46, c=DISNEY_BLUE, alpha=0.40, edgecolors="white", linewidths=0.5,
               label="Off-Peak days (observed)", zorder=2)
    ax.scatter(peak["Expected_Visitor_Count"], peak["Disney_Ticket_Price"],
               s=70, c=MAGIC_GOLD, alpha=0.85, edgecolors=INK, linewidths=0.6,
               marker="*", label="Peak / Holiday days (observed)", zorder=3)

    # --- Structural curves: SOLID over observed range only ---
    p_off = np.linspace(off_pmin, off_pmax, 200)
    ax.plot(demand_curve(p_off, ELASTICITY_OFFPEAK, *off_anchor), p_off,
            color=DISNEY_BLUE, lw=3.4, zorder=5,
            label=f"Off-Peak demand curve (\u03b5 = {ELASTICITY_OFFPEAK:.2f}, elastic)")

    p_peak = np.linspace(peak_pmin, peak_pmax, 200)
    ax.plot(demand_curve(p_peak, ELASTICITY_PEAK, *peak_anchor), p_peak,
            color=MINNIE_RED, lw=3.4, zorder=5,
            label=f"Peak demand curve (\u03b5 = {ELASTICITY_PEAK:.2f}, inelastic)")

    # --- Recommended peak price = extrapolation beyond data -> DASHED ---
    p_ext = np.linspace(peak_pmax, REC_PRICE_PEAK, 80)
    q_ext = demand_curve(p_ext, ELASTICITY_PEAK, *peak_anchor)
    ax.plot(q_ext, p_ext, color=MINNIE_RED, lw=2.2, ls="--", zorder=5,
            label="Peak: recommended price (extrapolated, no data)")
    ax.scatter([q_ext[-1]], [REC_PRICE_PEAK], s=130, c=MINNIE_RED, marker="X",
               edgecolors="white", linewidths=1.2, zorder=6)
    ax.annotate(f"Recommended peak\nprice  ${REC_PRICE_PEAK:.0f}\n(brand ceiling)",
                xy=(q_ext[-1], REC_PRICE_PEAK), xytext=(q_ext[-1] - 17_000, REC_PRICE_PEAK - 4),
                fontsize=9.5, color=MINNIE_RED, fontweight="bold", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#D9D2C5", alpha=0.92))

    # --- Annotations on the observed portion ---
    label_box = dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="#D9D2C5", alpha=0.92)
    ax.annotate("Peak: steep & inelastic\n(price moves, demand barely does)",
                xy=(peak_anchor[1], peak_anchor[0]), xytext=(peak_anchor[1] + 6_000, 196),
                fontsize=10, color=MINNIE_RED, fontweight="bold", ha="left", bbox=label_box,
                arrowprops=dict(arrowstyle="->", color=MINNIE_RED, lw=1.6))
    ax.annotate("Off-Peak: flat & elastic\n(discount to fill the park)",
                xy=(off_anchor[1], off_anchor[0]), xytext=(off_anchor[1] - 23_000, 158),
                fontsize=10, color=DISNEY_BLUE, fontweight="bold", ha="left", bbox=label_box,
                arrowprops=dict(arrowstyle="->", color=DISNEY_BLUE, lw=1.6))

    # --- Titles, labels, frame ---
    ax.set_title("\u2605  Disney Parks \u2014 The Demand Curve  \u2605",
                 fontsize=20, fontweight="bold", color=DISNEY_BLUE, pad=16)
    ax.text(0.5, 1.012,
            "Ticket Price vs. Expected Visitor Count  \u00b7  curves anchored at "
            "segment means, drawn over observed prices",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.5, color=INK, style="italic")
    ax.set_xlabel("Expected Visitor Count  (Quantity Demanded)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Disney Ticket Price  ($)", fontsize=12, fontweight="bold")
    ax.grid(True, color="#D9D2C5", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(18_000, 96_000)
    ax.set_ylim(100, 258)

    legend = ax.legend(loc="upper right", frameon=True, fontsize=9,
                       facecolor="white", edgecolor="#D9D2C5", framealpha=0.95)
    legend.get_frame().set_linewidth(1.0)

    fig.text(0.985, 0.02,
             "Synthetic data \u00b7 Experience-Driven Dynamic Pricing & Crowd Yield Management",
             ha="right", va="bottom", fontsize=8, color="#8A8270")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.115)
    fig.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved chart to {OUT_PNG.relative_to(ROOT)}")
    print(f"Off-peak anchor (mean): ${off_anchor[0]:.0f} / {off_anchor[1]:,.0f} guests "
          f"| observed price ${off_pmin:.0f}-${off_pmax:.0f}")
    print(f"Peak anchor (mean):     ${peak_anchor[0]:.0f} / {peak_anchor[1]:,.0f} guests "
          f"| observed price ${peak_pmin:.0f}-${peak_pmax:.0f}")


if __name__ == "__main__":
    main()
