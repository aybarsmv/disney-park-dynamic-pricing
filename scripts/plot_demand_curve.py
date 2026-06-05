"""
plot_demand_curve.py
====================
Disney-themed demand curve chart → images/demand_curve.png

Economic note
-------------
The structural demand curves use elasticities recovered from a CONTROLLED
log-log OLS regression (R²=0.94). The raw scatter appears to slope upward
because peak days simultaneously raise both price AND demand (simultaneity
bias). The regression controls for holiday, weather, weekday and month,
isolating the true ceteris-paribus price effect.

Run
---
    python scripts/plot_demand_curve.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DISNEY_BLUE  = "#113CCF"
MAGIC_GOLD   = "#E2B973"
MINNIE_RED   = "#E32525"
INK          = "#0A1A4F"
CREAM        = "#FBF8F1"

ELASTICITY_PEAK    = -0.17
ELASTICITY_OFFPEAK = -1.47
REC_PRICE_PEAK     = 249.0

ROOT     = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "disney_park_data.csv"
OUT_PNG  = ROOT / "images" / "demand_curve.png"


def demand_curve(prices, elasticity, ref_p, ref_q):
    return ref_q * (prices / ref_p) ** elasticity


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError("Run generate_dataset.py first.")

    df   = pd.read_csv(DATA_CSV)
    off  = df[df["Holiday_Flag"] == 0]
    peak = df[df["Holiday_Flag"] == 1]

    # Empirical anchors (segment means) — curves pass through cloud centres
    off_ap,  off_aq  = off["Disney_Ticket_Price"].mean(),  off["Expected_Visitor_Count"].mean()
    peak_ap, peak_aq = peak["Disney_Ticket_Price"].mean(), peak["Expected_Visitor_Count"].mean()
    off_pmin,  off_pmax  = off["Disney_Ticket_Price"].min(),  off["Disney_Ticket_Price"].max()
    peak_pmin, peak_pmax = peak["Disney_Ticket_Price"].min(), peak["Disney_Ticket_Price"].max()

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    # ── Scatter ──────────────────────────────────────────────────────────────
    ax.scatter(off["Expected_Visitor_Count"],  off["Disney_Ticket_Price"],
               s=46, c=DISNEY_BLUE, alpha=0.38, edgecolors="white",
               linewidths=0.5, label="Off-Peak days (observed)", zorder=2)
    ax.scatter(peak["Expected_Visitor_Count"], peak["Disney_Ticket_Price"],
               s=70, c=MAGIC_GOLD, alpha=0.85, edgecolors=INK,
               linewidths=0.6, marker="*",
               label="Peak / Holiday days (observed)", zorder=3)

    # ── Structural curves — solid over OBSERVED price range only ─────────────
    p_off  = np.linspace(off_pmin,  off_pmax,  200)
    p_peak = np.linspace(peak_pmin, peak_pmax, 200)

    ax.plot(demand_curve(p_off,  ELASTICITY_OFFPEAK, off_ap,  off_aq),  p_off,
            color=DISNEY_BLUE, lw=3.4, zorder=5,
            label=f"Off-Peak demand curve (\u03b5 = {ELASTICITY_OFFPEAK:.2f}, elastic)")
    ax.plot(demand_curve(p_peak, ELASTICITY_PEAK,    peak_ap, peak_aq), p_peak,
            color=MINNIE_RED,  lw=3.4, zorder=5,
            label=f"Peak demand curve (\u03b5 = {ELASTICITY_PEAK:.2f}, inelastic)")

    # ── Dashed extrapolation to recommended peak price ───────────────────────
    p_ext = np.linspace(peak_pmax, REC_PRICE_PEAK, 80)
    q_ext = demand_curve(p_ext, ELASTICITY_PEAK, peak_ap, peak_aq)
    ax.plot(q_ext, p_ext, color=MINNIE_RED, lw=2.2, ls="--", zorder=5,
            label="Peak: recommended price (extrapolated \u2014 no data above $177)")
    # X marker at recommended price — no directional arrow
    ax.scatter([q_ext[-1]], [REC_PRICE_PEAK], s=140, c=MINNIE_RED,
               marker="X", edgecolors="white", linewidths=1.2, zorder=6)

    # ── Annotation boxes — no directional arrows on the curves ───────────────
    label_box = dict(boxstyle="round,pad=0.45", facecolor="white",
                     edgecolor="#D9D2C5", alpha=0.93)

    # Peak annotation: point to the SOLID red curve midpoint, not the extrapolation
    peak_mid_p = (peak_pmin + peak_pmax) / 2
    peak_mid_q = demand_curve(peak_mid_p, ELASTICITY_PEAK, peak_ap, peak_aq)
    ax.annotate(
        "Peak: steep & inelastic\n"
        "Price \u2191 \u2192 Demand barely moves\n"
        "(controlled OLS, R\u00b2=0.94)",
        xy=(peak_mid_q, peak_mid_p),
        xytext=(peak_mid_q + 9_000, peak_mid_p + 28),
        fontsize=9.5, color=MINNIE_RED, fontweight="bold",
        ha="left", bbox=label_box,
        arrowprops=dict(arrowstyle="-|>", color=MINNIE_RED,
                        lw=1.5, mutation_scale=12),
    )

    # Off-peak annotation: point to the blue curve midpoint
    off_mid_p = (off_pmin + off_pmax) / 2
    off_mid_q = demand_curve(off_mid_p, ELASTICITY_OFFPEAK, off_ap, off_aq)
    ax.annotate(
        "Off-Peak: flat & elastic\n"
        "Price \u2191 \u2192 Demand drops sharply\n"
        "Discount to fill the park",
        xy=(off_mid_q, off_mid_p),
        xytext=(off_mid_q - 20_000, off_mid_p + 22),
        fontsize=9.5, color=DISNEY_BLUE, fontweight="bold",
        ha="left", bbox=label_box,
        arrowprops=dict(arrowstyle="-|>", color=DISNEY_BLUE,
                        lw=1.5, mutation_scale=12),
    )

    # Recommended price label — plain box, no arrow (it's the X marker)
    ax.annotate(
        f"Recommended peak price  ${REC_PRICE_PEAK:.0f}\n"
        "(brand ceiling \u2014 extrapolated)",
        xy=(q_ext[-1], REC_PRICE_PEAK),
        xytext=(q_ext[-1] - 16_000, REC_PRICE_PEAK - 6),
        fontsize=9, color=MINNIE_RED, fontweight="bold",
        ha="right", va="center", bbox=label_box,
        arrowprops=dict(arrowstyle="-|>", color=MINNIE_RED,
                        lw=1.2, mutation_scale=10),
    )

    # ── Titles, labels, frame ────────────────────────────────────────────────
    ax.set_title("\u2605  Disney Parks \u2014 The Demand Curve  \u2605",
                 fontsize=20, fontweight="bold", color=DISNEY_BLUE, pad=16)
    ax.text(0.5, 1.012,
            "Structural curves anchored at segment means \u00b7 "
            "solid = observed price range \u00b7 dashed = extrapolation",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=INK, style="italic")

    ax.set_xlabel("Expected Visitor Count  (Quantity Demanded)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel("Disney Ticket Price  ($)", fontsize=12, fontweight="bold")
    ax.grid(True, color="#D9D2C5", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(18_000, 96_000)
    ax.set_ylim(100, 258)

    legend = ax.legend(loc="upper right", frameon=True, fontsize=8.8,
                       facecolor="white", edgecolor="#D9D2C5", framealpha=0.95)
    legend.get_frame().set_linewidth(1.0)

    fig.text(0.985, 0.02,
             "Synthetic data \u00b7 "
             "Experience-Driven Dynamic Pricing & Crowd Yield Management",
             ha="right", va="bottom", fontsize=8, color="#8A8270")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.115)
    fig.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print(f"Saved → {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
