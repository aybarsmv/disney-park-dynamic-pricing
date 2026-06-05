"""
generate_dataset.py
===================
Generate a one-year daily synthetic dataset for the Disney (Orlando) pricing
project and write it to ``data/disney_park_data.csv``.

Design principle
----------------
Demand is built as a MULTIPLICATIVE function of its real-world drivers
(seasonality, day-of-week, holidays, weather, own price, competitor price) so
that the downstream econometric model has a true, recoverable structure to
estimate. A separate, independent price shock is injected so that price is not
perfectly collinear with the demand shifters -- this gives the regression the
identifying variation it needs to recover a clean own-price elasticity.

Run
---
    python scripts/generate_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Resolve paths relative to the repository root so the script runs from anywhere.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "data" / "disney_park_data.csv"

SEED = 42
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"


def build_dataset(seed: int = SEED) -> pd.DataFrame:
    """Return the full synthetic park dataset as a DataFrame."""
    rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------
    # 1. Calendar backbone
    # ----------------------------------------------------------------------
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n = len(dates)
    df = pd.DataFrame({"Date": dates})

    dow = df["Date"].dt.dayofweek.to_numpy()        # 0 = Mon ... 6 = Sun
    month = df["Date"].dt.month.to_numpy()
    is_weekend = np.isin(dow, [4, 5, 6]).astype(int)  # Fri/Sat/Sun = park "weekend"

    # ----------------------------------------------------------------------
    # 2. Holiday flag (major US holidays + peak vacation windows)
    # ----------------------------------------------------------------------
    holiday_dates: set[pd.Timestamp] = set()

    def add_range(start: str, end: str) -> None:
        for d in pd.date_range(start, end):
            holiday_dates.add(d.normalize())

    add_range("2024-01-01", "2024-01-02")   # New Year
    holiday_dates.add(pd.Timestamp("2024-01-15"))   # MLK Day
    add_range("2024-02-17", "2024-02-25")   # Presidents' week
    add_range("2024-03-09", "2024-04-07")   # Spring break window
    holiday_dates.add(pd.Timestamp("2024-05-27"))   # Memorial Day
    add_range("2024-07-03", "2024-07-07")   # Independence Day
    holiday_dates.add(pd.Timestamp("2024-09-02"))   # Labor Day
    add_range("2024-11-23", "2024-11-30")   # Thanksgiving week
    add_range("2024-12-21", "2024-12-31")   # Christmas -> New Year surge

    holiday_flag = df["Date"].dt.normalize().isin(holiday_dates).astype(int).to_numpy()

    # ----------------------------------------------------------------------
    # 3. Weather (Florida: wetter, stormier summers; drier winters)
    # ----------------------------------------------------------------------
    conditions = ["Sunny", "Cloudy", "Rainy", "Stormy"]

    def weather_probs(m: int) -> list[float]:
        if m in (6, 7, 8, 9):               # hurricane / storm season
            return [0.40, 0.25, 0.20, 0.15]
        if m in (12, 1, 2):                 # dry season
            return [0.58, 0.29, 0.10, 0.03]
        return [0.55, 0.25, 0.15, 0.05]     # shoulder seasons

    weather = np.array(
        [rng.choice(conditions, p=weather_probs(m)) for m in month]
    )
    weather_mult = (
        pd.Series(weather)
        .map({"Sunny": 1.05, "Cloudy": 1.00, "Rainy": 0.88, "Stormy": 0.75})
        .to_numpy()
    )

    # ----------------------------------------------------------------------
    # 4. Demand-shifter multipliers
    # ----------------------------------------------------------------------
    month_mult_map = {
        1: 0.85, 2: 0.90, 3: 1.15, 4: 1.00, 5: 0.95, 6: 1.20,
        7: 1.30, 8: 1.15, 9: 0.78, 10: 0.95, 11: 1.00, 12: 1.12,
    }
    season_mult = np.array([month_mult_map[m] for m in month])
    weekend_mult = np.where(is_weekend == 1, 1.15, 0.95)
    holiday_mult = np.where(holiday_flag == 1, 1.35, 1.00)

    # ----------------------------------------------------------------------
    # 5. Prices
    #    Disney already practices tiered pricing, so price is partly endogenous
    #    to demand peaks. We add an INDEPENDENT price shock so the regression
    #    can still identify the own-price elasticity (see module docstring).
    # ----------------------------------------------------------------------
    price_pressure = (
        1
        + 0.18 * holiday_flag
        + 0.06 * is_weekend
        + 0.12 * (season_mult - 1)
    )
    disney_price_shock = rng.normal(0, 12, n)       # independent variation for identification
    disney_price = np.clip(
        np.round(124.0 * price_pressure + disney_price_shock), 109, 209
    )
    universal_price = np.clip(
        np.round(114.0 * price_pressure + rng.normal(0, 6, n)), 105, 199
    )

    # ----------------------------------------------------------------------
    # 6. Hotel occupancy (shares the latent demand pressure + noise)
    # ----------------------------------------------------------------------
    occ = (
        0.60
        + 0.20 * holiday_flag
        + 0.08 * is_weekend
        + 0.20 * (season_mult - 1)
        + rng.normal(0, 0.035, n)
    )
    occ = np.clip(np.round(occ, 3), 0.45, 0.99)

    # ----------------------------------------------------------------------
    # 7. Expected visitor count (the demand curve)
    #    Key structure: demand is INELASTIC on peak/holiday days (captive,
    #    pre-committed travelers) and ELASTIC off-peak (discretionary visitors).
    # ----------------------------------------------------------------------
    p_ref = 124.0
    elasticity = np.where(holiday_flag == 1, -0.35, -1.50)
    own_price_term = (disney_price / p_ref) ** elasticity
    cross_price_term = (universal_price / 114.0) ** 0.20   # substitutes -> positive cross-elasticity

    base_demand = 42_000
    expected = (
        base_demand
        * season_mult
        * weekend_mult
        * holiday_mult
        * weather_mult
        * own_price_term
        * cross_price_term
    )
    expected *= rng.normal(1.0, 0.05, n)                   # multiplicative demand shock
    visitor_count = np.clip(np.round(expected), 8_000, 95_000).astype(int)

    # ----------------------------------------------------------------------
    # 8. Average wait time (rises convexly with park utilization)
    # ----------------------------------------------------------------------
    capacity = 72_000
    utilization = visitor_count / capacity
    wait = 18 + 70 * (utilization ** 1.6) + rng.normal(0, 4, n)
    wait = np.clip(np.round(wait), 10, 140).astype(int)

    # ----------------------------------------------------------------------
    # 9. Assemble
    # ----------------------------------------------------------------------
    df["Weather_Condition"] = weather
    df["Holiday_Flag"] = holiday_flag
    df["Hotel_Occupancy_Rate"] = occ
    df["Disney_Ticket_Price"] = disney_price          # required for OWN-price elasticity
    df["Universal_Studios_Price"] = universal_price
    df["Expected_Visitor_Count"] = visitor_count
    df["Avg_Wait_Time_Minutes"] = wait
    return df


def main() -> None:
    df = build_dataset()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV.relative_to(ROOT)}\n")
    print(df.head())


if __name__ == "__main__":
    main()
