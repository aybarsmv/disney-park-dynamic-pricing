<div align="center">

# 🏰 Experience-Driven Dynamic Pricing & Crowd Yield Management

### *Where Microeconomics Meets the Magic — Optimizing Revenue, Crowds & Guest Delight*

![Status](https://img.shields.io/badge/Status-Complete-E2B973?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-Pricing%20%26%20Behavioral%20Economics-113CCF?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-113CCF?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-E2B973?style=for-the-badge)

</div>

---

## ✨ Executive Summary

A theme park is a **fixed-capacity, perishable-inventory** business — an unsold seat
on the Friday after Thanksgiving is revenue lost forever, while an *over*-sold day
destroys the very experience guests pay a premium for. This project builds an
end-to-end analytics system that resolves that tension.

Combining **microeconomic price-elasticity modeling** with a **behavioral
crowd-redistribution engine**, the system answers two questions at once:
**how much should we charge?** and **where should guests go?** On a reproducible
synthetic dataset it recovers a clean demand curve, derives a constrained
segment-specific price schedule, and pairs it with a nudge-based marketing playbook
projected to add **$1M–$3M+ in annual incremental in-park spend** per park.

---

## 🎯 Business Problem

| Challenge | Cost of the status quo |
|---|---|
| 🔴 **Peak-day over-crowding** | Wait times spike → satisfaction & repeat-visit LTV erode |
| 🔵 **Off-peak under-utilization** | Perishable capacity sits idle → margin left on the table |
| 🟡 **Flat / static pricing** | Ignores that holiday demand is *inelastic* and off-peak demand is *elastic* |

---

## 🧠 Key Methodologies

### 1. Microeconomics — Price Elasticity & Yield Management
- **Log-log demand estimation** recovering constant-elasticity coefficients, with
  controls for holidays, weekends, weather and seasonality to defeat
  **simultaneity bias** (price is endogenous to demand).
- **Segmented elasticity** (peak vs. off-peak) via a price × holiday interaction.
- **Cross-price elasticity** vs. competitor (Universal) pricing — substitution effects.
- A **constrained profit optimizer** applying the **Lerner inverse-elasticity rule**,
  bounded by a **crowd-yield capacity cap** and a **brand-goodwill price ceiling**.

### 2. Behavioral Marketing — Nudge Theory & Gamification
- A 3-stage **gamified quest** redistributing crowds from hyper-crowded to
  under-utilized, high-margin zones.
- Levers: **endowed progress, social proof, scarcity, loss aversion, variable
  rewards, choice architecture** and **loss-leader nudges**.
- A fully auditable **ROI funnel** linking opt-in → redirection → incremental spend.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Language** | ![Python](https://img.shields.io/badge/Python-113CCF?logo=python&logoColor=white) |
| **Data / Compute** | `pandas` · `numpy` |
| **Econometrics** | `statsmodels` (OLS), with a built-in numpy fallback |
| **Optimization** | Custom constrained grid-search engine |
| **Visualization** | `matplotlib` · `seaborn` |
| **Workflow** | Jupyter · Git · modular `src/` package · `pytest` |

---

## 📊 Key Findings & ROI

> *All figures derive from a reproducible synthetic dataset (seed-fixed) and stated
> assumptions, and are reproduced by the scripts in this repository.*

| Metric | Result |
|---|---|
| 🔵 Model fit (aggregate R²) | **0.94** |
| 🔵 Aggregate own-price elasticity | **≈ −1.30** |
| 🔵 Cross-price elasticity (Universal) | **≈ +0.12** (substitutes) |
| 🔵 Peak / holiday elasticity | **≈ −0.17** (inelastic → room to raise price) |
| 🔵 Off-peak elasticity | **≈ −1.47** (elastic → discount to fill capacity) |
| 🟡 **Profit uplift** (dynamic vs. best single flat price) | **+6.2% annualized** |
| 🟡 **Incremental in-park spend** (behavioral nudge) | **$1M–$3M+ / yr / park** |
| 🔴 Crowd-yield cap | Enforced — peak pricing protects guest experience |

**Optimal dynamic schedule:** off-peak ≈ **\$119** · peak ≈ **\$249**
(🔴 brand-ceiling-bound on inelastic peak days — i.e. priced *below* the demand
drop-off, not at it).

---

## 📁 Repository Structure

```text
data/         → raw, processed & generated datasets
notebooks/    → narrative analysis (generation → EDA → elasticity → optimization)
scripts/      → reproducible CLI pipelines
src/          → importable pricing-engine package
marketing/    → behavioral nudge playbook
presentation/ → executive deck & summary
reports/      → exported figures
tests/        → unit tests
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/aybarsmv/disney-park-dynamic-pricing.git
cd experience-driven-dynamic-pricing
pip install -r requirements.txt

python scripts/generate_dataset.py     # 1) build the synthetic data
python scripts/elasticity_model.py      # 2) estimate elasticities
python scripts/dynamic_pricing.py       # 3) derive the optimal price schedule
python -m pytest -q                     # (optional) run the test suite
```

---

## 👤 About the Author

**Aybars Mavi** — Data Scientist · Behavioral Economist · Marketing Strategist

Bridging **quantitative pricing science** and **behavioral marketing** to turn data into decisions that grow revenue *and* delight customers.

📫 [GitHub](https://github.com/aybarsmv) · [Email](mailto:aybarsmv@gmail.com)

<div align="center">

*“It’s kind of fun to do the impossible.” — Walt Disney*

---

<sub>Independent portfolio project. Not affiliated with, endorsed by, or sponsored by
The Walt Disney Company or Universal Studios. All data is synthetic; company names
are used illustratively.</sub>

</div>
