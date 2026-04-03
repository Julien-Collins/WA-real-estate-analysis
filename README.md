# WA Real Estate Analysis

## Project Walkthrough Video
👉 [Watch the full walkthrough](https://drive.google.com/file/d/16soLqfzNbcLvcAjJr0xC6v-7zz1dsIlw/view?usp=drive_link)

This video walks through the full analysis, including data preparation, modeling, and key findings.

## Overview
I built this project to better understand how housing market conditions in Washington State actually impact outcomes like how fast homes sell and how strong pricing is.

Specifically, I focused on:
- Days on Market (DOM)
- Sale-to-List Price Ratio

The goal was to quantify how supply (measured through months of inventory) and seasonality affect both.

---

## Dataset
The dataset is structured at the county-month level and contains ~8,100 observations from 2012–2025.

Some of the key fields:
- Region
- Month of Period End
- Homes Sold
- Inventory
- Days on Market
- Average Sale To List

---

## Data Preparation
There were a few key issues I had to fix upfront:

- The `Region` column only had ~50 actual values and the rest missing → I forward-filled it
- Converted date fields using `pd.to_datetime`
- Converted numeric columns with `errors="coerce"` to handle bad values
- Dropped rows with missing core variables

I also created:

- **Months of Inventory = Inventory / Homes Sold**
- Time features (year, month, quarter)
- A seasonal variable (Winter, Spring, Summer, Fall)

To stabilize the analysis, I capped extreme values at the 95th percentile:
- Days on Market capped at **174 days**
- Months of Inventory capped at **~12.3 months** :contentReference[oaicite:1]{index=1}

---

## Exploratory Analysis

### Market Averages
- Average Days on Market: **~58 days**
- Average Sale-to-List Ratio: **~0.985** (homes typically sell slightly below list) :contentReference[oaicite:2]{index=2}

### Seasonal Differences

| Season | Avg DOM | Sale-to-List |
|--------|--------|--------------|
| Winter | 72.1   | 0.979 |
| Spring | 58.3   | 0.990 |
| Summer | 45.3   | 0.989 |
| Fall   | 56.0   | 0.982 |

Key takeaway:
- Homes sell **fastest in Summer**
- Homes sell **slowest in Winter**

---

### Above-List Pricing (% of periods)

- Winter: **19%**
- Spring: **34%**
- Summer: **31%**
- Fall: **20%**

 Strongest pricing environment = **Spring / Summer**

---

### Confidence Intervals (DOM)

Example:
- Winter: **72.15 ± 2.12 days**
- Summer: **45.27 ± 1.80 days**

 These ranges are tight, meaning the seasonal differences are **very reliable**

---

## Regression Modeling

I built two OLS regression models using `statsmodels`.

---

### Model 1: Days on Market

**R² = 0.527**

This model explains ~53% of variation in how long homes take to sell.

#### Key coefficients:

- Months of Inventory: **+10.80**
  - Each additional 1 month of inventory → **~11 extra days on market**

- Spring: **-8.48 days**
- Summer: **-20.27 days**
- Fall: **-10.45 days**

(All relative to Winter baseline)

 Interpretation:
- Inventory is the biggest driver
- Seasonality also matters a lot, especially Summer

---

### Model 2: Sale-to-List Ratio

**R² = 0.355**

#### Key coefficients:

- Months of Inventory: **-0.0061**
  - Each additional month of inventory → ~0.61% drop in sale-to-list ratio

- Spring: **+0.0075**
- Summer: **+0.0059**
- Fall: not statistically significant

 Interpretation:
- Higher inventory → weaker pricing power
- Spring/Summer slightly outperform Winter

---

## Model Diagnostics

I checked:

- Multicollinearity (VIF all ~1–1.5 → no issue)
- Residual distributions
- Q-Q plots
- Residual vs fitted plots

Overall:
- Models are reasonably well-behaved
- Some non-normality (expected with real-world housing data)

---

## Key Takeaways

- **Inventory is the #1 driver of market conditions**
  - More supply → slower sales and weaker pricing

- **Seasonality strongly affects speed**
  - Summer is significantly faster than Winter (~20+ days difference)

- **Pricing is less seasonal than speed**
  - Inventory matters more than season for pricing outcomes

- **Spring/Summer are peak competitive markets**
  - Highest likelihood of above-list pricing

- **Market dynamics are highly consistent**
  - Tight confidence intervals show stable seasonal patterns

---

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Statsmodels

---

## How to Run

1. Install dependencies:
```bash
pip install pandas numpy matplotlib statsmodels openpyxl
