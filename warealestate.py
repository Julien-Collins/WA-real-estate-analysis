# Python Analysis
# Washington State Real Estate
# Multiple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load Data
df = pd.read_excel("WA - Real Estate Data.xlsx")

# check shape, first few records, and data types 

print(df.shape)
print(df.head())
df.info()

print("\n Region column validation")

# Count missing Region values

missing_region_count = df["Region"].isna().sum()
total_rows = len(df)

print(f"Total rows: {total_rows}")
print(f"Missing Region values: {missing_region_count}")


# Forward fill the Region column to address the redfin issue
df["Region"] = df["Region"].ffill()

# Convert date column of months to datetime 
df["Month of Period End"] = pd.to_datetime(df["Month of Period End"], errors="coerce")

# Create time-based features
df["Year"] = df["Month of Period End"].dt.year
df["Month"] = df["Month of Period End"].dt.month
df["Quarter"] = df["Month of Period End"].dt.quarter

# Create categorical season variable
def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df["Season"] = df["Month"].apply(assign_season)

print("Season variable created.")
print(df[["Region", "Month of Period End", "Month", "Season"]].head(10))

df["Homes Sold"] = pd.to_numeric(df["Homes Sold"], errors="coerce")
df["Inventory"] = pd.to_numeric(df["Inventory"], errors="coerce")

# Months of Inventory = Inventory / Homes Sold
# New engineered variable for analysis 

df["Months_Inventory"] = np.where(df["Homes Sold"] > 0, df["Inventory"] / df["Homes Sold"], np.nan)

# Define analysis variables

analysis_vars = [
    "Region",
    "Month of Period End",
    "Season",
    "Months_Inventory",
    "Days on Market",
    "Average Sale To List"
]

# Create clean analysis dataframe

df_analysis = df[analysis_vars].dropna().copy()

# outlier handling with 95th percentile capping

dom_cap = df_analysis["Days on Market"].quantile(0.95)
moi_cap = df_analysis["Months_Inventory"].quantile(0.95)

# this logic ensures that values over dom_cap are converted into dom_cap, otherwise stay the same
# condition -> value if true -> value if false for dom and moi

df_analysis["Days_on_Market_capped"] = np.where(
    df_analysis["Days on Market"] > dom_cap,
    dom_cap,
    df_analysis["Days on Market"]
)

df_analysis["Months_Inventory_capped"] = np.where(
    df_analysis["Months_Inventory"] > moi_cap,
    moi_cap,
    df_analysis["Months_Inventory"]
)

print("days on market capped at:", round(dom_cap, 2))
print("months of inventory capped at:", round(moi_cap, 2))


print("new dataset shape:", df_analysis.shape)
print(df_analysis.head())

# Summary statistics
print(df_analysis.describe())



# VISUALIZATIONS for Analysis 

#  Days on Market — capped distribution 
plt.figure(figsize=(7,4))

plt.hist(
    df_analysis["Days_on_Market_capped"],
    bins=np.arange(0, 201, 10)  
)
plt.title("Distribution of Days on Market (0-200 Days)")
plt.xlabel("Days on Market")
plt.ylabel("Frequency")
plt.xlim(0, 200)
plt.tight_layout
plt.show()


# Sale-to-List ratio 

plt.figure(figsize=(7,4))
plt.hist(
    df_analysis["Average Sale To List"],
    bins=np.arange(0.90, 1.05, 0.005)
)
plt.title("Detailed Distribution of Sale-to-List Ratio (0.90-1.05)")
plt.xlabel("Sale-to-List Ratio")
plt.ylabel("Frequency")
plt.show()


#  Sale-to-List Ratio by Season — boxplot 

ax = df_analysis.boxplot(
    column="Average Sale To List",
    by="Season",
    showfliers=False,
    figsize=(7,4)
)
plt.title("Sale-to-List Ratio by Season")
plt.suptitle("")
plt.xlabel("Season")
plt.ylabel("Sale-to-List Ratio")
plt.tight_layout()
plt.show()
plt.close()

#  Seasonal Summary Statistics 
# group data by season, calc avg for variables, round 3 dec

season_summary = (
    df_analysis
    .groupby("Season")[["Average Sale To List", "Days_on_Market_capped"]]
    .mean()
    .round(3)
)

print("\nSeasonal Averages:")
print(season_summary)


season_order = ["Winter", "Spring", "Summer", "Fall"]

plot_df = df_analysis.copy()

# enfore the order of teh seasons
plot_df["Season"] = pd.Categorical(plot_df["Season"], categories=season_order, ordered=True) 
plot_df = plot_df.dropna(subset=["Season", "Days_on_Market_capped"])

# group data, compile stats, and then organize results with reindex
summary = (
    plot_df.groupby("Season", observed=False)["Days_on_Market_capped"]
    .agg(["count", "mean", "std"])
    .reindex(season_order)
)

# 95% CI 
# standard error will determine how precise the averages are

print("\n Standard Deviation")
print(summary["std"])

# use formula for se - std divided by the square root of the sample size
# calculating uncertainty of avgs
# 95% of values fall within plus or minus 1.96 standard errors

summary["se"] = summary["std"] / np.sqrt(summary["count"])
summary["ci95"] = 1.96 * summary["se"]

print(summary["ci95"])
# maps the indxing for seasons so that 0 = winter,  1 = spring, etc

summary = summary.dropna()

x = np.arange(len(season_order))

# map out the confidence interval error bars to see what they look like

plt.figure(figsize=(8,5))
plt.errorbar(
    x,
    summary["mean"].values,
    yerr=summary["ci95"].values,
    fmt="o",
    capsize=6
)
plt.xticks(x, season_order)
plt.title("Average Days on Market by Season - 95% Confidence Intervals")
plt.xlabel("Season")
plt.ylabel("Average Days on Market")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print(summary[["count", "mean", "ci95"]].round(2))



season_order = ["Winter", "Spring", "Summer", "Fall"]
plot_df = df_analysis.dropna(subset=["Months_Inventory_capped", "Average Sale To List", "Season"]).copy()

# all charts use the same scale - share x and y
fig, axes = plt.subplots(2, 2, figsize=(10,7), sharex=True, sharey=True)
axes = axes.ravel()
# compare relationship across seasons on same scale

# loops through seasons, filters data for season, and assigns to subplot
# goes through i position of (0,1,2,3) and s (season name)
#then ax takes the index position of i (or seasonal index) 
# filter to keep rows for selected season (s)
# scatter plot with x being months inv and y sale to list

for i, s in enumerate(season_order):
    ax = axes[i]
    subset = plot_df[plot_df["Season"] == s]
    ax.scatter(subset["Months_Inventory_capped"], subset["Average Sale To List"], s=8, alpha=0.25)
    
    # Trend line
    if len(subset) > 20:
        # fits a straight line through the data
        coef = np.polyfit(subset["Months_Inventory_capped"], subset["Average Sale To List"], 1)
        x_line = np.linspace(subset["Months_Inventory_capped"].min(), subset["Months_Inventory_capped"].max(), 100)
        # y = mx + b   (slope)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line, linewidth=2)

    ax.set_title(s)
    ax.grid(alpha=0.15)

fig.suptitle("Sale-to-List Ratio vs. Months of Inventory (By Season)", y=1.02)
fig.text(0.5, 0.04, "Months of Inventory", ha="center")
fig.text(0.04, 0.5, "Average Sale-to-List Ratio", va="center", rotation="vertical")
plt.tight_layout()
plt.show()

# ^ this is showing the relationship between months of inv and sales to list price ratio.
# the trend line going down suggests that as months of inventory increase, the ratio declines

bins = np.arange(0, 201, 10)  

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
# flatten the 2 x 2 list into one dimension - easier to access it
axes = axes.ravel()

# grab index and the value 
# enumerate lets you loop through list while keeping track of item and position

for i, season in enumerate(season_order):
    # picks the subplot to draw on
    ax = axes[i]
    season_df = df_analysis[
        (df_analysis["Season"] == season) &
        (~df_analysis["Days_on_Market_capped"].isna())
    ]

    ax.hist(
        season_df["Days_on_Market_capped"],
        bins=bins,
        alpha=0.85
    )

    ax.set_title(season)
    ax.set_xlim(0, 200)
    ax.grid(axis="y", alpha=0.3)

    median_dom = season_df["Days_on_Market_capped"].median()
    ax.axvline(median_dom, linestyle="--", linewidth=1)

# Shared labels
fig.suptitle("Distribution of Days on Market by Season", fontsize=14)
fig.text(0.5, 0.04, "Days on Market (Capped)", ha="center")
fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical")
plt.tight_layout(rect=[0.04, 0.05, 1, 0.95])
plt.show()

# Consistent bins and limits across seasons
bins = np.arange(0.94, 1.041, 0.01)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
axes = axes.ravel()

for i, season in enumerate(season_order):
    ax = axes[i]
    
    #create smaller dataframe that has rows for current season and rows where dom capped is not missing
    season_df = df_analysis[
        (df_analysis["Season"] == season) &
        (~df_analysis["Average Sale To List"].isna())
    ]

    ax.hist(
        season_df["Average Sale To List"],
        bins=bins,
        alpha=0.85
    )

    ax.set_title(season)
    ax.set_xlim(0.94, 1.04)
    ax.grid(axis="y", alpha=0.3)
    # sets the limits for avg sale to list price ratio on x ax

    # Median vertical line shown on each distribution with dashed line
    median_stl = season_df["Average Sale To List"].median()
    ax.axvline(median_stl, linestyle="--", linewidth=1)

# Shared labels for one title and shared x and y
fig.suptitle("Distribution of Sale-to-List Price Ratio by Season", fontsize=14)
fig.text(0.5, 0.04, "Sale-to-List Price Ratio", ha="center")
fig.text(0.04, 0.5, "Frequency", va="center", rotation="vertical")
plt.tight_layout(rect=[0.04, 0.05, 1, 0.95])
plt.show()


# create new clean dataframe and copy so safe for edits
plot_df = df_analysis.dropna(subset=["Season", "Average Sale To List"]).copy()

# Create binary for sold above list

plot_df["Above_List"] = plot_df["Average Sale To List"] > 1.00

# Calculate percentage by season for those that sold above list price
season_summary = (
    plot_df
    .groupby("Season")["Above_List"]
    .mean()
    .reindex(season_order)
    * 100
)

plt.figure(figsize=(7,5))
plt.bar(season_summary.index, season_summary.values) # one bar per season - bar high demonstrates mean result

plt.title("Percentage of County-Months with Above-List Average Pricing By Season")
plt.xlabel("Season")
plt.ylabel("Percent of Sales Above List Price")
plt.ylim(0, max(season_summary.values) * 1.15) # add some more room above tallest bar
plt.grid(axis="y", alpha=0.3)

# Add value labels and loop through each bar and have formatted % label above it w decimal place
for i, v in enumerate(season_summary.values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
print(season_summary.round(2))



# Select regression variables
df_reg = df_analysis[
    ["Days_on_Market_capped", "Average Sale To List", "Months_Inventory_capped", "Season"]
].dropna().copy()

# Lock season order and create a baseline
season_order = ["Winter", "Spring", "Summer", "Fall"]
df_reg["Season"] = pd.Categorical(df_reg["Season"], categories=season_order)

# Dummy encode Season
season_dummies = pd.get_dummies(df_reg["Season"], drop_first=True, dtype=float)

# Build X matrix
X = pd.concat([df_reg[["Months_Inventory_capped"]].astype(float), season_dummies], axis=1)

# Add intercept
X = sm.add_constant(X, has_constant="add")

# Ensure everything is numeric and float
X = X.astype(float)

# Dependent vars
y_dom = df_reg["Days_on_Market_capped"].astype(float)
y_stl = df_reg["Average Sale To List"].astype(float)


# MODEL 1: Days on Market

model_dom = sm.OLS(y_dom, X).fit()
print("\nMODEL 1: Days on Market Regression Results")
print(model_dom.summary())


# MODEL 2: Sale-to-List Ratio

model_stl = sm.OLS(y_stl, X).fit()
print("\nMODEL 2: Sale-to-List Ratio Regression Results")
print(model_stl.summary())


# multicollinearity check - are predictrs correlated 

vif_df = pd.DataFrame()
vif_df["Variable"] = X.columns
vif_df["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\nVariance Inflation Factors:")
print(vif_df)


# RESIDUAL DIAGNOSTIC CHECKS

# Model 1: Days on Market
resid_dom = model_dom.resid

plt.figure(figsize=(7,4))
plt.hist(resid_dom, bins=40)
plt.title("Residual Distribution: Days on Market Model")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

sm.qqplot(resid_dom, line="45")
plt.title("Q-Q Plot: Days on Market Model Residuals")
plt.show()


#Model 2: Sale-to-List Ratio 
resid_stl = model_stl.resid

plt.figure(figsize=(7,4))
plt.hist(resid_stl, bins=40)
plt.title("Residual Distribution: Sale-to-List Ratio Model")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

sm.qqplot(resid_stl, line="45")
plt.title("Q-Q Plot: Sale-to-List Ratio Model Residuals")
plt.show()

# HOMOSCEDASTICITY CHECK

# Model 1
plt.figure(figsize=(6,4))
plt.scatter(model_dom.fittedvalues, model_dom.resid, alpha=0.3)
plt.axhline(0)
plt.title("Residuals vs Fitted Values (Days on Market)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# Model 2
plt.figure(figsize=(6,4))
plt.scatter(model_stl.fittedvalues, model_stl.resid, alpha=0.3)
plt.axhline(0)
plt.title("Residuals vs Fitted Values (Sale-to-List Ratio)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# time-series analysis

# Ensure datetime index
df_analysis["Month of Period End"] = pd.to_datetime(df_analysis["Month of Period End"])

# Aggregate to statewide monthly averages
ts_df = (
    df_analysis
    .groupby("Month of Period End")[["Days_on_Market_capped", "Average Sale To List"]]
    .mean()
)

# Long-run time-series trends

fig, ax1 = plt.subplots(figsize=(12, 6))

# Left axis: Days on Market
line1, = ax1.plot(
    ts_df.index,
    ts_df["Days_on_Market_capped"],
    color="steelblue",
    linewidth=2,
    label="Avg Days on Market"
)
ax1.set_ylabel("Avg Days on Market", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

# Right axis: Sale-to-List Ratio
ax2 = ax1.twinx()
line2, = ax2.plot(
    ts_df.index,
    ts_df["Average Sale To List"],
    color="darkorange",
    linewidth=2,
    label="Avg Sale-to-List Ratio"
)
ax2.set_ylabel("Avg Sale-to-List Ratio", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

# Title, legend, and grid
plt.title("Time-Series Trends: Days on Market vs Sale-to-List Ratio (Washington State)")
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Average seasonal profile - month of year effect

# Extract month-of-year
df_analysis["Month"] = df_analysis["Month of Period End"].dt.month

seasonal_month = (
    df_analysis
    .groupby("Month")[["Days_on_Market_capped", "Average Sale To List"]]
    .mean()
)

fig, ax1 = plt.subplots(figsize=(12, 6))

# Left axis- Days on Market
line1, = ax1.plot(
    seasonal_month.index,
    seasonal_month["Days_on_Market_capped"],
    color="steelblue",
    marker="o",
    linewidth=2,
    label="Avg Days on Market"
)
ax1.set_ylabel("Avg Days on Market", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

# Right axis- stl ratio
ax2 = ax1.twinx()
line2, = ax2.plot(
    seasonal_month.index,
    seasonal_month["Average Sale To List"],
    color="darkorange",
    marker="o",
    linewidth=2,
    label="Avg Sale-to-List Ratio"
)
ax2.set_ylabel("Avg Sale-to-List Ratio", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

# X-axis labels
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

# Title, legend, grid
plt.title("Average Seasonal Profile (Month-of-Year): Days on Market vs Sale-to-List Ratio")
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()