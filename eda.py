"""Missing-value handling and basic EDA for the Telco customer churn dataset."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "data.csv"
CLEAN_PATH = ROOT / "data" / "data_clean.csv"
PLOTS_DIR = ROOT / "results" / "plots"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_and_clean(path):
    df = pd.read_csv(path)

    section("RAW DATA")
    print(f"Shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate customerIDs: {df['customerID'].duplicated().sum()}")

    # Blank strings (" ") are not seen as NaN by read_csv, so normalise them first.
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    # TotalCharges is read as text because of those blanks; make it numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    section("MISSING VALUES (before)")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() or "None")
    print("\nRows with missing TotalCharges:")
    print(df.loc[df["TotalCharges"].isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

    # Every missing TotalCharges row has tenure == 0: the customer just joined and
    # has not been billed yet, so the correct total is 0 (not the mean/median).
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    section("MISSING VALUES (after)")
    print(f"Total missing cells: {df.isna().sum().sum()}")

    # SeniorCitizen is a 0/1 flag; treat it as categorical like the other Yes/No columns.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def describe(df):
    categorical = [c for c in df.columns if c not in NUMERIC_COLS + ["customerID"]]

    section("DTYPES")
    print(df.dtypes.to_string())

    section("NUMERIC SUMMARY")
    print(df[NUMERIC_COLS].describe().round(2).to_string())

    section("CHURN DISTRIBUTION")
    counts = df["Churn"].value_counts()
    print(pd.DataFrame({"count": counts, "pct": (counts / len(df) * 100).round(2)}).to_string())

    section("CHURN RATE BY CATEGORY (%)")
    churn_flag = df["Churn"].eq("Yes")
    for col in categorical:
        if col == "Churn":
            continue
        rate = churn_flag.groupby(df[col]).mean().mul(100).round(1).sort_values(ascending=False)
        print(f"\n{col}:\n{rate.to_string()}")

    section("NUMERIC MEANS BY CHURN")
    print(df.groupby("Churn")[NUMERIC_COLS].mean().round(2).to_string())

    section("CORRELATION (numeric + churn flag)")
    corr = df[NUMERIC_COLS].assign(Churn=churn_flag.astype(int)).corr().round(3)
    print(corr.to_string())
    return categorical, corr


def plot(df, categorical, corr):
    PLOTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_title("Churn distribution")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "churn_distribution.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.histplot(data=df, x=col, hue="Churn", kde=True, element="step", ax=ax)
        ax.set_title(f"{col} by churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "numeric_distributions.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, NUMERIC_COLS):
        sns.boxplot(data=df, x="Churn", y=col, ax=ax)
        ax.set_title(f"{col} vs churn")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "numeric_boxplots.png", dpi=120)
    plt.close(fig)

    cats = [c for c in categorical if c != "Churn"]
    ncols = 4
    nrows = -(-len(cats) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    churn_flag = df["Churn"].eq("Yes")
    for ax, col in zip(axes.flat, cats):
        rate = churn_flag.groupby(df[col]).mean().mul(100).sort_values(ascending=False)
        sns.barplot(x=rate.index, y=rate.values, ax=ax)
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Churn rate (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes.flat[len(cats):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "churn_rate_by_category.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    print(f"\nPlots saved to {PLOTS_DIR}/")


def main():
    df = load_and_clean(DATA_PATH)
    categorical, corr = describe(df)
    plot(df, categorical, corr)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned data saved to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
