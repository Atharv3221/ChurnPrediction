"""Build all-result.md: one detailed document with every result, the progress and all plots.

Stitches together results/README.md (overview + progress), the EDA and feature
reports, the per-model reports and the cost-sensitive reports. Headings are
demoted one level. The file is written to results/, so links to plots/ and the
other reports stay relative and need no rewriting.
Rerun after the pipeline (run_pipeline.sh does this automatically).
"""
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on the path when run as a script
from churn.paths import RESULTS_DIR  # noqa: E402

RES = RESULTS_DIR
OUT = RESULTS_DIR / "all-result.md"

MODELS = [("logistic_regression", "Logistic Regression"), ("mlp", "MLP"), ("xgboost", "XGBoost")]
EDA_PLOTS = [
    ("churn_distribution.png", "Churn distribution (26.5% churn, imbalanced)"),
    ("numeric_distributions.png", "Tenure, MonthlyCharges and TotalCharges distributions by churn"),
    ("numeric_boxplots.png", "Numeric features vs churn"),
    ("churn_rate_by_category.png", "Churn rate for every categorical feature"),
    ("correlation_heatmap.png", "Correlation between numeric features and churn"),
]


def embed_md(path, demote=1):
    """Return a results/ markdown file with headings demoted."""
    out, in_code = [], False
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code and re.match(r"#{1,5} ", line):
            line = "#" * demote + line
        out.append(line)
    return "\n".join(out).strip() + "\n"


def embed_txt(path):
    return f"```text\n{path.read_text().rstrip()}\n```\n"


def main():
    sections = [
        ("1. Project Overview & Progress", "overview"),
        ("2. Exploratory Data Analysis", "eda"),
        ("3. Feature Extraction & Scaling", "features"),
        ("4. Model Results", "models"),
        ("5. Cost-Sensitive Analysis", "cost"),
        ("6. Reproducing the Results", "repro"),
    ]
    anchor = lambda t: re.sub(r"[^\w\- ]", "", t.lower()).replace(" ", "-")
    parts = [
        "# Customer Churn Prediction – All Results\n",
        f"_Generated {date.today().isoformat()} by `build_all_results.py` from the reports in `results/`. "
        "Do not edit by hand; rerun the script instead._\n",
        "## Contents\n",
        "\n".join(f"- [{t}](#{anchor(t)})" for t, _ in sections) + "\n",
    ]

    parts.append(f"## {sections[0][0]}\n")
    parts.append(embed_md(RES / "README.md", demote=2))

    parts.append(f"## {sections[1][0]}\n")
    parts.append("### Plots\n")
    for fname, caption in EDA_PLOTS:
        parts.append(f"**{caption}**\n\n![{caption}](plots/{fname})\n")
    parts.append("### Full EDA report (missing values, summary statistics, churn rate by category, correlations)\n")
    parts.append(embed_txt(RES / "eda_report.txt"))

    parts.append(f"## {sections[2][0]}\n")
    parts.append(embed_txt(RES / "features_report.txt"))

    parts.append(f"## {sections[3][0]}\n")
    for key, _ in MODELS:
        parts.append(embed_md(RES / f"{key}.md", demote=2))

    parts.append(f"## {sections[4][0]}\n")
    parts.append(embed_md(RES / "cost_comparison.md", demote=2))
    for key, _ in MODELS:
        parts.append(embed_md(RES / f"cost_{key}.md", demote=2))

    parts.append(f"## {sections[5][0]}\n")
    parts.append(
        "```bash\n"
        "./run_pipeline.sh                        # full pipeline (EDA → features → models → cost → this file)\n"
        "./run_pipeline.sh --c-fn 800 --c-fp 50   # with real business costs\n"
        "```\n"
    )

    OUT.write_text("\n".join(parts))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
