"""Read-only dashboard for the cost-sensitive churn project.

Everything shown here is read from data/, artifacts/ and results/. Nothing is
retrained: the cost explorer recomputes FN/FP/cost from the saved
validation/test probabilities in artifacts/<model>_probs.csv.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.paths import ARTIFACTS_DIR, PLOTS_DIR, RAW_DATA, RESULTS_DIR  # noqa: E402

MODELS = {"logistic_regression": "Logistic Regression", "mlp": "MLP", "xgboost": "XGBoost"}
DEFAULT_C_FN, DEFAULT_C_FP = 500, 100
GRID = np.round(np.arange(0.01, 0.9901, 0.01), 2)  # same grid as the cost scripts
SENS_CFN = [100, 200, 300, 500, 1000, 2000]

BLUE, GREY, MUTED, AMBER, GREEN = "#2F6FB3", "#7B8794", "#52606D", "#B7791F", "#2F855A"
INK, GRID_LINE, ROW_HIGHLIGHT = "#1F2933", "#E5E9F0", "#E6EFF9"

CSS = """
<style>
[data-testid="stMetric"] {padding: 14px 16px; border: 1px solid #E2E8F0; border-radius: 8px;}
.placeholder-banner {padding: 6px 12px; border: 1px solid #E2E8F0; border-left: 4px solid #B7791F;
    border-radius: 8px; background: #FDF6E9; color: #1F2933; font-size: 0.9rem; margin-bottom: 1rem;}
#MainMenu, footer, [data-testid="stMainMenu"] {visibility: hidden;}
</style>
"""


# ---------------------------------------------------------------- loading

@st.cache_data
def load_json(name):
    return json.loads((ARTIFACTS_DIR / name).read_text())


@st.cache_data
def load_probs(model):
    return pd.read_csv(ARTIFACTS_DIR / f"{model}_probs.csv")


@st.cache_data
def load_raw_data():
    return pd.read_csv(RAW_DATA)


@st.cache_data
def load_text(path):
    return Path(path).read_text()


@st.cache_data
def load_lr_coefficients():
    obj = joblib.load(ARTIFACTS_DIR / "logistic_regression.joblib")
    coef = obj["model"].coef_.ravel()
    return pd.DataFrame({"feature": obj["feature_names"], "coefficient": coef})


def lower_keys(d):
    """The cost JSONs mix FN/fn and savings_vs_target_nobody/savings_vs_nobody; normalise."""
    out = {k.lower(): v for k, v in d.items()}
    if "savings_vs_target_nobody" in out:
        out.setdefault("savings_vs_nobody", out["savings_vs_target_nobody"])
    return out


# ---------------------------------------------------------------- cost maths

def confusion(y, p, t):
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def cost_curve(y, p, c_fn, c_fp):
    """FN, FP and total cost at every grid threshold (prediction = proba >= t)."""
    pred = p[None, :] >= GRID[:, None]
    fn = np.sum(~pred & (y[None, :] == 1), axis=1)
    fp = np.sum(pred & (y[None, :] == 0), axis=1)
    return pd.DataFrame({"threshold": GRID, "fn": fn, "fp": fp, "cost": fn * c_fn + fp * c_fp})


def best_threshold(curve):
    """Minimum cost; ties go to the lower threshold, as in the cost scripts."""
    return float(curve.sort_values(["cost", "threshold"]).iloc[0]["threshold"])


def split_arrays(model, split):
    df = load_probs(model)
    s = df[df["split"] == split]
    return s["y_true"].to_numpy(), s["proba"].to_numpy()


def evaluate(model, t, c_fn, c_fp, split="test"):
    y, p = split_arrays(model, split)
    tp, fp, fn, tn = confusion(y, p, t)
    cost = fn * c_fn + fp * c_fp
    nobody = int(np.sum(y == 1)) * c_fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": cost,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "nobody_cost": nobody, "everyone_cost": int(np.sum(y == 0)) * c_fp,
        "savings": nobody - cost,
    }


@st.cache_data
def sensitivity_table(c_fp):
    rows = []
    for key, name in MODELS.items():
        yv, pv = split_arrays(key, "val")
        for c_fn in SENS_CFN:
            t = best_threshold(cost_curve(yv, pv, c_fn, c_fp))
            r = evaluate(key, t, c_fn, c_fp)
            rows.append({"Model": name, "C_FN": c_fn, "C_FN / C_FP": c_fn / c_fp,
                         "Best threshold (val)": t, "Test cost": r["cost"],
                         "Test recall": r["recall"], "Missed churners": r["fn"], "False alarms": r["fp"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- presentation helpers

def page_header(title, description):
    st.title(title)
    st.markdown(description)
    st.markdown(
        '<div class="placeholder-banner"><b>Costs are placeholders.</b> '
        "C_FN = 500 and C_FP = 100 stand in for real business costs (stage 5 is still to do).</div>",
        unsafe_allow_html=True,
    )


def money(x):
    return f"{x:,.0f}"


def style_chart(fig, height=420):
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="sans-serif", size=14, color=INK),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID_LINE, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID_LINE, zeroline=False, showline=False)
    return fig


def show_table(df, formats, highlight_col=None, highlight_value=None, height=None):
    # st.dataframe shows missing numbers as "None", so swap them for a dash before formatting.
    df = df.astype(object).where(df.notna(), "–")
    formats = {c: (lambda v, f=f: v if isinstance(v, str) else f.format(v)) for c, f in formats.items()}
    styler = df.style.format(formats)
    if highlight_col is not None:
        styler = styler.apply(
            lambda row: [f"background-color: {ROW_HIGHLIGHT}" if row[highlight_col] == highlight_value else ""
                         for _ in row], axis=1)
    kwargs = {"height": height} if height else {}
    numeric = {c: st.column_config.Column(alignment="right") for c in formats}
    st.dataframe(styler, hide_index=True, width="stretch", column_config=numeric, **kwargs)


def cost_summary():
    """Model-comparison rows built from artifacts/cost_<model>.json."""
    rows = []
    for key, name in MODELS.items():
        d = load_json(f"cost_{key}.json")
        opt = lower_keys(d["test"]["cost_optimal"])
        rows.append({
            "key": key, "Model": name, "ROC-AUC": d["test_roc_auc"], "PR-AUC": d["test_pr_auc"],
            "Threshold": d["val_optimal_threshold"], "Val cost": d["val_min_cost"],
            "Test cost": opt["total_cost"], "Missed churners (FN)": opt["fn"],
            "False alarms (FP)": opt["fp"], "Recall": opt["recall"],
            "Savings vs nobody": opt["savings_vs_nobody"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pages

def page_overview():
    page_header("Overview", "Cost-sensitive, explainable churn prediction on the IBM Telco dataset: data, splits and progress.")
    raw = load_raw_data()
    churn_rate = (raw["Churn"] == "Yes").mean()
    sizes = load_json("xgboost_metrics.json")["split_sizes"]
    summary = cost_summary()
    selected = summary.sort_values("Val cost").iloc[0]

    c = st.columns(4)
    c[0].metric("Customers", f"{len(raw):,}")
    c[1].metric("Churn rate", f"{churn_rate:.1%}")
    c[2].metric("Split (train / val / test)", "70 / 15 / 15",
                f"{sizes['train']:,} / {sizes['val']:,} / {sizes['test']:,} customers", delta_color="off", delta_arrow="off")
    c[3].metric(f"Selected: {selected['Model']}", f"t = {selected['Threshold']:.2f}",
                "lowest validation cost", delta_color="off", delta_arrow="off")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Project status")
        st.caption("From the progress table in results/README.md.")
        readme = load_text(RESULTS_DIR / "README.md")
        block = readme.split("## Progress", 1)[1].split("\n## ", 1)[0]
        rows = [[c.strip().replace("**", "") for c in line.strip("|").split("|")]
                for line in block.splitlines() if line.startswith("| ") and not line.startswith("| Stage")]
        for stage, status, _ in rows:
            done = status.lower().startswith("done")
            mark = "[x]" if done else "[ ]"
            extra = status[4:].strip() if done and len(status) > 4 else ""
            st.markdown(f"- {mark} **{stage}** — {'Done' if done else status} {extra}")
    with right:
        st.subheader("How the data is used")
        st.markdown(
            f"- Stratified **70 / 15 / 15** split, `random_state=42`.\n"
            f"- **Validation** ({sizes['val']:,} customers) is used for all tuning and threshold choices.\n"
            f"- **Test** ({sizes['test']:,} customers) is scored once, for reporting only.\n"
            f"- Selected model: **{selected['Model']}** at threshold **{selected['Threshold']:.2f}**, "
            f"which has the lowest validation cost. The three models are statistically tied (paired bootstrap).\n"
            f"- Next: real business costs (stage 5) and SHAP explanations (stage 6)."
        )


def page_models():
    page_header("Model comparison",
                "Test-set results for the three models at their validation cost-optimal thresholds, against two baselines.")
    s = cost_summary()
    selected = s.sort_values("Val cost").iloc[0]["Model"]
    base = lower_keys(load_json("cost_logistic_regression.json")["baselines"]["target_nobody"])
    every = lower_keys(load_json("cost_logistic_regression.json")["baselines"]["target_everyone"])

    best = s[s["Model"] == selected].iloc[0]
    c = st.columns(4)
    c[0].metric(f"Test cost, {selected}", money(best["Test cost"]))
    c[1].metric("Recall", f"{best['Recall']:.1%}")
    c[2].metric("Savings vs contacting nobody", money(best["Savings vs nobody"]),
                f"{best['Savings vs nobody'] / base['total_cost']:.1%} lower cost", delta_arrow="off")
    c[3].metric("Contact-nobody cost", money(base["total_cost"]))

    table = s.drop(columns=["key", "Savings vs nobody"])
    table.insert(1, "Status", np.where(table["Model"] == selected, "Selected", ""))
    baselines = pd.DataFrame([
        {"Model": "Contact nobody", "Status": "Baseline", "Test cost": base["total_cost"],
         "Missed churners (FN)": base["fn"], "False alarms (FP)": base["fp"], "Recall": 0.0},
        {"Model": "Contact everyone", "Status": "Baseline", "Test cost": every["total_cost"],
         "Missed churners (FN)": every["fn"], "False alarms (FP)": every["fp"], "Recall": 1.0},
    ])
    table = pd.concat([table, baselines], ignore_index=True)
    show_table(table, {"ROC-AUC": "{:.3g}", "PR-AUC": "{:.3g}", "Threshold": "{:.2f}", "Val cost": "{:,.0f}",
                       "Test cost": "{:,.0f}", "Missed churners (FN)": "{:,.0f}", "False alarms (FP)": "{:,.0f}",
                       "Recall": "{:.1%}"},
               highlight_col="Status", highlight_value="Selected")
    st.caption(
        f"Costs use C_FN = {DEFAULT_C_FN} and C_FP = {DEFAULT_C_FP}. The selected model has the lowest validation cost; "
        "test cost is reported once and never used for selection. ROC-AUC and PR-AUC are on the test set. "
        "Source: artifacts/cost_<model>.json."
    )


def page_cost_explorer():
    page_header("Cost explorer",
                "Change the costs and the decision threshold to see how missed churners, false alarms and total cost respond.")
    ctrl, info = st.columns(2, gap="large")
    with ctrl:
        model = st.selectbox("Model", list(MODELS), format_func=MODELS.get)
        c_fn = st.slider("C_FN: cost of a missed churner", 50, 3000, DEFAULT_C_FN, step=50)
        c_fp = st.slider("C_FP: cost of an unneeded retention offer", 10, 1000, DEFAULT_C_FP, step=10)

    yv, pv = split_arrays(model, "val")
    curve = cost_curve(yv, pv, c_fn, c_fp)
    t_opt = best_threshold(curve)
    t_theory = c_fp / (c_fp + c_fn)

    with info:
        use_opt = st.checkbox("Use the validation-optimal threshold", value=True)
        t = st.slider("Decision threshold", 0.01, 0.99, t_opt, step=0.01, disabled=use_opt,
                      help="A customer is flagged as a churner when the predicted probability is at or above this value.")
        if use_opt:
            t = t_opt
        st.markdown(
            f"Validation-optimal threshold: **{t_opt:.2f}** (validation cost {money(curve.loc[curve.threshold == t_opt, 'cost'].iloc[0])})  \n"
            f"Theoretical C_FP / (C_FP + C_FN): **{t_theory:.2f}**  \n"
            f"Threshold in use: **{t:.2f}**"
        )

    st.info("The optimal threshold is chosen on the **validation set only**. Test-set numbers below are for reporting "
            "and are never used to pick the threshold.")

    r = evaluate(model, t, c_fn, c_fp, "test")
    c = st.columns(4)
    c[0].metric("Test cost", money(r["cost"]))
    c[1].metric("Recall", f"{r['recall']:.1%}")
    c[2].metric("Precision", f"{r['precision']:.1%}")
    c[3].metric("Savings vs contacting nobody", money(r["savings"]),
                f"{r['savings'] / r['nobody_cost']:.1%} lower cost" if r["savings"] > 0 else "no saving",
                delta_color="normal" if r["savings"] > 0 else "off", delta_arrow="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.threshold, y=curve.cost, mode="lines", name="Validation cost",
                             line=dict(color=BLUE, width=2),
                             customdata=np.c_[curve.fn, curve.fp],
                             hovertemplate="Threshold %{x:.2f}<br>Cost %{y:,.0f}<br>FN %{customdata[0]} · FP %{customdata[1]}<extra></extra>"))
    ymax = curve.cost.max()
    fig.add_trace(go.Scatter(x=[0.5, 0.5], y=[0, ymax], mode="lines", name="Default 0.50",
                             line=dict(color=GREY, width=1.5, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[t_theory, t_theory], y=[0, ymax], mode="lines",
                             name=f"Theoretical C_FP/(C_FP+C_FN) = {t_theory:.2f}",
                             line=dict(color=AMBER, width=2, dash="dot"), hoverinfo="skip"))
    if not use_opt:
        fig.add_trace(go.Scatter(x=[t, t], y=[0, ymax], mode="lines", name=f"Chosen threshold {t:.2f}",
                                 line=dict(color=MUTED, width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[t_opt], y=[curve.loc[curve.threshold == t_opt, "cost"].iloc[0]], mode="markers+text",
                             name=f"Validation-optimal threshold {t_opt:.2f}",
                             marker=dict(color=GREEN, size=11, symbol="diamond", line=dict(color="white", width=2)),
                             text=[f"optimal {t_opt:.2f}"], textposition="bottom center", textfont=dict(color=INK)))
    fig.update_xaxes(title="Decision threshold", range=[0, 1], tickvals=np.round(np.arange(0.1, 1.0, 0.1), 1),
                     tickformat=".1f")
    fig.update_yaxes(title="Total cost (validation)", rangemode="tozero", tickformat=",")
    st.subheader(f"Validation cost by threshold: {MODELS[model]}")
    st.plotly_chart(style_chart(fig), width="stretch", theme=None)

    st.subheader(f"Test set at threshold {t:.2f}")
    table = pd.DataFrame([{
        "Missed churners (FN)": r["fn"], "False alarms (FP)": r["fp"], "Churners caught (TP)": r["tp"],
        "Correctly left alone (TN)": r["tn"], "Total cost": r["cost"],
        "Contact-nobody cost": r["nobody_cost"], "Contact-everyone cost": r["everyone_cost"],
    }])
    show_table(table, {c: "{:,.0f}" for c in table.columns})


def page_sensitivity():
    page_header("Sensitivity",
                "How the best validation threshold and the test cost move as the cost of a missed churner (C_FN) changes.")
    c_fp = st.slider("C_FP: cost of an unneeded retention offer", 10, 1000, DEFAULT_C_FP, step=10)
    df = sensitivity_table(c_fp)

    styles = {"Logistic Regression": dict(color=BLUE, dash="solid"),
              "MLP": dict(color=GREY, dash="dash"), "XGBoost": dict(color=MUTED, dash="dot")}
    symbols = {"Logistic Regression": "circle", "MLP": "square", "XGBoost": "triangle-up"}
    left, right = st.columns(2, gap="large")
    for col, metric, title, fmt in [(left, "Best threshold (val)", "Best threshold", ".2f"),
                                    (right, "Test cost", "Test cost", ",.0f")]:
        fig = go.Figure()
        for name in MODELS.values():
            d = df[df.Model == name]
            fig.add_trace(go.Scatter(x=d.C_FN, y=d[metric], mode="lines+markers", name=name,
                                     line=dict(width=2, **styles[name]),
                                     marker=dict(size=8, symbol=symbols[name], color=styles[name]["color"]),
                                     hovertemplate=f"{name}<br>C_FN %{{x:,}}<br>{title} %{{y:{fmt}}}<extra></extra>"))
        fig.update_xaxes(title="Cost of a missed churner (C_FN)", type="log", tickvals=SENS_CFN,
                         ticktext=[f"{v:,}" for v in SENS_CFN])
        fig.update_yaxes(title=title, tickformat=fmt, rangemode="tozero")
        with col:
            st.subheader(title)
            st.plotly_chart(style_chart(fig, 380), width="stretch", theme=None)

    show_table(df, {"C_FN": "{:,.0f}", "C_FN / C_FP": "{:.3g}", "Best threshold (val)": "{:.2f}",
                    "Test cost": "{:,.0f}", "Test recall": "{:.1%}", "Missed churners": "{:,.0f}",
                    "False alarms": "{:,.0f}"},
               highlight_col="Model", highlight_value="Logistic Regression", height=35 * (len(df) + 1) + 3)
    st.caption("Thresholds are chosen on validation for each C_FN; test cost is then reported at that threshold. "
               "Computed live from artifacts/<model>_probs.csv. Logistic Regression is the selected model (highlighted).")


# Rows of plots: compact figures share a row, the wide multi-panel figures get the full width.
EDA_PLOTS = [
    [("churn_distribution.png", "Churn distribution: 26.5% of customers churned, so the classes are imbalanced."),
     ("correlation_heatmap.png", "Correlation between the numeric features and churn.")],
    [("churn_rate_by_category.png", "Churn rate for every categorical feature.")],
    [("numeric_distributions.png", "Tenure, MonthlyCharges and TotalCharges distributions by churn.")],
    [("numeric_boxplots.png", "Numeric features against churn.")],
]


@st.cache_data
def churn_rates_by_category():
    text = load_text(RESULTS_DIR / "eda_report.txt")
    block = text.split("CHURN RATE BY CATEGORY (%)", 1)[1].split("=" * 70, 2)[1]
    rows, feature = [], None
    for line in block.splitlines():
        line = line.rstrip()
        if not line or line == feature:
            continue
        if line.endswith(":"):
            feature = line[:-1]
            continue
        m = re.match(r"(.+?)\s+([\d.]+)$", line)
        if m and feature:
            rows.append({"Feature": feature, "Category": m.group(1), "Churn rate": float(m.group(2)) / 100})
    return pd.DataFrame(rows)


def page_eda():
    page_header("Exploratory data analysis", "Charts and churn rates from the EDA step (results/plots and results/eda_report.txt).")
    rates = churn_rates_by_category()
    top = rates.sort_values("Churn rate", ascending=False).head(8)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Highest churn rates by category")
        show_table(top, {"Churn rate": "{:.1%}"})
    with right:
        st.subheader("Look up a feature")
        feat = st.selectbox("Feature", rates["Feature"].unique(), index=int(np.argmax(rates["Feature"].unique() == "Contract")))
        show_table(rates[rates.Feature == feat].sort_values("Churn rate", ascending=False), {"Churn rate": "{:.1%}"})

    for row in EDA_PLOTS:
        cols = st.columns(len(row), gap="large")
        for col, (fname, caption) in zip(cols, row):
            with col:
                st.image(str(PLOTS_DIR / fname), width="stretch")
                st.caption(caption)


REPORTS = ["logistic_regression.md", "mlp.md", "xgboost.md", "cost_logistic_regression.md", "cost_mlp.md",
           "cost_xgboost.md", "cost_comparison.md"]


def render_markdown(md):
    """st.markdown cannot resolve relative image links, so render those with st.image."""
    for part in re.split(r"(!\[[^\]]*\]\([^)]+\))", md):
        m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", part)
        if m:
            path = RESULTS_DIR / m.group(2)
            if path.exists():
                st.image(str(path), width="stretch")
                st.caption(m.group(1))
        elif part.strip():
            st.markdown(part)


def page_reports():
    page_header("Reports", "The generated markdown reports from results/, rendered as written.")
    name = st.selectbox("Report", REPORTS)
    with st.container(border=True):
        render_markdown(load_text(RESULTS_DIR / name))


def page_coefficients():
    page_header("Logistic Regression coefficients",
                "The features that push the selected model's churn prediction up or down most strongly.")
    n = st.slider("Features per side", 5, 15, 10)
    coefs = load_lr_coefficients()
    top = pd.concat([coefs.nlargest(n, "coefficient"), coefs.nsmallest(n, "coefficient")])
    top = top.sort_values("coefficient")
    raises = top.coefficient > 0
    fig = go.Figure()
    for mask, name, color in [(~raises, "Lowers churn odds", GREY), (raises, "Raises churn odds", BLUE)]:
        d = top[mask]
        fig.add_trace(go.Bar(x=d.coefficient, y=d.feature, orientation="h", name=name, marker_color=color,
                             hovertemplate="%{y}<br>coefficient %{x:.3f}<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.25)
    fig.update_xaxes(title="Coefficient (log-odds per unit)")
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=list(top.feature))
    st.plotly_chart(style_chart(fig, 28 * len(top) + 140), width="stretch", theme=None)
    st.caption("Continuous features are standardised (z-scores), binary flags are 0/1, so magnitudes are comparable "
               "within each group. Coefficients describe association in this model, not causation. "
               "Source: artifacts/logistic_regression.joblib.")


PAGES = {
    "Overview": page_overview,
    "Model comparison": page_models,
    "Cost explorer": page_cost_explorer,
    "Sensitivity": page_sensitivity,
    "EDA": page_eda,
    "Reports": page_reports,
    "LR coefficients": page_coefficients,
}


def main():
    st.set_page_config(page_title="Churn cost dashboard", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**Churn prediction**")
        page = st.radio("Page", list(PAGES), label_visibility="collapsed")
        st.caption("Read-only view of the saved results. Nothing is retrained.")
    PAGES[page]()


main()
