# app.py
# ✅ Streamlit app (FAST DEMO): loads saved artifacts + merged dataset for dashboard
# Tabs: Dashboard, Model Comparison, Prediction (inputs + outcome together)

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# ----------------------------
# Paths
# ----------------------------
ART_DIR = "artifacts"
RESULTS_PATH = os.path.join(ART_DIR, "results.csv")
MODEL_PATH = os.path.join(ART_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "feature_names.json")

DEFAULT_MERGED_DATA_PATH = os.path.join("data", "final_merged_palm_oil_dataset.csv")

# ----------------------------
# Column names
# ----------------------------
COL_DATE = "Date"
COL_PRICE = "Price"
COL_PROD = "Index Production"
COL_EXPORT = "Export Number (in Tonnes)"
COL_PRECIP = "Precip"
OPTIONAL_COLS = ["Temp", "Humidity", "USD"]

# ----------------------------
# Hard cut-off date (ONLY up to 31-05-2022)
# ----------------------------
MAX_DATE_STR = "2022-05-31"
MAX_DATE = pd.to_datetime(MAX_DATE_STR)

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Palm Oil Price Prediction App", layout="wide")
st.title("🌴 Palm Oil Price Forecasting Dashboard for Malaysia")
st.caption(f"Dashboard data is restricted to **up to {MAX_DATE_STR}** only.")

# ----------------------------
# Utility helpers
# ----------------------------
def stop_with_error(msg: str):
    st.error(msg)
    st.stop()

def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

@st.cache_data(show_spinner=False)
def load_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_feature_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        feats = json.load(f)
    if not isinstance(feats, list) or not feats:
        raise ValueError("feature_names.json must be a non-empty list of feature names.")
    return feats

def infer_best_model_name(results_df: pd.DataFrame) -> str:
    if "Model" not in results_df.columns:
        return "Best Model"
    if "RMSE" in results_df.columns:
        return str(results_df.sort_values("RMSE", ascending=True).iloc[0]["Model"])
    if "R-squared" in results_df.columns:
        return str(results_df.sort_values("R-squared", ascending=False).iloc[0]["Model"])
    if "R2" in results_df.columns:
        return str(results_df.sort_values("R2", ascending=False).iloc[0]["Model"])
    return str(results_df.iloc[0]["Model"])

def pick_best_row(results_df: pd.DataFrame) -> pd.Series:
    if "RMSE" in results_df.columns:
        return results_df.sort_values("RMSE", ascending=True).iloc[0]
    if "MAE" in results_df.columns:
        return results_df.sort_values("MAE", ascending=True).iloc[0]
    if "R-squared" in results_df.columns:
        return results_df.sort_values("R-squared", ascending=False).iloc[0]
    if "R2" in results_df.columns:
        return results_df.sort_values("R2", ascending=False).iloc[0]
    return results_df.iloc[0]

# ----------------------------
# Load artifacts (REQUIRED)
# ----------------------------
missing = [p for p in [RESULTS_PATH, MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not file_exists(p)]
if missing:
    stop_with_error(
        "Missing artifact files:\n\n"
        + "\n".join([f"- {m}" for m in missing])
        + "\n\nFix: run your training code ONCE to generate the `artifacts/` folder."
    )

results_df = load_results(RESULTS_PATH)
model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)
feature_names = load_feature_names(FEATURES_PATH)

best_model_name = infer_best_model_name(results_df)
best_row = pick_best_row(results_df)

# ----------------------------
# Session state
# ----------------------------
if "single_pred_value" not in st.session_state:
    st.session_state.single_pred_value = None
if "single_pred_inputs" not in st.session_state:
    st.session_state.single_pred_inputs = None

# ----------------------------
# Dashboard helpers (merged dataset)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_merged_dataset(path_or_file) -> pd.DataFrame:
    df = pd.read_csv(path_or_file) if isinstance(path_or_file, str) else pd.read_csv(path_or_file)

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE]).sort_values(COL_DATE).reset_index(drop=True)

    # 🔥 Hard cutoff (ONLY up to 31-05-2022)
    df = df[df[COL_DATE] <= MAX_DATE].copy()

    for c in [COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP] + OPTIONAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def daily_price_trend_plot(df: pd.DataFrame):
    d = df.copy()
    d[COL_DATE] = pd.to_datetime(d[COL_DATE], errors="coerce")
    d = d.dropna(subset=[COL_DATE]).sort_values(COL_DATE)
    d = d[d[COL_DATE] <= MAX_DATE]

    d[COL_PRICE] = pd.to_numeric(d[COL_PRICE], errors="coerce").ffill()
    d = d.dropna(subset=[COL_PRICE])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(d[COL_DATE], d[COL_PRICE])
    ax.set_title(f"Palm Oil Price Trend (Daily) — up to {MAX_DATE_STR}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (RM)")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True)

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df[COL_DATE] <= MAX_DATE].copy()
    df["Month"] = df[COL_DATE].dt.to_period("M").dt.to_timestamp()

    agg = {
        COL_PRICE: "mean",
        COL_PROD: "mean",
        COL_EXPORT: "mean",
        COL_PRECIP: "mean",
    }
    for c in OPTIONAL_COLS:
        if c in df.columns:
            agg[c] = "mean"

    out = df.groupby("Month", as_index=False).agg(agg)
    out = out[out["Month"] <= MAX_DATE].copy()
    return out

def line_plot(x, y, title, y_label, x_label="Month"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def scatter_plot(x, y, title, x_label, y_label):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x, y, s=18)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

def corr_heatmap(df: pd.DataFrame, cols: List[str], title: str):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        st.warning("Not enough numeric columns available for correlation heatmap.")
        return

    cdf = df[cols].copy()
    corr = cdf.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(corr.values)

    ax.set_title(title)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# Prediction helpers
# ----------------------------
def compute_feature_defaults(merged_path: str, feats: List[str]) -> Dict[str, float]:
    df = load_merged_dataset(merged_path)
    defaults: Dict[str, float] = {}
    for f in feats:
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce").dropna()
            defaults[f] = float(s.mean()) if len(s) else 0.0
        else:
            defaults[f] = 0.0
    return defaults

def predict_one(inputs: Dict[str, Any]) -> float:
    row = []
    for f in feature_names:
        if f not in inputs:
            raise ValueError(f"Missing input feature: {f}")
        row.append(inputs[f])

    X = np.array(row, dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)
    yhat = model.predict(Xs)
    return float(np.array(yhat).ravel()[0])

# ----------------------------
# Tabs
# ----------------------------
tab_dash, tab_compare, tab_pred = st.tabs(["📊 Dashboard", "🏆 Model Comparison", "🧠 Prediction"])

# ============================================================
# TAB 1: DASHBOARD (Monthly first, then Daily; remove monthly price trend plot)
# ============================================================
with tab_dash:
    st.subheader("📊 Dashboard : Trends & Insights")

    if not os.path.exists(DEFAULT_MERGED_DATA_PATH):
        stop_with_error(f"Missing merged dataset file: {DEFAULT_MERGED_DATA_PATH}")

    try:
        merged_df = load_merged_dataset(DEFAULT_MERGED_DATA_PATH)
    except Exception as e:
        stop_with_error(f"Failed to load merged dataset: {e}")

    required = [COL_DATE, COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP]
    miss = [c for c in required if c not in merged_df.columns]
    if miss:
        stop_with_error("Merged dataset missing required columns:\n" + "\n".join([f"- {m}" for m in miss]))

    if merged_df.empty:
        stop_with_error(f"No data available after cutoff ({MAX_DATE_STR}). Check your dataset dates.")

    # ----------------------------
    # 1) MONTHLY analysis FIRST
    # ----------------------------
    df_monthly = to_monthly(merged_df)

    if df_monthly.empty:
        st.warning("Monthly data is empty after aggregation.")
        st.stop()

    min_m = df_monthly["Month"].min()
    max_m = df_monthly["Month"].max()

    start_date, end_date = st.date_input(
        "Month range",
        value=(min_m.date(), max_m.date()),
        min_value=min_m.date(),
        max_value=max_m.date(),
    )

    mask = (df_monthly["Month"].dt.date >= start_date) & (df_monthly["Month"].dt.date <= end_date)
    df_m = df_monthly.loc[mask].copy()

    if df_m.empty:
        st.warning("No data in that range.")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Months", f"{len(df_m):,}")
    k2.metric("Average Price (Monthly)", f"RM {df_m[COL_PRICE].mean():,.2f}")
    k3.metric("Average Production (Monthly)", f"{df_m[COL_PROD].mean():,.2f}")
    k4.metric("Avg Monthly Export (Tonnes)", f"{df_m[COL_EXPORT].mean():,.0f}")

    st.divider()

    # ----------------------------
    # 2) DAILY price trend SECOND
    # ----------------------------
    st.markdown("### Palm Oil Daily Price Trend")
    st.caption(f"Daily trend is restricted to **up to {MAX_DATE_STR}** only.")
    daily_price_trend_plot(merged_df)


    # ✅ REMOVED: Palm Oil Monthly Price Trend (because we already show daily price)
    st.markdown("### Production Monthly Index Trend")
    line_plot(df_m["Month"], df_m[COL_PROD], "Production Index (Monthly Mean)", "Production Index", x_label="Month")

    st.markdown("### Export Monthly Volume Trend")
    line_plot(df_m["Month"], df_m[COL_EXPORT], "Export Volume (Monthly Average)", "Export (Tonnes)", x_label="Month")

    st.divider()

    st.markdown("### Price–Production Relationship (Monthly)")
    scatter_plot(df_m[COL_PROD], df_m[COL_PRICE], "Palm Oil Price vs Production (Monthly)", "Production Index", "Price (RM)")

    st.markdown("### Rainfall–Production Relationship (Monthly)")
    scatter_plot(df_m[COL_PRECIP], df_m[COL_PROD], "Rainfall vs Production (Monthly)", "Rainfall", "Production Index")

    st.divider()

    st.markdown("### Feature Correlation Overview (Monthly)")
    heat_cols = [COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP] + [c for c in OPTIONAL_COLS if c in df_m.columns]
    corr_heatmap(df_m.dropna(subset=heat_cols), heat_cols, "Monthly Feature Correlation Matrix")

    with st.expander("View Monthly Dataset"):
        st.dataframe(df_m, use_container_width=True)

    st.divider()


# ============================================================
# TAB 2: MODEL COMPARISON
# ============================================================
with tab_compare:
    st.subheader("🏆 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_model_name)

    if "RMSE" in results_df.columns:
        c2.metric("RMSE", f"{float(best_row.get('RMSE', np.nan)):.4f}")
    if "MAE" in results_df.columns:
        c3.metric("MAE", f"{float(best_row.get('MAE', np.nan)):.4f}")

    if "R-squared" in results_df.columns:
        c4.metric("R²", f"{float(best_row.get('R-squared', np.nan)):.4f}")
    elif "R2" in results_df.columns:
        c4.metric("R²", f"{float(best_row.get('R2', np.nan)):.4f}")

    if "Model" in results_df.columns and "RMSE" in results_df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(results_df["Model"], results_df["RMSE"])
        ax.set_title("RMSE by Model (Lower is better)")
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE")
        plt.xticks(rotation=25, ha="right")
        st.pyplot(fig, clear_figure=True)

# ============================================================
# TAB 3: PREDICTION
# ============================================================
with tab_pred:
    st.subheader("🧠 Palm Oil Price Estimation")

    st.markdown(
        f"""
        <div style="padding:12px 14px; border-radius:14px; background:#f3f6ff; border:1px solid #dbe4ff;">
          <div style="font-size:20px; font-weight:850;">Model used: {best_model_name}</div>
          <div style="font-size:12.5px; opacity:0.8;">
            Defaults are derived from dataset mean (restricted to up to {MAX_DATE_STR}).
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    if not os.path.exists(DEFAULT_MERGED_DATA_PATH):
        stop_with_error(f"Missing merged dataset file: {DEFAULT_MERGED_DATA_PATH}")

    defaults = compute_feature_defaults(DEFAULT_MERGED_DATA_PATH, feature_names)

    st.markdown("### Set Parameters for Single Prediction")
    st.caption(
        "All parameters are pre-filled using mean values from the dataset (up to 31-05-2022). "
        "Adjust any value to test different conditions."
    )

    inputs: Dict[str, Any] = {}
    cols = st.columns(2, gap="large")

    for i, feat in enumerate(feature_names):
        if feat.strip().lower() == "year":
            continue

        default_val = defaults.get(feat, 0.0)

        with cols[i % 2]:
            if feat == COL_EXPORT:
                inputs[feat] = st.number_input(
                    feat,
                    value=int(round(default_val)),
                    step=1,
                    format="%d",
                    key=f"single_{feat}"
                )
            else:
                inputs[feat] = st.number_input(
                    feat,
                    value=float(default_val),
                    step=0.1,
                    key=f"single_{feat}"
                )

    year_feat = next((f for f in feature_names if f.strip().lower() == "year"), None)
    if year_feat is not None:
        inputs[year_feat] = 2022

    st.write("")

    c1, c2 = st.columns([1, 1])
    with c1:
        do_pred = st.button("Predict Price ✅")
    with c2:
        reset = st.button("Reset output")

    if reset:
        st.session_state.single_pred_value = None
        st.session_state.single_pred_inputs = None

    if do_pred:
        try:
            pred = predict_one(inputs)
            st.session_state.single_pred_value = pred
            st.session_state.single_pred_inputs = inputs
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()
    st.markdown("### Outcome")

    if st.session_state.single_pred_value is None:
        st.info("No prediction yet. Adjust values and click Predict.")
    else:
        st.metric("Predicted Palm Oil Price per Tonne", f"RM {st.session_state.single_pred_value:,.2f}")
        with st.expander("Show inputs used"):
            st.json(st.session_state.single_pred_inputs)
