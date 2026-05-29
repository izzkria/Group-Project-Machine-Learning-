import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# ----------------------------
# Paths & Artifacts
# ----------------------------
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(ART_DIR, "results.csv")
MODEL_PATH = os.path.join(ART_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "feature_names.json")

# Dataset path
DEFAULT_MERGED_DATA_PATH = "price.csv"  # <- your dataset

# ----------------------------
# Columns
# ----------------------------
COL_DATE = "Date"
COL_PRICE = "Price"
COL_PROD = "Index Production"
COL_EXPORT = "Export Number (in Tonnes)"
COL_PRECIP = "Precip"
OPTIONAL_COLS = ["Temp", "Humidity", "USD"]

MAX_DATE_STR = "2022-05-31"
MAX_DATE = pd.to_datetime(MAX_DATE_STR)

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="Palm Oil Price Prediction App", layout="wide")
st.title("🌴 Palm Oil Price Forecasting Dashboard for Malaysia")
st.caption(f"Dashboard data is restricted to **up to {MAX_DATE_STR}** only.")

# ----------------------------
# Helpers
# ----------------------------
def stop_with_error(msg: str):
    st.error(msg)
    st.stop()

def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

@st.cache_data(show_spinner=False)
def load_results(path: str) -> pd.DataFrame:
    if not file_exists(path):
        return pd.DataFrame()  # return empty df if missing
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not file_exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_scaler(path: str):
    if not file_exists(path):
        return None
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_feature_names(path: str) -> List[str]:
    if not file_exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        feats = json.load(f)
    if not isinstance(feats, list) or not feats:
        return []
    return feats

def infer_best_model_name(results_df: pd.DataFrame) -> str:
    if results_df.empty or "Model" not in results_df.columns:
        return "N/A"
    if "MAE" in results_df.columns:
        return str(results_df.sort_values("MAE", ascending=True).iloc[0]["Model"])
    return str(results_df.iloc[0]["Model"])

def pick_best_row(results_df: pd.DataFrame) -> pd.Series:
    if results_df.empty:
        return pd.Series()
    if "MAE" in results_df.columns:
        return results_df.sort_values("MAE", ascending=True).iloc[0]
    return results_df.iloc[0]

# ----------------------------
# Load artifacts
# ----------------------------
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
# Load dataset
# ----------------------------
@st.cache_data(show_spinner=False)
def load_merged_dataset(path_or_file) -> pd.DataFrame:
    if not file_exists(path_or_file):
        stop_with_error(f"Missing dataset file: {path_or_file}")
    df = pd.read_csv(path_or_file)
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
    df = df.dropna(subset=[COL_DATE]).sort_values(COL_DATE).reset_index(drop=True)
    df = df[df[COL_DATE] <= MAX_DATE].copy()
    for c in [COL_PRICE, COL_PROD, COL_EXPORT, COL_PRECIP] + OPTIONAL_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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
    if model is None or scaler is None:
        stop_with_error("Model or scaler not loaded. Run training first.")
    row = [inputs[f] for f in feature_names]
    X = np.array(row, dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)
    yhat = model.predict(Xs)
    return float(np.array(yhat).ravel()[0])

# ----------------------------
# Tab structure
# ----------------------------
tab_dash, tab_compare, tab_pred = st.tabs(["📊 Dashboard", "🏆 Model Comparison", "🧠 Prediction"])

# ----------------------------
# TAB 1: DASHBOARD
# ----------------------------
with tab_dash:
    st.subheader("📊 Dashboard : Trend & Insight")
    merged_df = load_merged_dataset(DEFAULT_MERGED_DATA_PATH)
    
    df_monthly = merged_df.copy()
    df_monthly['Month'] = df_monthly[COL_DATE].dt.to_period('M').dt.to_timestamp()
    df_monthly = df_monthly.groupby('Month', as_index=False).agg({
        COL_PRICE: 'mean',
        COL_PROD: 'mean',
        COL_EXPORT: 'mean',
        COL_PRECIP: 'mean'
    })
    
    st.markdown("### Monthly Price Trend")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_monthly['Month'], df_monthly[COL_PRICE])
    ax.set_title("Palm Oil Price (Monthly Mean)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (RM)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# TAB 2: MODEL COMPARISON
# ----------------------------
with tab_compare:
    st.subheader("🏆 Model Comparison")
    if results_df.empty:
        st.info("No results found. Run the training script first.")
    else:
        st.dataframe(results_df, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", best_model_name)
        c2.metric("RMSE", f"{float(best_row.get('RMSE', np.nan)):.4f}")
        c3.metric("MAE", f"{float(best_row.get('MAE', np.nan)):.4f}")
        c4.metric("R²", f"{float(best_row.get('R2', best_row.get('R-squared', np.nan))):.4f}")

# ----------------------------
# TAB 3: PREDICTION
# ----------------------------
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

    defaults = compute_feature_defaults(DEFAULT_MERGED_DATA_PATH, feature_names)
    inputs: Dict[str, Any] = {}
    cols = st.columns(2, gap="large")

    for i, feat in enumerate(feature_names):
        default_val = defaults.get(feat, 0.0)
        with cols[i % 2]:
            if feat == COL_EXPORT:
                inputs[feat] = st.number_input(feat, value=int(round(default_val)), step=1, format="%d", key=f"single_{feat}")
            else:
                inputs[feat] = st.number_input(feat, value=float(default_val), step=0.1, key=f"single_{feat}")

    # Force year if exists
    year_feat = next((f for f in feature_names if f.strip().lower() == "year"), None)
    if year_feat:
        inputs[year_feat] = 2022

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
