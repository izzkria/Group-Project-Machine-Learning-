import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Palm Oil Price Prediction", layout="wide")
st.title("Palm Oil Price Prediction System")
st.markdown("Pure Machine Learning Regression (No Time-Series Features)")

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("price.csv", parse_dates=["Date"])

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==============================
# Feature Selection
# ==============================
target = "Price"
features = [c for c in df.columns if c not in ["Date", target]]

X = df[features]
y = df[target]

# ==============================
# Train/Test Split
# ==============================
split_ratio = st.slider("Training Data Percentage", 60, 90, 80)

X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
    X, y, df["Date"], test_size=(100 - split_ratio) / 100, shuffle=True, random_state=42
)

# ==============================
# Scaling
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Models
# ==============================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42)
}

param_grids = {
    "Decision Tree": {"max_depth": [3, 5, 10] ,'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
       'max_depth': [None, 5, 10, 15, 20],
       'min_samples_split': [2, 5, 10],
       'min_samples_leaf': [1, 2, 4]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10], 
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1],'criterion': ['friedman_mse', 'squared_error', 'absolute_error'],'max_depth': [3, 5]},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5, 7],
    'learning_rate': [0.01, 0.1]}
}

run_tuned = st.sidebar.checkbox("Run tuned models (GridSearch)", True)

# ==============================
# Train Models
# ==============================
results = []
predictions = {}

for name, model in models.items():

    # ---------- Non-tuned ----------
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "Type": "Non-Tuned",
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "Training Time (s)": train_time
    })

    predictions[f"{name} (Non-Tuned)"] = y_pred

    # ---------- Tuned ----------
    if run_tuned and name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring="r2", n_jobs=-1)

        start = time.time()
        grid.fit(X_train_scaled, y_train)
        train_time = time.time() - start

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        results.append({
            "Model": name,
            "Type": "Tuned",
            "R2": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "Training Time (s)": train_time
        })

        predictions[f"{name} (Tuned)"] = y_pred

# ==============================
# Results Table
# ==============================
st.subheader("Model Performance Comparison")
results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
st.dataframe(results_df)

# ==============================
# Prediction Chart
# ==============================
st.subheader("Actual vs Predicted Price")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=date_test,
    y=y_test,
    mode="markers",
    name="Actual Price"
))

for name, pred in predictions.items():
    fig.add_trace(go.Scatter(
        x=date_test,
        y=pred,
        mode="markers",
        name=name
    ))

fig.update_layout(
    title="Palm Oil Price Prediction (Machine Learning Only)",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)
