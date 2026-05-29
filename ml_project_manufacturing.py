# -*- coding: utf-8 -*-
"""
BSD3523 MACHINE LEARNING GROUP PROJECT

Group Name: CSM1

Group Leader:
YIP YOONG ENG (SD23048)

Group Members:
MUHAMMAD AMIRUL AMIER BIN MOHD HUSNI (SD23011)
ALIYA AFIFAH BINTI AL ABAS (SD23062)
NUR IZZATI BINTI ZAKARIA (SD23007)
ALIA AYUNNI BINTI MOHD SHUKRI (SD23054)

Run this script locally. Place all 7 CSV files in the same folder:
  weather.csv, price_2020.csv, price_2021.csv, price_2022.csv,
  production_index.csv, export_number.csv, exchange_rates.csv
"""

# ─────────────────────────────────────────────
# IMPORT LIBRARIES
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# LOAD DATA  (no Colab file upload needed)
# ─────────────────────────────────────────────
weather_df   = pd.read_csv('weather.csv')
weather_df['Date'] = pd.to_datetime(weather_df['Date'])

price2020_df = pd.read_csv('price_2020.csv')
price2020_df['Date'] = pd.to_datetime(price2020_df['Date'])

price2021_df = pd.read_csv('price_2021.csv')
price2021_df['Date'] = pd.to_datetime(price2021_df['Date'])

price2022_df = pd.read_csv('price_2022.csv')
price2022_df['Date'] = pd.to_datetime(price2022_df['Date'])

ipi_df       = pd.read_csv('production_index.csv')
ipi_df['Date'] = pd.to_datetime(ipi_df['Date'])

export_df    = pd.read_csv('export_number.csv')
export_df['Date'] = pd.to_datetime(export_df['Date'])

exchange_df  = pd.read_csv('exchange_rates.csv')
exchange_df['Date'] = pd.to_datetime(exchange_df['Date'], format='%d-%m-%y')

# ─────────────────────────────────────────────
# DATA INTEGRATION
# ─────────────────────────────────────────────

# --- Exchange rates: expand to daily frequency (forward-fill weekends) ---
if 'Date' not in exchange_df.columns:
    exchange_df.reset_index(inplace=True)

exchange_df.set_index('Date', inplace=True)
full_index = pd.date_range(start=exchange_df.index.min(),
                           end=exchange_df.index.max(), freq='D')
exchanges_expanded = exchange_df[~exchange_df.index.duplicated(keep='first')]
exchanges_expanded = exchanges_expanded.reindex(full_index).asfreq('D').ffill()
exchanges_expanded.index.name = 'Date'
exchanges_expanded = exchanges_expanded.reset_index()

# --- Export numbers: expand monthly → daily ---
export_df['Date'] = pd.to_datetime(export_df['Date'], errors='coerce')
export_df.dropna(subset=['Date'], inplace=True)
export_df['year']  = export_df['Date'].dt.year
export_df['month'] = export_df['Date'].dt.month

export_expanded = (
    export_df.assign(key=1)
    .merge(pd.DataFrame({'day': range(1, 32), 'key': 1}), on='key', how='left')
    .drop('key', axis=1)
)
export_expanded['Date'] = pd.to_datetime(
    export_expanded['year'].astype(str)  + '-' +
    export_expanded['month'].astype(str) + '-' +
    export_expanded['day'].astype(str),
    errors='coerce'
)
export_expanded = export_expanded[export_expanded['Date'] <= '2022-08-25']
export_expanded = export_expanded[['Date', 'Export Number (in Tonnes)']].dropna(subset=['Date'])

# --- IPI: expand monthly → daily ---
ipi_df['Date'] = pd.to_datetime(ipi_df['Date'], errors='coerce')
ipi_df.dropna(subset=['Date'], inplace=True)
ipi_df['year']  = ipi_df['Date'].dt.year
ipi_df['month'] = ipi_df['Date'].dt.month

ipi_expanded = (
    ipi_df.assign(key=1)
    .merge(pd.DataFrame({'day': range(1, 32), 'key': 1}), on='key', how='left')
    .drop('key', axis=1)
)
ipi_expanded['Date'] = pd.to_datetime(
    ipi_expanded['year'].astype(str)  + '-' +
    ipi_expanded['month'].astype(str) + '-' +
    ipi_expanded['day'].astype(str),
    errors='coerce'
)
ipi_expanded = ipi_expanded[ipi_expanded['Date'] <= '2022-08-25']
ipi_expanded = ipi_expanded[['Date', 'Index Production']].dropna(subset=['Date'])
ipi_expanded = ipi_expanded.drop_duplicates(subset=['Date']).reset_index(drop=True)

# --- Price 2020: expand to daily ---
def expand_price_year(price_df, year, end_date=None):
    price_df = price_df.copy()
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    price_df.dropna(subset=['Date'], inplace=True)

    months = pd.DataFrame({'month': range(1, 13), 'year': year})
    days   = pd.DataFrame({'day': range(1, 32)})
    expanded = months.merge(days, how='cross')
    expanded['Date'] = pd.to_datetime(
        expanded['year'].astype(str) + '-' +
        expanded['month'].astype(str) + '-' +
        expanded['day'].astype(str),
        errors='coerce'
    )
    expanded = expanded[['Date']].dropna()
    result = pd.merge(expanded, price_df[['Date', 'Price']], on='Date', how='left')
    if end_date:
        result = result[result['Date'] <= end_date]
    return result.sort_values('Date').reset_index(drop=True)

price2020_expanded = expand_price_year(price2020_df, 2020)
price2021_expanded = expand_price_year(price2021_df, 2021)
price2022_expanded = expand_price_year(price2022_df, 2022, end_date='2022-08-25')

# --- Combine all prices and forward-fill ---
all_prices_df = pd.concat([price2020_expanded, price2021_expanded, price2022_expanded],
                           ignore_index=True)
all_prices_df = all_prices_df.sort_values('Date').reset_index(drop=True)
all_prices_df.set_index('Date', inplace=True)

full_index = pd.date_range(start=all_prices_df.index.min(),
                           end=all_prices_df.index.max(), freq='D')
all_prices_df = all_prices_df[~all_prices_df.index.duplicated(keep='first')]
all_prices_df = all_prices_df.reindex(full_index).asfreq('D').ffill()
all_prices_df.index.name = 'Date'
all_prices_df = all_prices_df.reset_index()

# --- Merge everything into one dataframe ---
df = weather_df.copy()
df = pd.merge(df, ipi_expanded,       on='Date', how='left')
df = pd.merge(df, all_prices_df,      on='Date', how='left')
df = pd.merge(df, export_expanded,    on='Date', how='left')
df = pd.merge(df, exchanges_expanded, on='Date', how='left')

print("Merged dataframe shape:", df.shape)
print(df.dtypes)

# ─────────────────────────────────────────────
# DATA PREPROCESSING
# ─────────────────────────────────────────────

# Fix Export Number data type (may be stored as string with commas)
numeric_cols = ['Export Number (in Tonnes)']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicated rows:", df.duplicated().sum())

# --- Visualise distributions before imputation ---
for col in ['Index Production', 'Price', 'Export Number (in Tonnes)', 'USD']:
    plt.figure()
    df[col].hist(bins=30)
    title = col.replace('(in Tonnes)', '').replace('Index Production', 'IPI').strip()
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# --- Missing value imputation ---
df_imputed = df.set_index('Date').copy()

# Export Number: rolling 30-day mean imputation
col = 'Export Number (in Tonnes)'
rolling_mean = df_imputed[col].rolling(window=30, min_periods=1).mean()
df_imputed[col] = df_imputed[col].fillna(rolling_mean)
df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

# Price, IPI, USD: median imputation
for col in ['Price', 'Index Production', 'USD']:
    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

df = df_imputed.reset_index()

# Drop rows with missing Sealevelpressure (< 5 rows)
df = df.dropna(subset=['Sealevelpressure'])

print("\nMissing values after imputation:\n", df.isnull().sum())

# ─────────────────────────────────────────────
# EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

print("\nDescriptive statistics:")
print(df.describe(include='all'))

# Distribution histograms
sns.set_style("darkgrid")
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(14, 18))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df.drop(columns=['Date']).corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Box plots
for col in ['Price', 'Export Number (in Tonnes)', 'Index Production', 'USD']:
    plt.figure(figsize=(21, 7))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# Price by month
df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year

plt.figure(figsize=(14, 7))
sns.boxplot(x='Month', y='Price', data=df)
plt.title("Price Distribution by Month")
plt.tight_layout()
plt.show()

# Stacked bar: month-year distribution
month_year_dist = df.groupby(['Year', 'Month']).size().unstack().T
month_year_dist.plot(kind='bar', stacked=True, figsize=(14, 7))
plt.title('Month-wise Distribution by Year')
plt.xlabel('Month')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df[numerical_columns])
plt.suptitle('Pairwise Scatter Plot', y=1.02)
plt.tight_layout()
plt.show()

# Palm oil price trend
combined_price_df = all_prices_df.copy()
combined_price_df['Price'] = combined_price_df['Price'].ffill()
combined_price_df.dropna(subset=['Price'], inplace=True)
combined_price_df['Year'] = combined_price_df['Date'].dt.year

plt.figure(figsize=(15, 7))
sns.lineplot(data=combined_price_df, x='Date', y='Price', hue='Year', palette='viridis')
plt.title('Palm Oil Price Trends (2020–2022)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend(title='Year')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING & SELECTION
# ─────────────────────────────────────────────

df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

feature_cols = [
    'Temp', 'Dew', 'Humidity', 'Precip', 'Precipprob', 'Precipcover',
    'Windspeed', 'Winddir', 'Sealevelpressure', 'Cloudcover', 'Visibility',
    'Solarradiation', 'Solarenergy', 'Uvindex', 'Moonphase',
    'Index Production', 'Export Number (in Tonnes)', 'USD', 'Year', 'Month'
]

X = df[feature_cols]
y = df['Price']

# Pearson correlation with target
corr_with_target = X.corrwith(y)
print("\nCorrelation with Price:\n", corr_with_target)

# ANOVA F-test feature selection
anova_results = []
for feature in X.columns:
    F, p = f_classif(X[[feature]], y)
    anova_results.append({'Feature': feature, 'F-value': F[0], 'p-value': p[0]})
anova_results = pd.DataFrame(anova_results).sort_values(by='p-value')
print("\nANOVA Results:\n", anova_results)

# Keep features with |correlation| >= 0.1
selected_features = corr_with_target[abs(corr_with_target) >= 0.1].index
print("\nSelected features:", list(selected_features))
X_selected = X[selected_features]

# ─────────────────────────────────────────────
# PARTITIONING (80 / 20)
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# FEATURE SCALING
# ─────────────────────────────────────────────

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# HELPER: evaluate a model
# ─────────────────────────────────────────────

def evaluate(model, X_tr, y_tr, X_te, y_te, label):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0
    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mae  = mean_absolute_error(y_te, preds)
    r2   = r2_score(y_te, preds)
    print(f"\n{label}  |  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  "
          f"Train time={train_time:.3f}s")
    return preds, rmse, mae, r2, train_time

# ─────────────────────────────────────────────
# DECISION TREE REGRESSOR
# ─────────────────────────────────────────────

dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
dt_pred, rmse_dt, mae_dt, r2_dt, training_time_dt = evaluate(
    dt, X_train_scaled, y_train, X_test_scaled, y_test, "Decision Tree (Untuned)")

# Grid search
param_grid_dt = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gs_dt = GridSearchCV(DecisionTreeRegressor(random_state=42),
                     param_grid_dt, cv=5, scoring='neg_mean_squared_error', verbose=1)
gs_dt.fit(X_train_scaled, y_train)
print("Best DT params:", gs_dt.best_params_)

dt_tuned = DecisionTreeRegressor(**{k: v for k, v in gs_dt.best_params_.items()},
                                  random_state=42)
dt_tuned_pred, rmse_dt_tuned, mae_dt_tuned, r2_dt_tuned, training_time_dt_tuned = evaluate(
    dt_tuned, X_train_scaled, y_train, X_test_scaled, y_test, "Decision Tree (Tuned)")

# ─────────────────────────────────────────────
# RANDOM FOREST REGRESSOR
# ─────────────────────────────────────────────

rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
rf_pred, rmse_rf, mae_rf, r2_rf, training_time_rf = evaluate(
    rf, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest (Untuned)")

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
gs_rf = GridSearchCV(RandomForestRegressor(random_state=42),
                     param_grid_rf, cv=5, scoring='neg_mean_squared_error', verbose=1)
gs_rf.fit(X_train_scaled, y_train)
print("Best RF params:", gs_rf.best_params_)

rf_tuned = RandomForestRegressor(**gs_rf.best_params_, random_state=42)
rf_tuned_pred, rmse_rf_tuned, mae_rf_tuned, r2_rf_tuned, training_time_rf_tuned = evaluate(
    rf_tuned, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest (Tuned)")

# ─────────────────────────────────────────────
# XGBOOST
# ─────────────────────────────────────────────

xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                   subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb_pred, rmse_xgb, mae_xgb, r2_xgb, training_time_xgb = evaluate(
    xgb, X_train_scaled, y_train, X_test_scaled, y_test, "XGBoost (Untuned)")

param_grid_xgb = {
    'objective': ['reg:squarederror', 'reg:absoluteerror'],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}
gs_xgb = GridSearchCV(XGBRegressor(random_state=42),
                      param_grid_xgb, cv=5, scoring='neg_mean_squared_error', verbose=1)
gs_xgb.fit(X_train_scaled, y_train)
print("Best XGB params:", gs_xgb.best_params_)

xgb_tuned = XGBRegressor(**gs_xgb.best_params_, random_state=42)
xgb_tuned_pred, rmse_xgb_tuned, mae_xgb_tuned, r2_xgb_tuned, training_time_xgb_tuned = evaluate(
    xgb_tuned, X_train_scaled, y_train, X_test_scaled, y_test, "XGBoost (Tuned)")

# ─────────────────────────────────────────────
# GRADIENT BOOSTING REGRESSOR
# ─────────────────────────────────────────────

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                 max_depth=3, random_state=42)
gbr_pred, rmse_gbr, mae_gbr, r2_gbr, training_time_gbr = evaluate(
    gbr, X_train_scaled, y_train, X_test_scaled, y_test, "Gradient Boosting (Untuned)")

param_grid_gbr = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gs_gbr = GridSearchCV(GradientBoostingRegressor(random_state=42),
                      param_grid_gbr, cv=5, scoring='neg_mean_squared_error', verbose=1)
gs_gbr.fit(X_train_scaled, y_train)
print("Best GBR params:", gs_gbr.best_params_)

gbr_tuned = GradientBoostingRegressor(**gs_gbr.best_params_, random_state=42)
gbr_tuned_pred, rmse_gbr_tuned, mae_gbr_tuned, r2_gbr_tuned, training_time_gbr_tuned = evaluate(
    gbr_tuned, X_train_scaled, y_train, X_test_scaled, y_test, "Gradient Boosting (Tuned)")

# ─────────────────────────────────────────────
# SUPPORT VECTOR REGRESSION
# ─────────────────────────────────────────────

svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1))
svr_pred, rmse_svr, mae_svr, r2_svr, training_time_svr = evaluate(
    svr, X_train_scaled, y_train, X_test_scaled, y_test, "SVR")

# ─────────────────────────────────────────────
# RESULTS COMPARISON TABLE
# ─────────────────────────────────────────────

results_df = pd.DataFrame({
    'Model': [
        'Decision Tree',
        'Decision Tree (Tuned)',
        'Random Forest',
        'Random Forest (Tuned)',
        'XGBoost',
        'XGBoost (Tuned)',
        'Gradient Boosting',
        'Gradient Boosting (Tuned)',
        'SVR'
    ],
    'RMSE': [
        rmse_dt, rmse_dt_tuned,
        rmse_rf, rmse_rf_tuned,
        rmse_xgb, rmse_xgb_tuned,
        rmse_gbr, rmse_gbr_tuned,
        rmse_svr
    ],
    'MAE': [
        mae_dt, mae_dt_tuned,
        mae_rf, mae_rf_tuned,
        mae_xgb, mae_xgb_tuned,
        mae_gbr, mae_gbr_tuned,
        mae_svr
    ],
    'R-squared': [
        r2_dt, r2_dt_tuned,
        r2_rf, r2_rf_tuned,
        r2_xgb, r2_xgb_tuned,
        r2_gbr, r2_gbr_tuned,
        r2_svr
    ],
    'Training Time (s)': [
        training_time_dt, training_time_dt_tuned,
        training_time_rf, training_time_rf_tuned,
        training_time_xgb, training_time_xgb_tuned,
        training_time_gbr, training_time_gbr_tuned,
        training_time_svr
    ]
})

results_df = results_df.sort_values(by='R-squared', ascending=False).reset_index(drop=True)
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# VISUALISATION: ACTUAL vs PREDICTED
# ─────────────────────────────────────────────

# Combined overview
plt.figure(figsize=(15, 8))
plt.plot(np.arange(len(y_test)), y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(np.arange(len(y_test)), dt_pred,        label='Decision Tree',            linestyle='--')
plt.plot(np.arange(len(y_test)), dt_tuned_pred,  label='Decision Tree (Tuned)',    linestyle='-.')
plt.plot(np.arange(len(y_test)), rf_pred,         label='Random Forest',           linestyle=':')
plt.plot(np.arange(len(y_test)), rf_tuned_pred,  label='Random Forest (Tuned)',    linestyle='--')
plt.plot(np.arange(len(y_test)), xgb_pred,        label='XGBoost',                linestyle='-')
plt.plot(np.arange(len(y_test)), xgb_tuned_pred, label='XGBoost (Tuned)',          linestyle='-.')
plt.plot(np.arange(len(y_test)), gbr_pred,        label='Gradient Boosting',       linestyle=':')
plt.plot(np.arange(len(y_test)), gbr_tuned_pred, label='Gradient Boosting (Tuned)', linestyle='--')
plt.plot(np.arange(len(y_test)), svr_pred,        label='SVR',                    linestyle='-')
plt.title('All Models: Actual vs Predicted Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()


# Per-model plots
def plot_model_predictions(actual, untuned, tuned=None, name='Model'):
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(actual)), actual.values,
             label='Actual', color='black', linewidth=2)
    plt.plot(np.arange(len(actual)), untuned,
             label=f'{name} (Untuned)', linestyle='--')
    if tuned is not None:
        plt.plot(np.arange(len(actual)), tuned,
                 label=f'{name} (Tuned)', linestyle='-.')
    plt.title(f'Actual vs Predicted: {name}')
    plt.xlabel('Sample Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_model_predictions(y_test, dt_pred,  dt_tuned_pred,  name='Decision Tree')
plot_model_predictions(y_test, rf_pred,  rf_tuned_pred,  name='Random Forest')
plot_model_predictions(y_test, xgb_pred, xgb_tuned_pred, name='XGBoost')
plot_model_predictions(y_test, gbr_pred, gbr_tuned_pred, name='Gradient Boosting')
plot_model_predictions(y_test, svr_pred,                 name='SVR')
