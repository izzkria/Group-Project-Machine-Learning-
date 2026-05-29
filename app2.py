# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Palm Oil Price Predictor",
    page_icon="🌴",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3CB371;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .solution-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🌴 Palm Oil Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts palm oil prices using machine learning models. 
**Debug Mode:** Activated to diagnose negative R² issues.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Data Overview", "Model Predictions", "Results Comparison"])

# Cache data loading
@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        # Try to load data
        df = pd.read_csv("price.csv")
        
        # Log data loading
        st.sidebar.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Display column names for debugging
        st.sidebar.write("**Columns in dataset:**")
        for col in df.columns:
            st.sidebar.write(f"- {col}")
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            # If no Date column, create one with specified date range
            df['Date'] = pd.date_range(start='2020-01-01', end='2025-08-25', periods=len(df))
        
        # Add temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Check if we have a Price column
        if 'Price' not in df.columns:
            # Try common alternative names
            alt_names = ['price', 'PRICE', 'Price($)', 'price_usd', 'USD', 'value', 'Value']
            for alt in alt_names:
                if alt in df.columns:
                    df['Price'] = df[alt]
                    st.sidebar.success(f"Found price column: '{alt}' renamed to 'Price'")
                    break
        
        if 'Price' not in df.columns:
            st.error("⚠️ 'Price' column not found in dataset!")
            # Show all columns
            st.write("**Available columns:**", df.columns.tolist())
            return None
        
        # Check for missing values in Price
        missing_prices = df['Price'].isnull().sum()
        if missing_prices > 0:
            st.warning(f"Found {missing_prices} missing values in Price column. Filling with median.")
            df['Price'] = df['Price'].fillna(df['Price'].median())
        
        # Check if Price column has variance
        price_std = df['Price'].std()
        if price_std < 1e-10:
            st.error(f"⚠️ Price column has no variance (std: {price_std:.6f})!")
            st.write("Price statistics:", df['Price'].describe())
            return None
        
        # Basic data cleaning
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    except FileNotFoundError:
        st.error("File 'price.csv' not found. Please create a CSV file with your data.")
        st.info("""
        Required columns:
        1. 'Date' - Date of observation
        2. 'Price' - Palm oil price (target variable)
        Optional features: Solarradiation, Solarenergy, Uvindex, etc.
        """)
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_lag_features(df, price_col='Price', lags=[1, 2, 3, 7, 30]):
    """Create lag features for time series data"""
    df_lagged = df.copy()
    
    # Sort by date if not already sorted
    if 'Date' in df.columns:
        df_lagged = df_lagged.sort_values('Date')
    
    # Create lag features
    for lag in lags:
        df_lagged[f'Price_Lag_{lag}'] = df_lagged[price_col].shift(lag)
    
    # Create rolling statistics
    df_lagged['Price_Rolling_Mean_7'] = df_lagged[price_col].rolling(window=7, min_periods=1).mean()
    df_lagged['Price_Rolling_Std_7'] = df_lagged[price_col].rolling(window=7, min_periods=1).std()
    
    # Fill NaN values created by shifting
    df_lagged = df_lagged.fillna(method='bfill').fillna(method='ffill')
    
    return df_lagged

@st.cache_resource
def train_single_model_with_checks(X_train, y_train, X_test, y_test, model_name, tuned=True, feature_names=None):
    """Train a single model with comprehensive checks"""
    
    # Debug info
    debug_info = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_mean': y_train.mean(),
        'y_train_std': y_train.std(),
        'y_test_mean': y_test.mean(),
        'y_test_std': y_test.std()
    }
    
    # Check if we have enough data
    if len(X_train) < 10 or len(X_test) < 5:
        return None, None, None, f"Not enough data for training/testing: {debug_info}"
    
    # Check for variance in target
    if y_train.std() < 1e-10 or y_test.std() < 1e-10:
        return None, None, None, f"Target variable has no variance: {debug_info}"
    
    try:
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define model based on type with simpler hyperparameters
        if model_name == 'Random Forest':
            if tuned:
                # Simpler grid for debugging
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                }
                model = GridSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
        
        elif model_name == 'XGBoost':
            if tuned:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
                model = GridSearchCV(
                    XGBRegressor(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            else:
                model = XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
        
        elif model_name == 'Gradient Boosting':
            if tuned:
                param_grid = {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
                model = GridSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
        
        elif model_name == 'SVR' or 'Linear Regression':
            # SVR is sensitive to scaling, so we'll use StandardScaler
            if tuned:
                param_grid = {
                    'C': [1, 10],
                    'epsilon': [0.1, 0.2]
                }
                model = GridSearchCV(
                    SVR(kernel='rbf'),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            else:
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        
        elif model_name == 'Decision Tree':
            if tuned:
                param_grid = {
                    'max_depth': [3, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = GridSearchCV(
                    DecisionTreeRegressor(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=42
                )
        
        elif model_name == 'Linear Regression':
            # Always use Ridge regression for stability
            if tuned:
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0]
                }
                model = GridSearchCV(
                    Ridge(random_state=42),
                    param_grid,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            else:
                model = Ridge(alpha=1.0, random_state=42)
        
        else:
            return None, None, None, f"Unknown model: {model_name}"
        
        # Train the model
        if model_name == 'SVR':
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            y_pred = model.predict(X_test_scaled)
        else:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time 
            # Make predictions
            y_pred = model.predict(X_test)
        
        # Calculate metrics with checks
        try:
            r2 = r2_score(y_test, y_pred)
            
            # Check for absurd R² values
            if r2 < -10 or r2 > 10:  # Allow some extreme values for debugging
                st.warning(f"Extreme R² value for {model_name}: {r2}")
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate baseline (predicting mean)
            y_mean = np.full_like(y_test, y_train.mean())
            baseline_r2 = r2_score(y_test, y_mean)
            
            metrics = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Baseline_R²': baseline_r2,
                'Improvement_over_baseline': r2 - baseline_r2
            }
            
            return model, metrics, scaler, training_time
            
        except Exception as e:
            return None, None, None, f"Error calculating metrics: {str(e)}"
            
    except Exception as e:
        return None, None, None, f"Error training model {model_name}: {str(e)}"

@st.cache_resource
def train_all_models(X_train, y_train, X_test, y_test, use_lag_features=True):
    """Train all models (tuned and untuned)"""
    models = {}
    metrics = {}
    scalers = {}
    training_times = {}
    errors = {}
    
    # Include Linear Regression for baseline comparison
    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting', 'SVR', 'Decision Tree']
    
    # Train tuned models
    for model_name in model_names:
        model_key = f"{model_name} (Tuned)"
        model, model_metrics, scaler, train_time = train_single_model_with_checks(
            X_train, y_train, X_test, y_test, model_name, tuned=True
        )
        
        if model is not None:
            models[model_key] = model
            metrics[model_key] = model_metrics
            scalers[model_key] = scaler
            training_times[model_key] = train_time
        else:
            errors[model_key] = model_metrics
    
    # Train untuned models
    for model_name in model_names:
        model_key = f"{model_name} (Untuned)"
        model, model_metrics, scaler, train_time = train_single_model_with_checks(
            X_train, y_train, X_test, y_test, model_name, tuned=False
        )
        
        if model is not None:
            models[model_key] = model
            metrics[model_key] = model_metrics
            scalers[model_key] = scaler
            training_times[model_key] = train_time
        else:
            errors[model_key] = model_metrics
    
    return models, metrics, scalers, training_times, errors

# Load data
df = load_data()

if df is None:
    st.stop()

if page == "Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Display critical warnings first
    if 'Price' in df.columns:
        price_stats = df['Price'].describe()
        
        st.markdown("### 🔍 Price Column Analysis")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Mean", f"${price_stats['mean']:.2f}")
        with cols[1]:
            st.metric("Std Dev", f"${price_stats['std']:.2f}")
            if price_stats['std'] < 10:
                st.error("⚠️ VERY LOW VARIANCE!")
        with cols[2]:
            st.metric("Min", f"${price_stats['min']:.2f}")
        with cols[3]:
            st.metric("Max", f"${price_stats['max']:.2f}")
        
        # Check for data issues
        if price_stats['std'] < 1:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.error("🚨 CRITICAL ISSUE: Price column has almost no variation!")
            st.write(f"Standard deviation: ${price_stats['std']:.6f}")
            st.write("**This is the MAIN REASON for negative R²!**")
            st.write("A model predicting the mean will outperform any ML model.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="solution-box">', unsafe_allow_html=True)
            st.write("**SOLUTION:** Check your Price column:")
            st.write("1. Make sure it's numeric (not text)")
            st.write("2. Check if values are in correct units")
            st.write("3. Ensure there's real price variation")
            st.write("4. If testing, use realistic price data")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if df['Price'].isnull().sum() > 0:
            st.warning(f"⚠️ Found {df['Price'].isnull().sum()} missing values in Price column")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 First 10 Rows")
        st.dataframe(df.head(10), width='stretch')
    
    with col2:
        st.subheader("📊 Last 10 Rows")
        st.dataframe(df.tail(10), width='stretch')
    
    st.subheader("📈 Yearly Price Trend Chart")
    
    if 'Date' in df.columns and 'Price' in df.columns:
        # Create a copy of the dataframe for plotting
        plot_df = df.copy()
        
        # Ensure we have a Year column
        if 'Year' not in plot_df.columns:
            plot_df['Year'] = plot_df['Date'].dt.year
        
        # Create the line plot with yearly breakdown
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Get unique years in the data
        years = sorted(plot_df['Year'].unique())
        
        # Use viridis colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
        
        # Plot each year separately
        for year, color in zip(years, colors):
            year_data = plot_df[plot_df['Year'] == year]
            if len(year_data) > 0:
                ax.plot(year_data['Date'], year_data['Price'], 
                       label=str(year), color=color, linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Palm Oil Price Trends by Year', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Year', title_fontsize=12, fontsize=10)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistical analysis by year
        st.subheader("📊 Yearly Statistics")
        
        yearly_stats = []
        for year in years:
            year_data = plot_df[plot_df['Year'] == year]['Price']
            if len(year_data) > 0:
                yearly_stats.append({
                    'Year': year,
                    'Count': len(year_data),
                    'Mean': year_data.mean(),
                    'Std': year_data.std(),
                    'Min': year_data.min(),
                    'Max': year_data.max()
                })
        
        yearly_df = pd.DataFrame(yearly_stats)
        st.dataframe(yearly_df.style.format({
            'Mean': '${:.2f}',
            'Std': '${:.2f}',
            'Min': '${:.2f}',
            'Max': '${:.2f}'
        }), width='stretch')
    
    # Feature correlation analysis
    st.subheader("🔗 Feature Correlations with Price")
    
    if 'Price' in df.columns:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            # Calculate correlations
            correlations = numeric_df.corr()['Price'].sort_values(ascending=False)
            
            # Remove Price itself
            correlations = correlations.drop('Price', errors='ignore')
            
            if len(correlations) > 0:
                # Display correlation matrix
                corr_df = pd.DataFrame(correlations).reset_index()
                corr_df.columns = ['Feature', 'Correlation']
                
                # Color code correlations
                def color_correlation(val):
                    if val > 0.7:
                        return 'background-color: #4CAF50; color: white'
                    elif val > 0.5:
                        return 'background-color: #8BC34A'
                    elif val > 0.3:
                        return 'background-color: #CDDC39'
                    elif val > 0:
                        return 'background-color: #FFEB3B'
                    elif val > -0.3:
                        return 'background-color: #FF9800'
                    else:
                        return 'background-color: #F44336; color: white'
                
                st.dataframe(
                    corr_df.style.format({'Correlation': '{:.3f}'})
                    .applymap(color_correlation, subset=['Correlation']),
                    width='stretch'
                )
                
                # Warning about weak correlations
                strong_corrs = corr_df[corr_df['Correlation'].abs() > 0.3]
                if len(strong_corrs) == 0:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.warning("⚠️ NO FEATURES have strong correlation (>0.3) with Price!")
                    st.write("This explains negative R². Features cannot predict Price.")
                    st.write("**Solutions:**")
                    st.write("1. Add lag features (previous prices)")
                    st.write("2. Add external economic indicators")
                    st.write("3. Use time-based features")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write("No other numeric features found for correlation analysis.")
        else:
            st.write("Only Price column found (no other numeric features).")

elif page == "Model Predictions":
    st.markdown('<h2 class="sub-header">Model Training & Prediction</h2>', unsafe_allow_html=True)
    
    if 'Price' not in df.columns:
        st.error("'Price' column not found in dataset. Cannot proceed with modeling.")
        st.stop()
    
    # Data Quality Check Section
    st.markdown("### 🔍 Data Quality Check")
    
    price_std = df['Price'].std()
    if price_std < 1:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.error(f"🚨 MAJOR DATA ISSUE: Price std deviation = ${price_std:.6f}")
        st.write("**This WILL cause negative R²!**")
        st.write("The price data has almost no variation.")
        st.write("**Quick test:** Run this in your Python console:")
        st.code("""
import pandas as pd
df = pd.read_csv('price.csv')
print(f"Price stats:\\n{df['Price'].describe()}")
print(f"Unique prices: {df['Price'].nunique()}")
print(f"Sample prices: {df['Price'].head(10).tolist()}")
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    # Feature Engineering Section
    st.markdown("### 🔧 Feature Engineering")
    
    # Create enhanced dataframe with lag features
    df_enhanced = create_lag_features(df)
    
    # Select features - prioritize lag features
    possible_features = [
        # Lag features (most important for time series)
        'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_7', 'Price_Lag_30',
        'Price_Rolling_Mean_7', 'Price_Rolling_Std_7',
        # Temporal features
        'Year', 'Month', 'Day', 'DayOfYear', 'Quarter', 'WeekOfYear', 'DayOfWeek',
        # Original features
        'Solarradiation', 'Solarenergy', 'Uvindex',
        'Index Production', 'Export Number (in Tonnes)', 'USD'
    ]
    
    # Get available features
    available_features = [f for f in possible_features if f in df_enhanced.columns]
    
    # Add other numeric features (excluding Price)
    numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    if 'Price' in numeric_cols:
        numeric_cols.remove('Price')
    available_features = list(set(available_features + numeric_cols))
    
    if len(available_features) < 1:
        st.error("No features available for modeling!")
        st.info(f"Available columns: {df.columns.tolist()}")
        st.stop()
    
    # Display feature selection
    with st.expander("📋 Selected Features"):
        st.write(f"**Total features:** {len(available_features)}")
        st.write("**Features:**", ", ".join(available_features))
        
        # Show which are lag features
        lag_features = [f for f in available_features if 'Lag' in f or 'Rolling' in f]
        if lag_features:
            st.success(f"✅ {len(lag_features)} lag features included")
        else:
            st.warning("⚠️ No lag features - time series prediction will be difficult")
    
    # Prepare features and target
    X = df_enhanced[available_features].copy()
    y = df_enhanced['Price'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    # Check for constant columns
    constant_cols = []
    for col in X.columns:
        if X[col].std() < 1e-10:
            constant_cols.append(col)
    
    if constant_cols:
        st.warning(f"Removing constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)
        available_features = [f for f in available_features if f not in constant_cols]
    
    # Feature Selection Option
    st.sidebar.subheader("Feature Selection")
    use_feature_selection = st.sidebar.checkbox("Use Feature Selection", value=True)
    if use_feature_selection:
        n_features = st.sidebar.slider("Number of features to select", 3, min(20, len(available_features)), 10)
    
    # Split data with time series consideration
    st.sidebar.subheader("Data Splitting")
    test_size = st.sidebar.slider("Test size (%)", 10, 40, 20) / 100
    
    # Time-based split (for time series)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Apply feature selection if requested
    if use_feature_selection and len(available_features) > n_features:
        with st.spinner("Selecting best features..."):
            # Use mutual information for feature selection
            selector = SelectKBest(score_func=f_regression, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_features = X_train.columns[selected_mask].tolist()
            
            st.success(f"Selected {len(selected_features)} best features")
            st.write("**Selected features:**", ", ".join(selected_features))
            
            # Update X_train and X_test with selected features
            X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
            available_features = selected_features
    
    # Display data info
    with st.expander("📊 Data Split Information"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(X))
        with col2:
            st.metric("Training Samples", len(X_train))
        with col3:
            st.metric("Test Samples", len(X_test))
        with col4:
            st.metric("Features", len(available_features))
        
        st.write("**Target statistics:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training set:**")
            st.write(f"- Mean: ${y_train.mean():.2f}")
            st.write(f"- Std: ${y_train.std():.2f}")
            st.write(f"- Range: ${y_train.min():.2f} - ${y_train.max():.2f}")
        
        with col2:
            st.write("**Test set:**")
            st.write(f"- Mean: ${y_test.mean():.2f}")
            st.write(f"- Std: ${y_test.std():.2f}")
            st.write(f"- Range: ${y_test.min():.2f} - ${y_test.max():.2f}")
    
    # Model Selection
    st.sidebar.subheader("Model Selection")
    model_options = {
        'Linear Regression': st.sidebar.checkbox("Linear Regression", value=True),
        'Random Forest': st.sidebar.checkbox("Random Forest", value=True),
        'XGBoost': st.sidebar.checkbox("XGBoost", value=True),
        'Gradient Boosting': st.sidebar.checkbox("Gradient Boosting", value=True),
        'SVR': st.sidebar.checkbox("SVR", value=False),
        'Decision Tree': st.sidebar.checkbox("Decision Tree", value=True)
    }
    
    selected_models = [model for model, selected in model_options.items() if selected]
    
    if not selected_models:
        st.warning("Please select at least one model to train")
        st.stop()
    
    # Training options
    st.sidebar.subheader("Training Options")
    train_tuned = st.sidebar.checkbox("Train Tuned Models", value=True)
    train_untuned = st.sidebar.checkbox("Train Untuned Models", value=True)
    
    # Train models button
    if st.button("🚀 Train Selected Models", type="primary"):
        if not train_tuned and not train_untuned:
            st.warning("Please select at least one model type (tuned or untuned)")
            st.stop()
        
        with st.spinner("Training models... This may take a minute."):
            # Clear previous session state
            for key in ['all_models', 'all_metrics', 'all_scalers', 'training_times', 'training_errors']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Train models
            models = {}
            metrics = {}
            scalers = {}
            training_times = {}
            errors = {}
            
            # Train selected models
            for model_name in selected_models:
                # Train tuned version
                if train_tuned:
                    model_key = f"{model_name} (Tuned)"
                    model, model_metrics, scaler, train_time = train_single_model_with_checks(
                        X_train, y_train, X_test, y_test, model_name, tuned=True
                    )
                    
                    if model is not None:
                        models[model_key] = model
                        metrics[model_key] = model_metrics
                        scalers[model_key] = scaler
                        training_times[model_key] = train_time
                    else:
                        errors[model_key] = model_metrics
                
                # Train untuned version
                if train_untuned:
                    model_key = f"{model_name} (Untuned)"
                    model, model_metrics, scaler, train_time = train_single_model_with_checks(
                        X_train, y_train, X_test, y_test, model_name, tuned=False
                    )
                    
                    if model is not None:
                        models[model_key] = model
                        metrics[model_key] = model_metrics
                        scalers[model_key] = scaler
                        training_times[model_key] = train_time
                    else:
                        errors[model_key] = model_metrics
            
            # Store in session state
            st.session_state.all_models = models
            st.session_state.all_metrics = metrics
            st.session_state.all_scalers = scalers
            st.session_state.training_times = training_times
            st.session_state.training_errors = errors
            st.session_state.model_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
        
        # Display training results
        if errors:
            st.warning("⚠️ Some models failed to train:")
            for model_name, error_msg in errors.items():
                st.write(f"- **{model_name}:** {error_msg}")
        
        if metrics:
            st.success(f"✅ Successfully trained {len(metrics)} models!")
    
    # Display results if models are trained
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        if 'all_metrics' not in st.session_state or not st.session_state.all_metrics:
            st.warning("No models were successfully trained.")
            st.stop()
        
        all_metrics = st.session_state.all_metrics
        
        # Display all model performances
        st.subheader("📊 Model Performance Summary")
        
        # Create performance table
        perf_data = []
        for model_name, model_metrics in all_metrics.items():
            perf_data.append({
                'Model': model_name,
                'R²': model_metrics['R²'],
                'Baseline_R²': model_metrics.get('Baseline_R²', 0),
                'Improvement': model_metrics.get('Improvement_over_baseline', 0),
                'RMSE': model_metrics['RMSE'],
                'MAE': model_metrics['MAE']
            })
        
        perf_df = pd.DataFrame(perf_data)
        perf_df = perf_df.sort_values('R²', ascending=False)
        
        # Highlight models
        def highlight_performance(row):
            styles = [''] * len(row)
            if row['R²'] < 0:
                styles[1] = 'background-color: #ff4444; color: white; font-weight: bold'
            elif row['R²'] < 0.3:
                styles[1] = 'background-color: #ff9800'
            elif row['R²'] < 0.6:
                styles[1] = 'background-color: #ffeb3b'
            elif row['R²'] < 0.8:
                styles[1] = 'background-color: #8bc34a'
            else:
                styles[1] = 'background-color: #4CAF50; color: white'
            
            # Highlight improvement
            if row['Improvement'] > 0:
                styles[3] = 'background-color: #4CAF50; color: white'
            elif row['Improvement'] < 0:
                styles[3] = 'background-color: #ff4444; color: white'
            
            return styles
        
        # Display table
        styled_df = perf_df.style.format({
            'R²': '{:.4f}',
            'Baseline_R²': '{:.4f}',
            'Improvement': '{:.4f}',
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}'
        }).apply(highlight_performance, axis=1)
        
        st.dataframe(styled_df, width='stretch')
        
        # Analysis of results
        st.subheader("📈 Performance Analysis")
        
        # Check for negative R²
        negative_models = perf_df[perf_df['R²'] < 0]
        if len(negative_models) > 0:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.error(f"🚨 {len(negative_models)} models have NEGATIVE R²!")
            
            # Diagnostic information
            st.write("**Common causes:**")
            st.write("1. **Price data has no real variation** (check std deviation)")
            st.write("2. **Features don't correlate with Price**")
            st.write("3. **Data leakage or incorrect splitting**")
            st.write("4. **Overfitting to noise**")
            
            # Quick fixes
            st.write("**Quick fixes to try:**")
            st.write("1. Check your Price column values")
            st.write("2. Add lag features (previous prices)")
            st.write("3. Use simpler models (Linear Regression first)")
            st.write("4. Reduce number of features")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show baseline comparison
        baseline_r2 = perf_df['Baseline_R²'].mean() if 'Baseline_R²' in perf_df.columns else 0
        st.write(f"**Baseline (predicting mean):** R² = {baseline_r2:.4f}")
        st.write(f"**Best model improvement:** {perf_df.iloc[0]['Improvement']:.4f}")
        
        # Select model for detailed view
        st.subheader("🔍 Detailed Model Analysis")
        
        selected_model = st.selectbox(
            "Choose a model for detailed analysis:",
            list(all_metrics.keys()),
            index=0
        )
        
        if selected_model in st.session_state.all_models:
            model = st.session_state.all_models[selected_model]
            metrics = st.session_state.all_metrics[selected_model]
            scaler = st.session_state.all_scalers[selected_model]
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            # Make predictions
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{metrics['R²']:.4f}")
                if metrics['R²'] < 0:
                    st.error("Worse than predicting mean!")
                elif metrics['R²'] < 0.3:
                    st.warning("Poor predictive power")
                elif metrics['R²'] < 0.6:
                    st.info("Moderate predictive power")
                elif metrics['R²'] < 0.8:
                    st.success("Good predictive power")
                else:
                    st.success("Excellent predictive power!")
            
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
                st.caption(f"Avg error: ${metrics['RMSE']:.2f}")
            
            with col3:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
                st.caption(f"Avg absolute error: ${metrics['MAE']:.2f}")
            
            # Visualizations
            st.subheader("📊 Prediction Visualizations")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Actual vs Predicted scatter
            ax1.scatter(y_test, y_pred, alpha=0.5, color='#2E8B57')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', label='Perfect Prediction')
            ax1.set_xlabel('Actual Price ($)')
            ax1.set_ylabel('Predicted Price ($)')
            ax1.set_title(f'{selected_model}: Actual vs Predicted')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Residuals plot
            residuals = y_test - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.5, color='#FF6B6B')
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Predicted Price ($)')
            ax2.set_ylabel('Residuals (Actual - Predicted)')
            ax2.set_title(f'Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            # 3. Distribution of residuals
            ax3.hist(residuals, bins=30, color='#2196F3', edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='r', linestyle='--')
            ax3.set_xlabel('Residuals')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Residuals')
            ax3.grid(True, alpha=0.3)
            
            # 4. Time series of predictions
            ax4.plot(range(len(y_test)), y_test.values, label='Actual', color='#2E8B57', linewidth=2)
            ax4.plot(range(len(y_pred)), y_pred, label='Predicted', color='#FF9800', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Test Sample Index')
            ax4.set_ylabel('Price ($)')
            ax4.set_title('Time Series: Actual vs Predicted')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

elif page == "Results Comparison":
    st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    if 'model_trained' not in st.session_state or not st.session_state.model_trained:
        st.warning("⚠️ No models have been trained yet!")
        st.info("Please go to 'Model Predictions' page and train models first.")
        st.stop()
    
    if 'all_metrics' not in st.session_state or not st.session_state.all_metrics:
        st.warning("No model metrics available. Please train models first.")
        st.stop()
    
    all_metrics = st.session_state.all_metrics
    
    # Create comparison table
    st.subheader("📊 Tuned vs Untuned Performance Comparison")
    
    comparison_data = []
    for model_name, metrics in all_metrics.items():
        comparison_data.append({
            'Model': model_name,
            'Type': 'Tuned' if '(Tuned)' in model_name else 'Untuned',
            'Base Model': model_name.replace(' (Tuned)', '').replace(' (Untuned)', ''),
            'R²': metrics['R²'],
            'Improvement': metrics.get('Improvement_over_baseline', 0),
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Sort by R²
    comp_df = comp_df.sort_values('R²', ascending=False).reset_index(drop=True)
    
    # Display comparison
    def highlight_comparison(row):
        styles = [''] * len(row)
        
        # Highlight model type
        if row['Type'] == 'Tuned':
            styles[1] = 'background-color: #e8f5e9'
        else:
            styles[1] = 'background-color: #fff3e0'
        
        # Highlight R²
        if row['R²'] < 0:
            styles[3] = 'background-color: #ff4444; color: white; font-weight: bold'
        elif row['R²'] < 0.3:
            styles[3] = 'background-color: #ff9800'
        elif row['R²'] < 0.6:
            styles[3] = 'background-color: #ffeb3b'
        elif row['R²'] < 0.8:
            styles[3] = 'background-color: #8bc34a'
        else:
            styles[3] = 'background-color: #4CAF50; color: white'
        
        # Highlight improvement
        if row['Improvement'] > 0:
            styles[4] = 'background-color: #4CAF50; color: white'
        elif row['Improvement'] < 0:
            styles[4] = 'background-color: #ff4444; color: white'
        
        return styles
    
    st.dataframe(
        comp_df.style.format({
            'R²': '{:.4f}',
            'Improvement': '{:.4f}',
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}'
        }).apply(highlight_comparison, axis=1),
        width='stretch'
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed using Streamlit | BSD3523 Machine Learning Project</p>
    <p>Group: CSM1 | University Malaysia Pahang Al-Sultan Abdullah</p>
    <p style='font-size: 0.9em; color: #666;'>
        Debug Mode: Diagnosing negative R² issues
    </p>
</div>
""", unsafe_allow_html=True)
