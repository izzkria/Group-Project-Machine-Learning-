# Machine_Learning_Project : Predicting Malaysia’s Palm Oil Price Incorporating Climate Conditions and Industrial Production Index Using Machine Learning Approaches (CSM1)

# 🌴 Oil Palm Price Prediction Streamlit App

A machine learning web application for predicting oil palm prices based on weather conditions, production indices, export data, and currency exchange rates.

## Features

- **Multi-Model Training**: Train and compare 9 different ML models
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - XGBoost
  - Random Forest
  - Gradient Boosting
  - Decision Tree
  - Support Vector Regression (SVR)
  - Multi-Layer Perceptron (MLP)

- **Feature Selection**: Automatic feature selection using Recursive Feature Elimination (RFE)
- **Interactive Predictions**: Make predictions with custom input values
- **Model Comparison**: Visual comparison of model performance metrics
- **Data Integration**: Seamlessly merge multiple data sources

## Required Data Files

The app requires 7 CSV files:

1. **weather.csv** - Weather data with Date column
2. **price2020.csv** - Price data for 2020
3. **price2021.csv** - Price data for 2021
4. **price2022.csv** - Price data for 2022
5. **ipi.csv** - Industrial Production Index data
6. **exchange.csv** - Currency exchange rates
7. **export.csv** - Export data
