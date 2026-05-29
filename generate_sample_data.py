"""
Sample Data Generator for Oil Palm Price Prediction App
This script generates sample CSV files for testing the application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2020, 1, 1)
end_date = datetime(2022, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

print("Generating sample data files...")

# 1. Weather Data
print("Creating weather.csv...")
weather_data = {
    'Date': date_range,
    'Temperature': np.random.normal(28, 3, len(date_range)),
    'Humidity': np.random.normal(75, 10, len(date_range)),
    'Precipitation': np.random.gamma(2, 5, len(date_range)),
    'WindSpeed': np.random.normal(10, 3, len(date_range)),
    'Sealevelpressure': np.random.normal(1013, 5, len(date_range)),
    'Cloudcover': np.random.uniform(0, 100, len(date_range)),
    'Visibility': np.random.normal(10, 2, len(date_range)),
    'Solarradiation': np.random.normal(200, 50, len(date_range)),
    'Solarenergy': np.random.normal(15, 4, len(date_range)),
    'Uvindex': np.random.randint(1, 11, len(date_range)),
    'Dew': np.random.normal(24, 2, len(date_range))
}
weather_df = pd.DataFrame(weather_data)
weather_df.to_csv('weather.csv', index=False)

# 2. Price Data (with some trend and seasonality)
print("Creating price files...")
base_price = 2500
for year in [2020, 2021, 2022]:
    year_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    
    # Add trend and seasonality
    trend = np.linspace(0, 200, len(year_dates))
    seasonal = 100 * np.sin(2 * np.pi * np.arange(len(year_dates)) / 365)
    noise = np.random.normal(0, 50, len(year_dates))
    
    prices = base_price + trend + seasonal + noise
    
    price_data = {
        'Date': year_dates,
        'Price': prices
    }
    price_df = pd.DataFrame(price_data)
    price_df.to_csv(f'price{year}.csv', index=False)
    
    base_price += 200  # Year-over-year increase

# 3. Industrial Production Index (monthly data)
print("Creating ipi.csv...")
monthly_dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='MS')
ipi_data = {
    'Date': monthly_dates,
    'Index Production': np.random.normal(105, 5, len(monthly_dates))
}
ipi_df = pd.DataFrame(ipi_data)
ipi_df.to_csv('ipi.csv', index=False)

# 4. Exchange Rates (monthly data)
print("Creating exchange.csv...")
exchange_data = {
    'Date': monthly_dates,
    'USD': np.random.normal(4.2, 0.1, len(monthly_dates))
}
exchange_df = pd.DataFrame(exchange_data)
exchange_df.to_csv('exchange.csv', index=False)

# 5. Export Data (monthly data)
print("Creating export.csv...")
export_data = {
    'Date': monthly_dates,
    'Export Number (in Tonnes)': np.random.normal(150000, 20000, len(monthly_dates))
}
export_df = pd.DataFrame(export_data)
export_df.to_csv('export.csv', index=False)

print("\n✅ All sample data files created successfully!")
print("\nGenerated files:")
print("1. weather.csv")
print("2. price2020.csv")
print("3. price2021.csv")
print("4. price2022.csv")
print("5. ipi.csv")
print("6. exchange.csv")
print("7. export.csv")
print("\nYou can now upload these files to the Streamlit app for testing!")
