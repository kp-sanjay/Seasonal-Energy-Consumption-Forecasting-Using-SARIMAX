
# Seasonal Energy Consumption Forecasting App

This app forecasts monthly energy consumption using SARIMA and SARIMAX models.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate dummy data (optional, `app.py` has fallback):
   ```bash
   python generate_dummy_data.py
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Features
- **SARIMA**: Auto-tuning of (p,d,q)(P,D,Q)s parameters.
- **SARIMAX**: Includes Temperature as an exogenous variable.
- **Diagnostics**: Ljung-Box test, Residual plots, ACF/PACF.
- **Simulation**: Slider to adjust temperature and see impact on energy forecast.
