
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from utils.preprocessing import load_data, check_stationarity, decompose_series
from utils.evaluation import calculate_metrics
from utils.diagnostics import plot_diagnostics, perform_ljung_box_test, check_normality
from models.sarima_model import SARIMAModel
from models.sarimax_model import SARIMAXModel

# --- Configuration ---
st.set_page_config(page_title="Energy Forecasting App", layout="wide")
plt.style.use('ggplot')

# --- Header ---
st.title("⚡ Seasonal Energy Consumption Forecasting")
st.markdown("""
Predict monthly energy consumption using **SARIMA** and **SARIMAX** models.
Upload your dataset or use the default generated data.
""")

# --- Sidebar ---
st.sidebar.header("Configuration")
data_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

# Model Selection
model_option = st.sidebar.selectbox("Select Model", ["SARIMA", "SARIMAX"])

# Parameters
seasonal_period = st.sidebar.number_input("Seasonal Period (Months)", min_value=1, value=12)
train_split = st.sidebar.slider("Train Split (%)", 50, 95, 80) / 100.0

# --- Data Loading ---
@st.cache_data
def get_data(uploaded_file):
    if uploaded_file is not None:
        return load_data(uploaded_file)
    else:
        # Check if dummy data exists, if not, generate it (simulate by import if needed, or assume pre-run)
        try:
            return load_data('data/energy_data.csv')
        except:
             st.warning("No data found. Generating synthetic data...")
             # Fallback generation for demo if file missing
             dates = pd.date_range(start='2015-01-01', periods=120, freq='MS')
             df = pd.DataFrame({
                 'Date': dates,
                 'Energy_Consumption': np.linspace(2000, 3500, 120) + 500*np.sin(2*np.pi*dates.month/12) + np.random.normal(0,100, 120),
                 'Temperature': 20 + 10*np.sin(2*np.pi*(dates.month-3)/12) + np.random.normal(0,2,120)
             })
             # Ensure index is set
             df.set_index('Date', inplace=True)
             return df

# Get data
try:
    df = get_data(data_file)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if df is not None:
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.line_chart(df['Energy_Consumption'])
    with col2:
        st.write(df.head())
        st.write(df.describe())

    # --- Preprocessing & Stationarity ---
    with st.expander("🔍 Stationarity & Decomposition"):
        st.write("### Augmented Dickey-Fuller Test")
        adf_result = check_stationarity(df['Energy_Consumption'])
        st.json(adf_result)
        
        st.write("### Seasonal Decomposition")
        decomposition = decompose_series(df['Energy_Consumption'], period=seasonal_period)
        fig_decomp = decomposition.plot()
        fig_decomp.set_size_inches(10, 8)
        st.pyplot(fig_decomp)

    # --- Train/Test Split ---
    train_size = int(len(df) * train_split)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    st.markdown("---")
    st.subheader(f"⚙️ Model Training: {model_option}")
    
    if st.button("🚀 Train Model"):
        with st.spinner("Training model... Please wait."):
            # Prepare data
            y_train = train_data['Energy_Consumption']
            y_test = test_data['Energy_Consumption']
            
            # --- SARIMA ---
            if model_option == "SARIMA":
                model = SARIMAModel(seasonal_period=seasonal_period)
                # Auto-tune
                st.info("Optimizing parameters (Grid Search)...")
                try:
                    order, seasonal_order = model.optimize_parameters(y_train, m=seasonal_period)
                    st.success(f"Best Parameters: Order={order}, Seasonal={seasonal_order}")
                except Exception as e:
                    st.error(f"Optimization failed: {e}. Using defaults.")
                    order = (1,1,1)
                    seasonal_order = (1,1,1,seasonal_period)

                # Fit
                model.fit(y_train, order=order, seasonal_order=seasonal_order)
                
                # Forecast on Test Set
                forecast_steps = len(test_data)
                forecast_mean, conf_int = model.predict(forecast_steps)
                
            # --- SARIMAX ---
            elif model_option == "SARIMAX":
                if 'Temperature' not in df.columns:
                    st.error("SARIMAX requires 'Temperature' column in dataset.")
                    st.stop()
                
                exog_train = train_data[['Temperature']]
                exog_test = test_data[['Temperature']]
                
                model = SARIMAXModel(seasonal_period=seasonal_period)
                
                # Auto-tune
                st.info("Optimizing parameters with Exogenous variable...")
                try:
                    order, seasonal_order = model.optimize_parameters(y_train, exog=exog_train, m=seasonal_period)
                    st.success(f"Best Parameters: Order={order}, Seasonal={seasonal_order}")
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    order = (1,1,1)
                    seasonal_order = (1,1,1,seasonal_period)
                
                # Fit
                model.fit(y_train, exog=exog_train, order=order, seasonal_order=seasonal_order)
                
                # Forecast on Test Set
                forecast_mean, conf_int = model.predict(steps=len(test_data), future_exog=exog_test)

            # --- Results & Metrics ---
            
            # Align indices
            forecast_series = pd.Series(forecast_mean, index=test_data.index)
            conf_int_df = pd.DataFrame(conf_int, index=test_data.index, columns=['lower', 'upper']) # Typically conf_int is ndarray or DataFrame, handle both
            if isinstance(conf_int, pd.DataFrame):
                conf_int_df.columns = ['lower', 'upper'] # Ensure naming
            else:
                conf_int_df = pd.DataFrame(conf_int, index=test_data.index, columns=['lower', 'upper'])

            # Metrics
            metrics = calculate_metrics(y_test, forecast_series)
            
            result_col1, result_col2 = st.columns([1, 1])
            with result_col1:
                st.write("### Performance Metrics")
                st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: 'Value'}))
            
            with result_col2:
                st.write("### Model Summary")
                st.text(str(model.results.summary()))

            # Plotting Forecast
            st.write("### Forecast vs Actual")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_data.index, y_train, label='Train')
            ax.plot(test_data.index, y_test, label='Test (Actual)')
            ax.plot(test_data.index, forecast_series, label='Forecast', color='green')
            ax.fill_between(test_data.index, 
                            conf_int_df['lower'], 
                            conf_int_df['upper'], 
                            color='green', alpha=0.1, label='Confidence Interval')
            ax.legend()
            st.pyplot(fig)
            
            # --- Diagnostics ---
            st.write("### Residual Diagnostics")
            residuals = model.results.resid
            fig_diag = plot_diagnostics(residuals)
            st.pyplot(fig_diag)
            
            lb_test = perform_ljung_box_test(residuals)
            st.write("Ljung-Box Test Results:")
            st.dataframe(lb_test)
            
            # Store model for simulation
            st.session_state['model'] = model
            st.session_state['forecast_series'] = forecast_series
            st.session_state['test_data'] = test_data

    # --- Scenario Simulation (SARIMAX Only) ---
    if model_option == "SARIMAX" and 'model' in st.session_state:
        st.markdown("---")
        st.subheader("🔮 Scenario Simulation")
        st.markdown("Simulate the effect of temperature change on future consumption.")
        
        temp_change = st.slider("Adjust Future Temperature (°C)", -5.0, 5.0, 0.0, step=0.5)
        
        if st.checkbox("Show Simulation"):
            model = st.session_state['model']
            test_data = st.session_state['test_data']
            forecast_series = st.session_state['forecast_series']
            
            # Adjust test set variable
            test_exog_adjusted = test_data[['Temperature']] + temp_change
            
            sim_mean, _ = model.predict(steps=len(test_data), future_exog=test_exog_adjusted)
            sim_series = pd.Series(sim_mean, index=test_data.index)
            
            fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
            ax_sim.plot(test_data.index, test_data['Energy_Consumption'], label='Actual')
            ax_sim.plot(test_data.index, forecast_series, label='Original Forecast', linestyle='--')
            ax_sim.plot(test_data.index, sim_series, label='Simulated Forecast', color='red')
            ax_sim.legend()
            st.pyplot(fig_sim)

else:
    st.info("Awaiting data... Generating default dataset if none provided.")
