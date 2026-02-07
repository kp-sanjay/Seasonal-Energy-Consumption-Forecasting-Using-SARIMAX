
import pandas as pd
import numpy as np
import os

def generate_data(start_date='2015-01-01', periods=120, freq='MS'):
    """
    Generates a synthetic dataset with monthly energy consumption and temperature.
    Includes trend, seasonality, and noise.
    """
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Base trend
    trend = np.linspace(2000, 3500, periods)
    
    # Seasonality (Sine wave for yearly cycle)
    seasonality = 500 * np.sin(2 * np.pi * dates.month / 12)
    
    # Temperature (correlated with seasonality but inverted for heating/cooling load simulation)
    # Simulating a region where summer requires cooling (high temp -> high energy)
    # and winter requires heating (low temp -> high energy) - simplified to positive correlation here for summer peak
    # Let's verify standard SARIMAX assumption: Energy often U-shaped with Temp, but for linear SARIMAX, 
    # we often model one side or assume linear relationship in a range. 
    # Let's model a summer-peaking load: High Temp -> High Energy.
    temperature = 20 + 10 * np.sin(2 * np.pi * (dates.month - 3) / 12) + np.random.normal(0, 2, periods)
    
    # Add temperature effect to energy (Heating/Cooling degree days proxy)
    temp_effect = 20 * (temperature - 15)  # Simplified linear effect
    
    # Noise
    noise = np.random.normal(0, 100, periods)
    
    # Total Energy Consumption
    energy_consumption = trend + seasonality + temp_effect + noise
    
    df = pd.DataFrame({
        'Date': dates,
        'Energy_Consumption': energy_consumption,
        'Temperature': temperature
    })
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate_data()
    file_path = os.path.join('data', 'energy_data.csv')
    df.to_csv(file_path, index=False)
    print(f"Dummy data generated at {file_path}")
