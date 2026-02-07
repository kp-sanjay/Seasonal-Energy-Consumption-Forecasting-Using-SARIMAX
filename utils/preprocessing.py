
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def load_data(file):
    """
    Loads data from a CSV file.
    Expects 'Date' column.
    """
    try:
        df = pd.read_csv(file)
        # Attempt to parse 'Date' column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            # Infer frequency if possible, default to MS (Month Start) if monthly data looks likely
            if pd.infer_freq(df.index) is None:
                # Naive check for monthly
                if len(df) > 1:
                    diff = (df.index[1] - df.index[0]).days
                    if 28 <= diff <= 31:
                        df.index.freq = 'MS'
            else:
                 df.index.freq = pd.infer_freq(df.index)

        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def check_stationarity(timeseries):
    """
    Performs Augmented Dickey-Fuller test.
    Returns a dictionary of results.
    """
    # Drop NA just in case
    clean_series = timeseries.dropna()
    result = adfuller(clean_series)
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Stationary': result[1] < 0.05
    }

def decompose_series(timeseries, model='additive', period=12):
    """
    Performs seasonal decomposition.
    """
    # Handle missing values by interpolation for decomposition
    ts_filled = timeseries.interpolate(method='linear')
    decomposition = seasonal_decompose(ts_filled, model=model, period=period)
    return decomposition
