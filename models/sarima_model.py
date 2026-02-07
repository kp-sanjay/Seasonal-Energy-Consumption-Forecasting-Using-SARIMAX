
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAModel:
    def __init__(self, seasonal_period=12):
        self.seasonal_period = seasonal_period
        self.model = None
        self.results = None
        self.order = None
        self.seasonal_order = None

    def optimize_parameters(self, data, seasonal=True, m=12):
        """
        Uses auto_arima to find the best parameters.
        """
        # Fit auto_arima
        stepwise_model = pm.auto_arima(data, 
                                       start_p=0, start_q=0,
                                       max_p=3, max_q=3,
                                       m=m,
                                       start_P=0, seasonal=seasonal,
                                       d=None, D=1, trace=False,
                                       error_action='ignore',  
                                       suppress_warnings=True, 
                                       stepwise=True)
        
        self.order = stepwise_model.order
        self.seasonal_order = stepwise_model.seasonal_order
        return self.order, self.seasonal_order

    def fit(self, data, order=None, seasonal_order=None):
        """
        Fits the SARIMA model.
        """
        if order is None:
            if self.order is None:
                 # Default if not optimized or provided
                 order = (1, 1, 1)
            else:
                 order = self.order
        
        if seasonal_order is None:
             if self.seasonal_order is None:
                  # Default
                  seasonal_order = (1, 1, 1, self.seasonal_period)
             else:
                  seasonal_order = self.seasonal_order

        self.model = SARIMAX(data, 
                             order=order, 
                             seasonal_order=seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        self.results = self.model.fit(disp=False)
        return self.results

    def predict(self, steps):
        """
        Forecasts future values.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet.")
        
        forecast_result = self.results.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        return forecast_mean, conf_int

    def get_summary(self):
        if self.results:
            return self.results.summary()
        return "Model not fitted."
