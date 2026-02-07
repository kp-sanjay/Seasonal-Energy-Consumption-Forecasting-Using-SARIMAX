
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

def plot_diagnostics(residuals, figsize=(15, 10)):
    """
    Generates diagnostic plots: Residuals over time, ACF, Histogram/KDE, Q-Q Plot.
    Returns the figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Residuals over time
    axes[0, 0].plot(residuals)
    axes[0, 0].set_title('Residuals over Time')
    
    # ACF
    plot_acf(residuals, ax=axes[0, 1], lags=min(len(residuals)//2 - 1, 40))
    axes[0, 1].set_title('Autocorrelation')
    
    # Histogram / Density
    residuals.plot(kind='kde', ax=axes[1, 0], label='Density')
    axes[1, 0].set_title('Residual Distribution')
    
    # Q-Q Plot
    sm.qqplot(residuals, line='s', ax=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    
    plt.tight_layout()
    return fig

def perform_ljung_box_test(residuals, lags=10):
    """
    Performs Ljung-Box test for autocorrelation in residuals.
    """
    try:
        lb_test = sm.stats.acorr_ljungbox(residuals, lags=[lags], return_df=True)
        return lb_test
    except Exception as e:
        return str(e)

def check_normality(residuals):
    """
    Performs Shapiro-Wilk test for normality.
    """
    stat, p_value = stats.shapiro(residuals)
    return {
        'Statistic': stat,
        'p-value': p_value,
        'Normal': p_value > 0.05
    }
