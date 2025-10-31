import numpy as np

def load_dataset(filepath):
    """Load dataset from CSV or Parquet."""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    raise ValueError("Unsupported file format")

def seasonal_naive_forecast(context, prediction_length, season_length):
    """Seasonal naive forecast baseline."""
    return np.tile(context[-season_length:], (prediction_length // season_length + 1))[:prediction_length]

def mase(actual, predicted, context, season_length):
    """Mean Absolute Scaled Error."""
    naive = seasonal_naive_forecast(context, len(actual), season_length)
    mae_model = np.mean(np.abs(actual - predicted))
    mae_naive = np.mean(np.abs(actual - naive))
    return mae_model / mae_naive if mae_naive != 0 else np.nan

def wql(actual, quantile_preds, quantile_levels):
    """Weighted Quantile Loss."""
    losses = []
    for q, pred in zip(quantile_levels, quantile_preds.T):
        err = actual - pred
        loss = np.mean(2 * (q * err[err >= 0] + (q - 1) * err[err < 0]))
        losses.append(loss)
    return np.mean(losses)