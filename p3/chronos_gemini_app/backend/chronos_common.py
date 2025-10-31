import numpy as np
import pandas as pd

def load_dataset(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    if filepath.endswith('.parquet'):
        return pd.read_parquet(filepath)
    raise ValueError("Unsupported format")

def seasonal_naive_forecast(context, pred_len, season_len):
    return np.tile(context[-season_len:], (pred_len // season_len + 1))[:pred_len]

def mase(actual, pred, context, season_len):
    naive = seasonal_naive_forecast(context, len(actual), season_len)
    mae_model = np.mean(np.abs(actual - pred))
    mae_naive = np.mean(np.abs(actual - naive))
    return mae_model / mae_naive if mae_naive != 0 else np.nan

def wql(actual, quantile_preds, levels):
    losses = []
    for q, pred in zip(levels, quantile_preds.T):
        err = actual - pred
        loss = np.mean(2 * (q * err[err >= 0] + (q - 1) * err[err < 0]))
        losses.append(loss)
    return np.mean(losses)