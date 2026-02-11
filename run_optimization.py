import pandas as pd
import numpy as np
from aco_per_asset import ACOOptimizer
import json

def main():
    df = pd.read_pickle('cleaned_data.pkl')
    df_window = df.iloc[-180:]
    tickers = df.columns.get_level_values(0).unique()
    sma_range = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    roc_range = [10, 20, 30, 40, 60, 100, 125, 250]
    sl_type = 'peak'
    sl_range = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]

    best_params_all = {}
    for t in tickers:
        opt = ACOOptimizer(df_window, t, sma_range, roc_range, sl_range, sl_type, iterations=10, ants=10)
        params, calmar = opt.optimize()
        best_params_all[t] = {'sma': params[0], 'roc': params[1], 'sl_val': params[2], 'calmar': calmar}

    with open('optimal_params.json', 'w') as f:
        json.dump(best_params_all, f)

if __name__ == "__main__":
    main()
