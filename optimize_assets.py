import pandas as pd
import numpy as np
from itertools import combinations
import time

def calculate_calmar(equity_curve):
    if len(equity_curve) < 2: return -1
    total_return = equity_curve[-1] / equity_curve[0]
    years = len(equity_curve) / 52.18 # Approximate weeks in a year
    cagr = total_return**(1/years) - 1

    rolling_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_dd = np.min(drawdowns)

    if max_dd == 0: return cagr / 0.0001
    return cagr / abs(max_dd)

def optimize_asset(prices, asset_name):
    print(f"Optimizing {asset_name}...")
    start_time = time.time()

    # Pre-calculate ROCs
    # ROC(n) = P(t) / P(t-n) - 1
    # We use integers from 2 to 52
    n_range = list(range(2, 53))
    rocs = {}
    for n in n_range:
        rocs[n] = (prices / prices.shift(n) - 1).values

    # Pre-calculate weekly returns (shifted)
    # Signal at t determines return at t+1 (prices[t+1]/prices[t] - 1)
    returns = (prices.shift(-1) / prices - 1).values
    returns = np.nan_to_num(returns, nan=0.0)

    best_results = {
        'Single': {'calmar': -1, 'params': None},
        'Double': {'calmar': -1, 'params': None},
        'Triple': {'calmar': -1, 'params': None}
    }

    # Single
    for n in n_range:
        score = rocs[n]
        signal = (score > 0).astype(float)
        # Shift signal by 1 to align with returns?
        # Actually returns[t] is P(t+1)/P(t)-1.
        # signal[t] is based on P(t). So signal[t] applies to returns[t].
        # We need to handle NaNs in score. (score > 0) will be False for NaN.

        eq_ret = signal * returns
        equity = np.exp(np.cumsum(np.log1p(eq_ret)))
        calmar = calculate_calmar(equity)

        if calmar > best_results['Single']['calmar']:
            best_results['Single']['calmar'] = calmar
            best_results['Single']['params'] = (n,)

    # Double
    for n1, n2 in combinations(n_range, 2):
        score = (rocs[n1] + rocs[n2]) / 2
        signal = (score > 0).astype(float)
        eq_ret = signal * returns
        equity = np.exp(np.cumsum(np.log1p(eq_ret)))
        calmar = calculate_calmar(equity)

        if calmar > best_results['Double']['calmar']:
            best_results['Double']['calmar'] = calmar
            best_results['Double']['params'] = (n1, n2)

    # Triple
    # To speed up, we can use a step for triple if it's too slow.
    # But let's see. 20k iterations is fast in numpy.
    for n1, n2, n3 in combinations(n_range, 3):
        score = (rocs[n1] + rocs[n2] + rocs[n3]) / 3
        signal = (score > 0).astype(float)
        eq_ret = signal * returns
        equity = np.exp(np.cumsum(np.log1p(eq_ret)))
        calmar = calculate_calmar(equity)

        if calmar > best_results['Triple']['calmar']:
            best_results['Triple']['calmar'] = calmar
            best_results['Triple']['params'] = (n1, n2, n3)

    end_time = time.time()
    print(f"Finished {asset_name} in {end_time - start_time:.2f}s")
    return best_results

if __name__ == "__main__":
    from process_data import load_and_clean_data
    df = load_and_clean_data('資料2.xlsx')

    all_best_params = {}
    for col in df.columns:
        res = optimize_asset(df[col], col)
        all_best_params[col] = res

    import json
    # Convert tuples to lists for JSON
    serializable = {k: {cat: {'calmar': v[cat]['calmar'], 'params': list(v[cat]['params']) if v[cat]['params'] else None}
                        for cat in v} for k, v in all_best_params.items()}
    with open('best_params.json', 'w') as f:
        json.dump(serializable, f, indent=4)
