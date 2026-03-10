import pandas as pd
import numpy as np
import os

def load_and_clean_data(filepath):
    # Load data
    df = pd.read_excel(filepath)

    # Rename columns if necessary
    # Based on previous check, columns are ['時間', '台灣50 (0050)', ...]
    df['時間'] = pd.to_datetime(df['時間'])
    df.set_index('時間', inplace=True)
    df.sort_index(inplace=True)

    # Handling missing values
    # Memory says: Use .ffill() followed by .bfill() to handle internal gaps without look-ahead bias
    # But for momentum, we should be careful about bfill() before the asset actually exists.
    # Actually, if the asset doesn't exist yet (NaN at the start), we should leave it NaN.
    # ffill() is good for internal gaps.
    df = df.ffill()

    return df

def calculate_metrics(equity_curve):
    if len(equity_curve) < 2:
        return 0, 0, 0, 0

    # CAGR
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years == 0:
        cagr = 0
    else:
        cagr = total_return**(1/years) - 1

    # MaxDD
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    if max_dd == 0:
        calmar = 0
    else:
        calmar = cagr / abs(max_dd)

    # Win Rate (weekly)
    returns = equity_curve.pct_change().dropna()
    win_rate = (returns > 0).mean()

    return cagr, max_dd, calmar, win_rate

if __name__ == "__main__":
    df = load_and_clean_data('資料2.xlsx')
    print("Data loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 5 dates:", df.index[:5].tolist())
    print("Last 5 dates:", df.index[-5:].tolist())
