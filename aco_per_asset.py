import pandas as pd
import numpy as np
from strategy_logic import calculate_metrics
import random

def run_single_asset_backtest(prices, ticker, sma_len, roc_len, stop_loss_type, stop_loss_val, initial_capital=10_000_000):
    if isinstance(prices[ticker], pd.DataFrame):
        p = prices[ticker].iloc[:, 0].ffill().bfill()
    else:
        p = prices[ticker].ffill().bfill()
    sma = p.rolling(sma_len).mean()
    roc = p.pct_change(roc_len)

    ma_stop = None
    if stop_loss_type == 'ma':
        ma_stop = p.rolling(int(stop_loss_val)).mean()

    dates = p.index
    n = len(dates)
    cash = initial_capital
    shares = 0
    max_price = 0

    equity = pd.Series(index=dates, dtype=float)

    pending_buy = False
    pending_sell = False

    start_idx = max(sma_len, roc_len)
    if stop_loss_type == 'ma':
        start_idx = max(start_idx, int(stop_loss_val))

    if start_idx >= n:
        return pd.Series([initial_capital]*n, index=dates)

    for i in range(start_idx, n):
        curr_p = p.iloc[i]

        if pending_buy:
            shares = cash / curr_p
            cash = 0
            max_price = curr_p
            pending_buy = False
        elif pending_sell:
            cash = shares * curr_p
            shares = 0
            pending_sell = False

        equity.iloc[i] = cash + shares * curr_p

        if shares > 0:
            max_price = max(max_price, curr_p)
            stop = False
            if stop_loss_type == 'peak':
                if curr_p < max_price * (1 - stop_loss_val):
                    stop = True
            elif stop_loss_type == 'ma':
                if curr_p < ma_stop.iloc[i]:
                    stop = True

            if stop or curr_p < sma.iloc[i] or roc.iloc[i] <= 0:
                pending_sell = True
        else:
            if curr_p > sma.iloc[i] and roc.iloc[i] > 0:
                pending_buy = True

    return equity.ffill().fillna(initial_capital)

class ACOOptimizer:
    def __init__(self, prices, ticker, sma_range, roc_range, sl_range, stop_loss_type, iterations=10, ants=10):
        self.prices = prices
        self.ticker = ticker
        self.sma_range = sma_range
        self.roc_range = roc_range
        self.sl_range = sl_range
        self.stop_loss_type = stop_loss_type
        self.iterations = iterations
        self.ants = ants
        self.ph_sma = {v: 1.0 for v in sma_range}
        self.ph_roc = {v: 1.0 for v in roc_range}
        self.ph_sl = {v: 1.0 for v in sl_range}

    def select(self, pheromones):
        vals = list(pheromones.keys())
        probs = [pheromones[v] for v in vals]
        total = sum(probs)
        probs = [p/total for p in probs]
        return random.choices(vals, weights=probs)[0]

    def optimize(self):
        best_params = None
        best_calmar = -np.inf
        for _ in range(self.iterations):
            solutions = []
            for _ in range(self.ants):
                s = self.select(self.ph_sma)
                r = self.select(self.ph_roc)
                sl = self.select(self.ph_sl)
                eq = run_single_asset_backtest(self.prices, self.ticker, s, r, self.stop_loss_type, sl)
                calmar = calculate_metrics(eq)['Calmar']
                solutions.append(((s, r, sl), calmar))
                if calmar > best_calmar:
                    best_calmar = calmar
                    best_params = (s, r, sl)
            for d in [self.ph_sma, self.ph_roc, self.ph_sl]:
                for k in d: d[k] *= 0.7
            for params, calmar in solutions:
                if calmar > 0:
                    s, r, sl = params
                    self.ph_sma[s] += calmar
                    self.ph_roc[r] += calmar
                    self.ph_sl[sl] += calmar
        return best_params, best_calmar
