import pandas as pd
import numpy as np

def run_backtest_per_asset(df, asset_params, stop_loss_type, rb_period=5, rb_offset=0, initial_capital=30_000_000):
    """
    asset_params: dict {ticker: (sma_len, roc_len, stop_loss_val)}
    stop_loss_type: 'peak' or 'ma'
    """
    prices = df.copy()
    tickers = prices.columns.get_level_values(0)
    names = prices.columns.get_level_values(1)
    ticker_to_name = dict(zip(tickers, names))
    prices.columns = tickers

    # Pre-calculate indicators for each asset
    sma_matrix = pd.DataFrame(index=prices.index, columns=tickers)
    roc_matrix = pd.DataFrame(index=prices.index, columns=tickers)
    ma_stop_matrix = pd.DataFrame(index=prices.index, columns=tickers)

    for t in tickers:
        s_len, r_len, sl_val = asset_params[t]
        sma_matrix[t] = prices[t].rolling(s_len).mean()
        roc_matrix[t] = prices[t].pct_change(r_len)
        if stop_loss_type == 'ma':
            ma_stop_matrix[t] = prices[t].rolling(int(sl_val)).mean()

    dates = prices.index
    n_days = len(dates)

    cash = initial_capital
    holdings = {}

    equity = pd.Series(index=dates, dtype=float)
    trade_log = []
    holdings_log = []

    pending_trades = []

    # Determine start index
    max_lookback = 0
    for t in tickers:
        s, r, sl = asset_params[t]
        max_lookback = max(max_lookback, s, r)
        if stop_loss_type == 'ma':
            max_lookback = max(max_lookback, int(sl))

    if max_lookback >= n_days:
        return pd.Series([initial_capital]*n_days, index=dates), [], []

    for i in range(max_lookback, n_days):
        curr_date = dates[i]
        curr_prices = prices.iloc[i]

        # 1. Execute pending trades (at T+1 close)
        if pending_trades:
            sells = [t for t in pending_trades if t['type'] == 'sell']
            buys = [t for t in pending_trades if t['type'] == 'buy']

            for trade in sells:
                ticker = trade['ticker']
                p = curr_prices[ticker]
                shares = trade['shares']
                cash += shares * p
                entry_info = holdings.pop(ticker, None)
                if entry_info:
                    trade_log.append({
                        'Date': curr_date,
                        'Ticker': ticker,
                        'Name': ticker_to_name[ticker],
                        'Type': 'Sell',
                        'Price': p,
                        'Shares': shares,
                        'Reason': trade['reason'],
                        'Entry Date': entry_info['entry_date'],
                        'Entry Price': entry_info['entry_price'],
                        'Return': (p / entry_info['entry_price']) - 1 if entry_info['entry_price'] != 0 else 0
                    })

            for trade in buys:
                ticker = trade['ticker']
                p = curr_prices[ticker]
                if p > 0:
                    amount = min(trade['amount'], cash)
                    if amount > 1e-6:
                        shares = amount / p
                        holdings[ticker] = {
                            'shares': shares,
                            'max_price': p,
                            'entry_date': curr_date,
                            'entry_price': p
                        }
                        cash -= amount
                        trade_log.append({
                            'Date': curr_date,
                            'Ticker': ticker,
                            'Name': ticker_to_name[ticker],
                            'Type': 'Buy',
                            'Price': p,
                            'Shares': shares,
                            'Reason': trade['reason'],
                            'Momentum_Value': trade.get('momentum', 0)
                        })
            pending_trades = []

        # 2. Update Equity
        port_value = cash
        for ticker, info in holdings.items():
            val = info['shares'] * curr_prices[ticker]
            port_value += val
            holdings[ticker]['max_price'] = max(holdings[ticker]['max_price'], curr_prices[ticker])
        equity.iloc[i] = port_value

        # 3. Log holdings
        holdings_log.append({
            'Date': curr_date,
            'Holdings': {t: info['shares'] for t, info in holdings.items()},
            'Equity': port_value
        })

        # 4. Signal Generation
        if i == n_days - 1: continue

        current_held_tickers = list(holdings.keys())
        for ticker in current_held_tickers:
            info = holdings[ticker]
            s_len, r_len, sl_val = asset_params[ticker]
            stop_triggered = False
            reason = ""
            if stop_loss_type == 'peak':
                if curr_prices[ticker] < info['max_price'] * (1 - sl_val):
                    stop_triggered = True
                    reason = f"Peak-to-Trough Stop ({sl_val*100:.1f}%)"
            elif stop_loss_type == 'ma':
                ma_val = ma_stop_matrix.at[curr_date, ticker]
                if curr_prices[ticker] < ma_val:
                    stop_triggered = True
                    reason = f"MA Stop ({sl_val})"

            if stop_triggered:
                if not any(t['ticker'] == ticker and t['type'] == 'sell' for t in pending_trades):
                    pending_trades.append({'ticker': ticker, 'type': 'sell', 'shares': info['shares'], 'reason': reason})

        if (i - max_lookback) % rb_period == rb_offset:
            eligible = (prices.iloc[i] > sma_matrix.iloc[i]) & (roc_matrix.iloc[i] > 0)
            eligible_roc = roc_matrix.iloc[i][eligible].sort_values(ascending=False)
            top_3 = eligible_roc.head(3).index.tolist()

            for ticker in list(holdings.keys()):
                if ticker not in top_3:
                    if not any(t['ticker'] == ticker and t['type'] == 'sell' for t in pending_trades):
                        pending_trades.append({'ticker': ticker, 'type': 'sell', 'shares': holdings[ticker]['shares'], 'reason': 'Dropped from Top 3'})

            to_buy = [t for t in top_3 if t not in holdings and not any(tr['ticker'] == t and tr['type'] == 'buy' for tr in pending_trades)]
            if to_buy:
                current_remains = [t for t in holdings if t in top_3 and not any(tr['ticker'] == t and tr['type'] == 'sell' for tr in pending_trades)]
                open_slots = 3 - len(current_remains)
                if open_slots > 0:
                    amount_per_slot = port_value / 3
                    for ticker in to_buy[:open_slots]:
                        pending_trades.append({'ticker': ticker, 'type': 'buy', 'amount': amount_per_slot,
                                               'reason': f'Top 3 ROC ({roc_matrix.at[curr_date, ticker]:.4f})',
                                               'momentum': roc_matrix.at[curr_date, ticker]})

    equity = equity.ffill().fillna(initial_capital)
    return equity, trade_log, holdings_log

def calculate_metrics(equity):
    if len(equity.dropna()) < 2: return {'CAGR': 0, 'MaxDD': 0, 'Calmar': 0, 'WinRate': 0}
    eq = equity.dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    drawdown = (eq / eq.cummax()) - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    daily_returns = eq.pct_change().dropna()
    win_rate = (daily_returns > 0).mean()

    return {
        'CAGR': cagr,
        'MaxDD': max_dd,
        'Calmar': calmar,
        'WinRate': win_rate,
        'TotalReturn': total_return
    }
