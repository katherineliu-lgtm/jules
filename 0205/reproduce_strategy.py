import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta

# ==========================================
# 策略參數與設定
# ==========================================
INITIAL_CAPITAL = 30_000_000
MAX_HOLDINGS = 3
REBALANCE_INTERVAL = 5
SMA_PERIOD = 50
ROC_PERIOD = 50
FILE_PATH = '0205/data.xlsx'
RESULT_EXCEL = '0205/strategy_results.xlsx'
REPRODUCE_MD = '0205/reproduce_strategy.md'

# ==========================================
# 資料載入與預處理
# ==========================================
def load_and_preprocess_data(filepath):
    # Read the excel file with multi-index header
    df = pd.read_excel(filepath, header=[0, 1])

    # Handle duplicates in Level 0 (Ticker)
    tickers_all = df.columns.get_level_values(0)
    df = df.loc[:, ~tickers_all.duplicated()]

    # Extract dates
    dates = pd.to_datetime(df[('日期', '日期')])
    prices = df.drop(columns=[('日期', '日期')])

    # Tickers and Names
    tickers = prices.columns.get_level_values(0)
    names = prices.columns.get_level_values(1)
    ticker_to_name = dict(zip(tickers, names))

    # Simplify columns and index
    prices.columns = tickers
    prices.index = dates

    # Data Cleaning Rules:
    # 1. Fill gaps/trailing NaNs with forward fill (ffill)
    # 2. Fill leading NaNs with the first available price (bfill)
    # Note: ffill first, then bfill to avoid look-ahead bias for gaps in the middle.
    prices = prices.ffill().bfill()

    return prices, ticker_to_name

# ==========================================
# 回測模組
# ==========================================
def run_backtest(prices, sma_period, roc_period, ticker_to_name, initial_capital=INITIAL_CAPITAL):
    sma = prices.rolling(window=sma_period).mean()
    roc = (prices / prices.shift(roc_period)) - 1

    dates = prices.index
    equity = pd.Series(index=dates, data=0.0)
    equity.iloc[:sma_period] = initial_capital

    cash = initial_capital
    holdings = {}  # {ticker: {'shares': s, 'buy_price': p, 'buy_date': d, ...}}

    closed_trades = []
    daily_records = []
    holdings_log = []

    signal_date_t = None
    target_tickers = []
    pending_rebalance = False

    for i in range(sma_period, len(dates)):
        current_date = dates[i]

        # Calculate current equity
        current_val = cash
        for t, info in holdings.items():
            current_val += info['shares'] * prices.loc[current_date, t]
        equity.iloc[i] = current_val

        # Signal Generation (T Day)
        if (i - sma_period) % REBALANCE_INTERVAL == 0:
            signal_date_t = current_date
            curr_p = prices.iloc[i]
            curr_sma = sma.iloc[i]
            curr_roc = roc.iloc[i]

            # Condition: Price > SMA and ROC > 0
            mask = (curr_p > curr_sma) & (curr_roc.notna()) & (curr_roc > 0)
            eligible = curr_roc[mask].sort_values(ascending=False)
            target_tickers = eligible.head(MAX_HOLDINGS).index.tolist()

            pending_rebalance = True
            daily_records.append({'日期': current_date, '事件': '訊號產生', '細節': f'目標持股: {target_tickers}'})
        else:
            daily_records.append({'日期': current_date, '事件': '觀察', '細節': ''})

        # Execution (T+1 Day)
        if pending_rebalance and current_date > signal_date_t:
            trade_date = current_date

            # 1. Sell holdings not in target
            current_held_tickers = list(holdings.keys())
            for t in current_held_tickers:
                if t not in target_tickers:
                    info = holdings.pop(t)
                    sell_price = prices.loc[trade_date, t]
                    profit = (sell_price - info['buy_price']) * info['shares']
                    closed_trades.append({
                        'Ticker': t, '名稱': ticker_to_name.get(t, t),
                        '買入日': info['buy_date'], '賣出日': trade_date,
                        '買入價': info['buy_price'], '賣出價': sell_price,
                        '股數': info['shares'], '損益': profit,
                        '報酬率': (sell_price / info['buy_price']) - 1
                    })
                    cash += info['shares'] * sell_price

            # 2. Identify new tickers to buy
            new_tickers = [t for t in target_tickers if t not in holdings]

            # 3. Buy new tickers with available cash
            if new_tickers:
                # Divide available cash equally among NEW tickers
                # As per "等權重分配" interpreted as distributing available funds to new ones
                # while keeping existing shares.
                buy_budget = cash / len(new_tickers)
                for t in new_tickers:
                    p = prices.loc[trade_date, t]
                    shares = buy_budget // p
                    if shares > 0:
                        holdings[t] = {
                            'shares': shares,
                            'buy_date': trade_date,
                            'buy_price': p
                        }
                        cash -= shares * p

            pending_rebalance = False
            daily_records[-1]['事件'] = '執行交易'
            daily_records[-1]['細節'] = f'當前持股: {list(holdings.keys())}'

        holdings_log.append({
            '日期': current_date,
            '持股數': len(holdings),
            '持股明細': ', '.join(holdings.keys()),
            '現金': cash,
            '總資產': current_val
        })

    return equity, closed_trades, holdings_log, daily_records

# ==========================================
# 績效指標計算
# ==========================================
def calculate_metrics(equity):
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    days = (equity.index[-1] - equity.index[0]).days
    cagr = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return cagr, max_dd, calmar

# ==========================================
# 主程式
# ==========================================
def main():
    print("載入資料中...")
    prices, ticker_to_name = load_and_preprocess_data(FILE_PATH)

    print(f"執行策略 (SMA={SMA_PERIOD}, ROC={ROC_PERIOD})...")
    equity, trades, hold_log, daily_rec = run_backtest(prices, SMA_PERIOD, ROC_PERIOD, ticker_to_name)

    cagr, max_dd, calmar = calculate_metrics(equity)

    print(f"CAGR: {cagr:.2%}")
    print(f"MaxDD: {max_dd:.2%}")
    print(f"Calmar Ratio: {calmar:.2f}")

    # 儲存結果
    trades_df = pd.DataFrame(trades)
    hold_log_df = pd.DataFrame(hold_log)
    daily_rec_df = pd.DataFrame(daily_rec)

    with pd.ExcelWriter(RESULT_EXCEL, engine='xlsxwriter') as writer:
        trades_df.to_excel(writer, sheet_name='Trades', index=False)
        pd.DataFrame({'Equity': equity, 'Drawdown': (equity - equity.cummax())/equity.cummax()}).to_excel(writer, sheet_name='Equity_Curve')
        hold_log_df.to_excel(writer, sheet_name='Holdings', index=False)
        daily_rec_df.to_excel(writer, sheet_name='Daily_Record', index=False)
        summary = pd.DataFrame({
            'Metric': ['CAGR', 'MaxDD', 'Calmar Ratio', 'Final Equity'],
            'Value': [f"{cagr:.2%}", f"{max_dd:.2%}", f"{calmar:.2f}", f"{equity.iloc[-1]:,.0f}"]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

    # 產出 MD 報告
    with open(REPRODUCE_MD, 'w', encoding='utf-8') as f:
        f.write(f"# 策略執行報告\n\n")
        f.write(f"## 績效摘要\n")
        f.write(f"- **CAGR**: {cagr:.2%}\n")
        f.write(f"- **MaxDD**: {max_dd:.2%}\n")
        f.write(f"- **Calmar Ratio**: {calmar:.2f}\n")
        f.write(f"- **最終資產**: {equity.iloc[-1]:,.0f}\n\n")
        f.write(f"## 策略參數\n")
        f.write(f"- SMA 週期: {SMA_PERIOD}\n")
        f.write(f"- ROC 週期: {ROC_PERIOD}\n")
        f.write(f"- 持股上限: {MAX_HOLDINGS}\n")
        f.write(f"- 再平衡週期: {REBALANCE_INTERVAL} 天\n")
        f.write(f"- 初始資金: {INITIAL_CAPITAL:,.0f}\n")

    print(f"結果已儲存至 {RESULT_EXCEL} 與 {REPRODUCE_MD}")

if __name__ == "__main__":
    main()
