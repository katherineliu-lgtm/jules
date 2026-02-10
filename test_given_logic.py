import pandas as pd
import numpy as np
import itertools

def prepare_data(filepath):
    df = pd.read_excel(filepath)
    df['時間'] = pd.to_datetime(df['時間'])
    df.set_index('時間', inplace=True)
    return df.sort_index()

def calculate_momentum(prices, periods):
    returns = []
    for p in periods:
        ret = prices.pct_change(p)
        returns.append(ret)
    return pd.concat(returns, axis=1).mean(axis=1)

def run_backtest(price_df, momentum_params, initial_capital=10000000):
    mom_df = pd.DataFrame(index=price_df.index)
    for col in price_df.columns:
        p = momentum_params[col] if isinstance(momentum_params, dict) else momentum_params
        mom_df[col] = calculate_momentum(price_df[col], p)

    dates = price_df.index
    holdings = {}
    cash = initial_capital
    equity_curve = []

    for i in range(len(dates)):
        date = dates[i]
        curr_prices = price_df.iloc[i]
        curr_mom = mom_df.iloc[i]
        portfolio_value = cash + sum(s * curr_prices[a] for a, s in holdings.items() if not np.isnan(curr_prices[a]))
        equity_curve.append(portfolio_value)

        valid = curr_mom[curr_mom > 0].sort_values(ascending=False)
        targets = valid.head(2).index.tolist()

        # Rebalance - Sell
        for a in list(holdings.keys()):
            if a not in targets:
                cash += holdings.pop(a) * curr_prices[a]

        # Rebalance - Buy
        needed = [t for t in targets if t not in holdings]
        if needed:
            for a in needed:
                # This logic is a bit weird but let's follow it
                if len(holdings) == 0 and len(targets) == 2:
                    amount = portfolio_value / 2
                elif len(targets) == 2:
                    amount = cash # Use all remaining cash for the 2nd slot?
                else:
                    amount = min(cash, portfolio_value / 2)

                if amount > 0 and not np.isnan(curr_prices[a]) and curr_prices[a] > 0:
                    holdings[a] = amount / curr_prices[a]
                    cash -= amount

    return pd.Series(equity_curve, index=dates)

def calculate_metrics(eq):
    years = len(eq) / 52.18
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    mdd = ((eq - eq.cummax())/eq.cummax()).min()
    return {'CAGR': cagr, 'MaxDD': mdd, 'Calmar': cagr/abs(mdd) if mdd!=0 else 0}

df_prices = prepare_data('資料2.xlsx')
# Test their "Best Global" [6, 18]
eq = run_backtest(df_prices, [6, 18])
print("Metrics for [6, 18]:", calculate_metrics(eq))
