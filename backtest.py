import pandas as pd
import numpy as np
import json

class Backtester:
    def __init__(self, prices, asset_params, initial_capital=10_000_000, num_slots=2):
        self.prices = prices
        self.asset_params = asset_params # {asset_name: [n1, n2, ...]}
        self.initial_capital = initial_capital
        self.num_slots = num_slots

        # Initialize slots
        self.slots = [{'asset': None, 'shares': 0, 'cash': initial_capital / num_slots} for _ in range(num_slots)]
        self.history = []
        self.trades = []

    def calculate_scores(self, date):
        scores = {}
        for asset in self.prices.columns:
            params = self.asset_params.get(asset)
            if params is None or len(params) == 0:
                continue

            # Get current index for the date
            idx = self.prices.index.get_loc(date)

            p_current = self.prices.iloc[idx][asset]
            if np.isnan(p_current):
                scores[asset] = -np.inf
                continue

            roc_vals = []
            for n in params:
                if idx < n:
                    roc_vals.append(-np.inf)
                else:
                    p_past = self.prices.iloc[idx - n][asset]
                    if np.isnan(p_past) or p_past == 0:
                        roc_vals.append(-np.inf)
                    else:
                        roc_vals.append(p_current / p_past - 1)

            if -np.inf in roc_vals:
                scores[asset] = -np.inf
            else:
                scores[asset] = np.mean(roc_vals)
        return scores

    def run(self):
        for i, date in enumerate(self.prices.index):
            # 1. Calculate Scores
            scores = self.calculate_scores(date)

            # 2. Filter and Rank
            qualifying = [a for a, s in scores.items() if s > 0]
            qualifying.sort(key=lambda x: scores[x], reverse=True)
            target_assets = qualifying[:self.num_slots]

            # 3. Rebalance
            week_actions = []

            # First, handle sells or keeps
            assets_to_keep = []
            for j in range(self.num_slots):
                current_asset = self.slots[j]['asset']
                if current_asset is not None:
                    if current_asset in target_assets:
                        # Keep
                        assets_to_keep.append(current_asset)
                        week_actions.append(f"保留 {current_asset}")
                    else:
                        # Sell
                        price = self.prices.loc[date, current_asset]
                        proceeds = self.slots[j]['shares'] * price
                        self.slots[j]['cash'] += proceeds
                        week_actions.append(f"賣出 {current_asset} @ {price:.2f}")
                        self.slots[j]['asset'] = None
                        self.slots[j]['shares'] = 0

            # Second, handle buys
            remaining_targets = [a for a in target_assets if a not in assets_to_keep]
            for j in range(self.num_slots):
                if self.slots[j]['asset'] is None and remaining_targets:
                    new_asset = remaining_targets.pop(0)
                    price = self.prices.loc[date, new_asset]
                    if not np.isnan(price) and price > 0:
                        shares = self.slots[j]['cash'] / price
                        self.slots[j]['shares'] = shares
                        self.slots[j]['cash'] = 0
                        self.slots[j]['asset'] = new_asset
                        week_actions.append(f"買進 {new_asset} @ {price:.2f}")

            # 4. Record State
            total_value = 0
            holdings = []
            for j in range(self.num_slots):
                asset = self.slots[j]['asset']
                shares = self.slots[j]['shares']
                cash = self.slots[j]['cash']
                if asset:
                    price = self.prices.loc[date, asset]
                    val = shares * price
                    total_value += val
                    holdings.append({'asset': asset, 'shares': shares, 'price': price, 'value': val})
                else:
                    total_value += cash
                    holdings.append({'asset': 'CASH', 'shares': cash, 'price': 1, 'value': cash})

            self.history.append({
                'Date': date,
                'TotalValue': total_value,
                'Holdings': holdings,
                'Actions': "; ".join(week_actions),
                'Scores': {a: s for a, s in scores.items() if s > -np.inf}
            })

        return pd.DataFrame(self.history)

def get_best_params_set(best_params_json, category):
    # category can be 'Single', 'Double', 'Triple', or 'AbsoluteBest'
    with open(best_params_json, 'r') as f:
        data = json.load(f)

    res = {}
    for asset, cats in data.items():
        if category == 'AbsoluteBest':
            # Pick the category with max calmar
            best_cat = max(cats.keys(), key=lambda k: cats[k]['calmar'])
            res[asset] = cats[best_cat]['params']
        else:
            res[asset] = cats[category]['params']
    return res

if __name__ == "__main__":
    from process_data import load_and_clean_data
    df = load_and_clean_data('資料2.xlsx')

    for cat in ['Single', 'Double', 'Triple', 'AbsoluteBest']:
        print(f"Running backtest for {cat}...")
        params = get_best_params_set('best_params.json', cat)
        bt = Backtester(df, params)
        results = bt.run()
        results.to_pickle(f"results_{cat}.pkl")

        # Calculate final metrics
        from process_data import calculate_metrics
        cagr, max_dd, calmar, win_rate = calculate_metrics(results.set_index('Date')['TotalValue'])
        print(f"{cat} Metrics: CAGR={cagr:.2%}, MaxDD={max_dd:.2%}, Calmar={calmar:.2f}, WinRate={win_rate:.2%}")
