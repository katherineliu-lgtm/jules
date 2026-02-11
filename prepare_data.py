import pandas as pd
import numpy as np

def load_and_clean_data(path):
    df = pd.read_excel(path, header=[0, 1], index_col=0)
    df.index = pd.to_datetime([str(i)[:8] for i in df.index], format='%Y%m%d', errors='coerce')
    df = df[df.index.notnull()]
    new_level0 = [str(col).split('.')[0] for col in df.columns.get_level_values(0)]
    new_level1 = [str(col) for col in df.columns.get_level_values(1)]
    df.columns = pd.MultiIndex.from_tuples(zip(new_level0, new_level1), names=['Ticker', 'Name'])
    if ('股票代號', '日期') in df.columns:
        df = df.drop(columns=[('股票代號', '日期')])
    df = df.loc[:, ~df.columns.get_level_values(0).duplicated()]
    df = df.ffill().bfill()
    return df

if __name__ == "__main__":
    df = load_and_clean_data('個股1.xlsx')
    df.to_pickle('cleaned_data.pkl')
