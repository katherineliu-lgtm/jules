import pandas as pd
import xlsxwriter
import numpy as np

def generate_excel():
    data_file = '個股合-1.xlsx'
    output_file = 'trendstrategy_formulas_equity26.xlsx'

    # Read raw data
    df_raw = pd.read_excel(data_file, header=None)
    stock_codes = df_raw.iloc[0, 2:].astype(str).values
    stock_names = df_raw.iloc[1, 2:].astype(str).values
    dates_raw = pd.to_datetime(df_raw.iloc[2:, 1])
    dates = dates_raw.dt.strftime('%Y-%m-%d').values
    prices = df_raw.iloc[2:, 2:].astype(float).values

    num_stocks = len(stock_codes)
    num_dates = len(dates)

    workbook = xlsxwriter.Workbook(output_file)

    # Formats
    header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
    date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd'})
    pct_fmt = workbook.add_format({'num_format': '0.00%'})
    num_fmt = workbook.add_format({'num_format': '#,##0'})
    price_fmt = workbook.add_format({'num_format': '#,##0.00'})

    # 1. Prices Sheet
    prices_sheet = workbook.add_worksheet('Prices')
    prices_sheet.write(0, 0, 'Date', header_fmt)
    prices_sheet.write(1, 0, '', header_fmt)
    prices_sheet.write(0, 1, 'Ticker', header_fmt)
    prices_sheet.write(1, 1, 'Name', header_fmt)
    for j in range(num_stocks):
        prices_sheet.write(0, j + 2, stock_codes[j], header_fmt)
        prices_sheet.write(1, j + 2, stock_names[j], header_fmt)
    for i in range(num_dates):
        prices_sheet.write(i + 2, 0, dates[i], date_fmt)
        prices_sheet.write(i + 2, 1, i + 1)
        for j in range(num_stocks):
            val = prices[i, j]
            if not np.isnan(val):
                prices_sheet.write(i + 2, j + 2, val)

    # 2. Calculations Sheet
    calc_sheet = workbook.add_worksheet('Calculations')
    calc_sheet.write(0, 0, 'Date', header_fmt)
    for j in range(num_stocks):
        calc_sheet.write(0, j + 1, f'SMA_{stock_codes[j]}', header_fmt)
        calc_sheet.write(0, j + 1 + num_stocks, f'ROC_{stock_codes[j]}', header_fmt)
        calc_sheet.write(0, j + 1 + 2 * num_stocks, f'Rank_{stock_codes[j]}', header_fmt)

    for i in range(num_dates):
        row_idx = i + 1
        excel_row = i + 2
        calc_sheet.write(row_idx, 0, dates[i], date_fmt)
        # SMA64
        if i >= 63:
            for j in range(num_stocks):
                p_col = xlsxwriter.utility.xl_col_to_name(j + 2)
                calc_sheet.write_formula(row_idx, j + 1, f'=AVERAGE(Prices!{p_col}{excel_row-63+1}:{p_col}{excel_row+1})', price_fmt)
        # ROC23
        if i >= 23:
            for j in range(num_stocks):
                p_col = xlsxwriter.utility.xl_col_to_name(j + 2)
                calc_sheet.write_formula(row_idx, j + 1 + num_stocks, f'=(Prices!{p_col}{excel_row+1}/Prices!{p_col}{excel_row-23+1})-1', pct_fmt)
        # Rank: Using compatible RANK() instead of RANK.EQ()
        if i >= 64:
            roc_start = xlsxwriter.utility.xl_col_to_name(num_stocks + 1)
            roc_end = xlsxwriter.utility.xl_col_to_name(2 * num_stocks)
            roc_range = f'${roc_start}${excel_row+1}:${roc_end}${excel_row+1}'
            for j in range(num_stocks):
                p_col = xlsxwriter.utility.xl_col_to_name(j + 2)
                sma_col = xlsxwriter.utility.xl_col_to_name(j + 1)
                roc_col = xlsxwriter.utility.xl_col_to_name(j + 1 + num_stocks)
                calc_sheet.write_formula(row_idx, j + 1 + 2 * num_stocks,
                    f'=IF(AND(Prices!{p_col}{excel_row+1}>Calculations!{sma_col}{excel_row+1}, Calculations!{roc_col}{excel_row+1}>0), '
                    f'RANK(Calculations!{roc_col}{excel_row+1}, {roc_range}) + {j+1}/1000, 999)')

    # Top 3 Candidates
    calc_sheet.write(0, 3 * num_stocks + 1, 'T1', header_fmt)
    calc_sheet.write(0, 3 * num_stocks + 2, 'T2', header_fmt)
    calc_sheet.write(0, 3 * num_stocks + 3, 'T3', header_fmt)
    r_start = xlsxwriter.utility.xl_col_to_name(2 * num_stocks + 1)
    r_end = xlsxwriter.utility.xl_col_to_name(3 * num_stocks)
    for i in range(64, num_dates):
        row_idx = i + 1
        excel_row = i + 2
        r_range = f'${r_start}${excel_row+1}:${r_end}${excel_row+1}'
        calc_sheet.write_formula(row_idx, 3 * num_stocks + 1, f'=IFERROR(MATCH(SMALL({r_range}, 1), {r_range}, 0), 0)')
        calc_sheet.write_formula(row_idx, 3 * num_stocks + 2, f'=IFERROR(MATCH(SMALL({r_range}, 2), {r_range}, 0), 0)')
        calc_sheet.write_formula(row_idx, 3 * num_stocks + 3, f'=IFERROR(MATCH(SMALL({r_range}, 3), {r_range}, 0), 0)')

    # 3. Portfolio Sheet
    port_sheet = workbook.add_worksheet('Portfolio')
    # Columns: Date, Rebalance, T1, T2, T3, S1...S3 Logic, Trade IDs, Total Value, Cash, Equity
    port_headers = [
        'Date', 'Rebalance', 'T1', 'T2', 'T3',
        'S1_Stock', 'S1_Entry', 'S1_Max', 'S1_Shares', 'S1_Exit', 'S1_NextStock', 'S1_TradeID',
        'S2_Stock', 'S2_Entry', 'S2_Max', 'S2_Shares', 'S2_Exit', 'S2_NextStock', 'S2_TradeID',
        'S3_Stock', 'S3_Entry', 'S3_Max', 'S3_Shares', 'S3_Exit', 'S3_NextStock', 'S3_TradeID',
        'TotalValue', 'Cash', 'Equity'
    ]
    for col, h in enumerate(port_headers):
        port_sheet.write(0, col, h, header_fmt)

    start_idx = 65
    for i in range(num_dates):
        row_idx = i + 1
        excel_row = i + 2
        port_sheet.write(row_idx, 0, dates[i], date_fmt)
        if i >= start_idx:
            # Rebalance logic: Every 5 days
            port_sheet.write_formula(row_idx, 1, f'=IF(MOD({excel_row}-{start_idx+1}, 5)=0, 1, 0)')
            # Signal from T-1
            sig_row = excel_row
            t1_col_name = xlsxwriter.utility.xl_col_to_name(3 * num_stocks + 1)
            t2_col_name = xlsxwriter.utility.xl_col_to_name(3 * num_stocks + 2)
            t3_col_name = xlsxwriter.utility.xl_col_to_name(3 * num_stocks + 3)
            port_sheet.write_formula(row_idx, 2, f'=Calculations!{t1_col_name}{sig_row}')
            port_sheet.write_formula(row_idx, 3, f'=Calculations!{t2_col_name}{sig_row}')
            port_sheet.write_formula(row_idx, 4, f'=Calculations!{t3_col_name}{sig_row}')

            # Slot 1 Logic
            if i == start_idx:
                port_sheet.write_formula(row_idx, 5, f'=C{excel_row}')
                port_sheet.write_formula(row_idx, 11, f'=IF(F{excel_row+1}>0, 1, 0)')
            else:
                port_sheet.write_formula(row_idx, 5, f'=K{excel_row}')
                port_sheet.write_formula(row_idx, 11, f'=IF(F{excel_row+1}<>F{excel_row}, L{excel_row}+1, L{excel_row})')
            p_ref = f'INDEX(Prices!$C{excel_row+1}:$EZ{excel_row+1}, F{excel_row+1})'
            if i == start_idx:
                port_sheet.write_formula(row_idx, 6, f'={p_ref}')
                port_sheet.write_formula(row_idx, 7, f'={p_ref}')
                port_sheet.write_formula(row_idx, 8, f'=10000000/G{excel_row+1}')
            else:
                port_sheet.write_formula(row_idx, 6, f'=IF(F{excel_row+1}<>F{excel_row}, {p_ref}, G{excel_row})')
                port_sheet.write_formula(row_idx, 7, f'=IF(F{excel_row+1}<>F{excel_row}, {p_ref}, MAX(H{excel_row}, {p_ref}))')
                port_sheet.write_formula(row_idx, 8, f'=IF(F{excel_row+1}<>F{excel_row}, 10000000/G{excel_row+1}, I{excel_row})')
            # Exit: Stop loss or Rebalance out
            port_sheet.write_formula(row_idx, 9, f'=IF(F{excel_row+1}=0, 1, OR({p_ref} < H{excel_row+1}*0.91, AND(B{excel_row+1}=1, NOT(OR(F{excel_row+1}=C{excel_row+1}, F{excel_row+1}=D{excel_row+1}, F{excel_row+1}=E{excel_row+1})))))')

            # Slot 2 Logic
            if i == start_idx:
                port_sheet.write_formula(row_idx, 12, f'=D{excel_row}')
                port_sheet.write_formula(row_idx, 18, f'=IF(M{excel_row+1}>0, 1, 0)')
            else:
                port_sheet.write_formula(row_idx, 12, f'=R{excel_row}')
                port_sheet.write_formula(row_idx, 18, f'=IF(M{excel_row+1}<>M{excel_row}, S{excel_row}+1, S{excel_row})')
            p_ref2 = f'INDEX(Prices!$C{excel_row+1}:$EZ{excel_row+1}, M{excel_row+1})'
            if i == start_idx:
                port_sheet.write_formula(row_idx, 13, f'={p_ref2}')
                port_sheet.write_formula(row_idx, 14, f'={p_ref2}')
                port_sheet.write_formula(row_idx, 15, f'=10000000/N{excel_row+1}')
            else:
                port_sheet.write_formula(row_idx, 13, f'=IF(M{excel_row+1}<>M{excel_row}, {p_ref2}, N{excel_row})')
                port_sheet.write_formula(row_idx, 14, f'=IF(M{excel_row+1}<>M{excel_row}, {p_ref2}, MAX(O{excel_row}, {p_ref2}))')
                port_sheet.write_formula(row_idx, 15, f'=IF(M{excel_row+1}<>M{excel_row}, 10000000/N{excel_row+1}, P{excel_row})')
            port_sheet.write_formula(row_idx, 16, f'=IF(M{excel_row+1}=0, 1, OR({p_ref2} < O{excel_row+1}*0.91, AND(B{excel_row+1}=1, NOT(OR(M{excel_row+1}=C{excel_row+1}, M{excel_row+1}=D{excel_row+1}, M{excel_row+1}=E{excel_row+1})))))')

            # Slot 3 Logic
            if i == start_idx:
                port_sheet.write_formula(row_idx, 19, f'=E{excel_row}')
                port_sheet.write_formula(row_idx, 25, f'=IF(T{excel_row+1}>0, 1, 0)')
            else:
                port_sheet.write_formula(row_idx, 19, f'=Y{excel_row}')
                port_sheet.write_formula(row_idx, 25, f'=IF(T{excel_row+1}<>T{excel_row}, Z{excel_row}+1, Z{excel_row})')
            p_ref3 = f'INDEX(Prices!$C{excel_row+1}:$EZ{excel_row+1}, T{excel_row+1})'
            if i == start_idx:
                port_sheet.write_formula(row_idx, 20, f'={p_ref3}')
                port_sheet.write_formula(row_idx, 21, f'={p_ref3}')
                port_sheet.write_formula(row_idx, 22, f'=10000000/U{excel_row+1}')
            else:
                port_sheet.write_formula(row_idx, 20, f'=IF(T{excel_row+1}<>T{excel_row}, {p_ref3}, U{excel_row})')
                port_sheet.write_formula(row_idx, 21, f'=IF(T{excel_row+1}<>T{excel_row}, {p_ref3}, MAX(V{excel_row}, {p_ref3}))')
                port_sheet.write_formula(row_idx, 22, f'=IF(T{excel_row+1}<>T{excel_row}, 10000000/U{excel_row+1}, W{excel_row})')
            port_sheet.write_formula(row_idx, 23, f'=IF(T{excel_row+1}=0, 1, OR({p_ref3} < V{excel_row+1}*0.91, AND(B{excel_row+1}=1, NOT(OR(T{excel_row+1}=C{excel_row+1}, T{excel_row+1}=D{excel_row+1}, T{excel_row+1}=E{excel_row+1})))))')

            # Next Stock selection logic
            s1k = f'IF(J{excel_row+1}, 0, F{excel_row+1})'
            s2k = f'IF(Q{excel_row+1}, 0, M{excel_row+1})'
            s3k = f'IF(X{excel_row+1}, 0, T{excel_row+1})'
            a1 = f'IF(AND(C{excel_row+1}<>{s1k}, C{excel_row+1}<>{s2k}, C{excel_row+1}<>{s3k}), C{excel_row+1}, IF(AND(D{excel_row+1}<>{s1k}, D{excel_row+1}<>{s2k}, D{excel_row+1}<>{s3k}), D{excel_row+1}, IF(AND(E{excel_row+1}<>{s1k}, E{excel_row+1}<>{s2k}, E{excel_row+1}<>{s3k}), E{excel_row+1}, 0)))'
            a2 = f'IF({a1}=C{excel_row+1}, IF(AND(D{excel_row+1}<>{s1k}, D{excel_row+1}<>{s2k}, D{excel_row+1}<>{s3k}), D{excel_row+1}, IF(AND(E{excel_row+1}<>{s1k}, E{excel_row+1}<>{s2k}, E{excel_row+1}<>{s3k}), E{excel_row+1}, 0)), IF({a1}=D{excel_row+1}, IF(AND(E{excel_row+1}<>{s1k}, E{excel_row+1}<>{s2k}, E{excel_row+1}<>{s3k}), E{excel_row+1}, 0), 0))'
            a3 = f'IF(AND(E{excel_row+1}<>{s1k}, E{excel_row+1}<>{s2k}, E{excel_row+1}<>{s3k}, E{excel_row+1}<>{a1}, E{excel_row+1}<>{a2}), E{excel_row+1}, 0)'
            port_sheet.write_formula(row_idx, 10, f'=IF(J{excel_row+1}, {a1}, F{excel_row+1})')
            port_sheet.write_formula(row_idx, 17, f'=IF(Q{excel_row+1}, IF(J{excel_row+1}, {a2}, {a1}), M{excel_row+1})')
            port_sheet.write_formula(row_idx, 24, f'=IF(X{excel_row+1}, IF(AND(J{excel_row+1}, Q{excel_row+1}), {a3}, IF(OR(J{excel_row+1}, Q{excel_row+1}), {a2}, {a1})), T{excel_row+1})')

            # Equity Calculation
            v1 = f'I{excel_row+1}*IF(F{excel_row+1}=0, 0, {p_ref})'
            v2 = f'P{excel_row+1}*IF(M{excel_row+1}=0, 0, {p_ref2})'
            v3 = f'W{excel_row+1}*IF(T{excel_row+1}=0, 0, {p_ref3})'
            port_sheet.write_formula(row_idx, 26, f'={v1}+{v2}+{v3}')
            port_sheet.write_formula(row_idx, 27, f'=(30000000 - I{excel_row+1}*G{excel_row+1} - P{excel_row+1}*N{excel_row+1} - W{excel_row+1}*U{excel_row+1})')
            port_sheet.write_formula(row_idx, 28, f'=AA{excel_row+1}+AB{excel_row+1}')

    # 4. Performance Sheet
    perf_sheet = workbook.add_worksheet('Performance')
    perf_headers = ['日期', '累積資金曲線', '報酬率 (%)', '交易編號', '進場日期', '出場日期', '標的', '進場價', '出場價', '報酬率', '持有天數']
    for col, h in enumerate(perf_headers):
        perf_sheet.write(0, col, h, header_fmt)

    for i in range(num_dates):
        row_idx = i + 1
        excel_row = i + 2
        perf_sheet.write(row_idx, 0, dates[i], date_fmt)
        perf_sheet.write_formula(row_idx, 1, f'=Portfolio!AC{excel_row}', num_fmt)
        if i > start_idx:
            perf_sheet.write_formula(row_idx, 2, f'=IF(B{excel_row-1}=0, 0, B{excel_row}/B{excel_row-1}-1)', pct_fmt)

    # Expanded trade logs for all 3 slots
    slots = [('S1', 'F', 'G', 'L'), ('S2', 'M', 'N', 'S'), ('S3', 'T', 'U', 'Z')]
    current_row = 1
    for slot_name, stock_col, entry_col, trade_id_col in slots:
        for t in range(1, 151):
            row = current_row
            current_row += 1
            perf_sheet.write(row, 3, f'{slot_name}-{t}')
            perf_sheet.write_formula(row, 4, f'=IFERROR(INDEX(Portfolio!$A$2:$A$2000, MATCH({t}, Portfolio!${trade_id_col}$2:${trade_id_col}$2000, 0)), "")', date_fmt)
            perf_sheet.write_formula(row, 5, f'=IFERROR(INDEX(Portfolio!$A$2:$A$2000, MATCH({t}, Portfolio!${trade_id_col}$2:${trade_id_col}$2000, 1)), "")', date_fmt)
            perf_sheet.write_formula(row, 6, f'=IFERROR(INDEX(Prices!$C$2:$EZ$2, INDEX(Portfolio!${stock_col}$2:${stock_col}$2000, MATCH({t}, Portfolio!${trade_id_col}$2:${trade_id_col}$2000, 0))), "")')
            perf_sheet.write_formula(row, 7, f'=IFERROR(INDEX(Portfolio!${entry_col}$2:${entry_col}$2000, MATCH({t}, Portfolio!${trade_id_col}$2:${trade_id_col}$2000, 0)), "")', price_fmt)
            perf_sheet.write_formula(row, 8, f'=IF(F{row+1}<>"", IFERROR(INDEX(Prices!$C$2:$EZ$2000, MATCH(F{row+1}, Portfolio!$A$2:$A$2000, 0)+1, INDEX(Portfolio!${stock_col}$2:${stock_col}$2000, MATCH({t}, Portfolio!${trade_id_col}$2:${trade_id_col}$2000, 0))), ""), "")', price_fmt)
            perf_sheet.write_formula(row, 9, f'=IF(I{row+1}<>"", I{row+1}/H{row+1}-1, "")', pct_fmt)
            perf_sheet.write_formula(row, 10, f'=IF(F{row+1}<>"", F{row+1}-E{row+1}, "")')

    # 5. 總績效統計 Sheet
    stat_sheet = workbook.add_worksheet('總績效統計')
    stat_sheet.write(0, 0, '指標', header_fmt)
    stat_sheet.write(0, 1, '數值', header_fmt)
    n = num_dates + 1
    stat_sheet.write(1, 0, '總交易次數')
    stat_sheet.write_formula(1, 1, f'=COUNT(Performance!J:J)')
    stat_sheet.write(2, 0, '勝率')
    stat_sheet.write_formula(2, 1, f'=COUNTIF(Performance!J:J, ">0")/COUNT(Performance!J:J)', pct_fmt)
    stat_sheet.write(3, 0, '平均報酬率')
    stat_sheet.write_formula(3, 1, f'=AVERAGE(Performance!J:J)', pct_fmt)
    stat_sheet.write(4, 0, '最大回撤')
    stat_sheet.write_formula(4, 1, f'=MIN(Performance!B2:B{n}/MAX(Performance!$B$2:B{n})-1)', pct_fmt)
    stat_sheet.write(5, 0, '年化報酬率')
    stat_sheet.write_formula(5, 1, f'=(Performance!B{n}/30000000)^(365/(Performance!A{n}-Performance!A{start_idx+2}))-1', pct_fmt)

    workbook.close()

if __name__ == "__main__":
    generate_excel()
