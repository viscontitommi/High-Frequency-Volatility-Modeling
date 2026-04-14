import pandas as pd
import os
import numpy as np
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay, BMonthEnd
import seaborn as sns
import Brent_Functions as BF
import WTI_Functions as WTI_F
import TFM_Functions as TFM_F
import ECF_Functions as ECF_F
import Data_Functions as DF
import random

def calculate_naive_rv(df_clean):
    prices = df_clean.iloc[:, 0].values.astype(float)
    log_returns = np.diff(np.log(prices))
    # Qualsiasi rendimento istantaneo > 25% (0.25) viene considerato errore dati
    # e forzato a 0.0. Questo rimuove l'errore del prezzo 160.000 trovato nel primo file ad esempio.
    threshold = 0.25
    is_outlier = np.abs(log_returns) > threshold
    log_returns[is_outlier] = 0.0

    squared_returns = log_returns ** 2
    dates = df_clean.index[1:]
    series_sq_ret = pd.Series(squared_returns, index=dates)
    rv_daily = series_sq_ret.resample('D').sum()
    rv_daily = rv_daily[rv_daily > 0]
    
    return rv_daily

# ------------------------------------- DESCRIPTIVE STATISTICS TABLE -------------------------------------------------

def create_statistics_table(df_brent, df_wti, df_ecf, df_tfm):
    datasets = [
        ('BRN', df_brent),
        ('WTI', df_wti),
        ('ECF', df_ecf),
        ('TFM', df_tfm)
    ]
    
    stats_list = []
    
    for name, df in datasets:
        if 'Realized_Volatility' in df.columns:
            series = df['Realized_Volatility']
        else:
            series = df.iloc[:, 0]
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
    
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
    
        ann_vol = np.sqrt(mean_val) * np.sqrt(252) * 100
        
        stats_list.append({
            'Ticker': name,
            'Mean': mean_val,
            'Std Dev': std_val,
            'Min': min_val,
            'Max': max_val,
            'Ann. Vol (%)': ann_vol
        })

    df_table = pd.DataFrame(stats_list)
    return df_table.set_index('Ticker').T

# ------------------------------------- SIGNATURE PLOT -------------------------------------------------

def plot_signature(vol_data, freq_seconds, asset_name):
    means = []
    freq_labels = ['1s', '5s', '15s', '30s', '1min', '2min', '5min', '10min', '20min']
    
    for label in freq_labels:
        vals = vol_data[label]
        if len(vals) > 0:
            means.append(np.mean(vals))
        else:
            means.append(np.nan)

    plt.figure(figsize=(10, 6))
    plt.plot(freq_seconds, means, marker='o', linestyle='-', linewidth=2, color='royalblue')
    
    plt.title(f'Volatility Signature Plot: {asset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Sampling Frequency (Log Scale)', fontsize=12)
    plt.ylabel('Annualized Volatility (%)', fontsize=12)
    
    plt.xscale('log')
    plt.xticks(freq_seconds, freq_labels)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.show()

# ------------------------------------- TSRV -------------------------------------------------

def calculate_tsrv(prices_array, K):
    p = prices_array.astype(float)
    p = p[p > 0]
    n = len(p)
    if n <= K + 2:
        return np.nan
    
    log_p = np.log(p)
    # Fast Scale
    r_all = np.diff(log_p)
    rv_all = np.sum(r_all**2)
    # Slow Scale
    # log_p[i+K] - log_p[i]
    # Questo array contiene tutti i rendimenti a passo K sovrapposti
    r_slow = log_p[K:] - log_p[:-K]
    rv_avg = np.sum(r_slow**2) / K
    
    # Correzione Bias
    n_bar = (n - K + 1) / K
    denom = 1 - (n_bar / n)
    if denom == 0:
        return np.nan
    
    tsrv = (1 / denom) * (rv_avg - (n_bar / n) * rv_all)
    
    return max(0.0, tsrv)

def get_clean_data_for_file(file_path, filename, asset_name):
    
    if "TFM" in asset_name or "ECF" in asset_name:
        df = DF.load_order_eur(file_path)
        df = DF.filter_trading_hours(df, start_time='08:00', end_time='18:00')
    else:
        df = DF.load_order_doll(file_path)
        df = DF.filter_trading_hours(df, start_time='08:00', end_time='22:00')


    # Qua stiamo sistemando prezzi errati nei csv (troppo alti e non avevano senso)
    if 'Price (€)' in df.columns:
        condition1 = df['Price (€)'] > 800
        df.loc[condition1, 'Price (€)'] = df.loc[condition1, 'Price (€)'] / 10000
    elif 'Price ($)' in df.columns:
        condition2 = df['Price ($)'] > 800
        df.loc[condition2, 'Price ($)'] = df.loc[condition2, 'Price ($)'] / 10000
    
    if "BRN" in asset_name: 
        df = DF.filter_brent_expiry(df, filename)
    elif "WBS" in asset_name: 
        df = DF.filter_wti_expiry(df, filename)
    elif "TFM" in asset_name: 
        df = DF.filter_tfm_expiry(df, filename)
    elif "ECF" in asset_name: 
        df = DF.filter_ecf_expiry(df, filename)

    if df.empty: 
        return None

    if "TFM" in asset_name or "ECF" in asset_name:
        df_final = DF.apply_median_aggregation_eur(df, freq='1s')
    else:
        df_final = DF.apply_median_aggregation_doll(df, freq='1s')
    # Safety Check
    if len(df_final) < 200: 
        return None
            
    return df_final

def compute_tsrv_sensitivity_data(input_folder, asset_name, n_files=50):
    k_test_values = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300]
    results = {k: [] for k in k_test_values} #daily vol
    
    all_files = sorted([f for f in os.listdir(input_folder) if f.startswith(asset_name) and f.endswith('.csv')])
    
    if len(all_files) > n_files:
        selected_files = random.sample(all_files, n_files)
    else:
        selected_files = all_files

    for filename in selected_files:
        try:
            file_path = os.path.join(input_folder, filename)
            df_clean = get_clean_data_for_file(file_path, filename, asset_name)
            
            if df_clean is None or df_clean.empty:
                continue
            #Raggruppiamo i dati per data
            #calcoliamo la varianza di 1 giorno e la annualizziamo.
            for date, day_group in df_clean.groupby(df_clean.index.date):
                
                # Se il giorno ha meno di 100 tick, è inutile per il TSRV
                # dato dalla condizione N > K
                if len(day_group) < 100:
                    continue
                
                if isinstance(day_group, pd.DataFrame):
                    cols = day_group.select_dtypes(include=[np.number]).columns
                    if len(cols) > 0:
                        prices = day_group[cols[0]].values 
                    else:
                        prices = day_group.iloc[:, 0].values
                else:
                    prices = day_group.values
                
                prices = prices.astype(float)

                # Calcolo TSRV per ogni K su questo giorno
                for k in k_test_values:
                    # Se K è troppo grande per i dati di questo giorno specifico, salta
                    if len(prices) <= k + 2:
                        continue

                    val_tsrv = calculate_tsrv(prices, K=k)
                    
                    if not np.isnan(val_tsrv) and val_tsrv > 0:
                        ann_vol = np.sqrt(val_tsrv * 252) * 100
                        results[k].append(ann_vol)
                        
        except Exception as e:
            continue

    return results

def plot_tsrv_sensitivity(results_data, asset_name):
    k_values = sorted(results_data.keys())
    median_curve = []
    valid_k = []

    # Calculate Robust Median for each K
    for k in k_values:
        vals = results_data[k]
        if len(vals) > 0:
            median_curve.append(np.median(vals))
            valid_k.append(k)
            
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(valid_k, median_curve, marker='o', linestyle='-', color='purple', linewidth=2, label='TSRV (Median)')
    
    # Theoretical Optimal Region (Visual Guide)
    # Assuming n ~ 20,000 -> K ~ 73. We highlight the 50-100 zone.
    plt.axvspan(50, 100, color='gray', alpha=0.1, label='Theoretical Optimal Region ($n^{2/3}$)')
    
    plt.title(f'TSRV Sensitivity Analysis (K-Plot): {asset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Parameter K (Subsampling Scale)', fontsize=12)
    plt.ylabel('Annualized TSRV (%)', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.xscale('log') 
    
    plt.show()

def run_production_final(input_folder, asset_name):
    results = []
    all_files = sorted([
        f for f in os.listdir(input_folder) 
        if f.startswith(asset_name) and f.endswith('.csv')
    ])
    
    for i, filename in enumerate(all_files):
        try:
            file_path = os.path.join(input_folder, filename)
            
            df_clean = get_clean_data_for_file(file_path, filename, asset_name)
            if df_clean is None or df_clean.empty: 
                continue
            
            for date, day_group in df_clean.groupby(df_clean.index.date):
                n = len(day_group)
                if n < 100: 
                    continue 
                
                prices = day_group.iloc[:, 0].values
                # K = n^(2/3)
                k_opt = int(np.floor(n**(2/3)))
                # Safety check: K deve essere almeno 1
                if k_opt < 1: 
                    k_opt = 1
                
                tsrv_variance = calculate_tsrv(prices, K=k_opt)
                
                if not np.isnan(tsrv_variance) and tsrv_variance > 0:
                    # Annualizzazione: Radice(Varianza * 252) * 100
                    ann_vol = np.sqrt(tsrv_variance * 252) * 100
                    
                    results.append({
                        'Date': date,
                        'TSRV': ann_vol,
                        'n_obs': n,
                        'K_used': k_opt
                    })
                        
        except Exception as e:
            continue

    df_results = pd.DataFrame(results)
    
    if not df_results.empty:
        df_results['Date'] = pd.to_datetime(df_results['Date'])
        df_results.set_index('Date', inplace=True)
        df_results = df_results.sort_index()

    return df_results

def keep_most_liquid_daily(df):
    """
    Rimuove i duplicati giornalieri mantenendo solo il contratto 
    con il maggior numero di osservazioni (n_obs).
    """
    df_temp = df.reset_index()
    #ordina per Data e poi per n_obs (dal più grande al più piccolo)
    df_temp = df_temp.sort_values(by=['Date', 'n_obs'], ascending=[True, False])
    df_unique = df_temp.drop_duplicates(subset='Date', keep='first') # riga con n_obs massimo
    df_unique = df_unique.set_index('Date').sort_index()
    
    return df_unique

def plot_volatility_trends(df):
    """Grafico Serie Storiche: Oil vs Energy"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Pannello 1: Global Oil
    axes[0].plot(df.index, df['Brent'], label='Brent Crude', color='#004C99', linewidth=1, alpha=0.9)
    axes[0].plot(df.index, df['WTI'], label='WTI Crude', color='#CC0000', linewidth=1, alpha=0.7)
    axes[0].set_title('Global Oil Markets Volatility (TSRV)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Annualized Volatility (%)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Pannello 2: European Energy
    axes[1].plot(df.index, df['TTF_Gas'], label='Dutch TTF Gas', color='#00994C', linewidth=1.2)
    axes[1].plot(df.index, df['EUA_Carbon'], label='EUA Carbon', color='#E67E22', linewidth=1, alpha=0.9)
    axes[1].set_title('European Energy Markets Volatility (TSRV)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Annualized Volatility (%)')
    axes[1].set_xlabel('Year')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()