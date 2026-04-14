import pandas as pd
import os
import numpy as np
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay, BMonthEnd
from scipy.stats import norm
import seaborn as sns
import Brent_Functions as BF
import WTI_Functions as WTI_F
import TFM_Functions as TFM_F
import ECF_Functions as ECF_F
import Data_Functions as DF
import RV_Functions as RV
import LHAR_CJ as lhar_cj

def clean_wti_bad_data(df):
    """
    Funzione di Pulizia per il WTI.
    Gestisce gli errori trovati.
    """
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    col_price = df.columns[0] 
    # Rimuove Prezzi Negativi o zero
    # Rimuove i 97 tick problematici del 2020
    df_clean = df[df[col_price] > 0.01].copy()
    
    if df_clean.empty:
        return df_clean
    # RIMUOVE I 71.389 FLASH CRASH (Bad Ticks)
    prices = df_clean[col_price].values
    pct_changes = np.zeros(len(prices))
    pct_changes[1:] = np.abs(np.diff(prices) / prices[:-1])
    # Filtro: teniamo solo i prezzi che NON hanno fatto un salto > 15% in 1 secondo
    mask_valid = pct_changes < 0.15
    mask_valid[0] = True # Il primo prezzo è sempre buono
    
    df_final = df_clean[mask_valid]
    
    return df_final

def process_asset_folder(folder_path, asset_name):
    accumulated_data = []
    files = sorted(os.listdir(folder_path))
    
    for filename in files:
        if filename.endswith(".csv") and not filename.startswith('.'):
            file_path = os.path.join(folder_path, filename)
            df_clean = RV.get_clean_data_for_file(file_path, filename, asset_name)
            
            if df_clean is not None and not df_clean.empty:
                if "WTI" in asset_name or "WBS" in filename:
                    df_clean = clean_wti_bad_data(df_clean)
                if not df_clean.empty:
                    accumulated_data.append(df_clean)

    if accumulated_data:
        df_final = pd.concat(accumulated_data)
        df_final = df_final.sort_index()
        df_final = df_final[~df_final.index.duplicated(keep='first')]
        
        return df_final
    else:
        return None

def calculate_tbv_value(log_returns):
    abs_returns = np.abs(log_returns)
    n = len(log_returns)
    # Stima Bipower Semplice
    bv_prelim = (np.pi/2) * np.sum(abs_returns[1:] * abs_returns[:-1])
    local_vol_est = np.sqrt(bv_prelim / n)
    # Treshold: 3 deviazioni standard
    threshold = 3.0 * local_vol_est
    # Pulizia, se |r| > soglia -> 0
    returns_clean = np.where(abs_returns > threshold, 0, abs_returns)
    # TBV Calculus
    tbv_val = (np.pi / 2) * np.sum(returns_clean[1:] * returns_clean[:-1])
    
    return tbv_val, abs_returns

def apply_ctz_test_tsrv(rv_val, tbv_val, tsrv_val, log_returns, abs_returns, alpha=0.999):
    """
    Esegue il test statistico Z-Score (C-T-Z) usando il TSRV come riferimento per la volatilità totale.
    
    Input:
    - rv_val: Realized Volatility a 5 min (usata SOLO per il test)
    - tbv_val: Threshold Bipower Variation a 5 min (usata per il test)
    - tsrv_val: Two-Scale RV a 1 secondo (usata come 'Verità' per l'assegnazione finale)
    - log_returns: rendimenti a 5 min
    """
    
    # 1. Controlli preliminari (No Jump se TBV > RV o errori numerici)
    # Nota: Anche se RV < TBV, usiamo TSRV come componente continua totale
    if tbv_val <= 1e-9 or rv_val <= tbv_val:
        return tsrv_val, 0.0  # C = TSRV, J = 0
        
    # 2. Calcolo Tri-Power Quarticity (TPQ) sui dati a 5 min
    n = len(log_returns)
    mu4_3 = 0.83088 # Costante più precisa: 2^(2/3) * gamma(7/6) / gamma(1/2)
    
    # TPQ serve per stimare la varianza integrata quartica
    tpq = n * (mu4_3**-3) * np.sum((abs_returns[:-2] * abs_returns[1:-1] * abs_returns[2:])**(4/3))
    
    if tpq <= 1e-9:
        return tsrv_val, 0.0 # Test fallito -> No Jump -> Tutto TSRV

    # 3. Z-Score (usando RV e TBV a 5 minuti, perché il test è calibrato lì)
    theta_const = (np.pi**2 / 4) + np.pi - 5 # ~ 0.609
    var_asintotica = theta_const * (tpq / n)
    
    z_score = (rv_val - tbv_val) / np.sqrt(var_asintotica)
    
    # 4. Assegnazione Componenti (usando TSRV come Volatilità Totale)
    critical_value = norm.ppf(alpha) # ~3.09
    
    if z_score > critical_value:
        # H1: C'è un salto significativo
        # La componente continua è la TBV (robusta ai salti)
        C_var = tbv_val 
        
        # Il salto è la differenza tra la 'Verità Totale' (TSRV) e la continua
        # Usiamo max(0, ...) per sicurezza numerica
        J_var = max(0.0, tsrv_val - tbv_val)
    else:
        # H0: Il salto è solo rumore
        # Tutta la volatilità (TSRV) è attribuita alla componente continua
        J_var = 0.0
        C_var = tsrv_val
        
    return C_var, J_var

def calculate_jump_math_daily(df_clean_1s, alpha=0.999, K_tsrv=300):
    df_5min = df_clean_1s.resample('5min').last().dropna()
    unique_days = sorted(list(set(df_5min.index.date)))
    daily_results = []

    for day in unique_days:
        # Dati 5 minuti (per Test Z e TBV)
        day_str = str(day)
        try:
            group_5min = df_5min.loc[day_str]
            group_1s = df_clean_1s.loc[day_str]
        except KeyError:
            continue
        if len(group_5min) < 10: 
            continue

        # Calcoli su 5 Minuti (Input per il Test)
        prices_5m = group_5min.iloc[:, 0].values.astype(float)
        log_ret_5m = np.diff(np.log(prices_5m))
        # RV Classica 5min (serve solo come input per il test Z)
        rv_day_5min = np.sum(log_ret_5m**2)
        # TBV 5min
        tbv_val, abs_returns = calculate_tbv_value(log_ret_5m)
        # Calcolo TSRV su 1 Secondo (Input per la Volatilità Reale)
        prices_1s = group_1s.iloc[:, 0].values.astype(float)
        tsrv_val = RV.calculate_tsrv(prices_1s, K=K_tsrv)
        
        # Se TSRV fallisce (NaN), fallback sulla RV 5min o salta
        if np.isnan(tsrv_val):
            tsrv_val = rv_day_5min

        # Applicazione Test e Assegnazione
        c_var_day, j_var_day = apply_ctz_test_tsrv(
            rv_val=rv_day_5min, 
            tbv_val=tbv_val, 
            tsrv_val=tsrv_val, 
            log_returns=log_ret_5m, 
            abs_returns=abs_returns,
            alpha=alpha
        )
        tot_ret = np.sum(log_ret_5m)
        # Leverage
        neg_ret = np.sum(log_ret_5m[log_ret_5m < 0])
        daily_results.append({
            "Date": day,
            "C_var": c_var_day,
            "J_var": j_var_day,
            "neg_ret": neg_ret,
            "tot_ret": tot_ret,
            "TSRV": tsrv_val
        })

    return pd.DataFrame(daily_results).set_index("Date")
