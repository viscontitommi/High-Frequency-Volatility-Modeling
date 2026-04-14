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
import RV_Functions as RV
import Data_Functions as DF

def process_ecf_dataset(input_folder):
    all_files = os.listdir(input_folder)
    ecf_files = [f for f in all_files if f.startswith('ECF') and f.endswith('.csv')]
    
    rv_results = []
    for filename in ecf_files:
        file_path = os.path.join(input_folder, filename)

        df = DF.load_order_eur(file_path)
        df = DF.filter_ecf_expiry(df, filename)
        df = DF.filter_trading_hours(df, start_time='08:00', end_time='18:00')
        df_final = DF.apply_median_aggregation_eur(df, freq='1s')
        
        rv_daily = RV.calculate_naive_rv(df_final)
        rv = rv_daily.to_frame(name='Realized Volatility')
        rv_results.append(rv)
        
    if len(rv_results) > 0:
        full_history = pd.concat(rv_results).sort_index()
        # Poiché abbiamo ordinato i file all'inizio (sorted), il 'first' sarà 
        # sempre il contratto più vicino alla scadenza (Front Month).
        final_dataset = full_history[~full_history.index.duplicated(keep='first')]
        return final_dataset
    
def run_ecf_signature_analysis(input_folder):
    freqs = ['1s', '5s', '15s', '30s', '1min', '2min', '5min', '10min', '20min']
    freq_seconds = [1, 5, 15, 30, 60, 120, 300, 600, 1200]
    
    vol_data = {f: [] for f in freqs}
    all_files = sorted([f for f in os.listdir(input_folder) if f.startswith('ECF') and f.endswith('.csv')])

    for i, filename in enumerate(all_files):
        file_path = os.path.join(input_folder, filename)
            
        df = DF.load_order_eur(file_path)
        df = DF.filter_ecf_expiry(df, filename)
        df = DF.filter_trading_hours(df, start_time='08:00', end_time='18:00')
        df_1s = DF.apply_median_aggregation_eur(df, freq='1s')
            
        if len(df_1s) < 100: continue
            
        vols = DF.calculate_multiscale_volatility(df_1s, freqs)
            
        for f, val in zip(freqs, vols):
            if not np.isnan(val): vol_data[f].append(val)
    
        return vol_data, freq_seconds
    
