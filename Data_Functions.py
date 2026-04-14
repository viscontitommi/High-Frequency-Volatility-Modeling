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

def load_order_doll(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Price ($)', 'Datetime']
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed')
    df.sort_values('Datetime', inplace=True)
    df.set_index('Datetime', inplace=True)
    return df

def load_order_eur(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['Price (€)', 'Datetime']
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed')
    df.sort_values('Datetime', inplace=True)
    df.set_index('Datetime', inplace=True)
    return df

def filter_brent_expiry(df, filename):
    name_clean = os.path.splitext(os.path.basename(filename))[0] # es. BRN_F16
    parts = name_clean.split('_')
    maturity_code = parts[1]      # "F16"
    month_char = maturity_code[0] # "F"
    year_str = maturity_code[1:]  # "16"
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12} 
    contract_month = month_map[month_char.upper()]
    contract_year = 2000 + int(year_str)
    # Let us apply the Rules    
    is_old_rule = (contract_year < 2016) or (contract_year == 2016 and contract_month < 3)
    if is_old_rule:
        contract_start = pd.Timestamp(year=contract_year, month=contract_month, day=1)
        day_15_prior = contract_start - pd.Timedelta(days=15)
        if day_15_prior.weekday() >= 5: 
            ref_day = day_15_prior - BDay(1)
        else:
            ref_day = day_15_prior
        expiry_date = ref_day - BDay(1)
        
    else:
        if contract_month <= 2:
            expiry_month = contract_month + 10 
            expiry_year = contract_year - 1
        else:
            expiry_month = contract_month - 2
            expiry_year = contract_year

        base_expiry_date = pd.Timestamp(year=expiry_year, month=expiry_month, day=1) + BMonthEnd()
        # Exceptions Manage
        christmas = pd.Timestamp(year=expiry_year, month=12, day=25)
        new_year = pd.Timestamp(year=expiry_year+1, month=1, day=1)
        bd_pre_xmas = christmas - BDay(1)
        bd_pre_ny = new_year - BDay(1)
        
        final_expiry_date = base_expiry_date
        if base_expiry_date == bd_pre_xmas or base_expiry_date == bd_pre_ny: # we anticipate
            final_expiry_date = base_expiry_date - BDay(1)
            
        expiry_date = final_expiry_date

    # Common Filter
    # Set the limit to the end of the day of the expiry
    expiry_limit = expiry_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_filtered = df[df.index <= expiry_limit]
    
    removed = len(df) - len(df_filtered)
    if removed > 0:
        print(f"We removed {removed} trade after the expiry.")

    return df_filtered

def filter_trading_hours(df, start_time, end_time):
    df_filtered = df.between_time(start_time, end_time)
    return df_filtered

def plot_hourly_distribution(df, title, valid_start, valid_end):
    df.index = pd.to_datetime(df.index)
    counts = df.index.hour.value_counts().sort_index()
    full_range = range(0, 24)
    counts = counts.reindex(full_range, fill_value=0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(counts.index, counts.values, edgecolor='black', alpha=0.8)
    plt.title(f"{title} (Total Trades: {len(df)})", fontsize=14)
    plt.xlabel("Day Hour (0-23)", fontsize=12)
    plt.ylabel("Number of Trades (Ticks)", fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

def apply_median_aggregation_doll(df, freq='1s'):
    col_name = 'Price ($)'
    # Resample + Median
    # Calcoliamo la mediana per ogni secondo
    df_final = df[col_name].resample(freq).median()
    df_final.dropna(inplace=True)
    df_final = df_final.to_frame()

    return df_final

def apply_median_aggregation_eur(df, freq='1s'):
    col_name = 'Price (€)'
    df_final = df[col_name].resample(freq).median()
    df_final.dropna(inplace=True)
    df_final = df_final.to_frame()

    return df_final

def filter_wti_expiry(df, filename):
    """
    Regola Scadenza WTI (Specifiche ICE):
    4th US business day prior to the 25th day of the month preceding the contract month.
    """
    name_clean = os.path.splitext(os.path.basename(filename))[0]
    parts = name_clean.split('_')
    maturity_code = parts[1] # "F16"
    month_char = maturity_code[0]
    year_str = maturity_code[1:] 
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}  
    contract_month = month_map[month_char.upper()]
    contract_year = 2000 + int(year_str)

    if contract_month == 1:
        prev_month = 12
        prev_year = contract_year - 1
    else:
        prev_month = contract_month - 1
        prev_year = contract_year
            
    day_25 = pd.Timestamp(year=prev_year, month=prev_month, day=25)
        
    if day_25.weekday() >= 5: # 5=Sabato, 6=Domenica
        ref_day = day_25 - BDay(1) # Back to Friday
    else:
        ref_day = day_25 # È già feriale

    expiry_date = ref_day - BDay(4)
    safe_expiry_date = expiry_date - BDay(1)
    expiry_limit = safe_expiry_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_filtered = df[df.index <= expiry_limit]
    
    return df_filtered

def filter_tfm_expiry(df, filename):
    """
    Regola Scadenza TFM (Dutch TTF Gas):
    - Ufficiale: 2 Business Days prima del 1° del mese di consegna.
    - NOSTRO FILTRO: Togliamo 1 giorno extra di sicurezza.
    - Totale: Ci fermiamo 3 Business Days prima del mese di consegna.
    """
    name_clean = os.path.splitext(os.path.basename(filename))[0]
    parts = name_clean.split('_')       
    maturity_code = parts[1] # "F16"
    month_char = maturity_code[0]
    year_str = maturity_code[1:]  
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 
                'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    contract_month = month_map[month_char.upper()]
    contract_year = 2000 + int(year_str)
        
    delivery_start = pd.Timestamp(year=contract_year, month=contract_month, day=1)
    safe_expiry_date = delivery_start - BDay(3)
    expiry_limit = safe_expiry_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_filtered = df[df.index <= expiry_limit]
             
    return df_filtered

def filter_ecf_expiry(df, filename):
    """
    Regola Scadenza ECF (EUA Carbon Emissions):
    Scadenza: L'Ultimo Lunedì (Last Monday) del mese di contratto.
    """
    name_clean = os.path.splitext(os.path.basename(filename))[0]
    parts = name_clean.split('_')
    maturity_code = parts[1] # "Z16"
    month_char = maturity_code[0]
    year_str = maturity_code[1:]
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12} 
    contract_month = month_map[month_char.upper()]
    contract_year = 2000 + int(year_str)
    end_of_month = pd.Timestamp(year=contract_year, month=contract_month, day=1) + BMonthEnd()
        
    start_dt = pd.Timestamp(year=contract_year, month=contract_month, day=1)
    end_dt = start_dt + BMonthEnd()
    all_days = pd.date_range(start=start_dt, end=end_dt, freq='D')
    mondays = all_days[all_days.dayofweek == 0]
    if contract_month == 12:
        if len(mondays) >= 3:
            official_expiry = mondays[2] # Third Monday
        else:
            official_expiry = mondays[-1] 
    else:
        official_expiry = mondays[-1] #Last Monday for other months

    safe_expiry_date = official_expiry - BDay(1)
    expiry_limit = safe_expiry_date + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df_filtered = df[df.index <= expiry_limit]
    
    return df_filtered

def calculate_multiscale_volatility(df_1s, freqs):
    vol_results = []
    df_work = df_1s.copy()
    df_work['date'] = df_work.index.date
    for freq in freqs:
        daily_vols = []
        
        # day by day
        for day, group in df_work.groupby('date'):
            resampled = group.iloc[:, 0].resample(freq).last().dropna()
            
            prices = resampled.values.astype(float)
            prices = prices[prices > 0]
            
            log_ret = np.diff(np.log(prices))
            
            threshold = 0.25
            is_outlier = np.abs(log_ret) > threshold
            log_ret[is_outlier] = 0.0
            daily_rv = np.sum(log_ret**2)
            
            ann_vol_day = np.sqrt(daily_rv * 252) * 100
            daily_vols.append(ann_vol_day)
            
        if len(daily_vols) > 0:
            vol_results.append(np.mean(daily_vols))
        else:
            vol_results.append(np.nan)
        
    return vol_results

