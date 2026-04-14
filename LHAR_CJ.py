import pandas as pd
import os
import numpy as np
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay, BMonthEnd
from scipy.stats import norm
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import Brent_Functions as BF
import WTI_Functions as WTI_F
import TFM_Functions as TFM_F
import ECF_Functions as ECF_F
import Data_Functions as DF
import RV_Functions as RV
import Vol_Fun_Sep as VFS

def calculate_har_variables(df_daily):
    """
    Calcola le variabili HAR (Weekly/Monthly) secondo Corsi & Renò (2012).
    CONFORMITÀ:
    - Tutte le aggregazioni usano la MEDIA (rolling mean) come da Eq. 2 (fattore 1/h).
    """
    df = df_daily.copy()
    
    # --- 1. CONTINUOUS VOLATILITY (C) ---
    # Logaritmo della componente continua
    # Fix: gestiamo gli zeri prima del logaritmo
    c_safe = df['C_var'].replace(0, np.nan).fillna(1e-10)
    df['log_C_day'] = np.log(c_safe)
    
    # Aggregazione: MEDIA dei logaritmi (standard HAR)
    df['log_C_week'] = df['log_C_day'].rolling(window=5).mean()
    df['log_C_mon'] = df['log_C_day'].rolling(window=22).mean()

    # --- 2. JUMPS (J) ---
    # Regressore: log(1 + J_t^(h))
    # J_t^(h) è la MEDIA dei salti passati (intensità media)
    df['J_mean_week'] = df['J_var'].rolling(window=5).mean()
    df['J_mean_mon'] = df['J_var'].rolling(window=22).mean()
    
    # Trasformazione logaritmica (Eq. 8)
    df['log_1_J_day'] = np.log(1 + df['J_var'])
    df['log_1_J_week'] = np.log(1 + df['J_mean_week'])
    df['log_1_J_mon'] = np.log(1 + df['J_mean_mon'])

    # --- 3. LEVERAGE (Rendimenti Negativi) ---
    # Eq. 2: r_t^(h) è la MEDIA dei rendimenti
    df['ret_mean_week'] = df['tot_ret'].rolling(window=5).mean()
    df['ret_mean_mon'] = df['tot_ret'].rolling(window=22).mean()
    
    # Leverage Effect: min(r, 0)
    df['lev_day'] = np.minimum(df['tot_ret'], 0)
    df['lev_week'] = np.minimum(df['ret_mean_week'], 0)
    df['lev_mon'] = np.minimum(df['ret_mean_mon'], 0)

    # Rimuoviamo i NaN generati dalle rolling windows
    df_final = df.dropna()
    
    return df_final

def run_lhar_cj_model(df_input, h=1):
    """
    Stima il modello LHAR-CJ (Volatilità Totale).
    Target: Log(TSRV) futuro medio su orizzonte h.
    Features: C, J, Leverage passati.
    """
    df = df_input.copy()
    # Definizione Target (V)
    # Usiamo TSRV se c'è, altrimenti C+J
    if 'TSRV' in df.columns:
        target_series = df['TSRV']
    else:
        target_series = df['C_var'] + df['J_var']
    # Pulizia Zeri prima del log
    target_series = target_series.replace(0, np.nan)
    # Costruzione Target Futuro (Rolling Mean per h > 1)
    if h == 1:
        # Log del valore di domani
        df['TARGET'] = np.log(target_series).shift(-1)
    else:
        # Log della MEDIA dei prossimi h giorni
        # .rolling(h) mette il valore alla fine della finestra. 
        # .shift(-h) lo riporta indietro all'inizio (t).
        df['TARGET'] = np.log(target_series.rolling(window=h).mean().shift(-h))

    # Pulizia (Rimuove NaN e Inf)
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna()

    formula = """
        TARGET ~ log_C_day + log_C_week + log_C_mon + \
                 log_1_J_day + log_1_J_week + log_1_J_mon + \
                 lev_day + lev_week + lev_mon
    """
    
    nw_lags = int(2 + (2*h)) 
    model = smf.ols(formula=formula, data=df_clean)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    
    return result

def run_lhar_c_cj_model(df_input, h=1):
    """
    Stima il modello LHAR-C-CJ (Eq. 5 del Paper).
    Target: Log(C) futuro medio su orizzonte h.
    """
    df = df_input.copy()
    # Target Base (C_var)
    target_series = df['C_var'].replace(0, np.nan)
    # Costruzione Target Futuro (Rolling Mean)
    if h == 1:
        df['TARGET'] = np.log(target_series).shift(-1)
    else:
        # Previsione della volatilità continua MEDIA futura
        df['TARGET'] = np.log(target_series.rolling(window=h).mean().shift(-h))
    # Pulizia
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna()
    
    formula = """
        TARGET ~ log_C_day + log_C_week + log_C_mon + \
                 log_1_J_day + log_1_J_week + log_1_J_mon + \
                 lev_day + lev_week + lev_mon
    """
    nw_lags = int(2 + (2 * h))
    model = smf.ols(formula=formula, data=df_clean)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    
    return result

def run_lhar_j_cj_model(df_input, h=1):
    """
    Stima il modello LHAR-J-CJ (Eq. 8 del Paper).
    Target: Log(1 + J_mean) futuro su orizzonte h.
    """
    df = df_input.copy()
    # Target Base (J_var)
    # Nota: J può essere 0, quindi usiamo log(1+J) come da paper
    target_series = df['J_var']
    # Costruzione Target Futuro
    if h == 1:
        # Domani
        df['TARGET'] = np.log(1 + target_series).shift(-1)
    else:
        # Per h > 1, il paper usa l'intensità media dei salti
        # Log(1 + Media(J_t...t+h))
        j_mean_future = target_series.rolling(window=h).mean().shift(-h)
        df['TARGET'] = np.log(1 + j_mean_future)
    # Pulizia
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna()
    
    formula = """
        TARGET ~ log_C_day + log_C_week + log_C_mon + \
                 log_1_J_day + log_1_J_week + log_1_J_mon + \
                 lev_day + lev_week + lev_mon
    """
    nw_lags = int(2 + (2 * h))
    model = smf.ols(formula=formula, data=df_clean)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    
    return result

def analyze_risk_return_tradeoff(realized_returns, log_volatility_forecast):
    """
    Esegue la regressione Risk-Return Trade-off per il modello LHAR-CJ
    Parametri:
    - realized_returns: pd.Series dei rendimenti giornalieri veri (r_t)
    - log_volatility_forecast: pd.Series delle previsioni del LOGaritmo della volatilità 
                               (output del modello LHAR-CJ)
    """
    data = pd.DataFrame({
        'r_t': realized_returns,
        'log_pred': log_volatility_forecast
    }).dropna()
    # "exponential of the logarithmic forecasts"
    data['V_tilde'] = np.exp(data['log_pred'])
    Y = data['r_t']
    X = data['V_tilde']
    X = sm.add_constant(X) 
    model = sm.OLS(Y, X).fit()
    # Equazione: r_t = c + beta * V_tilde + error
    
    return model

def run_lhar_cj_plus_model(df, h=1):
    """
    Esegue la regressione LHAR-CJ+.
    TARGET: Log(TSRV).
    FEATURES: Tutte le variabili HAR calcolate.
    SICUREZZA: Gestisce log(0) -> -inf sostituendoli con NaN.
    """
    data = df.copy()
    # Definizione Segno Salti (Dummy variable logic)
    if 'tot_ret' in data.columns:
        ret_col = data['tot_ret']
    else:
        ret_col = data['neg_ret']
    # Creiamo le componenti asimmetriche (Positive/Negative)
    raw_J_pos = np.where(ret_col > 0, data['J_var'], 0)
    raw_J_neg = np.where(ret_col <= 0, data['J_var'], 0)
    data['log_J_pos_day'] = np.log(1 + raw_J_pos)
    data['log_J_neg_day'] = np.log(1 + raw_J_neg)
    
    features = [
        'log_C_day', 'log_C_week', 'log_C_mon',
        'log_J_pos_day', 'log_J_neg_day', 
        'log_1_J_week', 'log_1_J_mon',
        'lev_day', 'lev_week', 'lev_mon'
    ]
    # SHIFT TEMPORALE (Fondamentale: X=Ieri, Y=Oggi)
    for col in features:
        if col in data.columns:
            data[col] = data[col].shift(1)
        
    # DEFINIZIONE TARGET (Y)
    # Usiamo TSRV come proxy della Volatilità Totale (C+J)
    # Se TSRV manca, usiamo C_var + J_var
    if 'TSRV' in data.columns:
        target_series = data['TSRV']
    else:
        target_series = data['C_var'] + data['J_var']
    # Rimuovi gli zeri PRIMA del logaritmo
    target_series = target_series.replace(0, np.nan)
    if h == 1:
        data['target_h'] = np.log(target_series)
    else:
        # Per h>1, target è la media della volatilità futura
        rv_rolling = target_series.rolling(window=h).mean()
        data['target_h'] = np.log(rv_rolling.shift(-(h-1)))
    # PULIZIA FINALE (Rimuove righe con NaN o Inf)
    # Pandas dropna() non rimuove inf, lo facciamo esplicitamente
    data = data.replace([np.inf, -np.inf], np.nan)
    data_clean = data.dropna()
    # Controllo che ci siano dati sufficienti
    if len(data_clean) < 100:
        print(f"ATTENZIONE: Solo {len(data_clean)} osservazioni rimaste dopo la pulizia.")
        if len(data_clean) == 0:
            return None
    # STIMA (Newey-West HAC)
    formula = "target_h ~ " + " + ".join(features)
    nw_lags = 2 + 2 * h
    try:
        model = smf.ols(formula, data=data_clean).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        return model
    except Exception as e:
        print(f"Errore Stima: {e}")
        return None

def run_oos_lhar_cj_expanding(df_har_input, start_fraction=0.75):
    """
    Esegue OOS Expanding Window prendendo in input il DF con le variabili HAR già calcolate.
    (es. df_lhar_brn o df_lhar_wbs).
    Gestisce AUTOMATICAMENTE lo shift: Feature(t-1) -> Target(t).
    """
    data = df_har_input.copy()
    # Target: TSRV (Log)
    if 'TSRV' in data.columns:
        tgt = data['TSRV']
    else:
        # se non c'è TSRV (usa C+J)
        tgt = data['C_var'] + data['J_var']
        
    data['log_target'] = np.log(tgt.replace(0, np.nan))
    # SHIFT DELLE FEATURES (Il passaggio chiave)
    # Prendiamo tutte le colonne che sono features (log_... lev_...)
    features_cols = [
        'log_C_day', 'log_C_week', 'log_C_mon',
        'log_1_J_day', 'log_1_J_week', 'log_1_J_mon',
        'lev_day', 'lev_week', 'lev_mon'
    ]
    # Spostiamo le features indietro di 1 giorno
    # Così alla riga 't' (Oggi) avremo le features di 't-1' (Ieri)
    for col in features_cols:
        if col in data.columns:
            data[col] = data[col].shift(1)      
    # Ora rimuoviamo la prima riga che ha NaN per via dello shift
    data = data.dropna()
    # Definizione Formula (Modello LHAR-CJ Base)
    formula = (
        "log_target ~ "
        "log_C_day + log_C_week + log_C_mon + "
        "log_1_J_day + log_1_J_week + log_1_J_mon + "
        "lev_day + lev_week + lev_mon"
    )
    # Loop Expanding Window
    total_obs = len(data)
    start_index = int(total_obs * start_fraction)
    
    predictions = []
    real_values = []
    for t in range(start_index, total_obs):
        # Train: Tutto il passato disponibile
        train_data = data.iloc[:t]
        # Test: Oggi (contiene X ieri e Y oggi)
        test_row = data.iloc[t:t+1]
        try:
            model = smf.ols(formula, data=train_data).fit()
            pred = model.predict(test_row).values[0]
            
            predictions.append(pred)
            real_values.append(test_row['log_target'].values[0])
        except:
            predictions.append(np.nan)
            real_values.append(np.nan)   
    # Risultati
    results = pd.DataFrame({
        'Realized': real_values,
        'Forecast': predictions
    }, index=data.index[start_index:])
    
    results = results.dropna()
    # Mincer-Zarnowitz Regression
    mz_model = sm.OLS(results['Realized'], sm.add_constant(results['Forecast'])).fit()
    
    rmse = np.sqrt(((results['Forecast'] - results['Realized'])**2).mean())
    print(f"RMSE: {rmse:.4f}")
    
    return results, mz_model

def _prepare_oos_data(df_har_input, model_type, h):
    """
    Funzione di supporto.
    model_type: 'HAR' | 'HAR_CJ' | 'LHAR_CJ'
    h: orizzonte di previsione (1, 5, 10, 22)
    """
    data = df_har_input.copy()
    # TSRV come proxy della volatilità totale 
    if 'TSRV' in data.columns:
        tgt = data['TSRV'].replace(0, np.nan)
    else:
        tgt = (data['C_var'] + data['J_var']).replace(0, np.nan)

    if h == 1:
        data['log_target'] = np.log(tgt).shift(-1) # Log del valore del giorno successivo
    else:
        data['log_target'] = np.log(tgt.rolling(window=h).mean()).shift(-h) # Log della media mobile futura su h giorni (Eq. 2.4 paper)

    # FEATURES
    # Le features sono sempre al tempo t (facciamo previsione t+h partendo da t). NON shiftiamo qui: lo shift temporale è già implicito nel target futuro
    volatility_features = ['log_C_day', 'log_C_week', 'log_C_mon']
    jump_features       = ['log_1_J_day', 'log_1_J_week', 'log_1_J_mon']
    leverage_features   = ['lev_day', 'lev_week', 'lev_mon']

    # SELEZIONE FEATURES PER MODELLO
    if model_type == 'HAR':
        # HAR puro: solo volatilità eterogena, target = log(TSRV) e dunque sostituiamo log_C con log_TSRV
        # Usiamo log_C_day/week/mon come proxy (già calcolati su C_var)
        active_features = volatility_features
        formula = "log_target ~ log_C_day + log_C_week + log_C_mon"
    elif model_type == 'HAR_CJ':
        # HAR-CJ: volatilità + salti
        active_features = volatility_features + jump_features
        formula = ("log_target ~ ""log_C_day + log_C_week + log_C_mon + ""log_1_J_day + log_1_J_week + log_1_J_mon")

    elif model_type == 'LHAR_CJ':
        # LHAR-CJ completo: volatilità + salti + leverage eterogeneo
        active_features = volatility_features + jump_features + leverage_features
        formula = ("log_target ~ ""log_C_day + log_C_week + log_C_mon + ""log_1_J_day + log_1_J_week + log_1_J_mon + ""lev_day + lev_week + lev_mon")
    else:
        raise ValueError(f"model_type '{model_type}' non riconosciuto. Usa: HAR, HAR_CJ, LHAR_CJ")

    # PULIZIA
    data = data.replace([np.inf, -np.inf], np.nan)
    data_clean = data[['log_target'] + active_features].dropna()

    return data_clean, formula

def run_oos_multihorizon(df_har_input, horizons=None, start_fraction=0.75):
    """
    Esegue Out-Of-Sample con diverse Window per i tre modelli (HAR, HAR-CJ, LHAR-CJ) su tutti gli orizzonti h specificati.
    Parametri:
    df_har_input    : DataFrame con le variabili HAR già calcolate (output di calculate_har_variables)
    horizons        : lista di orizzonti in giorni, default [1, 5, 10, 22]
    start_fraction  : frazione dei dati usata per il training iniziale (default 0.75)
    """
    if horizons is None:
        horizons = [1, 5, 10, 22]
    model_types = ['HAR', 'HAR_CJ', 'LHAR_CJ']
    results = {m: {} for m in model_types}

    for i in model_types:
        for h in horizons:
            data_clean, formula = _prepare_oos_data(df_har_input, i, h) # Preparazione dati
            total_obs   = len(data_clean)
            start_index = int(total_obs * start_fraction)
            predictions  = []
            real_values  = []
            valid_index  = []

            # Loop espandente: train su [0:t], predict su t
            for t in range(start_index, total_obs):
                train_data = data_clean.iloc[:t]
                test_row = data_clean.iloc[t:t+1]
                try:
                    model_fit = smf.ols(formula, data=train_data).fit()
                    pred = model_fit.predict(test_row).values[0]
                    predictions.append(pred)
                    real_values.append(test_row['log_target'].values[0])
                    valid_index.append(data_clean.index[t])
                except Exception:
                    # In caso di errore numerico saltiamo l'osservazione
                    pass

            # DataFrame risultati
            oos_df = pd.DataFrame({'Realized': real_values, 'Forecast': predictions},index=valid_index).dropna()

            # Regressione Mincer-Zarnowitz: Realized = a + b * Forecast
            mz = sm.OLS(oos_df['Realized'], sm.add_constant(oos_df['Forecast'])).fit()
            mz_r2 = mz.rsquared
            rmse = np.sqrt(((oos_df['Forecast'] - oos_df['Realized'])**2).mean())

            results[i][h] = {'oos_df': oos_df, 'mz_r2': mz_r2, 'rmse': rmse, 'mz_model': mz}

    return results

def plot_mz_r2_figure4(oos_results_dict, asset_name, horizons=None, figsize=(10, 5)):
    """
    Produce il grafico MZ-R² per orizzonte temporale considerato
    Confronta HAR, HAR-CJ e LHAR-CJ sullo stesso pannello.
    """
    if horizons is None:
        horizons = [1, 5, 10, 22]

    # Etichette asse x
    horizon_labels = {1: '1d', 5: '1w', 10: '2w', 22: '1m'}
    x_labels = [horizon_labels.get(h, str(h)) for h in horizons]
    # Colori e stili
    styles = {'HAR'     : {'color': '#1f77b4', 'linestyle': '--',  'marker': 's', 'label': 'HAR'},
        'HAR_CJ'  : {'color': '#ff7f0e', 'linestyle': '-.',  'marker': '^', 'label': 'HAR-CJ'},
        'LHAR_CJ' : {'color': '#2ca02c', 'linestyle': '-',   'marker': 'o', 'label': 'LHAR-CJ'},
    }

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    for model_type, style in styles.items():
        r2_values = []
        for h in horizons:
            entry = oos_results_dict.get(model_type, {}).get(h)
            if entry is not None and 'mz_r2' in entry:
                r2_values.append(entry['mz_r2'])
            else:
                r2_values.append(np.nan)

        ax.plot(range(len(horizons)), r2_values, color = style['color'], linestyle = style['linestyle'], marker = style['marker'], linewidth = 2, markersize= 7, label = style['label'])

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel('Forecasting Horizon', fontsize=12)
    ax.set_ylabel('Mincer-Zarnowitz $R^2$', fontsize=12)
    ax.set_title(
        f'Out-of-Sample MZ-$R^2$ by Horizon — {asset_name}\n'
        f'(Expanding Window, start fraction = 75%)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_rmse_figure4(oos_results_dict, asset_name, horizons=None, figsize=(10, 5)):
    """
    Produce il grafico RMSE per orizzonte (pannello inferiore analogo a Figura 4).
    Confronta HAR, HAR-CJ e LHAR-CJ.
    """
    if horizons is None:
        horizons = [1, 5, 10, 22]

    horizon_labels = {1: '1d', 5: '1w', 10: '2w', 22: '1m'}
    x_labels = [horizon_labels.get(h, str(h)) for h in horizons]
    styles = {'HAR'     : {'color': '#1f77b4', 'linestyle': '--',  'marker': 's', 'label': 'HAR'},
        'HAR_CJ'  : {'color': '#ff7f0e', 'linestyle': '-.',  'marker': '^', 'label': 'HAR-CJ'},
        'LHAR_CJ' : {'color': '#2ca02c', 'linestyle': '-',   'marker': 'o', 'label': 'LHAR-CJ'},
    }

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    for model_type, style in styles.items():
        rmse_values = []
        for h in horizons:
            entry = oos_results_dict.get(model_type, {}).get(h)
            if entry is not None and 'rmse' in entry:
                rmse_values.append(entry['rmse'])
            else:
                rmse_values.append(np.nan)

        ax.plot(range(len(horizons)), rmse_values, color = style['color'], linestyle = style['linestyle'], marker = style['marker'], linewidth = 2, markersize= 7,label = style['label'])

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel('Forecasting Horizon', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(
        f'Out-of-Sample RMSE by Horizon — {asset_name}\n'
        f'(Expanding Window, start fraction = 75%)',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()