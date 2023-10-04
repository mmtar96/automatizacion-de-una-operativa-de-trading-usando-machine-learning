# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:54:01 2023

@author: mmtar96
"""

' TFM DEFINITIVO - V1 '


' IMPORTADO DE LIBRERIAS '

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import time
import seaborn as sns

# LIBRERIAS DESARROLLADAS
%cd code/libs
import ipdata as ip
import indtechv2 as it
%cd ..
%cd ..


' CARGA DE DATOS '


# Tratamiento y preprocesado de datos
def preprocesado(data_bid, data_ask):
    data_bid, data_ask = ip.correct_data(data_bid, data_ask)
    data_bidF = ip.import_data(data_bid)
    data_askF = ip.import_data(data_ask)
    dataset = ip.merge(data_bidF, data_askF) 
    ip.ploteo(dataset)
    return dataset



' CALCULO DE INDICADORES: '
  

def ind_tec(dataset, target_price, p1=14, p2=26, p3=50, p_rsi=14, p_macd1=14, p_macd2=26, amp=25):
    mPrice = dataset['Open']
    # SMA    
    sma14 = it.sma(mPrice, periodo=p1)
    sma29 = it.sma(mPrice, periodo=p2)
    sma50 = it.sma(mPrice, periodo=p3)
    
    fig = plt.figure()
    plt.plot(dataset.iloc[:,0], mPrice, color='green', linewidth=1)
    plt.plot(dataset.iloc[:,0], sma14, color='blue', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], sma29, color='red', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], sma50, color='black', linewidth=1, linestyle = "--")
    plt.ylim(min(mPrice)-amp, max(mPrice)+amp)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('SMA')
    plt.show()
    
    # SMA Ponderada 
    sma14_p = it.sma_p(mPrice, periodo=p1)
    sma29_p = it.sma_p(mPrice, periodo=p2)
    sma50_p = it.sma_p(mPrice, periodo=p3)
    
    fig = plt.figure()
    plt.plot(dataset.iloc[:,0], mPrice, color='green', linewidth=1)
    plt.plot(dataset.iloc[:,0], sma14_p, color='blue', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], sma29_p, color='red', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], sma50_p, color='black', linewidth=1, linestyle = "--")
    plt.ylim(min(mPrice)-amp, max(mPrice)+amp)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('SMA_P')
    plt.show()
    
    
    # EMA 
    ema14 = it.ema(mPrice, periodo=p1)
    ema29 = it.ema(mPrice, periodo=p2)
    ema50 = it.ema(mPrice, periodo=p3)
    
    fig = plt.figure()
    plt.plot(dataset.iloc[:,0], mPrice, color='green', linewidth=1)
    plt.plot(dataset.iloc[:,0], ema14, color='blue', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], ema29, color='red', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], ema50, color='black', linewidth=1, linestyle = "--")
    plt.ylim(min(mPrice)-amp, max(mPrice)+amp)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('EMA')
    plt.show()
    
    # Comparacion entre SMA - SMA_P - EMA
    fig = plt.figure()
    plt.plot(dataset.iloc[:,0], mPrice, color='green', linewidth=1)
    plt.plot(dataset.iloc[:,0], sma50, color='blue', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], sma50_p, color='red', linewidth=1, linestyle = "--")
    plt.plot(dataset.iloc[:,0], ema50, color='black', linewidth=1, linestyle = "--")
    plt.ylim(min(mPrice)-amp, max(mPrice)+amp)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('SMA50 - SMA_P50 - EMA50')
    plt.show()
    
    
    # RSI
    periodo = p_rsi
    rsi = it.rsi(mPrice, periodo)
    
    fig = plt.figure()
    plt.plot(dataset.iloc[:,0], rsi, color='green', linewidth=1)
    plt.ylim(0, 100)
    plt.xlabel('Time')
    plt.ylabel('RSI')
    plt.title('RSI')
    plt.show()
    
    
    # MACD 
    pme1 = p_macd1
    pme2 = p_macd2
    macd = it.macd(mPrice, pme1, pme2)
    
    fig = plt.figure()
    plt.plot(dataset.iloc[:,0], macd, color='green', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('MACD')
    plt.title('MACD')
    plt.show()

    ' DATASET DEFINITIVO '
    
    vol = dataset['Volume']
        
    data = pd.DataFrame({'Time' : dataset['Date'],
                         'Open Price' : dataset['Open'],
                         'Volume' : vol,
                         'SMA14' : sma14.iloc[:,0],
                         'SMA29' : sma29.iloc[:,0],
                         'SMA50' : sma50.iloc[:,0],
                         'SMA14_p' : sma14_p.iloc[:,0],
                         'SMA29_p' : sma29_p.iloc[:,0],
                         'SMA50_p' : sma50_p.iloc[:,0],
                         'EMA14' : ema14.iloc[:,0],
                         'EMA29' : ema29.iloc[:,0],
                         'EMA50' : ema50.iloc[:,0],
                         'RSI' : rsi.iloc[:,0],
                         'MACD' : macd.iloc[:,0].values,     
                         'Target_Price' : target_price})
    
    # Matriz de correlacion #correlacion de pearson
    correlation_matrix = data.corr(method='pearson')
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación')
    plt.show()
    
    return data, correlation_matrix
 


 
' PROCESADO Y TRATAMIENTO DE DATOS DEFINITIVO '

# Eliminar los datos nulos
def nulos(data):
    pos = []
    for i in range(len(data)):
            if  np.prod(data.iloc[i,1:]) == 0:
                pos.append(i)
    data.drop(pos, axis=0, inplace=True)  
    return data


def train_test(data, date='03.01.2022'):
    pos = list(data['Time']).index(date)
    # Division datos entrenamiento y test
    data_X_train = data.iloc[:pos,:-1]
    data_X_test = data.iloc[pos:,:-1]
    data_y_train = data.iloc[:pos,-1]
    data_y_test = data.iloc[pos:,-1]
    
    # Ploteo de datos
    fig = plt.figure()
    plt.plot(data_X_train.iloc[:,0], data_X_train.iloc[:,1], color='green', linewidth=1)
    plt.ylim(min(data_X_train.iloc[:,1])*0.95, max(data_X_train.iloc[:,1])*1.05)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Datos Entrenamiento')
    plt.show()
    
    fig = plt.figure()
    plt.plot(data_X_test.iloc[:,0], data_X_test.iloc[:,1], color='green', linewidth=1)
    plt.ylim(min(data_X_test.iloc[:,1])*0.95, max(data_X_test.iloc[:,1])*1.05)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Datos Test')
    plt.show()
    
    # Entrada de datos al modelo (Eliminamos la fecha)
    X_train = data_X_train.to_numpy()[:,1:]
    X_test = data_X_test.to_numpy()[:,1:]
    y_train = data_y_train.to_numpy()
    y_test = data_y_test.to_numpy()
    
    # Datos para excel anexo1
    data_init = pd.DataFrame({'X_test' : X_test[:,0],
                          'y_test' : y_test})
    
    X = data_X_train.append(data_X_test).to_numpy()[:,1:]
    y = data_y_train.append(data_y_test).to_numpy()
    
    # Escalado
    sc_X = StandardScaler()
    X_sc = sc_X.fit_transform(X)
    
    sc_y = StandardScaler()
    y_sc = sc_y.fit_transform(y.reshape(-1,1))
    
    X_train_sc = X_sc[:len(data_X_train),:]
    X_test_sc = X_sc[len(data_X_train):,:]
    y_train_sc = y_sc[:len(data_X_train)]
    y_test_sc = y_sc[len(data_X_train):]

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, y_train_sc, y_test_sc, sc_y, data_init


' ENTRENAMIENTO DE MODELOS'

def metrics(y_test, y_pred, k, title): # Funcion para calcular las metricas de los modelos generados
    #Error cuadratico medio
    from sklearn.metrics import mean_squared_error
    rms = mean_squared_error(y_test, y_pred)

    #Error absoluto medio
    mae = np.abs((y_test - y_pred)).mean()

    #Error absoluto medio porcentual
    mape = np.abs((y_test - y_pred)/y_test).mean()

    #Coeficiente de determinacion R2
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)

    #Coeficiente de determinacion R2 ajustado
    n = y_test.shape[0]
    num = n - 1
    den = n - k - 1
    r2_adj = 1 - (n - 1)/(n - k -1)*(1 - r2)
    print("########################################")
    print(f"{title}")
    print("RMS = ", rms)
    print("MAE = ", mae)
    print("MAPE = ", mape)
    print("R2 = ", r2)
    print("R2' = ", r2_adj)
    print("########################################")
    return(rms, mae, mape, r2, r2_adj)

def vResults(y_pred, y_test, amp, title):
    ax = np.linspace(0, len(y_pred), len(y_pred))
    fig = plt.figure()
    plt.plot(ax, y_test, color='green', linewidth=1, label='real')
    plt.plot(ax, y_pred, color='blue', linewidth=1, linestyle = "--", label='predict')
    plt.ylim(min(y_test)-amp, max(y_test)+amp)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, y_train_sc, y_test_sc, sc_y, amp=25):
    
    'Regresion Lineal'
    
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)
    #Prediccion
    y_pred_linear = model_linear.predict(X_test)  
    results_model_linear = metrics(y_test, y_pred_linear, len(X_train[1,:]), 'Resultados modelo lineal')
    
    # Ploteo de los datos de testeo
    vResults(y_pred_linear, y_test, amp, 'Regresion lineal')
    
    # Con datos escalados
    model_linear_sc = LinearRegression()
    model_linear_sc.fit(X_train_sc, y_train_sc)
    #Prediccion
    y_pred_linear_sc = model_linear_sc.predict(X_test_sc)  
    y_pred_linear_t = sc_y.inverse_transform(y_pred_linear_sc)
    results_model_linear_sc = metrics(y_test, y_pred_linear_t, len(X_train[1,:]), 'Resultados modelo lineal escalado')
    
    # Ploteo de los datos de testeo
    vResults(y_pred_linear_t, y_test, amp, 'Regresion lineal escalado')
    
    
    
    'Regresion polinomica'
    poly_reg = PolynomialFeatures(degree = 3)
    X_train_poly = poly_reg.fit_transform(X_train)
    X_test_poly = poly_reg.fit_transform(X_test)
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    #Prediccion
    y_pred_poly = model_poly.predict(X_test_poly)  
    results_model_poly = metrics(y_test, y_pred_poly, len(X_train[1,:]), "Resultados modelo regresion poly (3)")
    
    # Ploteo de los datos de testeo
    vResults(y_pred_poly, y_test, amp, 'Regresion Polinomica (3)')
    
    
    # Con datos escalados
    model_poly_sc = SVR(kernel='poly', degree=3)
    model_poly_sc.fit(X_train_sc, y_train_sc)
    #Prediccion
    y_pred_poly_sc = model_poly_sc.predict(X_test_sc)  
    y_pred_poly_t = sc_y.inverse_transform(y_pred_poly_sc)
    results_model_poly_sc = metrics(y_test, y_pred_poly_t, len(X_train[1,:]), 'Resultados modelo regresion poly (3) escalado')
    
    # Ploteo de los datos de testeo
    vResults(y_pred_poly_t, y_test, amp, 'Regresion Polinomica (3) escalado')
    
    
    
    ' Maquina de soporte vectorial - kernel: rbf'
    model_svr_rbf = SVR(kernel='rbf')
    model_svr_rbf.fit(X_train, y_train)
    #Prediccion
    y_pred_svr_rbf = model_svr_rbf.predict(X_test)  
    results_model_svr_rbf = metrics(y_test, y_pred_svr_rbf, len(X_train[1,:]), "Resultados modelo SVR - rbf")
    
    # Ploteo de los datos de testeo
    vResults(y_pred_svr_rbf, y_test, amp, 'SVM - kernel: rbf')
    
    
    # Con datos escalados
    model_svr_rbf_sc = SVR(kernel='rbf')
    model_svr_rbf_sc.fit(X_train_sc, y_train_sc)
    #Prediccion
    y_pred_svr_rbf_sc = model_svr_rbf_sc.predict(X_test_sc)  
    y_pred_svr_rbf_t = sc_y.inverse_transform(y_pred_svr_rbf_sc)
    results_model_svr_rbf_sc = metrics(y_test, y_pred_svr_rbf_t, len(X_train[1,:]), 'Resultados modelo regresion rbf (3) escalado')
    
    # Ploteo de los datos de 
    vResults(y_pred_svr_rbf_t, y_test, amp, 'SVM - kernel: rbf escalado')
    
    
    
    ' Arboles de decision'
    model_arbol_decision = DecisionTreeRegressor()
    model_arbol_decision.fit(X_train, y_train)
    #Prediccion
    y_pred_arbol_decision = model_arbol_decision.predict(X_test)  
    results_model_arbol_decision = metrics(y_test, y_pred_arbol_decision, len(X_train[1,:]), 'Resultados modelo arbol decision')
    
    # Ploteo de los datos de testeo
    vResults(y_pred_arbol_decision, y_test, amp, 'Arbol de decision')
    
    
    # Con datos escalados
    model_arbol_decision_sc = DecisionTreeRegressor()
    model_arbol_decision_sc.fit(X_train_sc, y_train_sc)
    #Prediccion
    y_pred_arbol_decision_sc = model_arbol_decision_sc.predict(X_test_sc)  
    y_pred_arbol_decision_t = sc_y.inverse_transform(y_pred_arbol_decision_sc)
    results_model_arbol_decision_sc = metrics(y_test, y_pred_arbol_decision_t, len(X_train[1,:]), 'Resultados modelo arbol decision escalado')
    
    # Ploteo de los datos de testeo
    vResults(y_pred_arbol_decision_t, y_test, amp, 'Arbol de decision escalado')
    
    
    
    ' Bosques aleatorios - criterion : friedman_mse'
    model_bosques_aleatorios_fmse = RandomForestRegressor(n_estimators=100, criterion="friedman_mse", random_state=17)
    model_bosques_aleatorios_fmse.fit(X_train, y_train)
    #Prediccion
    y_pred_bosques_aleatorios = model_bosques_aleatorios_fmse.predict(X_test)  
    results_model_bosques_aleatorios_fmse = metrics(y_test, y_pred_bosques_aleatorios, len(X_train[1,:]), 'Resultados modelo bosqueas aleatorios f_mse')
    
    # Ploteo de los datos de testeo
    vResults(y_pred_bosques_aleatorios, y_test, amp, 'Busques aleatorios f_mse')
    
    
    # Con datos escalados
    model_bosques_aleatorios_fmse_sc = DecisionTreeRegressor()
    model_bosques_aleatorios_fmse_sc.fit(X_train_sc, y_train_sc)
    #Prediccion
    y_pred_bosques_aleatorios_sc = model_bosques_aleatorios_fmse_sc.predict(X_test_sc)  
    y_pred_bosques_aleatorios_t = sc_y.inverse_transform(y_pred_bosques_aleatorios_sc)
    results_model_bosques_aleatorios_fmse_sc = metrics(y_test, y_pred_bosques_aleatorios_t, len(X_train[1,:]), 'Resultados modelo bosqueas aleatorios f_mse escalado')
    
    # Ploteo de los datos de testeo
    vResults(y_pred_bosques_aleatorios_t, y_test, amp, 'Busques aleatorios f_mse escalado')
    
    
    
    
    ' EXPORTAR RESULTADOS '
    
    medidas = ['RMS', 'MAE', 'MAPE', 'R2', 'R2_adj']
    
    resultados = pd.DataFrame({'Medidas' : medidas,
                               'Modelo Regresion Lineal' : results_model_linear,
                              'Modelo Regresion Polinomica' : results_model_poly,
                              'Modelo SVR - kernel : rbf' : results_model_svr_rbf,
                              'Modelo Arbol de decision' : results_model_arbol_decision,
                              'Modelo Bosques aleatorios - crit: f_mse' : results_model_bosques_aleatorios_fmse,
                              'Modelo Regresion Lineal sc' : results_model_linear_sc,
                             'Modelo Regresion Polinomica sc' : results_model_poly_sc,
                             'Modelo SVR - kernel : rbf sc' : results_model_svr_rbf_sc,
                             'Modelo Arbol de decision sc' : results_model_arbol_decision_sc,
                             'Modelo Bosques aleatorios - crit: f_mse sc' : results_model_bosques_aleatorios_fmse_sc,
                             })
    print(resultados)
    
    predicts = pd.DataFrame({'Modelo Regresion Lineal' : y_pred_linear,
                              'Modelo Regresion Polinomica' : y_pred_poly,
                              'Modelo SVM - kernel : rbf' : y_pred_svr_rbf,
                              'Modelo Arbol de decision' : y_pred_arbol_decision,
                              'Modelo Bosques aleatorios - crit: f_mse' : y_pred_bosques_aleatorios,
                              'Modelo Regresion Lineal sc' : y_pred_linear,
                                'Modelo Regresion Polinomica sc' : y_pred_poly_t,
                                'Modelo SVM - kernel : rbf sc' : y_pred_svr_rbf_t,
                                'Modelo Arbol de decision sc' : y_pred_arbol_decision_t,
                                'Modelo Bosques aleatorios - crit: f_mse sc' : y_pred_bosques_aleatorios_t})
    print(predicts)
    
    
    
    return resultados, predicts, [model_linear, model_linear_sc, model_poly, model_poly_sc, model_svr_rbf, model_svr_rbf_sc, model_arbol_decision, model_arbol_decision_sc, model_bosques_aleatorios_fmse, model_bosques_aleatorios_fmse_sc]


def analisis(data_bid, data_ask, date='03.01.2021'):

    # Datos brutos
    dataset = preprocesado(data_bid, data_ask)
    spread_medio = np.mean(dataset['Spread']+dataset['Spread'])
    
    # Datos finales (Entradas del modelo)
    data, correlation_matrix = ind_tec(dataset, dataset['Close'], p1=14, p2=26, p3=50, p_rsi=14, p_macd1=14, p_macd2=26)
    
     # Eliminar los datos nulos o 0
    data = nulos(data)
    
    # Datos de entrenamiento y escalados
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, y_train_sc, y_test_sc, sc_y, data_init = train_test(data, date)
    
    # Evaluacion de modelos
    resultados, predicts, modelos = models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, y_train_sc, y_test_sc, sc_y, amp=25)

    return resultados, predicts, modelos, data_init, spread_medio, correlation_matrix



' BACK '

# Datos 17 - 23
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_26.01.2017-31.08.2023.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_26.01.2017-31.08.2023.csv')
dataset = preprocesado(data_bid, data_ask)

# Carga de datos 2021-22 / 23
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2021-31.08.2023.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2021-31.08.2023.csv')

resultados2123, predicts2123, modelos2123, data_init2123, spread2123, correlation_matrix2123 = analisis(data_bid, data_ask, date='03.01.2023')

resultados2123.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados2123.xlsx')
predicts2123.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts2123.xlsx')
data_init2123.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init2123.xlsx')

# Carga de datos 2020-22 / 2023
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2020-31.08.2023.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2020-31.08.2023.csv')

resultados2023, predicts2023, data_init2023, spread2023 = analisis(data_bid, data_ask, date='03.01.2023')

resultados2023.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados2023.xlsx')
predicts2023.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts2023.xlsx')
data_init2023.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init2023.xlsx')


# DAYS

# Carga de datos 2022 / 2023
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2022-31.08.2023.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2022-31.08.2023.csv')

resultados2223, predicts2223, data_init2223, spread2223 = analisis(data_bid, data_ask, date='03.01.2023')

resultados2223.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados2223.xlsx')
predicts2223.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts2223.xlsx')
data_init2223.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init2223.xlsx')


# Carga de datos 2021 / 2022
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2021-31.12.2022.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2021-31.12.2022.csv')

resultados2122, predicts2122, data_init2122, spread2122 = analisis(data_bid, data_ask, date='03.01.2022')

resultados2122.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados2122.xlsx')
predicts2122.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts2122.xlsx')
data_init2122.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init2122.xlsx')

# Carga de datos 2020 / 2021
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2020-31.12.2021.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2020-31.12.2021.csv')

resultados2021, predicts2021, data_init2021, spread2021 = analisis(data_bid, data_ask, date='04.01.2021')

resultados2021.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados2021.xlsx')
predicts2021.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts2021.xlsx')
data_init2021.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init2021.xlsx')

# Carga de datos 2019 / 2020
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2019-31.12.2020.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2019-31.12.2020.csv')

resultados1920, predicts1920, data_init1920, spread1920 = analisis(data_bid, data_ask, date='03.01.2020')

resultados1920.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados1920.xlsx')
predicts1920.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts1920.xlsx')
data_init1920.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init1920.xlsx')

# Carga de datos 2018 / 2019
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2018-31.12.2019.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2018-31.12.2019.csv')

resultados1819, predicts1819, data_init1819, spread1819 = analisis(data_bid, data_ask, date='03.01.2019')

resultados1819.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados1819.xlsx')
predicts1819.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts1819.xlsx')
data_init1819.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init1819.xlsx')

# Carga de datos 2017 / 2018
data_bid = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_26.01.2017-31.12.2018.csv')
data_ask = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_26.01.2017-31.12.2018.csv')

resultados1718, predicts1718, data_init1718, spread1718 = analisis(data_bid, data_ask, date='03.01.2018')

resultados1718.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/resultados1718.xlsx')
predicts1718.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/predicts1718.xlsx')
data_init1718.to_excel('TFM_MUSI-Instrucciones_autor_LaTeX/resultados/data_init1718.xlsx')



# ESTRATEGIA TRADING









' EXPORTADO DE DATOS '



#data_init.to_excel('data_init.xlsx')

#predicts.to_excel('predicts.xlsx')
#resultados.to_excel('resultados.xlsx')
#data.to_excel('data.xlsx')








