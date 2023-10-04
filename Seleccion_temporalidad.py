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

# LIBRERIAS DESARROLLADAS
%cd code/libs
import ipdata as ip
import indtechv2 as it
%cd ..
%cd ..


' CARGA DE DATOS '

data_bid17 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_26.01.2017-31.12.2017.csv')
data_ask17 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_26.01.2017-31.12.2017.csv')

data_bid18 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2018-31.12.2018.csv')
data_ask18 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2018-31.12.2018.csv')

data_bid19 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2019-31.12.2019.csv')
data_ask19 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2019-31.12.2019.csv')

data_bid20 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2020-31.12.2020.csv')
data_ask20 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2020-31.12.2020.csv')

data_bid21 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2021-31.12.2021.csv')
data_ask21 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2021-31.12.2021.csv')

data_bid22 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2022-31.12.2022.csv')
data_ask22 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2022-31.12.2022.csv')

data_bid23 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_ASK_01.01.2023-31.08.2023.csv')
data_ask23 = pd.read_csv('data/MSFT.USUSD_Candlestick_1_D_BID_01.01.2023-31.08.2023.csv')


# Tratamiento y preprocesado de datos
def preprocesado(data_bid, data_ask):
    data_bid, data_ask = ip.correct_data(data_bid, data_ask)
    data_bidF = ip.import_data(data_bid)
    data_askF = ip.import_data(data_ask)
    dataset = ip.merge(data_bidF, data_askF) 
    ip.ploteo(dataset)
    return dataset

dataset17 = preprocesado(data_bid17, data_ask17)
dataset18 = preprocesado(data_bid18, data_ask18)
dataset19 = preprocesado(data_bid19, data_ask19)
dataset20 = preprocesado(data_bid20, data_ask20)
dataset21 = preprocesado(data_bid21, data_ask21)
dataset22 = preprocesado(data_bid22, data_ask22)
dataset23 = preprocesado(data_bid23, data_ask23)

#Volatilidad media (%)
volatilidad_media = np.mean(abs(((dataset['Open'] - dataset['Close'])/dataset['Open'])*100))
spread_medio = np.mean(dataset['Spread'])


# Estadistica descriptiva


# Precios de apertura

def desc(datos):
    media = np.mean(datos)
    mediana = np.median(datos)
    cuartiles = np.percentile(datos, [25, 50, 75])
    rango = np.ptp(datos)
    varianza = np.var(datos)
    desviacion_estandar = np.std(datos)
    result = [media, mediana, cuartiles[0], cuartiles[1], cuartiles[2], rango, varianza, desviacion_estandar]
    return result

desc17op = desc(dataset17['Open'])
desc18op = desc(dataset18['Open'])
desc19op = desc(dataset19['Open'])
desc20op = desc(dataset20['Open'])
desc21op = desc(dataset21['Open'])
desc22op = desc(dataset22['Open'])
desc23op = desc(dataset23['Open'])

metricas = ['Media', 'Mediana', 'Cuartil 25', 'Cuartil 50', 'Cuartil 75', 'Amplitud', 'Varianza', 'Desviacion Estandar']
desc = pd.DataFrame({'Metricas' : metricas,
                     '2017' : desc17op,
                     '2018' : desc18op,
                     '2019' : desc19op,
                     '2020' : desc20op,
                     '2021' : desc21op,
                     '2022' : desc22op,
                     '2023' : desc23op,})

desc.to_excel('metricas.xlsx')

fig = plt.figure()
plt.boxplot([dataset17['Open'], dataset18['Open'], dataset19['Open'], dataset20['Open'], dataset21['Open'], dataset22['Open'], dataset23['Open']], labels=['2017', '2018', '2019', '2020', '2021', '2022', '2023'])
plt.xlabel('Periodo')
plt.ylim(0,400)
plt.ylabel('Precio')
plt.grid(True)
plt.show()




