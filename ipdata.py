' IMPORTADO Y PREPROCESADO DE DATOS '

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Correccion de nº de datos
def correct_data(data_bid, data_ask):
    minimo = min(len(data_bid), len(data_ask))
    j = 0
    for i in range(minimo):
        if len(data_bid) > len(data_ask):
            if data_bid.iloc[j,0] != data_ask.iloc[i,0]:
                data_bid.drop(j, axis=0, inplace=True) 
        if len(data_bid) < len(data_ask):
            if data_bid.iloc[i,0] != data_ask.iloc[j,0]:
                data_ask.drop(j, axis=0, inplace=True) 
        j = j + 1
        
    data_bid = data_bid.reset_index().iloc[:,1:]
    data_ask = data_ask.reset_index().iloc[:,1:]
    
    return data_bid, data_ask

# Importado de datos
def import_data(data):  
    #Correccion de fecha
    date = []
    time = []
    for i in range(len(data)):
        aux1 = data.iloc[i,0]
        date_aux = aux1.split(' ')[0]
        time_aux = aux1.split(' ')[1].split('.')[0]
        date.append(date_aux)
        time.append(time_aux)
    dataFaux = pd.DataFrame({'Date' : date,
                             'Time' : time})
    final_data = pd.concat([dataFaux, data.iloc[:,1:]], axis = 1)
    
    return final_data

# Crear el dataset definitivo con la media de valores y añadir una columna mas calculando el spread
def merge(data_bid, data_ask):
    if data_bid.size == data_ask.size:
        mOpen_all = []
        mHigh_all = []
        mLow_all = []
        mClose_all = []
        mPrice_all = []
        mVolbid_all = []
        mVolask_all = []
        mVol_all = [] 
        mSpread_all = []
        Spreads = []
        for i in range(len(data_bid)):
            mOpen = (data_bid.iloc[i,2]+data_ask.iloc[i,2])/2
            spread_open = abs(-data_bid.iloc[i,2]+data_ask.iloc[i,2])/mOpen*100 # Spread en % sobre la media
            mHigh = (data_bid.iloc[i,3]+data_ask.iloc[i,3])/2
            spread_high = abs(-data_bid.iloc[i,3]+data_ask.iloc[i,3])/mHigh*100 # Spread en % sobre la media
            mLow = (data_bid.iloc[i,4]+data_ask.iloc[i,4])/2
            spread_low = abs(-data_bid.iloc[i,4]+data_ask.iloc[i,4])/mLow*100 # Spread en % sobre la media
            mClose = (data_bid.iloc[i,5]+data_ask.iloc[i,5])/2
            spread_close =  abs(-data_bid.iloc[i,5]+data_ask.iloc[i,5])/mClose*100 # Spread en % sobre la media
            mPrice = (mOpen + mHigh + mLow + mClose)/4 # El calculo del mPrice habria que revisarlo ya que es un PILOTO
            mSpread = (spread_open + spread_close)/2 # Media de los spreads para Open y Close
            mVolbid = data_bid.iloc[i,6]
            mVolask = data_ask.iloc[i,6]
            mVol = (data_bid.iloc[i,6]+data_ask.iloc[i,6]) 
            mOpen_all.append(mOpen)
            mHigh_all.append(mHigh)
            mLow_all.append(mLow)
            mClose_all.append(mClose)
            mPrice_all.append(mPrice)
            mVolask_all.append(mVolask)
            mVolbid_all.append(mVolbid)
            mVol_all.append(mVol)
            mSpread_all.append(mSpread)
            Spreads.append([spread_open, spread_high, spread_low, spread_close])
        data = pd.DataFrame({'Open' : mOpen_all,
                             'High' : mHigh_all,
                             'Low' : mLow_all,
                             'Close' : mClose_all,
                             'mPrice' : mPrice_all,
                             'Volume_Ask' : mVolask_all,
                             'Volume_Bid' : mVolbid_all,
                             'Volume' : mVol_all,
                             'Spread' : mSpread_all,
                             'All_spreads' : Spreads})
        dataset = pd.concat([data_bid.iloc[:,:2], data], axis = 1)
        return dataset

# Ploteo de los datos
def ploteo(dataset):
    fig = plt.figure()
    plt.plot(dataset.iloc[:,0], dataset.iloc[:,6], color='green', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.xticks()
    plt.xticks([2017, 2018, 2019, 2020, 2021, 2022, 2023])
    plt.show()