# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 10:24:19 2023

@author: mmtar96
"""

import pandas as pd

data_bid = pd.read_csv('data/USA500.IDXUSD_Candlestick_1_h_BID_01.01.2022-31.12.2022.csv')
data_ask = pd.read_csv('data/USA500.IDXUSD_Candlestick_1_h_ASK_01.01.2022-31.12.2022.csv')

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


# Calculo volatilidad

