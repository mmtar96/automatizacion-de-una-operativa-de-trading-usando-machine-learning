' CALCULO DE INDICADORES TÉCNICOS '

' V2 - Incorporacion de nombres de variables dinamicas para poder calcular varias medias en la misma funcion - NO SE AUN MUY BIEN COMO HACERLO'

import pandas as pd
import numpy as np

# Calculo de la media movil
def sma(mPrice, periodo): 
    sma10 = []
    sumatorio = 0
    for i in range(len(mPrice)):
        if i < periodo:
            sumatorio = mPrice[i] + sumatorio
            sma10.append(0)
        if i >= periodo:
            sumatorio = mPrice[i] + sumatorio - mPrice[i-periodo]
            sma = sumatorio / periodo
            sma10.append(sma)
    sma = pd.DataFrame({'SMA' : sma10})
    return sma

# Calculo de la media movil ponderarad
def sma_p(mPrice, periodo): # Costo de computo MUY elevado - INASUMIBLE
    sma10 = []
    pond = 0
    for h in range(periodo):
        pond = h + 1 + pond
    for i in range(len(mPrice)):
        if i < periodo:
            sma10.append(0)
        if i >= periodo:
            p = i - periodo + 1
            sumatorio = 0
            for j in range(periodo): # Se introduce un bucle que hace que el coste de computo sea "periodo" veces superior
                sumatorio = mPrice[p+j]*(j+1) + sumatorio
            sma = sumatorio / pond
            sma10.append(sma)
    sma_p = pd.DataFrame({'SMA_P' : sma10})
    return sma_p

# Calculo de la media exponencial
def ema(mPrice, periodo):
    ema = []
    sumatorio = 0
    mult = 2 / (periodo + 1)
    for i in range(len(mPrice)):
        if i < (periodo-1):
            sumatorio = mPrice[i] + sumatorio
            ema.append(0)
        if i == (periodo-1):
            sumatorio = mPrice[i] + sumatorio
            sma = sumatorio / periodo
            ema.append(sma)
        if i >= periodo:
            ema_val = (mPrice[i] - ema[i-1])*mult + ema[i-1]
            ema.append(ema_val)
    ema = pd.DataFrame({'EMA' : ema})    
    return ema

# Calculo del RSI
def rsi(mPrice, periodo):
    rsi = []
    for i in range(len(mPrice)):
        j = i - periodo + 1
        sma_asc = []
        sma_desc = []
        for p in range(periodo):
            try:
               if mPrice[j-1] < mPrice[j]:
                   asc = mPrice[j-1] - mPrice[j]
                   desc = 0
               if mPrice[j-1] > mPrice[j]:
                   desc = mPrice[j-1] - mPrice[j]
                   asc = 0 
               if mPrice[j-1] == mPrice[j]:
                   desc = 0
                   asc = 0     
               sma_asc.append(abs(asc))
               sma_desc.append(abs(desc))
               j = j + 1
            except:
               pass
        try:
            if j <= 0:
                rsi_val = 0
            else: 
                rsi_val = 100 - 100 / (1 + sum(sma_asc)/sum(sma_desc))
            rsi.append(rsi_val)
        except:
            pass
    rsi = pd.DataFrame({'RSI' : rsi})
    return rsi

# Calculo del MACD 
def macd(mPrice, pme1, pme2):
    ema1 = ema(mPrice, pme1)
    ema2 = ema(mPrice, pme2)
    macd = []
    for i in range(len(ema1)):
        if i < max(pme1, pme2):
            macd.append(0)
        if i >= max(pme1, pme2):
            macd_val = ema1.iloc[i,0] - ema2.iloc[i,0]
            macd.append(macd_val)
    macd = pd.DataFrame({'MACD' : macd})
    return macd
            

'''

# OSCILADOR McCLEAN [INDICE GLOBAL]

# INDICCE ACUMULATIVO McCLEAN [INDICE GLOBAL]

# INDICE ARMS [INDICE GLOBAL]

# INDICE NUEVO MAX - NUEVO MIN

def inmaxmin(cierre_max, cierre_min):
    maxin_all = []
    minim_all = []
    maxin_all.append(cierre_max[0], cierre_min[0])
    for i in range(len(cierre_max)):
        i = i + 1
        try:
            if  cierre_max[i-1] < cierre_max[i]:
                minim_all.append(cierre_max[i])
            else:
                minim_all.append(cierre_max[i-1])
        except:
            print("algo va mal al alza")
        
        try:
            if cierre_min[i-1] > cierre_min[i]:
                maxin_all.append(cierre_min[i])
            else:
                maxin_all.append(cierre_min[i-1])
        except:
            print("algo va mal al a la baja")
          
            
'''
            
    
            
    
