
# -*- coding: utf-8 -*-
"""
reference: http://gouthamanbalaraman.com/blog/calculating-stock-beta.html
"""
import pandas_datareader.data as web
import datetime
import numpy as np
import pandas as pd


# Grab time series data for 5-year history for the stock (here TEF.MC)
# and for IBEX 35 Index

edate =   datetime.datetime(2004,12, 31, 0, 0, 0, 0)
sdate = edate - datetime.timedelta(days=5*365)

ticker_symbol = 'TEF.MC'
ref_index = '^IBEX'

df_stock = web.DataReader(ticker_symbol,'yahoo',sdate,edate)
df_index = web.DataReader(ref_index,'yahoo',sdate,edate)

# create a time-series of monthly data points
df_stock = df_stock.resample('M').last()
df_index = df_index.resample('M').last()

df_stock['returns'] = df_stock['Adj Close']/ df_stock['Adj Close'].shift(1) -1
df_stock = df_stock.dropna()
df_index['returns'] = df_index['Adj Close']/ df_index['Adj Close'].shift(1) -1
df_index = df_index.dropna()

df = pd.DataFrame({'stock_returns' : df_stock['returns'],
                        'index_returns' : df_index['returns']},
                        index=df_stock.index)
df = df.dropna()


# reference - http://ci.columbia.edu/ci/premba_test/c0331/s7/s7_5.html
def covariance(a, b):
    if len(a) != len(b):
        return
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    sum = 0
    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))
    return sum/(len(a)-1)


print(edate)
numerator = covariance(df['stock_returns'],df['index_returns'])
print("COVARIANCE(stock, benchmark) = COVARIANCE("+ticker_symbol+", "+ref_index +") = " +str(numerator))
denominator = covariance(df['index_returns'],df['index_returns'])
print("VARIANCE(benchmark) = COVARIANCE(benchmark, benchmark) = COVARIANCE("+ref_index+", "+ref_index +") = " +str(denominator))

# BETA = Covariance (stock,index) / Variance (Index) = Covariance (stock,index) / Covariance (stock,stock)
print("BETA = COVARIANCE(stock, benchmark) / VARIANCE(benchmark) = " + str(numerator) + " / " + str(denominator) + " = " +str(covariance(df['stock_returns'],df['index_returns'])/covariance(df['index_returns'],df['index_returns'])))
#http://www.investopedia.com/ask/answers/070615/what-formula-calculating-beta.asp