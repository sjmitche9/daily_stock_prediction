import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import ta
from binance.client import Client
import config
import os

# os.environ['TZ'] = 'America/Los_Angeles'

def get_companies():
    payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    df = first_table
    symbols = df['Symbol'].values.tolist()
    return symbols

# str(dt.datetime.now() - dt.timedelta(hours=50)), str(dt.datetime.now()))
def get_all(hist_data, ticker):
    try:
        mom_data = ta.add_all_ta_features(hist_data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    except:
        print(f'No data for {ticker}.')
        mom_data = pd.DataFrame()
    return mom_data

def get_data(begin, finish, ticker, crypto, client, trade_frequency, training):
    intervals = {1: Client.KLINE_INTERVAL_1MINUTE, 5: Client.KLINE_INTERVAL_5MINUTE, 15: Client.KLINE_INTERVAL_15MINUTE,
    30: Client.KLINE_INTERVAL_30MINUTE, 60: Client.KLINE_INTERVAL_1HOUR, 240: Client.KLINE_INTERVAL_4HOUR, 720: Client.KLINE_INTERVAL_12HOUR, 
    1440: Client.KLINE_INTERVAL_1DAY}
    if crypto:
        if training:
            klines = client.get_historical_klines(ticker, intervals[trade_frequency],
            str(begin - dt.timedelta(hours=2000*trade_frequency)), str(finish))
        else:
            klines = client.get_historical_klines(ticker, intervals[trade_frequency],
            str(begin - dt.timedelta(hours=50*trade_frequency)), str(finish))
        opens = [float(x[1]) for x in klines]
        highs = [float(x[2]) for x in klines]
        lows = [float(x[3]) for x in klines]
        closes = [float(x[4]) for x in klines]
        volumes = [float(x[5]) for x in klines]
        # time = [float(x[6]) for x in klines]
        hist_data = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': volumes}) # add index=time eventually
    else:
        back_begin = begin - dt.timedelta(days=3000)
        company = yf.Ticker(ticker)
        hist_data = company.history(start=back_begin, end=finish)
    mom_data = get_all(hist_data, ticker)
    if crypto:
        mom_data = mom_data.iloc[100:, :]
        # print(f'Nulls to drop from get data: {mom_data.isnull().sum()}')
        # mom_data = mom_data.dropna(axis=1)
    else:
        mom_data = mom_data.loc[begin - dt.timedelta(2900):]

    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if all([col in mom_data.columns for col in columns]):
        if mom_data.shape[0] >= 280:
            return mom_data
        else:
            return None
    else:
        return None