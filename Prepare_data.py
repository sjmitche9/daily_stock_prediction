import os
import time
import unicodedata
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import yfinance as yf

class Prepare_data():

    def __init__(self, ticker, features, num_features, drop_features, delta, scaler, crypto, training, to_predict, scale, live):
        self.scaler = scaler
        self.scale = scale
        self.delta = delta
        self.ticker = ticker
        self.features = features
        self.num_features = num_features
        self.drop_features = drop_features
        self.training = training
        self.crypto = crypto
        self.to_predict = to_predict
        self.live = live

        print(f'Preparing data: Live is {live}')
        if not live:
            self.file_path = f'{os.getcwd()}/Data/{ticker}.csv'
        else:
            self.file_path = f'{os.getcwd()}/Data/{ticker}_live.csv'
        if not crypto:
            self.data = pd.read_csv(self.file_path, index_col='Date')
        else:
            self.data = pd.read_csv(self.file_path, index_col=0)
        
        self.data = self.data.replace('null', np.nan).fillna(0).astype('float')
      
        days_list = [10, 20, 30, 60]
        for days in days_list:
            self.data = self.CCI(self.data, days, to_predict)

        days_list = [14, 20, 30, 60]
        for days in days_list:
            self.data = self.EVM(self.data, days)

        days_list = [10, 50, 100, 200]
        for days in days_list:
            self.data = self.SMA(self.data, days, to_predict)
            self.data = self.EWMA(self.data, days, to_predict)

        days_list = [5, 10, 15, 20]
        for days in days_list:
            self.data = self.ROC(self.data, days, to_predict)

        days_list = [20, 30, 50, 60, 80]
        for days in days_list:
            self.data = self.bbands(self.data, days, to_predict)

        days_list = range(1, 11)
        for days in days_list:
            self.data = self.ForceIndex(self.data, days, to_predict)

        self.data = self.keep_feats(self.data, features, num_features, drop_features, to_predict)

        self.train, self.test = self.split(self.data, self.delta)
        if live:
            self.test = self.data
        self.preprocessed_train, self.scale = self.rescale_data(self.train, scaler, self.scale, self.live, split='train')
        self.preprocessed_test, self.scale = self.rescale_data(self.test, scaler, self.scale, self.live, split='test')

        # self.preprocessed_live, scale = self.rescale_data(self.data, scaler, scale, split='test')


    def CCI(self, data, days, to_predict):
        TP = (data['High'] + data['Low'] + data[to_predict]) / 3
        CCI = pd.Series((TP - TP.rolling(days).mean()) / (0.015 * TP.rolling(days).std()), name='CCI' + str(days))
        data = data.join(CCI)
        return data

    def EVM(self, data, days):
        dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
        br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
        EVM = dm / br
        EVM_MA = pd.Series(EVM.rolling(days).mean(), name = 'EVM' + str(days))
        data = data.join(EVM_MA)
        return data

    def SMA(self, data, days, to_predict): 
        sma = pd.Series(data[to_predict].rolling(days).mean(), name = 'SMA ' + str(days))
        data = data.join(sma) 
        return data

    def EWMA(self, data, days, to_predict):
        ema = pd.Series(data[to_predict].ewm(span = days, min_periods = days - 1).mean(), name = 'EWMA ' + str(days))
        data = data.join(ema)
        return data

    def ROC(self, data, days, to_predict):
        N = data[to_predict].diff(days)
        D = data[to_predict].shift(days)
        roc = pd.Series(N/D,name='ROW' + str(days))
        data = data.join(roc)
        return data 

    def bbands(self, data, days, to_predict):
        MA = data[to_predict].rolling(window=days).mean()
        SD = data[to_predict].rolling(window=days).std()
        data['UpperBB' + str(days)] = MA + (2 * SD) 
        data['LowerBB' + str(days)] = MA - (2 * SD)
        return data

    def ForceIndex(self, data, days, to_predict): 
        FI = pd.Series(data[to_predict].diff(days) * data['Volume'], name = 'Force Index' + str(days)) 
        data = data.join(FI)
        return data

    def keep_feats(self, data, features, num_features, drop_features, to_predict):
        if drop_features:
            open_close = data[to_predict]
            to_keep = [f for f in features[:num_features] if f in data.columns]
            data = data[to_keep]
            data[to_predict] = open_close
        return data

    def split(self, data, delta):
        data = data.iloc[200:, :]
        train = data.iloc[:-delta, :]
        test = data.iloc[-delta:, :]
        return train, test

    def rescale_data(self, data, scaler, scale, live, split):
        if live == True and split == 'train':
            return None, scale
        self.col_names = data.columns
        if scaler == 'MinMax' and split == 'train':
            scale = MinMaxScaler(feature_range=(0,1))
        elif scaler == 'Standard' and split == 'train':
            scale = StandardScaler()
        elif scaler is None:
            print('Warning, no scaler selected!')
        if len(data > 0) and split == 'train':
            try:
                scale.fit(data)
                data = scale.transform(data)
            except:
                data = data
                print("Scaling didn't work")
            data = pd.DataFrame(data, columns=self.col_names)
            return data, scale
        elif len(data > 0) and split == 'test':
            try:
                data = scale.transform(data)
            except:
                data = data
                print("Scaling didn't work")
            data = pd.DataFrame(data, columns=self.col_names)
            return data, scale
		
if __name__ == '__main__':
	prepare_data = Prepare_data()