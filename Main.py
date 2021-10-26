from binance.client import Client
from binance.enums import *
from keras.models import model_from_json
import statistics as stats
import numpy as np
import pandas as pd
import sklearn, json, collections, Get_data, trade, live_trade, pprint, config, time, pickle, schedule, os, keras
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sns.set(color_codes=True)
import datetime as dt
import yfinance as yf
import Prepare_data as pdat
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
import crypto_lstm
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###----------------------------------------------------------------------------------------------------------------------------------###
# enter variables here, choose 'MinMax' or 'Standard' for scalers, fill in features to keep after analyzing feature contribution
# drop features to use that number of features, set ticker for a single company or leave None for multiple companies

begin = 'Jan 2 2021'
finish = 'Jun 4 2021'   #May 23 2021
ticker = 'LTCUSDT'

# ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK']

lower_threshold = .4
upper_threshold = .6
crypto = True
live = True
scaler = 'MinMax'
num_features = 128 ### keep this at 128
drop_features = True
same_drop = True
graph_features = False
graph_results = True
shares = 1
num_companies = len(ticker) if type(ticker) == list else 1
trade_frequency = 240
train_xg = False
train_lstm = False
to_predict = 'Close'
predict_xg_probs = True
start_in_position = False # use this only for simulation purposes
use_all_feats_lstm = False
lookback = 30
lstm_prep_saved = False
client = Client(config.API_KEY, config.API_SECRET)

all_features = ['volatility_dcm', 'momentum_kama', 'volatility_bbh', 'trend_ichimoku_base', 'trend_visual_ichimoku_a',
'volume_nvi', 'trend_visual_ichimoku_b', 'LowerBB50', 'trend_ichimoku_conv', 'LowerBB80', 'volatility_kcc', 'EWMA 50',
'volatility_kcl', 'trend_ichimoku_a', 'momentum_tsi', 'LowerBB60', 'CCI20', 'EWMA 200', 'EWMA 10', 'volatility_bbl',
'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'SMA 50', 'trend_vortex_ind_diff', 'SMA 200', 'others_cr', 'Force Index6',
'volatility_atr', 'Force Index9', 'UpperBB80', 'Low', 'trend_macd_diff', 'Close', 'trend_adx_pos', 'volatility_ui', 'ROW10',
'trend_dpo', 'volume_em', 'momentum_stoch_rsi', 'volatility_dcp', 'volatility_kchi', 'trend_mass_index', 'Force Index2',
'Open', 'Force Index7', 'momentum_wr', 'UpperBB20', 'volatility_dch', 'momentum_stoch', 'trend_macd_signal', 'trend_cci',
'trend_aroon_ind', 'CCI60', 'volatility_bbp', 'momentum_ppo_signal', 'trend_stc', 'EVM14', 'volume_mfi', 'CCI30',
'volatility_kcp', 'SMA 100', 'volume_adi', 'volatility_bbw', 'EVM30', 'others_dr', 'volume_fi', 'trend_aroon_down',
'others_dlr', 'trend_vortex_ind_neg', 'trend_kst_diff', 'trend_psar_up', 'momentum_ppo_hist', 'volume_sma_em', 'Force Index8',
'volatility_kcw', 'momentum_uo', 'UpperBB30', 'Volume', 'Force Index5', 'Force Index4', 'momentum_stoch_signal', 'CCI10',
'ROW20', 'trend_vortex_ind_pos', 'trend_kst_sig', 'Force Index3', 'volatility_dcl', 'Force Index10', 'EVM20', 'UpperBB50',
'volatility_dcw', 'trend_adx_neg', 'LowerBB20', 'trend_trix', 'EWMA 100', 'trend_macd', 'trend_adx', 'volume_obv', 'SMA 10',
'momentum_ppo', 'EVM60', 'momentum_rsi', 'volatility_kch', 'trend_psar_down', 'volume_cmf', 'momentum_roc', 'Force Index1',
'ROW5', 'LowerBB30', 'momentum_ao', 'volatility_bbm', 'volume_vpt', 'trend_aroon_up', 'ROW15', 'UpperBB60', 'trend_kst',
'trend_sma_slow', 'volume_vwap', 'High', 'trend_ema_slow', 'trend_ichimoku_b', 'volatility_bbhi', 'volatility_bbli',
'volatility_kcli', 'trend_ema_fast', 'trend_psar_down_indicator', 'trend_psar_up_indicator', 'trend_sma_fast']

if train_xg or train_lstm:
    training = True
else:
    training = False
if training:
    response = input('Are you sure you want to train models?')
    if response in ['no', 'n', 'No', 'N']:
        sys.exit()

pd.set_option('display.max_rows', 150)

class Main(object):

    def __init__(self):
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
    
    def prepare_X_y(self, data, features, lookback):
        ind = list(data.columns).index(to_predict)
        ind2 = list(data.columns).index('Close')
        X = data.values
        y = []
        if to_predict == 'Close':
            for i in range(X.shape[0] - 1):
                if (X[i+1,ind]-X[i,ind]) > 0:
                    y.append(1)
                else:
                    y.append(0)
        elif to_predict == 'Open':
            for i in range(X.shape[0] - 1):
                if (X[i+1,ind]-X[i,ind2]) > 0:
                    y.append(1)
                else:
                    y.append(0)
        y = np.array(y)
        y = y[lookback:]
        X = data
        X = X[features]
        if not train_xg:
            X = X[new_features]
        X = np.array(X)
        X = X[:-1]
        X = X[lookback:, :]
        # saved_data = data.iloc[lookback:-1, :]
        # saved_data['target'] = y
        # saved_data.to_csv(f'{os.getcwd()}/saved_data_{len(y)}.csv')
        return X, y
    
    def prepare_lstm(self, data, features, lookback):
        X = []
        y = []
        if use_all_feats_lstm:
            for i in range(lookback, len(data) - 1):
                if i == lookback:
                    X = np.array(data[features].iloc[i-lookback : i, :100])
                else:
                    X = np.append(X, np.array(data[features].iloc[i-lookback : i, :100]), axis=0)
                y.append(int(data[to_predict].iloc[i] < data[to_predict].iloc[i+1]))
        else:
            for i in range(lookback, len(data) - 1):
                X.append(data[to_predict][i-lookback : i])
                y.append(int(data[to_predict].iloc[i] < data[to_predict].iloc[i+1]))
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def split_train_test(self,X,y):
        split_ratio=0.9
        train_size = int(round(split_ratio * X.shape[0]))
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        return X_train, X_test, y_train, y_test
        
    def class_balance(self, train):
        count_class_0, count_class_1 = train['target'].value_counts()
        train_class_0 = train[train['target'] == 0]
        train_class_1 = train[train['target'] == 1]

        if count_class_0 > count_class_1:
            train_class_0_under = train_class_0.sample(count_class_1)
            train_sampled = pd.concat([train_class_0_under, train_class_1], axis=0)
        else:
            train_class_1_under = train_class_1.sample(count_class_0)
            train_sampled = pd.concat([train_class_0, train_class_1_under], axis=0)
        
        print(train_sampled['target'].value_counts())
        train_sampled['target'].value_counts().plot(kind='bar', title='Count (target)')
        plt.show()
        return train_sampled
    
    def train_model(self, X_train, y_train):
        model = xgb.XGBClassifier(eval_metric='logloss', n_estimators=600,
        objective='binary:logistic', nthread=1, use_label_encoder=False)
        params = {
        'min_child_weight': [1, 5, 7.5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [.01, .02, .05, .1, .15, .2]
        }
        random_search = RandomizedSearchCV(model, param_distributions=params, scoring='accuracy', n_jobs=-1,
        verbose=0, random_state=1001)
        random_search.fit(X_train, y_train)
        # print('\n All results:')
        # print(random_search.cv_results_)
        # print('\n Best estimator:')
        # print(random_search.best_estimator_)
        # print(random_search.best_score_ * 2 - 1)
        # print('\n Best hyperparameters:')
        # print(random_search.best_params_)
        results = pd.DataFrame(random_search.cv_results_)
        # results.to_csv(f'{os.getcwd()}/Data/xgb-random-grid-search-results-{ticker[i]}_{trade_frequency}.csv', index=False)
        model = random_search.best_estimator_
        model.fit(X_train, y_train)
        return model

    def predict(self, model, X_test, y_test):
        if predict_xg_probs:
            y_pred = model.predict_proba(X_test)
            y_pred = [x[1] for x in y_pred]
        else:
            y_pred = model.predict(X_test)
            y_pred = [round(value) for value in y_pred]
        if y_test is not None and predict_xg_probs:
            class_pred = [0 if x < .5 else 1 for x in y_pred]
            accuracy = accuracy_score(y_test, class_pred)
            precision = precision_score(y_test, class_pred)
            recall = recall_score(y_test, class_pred)
            return y_pred, accuracy, precision, recall
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            return y_pred, accuracy, precision, recall
    
    def plot_feature_imp(self, data, model):
        imp_score = pd.DataFrame(model.feature_importances_, columns=['Importance Score'])
        features = pd.DataFrame(columns, columns=['Features'])
        feature_imp = pd.concat([features, imp_score], axis=1)
        feature_imp = feature_imp.sort_values(by='Importance Score', ascending=False).reset_index()
        features = feature_imp['Features'].tolist()
        print(feature_imp)
        print()
        if graph_features:
            sns.barplot(x=feature_imp['Importance Score'], y=feature_imp['Features'])
            plt.show()
        return features

if __name__ == '__main__':
    Main = Main()

    if ticker is None:
        ticker = Get_data.get_companies()
    elif type(ticker) == list:
        ticker = ticker
    else:
        ticker = [ticker]
    combine = []
    all_day_values_df = pd.DataFrame()
    daily_worth_df = pd.DataFrame()
    warnings = []
    for i in range(num_companies):
        if i == 0 or not same_drop:
            features = all_features
            num_features = len(features)
        if not train_xg:
            if crypto:
                load = pd.read_csv(f'{os.getcwd()}/Data/features_{ticker[i]}_{trade_frequency}.csv', index_col=0)
            else:
                load = pd.read_csv(f'{os.getcwd()}/Data/features_{ticker[i]}.csv', index_col=0)
            features = [x[0] for x in load.values]
            new_features = features
            num_features = len(features)
            print('Number of features:', num_features)
        else:
            save = pd.Series(features)
            if crypto:
                save.to_csv(f'{os.getcwd()}/Data/features_{ticker[i]}_{trade_frequency}.csv')
            else:
                save.to_csv(f'{os.getcwd()}/Data/features_{ticker[i]}.csv')
        if crypto:
            filename = f'{ticker[i]}_model_{trade_frequency}'
            lstm_filename = f'{ticker[i]}_lstm_model_{trade_frequency}_{lookback}_{use_all_feats_lstm}'
        else:
            filename = f'{ticker[i]}_model'
            lstm_filename = f'{ticker[i]}_lstm_model_{use_all_feats_lstm}'
        print(str(i) + " iteration")
        start = dt.datetime.strptime(begin, "%b %d %Y")
        end = dt.datetime.strptime(finish, "%b %d %Y")
        df = Get_data.get_data(start, end, ticker[i], crypto, client, trade_frequency, training)
        if df is None:
            print(f'Skipping {ticker[i]} due to lack of data.')
            continue
        df.to_csv(f'{os.getcwd()}/Data/{ticker[i]}.csv')
        all_day_values_df[ticker[i]] =  df[to_predict]
        if i == 0:
            all_day_values_df = all_day_values_df.set_index(df.index)
        if not crypto:
            delta = sum([1 for x in pd.date_range(start, end) if x in df.index])
        else:
            delta = int((end - start).days * 1440 / trade_frequency)
        scale = None
        Prepare = pdat.Prepare_data(ticker[i], features, num_features, drop_features, delta, scaler, crypto, 
        training, to_predict, scale, live=False)
        Main.train = Prepare.preprocessed_train
        Main.test = Prepare.preprocessed_test
        scale = Prepare.scale
        Main.train = Main.train.replace([np.inf, -np.inf], np.nan)
        Main.test = Main.test.replace([np.inf, -np.inf], np.nan)
        nulls1 = Main.train.isnull().sum().values
        nulls2 = Main.test.isnull().sum().values
        total_nulls = sum(nulls1) + sum(nulls2)
        if total_nulls != 0:
            warning = f'WARNING: {total_nulls} null values in {len(nulls1) + len(nulls2)} columns'
            print(warning)
            warnings.append(warning)
            if total_nulls > 500:
                warning = f'Too many nulls, skipping {ticker[i]}'
                print(warning)
                warnings.append(warning)
                continue
        Main.train = Main.train.fillna(0)
        Main.test = Main.test.fillna(0)
        columns = Main.test.columns
        Main.X_train, Main.y_train = Main.prepare_X_y(Main.train, columns, lookback)
        Main.X_test, Main.y_test = Main.prepare_X_y(Main.test, columns, lookback)
        # if not lstm_prep_saved:
            # Main.lstm_X_train, Main.lstm_y_train = Main.prepare_lstm(Main.train, columns, lookback)
            # Main.lstm_X_test, Main.lstm_y_test = Main.prepare_lstm(Main.test, columns, lookback)
        if not lstm_prep_saved and use_all_feats_lstm:
            np.savetxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_Xtrain.csv', Main.lstm_X_train, delimiter=',')
            np.savetxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_Xtest.csv', Main.lstm_X_test, delimiter=',')
            np.savetxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_ytrain.csv', Main.lstm_y_train, delimiter=',')
            np.savetxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_ytest.csv', Main.lstm_y_test, delimiter=',')
        if lstm_prep_saved:
            Main.lstm_X_train = np.genfromtxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_Xtrain.csv', delimiter= ',')
            Main.lstm_X_test = np.genfromtxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_Xtest.csv', delimiter= ',')
            Main.lstm_y_train = np.genfromtxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_ytrain.csv', delimiter= ',')
            Main.lstm_y_test = np.genfromtxt(f'{os.getcwd()}/Data/{lstm_filename}_prep_ytest.csv', delimiter= ',')
        if train_lstm:
            print(f'Training ups and downs counts: {collections.Counter(Main.y_train)}')
            from keras.wrappers.scikit_learn import KerasClassifier
            model_CV = KerasClassifier(build_fn=crypto_lstm.train_model, verbose=1)
            init_mode = ['uniform']#['uniform', 'normal', 'zero']# ['uniform', 'lecun_uniform', 'normal', 'zero', 
                #  'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
            EPOCHS = [1]
            BATCH_SIZE = [16]#[4, 8, 16]
            lookbacks = [15, 30, 45, 60, 75, 90, 120]
            accuracies = []
            # for x in lookbacks:
            if True:
                # lookback = x
                if use_all_feats_lstm:
                    Main.lstm_X_train = Main.lstm_X_train.reshape((Main.lstm_X_train.shape[0] // lookback, lookback, 100))
                    Main.lstm_X_test = Main.lstm_X_test.reshape((Main.lstm_X_test.shape[0] // lookback, lookback, 100))
                else:
                    Main.lstm_X_train = Main.lstm_X_train.reshape((Main.lstm_X_train.shape[0], Main.lstm_X_train.shape[1], 1))
                    Main.lstm_X_test = Main.lstm_X_test.reshape((Main.lstm_X_test.shape[0], Main.lstm_X_test.shape[1], 1))
                param_grid = dict(init_mode=init_mode, epochs=EPOCHS, batch_size=BATCH_SIZE, lookback=[lookback])
                grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, verbose=1)
                grid_result = grid.fit(Main.lstm_X_train, Main.lstm_y_train, validation_data=(Main.lstm_X_test, Main.lstm_y_test))
                print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
                accuracies.append(grid_result.best_score_)
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                results = pd.DataFrame(grid_result.cv_results_)
                # results.to_csv(f'{os.getcwd()}/Data/lstm-grid-search-results-{lstm_filename}.csv', index=False)
                for mean, stdev, param in zip(means, stds, params):
                    print(f'mean={mean:.4}, std={stdev:.4} using {param}')
            print(accuracies)

            lstm_model = grid_result.best_estimator_
            lstm_model.fit(Main.lstm_X_train, Main.lstm_y_train, validation_data=(Main.lstm_X_test, Main.lstm_y_test))
            lstm_model = lstm_model.model
            lstm_model.save(f"{os.getcwd()}/Data/{lstm_filename}")
            print("LSTM model saved")
        else:
            lstm_model = None #keras.models.load_model(f"{os.getcwd()}/Data/{lstm_filename}")
            print("LSTM model loaded")
        if train_xg:
            Main.model = Main.train_model(Main.X_train, Main.y_train)
            Main.model.save_model(f'{os.getcwd()}/Data/{filename}.json')
            print("xgboost model saved")
        else:
            Main.model = xgb.XGBClassifier()
            Main.model.load_model(f'{os.getcwd()}/Data/{filename}.json')
            print("xgboost model loaded")

        train_pred, accuracy, precision, recall = Main.predict(Main.model, Main.X_train, Main.y_train)
        train_acc = round(accuracy * 100, 2)
        y_pred, accuracy, precision, recall = Main.predict(Main.model, Main.X_test, Main.y_test)
        
        # if not train_lstm:
            # if use_all_feats_lstm:
                # Main.lstm_X_test = Main.lstm_X_test.reshape((Main.lstm_X_test.shape[0] // lookback, lookback, 100))
            # else:
                # Main.lstm_X_train = Main.lstm_X_train.reshape((Main.lstm_X_train.shape[0], Main.lstm_X_train.shape[1], 1))
                # Main.lstm_X_test = Main.lstm_X_test.reshape((Main.lstm_X_test.shape[0], Main.lstm_X_test.shape[1], 1))
        lstm_pred = [0 for i in range(len(y_pred))] #crypto_lstm.predict_lstm(lstm_model, Main.lstm_X_train, Main.lstm_y_train, Main.lstm_X_test, Main.lstm_y_test,
        # live)
        # lstm_pred = [x[1] for x in lstm_pred]
        lstm_binary = lstm_pred #[0 if x < .5 else 1 for x in lstm_pred]
        # print(f'LSTM ups and downs counts: {collections.Counter(lstm_binary)}')
        pred_df = pd.DataFrame({'y_test': Main.y_test, 'y_pred': y_pred, 'lstm_pred': lstm_pred, 'lstm_binary': lstm_binary})
        if train_xg:
            Main.data = np.concatenate((Main.train, Main.test), axis=0)
            features = Main.plot_feature_imp(Main.data, Main.model)
        metrics = pd.Series([accuracy, precision, recall])
        if drop_features and train_xg:
            if i == 0 or not same_drop:
                num_features = int(input("Enter number of features to keep: "))
            print()
            Prepare = pdat.Prepare_data(ticker[i], features, num_features, drop_features, delta, scaler, crypto, training, to_predict, scale,
            live = False)
            Main.train = Prepare.preprocessed_train
            Main.test = Prepare.preprocessed_test
            Main.train = Main.train.replace([np.inf, -np.inf], np.nan)
            Main.test = Main.test.replace([np.inf, -np.inf], np.nan)
            Main.train = Main.train.fillna(0)
            Main.test = Main.test.fillna(0)
            features = Main.train.columns
            print(features)
            save = pd.Series(features)
            if crypto:
                save.to_csv(f'{os.getcwd()}/Data/features_{ticker[i]}_{trade_frequency}.csv')
            else:
                save.to_csv(f'{os.getcwd()}/Data/features_{ticker[i]}.csv')
            Main.X_train, Main.y_train = Main.prepare_X_y(Main.train, features, lookback)
            Main.X_test, Main.y_test = Main.prepare_X_y(Main.test, features, lookback)
            Main.model = Main.train_model(Main.X_train, Main.y_train)
            Main.model.save_model(f'{os.getcwd()}/Data/{filename}.json')
            
            train_pred, accuracy, precision, recall = Main.predict(Main.model, Main.X_train, Main.y_train)
            train_acc = round(accuracy * 100, 2)
            y_pred, accuracy, precision, recall = Main.predict(Main.model, Main.X_test, Main.y_test)
            pred_df = pd.DataFrame({'y_test': Main.y_test, 'y_pred': y_pred, 'lstm_pred': lstm_pred})
            Main.data = np.concatenate((Main.train, Main.test), axis=0)
            metrics = pd.Series([accuracy, precision, recall])
        if not crypto:
            dates = [x for x in pd.date_range(start, end) if x in df.index][30:]
            close_values = [df.loc[df.index == x, 'Close'][0] for x in dates]
            open_values = [df.loc[df.index == x, 'Open'][0] for x in dates]
        else:
            dates = list(df.index[-delta + lookback:])
            close_values = list(df['Close'][-delta + lookback:])
            open_values = list(df['Open'][-delta + lookback:])
            pred_df.index = dates[:-1]
            pred_df['close_values'] = close_values[:-1]
            pred_df['open_values'] = open_values[:-1]
            if not drop_features:
                num_features = len(features)
            results, daily_worth = trade.historical(close_values, open_values, pred_df, dates, ticker[i], delta, num_features, metrics,
            crypto, shares, lower_threshold, upper_threshold, to_predict)
            results['Train Accuracy %'] = train_acc
            print(results)
            combine.append(results)
            daily_worth_df[ticker[i]] = daily_worth
            daily_worth_df = daily_worth_df.set_index([dates[:-1]])
        if not crypto:
            pred_df.index = dates[:-1]
            pred_df['close_values'] = close_values[:-1]
            pred_df['open_values'] = open_values[:-1]
            results, daily_worth = trade.historical(close_values, open_values, pred_df, dates, ticker[i], delta, num_features, metrics,
            crypto, shares, lower_threshold, upper_threshold, to_predict)
            results['Train Accuracy %'] = train_acc
            print(results)
            combine.append(results)
            daily_worth_df[ticker[i]] = daily_worth
            daily_worth_df = daily_worth_df.set_index([dates[:-1]])
        if graph_results and to_predict == 'Close':
            trade_worth = [daily_worth[x] + close_values[0] for x in range(len(daily_worth))]
            graph_dates = pd.date_range(start, end, periods=len(trade_worth))
            fig, ax = plt.subplots()
            ax.plot(graph_dates, close_values[:-1], label='Close Values')
            ax.plot(graph_dates, trade_worth, label='Trade Worth')
            # axis labels
            plt.xlabel('Date', fontsize=20)
            plt.ylabel('USD', fontsize=20)
            # show the legend
            plt.legend(fontsize=20)
            # show the title
            plt.title(ticker)
            # show the plot
            plt.show()
        with open(f'{os.getcwd()}/Data/combine_{begin.replace(" ", "-")}.json', 'w') as f:
            f.write(json.dumps(combine, indent=4))
        daily_worth_df.to_csv(f'{os.getcwd()}/Data/daily_worth_df.csv')
        all_day_values_df.to_csv(f'{os.getcwd()}/Data/all_day_values.csv')
    print(warnings)

if live:
    training = False
    intervals = {1: Client.KLINE_INTERVAL_1MINUTE, 5: Client.KLINE_INTERVAL_5MINUTE, 15: Client.KLINE_INTERVAL_15MINUTE,
    30: Client.KLINE_INTERVAL_30MINUTE, 60: Client.KLINE_INTERVAL_1HOUR, 240: Client.KLINE_INTERVAL_4HOUR, 720: Client.KLINE_INTERVAL_12HOUR, 
    1440: Client.KLINE_INTERVAL_1DAY}
    k_line_now = client.get_historical_klines(ticker[0], intervals[trade_frequency], f'{trade_frequency} minutes ago UTC')[0]
    start_time = k_line_now[6]
    if start_in_position:
        print("Overriding true value of stock, USE THIS ONLY FOR SIMULATION PURPOSES!")
        in_position = True
        stock = shares * float(k_line_now[4])
        wallet = -stock
        num_shares = shares
        investment = wallet * -1
        buys = 1
    else:
        in_position = False
        wallet = 0
        num_shares = 0
        stock = 0
        investment = 0
        buys = 0
    
    fees_lst = []
    y_true_lst = []
    y_pred_lst = []
    guess_lst = []
    lstm_pred_lst = []
    action_lst = []
    static_investment = []
    time_count = 0
    trade_count = 0
    zeros = 0
    ones = 0
    start_value = 0
    warnings = []

    def go(client, ticker, features, num_features, drop_features, delta, scaler, crypto, training, trade_frequency, investment):
        global wallet, in_position, num_shares, time_count, trade_count, ones, zeros, stock, buys, start_value, warnings
        print(dt.datetime.fromtimestamp(client.get_server_time()['serverTime'] / 1000))
        last, working_df, open_close, warning = live_trade.close_function(client, ticker[0], features, num_features, drop_features, delta,
        scaler, crypto, training, trade_frequency, to_predict, Prepare, scale, live)

        warnings.append(warning)

        y_pred, accuracy, precision, recall = Main.predict(Main.model, last, [1])
        print(dt.datetime.now())
        if use_all_feats_lstm:
            z = np.array(working_df[features].iloc[-lookback:, :])
            z = z.reshape(z.shape[0], z.shape[1], 1)
        else:
            z = np.array(working_df[to_predict].iloc[-lookback:])
            z = z.reshape(1, lookback, 1)
        lstm_pred = [[0, 0]] #crypto_lstm.predict_lstm(lstm_model, Main.lstm_X_train, Main.lstm_y_train, z, Main.lstm_y_test[-1], live)
        print(lstm_pred)
        print(dt.datetime.now())
        lstm_pred_lst.append(lstm_pred[0][1])

        y_pred_lst.append(y_pred)
        in_position, num_shares, open_close, wallet, fees, action = live_trade.trade(y_pred[0], shares, in_position, ticker[0], client, wallet,
        num_shares, open_close, crypto, lower_threshold, upper_threshold, lstm_pred[0][1])
        time_count += 1
        if time_count == 1:
            start_value = open_close
        if action == 0 or action == 1:
            trade_count += 1
        print(f'TOTAL BUYS: {buys}')
        if action == 1:
            buys += 1
            static_investment.append(round(open_close * shares, 2))
        if len(static_investment) >= 1:
            stat_invest = static_investment[0]
        else:
            stat_invest = 0
        print(f'{to_predict} $: {open_close}, Number of Shares: {num_shares}')
        print()
        stock = open_close * num_shares
        print(f'Stock $: {round(stock, 2)}, Wallet $ {round(wallet, 2)}, Fees $: {round(fees, 2)}')
        print()
        fees_lst.append(fees)
        y_true_lst.append(open_close)
        action_lst.append(action)
        print(f'Fees:{fees_lst}')
        print()
        if len(y_true_lst) > 1:
            if (action_lst[-2] == 1 and y_true_lst[-1] > y_true_lst[-2]) or (action_lst[-2] == 0 and y_true_lst[-1] < y_true_lst[-2]):
                guess_lst.append(True)
            elif (action_lst[-2] == 1 and y_true_lst[-1] < y_true_lst[-2]) or (action_lst[-2] == 0 and y_true_lst[-1] > y_true_lst[-2]):
                guess_lst.append(False)
            else:
                pass
            if y_true_lst[-1] > y_true_lst[-2]:
                ones += 1
            else:
                zeros += 1
        if len(guess_lst) > 1:
            acc = round(guess_lst.count(True) / (len(guess_lst)), 2) * 100
            print(f'Accuracy: {acc}')
        else:
            acc = 0
        print(f'Predictions: {guess_lst}')
        tally = stock + wallet - sum(fees_lst)
        percent_change = round((y_true_lst[-1] - stat_invest) / stat_invest * 100, 2)
        percent_profit = round(tally / stat_invest * 100, 2)
        amount_change = round(y_true_lst[-1] - stat_invest, 2)
        adjusted_profit = round(tally - amount_change, 2)
        percent_adj_profit = round(percent_profit - percent_change, 2)
        stats = {'Features': int(num_features), 'Actual Days Up': int(ones), 'Actual Days Down': int(zeros),
        'Pred Days Up Agreed': 0, 'Pred Days Down Agreed': 0, 'Stock Start Value $': start_value, 'Total Investment $': stat_invest,
        'Profit $': float(tally), 'Stock Change %': float(percent_change), 'Profit %': float(percent_profit),
        'Transaction Accuracy %': acc, 'Total Fees $': round(sum(fees_lst), 2),
        'Net Profit $': round(float(tally) - sum(fees_lst), 2), 'Adjusted Profit $': adjusted_profit, 
        'Adjusted Profit %': percent_adj_profit, 'Trade Count': int(trade_count), 'Candlestick Count:': int(time_count)}
        with open(f'{os.getcwd()}/Data/stats_{finish.replace(" ", "-")}.json', 'w') as f:
            f.write(json.dumps(stats, indent=4))
        
        print(warning)
        print()
        print(stats)
    

    time_now = dt.datetime.now()
    minutes = (time_now.minute // trade_frequency * trade_frequency) + trade_frequency
    if minutes == 60 or minutes == 240:
        minutes = 0

    one_min = [i for i in range(60)]
    five_mins = [i for i in range(0, 60, 5)]
    fifteen_mins = [i for i in range(0, 60, 15)]
    thirty_mins = [0, 30]
    sixty_mins = 0
    twoforty_mins = 0
    twoforty_hours = [1, 5, 9, 13, 17, 21]

    if trade_frequency == 1:
        interval = []
    elif trade_frequency == 5:
        interval = five_mins
    elif trade_frequency == 15:
        interval = fifteen_mins
    elif trade_frequency == 30:
        interval = thirty_mins
    elif trade_frequency == 60:
        interval = sixty_mins
    elif trade_frequency == 240:
        interval = twoforty_mins

    while True:
        if trade_frequency == 240:
            if dt.datetime.now().minute == minutes and dt.datetime.now().hour in twoforty_hours:
                print(f'Trading time interval reached, commencing trading in {trade_frequency} minutes...')
                break
        elif dt.datetime.now().minute == minutes:
            print(f'Trading time interval reached, commencing trading in {trade_frequency} minutes...')
            if trade_frequency == 1:
                time.sleep(15)
            break
        else:
            continue

    while True:
        if trade_frequency == 1:
            if dt.datetime.now().second == 0:
                go(client=client, ticker=ticker, features=features,num_features=num_features,
                drop_features=drop_features, delta=delta, scaler=scaler, crypto=crypto, training=training,
                trade_frequency=trade_frequency, investment=investment)
            else:
                continue
        elif trade_frequency == 240:
            if dt.datetime.now().minute == minutes and dt.datetime.now().hour in twoforty_hours:
                print(dt.datetime.now())
                go(client=client, ticker=ticker, features=features,num_features=num_features,
                drop_features=drop_features, delta=delta, scaler=scaler, crypto=crypto, training=training,
                trade_frequency=trade_frequency, investment=investment)
                time.sleep(60)
            else:
                continue
        elif dt.datetime.now().minute in interval:
            print(dt.datetime.now())
            go(client=client, ticker=ticker, features=features,num_features=num_features,
            drop_features=drop_features, delta=delta, scaler=scaler, crypto=crypto, training=training,
            trade_frequency=trade_frequency, investment=investment)
            time.sleep(60)
            'interval sleep over'
        else:
            continue

    # schedule.every(trade_frequency).minutes.do(go, client=client, ticker=ticker, features=features,num_features=num_features,
    # drop_features=drop_features, delta=delta, scaler=scaler, crypto=crypto, training=training, trade_frequency=trade_frequency,
    # investment=investment)


    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)