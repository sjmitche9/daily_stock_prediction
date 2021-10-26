from binance.client import Client
from binance.enums import *
import datetime as dt
import pandas as pd
import numpy as np
import Get_data, os
import Prepare_data as pdat

predictions = []

def order(side, quantity, symbol, client, order_type=ORDER_TYPE_MARKET):
    try:
        print("Sending order")
        print()
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=quantity)
        # print(order)
    except Exception as e:
        print("An exception occured - {}".format(e))
        return False
    return True 
def close_function(client, ticker, features, num_features, drop_features, delta, scaler, crypto, training, trade_frequency,
to_predict, Prepare, scale, live):
    # print(f'this is the scale object: {scale}')
    intervals = {1: Client.KLINE_INTERVAL_1MINUTE, 5: Client.KLINE_INTERVAL_5MINUTE, 15: Client.KLINE_INTERVAL_15MINUTE,
    30: Client.KLINE_INTERVAL_30MINUTE, 60: Client.KLINE_INTERVAL_1HOUR, 240: Client.KLINE_INTERVAL_4HOUR, 720: Client.KLINE_INTERVAL_12HOUR, 
    1440: Client.KLINE_INTERVAL_1DAY}
    klines = client.get_historical_klines(ticker, intervals[trade_frequency], f'{800 * trade_frequency} minutes ago PDT')
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    for i in range(len(klines)):
        opens.append(float(klines[i][1]))
        highs.append(float(klines[i][2]))
        lows.append(float(klines[i][3]))
        closes.append(float(klines[i][4]))
        volumes.append(float(klines[i][5]))
    close_time = dt.datetime.fromtimestamp(klines[-1][6] / 1000)
    working_df = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes, 'Volume': volumes}, index=range(len(opens)))
    open_close = working_df[to_predict].iloc[-1]
    print(working_df.iloc[-1, :], 'at', close_time)
    print()
    X = Get_data.get_all(working_df, ticker)
    # X = X.replace([np.inf, -np.inf], np.nan)
    X = X.iloc[600:, :]
    nulls = X.isnull().sum().values
    pd.set_option('display.max_rows', 150)
    pd.set_option('display.max_columns', 150)
    if sum(nulls) != 0:
        warning = f'WARNING: {sum(nulls)} null values in {len(nulls)} columns'
        print(warning)
    X = X.fillna(0)
    X.to_csv(f'{os.getcwd()}/Data/{ticker}_live.csv')
    Prepare = pdat.Prepare_data(ticker, features, num_features, drop_features, delta, scaler, crypto, training, to_predict, scale, live=True)
    prepped = Prepare.preprocessed_test
    prepped = prepped[features[:num_features]]
    prepped = np.array(prepped)
    last = prepped[-1, :]
    last = np.array([last])
    last = np.reshape(last, (1, last.shape[1]))
    print(f'PRINTING LAST: {last}')
    return last, working_df, open_close, warning

def trade(y_pred, shares, in_position, ticker, client, wallet, num_shares, to_predict, crypto, lower, upper, lstm_pred=0):
    trade_shares = shares
    fees = 0
    predictions.append(y_pred)
    print(f'Predictions list: {predictions}')
    print()
    if num_shares >= trade_shares:
        in_position = True
    else:
        in_position = False
    print(y_pred)
    if y_pred > upper:
        if in_position:
            print("Time to buy but we already own.")
            print()
            action = 2
        else:
            print("We don't own, BUY!")
            print()
            # order_succeeded = order(Client.SIDE_BUY, quantity=shares, symbol=ticker, client=client)
            order_succeeded = True
            if order_succeeded:
                in_position = True
                wallet -=  to_predict * trade_shares
                num_shares += trade_shares
                action = 1
                if crypto:
                    fees = trade_shares * to_predict * .001
    # elif y_pred[0] and lstm_pred < .5 or (abs(y_pred[0] - lstm_pred) >= .1 and (y_pred[0] or lstm_pred < .45)):
    elif y_pred < lower:
        if in_position:
            print("Sell! Sell! Sell!")
            print()
            # order_succeeded = order(Client.SIDE_SELL, quantity=shares, symbol=ticker, client=client)
            order_succeeded = True
            if order_succeeded:
                in_position = False
                wallet += to_predict * trade_shares
                num_shares -= trade_shares
                action = 0
                if crypto:
                    fees = trade_shares * to_predict * .001
        else:
            print("Time to sell but we don't own.")
            print()
            action = 2
    else:
        print("Prediction not accurate enough. Nothing to do.")
        action = 2
        # else:
        # 	print("Time to sell but we don't own. Nothing to do.")
        # 	print()
        # 	action = 2
        # action = 2
        # return in_position, num_shares, to_predict, wallet, fees, action
    # elif y_pred[0] == 1 or lstm_pred >= .53:
        # if in_position:
        # 	print("Time to buy but we already own.")
        # 	print()
        # 	action = 2
        # else:
        # 	print("We don't own, BUY!")
        # 	print()
        # 	# order_succeeded = order(Client.SIDE_BUY, quantity=shares, symbol=ticker, client=client)
        # 	order_succeeded = True
        # 	if order_succeeded:
        # 		in_position = True
        # 		wallet -=  to_predict * trade_shares
        # 		num_shares += trade_shares
        # 		action = 1
        # 		if crypto:
        # 		    fees = trade_shares * to_predict * .001
    # elif y_pred[0] == 0 or lstm_pred <= .47:
        # if in_position:
        # 	print("Sell! Sell! Sell!")
        # 	print()
        # 	# order_succeeded = order(Client.SIDE_SELL, quantity=shares, symbol=ticker, client=client)
        # 	order_succeeded = True
        # 	if order_succeeded:
        # 		in_position = False
        # 		wallet += to_predict * trade_shares
        # 		num_shares -= trade_shares
        # 		action = 0
        # 		if crypto:
        # 		    fees = trade_shares * to_predict * .001
        # else:
        # 	print("Time to sell but we don't own. Nothing to do.")
        # 	print()
        # 	action = 2
    return in_position, num_shares, to_predict, wallet, fees, action