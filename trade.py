import pandas as pd
import numpy as np
import json
import random

def is_jsonable(x):
	try:
	    json.dumps(x)
	    return True
	except:
	    return False

def buy_signal(pred, lstm_pred, bought, wallet, num_shares, trade_shares, y_true_today, y_true_tomorrow, close_today,
to_predict, crypto):
	fees = 0
	if bought:
		action = 2
		trades = False
		trans_acc = 2
	elif to_predict == 'Close':
		bought = True
		wallet -= y_true_today * trade_shares
		num_shares += trade_shares
		action = 1
		trades = True
		if y_true_tomorrow > y_true_today:
			trans_acc = 1
		else:
			trans_acc = 0
		if crypto:
		    fees = trade_shares * y_true_today * .001
	elif to_predict == 'Open':
		bought = True
		wallet -= close_today * trade_shares
		num_shares += trade_shares
		action = 1
		trades = True
		if y_true_tomorrow > close_today:
			trans_acc = 1
		else:
			trans_acc = 0
		if crypto:
		    fees = trade_shares * y_true_today * .001
	
	return bought, wallet, num_shares, action, trades, fees, trans_acc

def sell_signal(pred, lstm_pred, bought, wallet, num_shares, trade_shares, y_true_today, y_true_tomorrow, close_today,
to_predict, crypto):
	fees = 0
	if bought:
		trades = True
		bought = False
		wallet += y_true_today * trade_shares
		if y_true_tomorrow < y_true_today:
			trans_acc = 1
		else:
			trans_acc = 0
		num_shares -= trade_shares
		action = 0
		if crypto:
		    fees = trade_shares * y_true_today * .001
	else:
		action = 2
		trades = False
		trans_acc = 2
	
	return bought, wallet, num_shares, action, trades, fees, trans_acc

def historical(close_values, open_values, pred_df, dates, ticker, delta, num_features, metrics, crypto, num_shares,
lower, upper, to_predict):
	begin_close = close_values[0]
	begin_open = open_values[0]
	if to_predict == 'Open':
		begin_value = begin_open
	elif to_predict == 'Close':
		begin_value = begin_close
	trade_shares = num_shares
	wallet = -num_shares * begin_value
	stock = -1 * wallet
	y_pred = list(pred_df['y_pred'])
	lstm_pred_lst = list(pred_df['lstm_pred'])
	investment = stock
	daily_worth = []
	transaction_accuracy = []
	fees_lst = []
	trades_lst = []
	combined_up_down = []

	if num_shares:
		bought = True
	else:
		bought = False
	for day in range(len(dates) - 1):
		pred = y_pred[day]
		lstm_pred = lstm_pred_lst[day]
		if to_predict == 'Open':
			y_true_tomorrow = float(open_values[day + 1])
			y_true_today = float(open_values[day])
			close_today = float(close_values[day])
			open_today = float(open_values[day])
			open_tomorrow = float(open_values[day+1])
		elif to_predict == 'Close':
			y_true_tomorrow = float(close_values[day + 1])
			y_true_today = float(close_values[day])
			close_today = float(close_values[day])
			open_today = float(open_values[day])
		avg_pred = (pred + lstm_pred) / 2
		if avg_pred >= .5:
			combined_up_down.append(1)
		else:
			combined_up_down.append(0)

		if pred >= upper:
			bought, wallet, num_shares, action, trades, fees, trans_acc = buy_signal(pred, lstm_pred, bought, wallet,
			num_shares, trade_shares, y_true_today, y_true_tomorrow, close_today, to_predict, crypto)
			if trans_acc == 0 or trans_acc == 1:
				transaction_accuracy.append(trans_acc)
			trades_lst.append(int(trades))
			fees_lst.append(fees)
			
		# if pred >= .54:
		# 	if bought:
		# 		action = 2
		# 	else:
		# 		bought = True
		# 		wallet -=  y_true_today * trade_shares
		# 		num_shares += trade_shares
		# 		action = 1
		# 		trades += 1
		# 		if crypto:
		# 		    fees += trade_shares * y_true_today * .001
		# 	if y_true_tomorrow > y_true_today:
		# 		combined_accuracy.append(1)
		# 	else:
		# 		combined_accuracy.append(0)
		# elif pred and lstm_pred < .5 or (abs(pred - lstm_pred) >= .1 and (pred or lstm_pred < .45)):
		# elif pred < .5 and lstm_pred < .5:
		elif pred < lower and to_predict == 'Close':
			bought, wallet, num_shares, action, trades, fees, trans_acc = sell_signal(pred, lstm_pred, bought, wallet,
			num_shares, trade_shares, y_true_today, y_true_tomorrow, close_today, to_predict, crypto)
			if trans_acc == 0 or trans_acc == 1:
				transaction_accuracy.append(trans_acc)
			trades_lst.append(int(trades))
			fees_lst.append(fees)
			# if bought:
			# 	bought = False
			# 	wallet += y_true_today * trade_shares
			# 	num_shares -= trade_shares
			# 	action = 0
			# 	if crypto:
			# 	    fees += trade_shares * y_true_today * .001
			# else:
			# 	action = 2
			# if y_true_tomorrow < y_true_today:
			# 	combined_accuracy.append(1)
			# else:
			# 	combined_accuracy.append(0)
		else:
			# print("Prediction not accurate enough. Nothing to do.")
			action = 2
			fees = 0
		# if (pred == 0 and lstm_pred > .53) or (pred == 1 and lstm_pred < .47):
		# 	pass
		# elif pred == 1 and lstm_pred >= .53:
		# 	if y_true_tomorrow > y_true_today:
		# 		combined_accuracy.append(1)
		# 	else:
		# 		combined_accuracy.append(0)
		# 	if not bought:
		# 		wallet -=  y_true_today * trade_shares
		# 		num_shares += trade_shares
		# 		bought = True
		# 		trades += 1
		# 		if crypto:
		# 			fees += trade_shares * y_true_today * .001
		# 	else:
		# 		pass
		# elif pred == 0 and lstm_pred <= .47:
		# 	if y_true_tomorrow < y_true_today:
		# 		combined_accuracy.append(1)
		# 	else:
		# 		combined_accuracy.append(0)
		# 	if bought:
		# 		wallet += y_true_today * trade_shares
		# 		num_shares -= trade_shares
		# 		bought = False
		# 		trades += 1
		# 		if crypto:
		# 			fees += trade_shares * y_true_today * .001
		# else:
		# 	combined_accuracy.append(1)
		if to_predict == 'Close':
			stock = y_true_today * num_shares
		elif to_predict == 'Open':
			if bought:
				stock = open_tomorrow * num_shares
				bought = False
				wallet += stock * num_shares
				stock = 0
				if num_shares > 0:
					num_shares -= trade_shares
		day_wor = stock + wallet - fees
		daily_worth.append(day_wor)
		# print(stock, wallet, num_shares, day_wor)
	if to_predict == 'Close':
		stock = y_true_tomorrow * num_shares
	ones_pred = combined_up_down.count(1)
	zeros_pred = combined_up_down.count(0)
	totals_actual = pred_df['y_test'].value_counts()
	try:
		ones = totals_actual[0]
	except:
		ones = 0
	try:
		zeros = totals_actual[1]
	except:
		zeros = 0
	xg_pred_up_binary = [1 for j in y_pred if j >= .5]
	xg_pred_down_binary = [0 for j in y_pred if j < .5]
	xg_pred_up = xg_pred_up_binary.count(1)
	xg_pred_down = xg_pred_down_binary.count(0)
	# pd.set_option('display.max_rows', 1500)
	# print(pred_df['y_pred'])
	# print()
	# print(totals_pred)
	# print()
	# print(f'Wallet value: ${round(wallet, 2):,}, stock value: ${round(stock, 2):,}')
	# print()
	tally = round(wallet + stock, 2)
	# print(f'Stock value on first day: ${round(begin_value, 2):,}, stock value on last day: ${round(values[-1], 2):,}')
	if to_predict == 'Open':
		end_value = open_values[-1]
	elif to_predict == 'Close':
		end_value = close_values[-1]
	percent_change = round((end_value - begin_value) / begin_value * 100, 2)
	# print(f'Stock changed {percent_change}%')
	# print()
	percent_profit = round(tally / investment * 100, 2)
	# print(f'Profit: {round(percent_profit, 2)}%')
	net_profit = tally - sum(fees_lst)
	try:
		transaction_acc_total = sum(transaction_accuracy) / len(transaction_accuracy)
	except:
		transaction_acc_total = 0
	days = int(delta - 60)
	results = {'Ticker': ticker, 'Days': days, 'Features': int(num_features), 'Actual Days Up': int(ones),
	'Actual Days Down': int(zeros), 'XG Pred Days Up': int(xg_pred_up), 'XG Pred Days Down': int(xg_pred_down), 'Total Investment $': round(investment, 2),
	'Profit $': float(tally), 'Stock Change %': float(percent_change), 'Profit %': float(percent_profit),
	'Start Value $': round(begin_value, 2), 'End Value $': round(end_value, 2), 'XG Accuracy %': float(round(metrics[0] * 100, 2)),
	'Transaction Accuracy %': round(transaction_acc_total * 100, 2), 'Fees $': round(sum(fees_lst), 2),
	'Net Profit $': round(net_profit, 2), 'Trade Count:': sum(trades_lst)}
	print()
	keys_to_delete = []
	for key,value in results.items():
		if not is_jsonable(value):
			keys_to_delete.append(key)
	if len(keys_to_delete) >= 1:
		print(keys_to_delete)
	return results, daily_worth