import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# try using these macd and change in macd pairs: [10, 30], [5, 10], [2, 10]

# try using RSI over 5, 10, 30, 60 days

# Average daily/weekly/monthly returns over last 5, 10, 30, 60 days

# Ratio of avg close price over last 2 days and current close price

# mean daily returns last 5, 10, 30, 60 days

# loop for selecting number of features

# grid search for xg boost parameters

# change test plot so that the graph is at the end and has all the predicted days

# change graph x axis to dates

# can I get business financial metrics???

# df = pd.read_csv('C:/Users/sjmit/practice/lighthouse/stocks/xg_boost/Data/all_day_values.csv', index_col=0)
# df2 = pd.read_csv('C:/Users/sjmit/practice/lighthouse/stocks/xg_boost/Data/daily_worth_df.csv', index_col=0)

pd.set_option("display.max_rows", None, "display.max_columns", None)
with open('C:/Users/sjmit/practice/lighthouse/stocks/xg_boost/Data/combine_May-23-2019.json') as f:
    data = json.load(f)
days_up = sum([x[0]['Actual Days Up'] for x in data])
days_down = sum([x[0]['Actual Days Down'] for x in data])
pred_days_up = sum([x[0]['XG Pred Days Up'] for x in data])
pred_days_down = sum([x[0]['XG Pred Days Down'] for x in data])
total_invest = round(sum([x[0]['Total Investment $'] for x in data]), 2)
avg_invest = round(total_invest / len(data), 2)
total_profit = round(sum([x[0]['Profit $'] for x in data]), 2)
avg_profit = round(total_profit / len(data), 2)
avg_stock_change = round(sum([x[0]['Stock Change %'] for x in data]) / len(data), 2)
avg_profit_percent = round(sum([x[0]['Profit %'] for x in data]) / len(data), 2)
avg_profit_med_percent = round(sum([x[1]['Profit %'] for x in data]) / len(data), 2)
best_case_percent = round(sum([max([x[0]['Profit %'], x[1]['Profit %']]) for x in data]) / len(data), 2)
avg_accuracy = round(sum([x[0]['XG Accuracy %'] for x in data]) / len(data), 2)
# avg_precision = round(sum([x['Precision %'] for x in data]) / len(data), 2)
# avg_recall = round(sum([x['Recall %'] for x in data]) / len(data), 2)
# avg_metric = round(sum([avg_accuracy, avg_precision, avg_recall]) / 3, 2)
num_stock_losses = round(len([x[0]['Profit $'] for x in data if x[0]['Profit $'] < 0]), 2)
total_losses = round(sum([x[0]['Profit $'] for x in data if x[0]['Profit $'] < 0]), 2)
max_profit = round(max([x[0]['Profit $'] for x in data if x[0]['Profit $']]), 2)
min_profit = round(min([x[0]['Profit $'] for x in data if x[0]['Profit $']]), 2)
avg_train_acc = round(sum([x[0]['Train Accuracy %'] for x in data]) / len(data), 2)
df = pd.DataFrame({'Ticker': [x[0]['Ticker'] for x in data], 'Profit': [x[0]['Profit %'] for x in data],
'XG Accuracy': [x[0]['XG Accuracy %'] for x in data],
'Transaction Accuracy %': [x[0]['Transaction Accuracy %'] for x in data],
'Stock Change %': [x[0]['Stock Change %'] for x in data], 'Train Accuracy %': [x[0]['Train Accuracy %'] for x in data]})
print('Total Days Up:', days_up)
print('Total Days Down:', days_down)
print('Pred Days Up:', pred_days_up)
print('Pred Days Down:', pred_days_down)
print('Total Investment $:', total_invest)
print('Avg Investment $:', avg_invest)
print('Total Profit $:', total_profit)
print('Avg Profit $:', avg_profit)
print('Avg Stock Change %:', avg_stock_change)
print('Avg Profit %:', avg_profit_percent)
print('Avg Profit Median %', avg_profit_med_percent)
print('Best Case Scenario %', best_case_percent)
print('Avg Accuracy %:', avg_accuracy)
print('Avg Train Accuracy %', avg_train_acc)
# print('Avg Precision %:', avg_precision)
# print('Avg Recall %:', avg_recall)
# print('Avg Metric %:', avg_metric)
print('Total stock losses:', num_stock_losses)
print('Total stock losses $:', total_losses)
print('Max profit $:', max_profit)
print('Min profit $:', min_profit)


top_picks = df.sort_values('Train Accuracy %', ascending=False).iloc[:5, :]
top_tickers = top_picks['Ticker']
top_tickers = top_tickers.tolist()
print()
print(top_picks)
print()

days_up = sum([x[0]['Actual Days Up'] for x in data if x[0]['Ticker'] in top_tickers])
days_down = sum([x[0]['Actual Days Down'] for x in data if x[0]['Ticker'] in top_tickers])
pred_days_up = sum([x[0]['XG Pred Days Up'] for x in data if x[0]['Ticker'] in top_tickers])
pred_days_down = sum([x[0]['XG Pred Days Down'] for x in data if x[0]['Ticker'] in top_tickers])
total_invest = round(sum([x[0]['Total Investment $'] for x in data if x[0]['Ticker'] in top_tickers]), 2)
avg_invest = round(total_invest / len(top_tickers), 2)
total_profit = round(sum([x[0]['Profit $'] for x in data if x[0]['Ticker'] in top_tickers]), 2)
avg_profit = round(total_profit / len(top_tickers), 2)
avg_stock_change = round(sum([x[0]['Stock Change %'] for x in data if x[0]['Ticker'] in top_tickers]) / len(top_tickers), 2)
avg_profit_percent = round(sum([x[0]['Profit %'] for x in data if x[0]['Ticker'] in top_tickers]) / len(top_tickers), 2)
avg_profit_med_percent = round(sum([x[1]['Profit %'] for x in data if x[0]['Ticker'] in top_tickers]) / len(top_tickers), 2)
best_case_percent = round(sum([max([x[0]['Profit %'], x[1]['Profit %']]) for x in data if x[0]['Ticker'] in top_tickers]) / len(top_tickers), 2)
avg_accuracy = round(sum([x[0]['XG Accuracy %'] for x in data if x[0]['Ticker'] in top_tickers]) / len(top_tickers), 2)
avg_train_acc = round(sum([x[0]['Train Accuracy %'] for x in data if x[0]['Ticker'] in top_tickers]) / len(top_tickers), 2)
# avg_precision = round(sum([x['Precision %'] for x in data]) / len(data), 2)
# avg_recall = round(sum([x['Recall %'] for x in data]) / len(data), 2)
# avg_metric = round(sum([avg_accuracy, avg_precision, avg_recall]) / 3, 2)
num_stock_losses = round(len([x[0]['Profit $'] for x in data if x[0]['Profit $'] < 0]), 2)
total_losses = round(sum([x[0]['Profit $'] for x in data if x[0]['Profit $'] < 0]), 2)
max_profit = round(max([x[0]['Profit $'] for x in data if x[0]['Profit $']]), 2)
min_profit = round(min([x[0]['Profit $'] for x in data if x[0]['Profit $']]), 2)

print('Total Days Up:', days_up)
print('Total Days Down:', days_down)
print('Pred Days Up:', pred_days_up)
print('Pred Days Down:', pred_days_down)
print('Total Investment $:', total_invest)
print('Avg Investment $:', avg_invest)
print('Total Profit $:', total_profit)
print('Avg Profit $:', avg_profit)
print('Avg Stock Change %:', avg_stock_change)
print('Avg Profit %:', avg_profit_percent)
print('Avg Profit Median %', avg_profit_med_percent)
print('Best Case Scenario %', best_case_percent)
print('Avg Accuracy %:', avg_accuracy)
print('Avg Train Accuracy %:', avg_train_acc)
# print('Avg Precision %:', avg_precision)
# print('Avg Recall %:', avg_recall)
# print('Avg Metric %:', avg_metric)
print('Total stock losses:', num_stock_losses)
print('Total stock losses $:', total_losses)
print('Max profit $:', max_profit)
print('Min profit $:', min_profit)

# with open('C:/Users/sjmit/practice/lighthouse/stocks/xg_boost/Data/stats_May-5-2021.json') as f:
#     data = json.load(f)

# print(data)