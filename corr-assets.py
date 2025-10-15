# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 18:09:30 2025

@author: quant
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz

import statsmodels.api as sm
import xgboost
import pmdarima as pm

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# Core-Requisites
# =============================================================================

calendar = get_calendar("NYSE")
all_trading_dates = calendar.schedule(start_date = "2001-01-01", end_date = datetime.now(pytz.timezone("America/New_York"))+timedelta(days=45)).index.strftime("%Y-%m-%d").values

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"

# =============================================================================
# L/S Data
# =============================================================================

# trading_dates = calendar.schedule(start_date = "2025-01-01", end_date = (datetime.today()-timedelta(days = 1))).index.strftime("%Y-%m-%d").values
trading_dates = calendar.schedule(start_date = "2025-02-01", end_date = "2025-06-01").index.strftime("%Y-%m-%d").values
lookback_date = np.sort(all_trading_dates[all_trading_dates < trading_dates[0]])[-252]

tickers = np.array([["RGTI", "IONQ"], ["RIOT", "MARA"], ["OKLO","SMR"], ["CVNA", "KMX"]])

target_pnl = 1

# Keep 1 session active (same headers, etc.)
connection_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=500)
connection_session.mount('https://', adapter)

notional = 10000

all_trades_list = []

# ticker_set = tickers[0]
for ticker_set in tickers:
    try:
        
        long_ticker = ticker_set[0]
        short_ticker = ticker_set[1]
        
        long_ticker_data = pd.json_normalize(connection_session.get(f"https://api.polygon.io/v2/aggs/ticker/{long_ticker}/range/1/day/{lookback_date}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        long_ticker_data.index = pd.to_datetime(long_ticker_data.index, unit="ms", utc=True).tz_convert("America/New_York")    
        long_ticker_data["date"] = long_ticker_data.index.strftime("%Y-%m-%d")
        long_ticker_data["daily_return"] = round(long_ticker_data["c"].pct_change()*100, 2)
        
        long_ticker_data["log_ret_sq"] = np.log(long_ticker_data["c"] / long_ticker_data["c"].shift(1)) ** 2 
        long_ticker_data["var"] = long_ticker_data["log_ret_sq"].rolling(window=20, closed = "left").sum()
        long_ticker_data["vol"] = round((np.sqrt(long_ticker_data["var"]) / np.sqrt(252/20))*100, 2)
        
        short_ticker_data = pd.json_normalize(connection_session.get(f"https://api.polygon.io/v2/aggs/ticker/{short_ticker}/range/1/day/{lookback_date}/{trading_dates[-1]}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        short_ticker_data.index = pd.to_datetime(short_ticker_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        short_ticker_data["date"] = short_ticker_data.index.strftime("%Y-%m-%d")
        short_ticker_data["daily_return"] = round(short_ticker_data["c"].pct_change()*100, 2)
        
        short_ticker_data["log_ret_sq"] = np.log(short_ticker_data["c"] / short_ticker_data["c"].shift(1)) ** 2 
        short_ticker_data["var"] = short_ticker_data["log_ret_sq"].rolling(window=20, closed = "left").sum()
        short_ticker_data["vol"] = round((np.sqrt(short_ticker_data["var"]) / np.sqrt(252/20))*100, 2)
        
        combined_daily_data = pd.merge(long_ticker_data[["date", "c","daily_return", "vol"]], short_ticker_data[["date","c", "daily_return", "vol"]], on="date")
        combined_daily_data["rolling_corr"] = combined_daily_data["daily_return_x"].rolling(window=5, closed = "left").corr(combined_daily_data["daily_return_y"])
        
        combined_daily_data["long_wt"] = round(combined_daily_data["vol_y"] / combined_daily_data["vol_x"], 2)
        
        corr_data = combined_daily_data.copy()
        
        # high_corr = corr_data[corr_data["rolling_corr"] >= .8].copy()
        
        # =============================================================================
        # Modeling
        # =============================================================================
        
        intraday_data_list = []
        
        # date = trading_dates[0]
        for date in trading_dates:
            
            try:
            
                date_info = corr_data[corr_data["date"] == date].copy()
                
                long_ticker_data = pd.json_normalize(connection_session.get(f"https://api.polygon.io/v2/aggs/ticker/{long_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                long_ticker_data.index = pd.to_datetime(long_ticker_data.index, unit="ms", utc=True).tz_convert("America/New_York")    
                long_ticker_data["date"] = long_ticker_data.index.strftime("%Y-%m-%d")
                
                long_ticker_data = long_ticker_data[(long_ticker_data.index.time >= pd.Timestamp("09:30").time()) & (long_ticker_data.index.time <= pd.Timestamp("16:00").time())].copy().sort_index()    
                long_ticker_data["intraday_returns"] = round(((long_ticker_data["c"] - long_ticker_data["c"].iloc[0]) / long_ticker_data["c"].iloc[0])*100, 2)
            
                short_ticker_data = pd.json_normalize(connection_session.get(f"https://api.polygon.io/v2/aggs/ticker/{short_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                short_ticker_data.index = pd.to_datetime(short_ticker_data.index, unit="ms", utc=True).tz_convert("America/New_York")
                short_ticker_data["date"] = short_ticker_data.index.strftime("%Y-%m-%d")
                
                short_ticker_data = short_ticker_data[(short_ticker_data.index.time >= pd.Timestamp("09:30").time()) & (short_ticker_data.index.time <= pd.Timestamp("16:00").time())].copy().sort_index()    
                short_ticker_data["intraday_returns"] = round(((short_ticker_data["c"] - short_ticker_data["c"].iloc[0]) / short_ticker_data["c"].iloc[0])*100, 2)
            
                combined_intraday_data = pd.merge(long_ticker_data[["date", "intraday_returns"]], short_ticker_data[["intraday_returns"]], left_index=True, right_index=True)    
                
                combined_intraday_data["long_wt"] = date_info["long_wt"].iloc[0]
                
                combined_intraday_data["adj_long_returns"] = combined_intraday_data["intraday_returns_x"] * combined_intraday_data["long_wt"]
                
                combined_intraday_data["gross_return"] = combined_intraday_data["adj_long_returns"] - combined_intraday_data["intraday_returns_y"]
                
                intraday_data_list.append(combined_intraday_data)
                
            except Exception as error:
                print(error)
                continue
        
        full_dataset = pd.concat(intraday_data_list)
        
        # =============================================================================
        # Trade Simulation
        # =============================================================================
        
        dates_in_coverage = np.sort(full_dataset["date"].drop_duplicates().values)
        
        trade_list = []
        
        # trade_date = dates_in_coverage[0]
        for trade_date in dates_in_coverage:
            
            daily_trade_info = full_dataset[full_dataset["date"] == trade_date].copy().sort_index()
            
            signals = daily_trade_info[daily_trade_info["gross_return"] >= target_pnl].sort_index().copy()
            
            if len(signals) < 1:
                
                trade = pd.DataFrame([{"date": trade_date, "return": daily_trade_info["gross_return"].iloc[-1],
                                       "signal": 0}])
                
                trade = pd.merge(daily_trade_info.tail(1), trade, on="date")
            else:
                
                trade = pd.DataFrame([{"date": trade_date, "return": signals["gross_return"].iloc[0],
                                       "signal": 1}])
                
                trade = pd.merge(daily_trade_info.tail(1), trade, on="date")
                
            trade_list.append(trade)
            
        all_trades = pd.concat(trade_list)
        all_trades["long_ticker"] = long_ticker
        all_trades["short_ticker"] = short_ticker
        
        all_trades["dollar_pnl"] = ((all_trades["gross_return"]/100) * notional)
        
        all_trades["return_on_size"] = round((all_trades["dollar_pnl"] / notional)*100, 2)
        all_trades["rolling_std"] = all_trades["return_on_size"].rolling(window=5).std()
        all_trades["capital"] = notional + all_trades["dollar_pnl"].cumsum()
        
        all_trades["strategy_return"] = round(((all_trades["capital"] - notional) / notional)*100, 2)
        all_trades["strategy_sharpe"] = round(all_trades["strategy_return"] / (all_trades["rolling_std"] * np.sqrt(5)), 2)
        
        all_trades_list.append(all_trades)
        
    except Exception as error:
        print(error)
        continue
    
dates_to_model = dates_in_coverage[253:]

prediction_data_list = []

# date = dates_to_model[-15]
for date in dates_to_model:
    
    try:
        
        historical_data = full_dataset[full_dataset["date"] < date].copy().reset_index(drop=True).tail(252)

        # Create an array like you would in R
        X = historical_data["error"]
        
        # Compute an auto-correlation like you would in R:
        # pm.acf(X)
        
        # # Plot an auto-correlation:
        # pm.plot_acf(X)
        
        stepwise_fit = pm.auto_arima(X, stepwise=True)
        stepwise_fit.summary()
        
        forecast = stepwise_fit.predict(n_periods = 1).iloc[0]
        
        # X = historical_data[["error"]].values
        # Y = historical_data[["next_error"]].values
        
        # Model = RandomForestRegressor(n_estimators=1000).fit(X, Y)
        
        # Model = sm.tsa.ARIMA(X, order=(1, 0, 0)).fit()      
        
        # Make a one-step-ahead forecast
        # forecast = Model.forecast(steps=1)
        
        oos_data = full_dataset[full_dataset["date"] == date].copy()
        
        # X_oos = oos_data[["error"]].values
        
        # forecast = Model.predict(X_oos)[0]
        
        # actual = full_dataset["error"].iloc[0]
        actual = oos_data["error"].iloc[0]
        
        # prediction_data = pd.DataFrame([{"date": date, "pred": forecast.iloc[0], "actual": actual}])
        prediction_data = pd.DataFrame([{"date": date, "pred": forecast, "actual": actual}])
        
        oos_data = pd.merge(oos_data, prediction_data, on="date")
    
        prediction_data_list.append(oos_data)
        
    except Exception as error:
        print(error)
        continue

# =============================================================================
# Pnl + Ev Calcs
# =============================================================================

all_trades = pd.concat(prediction_data_list)

notional = 10000

costs = notional * .0025

all_trades["base_pnl"] = (notional * (all_trades["error"]/100)) - costs
all_trades["model_pnl"] = all_trades.apply(lambda x: x["base_pnl"] if x["pred"] > 0 else 0, axis = 1)

all_trades["return_on_size"] = round((all_trades["model_pnl"] / notional)*100, 2)

all_trades["base_capital"] = notional + (all_trades["base_pnl"]).cumsum()
all_trades["model_capital"] = notional + (all_trades["model_pnl"]).cumsum()

plt.figure(dpi=200)
plt.xticks(rotation = 45)
plt.suptitle(f"Long: {long_ticker}, Short: {short_ticker}")
plt.plot(np.arange(0, len(all_trades)), all_trades["base_capital"].values)
plt.plot(np.arange(0, len(all_trades)), all_trades["model_capital"].values)
plt.legend(["baseline", "model"])
plt.show()
plt.close()

active_trades = all_trades[all_trades["model_pnl"] != 0].copy()

len(active_trades) / len(all_trades)

len(active_trades[active_trades["model_pnl"] > 0]) / len(active_trades)

# =============================================================================
# EV / Risk Calcs
# =============================================================================

pnl_choice = "model_pnl"
capital_choice = "model_capital"

portfolio_size = 10000

all_trades["rolling_5d_pnl"] = all_trades[pnl_choice].rolling(window=5, closed="left", min_periods = 1).sum()
all_trades["rolling_30d_pnl"] = all_trades[pnl_choice].rolling(window=20, closed="left", min_periods=1).sum()

worst_5d_drawdown = round((all_trades["rolling_5d_pnl"].min() / portfolio_size)*100, 2)
worst_30d_drawdown = round((all_trades["rolling_30d_pnl"].min() / portfolio_size)*100, 2)

median_5d_pnl = all_trades["rolling_5d_pnl"].median()
median_30d_pnl = all_trades["rolling_30d_pnl"].median()

active_trades = all_trades[all_trades[pnl_choice] != 0].copy()

wins = active_trades[active_trades[pnl_choice] > 0].copy()
losses = active_trades[active_trades[pnl_choice] < 0].copy()

worst_loss = round((losses["return_on_size"]).min())
best_win = round((wins["return_on_size"]).max())

avg_win = round((wins["return_on_size"]).mean(), 2)
avg_loss = round((losses["return_on_size"]).mean(), 2)

win_rate = len(wins) / len(active_trades)
loss_rate = len(losses) / len(active_trades)

theo_ev_per_trade = (win_rate * (avg_win)) + (loss_rate*(avg_loss))

full_gross_pnl = round(((all_trades[capital_choice].iloc[-1] - portfolio_size) / portfolio_size)*100, 2)

summary = f"""
Trade Performance Summary
--------------------------
Win Rate     : {win_rate:.2%}
Loss Rate    : {loss_rate:.2%}
Avg Win      : +{avg_win}%
Avg Loss     : -{avg_loss}%
Best Win     : +{best_win}%
Worst Loss   : {worst_loss}%
Expected EV  : {theo_ev_per_trade:.2f}%
Gross PnL    : {full_gross_pnl}% | {round(all_trades[capital_choice].iloc[-1] - portfolio_size, 2)}
Max 5d DD    : {worst_5d_drawdown}% | ${round((worst_5d_drawdown/100)*portfolio_size, 2)}
Max 30d DD   : {worst_30d_drawdown}% + | ${round((worst_30d_drawdown/100)*portfolio_size, 2)}
Median 5D $  : {round(median_5d_pnl, 2)}
Median 30D $ : {round(median_30d_pnl, 2)}
2SD Loss     : {round(losses["return_on_size"].mean() - (losses["return_on_size"].std()*2), 2)}

"""

print(summary)