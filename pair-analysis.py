# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 01:39:32 2025

@author: Local User
"""


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytz

import sqlalchemy
import mysql.connector
import statsmodels.api as sm
import xgboost
import pmdarima as pm
import pyarrow
import pyarrow.parquet as pq

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# Core-Requisites
# =============================================================================

engine = sqlalchemy.create_engine('mysql+mysqlconnector://dbadmin:XBCy9erLMMWC2xUJesy5@qg-aws-v2.cm3csfkhhqeu.us-east-1.rds.amazonaws.com:3306/qgv2')
universe = pd.read_sql("historical_liquid_tickers_polygon", con = engine).drop_duplicates(subset=["date", "ticker"])

calendar = get_calendar("NYSE")
all_trading_dates = calendar.schedule(start_date = "2001-01-01", end_date = datetime.now(pytz.timezone("America/New_York"))+timedelta(days=45)).index.strftime("%Y-%m-%d").values

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"

long_ticker = "RGTI"
short_ticker = "QUBT"

# =============================================================================
# L/S Data
# =============================================================================

trading_dates = calendar.schedule(start_date = "2025-01-01", end_date = (datetime.today()-timedelta(days = 1))).index.strftime("%Y-%m-%d").values

# Keep 1 session active (same headers, etc.)
connection_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=500)
connection_session.mount('https://', adapter)

full_basket_list = []
times = []

target_pnl = 1

trade_list = []

# date = trading_dates[0] # np.where(trading_dates==date)
for date in trading_dates:
    
    try:
        
        start_time = datetime.now()
 
        # =============================================================================
        # Modeling
        # =============================================================================
        
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
        
        # combined_intraday_data["long_wt"] = pair_data["long_wt"].iloc[0]
        
        # combined_intraday_data["adj_long_returns"] = combined_intraday_data["intraday_returns_x"] * combined_intraday_data["long_wt"]
        
        combined_intraday_data["gross_return"] = combined_intraday_data["intraday_returns_x"] - combined_intraday_data["intraday_returns_y"]
        
        signals = combined_intraday_data[combined_intraday_data["gross_return"] >= target_pnl].sort_index().copy()
        
        if len(signals) < 1:
            
            trade = pd.DataFrame([{"date": date, "long_ticker": long_ticker,
                                   "short_ticker": short_ticker,
                                   "return": combined_intraday_data["gross_return"].iloc[-1],
                                   "signal": 0}])
            
        else:
            
            trade = pd.DataFrame([{"date": date, "long_ticker": long_ticker,
                                   "short_ticker": short_ticker, "return": signals["gross_return"].iloc[0],
                                   "signal": 1}])
            
        trade_list.append(trade)
    
    
    except Exception as error:
        print(error)
        continue
            
    
notional =10000    
all_trades = pd.concat(trade_list)#.drop_duplicates(subset=["date"], keep="first").copy()#.groupby("date").mean(numeric_only=True).reset_index()
# all_trades = all_trades[all_trades["px_corr"]>=.9].drop_duplicates(subset=["date"], keep="first").copy()
# all_trades["long_ticker"] = long_ticker
# all_trades["short_ticker"] = short_ticker

all_trades["dollar_pnl"] = ((all_trades["return"]/100) * notional)

all_trades["return_on_size"] = round((all_trades["dollar_pnl"] / notional)*100, 2)
all_trades["rolling_std"] = all_trades["return_on_size"].rolling(window=5).std()
all_trades["capital"] = notional + all_trades["dollar_pnl"].cumsum()

all_trades["strategy_return"] = round(((all_trades["capital"] - notional) / notional)*100, 2)
all_trades["strategy_sharpe"] = round(all_trades["strategy_return"] / (all_trades["rolling_std"] * np.sqrt(len(all_trades))), 2)

plt.figure(dpi=200)
plt.xticks(rotation = 45)
plt.suptitle(f"Optimal-Ranked Pairs")
plt.plot(pd.to_datetime(all_trades["date"]).values, all_trades["capital"].values)
plt.legend(["capital"])
plt.show()
plt.close()


len(all_trades[all_trades["return_on_size"] > 0]) / len(all_trades)

from catboost import CatBoostClassifier


complete_trade_data = pd.concat(trade_list)

completed_trade_dates = np.sort(complete_trade_data["date"].drop_duplicates().values)[20:]



model_features = ["long_ticker_x", "short_ticker_y","px_corr", "ret_corr", "long_wt"]
target = "signal"

complete_trade_list = []

# complete_date = completed_trade_dates[0]
for complete_date in completed_trade_dates:
    
    prior_trade_date = np.sort(all_trading_dates[all_trading_dates < complete_date])[-1]
    prior_trade_start = np.sort(all_trading_dates[all_trading_dates < complete_date])[-20]
    
    historical_data = complete_trade_data[(complete_trade_data["date"] >= prior_trade_start) & (complete_trade_data["date"] <= prior_trade_date)].copy()
    
    X = historical_data[model_features]
    Y = historical_data[target]

    Model = CatBoostClassifier(cat_features=["long_ticker_x", "short_ticker_y"], iterations=100).fit(X, Y)
    
    oos_data = complete_trade_data[(complete_trade_data["date"] == complete_date)].copy()
    
    X_oos = oos_data[model_features]
    
    predictions = Model.predict(X_oos)
    probas = Model.predict_proba(X_oos)

    oos_data["prediction"] = predictions
    oos_data["prob_0"] = np.float64(probas[:, 0])
    oos_data["prob_1"] = np.float64(probas[:, 1])
    
    oos_best_performer = oos_data.copy().sort_values(by="prob_1", ascending=False).head(1)
    
    complete_trade_list.append(oos_best_performer)
    
optimal_trades = pd.concat(complete_trade_list)
optimal_trades = optimal_trades[optimal_trades["prob_1"] >= .80].copy()

optimal_trades["adj_pnl"] = (optimal_trades["return"]/100) * notional
optimal_trades["adj_capital"] = notional + optimal_trades["adj_pnl"].cumsum()    

plt.figure(dpi=200)
plt.xticks(rotation = 45)
plt.suptitle(f"Optimal-Ranked Pairs")
plt.plot(pd.to_datetime(optimal_trades["date"]).values, optimal_trades["adj_capital"].values)
plt.legend(["optimal"])
plt.show()
plt.close()

len(optimal_trades[optimal_trades["adj_pnl"] > 0]) / len(optimal_trades)

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