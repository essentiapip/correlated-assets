# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 23:33:29 2025

@author: quant
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

# date = trading_dates[0]
for date in trading_dates:
    
    try:
        
        start_time = datetime.now()
        
        # Rebalancing near EOD, so tickers on that day would be available
        point_in_time_dates = np.sort(universe[universe["date"] <= date]["date"].drop_duplicates().values)
        point_in_time_date = point_in_time_dates[-1]        
        point_in_time_universe = universe[universe["date"] == point_in_time_date].drop_duplicates(subset=["ticker"], keep = "last")
    
        tickers = point_in_time_universe["ticker"].drop_duplicates().values
        
        lookback_date = np.sort(all_trading_dates[all_trading_dates < date])[-252]
        prior_date = np.sort(all_trading_dates[all_trading_dates < date])[-1]
        
        ticker_data_list = []
        
        # ticker = tickers[0]
        for ticker in tickers:
        
            try:
            
                # stock_info = universe[universe["Symbol"] == ticker].copy()
                
                underlying_data_original = pq.read_table(source=f"Imported Adjusted Daily Data/{ticker}/{ticker}.parquet").to_pandas().set_index("t")
                underlying_data = underlying_data_original[(underlying_data_original["date"] >= lookback_date) & (underlying_data_original["date"] < date)].copy().sort_index()
                
                
                if len(underlying_data) < 20:
                    continue
                
                underlying_data["year"] = underlying_data.index.year
                underlying_data["month"] = underlying_data.index.month
                
                underlying_data["pct_chg"] = round(underlying_data["c"].pct_change()*100, 2)
                underlying_data["log_ret"] = np.log(underlying_data["c"] / underlying_data["c"].shift(1))
                underlying_data["log_ret_sq"] = underlying_data["log_ret"] ** 2
                
                underlying_data["var"] = underlying_data["log_ret_sq"].rolling(window=20).sum()
                
                underlying_data["vol"] = round((np.sqrt(underlying_data["var"]) / np.sqrt(252/20))*100, 2)
                
                ticker_px_data = underlying_data.tail(20).copy()
                ticker_data_list.append(ticker_px_data)
                
                
            except Exception as error:
                print(error)
                continue
            
        full_ticker_prices = pd.concat(ticker_data_list)
        
        ticker_dates_in_coverage = np.sort(full_ticker_prices["date"].drop_duplicates().values)
        
        prior_day_uni = full_ticker_prices[full_ticker_prices["date"] == prior_date].copy().sort_values(by="vol", ascending=False)
        
        sample = prior_day_uni.head(25)
        sample_tickers = sample["ticker"].drop_duplicates().values
        
        corr_pair_list = []
        
        # sample_ticker = sample_tickers[0]
        for sample_ticker in sample_tickers:
            
            sample_ticker_data = full_ticker_prices[full_ticker_prices["ticker"] == sample_ticker].copy().sort_index()
            ticker_date_universe = full_ticker_prices[(full_ticker_prices["date"] == ticker_date) & (full_ticker_prices["ticker"] != sample_ticker)].copy()#.sort_index()

            tickers_on_date = ticker_date_universe["ticker"].drop_duplicates().values
            
            ticker_corr_list = []

            # date_ticker = tickers_on_date[0]
            for date_ticker in tickers_on_date:
            
                sample_ticker_data_for_comparison = full_ticker_prices[full_ticker_prices["ticker"] == date_ticker].copy().sort_index()
                
                ticker_corr_sample = pd.merge(sample_ticker_data[["date","c", "pct_chg"]], sample_ticker_data_for_comparison[["date","c", "pct_chg"]], on = "date")
                
                px_corr = ticker_corr_sample["c_x"].corr(ticker_corr_sample["c_y"])
                ret_corr = ticker_corr_sample["pct_chg_x"].corr(ticker_corr_sample["pct_chg_y"])
                
                ticker_corr_data = pd.DataFrame([{"ticker": date_ticker, "px_corr": px_corr, "ret_corr": ret_corr}])
                
                ticker_corr_list.append(ticker_corr_data)
                
            full_ticker_corr_data = pd.concat(ticker_corr_list).sort_values(by="ret_corr", ascending=False)
            
            most_correlated = full_ticker_corr_data.head(1)
            
            corr_pair_data = pd.DataFrame([{"date": date, "long_ticker": sample_ticker, "short_ticker": most_correlated["ticker"].iloc[0],
                                            "px_corr": most_correlated["px_corr"].iloc[0], "ret_corr": most_correlated["ret_corr"].iloc[0]}])
            
            corr_pair_list.append(corr_pair_data)
            
        full_corr_pairs = pd.concat(corr_pair_list)
            
            
    
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
    
    
complete_trade_data = pd.concat(all_trades_list)

completed_trade_dates = np.sort(complete_trade_data["date"].drop_duplicates().values)[20:]

complete_trade_list = []

# complete_date = completed_trade_dates[0]
for complete_date in completed_trade_dates:
    
    prior_trade_date = np.sort(all_trading_dates[all_trading_dates < complete_date])[-1]
    
    prior_day_trades = complete_trade_data[complete_trade_data["date"] == prior_trade_date].copy()
    
    best_performer = prior_day_trades.sort_values(by="strategy_sharpe", ascending=False).head(1)
    
    trade_day_data = complete_trade_data[complete_trade_data["date"] == complete_date].copy()
    
    oos_best_performer = trade_day_data[trade_day_data["long_ticker"] == best_performer["long_ticker"].iloc[0]].copy()
    
    complete_trade_list.append(oos_best_performer)
    
optimal_trades = pd.concat(complete_trade_list)

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