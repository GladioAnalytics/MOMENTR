import yfinance as yf 
from yahoo_fin.stock_info import *
## Standard Python Data Science stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import math
import statsmodels.api as sm
import datetime as dt
import math
from scipy.optimize import minimize
from scipy.stats import (norm as norm, linregress as linregress)
plt.rcParams['figure.figsize'] = [20, 10]
#from SHARPR_backend import *
from fredapi import Fred
fred = Fred(api_key='3cc2743ce40daec36ca56954fefedca7')
from FedTools import MonetaryPolicyCommittee
from FedTools import BeigeBooks
from FedTools import FederalReserveMins
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import ceil
import streamlit as st


def rolling_cumulative_returns(returns, window):
    return (returns.add(1).rolling(window=window).apply(np.prod, raw=True) - 1)

def top_n_columns_by_last_row(cum_rets, n):
    last_row = cum_rets.iloc[-1]
    top_columns = last_row.nlargest(n).index.tolist()
    return top_columns

def top_n_columns_by_row(cum_rets, n, row_num):
    last_row = cum_rets.iloc[row_num]
    top_columns = last_row.nlargest(n).index.tolist()
    return top_columns

def top_n_columns_by_all_rows(cum_rets, n):
    top_columns_df = pd.DataFrame(
        index=cum_rets.index,
        columns=[f"Top_{i+1}" for i in range(n)])
    for idx, row in cum_rets.iterrows():
        top_columns_df.loc[idx] = row.nlargest(n).index.tolist()
    return top_columns_df


def triple_momentum_filter(returns,lookback_periods = (12,8,4),top_counts = (50,25,5),holding_length=1):

    roll_let_long = rolling_cumulative_returns(returns,lookback_periods[0]).iloc[(lookback_periods[0]-1):-holding_length]
    roll_let_medium = rolling_cumulative_returns(returns,lookback_periods[1]).iloc[(lookback_periods[0]-1):-holding_length]
    roll_let_short = rolling_cumulative_returns(returns,lookback_periods[2]).iloc[(lookback_periods[0]-1):-holding_length]

    holding_cum_ret = (returns.iloc[(lookback_periods[0]):-holding_length].add(1).rolling(window=1).apply(np.prod, raw=True) - 1)

    portfolio_formation_dates = roll_let_long.index
    stock_picks = []
    for i in range(len(portfolio_formation_dates)):
        top_long = top_n_columns_by_row(roll_let_long,top_counts[0],row_num=i)
        top_medium = top_n_columns_by_row(roll_let_medium[top_long],top_counts[1],row_num=i)
        top_short = top_n_columns_by_row(roll_let_short[top_medium],top_counts[2],row_num=i)
        stock_picks.append(top_short)
    stock_picks = pd.DataFrame(stock_picks)
    stock_picks.index = portfolio_formation_dates
    stock_picks.columns = [f"Top_{i+1}" for i in range(len(stock_picks.columns))]

    earn_dates = holding_cum_ret.index
    raw_returns = []
    for i in range(len(earn_dates)):
        picks = stock_picks.iloc[i].values
        raw_rets = holding_cum_ret[picks].iloc[i].values
        raw_returns.append(raw_rets)
    raw_returns = pd.DataFrame(raw_returns)
    raw_returns.index = earn_dates
    raw_returns.columns = [f"Top_{i+1}" for i in range(len(raw_returns.columns))]

    strategy_return = raw_returns.mean(axis=1)

    return {
        "Strategy Return": strategy_return,
        "Raw Returns": raw_returns,
        "Stock Picks": stock_picks}

def plot_cumulative_returns(strategy_return, benchmark):
    strat = (strategy_return + 1).cumprod()
    bench = (benchmark + 1).cumprod()

    aligned_strat, aligned_bench = strat.align(bench, join='inner')

    aligned_strat = aligned_strat.squeeze()
    aligned_bench = aligned_bench.squeeze()

    combined_df = pd.DataFrame({
        'Strategy Return': aligned_strat,
        'Benchmark Return': aligned_bench})

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=combined_df.index, 
        y=combined_df['Strategy Return'],
        mode='lines',
        name='Strategy Return',
        line=dict(width=2, color='blue')))

    fig.add_trace(go.Scatter(
        x=combined_df.index, 
        y=combined_df['Benchmark Return'],
        mode='lines',
        name='Benchmark Return',
        line=dict(width=2, color='orange')))

    fig.update_layout(
        title='Cumulative Returns: Strategy vs Benchmark',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        legend=dict(font=dict(size=12)),
        template='plotly_white',
        width=800,
        height=600)
    return fig

def plot_rolling_cumulative_return(rolling_cumulative_return):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_cumulative_return.index,
        y=rolling_cumulative_return.values,
        mode='lines',
        name='Rolling Cumulative Return',
        line=dict(width=2, color='blue')))

    fig.add_trace(go.Scatter(
        x=rolling_cumulative_return.index,
        y=[1] * len(rolling_cumulative_return),
        mode='lines',
        name='Baseline',
        line=dict(width=1, dash='dash', color='black')))

    fig.update_layout(
        title="Rolling 1-year Cumulative Return",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_top_stocks(returns, lookbacks, tickers_to_plot):
    tickers_to_plot = [ticker for ticker in tickers_to_plot if ticker in returns.columns]
    data_to_plot = (returns.loc["2024", tickers_to_plot] + 1).cumprod()
    n_cols = 4
    n_rows = ceil(len(data_to_plot.columns) / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=data_to_plot.columns)
    lookback_indices = [data_to_plot.index[-lb] for lb in lookbacks if lb <= len(data_to_plot.index)]
    for i, col in enumerate(data_to_plot.columns):
        row = i // n_cols + 1
        col_position = i % n_cols + 1
        series = data_to_plot[col]
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name=col,
            legendgroup=col,
            line=dict(width=2)
        ), row=row, col=col_position)
        for lb_index in lookback_indices:
            fig.add_shape(
                type="line",
                x0=lb_index, x1=lb_index,
                y0=series.min(), y1=series.max(),
                line=dict(color="black", dash="dash"),
                row=row, col=col_position)
    fig.update_layout(
        title="Top Stocks Cumulative Returns",
        height=300 * n_rows,
        width=1200,
        template="plotly_white")
    return fig

def dual_momentum_filter(returns, lookback_periods=(12, 6), top_counts=(50, 10), holding_length=1):
    roll_let_long = rolling_cumulative_returns(returns, lookback_periods[0]).iloc[(lookback_periods[0] - 1):-holding_length]
    roll_let_short = rolling_cumulative_returns(returns, lookback_periods[1]).iloc[(lookback_periods[0] - 1):-holding_length]


    holding_cum_ret = (returns.iloc[(lookback_periods[0]):-holding_length].add(1).rolling(window=1).apply(np.prod, raw=True) - 1)
    portfolio_formation_dates = roll_let_long.index
    stock_picks = []

    for i in range(len(portfolio_formation_dates)):
        top_long = top_n_columns_by_row(roll_let_long, top_counts[0], row_num=i)
        top_short = top_n_columns_by_row(roll_let_short[top_long], top_counts[1], row_num=i)
        stock_picks.append(top_short)

    stock_picks = pd.DataFrame(stock_picks)
    stock_picks.index = portfolio_formation_dates
    stock_picks.columns = [f"Top_{i+1}" for i in range(len(stock_picks.columns))]

    earn_dates = holding_cum_ret.index
    raw_returns = []

    for i in range(len(earn_dates)):
        picks = stock_picks.iloc[i].values
        raw_rets = holding_cum_ret[picks].iloc[i].values
        raw_returns.append(raw_rets)

    raw_returns = pd.DataFrame(raw_returns)
    raw_returns.index = earn_dates
    raw_returns.columns = [f"Top_{i+1}" for i in range(len(raw_returns.columns))]

    strategy_return = raw_returns.mean(axis=1)
    return {
        "Strategy Return": strategy_return,
        "Raw Returns": raw_returns,
        "Stock Picks": stock_picks}

def simple_momentum_filter(returns, lookback_periods=12, top_counts=10, holding_length=1):
    roll_let = rolling_cumulative_returns(returns, lookback_periods).iloc[(lookback_periods - 1):-holding_length]

    holding_cum_ret = (returns.iloc[(lookback_periods):-holding_length]
                       .add(1).rolling(window=1).apply(np.prod, raw=True) - 1)

    portfolio_formation_dates = roll_let.index
    stock_picks = []

    for i in range(len(portfolio_formation_dates)):
        top_stocks = top_n_columns_by_row(roll_let, top_counts, row_num=i)
        stock_picks.append(top_stocks)

    stock_picks = pd.DataFrame(stock_picks)
    stock_picks.index = portfolio_formation_dates
    stock_picks.columns = [f"Top_{i+1}" for i in range(len(stock_picks.columns))]

    earn_dates = holding_cum_ret.index
    raw_returns = []

    for i in range(len(earn_dates)):
        picks = stock_picks.iloc[i].values
        raw_rets = holding_cum_ret[picks].iloc[i].values
        raw_returns.append(raw_rets)
    raw_returns = pd.DataFrame(raw_returns)
    raw_returns.index = earn_dates
    raw_returns.columns = [f"Top_{i+1}" for i in range(len(raw_returns.columns))]

    strategy_return = raw_returns.mean(axis=1)

    return {
        "Strategy Return": strategy_return,
        "Raw Returns": raw_returns,
        "Stock Picks": stock_picks}

def get_latest_picks_triple(returns,lookback_periods = (12,8,4),top_counts = (50,25,5)):
    returns = returns.iloc[-(lookback_periods[0]+1):-1]
    roll_let_long = rolling_cumulative_returns(returns,lookback_periods[0])
    roll_let_medium = rolling_cumulative_returns(returns,lookback_periods[1])
    roll_let_short = rolling_cumulative_returns(returns,lookback_periods[2])
    top_long = top_n_columns_by_last_row(roll_let_long,top_counts[0])
    top_medium = top_n_columns_by_last_row(roll_let_medium[top_long],top_counts[1])
    top_short = top_n_columns_by_last_row(roll_let_short[top_medium],top_counts[2])
    return top_short

def get_latest_picks_dual(returns,lookback_periods = (12,4),top_counts = (50,5)):
    returns = returns.iloc[-(lookback_periods[0]+1):-1]
    roll_let_long = rolling_cumulative_returns(returns,lookback_periods[0])
    roll_let_short = rolling_cumulative_returns(returns,lookback_periods[1])
    top_long = top_n_columns_by_last_row(roll_let_long,top_counts[0])
    top_short = top_n_columns_by_last_row(roll_let_short[top_long],top_counts[1])
    return top_short

def get_latest_picks_simple(returns,lookback_periods = 12,top_counts = 5):
    returns = returns.iloc[-(lookback_periods+1):-1]
    roll_let = rolling_cumulative_returns(returns,top_counts)
    top = top_n_columns_by_last_row(roll_let,top_counts)
    return top

def calc_rolling_cum_ret(return_series,freq=52):
    rolling_cumulative_return = ((return_series.add(1).rolling(window=freq).apply(lambda x: x.prod(), raw=True))).dropna()
    return rolling_cumulative_return


def plot_rolling_cumulative_return_2(series, constant_value=1):
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        raise ValueError("The index of the input Series must be datetime-like.")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        line=dict(color='blue'),
        name='Strategy'
    ))

    fig.add_trace(go.Scatter(
        x=series.index,
        y=[constant_value] * len(series),
        mode='lines',
        line=dict(color='black', dash='dash'),
        name=f'Start Capital ({constant_value})'))

    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.where(series < constant_value, constant_value),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False))

    fig.update_layout(
        title="1-year Rolling Cumulative Return",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=600)

    return fig


def update_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)
    sp500_df = table[0]  
    sp500_tickers = sp500_df['Symbol'].tolist()

    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    table = pd.read_html(url, header=0)
    dow_df = table[2]  
    dow_tickers = dow_df['Symbol'].tolist()

    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    table = pd.read_html(url,header=0)
    nasdaq_df = table[4] 
    nasdaq_tickers = nasdaq_df.Symbol.to_list()

    tickers = set(sp500_tickers) | set(dow_tickers) | set(nasdaq_tickers)

    prices = [] 
    symbols = []
    start_date = "2010-01-01"

    progress_bar = st.progress(0)
    counter_text = st.empty()
    total_tickers = len(tickers)

    for i, symbol in enumerate(tickers, start=1):
        df = yf.download(symbol, start=start_date)['Adj Close']
        if not df.empty:
            prices.append(df)
            symbols.append(symbol)
        progress_bar.progress(i / total_tickers)
        counter_text.text(f"Processed {i}/{total_tickers} tickers")

    all_prices = pd.concat(prices, axis=1)
    all_prices.columns = symbols

    all_m_ret = all_prices.pct_change(fill_method=None).resample('ME').agg(lambda x: (x + 1).prod() - 1).dropna().iloc[:-1]
    all_w_ret = all_prices.pct_change(fill_method=None).resample('W').agg(lambda x: (x + 1).prod() - 1).dropna().iloc[:-1]

    progress_bar.empty()
    counter_text.empty()

    return all_m_ret, all_w_ret