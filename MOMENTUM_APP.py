import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
from MOMENTR import *  

all_m_ret = pd.read_csv("all_monthly_returns.csv", index_col=0, parse_dates=True)
all_w_ret = pd.read_csv("all_weekly_returns.csv", index_col=0, parse_dates=True)

benchmark_ticker = st.sidebar.text_input("Enter Benchmark Ticker", value="SPY")
risk_free_ticker = st.sidebar.text_input("Enter Risk-Free Ticker", value="BIL")

@st.cache_data
def load_benchmark_and_rf(benchmark_ticker, risk_free_ticker):
    benchmark = yf.download(benchmark_ticker,auto_adjust=False, start='2010-01-01')[['Adj Close']]
    benchmark.columns = ["Benchmark"]
    risk_free = yf.download(risk_free_ticker, auto_adjust=False,start='2010-01-01')[['Adj Close']]
    risk_free.columns = ["Risk Free Return"]
    benchmark_m = benchmark.pct_change().resample('ME').apply(lambda x: (x + 1).prod() - 1).dropna()
    benchmark_w = benchmark.pct_change().resample('W').apply(lambda x: (x + 1).prod() - 1).dropna()
    risk_free_m = risk_free.pct_change().resample('ME').apply(lambda x: (x + 1).prod() - 1).dropna()
    risk_free_w = risk_free.pct_change().resample('W').apply(lambda x: (x + 1).prod() - 1).dropna()
    return benchmark_m, benchmark_w, risk_free_m, risk_free_w

benchmark_m, benchmark_w, risk_free_m, risk_free_w = load_benchmark_and_rf(benchmark_ticker, risk_free_ticker)

st.title("Momentum Filter Backtest")
st.write("Choose between simple-, double-, and triple momentum filters for the stock universe of the S&P 500, Dow Jones, and NASDAQ, since 2010")
st.write("The Lookback Period indicates over what time frame(s) you want to assess a stock's momentum")
st.write("The Top Count selects how many stocks you want to pick from the stock universe.")
st.write("The Holding Length specifies how long you want to hold the stocks. The default is one period.")
st.write("A simple filter picks the number of best-performing stocks over the lookback period you specify (default: the 10 best stocks over 12 periods)")
st.write("A dual filter picks among those stocks for another Lookback Period you specify, and a triple filter further picks the best over a third Lookback Period.")
st.write("Be careful: this strategy has very high turnover, which leads to substantial trading- and tax costs, which are not considered.")


st.sidebar.header("Backtest Parameters")
frequency = st.sidebar.selectbox("Select Frequency", ["Weekly", "Monthly"], index=0)
returns = all_w_ret if frequency == "Weekly" else all_m_ret
benchmark = benchmark_w if frequency == "Weekly" else benchmark_m
freq = 52 if frequency == "Weekly" else 12

last_date = returns.index[-1]

# Dropdown menu to select backtest type
backtest_type = st.sidebar.selectbox("Select Backtest Type", ["Triple", "Dual", "Simple"], index=0)

if backtest_type == "Triple":
    lookback_periods = st.sidebar.text_input("Lookback Periods (comma-separated)", "12,8,4")
    lookback_periods = tuple(map(int, lookback_periods.split(",")))

    top_counts = st.sidebar.text_input("Top Counts (comma-separated)", "50,25,10")
    top_counts = tuple(map(int, top_counts.split(",")))

elif backtest_type == "Dual":
    lookback_periods = st.sidebar.text_input("Lookback Periods (comma-separated)", "12,6")
    lookback_periods = tuple(map(int, lookback_periods.split(",")))

    top_counts = st.sidebar.text_input("Top Counts (comma-separated)", "50,10")
    top_counts = tuple(map(int, top_counts.split(",")))

elif backtest_type == "Simple":
    lookback_periods = st.sidebar.number_input("Lookback Period", min_value=1, value=12)
    top_counts = st.sidebar.number_input("Top Count", min_value=1, value=10)

holding_length = st.sidebar.number_input("Holding Length", min_value=1, value=1)

if st.sidebar.button("Run Analysis"):
    if backtest_type == "Triple":
        backtest = triple_momentum_filter(
            returns=returns,
            lookback_periods=lookback_periods,
            top_counts=top_counts,
            holding_length=holding_length)
    elif backtest_type == "Dual":
        backtest = dual_momentum_filter(
            returns=returns,
            lookback_periods=lookback_periods,
            top_counts=top_counts,
            holding_length=holding_length)
    elif backtest_type == "Simple":
        backtest = simple_momentum_filter(
            returns=returns,
            lookback_periods=lookback_periods,
            top_counts=top_counts,
            holding_length=holding_length)

    rolling_cumulative_return = calc_rolling_cum_ret(backtest["Strategy Return"], freq=freq)
    if backtest_type == "Triple":
        tickers_to_plot = get_latest_picks_triple(
            returns,
            lookback_periods=lookback_periods,
            top_counts=top_counts)
    elif backtest_type == "Dual":
        tickers_to_plot = get_latest_picks_dual(
            returns,
            lookback_periods=lookback_periods,
            top_counts=top_counts)
    elif backtest_type == "Simple":
        tickers_to_plot = get_latest_picks_simple(
            returns,
            lookback_periods=lookback_periods,
            top_counts=top_counts)

    st.subheader("Backtest Results")
    st.write("### Strategy Returns")
    st.write("This is how your Strategy performs over the sample period.")
    st.write(f"The backtest uses data until {last_date.strftime('%Y-%m-%d')}.")
    strat_fig = plot_cumulative_returns(backtest["Strategy Return"], benchmark)
    st.plotly_chart(strat_fig)

    st.write("### Rolling Cumulative Returns")
    st.write("This is the Rolling 1-year Cumulative Return of your Strategy.")
    rolling_fig = plot_rolling_cumulative_return_2(rolling_cumulative_return)
    st.plotly_chart(rolling_fig)

    st.write("### Top Stocks")
    st.write("These are the most current stock picks according to the filter you chose.")
    lookbacks_for_plot = lookback_periods if backtest_type != "Simple" else (lookback_periods,)
    top_stocks_fig = plot_top_stocks(returns=returns, lookbacks=lookbacks_for_plot, tickers_to_plot=tickers_to_plot)
    st.plotly_chart(top_stocks_fig)
