import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Stock Momentum Analyzer",
    layout="wide"
)

# ----------------------------------
# STOCK UNIVERSE
# ----------------------------------
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD", "NFLX", "INTC",
    "JPM", "BAC", "XOM", "CVX", "KO"
]

# ----------------------------------
# SIDEBAR CONTROLS
# ----------------------------------
st.sidebar.header("Parameters")

lookback_days = st.sidebar.number_input(
    "Lookback Days",
    min_value=10,
    max_value=180,
    value=30,
    step=5
)

predict_days = st.sidebar.number_input(
    "Predict Days",
    min_value=1,
    max_value=30,
    value=5
)

top_n = st.sidebar.number_input(
    "Top N Stocks",
    min_value=1,
    max_value=10,
    value=5
)

run = st.sidebar.button("Run Analysis")

# ----------------------------------
# DATA LOADING (CACHED)
# ----------------------------------
@st.cache_data(ttl=3600)
def load_data(lookback_days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days + 40)

    data = yf.download(
        STOCK_UNIVERSE,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    return data.dropna(axis=1)

# ----------------------------------
# ANALYSIS
# ----------------------------------
def analyze(data):
    results = []

    for ticker in data.columns:
        prices = data[ticker].dropna()
        if len(prices) < lookback_days:
            continue

        recent_prices = prices[-lookback_days:]
        returns = recent_prices.pct_change().dropna()

        X = np.arange(len(recent_prices)).reshape(-1, 1)
        y = recent_prices.values.reshape(-1, 1)

        model = LinearRegression().fit(X, y)
        slope = model.coef_[0][0]
        volatility = returns.std()

        expected_return = (slope * predict_days) / recent_prices.iloc[-1]
        score = expected_return / volatility if volatility != 0 else 0

        results.append({
            "Ticker": ticker,
            "Expected {}D Return %".format(predict_days): expected_return * 100,
            "Volatility": volatility,
            "Score": score
        })

    df = pd.DataFrame(results).sort_values("Score", ascending=False)
    return df.head(top_n)

# ----------------------------------
# MAIN UI
# ----------------------------------
st.title("📈 Stock Momentum Analyzer")

if run:
    with st.spinner("Downloading & analyzing data..."):
        data = load_data(lookback_days)
        top_df = analyze(data)

    st.subheader("Top Ranked Stocks")
    st.dataframe(top_df, use_container_width=True)

    st.subheader("Price Movement (Last {} Days)".format(lookback_days))

    fig, ax = plt.subplots(figsize=(12, 6))

    for ticker in top_df["Ticker"]:
        ax.plot(data[ticker][-lookback_days:], label=ticker)

    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
else:
    st.info("Set parameters and click **Run Analysis**.")
