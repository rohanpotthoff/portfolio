import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime

# Set page config
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

st.title("📈 Portfolio Tracker Dashboard")

# Performance timeframes
period_map = {
    "1D": "1d",
    "3D": "5d",
    "5D": "5d",
    "1W": "7d",
    "1M": "1mo",
    "1Y": "1y",
    "5Y": "5y",
    "10Y": "10y",
    "20Y": "20y"
}

# Comparison tickers
comparison_tickers = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Euro Stoxx 50": "^STOXX50E"
}

# Timeframe selection
st.sidebar.subheader("Performance Comparison")
selected_period = st.sidebar.selectbox("Select time range", list(period_map.keys()), index=0)

# Fetch benchmark data
st.subheader(f"📊 Portfolio vs Benchmarks ({selected_period})")
period = period_map[selected_period]
benchmark_data = {}
benchmark_series = []

for label, symbol in comparison_tickers.items():
    try:
        hist = yf.Ticker(symbol).history(period=period)
        if not hist.empty:
            pct_change = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
            benchmark_data[label] = pct_change
            benchmark_series.append(pd.DataFrame({
                "Date": hist.index,
                "Price": hist["Close"].values,
                "Index": label
            }))
        else:
            benchmark_data[label] = None
    except Exception:
        benchmark_data[label] = None

# Placeholder for portfolio performance (real calculation below if data available)
portfolio_change = None

# File uploader
uploaded_file = st.file_uploader("Upload your holdings CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Normalize columns
    df.columns = [col.strip().capitalize() for col in df.columns]
    required_columns = ["Ticker", "Quantity"]

    if all(col in df.columns for col in required_columns):
        st.success("Holdings file loaded successfully!")

        # Fetch current and historical prices
        tickers = df["Ticker"].unique().tolist()
        data = []
        portfolio_start_value = 0
        portfolio_end_value = 0

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                info = stock.info
                price = info.get("regularMarketPrice") or info.get("currentPrice")
                sector = info.get("sector", "Unknown")
                df_ticker = df[df["Ticker"] == ticker]
                quantity = df_ticker["Quantity"].sum()
                start_price = hist["Close"].iloc[0] if not hist.empty else price
                end_price = hist["Close"].iloc[-1] if not hist.empty else price
                portfolio_start_value += quantity * start_price
                portfolio_end_value += quantity * end_price
                data.append({"Ticker": ticker, "Current Price": end_price, "Sector": sector})
            except Exception:
                data.append({"Ticker": ticker, "Current Price": None, "Sector": "Unknown"})

        if portfolio_start_value > 0:
            portfolio_change = (portfolio_end_value / portfolio_start_value - 1) * 100

        price_df = pd.DataFrame(data)
        df = df.merge(price_df, on="Ticker", how="left")
        df["Market Value"] = df["Quantity"] * df["Current Price"]

        # Account filter
        if "Account" in df.columns:
            accounts = df["Account"].dropna().unique().tolist()
            selected_accounts = st.multiselect("Filter by account(s):", accounts, default=accounts)
            df = df[df["Account"].isin(selected_accounts)]

        st.subheader("📊 Portfolio Overview")
        st.dataframe(df)

        # Summary stats
        total_value = df["Market Value"].sum()
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Sector Allocation")
            sector_group = df.groupby("Sector")["Market Value"].sum().reset_index()
            fig_sector = px.pie(sector_group, values="Market Value", names="Sector", title="By Sector")
            st.plotly_chart(fig_sector, use_container_width=True)

        with col2:
            if "Asset Class" in df.columns:
                st.subheader("Asset Class Allocation")
                asset_group = df.groupby("Asset Class")["Market Value"].sum().reset_index()
                fig_asset = px.pie(asset_group, values="Market Value", names="Asset Class", title="By Asset Class")
                st.plotly_chart(fig_asset, use_container_width=True)

        with col3:
            if "Account" in df.columns:
                st.subheader("Account Allocation")
                account_group = df.groupby("Account")["Market Value"].sum().reset_index()
                fig_account = px.pie(account_group, values="Market Value", names="Account", title="By Account")
                st.plotly_chart(fig_account, use_container_width=True)

# Performance Metrics Display
st.subheader("🔁 Performance Summary")
if portfolio_change is not None:
    st.metric("Portfolio", f"{portfolio_change:+.2f}%", delta_color="normal")
else:
    st.metric("Portfolio", "N/A")

for label, change in benchmark_data.items():
    if change is not None:
        st.metric(label, f"{change:+.2f}%")
    else:
        st.metric(label, "N/A")

# Chart for benchmark comparison
if benchmark_series:
    combined_df = pd.concat(benchmark_series)
    fig = px.line(combined_df, x="Date", y="Price", color="Index", title="Benchmark Performance")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Benchmark data unavailable for the selected timeframe.")
