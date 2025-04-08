import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import datetime

# Set page config
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# Version header with tooltip
st.title("📈 Portfolio Tracker Dashboard")
st.caption("Version 1.0.0.1")
st.markdown("<small title='Created by Rohan Potthoff (rohanpotthoff@gmail.com)'>ℹ️ Hover for contact info</small>", unsafe_allow_html=True)

# Performance timeframes
period_map = {
    "Today": "1d",
    "3D": "5d",
    "1W": "7d",
    "1M": "1mo",
    "YTD": "ytd",
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

# Sidebar Filters
st.sidebar.subheader("Filters")
selected_period = st.sidebar.selectbox("Select time range", list(period_map.keys()), index=0)

uploaded_files = st.sidebar.file_uploader("Upload your holdings CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=True)

# Normalize tickers and auto-clean
def clean_portfolio(df):
    df.columns = [col.strip().capitalize() for col in df.columns]
    df = df[df['Ticker'].str.upper() != "CUR:USD"]
    df['Ticker'] = df['Ticker'].str.replace("CUR:GE", "GE")
    return df

# Merge multiple portfolios and warn about potential duplicates
if uploaded_files:
    dataframes = []
    tickers_seen = set()
    duplicate_tickers = set()

    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df = clean_portfolio(df)
        df.columns = [col.strip().capitalize() for col in df.columns]
        if 'Ticker' in df.columns:
            dup = set(df['Ticker'].unique()).intersection(tickers_seen)
            duplicate_tickers.update(dup)
            tickers_seen.update(df['Ticker'].unique())
            dataframes.append(df)

    if duplicate_tickers:
        with st.expander("⚠️ Potential duplicate tickers found", expanded=True):
            st.warning(f"The following tickers appear in multiple uploaded files: {', '.join(duplicate_tickers)}")

    df = pd.concat(dataframes, ignore_index=True)
    df.columns = [col.strip().capitalize() for col in df.columns]
    required_columns = ["Ticker", "Quantity"]

    if all(col in df.columns for col in required_columns):
        st.success("Holdings file(s) loaded successfully!")

        # Performance section
        st.subheader(f"📊 Portfolio vs Benchmarks ({selected_period})")
        period = period_map[selected_period]
        benchmark_data = {}
        benchmark_series = []
        portfolio_change = None

        for label, symbol in comparison_tickers.items():
            try:
                hist = yf.Ticker(symbol).history(period=period)
                if not hist.empty:
                    norm_price = hist["Close"] / hist["Close"].iloc[0] * 100
                    pct_change = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                    benchmark_data[label] = pct_change
                    benchmark_series.append(pd.DataFrame({
                        "Date": hist.index,
                        "Normalized Price": norm_price,
                        "Index": label
                    }))
                else:
                    benchmark_data[label] = None
            except Exception:
                benchmark_data[label] = None

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
                sector = info.get("sector", "Cryptocurrency" if "-USD" in ticker else "Unknown")
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
            selected_accounts = st.sidebar.multiselect("Filter by account(s):", accounts, default=accounts)
            df = df[df["Account"].isin(selected_accounts)]

        # Styled performance grid
        st.subheader("🔁 Performance Summary")
        col1, col2 = st.columns(2)
        with col1:
            delta_color = "normal" if portfolio_change is None else ("inverse" if portfolio_change < 0 else "normal")
            st.metric("📦 Portfolio", f"{portfolio_change:+.2f}%" if portfolio_change is not None else "N/A", delta_color=delta_color)
        with col2:
            for idx, label in enumerate(comparison_tickers):
                change = benchmark_data.get(label)
                if change is not None:
                    icon = "⬆️" if change >= 0 else "⬇️"
                    st.metric(f"📊 {label}", f"{change:+.2f}% {icon}")
                else:
                    st.metric(f"📊 {label}", "N/A")

        # Normalized benchmark chart
        if benchmark_series:
            combined_df = pd.concat(benchmark_series)
            fig = px.line(combined_df, x="Date", y="Normalized Price", color="Index", title="Normalized Benchmark Performance")
            st.plotly_chart(fig, use_container_width=True)

        # Portfolio overview
        st.subheader("📊 Portfolio Overview")
        st.dataframe(df)

        total_value = df["Market Value"].sum()
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            sector_group = df.groupby("Sector")["Market Value"].sum().reset_index()
            fig_sector = px.pie(sector_group, values="Market Value", names="Sector", title="By Sector")
            st.plotly_chart(fig_sector, use_container_width=True)

        with col2:
            if "Asset Class" in df.columns:
                asset_group = df.groupby("Asset Class")["Market Value"].sum().reset_index()
                fig_asset = px.pie(asset_group, values="Market Value", names="Asset Class", title="By Asset Class")
                st.plotly_chart(fig_asset, use_container_width=True)

        with col3:
            if "Account" in df.columns:
                account_group = df.groupby("Account")["Market Value"].sum().reset_index()
                fig_account = px.pie(account_group, values="Market Value", names="Account", title="By Account")
                st.plotly_chart(fig_account, use_container_width=True)
else:
    st.info("Upload at least one CSV or Excel portfolio file to get started.")
