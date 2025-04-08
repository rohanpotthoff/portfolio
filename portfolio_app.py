import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

# Set page config
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

st.title("📈 Portfolio Tracker Dashboard")

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

        # Fetch current prices and sectors
        tickers = df["Ticker"].unique().tolist()
        data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                price = info.get("currentPrice", None)
                sector = info.get("sector", "Unknown")
                data.append({"Ticker": ticker, "Current Price": price, "Sector": sector})
            except Exception:
                data.append({"Ticker": ticker, "Current Price": None, "Sector": "Unknown"})

        price_df = pd.DataFrame(data)
        df = df.merge(price_df, on="Ticker", how="left")
        df["Market Value"] = df["Quantity"] * df["Current Price"]

        st.subheader("📊 Portfolio Overview")
        st.dataframe(df)

        # Summary stats
        total_value = df["Market Value"].sum()
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")

        # Pie chart by sector
        st.subheader("Sector Allocation")
        sector_group = df.groupby("Sector")["Market Value"].sum().reset_index()
        fig_sector = px.pie(sector_group, values="Market Value", names="Sector", title="Allocation by Sector")
        st.plotly_chart(fig_sector, use_container_width=True)

        # Optional: Allocation by Asset Class
        if "Asset Class" in df.columns:
            st.subheader("Asset Class Allocation")
            asset_group = df.groupby("Asset Class")["Market Value"].sum().reset_index()
            fig_asset = px.pie(asset_group, values="Market Value", names="Asset Class", title="Allocation by Asset Class")
            st.plotly_chart(fig_asset, use_container_width=True)

        # Optional: Allocation by Account
        if "Account" in df.columns:
            st.subheader("Account Allocation")
            account_group = df.groupby("Account")["Market Value"].sum().reset_index()
            fig_account = px.pie(account_group, values="Market Value", names="Account", title="Allocation by Account")
            st.plotly_chart(fig_account, use_container_width=True)

    else:
        st.error("Your file must contain at least 'Ticker' and 'Quantity' columns.")
else:
    st.info("Upload a CSV or Excel file with at least 'Ticker' and 'Quantity' columns to get started.")
