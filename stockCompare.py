import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Functions (same as before)

def get_fundamental_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    bs = stock.balance_sheet

    if bs.empty:
        return None

    def safe_get(item):
        return bs.loc[item][0] if item in bs.index else 0

    current_assets = (
        safe_get("Cash") +
        safe_get("Inventory") +
        safe_get("Receivables") +
        safe_get("Other Current Assets")
    )
    current_liabilities = (
        safe_get("Current Liabilities") +
        safe_get("Accounts Payable") +
        safe_get("Other Current Liabilities")
    )

    if current_assets == 0 or current_liabilities == 0:
        return None

    current_ratio = current_assets / current_liabilities if current_liabilities != 0 else np.nan

    total_assets = safe_get("Total Assets")
    retained_earnings = safe_get("Retained Earnings")
    ebit = info.get("ebitda", np.nan)
    market_cap = info.get("marketCap", np.nan)
    total_liabilities = safe_get("Total Liab")
    sales = info.get("totalRevenue", np.nan)

    if not all([total_assets, retained_earnings, ebit, market_cap, total_liabilities, sales]):
        z_score = np.nan
    else:
        A = (current_assets - current_liabilities) / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / total_liabilities
        E = sales / total_assets
        z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E

    return {
        "current_ratio": current_ratio,
        "altman_z": z_score
    }


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")

    if df.empty:
        return None

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)

    ma_signal = "Bullish" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "Bearish"
    rsi_signal = "Overbought" if df["RSI"].iloc[-1] > 70 else ("Oversold" if df["RSI"].iloc[-1] < 30 else "Neutral")

    return {
        "MA20": df["MA20"].iloc[-1],
        "MA50": df["MA50"].iloc[-1],
        "RSI": df["RSI"].iloc[-1],
        "MA_signal": ma_signal,
        "RSI_signal": rsi_signal
    }


def get_sharpe_ratio(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")

    if df.empty:
        return np.nan

    returns = df["Close"].pct_change().dropna()
    excess_return = returns - 0.02 / 252
    sharpe = np.sqrt(252) * excess_return.mean() / excess_return.std()
    return sharpe


def generate_recommendation(z_score, sharpe, tech_signal):
    if z_score is not np.nan and z_score > 3 and sharpe > 1 and tech_signal == "Bullish":
        return "Buy"
    elif z_score is not np.nan and z_score < 1.8:
        return "Avoid"
    else:
        return "Watchlist"


def analyze_ticker(ticker):
    fund = get_fundamental_ratios(ticker)
    tech = get_technical_indicators(ticker)
    sharpe = get_sharpe_ratio(ticker)

    if fund is None or tech is None or np.isnan(sharpe):
        return None

    reco = generate_recommendation(fund["altman_z"], sharpe, tech["MA_signal"])

    return {
        "Current Ratio": round(fund["current_ratio"], 2),
        "Altman Z-Score": round(fund["altman_z"], 2) if fund["altman_z"] is not np.nan else "N/A",
        "Sharpe Ratio": round(sharpe, 2),
        "MA20": round(tech["MA20"], 2),
        "MA50": round(tech["MA50"], 2),
        "RSI": round(tech["RSI"], 2),
        "MA Signal": tech["MA_signal"],
        "RSI Signal": tech["RSI_signal"],
        "Recommendation": reco
    }


# ---------------- Streamlit App ----------------

st.title("📊 Stock Ticker Comparison Tool")

ticker1 = st.text_input("Enter first ticker symbol (e.g., GIII):", "GIII")
ticker2 = st.text_input("Enter second ticker symbol (e.g., LULU):", "LULU")

if st.button("Compare"):
    results = {}
    for tk in [ticker1, ticker2]:
        st.subheader(f"🔎 {tk}")
        data = analyze_ticker(tk)
        if data:
            st.dataframe(pd.DataFrame(data.items(), columns=["Metric", "Value"]))
        else:
            st.warning(f"Data incomplete or unavailable for {tk}.")