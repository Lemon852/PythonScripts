import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------
# Technical Analysis Functions
# ---------------------------------------------

def compute_rsi(series, period=14):
    """Compute RSI indicator"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_supertrend(df, period=10, multiplier=3):
    """Compute SuperTrend indicator"""
    hl2 = (df['High'] + df['Low']) / 2
    atr = compute_atr(df, period)
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] <= upper_band.iloc[i-1]:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        elif df['Close'].iloc[i] >= lower_band.iloc[i-1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
    
    return supertrend, direction


def compute_atr(df, period=14):
    """Compute Average True Range"""
    high_low = df['High'] - df['Low']
    high_close_prev = np.abs(df['High'] - df['Close'].shift())
    low_close_prev = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def get_technical_indicators(ticker):
    """Get technical indicators for a stock"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y", interval="1d")
        
        if df.empty or len(df) < 100:
            return None, None
        
        # Clean data
        df = df.dropna(subset=['Close', 'High', 'Low'])
        
        if df['Close'].min() <= 0:
            return None, None
        
        # Calculate technical indicators
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["RSI"] = compute_rsi(df["Close"], 14)
        df["SuperTrend"], df["ST_Direction"] = compute_supertrend(df)
        
        # Get current values
        current_price = df["Close"].iloc[-1]
        ma20 = df["MA20"].iloc[-1]
        ma50 = df["MA50"].iloc[-1]
        rsi = df["RSI"].iloc[-1]
        supertrend = df["SuperTrend"].iloc[-1]
        st_direction = df["ST_Direction"].iloc[-1]
        
        # Calculate returns
        returns_1d = ((current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100
        returns_7d = ((current_price - df["Close"].iloc[-8]) / df["Close"].iloc[-8]) * 100
        returns_30d = ((current_price - df["Close"].iloc[-31]) / df["Close"].iloc[-31]) * 100
        
        # Generate signals
        ma_signal = "Bullish" if ma20 > ma50 else "Bearish"
        
        if pd.notna(rsi):
            if rsi > 70:
                rsi_signal = "Overbought"
            elif rsi < 30:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"
        else:
            rsi_signal = "Unknown"
        
        st_signal = "Bullish" if st_direction == 1 else "Bearish"
        
        # Overall trend analysis
        bullish_signals = 0
        bearish_signals = 0
        
        if ma_signal == "Bullish":
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if st_signal == "Bullish":
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if rsi_signal == "Oversold":
            bullish_signals += 1
        elif rsi_signal == "Overbought":
            bearish_signals += 1
        
        overall_trend = "Bullish" if bullish_signals > bearish_signals else "Bearish"
        
        tech_dict = {
            "current_price": round(current_price, 2),
            "returns_1d": round(returns_1d, 2),
            "returns_7d": round(returns_7d, 2),
            "returns_30d": round(returns_30d, 2),
            "MA20": round(ma20, 2) if pd.notna(ma20) else "N/A",
            "MA50": round(ma50, 2) if pd.notna(ma50) else "N/A",
            "RSI": round(rsi, 2) if pd.notna(rsi) else "N/A",
            "SuperTrend": round(supertrend, 2) if pd.notna(supertrend) else "N/A",
            "MA_signal": ma_signal,
            "RSI_signal": rsi_signal,
            "ST_signal": st_signal,
            "overall_trend": overall_trend
        }
        
        return tech_dict, df
    
    except Exception as e:
        print(f"Error getting technical data for {ticker}: {e}")
        return None, None


def predict_future_trend(df, days=30):
    """Predict future price trend using linear regression"""
    try:
        # Prepare data for prediction
        df_clean = df.dropna(subset=['Close'])
        
        if len(df_clean) < 50:
            return None, None
        
        # Use last 60 days for prediction
        recent_data = df_clean.tail(60)
        
        # Create features (days as numbers)
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data['Close'].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future prices
        future_X = np.arange(len(recent_data), len(recent_data) + days).reshape(-1, 1)
        future_prices = model.predict(future_X)
        
        # Create future dates
        last_date = recent_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        # Calculate expected return
        current_price = recent_data['Close'].iloc[-1]
        future_price = future_prices[-1]
        expected_return = ((future_price - current_price) / current_price) * 100
        
        return future_dates, future_prices, expected_return
    
    except Exception as e:
        print(f"Error predicting trend: {e}")
        return None, None, None


def get_trend_color(trend):
    """Get color for trend display"""
    return "green" if trend == "Bullish" else "red"


def analyze_ticker(ticker):
    """Analyze a single ticker - Technical Analysis Only"""
    try:
        tech, df = get_technical_indicators(ticker)
        if tech is None or df is None:
            return None, None, None, None, None
        
        # Get prediction
        future_dates, future_prices, expected_return = predict_future_trend(df)
        
        return tech, df, future_dates, future_prices, expected_return
    
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None, None, None, None, None


# ---------------------------------------------
# Streamlit App
# ---------------------------------------------

st.set_page_config(page_title="Technical Analysis Tool", layout="wide")

# Header
st.title("📈 Technical Analysis & Trend Prediction")
st.markdown("**Focus on Technical Indicators Only** - Clean, Simple, Powerful")

# Sidebar with expandable technical explanations
with st.sidebar:
    st.header("📚 Technical Indicators Guide")
    
    with st.expander("🔄 Moving Averages (MA)"):
        st.markdown("""
        **MA20 vs MA50 Signal:**
        - 🟢 **Bullish**: MA20 > MA50 (Golden Cross)
        - 🔴 **Bearish**: MA20 < MA50 (Death Cross)
        
        **What it means:**
        - Short-term trend vs long-term trend
        - Price above MA = Uptrend
        - Price below MA = Downtrend
        """)
    
    with st.expander("📊 RSI (Relative Strength Index)"):
        st.markdown("""
        **RSI Levels:**
        - 🔴 **Overbought**: RSI > 70 (Sell Signal)
        - 🟢 **Oversold**: RSI < 30 (Buy Signal)
        - 🟡 **Neutral**: RSI 30-70 (Hold)
        
        **What it means:**
        - Measures momentum
        - Overbought = Potential price drop
        - Oversold = Potential price rise
        - Divergence can signal trend reversal
        """)
    
    with st.expander("⚡ SuperTrend Indicator"):
        st.markdown("""
        **SuperTrend Signals:**
        - 🟢 **Bullish**: Price above SuperTrend line
        - 🔴 **Bearish**: Price below SuperTrend line
        
        **What it means:**
        - Trend-following indicator
        - Uses volatility (ATR) to set levels
        - Green line = Buy and hold
        - Red line = Sell or avoid
        - Very reliable for trending markets
        """)
    
    with st.expander("🎯 Overall Trend Analysis"):
        st.markdown("""
        **How we determine trend:**
        - Combines all 3 indicators
        - MA Signal + SuperTrend + RSI
        - Majority vote determines trend
        
        **Confidence levels:**
        - 3/3 signals = Very High
        - 2/3 signals = Medium
        - 1/3 signals = Low
        """)
    
    with st.expander("📈 Future Trend Prediction"):
        st.markdown("""
        **Prediction Method:**
        - Uses last 60 days of data
        - Linear regression analysis
        - Projects 30 days ahead
        
        **Important Notes:**
        - ⚠️ Not financial advice
        - Based on historical patterns
        - Market can be unpredictable
        - Use with other analysis
        """)

# Main input section
col_input1, col_input2 = st.columns(2)
with col_input1:
    ticker1 = st.text_input("Enter first ticker symbol:", "AAPL")
with col_input2:
    ticker2 = st.text_input("Enter second ticker symbol:", "TSLA")

if st.button("🔍 Analyze Technical Indicators", type="primary"):
    if ticker1 and ticker2:
        col1, col2 = st.columns(2)
        
        for ticker, col in zip([ticker1.upper(), ticker2.upper()], [col1, col2]):
            with col:
                st.subheader(f"📊 {ticker}")
                
                with st.spinner(f"Analyzing {ticker}..."):
                    tech, df, future_dates, future_prices, expected_return = analyze_ticker(ticker)
                
                if tech and df is not None:
                    # Overall trend with color
                    trend_color = get_trend_color(tech["overall_trend"])
                    st.markdown(
                        f"<div style='background-color:{trend_color}; color:white; padding:10px; border-radius:5px; text-align:center; font-weight:bold; margin-bottom:10px;'>"
                        f"📈 Overall Trend: {tech['overall_trend']}</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Key metrics in clean format
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Current Price", f"${tech['current_price']}", f"{tech['returns_1d']:+.2f}%")
                        st.metric("MA20", f"${tech['MA20']}", delta=None)
                        st.metric("RSI", tech["RSI"], delta=None)
                    
                    with col_b:
                        st.metric("SuperTrend", f"${tech['SuperTrend']}", delta=None)
                        st.metric("7D Return", f"{tech['returns_7d']:+.2f}%", delta=None)
                        st.metric("30D Return", f"{tech['returns_30d']:+.2f}%", delta=None)
                    
                    # Signal summary
                    st.subheader("🎯 Signal Summary")
                    signals_df = pd.DataFrame({
                        "Indicator": ["Moving Average", "RSI", "SuperTrend"],
                        "Signal": [tech["MA_signal"], tech["RSI_signal"], tech["ST_signal"]]
                    })
                    st.dataframe(signals_df, use_container_width=True, hide_index=True)
                    
                    # Price chart with technical indicators
                    st.subheader("📉 Technical Analysis Chart")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
                    
                    # Price chart
                    recent_df = df.tail(120)  # Last 6 months
                    ax1.plot(recent_df.index, recent_df["Close"], label="Close", color="black", linewidth=2)
                    ax1.plot(recent_df.index, recent_df["MA20"], label="MA20", color="blue", alpha=0.7)
                    ax1.plot(recent_df.index, recent_df["MA50"], label="MA50", color="orange", alpha=0.7)
                    
                    # SuperTrend
                    bullish_st = recent_df["SuperTrend"].where(recent_df["ST_Direction"] == 1)
                    bearish_st = recent_df["SuperTrend"].where(recent_df["ST_Direction"] == -1)
                    ax1.plot(recent_df.index, bullish_st, label="SuperTrend (Bullish)", color="green", linewidth=2)
                    ax1.plot(recent_df.index, bearish_st, label="SuperTrend (Bearish)", color="red", linewidth=2)
                    
                    ax1.set_title(f"{ticker} - Technical Analysis", fontsize=14, fontweight='bold')
                    ax1.set_ylabel("Price ($)")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # RSI chart
                    ax2.plot(recent_df.index, recent_df["RSI"], color="purple", linewidth=2)
                    ax2.axhline(70, color="red", linestyle="--", alpha=0.7)
                    ax2.axhline(30, color="green", linestyle="--", alpha=0.7)
                    ax2.axhline(50, color="gray", linestyle="-", alpha=0.3)
                    ax2.set_ylim([0, 100])
                    ax2.set_title("RSI (14-day)", fontsize=12)
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("RSI")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Future trend prediction
                    if future_dates is not None and future_prices is not None:
                        st.subheader("🔮 30-Day Trend Prediction")
                        
                        # Show expected return
                        return_color = "green" if expected_return > 0 else "red"
                        st.markdown(
                            f"<div style='background-color:{return_color}; color:white; padding:8px; border-radius:5px; text-align:center; margin-bottom:10px;'>"
                            f"Expected 30-Day Return: {expected_return:+.2f}%</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Prediction chart
                        fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
                        
                        # Historical data (last 60 days)
                        historical = df.tail(60)
                        ax_pred.plot(historical.index, historical["Close"], 
                                   label="Historical Price", color="black", linewidth=2)
                        
                        # Predicted data
                        ax_pred.plot(future_dates, future_prices, 
                                   label="Predicted Price", color="red", linewidth=2, linestyle="--")
                        
                        # Connect last historical point to first predicted point
                        ax_pred.plot([historical.index[-1], future_dates[0]], 
                                   [historical["Close"].iloc[-1], future_prices[0]], 
                                   color="red", linewidth=2, linestyle="--")
                        
                        ax_pred.set_title(f"{ticker} - 30-Day Price Prediction", fontsize=14, fontweight='bold')
                        ax_pred.set_xlabel("Date")
                        ax_pred.set_ylabel("Price ($)")
                        ax_pred.legend()
                        ax_pred.grid(True, alpha=0.3)
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_pred)
                        
                        # Prediction details
                        st.info(f"📊 **Prediction Summary:**\n"
                               f"- Current Price: ${tech['current_price']}\n"
                               f"- Predicted Price (30 days): ${future_prices[-1]:.2f}\n"
                               f"- Expected Return: {expected_return:+.2f}%")
                    
                else:
                    st.error(f"❌ Unable to analyze {ticker}. Please check the ticker symbol.")
    else:
        st.warning("⚠️ Please enter both ticker symbols.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>
⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
Past performance does not guarantee future results. Always consult with a financial advisor.
</small>
</div>
""", unsafe_allow_html=True)