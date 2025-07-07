import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.title("🚀 VWAP & SMA Crypto Strategy Backtest (KuCoin)")

# --------------------------
# Initialize Exchange
# --------------------------
@st.cache_resource
def init_exchange():
    exchange = ccxt.kucoin({
        'rateLimit': True,
        'enableRateLimit': True,
    })
    return exchange

exchange = init_exchange()

# --------------------------
# Get Trading Pairs
# --------------------------
@st.cache_data(ttl=3600)
def get_trading_pairs(_exchange):
    markets = _exchange.load_markets()
    usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
    popular = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'BNB/USDT', 'MATIC/USDT', 'AVAX/USDT']
    sorted_pairs = [p for p in popular if p in usdt_pairs]
    remaining = list(set(usdt_pairs) - set(sorted_pairs))
    sorted_pairs += sorted(remaining)
    return sorted_pairs[:100]

trading_pairs = get_trading_pairs(exchange)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("🔧 Strategy Settings")
ticker = st.sidebar.selectbox("Select Pair", trading_pairs, index=0)
timeframe = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '4h'], index=1)
sma_period = st.sidebar.selectbox("SMA Period", [10, 20, 50, 100], index=1)
vwap_hours = st.sidebar.selectbox("VWAP Period (hours)", [1, 4, 12, 24], index=2)

profit_target = st.sidebar.number_input("Profit Target (%)", value=1.5, min_value=0.5, max_value=20.0, step=0.1)
stop_loss = st.sidebar.number_input("Stop Loss (%)", value=1.0, min_value=0.5, max_value=20.0, step=0.1)
initial_capital = st.sidebar.number_input("Initial Capital (USDT)", value=10000)
risk_per_trade = st.sidebar.number_input("Risk per Trade (%)", value=2.0, min_value=0.5, max_value=10.0)

max_trades_per_day = st.sidebar.slider("Max Trades Per Day", 1, 20, 5)

st.sidebar.header("🛡️ Filters")
use_rsi = st.sidebar.checkbox("Use RSI Filter", value=True)
use_volume = st.sidebar.checkbox("Use Volume Spike Filter", value=True)
use_volatility = st.sidebar.checkbox("Use Volatility Filter", value=True)

# --------------------------
# Fetch data
# --------------------------
def fetch_data(exchange, symbol, tf, limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    return df

# --------------------------
# Indicators
# --------------------------
def calculate_indicators(df, sma_period, vwap_hours, tf):
    df[f'SMA{sma_period}'] = df['Close'].rolling(window=sma_period).mean()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['price_volume'] = typical_price * df['Volume']
    window_size = get_vwap_window(tf, vwap_hours)
    df['rolling_pv'] = df['price_volume'].rolling(window=window_size).sum()
    df['rolling_volume'] = df['Volume'].rolling(window=window_size).sum()
    df['VWAP'] = df['rolling_pv'] / df['rolling_volume']

    df['RSI'] = calculate_rsi(df['Close'])
    df['volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['volume_spike'] = df['Volume'] > (df['volume_ma'] * 1.5)
    return df

def get_vwap_window(tf, hours):
    tf_map = {'1m': 60, '5m': 12, '15m': 4, '1h': 1, '4h': max(1, hours // 4)}
    if tf in tf_map:
        return max(1, hours * tf_map[tf])
    else:
        return hours

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --------------------------
# Generate signals
# --------------------------
def generate_signals(df, sma_period, use_rsi, use_volume, use_volatility):
    if df.empty:
        return df
    sma_col = f'SMA{sma_period}'
    df['prev_vwap'] = df['VWAP'].shift(1)
    df['prev_sma'] = df[sma_col].shift(1)
    base_signal = (
        (df['VWAP'] > df[sma_col]) &
        (df['prev_vwap'] <= df['prev_sma'])
    )
    filters = base_signal
    if use_rsi:
        filters = filters & (df['RSI'] > 30) & (df['RSI'] < 70)
    if use_volume:
        filters = filters & (df['volume_spike'] == True)
    if use_volatility:
        filters = filters & (df['volatility'] < 0.05)

    df['signal'] = np.where(filters, 1, 0)
    return df

# --------------------------
# Backtest simulation
# --------------------------
def simulate_trades(df):
    trades = []
    capital = initial_capital
    daily_trades = {}

    for idx, row in df[df['signal'] == 1].iterrows():
        date = idx.date()
        if date not in daily_trades:
            daily_trades[date] = 0
        if daily_trades[date] >= max_trades_per_day:
            continue

        entry_price = row['Close']
        risk_amount = capital * (risk_per_trade / 100)
        stop_loss_amount = entry_price * (stop_loss / 100)
        if stop_loss_amount <= 0:
            continue
        position_size = risk_amount / stop_loss_amount
        position_value = position_size * entry_price

        exit_found = False
        future_df = df.loc[idx:].iloc[1:]
        for e_idx, e_row in future_df.iterrows():
            change_pct = (e_row['Close'] - entry_price) / entry_price * 100
            if change_pct >= profit_target:
                exit_price = e_row['Close']
                exit_time = e_idx
                exit_found = True
                break
            elif change_pct <= -stop_loss:
                exit_price = e_row['Close']
                exit_time = e_idx
                exit_found = True
                break
        if not exit_found:
            exit_price = future_df['Close'].iloc[-1] if len(future_df) > 0 else entry_price
            exit_time = future_df.index[-1] if len(future_df) > 0 else idx

        pnl = (exit_price - entry_price) * position_size
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        capital += pnl
        fee = (position_value * 0.001) * 2
        capital -= fee

        trades.append({
            'Entry Time': idx,
            'Entry Price': round(entry_price, 4),
            'Exit Time': exit_time,
            'Exit Price': round(exit_price, 4),
            'P&L (%)': round(pnl_pct, 2),
            'P&L ($)': round(pnl, 2),
            'Capital': round(capital, 2),
        })
        daily_trades[date] += 1

    return pd.DataFrame(trades), capital

# --------------------------
# Performance metrics
# --------------------------
def get_metrics(trades_df, start_capital, end_capital):
    if trades_df.empty:
        st.warning("⚠️ No trades were executed. Adjust your filters or parameters.")
        return {}
    wins = trades_df[trades_df['P&L ($)'] > 0]
    losses = trades_df[trades_df['P&L ($)'] <= 0]
    win_rate = len(wins) / len(trades_df) * 100
    total_return = (end_capital - start_capital) / start_capital * 100
    return {
        'Total Trades': len(trades_df),
        'Wins': len(wins),
        'Losses': len(losses),
        'Win Rate (%)': round(win_rate, 2),
        'Total Return (%)': round(total_return, 2),
        'Final Capital ($)': round(end_capital, 2),
    }

# --------------------------
# Run
# --------------------------
if st.button("🚀 Run Backtest"):
    st.write("Fetching data...")
    df = fetch_data(exchange, ticker, timeframe)
    st.success(f"Data loaded: {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    df = calculate_indicators(df, sma_period, vwap_hours, timeframe)
    df = generate_signals(df, sma_period, use_rsi, use_volume, use_volatility)

    trades_df, final_cap = simulate_trades(df)
    metrics = get_metrics(trades_df, initial_capital, final_cap)

    if metrics:
        st.subheader("📊 Performance Metrics")
        st.json(metrics)

        st.subheader("💰 Trade Details")
        st.dataframe(trades_df)

        st.subheader("📉 Chart")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df['Close'], label='Close', color='blue', linewidth=1)
        ax.plot(df.index, df['VWAP'], label='VWAP', color='orange')
        ax.plot(df.index, df[f'SMA{sma_period}'], label=f'SMA{sma_period}', color='green')
        signals = df[df['signal'] == 1]
        ax.scatter(signals.index, signals['Close'], marker='^', color='red', s=100, label='Signal')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** For educational purposes only. Not financial advice.")