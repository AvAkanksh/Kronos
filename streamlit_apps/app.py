
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import yfinance as yf
from datetime import datetime, timedelta

# To import from parent directory
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

st.set_page_config(layout="wide", page_title="Kronos Financial Analysis Suite")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the Kronos model and tokenizer."""
    with st.spinner("Loading Kronos model and tokenizer... This may take a moment."):
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=1000)
    return predictor

# --- Helper Functions ---
def calculate_volatility(df, window):
    """Calculates annualized rolling volatility."""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)

# --- UI Pages ---

def render_exploratory_page(predictor):
    st.header("Exploratory Stock Prediction")

    st.sidebar.subheader("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365)
    date_range = st.sidebar.date_input("Select Date Range", [start_date_default, end_date], key="exp_date")
    start_date, end_date = date_range if len(date_range) > 1 else (date_range[0], datetime.today())

    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365, key="exp_lookback")
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90, key="exp_pred_len")
    temperature = st.sidebar.slider("Temperature (T)", 0.1, 2.0, 1.0, 0.1, key="exp_temp")
    top_p = st.sidebar.slider("Top-p Sampling", 0.1, 1.0, 0.9, 0.05, key="exp_topp")
    sample_count = st.sidebar.slider("Sample Count for Averaging", 1, 20, 1, key="exp_sample_count")

    if st.sidebar.button("Run Prediction"):
        st.write(f"Fetching stock data for **{ticker}**...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if data.empty:
                st.error(f"No data found for ticker {ticker}.")
                return
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

        data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        data['amount'] = data['close'] * data['volume']
        data['timestamps'] = data.index
        st.write(f"Data for {ticker} fetched. Shape: {data.shape}")
        
        if len(data) < lookback + pred_len:
            st.warning(f"Not enough data. Need {lookback + pred_len} points, got {len(data)}.")
            return

        x_df = data.iloc[-lookback-pred_len:-pred_len]
        y_df_ground_truth = data.iloc[-pred_len:]
        x_timestamp, y_timestamp = x_df['timestamps'], y_df_ground_truth['timestamps']

        with st.spinner("Running prediction..."):
            pred_df = predictor.predict(
                df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=pred_len,
                T=temperature, top_p=top_p, sample_count=sample_count, verbose=False
            )
        st.write("Prediction complete.")

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(x_df.index, x_df['close'], label='Historical Data', color='blue')
        ax.plot(y_df_ground_truth.index, y_df_ground_truth['close'], label='Ground Truth', color='green')
        ax.plot(pred_df.index, pred_df['close'], label='Prediction', color='red', linestyle='--')
        ax.set_title(f'{ticker} Stock Price Prediction')
        ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)'); ax.legend(); ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.subheader("Forecasted Data")
        st.dataframe(pred_df)

def render_anomaly_page(predictor):
    st.header("Stock Anomaly Detection")

    st.sidebar.subheader("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", "GOOG").upper()
    
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365)
    date_range = st.sidebar.date_input("Select Date Range", [start_date_default, end_date], key="anom_date")
    start_date, end_date = date_range if len(date_range) > 1 else (date_range[0], datetime.today())

    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365, key="anom_lookback")
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90, key="anom_pred_len")
    sample_count = st.sidebar.slider("Sample Count for CI", 5, 50, 20, key="anom_sample")
    confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95, key="anom_conf")
    
    z_score = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}.get(confidence_level, 1.96)

    if st.sidebar.button("Run Anomaly Detection"):
        st.write(f"Fetching stock data for **{ticker}**...")
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        if data.empty:
            st.error(f"No data found for ticker {ticker}.")
            return

        data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        data['amount'] = data['close'] * data['volume']
        data['timestamps'] = data.index
        
        if len(data) < lookback + pred_len:
            st.warning(f"Not enough data. Need {lookback + pred_len}, got {len(data)}.")
            return

        x_df = data.iloc[-lookback-pred_len:-pred_len]
        y_df_ground_truth = data.iloc[-pred_len:]
        x_timestamp, y_timestamp = x_df['timestamps'], y_df_ground_truth['timestamps']

        progress_bar = st.progress(0); status_text = st.empty()
        all_preds = []
        for i in range(sample_count):
            status_text.text(f"Running sample {i+1}/{sample_count}")
            pred_df_sample = predictor.predict(
                df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=pred_len,
                T=1.1, top_p=0.95, sample_count=1, verbose=False
            )
            all_preds.append(pred_df_sample['close'].values)
            progress_bar.progress((i + 1) / sample_count)
        status_text.text("Prediction complete.")

        pred_matrix = np.array(all_preds)
        pred_mean = pred_matrix.mean(axis=0)
        pred_std = pred_matrix.std(axis=0)
        pred_upper = pred_mean + z_score * pred_std
        pred_lower = pred_mean - z_score * pred_std
        pred_df = pd.DataFrame({'close': pred_mean, 'upper_bound': pred_upper, 'lower_bound': pred_lower}, index=y_timestamp)

        anomalies = y_df_ground_truth[(y_df_ground_truth['close'] > pred_df['upper_bound']) | (y_df_ground_truth['close'] < pred_df['lower_bound'])]

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(x_df.index, x_df['close'], label='Historical Data', color='blue')
        ax.plot(y_df_ground_truth.index, y_df_ground_truth['close'], label='Actual Price', color='green')
        ax.plot(pred_df.index, pred_df['close'], label='Predicted Mean', color='orange', linestyle='--')
        ax.fill_between(pred_df.index, pred_df['lower_bound'], pred_df['upper_bound'], color='orange', alpha=0.2, label=f'{confidence_level}% CI')
        if not anomalies.empty:
            ax.scatter(anomalies.index, anomalies['close'], color='red', s=50, zorder=5, label='Anomaly')
        ax.set_title(f'Anomaly Detection for {ticker}'); ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)'); ax.legend(); ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        if not anomalies.empty:
            st.subheader(f"Detected {len(anomalies)} Anomalies"); st.dataframe(anomalies)
        else:
            st.success("No anomalies detected.")

def render_volatility_page(predictor):
    st.header("Stock Volatility Analysis")

    st.sidebar.subheader("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", "TSLA").upper()
    
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365)
    date_range = st.sidebar.date_input("Select Date Range", [start_date_default, end_date], key="vol_date")
    start_date, end_date = date_range if len(date_range) > 1 else (date_range[0], datetime.today())

    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365, key="vol_lookback")
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90, key="vol_pred_len")
    sample_count = st.sidebar.slider("Sample Count", 1, 20, 5, key="vol_sample")
    volatility_window = st.sidebar.slider("Volatility Window (days)", 10, 90, 30, key="vol_window")

    if st.sidebar.button("Run Volatility Analysis"):
        st.write(f"Fetching stock data for **{ticker}**...")
        data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        if data.empty:
            st.error(f"No data found for ticker {ticker}."); return

        data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        data['amount'] = data['close'] * data['volume']
        data['timestamps'] = data.index
        
        if len(data) < lookback + pred_len:
            st.warning(f"Not enough data. Need {lookback + pred_len}, got {len(data)}."); return

        x_df = data.iloc[-lookback-pred_len:-pred_len]
        y_df_ground_truth = data.iloc[-pred_len:]
        x_timestamp, y_timestamp = x_df['timestamps'], y_df_ground_truth['timestamps']

        with st.spinner("Running prediction..."):
            pred_df = predictor.predict(
                df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=pred_len,
                T=1.0, top_p=0.9, sample_count=sample_count, verbose=False
            )
        st.write("Prediction complete.")

        historical_vol = calculate_volatility(pd.concat([x_df, y_df_ground_truth]), window=volatility_window)
        predicted_vol = calculate_volatility(pd.concat([x_df.iloc[-volatility_window:], pred_df]), window=volatility_window)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        ax1.plot(x_df.index, x_df['close'], label='Historical Data', color='blue')
        ax1.plot(y_df_ground_truth.index, y_df_ground_truth['close'], label='Ground Truth', color='green')
        ax1.plot(pred_df.index, pred_df['close'], label='Prediction', color='red', linestyle='--')
        ax1.set_title(f'{ticker} Stock Price Prediction'); ax1.set_ylabel('Price (USD)'); ax1.legend(); ax1.grid(True)
        
        ax2.plot(historical_vol.index, historical_vol, label='Historical Volatility', color='blue')
        ax2.plot(predicted_vol.index, predicted_vol, label='Predicted Volatility', color='orange', linestyle='--')
        ax2.set_title(f'{ticker} {volatility_window}-Day Rolling Volatility (Annualized)'); ax2.set_ylabel('Volatility'); ax2.set_xlabel('Date'); ax2.legend(); ax2.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        col1.metric("Average Historical Volatility", f"{historical_vol.mean():.4f}")
        col2.metric("Average Predicted Volatility", f"{predicted_vol.mean():.4f}")

def render_batch_page(predictor):
    st.header("Batch Stock Analysis")

    st.sidebar.subheader("Configuration")
    tickers_input = st.sidebar.text_area("Stock Tickers (comma-separated)", "AAPL, GOOG, TSLA, MSFT")
    
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365)
    date_range = st.sidebar.date_input("Select Date Range", [start_date_default, end_date], key="batch_date")
    start_date, end_date = date_range if len(date_range) > 1 else (date_range[0], datetime.today())

    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365, key="batch_lookback")
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90, key="batch_pred_len")
    sample_count = st.sidebar.slider("Sample Count (Anomaly/Pred)", 5, 50, 20, key="batch_sample")
    volatility_window = st.sidebar.slider("Volatility Window (days)", 10, 90, 30, key="batch_vol_window")

    if st.sidebar.button("Run Batch Analysis"):
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        summary_data = []

        for ticker in tickers:
            with st.expander(f"Analysis for {ticker}", expanded=True):
                st.write(f"----- Analyzing {ticker} -----")
                try:
                    st.write(f"Fetching data for {ticker}...")
                    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
                    if data.empty:
                        st.warning(f"No data for {ticker}. Skipping."); continue

                    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    data['amount'] = data['close'] * data['volume']
                    data['timestamps'] = data.index
                    
                    if len(data) < lookback + pred_len:
                        st.warning(f"Not enough data for {ticker}. Skipping."); continue

                    x_df = data.iloc[-lookback-pred_len:-pred_len]
                    y_df_ground_truth = data.iloc[-pred_len:]
                    x_timestamp, y_timestamp = x_df['timestamps'], y_df_ground_truth['timestamps']

                    st.write(f"Running prediction and anomaly detection for {ticker}...")
                    all_preds = []
                    progress_bar = st.progress(0)
                    for i in range(sample_count):
                        pred_df_sample = predictor.predict(
                            df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                            x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=pred_len,
                            T=1.1, top_p=0.95, sample_count=1, verbose=False
                        )
                        all_preds.append(pred_df_sample['close'].values)
                        progress_bar.progress((i + 1) / sample_count)

                    pred_matrix = np.array(all_preds)
                    pred_mean = pred_matrix.mean(axis=0)
                    pred_std = pred_matrix.std(axis=0)
                    pred_df = pd.DataFrame({
                        'close': pred_mean, 
                        'upper_bound': pred_mean + 1.96 * pred_std, 
                        'lower_bound': pred_mean - 1.96 * pred_std
                    }, index=y_timestamp)

                    anomalies = y_df_ground_truth[(y_df_ground_truth['close'] > pred_df['upper_bound']) | (y_df_ground_truth['close'] < pred_df['lower_bound'])]
                    
                    historical_vol = calculate_volatility(pd.concat([x_df, y_df_ground_truth]), window=volatility_window)
                    predicted_vol = calculate_volatility(pd.concat([x_df.iloc[-volatility_window:], pred_df]), window=volatility_window)

                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
                    ax1.plot(x_df.index, x_df['close'], label='Historical Data', color='blue')
                    ax1.plot(y_df_ground_truth.index, y_df_ground_truth['close'], label='Actual Price', color='green')
                    ax1.plot(pred_df.index, pred_df['close'], label='Predicted Mean', color='orange', linestyle='--')
                    ax1.fill_between(pred_df.index, pred_df['lower_bound'], pred_df['upper_bound'], color='orange', alpha=0.2, label='95% CI')
                    if not anomalies.empty:
                        ax1.scatter(anomalies.index, anomalies['close'], color='red', s=50, zorder=5, label='Anomaly')
                    ax1.set_title(f'{ticker} - Price Prediction & Anomaly Detection'); ax1.set_ylabel('Price (USD)'); ax1.legend(); ax1.grid(True)

                    ax2.plot(historical_vol.index, historical_vol, label='Historical Volatility', color='blue')
                    ax2.plot(predicted_vol.index, predicted_vol, label='Predicted Volatility', color='orange', linestyle='--')
                    ax2.set_title(f'{ticker} - {volatility_window}-Day Rolling Volatility'); ax2.set_ylabel('Volatility'); ax2.set_xlabel('Date'); ax2.legend(); ax2.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Anomalies Detected", len(anomalies))
                    col2.metric("Avg. Historical Volatility", f"{historical_vol.mean():.4f}")
                    col3.metric("Avg. Predicted Volatility", f"{predicted_vol.mean():.4f}")

                    summary_data.append({
                        "Ticker": ticker, "Anomalies": len(anomalies),
                        "Historical Volatility": historical_vol.mean(),
                        "Predicted Volatility": predicted_vol.mean()
                    })
                except Exception as e:
                    st.error(f"An error occurred while analyzing {ticker}: {e}")

        if summary_data:
            st.subheader("Batch Analysis Summary")
            st.dataframe(pd.DataFrame(summary_data))

# --- Main App ---
def main():
    st.sidebar.title("Kronos Analysis Suite")
    
    app_mode = st.sidebar.selectbox(
        "Choose the analysis tool",
        ["Exploratory Prediction", "Anomaly Detection", "Volatility Analysis", "Batch Analysis"]
    )

    predictor = load_model()

    if app_mode == "Exploratory Prediction":
        render_exploratory_page(predictor)
    elif app_mode == "Anomaly Detection":
        render_anomaly_page(predictor)
    elif app_mode == "Volatility Analysis":
        render_volatility_page(predictor)
    elif app_mode == "Batch Analysis":
        render_batch_page(predictor)

if __name__ == "__main__":
    main()
