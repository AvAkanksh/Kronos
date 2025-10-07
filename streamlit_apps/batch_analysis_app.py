
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import yfinance as yf
import os
from datetime import datetime, timedelta

# To import from parent directory
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    """Loads the Kronos model and tokenizer."""
    with st.spinner("Loading Kronos model and tokenizer... This may take a moment."):
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=1000)
    return predictor

def calculate_volatility(df, window):
    """Calculates annualized rolling volatility."""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)

def plot_stock_analysis(ticker, x_df, y_df_ground_truth, pred_df, anomalies, historical_vol, predicted_vol):
    """Plots the comprehensive analysis for a single stock."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Price Prediction and Anomaly Detection
    ax1.plot(x_df.index, x_df['close'], label='Historical Data', color='blue')
    ax1.plot(y_df_ground_truth.index, y_df_ground_truth['close'], label='Actual Price', color='green')
    ax1.plot(pred_df.index, pred_df['close'], label='Predicted Mean', color='orange', linestyle='--')
    ax1.fill_between(pred_df.index, pred_df['lower_bound'], pred_df['upper_bound'], color='orange', alpha=0.2, label='95% Confidence Interval')
    if not anomalies.empty:
        ax1.scatter(anomalies.index, anomalies['close'], color='red', s=50, zorder=5, label='Anomaly')
    ax1.set_title(f'{ticker} - Price Prediction and Anomaly Detection')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Volatility Analysis
    ax2.plot(historical_vol.index, historical_vol, label='Historical Volatility', color='blue')
    ax2.plot(predicted_vol.index, predicted_vol, label='Predicted Volatility', color='orange', linestyle='--')
    ax2.set_title(f'{ticker} - 30-Day Rolling Volatility (Annualized)')
    ax2.set_ylabel('Volatility')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("Batch Stock Analysis with Kronos")

    st.sidebar.header("Configuration")
    tickers_input = st.sidebar.text_area("Stock Tickers (comma-separated)", "AAPL, GOOG, TSLA, MSFT")
    
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365)
    date_range = st.sidebar.date_input("Select Date Range", [start_date_default, end_date])
    start_date, end_date = date_range

    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365)
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90)
    sample_count_anomaly = st.sidebar.slider("Sample Count (Anomaly/Prediction)", 5, 50, 20)
    volatility_window = st.sidebar.slider("Volatility Window (days)", 10, 90, 30)

    if st.sidebar.button("Run Batch Analysis"):
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        predictor = load_model()
        
        summary_data = []

        for ticker in tickers:
            with st.expander(f"Analysis for {ticker}", expanded=True):
                st.write(f"----- Analyzing {ticker} -----")
                
                try:
                    # 1. Fetch Data
                    st.write(f"Fetching data for {ticker}...")
                    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
                    if data.empty:
                        st.warning(f"Could not fetch data for {ticker}. Skipping.")
                        continue

                    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    data['amount'] = data['close'] * data['volume']
                    data['timestamps'] = data.index
                    
                    if len(data) < lookback + pred_len:
                        st.warning(f"Not enough data for {ticker} with the selected settings. Skipping.")
                        continue

                    x_df = data.iloc[-lookback-pred_len:-pred_len]
                    y_df_ground_truth = data.iloc[-pred_len:]
                    x_timestamp = x_df['timestamps']
                    y_timestamp = y_df_ground_truth['timestamps']

                    # 2. Perform Prediction and Anomaly Detection
                    st.write(f"Running prediction and anomaly detection for {ticker}...")
                    all_preds = []
                    progress_bar = st.progress(0)
                    for i in range(sample_count_anomaly):
                        pred_df_sample = predictor.predict(
                            df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                            x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=pred_len,
                            T=1.1, top_p=0.95, sample_count=1, verbose=False
                        )
                        all_preds.append(pred_df_sample['close'].values)
                        progress_bar.progress((i + 1) / sample_count_anomaly)

                    pred_matrix = np.array(all_preds)
                    pred_mean = pred_matrix.mean(axis=0)
                    pred_std = pred_matrix.std(axis=0)
                    pred_upper = pred_mean + 1.96 * pred_std
                    pred_lower = pred_mean - 1.96 * pred_std

                    pred_df = pd.DataFrame({'close': pred_mean, 'upper_bound': pred_upper, 'lower_bound': pred_lower}, index=y_timestamp)

                    # 3. Detect Anomalies
                    ground_truth_values = y_df_ground_truth['close'].values
                    anomalies = y_df_ground_truth[(ground_truth_values > pred_upper) | (ground_truth_values < pred_lower)]
                    
                    # 4. Analyze Volatility
                    historical_vol = calculate_volatility(pd.concat([x_df, y_df_ground_truth]), window=volatility_window)
                    predicted_vol = calculate_volatility(pd.concat([x_df.iloc[-volatility_window:], pred_df]), window=volatility_window)

                    # 5. Display Results
                    plot_stock_analysis(ticker, x_df, y_df_ground_truth, pred_df, anomalies, historical_vol, predicted_vol)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Anomalies Detected", len(anomalies))
                    col2.metric("Avg. Historical Volatility", f"{historical_vol.mean():.4f}")
                    col3.metric("Avg. Predicted Volatility", f"{predicted_vol.mean():.4f}")

                    if not anomalies.empty:
                        st.write("Detected Anomalies:")
                        st.dataframe(anomalies)
                    
                    summary_data.append({
                        "Ticker": ticker,
                        "Anomalies": len(anomalies),
                        "Historical Volatility": historical_vol.mean(),
                        "Predicted Volatility": predicted_vol.mean()
                    })

                except Exception as e:
                    st.error(f"An error occurred while analyzing {ticker}: {e}")

        if summary_data:
            st.subheader("Batch Analysis Summary")
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)

if __name__ == "__main__":
    main()
