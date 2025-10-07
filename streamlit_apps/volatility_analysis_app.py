
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

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    """Loads the Kronos model and tokenizer."""
    with st.spinner("Loading Kronos model and tokenizer... This may take a moment."):
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=1000)
    return predictor

def calculate_volatility(df, window=30):
    """
    Calculates the rolling volatility of a stock.
    """
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252) # Annualized volatility

def plot_volatility_analysis(historical_df, ground_truth_df, prediction_df, historical_vol, predicted_vol, ticker):
    """
    Plots the price prediction and the volatility analysis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Plot 1: Price Prediction
    ax1.plot(historical_df.index, historical_df['close'], label='Historical Data', color='blue')
    ax1.plot(ground_truth_df.index, ground_truth_df['close'], label='Ground Truth', color='green')
    ax1.plot(prediction_df.index, prediction_df['close'], label='Prediction', color='red', linestyle='--')
    ax1.set_title(f'{ticker} Stock Price Prediction')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Volatility
    ax2.plot(historical_vol.index, historical_vol, label='Historical Volatility', color='blue')
    ax2.plot(predicted_vol.index, predicted_vol, label='Predicted Volatility', color='orange', linestyle='--')
    ax2.set_title(f'{ticker} 30-Day Rolling Volatility (Annualized)')
    ax2.set_ylabel('Volatility')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Stock Volatility Analysis with Kronos")

    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", "TSLA").upper()
    
    # Date range selection
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365) # Default to 3 years of data
    date_range = st.sidebar.date_input("Select Date Range for Data", [start_date_default, end_date])
    
    start_date = date_range[0]
    end_date = date_range[1] if len(date_range) > 1 else datetime.today()

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365, key="vol_lookback")
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90, key="vol_pred_len")
    sample_count = st.sidebar.slider("Sample Count for Prediction", 1, 20, 5, key="vol_sample_count")
    volatility_window = st.sidebar.slider("Volatility Window (days)", 10, 90, 30, key="vol_window")

    if st.sidebar.button("Run Volatility Analysis"):
        predictor = load_model()

        # ### 2. Fetch Data
        st.write(f"Fetching stock data for **{ticker}** from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if data.empty:
                st.error(f"No data found for ticker {ticker}. Please check the ticker or date range.")
                return
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

        # Prepare data
        data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        data['amount'] = data['close'] * data['volume']
        data['timestamps'] = data.index
        st.write(f"Data for {ticker} fetched. Shape: {data.shape}")

        if len(data) < lookback + pred_len:
            st.warning(f"Not enough data for the selected lookback ({lookback}) and prediction ({pred_len}) periods. Need at least {lookback + pred_len} data points, but got {len(data)}. Please select a larger date range or smaller periods.")
            return

        # ### 3. Make Prediction
        st.write("\nMaking predictions...")
        
        x_df = data.iloc[-lookback-pred_len:-pred_len]
        y_df_ground_truth = data.iloc[-pred_len:]

        x_timestamp = x_df['timestamps']
        y_timestamp = y_df_ground_truth['timestamps']

        with st.spinner("Running predictions..."):
            pred_df = predictor.predict(
                df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0, top_p=0.9, sample_count=sample_count,
                verbose=False
            )
        st.write("Prediction complete.")

        # ### 4. Analyze and Visualize Volatility
        st.write("\nAnalyzing and visualizing volatility...")
        
        # Calculate volatility
        historical_volatility = calculate_volatility(pd.concat([x_df, y_df_ground_truth]), window=volatility_window)
        
        # For predicted volatility, we need some historical data to start the rolling window
        combined_for_pred_vol = pd.concat([x_df.iloc[-volatility_window:], pred_df])
        predicted_volatility = calculate_volatility(combined_for_pred_vol, window=volatility_window)


        st.subheader(f"Volatility Analysis for {ticker}")
        plot_volatility_analysis(x_df, y_df_ground_truth, pred_df, historical_volatility, predicted_volatility, ticker)

        # Display volatility metrics
        st.subheader("Volatility Metrics")
        avg_hist_vol = historical_volatility.mean()
        avg_pred_vol = predicted_volatility.mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Average Historical Volatility", f"{avg_hist_vol:.4f}")
        col2.metric("Average Predicted Volatility", f"{avg_pred_vol:.4f}")

        st.write("Note: Volatility is the annualized standard deviation of daily logarithmic returns.")


if __name__ == "__main__":
    main()
