
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

def plot_anomaly_detection(x_df, y_df_ground_truth, pred_df, anomalies, ticker):
    """
    Plots the anomaly detection results.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot historical data
    ax.plot(x_df.index, x_df['close'], label='Historical Data', color='blue')
    
    # Plot ground truth for the prediction period
    ax.plot(y_df_ground_truth.index, y_df_ground_truth['close'], label='Actual Price', color='green')
    
    # Plot predicted mean
    ax.plot(pred_df.index, pred_df['close'], label='Predicted Mean', color='orange', linestyle='--')
    
    # Plot confidence interval
    ax.fill_between(pred_df.index, pred_df['lower_bound'], pred_df['upper_bound'], color='orange', alpha=0.2, label='95% Confidence Interval')
    
    # Highlight anomalies
    if not anomalies.empty:
        ax.scatter(anomalies.index, anomalies['close'], color='red', s=50, zorder=5, label='Anomaly')
        
    ax.set_title(f'Anomaly Detection for {ticker}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (USD)')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Stock Anomaly Detection with Kronos")

    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", "GOOG").upper()
    
    # Date range selection
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365) # Default to 3 years of data
    date_range = st.sidebar.date_input("Select Date Range for Data", [start_date_default, end_date])
    
    start_date = date_range[0]
    end_date = date_range[1] if len(date_range) > 1 else datetime.today()


    # Model parameters
    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365)
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90)
    sample_count = st.sidebar.slider("Sample Count for Confidence Interval", 5, 50, 20)
    confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95)
    
    z_score = {
        80: 1.28,
        85: 1.44,
        90: 1.645,
        95: 1.96,
        99: 2.576
    }.get(confidence_level, 1.96)


    if st.sidebar.button("Run Anomaly Detection"):
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
        st.write("\nMaking predictions for anomaly detection...")
        
        x_df = data.iloc[-lookback-pred_len:-pred_len]
        y_df_ground_truth = data.iloc[-pred_len:]

        x_timestamp = x_df['timestamps']
        y_timestamp = y_df_ground_truth['timestamps']

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_preds = []
        for i in range(sample_count):
            status_text.text(f"Running sample {i+1}/{sample_count}")
            pred_df_sample = predictor.predict(
                df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.1, top_p=0.95, sample_count=1,
                verbose=False
            )
            all_preds.append(pred_df_sample['close'].values)
            progress_bar.progress((i + 1) / sample_count)

        status_text.text("Prediction complete. Calculating anomalies...")

        # Calculate mean prediction and confidence interval
        pred_matrix = np.array(all_preds)
        pred_mean = pred_matrix.mean(axis=0)
        pred_std = pred_matrix.std(axis=0)
        
        pred_upper = pred_mean + z_score * pred_std
        pred_lower = pred_mean - z_score * pred_std

        pred_df = pd.DataFrame({
            'close': pred_mean,
            'upper_bound': pred_upper,
            'lower_bound': pred_lower
        }, index=y_timestamp)

        # ### 4. Detect and Visualize Anomalies
        st.write("\nDetecting and visualizing anomalies...")
        
        ground_truth_values = y_df_ground_truth['close'].values
        upper_bound_values = pred_df['upper_bound'].values
        lower_bound_values = pred_df['lower_bound'].values

        anomalies = y_df_ground_truth[(ground_truth_values > upper_bound_values) | 
                                      (ground_truth_values < lower_bound_values)]

        st.subheader(f"Anomaly Detection Results for {ticker}")
        plot_anomaly_detection(x_df, y_df_ground_truth, pred_df, anomalies, ticker)

        if not anomalies.empty:
            st.subheader(f"Detected {len(anomalies)} Anomalies")
            st.dataframe(anomalies)
        else:
            st.success("No anomalies detected for the given period and confidence level.")

if __name__ == "__main__":
    main()
