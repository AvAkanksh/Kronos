
import streamlit as st
import pandas as pd
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

def plot_prediction(historical_df, ground_truth_df, prediction_df, ticker):
    """
    Plots the historical data, ground truth, and prediction.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(historical_df.index, historical_df['close'], label='Historical Data', color='blue')
    ax.plot(ground_truth_df.index, ground_truth_df['close'], label='Ground Truth', color='green')
    ax.plot(prediction_df.index, prediction_df['close'], label='Prediction', color='red', linestyle='--')
    
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Exploratory Stock Prediction with Kronos")

    st.sidebar.header("Configuration")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
    
    # Date range selection
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=3*365)
    date_range = st.sidebar.date_input("Select Date Range for Data", [start_date_default, end_date])
    
    start_date = date_range[0]
    end_date = date_range[1] if len(date_range) > 1 else datetime.today()

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Period (days)", 30, 730, 365, key="exp_lookback")
    pred_len = st.sidebar.slider("Prediction Length (days)", 30, 365, 90, key="exp_pred_len")
    temperature = st.sidebar.slider("Temperature (T)", 0.1, 2.0, 1.0, 0.1, key="exp_temp")
    top_p = st.sidebar.slider("Top-p Sampling", 0.1, 1.0, 0.9, 0.05, key="exp_topp")
    sample_count = st.sidebar.slider("Sample Count for Averaging", 1, 20, 1, key="exp_sample_count")

    if st.sidebar.button("Run Prediction"):
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
        
        st.subheader("Raw Data")
        st.dataframe(data.tail())

        if len(data) < lookback + pred_len:
            st.warning(f"Not enough data for the selected lookback ({lookback}) and prediction ({pred_len}) periods. Need at least {lookback + pred_len} data points, but got {len(data)}. Please select a larger date range or smaller periods.")
            return

        # ### 3. Make Prediction
        st.write("\nMaking predictions...")
        
        x_df = data.iloc[-lookback-pred_len:-pred_len]
        y_df_ground_truth = data.iloc[-pred_len:]

        x_timestamp = x_df['timestamps']
        y_timestamp = y_df_ground_truth['timestamps']

        with st.spinner("Running prediction..."):
            pred_df = predictor.predict(
                df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False
            )
        st.write("Prediction complete.")

        # ### 4. Visualize the Results
        st.subheader(f"Prediction Results for {ticker}")
        plot_prediction(x_df, y_df_ground_truth, pred_df, ticker)
        
        st.subheader("Forecasted Data")
        st.dataframe(pred_df)

if __name__ == "__main__":
    main()
