# Exploratory Analysis of Kronos Model
#
# This script allows for experimenting with the Kronos model. It will:
# 1. Load the pre-trained Kronos model and tokenizer.
# 2. Fetch financial data using `yfinance`.
# 3. Use the model to make predictions on the fetched data.
# 4. Visualize the results.

# Before running, make sure you have yfinance installed:
# pip install yfinance

import pandas as pd
import matplotlib.pyplot as plt
import sys
import yfinance as yf

# To import from parent directory
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    """
    Main function to run the exploratory analysis.
    """
    # ### 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    # Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=1000)
    print("Model and tokenizer loaded.")

    # ### 2. Fetch Data using yfinance
    print("\nFetching stock data...")
    ticker = "AAPL"
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01", interval="1d")

    # Prepare data for the model
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })
    data['amount'] = data['close'] * data['volume']
    data['timestamps'] = data.index

    print("Data head:")
    print(data.head())

    # ### 3. Make Prediction
    print("\nMaking predictions...")
    lookback = 365
    pred_len = 90

    x_df = data.iloc[-lookback-pred_len:-pred_len]
    y_df_ground_truth = data.iloc[-pred_len:]

    x_timestamp = x_df['timestamps']
    y_timestamp = y_df_ground_truth['timestamps']

    # Make Prediction
    pred_df = predictor.predict(
        df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )

    print("\nForecasted Data Head:")
    print(pred_df.head())

    # ### 4. Visualize the Results
    print("\nVisualizing results...")
    plot_prediction(x_df, y_df_ground_truth, pred_df, ticker)
    print("Plot displayed. Script finished.")

def plot_prediction(historical_df, ground_truth_df, prediction_df, ticker):
    """
    Plots the historical data, ground truth, and prediction.
    """
    plt.figure(figsize=(15, 7))
    
    plt.plot(historical_df.index, historical_df['close'], label='Historical Data', color='blue')
    plt.plot(ground_truth_df.index, ground_truth_df['close'], label='Ground Truth', color='green')
    plt.plot(prediction_df.index, prediction_df['close'], label='Prediction', color='red', linestyle='--')
    
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
