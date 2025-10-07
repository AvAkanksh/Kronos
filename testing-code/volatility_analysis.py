# Volatility Analysis using Kronos Model
#
# This script performs volatility analysis by:
# 1. Loading the pre-trained Kronos model and tokenizer.
# 2. Fetching historical stock data.
# 3. Predicting future stock prices.
# 4. Calculating and comparing the historical and predicted volatility.
# 5. Visualizing the price predictions and volatility analysis.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import yfinance as yf

# To import from parent directory
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

def calculate_volatility(df, window=30):
    """
    Calculates the rolling volatility of a stock.
    """
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252) # Annualized volatility

def main():
    """
    Main function to run the volatility analysis.
    """
    # ### 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=1000)
    print("Model and tokenizer loaded.")

    # ### 2. Fetch Data
    print("\nFetching stock data for volatility analysis...")
    ticker = "TSLA" # Using a more volatile stock for this example
    data = yf.download(ticker, start="2022-01-01", end="2025-07-01", interval="1d")

    # Prepare data
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    data['amount'] = data['close'] * data['volume']
    data['timestamps'] = data.index
    print(f"Data for {ticker} fetched. Shape: {data.shape}")

    # ### 3. Make Prediction
    print("\nMaking predictions...")
    lookback = 365
    pred_len = 90

    x_df = data.iloc[-lookback-pred_len:-pred_len]
    y_df_ground_truth = data.iloc[-pred_len:]

    x_timestamp = x_df['timestamps']
    y_timestamp = y_df_ground_truth['timestamps']

    pred_df = predictor.predict(
        df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0, top_p=0.9, sample_count=5, # Using more samples for a better estimate
        verbose=True
    )
    print("\nForecasted Data Head:")
    print(pred_df.head())

    # ### 4. Analyze and Visualize Volatility
    print("\nAnalyzing and visualizing volatility...")
    
    # Calculate volatility
    historical_volatility = calculate_volatility(pd.concat([x_df, y_df_ground_truth]))
    predicted_volatility = calculate_volatility(pd.concat([x_df.iloc[-30:], pred_df]))

    plot_volatility_analysis(x_df, y_df_ground_truth, pred_df, historical_volatility, predicted_volatility, ticker)
    print("Plot displayed and saved. Script finished.")

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

    plt.tight_layout()
    plt.savefig(f"{ticker}_volatility_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
