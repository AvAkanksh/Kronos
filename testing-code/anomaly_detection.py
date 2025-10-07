# Anomaly Detection in Stock Prices using Kronos Model
#
# This script uses the Kronos model to detect anomalies in stock price data.
# The main idea is to:
# 1. Predict a range of future prices (using multiple samples).
# 2. Compare the actual prices against this predicted range.
# 3. Flag any actual prices that fall outside the predicted confidence interval as anomalies.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import yfinance as yf

# To import from parent directory
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

def main():
    """
    Main function to run the anomaly detection analysis.
    """
    # ### 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=1000)
    print("Model and tokenizer loaded.")

    # ### 2. Fetch Data
    print("\nFetching stock data for anomaly detection...")
    ticker = "GOOG"
    data = yf.download(ticker, start="2022-01-01", end="2025-07-01", interval="1d")

    # Prepare data
    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    data['amount'] = data['close'] * data['volume']
    data['timestamps'] = data.index
    print(f"Data for {ticker} fetched. Shape: {data.shape}")

    # ### 3. Make Prediction with multiple samples to create a confidence interval
    print("\nMaking predictions for anomaly detection...")
    lookback = 365
    pred_len = 90
    sample_count = 20 # More samples to create a better confidence interval

    x_df = data.iloc[-lookback-pred_len:-pred_len]
    y_df_ground_truth = data.iloc[-pred_len:]

    x_timestamp = x_df['timestamps']
    y_timestamp = y_df_ground_truth['timestamps']

    # We need to get the raw predictions from multiple samples
    # The current predictor averages them, so we'll call the generate method multiple times
    # Note: This is a workaround. A more efficient implementation would modify the predictor.
    
    all_preds = []
    for i in range(sample_count):
        print(f"Running sample {i+1}/{sample_count}")
        pred_df_sample = predictor.predict(
            df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.1, top_p=0.95, sample_count=1, # Higher temperature for more variance
            verbose=False
        )
        all_preds.append(pred_df_sample['close'].values)

    # Calculate mean prediction and confidence interval
    pred_matrix = np.array(all_preds)
    pred_mean = pred_matrix.mean(axis=0)
    pred_std = pred_matrix.std(axis=0)
    
    pred_upper = pred_mean + 1.96 * pred_std # 95% confidence interval
    pred_lower = pred_mean - 1.96 * pred_std

    pred_df = pd.DataFrame({
        'close': pred_mean,
        'upper_bound': pred_upper,
        'lower_bound': pred_lower
    }, index=y_timestamp)

    # ### 4. Detect and Visualize Anomalies
    print("\nDetecting and visualizing anomalies...")
    
    anomalies = y_df_ground_truth[(y_df_ground_truth['close'] > pred_df['upper_bound']) | 
                                  (y_df_ground_truth['close'] < pred_df['lower_bound'])]

    print(f"Found {len(anomalies)} anomalies.")
    if not anomalies.empty:
        print(anomalies)

    plot_anomaly_detection(x_df, y_df_ground_truth, pred_df, anomalies, ticker)
    print("Plot displayed and saved. Script finished.")

def plot_anomaly_detection(historical_df, ground_truth_df, prediction_df, anomalies, ticker):
    """
    Plots the anomaly detection results.
    """
    plt.figure(figsize=(15, 7))
    
    plt.plot(historical_df.index, historical_df['close'], label='Historical Data', color='blue')
    plt.plot(ground_truth_df.index, ground_truth_df['close'], label='Ground Truth', color='green')
    plt.plot(prediction_df.index, prediction_df['close'], label='Mean Prediction', color='red', linestyle='--')
    
    plt.fill_between(prediction_df.index, prediction_df['lower_bound'], prediction_df['upper_bound'], 
                     color='red', alpha=0.2, label='95% Confidence Interval')

    if not anomalies.empty:
        plt.scatter(anomalies.index, anomalies['close'], color='purple', s=100, zorder=5, label='Anomaly')

    plt.title(f'{ticker} Stock Price Anomaly Detection')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{ticker}_anomaly_detection.png")
    plt.show()

if __name__ == "__main__":
    main()
