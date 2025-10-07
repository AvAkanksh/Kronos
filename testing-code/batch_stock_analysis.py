# Comprehensive Batch Stock Analysis using Kronos Model
#
# This script performs a batch analysis on a list of specified stocks.
# For each stock, it conducts three types of analysis:
# 1. Price Prediction: Forecasts future prices.
# 2. Volatility Analysis: Compares historical and predicted volatility.
# 3. Anomaly Detection: Identifies unusual price movements.
#
# All results, including individual plots and metrics, are saved in the
# `batch_analysis_results` directory. Finally, it generates a summary
# report in Markdown format.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import yfinance as yf
import os
import json
from datetime import datetime

# To import from parent directory
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

# --- Configuration ---
STOCKS_TO_ANALYZE = ["AAPL", "GOOG", "TSLA", "JPM", "AMZN", "MSFT"]
RESULTS_DIR = "batch_analysis_results"
START_DATE = "2022-01-01"
END_DATE = "2025-07-01"
LOOKBACK = 365
PRED_LEN = 90
SAMPLE_COUNT_ANOMALY = 20
VOLATILITY_WINDOW = 30

# --- Helper Functions ---

def calculate_volatility(df, window=VOLATILITY_WINDOW):
    """Calculates annualized rolling volatility."""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)

def create_results_directory():
    """Creates the directory for saving results if it doesn't exist."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

# --- Main Analysis Function ---

def analyze_stock(ticker, predictor):
    """
    Performs the full analysis suite for a single stock ticker.
    """
    print(f"\n----- Analyzing {ticker} -----")

    # 1. Fetch Data
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True)
    if data.empty:
        print(f"Could not fetch data for {ticker}. Skipping.")
        return None

    data = data.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    data['amount'] = data['close'] * data['volume']
    data['timestamps'] = data.index
    
    # Define data splits
    x_df = data.iloc[-LOOKBACK-PRED_LEN:-PRED_LEN]
    y_df_ground_truth = data.iloc[-PRED_LEN:]
    x_timestamp = x_df['timestamps']
    y_timestamp = y_df_ground_truth['timestamps']

    # 2. Perform Anomaly Detection & Prediction
    print(f"Running prediction and anomaly detection for {ticker}...")
    all_preds = []
    for i in range(SAMPLE_COUNT_ANOMALY):
        if (i + 1) % 5 == 0:
            print(f"  Running sample {i+1}/{SAMPLE_COUNT_ANOMALY}...")
        pred_df_sample = predictor.predict(
            df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=1.1, top_p=0.95, sample_count=1,
            verbose=False
        )
        all_preds.append(pred_df_sample['close'].values)

    pred_matrix = np.array(all_preds)
    pred_mean = pred_matrix.mean(axis=0)
    pred_std = pred_matrix.std(axis=0)
    pred_upper = pred_mean + 1.96 * pred_std
    pred_lower = pred_mean - 1.96 * pred_std

    pred_df = pd.DataFrame({
        'close': pred_mean,
        'upper_bound': pred_upper,
        'lower_bound': pred_lower
    }, index=y_timestamp)

    ground_truth_values = y_df_ground_truth['close'].values
    anomalies = y_df_ground_truth[(ground_truth_values > pred_upper) | (ground_truth_values < pred_lower)]
    
    # 3. Perform Volatility Analysis
    print(f"Running volatility analysis for {ticker}...")
    historical_volatility = calculate_volatility(pd.concat([x_df, y_df_ground_truth]))
    # Combine last 30 days of history with prediction for a smoother transition
    combined_for_vol = pd.concat([x_df.iloc[-VOLATILITY_WINDOW:], pred_df])
    predicted_volatility = calculate_volatility(combined_for_vol)

    # 4. Save Results and Plot
    plot_path = os.path.join(RESULTS_DIR, f"{ticker}_analysis_plot.png")
    plot_full_analysis(x_df, y_df_ground_truth, pred_df, anomalies, historical_volatility, predicted_volatility, ticker, plot_path)

    # 5. Compile Metrics
    metrics = {
        "ticker": ticker,
        "anomalies_found": len(anomalies),
        "average_predicted_volatility": predicted_volatility['close'].mean(),
        "plot_path": plot_path
    }
    
    json_path = os.path.join(RESULTS_DIR, f"{ticker}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"----- Analysis for {ticker} complete. Results saved. -----")
    return metrics

def plot_full_analysis(historical_df, ground_truth_df, prediction_df, anomalies, historical_vol, predicted_vol, ticker, save_path):
    """Plots the combined analysis for a stock and saves it to a file."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Comprehensive Analysis for {ticker}', fontsize=20)

    # Plot 1: Price Prediction and Anomaly Detection
    ax1.plot(historical_df.index, historical_df['close'], label='Historical Data', color='blue', linewidth=1.5)
    ax1.plot(ground_truth_df.index, ground_truth_df['close'], label='Ground Truth', color='green', linewidth=1.5)
    ax1.plot(prediction_df.index, prediction_df['close'], label='Mean Prediction', color='red', linestyle='--', linewidth=1.5)
    
    ax1.fill_between(prediction_df.index, prediction_df['lower_bound'], prediction_df['upper_bound'], 
                     color='red', alpha=0.2, label='95% Confidence Interval')

    if not anomalies.empty:
        ax1.scatter(anomalies.index, anomalies['close'], color='purple', s=120, zorder=5, label='Anomaly', edgecolors='black')

    ax1.set_title('Price Prediction & Anomaly Detection', fontsize=16)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Volatility Analysis
    ax2.plot(historical_vol.index, historical_vol, label='Historical Volatility', color='blue')
    ax2.plot(predicted_vol.index, predicted_vol, label='Predicted Volatility', color='orange', linestyle='--')
    ax2.set_title(f'{VOLATILITY_WINDOW}-Day Rolling Volatility (Annualized)', fontsize=16)
    ax2.set_ylabel('Volatility', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close() # Close the plot to free up memory
    print(f"Saved plot to {save_path}")

def generate_summary_report(all_metrics):
    """Generates a Markdown summary report from all the collected metrics."""
    report_path = os.path.join(RESULTS_DIR, "summary_report.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Kronos Batch Stock Analysis Report\n")
        f.write(f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary Inference Table\n\n")
        
        # Header
        f.write("| Ticker | Anomalies Found | Avg. Predicted Volatility | Detailed Plot |\n")
        f.write("|--------|-----------------|---------------------------|---------------|\n")
        
        # Rows
        for metrics in all_metrics:
            if metrics:
                volatility_str = f"{metrics['average_predicted_volatility']:.4f}" if metrics['average_predicted_volatility'] is not None else "N/A"
                f.write(f"| **{metrics['ticker']}** | {metrics['anomalies_found']} | {volatility_str} | [View Plot]({os.path.basename(metrics['plot_path'])}) |\n")
        
        f.write("\n## General Inferences\n")
        f.write("- **Volatility Insights**: Compare the `Avg. Predicted Volatility` across different stocks. Higher values suggest the model anticipates more price fluctuation.\n")
        f.write("- **Anomaly Hotspots**: Stocks with a higher number of `Anomalies Found` may be experiencing unusual market events that the model's historical context did not fully capture, warranting further investigation.\n")
        f.write("- **Model Confidence**: The width of the red shaded area in the plots indicates the model's uncertainty. Wider bands mean lower confidence.\n")

    print(f"\nSummary report generated at {report_path}")

# --- Main Execution ---

def main():
    """
    Main function to orchestrate the batch analysis.
    """
    create_results_directory()

    print("Loading Kronos model and tokenizer (this may take a moment)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=1000)
    print("Model and tokenizer loaded successfully.")

    all_results = []
    for ticker in STOCKS_TO_ANALYZE:
        result = analyze_stock(ticker, predictor)
        if result:
            all_results.append(result)
            
    if all_results:
        generate_summary_report(all_results)
    else:
        print("No analysis was completed. Summary report not generated.")

    print("\nBatch analysis finished.")

if __name__ == "__main__":
    main()
