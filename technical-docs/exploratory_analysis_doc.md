# Documentation: Exploratory Analysis Script

## 1. Purpose
The `exploratory_analysis.py` script serves as a basic demonstration of the Kronos model's forecasting capabilities. Its primary goal is to take a historical stock price series, predict a future period, and visualize the prediction against the actual data to allow for a qualitative assessment of the model's performance.

## 2. Methodology
The script follows a straightforward process:
1.  **Initialization**: It loads the pre-trained `Kronos` model and its corresponding `KronosTokenizer`.
2.  **Data Fetching**: It uses the `yfinance` library to download historical daily stock data for a specified ticker and date range.
3.  **Data Preparation**: The downloaded data is formatted to match the input requirements of the Kronos model, including renaming columns (`Open` -> `open`, etc.) and creating a `timestamps` column.
4.  **Prediction**: It feeds a `lookback` period of historical data (e.g., 365 days) to the model's `predict` function to forecast the next `pred_len` days (e.g., 90 days).
5.  **Visualization**: It generates and displays a plot that shows:
    *   The historical data used for the prediction (in blue).
    *   The actual stock prices for the forecasted period (the ground truth, in green).
    *   The model's prediction (in red).

## 3. Key Parameters
You can modify these variables in the `main` function to experiment with different scenarios:
*   `ticker`: The stock symbol to analyze (e.g., `"AAPL"`, `"MSFT"`).
*   `start`, `end`: The date range for fetching historical data.
*   `lookback`: The number of historical data points to feed into the model.
*   `pred_len`: The number of future data points to predict.

## 4. Output
*   **Console Output**: The script prints status messages, the head of the fetched data, and the head of the forecasted data.
*   **Plot**: A matplotlib window appears, showing the price prediction chart.
*   **Image File**: The plot is saved to the root directory as `<ticker>_prediction_plot.png`.
