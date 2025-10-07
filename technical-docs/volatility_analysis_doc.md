# Documentation: Volatility Analysis Script

## 1. Purpose
The `volatility_analysis.py` script is designed to assess the Kronos model's ability to capture and predict market volatility. It forecasts future prices and then compares the annualized rolling volatility of the predicted data against the historical volatility.

## 2. Methodology
1.  **Initialization**: Loads the `Kronos` model and `KronosTokenizer`.
2.  **Data Fetching**: Downloads historical data for a specified stock. It is recommended to use a stock known for its price fluctuations (e.g., `"TSLA"`) to better test the model's capabilities.
3.  **Prediction**: It uses a `lookback` window of historical data to predict a future period (`pred_len`). For this task, it uses a higher `sample_count` (e.g., 5) to generate a more stable and representative average prediction.
4.  **Volatility Calculation**:
    *   **Historical Volatility**: It calculates the 30-day annualized rolling volatility on the actual historical and ground truth data.
    *   **Predicted Volatility**: It calculates the same volatility metric on a combined series of the last 30 days of historical data and the newly generated predictions. This overlap ensures a smoother transition.
5.  **Visualization**: The script produces a two-part plot:
    *   **Top Plot**: Shows the standard price prediction (historical vs. ground truth vs. prediction).
    *   **Bottom Plot**: Compares the historical volatility (blue line) with the predicted volatility (orange dashed line).

## 3. Key Parameters
*   `ticker`: The stock symbol to analyze. Using a more volatile stock like `"TSLA"` or a cryptocurrency pair is effective.
*   `start`, `end`: The date range for fetching data.
*   `lookback`: The number of historical days to use for the prediction.
*   `pred_len`: The number of future days to predict.
*   `window`: The rolling window size (in days) for the volatility calculation, set to 30 by default.

## 4. Output
*   **Console Output**: Prints status messages and the head of the forecasted data.
*   **Plot**: A matplotlib window with two subplots for price and volatility.
*   **Image File**: The combined plot is saved to the root directory as `<ticker>_volatility_analysis.png`.
