# Documentation: Anomaly Detection Script

## 1. Purpose
The `anomaly_detection.py` script leverages the Kronos model to identify unusual or anomalous price movements in a stock's history. It works by treating any price that falls outside the model's high-confidence prediction range as an anomaly.

## 2. Methodology
The script's logic is based on creating a predictive confidence interval:
1.  **Initialization**: Loads the `Kronos` model and `KronosTokenizer`.
2.  **Data Fetching**: Downloads historical data for a given stock.
3.  **Generate Predictive Distribution**: Instead of making a single prediction, the script runs the prediction process many times (`sample_count = 20`). By using a higher "temperature" (`T=1.1`), it encourages a diverse range of plausible outcomes. This creates a distribution of possible future prices.
4.  **Calculate Confidence Interval**: From the 20 prediction samples, it calculates the mean and standard deviation for each future day. It then computes a **95% confidence interval** (defined as `mean ± 1.96 * std_dev`). This interval represents the range where the model is highly confident the actual price will fall.
5.  **Identify Anomalies**: The script compares the actual stock prices (ground truth) against this confidence interval. If an actual price point falls **outside** this range (either above the upper bound or below the lower bound), it is flagged as an **anomaly**.
6.  **Visualization**: It generates a plot showing:
    *   Historical data.
    *   The mean prediction (red dashed line).
    *   The 95% confidence interval (a shaded red area).
    *   The ground truth data (green line).
    *   Any detected anomalies marked with large purple dots.

## 3. Key Parameters
*   `ticker`: The stock symbol to analyze.
*   `start`, `end`: The date range for fetching data.
*   `lookback`: The number of historical days for the model's context.
*   `pred_len`: The number of future days to predict and check for anomalies.
*   `sample_count`: The number of samples to generate for the predictive distribution. A higher number (e.g., 20) creates a more reliable confidence interval.
*   `T` (temperature): A sampling parameter. A value > 1.0 increases variance in predictions, which is useful for creating the distribution.

## 4. Output
*   **Console Output**: Prints status messages, the number of anomalies found, and the details of each anomaly (if any).
*   **Plot**: A matplotlib window showing the price chart with the confidence interval and highlighted anomalies.
*   **Image File**: The plot is saved to the root directory as `<ticker>_anomaly_detection.png`.
