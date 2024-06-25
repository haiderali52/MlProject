import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the dataset
df = pd.read_csv(r'/content/drive/MyDrive/supermarket_sales - Sheet1.csv')

# Display the first few rows of the dataframe to understand its structure
print(df.head())
print(df.columns)

# Ensure the 'Date' column is in datetime format with error handling
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
except ValueError:
    print("Error: Unable to parse dates. Please check the date format in the dataset.")
    exit(1)

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Resample to daily data and sum the sales
daily_sales = df['Total'].resample('D').sum()

# Plot the daily sales data
plt.figure(figsize=(12, 6))
plt.plot(daily_sales, label='Daily Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Daily Sales Over Time')
plt.legend()
plt.show()

# Fit the SARIMA model with error handling
try:
    sarima_model = SARIMAX(daily_sales, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    sarima_result = sarima_model.fit(disp=False)  # Set disp=False to suppress convergence messages
except ValueError as ve:
    print(f"Error: {ve}")
    exit(1)

# Forecast the next 30 days
forecast = sarima_result.get_forecast(steps=30)
forecast_index = pd.date_range(start=daily_sales.index[-1] + pd.DateOffset(days=1), periods=30, freq='D')
forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

# Backcast the previous 60 days using in-sample predictions
in_sample_prediction = sarima_result.predict(start=daily_sales.index[0], end=daily_sales.index[-1])
backcast_series = in_sample_prediction.head(60)[::-1]
backcast_index = pd.date_range(end=daily_sales.index[0] - pd.DateOffset(days=1), periods=60, freq='D')[::-1]
backcast_series.index = backcast_index

# Plot the forecast and backcast
plt.figure(figsize=(12, 6))
plt.plot(daily_sales, label='Observed')
plt.plot(forecast_series, label='Forecast', color='red')
plt.plot(backcast_series, label='Backcast', color='black')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast and Backcast')
plt.legend()
plt.show()

# Identify the day with the maximum and minimum forecasted sales for each month
max_sales_day_forecast = forecast_series.idxmax()
min_sales_day_forecast = forecast_series.idxmin()

print(f"Day with the highest forecasted sales: {max_sales_day_forecast}, Sales: {forecast_series[max_sales_day_forecast]:.2f}")
print(f"Day with the lowest forecasted sales: {min_sales_day_forecast}, Sales: {forecast_series[min_sales_day_forecast]:.2f}")

# Identify the day with the maximum and minimum sales for the previous 2 months
previous_2_months_sales = daily_sales[-60:]
max_sales_day_previous = previous_2_months_sales.idxmax()
min_sales_day_previous = previous_2_months_sales.idxmin()

print(f"Day with the highest sales in the previous 2 months: {max_sales_day_previous}, Sales: {previous_2_months_sales[max_sales_day_previous]:.2f}")
print(f"Day with the lowest sales in the previous 2 months: {min_sales_day_previous}, Sales: {previous_2_months_sales[min_sales_day_previous]:.2f}")

# Save the forecasted data to a CSV file if needed
try:
    forecast_series.to_csv('/sales1_forecast.csv', header=['Sales Forecast'])
    print("Forecast data saved successfully.")
except IOError:
    print("Error: Unable to save forecast data. Check write permissions.")
    exit(1)

# Evaluate the model
try:
    # Split data for evaluation
    train_size = int(len(daily_sales) * 0.5)
    train, test = daily_sales[:train_size], daily_sales[train_size:]

    # Forecast using the trained model on the test set
    forecast_test = sarima_result.get_forecast(steps=len(test))
    forecast_mean_test = forecast_test.predicted_mean

    # Calculate evaluation metrics
    mse = mean_squared_error(test, forecast_mean_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, forecast_mean_test)

    print(f"Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

except ValueError as ve:
    print(f"Error: {ve}")
    exit(1)
