import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Load and Preprocess Data
# Replace 'your_dataset.csv' with the actual path to your dataset
data = pd.read_csv('dataset.csv')
# Assuming 'Year' is a column representing the time
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)
your_time_series_data = data['Total']  # Replace 'Total' with the actual column name

# Step 2: Visualize Time Series
"""
plt.plot(your_time_series_data)
plt.title('Crime Counts Over Time')
plt.xlabel('Time')
plt.ylabel('Crime Counts')
plt.show()
"""

# Step 3: Determine Order of Differencing (d) and Seasonal Differencing (D)
plot_acf(your_time_series_data, lags=40)
plot_pacf(your_time_series_data, lags=40)

# Step 4: Fit SARIMA Model
# Replace placeholders with actual values based on your analysis
p, d, q = 1, 1, 1  # Example values, adjust as needed
P, D, Q, s = 1, 1, 1, 12  # Example values, adjust as needed

# Define the SARIMA model
order = (p, d, q)
seasonal_order = (P, D, Q, s)
model = SARIMAX(your_time_series_data, order=order, seasonal_order=seasonal_order)

# Fit the model
results = model.fit()

# Print the model summary
print(results.summary())

# Step 5: Make Predictions
forecast_steps = 12  # Adjust as needed
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# Step 6: Visualize Predictions
plt.plot(your_time_series_data, label='Observed')
plt.plot(forecast_mean, label='Forecast', color='red')
plt.title('SARIMA Forecast')
plt.xlabel('Time')
plt.ylabel('Crime Counts')
plt.legend()
plt.show()
