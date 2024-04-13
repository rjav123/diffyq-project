import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define the tickers
vix_ticker = "^VIX"  # VIX for market sentiment
sp500_ticker = "^GSPC"  # S&P 500

# Define the start and end dates for the data
start_date = "2023-01-01"
end_date = "2023-12-31"

# Fetch the data
vix_data = yf.download(vix_ticker, start=start_date, end=end_date)
sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date)

# For simplicity, we'll focus on 'Close' for price and 'Volume'
vix_close = vix_data['Close'].rename('VIX_Close')
sp500_volume = sp500_data['Volume'].rename('SP500_Volume')
sp500_close = sp500_data['Close'].rename('SP500_Close')

# Combine the data into a single DataFrame
combined_data = pd.concat([vix_close, sp500_volume, sp500_close], axis=1)

# Display the first few rows of the dataset
print(combined_data.head())

# Check for missing values
missing_values = combined_data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values
# For simplicity, you might fill missing values with the previous day's data (forward fill)
combined_data.fillna(method='ffill', inplace=True)

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Select columns to scale
columns_to_scale = ['VIX_Close', 'SP500_Volume', 'SP500_Close']
combined_data[columns_to_scale] = scaler.fit_transform(combined_data[columns_to_scale])

print("Data after normalization:\n", combined_data.head())

# Sort the DataFrame by the index (date) to ensure proper alignment
combined_data.sort_index(inplace=True)

# Save the preprocessed data to a CSV file
combined_data.to_csv("preprocessed_financial_data.csv", index=True)

print("Preprocessed data saved successfully.")

import pandas as pd

# If starting a new session, load the preprocessed data
combined_data = pd.read_csv("preprocessed_financial_data.csv", index_col='Date', parse_dates=True)

# Display summary statistics
print(combined_data.describe())

# Calculate and display the correlation matrix
correlation_matrix = combined_data.corr()
print(correlation_matrix)

import matplotlib.pyplot as plt

# Plotting each variable over time
combined_data.plot(subplots=True, figsize=(10, 8))
plt.title('Time Series Plot of VIX, S&P 500 Volume, and S&P 500 Close')
plt.show()

# Scatter plot of VIX_Close vs. SP500_Close
plt.scatter(combined_data['VIX_Close'], combined_data['SP500_Close'])
plt.title('VIX_Close vs. SP500_Close')
plt.xlabel('Normalized VIX Close')
plt.ylabel('Normalized S&P 500 Close')
plt.show()

# Scatter plot of SP500_Volume vs. SP500_Close
plt.scatter(combined_data['SP500_Volume'], combined_data['SP500_Close'])
plt.title('SP500_Volume vs. SP500_Close')
plt.xlabel('Normalized S&P 500 Volume')
plt.ylabel('Normalized S&P 500 Close')
plt.show()

# Model parameters
a = 0.1
b = 0.05
c = 0.03
S_0 = 0.5  # Equilibrium sentiment level

# Model functions
def f(S, V):
    return a * S - b * V

def g(S):
    return -c * (S - S_0)

# Time settings
T = 100  # total time
dt = 0.1  # timestep
N = int(T / dt)  # number of steps

# Initial conditions
S = np.zeros(N)
V = np.zeros(N)
S[0] = 0.6  # initial sentiment
V[0] = 0.4  # initial volume

# Euler's method to solve the system
for t in range(1, N):
    S[t] = S[t-1] + dt * g(S[t-1])
    V[t] = V[t-1] + dt * f(S[t-1], V[t-1])

# Creating the time array
time = np.arange(0, T, dt)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, S, label='Market Sentiment')
plt.ylabel('Sentiment')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, V, label='Trading Volume')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()

plt.suptitle('Financial Market Dynamics')
plt.show()
