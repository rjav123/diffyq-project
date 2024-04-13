# Step 1: Import Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Step 2: Fetch and Preprocess Data
# Define the tickers
vix_ticker = "^VIX"  # VIX for market sentiment
sp500_ticker = "^GSPC"  # S&P 500

# Define the start and end dates for the data
start_date = "2023-01-01"
end_date = "2023-12-31"

# Fetch the data
vix_data = yf.download(vix_ticker, start=start_date, end=end_date)
sp500_data = yf.download(sp500_ticker, start=start_date, end=end_date)

# Focus on 'Close' for price and 'Volume'
vix_close = vix_data['Close'].rename('VIX_Close')
sp500_volume = sp500_data['Volume'].rename('SP500_Volume')
sp500_close = sp500_data['Close'].rename('SP500_Close')

# Combine the data into a single DataFrame
combined_data = pd.concat([vix_close, sp500_volume, sp500_close], axis=1)

# Check for missing values and handle them
combined_data.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
columns_to_scale = ['VIX_Close', 'SP500_Volume', 'SP500_Close']
combined_data[columns_to_scale] = scaler.fit_transform(combined_data[columns_to_scale])

# Save the preprocessed data
combined_data.to_csv("preprocessed_financial_data.csv", index=True)

# Step 3: Feature Engineering for Predictive Modeling
# Generate lagged features for the past 5 days
for i in range(1, 6):
    combined_data[f'VIX_Close_Lag_{i}'] = combined_data['VIX_Close'].shift(i)
    combined_data[f'SP500_Volume_Lag_{i}'] = combined_data['SP500_Volume'].shift(i)
    combined_data[f'SP500_Close_Lag_{i}'] = combined_data['SP500_Close'].shift(i)

# Drop rows with NaN values
combined_data.dropna(inplace=True)

# Step 4: Define Features and Target for Prediction
X = combined_data[[f'VIX_Close_Lag_{i}' for i in range(1, 6)] +
                  [f'SP500_Volume_Lag_{i}' for i in range(1, 6)] +
                  [f'SP500_Close_Lag_{i}' for i in range(1, 6)]]
y = combined_data['SP500_Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict the S&P 500 Price for the Next 5 Trading Days
def predict_next_5_days(model, recent_data, feature_names):
    future_prices = []
    input_features = recent_data.to_frame().T  # Convert series to dataframe with correct feature names

    for _ in range(5):
        next_day_price = model.predict(input_features)[0]
        future_prices.append(next_day_price)

        # Shift the input features left and insert the predicted price as the most recent lag feature
        input_features = input_features.shift(-1, axis=1)
        input_features[feature_names[-1]] = next_day_price  # Use the name of the last lag feature for the predicted price

    return future_prices

# Assuming 'recent_data' is the last row from your features DataFrame 'X'
recent_data = X.iloc[-1]
feature_names = X.columns
predicted_prices = predict_next_5_days(model, recent_data, feature_names)
print("Predicted S&P 500 prices for the next 5 trading days:", predicted_prices)

# Assuming 'scaler' is your MinMaxScaler instance and it's still in scope
# If 'scaler' is not in scope, you might need to load it or refit it

# Create a dummy array with the shape of the scaled data
dummy_features = np.zeros((len(predicted_prices), combined_data[columns_to_scale].shape[1]))

# Assume S&P 500 Close prices were the last column scaled, replace its values with predictions
dummy_features[:, -1] = predicted_prices

# Inverse transform the dummy array
actual_prices = scaler.inverse_transform(dummy_features)[:, -1]

print("Actual S&P 500 prices for the next 5 trading days:", actual_prices)