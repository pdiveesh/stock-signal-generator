import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time  # Import the time module for introducing delays

# Function to retrieve historical stock data from Alpha Vantage
def get_stock_data(api_key, symbol, interval='1min', output_size='full'):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize=output_size)
    return data

# Function to create features from historical stock data
def create_features(data):
    data['SMA5'] = data['4. close'].rolling(window=5).mean()
    data['SMA20'] = data['4. close'].rolling(window=20).mean()
    data['SMA50'] = data['4. close'].rolling(window=50).mean()
    data['SMA200'] = data['4. close'].rolling(window=200).mean()

    data['RSI'] = calculate_rsi(data['4. close'], window=14)

    return data

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    diff = data.diff(1)

    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Function to create additional features
def create_additional_features(data):
    data['EMA12'] = data['4. close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['4. close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to generate trading signals
def generate_signals(data):
    data['Signal'] = np.where(data['SMA5'] > data['SMA20'], 1, 0)
    data['Signal'] = np.where(data['SMA5'] < data['SMA20'], -1, data['Signal'])
    return data

# Function to prepare data for machine learning
def prepare_data(data):
    data = data.dropna()
    X = data[['SMA5', 'SMA20', 'SMA50', 'SMA200', 'RSI', 'EMA12', 'EMA26', 'MACD', 'Signal_Line']]
    y = data['Signal']
    return X, y

# Train and evaluate the model
def train_evaluate_model(api_key):
    training_symbol = 'AAPL'  # Replace with your training stock symbol
    training_data = get_stock_data(api_key, training_symbol)

    training_data = create_features(training_data)
    training_data = create_additional_features(training_data)
    training_data = generate_signals(training_data)

    X_train, y_train = prepare_data(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy on Test Set: {accuracy:.2f}')

    print('\nClassification Report on Test Set:\n', classification_report(y_test, predictions, zero_division=1))

    return model

# Function to predict for a new stock based on user input
def predict_for_stock(api_key, model):
    while True:
        stock_symbol = input("Enter the stock symbol for prediction (or 'exit' to stop): ").upper()

        if stock_symbol.lower() == 'exit':
            print("Exiting real-time prediction.")
            break

        prediction_data = get_stock_data(api_key, stock_symbol)
        prediction_data = create_features(prediction_data)
        prediction_data = create_additional_features(prediction_data)
        prediction_data = generate_signals(prediction_data)

        X_predict, _ = prepare_data(prediction_data)

        predictions = model.predict(X_predict)

        prediction_data['Prediction'] = np.concatenate((np.zeros(len(prediction_data) - len(predictions)), predictions))

        print(prediction_data[['4. close', 'Signal', 'Prediction']])

        print("\nAdditional Information:")
        print("Features used for prediction:")
        print(X_predict.head())

        plt.figure(figsize=(10, 6))
        prediction_data['4. close'].plot(label='Close Price')

        buy_signals = prediction_data[prediction_data['Prediction'] == 1]
        sell_signals = prediction_data[prediction_data['Prediction'] == -1]

        plt.scatter(buy_signals.index, buy_signals['4. close'], marker='^', color='g', label='Buy Signal')
        plt.scatter(sell_signals.index, sell_signals['4. close'], marker='v', color='r', label='Sell Signal')

        plt.legend()
        plt.show()

        time.sleep(300)

if __name__ == '__main__':
    api_key = 'YOUR_API_KEY'
    trained_model = train_evaluate_model(api_key)
    predict_for_stock(api_key, trained_model)
