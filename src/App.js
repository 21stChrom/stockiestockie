import React from "react";
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load data
data = pd.read_csv('stock_prices.csv')

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Split data into training and testing sets
training_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - training_size
train_data, test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:]

# Create features and labels
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]
                        dataX.append(a)
                                dataY.append(dataset[i + time_step, 0])
                                    return np.array(dataX), np.array(dataY)

                                    time_step = 100
                                    X_train, y_train = create_dataset(train_data, time_step)
                                    X_test, y_test = create_dataset(test_data, time_step)

                                    # Reshape input to be [samples, time steps, features]
                                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                                    # Define LSTM model
                                    model = Sequential()
                                    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                                    model.add(LSTM(50, return_sequences=True))
                                    model.add(LSTM(50))
                                    model.add(Dense(1))
                                    model.compile(loss='mean_squared_error', optimizer='adam')

                                    # Train model
                                    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

                                    # Make predictions
                                    train_predict = model.predict(X_train)
                                    test_predict = model.predict(X_test)

                                    # Invert predictions to get actual stock prices
                                    train_predict = scaler.inverse_transform(train_predict)
                                    y_train = scaler.inverse_transform([y_train])
                                    test_predict = scaler.inverse_transform(test_predict)
                                    y_test = scaler.inverse_transform([y_test])

                                    # Plot predictions
                                    plt.plot(y_test[0], label='Actual')
                                    plt.plot(test_predict[:,0], label='Predicted')
                                    plt.legend()
                                    plt.show()
                                    
import "./style.css";

export default function App() {
  return (
    <div>
      <h1></h1>
      <p></p>
    </div>
  );
}
