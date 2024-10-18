import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

class LSTMModel:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess_data(self):
        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
        return scaled_data

    def create_sequences(self, data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_future(self, X_test):
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
