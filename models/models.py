import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.model = None
        self.scaler = MinMaxScaler()

    def preprocess_data(self, macro_data, sentiment_data):
        # Preprocessing: Scaling and feature engineering with macro and sentiment data
        self.stock_data['Close'] = self.scaler.fit_transform(self.stock_data['Close'].values.reshape(-1, 1))
        # Aggiungi altre feature come macro_data e sentiment_data

    def create_sequences(self, sequence_length=60):
        X, y = [], []
        for i in range(sequence_length, len(self.stock_data['Close'])):
            X.append(self.stock_data['Close'][i-sequence_length:i])
            y.append(self.stock_data['Close'][i])
        return np.array(X), np.array(y)

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_future(self, projection_range):
        # Implementa la logica per la previsione futura
        return self.model.predict(self.stock_data[-60:].reshape(1, 60, 1))

class TransformerModel:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        # Implementa un modello Transformer o un altro tipo di modello avanzato
        pass
