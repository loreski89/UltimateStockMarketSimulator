import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import TFAutoModel

class LSTMModel:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    
    def preprocess_data(self, macro_data=None, sentiment_data=None):
        # Normalizza i dati storici
        scaled_data = self.scaler.fit_transform(self.stock_data['Close'].values.reshape(-1, 1))
        return scaled_data
    
    def create_sequences(self, time_step=60):
        data = self.preprocess_data()
        X_train, y_train = [], []
        for i in range(time_step, len(data)):
            X_train.append(data[i - time_step:i, 0])
            y_train.append(data[i, 0])
        return np.array(X_train), np.array(y_train)

    def build_model(self):
        # Definisce il modello LSTM
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def predict_future(self, last_sequence, projection_range='1M'):
        # Genera previsioni future
        predictions = []
        input_sequence = last_sequence
        for _ in range(60):  # Esegue previsioni per i prossimi 60 giorni
            pred = self.model.predict(input_sequence.reshape(1, 60, 1))
            predictions.append(pred[0, 0])
            input_sequence = np.append(input_sequence[1:], pred)
        return np.array(predictions)

class TransformerModel:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.model = TFAutoModel.from_pretrained("bert-base-uncased")

    def preprocess_data(self):
        pass  # Preprocessamento specifico per il modello Transformer
