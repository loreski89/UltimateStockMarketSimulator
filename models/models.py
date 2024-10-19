import numpy as np
import pandas as pd
import tensorflow as tf

class LSTMModel:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.model = None

    def preprocess_data(self, macro_data, sentiment_data):
        # Preprocessamento dei dati
        self.stock_data['Normalized'] = (self.stock_data['Close'] - self.stock_data['Close'].mean()) / self.stock_data['Close'].std()

    def create_sequences(self, seq_length=50):
        # Crea sequenze per LSTM
        sequences = []
        labels = []
        data = self.stock_data['Normalized'].values
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length])
        X_train = np.array(sequences)
        y_train = np.array(labels)
        return X_train, y_train

    def build_model(self, seq_length=50):
        # Costruisci il modello LSTM
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        # Allenamento del modello
        X_train = np.expand_dims(X_train, axis=-1)  # Aggiungi dimensione per LSTM
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_future(self, projection_days):
        # Previsione dei prezzi futuri
        future_predictions = []
        last_data = np.expand_dims(self.stock_data['Normalized'].values[-50:], axis=0)

        for _ in range(projection_days):
            next_prediction = self.model.predict(last_data)
            future_predictions.append(next_prediction[0, 0])
            last_data = np.roll(last_data, -1, axis=1)
            last_data[0, -1] = next_prediction[0, 0]

        return np.array(future_predictions)

class TransformerModel:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.model = None

    def preprocess_data(self, macro_data, sentiment_data):
        # Preprocessamento dei dati
        self.stock_data['Normalized'] = (self.stock_data['Close'] - self.stock_data['Close'].mean()) / self.stock_data['Close'].std()

    def create_sequences(self, seq_length=50):
        # Crea sequenze per Transformer
        sequences = []
        labels = []
        data = self.stock_data['Normalized'].values
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            labels.append(data[i + seq_length])
        X_train = np.array(sequences)
        y_train = np.array(labels)
        return X_train, y_train

    def build_model(self, seq_length=50):
        # Costruisci il modello Transformer
        input_layer = tf.keras.layers.Input(shape=(seq_length, 1))
        transformer_block = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)(input_layer, input_layer)
        x = tf.keras.layers.GlobalAveragePooling1D()(transformer_block)
        output_layer = tf.keras.layers.Dense(1)(x)

        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        # Allenamento del modello
        X_train = np.expand_dims(X_train, axis=-1)  # Aggiungi dimensione per il Transformer
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict_future(self, projection_days):
        # Previsione dei prezzi futuri
        future_predictions = []
        last_data = np.expand_dims(self.stock_data['Normalized'].values[-50:], axis=0)

        for _ in range(projection_days):
            next_prediction = self.model.predict(last_data)
            future_predictions.append(next_prediction[0, 0])
            last_data = np.roll(last_data, -1, axis=1)
            last_data[0, -1] = next_prediction[0, 0]

        return np.array(future_predictions)
