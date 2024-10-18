import pandas as pd
import numpy as np

def fetch_stock_data(ticker):
    # Simulazione per esempio, puoi sostituirlo con yfinance o un'API reale
    dates = pd.date_range(start='2015-01-01', periods=200)
    prices = np.random.randn(200).cumsum() + 100  # Genera prezzi casuali
    data = pd.DataFrame(data={'Close': prices}, index=dates)
    return data

def fetch_macro_data():
    # Simula i dati macroeconomici (inflazione, tassi di interesse, ecc.)
    macro_data = {
        'inflation': 2.5,
        'interest_rates': 1.75
    }
    return macro_data

def fetch_sentiment_data(ticker):
    # Simula i dati di sentiment
    sentiment_data = {
        'average_sentiment': 0.5,
        'positive': 60,
        'negative': 40
    }
    return sentiment_data
