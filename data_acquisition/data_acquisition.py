import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker):
    stock_data = yf.download(ticker, period="max")
    return stock_data

def fetch_macro_data():
    # Aggiungi logica per acquisire dati macroeconomici
    return {
        'inflation': 2.5,
        'interest_rates': 1.5
    }

def fetch_sentiment_data(ticker):
    # Aggiungi logica per acquisire dati di sentiment
    return {
        'average_sentiment': 0.6,
        'positive': 70,
        'negative': 30
    }
