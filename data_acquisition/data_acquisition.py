import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker):
    # Funzione per recuperare i dati storici delle azioni da Yahoo Finance
    stock_data = yf.download(ticker, period='1y')  # Scarica i dati dell'ultimo anno
    if stock_data.empty:
        raise KeyError(f"No data found for ticker: {ticker}")
    return stock_data

def fetch_macro_data():
    # Funzione per recuperare i dati macroeconomici
    return {
        'inflation': 2.5,
        'interest_rates': 0.5
    }

def fetch_sentiment_data(ticker):
    # Funzione per recuperare i dati di sentiment
    return {
        'average_sentiment': 0.6,
        'positive': 70,
        'negative': 30
    }
