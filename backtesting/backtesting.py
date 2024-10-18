import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.initial_capital = 100000  # Capitale iniziale
        self.results = None
    
    def run_backtest(self):
        # Simulazione di strategia semplice (es. media mobile)
        short_window = 40
        long_window = 100
        
        signals = pd.DataFrame(index=self.stock_data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = self.stock_data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = self.stock_data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()

        # Simulazione portafoglio
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions['stock'] = signals['signal']
        portfolio = positions.multiply(self.stock_data['Close'], axis=0)
        portfolio['holdings'] = portfolio.sum(axis=1)
        portfolio['cash'] = self.initial_capital - (positions.diff().multiply(self.stock_data['Close'], axis=0)).sum(axis=1).cumsum()
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
