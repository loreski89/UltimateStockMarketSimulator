import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, data):
        self.data = data

    def moving_average_crossover(self, short_window=40, long_window=100):
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0

        signals['short_mavg'] = self.data['Close'].rolling(window=short_window, min_periods=1).mean()
        signals['long_mavg'] = self.data['Close'].rolling(window=long_window, min_periods=1).mean()

        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()

        return signals

    def run_backtest(self, strategy='moving_average_crossover'):
        if strategy == 'moving_average_crossover':
            signals = self.moving_average_crossover()
            initial_capital = 100000.0  # capitale iniziale
            positions = pd.DataFrame(index=signals.index).fillna(0.0)
            positions['Position'] = signals['signal'] * 100  # Numero di azioni da comprare
            portfolio = positions.multiply(self.data['Close'], axis=0)
            portfolio['holdings'] = (positions.multiply(self.data['Close'], axis=0)).sum(axis=1)
            portfolio['cash'] = initial_capital - (positions.diff().multiply(self.data['Close'], axis=0)).sum(axis=1).cumsum()
            portfolio['total'] = portfolio['cash'] + portfolio['holdings']
            return portfolio
