import pandas as pd
import numpy as np
import cvxpy as cp

class PortfolioOptimizer:
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.asset_weights = None

    def optimize(self, risk_free_rate=0.01):
        # Calcola i rendimenti attesi e la matrice di covarianza dai dati storici
        returns = self.stock_data['Close'].pct_change().dropna()  # Rendimenti giornalieri
        expected_returns = np.mean(returns) * 252  # Rendimento annuo atteso
        cov_matrix = np.cov(returns, rowvar=False) * 252  # Matrice di covarianza annualizzata

        # Numero di asset
        n = len(expected_returns)

        # Variabile dei pesi
        w = cp.Variable(n)

        # Funzione obiettivo: minimizzare la varianza del portafoglio
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)

        # Risoluzione del problema di ottimizzazione
        problem = cp.Problem(cp.Minimize(portfolio_variance - risk_free_rate * portfolio_return), 
                             [cp.sum(w) == 1, w >= 0])
        problem.solve()

        self.asset_weights = w.value
        return self._generate_portfolio()

    def _generate_portfolio(self):
        return pd.DataFrame({
            'Asset': ['Stock'],
            'Weight': self.asset_weights
        })
