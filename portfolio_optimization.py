import numpy as np
import cvxpy as cp
import pandas as pd

class PortfolioOptimizer:
    def __init__(self):
        self.asset_weights = None

    def optimize(self, expected_returns, cov_matrix, risk_free_rate=0.01):
        n = len(expected_returns)
        w = cp.Variable(n)
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        
        # Ottimizzazione con rischio minimo
        problem = cp.Problem(cp.Minimize(portfolio_variance - risk_free_rate * portfolio_return), [cp.sum(w) == 1, w >= 0])
        problem.solve()
        
        self.asset_weights = w.value
        return self._generate_portfolio()

    def _generate_portfolio(self):
        return pd.DataFrame({
            'Asset': [f'Asset {i+1}' for i in range(len(self.asset_weights))],
            'Weight': self.asset_weights
        })
