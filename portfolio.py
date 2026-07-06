import numpy as np
from scipy.optimize import minimize

def annualize(returns):
    """Annualise les rendements et la covariance (base 252 jours de trading)."""
    mu    = returns.mean() * 252
    sigma = returns.cov() * 252
    return mu, sigma

def portfolio_perf(weights, mu, sigma):
    """Rendement, volatilité et Sharpe d'un portefeuille."""
    ret  = weights @ mu
    vol  = np.sqrt(weights @ sigma @ weights)
    sharpe = ret / vol
    return ret, vol, sharpe

def min_variance(mu, sigma):
    """Portefeuille de variance minimale."""
    n      = len(mu)
    w0     = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        lambda w: portfolio_perf(w, mu, sigma)[1],  # minimise la volatilité
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result.x

def max_sharpe(mu, sigma, rf=0.05):
    """Portefeuille maximisant le ratio de Sharpe."""
    n      = len(mu)
    w0     = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        lambda w: -(portfolio_perf(w, mu, sigma)[0] - rf) / portfolio_perf(w, mu, sigma)[1],
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result.x

def efficient_frontier(mu, sigma, n_points=200):
    """Calcule la frontière efficiente."""
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    frontier_vols  = []
    n = len(mu)

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_perf(w, mu, sigma)[0] - t}
        ]
        result = minimize(
            lambda w: portfolio_perf(w, mu, sigma)[1],
            np.ones(n) / n,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=constraints
        )
        frontier_vols.append(result.fun if result.success else np.nan)

    return target_returns, np.array(frontier_vols)