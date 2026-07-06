import numpy as np

def historical_var(returns, weights, confidence=0.95):
    """
    VaR historique — pas d'hypothèse sur la distribution,
    on utilise les rendements réels observés.
    """
    portfolio_returns = returns @ weights
    var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
    return var

def historical_cvar(returns, weights, confidence=0.95):
    """
    CVaR (Expected Shortfall) — moyenne des pertes au-delà de la VaR.
    Mesure ce qu'on perd en moyenne dans les pires cas.
    """
    portfolio_returns = returns @ weights
    var               = historical_var(returns, weights, confidence)
    cvar              = -portfolio_returns[portfolio_returns < -var].mean()
    return cvar

def parametric_var(returns, weights, confidence=0.95):
    """
    VaR paramétrique — suppose que les rendements suivent une loi normale.
    Plus simple mais moins robuste que la VaR historique.
    """
    from scipy.stats import norm
    portfolio_returns = returns @ weights
    mu_p  = portfolio_returns.mean()
    sig_p = portfolio_returns.std()
    var   = -(mu_p + norm.ppf(1 - confidence) * sig_p)
    return var