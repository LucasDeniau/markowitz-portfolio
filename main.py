import numpy as np
from data import get_prices, get_returns
from portfolio import annualize, portfolio_perf, min_variance, max_sharpe
from var import historical_var, historical_cvar, parametric_var

TICKERS = ["AAPL", "MSFT", "JPM", "GLD", "TLT"]
RF      = 0.05  # taux sans risque annuel

# ── DONNÉES ──────────────────────────────────────────────
prices  = get_prices()
returns = get_returns(prices)
mu, sigma = annualize(returns)

# ── PORTEFEUILLES OPTIMAUX ────────────────────────────────
w_minvar = min_variance(mu, sigma)
w_sharpe = max_sharpe(mu, sigma, RF)
w_equal  = np.ones(len(TICKERS)) / len(TICKERS)

# ── AFFICHAGE ─────────────────────────────────────────────
def print_portfolio(name, weights, returns, mu, sigma, rf=RF):
    ret, vol, sharpe = portfolio_perf(weights, mu, sigma)
    var  = historical_var(returns, weights)
    cvar = historical_cvar(returns, weights)
    pvar = parametric_var(returns, weights)

    print(f"\n── {name} ──")
    print(f"  Allocations :")
    for t, w in zip(TICKERS, weights):
        print(f"    {t:<6} {w*100:>6.2f}%")
    print(f"  Rendement annuel   : {ret*100:.2f}%")
    print(f"  Volatilité annuelle: {vol*100:.2f}%")
    print(f"  Ratio de Sharpe    : {sharpe:.4f}")
    print(f"  VaR 95% (histo)    : {var*100:.2f}%")
    print(f"  CVaR 95% (histo)   : {cvar*100:.2f}%")
    print(f"  VaR 95% (param)    : {pvar*100:.2f}%")

print_portfolio("Équipondéré",        w_equal,  returns, mu, sigma)
print_portfolio("Variance Minimale",  w_minvar, returns, mu, sigma)
print_portfolio("Max Sharpe",         w_sharpe, returns, mu, sigma)