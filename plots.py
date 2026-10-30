import numpy as np
import matplotlib.pyplot as plt
from data import get_prices, get_returns
from portfolio import annualize, portfolio_perf, min_variance, max_sharpe, efficient_frontier
from var import historical_var, historical_cvar

RF = 0.05

prices = get_prices()
returns = get_returns(prices)
TICKERS = list(returns.columns)  # noms pris directement dans les données
mu, sigma = annualize(returns)

w_minvar = min_variance(mu, sigma)
w_sharpe = max_sharpe(mu, sigma, RF)
w_equal = np.ones(len(TICKERS)) / len(TICKERS)


def plot_efficient_frontier():
    target_rets, frontier_vols = efficient_frontier(mu, sigma)
    r_min, v_min, s_min = portfolio_perf(w_minvar, mu, sigma, RF)
    r_shp, v_shp, s_shp = portfolio_perf(w_sharpe, mu, sigma, RF)
    r_eq, v_eq, s_eq = portfolio_perf(w_equal, mu, sigma, RF)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Frontière efficiente
    ax.plot(frontier_vols * 100, target_rets * 100,
            color="steelblue", linewidth=2, label="Efficient Frontier")

    # Portefeuilles remarquables
    ax.scatter(v_min * 100, r_min * 100, color="green", s=80, zorder=5, label=f"Min Variance  (Sharpe {s_min:.2f})")
    ax.scatter(v_shp * 100, r_shp * 100, color="tomato", s=80, zorder=5, label=f"Max Sharpe    (Sharpe {s_shp:.2f})")
    ax.scatter(v_eq * 100, r_eq * 100, color="orange", s=80, zorder=5, label=f"Equal Weight  (Sharpe {s_eq:.2f})")

    # Actifs individuels
    for i, t in enumerate(TICKERS):
        ax.scatter(np.sqrt(sigma.iloc[i, i]) * 100, mu.iloc[i] * 100,
                   marker="x", color="gray", s=60, zorder=4)
        ax.annotate(t, (np.sqrt(sigma.iloc[i, i]) * 100, mu.iloc[i] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="gray")

    ax.set_xlabel("Annual Volatility (%)")
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Efficient Frontier — Markowitz (2019–2023)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150)
    plt.show()


def plot_allocations():
    portfolios = {
        "Equal Weight": w_equal,
        "Min Variance": w_minvar,
        "Max Sharpe": w_sharpe,
    }
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (name, weights) in zip(axes, portfolios.items()):
        wedges, texts, autotexts = ax.pie(
            weights,
            labels=[t if w > 0.01 else "" for t, w in zip(TICKERS, weights)],
            autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
            colors=colors,
            startangle=90
        )
        ax.set_title(name, fontsize=10, fontweight="bold")

    plt.suptitle("Portfolio Allocations", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("allocations.png", dpi=150)
    plt.show()


def plot_var():
    portfolio_returns = returns @ w_sharpe
    var = historical_var(returns, w_sharpe)
    cvar = historical_cvar(returns, w_sharpe)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(portfolio_returns * 100, bins=80, color="steelblue",
            alpha=0.7, edgecolor="white", label="Daily Returns")
    ax.axvline(-var * 100, color="tomato", linewidth=2,
               linestyle="--", label=f"VaR 95% = -{var*100:.2f}%")
    ax.axvline(-cvar * 100, color="darkred", linewidth=2,
               linestyle="--", label=f"CVaR 95% = -{cvar*100:.2f}%")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Return Distribution — Max Sharpe Portfolio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("var_distribution.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_efficient_frontier()
    plot_allocations()
    plot_var()
