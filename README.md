# Markowitz Portfolio Optimization

> Mean-variance portfolio optimization with efficient frontier, Sharpe ratio maximization and risk metrics (VaR/CVaR).

---

## Overview

For my second quantitative finance project, I implemented Harry Markowitz's Modern Portfolio Theory from scratch — the mathematical framework behind rational asset allocation. Given a set of risky assets, the model finds the optimal capital allocation that either minimizes risk for a target return, or maximizes the risk-adjusted return (Sharpe ratio). Risk is further quantified through Value-at-Risk and Conditional Value-at-Risk metrics.

Built on 5 years of real daily market data (2019–2023) across five assets: AAPL, MSFT, JPM, GLD (gold) and TLT (long-term US Treasuries) — intentionally diversified across equities, commodities and fixed income.

---

## Theory

### Returns & Covariance

For a portfolio of **n assets** with weight vector **w** (∑wᵢ = 1, wᵢ ≥ 0, no short selling):

$$\mu_p = w^\top \mu, \qquad \sigma_p^2 = w^\top \Sigma w$$

where **μ** is the vector of annualized expected returns and **Σ** the annualized covariance matrix (daily estimates scaled by 252 trading days).

### Sharpe Ratio

$$S = \frac{\mu_p - r_f}{\sigma_p}$$

The Sharpe ratio measures excess return per unit of risk. We use a constant $r_f = 5\%$ as a simplifying assumption (close to the US short-term rate at the end of the period).

### Efficient Frontier

The efficient frontier is the set of portfolios that minimize volatility for each level of target return. It is obtained by solving, for each target $\mu^*$:

$$\min_w \; w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mu = \mu^*, \quad \sum w_i = 1, \quad w_i \geq 0$$

Only the upper branch, above the minimum variance portfolio, is efficient. Two portfolios of particular interest lie on the frontier:

- **Minimum Variance Portfolio** — lowest achievable risk (does not depend on expected returns)
- **Maximum Sharpe Portfolio** — best risk-adjusted return (tangency portfolio)

Optimizations are solved with SLSQP (`scipy.optimize.minimize`).

### Value-at-Risk & CVaR

**Historical VaR** at confidence level α: the daily loss not exceeded with probability α, computed directly from observed portfolio returns — no distributional assumption.

**Parametric VaR**: same measure assuming normally distributed returns, $\text{VaR}_\alpha = -(\mu + z_{1-\alpha}\,\sigma)$.

**CVaR (Expected Shortfall)**: the average loss in the worst $(1-\alpha)$ scenarios:

$$\text{CVaR}_\alpha = -\mathbb{E}[r \mid r < -\text{VaR}_\alpha]$$

CVaR is a coherent (sub-additive) risk measure and captures the severity of tail losses, which VaR ignores.

---

## Results

Parameters: **5 assets, daily data 2019–2023, $r_f = 5\%$, confidence = 95%**

### Portfolio Comparison

| Metric                       | Equal Weight | Min Variance | Max Sharpe |
|------------------------------|--------------|--------------|------------|
| **Annual Return**            | 19.70%       | 9.07%        | 27.40%     |
| **Annual Volatility**        | 16.07%       | 10.81%       | 21.05%     |
| **Sharpe Ratio**             | 0.91         | 0.38         | 1.06       |
| **VaR 95% (daily, hist.)**   | 1.41%        | 1.05%        | 2.04%      |
| **VaR 95% (daily, param.)**  | 1.59%        | 1.08%        | 2.07%      |
| **CVaR 95% (daily, hist.)**  | 2.30%        | 1.47%        | 3.01%      |

### Allocations

| Asset | Min Variance | Max Sharpe |
|-------|--------------|------------|
| AAPL  | 2.30%        | 49.27%     |
| MSFT  | 3.87%        | 16.46%     |
| JPM   | 16.50%       | 0.00%      |
| GLD   | 39.23%       | 34.28%     |
| TLT   | 38.10%       | 0.00%      |

### Interpretation

- **Minimum variance** relies on the two least volatile assets, gold and long-term Treasuries (≈ 77% combined), and reaches a volatility of 10.81%, below that of any single asset in the universe. This illustrates the diversification effect of combining weakly correlated assets.
- **Maximum Sharpe** combines Apple with a large allocation to gold (34%): gold has a high return over the period and a low correlation with tech stocks, which improves the risk/return trade-off. JPM and TLT are excluded — TLT suffered from the 2022 rate hikes.
- **Equal weight** lies inside the frontier: the same return could be achieved with lower volatility.
- In the tails, the Max Sharpe CVaR (3.01%) is about 1.5× its VaR (2.04%), showing that losses beyond the VaR threshold can be severe.

### Limitations

- **In-sample bias**: weights are optimized on the same period used to measure performance; 2019–2023 was exceptional for US tech.
- **Estimation error**: mean-variance optimization is highly sensitive to expected-return estimates, which leads to concentrated portfolios. Natural extensions: covariance shrinkage (Ledoit-Wolf), Black-Litterman, out-of-sample backtesting.
- **Constant risk-free rate** of 5% over a period where rates were near zero until 2022.
- **Parametric VaR** assumes normality and underestimates tail risk when returns are fat-tailed.

### Visualizations

![Efficient Frontier](efficient_frontier.png)
*Efficient frontier with individual assets (×), minimum variance portfolio (green), maximum Sharpe portfolio (red) and equal weight portfolio (orange).*

![Portfolio Allocations](allocations.png)
*Capital allocation across the three portfolio strategies.*

![VaR Distribution](var_distribution.png)
*Daily return distribution of the Max Sharpe portfolio with VaR and CVaR thresholds.*

---

## Project Structure

```
markowitz-portfolio/
├── data.py          # Price download via yfinance (2019–2023)
├── portfolio.py     # Annualization, optimization, efficient frontier
├── var.py           # Historical VaR, CVaR, parametric VaR
├── plots.py         # Efficient frontier, allocations, return distribution
└── main.py          # Entry point — prints all portfolio statistics
```

---

## Usage

```bash
# Install dependencies
pip install numpy scipy matplotlib yfinance pandas

# Run portfolio optimization
python main.py

# Generate visualizations
python plots.py
```

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Rockafellar, R. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk.
- Hull, J. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
