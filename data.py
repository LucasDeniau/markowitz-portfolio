import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "MSFT", "JPM", "GLD", "TLT"]
START = "2019-01-01"
END = "2024-01-01"


def get_prices():
    """Télécharge les prix de clôture ajustés sur 5 ans (2019-2023)."""
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    # yfinance trie les colonnes par ordre alphabétique :
    # on les remet dans l'ordre de TICKERS pour que les étiquettes correspondent aux données
    prices = raw["Close"][TICKERS]
    prices = prices.dropna()
    return prices


def get_returns(prices):
    """Calcule les rendements journaliers."""
    returns = prices.pct_change().dropna()
    return returns


if __name__ == "__main__":
    prices = get_prices()
    returns = get_returns(prices)
    print(f"Prix : {prices.shape}")
    print(f"Rendements : {returns.shape}")
    print(returns.head())
