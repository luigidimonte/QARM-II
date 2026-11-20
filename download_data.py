import yfinance as yf

CORE = ["SPY", "EFA", "EEM", "IEF", "LQD", "TIP", "GLD", "VNQ"]

prices = yf.download(CORE, start="2000-01-01", auto_adjust=True)["Close"]
daily_return = prices.pct_change().dropna()

# same format as the CSV used in the labs: date, ticker, daily_return
data = daily_return.stack().reset_index()
data.columns = ["date", "ticker", "daily_return"]
data.to_csv("etf_returns.csv", index=False)
print(data.head())


