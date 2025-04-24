import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

file_path = 'portfolio_data.csv'
df = pd.read_csv(file_path)

price_data = df[['AAPL', 'MSFT', 'GOOGL', 'TSLA']]
tickers = price_data.columns.tolist()

returns = price_data.pct_change().dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

annual_returns = mean_returns * 252
annual_cov_matrix = cov_matrix * 252

try:
    investment_amount = float(input("Enter investment amount (default 10000): ") or 10000)
    risk_tolerance = float(input("Enter max risk (std dev as decimal, default 0.05): ") or 0.05)
except ValueError:
    print("Invalid input, using default values.")
    investment_amount = 10000
    risk_tolerance = 0.05

def portfolio_return(weights):
    return np.dot(weights, annual_returns)

def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights)))

def portfolio_sharpe_ratio(weights, risk_free_rate=0.02):
    return (portfolio_return(weights) - risk_free_rate) / portfolio_volatility(weights)

def objective(weights):
    return -portfolio_return(weights)

constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: risk_tolerance - portfolio_volatility(x)}
)

bounds = tuple((0, 1) for _ in range(len(tickers)))
init_guess = np.array([1 / len(tickers)] * len(tickers))

result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = result.x

allocation = opt_weights * investment_amount

final_return = portfolio_return(opt_weights)
final_volatility = portfolio_volatility(opt_weights)
final_sharpe = portfolio_sharpe_ratio(opt_weights)

print("\nOptimized Portfolio Allocation:")
for ticker, alloc in zip(tickers, allocation):
    print(f"{ticker}: ${alloc:.2f}")

print(f"\nExpected Annual Return: {final_return:.2%}")
print(f"Annual Volatility (Risk): {final_volatility:.2%}")
print(f"Sharpe Ratio: {final_sharpe:.2f}")

plt.figure(figsize=(10, 6))
plt.bar(tickers, allocation, color='lightgreen')
plt.title(f'Optimized Portfolio Allocation (Investment: ${investment_amount:.2f})')
plt.ylabel('Allocation ($)')
plt.grid(True)
plt.tight_layout()
plt.show()
