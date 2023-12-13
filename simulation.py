import pandas as pd
from datetime import datetime, timedelta
from GSPC_model import process_data_and_backtest
import yfinance as yf

def simulate_trading_algorithm(starting_balance, data, index_date):
    # Initialize variables
    current_balance = starting_balance
    shares_held = 0
    cost_basis = 0

    for i, current_date in enumerate(index_date[:-1]): 
        # Extract relevant data
        row = data.loc[current_date]
        prediction = row['Predictions_SP500']
        vfiax_close = row['VFIAX_close']

        # Buy decision
        if prediction == 1 and shares_held == 0:
            shares_bought = current_balance / vfiax_close
            cost_basis = current_balance
            shares_held += shares_bought
            current_balance = 0
            print("------")
            print(f"Date: {current_date}\nDecision: Buy {shares_bought} shares at {vfiax_close} per share.\nCurrent balance: {current_balance}\n")
            print("------")

        # Hold decision
        elif prediction == 1 and shares_held > 0:
            current_balance = shares_held * vfiax_close
            print("------")
            print(f"Date: {current_date}\nDecision: Hold.\nCurrent balance: {current_balance}\n")
            print("------")

        # Sell decision
        elif prediction == 0 and shares_held > 0:
            current_balance = shares_held * vfiax_close
            profit_loss = current_balance - cost_basis
            shares_held = 0
            cost_basis = 0
            print("------")
            print(f"Date: {current_date}\nDecision:Sell at {vfiax_close} per share.\nProfit/Loss: {profit_loss}.\nCurrent balance: {current_balance}\n")
            print("------")

    return current_balance

if __name__ == "__main__":
    vfiax = yf.Ticker("VFIAX")
    vfiax = vfiax.history(start="1990-01-02", end=None)

    sp500 = process_data_and_backtest()

    sp500 = pd.merge(sp500, vfiax['Close'].rename('VFIAX_close'), left_index=True, right_index=True, how='left')

    result = simulate_trading_algorithm(10000, sp500, sp500.index)

    print("Final balance:", result)

