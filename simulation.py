import pandas as pd
from datetime import datetime, timedelta
from GSPC_model import process_data_and_backtest
import yfinance as yf

def simulate_trading_algorithm(starting_balance, data, index_date):
    current_balance = starting_balance
    shares_held = 0
    cost_basis = 0

    trading_log = []

    for i, current_date in enumerate(index_date[:-1]):

        row = data.loc[current_date]
        prediction = row['Predictions_SP500']
        vfiax_close = row['VFIAX_close']

        # Buy
        if prediction == 1 and shares_held == 0:
            shares_bought = current_balance / vfiax_close
            cost_basis = current_balance
            shares_held += shares_bought
            current_balance = 0

            trading_log.append({
                'Date': current_date,
                'Prediction': prediction,
                'Decision': 'Buy',
                'Shares': shares_bought,
                'Price': vfiax_close,
                'Profit/Loss': 0,
                'Current Balance': current_balance
            })

        # Hold
        elif prediction == 1 and shares_held > 0:
            current_balance = shares_held * vfiax_close

            trading_log.append({
                'Date': current_date,
                'Prediction': prediction,
                'Decision': 'Hold',
                'Shares': shares_held,
                'Price': vfiax_close,
                'Profit/Loss': 0,
                'Current Balance': current_balance
            })

        # Sell
        elif prediction == 0 and shares_held > 0:
            current_balance = shares_held * vfiax_close
            profit_loss = current_balance - cost_basis
            shares_held = 0
            cost_basis = 0

            trading_log.append({
                'Date': current_date,
                'Prediction': prediction,
                'Decision': 'Sell',
                'Shares': 0,
                'Price': vfiax_close,
                'Profit/Loss': profit_loss,
                'Current Balance': current_balance
            })

    trading_log_df = pd.concat([pd.DataFrame([x]).round({'Shares': 2, 'Price': 2, 'Profit/Loss': 2, 'Current Balance': 2}) for x in trading_log], ignore_index=True)

    return current_balance, trading_log_df

if __name__ == "__main__":
    vfiax = yf.Ticker("VFIAX")
    vfiax = vfiax.history(start="1990-01-02", end=None)

    sp500 = process_data_and_backtest()

    sp500 = pd.merge(sp500, vfiax['Close'].rename('VFIAX_close'), left_index=True, right_index=True, how='left')

    result, trading_log = simulate_trading_algorithm(10000, sp500, sp500.index)

    print("Final balance:", result)

    print("\nTrading Log:")
    print(trading_log)
