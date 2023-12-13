import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def fetch_stock_data(ticker, start_date="1990-01-02 00:00:00-05:00", end_date=None):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date)
    
    # Check if 'Dividends' and 'Stock Splits' columns exist before dropping
    columns_to_drop = ["Dividends", "Stock Splits"]
    stock_data = stock_data.drop(columns=[col for col in columns_to_drop if col in stock_data.columns], errors='ignore')

    # Calculate Magnitude and Velocity
    stock_data = calculate_magnitude(stock_data, window=10)
    stock_data = calculate_velocity(stock_data, window=10)

    return stock_data

def calculate_magnitude(data, window=10):
    data['Magnitude'] = data['High'] - data['Low']
    data['Magnitude'] = data['Magnitude'].rolling(window=window).mean()
    return data

def calculate_velocity(data, window=10):
    data['Velocity'] = data['Close'].pct_change() * 100  # Percentage change as a proxy for velocity
    data['Velocity'] = data['Velocity'].rolling(window=window).mean()
    return data

def create_target_column(data, horizon=1):
    data["Tomorrow"] = data["Close"].shift(-horizon)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data

def train_random_forest_model(train_data, predictors, target):
    train_data[predictors] = train_data[predictors].fillna(0)

    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
    model.fit(train_data[predictors], train_data[target])
    return model

def predict(train, test, predictors, model):
    test[predictors] = test[predictors].fillna(0)

    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def generate_additional_ticker_features(sp500, ticker_data, horizon, ticker):
    ticker_data = create_target_column(ticker_data, horizon=1)
    ticker_predictors = ["Close", "Volume", "Open", "High", "Low", "Magnitude", "Velocity"]

    # print(f"\nAdditional Ticker: {ticker} - Before Training\n{ticker_data.head()}")

    # Train model for the additional ticker
    ticker_model = train_random_forest_model(ticker_data, ticker_predictors, "Target")
    
    # Generate predictions for the additional ticker
    ticker_predictions = predict(ticker_data, ticker_data, ticker_predictors, ticker_model)
    
    # Create a new dataframe to store predictions for the additional ticker
    ticker_predictions_df = pd.DataFrame(index=sp500.index)
    ticker_predictions_df[f"Prediction_{ticker}"] = ticker_predictions["Predictions"]

    return ticker_predictions_df

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

        data.loc[test.index, 'Predictions_SP500'] = predictions['Predictions']

    return pd.concat(all_predictions)

def calculate_precision(predictions):
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print("Precision:", precision)

def display_value_counts(predictions):
    value_counts = predictions["Predictions"].value_counts()
    print("Value Counts of Predictions:")
    print(value_counts)

def process_data_and_backtest():
    sp500 = fetch_stock_data("^GSPC", start_date="1990-01-02")
    sp500 = create_target_column(sp500, horizon=1)

    # Train the S&P 500 model first
    predictors = ["Close", "Volume", "Open", "High", "Low", "Magnitude", "Velocity"]
    model = train_random_forest_model(sp500, predictors, "Target")

    # Process data/train model for additional tickers
    additional_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "UNH"]
    additional_predictions_dfs = []

    for ticker in additional_tickers:
        additional_data = fetch_stock_data(ticker, start_date="1990-01-02")
        ticker_predictions_df = generate_additional_ticker_features(sp500, additional_data, horizon=1, ticker=ticker)
        additional_predictions_dfs.append(ticker_predictions_df)

    # Merge all additional ticker predictions into the S&P 500 dataframe
    for i, ticker in enumerate(additional_tickers):
        sp500 = pd.merge(sp500, additional_predictions_dfs[i], left_index=True, right_index=True)

    combined_predictors = predictors + [f"Prediction_{ticker}" for ticker in additional_tickers]

    sp500 = sp500.ffill().dropna()

    final_model = train_random_forest_model(sp500, combined_predictors, "Target")

    backtest_results = backtest(sp500, final_model, combined_predictors)
    calculate_precision(backtest_results)
    display_value_counts(backtest_results)

    sp500['Predictions_SP500'] = backtest_results['Predictions']
    

    return sp500[['Predictions_SP500']]

if __name__ == "__main__":
    result_df = process_data_and_backtest()
    print(result_df)