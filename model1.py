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

    return stock_data

def create_target_column(data, horizon=1):
    data["Tomorrow"] = data["Close"].shift(-horizon)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data

def train_random_forest_model(train_data, predictors, target):
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
    model.fit(train_data[predictors], train_data[target])
    return model

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def generate_additional_ticker_features(sp500, ticker_data, horizon, ticker):
    ticker_data = create_target_column(ticker_data, horizon=1)
    ticker_predictors = ["Close", "Volume", "Open", "High", "Low"]
    
    # Train model for the additional ticker
    ticker_model = train_random_forest_model(ticker_data, ticker_predictors, "Target")
    
    # Generate predictions for the additional ticker
    ticker_predictions = predict(ticker_data, ticker_data, ticker_predictors, ticker_model)
    
    # Add the predictions as a feature for the S&P 500 dataframe
    sp500[f"Prediction_{ticker}"] = ticker_predictions["Predictions"]

    # Forward-fill missing values in the dataframe
    sp500 = sp500.ffill()

    return sp500

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)

def calculate_precision(predictions):
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print("Precision:", precision)

def display_value_counts(predictions):
    value_counts = predictions["Predictions"].value_counts()
    print("Value Counts of Predictions:")
    print(value_counts)

if __name__ == "__main__":
    sp500 = fetch_stock_data("^GSPC", start_date="1990-01-02")
    sp500 = create_target_column(sp500, horizon=1)

    # Train the S&P 500 model first
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    model = train_random_forest_model(sp500, predictors, "Target")

    # Fetch and process data for additional tickers
    additional_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "UNH"]
    for ticker in additional_tickers:
        additional_data = fetch_stock_data(ticker, start_date="1990-01-02")
        sp500 = generate_additional_ticker_features(sp500, additional_data, horizon=1, ticker=ticker)

    # Update predictors with the additional features
    combined_predictors = predictors + [f"Prediction_{ticker}" for ticker in additional_tickers]

    # Impute missing values in the final dataframe
    sp500 = sp500.ffill().dropna()

    # Train the final model
    final_model = train_random_forest_model(sp500, combined_predictors, "Target")

    # Perform backtesting and evaluation as before
    backtest_results = backtest(sp500, final_model, combined_predictors)
    calculate_precision(backtest_results)
    display_value_counts(backtest_results)