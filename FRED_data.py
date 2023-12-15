from fredapi import Fred
import pandas as pd

FRED_API_KEY = 'c8ffb85d56a65bb030644d9d02528564'

important_series = [
    "CPIAUCNS",  # Consumer price index
    "DJIA",  # Dow Jones Industrial Average index
    "FEDFUNDS",  # Federal funds interest rate
    "GS10",  # 10-Year treasury yield
    "M2",  # Money stock measures (i.e., savings and related balances)
    "MICH",  # University of Michigan: inflation expectation
    "UMCSENT",  # University of Michigan: consumer sentiment
    "UNRATE",  # Unemployment rate
    "WALCL",  # US assets, total assets (less eliminations from consolidation)
]

def fetch_fred_data(indicators, start_date):
    fred = Fred(api_key=FRED_API_KEY)
    dfs = []

    for series_id in indicators:
        data = fred.get_series(series_id, start_date)
        df = pd.DataFrame({f"{series_id}": data})
        dfs.append(df)

    result_df = pd.concat(dfs, axis=1)

    result_df.ffill(inplace=True)

    return result_df
