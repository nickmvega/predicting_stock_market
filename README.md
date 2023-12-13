# Stock Market Prediction and Trading Simulation

## Overview

This project includes Python scripts for fetching historical stock data, training machine learning models, and simulating a trading algorithm based on predictions. The primary files are:

- **GSPC_model.py:** Python script for data processing, model training, and backtesting.
- **GSPC_model.ipynb:** Jupyter Notebook for interactive data exploration and visualization.
- **simulation.py:** Python script for simulating a trading algorithm using model predictions.
- **simulation.ipynb:** Jupyter Notebook for interactive simulation of the trading algorithm.

## GSPC_model.py

This script utilizes the `yfinance` library to fetch historical stock data, calculates additional features such as magnitude and velocity, and trains a random forest model. It also includes functions for backtesting and generating predictions for the S&P 500 index and additional tickers.

## GSPC_model.ipynb

This Jupyter Notebook contains the same code as GSPC_model.py, providing an interactive environment for data exploration, visualization, and experimentation.

## simulation.py

This script simulates a simple trading algorithm based on the predictions generated by the model in GSPC_model.py made for the S&P500. It defines a function `simulate_trading_algorithm` that takes a starting balance, stock data, and index dates as input and simulates trading decisions (buy, hold, sell) for the VFIAX (Vanguard 500 Index Fund).

## simulation.ipynb

This Jupyter Notebook provides an interactive environment for simulating the trading algorithm using the functions defined in simulation.py. It allows users to experiment with different parameters and visualize the results.

## Usage

1. Ensure you have the required dependencies installed:

    ```
    pip install yfinance pandas scikit-learn
    ```

2. Run GSPC_model.py to fetch data, train models, and generate predictions:

    ```
    python3 GSPC_model.py
    ```

3. Explore and visualize data interactively using GSPC_model.ipynb.

4. Simulate the trading algorithm using simulation.py:

    ```
    python3 simulation.py
    ```

5. Interactively simulate and experiment with the trading algorithm using simulation.ipynb.

## Dependencies

- `yfinance`: Fetch historical stock data.
- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning library for model training and evaluation.
- `jupyterlab`: Interactive computing in a Jupyter environment (required for running Jupyter Notebooks).
