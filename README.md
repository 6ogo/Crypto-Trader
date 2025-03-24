# Advanced ML Trading Bot

This repository contains a Python-based trading bot that utilizes machine learning and physics-inspired models to generate trading signals for financial markets.

## Overview

The trading bot implements an advanced strategy combining several technical indicators, machine learning models, and market regime detection to identify potential trading opportunities. It's designed for algorithmic trading across different market conditions (trending, mean-reverting, volatile, or neutral).

## Features

- **Machine Learning Models**: Uses Random Forest for direction prediction and Gradient Boosting for returns magnitude prediction
- **Market Regime Detection**: Dynamically adapts to changing market conditions using Hurst exponent analysis
- **Physics-Inspired Features**: Incorporates concepts like momentum, force, energy, and entropy for market analysis
- **Adaptive Strategy**: Falls back to simpler moving average strategies when insufficient data is available
- **Automatic Stop Loss**: Calculates appropriate stop levels based on Average True Range (ATR)
- **Comprehensive Technical Indicators**: Includes volatility metrics, trend strength indicators, mean reversion signals, and volume analysis

## Requirements

The trading bot requires the following Python packages:
```
numpy
pandas
scipy
scikit-learn
statsmodels
matplotlib (optional, for visualization)
```

You can install these dependencies using:

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

## Usage

### Basic Example

To run a backtest with the default settings and sample data:

```python
from trading_bot import run_backtest
import pandas as pd
import numpy as np

# Generate sample data
dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
np.random.seed(42)
closes = (np.random.normal(loc=0.001, scale=0.01, size=200) + 0.001).cumsum() + 100
    
df = pd.DataFrame({
    'timestamp': dates,
    'open': closes * np.random.normal(loc=1, scale=0.005, size=200),
    'high': closes * np.random.normal(loc=1.02, scale=0.005, size=200),
    'low': closes * np.random.normal(loc=0.98, scale=0.005, size=200),
    'close': closes,
    'volume': np.random.normal(loc=1000000, scale=500000, size=200)
})
    
df.set_index('timestamp', inplace=True)

# Run backtest with ML strategy
print("Running backtest with ML strategy:")
run_backtest(df, use_ml_strategy=True, train_size=150)
```

### With Your Own Data

To use your own OHLCV (Open, High, Low, Close, Volume) data:

```python
import pandas as pd
from trading_bot import run_backtest

# Load your data
df = pd.read_csv('your_data.csv')

# Ensure your dataframe has the required columns: timestamp (or date), open, high, low, close, volume
df.set_index('timestamp', inplace=True)

# Run backtest
run_backtest(df, use_ml_strategy=True, train_size=150)
```

### Using Just the Strategy

If you want to directly use the `AdvancedStrategy` class for your own trading system:

```python
from trading_bot import AdvancedStrategy
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')
df.set_index('timestamp', inplace=True)

# Initialize strategy
strategy = AdvancedStrategy()

# Train the strategy
train_data = df.iloc[:150]
strategy.train(train_data)

# Generate signals for latest data
latest_data = df.iloc[:200]  # some overlap with training data is needed
buy_signals, sell_signals, stop_signals = strategy.generate_signals(latest_data)

# Print latest signal
print(f"Latest buy signal: {buy_signals.iloc[-1]}")
print(f"Latest sell signal: {sell_signals.iloc[-1]}")
```

## Technical Details

### AdvancedStrategy Class

The core of the trading bot is the `AdvancedStrategy` class, which:

1. **Preprocesses data** - Creates feature engineering from raw OHLCV data
2. **Trains multiple models** - A classifier for direction and a regressor for magnitude
3. **Detects market regimes** - Categorizes market as trending, mean-reverting, or volatile
4. **Generates signals** - Produces buy/sell/stop signals based on model predictions and market conditions

Key parameters of the `AdvancedStrategy` class:
- `lookback_period`: Number of historical bars used for feature calculation (default: 60)
- `prediction_horizon`: Number of bars to predict ahead (default: 5)
- `volatility_window`: Window size for volatility calculations (default: 20)
- `trend_window`: Window size for trend calculations (default: 50)
- `mean_reversion_window`: Window size for mean reversion features (default: 14)

### MLPhysicsStrategy

This is a wrapper class providing a combined strategy that:
- Uses the `AdvancedStrategy` when sufficient data is available
- Falls back to a simple Moving Average crossover strategy when data is limited

### Backtesting Function

The `run_backtest` function allows testing the strategy on historical data:
- `df`: DataFrame with OHLCV data
- `use_ml_strategy`: Whether to use ML models or fall back to simpler strategies
- `train_size`: Number of bars to use for initial training (default: 150)

## Feature Categories

The strategy uses several categories of features:

1. **Basic Features**:
   - Returns and log returns

2. **Volatility Features**:
   - Standard deviation of returns
   - Average True Range (ATR)
   - Normalized range

3. **Trend Features**:
   - Moving averages of different lengths
   - Trend strength indicators

4. **Mean Reversion Features**:
   - Z-score
   - RSI (Relative Strength Index)
   - Bollinger Band position

5. **Volume Features**:
   - Volume moving average
   - Volume ratio

6. **Physics-Inspired Features**:
   - Momentum (price velocity)
   - Force Index
   - Market "energy"
   - Price acceleration
   - Approximate entropy

## Example Output

When running the backtest, you'll see output like:

```
Running backtest with ML strategy:
Prediction features shape: (1, 18)
Buy signals: 12
Sell signals: 8
Stop signals: 5
```

This indicates the strategy generated 12 buy signals, 8 sell signals, and 5 stop signals during the backtest period.