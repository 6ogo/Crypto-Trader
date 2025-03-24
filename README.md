# Crypto Trading ML Platform

A machine learning-powered platform for cryptocurrency price prediction and automated trading.

## Overview

This application combines machine learning algorithms with cryptocurrency market data to predict price movements and execute trades. It uses historical data to train models that forecast whether crypto prices will rise or fall in the next 12 hours.

## Features

- **ML-Powered Price Predictions**: Uses XGBoost models to predict price movements for BTC, ETH, SOL, AVAX, and XRP
- **Multiple Cryptocurrency Support**: Tracks and trades 5 major cryptocurrencies
- **Real-time Trading**: Connects to Kraken exchange API for live trading
- **Technical Indicators**: Incorporates 50+ technical indicators for prediction models
- **Portfolio Management**: Tracks positions, realized and unrealized P&L
- **Position Tracking**: Manages long and short positions with proper P&L calculations
- **Performance Analytics**: Visualizes trade history and prediction accuracy
- **Secure User Authentication**: Supports multiple users with separate portfolios

## How the Trading Works

### Data Collection
- Historical hourly OHLCV data is fetched from Kraken for the past 4 years
- The `fetch_crypto_data.py` script handles data retrieval and preprocessing

### Machine Learning Model
- The `train_crypto_model.py` script creates and trains XGBoost models for each cryptocurrency
- Features include technical indicators (RSI, MACD, Bollinger Bands), moving averages, volume metrics, and more
- Models are trained to predict price direction (up/down) 12 hours into the future

### Trading System
1. **Prediction Generation**
   - Models analyze current market conditions every hour
   - Each prediction includes direction (up/down) and probability score

2. **Position Management**
   - Users can manually execute trades based on predictions
   - The system tracks positions, calculating average entry prices and P&L
   - Positions can be long (buy) or short (sell)

3. **P&L Calculation**
   - Realized P&L is calculated when positions are closed
   - Unrealized P&L is continuously calculated based on current market prices
   - The system handles partial position closures correctly

4. **API Integration**
   - Trades are executed through the Kraken exchange API
   - Users must provide their Kraken API keys to enable trading

### Prediction Accuracy Tracking
- The system records all predictions and checks outcomes when the prediction horizon passes
- Historical accuracy metrics are displayed for each cryptocurrency

## Components

- **Flask Web Application**: Backend server handling user authentication, trading, and predictions
- **SQLite Database**: Stores user data, trades, positions, and prediction history
- **Machine Learning Pipeline**: Trains and updates prediction models
- **Kraken API Integration**: Connects to the exchange for real-time data and trading

## Dashboard

The main dashboard provides:
- Portfolio value and position overview
- Latest price predictions with trading buttons
- Current positions with unrealized P&L
- Recent trade history
- Current market prices

## Security

- Passwords are stored using secure hashing
- API secrets are protected for each user
- Users can only access their own trading data