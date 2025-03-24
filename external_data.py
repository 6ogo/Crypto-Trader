import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalDataCollector:
    """Collects external data sources for crypto prediction models"""
    
    def __init__(self, data_dir='data'):
        """Initialize with data directory path"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.cache = {}
        
    def fetch_crypto_fear_greed_index(self, days=30):
        """
        Fetch Crypto Fear & Greed Index
        Data source: Alternative.me API
        """
        url = f"https://api.alternative.me/fng/?limit={days}"
        filename = os.path.join(self.data_dir, 'crypto_fear_greed.csv')
        
        try:
            logger.info(f"Fetching Crypto Fear & Greed Index for past {days} days")
            response = requests.get(url)
            data = response.json()
            
            if data.get('data'):
                df = pd.DataFrame(data['data'])
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['value'] = pd.to_numeric(df['value'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Save to CSV
                df.to_csv(filename)
                logger.info(f"Saved Crypto Fear & Greed data to {filename}")
                return df
            else:
                logger.error("Failed to parse Fear & Greed Index data")
                return None
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {str(e)}")
            
            # Try to load from cache if fetch failed
            if os.path.exists(filename):
                logger.info(f"Loading Fear & Greed Index from cache: {filename}")
                return pd.read_csv(filename, index_col=0, parse_dates=True)
            return None
    
    def fetch_market_sentiment(self, days=30):
        """
        Fetch general market sentiment data (using a proxy, since this would normally
        require a premium API like AAII, StockTwits, or Twitter sentiment)
        
        For this example, we'll simulate sentiment based on market performance.
        In a real system, you'd use a proper sentiment API.
        """
        filename = os.path.join(self.data_dir, 'market_sentiment.csv')
        
        try:
            # If we already have a cached version that's recent, use it
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                latest_date = df.index.max()
                
                if latest_date >= (datetime.now() - timedelta(days=2)).date():
                    logger.info(f"Using cached market sentiment from {filename}")
                    return df
            
            # In a real system, you'd call a proper API here. 
            # For this example, we'll generate proxy sentiment based on BTC price.
            # First, load BTC price data
            btc_file = os.path.join(self.data_dir, 'BTCUSD_hourly_4y.csv')
            if not os.path.exists(btc_file):
                logger.error(f"BTC price data not found at {btc_file}")
                return None
                
            btc_data = pd.read_csv(btc_file, index_col=0, parse_dates=True)
            
            # Resample to daily
            daily_data = btc_data['close'].resample('D').last()
            
            # Calculate daily returns
            daily_returns = daily_data.pct_change()
            
            # Get last 30 days
            recent_returns = daily_returns.tail(days)
            
            # Create sentiment index (simple function of returns)
            # In reality, this would come from a sentiment API
            sentiment = pd.DataFrame({
                'sentiment_value': 50 + (recent_returns * 500),  # Scale to 0-100
                'sentiment_label': recent_returns.apply(
                    lambda x: 'very_bearish' if x < -0.05 else
                             'bearish' if x < -0.01 else
                             'neutral' if x < 0.01 else
                             'bullish' if x < 0.05 else
                             'very_bullish'
                )
            })
            
            # Ensure values are in 0-100 range
            sentiment['sentiment_value'] = sentiment['sentiment_value'].clip(0, 100)
            
            # Add a 3-day moving average
            sentiment['sentiment_ma3'] = sentiment['sentiment_value'].rolling(3).mean()
            
            # Save to CSV
            sentiment.to_csv(filename)
            logger.info(f"Generated and saved proxy market sentiment data to {filename}")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error generating market sentiment: {str(e)}")
            
            # Try to load from cache if failed
            if os.path.exists(filename):
                return pd.read_csv(filename, index_col=0, parse_dates=True)
            return None
    
    def fetch_m2_money_supply(self):
        """
        Fetch M2 Money Supply data (using FRED API)
        Normally you'd use an API key, but we'll use a static file for this example
        """
        filename = os.path.join(self.data_dir, 'm2_money_supply.csv')
        
        try:
            # Check if we need to update (monthly data, so only update once a month)
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                latest_date = df.index.max()
                
                if latest_date >= (datetime.now() - timedelta(days=30)).date():
                    logger.info(f"Using cached M2 money supply from {filename}")
                    return df
            
            # In a real system, you'd use the FRED API:
            # url = f"https://api.stlouisfed.org/fred/series/observations?series_id=M2&api_key={api_key}&file_type=json"
            
            # For this example, we'll create a synthetic dataset based on recent trends
            # Start with a base value
            base_value = 21400  # Approx M2 in billions as of 2022
            growth_rates = [0.005, 0.004, 0.003, 0.001, -0.001, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003]
            
            # Generate dates (monthly for past 3 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
            
            # Generate values
            values = [base_value]
            for rate in growth_rates:
                next_val = values[-1] * (1 + rate)
                values.append(next_val)
            
            # Extend to match number of dates
            while len(values) < len(dates):
                next_val = values[-1] * (1 + growth_rates[len(values) % len(growth_rates)])
                values.append(next_val)
            
            # Trim to match
            values = values[:len(dates)]
            
            # Create DataFrame
            df = pd.DataFrame({
                'M2_Money_Supply_Billions': values
            }, index=dates)
            
            # Calculate month-over-month and year-over-year changes
            df['M2_MoM_Change'] = df['M2_Money_Supply_Billions'].pct_change()
            df['M2_YoY_Change'] = df['M2_Money_Supply_Billions'].pct_change(12)
            
            # Save to CSV
            df.to_csv(filename)
            logger.info(f"Generated and saved synthetic M2 money supply data to {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating M2 money supply data: {str(e)}")
            
            # Try to load from cache if failed
            if os.path.exists(filename):
                return pd.read_csv(filename, index_col=0, parse_dates=True)
            return None
    
    def fetch_dxy_data(self, days=365):
        """
        Fetch Dollar Index (DXY) data
        In a real system, you'd use a financial API like Alpha Vantage, FRED, or Yahoo Finance
        """
        filename = os.path.join(self.data_dir, 'dxy_index.csv')
        
        try:
            # Check if we need to update (daily data, only update once a day)
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                latest_date = df.index.max()
                
                if latest_date >= (datetime.now() - timedelta(days=2)).date():
                    logger.info(f"Using cached DXY index data from {filename}")
                    return df
            
            # In a real system, you'd call an API like:
            # url = f"https://api.example.com/forex/DXY/historical?days={days}&api_key={api_key}"
            
            # For this example, we'll generate proxy DXY data based on relative strength patterns
            # Start with a base value
            base_value = 102.0  # Approximate recent DXY value
            
            # Simulate fluctuations with slight upward trend
            volatility = 0.005
            trend = 0.0001
            
            # Generate dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate values with random walk + trend
            values = [base_value]
            for i in range(1, len(dates)):
                change = np.random.normal(trend, volatility)
                next_val = values[-1] * (1 + change)
                values.append(next_val)
            
            # Create DataFrame
            df = pd.DataFrame({
                'DXY_Close': values,
            }, index=dates)
            
            # Calculate day-over-day and week-over-week changes
            df['DXY_DoD_Change'] = df['DXY_Close'].pct_change()
            df['DXY_WoW_Change'] = df['DXY_Close'].pct_change(7)
            
            # Save to CSV
            df.to_csv(filename)
            logger.info(f"Generated and saved synthetic DXY index data to {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating DXY index data: {str(e)}")
            
            # Try to load from cache if failed
            if os.path.exists(filename):
                return pd.read_csv(filename, index_col=0, parse_dates=True)
            return None

    def fetch_onchain_metrics(self, symbol='BTC', days=180):
        """
        Fetch on-chain metrics for Bitcoin/crypto
        In a real system, you'd use an API like Glassnode, CryptoQuant, or Coin Metrics
        """
        filename = os.path.join(self.data_dir, f'{symbol}_onchain.csv')
        
        try:
            # Check if we need to update
            if os.path.exists(filename):
                df = pd.read_csv(filename, index_col=0, parse_dates=True)
                latest_date = df.index.max()
                
                if latest_date >= (datetime.now() - timedelta(days=3)).date():
                    logger.info(f"Using cached on-chain data from {filename}")
                    return df
            
            # Get price data to simulate realistic on-chain metrics
            price_file = os.path.join(self.data_dir, f"{symbol}USD_hourly_4y.csv")
            if not os.path.exists(price_file):
                logger.error(f"{symbol} price data not found at {price_file}")
                return None
                
            price_data = pd.read_csv(price_file, index_col=0, parse_dates=True)
            daily_prices = price_data['close'].resample('D').last()
            recent_prices = daily_prices.tail(days)
            
            # Generate dates
            dates = recent_prices.index
            
            # Generate synthetic on-chain metrics based on price action
            # Active addresses
            active_addresses_base = 1000000 if symbol == 'BTC' else 500000
            active_addresses = active_addresses_base + recent_prices * (10 if symbol == 'BTC' else 5)
            active_addresses = active_addresses * (1 + np.random.normal(0, 0.1, len(dates)))
            
            # Transaction count
            tx_count_base = 300000 if symbol == 'BTC' else 1200000
            tx_count = tx_count_base + recent_prices * (5 if symbol == 'BTC' else 20)
            tx_count = tx_count * (1 + np.random.normal(0, 0.15, len(dates)))
            
            # Exchange inflow/outflow
            exchange_inflow = recent_prices * (0.1 + np.random.normal(0, 0.05, len(dates)))
            exchange_outflow = recent_prices * (0.09 + np.random.normal(0, 0.06, len(dates)))
            net_flow = exchange_outflow - exchange_inflow
            
            # Create DataFrame
            df = pd.DataFrame({
                'active_addresses': active_addresses,
                'transaction_count': tx_count,
                'exchange_inflow': exchange_inflow,
                'exchange_outflow': exchange_outflow,
                'net_exchange_flow': net_flow,
                'price': recent_prices
            })
            
            # Add derived metrics
            df['active_addresses_change'] = df['active_addresses'].pct_change(7)
            df['transaction_count_change'] = df['transaction_count'].pct_change(7)
            df['net_flow_intensity'] = df['net_exchange_flow'] / df['price']
            
            # Save to CSV
            df.to_csv(filename)
            logger.info(f"Generated and saved synthetic on-chain data for {symbol} to {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error generating on-chain data: {str(e)}")
            
            # Try to load from cache if failed
            if os.path.exists(filename):
                return pd.read_csv(filename, index_col=0, parse_dates=True)
            return None

    def collect_all_external_data(self):
        """Modified version to add additional data sources"""
        # Fetch existing data sources
        fear_greed = self.fetch_crypto_fear_greed_index()
        market_sentiment = self.fetch_market_sentiment()
        m2_supply = self.fetch_m2_money_supply()
        
        # New data sources
        dxy_data = self.fetch_dxy_data()
        btc_onchain = self.fetch_onchain_metrics('BTC')
        eth_onchain = self.fetch_onchain_metrics('ETH')
        
        # Create a daily date range for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a base DataFrame with the date range
        combined_df = pd.DataFrame(index=date_range)
        
        # Add existing data
        if fear_greed is not None:
            # Resample to daily just in case
            fear_greed_daily = fear_greed.resample('D').last()
            combined_df['fear_greed_value'] = fear_greed_daily['value']
            combined_df['fear_greed_class'] = fear_greed_daily['value_classification']
            
            # Add contrarian indicator (100 - fear_greed)
            combined_df['fear_greed_contrarian'] = 100 - combined_df['fear_greed_value']
                
        if market_sentiment is not None:
            combined_df['market_sentiment'] = market_sentiment['sentiment_value']
            combined_df['market_sentiment_ma3'] = market_sentiment['sentiment_ma3']
            combined_df['market_sentiment_label'] = market_sentiment['sentiment_label']
            
        if m2_supply is not None:
            # Forward fill to daily
            m2_daily = m2_supply.resample('D').ffill()
            combined_df['m2_money_supply'] = m2_daily['M2_Money_Supply_Billions']
            combined_df['m2_mom_change'] = m2_daily['M2_MoM_Change']
            combined_df['m2_yoy_change'] = m2_daily['M2_YoY_Change']
            
            # Add lagged M2 values (57-day lag as mentioned)
            combined_df['m2_lag57'] = combined_df['m2_money_supply'].shift(57)
            combined_df['m2_change_lag57'] = combined_df['m2_yoy_change'].shift(57)
        
        # Add new data sources
        if dxy_data is not None:
            dxy_daily = dxy_data.resample('D').ffill()
            combined_df['dxy_index'] = dxy_daily['DXY_Close']
            combined_df['dxy_dod_change'] = dxy_daily['DXY_DoD_Change']
            combined_df['dxy_wow_change'] = dxy_daily['DXY_WoW_Change']
            
            # Add inverse DXY (to capture expected inverse relationship with crypto)
            combined_df['inverse_dxy'] = 1 / combined_df['dxy_index'] * 100
        
        if btc_onchain is not None:
            btc_daily = btc_onchain.resample('D').ffill()
            combined_df['btc_active_addresses'] = btc_daily['active_addresses']
            combined_df['btc_txn_count'] = btc_daily['transaction_count']
            combined_df['btc_net_exchange_flow'] = btc_daily['net_exchange_flow']
            combined_df['btc_net_flow_intensity'] = btc_daily['net_flow_intensity']
        
        if eth_onchain is not None:
            eth_daily = eth_onchain.resample('D').ffill()
            combined_df['eth_active_addresses'] = eth_daily['active_addresses']
            combined_df['eth_txn_count'] = eth_daily['transaction_count']
            combined_df['eth_net_exchange_flow'] = eth_daily['net_exchange_flow']
            combined_df['eth_net_flow_intensity'] = eth_daily['net_flow_intensity']
        
        # Forward fill any missing values
        combined_df = combined_df.ffill()
        
        # Save combined data
        output_file = os.path.join(self.data_dir, 'combined_external_data.csv')
        combined_df.to_csv(output_file)
        logger.info(f"Saved combined external data to {output_file}")
        
        return combined_df
    
    def merge_with_price_data(self, symbol):
        """Merge external data with price data for a specific symbol"""
        # Load price data
        price_file = os.path.join(self.data_dir, f"{symbol}_hourly_4y.csv")
        if not os.path.exists(price_file):
            logger.error(f"Price data for {symbol} not found at {price_file}")
            return None
            
        price_data = pd.read_csv(price_file, index_col=0, parse_dates=True)
        
        # Load external data
        external_file = os.path.join(self.data_dir, 'combined_external_data.csv')
        if not os.path.exists(external_file):
            # If it doesn't exist, collect it
            external_data = self.collect_all_external_data()
        else:
            external_data = pd.read_csv(external_file, index_col=0, parse_dates=True)
        
        # Resample price data to daily for easier merging
        daily_price = price_data.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Merge data
        merged_data = daily_price.merge(external_data, 
                                        left_index=True, 
                                        right_index=True, 
                                        how='left')
        
        # Forward fill missing external data
        external_cols = [col for col in merged_data.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume']]
        merged_data[external_cols] = merged_data[external_cols].ffill().bfill()
        
        # Save merged data
        output_file = os.path.join(self.data_dir, f"{symbol}_with_external_data.csv")
        merged_data.to_csv(output_file)
        logger.info(f"Merged external data with {symbol} price data and saved to {output_file}")
        
        return merged_data

# Add this function to train_crypto_model.py to use the external data

def add_external_features(self, df):
    """Add external data features to the dataframe"""
    symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTCUSD'
    
    # Try to load merged data
    external_file = os.path.join(self.data_dir, f"{symbol}_with_external_data.csv")
    
    if not os.path.exists(external_file):
        # If file doesn't exist, try to create it
        from external_data import ExternalDataCollector
        collector = ExternalDataCollector(self.data_dir)
        merged_data = collector.merge_with_price_data(symbol)
    else:
        merged_data = pd.read_csv(external_file, index_col=0, parse_dates=True)
    
    if merged_data is None:
        print(f"Warning: No external data available for {symbol}")
        return df
    
    # Resample external data to match df's frequency (likely hourly)
    # For this, we'll need to determine df's frequency
    freq = pd.infer_freq(df.index)
    if freq is None:
        # If we can't infer frequency, assume it's hourly
        freq = 'H'
    
    # Resample by forward filling
    resampled = merged_data.resample(freq).ffill()
    
    # Get external columns
    external_cols = [col for col in resampled.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    # Merge with original df (on index)
    result = df.copy()
    for col in external_cols:
        result[f'ext_{col}'] = resampled[col]
    
    # Handle any NaN values
    for col in [c for c in result.columns if c.startswith('ext_')]:
        result[col] = result[col].ffill().bfill()
        
    return result