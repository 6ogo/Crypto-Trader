import pandas as pd
import time
import datetime
import os
from pykrakenapi import KrakenAPI
import krakenex
import numpy as np
from tqdm import tqdm

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Symbol mapping for Kraken API
SYMBOL_MAPPING = {
    "BTC/USD": "XXBTZUSD",
    "ETH/USD": "XETHZUSD",
    "SOL/USD": "SOLUSD",  
    "AVAX/USD": "AVAXUSD",
    "XRP/USD": "XXRPZUSD"
}

def fetch_kraken_historical_data(symbol_pair, interval=60, start_date=None):
    """
    Fetch historical OHLCV data from Kraken for a specific symbol
    
    Parameters:
    symbol_pair (str): Trading pair in Kraken format (e.g., 'XXBTZUSD')
    interval (int): Time interval in minutes
    start_date (datetime): Starting date for data collection
    
    Returns:
    pandas.DataFrame: Historical OHLCV data
    """
    # Initialize the API
    kraken = krakenex.API()
    api = KrakenAPI(kraken)
    
    # Convert start_date to UNIX timestamp
    if start_date:
        since = int(start_date.timestamp())
    else:
        # Default to 4 years ago
        since = int(time.time() - (4 * 365 * 24 * 60 * 60))
    
    all_data = []
    last = since
    
    # Kraken limits results to 720 data points per request
    # We need to make multiple requests to get all the data
    with tqdm(desc=f"Fetching {symbol_pair}", unit="batch") as pbar:
        while True:
            try:
                # Fetch OHLC data
                ohlc, last_id = api.get_ohlc_data(symbol_pair, interval=interval, since=last)
                
                # If no new data was returned or we reached current time, break
                if len(ohlc) == 0 or (last_id <= last):
                    break
                
                all_data.append(ohlc)
                last = last_id
                
                # Update progress bar
                pbar.update(1)
                
                # Respect Kraken's API rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Error fetching {symbol_pair}: {str(e)}")
                # Wait longer on error (might be rate limiting)
                time.sleep(10)
                continue
    
    # Combine all fetched data
    if not all_data:
        print(f"No data retrieved for {symbol_pair}")
        return None
    
    combined_df = pd.concat(all_data)
    
    # Remove duplicates (can happen when fetching batches)
    combined_df = combined_df.drop_duplicates()
    
    # Sort by timestamp
    combined_df.sort_index(inplace=True)
    
    return combined_df

def process_and_save_data(symbol, data):
    """
    Process the raw data and save to CSV
    """
    if data is None:
        return
    
    # Rename columns
    data = data.rename(columns={
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })
    
    # Keep only required columns
    data = data[['open', 'high', 'low', 'close', 'volume']]
    
    # Add symbol column
    data['symbol'] = symbol.replace('/', '')
    
    # Convert index to datetime for readability
    data.index = pd.to_datetime(data.index, unit='s')
    
    # Save to CSV
    filename = f"data/{symbol.replace('/', '')}_hourly_4y.csv"
    data.to_csv(filename)
    print(f"Saved {len(data)} records to {filename}")
    
    return data

def fetch_all_crypto_data():
    """
    Fetch data for all specified cryptocurrencies
    """
    # Calculate start date (4 years ago)
    start_date = datetime.datetime.now() - datetime.timedelta(days=4*365)
    
    all_symbol_data = {}
    
    for symbol, kraken_symbol in SYMBOL_MAPPING.items():
        print(f"\nFetching {symbol} ({kraken_symbol}) data from {start_date}...")
        
        # Fetch the data
        data = fetch_kraken_historical_data(kraken_symbol, interval=60, start_date=start_date)
        
        # Process and save
        if data is not None:
            processed_data = process_and_save_data(symbol, data)
            all_symbol_data[symbol] = processed_data
    
    return all_symbol_data

if __name__ == "__main__":
    print("Starting to fetch 4 years of hourly data for BTC/USD, ETH/USD, SOL/USD, AVAX/USD, XRP/USD")
    fetch_all_crypto_data()
    print("\nData fetching completed!")