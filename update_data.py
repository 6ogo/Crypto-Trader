# Create a file called update_data.py

from external_data import ExternalDataCollector
from fetch_crypto_data import fetch_all_crypto_data
import os

def update_all_data():
    """Update all data sources"""
    print("Updating cryptocurrency price data...")
    fetch_all_crypto_data()
    
    print("Updating external data sources...")
    collector = ExternalDataCollector('data')
    collector.collect_all_external_data()
    
    # Merge data for each symbol
    for symbol in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'XRPUSD']:
        print(f"Merging external data with {symbol} price data...")
        collector.merge_with_price_data(symbol)
    
    print("Data update completed successfully!")

if __name__ == "__main__":
    update_all_data()