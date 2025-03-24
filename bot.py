import ccxt
import pandas as pd
import numpy as np
import time
import datetime
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dotenv import load_dotenv  # Added for secure API key handling

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

# Configuration class for the trading bot
@dataclass
class TradeConfig:
    """Configuration parameters for the trading bot"""
    exchange_id: str
    api_key: str
    api_secret: str
    trading_pairs: Optional[List[str]] = None  # List of pairs or None for high-volume pairs
    timeframe: str = '1h'
    strategy: str = 'combined'
    max_position_size: float = 0.02  # As percentage of account balance
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.05
    leverage: float = 1.0  # 1.0 means no leverage
    backtest_days: int = 30
    high_volume_top_n: int = 10  # Number of top high-volume pairs if trading_pairs is None
    paper_trading: bool = True  # Added paper trading mode flag
    risk_per_trade_pct: float = 0.01  # Added risk per trade as percentage

# Exchange operations handler
class Exchange:
    """Handles all exchange-related operations"""
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.exchange = self._initialize_exchange()
        self.paper_balance = {"USDT": 1000.0}  # Default paper trading balance
        self.paper_positions = {}  # Track paper trading positions
        
    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange_id)
            exchange = exchange_class({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}  # Kraken uses 'spot' for cryptocurrency trading
            })
            logger.info(f"Successfully connected to {self.config.exchange_id}")
            
            # Load paper trading balance from file if exists
            if self.config.paper_trading:
                paper_file = "paper_trading_state.json"
                if os.path.exists(paper_file):
                    with open(paper_file, 'r') as f:
                        state = json.load(f)
                        self.paper_balance = state.get('balance', self.paper_balance)
                        self.paper_positions = state.get('positions', {})
                logger.info(f"Paper trading mode enabled with balance: {self.paper_balance}")
                
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def _save_paper_trading_state(self):
        """Save paper trading state to file"""
        if self.config.paper_trading:
            with open("paper_trading_state.json", 'w') as f:
                json.dump({
                    'balance': self.paper_balance,
                    'positions': self.paper_positions
                }, f, indent=2)
            
    def fetch_ohlcv(self, pair: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data from exchange for a specific pair"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=pair,
                timeframe=self.config.timeframe,
                limit=limit
            )
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.info(f"Fetched {len(df)} OHLCV records for {pair}")
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {pair}: {e}")
            raise
            
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        try:
            if self.config.paper_trading:
                result = {
                    'total': self.paper_balance.get('USDT', 0),
                    'free': self.paper_balance.get('USDT', 0),
                    'used': 0
                }
                logger.info(f"Paper trading account balance: {result}")
                return result
            else:
                balance = self.exchange.fetch_balance()
                quote_currency = 'USDT'  # Assuming USDT is the quote currency
                result = {
                    'total': balance['total'].get(quote_currency, 0),
                    'free': balance['free'].get(quote_currency, 0),
                    'used': balance['used'].get(quote_currency, 0)
                }
                logger.info(f"Account balance: {result}")
                return result
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise
    
    def get_position_size(self, pair: str) -> float:
        """Get current position size for a given pair"""
        if self.config.paper_trading:
            return self.paper_positions.get(pair, {}).get('size', 0)
        else:
            # Extract the base currency from the pair (e.g., 'BTC' from 'BTC/USDT')
            base_currency = pair.split('/')[0]
            try:
                balance = self.exchange.fetch_balance()
                return balance['total'].get(base_currency, 0)
            except Exception as e:
                logger.error(f"Error fetching position size for {pair}: {e}")
                return 0
            
    def create_order(self, pair: str, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Create a new order for a specific pair"""
        try:
            order_type = 'market' if price is None else 'limit'
            
            if self.config.paper_trading:
                # Simulate order in paper trading mode
                current_price = self.get_current_price(pair)
                order_price = price if price else current_price
                
                # Generate a random order ID for paper trading
                order_id = f"paper_{int(time.time())}_{hash(pair+side)}"
                
                # Update paper trading balance
                if side == 'buy':
                    cost = amount * order_price
                    if self.paper_balance.get('USDT', 0) >= cost:
                        self.paper_balance['USDT'] = self.paper_balance.get('USDT', 0) - cost
                        
                        # Add to positions
                        if pair not in self.paper_positions:
                            self.paper_positions[pair] = {'size': 0, 'cost': 0}
                        
                        self.paper_positions[pair]['size'] = self.paper_positions[pair]['size'] + amount
                        self.paper_positions[pair]['cost'] = self.paper_positions[pair]['cost'] + cost
                    else:
                        raise Exception(f"Insufficient balance for paper trading: {self.paper_balance.get('USDT', 0)} < {cost}")
                else:  # sell
                    if pair in self.paper_positions and self.paper_positions[pair]['size'] >= amount:
                        revenue = amount * order_price
                        self.paper_balance['USDT'] = self.paper_balance.get('USDT', 0) + revenue
                        
                        # Update positions
                        self.paper_positions[pair]['size'] = self.paper_positions[pair]['size'] - amount
                        
                        # If position is fully closed, calculate P/L
                        if self.paper_positions[pair]['size'] <= 0:
                            avg_cost = self.paper_positions[pair]['cost'] / amount if amount > 0 else 0
                            profit_loss = revenue - self.paper_positions[pair]['cost']
                            logger.info(f"Paper trading closed position for {pair}: P/L = ${profit_loss:.2f}")
                            self.paper_positions[pair] = {'size': 0, 'cost': 0}
                    else:
                        raise Exception(f"Insufficient position size for paper trading: {self.paper_positions.get(pair, {}).get('size', 0)} < {amount}")
                
                # Save paper trading state
                self._save_paper_trading_state()
                
                order = {
                    'id': order_id,
                    'symbol': pair,
                    'type': order_type,
                    'side': side,
                    'amount': amount,
                    'price': order_price,
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.datetime.now().isoformat(),
                    'status': 'closed'  # Paper orders are immediately filled
                }
                
                logger.info(f"Created paper trading {side} order for {pair}: {order_id}")
                return order
            else:
                order = self.exchange.create_order(
                    symbol=pair,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price
                )
                logger.info(f"Created {side} order for {pair}: {order['id']}")
                return order
        except Exception as e:
            logger.error(f"Error creating order for {pair}: {e}")
            raise
    
    def get_current_price(self, pair: str) -> float:
        """Get current price for a pair"""
        try:
            ticker = self.exchange.fetch_ticker(pair)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching current price for {pair}: {e}")
            # Fallback to last OHLCV close price
            ohlcv = self.fetch_ohlcv(pair, limit=1)
            return ohlcv['close'].iloc[0]
            
    def fetch_open_orders(self, pair: str) -> List[Dict]:
        """Fetch all open orders for a specific pair"""
        try:
            if self.config.paper_trading:
                # Paper trading doesn't have open orders (all filled immediately)
                return []
            else:
                open_orders = self.exchange.fetch_open_orders(symbol=pair)
                logger.info(f"Fetched {len(open_orders)} open orders for {pair}")
                return open_orders
        except Exception as e:
            logger.error(f"Error fetching open orders for {pair}: {e}")
            raise

    def fetch_high_volume_pairs(self, top_n: int = 10) -> List[str]:
        """Fetch the top N high-volume trading pairs from Kraken"""
        try:
            markets = self.exchange.fetch_markets()
            # Filter for USDT pairs
            usdt_pairs = [m['symbol'] for m in markets if '/USDT' in m['symbol']]
            
            # Fetch tickers for all USDT pairs to get volume data
            tickers = self.exchange.fetch_tickers(usdt_pairs)
            
            # Sort by 24h volume
            pairs_with_volume = [(pair, tickers[pair]['quoteVolume'] if 'quoteVolume' in tickers[pair] else 0) 
                                for pair in usdt_pairs if pair in tickers]
            sorted_pairs = [pair for pair, volume in sorted(pairs_with_volume, key=lambda x: x[1], reverse=True)]
            
            top_pairs = sorted_pairs[:top_n]
            logger.info(f"Top {top_n} high-volume pairs: {top_pairs}")
            return top_pairs
        except Exception as e:
            logger.error(f"Error fetching high-volume pairs: {e}")
            # Fallback to common pairs if fetching fails
            fallback_pairs = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT"]
            logger.info(f"Using fallback pairs: {fallback_pairs}")
            return fallback_pairs

# Technical indicators
class Indicators:
    """Technical indicators for strategy implementation"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators to the dataframe"""
        df = Indicators.add_moving_averages(df)
        df = Indicators.add_rsi(df)
        df = Indicators.add_bollinger_bands(df)
        df = Indicators.add_macd(df)
        df = Indicators.add_atr(df)
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages to the dataframe"""
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add RSI indicator to the dataframe"""
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Handle division by zero
        avg_loss = avg_loss.replace(0, np.nan)
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values
        df['rsi'] = df['rsi'].fillna(50)  # Neutral RSI for NaN values
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands to the dataframe"""
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        df['bb_std'] = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator to the dataframe"""
        df['macd_line'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_line'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Average True Range indicator to the dataframe"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=window).mean()
        
        # Fill NaN values
        df['atr'] = df['atr'].fillna(method='bfill')
        return df

# Trading strategy
class Strategy:
    """Trading strategy implementations"""
    
    @staticmethod
    def combined_strategy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Combined strategy using multiple indicators
        Returns buy signals, sell signals, and stop signals
        """
        buy_signals = pd.Series(index=df.index, data=False)
        sell_signals = pd.Series(index=df.index, data=False)
        stop_signals = pd.Series(index=df.index, data=False)
        
        # Ensure all required indicators are present
        required_columns = ['sma_20', 'sma_50', 'rsi', 'bb_lower', 'bb_upper', 
                          'macd_line', 'macd_signal', 'atr', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column {col} for strategy calculation")
                return buy_signals, sell_signals, stop_signals
        
        # Ensure all indicator data is valid (no NaN values)
        df_valid = df.dropna(subset=required_columns)
        if len(df_valid) < 20:  # Require at least 20 valid data points
            logger.warning(f"Insufficient valid data points for strategy: {len(df_valid)}")
            return buy_signals, sell_signals, stop_signals
        
        df['ma_trend'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ma_crossover_buy'] = ((df['ma_trend'] == 1) & (df['ma_trend'].shift(1) == 0)).astype(int)
        df['ma_crossover_sell'] = ((df['ma_trend'] == 0) & (df['ma_trend'].shift(1) == 1)).astype(int)
        
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        df['bb_buy'] = (df['close'] < df['bb_lower']).astype(int)
        df['bb_sell'] = (df['close'] > df['bb_upper']).astype(int)
        
        df['macd_buy'] = ((df['macd_line'] > df['macd_signal']) & 
                          (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_sell'] = ((df['macd_line'] < df['macd_signal']) & 
                           (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        df['buy_score'] = (df['ma_crossover_buy'] * 0.3 + 
                           df['rsi_oversold'] * 0.3 + 
                           df['bb_buy'] * 0.2 + 
                           df['macd_buy'] * 0.2)
        
        df['sell_score'] = (df['ma_crossover_sell'] * 0.3 + 
                            df['rsi_overbought'] * 0.3 + 
                            df['bb_sell'] * 0.2 + 
                            df['macd_sell'] * 0.2)
        
        buy_signals = df['buy_score'] >= 0.5
        sell_signals = df['sell_score'] >= 0.5
        
        df['stop_level'] = np.where(buy_signals, df['close'] - 2 * df['atr'], 
                                   np.where(sell_signals, df['close'] + 2 * df['atr'], np.nan))
        df['stop_level'] = df['stop_level'].ffill()
        
        stop_signals = ((df['close'] < df['stop_level']) & (df['stop_level'].shift(1).notna()))
        
        return buy_signals, sell_signals, stop_signals

# Risk management
class RiskManager:
    """Handles position sizing and risk management"""
    
    def __init__(self, config: TradeConfig, exchange: Exchange):
        self.config = config
        self.exchange = exchange
        
    def calculate_position_size(self, pair: str, current_price: float, stop_price: float = None) -> float:
        """
        Calculate position size based on risk parameters for a specific pair
        """
        try:
            balance = self.exchange.get_balance()
            account_value = balance['total']
            max_position_value = account_value * self.config.max_position_size
            
            if stop_price:
                risk_per_trade = account_value * self.config.risk_per_trade_pct  # Risk per trade
                price_diff_pct = abs(current_price - stop_price) / current_price
                if price_diff_pct > 0:
                    position_value = risk_per_trade / price_diff_pct
                    position_value = min(position_value, max_position_value)
                else:
                    position_value = max_position_value
            else:
                position_value = max_position_value
            
            position_size = position_value / current_price
            
            # Round position size to appropriate precision for the asset
            if current_price < 0.1:
                position_size = round(position_size, 0)  # Integer for very low-priced assets
            elif current_price < 10:
                position_size = round(position_size, 2)
            elif current_price < 1000:
                position_size = round(position_size, 4)
            else:
                position_size = round(position_size, 6)
            
            logger.info(f"Calculated position size for {pair}: {position_size} units (${position_value:.2f})")
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size for {pair}: {e}")
            return 0.0

# Performance tracking
class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self, data_dir: str = "performance_data"):
        self.data_dir = data_dir
        self.trades_file = os.path.join(data_dir, "trades.json")
        self.performance_file = os.path.join(data_dir, "performance.json")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        if not os.path.exists(self.trades_file):
            self.save_trades([])
            
        if not os.path.exists(self.performance_file):
            self.save_performance({
                "starting_balance": 0,
                "current_balance": 0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_profit_loss": 0,
                "max_drawdown": 0,
                "daily_returns": []
            })
    
    def load_trades(self) -> List[Dict]:
        """Load trades from file"""
        with open(self.trades_file, 'r') as f:
            return json.load(f)
    
    def save_trades(self, trades: List[Dict]) -> None:
        """Save trades to file"""
        with open(self.trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def load_performance(self) -> Dict:
        """Load performance data from file"""
        with open(self.performance_file, 'r') as f:
            return json.load(f)
    
    def save_performance(self, performance: Dict) -> None:
        """Save performance data to file"""
        with open(self.performance_file, 'w') as f:
            json.dump(performance, f, indent=2)
    
    def add_trade(self, trade: Dict) -> None:
        """Add a new trade to the records"""
        trades = self.load_trades()
        trades.append(trade)
        self.save_trades(trades)
        self.update_performance_metrics()
        logger.info(f"Added trade: {trade['id']} to records")
    
    def update_performance_metrics(self) -> None:
        """Update performance metrics based on trades"""
        trades = self.load_trades()
        performance = self.load_performance()
        
        if not trades:
            return
        
        performance["total_trades"] = len(trades)
        
        winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
        losing_trades = [t for t in trades if t.get("profit_loss", 0) <= 0]
        
        performance["winning_trades"] = len(winning_trades)
        performance["losing_trades"] = len(losing_trades)
        
        if performance["total_trades"] > 0:
            performance["win_rate"] = performance["winning_trades"] / performance["total_trades"]
        
        total_profits = sum(t.get("profit_loss", 0) for t in winning_trades)
        total_losses = abs(sum(t.get("profit_loss", 0) for t in losing_trades))
        
        performance["total_profit_loss"] = total_profits - total_losses
        
        if total_losses > 0:
            performance["profit_factor"] = total_profits / total_losses
        
        daily_trades = {}
        for trade in trades:
            date = trade.get("close_time", "").split("T")[0]
            if date:
                if date not in daily_trades:
                    daily_trades[date] = 0
                daily_trades[date] += trade.get("profit_loss", 0)
        
        performance["daily_returns"] = [{"date": k, "return": v} for k, v in daily_trades.items()]
        
        cumulative_returns = []
        balance = performance.get("starting_balance", 0)
        for date, ret in sorted(daily_trades.items()):
            balance += ret
            cumulative_returns.append({"date": date, "balance": balance})
        
        if cumulative_returns:
            peak = 0
            max_drawdown = 0
            for cr in cumulative_returns:
                if cr["balance"] > peak:
                    peak = cr["balance"]
                drawdown = (peak - cr["balance"]) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            performance["max_drawdown"] = max_drawdown
            performance["current_balance"] = cumulative_returns[-1]["balance"]
        
        self.save_performance(performance)
        logger.info(f"Updated performance metrics: Win rate {performance['win_rate']:.2f}, P/L ${performance['total_profit_loss']:.2f}")
    
    def plot_performance(self, save_path: str = None) -> None:
        """Plot performance metrics"""
        try:
            performance = self.load_performance()
            trades = self.load_trades()
            
            # Create a figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot 1: Equity Curve
            dates = [t["date"] for t in performance["daily_returns"]]
            balances = []
            cumulative_return = performance["starting_balance"]
            for ret in [t["return"] for t in performance["daily_returns"]]:
                cumulative_return += ret
                balances.append(cumulative_return)
            
            if dates and balances:
                axs[0].plot(dates, balances)
                axs[0].set_title('Equity Curve')
                axs[0].set_ylabel('Account Value')
                axs[0].grid(True)
                axs[0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Trade P/L
            trade_ids = [f"{t['pair']}-{t['id'][-6:]}" for t in trades]
            trade_pls = [t["profit_loss"] for t in trades]
            
            if trade_ids and trade_pls:
                colors = ['green' if pl > 0 else 'red' for pl in trade_pls]
                axs[1].bar(trade_ids, trade_pls, color=colors)
                axs[1].set_title('Trade P/L')
                axs[1].set_ylabel('Profit/Loss')
                axs[1].grid(True)
                axs[1].tick_params(axis='x', rotation=90)
            
            # Plot 3: Performance Metrics
            metrics = {
                'Win Rate': performance["win_rate"],
                'Profit Factor': performance.get("profit_factor", 0),
                'Max Drawdown': performance["max_drawdown"]
            }
            
            axs[2].bar(metrics.keys(), metrics.values())
            axs[2].set_title('Performance Metrics')
            axs[2].set_ylim(0, 1.2)
            axs[2].grid(True)
            
            # Add values on top of bars
            for i, v in enumerate(metrics.values()):
                axs[2].text(i, v + 0.05, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Performance chart saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting performance: {e}")

# Main trading bot
class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, config: TradeConfig):
        self.config = config
        self.exchange = Exchange(config)
        self.risk_manager = RiskManager(config, self.exchange)
        self.performance_tracker = PerformanceTracker()
        
        self.current_position = None
        self.current_pair = None
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # Determine trading pairs
        if self.config.trading_pairs is None:
            self.trading_pairs = self.exchange.fetch_high_volume_pairs(self.config.high_volume_top_n)
        else:
            self.trading_pairs = self.config.trading_pairs
        
        logger.info(f"Initialized trading bot for pairs: {self.trading_pairs}")
    
    def run(self, backtest_mode: bool = False) -> None:
        """Run the trading bot in live or backtest mode"""
        if backtest_mode:
            self.run_backtest()
        else:
            self.run_live()
    
    def run_live(self) -> None:
        """Run the trading bot in live mode"""
        logger.info("Starting trading bot in live mode")
        
        try:
            balance = self.exchange.get_balance()
            performance = self.performance_tracker.load_performance()
            
            if performance["starting_balance"] == 0:
                performance["starting_balance"] = balance["total"]
                performance["current_balance"] = balance["total"]
                self.performance_tracker.save_performance(performance)
            
            while True:
                # Fetch data and generate signals for all pairs
                pair_signals = {}
                for pair in self.trading_pairs:
                    try:
                        ohlcv_data = self.exchange.fetch_ohlcv(pair, limit=200)
                        if len(ohlcv_data) >= 50:  # Ensure we have enough data points
                            df = Indicators.add_all_indicators(ohlcv_data)
                            buy_signals, sell_signals, stop_signals = Strategy.combined_strategy(df)
                            pair_signals[pair] = {
                                'buy': buy_signals.iloc[-1],
                                'sell': sell_signals.iloc[-1],
                                'stop': stop_signals.iloc[-1],
                                'close': df['close'].iloc[-1]
                            }
                            logger.info(f"Generated signals for {pair}: buy={buy_signals.iloc[-1]}, sell={sell_signals.iloc[-1]}")
                        else:
                            logger.warning(f"Insufficient data for {pair}: {len(ohlcv_data)} points")
                    except Exception as e:
                        logger.error(f"Error processing pair {pair}: {e}")
                
                # Determine action based on signals
                if self.current_position:
                    if self.current_pair in pair_signals:
                        pair_data = pair_signals[self.current_pair]
                        current_price = pair_data['close']
                        
                        # Check for exit conditions
                        if (self.current_position == 'long' and (current_price <= self.stop_loss_price or pair_data['sell'] or pair_data['stop'])) or \
                           (self.current_position == 'short' and (current_price >= self.stop_loss_price or pair_data['buy'] or pair_data['stop'])):
                            self.exit_position(self.current_pair, current_price, 'signal')
                else:
                    # Find the pair with the strongest buy or sell signal
                    strongest_buy = None
                    strongest_sell = None
                    
                    for pair, data in pair_signals.items():
                        if data['buy'] and (strongest_buy is None or data['close'] > pair_signals[strongest_buy]['close']):
                            strongest_buy = pair
                        if data['sell'] and (strongest_sell is None or data['close'] < pair_signals[strongest_sell]['close']):
                            strongest_sell = pair
                    
                    if strongest_buy:
                        self.enter_position(strongest_buy, 'long', pair_signals[strongest_buy]['close'])
                    elif strongest_sell and not self.config.paper_trading:  # Only allow shorts in real trading
                        self.enter_position(strongest_sell, 'short', pair_signals[strongest_sell]['close'])
                
                # Sleep for the specified timeframe
                timeframe_seconds = {
                    '1m': 60,
                    '5m': 300,
                    '15m': 900,
                    '30m': 1800,
                    '1h': 3600,
                    '4h': 14400,
                    '1d': 86400
                }.get(self.config.timeframe, 3600)  # Default to 1 hour
                
                # Sleep for 1/4 of the timeframe period
                sleep_time = max(60, timeframe_seconds // 4)  # Min 60 seconds
                logger.info(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
            raise
    
    def run_backtest(self) -> None:
        """Run a backtest simulation"""
        logger.info(f"Starting backtest for the last {self.config.backtest_days} days")
        
        backtest_results = {}
        overall_trades = []
        
        for pair in self.trading_pairs:
            try:
                # Fetch enough data for the backtest
                days_of_data = self.config.backtest_days + 30  # Extra data for indicators
                bars_needed = days_of_data * 24  # Assuming 1-hour bars
                
                ohlcv_data = self.exchange.fetch_ohlcv(pair, limit=min(1000, bars_needed))
                
                if len(ohlcv_data) < 100:
                    logger.warning(f"Insufficient historical data for {pair}, skipping")
                    continue
                
                # Calculate indicators
                df = Indicators.add_all_indicators(ohlcv_data)
                
                # Generate signals
                buy_signals, sell_signals, stop_signals = Strategy.combined_strategy(df)
                
                # Initialize backtest variables
                position = None
                entry_price = 0
                entry_time = None
                stop_loss = 0
                take_profit = 0
                trades = []
                
                # Simulate trading
                for i in range(50, len(df)):  # Skip the first 50 rows for indicator warmup
                    date = df.index[i]
                    current_price = df['close'].iloc[i]
                    
                    # Check for exit if in position
                    if position:
                        if (position == 'long' and (current_price <= stop_loss or sell_signals.iloc[i])) or \
                           (position == 'short' and (current_price >= stop_loss or buy_signals.iloc[i])):
                            
                            # Calculate P/L
                            if position == 'long':
                                profit_loss = (current_price - entry_price) / entry_price
                            else:  # short
                                profit_loss = (entry_price - current_price) / entry_price
                            
                            # Record trade
                            trade = {
                                "id": f"backtest_{len(trades)}",
                                "pair": pair,
                                "side": position,
                                "entry_price": entry_price,
                                "exit_price": current_price,
                                "position_size": 1.0,  # Normalized to 1.0 for backtest
                                "entry_time": entry_time.isoformat(),
                                "close_time": date.isoformat(),
                                "profit_loss": profit_loss,
                                "exit_reason": "stop_loss" if (position == 'long' and current_price <= stop_loss) or 
                                                            (position == 'short' and current_price >= stop_loss) 
                                                    else "signal"
                            }
                            trades.append(trade)
                            overall_trades.append(trade)
                            
                            # Reset position
                            position = None
                            entry_price = 0
                            entry_time = None
                            
                    # Check for entry
                    elif not position:
                        if buy_signals.iloc[i]:
                            position = 'long'
                            entry_price = current_price
                            entry_time = date
                            stop_loss = current_price * (1 - self.config.stop_loss_pct)
                            take_profit = current_price * (1 + self.config.take_profit_pct)
                        elif sell_signals.iloc[i]:
                            position = 'short'
                            entry_price = current_price
                            entry_time = date
                            stop_loss = current_price * (1 + self.config.stop_loss_pct)
                            take_profit = current_price * (1 - self.config.take_profit_pct)
                
                # Calculate backtest statistics
                if trades:
                    winning_trades = [t for t in trades if t["profit_loss"] > 0]
                    win_rate = len(winning_trades) / len(trades) if trades else 0
                    avg_profit = sum(t["profit_loss"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
                    
                    losing_trades = [t for t in trades if t["profit_loss"] <= 0]
                    avg_loss = sum(t["profit_loss"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
                    
                    # Calculate equity curve
                    equity = 1000  # Start with $1000
                    equity_curve = [equity]
                    for trade in trades:
                        equity *= (1 + trade["profit_loss"])
                        equity_curve.append(equity)
                    
                    # Calculate max drawdown
                    peak = equity_curve[0]
                    max_drawdown = 0
                    for value in equity_curve:
                        if value > peak:
                            peak = value
                        drawdown = (peak - value) / peak if peak > 0 else 0
                        max_drawdown = max(max_drawdown, drawdown)
                    
                    backtest_results[pair] = {
                        "total_trades": len(trades),
                        "win_rate": win_rate,
                        "avg_profit": avg_profit,
                        "avg_loss": avg_loss,
                        "final_equity": equity_curve[-1],
                        "return_pct": (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100,
                        "max_drawdown": max_drawdown,
                        "profit_factor": abs(sum(t["profit_loss"] for t in winning_trades) / 
                                           sum(t["profit_loss"] for t in losing_trades)) if sum(t["profit_loss"] for t in losing_trades) != 0 else float('inf')
                    }
                    
                    logger.info(f"Backtest results for {pair}: Win rate: {win_rate:.2f}, Return: {backtest_results[pair]['return_pct']:.2f}%")
                else:
                    logger.info(f"No trades generated for {pair} during backtest period")
            
            except Exception as e:
                logger.error(f"Error during backtest for {pair}: {e}")
        
        # Aggregate results
        if overall_trades:
            # Save all backtest trades
            with open("backtest_trades.json", "w") as f:
                json.dump(overall_trades, f, indent=2)
            
            # Calculate overall statistics
            winning_trades = [t for t in overall_trades if t["profit_loss"] > 0]
            win_rate = len(winning_trades) / len(overall_trades)
            avg_profit = sum(t["profit_loss"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            
            losing_trades = [t for t in overall_trades if t["profit_loss"] <= 0]
            avg_loss = sum(t["profit_loss"] for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            logger.info(f"Overall backtest results: {len(overall_trades)} trades, Win rate: {win_rate:.2f}")
            logger.info(f"Average profit: {avg_profit:.4f}, Average loss: {avg_loss:.4f}")
            
            # Plot results
            self.plot_backtest_results(backtest_results, overall_trades)
        else:
            logger.warning("No trades were generated during the backtest period")
    
    def plot_backtest_results(self, results: Dict, trades: List[Dict]) -> None:
        """Plot backtest results"""
        try:
            # Create a figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot 1: Equity curves for all pairs
            for pair, data in results.items():
                pair_trades = [t for t in trades if t["pair"] == pair]
                equity = 1000
                equity_curve = [equity]
                dates = [datetime.datetime.fromisoformat(pair_trades[0]["entry_time"])]
                
                for trade in pair_trades:
                    equity *= (1 + trade["profit_loss"])
                    equity_curve.append(equity)
                    dates.append(datetime.datetime.fromisoformat(trade["close_time"]))
                
                axs[0].plot(dates, equity_curve, label=pair)
            
            axs[0].set_title('Equity Curves')
            axs[0].set_ylabel('Account Value ($)')
            axs[0].grid(True)
            axs[0].legend()
            
            # Plot 2: Return percentages by pair
            pairs = list(results.keys())
            returns = [results[pair]["return_pct"] for pair in pairs]
            
            colors = ['green' if r > 0 else 'red' for r in returns]
            axs[1].bar(pairs, returns, color=colors)
            axs[1].set_title('Return Percentage by Pair')
            axs[1].set_ylabel('Return (%)')
            axs[1].grid(True)
            
            # Add values on top of bars
            for i, v in enumerate(returns):
                axs[1].text(i, v + 1, f"{v:.1f}%", ha='center')
            
            # Plot 3: Key metrics by pair
            metrics = {
                'Win Rate': [results[pair]["win_rate"] for pair in pairs],
                'Profit Factor': [min(results[pair]["profit_factor"], 5) for pair in pairs],  # Cap at 5 for visualization
                'Max Drawdown': [results[pair]["max_drawdown"] for pair in pairs]
            }
            
            bar_width = 0.25
            x = np.arange(len(pairs))
            
            for i, (metric, values) in enumerate(metrics.items()):
                axs[2].bar(x + i * bar_width, values, width=bar_width, label=metric)
            
            axs[2].set_title('Performance Metrics by Pair')
            axs[2].set_xticks(x + bar_width)
            axs[2].set_xticklabels(pairs)
            axs[2].legend()
            axs[2].grid(True)
            
            plt.tight_layout()
            plt.savefig("backtest_results.png")
            logger.info("Backtest results chart saved to backtest_results.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")
    
    def enter_position(self, pair: str, side: str, price: float) -> None:
        """Enter a trading position for a specific pair"""
        try:
            stop_price = price * (1 - self.config.stop_loss_pct) if side == 'long' else price * (1 + self.config.stop_loss_pct)
            take_profit_price = price * (1 + self.config.take_profit_pct) if side == 'long' else price * (1 - self.config.take_profit_pct)
            position_size = self.risk_manager.calculate_position_size(pair, price, stop_price)
            
            if position_size <= 0:
                logger.warning(f"Cannot enter position: calculated position size is {position_size}")
                return
            
            order_side = 'buy' if side == 'long' else 'sell'
            order = self.exchange.create_order(pair, order_side, position_size)
            
            self.current_position = side
            self.current_pair = pair
            self.stop_loss_price = stop_price
            self.take_profit_price = take_profit_price
            
            logger.info(f"Entered {side} position for {pair} at {price}: size={position_size}, stop={stop_price}, target={take_profit_price}")
        except Exception as e:
            logger.error(f"Error entering position for {pair}: {e}")
    
    def exit_position(self, pair: str, price: float, reason: str) -> None:
        """Exit a trading position for a specific pair"""
        try:
            if not self.current_position:
                logger.warning("No position to exit")
                return
            
            order_side = 'sell' if self.current_position == 'long' else 'buy'
            
            # Get actual position size
            position_size = self.exchange.get_position_size(pair)
            
            if position_size <= 0:
                logger.warning(f"Cannot exit position: current position size is {position_size}")
                self.current_position = None
                self.current_pair = None
                return
            
            order = self.exchange.create_order(pair, order_side, position_size)
            
            # Estimate entry price based on stop loss and take profit
            estimated_entry_price = None
            if self.current_position == 'long':
                estimated_entry_price = self.stop_loss_price / (1 - self.config.stop_loss_pct)
            else:  # short
                estimated_entry_price = self.stop_loss_price / (1 + self.config.stop_loss_pct)
            
            # Record the trade
            trade = {
                "id": order['id'],
                "pair": pair,
                "side": self.current_position,
                "entry_price": estimated_entry_price,
                "exit_price": price,
                "position_size": position_size,
                "entry_time": (datetime.datetime.now() - datetime.timedelta(hours=24)).isoformat(),  # Estimated
                "close_time": datetime.datetime.now().isoformat(),
                "profit_loss": 0,  # To be calculated
                "exit_reason": reason
            }
            
            if self.current_position == 'long':
                profit_loss = (price - trade["entry_price"]) * position_size
            else:
                profit_loss = (trade["entry_price"] - price) * position_size
            
            trade["profit_loss"] = profit_loss
            self.performance_tracker.add_trade(trade)
            
            self.current_position = None
            self.current_pair = None
            self.stop_loss_price = None
            self.take_profit_price = None
            
            logger.info(f"Exited position for {pair} at {price}: reason={reason}, profit/loss=${profit_loss:.2f}")
        except Exception as e:
            logger.error(f"Error exiting position for {pair}: {e}")

def load_config_from_env():
    """Load configuration from environment variables or .env file"""
    api_key = os.getenv('KRAKEN_API_KEY', '')
    api_secret = os.getenv('KRAKEN_API_SECRET', '')
    
    # Parse trading pairs from environment if provided
    trading_pairs_str = os.getenv('TRADING_PAIRS', '')
    trading_pairs = trading_pairs_str.split(',') if trading_pairs_str else None
    
    # Create configuration
    config = TradeConfig(
        exchange_id="kraken",
        api_key=api_key,
        api_secret=api_secret,
        trading_pairs=trading_pairs,
        timeframe=os.getenv('TIMEFRAME', '1h'),
        strategy=os.getenv('STRATEGY', 'combined'),
        max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.02')),
        stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', '0.03')),
        take_profit_pct=float(os.getenv('TAKE_PROFIT_PCT', '0.05')),
        leverage=float(os.getenv('LEVERAGE', '1.0')),
        backtest_days=int(os.getenv('BACKTEST_DAYS', '30')),
        high_volume_top_n=int(os.getenv('HIGH_VOLUME_TOP_N', '10')),
        paper_trading=os.getenv('PAPER_TRADING', 'True').lower() in ('true', 'yes', '1'),
        risk_per_trade_pct=float(os.getenv('RISK_PER_TRADE_PCT', '0.01'))
    )
    
    return config

# Example usage
if __name__ == "__main__":
    # Load configuration from environment variables
    config = load_config_from_env()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    args = parser.parse_args()
    
    # Override paper trading setting if specified
    if args.paper:
        config.paper_trading = True
    elif args.live:
        config.paper_trading = False
    
    # Create and run the bot
    bot = TradingBot(config)
    bot.run(backtest_mode=args.backtest)