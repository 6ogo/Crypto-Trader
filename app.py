from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import os
import json
import uuid
import datetime
import joblib
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crypto_trading.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the predictor after initializing app and logger
from train_crypto_model import CryptoPredictor
predictor = CryptoPredictor()

# Database models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    api_key = db.Column(db.String(200))
    api_secret = db.Column(db.String(200))
    trades = db.relationship('Trade', backref='user', lazy=True)
    positions = db.relationship('Position', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    order_type = db.Column(db.String(10), nullable=False)  # buy, sell
    amount = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    order_id = db.Column(db.String(50))
    status = db.Column(db.String(20), default='pending')  # pending, completed, failed
    realized_pnl = db.Column(db.Float, nullable=True)  # Realized P&L from this trade
    
class Position(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    amount = db.Column(db.Float, default=0.0)  # Current position size (positive for long, negative for short)
    avg_entry_price = db.Column(db.Float, nullable=True)  # Average entry price
    realized_pnl = db.Column(db.Float, default=0.0)  # Realized P&L for this symbol
    last_updated = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    def calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L at current price"""
        if self.amount == 0 or not self.avg_entry_price:
            return 0.0
            
        if self.amount > 0:  # Long position
            return self.amount * (current_price - self.avg_entry_price)
        else:  # Short position
            return -self.amount * (self.avg_entry_price - current_price)
    
    def update_position(self, trade_type, amount, price):
        """Update position based on a new trade"""
        # Record current time
        self.last_updated = datetime.datetime.utcnow()
        
        # Calculate position value before trade
        old_position_value = self.amount * self.avg_entry_price if self.avg_entry_price else 0
        trade_realized_pnl = 0.0
        
        if trade_type == 'buy':
            # Calculate new average entry price for long positions
            if self.amount < 0:  # Currently short
                # First, calculate P&L for the amount we're covering
                cover_amount = min(amount, abs(self.amount))
                if cover_amount > 0:
                    # Realize P&L on the covered portion
                    cover_pnl = cover_amount * (self.avg_entry_price - price)
                    self.realized_pnl += cover_pnl
                    trade_realized_pnl += cover_pnl
                    
                    # Update position size
                    self.amount += cover_amount
                    
                    # If we're buying more than our short position
                    extra_amount = amount - cover_amount
                    if extra_amount > 0 and self.amount == 0:
                        # Reset average price for the new long position
                        self.avg_entry_price = price
                        self.amount = extra_amount
                    elif extra_amount > 0:
                        # Add to existing long position
                        new_total = self.amount + extra_amount
                        self.avg_entry_price = (old_position_value + (extra_amount * price)) / new_total
                        self.amount = new_total
                else:
                    # Just reduce short position
                    self.amount += amount
            else:
                # Adding to long position or creating new long
                new_total = self.amount + amount
                new_value = (old_position_value + (amount * price))
                self.avg_entry_price = new_value / new_total if new_total > 0 else None
                self.amount = new_total
                
        elif trade_type == 'sell':
            # Calculate new average entry price for short positions
            if self.amount > 0:  # Currently long
                # First, calculate P&L for the amount we're selling
                sell_amount = min(amount, self.amount)
                if sell_amount > 0:
                    # Realize P&L on the sold portion
                    sell_pnl = sell_amount * (price - self.avg_entry_price)
                    self.realized_pnl += sell_pnl
                    trade_realized_pnl += sell_pnl
                    
                    # Update position size
                    self.amount -= sell_amount
                    
                    # If we're selling more than our long position
                    extra_amount = amount - sell_amount
                    if extra_amount > 0 and self.amount == 0:
                        # Reset average price for the new short position
                        self.avg_entry_price = price
                        self.amount = -extra_amount
                    elif extra_amount > 0:
                        # Add to existing short position
                        new_total = self.amount - extra_amount
                        self.avg_entry_price = (old_position_value + (extra_amount * price)) / abs(new_total)
                        self.amount = new_total
                else:
                    # Just reduce long position
                    self.amount -= amount
            else:
                # Adding to short position or creating new short
                new_total = self.amount - amount
                new_value = (old_position_value + (amount * price))
                self.avg_entry_price = new_value / abs(new_total) if new_total < 0 else None
                self.amount = new_total
        
        # If position is closed completely, reset avg entry price
        if self.amount == 0:
            self.avg_entry_price = None
            
        return trade_realized_pnl

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    prediction = db.Column(db.Boolean, nullable=False)  # True for up, False for down
    probability = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    horizon_hours = db.Column(db.Integer, default=12)
    accuracy = db.Column(db.Float)  # To be filled when outcome is known
    outcome = db.Column(db.Boolean)  # Actual outcome when available

# Now you can import and initialize SignalCombiner after all models are defined
from combined_signals import SignalCombiner
signal_combiner = SignalCombiner(db, Prediction, Trade, User, logger)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Basic validation
        if not username or not email or not password:
            flash('All fields are required')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            flash('An error occurred during registration')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/external_data')
@login_required
def external_data():
    """Page to explore external data influences"""
    # Load combined data for BTC
    external_file = os.path.join('data', 'BTCUSD_with_external_data.csv')
    
    if not os.path.exists(external_file):
        flash('External data not yet available. Please run data collection first.')
        return redirect(url_for('dashboard'))
    
    df = pd.read_csv(external_file, index_col=0, parse_dates=True)
    
    # Calculate correlations
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    external_cols = [col for col in df.columns if col not in price_cols]
    
    # Get correlations with close price
    correlations = df[[*price_cols, *external_cols]].corr()['close'].sort_values(ascending=False)
    
    # Get recent external data
    recent_data = df.tail(30)[external_cols]
    
    return render_template('external_data.html',
                          recent_data=recent_data.to_dict('records'),
                          correlations=correlations.to_dict(),
                          dates=recent_data.index.strftime('%Y-%m-%d').tolist())

@app.route('/dashboard')
@login_required
def dashboard():
    # Get latest predictions
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(25).all()
    
    # Get user's trades
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.timestamp.desc()).limit(10).all()
    
    # Get current market prices
    prices = get_current_prices()
    
    # Get user's positions with P&L calculations
    positions = get_user_positions_with_pnl(current_user.id, prices)
    
    # Calculate total portfolio value
    portfolio_value = calculate_portfolio_value(positions, prices)
    
    # Get combined signals
    combined_signals = signal_combiner.generate_market_signals()
    
    return render_template('dashboard.html', 
                          predictions=predictions, 
                          trades=trades, 
                          prices=prices,
                          positions=positions,
                          portfolio_value=portfolio_value,
                          combined_signals=combined_signals,
                          user=current_user)

@app.route('/api_settings', methods=['GET', 'POST'])
@login_required
def api_settings():
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        api_secret = request.form.get('api_secret')
        
        # Validate API key and secret with Kraken
        if validate_api_credentials(api_key, api_secret):
            current_user.api_key = api_key
            current_user.api_secret = api_secret
            db.session.commit()
            flash('API credentials updated successfully')
        else:
            flash('Invalid API credentials')
        
        return redirect(url_for('api_settings'))
    
    return render_template('api_settings.html')

@app.route('/place_trade', methods=['POST'])
@login_required
def place_trade():
    if not current_user.api_key or not current_user.api_secret:
        return jsonify({'success': False, 'message': 'API credentials not set'})
    
    symbol = request.form.get('symbol')
    order_type = request.form.get('order_type')
    amount = float(request.form.get('amount'))
    
    if not symbol or not order_type or not amount:
        return jsonify({'success': False, 'message': 'Missing required fields'})
    
    # Get current price
    prices = get_current_prices()
    if symbol not in prices:
        return jsonify({'success': False, 'message': f'Price for {symbol} not available'})
    
    price = prices[symbol]
    
    # Place order on Kraken
    order_id = place_kraken_order(current_user.api_key, current_user.api_secret, 
                                 symbol, order_type, amount)
    
    if not order_id:
        return jsonify({'success': False, 'message': 'Order placement failed'})
    
    # Update position and calculate realized P&L
    position, realized_pnl = update_user_position(current_user.id, symbol, order_type, amount, price)
    
    # Record trade in database
    new_trade = Trade(
        user_id=current_user.id,
        symbol=symbol,
        order_type=order_type,
        amount=amount,
        price=price,
        order_id=order_id,
        status='completed',
        realized_pnl=realized_pnl
    )
    
    db.session.add(new_trade)
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'message': f'{order_type.capitalize()} order placed successfully',
        'trade_id': new_trade.id,
        'realized_pnl': realized_pnl
    })

@app.route('/predictions')
@login_required
def predictions():
    # Get all predictions for analysis
    all_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    
    # Calculate accuracy statistics
    prediction_stats = calculate_prediction_statistics()
    
    return render_template('predictions.html', 
                          predictions=all_predictions,
                          stats=prediction_stats)

@app.route('/trades')
@login_required
def trades():
    # Get all user trades
    all_trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.timestamp.desc()).all()
    
    # Calculate trade statistics
    trade_stats = calculate_trade_statistics(current_user.id)
    
    return render_template('trades.html', 
                          trades=all_trades,
                          stats=trade_stats)

@app.route('/positions')
@login_required
def positions():
    # Get current market prices
    prices = get_current_prices()
    
    # Get user's positions with P&L calculations
    user_positions = get_user_positions_with_pnl(current_user.id, prices)
    
    # Calculate total P&L and other statistics
    total_realized_pnl = sum(p.realized_pnl for p in user_positions)
    total_unrealized_pnl = sum(p['unrealized_pnl'] for p in user_positions if p['unrealized_pnl'])
    total_pnl = total_realized_pnl + total_unrealized_pnl
    
    return render_template('positions.html',
                          positions=user_positions,
                          prices=prices,
                          total_realized_pnl=total_realized_pnl,
                          total_unrealized_pnl=total_unrealized_pnl,
                          total_pnl=total_pnl)

@app.route('/api/predictions')
@login_required
def api_predictions():
    # Get latest predictions for API
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    
    predictions_data = [{
        'symbol': p.symbol,
        'prediction': 'UP' if p.prediction else 'DOWN',
        'probability': p.probability,
        'timestamp': p.timestamp.isoformat(),
        'horizon_hours': p.horizon_hours
    } for p in predictions]
    
    return jsonify(predictions_data)

@app.route('/api/combined_signals')
@login_required
def api_combined_signals():
    """API endpoint for combined market signals"""
    signals = signal_combiner.generate_market_signals()
    return jsonify(signals)

@app.route('/api/positions')
@login_required
def api_positions():
    # Get current prices
    prices = get_current_prices()
    
    # Get positions with P&L
    positions = get_user_positions_with_pnl(current_user.id, prices)
    
    # Format for API
    positions_data = [{
        'symbol': p.symbol,
        'amount': p.amount,
        'avg_entry_price': p.avg_entry_price,
        'current_price': prices.get(p.symbol, 0),
        'realized_pnl': p.realized_pnl,
        'unrealized_pnl': p['unrealized_pnl'],
        'total_pnl': p.realized_pnl + p['unrealized_pnl'],
        'last_updated': p.last_updated.isoformat()
    } for p in positions]
    
    return jsonify(positions_data)

@app.route('/api/market_data')
@login_required
def api_market_data():
    # Get current market data for API
    prices = get_current_prices()
    
    # Get recent price history
    history = get_price_history()
    
    return jsonify({
        'current_prices': prices,
        'price_history': history
    })

# Helper functions
def validate_api_credentials(api_key, api_secret):
    """Validate Kraken API credentials"""
    try:
        kraken = krakenex.API(key=api_key, secret=api_secret)
        response = kraken.query_private('Balance')
        return 'error' not in response
    except Exception as e:
        logger.error(f"API validation error: {str(e)}")
        return False

def place_kraken_order(api_key, api_secret, symbol, order_type, amount):
    """Place an order on Kraken"""
    try:
        kraken = krakenex.API(key=api_key, secret=api_secret)
        
        # Convert symbol format (BTCUSD -> XBTUSD)
        kraken_symbol = symbol
        if symbol.startswith('BTC'):
            kraken_symbol = 'XBT' + symbol[3:]
        
        # Prepare order parameters
        params = {
            'pair': kraken_symbol,
            'type': order_type.lower(),
            'ordertype': 'market',
            'volume': str(amount)
        }
        
        # Place order
        response = kraken.query_private('AddOrder', params)
        
        if 'error' in response and response['error']:
            logger.error(f"Kraken order error: {response['error']}")
            return None
        
        # Return order ID
        return response['result']['txid'][0]
    except Exception as e:
        logger.error(f"Order placement error: {str(e)}")
        return None

def update_user_position(user_id, symbol, order_type, amount, price):
    """Update user position after a trade"""
    # Find or create position for this user and symbol
    position = Position.query.filter_by(user_id=user_id, symbol=symbol).first()
    
    if not position:
        position = Position(
            user_id=user_id,
            symbol=symbol,
            amount=0,
            realized_pnl=0.0
        )
        db.session.add(position)
    
    # Update position
    realized_pnl = position.update_position(order_type, amount, price)
    
    # Save changes
    db.session.commit()
    
    return position, realized_pnl

def get_user_positions_with_pnl(user_id, prices=None):
    """Get user positions with unrealized P&L calculations"""
    if not prices:
        prices = get_current_prices()
        
    positions = Position.query.filter_by(user_id=user_id).all()
    
    # Calculate unrealized P&L for each position
    for position in positions:
        current_price = prices.get(position.symbol)
        if current_price:
            unrealized_pnl = position.calculate_unrealized_pnl(current_price)
            position.unrealized_pnl = unrealized_pnl
            position.current_price = current_price
            position.position_value = abs(position.amount) * current_price
        else:
            position.unrealized_pnl = 0
            position.current_price = 0
            position.position_value = 0
    
    return positions

def calculate_portfolio_value(positions, prices):
    """Calculate total portfolio value including cash and positions"""
    # This is a simplified version, in a real system you'd query the actual account balance
    position_value = sum(abs(p.amount) * prices.get(p.symbol, 0) for p in positions)
    
    # For this example, we'll assume cash is fixed at $10,000
    # In a real system, you'd get this from the exchange API
    cash = 10000.0  
    
    return {
        'position_value': position_value,
        'cash': cash,
        'total': position_value + cash
    }

def get_current_prices():
    """Get current prices for all symbols"""
    try:
        kraken = krakenex.API()
        api = KrakenAPI(kraken)
        
        # Define symbols to fetch
        symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'XRPUSD']
        prices = {}
        
        # Convert to Kraken symbols
        kraken_symbols = []
        for symbol in symbols:
            if symbol.startswith('BTC'):
                kraken_symbols.append('XXBT' + symbol[3:])
            elif symbol.startswith('ETH'):
                kraken_symbols.append('XETH' + symbol[3:])
            elif symbol.startswith('XRP'):
                kraken_symbols.append('XXRP' + symbol[3:])
            else:
                kraken_symbols.append(symbol)
        
        # Get ticker info
        ticker_info = api.get_ticker_information(','.join(kraken_symbols))
        
        # Extract last price
        for i, symbol in enumerate(symbols):
            kraken_symbol = kraken_symbols[i]
            if kraken_symbol in ticker_info.index:
                prices[symbol] = float(ticker_info.loc[kraken_symbol]['c'][0])
        
        return prices
    except Exception as e:
        logger.error(f"Price fetching error: {str(e)}")
        return {}

def get_price_history():
    """Get recent price history for charting"""
    try:
        # For simplicity, we'll use the data we already have
        history = {}
        
        for symbol in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'XRPUSD']:
            file_path = os.path.join('data', f"{symbol}_hourly_4y.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Get last 7 days of data
                recent_data = df.iloc[-168:][['close']]
                history[symbol] = recent_data.to_dict()['close']
        
        return history
    except Exception as e:
        logger.error(f"History fetching error: {str(e)}")
        return {}

def calculate_prediction_statistics():
    """Calculate accuracy statistics for predictions"""
    stats = {
        'total': 0,
        'correct': 0,
        'accuracy': 0,
        'by_symbol': {}
    }
    
    predictions = Prediction.query.filter(Prediction.outcome.isnot(None)).all()
    
    if not predictions:
        return stats
    
    stats['total'] = len(predictions)
    stats['correct'] = sum(1 for p in predictions if p.prediction == p.outcome)
    stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # Calculate by symbol
    for symbol in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'XRPUSD']:
        symbol_preds = [p for p in predictions if p.symbol == symbol]
        if symbol_preds:
            correct = sum(1 for p in symbol_preds if p.prediction == p.outcome)
            stats['by_symbol'][symbol] = {
                'total': len(symbol_preds),
                'correct': correct,
                'accuracy': correct / len(symbol_preds)
            }
    
    return stats

def calculate_trade_statistics(user_id):
    """Calculate statistics for user trades"""
    stats = {
        'total_trades': 0,
        'profitable_trades': 0,
        'total_profit_loss': 0,
        'by_symbol': {}
    }
    
    trades = Trade.query.filter_by(user_id=user_id).all()
    
    if not trades:
        return stats
    
    stats['total_trades'] = len(trades)
    
    # Calculate profit/loss statistics
    trades_with_pnl = [t for t in trades if t.realized_pnl is not None]
    if trades_with_pnl:
        stats['profitable_trades'] = sum(1 for t in trades_with_pnl if t.realized_pnl > 0)
        stats['total_profit_loss'] = sum(t.realized_pnl for t in trades_with_pnl)
        
        # Calculate by symbol
        for symbol in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'XRPUSD']:
            symbol_trades = [t for t in trades_with_pnl if t.symbol == symbol]
            if symbol_trades:
                profitable = sum(1 for t in symbol_trades if t.realized_pnl > 0)
                total_pnl = sum(t.realized_pnl for t in symbol_trades)
                stats['by_symbol'][symbol] = {
                    'total': len(symbol_trades),
                    'profitable': profitable,
                    'win_rate': profitable / len(symbol_trades),
                    'pnl': total_pnl
                }
    
    return stats

def update_predictions():
    """Update predictions for all symbols with confidence thresholds"""
    logger.info("Updating predictions...")
    
    try:
        # Load predictor if needed
        if not hasattr(predictor, 'models') or not predictor.models:
            predictor.load_data()
            
            # Load existing models
            for symbol in predictor.symbols:
                model_path = f'models/{symbol}_model.joblib'
                if os.path.exists(model_path):
                    predictor.models[symbol] = joblib.load(model_path)
                    scaler_path = f'models/{symbol}_scaler.joblib'
                    if os.path.exists(scaler_path):
                        predictor.scalers[symbol] = joblib.load(scaler_path)
        
        # Generate new predictions
        actionable_predictions = 0
        for symbol in predictor.symbols:
            if symbol in predictor.data:
                # Use threshold of 0.3 (60% confidence) - adjust as needed
                prediction_result = predictor.make_predictions_with_threshold(symbol, confidence_threshold=0.3)
                
                if prediction_result:
                    # Add to database
                    new_prediction = Prediction(
                        symbol=symbol,
                        prediction=prediction_result['prediction'] == 1,
                        probability=prediction_result['probability'],
                        horizon_hours=predictor.prediction_horizon
                    )
                    
                    db.session.add(new_prediction)
                    
                    # Log with actionability indicator
                    if prediction_result.get('actionable', False):
                        actionable_predictions += 1
                        logger.info(f"ACTIONABLE prediction for {symbol}: {'UP' if new_prediction.prediction else 'DOWN'} with {new_prediction.probability:.4f} probability")
                    else:
                        logger.info(f"Low confidence prediction for {symbol}: {'UP' if new_prediction.prediction else 'DOWN'} with {new_prediction.probability:.4f} probability")
        
        db.session.commit()
        logger.info(f"Predictions updated successfully. {actionable_predictions} actionable predictions generated.")
    except Exception as e:
        logger.error(f"Prediction update error: {str(e)}")

def check_prediction_outcomes():
    """Check outcomes of past predictions"""
    logger.info("Checking prediction outcomes...")
    
    try:
        # Get predictions without outcomes
        unchecked_predictions = Prediction.query.filter_by(outcome=None).all()
        
        for pred in unchecked_predictions:
            # Skip if prediction horizon hasn't passed yet
            if datetime.datetime.utcnow() < pred.timestamp + datetime.timedelta(hours=pred.horizon_hours):
                continue
            
            # Get actual price data to determine outcome
            symbol = pred.symbol
            file_path = os.path.join('data', f"{symbol}_hourly_4y.csv")
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Find price at prediction time and horizon hours later
                pred_time = pred.timestamp
                target_time = pred_time + datetime.timedelta(hours=pred.horizon_hours)
                
                # Get closest data points
                price_before = df[df.index <= pred_time].iloc[-1]['close'] if not df[df.index <= pred_time].empty else None
                price_after = df[df.index >= target_time].iloc[0]['close'] if not df[df.index >= target_time].empty else None
                
                if price_before is not None and price_after is not None:
                    # Determine actual outcome
                    actual_up = price_after > price_before
                    pred.outcome = actual_up
                    pred.accuracy = 1 if pred.prediction == actual_up else 0
                    
                    logger.info(f"Updated outcome for {symbol} prediction: actual {'UP' if actual_up else 'DOWN'}, predicted {'UP' if pred.prediction else 'DOWN'}")
        
        db.session.commit()
        logger.info("Prediction outcomes updated successfully")
    except Exception as e:
        logger.error(f"Outcome checking error: {str(e)}")

# Set up scheduled tasks
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_predictions, trigger="interval", hours=1)
scheduler.add_job(func=check_prediction_outcomes, trigger="interval", hours=6)
scheduler.start()

# Shut down scheduler when app is closing
atexit.register(lambda: scheduler.shutdown())

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    # Ensure folders exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initial prediction update
    with app.app_context():
        update_predictions()
    
    app.run(debug=True)