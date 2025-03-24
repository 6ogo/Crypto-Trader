# Add this class to a new file named combined_signals.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import desc

class SignalCombiner:
    """Combines signals from multiple prediction sources to generate trading signals"""
    
    def __init__(self, db, Prediction, Trade, User, logger):
        """Initialize with database models"""
        self.db = db
        self.Prediction = Prediction
        self.Trade = Trade
        self.User = User
        self.logger = logger
        
    def get_recent_predictions(self, hours=24):
        """Get predictions from the last X hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        predictions = self.Prediction.query.filter(
            self.Prediction.timestamp >= cutoff_time
        ).order_by(desc(self.Prediction.timestamp)).all()
        
        return predictions
    
    def predictions_to_dataframe(self, predictions):
        """Convert prediction objects to a DataFrame"""
        data = []
        for p in predictions:
            data.append({
                'symbol': p.symbol,
                'prediction': 1 if p.prediction else -1,  # 1 for UP, -1 for DOWN
                'probability': p.probability,
                'timestamp': p.timestamp,
                'horizon_hours': p.horizon_hours,
                'confidence': abs(p.probability - 0.5) * 2  # Scale to 0-1
            })
        return pd.DataFrame(data)
    
    def generate_market_signals(self):
        """Generate combined market signals for all symbols"""
        # Get recent predictions
        recent_preds = self.get_recent_predictions(hours=48)
        
        if not recent_preds:
            self.logger.warning("No recent predictions found")
            return {}
            
        # Convert to DataFrame
        df = self.predictions_to_dataframe(recent_preds)
        
        # Group by symbol and get latest prediction for each
        symbols = df['symbol'].unique()
        
        signals = {}
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol]
            
            # Skip if not enough data
            if len(symbol_df) < 3:
                continue
                
            # Calculate various signal components:
            # 1. Latest prediction
            latest = symbol_df.sort_values('timestamp', ascending=False).iloc[0]
            
            # 2. Weighted average of recent predictions (weighted by confidence)
            recent_df = symbol_df.iloc[:5]  # Last 5 predictions
            weighted_signal = np.average(
                recent_df['prediction'], 
                weights=recent_df['confidence']
            ) if not recent_df.empty else 0
            
            # 3. Trend of predictions
            if len(symbol_df) >= 3:
                sorted_df = symbol_df.sort_values('timestamp')
                trend = np.polyfit(range(len(sorted_df)), sorted_df['prediction'], 1)[0]
            else:
                trend = 0
                
            # 4. Confidence-weighted consensus
            consensus = np.mean(symbol_df['prediction'])
            
            # Combine signals (simple weighted average for now)
            raw_signal = (
                0.4 * latest['prediction'] +  # Latest prediction (40% weight)
                0.3 * weighted_signal +       # Weighted recent average (30% weight)
                0.2 * consensus +             # Overall consensus (20% weight)
                0.1 * trend                   # Trend direction (10% weight)
            )
            
            # Scale to -1 to 1 and add confidence measure
            signal_strength = max(min(raw_signal, 1.0), -1.0)
            confidence = abs(signal_strength) * latest['confidence']
            
            signals[symbol] = {
                'signal': 'UP' if signal_strength > 0 else 'DOWN',
                'strength': abs(signal_strength),
                'confidence': confidence,
                'raw_value': raw_signal,
                'actionable': confidence > 0.4,  # Threshold for actionable signals
                'components': {
                    'latest': latest['prediction'],
                    'weighted_recent': weighted_signal,
                    'consensus': consensus,
                    'trend': trend
                }
            }
            
        return signals