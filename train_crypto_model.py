import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.pipeline import Pipeline
from scipy import stats
import talib
import warnings
warnings.filterwarnings('ignore')

class CryptoPredictor:
    def __init__(self, base_folder='data'):
        self.base_folder = base_folder
        self.symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'AVAXUSD', 'XRPUSD']
        self.models = {}
        self.scalers = {}
        self.prediction_horizon = 12  # Hours to predict ahead
        self.feature_importance = {}
    
    def load_data(self):
        """Load data for all cryptocurrencies"""
        self.data = {}
        
        for symbol in self.symbols:
            file_path = os.path.join(self.base_folder, f"{symbol}_hourly_4y.csv")
            if os.path.exists(file_path):
                print(f"Loading {symbol} data...")
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Convert columns to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                self.data[symbol] = df
            else:
                print(f"Warning: Data file for {symbol} not found at {file_path}")
        
        return self.data
    
    def add_features(self, df):
        """Add technical indicators and custom features to dataframe"""
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for ma_period in [7, 14, 21, 50, 100, 200]:
            df[f'ma_{ma_period}'] = df['close'].rolling(window=ma_period).mean()
            df[f'ma_vol_{ma_period}'] = df['volume'].rolling(window=ma_period).mean()
        
        # Price relative to moving averages
        for ma_period in [7, 21, 50, 100]:
            df[f'close_over_ma_{ma_period}'] = df['close'] / df[f'ma_{ma_period}']
        
        # Moving average crossovers
        df['ma_7_over_21'] = df['ma_7'] / df['ma_21']
        df['ma_21_over_50'] = df['ma_21'] / df['ma_50']
        df['ma_50_over_100'] = df['ma_50'] / df['ma_100']
        
        # Volatility indicators
        df['atr_14'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['natr_14'] = talib.NATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # Trend indicators
        df['adx_14'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Momentum indicators
        df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14)
        df['rsi_21'] = talib.RSI(df['close'].values, timeperiod=21)
        df['mfi_14'] = talib.MFI(df['high'].values, df['low'].values, 
                               df['close'].values, df['volume'].values, timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        df['obv_change'] = df['obv'].pct_change(5)
        df['volume_change'] = df['volume'].pct_change(1)
        
        # Advanced indicators
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'].values, df['low'].values, 
                                                 df['close'].values, fastk_period=14, 
                                                 slowk_period=3, slowk_matype=0, 
                                                 slowd_period=3, slowd_matype=0)
        
        # Ichimoku Cloud elements
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        
        # Price channels
        df['upper_channel'] = df['high'].rolling(20).max()
        df['lower_channel'] = df['low'].rolling(20).min()
        df['channel_width'] = (df['upper_channel'] - df['lower_channel']) / df['close']
        df['channel_position'] = (df['close'] - df['lower_channel']) / (df['upper_channel'] - df['lower_channel'])
        
        # Hurst exponent (market efficiency)
        df['hurst_exponent'] = df['close'].rolling(window=100).apply(
            lambda x: self.calculate_hurst_exponent(x) if len(x.dropna()) > 50 else np.nan, raw=False
        )
        
        # Cycle measures
        df['wpr_14'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['wpr_21'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=21)
        
        # Additional trend features
        df['close_over_open'] = df['close'] / df['open']
        df['high_over_open'] = df['high'] / df['open']
        df['low_over_open'] = df['low'] / df['open']
        
        # Market regime features
        df['volatility_21'] = df['returns'].rolling(window=21).std()
        df['volatility_change'] = df['volatility_21'].pct_change(5)
        
        # Smoothed momentum
        df['momentum_14'] = df['close'] - df['close'].shift(14)
        df['momentum_smoothed'] = df['momentum_14'].rolling(window=3).mean()
        
        # Time features (hourly seasonality)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Target variables
        for hours in [4, 8, 12, 24]:
            future_returns = df['close'].pct_change(hours).shift(-hours)
            df[f'future_return_{hours}h'] = future_returns
            df[f'target_{hours}h'] = (future_returns > 0).astype(int)
        
        # Clean up NaN values
        df = df.dropna()
        
        return df
    
    def calculate_hurst_exponent(self, series, max_lag=20):
        """Calculate Hurst exponent for time series"""
        lags = range(2, min(max_lag, len(series) // 4))
        tau = [np.std(np.subtract(series.values[lag:], series.values[:-lag])) for lag in lags]
        if not all(tau):
            return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    def prepare_features_and_target(self, df, horizon=12):
        """Prepare feature matrix X and target vector y"""
        # Define features to exclude from training
        exclude_cols = [col for col in df.columns if 'future_return' in col or 'target_' in col]
        exclude_cols += ['symbol', 'open', 'high', 'low', 'close', 'volume', 'chikou_span']
        
        # Select features and target
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[f'target_{horizon}h'].copy()
        
        return X, y
    
    def train_all_models(self):
        """Train models for all cryptocurrencies"""
        if not hasattr(self, 'data'):
            self.load_data()
        
        for symbol, df in self.data.items():
            print(f"\nTraining model for {symbol}...")
            
            # Add features
            df_with_features = self.add_features(df)
            print(f"Data shape after adding features: {df_with_features.shape}")
            
            # Prepare features and target
            X, y = self.prepare_features_and_target(df_with_features, horizon=self.prediction_horizon)
            
            # Save the feature list for future reference
            feature_list = X.columns.tolist()
            joblib.dump(feature_list, f'models/{symbol}_features.joblib')
            
            # Train model with cross-validation
            self.train_model_with_cv(X, y, symbol)
            
            # Train final model on all data
            self.train_final_model(X, y, symbol)
    
    def train_model_with_cv(self, X, y, symbol):
        """Train model with time series cross-validation"""
        print(f"Training with time series cross-validation...")
        
        # Define cross-validation strategy
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42
            ))
        ])
        
        # Perform cross-validation
        cv_results = cross_validate(
            pipeline, X, y,
            cv=tscv,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            return_train_score=True
        )
        
        # Print cross-validation results
        print(f"Cross-validation results for {symbol}:")
        print(f"  Mean accuracy: {cv_results['test_accuracy'].mean():.4f}")
        print(f"  Mean precision: {cv_results['test_precision'].mean():.4f}")
        print(f"  Mean recall: {cv_results['test_recall'].mean():.4f}")
        print(f"  Mean F1 score: {cv_results['test_f1'].mean():.4f}")
        
        # Check for potential overfitting
        train_test_diff = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
        print(f"  Train-test accuracy difference: {train_test_diff:.4f} ({'High overfitting' if train_test_diff > 0.1 else 'Acceptable'})")
    
    def train_final_model(self, X, y, symbol):
        """Train final model on all data and save it"""
        print(f"Training final model for {symbol}...")
        
        # Create and save directory for models
        os.makedirs('models', exist_ok=True)
        
        # Create and fit the scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and fit the model
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump(model, f'models/{symbol}_model.joblib')
        joblib.dump(scaler, f'models/{symbol}_scaler.joblib')
        
        # Store models and scalers in memory
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save and store feature importance
        self.feature_importance[symbol] = feature_importance
        feature_importance.to_csv(f'models/{symbol}_feature_importance.csv', index=False)
        
        # Print top 15 features
        print("\nTop 15 important features:")
        print(feature_importance.head(15))
    
    def make_predictions(self, symbol, new_data=None):
        """Make predictions using trained model"""
        # Load model and scaler if not in memory
        if symbol not in self.models:
            model_path = f'models/{symbol}_model.joblib'
            scaler_path = f'models/{symbol}_scaler.joblib'
            feature_path = f'models/{symbol}_features.joblib'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_path):
                self.models[symbol] = joblib.load(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                self.feature_list = joblib.load(feature_path)
            else:
                print(f"Error: Model files for {symbol} not found!")
                return None
        
        # Use latest data from loaded dataframe if new_data not provided
        if new_data is None:
            if not hasattr(self, 'data') or symbol not in self.data:
                self.load_data()
            
            df = self.data[symbol].copy()
            df_with_features = self.add_features(df)
            
            # Get latest data point
            latest_data = df_with_features.iloc[-1:]
            
            # Prepare features
            X, _ = self.prepare_features_and_target(latest_data, horizon=self.prediction_horizon)
        else:
            # Process new data
            new_data_with_features = self.add_features(new_data)
            X, _ = self.prepare_features_and_target(new_data_with_features, horizon=self.prediction_horizon)
        
        # Scale features
        X_scaled = self.scalers[symbol].transform(X)
        
        # Make predictions
        pred_prob = self.models[symbol].predict_proba(X_scaled)[0, 1]
        pred_class = 1 if pred_prob > 0.5 else 0
        
        return {
            'symbol': symbol,
            'prediction': pred_class,  # 1 for up, 0 for down
            'probability': pred_prob,
            'confidence': max(pred_prob, 1 - pred_prob),
            'direction': 'UP' if pred_class == 1 else 'DOWN',
            'horizon_hours': self.prediction_horizon,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models on test data"""
        results = {}
        
        for symbol in self.symbols:
            if symbol not in self.data:
                continue
                
            # Add features
            df = self.add_features(self.data[symbol])
            
            # Split into train and test (last 30 days as test)
            test_size = 30 * 24  # 30 days of hourly data
            if len(df) <= test_size:
                test_size = len(df) // 5  # 20% of data
                
            train_df = df.iloc[:-test_size]
            test_df = df.iloc[-test_size:]
            
            # Prepare features and target
            X_test, y_test = self.prepare_features_and_target(test_df, horizon=self.prediction_horizon)
            
            # Load model and scaler
            model_path = f'models/{symbol}_model.joblib'
            scaler_path = f'models/{symbol}_scaler.joblib'
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"Model files for {symbol} not found. Skipping evaluation.")
                continue
                
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Scale features and predict
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Print results
            print(f"\nEvaluation results for {symbol}:")
            print(f"  Test size: {len(X_test)} data points")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("  Confusion Matrix:")
            print(f"    TN: {cm[0, 0]}, FP: {cm[0, 1]}")
            print(f"    FN: {cm[1, 0]}, TP: {cm[1, 1]}")
            
            # Store results
            results[symbol] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm
            }
            
            # Plot precision-recall curve
            plt.figure(figsize=(10, 6))
            plt.scatter(y_prob, y_test, alpha=0.3)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Actual Label')
            plt.title(f'{symbol} - Predicted Probabilities vs Actual Labels')
            plt.grid(True)
            plt.savefig(f'models/{symbol}_predictions.png')
            plt.close()
        
        return results

if __name__ == "__main__":
    predictor = CryptoPredictor()
    predictor.load_data()
    predictor.train_all_models()
    evaluation_results = predictor.evaluate_all_models()
    
    print("\nTraining and evaluation completed!")
    
    # Make predictions for all symbols
    print("\nCurrent predictions:")
    for symbol in predictor.symbols:
        if symbol in predictor.data:
            prediction = predictor.make_predictions(symbol)
            if prediction:
                print(f"{symbol}: {prediction['direction']} with {prediction['probability']:.2f} probability")