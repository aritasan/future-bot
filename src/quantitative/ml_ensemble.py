"""
WorldQuant Machine Learning Ensemble Implementation
Ensemble ML with Random Forest, Gradient Boosting, Neural Networks, SVM.
Implements feature engineering, cross-validation with time-series splits, and model interpretability.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
import os
from datetime import datetime

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Model interpretability will be limited.")

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WorldQuantMLEnsemble:
    """
    WorldQuant-level machine learning ensemble for quantitative trading.
    Implements ensemble methods, feature engineering, and model interpretability.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize WorldQuant ML Ensemble.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # ML model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layer_sizes': (100, 50, 25),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': 42
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1,
                'gamma': 'scale'
            }
        }
        
        # Feature engineering parameters
        self.feature_params = {
            'technical_indicators': True,
            'price_features': True,
            'volume_features': True,
            'volatility_features': True,
            'momentum_features': True,
            'lag_features': [1, 2, 3, 5, 10, 20],
            'rolling_windows': [5, 10, 20, 50],
            'cross_features': True
        }
        
        # Cross-validation parameters
        self.cv_params = {
            'n_splits': 5,
            'test_size': 0.2,
            'gap': 10  # Gap between train and test sets
        }
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.shap_values = {}
        self.performance_metrics = {}
        
        # Training history
        self.training_history = {}
        self.validation_history = {}
        
        logger.info("WorldQuantMLEnsemble initialized")
    
    async def initialize(self) -> bool:
        """Initialize the ML ensemble."""
        try:
            # Initialize models
            self.models = {
                'random_forest': RandomForestRegressor(**self.model_configs['random_forest']),
                'gradient_boosting': GradientBoostingRegressor(**self.model_configs['gradient_boosting']),
                'neural_network': MLPRegressor(**self.model_configs['neural_network']),
                'svm': SVR(**self.model_configs['svm'])
            }
            
            # Initialize scalers
            self.scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler()
            }
            
            logger.info("WorldQuantMLEnsemble initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing WorldQuantMLEnsemble: {str(e)}")
            return False
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features for ML models.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()
            
            # Price-based features
            if self.feature_params['price_features']:
                # Returns
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                # Price levels
                df['price_level'] = (df['close'] - df['close'].rolling(20).min()) / \
                                   (df['close'].rolling(20).max() - df['close'].rolling(20).min())
                
                # Price momentum
                for window in self.feature_params['rolling_windows']:
                    df[f'momentum_{window}'] = df['close'].pct_change(window)
                    df[f'sma_{window}'] = df['close'].rolling(window).mean()
                    df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # Volume-based features
            if self.feature_params['volume_features']:
                df['volume_ma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
                df['volume_momentum'] = df['volume'].pct_change()
                
                # Volume-price relationship
                df['volume_price_trend'] = df['volume'] * df['returns']
                df['volume_price_ratio'] = df['volume'] / df['close']
            
            # Volatility features
            if self.feature_params['volatility_features']:
                for window in self.feature_params['rolling_windows']:
                    df[f'volatility_{window}'] = df['returns'].rolling(window).std()
                    df[f'high_low_ratio_{window}'] = (df['high'] - df['low']) / df['close']
                    df[f'atr_{window}'] = self._calculate_atr(df, window)
            
            # Momentum features
            if self.feature_params['momentum_features']:
                # RSI
                df['rsi'] = self._calculate_rsi(df['close'], 14)
                
                # MACD
                df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                
                # Stochastic
                df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df)
            
            # Technical indicators
            if self.feature_params['technical_indicators']:
                # Bollinger Bands
                df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                # Williams %R
                df['williams_r'] = self._calculate_williams_r(df)
                
                # Commodity Channel Index
                df['cci'] = self._calculate_cci(df)
            
            # Lag features
            for lag in self.feature_params['lag_features']:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
            
            # Cross features
            if self.feature_params['cross_features']:
                df['volume_price_cross'] = df['volume'] * df['returns']
                df['volatility_momentum_cross'] = df['volatility_20'] * df['momentum_20']
                df['rsi_macd_cross'] = df['rsi'] * df['macd']
            
            # Target variable (next period return)
            df['target'] = df['returns'].shift(-1)
            
            # Remove NaN values
            df = df.dropna()
            
            logger.info(f"Feature engineering completed: {df.shape[1]} features created")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return data
    
    def _calculate_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window).mean()
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=df.index)
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD."""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            return macd, signal
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(index=prices.index), pd.Series(index=prices.index)
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        try:
            low_min = df['low'].rolling(14).min()
            high_max = df['high'].rolling(14).max()
            
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            d = k.rolling(3).mean()
            
            return k, d
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return pd.Series(index=df.index), pd.Series(index=df.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = prices.rolling(20).mean()
            std = prices.rolling(20).std()
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            return upper_band, lower_band
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series(index=prices.index), pd.Series(index=prices.index)
    
    def _calculate_williams_r(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R."""
        try:
            highest_high = df['high'].rolling(14).max()
            lowest_low = df['low'].rolling(14).min()
            
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            return williams_r
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series(index=df.index)
    
    def _calculate_cci(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Commodity Channel Index."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma = typical_price.rolling(20).mean()
            mean_deviation = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            
            return cci
        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            return pd.Series(index=df.index)
    
    async def train_ensemble(self, data: pd.DataFrame, target_col: str = 'target') -> Dict[str, Any]:
        """
        Train ensemble of ML models.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("Starting ensemble training...")
            
            # Prepare features and target
            feature_cols = [col for col in data.columns if col != target_col]
            X = data[feature_cols]
            y = data[target_col]
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(
                n_splits=self.cv_params['n_splits'],
                test_size=int(len(data) * self.cv_params['test_size']),
                gap=self.cv_params['gap']
            )
            
            # Training results
            training_results = {}
            validation_results = {}
            
            # Train each model
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                model_results = {
                    'train_scores': [],
                    'val_scores': [],
                    'feature_importance': None,
                    'shap_values': None
                }
                
                # Cross-validation
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features for certain models
                    if model_name in ['neural_network', 'svm']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        
                        # Train model
                        model.fit(X_train_scaled, y_train)
                        
                        # Predictions
                        y_train_pred = model.predict(X_train_scaled)
                        y_val_pred = model.predict(X_val_scaled)
                    else:
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Predictions
                        y_train_pred = model.predict(X_train)
                        y_val_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    train_score = r2_score(y_train, y_train_pred)
                    val_score = r2_score(y_val, y_val_pred)
                    
                    model_results['train_scores'].append(train_score)
                    model_results['val_scores'].append(val_score)
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    model_results['feature_importance'] = dict(zip(feature_cols, model.feature_importances_))
                
                # Calculate SHAP values for interpretability
                if model_name in ['random_forest', 'gradient_boosting'] and SHAP_AVAILABLE:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X.iloc[-100:])  # Last 100 samples
                        model_results['shap_values'] = shap_values
                    except Exception as e:
                        logger.warning(f"Could not calculate SHAP values for {model_name}: {str(e)}")
                elif model_name in ['random_forest', 'gradient_boosting'] and not SHAP_AVAILABLE:
                    logger.warning(f"SHAP not available, skipping SHAP values for {model_name}")
                
                training_results[model_name] = model_results
                
                # Store performance metrics
                self.performance_metrics[model_name] = {
                    'mean_train_score': np.mean(model_results['train_scores']),
                    'mean_val_score': np.mean(model_results['val_scores']),
                    'std_val_score': np.std(model_results['val_scores'])
                }
                
                logger.info(f"{model_name} training completed - Val Score: {np.mean(model_results['val_scores']):.4f}")
            
            # Store results
            self.training_history = training_results
            self.validation_history = validation_results
            
            logger.info("Ensemble training completed successfully")
            return {
                'training_results': training_results,
                'performance_metrics': self.performance_metrics,
                'feature_importance': self._aggregate_feature_importance(training_results)
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble training: {str(e)}")
            return {}
    
    def _aggregate_feature_importance(self, training_results: Dict) -> Dict[str, float]:
        """Aggregate feature importance across all models."""
        try:
            feature_importance = {}
            
            for model_name, results in training_results.items():
                if results['feature_importance']:
                    for feature, importance in results['feature_importance'].items():
                        if feature not in feature_importance:
                            feature_importance[feature] = []
                        feature_importance[feature].append(importance)
            
            # Calculate mean importance
            aggregated_importance = {}
            for feature, importances in feature_importance.items():
                aggregated_importance[feature] = np.mean(importances)
            
            # Sort by importance
            sorted_importance = dict(sorted(aggregated_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error aggregating feature importance: {str(e)}")
            return {}
    
    async def predict_ensemble(self, data: pd.DataFrame, model_weights: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make ensemble predictions.
        
        Args:
            data: DataFrame with features
            model_weights: Optional weights for ensemble combination
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Prepare features
            feature_cols = [col for col in data.columns if col != 'target']
            X = data[feature_cols]
            
            # Individual model predictions
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    # Scale features for certain models
                    if model_name in ['neural_network', 'svm']:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        pred = model.predict(X_scaled)
                    else:
                        pred = model.predict(X)
                    
                    predictions[model_name] = pred
                    
                except Exception as e:
                    logger.warning(f"Error predicting with {model_name}: {str(e)}")
                    predictions[model_name] = np.zeros(len(X))
            
            # Ensemble prediction
            if model_weights is None:
                # Equal weights
                model_weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
            
            ensemble_prediction = np.zeros(len(X))
            for model_name, pred in predictions.items():
                weight = model_weights.get(model_name, 0)
                ensemble_prediction += weight * pred
            
            # Calculate prediction confidence
            prediction_std = np.std([pred for pred in predictions.values()], axis=0)
            confidence = 1.0 / (1.0 + prediction_std)
            
            return {
                'ensemble_prediction': ensemble_prediction,
                'individual_predictions': predictions,
                'confidence': confidence,
                'model_weights': model_weights
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return {}
    
    def get_model_interpretability(self, data: pd.DataFrame, model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Get model interpretability using SHAP values.
        
        Args:
            data: DataFrame with features
            model_name: Name of model to interpret
            
        Returns:
            Dictionary with interpretability results
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return {}
            
            model = self.models[model_name]
            feature_cols = [col for col in data.columns if col != 'target']
            X = data[feature_cols]
            
            # Calculate SHAP values
            if model_name in ['random_forest', 'gradient_boosting'] and SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                    
                    # Feature importance from SHAP
                    feature_importance = np.abs(shap_values).mean(0)
                    feature_importance_dict = dict(zip(feature_cols, feature_importance))
                    
                    # Sort by importance
                    sorted_importance = dict(sorted(feature_importance_dict.items(), 
                                                  key=lambda x: x[1], reverse=True))
                    
                    return {
                        'shap_values': shap_values,
                        'feature_importance': sorted_importance,
                        'expected_value': explainer.expected_value,
                        'model_name': model_name
                    }
                except Exception as e:
                    logger.warning(f"Error calculating SHAP values for {model_name}: {str(e)}")
                    return {}
            elif model_name in ['random_forest', 'gradient_boosting'] and not SHAP_AVAILABLE:
                logger.warning(f"SHAP not available for {model_name}, using feature importances only")
                if hasattr(model, 'feature_importances_'):
                    feature_importance_dict = dict(zip(feature_cols, model.feature_importances_))
                    sorted_importance = dict(sorted(feature_importance_dict.items(), 
                                                  key=lambda x: x[1], reverse=True))
                    return {
                        'feature_importance': sorted_importance,
                        'model_name': model_name
                    }
                else:
                    return {}
            
            elif model_name == 'neural_network':
                # For neural networks, use permutation importance
                from sklearn.inspection import permutation_importance
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Calculate permutation importance
                perm_importance = permutation_importance(model, X_scaled, 
                                                      n_repeats=10, random_state=42)
                
                feature_importance_dict = dict(zip(feature_cols, perm_importance.importances_mean))
                sorted_importance = dict(sorted(feature_importance_dict.items(), 
                                              key=lambda x: x[1], reverse=True))
                
                return {
                    'feature_importance': sorted_importance,
                    'permutation_importance': perm_importance,
                    'model_name': model_name
                }
            
            else:
                logger.warning(f"SHAP interpretation not available for {model_name}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in model interpretability: {str(e)}")
            return {}
    
    async def get_ml_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML ensemble summary."""
        try:
            summary = {
                'total_models': len(self.models),
                'model_names': list(self.models.keys()),
                'performance_metrics': self.performance_metrics,
                'feature_importance': self._aggregate_feature_importance(self.training_history),
                'training_history': len(self.training_history),
                'validation_history': len(self.validation_history)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting ML summary: {str(e)}")
            return {}
    
    async def predict_signal_outcome(self, signal: Dict, market_data: Dict) -> Dict[str, Any]:
        """
        Predict outcome for a trading signal.
        
        Args:
            signal: Trading signal dictionary
            market_data: Market data dictionary
            
        Returns:
            Dictionary with ML prediction results
        """
        try:
            prediction_results = {
                'positive_prediction': False,
                'prediction_confidence': 0.0,
                'predicted_return': 0.0,
                'model_agreement': 0.0,
                'feature_importance': {}
            }
            
            # For now, return default prediction
            # In a real implementation, this would use trained models to predict signal outcome
            signal_strength = signal.get('strength', 0.0)
            confidence = signal.get('confidence', 0.0)
            
            # Simple prediction logic based on signal parameters
            if signal_strength >= 0.6 and confidence >= 0.7:
                prediction_results['positive_prediction'] = True
                prediction_results['prediction_confidence'] = 0.8
                prediction_results['predicted_return'] = 0.02  # 2% predicted return
                prediction_results['model_agreement'] = 0.75
            elif signal_strength >= 0.4 and confidence >= 0.5:
                prediction_results['positive_prediction'] = True
                prediction_results['prediction_confidence'] = 0.6
                prediction_results['predicted_return'] = 0.01  # 1% predicted return
                prediction_results['model_agreement'] = 0.6
            else:
                prediction_results['positive_prediction'] = False
                prediction_results['prediction_confidence'] = 0.3
                prediction_results['predicted_return'] = -0.01  # -1% predicted return
                prediction_results['model_agreement'] = 0.4
            
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error predicting signal outcome: {str(e)}")
            return {
                'positive_prediction': False,
                'prediction_confidence': 0.0,
                'error': str(e)
            }
    
    async def predict_symbol_movement(self, symbol: str) -> Dict[str, Any]:
        """
        Predict movement for a trading symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with symbol movement prediction
        """
        try:
            prediction_results = {
                'symbol': symbol,
                'positive_prediction': False,
                'prediction_confidence': 0.0,
                'predicted_direction': 'neutral',
                'predicted_magnitude': 0.0,
                'model_agreement': 0.0
            }
            
            # For now, return default prediction
            # In a real implementation, this would use trained models to predict symbol movement
            prediction_results['prediction_confidence'] = 0.6
            prediction_results['predicted_direction'] = 'up'
            prediction_results['predicted_magnitude'] = 0.015  # 1.5% predicted movement
            prediction_results['model_agreement'] = 0.65
            
            # Determine if prediction is positive
            if prediction_results['predicted_direction'] == 'up' and prediction_results['prediction_confidence'] >= 0.5:
                prediction_results['positive_prediction'] = True
            
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error predicting symbol movement for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'positive_prediction': False,
                'prediction_confidence': 0.0,
                'error': str(e)
            }

    async def close(self) -> None:
        """Close the worldquantmlensemble and cleanup resources."""
        try:
            logger.info("Closing WorldQuantMLEnsemble...")
            
            # Clear any stored data
            if hasattr(self, 'analysis_cache'):
                self.analysis_cache.clear()
            if hasattr(self, 'history'):
                self.history.clear()
            if hasattr(self, 'metrics_history'):
                self.metrics_history.clear()
            
            logger.info("WorldQuantMLEnsemble closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing WorldQuantMLEnsemble: {str(e)}")
            raise
