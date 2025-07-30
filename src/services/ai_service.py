"""
AI service for real-time trading signal analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)

class AIService:
    """Service for AI-powered trading signal analysis."""
    
    def __init__(self, config: Dict):
        """Initialize AI service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._is_initialized = False
        self._is_closed = False
        
        # Model paths
        self.model_dir = config.get('model_dir', 'models')
        self.price_prediction_model = None
        self.signal_classification_model = None
        self.anomaly_detection_model = None
        
        # Feature scalers
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        
        # Cache for real-time predictions
        self._prediction_cache = {}
        self._cache_ttl = 60  # 1 minute cache TTL
        
    async def initialize(self) -> bool:
        """Initialize AI service and load models."""
        try:
            # Create model directory if not exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Load or initialize models
            await self._load_models()
            
            self._is_initialized = True
            logger.info("AI service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing AI service: {str(e)}")
            return False
            
    async def _load_models(self) -> None:
        """Load or initialize ML models."""
        try:
            # Load price prediction model
            price_model_path = os.path.join(self.model_dir, 'price_prediction_model.h5')
            if os.path.exists(price_model_path):
                self.price_prediction_model = tf.keras.models.load_model(price_model_path)
            else:
                self.price_prediction_model = self._create_price_prediction_model()
                
            # Load signal classification model
            signal_model_path = os.path.join(self.model_dir, 'signal_classification_model.h5')
            if os.path.exists(signal_model_path):
                self.signal_classification_model = tf.keras.models.load_model(signal_model_path)
            else:
                self.signal_classification_model = self._create_signal_classification_model()
                
            # Load anomaly detection model
            anomaly_model_path = os.path.join(self.model_dir, 'anomaly_detection_model.joblib')
            if os.path.exists(anomaly_model_path):
                self.anomaly_detection_model = joblib.load(anomaly_model_path)
            else:
                self.anomaly_detection_model = self._create_anomaly_detection_model()
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
            
    def _create_price_prediction_model(self) -> tf.keras.Model:
        """Create LSTM model for price prediction."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(60, 10), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _create_signal_classification_model(self) -> tf.keras.Model:
        """Create CNN model for signal classification."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(60, 10)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='softmax')  # Buy, Sell, Hold
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _create_anomaly_detection_model(self) -> object:
        """Create isolation forest model for anomaly detection."""
        from sklearn.ensemble import IsolationForest
        
        return IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
    async def analyze_realtime_signal(self, market_data: Dict) -> Dict:
        """Analyze real-time trading signal using AI models.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dict: Analysis results including predictions and confidence scores
        """
        try:
            if not self._is_initialized:
                logger.error("AI service not initialized")
                return {}
                
            # Prepare features
            features = self._prepare_features(market_data)
            if features is None:
                return {}
                
            # Get predictions from all models
            price_prediction = await self._predict_price(features)
            signal_classification = await self._classify_signal(features)
            anomaly_score = await self._detect_anomalies(features)
            
            # Combine predictions
            analysis = {
                'price_prediction': price_prediction,
                'signal_classification': signal_classification,
                'anomaly_score': anomaly_score,
                'confidence_score': self._calculate_confidence_score(
                    price_prediction,
                    signal_classification,
                    anomaly_score
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache results
            self._prediction_cache[market_data['symbol']] = {
                'analysis': analysis,
                'timestamp': datetime.now()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing real-time signal: {str(e)}")
            return {}
            
    def _prepare_features(self, market_data: Dict) -> Optional[np.ndarray]:
        """Prepare features for model input."""
        try:
            # Extract features from market data
            features = []
            
            # Price features
            features.extend([
                market_data['close'],
                market_data['high'],
                market_data['low'],
                market_data['volume']
            ])
            
            # Technical indicators
            features.extend([
                market_data.get('rsi', 50),
                market_data.get('macd', 0),
                market_data.get('macd_signal', 0),
                market_data.get('macd_hist', 0),
                market_data.get('bb_upper', 0),
                market_data.get('bb_lower', 0)
            ])
            
            # Convert to numpy array and reshape
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features = self.feature_scaler.fit_transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
            
    async def _predict_price(self, features: np.ndarray) -> Dict:
        """Predict future price movement."""
        try:
            # Reshape features for LSTM input
            lstm_features = features.reshape(1, 60, 10)
            
            # Get prediction
            prediction = self.price_prediction_model.predict(lstm_features)
            
            return {
                'predicted_price': float(prediction[0][0]),
                'confidence': float(self._calculate_prediction_confidence(prediction))
            }
            
        except Exception as e:
            logger.error(f"Error predicting price: {str(e)}")
            return {}
            
    async def _classify_signal(self, features: np.ndarray) -> Dict:
        """Classify trading signal."""
        try:
            # Reshape features for CNN input
            cnn_features = features.reshape(1, 60, 10)
            
            # Get prediction
            prediction = self.signal_classification_model.predict(cnn_features)
            
            # Get class with highest probability
            signal_class = np.argmax(prediction[0])
            confidence = float(prediction[0][signal_class])
            
            signal_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
            
            return {
                'signal': signal_map[signal_class],
                'confidence': confidence,
                'probabilities': {
                    'BUY': float(prediction[0][0]),
                    'SELL': float(prediction[0][1]),
                    'HOLD': float(prediction[0][2])
                }
            }
            
        except Exception as e:
            logger.error(f"Error classifying signal: {str(e)}")
            return {}
            
    async def _detect_anomalies(self, features: np.ndarray) -> Dict:
        """Detect market anomalies."""
        try:
            # Get anomaly score
            score = self.anomaly_detection_model.score_samples(features)
            
            # Convert score to probability
            probability = 1 / (1 + np.exp(-score))
            
            return {
                'is_anomaly': bool(probability > 0.7),
                'anomaly_score': float(probability),
                'threshold': 0.7
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return {}
            
    def _calculate_confidence_score(self, price_prediction: Dict,
                                  signal_classification: Dict,
                                  anomaly_score: Dict) -> float:
        """Calculate overall confidence score."""
        try:
            # Weights for different components
            weights = {
                'price': 0.4,
                'signal': 0.4,
                'anomaly': 0.2
            }
            
            # Calculate weighted score
            confidence = (
                weights['price'] * price_prediction.get('confidence', 0) +
                weights['signal'] * signal_classification.get('confidence', 0) +
                weights['anomaly'] * (1 - anomaly_score.get('anomaly_score', 0))
            )
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.0
            
    def _calculate_prediction_confidence(self, prediction: np.ndarray) -> float:
        """Calculate confidence score for price prediction."""
        try:
            # Calculate confidence based on prediction variance
            variance = np.var(prediction)
            confidence = 1 / (1 + variance)
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {str(e)}")
            return 0.0
            
    async def train_models(self, training_data: pd.DataFrame) -> bool:
        """Train AI models with new data."""
        try:
            if not self._is_initialized:
                logger.error("AI service not initialized")
                return False
                
            # Prepare training data
            X_train, y_train = self._prepare_training_data(training_data)
            
            # Train price prediction model
            self.price_prediction_model.fit(
                X_train['price'],
                y_train['price'],
                epochs=50,
                batch_size=32,
                validation_split=0.2
            )
            
            # Train signal classification model
            self.signal_classification_model.fit(
                X_train['signal'],
                y_train['signal'],
                epochs=50,
                batch_size=32,
                validation_split=0.2
            )
            
            # Train anomaly detection model
            self.anomaly_detection_model.fit(X_train['anomaly'])
            
            # Save models
            self._save_models()
            
            logger.info("Models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
            
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Prepare training data for models."""
        try:
            # Split features and targets
            X = {
                'price': data[['close', 'high', 'low', 'volume', 'rsi', 'macd',
                             'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower']].values,
                'signal': data[['close', 'high', 'low', 'volume', 'rsi', 'macd',
                              'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower']].values,
                'anomaly': data[['close', 'high', 'low', 'volume', 'rsi', 'macd',
                               'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower']].values
            }
            
            y = {
                'price': data['close'].shift(-1).dropna().values,
                'signal': pd.get_dummies(data['signal']).values,
                'anomaly': data['is_anomaly'].values
            }
            
            # Scale features
            for key in X:
                X[key] = self.feature_scaler.fit_transform(X[key])
                
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
            
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            # Save price prediction model
            self.price_prediction_model.save(
                os.path.join(self.model_dir, 'price_prediction_model.h5')
            )
            
            # Save signal classification model
            self.signal_classification_model.save(
                os.path.join(self.model_dir, 'signal_classification_model.h5')
            )
            
            # Save anomaly detection model
            joblib.dump(
                self.anomaly_detection_model,
                os.path.join(self.model_dir, 'anomaly_detection_model.joblib')
            )
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
            
    async def close(self):
        """Close AI service and save models."""
        try:
            if self._is_initialized:
                self._save_models()
                self._is_closed = True
                logger.info("AI service closed")
        except Exception as e:
            logger.error(f"Error closing AI service: {str(e)}") 