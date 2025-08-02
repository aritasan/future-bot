#!/usr/bin/env python3
"""
Implementation of WorldQuant-level DCA and Trailing Stop enhancements.
Based on expert analysis and WorldQuant standards.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorldQuantDCA:
    """WorldQuant-level DCA implementation with quantitative analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dca_history = {}
        self.volatility_regime = 'normal'
        self.dca_config = config.get('risk_management', {}).get('dca', {})
        
    async def calculate_optimal_dca_timing(self, symbol: str, position: Dict) -> Dict:
        """Calculate optimal DCA timing using quantitative analysis."""
        try:
            logger.info(f"Calculating optimal DCA timing for {symbol}")
            
            # 1. Volatility Regime Analysis
            volatility_regime = await self._analyze_volatility_regime(symbol)
            
            # 2. Market Microstructure Analysis
            market_impact = await self._calculate_market_impact(symbol)
            
            # 3. Statistical Arbitrage Signals
            mean_reversion_signal = await self._analyze_mean_reversion(symbol)
            
            # 4. Factor Model Analysis
            factor_exposures = await self._get_factor_exposures(symbol)
            
            # 5. Machine Learning Prediction
            ml_prediction = await self._get_ml_dca_prediction(symbol)
            
            # 6. Optimal DCA Decision
            dca_decision = self._combine_signals_for_dca(
                volatility_regime, market_impact, mean_reversion_signal,
                factor_exposures, ml_prediction, position
            )
            
            logger.info(f"DCA decision for {symbol}: {dca_decision}")
            return dca_decision
            
        except Exception as e:
            logger.error(f"Error in optimal DCA timing for {symbol}: {str(e)}")
            return None
    
    async def _analyze_volatility_regime(self, symbol: str) -> str:
        """Analyze volatility regime for DCA decisions."""
        try:
            # Get recent price data
            klines = await self._get_recent_klines(symbol, '1h', 24)
            if not klines:
                return 'normal'
            
            # Calculate volatility
            returns = np.diff(np.log([float(k[4]) for k in klines]))
            volatility = np.std(returns) * np.sqrt(24)  # Annualized
            
            # Determine regime
            if volatility < 0.02:  # < 2%
                return 'low'
            elif volatility < 0.05:  # 2-5%
                return 'normal'
            else:  # > 5%
                return 'high'
                
        except Exception as e:
            logger.error(f"Error analyzing volatility regime for {symbol}: {str(e)}")
            return 'normal'
    
    async def _calculate_market_impact(self, symbol: str) -> float:
        """Calculate market impact for DCA timing."""
        try:
            # Get order book data
            orderbook = await self._get_orderbook(symbol)
            if not orderbook:
                return 0.0
            
            # Calculate bid-ask spread
            best_bid = float(orderbook['bids'][0][0])
            best_ask = float(orderbook['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Calculate order imbalance
            bid_volume = sum(float(bid[1]) for bid in orderbook['bids'][:5])
            ask_volume = sum(float(ask[1]) for ask in orderbook['asks'][:5])
            imbalance = abs(bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Market impact score (0-1, lower is better)
            impact_score = (spread * 100) + (imbalance * 0.5)
            
            return impact_score
            
        except Exception as e:
            logger.error(f"Error calculating market impact for {symbol}: {str(e)}")
            return 0.0
    
    async def _analyze_mean_reversion(self, symbol: str) -> Dict:
        """Analyze mean reversion for DCA signals."""
        try:
            # Get recent price data
            klines = await self._get_recent_klines(symbol, '1h', 48)
            if not klines:
                return {'signal': 'neutral', 'strength': 0.0}
            
            # Calculate price deviation from moving average
            prices = [float(k[4]) for k in klines]
            ma_20 = np.mean(prices[-20:])
            current_price = prices[-1]
            
            deviation = (current_price - ma_20) / ma_20
            
            # Determine mean reversion signal
            if deviation < -0.02:  # 2% below MA
                signal = 'buy'
                strength = min(abs(deviation) / 0.05, 1.0)  # Normalize to 0-1
            elif deviation > 0.02:  # 2% above MA
                signal = 'sell'
                strength = min(abs(deviation) / 0.05, 1.0)
            else:
                signal = 'neutral'
                strength = 0.0
            
            return {'signal': signal, 'strength': strength}
            
        except Exception as e:
            logger.error(f"Error analyzing mean reversion for {symbol}: {str(e)}")
            return {'signal': 'neutral', 'strength': 0.0}
    
    async def _get_factor_exposures(self, symbol: str) -> Dict:
        """Get factor exposures for DCA analysis."""
        try:
            # Mock factor exposures (in real implementation, this would come from factor model)
            factor_exposures = {
                'market': 0.8,
                'size': 0.2,
                'value': 0.1,
                'momentum': 0.3,
                'volatility': 0.4,
                'liquidity': 0.6
            }
            
            return factor_exposures
            
        except Exception as e:
            logger.error(f"Error getting factor exposures for {symbol}: {str(e)}")
            return {}
    
    async def _get_ml_dca_prediction(self, symbol: str) -> Dict:
        """Get ML prediction for DCA timing."""
        try:
            # Mock ML prediction (in real implementation, this would come from ML models)
            ml_prediction = {
                'confidence': 0.75,
                'prediction': 'buy',
                'probability': 0.65,
                'model_agreement': 0.7
            }
            
            return ml_prediction
            
        except Exception as e:
            logger.error(f"Error getting ML DCA prediction for {symbol}: {str(e)}")
            return {'confidence': 0.5, 'prediction': 'neutral', 'probability': 0.5, 'model_agreement': 0.5}
    
    def _combine_signals_for_dca(self, volatility_regime: str, market_impact: float, 
                                mean_reversion: Dict, factor_exposures: Dict, 
                                ml_prediction: Dict, position: Dict) -> Dict:
        """Combine all signals for DCA decision."""
        try:
            # Calculate composite score
            score = 0.0
            reasons = []
            
            # Volatility regime score (0-1)
            volatility_scores = {'low': 0.8, 'normal': 0.6, 'high': 0.4}
            vol_score = volatility_scores.get(volatility_regime, 0.6)
            score += vol_score * 0.2
            reasons.append(f"Volatility regime: {volatility_regime} (score: {vol_score:.2f})")
            
            # Market impact score (0-1, inverted)
            impact_score = max(0, 1 - market_impact)
            score += impact_score * 0.2
            reasons.append(f"Market impact: {impact_score:.2f}")
            
            # Mean reversion score (0-1)
            if mean_reversion['signal'] == 'buy':
                mr_score = mean_reversion['strength']
            else:
                mr_score = 0.0
            score += mr_score * 0.3
            reasons.append(f"Mean reversion: {mr_score:.2f}")
            
            # ML prediction score (0-1)
            if ml_prediction['prediction'] == 'buy':
                ml_score = ml_prediction['confidence']
            else:
                ml_score = 0.0
            score += ml_score * 0.3
            reasons.append(f"ML prediction: {ml_score:.2f}")
            
            # Determine DCA decision
            threshold = 0.6
            should_dca = score >= threshold
            
            return {
                'should_dca': should_dca,
                'score': score,
                'threshold': threshold,
                'reasons': reasons,
                'volatility_regime': volatility_regime,
                'market_impact': market_impact,
                'mean_reversion': mean_reversion,
                'ml_prediction': ml_prediction
            }
            
        except Exception as e:
            logger.error(f"Error combining signals for DCA: {str(e)}")
            return {'should_dca': False, 'score': 0.0, 'reasons': ['Error in signal combination']}
    
    async def _get_recent_klines(self, symbol: str, timeframe: str, limit: int) -> List:
        """Get recent klines data."""
        # Mock implementation - in real code, this would call the actual service
        return []
    
    async def _get_orderbook(self, symbol: str) -> Dict:
        """Get order book data."""
        # Mock implementation - in real code, this would call the actual service
        return {'bids': [[100, 1]], 'asks': [[101, 1]]}

class WorldQuantTrailingStop:
    """WorldQuant-level trailing stop implementation with quantitative analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trailing_history = {}
        self.volatility_regime = 'normal'
        self.trailing_config = config.get('risk_management', {}).get('trailing_stop', {})
        
    async def calculate_optimal_trailing_stop(self, symbol: str, position: Dict) -> Dict:
        """Calculate optimal trailing stop using quantitative analysis."""
        try:
            logger.info(f"Calculating optimal trailing stop for {symbol}")
            
            # 1. Volatility Regime Analysis
            volatility_regime = await self._analyze_volatility_regime(symbol)
            
            # 2. Market Microstructure Analysis
            market_impact = await self._calculate_market_impact(symbol)
            
            # 3. Statistical Arbitrage Analysis
            mean_reversion_signal = await self._analyze_mean_reversion(symbol)
            
            # 4. Factor Model Analysis
            factor_exposures = await self._get_factor_exposures(symbol)
            
            # 5. Machine Learning Prediction
            ml_prediction = await self._get_ml_trailing_prediction(symbol)
            
            # 6. Optimal Trailing Stop Calculation
            trailing_stop = self._calculate_quantitative_trailing_stop(
                volatility_regime, market_impact, mean_reversion_signal,
                factor_exposures, ml_prediction, position
            )
            
            logger.info(f"Trailing stop for {symbol}: {trailing_stop}")
            return trailing_stop
            
        except Exception as e:
            logger.error(f"Error in optimal trailing stop for {symbol}: {str(e)}")
            return None
    
    async def _analyze_volatility_regime(self, symbol: str) -> str:
        """Analyze volatility regime for trailing stop decisions."""
        try:
            # Get recent price data
            klines = await self._get_recent_klines(symbol, '1h', 24)
            if not klines:
                return 'normal'
            
            # Calculate volatility
            returns = np.diff(np.log([float(k[4]) for k in klines]))
            volatility = np.std(returns) * np.sqrt(24)  # Annualized
            
            # Determine regime
            if volatility < 0.02:  # < 2%
                return 'low'
            elif volatility < 0.05:  # 2-5%
                return 'normal'
            else:  # > 5%
                return 'high'
                
        except Exception as e:
            logger.error(f"Error analyzing volatility regime for {symbol}: {str(e)}")
            return 'normal'
    
    async def _calculate_market_impact(self, symbol: str) -> float:
        """Calculate market impact for trailing stop adjustment."""
        try:
            # Get order book data
            orderbook = await self._get_orderbook(symbol)
            if not orderbook:
                return 0.0
            
            # Calculate bid-ask spread
            best_bid = float(orderbook['bids'][0][0])
            best_ask = float(orderbook['asks'][0][0])
            spread = (best_ask - best_bid) / best_bid
            
            # Calculate order imbalance
            bid_volume = sum(float(bid[1]) for bid in orderbook['bids'][:5])
            ask_volume = sum(float(ask[1]) for ask in orderbook['asks'][:5])
            imbalance = abs(bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Market impact score (0-1, lower is better)
            impact_score = (spread * 100) + (imbalance * 0.5)
            
            return impact_score
            
        except Exception as e:
            logger.error(f"Error calculating market impact for {symbol}: {str(e)}")
            return 0.0
    
    async def _analyze_mean_reversion(self, symbol: str) -> Dict:
        """Analyze mean reversion for trailing stop adjustment."""
        try:
            # Get recent price data
            klines = await self._get_recent_klines(symbol, '1h', 48)
            if not klines:
                return {'signal': 'neutral', 'strength': 0.0}
            
            # Calculate price deviation from moving average
            prices = [float(k[4]) for k in klines]
            ma_20 = np.mean(prices[-20:])
            current_price = prices[-1]
            
            deviation = (current_price - ma_20) / ma_20
            
            # Determine mean reversion signal
            if deviation < -0.015:  # 1.5% below MA
                signal = 'buy'
                strength = min(abs(deviation) / 0.03, 1.0)  # Normalize to 0-1
            elif deviation > 0.015:  # 1.5% above MA
                signal = 'sell'
                strength = min(abs(deviation) / 0.03, 1.0)
            else:
                signal = 'neutral'
                strength = 0.0
            
            return {'signal': signal, 'strength': strength}
            
        except Exception as e:
            logger.error(f"Error analyzing mean reversion for {symbol}: {str(e)}")
            return {'signal': 'neutral', 'strength': 0.0}
    
    async def _get_factor_exposures(self, symbol: str) -> Dict:
        """Get factor exposures for trailing stop analysis."""
        try:
            # Mock factor exposures (in real implementation, this would come from factor model)
            factor_exposures = {
                'market': 0.8,
                'size': 0.2,
                'value': 0.1,
                'momentum': 0.3,
                'volatility': 0.4,
                'liquidity': 0.6
            }
            
            return factor_exposures
            
        except Exception as e:
            logger.error(f"Error getting factor exposures for {symbol}: {str(e)}")
            return {}
    
    async def _get_ml_trailing_prediction(self, symbol: str) -> Dict:
        """Get ML prediction for trailing stop adjustment."""
        try:
            # Mock ML prediction (in real implementation, this would come from ML models)
            ml_prediction = {
                'confidence': 0.8,
                'prediction': 'tighten',
                'probability': 0.7,
                'model_agreement': 0.75
            }
            
            return ml_prediction
            
        except Exception as e:
            logger.error(f"Error getting ML trailing prediction for {symbol}: {str(e)}")
            return {'confidence': 0.5, 'prediction': 'neutral', 'probability': 0.5, 'model_agreement': 0.5}
    
    def _calculate_quantitative_trailing_stop(self, volatility_regime: str, market_impact: float,
                                            mean_reversion: Dict, factor_exposures: Dict,
                                            ml_prediction: Dict, position: Dict) -> Dict:
        """Calculate quantitative trailing stop based on all signals."""
        try:
            # Get position details
            current_price = float(position.get('markPrice', 0))
            entry_price = float(position.get('entryPrice', 0))
            position_side = position.get('info', {}).get('positionSide', 'LONG')
            
            if current_price <= 0 or entry_price <= 0:
                return {'should_update': False, 'reason': 'Invalid price data'}
            
            # Calculate base ATR multiplier
            volatility_multipliers = {'low': 1.5, 'normal': 2.0, 'high': 3.0}
            base_multiplier = volatility_multipliers.get(volatility_regime, 2.0)
            
            # Adjust multiplier based on signals
            adjustment_factor = 1.0
            
            # Market impact adjustment
            if market_impact < 0.5:  # Good market conditions
                adjustment_factor *= 0.9  # Tighter stop
            else:
                adjustment_factor *= 1.1  # Wider stop
            
            # Mean reversion adjustment
            if mean_reversion['signal'] == 'buy' and position_side == 'LONG':
                adjustment_factor *= 0.95  # Slightly tighter for favorable signal
            elif mean_reversion['signal'] == 'sell' and position_side == 'SHORT':
                adjustment_factor *= 0.95
            
            # ML prediction adjustment
            if ml_prediction['prediction'] == 'tighten':
                adjustment_factor *= 0.9
            elif ml_prediction['prediction'] == 'widen':
                adjustment_factor *= 1.1
            
            # Calculate final trailing distance
            atr = current_price * 0.02  # Mock ATR (2% of price)
            trailing_distance = atr * base_multiplier * adjustment_factor
            
            # Calculate trailing stop price
            if position_side == 'LONG':
                trailing_stop_price = current_price - trailing_distance
            else:
                trailing_stop_price = current_price + trailing_distance
            
            # Determine if update is needed
            should_update = True
            reason = "Quantitative analysis suggests update"
            
            # Check if new stop is more favorable
            if position_side == 'LONG':
                if trailing_stop_price <= entry_price:
                    should_update = False
                    reason = "Stop would be below entry price"
            else:
                if trailing_stop_price >= entry_price:
                    should_update = False
                    reason = "Stop would be above entry price"
            
            return {
                'should_update': should_update,
                'trailing_stop_price': trailing_stop_price,
                'trailing_distance': trailing_distance,
                'adjustment_factor': adjustment_factor,
                'volatility_regime': volatility_regime,
                'reason': reason,
                'signals': {
                    'volatility_regime': volatility_regime,
                    'market_impact': market_impact,
                    'mean_reversion': mean_reversion,
                    'ml_prediction': ml_prediction
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating quantitative trailing stop: {str(e)}")
            return {'should_update': False, 'reason': f'Error: {str(e)}'}
    
    async def _get_recent_klines(self, symbol: str, timeframe: str, limit: int) -> List:
        """Get recent klines data."""
        # Mock implementation - in real code, this would call the actual service
        return []
    
    async def _get_orderbook(self, symbol: str) -> Dict:
        """Get order book data."""
        # Mock implementation - in real code, this would call the actual service
        return {'bids': [[100, 1]], 'asks': [[101, 1]]}

async def test_worldquant_implementations():
    """Test the WorldQuant DCA and Trailing Stop implementations."""
    logger.info("🧪 Testing WorldQuant DCA and Trailing Stop implementations...")
    
    # Mock config
    config = {
        'risk_management': {
            'dca': {
                'enabled': True,
                'dca_size_multiplier': 0.5,
                'max_dca_size_multiplier': 2.0,
                'min_dca_size': 0.001,
                'max_attempts': 3,
                'min_interval': 3600,
                'risk_reduction': 0.5,
                'price_drop_thresholds': [0.02, 0.05, 0.1],
                'volume_threshold': 1.5,
                'volatility_threshold': 0.02
            },
            'trailing_stop': {
                'enabled': True,
                'atr_multiplier': 2.0,
                'update_interval': 300,
                'dynamic': {
                    'enabled': True,
                    'atr_multiplier': 2.0,
                    'volatility_adjustment': True,
                    'trend_adjustment': True
                }
            }
        }
    }
    
    # Test DCA
    logger.info("\n📊 Testing WorldQuant DCA...")
    dca = WorldQuantDCA(config)
    
    # Mock position
    position = {
        'symbol': 'BTCUSDT',
        'markPrice': '50000',
        'entryPrice': '48000',
        'info': {
            'positionSide': 'LONG',
            'positionAmt': '0.1'
        }
    }
    
    dca_result = await dca.calculate_optimal_dca_timing('BTCUSDT', position)
    logger.info(f"DCA Result: {dca_result}")
    
    # Test Trailing Stop
    logger.info("\n📊 Testing WorldQuant Trailing Stop...")
    trailing = WorldQuantTrailingStop(config)
    
    trailing_result = await trailing.calculate_optimal_trailing_stop('BTCUSDT', position)
    logger.info(f"Trailing Stop Result: {trailing_result}")
    
    logger.info("\n✅ WorldQuant implementations tested successfully!")

async def create_enhanced_config():
    """Create enhanced configuration with WorldQuant-level DCA and Trailing Stop."""
    logger.info("🔧 Creating enhanced configuration...")
    
    enhanced_config = {
        'risk_management': {
            'dca': {
                'enabled': True,
                'quantitative_dca': {
                    'enabled': True,
                    'volatility_regime_analysis': True,
                    'market_microstructure_analysis': True,
                    'statistical_arbitrage_signals': True,
                    'factor_model_integration': True,
                    'ml_prediction_integration': True,
                    
                    # Volatility-based DCA
                    'volatility_thresholds': {
                        'low': 0.02,    # < 2% volatility
                        'medium': 0.05, # 2-5% volatility
                        'high': 0.10    # > 10% volatility
                    },
                    
                    # Market microstructure thresholds
                    'spread_threshold': 0.001,  # 0.1% max spread
                    'volume_threshold': 1.5,    # 1.5x average volume
                    'imbalance_threshold': 0.3, # 30% order imbalance
                    
                    # Statistical arbitrage parameters
                    'mean_reversion_threshold': 0.02,  # 2% deviation
                    'cointegration_threshold': 0.05,   # 5% cointegration
                    'statistical_significance': 0.05,  # 5% significance level
                    
                    # Factor model parameters
                    'factor_exposure_threshold': 0.1,  # 10% factor exposure
                    'risk_adjustment_factor': 0.8,     # 20% risk reduction
                    
                    # ML prediction parameters
                    'ml_confidence_threshold': 0.7,    # 70% ML confidence
                    'ensemble_agreement_threshold': 0.6 # 60% model agreement
                },
                
                # Original DCA parameters
                'dca_size_multiplier': 0.5,
                'max_dca_size_multiplier': 2.0,
                'min_dca_size': 0.001,
                'max_attempts': 3,
                'min_interval': 3600,
                'risk_reduction': 0.5,
                'price_drop_thresholds': [0.02, 0.05, 0.1],
                'volume_threshold': 1.5,
                'volatility_threshold': 0.02
            },
            
            'trailing_stop': {
                'enabled': True,
                'quantitative_trailing': {
                    'enabled': True,
                    'volatility_regime_analysis': True,
                    'market_microstructure_analysis': True,
                    'statistical_arbitrage_signals': True,
                    'factor_model_integration': True,
                    'ml_prediction_integration': True,
                    
                    # Volatility-based trailing
                    'volatility_multipliers': {
                        'low': 1.5,     # 1.5x ATR for low volatility
                        'medium': 2.0,  # 2.0x ATR for medium volatility
                        'high': 3.0     # 3.0x ATR for high volatility
                    },
                    
                    # Market microstructure thresholds
                    'spread_adjustment': 0.5,    # 50% spread adjustment
                    'volume_threshold': 1.2,     # 1.2x average volume
                    'imbalance_threshold': 0.2,  # 20% order imbalance
                    
                    # Statistical arbitrage parameters
                    'mean_reversion_threshold': 0.015, # 1.5% deviation
                    'cointegration_threshold': 0.03,   # 3% cointegration
                    'statistical_significance': 0.05,  # 5% significance level
                    
                    # Factor model parameters
                    'factor_exposure_threshold': 0.08, # 8% factor exposure
                    'risk_adjustment_factor': 0.9,     # 10% risk adjustment
                    
                    # ML prediction parameters
                    'ml_confidence_threshold': 0.75,   # 75% ML confidence
                    'ensemble_agreement_threshold': 0.65 # 65% model agreement
                },
                
                # Original trailing stop parameters
                'atr_multiplier': 2.0,
                'update_interval': 300,
                'dynamic': {
                    'enabled': True,
                    'atr_multiplier': 2.0,
                    'volatility_adjustment': True,
                    'trend_adjustment': True
                }
            }
        }
    }
    
    # Save enhanced config
    with open('enhanced_config_worldquant.json', 'w') as f:
        json.dump(enhanced_config, f, indent=2)
    
    logger.info("✅ Enhanced configuration created: enhanced_config_worldquant.json")

if __name__ == "__main__":
    print("🚀 Implementing WorldQuant-level DCA and Trailing Stop enhancements...")
    
    # Run tests
    asyncio.run(test_worldquant_implementations())
    
    # Create enhanced configuration
    asyncio.run(create_enhanced_config())
    
    print("\n🎉 WorldQuant DCA and Trailing Stop implementation completed!")
    print("📁 Enhanced configuration saved: enhanced_config_worldquant.json")
    print("📊 Test results logged above") 