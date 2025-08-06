#!/usr/bin/env python3
"""
Advanced Market Microstructure Module
WorldQuant Standards Implementation - Phase 3

Implements:
- Advanced order flow analysis
- Market impact modeling
- Liquidity crisis detection
- High-frequency microstructure patterns
- Order book dynamics
- Market microstructure signals
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OrderFlowMetrics:
    """Advanced order flow metrics."""
    vpin: float  # Volume-synchronized Probability of Informed Trading
    toxicity_score: float
    order_imbalance: float
    market_impact: float
    informed_trading_prob: float

@dataclass
class MarketImpactModel:
    """Market impact model parameters."""
    permanent_impact: float
    temporary_impact: float
    decay_factor: float
    impact_elasticity: float

class AdvancedMarketMicrostructureAnalyzer:
    """
    WorldQuant-Level Advanced Market Microstructure Analyzer.
    
    Features:
    - Advanced order flow analysis
    - Market impact modeling
    - Liquidity crisis detection
    - High-frequency microstructure patterns
    - Order book dynamics analysis
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Advanced Market Microstructure Analyzer."""
        self.config = config or {}
        
        # Advanced microstructure parameters
        self.vpin_buckets = self.config.get('vpin_buckets', 50)
        self.toxicity_threshold = self.config.get('toxicity_threshold', 0.7)
        self.impact_decay_factor = self.config.get('impact_decay_factor', 0.95)
        self.liquidity_crisis_threshold = self.config.get('liquidity_crisis_threshold', 0.8)
        
        # Data structures
        self.order_flow_history = deque(maxlen=10000)
        self.market_impact_models = {}
        self.liquidity_metrics = {}
        self.microstructure_signals = {}
        
        logger.info("Advanced Market Microstructure Analyzer initialized")
    
    def analyze_advanced_order_flow(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform advanced order flow analysis.
        
        Args:
            orderbook_data: Order book data
            trade_data: Trade data DataFrame
            
        Returns:
            Advanced order flow analysis
        """
        try:
            analysis = {}
            
            # 1. VPIN Analysis
            analysis['vpin_analysis'] = self._calculate_vpin(trade_data)
            
            # 2. Order Flow Toxicity
            analysis['toxicity_analysis'] = self._analyze_order_flow_toxicity(trade_data)
            
            # 3. Market Impact Modeling
            analysis['market_impact'] = self._model_market_impact(orderbook_data, trade_data)
            
            # 4. Liquidity Analysis
            analysis['liquidity_analysis'] = self._analyze_liquidity(orderbook_data, trade_data)
            
            # 5. Order Book Dynamics
            analysis['order_book_dynamics'] = self._analyze_order_book_dynamics(orderbook_data)
            
            # 6. Microstructure Signals
            analysis['microstructure_signals'] = self._generate_microstructure_signals(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in advanced order flow analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_vpin(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate VPIN (Volume-synchronized Probability of Informed Trading)."""
        try:
            if trade_data.empty or len(trade_data) < self.vpin_buckets:
                return {'vpin': 0.0, 'vpin_level': 'low', 'confidence': 0.0}
            
            # Calculate trade direction
            trade_data['direction'] = np.where(trade_data['price'] > trade_data['price'].shift(1), 1,
                                             np.where(trade_data['price'] < trade_data['price'].shift(1), -1, 0))
            
            # Calculate signed volume
            trade_data['signed_volume'] = trade_data['direction'] * trade_data['volume']
            
            # Calculate VPIN
            total_volume = trade_data['volume'].sum()
            absolute_signed_volume = abs(trade_data['signed_volume']).sum()
            
            if total_volume > 0:
                vpin = absolute_signed_volume / total_volume
            else:
                vpin = 0.0
            
            # Determine VPIN level
            if vpin > 0.7:
                vpin_level = 'high'
            elif vpin > 0.4:
                vpin_level = 'medium'
            else:
                vpin_level = 'low'
            
            # Calculate confidence
            confidence = min(vpin * 1.2, 1.0)
            
            return {
                'vpin': float(vpin),
                'vpin_level': vpin_level,
                'confidence': float(confidence),
                'total_volume': float(total_volume),
                'absolute_signed_volume': float(absolute_signed_volume)
            }
            
        except Exception as e:
            logger.error(f"Error calculating VPIN: {str(e)}")
            return {'vpin': 0.0, 'vpin_level': 'low', 'confidence': 0.0}
    
    def _analyze_order_flow_toxicity(self, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order flow toxicity using advanced metrics."""
        try:
            if trade_data.empty:
                return {'toxicity_score': 0.0, 'informed_trading_prob': 0.0, 'toxicity_level': 'low'}
            
            # Calculate trade direction
            trade_data['direction'] = np.where(trade_data['price'] > trade_data['price'].shift(1), 1,
                                             np.where(trade_data['price'] < trade_data['price'].shift(1), -1, 0))
            
            # Calculate volume-weighted metrics
            trade_data['signed_volume'] = trade_data['direction'] * trade_data['volume']
            trade_data['volume_imbalance'] = abs(trade_data['signed_volume'])
            
            # Calculate toxicity score
            total_volume = trade_data['volume'].sum()
            absolute_signed_volume = trade_data['volume_imbalance'].sum()
            
            if total_volume > 0:
                toxicity_score = absolute_signed_volume / total_volume
            else:
                toxicity_score = 0.0
            
            # Determine toxicity level
            if toxicity_score > self.toxicity_threshold:
                toxicity_level = 'high'
            elif toxicity_score > self.toxicity_threshold * 0.5:
                toxicity_level = 'medium'
            else:
                toxicity_level = 'low'
            
            # Estimate informed trading probability
            informed_trading_prob = toxicity_score * 0.8
            
            return {
                'toxicity_score': float(toxicity_score),
                'informed_trading_prob': float(informed_trading_prob),
                'toxicity_level': toxicity_level,
                'total_volume': float(total_volume),
                'absolute_signed_volume': float(absolute_signed_volume)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order flow toxicity: {str(e)}")
            return {'toxicity_score': 0.0, 'informed_trading_prob': 0.0, 'toxicity_level': 'low'}
    
    def _model_market_impact(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Model market impact of trades."""
        try:
            if trade_data.empty:
                return {'permanent_impact': 0.0, 'temporary_impact': 0.0, 'total_impact': 0.0}
            
            # Calculate price changes
            trade_data['price_change'] = trade_data['price'].diff()
            trade_data['volume_impact'] = trade_data['volume'] * trade_data['price_change']
            
            # Permanent impact (long-term price change)
            if len(trade_data) >= 20:
                permanent_impact = trade_data['price_change'].rolling(window=20).mean().iloc[-1]
            else:
                permanent_impact = trade_data['price_change'].mean() if len(trade_data) > 0 else 0
            
            # Temporary impact (immediate price change)
            temporary_impact = trade_data['price_change'].iloc[-1] if len(trade_data) > 0 else 0
            
            # Total impact
            total_impact = permanent_impact + temporary_impact
            
            return {
                'permanent_impact': float(permanent_impact),
                'temporary_impact': float(temporary_impact),
                'total_impact': float(total_impact),
                'impact_decay': float(self.impact_decay_factor),
                'avg_volume_impact': float(trade_data['volume_impact'].mean()) if 'volume_impact' in trade_data.columns else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error modeling market impact: {str(e)}")
            return {'permanent_impact': 0.0, 'temporary_impact': 0.0, 'total_impact': 0.0}
    
    def _analyze_liquidity(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market liquidity."""
        try:
            # Calculate basic liquidity metrics
            spread_analysis = self._analyze_spread(orderbook_data)
            depth_analysis = self._analyze_market_depth(orderbook_data)
            
            # Calculate Amihud illiquidity ratio
            if not trade_data.empty:
                returns = trade_data['price'].pct_change().abs()
                volume = trade_data['volume']
                amihud_ratio = (returns / volume).mean() if volume.sum() > 0 else 0
            else:
                amihud_ratio = 0
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(spread_analysis, depth_analysis, amihud_ratio)
            
            # Detect liquidity crisis
            crisis_detection = self._detect_liquidity_crisis(spread_analysis, depth_analysis, trade_data)
            
            return {
                'spread': spread_analysis['spread'],
                'spread_ratio': spread_analysis['spread_ratio'],
                'total_depth': depth_analysis['total_depth'],
                'depth_imbalance': depth_analysis['depth_imbalance'],
                'amihud_ratio': float(amihud_ratio),
                'liquidity_score': float(liquidity_score),
                'crisis_detected': crisis_detection['crisis_detected'],
                'crisis_score': float(crisis_detection['crisis_score'])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {'liquidity_score': 0.0, 'crisis_detected': False}
    
    def _analyze_spread(self, orderbook_data: Dict) -> Dict[str, float]:
        """Analyze bid-ask spread."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if not bids or not asks:
                return {'spread': 0.0, 'spread_ratio': 0.0, 'mid_price': 0.0}
            
            best_bid = max(bid[0] for bid in bids)
            best_ask = min(ask[0] for ask in asks)
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_ratio = spread / mid_price if mid_price > 0 else 0
            
            return {
                'spread': float(spread),
                'spread_ratio': float(spread_ratio),
                'mid_price': float(mid_price)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spread: {str(e)}")
            return {'spread': 0.0, 'spread_ratio': 0.0, 'mid_price': 0.0}
    
    def _analyze_market_depth(self, orderbook_data: Dict) -> Dict[str, float]:
        """Analyze market depth."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            bid_depth = sum(bid[1] for bid in bids[:10]) if len(bids) >= 10 else sum(bid[1] for bid in bids)
            ask_depth = sum(ask[1] for ask in asks[:10]) if len(asks) >= 10 else sum(ask[1] for ask in asks)
            total_depth = bid_depth + ask_depth
            
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            return {
                'bid_depth': float(bid_depth),
                'ask_depth': float(ask_depth),
                'total_depth': float(total_depth),
                'depth_imbalance': float(depth_imbalance)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market depth: {str(e)}")
            return {'bid_depth': 0.0, 'ask_depth': 0.0, 'total_depth': 0.0, 'depth_imbalance': 0.0}
    
    def _calculate_liquidity_score(self, spread_analysis: Dict, depth_analysis: Dict, amihud_ratio: float) -> float:
        """Calculate overall liquidity score."""
        try:
            # Normalize metrics
            spread_score = max(0, 1 - spread_analysis['spread_ratio'] * 100)
            depth_score = min(1, depth_analysis['total_depth'] / 1000)
            amihud_score = max(0, 1 - amihud_ratio * 100)
            
            # Combined liquidity score
            liquidity_score = (spread_score * 0.4 + depth_score * 0.4 + amihud_score * 0.2)
            
            return float(liquidity_score)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.0
    
    def _detect_liquidity_crisis(self, spread_analysis: Dict, depth_analysis: Dict, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect liquidity crisis conditions."""
        try:
            crisis_score = 0.0
            warning_signals = []
            
            # High spread indicator
            if spread_analysis['spread_ratio'] > 0.01:
                crisis_score += 0.3
                warning_signals.append('High bid-ask spread')
            
            # Low depth indicator
            if depth_analysis['total_depth'] < 100:
                crisis_score += 0.3
                warning_signals.append('Low market depth')
            
            # Volume spike indicator
            if not trade_data.empty:
                recent_volume = trade_data['volume'].tail(10).mean()
                avg_volume = trade_data['volume'].mean()
                if recent_volume > avg_volume * 3:
                    crisis_score += 0.2
                    warning_signals.append('Volume spike detected')
            
            # Price volatility indicator
            if not trade_data.empty:
                price_volatility = trade_data['price'].pct_change().std()
                if price_volatility > 0.05:
                    crisis_score += 0.2
                    warning_signals.append('High price volatility')
            
            crisis_detected = crisis_score > self.liquidity_crisis_threshold
            
            return {
                'crisis_detected': crisis_detected,
                'crisis_score': float(crisis_score),
                'warning_signals': warning_signals
            }
            
        except Exception as e:
            logger.error(f"Error detecting liquidity crisis: {str(e)}")
            return {'crisis_detected': False, 'crisis_score': 0.0, 'warning_signals': []}
    
    def _analyze_order_book_dynamics(self, orderbook_data: Dict) -> Dict[str, Any]:
        """Analyze order book dynamics."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if not bids or not asks:
                return {'depth_imbalance': 0.0, 'spread_dynamics': 0.0, 'order_flow_pressure': 0.0}
            
            # Calculate depth imbalance
            bid_depth = sum(bid[1] for bid in bids[:10])
            ask_depth = sum(ask[1] for ask in asks[:10])
            total_depth = bid_depth + ask_depth
            
            if total_depth > 0:
                depth_imbalance = (bid_depth - ask_depth) / total_depth
            else:
                depth_imbalance = 0.0
            
            # Calculate spread dynamics
            best_bid = max(bid[0] for bid in bids)
            best_ask = min(ask[0] for ask in asks)
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_ratio = spread / mid_price if mid_price > 0 else 0
            
            # Calculate order flow pressure
            bid_pressure = sum(bid[1] for bid in bids[:5])
            ask_pressure = sum(ask[1] for ask in asks[:5])
            
            if bid_pressure + ask_pressure > 0:
                flow_pressure = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure)
            else:
                flow_pressure = 0.0
            
            return {
                'depth_imbalance': float(depth_imbalance),
                'spread_dynamics': float(spread_ratio),
                'order_flow_pressure': float(flow_pressure),
                'bid_depth': float(bid_depth),
                'ask_depth': float(ask_depth),
                'spread': float(spread),
                'mid_price': float(mid_price)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book dynamics: {str(e)}")
            return {'depth_imbalance': 0.0, 'spread_dynamics': 0.0, 'order_flow_pressure': 0.0}
    
    def _generate_microstructure_signals(self, analysis: Dict) -> Dict[str, Any]:
        """Generate microstructure-based trading signals."""
        try:
            signals = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': [],
                'risk_level': 'low'
            }
            
            # Extract key metrics
            vpin = analysis.get('vpin_analysis', {}).get('vpin', 0.0)
            toxicity = analysis.get('toxicity_analysis', {}).get('toxicity_score', 0.0)
            liquidity = analysis.get('liquidity_analysis', {}).get('liquidity_score', 0.0)
            dynamics = analysis.get('order_book_dynamics', {})
            
            depth_imbalance = dynamics.get('depth_imbalance', 0.0)
            flow_pressure = dynamics.get('order_flow_pressure', 0.0)
            
            confidence_factors = []
            reasoning = []
            
            # High toxicity warning
            if toxicity > self.toxicity_threshold:
                signals['risk_level'] = 'high'
                reasoning.append(f'High order flow toxicity: {toxicity:.3f}')
            
            # Strong buy signals
            if (depth_imbalance > 0.2 and flow_pressure > 0.1 and 
                toxicity < self.toxicity_threshold * 0.5):
                signals['action'] = 'buy'
                confidence_factors.append(0.4)
                reasoning.append('Strong bid depth imbalance')
                reasoning.append('Positive order flow pressure')
                reasoning.append('Low toxicity environment')
            
            # Strong sell signals
            elif (depth_imbalance < -0.2 and flow_pressure < -0.1 and 
                  toxicity < self.toxicity_threshold * 0.5):
                signals['action'] = 'sell'
                confidence_factors.append(0.4)
                reasoning.append('Strong ask depth imbalance')
                reasoning.append('Negative order flow pressure')
                reasoning.append('Low toxicity environment')
            
            # Moderate signals
            elif depth_imbalance > 0.1 and flow_pressure > 0.05:
                signals['action'] = 'buy'
                confidence_factors.append(0.2)
                reasoning.append('Moderate positive microstructure')
            
            elif depth_imbalance < -0.1 and flow_pressure < -0.05:
                signals['action'] = 'sell'
                confidence_factors.append(0.2)
                reasoning.append('Moderate negative microstructure')
            
            # Liquidity consideration
            if liquidity < 0.3:
                reasoning.append('Low liquidity - exercise caution')
                confidence_factors = [c * 0.7 for c in confidence_factors]
            
            # Calculate final confidence
            if confidence_factors:
                signals['confidence'] = min(sum(confidence_factors), 1.0)
            else:
                reasoning.append('No clear microstructure signal')
            
            signals['reasoning'] = reasoning
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating microstructure signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def get_advanced_microstructure_summary(self) -> Dict[str, Any]:
        """Get comprehensive advanced microstructure analysis summary."""
        try:
            summary = {
                'analyzer_status': 'active',
                'total_analyses': len(self.order_flow_history),
                'vpin_analyses': len([a for a in self.order_flow_history if 'vpin' in a]),
                'toxicity_analyses': len([a for a in self.order_flow_history if 'toxicity' in a]),
                'impact_models': len(self.market_impact_models),
                'liquidity_metrics': len(self.liquidity_metrics),
                'microstructure_signals': len(self.microstructure_signals),
                'performance_metrics': self._calculate_microstructure_performance()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting microstructure summary: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_microstructure_performance(self) -> Dict[str, float]:
        """Calculate microstructure analysis performance metrics."""
        try:
            if not self.order_flow_history:
                return {'accuracy': 0.0, 'latency_ms': 0.0, 'coverage': 0.0}
            
            # Calculate performance metrics
            total_analyses = len(self.order_flow_history)
            successful_analyses = len([a for a in self.order_flow_history if 'error' not in a])
            
            accuracy = successful_analyses / total_analyses if total_analyses > 0 else 0
            latency_ms = 0.5  # Simulated average latency
            coverage = min(1.0, total_analyses / 1000)  # Coverage based on analysis count
            
            return {
                'accuracy': float(accuracy),
                'latency_ms': float(latency_ms),
                'coverage': float(coverage)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'accuracy': 0.0, 'latency_ms': 0.0, 'coverage': 0.0} 