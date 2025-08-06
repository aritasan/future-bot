#!/usr/bin/env python3
"""
Market Microstructure Module
WorldQuant Standards Implementation

Implements:
- Order Flow Analysis
- High-Frequency Trading capabilities
- Market Impact Modeling
- Liquidity Analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MarketMicrostructureAnalyzer:
    """
    Advanced Market Microstructure Analysis System following WorldQuant standards.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Market Microstructure Analyzer."""
        self.config = config or {}
        self.order_flow_data = {}
        self.liquidity_metrics = {}
        self.market_impact_models = {}
        self.hft_signals = {}
        
        # Microstructure parameters
        self.order_flow_window = self.config.get('order_flow_window', 100)
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.1)
        self.impact_decay_factor = self.config.get('impact_decay_factor', 0.95)
        
        logger.info("Market Microstructure Analyzer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the market microstructure analyzer."""
        try:
            # Initialize any required components
            self.order_flow_data = {}
            self.liquidity_metrics = {}
            self.market_impact_models = {}
            self.hft_signals = {}
            
            logger.info("Market Microstructure Analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Market Microstructure Analyzer: {str(e)}")
            return False
    
    def analyze_order_flow(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze order flow patterns.
        
        Args:
            orderbook_data: Order book data
            trade_data: Trade data DataFrame
            
        Returns:
            Order flow analysis results
        """
        try:
            order_flow_analysis = {}
            
            # 1. Bid-Ask Imbalance
            order_flow_analysis['bid_ask_imbalance'] = self._calculate_bid_ask_imbalance(orderbook_data)
            
            # 2. Order Flow Toxicity
            order_flow_analysis['order_flow_toxicity'] = self._calculate_order_flow_toxicity(trade_data)
            
            # 3. Order Flow Imbalance
            order_flow_analysis['order_flow_imbalance'] = self._calculate_order_flow_imbalance(trade_data)
            
            # 4. Market Impact
            order_flow_analysis['market_impact'] = self._calculate_market_impact(orderbook_data, trade_data)
            
            # 5. Order Flow Signals
            order_flow_analysis['signals'] = self._generate_order_flow_signals(order_flow_analysis)
            
            return order_flow_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing order flow: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_bid_ask_imbalance(self, orderbook_data: Dict) -> Dict[str, float]:
        """
        Calculate bid-ask imbalance.
        
        Args:
            orderbook_data: Order book data
            
        Returns:
            Bid-ask imbalance metrics
        """
        try:
            imbalance_metrics = {}
            
            # Extract bid and ask data
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if not bids or not asks:
                return {'imbalance': 0.0, 'spread': 0.0, 'depth': 0.0}
            
            # Calculate total bid and ask volume
            total_bid_volume = sum(bid[1] for bid in bids)
            total_ask_volume = sum(ask[1] for ask in asks)
            
            # Calculate imbalance
            total_volume = total_bid_volume + total_ask_volume
            if total_volume > 0:
                imbalance = (total_bid_volume - total_ask_volume) / total_volume
            else:
                imbalance = 0.0
            
            # Calculate spread
            best_bid = max(bid[0] for bid in bids) if bids else 0
            best_ask = min(ask[0] for ask in asks) if asks else 0
            spread = best_ask - best_bid if best_ask > best_bid else 0
            
            # Calculate depth
            depth = min(total_bid_volume, total_ask_volume)
            
            imbalance_metrics = {
                'imbalance': float(imbalance),
                'spread': float(spread),
                'depth': float(depth),
                'total_bid_volume': float(total_bid_volume),
                'total_ask_volume': float(total_ask_volume)
            }
            
            return imbalance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating bid-ask imbalance: {str(e)}")
            return {'imbalance': 0.0, 'spread': 0.0, 'depth': 0.0}
    
    def _calculate_order_flow_toxicity(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate order flow toxicity using VPIN (Volume-synchronized Probability of Informed Trading).
        
        Args:
            trade_data: Trade data DataFrame
            
        Returns:
            Order flow toxicity metrics
        """
        try:
            toxicity_metrics = {}
            
            if trade_data.empty:
                return {'vpin': 0.0, 'toxicity_score': 0.0, 'informed_trading_prob': 0.0}
            
            # Calculate trade direction
            trade_data['direction'] = np.where(trade_data['price'] > trade_data['price'].shift(1), 1,
                                             np.where(trade_data['price'] < trade_data['price'].shift(1), -1, 0))
            
            # Calculate volume-weighted trade direction
            trade_data['vwap'] = (trade_data['price'] * trade_data['volume']).cumsum() / trade_data['volume'].cumsum()
            trade_data['signed_volume'] = trade_data['direction'] * trade_data['volume']
            
            # Calculate VPIN
            total_volume = trade_data['volume'].sum()
            if total_volume > 0:
                # Simplified VPIN calculation
                absolute_signed_volume = abs(trade_data['signed_volume']).sum()
                vpin = absolute_signed_volume / total_volume
            else:
                vpin = 0.0
            
            # Calculate toxicity score
            toxicity_score = min(vpin * 10, 1.0)  # Normalize to [0, 1]
            
            # Estimate probability of informed trading
            informed_trading_prob = toxicity_score * 0.8  # Simplified estimate
            
            toxicity_metrics = {
                'vpin': float(vpin),
                'toxicity_score': float(toxicity_score),
                'informed_trading_prob': float(informed_trading_prob),
                'total_volume': float(total_volume),
                'absolute_signed_volume': float(absolute_signed_volume) if total_volume > 0 else 0.0
            }
            
            return toxicity_metrics
            
        except Exception as e:
            logger.error(f"Error calculating order flow toxicity: {str(e)}")
            return {'vpin': 0.0, 'toxicity_score': 0.0, 'informed_trading_prob': 0.0}
    
    def _calculate_order_flow_imbalance(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate order flow imbalance.
        
        Args:
            trade_data: Trade data DataFrame
            
        Returns:
            Order flow imbalance metrics
        """
        try:
            imbalance_metrics = {}
            
            if trade_data.empty:
                return {'buy_volume': 0.0, 'sell_volume': 0.0, 'imbalance': 0.0, 'imbalance_ratio': 0.0}
            
            # Calculate buy and sell volumes
            buy_trades = trade_data[trade_data['side'] == 'buy'] if 'side' in trade_data.columns else trade_data
            sell_trades = trade_data[trade_data['side'] == 'sell'] if 'side' in trade_data.columns else trade_data
            
            buy_volume = buy_trades['volume'].sum() if not buy_trades.empty else 0
            sell_volume = sell_trades['volume'].sum() if not sell_trades.empty else 0
            
            total_volume = buy_volume + sell_volume
            
            # Calculate imbalance
            if total_volume > 0:
                imbalance = (buy_volume - sell_volume) / total_volume
                imbalance_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            else:
                imbalance = 0.0
                imbalance_ratio = 1.0
            
            imbalance_metrics = {
                'buy_volume': float(buy_volume),
                'sell_volume': float(sell_volume),
                'imbalance': float(imbalance),
                'imbalance_ratio': float(imbalance_ratio) if imbalance_ratio != float('inf') else 1.0,
                'total_volume': float(total_volume)
            }
            
            return imbalance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating order flow imbalance: {str(e)}")
            return {'buy_volume': 0.0, 'sell_volume': 0.0, 'imbalance': 0.0, 'imbalance_ratio': 0.0}
    
    def _calculate_market_impact(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate market impact of trades.
        
        Args:
            orderbook_data: Order book data
            trade_data: Trade data DataFrame
            
        Returns:
            Market impact metrics
        """
        try:
            impact_metrics = {}
            
            if trade_data.empty:
                return {'permanent_impact': 0.0, 'temporary_impact': 0.0, 'total_impact': 0.0}
            
            # Calculate price impact
            trade_data['price_change'] = trade_data['price'].diff()
            trade_data['volume_impact'] = trade_data['volume'] * trade_data['price_change']
            
            # Permanent impact (long-term price change)
            permanent_impact = trade_data['price_change'].rolling(window=20).mean().iloc[-1] if len(trade_data) >= 20 else 0
            
            # Temporary impact (immediate price change)
            temporary_impact = trade_data['price_change'].iloc[-1] if len(trade_data) > 0 else 0
            
            # Total impact
            total_impact = permanent_impact + temporary_impact
            
            # Impact decay
            impact_decay = self.impact_decay_factor
            
            impact_metrics = {
                'permanent_impact': float(permanent_impact),
                'temporary_impact': float(temporary_impact),
                'total_impact': float(total_impact),
                'impact_decay': float(impact_decay),
                'avg_volume_impact': float(trade_data['volume_impact'].mean()) if 'volume_impact' in trade_data.columns else 0.0
            }
            
            return impact_metrics
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {str(e)}")
            return {'permanent_impact': 0.0, 'temporary_impact': 0.0, 'total_impact': 0.0}
    
    def _generate_order_flow_signals(self, order_flow_analysis: Dict) -> Dict[str, Any]:
        """
        Generate trading signals based on order flow analysis.
        
        Args:
            order_flow_analysis: Order flow analysis results
            
        Returns:
            Trading signals
        """
        try:
            signals = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Extract metrics
            imbalance = order_flow_analysis.get('bid_ask_imbalance', {}).get('imbalance', 0.0)
            toxicity = order_flow_analysis.get('order_flow_toxicity', {}).get('toxicity_score', 0.0)
            flow_imbalance = order_flow_analysis.get('order_flow_imbalance', {}).get('imbalance', 0.0)
            market_impact = order_flow_analysis.get('market_impact', {}).get('total_impact', 0.0)
            
            # Generate signals based on order flow patterns
            confidence_factors = []
            
            # Strong buy signal conditions
            if imbalance > 0.2 and flow_imbalance > 0.1 and toxicity < 0.3:
                signals['action'] = 'buy'
                confidence_factors.append(0.3)
                signals['reasoning'].append('Strong bid-ask imbalance favoring buys')
                signals['reasoning'].append('Positive order flow imbalance')
                signals['reasoning'].append('Low order flow toxicity')
            
            # Strong sell signal conditions
            elif imbalance < -0.2 and flow_imbalance < -0.1 and toxicity < 0.3:
                signals['action'] = 'sell'
                confidence_factors.append(0.3)
                signals['reasoning'].append('Strong bid-ask imbalance favoring sells')
                signals['reasoning'].append('Negative order flow imbalance')
                signals['reasoning'].append('Low order flow toxicity')
            
            # Moderate signals
            elif imbalance > 0.1 and flow_imbalance > 0.05:
                signals['action'] = 'buy'
                confidence_factors.append(0.2)
                signals['reasoning'].append('Moderate positive order flow')
            
            elif imbalance < -0.1 and flow_imbalance < -0.05:
                signals['action'] = 'sell'
                confidence_factors.append(0.2)
                signals['reasoning'].append('Moderate negative order flow')
            
            # High toxicity warning
            if toxicity > 0.7:
                signals['reasoning'].append('High order flow toxicity - exercise caution')
                confidence_factors = [c * 0.5 for c in confidence_factors]  # Reduce confidence
            
            # Calculate final confidence
            if confidence_factors:
                signals['confidence'] = min(sum(confidence_factors), 1.0)
            else:
                signals['reasoning'].append('No clear order flow signal')
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating order flow signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def analyze_liquidity(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market liquidity.
        
        Args:
            orderbook_data: Order book data
            trade_data: Trade data DataFrame
            
        Returns:
            Liquidity analysis results
        """
        try:
            liquidity_analysis = {}
            
            # 1. Bid-Ask Spread Analysis
            liquidity_analysis['spread_analysis'] = self._analyze_spread(orderbook_data)
            
            # 2. Market Depth Analysis
            liquidity_analysis['depth_analysis'] = self._analyze_market_depth(orderbook_data)
            
            # 3. Liquidity Crisis Detection
            liquidity_analysis['crisis_detection'] = self._detect_liquidity_crisis(orderbook_data, trade_data)
            
            # 4. Liquidity Metrics
            liquidity_analysis['metrics'] = self._calculate_liquidity_metrics(orderbook_data, trade_data)
            
            return liquidity_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_spread(self, orderbook_data: Dict) -> Dict[str, float]:
        """
        Analyze bid-ask spread patterns.
        """
        try:
            spread_analysis = {}
            
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if not bids or not asks:
                return {'spread': 0.0, 'spread_ratio': 0.0, 'mid_price': 0.0}
            
            # Calculate spread
            best_bid = max(bid[0] for bid in bids) if bids else 0
            best_ask = min(ask[0] for ask in asks) if asks else 0
            spread = best_ask - best_bid if best_ask > best_bid else 0
            
            # Calculate mid price
            mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
            
            # Calculate spread ratio
            spread_ratio = spread / mid_price if mid_price > 0 else 0
            
            spread_analysis = {
                'spread': float(spread),
                'spread_ratio': float(spread_ratio),
                'mid_price': float(mid_price),
                'best_bid': float(best_bid),
                'best_ask': float(best_ask)
            }
            
            return spread_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing spread: {str(e)}")
            return {'spread': 0.0, 'spread_ratio': 0.0, 'mid_price': 0.0}
    
    def _analyze_market_depth(self, orderbook_data: Dict) -> Dict[str, float]:
        """
        Analyze market depth at different price levels.
        """
        try:
            depth_analysis = {}
            
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            # Calculate depth at different levels
            bid_depth_1 = sum(bid[1] for bid in bids[:5]) if len(bids) >= 5 else sum(bid[1] for bid in bids)
            ask_depth_1 = sum(ask[1] for ask in asks[:5]) if len(asks) >= 5 else sum(ask[1] for ask in asks)
            
            bid_depth_2 = sum(bid[1] for bid in bids[:10]) if len(bids) >= 10 else bid_depth_1
            ask_depth_2 = sum(ask[1] for ask in asks[:10]) if len(asks) >= 10 else ask_depth_1
            
            total_depth = bid_depth_1 + ask_depth_1
            
            depth_analysis = {
                'bid_depth_1': float(bid_depth_1),
                'ask_depth_1': float(ask_depth_1),
                'bid_depth_2': float(bid_depth_2),
                'ask_depth_2': float(ask_depth_2),
                'total_depth': float(total_depth),
                'depth_imbalance': float((bid_depth_1 - ask_depth_1) / total_depth) if total_depth > 0 else 0.0
            }
            
            return depth_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market depth: {str(e)}")
            return {'bid_depth_1': 0.0, 'ask_depth_1': 0.0, 'total_depth': 0.0, 'depth_imbalance': 0.0}
    
    def _detect_liquidity_crisis(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect liquidity crisis conditions.
        """
        try:
            crisis_analysis = {
                'crisis_detected': False,
                'crisis_score': 0.0,
                'warning_signals': []
            }
            
            # Calculate crisis indicators
            spread_analysis = self._analyze_spread(orderbook_data)
            depth_analysis = self._analyze_market_depth(orderbook_data)
            
            crisis_score = 0.0
            warning_signals = []
            
            # High spread indicator
            if spread_analysis['spread_ratio'] > 0.01:  # 1% spread
                crisis_score += 0.3
                warning_signals.append('High bid-ask spread')
            
            # Low depth indicator
            if depth_analysis['total_depth'] < self.liquidity_threshold:
                crisis_score += 0.3
                warning_signals.append('Low market depth')
            
            # Volume spike indicator
            if not trade_data.empty:
                recent_volume = trade_data['volume'].tail(10).mean()
                avg_volume = trade_data['volume'].mean()
                if recent_volume > avg_volume * 3:  # 3x average volume
                    crisis_score += 0.2
                    warning_signals.append('Volume spike detected')
            
            # Price volatility indicator
            if not trade_data.empty:
                price_volatility = trade_data['price'].pct_change().std()
                if price_volatility > 0.05:  # 5% volatility
                    crisis_score += 0.2
                    warning_signals.append('High price volatility')
            
            crisis_analysis['crisis_score'] = min(crisis_score, 1.0)
            crisis_analysis['crisis_detected'] = crisis_score > 0.5
            crisis_analysis['warning_signals'] = warning_signals
            
            return crisis_analysis
            
        except Exception as e:
            logger.error(f"Error detecting liquidity crisis: {str(e)}")
            return {'crisis_detected': False, 'crisis_score': 0.0, 'warning_signals': []}
    
    def _calculate_liquidity_metrics(self, orderbook_data: Dict, trade_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive liquidity metrics.
        """
        try:
            metrics = {}
            
            # Extract basic metrics
            spread_analysis = self._analyze_spread(orderbook_data)
            depth_analysis = self._analyze_market_depth(orderbook_data)
            
            # Amihud illiquidity ratio
            if not trade_data.empty:
                returns = trade_data['price'].pct_change().abs()
                volume = trade_data['volume']
                amihud_ratio = (returns / volume).mean() if volume.sum() > 0 else 0
            else:
                amihud_ratio = 0
            
            # Kyle's lambda (price impact)
            if not trade_data.empty and len(trade_data) > 1:
                signed_volume = trade_data['volume'] * np.sign(trade_data['price'].diff())
                price_changes = trade_data['price'].diff()
                
                # Simple linear regression
                if len(signed_volume) > 1:
                    correlation = np.corrcoef(signed_volume.dropna(), price_changes.dropna())[0, 1]
                    kyle_lambda = abs(correlation) if not np.isnan(correlation) else 0
                else:
                    kyle_lambda = 0
            else:
                kyle_lambda = 0
            
            metrics = {
                'spread': spread_analysis['spread'],
                'spread_ratio': spread_analysis['spread_ratio'],
                'total_depth': depth_analysis['total_depth'],
                'depth_imbalance': depth_analysis['depth_imbalance'],
                'amihud_ratio': float(amihud_ratio),
                'kyle_lambda': float(kyle_lambda),
                'liquidity_score': self._calculate_liquidity_score(spread_analysis, depth_analysis)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {str(e)}")
            return {'spread': 0.0, 'total_depth': 0.0, 'liquidity_score': 0.0}
    
    def _calculate_liquidity_score(self, spread_analysis: Dict, depth_analysis: Dict) -> float:
        """
        Calculate overall liquidity score.
        """
        try:
            # Normalize metrics
            spread_score = max(0, 1 - spread_analysis['spread_ratio'] * 100)  # Lower spread = higher score
            depth_score = min(1, depth_analysis['total_depth'] / 1000)  # Normalize depth
            
            # Combined liquidity score
            liquidity_score = (spread_score * 0.6 + depth_score * 0.4)
            
            return float(liquidity_score)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.0
    
    def generate_hft_signals(self, market_data: Dict) -> Dict[str, Any]:
        """
        Generate High-Frequency Trading signals.
        
        Args:
            market_data: Market data including orderbook and trades
            
        Returns:
            HFT signals
        """
        try:
            hft_signals = {}
            
            # 1. Latency Arbitrage Signals
            hft_signals['latency_arbitrage'] = self._generate_latency_arbitrage_signals(market_data)
            
            # 2. Market Making Signals
            hft_signals['market_making'] = self._generate_market_making_signals(market_data)
            
            # 3. Statistical Arbitrage Signals
            hft_signals['statistical_arbitrage'] = self._generate_statistical_arbitrage_signals(market_data)
            
            # 4. Microsecond-level signals
            hft_signals['microsecond_signals'] = self._generate_microsecond_signals(market_data)
            
            return hft_signals
            
        except Exception as e:
            logger.error(f"Error generating HFT signals: {str(e)}")
            return {'error': str(e)}
    
    def _generate_latency_arbitrage_signals(self, market_data: Dict) -> Dict[str, Any]:
        """
        Generate latency arbitrage signals.
        """
        try:
            signals = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Placeholder for latency arbitrage logic
            # In production, this would analyze cross-exchange price differences
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating latency arbitrage signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def _generate_market_making_signals(self, market_data: Dict) -> Dict[str, Any]:
        """
        Generate market making signals.
        """
        try:
            signals = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Placeholder for market making logic
            # In production, this would analyze spread and depth for market making opportunities
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating market making signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def _generate_statistical_arbitrage_signals(self, market_data: Dict) -> Dict[str, Any]:
        """
        Generate statistical arbitrage signals at microsecond level.
        """
        try:
            signals = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Placeholder for statistical arbitrage logic
            # In production, this would analyze short-term price patterns
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating statistical arbitrage signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def _generate_microsecond_signals(self, market_data: Dict) -> Dict[str, Any]:
        """
        Generate microsecond-level trading signals.
        """
        try:
            signals = {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': []
            }
            
            # Placeholder for microsecond signal logic
            # In production, this would analyze ultra-high-frequency patterns
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating microsecond signals: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0, 'reasoning': ['Error in signal generation']}
    
    def get_microstructure_summary(self) -> Dict[str, Any]:
        """Get comprehensive microstructure analysis summary."""
        try:
            summary = {
                'microstructure_analyzer_status': 'active',
                'order_flow_analyses': len(self.order_flow_data),
                'liquidity_analyses': len(self.liquidity_metrics),
                'market_impact_models': len(self.market_impact_models),
                'hft_signals_generated': len(self.hft_signals),
                'total_analyses': (
                    len(self.order_flow_data) +
                    len(self.liquidity_metrics) +
                    len(self.market_impact_models) +
                    len(self.hft_signals)
                )
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting microstructure summary: {str(e)}")
            return {'error': str(e)}
