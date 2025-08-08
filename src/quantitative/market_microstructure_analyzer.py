"""
WorldQuant-Level Market Microstructure Analyzer
Advanced quantitative analysis with institutional-grade algorithms
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength enumeration for quantitative analysis."""
    VERY_WEAK = 0.1
    WEAK = 0.3
    NEUTRAL = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    CALM = "CALM"

@dataclass
class MicrostructureMetrics:
    """Advanced microstructure metrics container."""
    order_flow_imbalance: float
    liquidity_score: float
    volume_profile_score: float
    market_impact_score: float
    spread_score: float
    depth_score: float
    overall_score: float
    confidence_level: float
    regime: MarketRegime
    timestamp: datetime

class WorldQuantMarketMicrostructureAnalyzer:
    """WorldQuant-level market microstructure analyzer with advanced quantitative algorithms."""
    
    def __init__(self, binance_service, config: Dict):
        self.binance_service = binance_service
        self.config = config
        self.order_flow_window = config.get('order_flow_window', 200)
        self.market_impact_threshold = config.get('market_impact_threshold', 0.03)
        self.liquidity_threshold = config.get('liquidity_threshold', 0.001)
        self.volume_profile_periods = config.get('volume_profile_periods', 168)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}
        self.cache_ttl = 30  # seconds
        logger.info("WorldQuant Market Microstructure Analyzer initialized with advanced features")
    
    async def analyze_market_microstructure(self, symbol: str) -> Dict:
        """Perform comprehensive market microstructure analysis with advanced quantitative algorithms."""
        try:
            # Check cache first
            cache_key = f"microstructure_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Parallel execution of analysis components
            tasks = [
                self._analyze_order_flow_advanced(symbol),
                self._analyze_liquidity_levels_advanced(symbol),
                self._analyze_volume_profile_advanced(symbol),
                self._analyze_market_impact_advanced(symbol),
                self._analyze_bid_ask_spread_advanced(symbol),
                self._analyze_order_book_depth_advanced(symbol),
                self._analyze_market_regime(symbol)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            analysis_components = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error in analysis component {i}: {result}")
                    analysis_components[f"component_{i}"] = self._get_default_analysis()
                else:
                    analysis_components[f"component_{i}"] = result
            
            # Advanced microstructure analysis
            microstructure_analysis = await self._perform_advanced_analysis(symbol, analysis_components)
            
            # Cache the results
            self._cache_result(cache_key, microstructure_analysis)
            
            logger.info(f"Advanced market microstructure analysis completed for {symbol}")
            return microstructure_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market microstructure for {symbol}: {str(e)}")
            return self._get_default_microstructure_analysis(symbol)
    
    async def _analyze_order_flow_advanced(self, symbol: str) -> Dict:
        """Advanced order flow analysis with institutional-grade algorithms."""
        try:
            trades = await self.binance_service.get_recent_trades(symbol, limit=self.order_flow_window)
            
            if not trades or len(trades) < 10:
                return self._get_default_order_flow_analysis()
            
            # Advanced order flow analysis
            trade_data = []
            for trade in trades:
                price = float(trade['price'])
                quantity = float(trade['qty'])
                is_buyer_maker = trade['isBuyerMaker']
                timestamp = trade.get('time', 0)
                
                trade_data.append({
                    'price': price,
                    'quantity': quantity,
                    'is_buy': not is_buyer_maker,
                    'timestamp': timestamp
                })
            
            # Calculate advanced metrics
            buy_volume = sum(t['quantity'] for t in trade_data if t['is_buy'])
            sell_volume = sum(t['quantity'] for t in trade_data if not t['is_buy'])
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return self._get_default_order_flow_analysis()
            
            # Volume imbalance with confidence weighting
            volume_imbalance = (buy_volume - sell_volume) / total_volume
            volume_confidence = min(abs(volume_imbalance) * 2, 1.0)
            
            # Price momentum with exponential weighting
            prices = [t['price'] for t in trade_data]
            if len(prices) > 1:
                weights = np.exp(np.linspace(0, 1, len(prices)))
                weighted_prices = np.average(prices, weights=weights)
                price_momentum = (weighted_prices - prices[0]) / prices[0]
            else:
                price_momentum = 0
            
            # Order flow signal strength
            signal_strength = self._calculate_signal_strength(volume_imbalance, price_momentum)
            
            # Market microstructure patterns
            patterns = self._identify_order_flow_patterns(trade_data)
            
            return {
                'volume_imbalance': volume_imbalance,
                'volume_confidence': volume_confidence,
                'price_momentum': price_momentum,
                'signal_strength': signal_strength.value,
                'patterns': patterns,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'total_volume': total_volume,
                'trade_count': len(trade_data)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced order flow analysis for {symbol}: {str(e)}")
            return self._get_default_order_flow_analysis()
    
    async def _analyze_liquidity_levels_advanced(self, symbol: str) -> Dict:
        """Advanced liquidity analysis with institutional-grade depth analysis."""
        try:
            order_book = await self.binance_service.get_order_book(symbol)
            
            if not order_book:
                return self._get_default_liquidity_analysis()
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            if not bids or not asks:
                return self._get_default_liquidity_analysis()
            
            # Advanced liquidity analysis
            bid_liquidity_levels = self._calculate_liquidity_levels(bids, 'bid')
            ask_liquidity_levels = self._calculate_liquidity_levels(asks, 'ask')
            
            # Liquidity concentration analysis
            bid_concentration = self._calculate_liquidity_concentration(bid_liquidity_levels)
            ask_concentration = self._calculate_liquidity_concentration(ask_liquidity_levels)
            
            # Spread analysis
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_percentage = spread / best_bid
            
            # Liquidity score calculation
            liquidity_score = self._calculate_liquidity_score(
                bid_liquidity_levels, ask_liquidity_levels, spread_percentage
            )
            
            return {
                'bid_liquidity_levels': bid_liquidity_levels,
                'ask_liquidity_levels': ask_liquidity_levels,
                'bid_concentration': bid_concentration,
                'ask_concentration': ask_concentration,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'liquidity_score': liquidity_score,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': (best_bid + best_ask) / 2
            }
            
        except Exception as e:
            logger.error(f"Error in advanced liquidity analysis for {symbol}: {str(e)}")
            return self._get_default_liquidity_analysis()
    
    async def _analyze_volume_profile_advanced(self, symbol: str) -> Dict:
        """Advanced volume profile analysis with institutional-grade algorithms."""
        try:
            klines = await self.binance_service.get_klines(symbol, '1h', limit=self.volume_profile_periods)
            
            if not klines or len(klines) < 24:
                return self._get_default_volume_profile_analysis()
            
            # Advanced volume profile analysis
            volume_data = []
            for kline in klines:
                high = float(kline[2])
                low = float(kline[3])
                close = float(kline[4])
                volume = float(kline[5])
                
                volume_data.append({
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'price_range': high - low
                })
            
            # Volume-weighted average price (VWAP)
            total_volume = sum(d['volume'] for d in volume_data)
            vwap = sum(d['close'] * d['volume'] for d in volume_data) / total_volume if total_volume > 0 else 0
            
            # Volume profile analysis
            current_price = volume_data[-1]['close']
            vwap_position = (current_price - vwap) / vwap if vwap > 0 else 0
            
            # Volume nodes identification
            volume_nodes = self._identify_volume_nodes(volume_data)
            
            # Volume profile score
            volume_profile_score = self._calculate_volume_profile_score(
                volume_data, vwap_position, volume_nodes
            )
            
            return {
                'vwap': vwap,
                'vwap_position': vwap_position,
                'current_price': current_price,
                'volume_nodes': volume_nodes,
                'volume_profile_score': volume_profile_score,
                'total_volume': total_volume,
                'avg_volume': total_volume / len(volume_data)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced volume profile analysis for {symbol}: {str(e)}")
            return self._get_default_volume_profile_analysis()
    
    async def _analyze_market_impact_advanced(self, symbol: str) -> Dict:
        """Advanced market impact analysis with institutional-grade algorithms."""
        try:
            trades = await self.binance_service.get_recent_trades(symbol, limit=100)
            
            if not trades or len(trades) < 10:
                return self._get_default_market_impact_analysis()
            
            # Advanced market impact analysis
            impact_data = []
            for i in range(1, len(trades)):
                prev_trade = trades[i-1]
                curr_trade = trades[i]
                
                prev_price = float(prev_trade['price'])
                curr_price = float(curr_trade['price'])
                curr_quantity = float(curr_trade['qty'])
                
                price_impact = (curr_price - prev_price) / prev_price
                impact_data.append({
                    'price_impact': price_impact,
                    'quantity': curr_quantity,
                    'timestamp': curr_trade.get('time', 0)
                })
            
            if not impact_data:
                return self._get_default_market_impact_analysis()
            
            # Impact analysis with volume weighting
            impacts = [d['price_impact'] for d in impact_data]
            quantities = [d['quantity'] for d in impact_data]
            
            # Volume-weighted average impact
            total_quantity = sum(quantities)
            weighted_impact = sum(impacts[i] * quantities[i] for i in range(len(impacts))) / total_quantity if total_quantity > 0 else 0
            
            # Impact volatility
            impact_volatility = np.std(impacts) if len(impacts) > 1 else 0
            
            # Market impact score
            impact_score = self._calculate_market_impact_score(weighted_impact, impact_volatility)
            
            return {
                'weighted_impact': weighted_impact,
                'impact_volatility': impact_volatility,
                'impact_score': impact_score,
                'impact_threshold': self.market_impact_threshold,
                'impact_confidence': min(abs(weighted_impact) / self.market_impact_threshold, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced market impact analysis for {symbol}: {str(e)}")
            return self._get_default_market_impact_analysis()
    
    async def _analyze_bid_ask_spread_advanced(self, symbol: str) -> Dict:
        """Advanced bid-ask spread analysis with institutional-grade algorithms."""
        try:
            order_book = await self.binance_service.get_order_book(symbol)
            
            if not order_book:
                return self._get_default_bid_ask_analysis()
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            if not bids or not asks:
                return self._get_default_bid_ask_analysis()
            
            # Advanced spread analysis
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_percentage = spread / best_bid
            
            # Spread dynamics analysis
            spread_levels = self._analyze_spread_levels(bids, asks)
            
            # Spread score calculation
            spread_score = self._calculate_spread_score(spread_percentage, spread_levels)
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'spread_levels': spread_levels,
                'spread_score': spread_score,
                'spread_quality': self._assess_spread_quality(spread_percentage)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced bid-ask spread analysis for {symbol}: {str(e)}")
            return self._get_default_bid_ask_analysis()
    
    async def _analyze_order_book_depth_advanced(self, symbol: str) -> Dict:
        """Advanced order book depth analysis with institutional-grade algorithms."""
        try:
            order_book = await self.binance_service.get_order_book(symbol)
            
            if not order_book:
                return self._get_default_order_book_analysis()
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            if not bids or not asks:
                return self._get_default_order_book_analysis()
            
            # Advanced depth analysis
            bid_depth_levels = self._analyze_depth_levels(bids, 'bid')
            ask_depth_levels = self._analyze_depth_levels(asks, 'ask')
            
            # Depth imbalance analysis
            depth_imbalance = self._calculate_depth_imbalance(bid_depth_levels, ask_depth_levels)
            
            # Depth concentration analysis
            depth_concentration = self._calculate_depth_concentration(bid_depth_levels, ask_depth_levels)
            
            # Depth score calculation
            depth_score = self._calculate_depth_score(depth_imbalance, depth_concentration)
            
            return {
                'bid_depth_levels': bid_depth_levels,
                'ask_depth_levels': ask_depth_levels,
                'depth_imbalance': depth_imbalance,
                'depth_concentration': depth_concentration,
                'depth_score': depth_score,
                'total_depth': sum(level['quantity'] for level in bid_depth_levels + ask_depth_levels)
            }
            
        except Exception as e:
            logger.error(f"Error in advanced order book depth analysis for {symbol}: {str(e)}")
            return self._get_default_order_book_analysis()
    
    async def _analyze_market_regime(self, symbol: str) -> Dict:
        """Analyze market regime using advanced quantitative algorithms."""
        try:
            # Get historical data for regime analysis
            klines = await self.binance_service.get_klines(symbol, '1h', limit=168)
            
            if not klines or len(klines) < 24:
                return {'regime': MarketRegime.SIDEWAYS.value, 'confidence': 0.5}
            
            # Calculate regime indicators
            prices = [float(kline[4]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            
            # Trend analysis
            price_changes = np.diff(prices)
            trend_strength = np.mean(price_changes) / np.std(price_changes) if np.std(price_changes) > 0 else 0
            
            # Volatility analysis
            volatility = np.std(price_changes)
            avg_volatility = np.mean([np.std(price_changes[i:i+24]) for i in range(0, len(price_changes)-24, 24)]) if len(price_changes) >= 24 else volatility
            
            # Volume analysis
            volume_trend = np.corrcoef(range(len(volumes)), volumes)[0, 1] if len(volumes) > 1 else 0
            
            # Regime classification
            regime, confidence = self._classify_market_regime(trend_strength, volatility, avg_volatility, volume_trend)
            
            return {
                'regime': regime.value,
                'confidence': confidence,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            logger.error(f"Error in market regime analysis for {symbol}: {str(e)}")
            return {'regime': MarketRegime.SIDEWAYS.value, 'confidence': 0.5}
    
    async def _perform_advanced_analysis(self, symbol: str, components: Dict) -> Dict:
        """Perform advanced microstructure analysis combining all components."""
        try:
            # Extract components
            order_flow = components.get('component_0', {})
            liquidity = components.get('component_1', {})
            volume_profile = components.get('component_2', {})
            market_impact = components.get('component_3', {})
            bid_ask = components.get('component_4', {})
            order_book = components.get('component_5', {})
            regime = components.get('component_6', {})
            
            # Calculate advanced metrics
            metrics = MicrostructureMetrics(
                order_flow_imbalance=order_flow.get('volume_imbalance', 0),
                liquidity_score=liquidity.get('liquidity_score', 0),
                volume_profile_score=volume_profile.get('volume_profile_score', 0),
                market_impact_score=market_impact.get('impact_score', 0),
                spread_score=bid_ask.get('spread_score', 0),
                depth_score=order_book.get('depth_score', 0),
                overall_score=0,  # Will be calculated
                confidence_level=0,  # Will be calculated
                regime=MarketRegime(regime.get('regime', MarketRegime.SIDEWAYS.value)),
                timestamp=datetime.now()
            )
            
            # Calculate overall score with advanced weighting
            metrics.overall_score = self._calculate_overall_score(metrics)
            metrics.confidence_level = self._calculate_confidence_level(metrics)
            
            # Compile final analysis
            analysis = {
                'order_flow': order_flow,
                'liquidity': liquidity,
                'volume_profile': volume_profile,
                'market_impact': market_impact,
                'bid_ask_spread': bid_ask,
                'order_book_depth': order_book,
                'market_regime': regime,
                'metrics': {
                    'order_flow_imbalance': metrics.order_flow_imbalance,
                    'liquidity_score': metrics.liquidity_score,
                    'volume_profile_score': metrics.volume_profile_score,
                    'market_impact_score': metrics.market_impact_score,
                    'spread_score': metrics.spread_score,
                    'depth_score': metrics.depth_score,
                    'overall_score': metrics.overall_score,
                    'confidence_level': metrics.confidence_level,
                    'regime': metrics.regime.value
                },
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in advanced analysis for {symbol}: {str(e)}")
            return self._get_default_microstructure_analysis(symbol)
    
    # Advanced helper methods
    def _calculate_signal_strength(self, volume_imbalance: float, price_momentum: float) -> SignalStrength:
        """Calculate signal strength based on volume imbalance and price momentum."""
        try:
            # Combine volume imbalance and price momentum
            combined_signal = (abs(volume_imbalance) + abs(price_momentum)) / 2
            
            if combined_signal > 0.3:
                return SignalStrength.VERY_STRONG
            elif combined_signal > 0.2:
                return SignalStrength.STRONG
            elif combined_signal > 0.1:
                return SignalStrength.WEAK
            elif combined_signal > 0.05:
                return SignalStrength.VERY_WEAK
            else:
                return SignalStrength.NEUTRAL
        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return SignalStrength.NEUTRAL
    
    def _identify_order_flow_patterns(self, trade_data: List[Dict]) -> List[str]:
        """Identify order flow patterns using advanced algorithms."""
        try:
            patterns = []
            
            if len(trade_data) < 10:
                return patterns
            
            # Analyze trade sequences
            buy_sequence = [t['is_buy'] for t in trade_data]
            quantities = [t['quantity'] for t in trade_data]
            prices = [t['price'] for t in trade_data]
            
            # Large order detection
            avg_quantity = np.mean(quantities)
            large_orders = [i for i, q in enumerate(quantities) if q > avg_quantity * 2]
            if len(large_orders) > len(trade_data) * 0.1:
                patterns.append('LARGE_ORDERS')
            
            # Momentum patterns
            if len(prices) > 5:
                recent_prices = prices[-5:]
                if all(recent_prices[i] <= recent_prices[i+1] for i in range(len(recent_prices)-1)):
                    patterns.append('PRICE_MOMENTUM_UP')
                elif all(recent_prices[i] >= recent_prices[i+1] for i in range(len(recent_prices)-1)):
                    patterns.append('PRICE_MOMENTUM_DOWN')
            
            # Volume patterns
            buy_ratio = sum(buy_sequence) / len(buy_sequence)
            if buy_ratio > 0.7:
                patterns.append('BUY_PRESSURE')
            elif buy_ratio < 0.3:
                patterns.append('SELL_PRESSURE')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error identifying order flow patterns: {str(e)}")
            return []
    
    def _calculate_liquidity_levels(self, orders: List, side: str) -> List[Dict]:
        """Calculate liquidity levels with advanced algorithms."""
        try:
            levels = []
            cumulative_quantity = 0
            
            for i, order in enumerate(orders[:20]):  # Analyze top 20 levels
                price = float(order[0])
                quantity = float(order[1])
                cumulative_quantity += quantity
                
                levels.append({
                    'level': i + 1,
                    'price': price,
                    'quantity': quantity,
                    'cumulative_quantity': cumulative_quantity,
                    'side': side
                })
            
            return levels
            
        except Exception as e:
            logger.error(f"Error calculating liquidity levels: {str(e)}")
            return []
    
    def _calculate_liquidity_concentration(self, levels: List[Dict]) -> float:
        """Calculate liquidity concentration score."""
        try:
            if not levels:
                return 0.0
            
            quantities = [level['quantity'] for level in levels]
            total_quantity = sum(quantities)
            
            if total_quantity == 0:
                return 0.0
            
            # Calculate concentration using Herfindahl index
            concentration = sum((q / total_quantity) ** 2 for q in quantities)
            return concentration
            
        except Exception as e:
            logger.error(f"Error calculating liquidity concentration: {str(e)}")
            return 0.0
    
    def _calculate_liquidity_score(self, bid_levels: List[Dict], ask_levels: List[Dict], spread_percentage: float) -> float:
        """Calculate comprehensive liquidity score."""
        try:
            # Base score from spread
            spread_score = max(0, 1 - spread_percentage * 100)
            
            # Depth score
            bid_depth = sum(level['quantity'] for level in bid_levels)
            ask_depth = sum(level['quantity'] for level in ask_levels)
            total_depth = bid_depth + ask_depth
            
            if total_depth == 0:
                depth_score = 0
            else:
                depth_score = min(1, total_depth / 1000)  # Normalize to reasonable range
            
            # Concentration penalty
            bid_concentration = self._calculate_liquidity_concentration(bid_levels)
            ask_concentration = self._calculate_liquidity_concentration(ask_levels)
            concentration_penalty = (bid_concentration + ask_concentration) / 2
            
            # Final score
            final_score = (spread_score * 0.4 + depth_score * 0.4 + (1 - concentration_penalty) * 0.2)
            return max(0, min(1, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.0
    
    def _identify_volume_nodes(self, volume_data: List[Dict]) -> List[Dict]:
        """Identify high volume nodes in the volume profile."""
        try:
            nodes = []
            avg_volume = np.mean([d['volume'] for d in volume_data])
            
            for i, data in enumerate(volume_data):
                if data['volume'] > avg_volume * 1.5:
                    nodes.append({
                        'index': i,
                        'price': data['close'],
                        'volume': data['volume'],
                        'volume_ratio': data['volume'] / avg_volume
                    })
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error identifying volume nodes: {str(e)}")
            return []
    
    def _calculate_volume_profile_score(self, volume_data: List[Dict], vwap_position: float, volume_nodes: List[Dict]) -> float:
        """Calculate volume profile score."""
        try:
            # VWAP position score
            vwap_score = max(0, 1 - abs(vwap_position))
            
            # Volume node proximity score
            if volume_nodes:
                current_price = volume_data[-1]['close']
                nearest_node_distance = min(abs(node['price'] - current_price) / current_price for node in volume_nodes)
                node_score = max(0, 1 - nearest_node_distance * 10)
            else:
                node_score = 0.5
            
            # Volume consistency score
            volumes = [d['volume'] for d in volume_data]
            volume_consistency = 1 - np.std(volumes) / np.mean(volumes) if np.mean(volumes) > 0 else 0
            consistency_score = max(0, min(1, volume_consistency))
            
            # Final score
            final_score = (vwap_score * 0.4 + node_score * 0.3 + consistency_score * 0.3)
            return max(0, min(1, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating volume profile score: {str(e)}")
            return 0.0
    
    def _calculate_market_impact_score(self, weighted_impact: float, impact_volatility: float) -> float:
        """Calculate market impact score."""
        try:
            # Impact magnitude score (lower is better)
            impact_magnitude_score = max(0, 1 - abs(weighted_impact) / self.market_impact_threshold)
            
            # Impact stability score (lower volatility is better)
            stability_score = max(0, 1 - impact_volatility * 10)
            
            # Final score
            final_score = (impact_magnitude_score * 0.7 + stability_score * 0.3)
            return max(0, min(1, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating market impact score: {str(e)}")
            return 0.0
    
    def _analyze_spread_levels(self, bids: List, asks: List) -> Dict:
        """Analyze spread levels and dynamics."""
        try:
            if not bids or not asks:
                return {}
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            # Calculate spread at different levels
            levels = []
            for i in range(min(5, len(bids), len(asks))):
                bid_price = float(bids[i][0])
                ask_price = float(asks[i][0])
                spread = ask_price - bid_price
                spread_pct = spread / bid_price
                
                levels.append({
                    'level': i + 1,
                    'spread': spread,
                    'spread_percentage': spread_pct
                })
            
            return {
                'levels': levels,
                'best_spread': levels[0]['spread'] if levels else 0,
                'spread_curve': [level['spread_percentage'] for level in levels]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spread levels: {str(e)}")
            return {}
    
    def _calculate_spread_score(self, spread_percentage: float, spread_levels: Dict) -> float:
        """Calculate spread quality score."""
        try:
            # Base spread score
            base_score = max(0, 1 - spread_percentage * 100)
            
            # Spread curve consistency
            if 'spread_curve' in spread_levels and len(spread_levels['spread_curve']) > 1:
                curve_consistency = 1 - np.std(spread_levels['spread_curve'])
                curve_score = max(0, min(1, curve_consistency))
            else:
                curve_score = 0.5
            
            # Final score
            final_score = (base_score * 0.7 + curve_score * 0.3)
            return max(0, min(1, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating spread score: {str(e)}")
            return 0.0
    
    def _assess_spread_quality(self, spread_percentage: float) -> str:
        """Assess spread quality."""
        if spread_percentage < 0.0005:
            return 'EXCELLENT'
        elif spread_percentage < 0.001:
            return 'GOOD'
        elif spread_percentage < 0.002:
            return 'FAIR'
        else:
            return 'POOR'
    
    def _analyze_depth_levels(self, orders: List, side: str) -> List[Dict]:
        """Analyze order book depth levels."""
        try:
            levels = []
            cumulative_quantity = 0
            
            for i, order in enumerate(orders[:10]):
                price = float(order[0])
                quantity = float(order[1])
                cumulative_quantity += quantity
                
                levels.append({
                    'level': i + 1,
                    'price': price,
                    'quantity': quantity,
                    'cumulative_quantity': cumulative_quantity,
                    'side': side
                })
            
            return levels
            
        except Exception as e:
            logger.error(f"Error analyzing depth levels: {str(e)}")
            return []
    
    def _calculate_depth_imbalance(self, bid_levels: List[Dict], ask_levels: List[Dict]) -> float:
        """Calculate depth imbalance."""
        try:
            bid_depth = sum(level['quantity'] for level in bid_levels)
            ask_depth = sum(level['quantity'] for level in ask_levels)
            total_depth = bid_depth + ask_depth
            
            if total_depth == 0:
                return 0.0
            
            return (bid_depth - ask_depth) / total_depth
            
        except Exception as e:
            logger.error(f"Error calculating depth imbalance: {str(e)}")
            return 0.0
    
    def _calculate_depth_concentration(self, bid_levels: List[Dict], ask_levels: List[Dict]) -> float:
        """Calculate depth concentration."""
        try:
            all_levels = bid_levels + ask_levels
            if not all_levels:
                return 0.0
            
            quantities = [level['quantity'] for level in all_levels]
            total_quantity = sum(quantities)
            
            if total_quantity == 0:
                return 0.0
            
            # Herfindahl concentration index
            concentration = sum((q / total_quantity) ** 2 for q in quantities)
            return concentration
            
        except Exception as e:
            logger.error(f"Error calculating depth concentration: {str(e)}")
            return 0.0
    
    def _calculate_depth_score(self, depth_imbalance: float, depth_concentration: float) -> float:
        """Calculate depth quality score."""
        try:
            # Balance score (prefer balanced depth)
            balance_score = 1 - abs(depth_imbalance)
            
            # Concentration score (prefer less concentration)
            concentration_score = 1 - depth_concentration
            
            # Final score
            final_score = (balance_score * 0.6 + concentration_score * 0.4)
            return max(0, min(1, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating depth score: {str(e)}")
            return 0.0
    
    def _classify_market_regime(self, trend_strength: float, volatility: float, avg_volatility: float, volume_trend: float) -> Tuple[MarketRegime, float]:
        """Classify market regime with confidence level."""
        try:
            # Determine regime based on indicators
            if trend_strength > 0.1 and volume_trend > 0.3:
                regime = MarketRegime.TRENDING_UP
                confidence = min(abs(trend_strength) + abs(volume_trend), 1.0)
            elif trend_strength < -0.1 and volume_trend < -0.3:
                regime = MarketRegime.TRENDING_DOWN
                confidence = min(abs(trend_strength) + abs(volume_trend), 1.0)
            elif volatility > avg_volatility * 1.5:
                regime = MarketRegime.VOLATILE
                confidence = min(volatility / avg_volatility, 1.0)
            elif volatility < avg_volatility * 0.5:
                regime = MarketRegime.CALM
                confidence = min(avg_volatility / volatility, 1.0)
            else:
                regime = MarketRegime.SIDEWAYS
                confidence = 0.5
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error classifying market regime: {str(e)}")
            return MarketRegime.SIDEWAYS, 0.5
    
    def _calculate_overall_score(self, metrics: MicrostructureMetrics) -> float:
        """Calculate overall microstructure score with advanced weighting."""
        try:
            # Weighted combination of all scores
            weights = {
                'order_flow': 0.25,
                'liquidity': 0.20,
                'volume_profile': 0.20,
                'market_impact': 0.15,
                'spread': 0.10,
                'depth': 0.10
            }
            
            overall_score = (
                abs(metrics.order_flow_imbalance) * weights['order_flow'] +
                metrics.liquidity_score * weights['liquidity'] +
                metrics.volume_profile_score * weights['volume_profile'] +
                metrics.market_impact_score * weights['market_impact'] +
                metrics.spread_score * weights['spread'] +
                metrics.depth_score * weights['depth']
            )
            
            return max(0, min(1, overall_score))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {str(e)}")
            return 0.0
    
    def _calculate_confidence_level(self, metrics: MicrostructureMetrics) -> float:
        """Calculate confidence level for the analysis."""
        try:
            # Confidence based on data quality and consistency
            confidence_factors = [
                metrics.liquidity_score,
                metrics.spread_score,
                metrics.depth_score,
                1 - abs(metrics.order_flow_imbalance)  # More balanced = higher confidence
            ]
            
            confidence = np.mean(confidence_factors)
            return max(0, min(1, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence level: {str(e)}")
            return 0.5
    
    # Cache management methods
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        age = time.time() - cache_entry['timestamp']
        return age < self.cache_ttl
    
    def _cache_result(self, cache_key: str, data: Dict):
        """Cache analysis result."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _get_default_analysis(self) -> Dict:
        """Get default analysis structure."""
        return {
            'error': 'Analysis failed',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _analyze_order_flow(self, symbol: str) -> Dict:
        """Analyze order flow imbalance and patterns."""
        try:
            trades = await self.binance_service.get_recent_trades(symbol, limit=self.order_flow_window)
            
            if not trades:
                return self._get_default_order_flow_analysis()
            
            buy_volume = 0
            sell_volume = 0
            buy_count = 0
            sell_count = 0
            price_changes = []
            
            for trade in trades:
                price = float(trade['price'])
                quantity = float(trade['qty'])
                is_buyer_maker = trade['isBuyerMaker']
                
                if is_buyer_maker:  # Sell order
                    sell_volume += quantity
                    sell_count += 1
                else:  # Buy order
                    buy_volume += quantity
                    buy_count += 1
                
                price_changes.append(price)
            
            total_volume = buy_volume + sell_volume
            volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            trade_imbalance = (buy_count - sell_count) / len(trades) if trades else 0
            
            if len(price_changes) > 1:
                price_momentum = (price_changes[-1] - price_changes[0]) / price_changes[0]
            else:
                price_momentum = 0
            
            # Determine order flow signal
            if volume_imbalance > 0.1 and price_momentum > 0:
                flow_signal = 'STRONG_BUY'
            elif volume_imbalance < -0.1 and price_momentum < 0:
                flow_signal = 'STRONG_SELL'
            elif volume_imbalance > 0.05:
                flow_signal = 'BUY'
            elif volume_imbalance < -0.05:
                flow_signal = 'SELL'
            else:
                flow_signal = 'NEUTRAL'
            
            return {
                'volume_imbalance': volume_imbalance,
                'trade_imbalance': trade_imbalance,
                'buy_ratio': buy_volume / total_volume if total_volume > 0 else 0.5,
                'sell_ratio': sell_volume / total_volume if total_volume > 0 else 0.5,
                'price_momentum': price_momentum,
                'flow_signal': flow_signal,
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_count': buy_count,
                'sell_count': sell_count
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order flow for {symbol}: {str(e)}")
            return self._get_default_order_flow_analysis()
    
    async def _analyze_liquidity_levels(self, symbol: str) -> Dict:
        """Analyze liquidity levels and depth."""
        try:
            order_book = await self.binance_service.get_order_book(symbol)
            
            if not order_book:
                return self._get_default_liquidity_analysis()
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            bid_liquidity = sum(float(bid[1]) for bid in bids[:10])
            ask_liquidity = sum(float(ask[1]) for ask in asks[:10])
            total_liquidity = bid_liquidity + ask_liquidity
            liquidity_ratio = bid_liquidity / ask_liquidity if ask_liquidity > 0 else 1
            
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0
            
            if liquidity_ratio > 1.5 and spread < 0.001:
                liquidity_signal = 'HIGH_LIQUIDITY'
            elif liquidity_ratio < 0.7 or spread > 0.005:
                liquidity_signal = 'LOW_LIQUIDITY'
            else:
                liquidity_signal = 'NORMAL_LIQUIDITY'
            
            return {
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'total_liquidity': total_liquidity,
                'liquidity_ratio': liquidity_ratio,
                'spread': spread,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'mid_price': (best_bid + best_ask) / 2,
                'liquidity_signal': liquidity_signal
            }
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity for {symbol}: {str(e)}")
            return self._get_default_liquidity_analysis()
    
    async def _analyze_volume_profile(self, symbol: str) -> Dict:
        """Analyze volume profile and distribution."""
        try:
            klines = await self.binance_service.get_klines(symbol, '1h', limit=168)
            
            if not klines or len(klines) < 24:
                return self._get_default_volume_profile_analysis()
            
            prices = [float(kline[4]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            
            vwap = sum(prices[i] * volumes[i] for i in range(len(prices))) / sum(volumes)
            current_price = prices[-1]
            vwap_position = (current_price - vwap) / vwap if vwap > 0 else 0
            
            total_volume = sum(volumes)
            avg_volume = total_volume / len(volumes)
            
            high_volume_nodes = []
            for i, volume in enumerate(volumes):
                if volume > avg_volume * 1.5:
                    high_volume_nodes.append({
                        'price_level': prices[i],
                        'volume': volume,
                        'volume_ratio': volume / avg_volume
                    })
            
            if len(high_volume_nodes) > 0:
                nearest_node = min(high_volume_nodes, key=lambda x: abs(x['price_level'] - current_price))
                if abs(nearest_node['price_level'] - current_price) / current_price < 0.01:
                    volume_signal = 'NEAR_VOLUME_NODE'
                else:
                    volume_signal = 'AWAY_FROM_VOLUME_NODE'
            else:
                volume_signal = 'NO_VOLUME_NODES'
            
            return {
                'high_volume_nodes': high_volume_nodes,
                'vwap': vwap,
                'vwap_position': vwap_position,
                'current_price': current_price,
                'total_volume': total_volume,
                'avg_volume': avg_volume,
                'volume_signal': volume_signal
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile for {symbol}: {str(e)}")
            return self._get_default_volume_profile_analysis()
    
    async def _analyze_market_impact(self, symbol: str) -> Dict:
        """Analyze market impact of trades."""
        try:
            trades = await self.binance_service.get_recent_trades(symbol, limit=50)
            
            if not trades:
                return self._get_default_market_impact_analysis()
            
            price_impacts = []
            for i in range(1, len(trades)):
                prev_price = float(trades[i-1]['price'])
                curr_price = float(trades[i]['price'])
                price_impact = (curr_price - prev_price) / prev_price
                price_impacts.append(price_impact)
            
            if not price_impacts:
                return self._get_default_market_impact_analysis()
            
            avg_price_impact = np.mean(price_impacts)
            impact_volatility = np.std(price_impacts)
            
            if abs(avg_price_impact) > self.market_impact_threshold:
                if avg_price_impact > 0:
                    impact_signal = 'HIGH_POSITIVE_IMPACT'
                else:
                    impact_signal = 'HIGH_NEGATIVE_IMPACT'
            elif abs(avg_price_impact) > self.market_impact_threshold / 2:
                if avg_price_impact > 0:
                    impact_signal = 'MODERATE_POSITIVE_IMPACT'
                else:
                    impact_signal = 'MODERATE_NEGATIVE_IMPACT'
            else:
                impact_signal = 'LOW_IMPACT'
            
            return {
                'avg_price_impact': avg_price_impact,
                'impact_volatility': impact_volatility,
                'impact_signal': impact_signal,
                'impact_threshold': self.market_impact_threshold
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market impact for {symbol}: {str(e)}")
            return self._get_default_market_impact_analysis()
    
    async def _analyze_bid_ask_spread(self, symbol: str) -> Dict:
        """Analyze bid-ask spread dynamics."""
        try:
            order_book = await self.binance_service.get_order_book(symbol)
            
            if not order_book:
                return self._get_default_bid_ask_analysis()
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            if not bids or not asks:
                return self._get_default_bid_ask_analysis()
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_percentage = spread / best_bid
            
            if spread_percentage < 0.0005:
                spread_signal = 'TIGHT_SPREAD'
            elif spread_percentage > 0.002:
                spread_signal = 'WIDE_SPREAD'
            else:
                spread_signal = 'NORMAL_SPREAD'
            
            return {
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_percentage': spread_percentage,
                'spread_signal': spread_signal
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bid-ask spread for {symbol}: {str(e)}")
            return self._get_default_bid_ask_analysis()
    
    async def _analyze_order_book_depth(self, symbol: str) -> Dict:
        """Analyze order book depth and structure."""
        try:
            order_book = await self.binance_service.get_order_book(symbol)
            
            if not order_book:
                return self._get_default_order_book_analysis()
            
            bids = order_book['bids']
            asks = order_book['asks']
            
            bid_depth = sum(float(bid[1]) for bid in bids[:10])
            ask_depth = sum(float(ask[1]) for ask in asks[:10])
            total_depth = bid_depth + ask_depth
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            if depth_imbalance > 0.2:
                depth_signal = 'BID_HEAVY'
            elif depth_imbalance < -0.2:
                depth_signal = 'ASK_HEAVY'
            else:
                depth_signal = 'BALANCED_DEPTH'
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'depth_imbalance': depth_imbalance,
                'depth_signal': depth_signal
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order book depth for {symbol}: {str(e)}")
            return self._get_default_order_book_analysis()
    
    def _calculate_microstructure_score(self, analysis: Dict) -> float:
        """Calculate overall microstructure score."""
        try:
            score = 0.0
            factors = 0
            
            if 'order_flow' in analysis:
                flow = analysis['order_flow']
                volume_imbalance = abs(flow.get('volume_imbalance', 0))
                score += min(volume_imbalance * 2, 0.3)
                factors += 1
            
            if 'liquidity' in analysis:
                liquidity = analysis['liquidity']
                spread = liquidity.get('spread', 0)
                if spread < 0.001:
                    score += 0.2
                elif spread < 0.002:
                    score += 0.1
                factors += 1
            
            if 'volume_profile' in analysis:
                profile = analysis['volume_profile']
                if profile.get('volume_signal') == 'NEAR_VOLUME_NODE':
                    score += 0.2
                factors += 1
            
            if factors > 0:
                score = score / factors
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating microstructure score: {str(e)}")
            return 0.0
    
    def _get_default_microstructure_analysis(self, symbol: str) -> Dict:
        return {
            'order_flow': self._get_default_order_flow_analysis(),
            'liquidity': self._get_default_liquidity_analysis(),
            'volume_profile': self._get_default_volume_profile_analysis(),
            'market_impact': self._get_default_market_impact_analysis(),
            'bid_ask_spread': self._get_default_bid_ask_analysis(),
            'order_book_depth': self._get_default_order_book_analysis(),
            'microstructure_score': 0.0,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }
    
    def _get_default_order_flow_analysis(self) -> Dict:
        return {
            'volume_imbalance': 0.0,
            'trade_imbalance': 0.0,
            'buy_ratio': 0.5,
            'sell_ratio': 0.5,
            'price_momentum': 0.0,
            'flow_signal': 'NEUTRAL',
            'total_volume': 0.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'buy_count': 0,
            'sell_count': 0
        }
    
    def _get_default_liquidity_analysis(self) -> Dict:
        return {
            'bid_liquidity': 0.0,
            'ask_liquidity': 0.0,
            'total_liquidity': 0.0,
            'liquidity_ratio': 1.0,
            'spread': 0.0,
            'best_bid': 0.0,
            'best_ask': 0.0,
            'mid_price': 0.0,
            'liquidity_signal': 'NORMAL_LIQUIDITY'
        }
    
    def _get_default_volume_profile_analysis(self) -> Dict:
        return {
            'high_volume_nodes': [],
            'vwap': 0.0,
            'vwap_position': 0.0,
            'current_price': 0.0,
            'total_volume': 0.0,
            'avg_volume': 0.0,
            'volume_signal': 'NO_VOLUME_NODES'
        }
    
    def _get_default_market_impact_analysis(self) -> Dict:
        return {
            'avg_price_impact': 0.0,
            'impact_volatility': 0.0,
            'impact_signal': 'LOW_IMPACT',
            'impact_threshold': self.market_impact_threshold
        }
    
    def _get_default_bid_ask_analysis(self) -> Dict:
        return {
            'best_bid': 0.0,
            'best_ask': 0.0,
            'spread': 0.0,
            'spread_percentage': 0.0,
            'spread_signal': 'NORMAL_SPREAD'
        }
    
    def _get_default_order_book_analysis(self) -> Dict:
        return {
            'bid_depth': 0.0,
            'ask_depth': 0.0,
            'total_depth': 0.0,
            'depth_imbalance': 0.0,
            'depth_signal': 'BALANCED_DEPTH'
        }
