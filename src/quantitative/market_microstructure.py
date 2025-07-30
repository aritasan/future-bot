import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MarketMicrostructureAnalyzer:
    """
    Advanced market microstructure analysis for quantitative trading.
    Implements bid-ask spread, order flow, and market impact analysis.
    """
    
    def __init__(self, min_tick_size: float = 0.0001):
        self.min_tick_size = min_tick_size
        self.analysis_history = []
        
    def analyze_market_structure(self, orderbook_data: Dict, trade_data: pd.DataFrame = None) -> Dict:
        """
        Comprehensive market microstructure analysis.
        
        Args:
            orderbook_data: Dictionary containing bid/ask data
            trade_data: DataFrame of trade data
            
        Returns:
            Dict: Market microstructure metrics
        """
        try:
            results = {
                'bid_ask_spread': self._calculate_bid_ask_spread(orderbook_data),
                'spread_analysis': self._analyze_spread_dynamics(orderbook_data),
                'order_flow_imbalance': self._calculate_order_flow_imbalance(orderbook_data),
                'market_depth': self._analyze_market_depth(orderbook_data),
                'price_impact': self._estimate_price_impact(orderbook_data),
                'liquidity_metrics': self._calculate_liquidity_metrics(orderbook_data),
                'volatility_metrics': self._calculate_volatility_metrics(orderbook_data),
                'market_efficiency': self._assess_market_efficiency(orderbook_data)
            }
            
            if trade_data is not None:
                results.update({
                    'trade_analysis': self._analyze_trade_patterns(trade_data),
                    'volume_analysis': self._analyze_volume_patterns(trade_data),
                    'time_and_sales': self._analyze_time_and_sales(trade_data)
                })
            
            self._store_analysis_result(results)
            return results
            
        except Exception as e:
            logger.error(f"Error in market microstructure analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_bid_ask_spread(self, orderbook_data: Dict) -> Dict:
        """Calculate bid-ask spread metrics."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            # Handle different orderbook formats
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Check if data is in list format [price, size]
            if isinstance(bids, list) and len(bids) > 0:
                if isinstance(bids[0], list):
                    # Format: [[price, size], [price, size], ...]
                    best_bid = max(bids, key=lambda x: float(x[0]))
                    best_ask = min(asks, key=lambda x: float(x[0]))
                    
                    best_bid_price = float(best_bid[0])
                    best_bid_size = float(best_bid[1])
                    best_ask_price = float(best_ask[0])
                    best_ask_size = float(best_ask[1])
                else:
                    # Format: [{'price': price, 'size': size}, ...]
                    best_bid = max(bids, key=lambda x: float(x['price']))
                    best_ask = min(asks, key=lambda x: float(x['price']))
                    
                    best_bid_price = float(best_bid['price'])
                    best_bid_size = float(best_bid['size'])
                    best_ask_price = float(best_ask['price'])
                    best_ask_size = float(best_ask['size'])
            else:
                return {'error': 'Invalid orderbook format'}
            
            spread = best_ask_price - best_bid_price
            spread_bps = (spread / best_bid_price) * 10000  # Basis points
            mid_price = (best_bid_price + best_ask_price) / 2
            
            return {
                'absolute_spread': spread,
                'relative_spread_bps': spread_bps,
                'mid_price': mid_price,
                'best_bid': best_bid_price,
                'best_ask': best_ask_price,
                'bid_size': best_bid_size,
                'ask_size': best_ask_size
            }
            
        except Exception as e:
            logger.error(f"Error calculating bid-ask spread: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_spread_dynamics(self, orderbook_data: Dict) -> Dict:
        """Analyze spread dynamics and patterns."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Handle different orderbook formats
            if isinstance(bids, list) and len(bids) > 0:
                if isinstance(bids[0], list):
                    # Format: [[price, size], [price, size], ...]
                    bid_levels = sorted(bids, key=lambda x: float(x[0]), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x[0]))
                else:
                    # Format: [{'price': price, 'size': size}, ...]
                    bid_levels = sorted(bids, key=lambda x: float(x['price']), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x['price']))
            else:
                return {'error': 'Invalid orderbook format'}
            
            spreads = []
            for i in range(min(5, len(bid_levels), len(ask_levels))):
                if isinstance(bid_levels[0], list):
                    spread = float(ask_levels[i][0]) - float(bid_levels[i][0])
                else:
                    spread = float(ask_levels[i]['price']) - float(bid_levels[i]['price'])
                spreads.append(spread)
            
            # Calculate spread statistics
            spread_stats = {
                'mean_spread': np.mean(spreads),
                'std_spread': np.std(spreads),
                'min_spread': np.min(spreads),
                'max_spread': np.max(spreads),
                'spread_skewness': self._calculate_skewness(spreads),
                'spread_kurtosis': self._calculate_kurtosis(spreads)
            }
            
            return spread_stats
            
        except Exception as e:
            logger.error(f"Error analyzing spread dynamics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_order_flow_imbalance(self, orderbook_data: Dict) -> Dict:
        """Calculate order flow imbalance metrics."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Handle different orderbook formats
            if isinstance(bids, list) and len(bids) > 0:
                if isinstance(bids[0], list):
                    # Format: [[price, size], [price, size], ...]
                    total_bid_volume = sum(float(bid[1]) for bid in bids)
                    total_ask_volume = sum(float(ask[1]) for ask in asks)
                    weighted_bid_price = sum(float(bid[0]) * float(bid[1]) for bid in bids) / total_bid_volume if total_bid_volume > 0 else 0
                    weighted_ask_price = sum(float(ask[0]) * float(ask[1]) for ask in asks) / total_ask_volume if total_ask_volume > 0 else 0
                else:
                    # Format: [{'price': price, 'size': size}, ...]
                    total_bid_volume = sum(float(bid['size']) for bid in bids)
                    total_ask_volume = sum(float(ask['size']) for ask in asks)
                    weighted_bid_price = sum(float(bid['price']) * float(bid['size']) for bid in bids) / total_bid_volume if total_bid_volume > 0 else 0
                    weighted_ask_price = sum(float(ask['price']) * float(ask['size']) for ask in asks) / total_ask_volume if total_ask_volume > 0 else 0
            else:
                return {'error': 'Invalid orderbook format'}
            
            # Calculate imbalance metrics
            total_volume = total_bid_volume + total_ask_volume
            imbalance_ratio = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0
            
            return {
                'total_bid_volume': total_bid_volume,
                'total_ask_volume': total_ask_volume,
                'imbalance_ratio': imbalance_ratio,
                'weighted_bid_price': weighted_bid_price,
                'weighted_ask_price': weighted_ask_price,
                'volume_ratio': total_bid_volume / total_ask_volume if total_ask_volume > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error calculating order flow imbalance: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_market_depth(self, orderbook_data: Dict) -> Dict:
        """Analyze market depth and liquidity distribution."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Handle different orderbook formats
            if isinstance(bids, list) and len(bids) > 0:
                if isinstance(bids[0], list):
                    # Format: [[price, size], [price, size], ...]
                    bid_levels = sorted(bids, key=lambda x: float(x[0]), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x[0]))
                    
                    cumulative_bid_volume = []
                    cumulative_ask_volume = []
                    
                    for i, bid in enumerate(bid_levels[:10]):  # Top 10 levels
                        cumulative_volume = sum(float(bid_levels[j][1]) for j in range(i + 1))
                        cumulative_bid_volume.append({
                            'level': i + 1,
                            'price': float(bid[0]),
                            'cumulative_volume': cumulative_volume
                        })
                    
                    for i, ask in enumerate(ask_levels[:10]):  # Top 10 levels
                        cumulative_volume = sum(float(ask_levels[j][1]) for j in range(i + 1))
                        cumulative_ask_volume.append({
                            'level': i + 1,
                            'price': float(ask[0]),
                            'cumulative_volume': cumulative_volume
                        })
                    
                    # Calculate depth metrics
                    total_bid_depth = sum(float(bid[1]) for bid in bid_levels[:5])
                    total_ask_depth = sum(float(ask[1]) for ask in ask_levels[:5])
                else:
                    # Format: [{'price': price, 'size': size}, ...]
                    bid_levels = sorted(bids, key=lambda x: float(x['price']), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x['price']))
                    
                    cumulative_bid_volume = []
                    cumulative_ask_volume = []
                    
                    for i, bid in enumerate(bid_levels[:10]):  # Top 10 levels
                        cumulative_volume = sum(float(bid_levels[j]['size']) for j in range(i + 1))
                        cumulative_bid_volume.append({
                            'level': i + 1,
                            'price': float(bid['price']),
                            'cumulative_volume': cumulative_volume
                        })
                    
                    for i, ask in enumerate(ask_levels[:10]):  # Top 10 levels
                        cumulative_volume = sum(float(ask_levels[j]['size']) for j in range(i + 1))
                        cumulative_ask_volume.append({
                            'level': i + 1,
                            'price': float(ask['price']),
                            'cumulative_volume': cumulative_volume
                        })
                    
                    # Calculate depth metrics
                    total_bid_depth = sum(float(bid['size']) for bid in bid_levels[:5])
                    total_ask_depth = sum(float(ask['size']) for ask in ask_levels[:5])
            else:
                return {'error': 'Invalid orderbook format'}
            
            return {
                'cumulative_bid_depth': cumulative_bid_volume,
                'cumulative_ask_depth': cumulative_ask_volume,
                'total_bid_depth_5_levels': total_bid_depth,
                'total_ask_depth_5_levels': total_ask_depth,
                'depth_imbalance': (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth) if (total_bid_depth + total_ask_depth) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market depth: {str(e)}")
            return {'error': str(e)}
    
    def _estimate_price_impact(self, orderbook_data: Dict) -> Dict:
        """Estimate price impact of trades."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Handle different orderbook formats
            if isinstance(bids, list) and len(bids) > 0:
                if isinstance(bids[0], list):
                    # Format: [[price, size], [price, size], ...]
                    bid_levels = sorted(bids, key=lambda x: float(x[0]), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x[0]))
                else:
                    # Format: [{'price': price, 'size': size}, ...]
                    bid_levels = sorted(bids, key=lambda x: float(x['price']), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x['price']))
            else:
                return {'error': 'Invalid orderbook format'}
            
            # Calculate price impact for different trade sizes
            trade_sizes = [1000, 5000, 10000, 50000, 100000]  # Example trade sizes
            price_impacts = {}
            
            for size in trade_sizes:
                # Buy impact (consuming asks)
                remaining_size = size
                total_cost = 0
                ask_impact = 0
                
                for ask in ask_levels:
                    if remaining_size <= 0:
                        break
                    if isinstance(ask, list):
                        fill_size = min(remaining_size, float(ask[1]))
                        total_cost += fill_size * float(ask[0])
                    else:
                        fill_size = min(remaining_size, float(ask['size']))
                        total_cost += fill_size * float(ask['price'])
                    remaining_size -= fill_size
                
                if size - remaining_size > 0:
                    avg_buy_price = total_cost / (size - remaining_size)
                    if isinstance(ask_levels[0], list):
                        base_price = float(ask_levels[0][0])
                    else:
                        base_price = float(ask_levels[0]['price'])
                    ask_impact = (avg_buy_price - base_price) / base_price
                
                # Sell impact (consuming bids)
                remaining_size = size
                total_proceeds = 0
                bid_impact = 0
                
                for bid in bid_levels:
                    if remaining_size <= 0:
                        break
                    if isinstance(bid, list):
                        fill_size = min(remaining_size, float(bid[1]))
                        total_proceeds += fill_size * float(bid[0])
                    else:
                        fill_size = min(remaining_size, float(bid['size']))
                        total_proceeds += fill_size * float(bid['price'])
                    remaining_size -= fill_size
                
                if size - remaining_size > 0:
                    avg_sell_price = total_proceeds / (size - remaining_size)
                    if isinstance(bid_levels[0], list):
                        base_price = float(bid_levels[0][0])
                    else:
                        base_price = float(bid_levels[0]['price'])
                    bid_impact = (base_price - avg_sell_price) / base_price
                
                price_impacts[f'trade_size_{size}'] = {
                    'buy_impact_bps': ask_impact * 10000,
                    'sell_impact_bps': bid_impact * 10000,
                    'avg_impact_bps': (ask_impact + bid_impact) * 5000
                }
            
            return price_impacts
            
        except Exception as e:
            logger.error(f"Error estimating price impact: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_liquidity_metrics(self, orderbook_data: Dict) -> Dict:
        """Calculate liquidity metrics."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            # Calculate Amihud illiquidity ratio (proxy)
            spread = self._calculate_bid_ask_spread(orderbook_data)
            if 'error' in spread:
                return {'error': spread['error']}
            
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Handle different orderbook formats
            if isinstance(bids, list) and len(bids) > 0:
                if isinstance(bids[0], list):
                    # Format: [[price, size], ...]
                    total_volume = sum(float(bid[1]) for bid in bids) + sum(float(ask[1]) for ask in asks)
                    bid_levels = sorted(bids, key=lambda x: float(x[0]), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x[0]))
                    # Top 5 levels
                    total_bid_volume = sum(float(bid[1]) for bid in bid_levels[:5])
                    total_ask_volume = sum(float(ask[1]) for ask in ask_levels[:5])
                else:
                    # Format: [{'price': price, 'size': size}, ...]
                    total_volume = sum(float(bid['size']) for bid in bids) + sum(float(ask['size']) for ask in asks)
                    bid_levels = sorted(bids, key=lambda x: float(x['price']), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x['price']))
                    total_bid_volume = sum(float(bid['size']) for bid in bid_levels[:5])
                    total_ask_volume = sum(float(ask['size']) for ask in ask_levels[:5])
            else:
                return {'error': 'Invalid orderbook format'}
            
            mid_price = spread['mid_price']
            # Simple Kyle's lambda approximation
            kyle_lambda = spread['absolute_spread'] / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) > 0 else 0
            
            return {
                'amihud_illiquidity': spread['absolute_spread'] / (mid_price * total_volume) if (mid_price * total_volume) > 0 else float('inf'),
                'turnover_ratio': total_volume / mid_price if mid_price > 0 else 0,
                'kyle_lambda': kyle_lambda,
                'liquidity_score': 1 / (1 + kyle_lambda) if kyle_lambda > 0 else 1
            }
        except Exception as e:
            logger.error(f"Error calculating liquidity metrics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_volatility_metrics(self, orderbook_data: Dict) -> Dict:
        """Calculate volatility metrics from orderbook."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Handle different orderbook formats
            if isinstance(bids, list) and len(bids) > 0:
                if isinstance(bids[0], list):
                    # Format: [[price, size], [price, size], ...]
                    bid_levels = sorted(bids, key=lambda x: float(x[0]), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x[0]))
                else:
                    # Format: [{'price': price, 'size': size}, ...]
                    bid_levels = sorted(bids, key=lambda x: float(x['price']), reverse=True)
                    ask_levels = sorted(asks, key=lambda x: float(x['price']))
            else:
                return {'error': 'Invalid orderbook format'}
            
            # Calculate price volatility from bid-ask spreads
            spreads = []
            
            for i in range(min(5, len(bid_levels), len(ask_levels))):
                if isinstance(bid_levels[0], list):
                    spread = float(ask_levels[i][0]) - float(bid_levels[i][0])
                else:
                    spread = float(ask_levels[i]['price']) - float(bid_levels[i]['price'])
                spreads.append(spread)
            
            if len(spreads) > 1:
                spread_volatility = np.std(spreads)
                spread_mean = np.mean(spreads)
                coefficient_of_variation = spread_volatility / spread_mean if spread_mean > 0 else 0
            else:
                spread_volatility = 0
                spread_mean = spreads[0] if spreads else 0
                coefficient_of_variation = 0
            
            return {
                'spread_volatility': spread_volatility,
                'spread_mean': spread_mean,
                'coefficient_of_variation': coefficient_of_variation,
                'spread_range': np.max(spreads) - np.min(spreads) if spreads else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {'error': str(e)}
    
    def _assess_market_efficiency(self, orderbook_data: Dict) -> Dict:
        """Assess market efficiency metrics."""
        try:
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                return {'error': 'Missing bid/ask data'}
            
            # Calculate efficiency metrics
            spread = self._calculate_bid_ask_spread(orderbook_data)
            if 'error' in spread:
                return {'error': spread['error']}
            
            # Market efficiency score (lower spread = more efficient)
            efficiency_score = 1 / (1 + spread['relative_spread_bps'] / 100)
            
            # Calculate order book imbalance
            imbalance = self._calculate_order_flow_imbalance(orderbook_data)
            if 'error' in imbalance:
                return {'error': imbalance['error']}
            
            return {
                'efficiency_score': efficiency_score,
                'spread_efficiency': 1 / (1 + spread['relative_spread_bps']),
                'depth_efficiency': 1 / (1 + abs(imbalance['imbalance_ratio'])),
                'overall_efficiency': (efficiency_score + (1 / (1 + abs(imbalance['imbalance_ratio'])))) / 2
            }
            
        except Exception as e:
            logger.error(f"Error assessing market efficiency: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_trade_patterns(self, trade_data: pd.DataFrame) -> Dict:
        """Analyze trade patterns and characteristics."""
        try:
            if trade_data.empty:
                return {'error': 'No trade data available'}
            
            # Calculate trade statistics
            trade_sizes = trade_data['size'] if 'size' in trade_data.columns else pd.Series([1] * len(trade_data))
            trade_prices = trade_data['price'] if 'price' in trade_data.columns else pd.Series([0] * len(trade_data))
            
            # Calculate trade patterns
            large_trades = trade_sizes[trade_sizes > trade_sizes.quantile(0.9)]
            small_trades = trade_sizes[trade_sizes < trade_sizes.quantile(0.1)]
            
            return {
                'total_trades': len(trade_data),
                'avg_trade_size': trade_sizes.mean(),
                'median_trade_size': trade_sizes.median(),
                'large_trade_ratio': len(large_trades) / len(trade_data),
                'small_trade_ratio': len(small_trades) / len(trade_data),
                'price_volatility': trade_prices.std(),
                'size_volatility': trade_sizes.std()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_volume_patterns(self, trade_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns and distribution."""
        try:
            if trade_data.empty:
                return {'error': 'No trade data available'}
            
            # Calculate volume metrics
            volumes = trade_data['size'] if 'size' in trade_data.columns else pd.Series([1] * len(trade_data))
            
            # Volume distribution analysis
            volume_quantiles = volumes.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            
            # Calculate volume concentration
            total_volume = volumes.sum()
            top_10_percent_volume = volumes.nlargest(int(len(volumes) * 0.1)).sum()
            concentration_ratio = top_10_percent_volume / total_volume if total_volume > 0 else 0
            
            return {
                'total_volume': total_volume,
                'avg_volume': volumes.mean(),
                'volume_std': volumes.std(),
                'volume_quantiles': volume_quantiles.to_dict(),
                'concentration_ratio': concentration_ratio,
                'volume_skewness': self._calculate_skewness(volumes),
                'volume_kurtosis': self._calculate_kurtosis(volumes)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_time_and_sales(self, trade_data: pd.DataFrame) -> Dict:
        """Analyze time and sales data."""
        try:
            if trade_data.empty:
                return {'error': 'No trade data available'}
            
            # Time-based analysis
            if 'timestamp' in trade_data.columns:
                trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
                trade_data = trade_data.sort_values('timestamp')
                
                # Calculate time intervals between trades
                time_intervals = trade_data['timestamp'].diff().dropna()
                avg_interval = time_intervals.mean()
                
                # Calculate trade frequency
                total_time = (trade_data['timestamp'].max() - trade_data['timestamp'].min()).total_seconds()
                trade_frequency = len(trade_data) / total_time if total_time > 0 else 0
                
                return {
                    'avg_time_interval': avg_interval,
                    'trade_frequency': trade_frequency,
                    'time_interval_std': time_intervals.std(),
                    'total_trading_time': total_time
                }
            else:
                return {'error': 'No timestamp data available'}
                
        except Exception as e:
            logger.error(f"Error analyzing time and sales: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_skewness(self, data: pd.Series) -> float:
        """Calculate skewness of data."""
        try:
            return data.skew()
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: pd.Series) -> float:
        """Calculate kurtosis of data."""
        try:
            return data.kurtosis()
        except:
            return 0.0
    
    def _store_analysis_result(self, result: Dict):
        """Store analysis result in history."""
        self.analysis_history.append({
            'timestamp': pd.Timestamp.now(),
            'result': result
        })
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of all analysis results."""
        if not self.analysis_history:
            return {'message': 'No analysis history available'}
        
        summary = {
            'total_analyses': len(self.analysis_history),
            'recent_analyses': self.analysis_history[-5:],
            'average_spreads': [],
            'average_imbalances': []
        }
        
        for record in self.analysis_history:
            result = record['result']
            
            if 'bid_ask_spread' in result and 'relative_spread_bps' in result['bid_ask_spread']:
                summary['average_spreads'].append(result['bid_ask_spread']['relative_spread_bps'])
            
            if 'order_flow_imbalance' in result and 'imbalance_ratio' in result['order_flow_imbalance']:
                summary['average_imbalances'].append(result['order_flow_imbalance']['imbalance_ratio'])
        
        if summary['average_spreads']:
            summary['avg_spread_bps'] = np.mean(summary['average_spreads'])
        
        if summary['average_imbalances']:
            summary['avg_imbalance'] = np.mean(summary['average_imbalances'])
        
        return summary 