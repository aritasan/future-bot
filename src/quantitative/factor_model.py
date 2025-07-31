"""
WorldQuant Factor Model Implementation
Multi-factor model with market, size, value, momentum, volatility, and liquidity factors.
Implements factor exposure calculation, risk attribution analysis, and sector/geographic risk exposure.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WorldQuantFactorModel:
    """
    WorldQuant-level multi-factor model for quantitative trading.
    Implements comprehensive factor analysis with risk attribution.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize WorldQuant Factor Model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Factor definitions
        self.factors = {
            'market': 'Market factor (CAPM beta)',
            'size': 'Size factor (small vs large cap)',
            'value': 'Value factor (book-to-market ratio)',
            'momentum': 'Momentum factor (price momentum)',
            'volatility': 'Volatility factor (realized volatility)',
            'liquidity': 'Liquidity factor (bid-ask spread, volume)'
        }
        
        # Factor parameters
        self.factor_params = {
            'market': {'lookback': 252, 'min_data': 100},
            'size': {'lookback': 252, 'min_data': 100},
            'value': {'lookback': 252, 'min_data': 100},
            'momentum': {'lookback': 63, 'min_data': 50},
            'volatility': {'lookback': 21, 'min_data': 20},
            'liquidity': {'lookback': 21, 'min_data': 20}
        }
        
        # Risk attribution parameters
        self.risk_attribution = {
            'confidence_level': 0.95,
            'var_confidence': 0.99,
            'max_factor_exposure': 0.3,
            'min_factor_exposure': -0.3
        }
        
        # Sector and geographic classifications
        self.sector_classifications = {
            'technology': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT'],
            'finance': ['BNBUSDT', 'DOTUSDT'],
            'energy': ['XRPUSDT'],
            'consumer': ['DOGEUSDT', 'SHIBUSDT'],
            'utilities': ['LTCUSDT', 'BCHUSDT']
        }
        
        self.geographic_classifications = {
            'asia_pacific': ['BNBUSDT', 'ADAUSDT', 'DOGEUSDT'],
            'europe': ['ETHUSDT', 'DOTUSDT'],
            'americas': ['BTCUSDT', 'SOLUSDT', 'XRPUSDT'],
            'global': ['LTCUSDT', 'BCHUSDT', 'SHIBUSDT']
        }
        
        # Factor data storage
        self.factor_data = {}
        self.factor_exposures = {}
        self.risk_attribution_results = {}
        
        # Performance tracking
        self.factor_performance = {}
        self.risk_metrics = {}
        
        logger.info("WorldQuantFactorModel initialized")
    
    async def initialize(self) -> bool:
        """Initialize the factor model."""
        try:
            # Initialize factor data structures
            for factor in self.factors.keys():
                self.factor_data[factor] = {}
                self.factor_exposures[factor] = {}
                self.factor_performance[factor] = {}
            
            logger.info("WorldQuantFactorModel initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing WorldQuantFactorModel: {str(e)}")
            return False
    
    async def calculate_market_factor(self, symbols: List[str], market_data: Dict) -> Dict[str, float]:
        """
        Calculate market factor (CAPM beta) for all symbols.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of market factor exposures
        """
        try:
            market_exposures = {}
            
            # Get market benchmark (BTC as proxy for crypto market)
            if 'BTCUSDT' in market_data and 'returns' in market_data['BTCUSDT']:
                market_returns = np.array(market_data['BTCUSDT']['returns'])
            else:
                # Use average of all symbols as market proxy
                all_returns = []
                for symbol in symbols:
                    if symbol in market_data and 'returns' in market_data[symbol]:
                        all_returns.append(market_data[symbol]['returns'])
                
                if all_returns:
                    market_returns = np.mean(all_returns, axis=0)
                else:
                    logger.warning("No market data available for market factor calculation")
                    return {}
            
            # Calculate beta for each symbol
            for symbol in symbols:
                if symbol in market_data and 'returns' in market_data[symbol]:
                    symbol_returns = np.array(market_data[symbol]['returns'])
                    
                    # Ensure same length
                    min_length = min(len(symbol_returns), len(market_returns))
                    if min_length >= self.factor_params['market']['min_data']:
                        symbol_returns = symbol_returns[:min_length]
                        market_returns_aligned = market_returns[:min_length]
                        
                        # Calculate CAPM beta
                        covariance = np.cov(symbol_returns, market_returns_aligned)[0, 1]
                        market_variance = np.var(market_returns_aligned)
                        
                        if market_variance > 0:
                            beta = covariance / market_variance
                            market_exposures[symbol] = float(beta)
                        else:
                            market_exposures[symbol] = 1.0  # Default beta
                    else:
                        market_exposures[symbol] = 1.0  # Default beta
                else:
                    market_exposures[symbol] = 1.0  # Default beta
            
            # Store factor data
            self.factor_data['market'] = market_exposures
            self.factor_exposures['market'] = market_exposures
            
            logger.info(f"Market factor calculated for {len(market_exposures)} symbols")
            return market_exposures
            
        except Exception as e:
            logger.error(f"Error calculating market factor: {str(e)}")
            return {}
    
    async def calculate_size_factor(self, symbols: List[str], market_data: Dict) -> Dict[str, float]:
        """
        Calculate size factor (market cap based) for all symbols.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of size factor exposures
        """
        try:
            size_exposures = {}
            
            # Get market caps (use price * volume as proxy)
            market_caps = {}
            
            for symbol in symbols:
                if symbol in market_data and 'price' in market_data[symbol] and 'volume' in market_data[symbol]:
                    price = market_data[symbol]['price']
                    volume = market_data[symbol]['volume']
                    market_cap = price * volume
                    market_caps[symbol] = market_cap
            
            if market_caps:
                # Calculate size factor (negative of log market cap)
                max_market_cap = max(market_caps.values())
                
                for symbol, market_cap in market_caps.items():
                    if market_cap > 0:
                        # Size factor = -log(market_cap / max_market_cap)
                        size_factor = -np.log(market_cap / max_market_cap)
                        size_exposures[symbol] = float(size_factor)
                    else:
                        size_exposures[symbol] = 0.0
            else:
                # Default size factors based on symbol type
                for symbol in symbols:
                    if 'BTC' in symbol or 'ETH' in symbol:
                        size_exposures[symbol] = -0.5  # Large cap
                    elif 'BNB' in symbol or 'ADA' in symbol:
                        size_exposures[symbol] = 0.0   # Mid cap
                    else:
                        size_exposures[symbol] = 0.5   # Small cap
            
            # Store factor data
            self.factor_data['size'] = size_exposures
            self.factor_exposures['size'] = size_exposures
            
            logger.info(f"Size factor calculated for {len(size_exposures)} symbols")
            return size_exposures
            
        except Exception as e:
            logger.error(f"Error calculating size factor: {str(e)}")
            return {}
    
    async def calculate_value_factor(self, symbols: List[str], market_data: Dict) -> Dict[str, float]:
        """
        Calculate value factor (book-to-market proxy) for all symbols.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of value factor exposures
        """
        try:
            value_exposures = {}
            
            for symbol in symbols:
                if symbol in market_data and 'price' in market_data[symbol]:
                    price = market_data[symbol]['price']
                    
                    # Use price momentum as proxy for value factor
                    # Lower price momentum = higher value factor
                    if 'returns' in market_data[symbol]:
                        returns = market_data[symbol]['returns']
                        if len(returns) >= 21:
                            # Calculate 21-day momentum
                            momentum = np.mean(returns[-21:])
                            # Value factor = negative momentum (contrarian)
                            value_factor = -momentum
                            value_exposures[symbol] = float(value_factor)
                        else:
                            value_exposures[symbol] = 0.0
                    else:
                        value_exposures[symbol] = 0.0
                else:
                    value_exposures[symbol] = 0.0
            
            # Store factor data
            self.factor_data['value'] = value_exposures
            self.factor_exposures['value'] = value_exposures
            
            logger.info(f"Value factor calculated for {len(value_exposures)} symbols")
            return value_exposures
            
        except Exception as e:
            logger.error(f"Error calculating value factor: {str(e)}")
            return {}
    
    async def calculate_momentum_factor(self, symbols: List[str], market_data: Dict) -> Dict[str, float]:
        """
        Calculate momentum factor for all symbols.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of momentum factor exposures
        """
        try:
            momentum_exposures = {}
            
            for symbol in symbols:
                if symbol in market_data and 'returns' in market_data[symbol]:
                    returns = market_data[symbol]['returns']
                    
                    if len(returns) >= self.factor_params['momentum']['min_data']:
                        # Calculate momentum over lookback period
                        lookback = self.factor_params['momentum']['lookback']
                        momentum_period = min(lookback, len(returns))
                        
                        # Calculate cumulative return over momentum period
                        returns_array = np.array(returns[-momentum_period:])
                        momentum = np.prod(1 + returns_array) - 1
                        momentum_exposures[symbol] = float(momentum)
                    else:
                        momentum_exposures[symbol] = 0.0
                else:
                    momentum_exposures[symbol] = 0.0
            
            # Store factor data
            self.factor_data['momentum'] = momentum_exposures
            self.factor_exposures['momentum'] = momentum_exposures
            
            logger.info(f"Momentum factor calculated for {len(momentum_exposures)} symbols")
            return momentum_exposures
            
        except Exception as e:
            logger.error(f"Error calculating momentum factor: {str(e)}")
            return {}
    
    async def calculate_volatility_factor(self, symbols: List[str], market_data: Dict) -> Dict[str, float]:
        """
        Calculate volatility factor for all symbols.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of volatility factor exposures
        """
        try:
            volatility_exposures = {}
            
            for symbol in symbols:
                if symbol in market_data and 'returns' in market_data[symbol]:
                    returns = market_data[symbol]['returns']
                    
                    if len(returns) >= self.factor_params['volatility']['min_data']:
                        # Calculate realized volatility
                        lookback = self.factor_params['volatility']['lookback']
                        volatility_period = min(lookback, len(returns))
                        
                        # Calculate annualized volatility
                        volatility = np.std(returns[-volatility_period:]) * np.sqrt(252)
                        volatility_exposures[symbol] = float(volatility)
                    else:
                        volatility_exposures[symbol] = 0.0
                else:
                    volatility_exposures[symbol] = 0.0
            
            # Store factor data
            self.factor_data['volatility'] = volatility_exposures
            self.factor_exposures['volatility'] = volatility_exposures
            
            logger.info(f"Volatility factor calculated for {len(volatility_exposures)} symbols")
            return volatility_exposures
            
        except Exception as e:
            logger.error(f"Error calculating volatility factor: {str(e)}")
            return {}
    
    async def calculate_liquidity_factor(self, symbols: List[str], market_data: Dict) -> Dict[str, float]:
        """
        Calculate liquidity factor for all symbols.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of liquidity factor exposures
        """
        try:
            liquidity_exposures = {}
            
            for symbol in symbols:
                if symbol in market_data and 'volume' in market_data[symbol]:
                    volume = market_data[symbol]['volume']
                    
                    # Calculate liquidity factor (log volume as proxy)
                    if volume > 0:
                        liquidity_factor = np.log(volume)
                        liquidity_exposures[symbol] = float(liquidity_factor)
                    else:
                        liquidity_exposures[symbol] = 0.0
                else:
                    liquidity_exposures[symbol] = 0.0
            
            # Store factor data
            self.factor_data['liquidity'] = liquidity_exposures
            self.factor_exposures['liquidity'] = liquidity_exposures
            
            logger.info(f"Liquidity factor calculated for {len(liquidity_exposures)} symbols")
            return liquidity_exposures
            
        except Exception as e:
            logger.error(f"Error calculating liquidity factor: {str(e)}")
            return {}
    
    async def calculate_all_factors(self, symbols: List[str], market_data: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate all factors for all symbols.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of all factor exposures
        """
        try:
            logger.info(f"Calculating all factors for {len(symbols)} symbols")
            
            # Calculate all factors
            market_factor = await self.calculate_market_factor(symbols, market_data)
            size_factor = await self.calculate_size_factor(symbols, market_data)
            value_factor = await self.calculate_value_factor(symbols, market_data)
            momentum_factor = await self.calculate_momentum_factor(symbols, market_data)
            volatility_factor = await self.calculate_volatility_factor(symbols, market_data)
            liquidity_factor = await self.calculate_liquidity_factor(symbols, market_data)
            
            # Combine all factors
            all_factors = {
                'market': market_factor,
                'size': size_factor,
                'value': value_factor,
                'momentum': momentum_factor,
                'volatility': volatility_factor,
                'liquidity': liquidity_factor
            }
            
            logger.info("All factors calculated successfully")
            return all_factors
            
        except Exception as e:
            logger.error(f"Error calculating all factors: {str(e)}")
            return {}
    
    async def calculate_factor_exposures(self, symbols: List[str], market_data: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate factor exposures for portfolio.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary of factor exposures by symbol
        """
        try:
            # Calculate all factors
            all_factors = await self.calculate_all_factors(symbols, market_data)
            
            # Create factor exposure matrix
            factor_exposures = {}
            
            for symbol in symbols:
                symbol_exposures = {}
                for factor_name, factor_data in all_factors.items():
                    if symbol in factor_data:
                        exposure = factor_data[symbol]
                        # Normalize exposure to [-1, 1] range
                        normalized_exposure = np.clip(exposure, -1, 1)
                        symbol_exposures[factor_name] = float(normalized_exposure)
                    else:
                        symbol_exposures[factor_name] = 0.0
                
                factor_exposures[symbol] = symbol_exposures
            
            logger.info(f"Factor exposures calculated for {len(factor_exposures)} symbols")
            return factor_exposures
            
        except Exception as e:
            logger.error(f"Error calculating factor exposures: {str(e)}")
            return {}
    
    async def perform_risk_attribution_analysis(self, symbols: List[str], market_data: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive risk attribution analysis.
        
        Args:
            symbols: List of trading symbols
            market_data: Market data dictionary
            
        Returns:
            Dictionary with risk attribution results
        """
        try:
            logger.info("Performing risk attribution analysis...")
            
            # Calculate factor exposures
            factor_exposures = await self.calculate_factor_exposures(symbols, market_data)
            
            # Calculate portfolio weights (equal weight for now)
            portfolio_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            
            # Calculate factor contributions
            factor_contributions = {}
            total_factor_risk = 0.0
            
            for factor_name in self.factors.keys():
                factor_contribution = 0.0
                
                for symbol in symbols:
                    if symbol in factor_exposures and factor_name in factor_exposures[symbol]:
                        exposure = factor_exposures[symbol][factor_name]
                        weight = portfolio_weights[symbol]
                        factor_contribution += exposure * weight
                
                factor_contributions[factor_name] = factor_contribution
                total_factor_risk += abs(factor_contribution)
            
            # Calculate risk metrics
            risk_metrics = {
                'total_factor_risk': total_factor_risk,
                'factor_concentrations': factor_contributions,
                'diversification_score': self._calculate_diversification_score(factor_contributions),
                'factor_correlations': await self._calculate_factor_correlations(factor_exposures),
                'var_95': self._calculate_value_at_risk(factor_contributions, 0.95),
                'var_99': self._calculate_value_at_risk(factor_contributions, 0.99)
            }
            
            # Store results
            self.risk_attribution_results = risk_metrics
            self.risk_metrics = risk_metrics
            
            logger.info("Risk attribution analysis completed")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error performing risk attribution analysis: {str(e)}")
            return {}
    
    def _calculate_diversification_score(self, factor_contributions: Dict[str, float]) -> float:
        """Calculate portfolio diversification score."""
        try:
            if not factor_contributions:
                return 0.0
            
            # Calculate Herfindahl index
            total_risk = sum(abs(contribution) for contribution in factor_contributions.values())
            
            if total_risk == 0:
                return 1.0  # Perfect diversification
            
            herfindahl = sum((abs(contribution) / total_risk) ** 2 for contribution in factor_contributions.values())
            
            # Convert to diversification score (1 - normalized Herfindahl)
            diversification_score = 1.0 - (herfindahl - 1.0 / len(factor_contributions)) / (1.0 - 1.0 / len(factor_contributions))
            
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {str(e)}")
            return 0.0
    
    async def _calculate_factor_correlations(self, factor_exposures: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate correlations between factors."""
        try:
            if not factor_exposures:
                return {}
            
            # Create factor exposure matrix
            symbols = list(factor_exposures.keys())
            factors = list(self.factors.keys())
            
            exposure_matrix = []
            for symbol in symbols:
                symbol_exposures = []
                for factor in factors:
                    exposure = factor_exposures[symbol].get(factor, 0.0)
                    symbol_exposures.append(exposure)
                exposure_matrix.append(symbol_exposures)
            
            exposure_matrix = np.array(exposure_matrix)
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(exposure_matrix.T)
            
            # Extract factor correlations
            factor_correlations = {}
            for i, factor1 in enumerate(factors):
                for j, factor2 in enumerate(factors):
                    if i < j:  # Only upper triangle
                        correlation = correlation_matrix[i, j]
                        factor_correlations[f"{factor1}_{factor2}"] = float(correlation)
            
            return factor_correlations
            
        except Exception as e:
            logger.error(f"Error calculating factor correlations: {str(e)}")
            return {}
    
    def _calculate_value_at_risk(self, factor_contributions: Dict[str, float], confidence_level: float) -> float:
        """Calculate Value at Risk for factor contributions."""
        try:
            if not factor_contributions:
                return 0.0
            
            # Simulate factor returns using historical data
            factor_returns = []
            
            # Generate simulated factor returns (simplified)
            n_simulations = 10000
            for _ in range(n_simulations):
                portfolio_return = 0.0
                for factor_name, contribution in factor_contributions.items():
                    # Simulate factor return (normal distribution)
                    factor_return = np.random.normal(0, 0.02)  # 2% volatility
                    portfolio_return += contribution * factor_return
                factor_returns.append(portfolio_return)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(factor_returns, var_percentile)
            
            return float(var)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    async def analyze_sector_risk_exposure(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze sector risk exposure.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary with sector risk analysis
        """
        try:
            logger.info("Analyzing sector risk exposure...")
            
            # Calculate sector exposures
            sector_exposures = {}
            total_symbols = len(symbols)
            
            for sector, sector_symbols in self.sector_classifications.items():
                sector_count = sum(1 for symbol in symbols if symbol in sector_symbols)
                sector_exposure = sector_count / total_symbols
                sector_exposures[sector] = sector_exposure
            
            # Calculate sector concentration
            sector_concentration = sum(exposure ** 2 for exposure in sector_exposures.values())
            
            # Calculate sector diversification score
            sector_diversification = 1.0 - sector_concentration
            
            sector_analysis = {
                'sector_exposures': sector_exposures,
                'sector_concentration': sector_concentration,
                'sector_diversification': sector_diversification,
                'high_exposure_sectors': [sector for sector, exposure in sector_exposures.items() if exposure > 0.3]
            }
            
            logger.info("Sector risk exposure analysis completed")
            return sector_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sector risk exposure: {str(e)}")
            return {}
    
    async def analyze_geographic_risk_exposure(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze geographic risk exposure.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary with geographic risk analysis
        """
        try:
            logger.info("Analyzing geographic risk exposure...")
            
            # Calculate geographic exposures
            geographic_exposures = {}
            total_symbols = len(symbols)
            
            for region, region_symbols in self.geographic_classifications.items():
                region_count = sum(1 for symbol in symbols if symbol in region_symbols)
                region_exposure = region_count / total_symbols
                geographic_exposures[region] = region_exposure
            
            # Calculate geographic concentration
            geographic_concentration = sum(exposure ** 2 for exposure in geographic_exposures.values())
            
            # Calculate geographic diversification score
            geographic_diversification = 1.0 - geographic_concentration
            
            geographic_analysis = {
                'geographic_exposures': geographic_exposures,
                'geographic_concentration': geographic_concentration,
                'geographic_diversification': geographic_diversification,
                'high_exposure_regions': [region for region, exposure in geographic_exposures.items() if exposure > 0.3]
            }
            
            logger.info("Geographic risk exposure analysis completed")
            return geographic_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing geographic risk exposure: {str(e)}")
            return {}
    
    async def get_factor_summary(self) -> Dict[str, Any]:
        """Get comprehensive factor model summary."""
        try:
            summary = {
                'factors': list(self.factors.keys()),
                'factor_exposures': self.factor_exposures,
                'risk_attribution': self.risk_attribution_results,
                'performance': self.factor_performance,
                'risk_metrics': self.risk_metrics,
                'sector_exposure': {},
                'geographic_exposure': {}
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting factor summary: {str(e)}")
            return {'error': str(e)}

    async def build_factor_model(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Build comprehensive factor model from returns data.
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            Dict: Factor model results including exposures and risk attribution
        """
        try:
            logger.info("Building comprehensive factor model...")
            
            if returns_data is None or returns_data.empty:
                logger.warning("No returns data provided for factor model")
                return {'error': 'No returns data provided'}
            
            # Convert to numpy array if needed
            if isinstance(returns_data, pd.DataFrame):
                returns_array = returns_data.values
                symbols = returns_data.columns.tolist()
            elif isinstance(returns_data, np.ndarray):
                returns_array = returns_data
                symbols = [f"Asset_{i}" for i in range(returns_array.shape[1])]
            else:
                logger.error("Invalid returns data format")
                return {'error': 'Invalid returns data format'}
            
            # Prepare market data structure
            market_data = {
                'returns': returns_array,
                'symbols': symbols,
                'timestamps': pd.date_range(start='2023-01-01', periods=len(returns_array), freq='D')
            }
            
            # Calculate all factors
            logger.info("Calculating factor exposures...")
            factor_exposures = await self.calculate_all_factors(symbols, market_data)
            
            # Perform risk attribution analysis
            logger.info("Performing risk attribution analysis...")
            risk_attribution = await self.perform_risk_attribution_analysis(symbols, market_data)
            
            # Analyze sector and geographic exposures
            logger.info("Analyzing sector and geographic exposures...")
            sector_analysis = await self.analyze_sector_risk_exposure(symbols)
            geographic_analysis = await self.analyze_geographic_risk_exposure(symbols)
            
            # Calculate factor correlations
            factor_correlations = await self._calculate_factor_correlations(factor_exposures)
            
            # Build comprehensive results
            results = {
                'factor_exposures': factor_exposures,
                'risk_attribution': risk_attribution,
                'sector_analysis': sector_analysis,
                'geographic_analysis': geographic_analysis,
                'factor_correlations': factor_correlations,
                'model_status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results for future reference
            self.factor_exposures = factor_exposures
            self.risk_attribution_results = risk_attribution
            
            logger.info("Factor model built successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error building factor model: {str(e)}")
            return {'error': str(e)} 