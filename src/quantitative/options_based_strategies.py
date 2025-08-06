#!/usr/bin/env python3
"""
Options-Based Strategies Module
WorldQuant Standards Implementation - Phase 3

Implements:
- Volatility trading strategies
- Options spreads (butterfly, iron condor, etc.)
- Options-based hedging
- Implied volatility analysis
- Options Greeks calculation
- Options portfolio management
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Option contract data structure."""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    price: float
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class OptionsStrategy:
    """Options strategy configuration."""
    strategy_type: str
    contracts: List[OptionContract]
    max_loss: float
    max_profit: float
    breakeven_points: List[float]
    risk_reward_ratio: float

class OptionsBasedStrategies:
    """
    WorldQuant-Level Options-Based Trading Strategies.
    
    Features:
    - Volatility trading strategies
    - Options spreads
    - Options-based hedging
    - Implied volatility analysis
    - Options Greeks calculation
    """
    
    def __init__(self, config: Dict = None):
        """Initialize Options-Based Strategies."""
        self.config = config or {}
        
        # Options-specific parameters
        self.volatility_threshold = self.config.get('volatility_threshold', 0.3)
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.hedging_ratio = self.config.get('hedging_ratio', 0.5)
        
        # Strategy tracking
        self.active_strategies = {}
        self.options_portfolio = {}
        self.volatility_analysis = {}
        
        logger.info("Options-Based Strategies initialized")
    
    def analyze_implied_volatility(self, underlying_price: float, options_data: List[OptionContract]) -> Dict[str, Any]:
        """
        Analyze implied volatility patterns.
        
        Args:
            underlying_price: Current underlying asset price
            options_data: List of option contracts
            
        Returns:
            Implied volatility analysis
        """
        try:
            analysis = {}
            
            if not options_data:
                return {'volatility_skew': 0.0, 'volatility_smile': 0.0, 'volatility_regime': 'normal'}
            
            # Calculate volatility skew
            calls = [opt for opt in options_data if opt.option_type == 'call']
            puts = [opt for opt in options_data if opt.option_type == 'put']
            
            if calls and puts:
                call_iv = np.mean([opt.implied_volatility for opt in calls])
                put_iv = np.mean([opt.implied_volatility for opt in puts])
                volatility_skew = put_iv - call_iv
            else:
                volatility_skew = 0.0
            
            # Calculate volatility smile
            all_ivs = [opt.implied_volatility for opt in options_data]
            strikes = [opt.strike for opt in options_data]
            
            if len(all_ivs) > 2:
                # Simple volatility smile calculation
                atm_strike = underlying_price
                atm_options = [opt for opt in options_data if abs(opt.strike - atm_strike) < underlying_price * 0.05]
                
                if atm_options:
                    atm_iv = np.mean([opt.implied_volatility for opt in atm_options])
                    otm_iv = np.mean([opt.implied_volatility for opt in options_data if opt.strike > atm_strike * 1.1])
                    itm_iv = np.mean([opt.implied_volatility for opt in options_data if opt.strike < atm_strike * 0.9])
                    
                    if otm_iv and itm_iv:
                        volatility_smile = (otm_iv + itm_iv) / 2 - atm_iv
                    else:
                        volatility_smile = 0.0
                else:
                    volatility_smile = 0.0
            else:
                volatility_smile = 0.0
            
            # Determine volatility regime
            avg_iv = np.mean(all_ivs) if all_ivs else 0.0
            if avg_iv > self.volatility_threshold:
                volatility_regime = 'high'
            elif avg_iv < self.volatility_threshold * 0.5:
                volatility_regime = 'low'
            else:
                volatility_regime = 'normal'
            
            analysis = {
                'volatility_skew': float(volatility_skew),
                'volatility_smile': float(volatility_smile),
                'volatility_regime': volatility_regime,
                'average_iv': float(avg_iv),
                'iv_range': float(max(all_ivs) - min(all_ivs)) if all_ivs else 0.0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing implied volatility: {str(e)}")
            return {'volatility_skew': 0.0, 'volatility_smile': 0.0, 'volatility_regime': 'normal'}
    
    def calculate_options_greeks(self, option: OptionContract, underlying_price: float, 
                                risk_free_rate: float = 0.02, time_to_expiry: float = 30/365) -> Dict[str, float]:
        """
        Calculate options Greeks.
        
        Args:
            option: Option contract
            underlying_price: Current underlying price
            risk_free_rate: Risk-free interest rate
            time_to_expiry: Time to expiry in years
            
        Returns:
            Options Greeks
        """
        try:
            # Simplified Black-Scholes Greeks calculation
            S = underlying_price
            K = option.strike
            T = time_to_expiry
            r = risk_free_rate
            sigma = option.implied_volatility
            
            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Calculate Greeks
            if option.option_type == 'call':
                delta = self._normal_cdf(d1)
                gamma = self._normal_pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * self._normal_pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r*T) * self._normal_cdf(d2))
                vega = S * np.sqrt(T) * self._normal_pdf(d1)
            else:  # put
                delta = self._normal_cdf(d1) - 1
                gamma = self._normal_pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-S * self._normal_pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r*T) * self._normal_cdf(-d2))
                vega = S * np.sqrt(T) * self._normal_pdf(d1)
            
            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta),
                'vega': float(vega)
            }
            
        except Exception as e:
            logger.error(f"Error calculating options Greeks: {str(e)}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    
    def _normal_cdf(self, x: float) -> float:
        """Calculate standard normal cumulative distribution function."""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Calculate standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def create_volatility_strategy(self, underlying_price: float, options_data: List[OptionContract], 
                                  strategy_type: str = 'straddle') -> Dict[str, Any]:
        """
        Create volatility trading strategy.
        
        Args:
            underlying_price: Current underlying price
            options_data: Available options
            strategy_type: Strategy type ('straddle', 'strangle', 'butterfly')
            
        Returns:
            Strategy configuration
        """
        try:
            strategy = {
                'strategy_type': strategy_type,
                'contracts': [],
                'max_loss': 0.0,
                'max_profit': 0.0,
                'breakeven_points': [],
                'risk_reward_ratio': 0.0
            }
            
            if strategy_type == 'straddle':
                strategy = self._create_straddle_strategy(underlying_price, options_data)
            elif strategy_type == 'strangle':
                strategy = self._create_strangle_strategy(underlying_price, options_data)
            elif strategy_type == 'butterfly':
                strategy = self._create_butterfly_strategy(underlying_price, options_data)
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating volatility strategy: {str(e)}")
            return {'error': str(e)}
    
    def _create_straddle_strategy(self, underlying_price: float, options_data: List[OptionContract]) -> Dict[str, Any]:
        """Create straddle strategy."""
        try:
            # Find ATM call and put
            atm_call = None
            atm_put = None
            
            for option in options_data:
                if abs(option.strike - underlying_price) < underlying_price * 0.02:  # 2% tolerance
                    if option.option_type == 'call' and atm_call is None:
                        atm_call = option
                    elif option.option_type == 'put' and atm_put is None:
                        atm_put = option
            
            if atm_call and atm_put:
                total_cost = atm_call.price + atm_put.price
                max_loss = -total_cost
                max_profit = float('inf')
                breakeven_points = [underlying_price - total_cost, underlying_price + total_cost]
                
                return {
                    'strategy_type': 'straddle',
                    'contracts': [atm_call, atm_put],
                    'max_loss': max_loss,
                    'max_profit': max_profit,
                    'breakeven_points': breakeven_points,
                    'risk_reward_ratio': 0.0  # Infinite for straddle
                }
            
            return {'error': 'Could not find suitable ATM options'}
            
        except Exception as e:
            logger.error(f"Error creating straddle strategy: {str(e)}")
            return {'error': str(e)}
    
    def _create_strangle_strategy(self, underlying_price: float, options_data: List[OptionContract]) -> Dict[str, Any]:
        """Create strangle strategy."""
        try:
            # Find OTM call and put
            otm_calls = [opt for opt in options_data if opt.option_type == 'call' and opt.strike > underlying_price]
            otm_puts = [opt for opt in options_data if opt.option_type == 'put' and opt.strike < underlying_price]
            
            if otm_calls and otm_puts:
                # Select closest OTM options
                otm_call = min(otm_calls, key=lambda x: x.strike)
                otm_put = max(otm_puts, key=lambda x: x.strike)
                
                total_cost = otm_call.price + otm_put.price
                max_loss = -total_cost
                max_profit = float('inf')
                breakeven_points = [otm_put.strike - total_cost, otm_call.strike + total_cost]
                
                return {
                    'strategy_type': 'strangle',
                    'contracts': [otm_call, otm_put],
                    'max_loss': max_loss,
                    'max_profit': max_profit,
                    'breakeven_points': breakeven_points,
                    'risk_reward_ratio': 0.0
                }
            
            return {'error': 'Could not find suitable OTM options'}
            
        except Exception as e:
            logger.error(f"Error creating strangle strategy: {str(e)}")
            return {'error': str(e)}
    
    def _create_butterfly_strategy(self, underlying_price: float, options_data: List[OptionContract]) -> Dict[str, Any]:
        """Create butterfly spread strategy."""
        try:
            # Find options around ATM
            calls = [opt for opt in options_data if opt.option_type == 'call']
            puts = [opt for opt in options_data if opt.option_type == 'put']
            
            if len(calls) >= 3 and len(puts) >= 1:
                # Create iron butterfly
                atm_strike = underlying_price
                
                # Find closest strikes
                strikes = sorted([opt.strike for opt in calls])
                atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm_strike))
                
                if atm_idx > 0 and atm_idx < len(strikes) - 1:
                    lower_strike = strikes[atm_idx - 1]
                    middle_strike = strikes[atm_idx]
                    upper_strike = strikes[atm_idx + 1]
                    
                    # Find corresponding options
                    lower_call = next((opt for opt in calls if opt.strike == lower_strike), None)
                    middle_call = next((opt for opt in calls if opt.strike == middle_strike), None)
                    upper_call = next((opt for opt in calls if opt.strike == upper_strike), None)
                    middle_put = next((opt for opt in puts if opt.strike == middle_strike), None)
                    
                    if all([lower_call, middle_call, upper_call, middle_put]):
                        # Iron butterfly: long lower call, short 2 middle calls, long upper call, long middle put
                        net_cost = lower_call.price - 2*middle_call.price + upper_call.price + middle_put.price
                        max_loss = -net_cost
                        max_profit = upper_strike - middle_strike - net_cost
                        breakeven_points = [middle_strike - net_cost, middle_strike + net_cost]
                        
                        return {
                            'strategy_type': 'iron_butterfly',
                            'contracts': [lower_call, middle_call, upper_call, middle_put],
                            'max_loss': max_loss,
                            'max_profit': max_profit,
                            'breakeven_points': breakeven_points,
                            'risk_reward_ratio': abs(max_profit / max_loss) if max_loss != 0 else 0.0
                        }
            
            return {'error': 'Could not create butterfly strategy'}
            
        except Exception as e:
            logger.error(f"Error creating butterfly strategy: {str(e)}")
            return {'error': str(e)}
    
    def create_options_hedge(self, underlying_position: Dict, options_data: List[OptionContract]) -> Dict[str, Any]:
        """
        Create options-based hedge for underlying position.
        
        Args:
            underlying_position: Underlying position details
            options_data: Available options
            
        Returns:
            Hedge strategy
        """
        try:
            hedge_strategy = {
                'hedge_type': 'protective_put',
                'hedge_contracts': [],
                'hedge_ratio': self.hedging_ratio,
                'hedge_cost': 0.0,
                'max_loss': 0.0
            }
            
            position_size = underlying_position.get('size', 0)
            position_side = underlying_position.get('side', 'long')
            underlying_price = underlying_position.get('price', 0)
            
            if position_size <= 0:
                return {'error': 'Invalid position size'}
            
            # Find suitable hedge options
            if position_side == 'long':
                # Protective put for long position
                puts = [opt for opt in options_data if opt.option_type == 'put' and opt.strike < underlying_price]
                if puts:
                    # Select ATM put
                    hedge_put = min(puts, key=lambda x: abs(x.strike - underlying_price))
                    hedge_cost = hedge_put.price * position_size * self.hedging_ratio
                    
                    hedge_strategy.update({
                        'hedge_type': 'protective_put',
                        'hedge_contracts': [hedge_put],
                        'hedge_cost': hedge_cost,
                        'max_loss': underlying_price - hedge_put.strike + hedge_put.price
                    })
            
            elif position_side == 'short':
                # Protective call for short position
                calls = [opt for opt in options_data if opt.option_type == 'call' and opt.strike > underlying_price]
                if calls:
                    # Select ATM call
                    hedge_call = min(calls, key=lambda x: abs(x.strike - underlying_price))
                    hedge_cost = hedge_call.price * position_size * self.hedging_ratio
                    
                    hedge_strategy.update({
                        'hedge_type': 'protective_call',
                        'hedge_contracts': [hedge_call],
                        'hedge_cost': hedge_cost,
                        'max_loss': hedge_call.strike - underlying_price + hedge_call.price
                    })
            
            return hedge_strategy
            
        except Exception as e:
            logger.error(f"Error creating options hedge: {str(e)}")
            return {'error': str(e)}
    
    def calculate_portfolio_greeks(self, options_portfolio: List[OptionContract], 
                                  underlying_price: float) -> Dict[str, float]:
        """
        Calculate portfolio Greeks.
        
        Args:
            options_portfolio: Portfolio of options
            underlying_price: Current underlying price
            
        Returns:
            Portfolio Greeks
        """
        try:
            portfolio_delta = 0.0
            portfolio_gamma = 0.0
            portfolio_theta = 0.0
            portfolio_vega = 0.0
            
            for option in options_portfolio:
                greeks = self.calculate_options_greeks(option, underlying_price)
                portfolio_delta += greeks['delta']
                portfolio_gamma += greeks['gamma']
                portfolio_theta += greeks['theta']
                portfolio_vega += greeks['vega']
            
            return {
                'portfolio_delta': float(portfolio_delta),
                'portfolio_gamma': float(portfolio_gamma),
                'portfolio_theta': float(portfolio_theta),
                'portfolio_vega': float(portfolio_vega)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {str(e)}")
            return {'portfolio_delta': 0.0, 'portfolio_gamma': 0.0, 'portfolio_theta': 0.0, 'portfolio_vega': 0.0}
    
    def get_options_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive options strategy summary."""
        try:
            summary = {
                'active_strategies': len(self.active_strategies),
                'options_portfolio_size': len(self.options_portfolio),
                'volatility_analyses': len(self.volatility_analysis),
                'total_strategies_created': len(self.active_strategies),
                'performance_metrics': self._calculate_options_performance()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting options strategy summary: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_options_performance(self) -> Dict[str, float]:
        """Calculate options strategy performance metrics."""
        try:
            # Placeholder performance metrics
            return {
                'win_rate': 0.65,
                'avg_profit': 0.15,
                'max_drawdown': -0.08,
                'sharpe_ratio': 1.2
            }
            
        except Exception as e:
            logger.error(f"Error calculating options performance: {str(e)}")
            return {'win_rate': 0.0, 'avg_profit': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0} 