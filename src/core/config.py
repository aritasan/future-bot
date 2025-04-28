"""
Configuration settings for the trading bot.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading parameters
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "10.0"))  # 10% max drawdown
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
MAX_ORDERS_PER_SYMBOL = int(os.getenv("MAX_ORDERS_PER_SYMBOL", "3"))

# Position sizing parameters
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))  # 2% risk per trade

# Technical analysis parameters
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", "1.2"))
MAX_VOLATILITY_RATIO = float(os.getenv("MAX_VOLATILITY_RATIO", "2.0"))
MIN_ADX = float(os.getenv("MIN_ADX", "25"))
MAX_BB_WIDTH = float(os.getenv("MAX_BB_WIDTH", "0.1"))

# Timeframe weights
TIMEFRAME_WEIGHTS = {
    "5m": 0.2,
    "15m": 0.3,
    "1h": 0.3,
    "4h": 0.2
}


# Drawdown warning levels
DRAWDOWN_WARNING_LEVELS = [0.5, 0.7, 0.9]

# API settings
BINANCE_API_KEY = os.getenv("BINANCE_MAINNET_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_MAINNET_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID"))  # Convert to string to handle negative IDs

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Cache settings
PRICE_CACHE_TTL = 60  # seconds
POSITION_CACHE_TTL = 300  # seconds

# Circuit breaker settings
FAILURE_THRESHOLD = 5
RESET_TIMEOUT = 300  # seconds

# Health monitor settings
CPU_THRESHOLD = 80  # percent
MEMORY_THRESHOLD = 80  # percent
DISK_THRESHOLD = 80  # percent


# Timeframes
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']

# Model paths
MODEL_DIR = 'models'
POSITIONS_FILE = 'positions.json'

# Trading settings
DEFAULT_TIMEFRAME = "5m"
DEFAULT_LEVERAGE = 10
MAX_LEVERAGE = 20
MIN_ORDER_SIZE = 10  # USDT
PRICE_PRECISION = 8  # Number of decimal places for price values

# Risk management
MAX_CORRELATION = 0.7    # Maximum allowed correlation between positions
MAX_VOLATILITY = 0.02    # Maximum allowed volatility (2%)

# Risk management parameters
BASE_STOP_DISTANCE = float(os.getenv("BASE_STOP_DISTANCE", "0.02"))  # 2% base stop distance
VOLATILITY_MULTIPLIER = float(os.getenv("VOLATILITY_MULTIPLIER", "1.5"))  # Volatility adjustment factor
TREND_MULTIPLIER = float(os.getenv("TREND_MULTIPLIER", "1.2"))  # Trend strength adjustment factor
TAKE_PROFIT_MULTIPLIER = float(os.getenv("TAKE_PROFIT_MULTIPLIER", "2.0"))  # Risk:Reward ratio
DCA_MULTIPLIER = float(os.getenv("DCA_MULTIPLIER", "0.5"))  # DCA size relative to initial position
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.5"))  # ATR multiplier for dynamic stops
STOP_LOSS_ATR_MULTIPLIER = float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "1.5"))  # ATR multiplier for stop loss calculation

# New risk management parameters
MIN_STOP_DISTANCE = float(os.getenv("MIN_STOP_DISTANCE", "0.005"))  # 0.5% minimum distance for stop loss
MIN_TP_DISTANCE = float(os.getenv("MIN_TP_DISTANCE", "0.01"))     # 1% minimum distance for take profit
TRAILING_STOP_ACTIVATION = float(os.getenv("TRAILING_STOP_ACTIVATION", "0.02"))
TRAILING_STOP_DISTANCE = float(os.getenv("TRAILING_STOP_DISTANCE", "0.01"))
DCA_ENABLED = bool(os.getenv("DCA_ENABLED", "true"))
MAX_DCA_ATTEMPTS = int(os.getenv("MAX_DCA_ATTEMPTS", "3"))
DCA_DISTANCE = float(os.getenv("DCA_DISTANCE", "0.02"))

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and config file."""
    try:
        # Load environment variables
        config = {
            'api': {
                'binance': {
                    'use_testnet': os.getenv('USE_TESTNET', 'false').lower() == 'true',
                    'mainnet': {
                        'api_key': os.getenv('BINANCE_MAINNET_API_KEY'),
                        'api_secret': os.getenv('BINANCE_MAINNET_API_SECRET')
                    },
                    'testnet': {
                        'api_key': os.getenv('BINANCE_TESTNET_API_KEY'),
                        'api_secret': os.getenv('BINANCE_TESTNET_API_SECRET')
                    }
                },
                'telegram': {
                    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                    'chat_id': os.getenv('TELEGRAM_CHAT_ID')
                },
                'twitter': {
                    'api_key': TWITTER_API_KEY,
                    'api_secret': TWITTER_API_SECRET,
                    'access_token': TWITTER_ACCESS_TOKEN,
                    'access_token_secret': TWITTER_ACCESS_TOKEN_SECRET
                },
                'news': {
                    'api_key': NEWS_API_KEY
                }
            },
            'trading': {
                'min_volume': 1000000,
                'min_volatility': 0.02,
                'max_btc_correlation': 0.8,
                'max_drawdown': MAX_DRAWDOWN,
                'atr_period': ATR_PERIOD,
                'max_orders_per_symbol': MAX_ORDERS_PER_SYMBOL,
                'risk_per_trade': RISK_PER_TRADE,
                'min_volume_ratio': MIN_VOLUME_RATIO,
                'max_volatility_ratio': MAX_VOLATILITY_RATIO,
                'min_adx': MIN_ADX,
                'max_bb_width': MAX_BB_WIDTH,
                'timeframe_weights': TIMEFRAME_WEIGHTS,
                'drawdown_warning_levels': DRAWDOWN_WARNING_LEVELS,
                'timeframes': TIMEFRAMES,
                'default_timeframe': DEFAULT_TIMEFRAME,
                'default_leverage': DEFAULT_LEVERAGE,
                'max_leverage': MAX_LEVERAGE,
                'min_order_size': MIN_ORDER_SIZE,
                'max_correlation': MAX_CORRELATION,
                'price_precision': PRICE_PRECISION,
                'amount_precision': PRICE_PRECISION,
                'max_order_size': 1000,
                'leverage': DEFAULT_LEVERAGE,
                'buy_threshold': 0.25,
                'sell_threshold': -0.25,
                'position_mode': "hedge"
            },
            'risk_management': {
                # General risk parameters
                'max_risk_per_trade': 0.02,  # 2% of account balance
                'max_risk_per_position': 0.05,  # 5% of account balance
                'min_stop_distance': 0.01,  # 0.5% minimum stop distance
                'min_tp_distance': 0.03,  # 1% minimum take profit distance
                'take_profit_multiplier': 2.0,  # R:R ratio
                
                # Stop loss calculation parameters
                'atr_multiplier': 1.5,  # ATR multiplier for stop loss calculation
                'volatility_multiplier': 1.5,  # Volatility adjustment factor
                'trend_multiplier': 1.2,  # Trend strength adjustment factor
                'base_stop_distance': 0.02,  # Base stop distance (2%)
                'stop_loss_atr_multiplier': 1.5,  # ATR multiplier for stop loss calculation
                
                # DCA parameters
                'dca': {
                    'enabled': True,
                    'max_attempts': 3,  # Maximum number of DCA attempts per position
                    'risk_reduction': 0.5,  # Reduce risk by 50% for each DCA attempt
                    'price_drop_thresholds': [0.02, 0.05, 0.1],  # Price drop thresholds
                    'volume_threshold': 1.5,  # Minimum volume ratio
                    'volatility_threshold': 0.02,  # Maximum volatility
                    'rsi_thresholds': {
                        'oversold': 30,
                        'overbought': 70
                    },
                    'min_time_between_attempts': 3600,  # Minimum 1 hour between attempts
                    'max_positions': 5,  # Maximum number of positions in DCA
                    'btc_correlation_threshold': 0.7,  # Minimum BTC correlation
                    
                    # New time-based adjustment parameters
                    'time_based_adjustment': {
                        'enabled': True,
                        'time_windows': [3600, 7200, 14400],  # 1h, 2h, 4h
                        'size_multipliers': [1.0, 0.8, 0.6]   # Reduce DCA size over time
                    },
                    
                    # New risk control parameters
                    'risk_control': {
                        'max_drawdown': 0.1,  # 10% max drawdown for DCA
                        'max_position_size': 0.2,  # 20% of account
                        'min_profit_target': 0.03  # 3% minimum profit target
                    }
                },
                
                # Trailing stop parameters
                'trailing_stop': {
                    'update_interval': 600,  # Update every 600 seconds
                    'break_even': {
                        'min_profit': 0.1,  # Minimum 10% profit
                        'min_time': 300  # Minimum 5 minutes
                    },
                    'partial_profit': {
                        'min_profit': 0.2,  # Minimum 20% profit
                        'min_time': 600,  # Minimum 10 minutes
                        'close_ratio': 0.5  # Close 50% of position
                    },
                    
                    # New dynamic trailing stop parameters
                    'dynamic': {
                        'enabled': True,
                        'atr_multiplier': 1.5,
                        'volatility_adjustment': True,
                        'trend_adjustment': True
                    },
                    
                    # New time-based parameters
                    'time_based': {
                        'enabled': True,
                        'time_windows': [1800, 3600, 7200],  # 30m, 1h, 2h
                        'distance_multipliers': [1.0, 1.2, 1.5]  # Increase distance over time
                    }
                },
                
                # Emergency stop parameters
                'emergency_stop': {
                    'volatility_threshold': 0.1,  # 10% volatility threshold
                    'volume_threshold': 2.0,  # 2x average volume threshold
                    'distance': 0.02  # 2% emergency stop distance
                },

                # New order frequency control parameters
                'order_frequency': {
                    'min_time_between_orders': 300,  # 5 minutes minimum between orders
                    'max_orders_per_hour': 12,  # Maximum 12 orders per hour
                    'max_orders_per_day': 50,  # Maximum 50 orders per day
                    'cool_down_period': 3600,  # 1 hour cool down if limits exceeded
                    'time_windows': {
                        'hourly': 3600,
                        'daily': 86400
                    }
                }
            },
            'cache': {
                'price_ttl': PRICE_CACHE_TTL,
                'position_ttl': POSITION_CACHE_TTL
            },
            'circuit_breaker': {
                'failure_threshold': FAILURE_THRESHOLD,
                'reset_timeout': RESET_TIMEOUT
            },
            'health_monitor': {
                'cpu_threshold': CPU_THRESHOLD,
                'memory_threshold': MEMORY_THRESHOLD,
                'disk_threshold': DISK_THRESHOLD
            },
            'model': {
                'dir': MODEL_DIR,
                'positions_file': POSITIONS_FILE
            }
        }
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {} 