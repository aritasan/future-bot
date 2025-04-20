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

# Signal score weights
SIGNAL_SCORE_WEIGHTS = {
    "technical": 0.3,
    "market": 0.2,
    "timeframe": 0.2,
    "btc": 0.15,
    "sentiment": 0.15
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

# Trading pairs
TRADING_PAIRS = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'ADA/USDT',
    'DOGE/USDT',
    'XRP/USDT',
    'DOT/USDT',
    'UNI/USDT',
    'LINK/USDT',
    'LTC/USDT'
]

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
                'max_drawdown': MAX_DRAWDOWN,
                'atr_period': ATR_PERIOD,
                'max_orders_per_symbol': MAX_ORDERS_PER_SYMBOL,
                'risk_per_trade': RISK_PER_TRADE,
                'min_volume_ratio': MIN_VOLUME_RATIO,
                'max_volatility_ratio': MAX_VOLATILITY_RATIO,
                'min_adx': MIN_ADX,
                'max_bb_width': MAX_BB_WIDTH,
                'timeframe_weights': TIMEFRAME_WEIGHTS,
                'signal_score_weights': SIGNAL_SCORE_WEIGHTS,
                'drawdown_warning_levels': DRAWDOWN_WARNING_LEVELS,
                'trading_pairs': TRADING_PAIRS,
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
                'position_mode': "hedge",
                'risk_management': {
                    'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.1')),
                    'max_leverage': int(os.getenv('MAX_LEVERAGE', '20')),
                    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.01')),
                    'min_stop_distance': float(os.getenv('MIN_STOP_DISTANCE', '0.005')),
                    'take_profit_multiplier': float(os.getenv('TAKE_PROFIT_MULTIPLIER', '2.0'))
                },
                'strategy': {
                    'timeframes': ['5m', '15m', '1h', '4h'],
                    'indicators': {
                        'rsi': {'period': 14},
                        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                        'bollinger': {'period': 20, 'std': 2},
                        'atr': {'period': 14}
                    }
                }
            },
            'risk_management': {
                'base_stop_distance': BASE_STOP_DISTANCE,
                'volatility_multiplier': VOLATILITY_MULTIPLIER,
                'trend_multiplier': TREND_MULTIPLIER,
                'take_profit_multiplier': TAKE_PROFIT_MULTIPLIER,
                'dca_multiplier': DCA_MULTIPLIER,
                'atr_multiplier': ATR_MULTIPLIER,
                'risk_per_trade': RISK_PER_TRADE,
                'max_drawdown': MAX_DRAWDOWN,
                'max_open_positions': 5,
                'stop_loss_multiplier': 1.5,
                'min_stop_distance': MIN_STOP_DISTANCE,
                'min_tp_distance': MIN_TP_DISTANCE,
                'trailing_stop_activation': TRAILING_STOP_ACTIVATION,
                'trailing_stop_distance': TRAILING_STOP_DISTANCE,
                'dca_enabled': DCA_ENABLED,
                'max_dca_attempts': MAX_DCA_ATTEMPTS,
                'dca_distance': DCA_DISTANCE,
                'dca_multiplier': DCA_MULTIPLIER,
                'max_position_size': 0.1,  # Maximum position size as percentage of account
                'max_leverage': 20,  # Maximum leverage allowed
                'max_drawdown': 0.1,  # Maximum drawdown before stopping trading
                'max_open_positions': 5,  # Maximum number of open positions
                'min_stop_distance': 0.005,  # Minimum stop loss distance (0.5%)
                'take_profit_multiplier': 2.0,  # Take profit distance multiplier relative to stop loss
                'trailing_stop_update_interval': 60,  # Update trailing stop every 60 seconds
                'break_even_min_profit': 0.01,  # Minimum 1% profit to move to break-even
                'break_even_min_time': 300,  # Minimum 5 minutes to move to break-even
                'partial_profit_min_profit': 0.02,  # Minimum 2% profit to take partial profit
                'partial_profit_min_time': 600,  # Minimum 10 minutes to take partial profit
                'partial_profit_close_ratio': 0.5,  # Close 50% of position when taking partial profit
                'emergency_stop_volatility_threshold': 0.03,  # 3% volatility threshold
                'emergency_stop_volume_threshold': 2.0,  # 2x average volume threshold
                'emergency_stop_trend_threshold': 0.7,  # 70% trend strength threshold
                'emergency_stop_distance': 0.02  # 2% emergency stop distance
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