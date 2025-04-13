"""
Configuration settings for the trading bot.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Trading parameters
ORDER_RISK_PERCENT = float(os.getenv("ORDER_RISK_PERCENT", "1.0"))  # 100% risk per trade
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
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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

# Risk management
MAX_CORRELATION = 0.7    # Maximum allowed correlation between positions
MAX_VOLATILITY = 0.02    # Maximum allowed volatility (2%)

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables and return as dictionary.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = {
        "api": {
            "binance": {
                "api_key": BINANCE_API_KEY,
                "api_secret": BINANCE_API_SECRET
            },
            "telegram": {
                "bot_token": TELEGRAM_BOT_TOKEN,
                "chat_id": TELEGRAM_CHAT_ID
            },
            "twitter": {
                "api_key": TWITTER_API_KEY,
                "api_secret": TWITTER_API_SECRET,
                "access_token": TWITTER_ACCESS_TOKEN,
                "access_token_secret": TWITTER_ACCESS_TOKEN_SECRET
            },
            "news": {
                "api_key": NEWS_API_KEY
            }
        },
        "trading": {
            "risk_per_trade": RISK_PER_TRADE,
            "min_volume_ratio": MIN_VOLUME_RATIO,
            "max_volatility_ratio": MAX_VOLATILITY_RATIO,
            "min_adx": MIN_ADX,
            "max_bb_width": MAX_BB_WIDTH,
            "timeframe_weights": TIMEFRAME_WEIGHTS,
            "signal_score_weights": SIGNAL_SCORE_WEIGHTS,
            "leverage": DEFAULT_LEVERAGE
        },
        "health_monitor": {
            "max_errors": FAILURE_THRESHOLD,
            "cpu_threshold": CPU_THRESHOLD,
            "memory_threshold": MEMORY_THRESHOLD,
            "disk_threshold": DISK_THRESHOLD
        },
        "risk_management": {
            "max_correlation": MAX_CORRELATION,
            "max_volatility": MAX_VOLATILITY
        },
        "trading_pairs": TRADING_PAIRS
    }
    
    return config 