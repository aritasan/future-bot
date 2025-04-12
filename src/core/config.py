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

# Drawdown warning levels
DRAWDOWN_WARNING_LEVELS = [0.5, 0.7, 0.9]

# API settings
BINANCE_API_KEY = os.getenv("BINANCE_MAINNET_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_MAINNET_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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
DEFAULT_LEVERAGE = 1
MAX_LEVERAGE = 20
MIN_ORDER_SIZE = 10  # USDT

# Risk management
MAX_POSITION_SIZE = 0.1  # 10% of account balance
MAX_CORRELATION = 0.7    # Maximum allowed correlation between positions
MAX_VOLATILITY = 0.02    # Maximum allowed volatility (2%)

def load_config() -> Dict[str, Any]:
    """
    Load and return the configuration settings.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return {
        "trading": {
            "order_risk_percent": ORDER_RISK_PERCENT,
            "max_drawdown": MAX_DRAWDOWN,
            "atr_period": ATR_PERIOD,
            "max_orders_per_symbol": MAX_ORDERS_PER_SYMBOL,
            "drawdown_warning_levels": DRAWDOWN_WARNING_LEVELS,
            "default_timeframe": DEFAULT_TIMEFRAME,
            "default_leverage": DEFAULT_LEVERAGE,
            "max_leverage": MAX_LEVERAGE,
            "min_order_size": MIN_ORDER_SIZE
        },
        "risk_management": {
            "max_position_size": MAX_POSITION_SIZE,
            "max_correlation": MAX_CORRELATION,
            "max_volatility": MAX_VOLATILITY
        },
        "api": {
            "binance": {
                "api_key": BINANCE_API_KEY,
                "api_secret": BINANCE_API_SECRET
            },
            "telegram": {
                "bot_token": TELEGRAM_BOT_TOKEN,
                "chat_id": TELEGRAM_CHAT_ID
            }
        },
        "cache": {
            "price_cache_ttl": PRICE_CACHE_TTL,
            "position_cache_ttl": POSITION_CACHE_TTL
        },
        "circuit_breaker": {
            "failure_threshold": FAILURE_THRESHOLD,
            "reset_timeout": RESET_TIMEOUT
        },
        "health": {
            "cpu_threshold": CPU_THRESHOLD,
            "memory_threshold": MEMORY_THRESHOLD,
            "disk_threshold": DISK_THRESHOLD
        },
        "trading_pairs": TRADING_PAIRS,
        "timeframes": TIMEFRAMES,
        "model_dir": MODEL_DIR,
        "positions_file": POSITIONS_FILE
    } 