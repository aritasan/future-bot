"""
Helper functions for the trading bot.
"""
import logging
from typing import Dict, Any
import ccxt
from collections import defaultdict

logger = logging.getLogger(__name__)

# Constants for Telegram message handling
MESSAGE_COOLDOWN = 60  # seconds
message_history = defaultdict(float)
last_message_content = None
last_message_time = 0

def normalize_symbol(symbol: str) -> str:
    """Normalize symbol format"""
    return symbol.upper().replace('-', '/')

def get_exchange_info() -> Dict[str, Any]:
    """
    Get exchange information from Binance.
    """
    try:
        # Initialize Binance client
        client = ccxt.binance()
        
        # Get exchange info
        info = client.fetch_markets()
        return info
        
    except Exception as e:
        logger.error(f"Error getting exchange info: {e}")
        return {} 
    
def is_same_symbol(symbol1: str, symbol2: str) -> bool:
    """Check if two symbols are the same"""
    symbol1 = symbol1.split(':')[0].replace('/', '')
    symbol2 = symbol2.split(':')[0].replace('/', '')
    return symbol1.lower() == symbol2.lower()

def is_same_side(side1: str, side2: str) -> bool:
    """Check if two sides are the same"""
    is_side1_long = is_long_side(side1)
    if is_side1_long:
        return is_long_side(side2)
    
    return is_short_side(side2)

def is_long_side(side: str) -> bool:
    """Check if a side is long"""
    return side.upper() == "LONG" or side.upper() == "BUY"

def is_short_side(side: str) -> bool:
    """Check if a side is short"""
    return side.upper() == "SHORT" or side.upper() == "SELL"
