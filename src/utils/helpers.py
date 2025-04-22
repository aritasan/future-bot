"""
Helper functions for the trading bot.
"""
import logging
from typing import Dict, Any
import ccxt
import requests
from datetime import datetime, timedelta
import os
import json
import time
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

def get_min_order_size(symbol: str) -> float:
    """Get minimum order size for a symbol"""
    try:
        # Initialize Binance client
        client = ccxt.binance()
        
        # Get exchange info
        exchange_info = client.get_exchange_info()
        
        # Find the symbol info
        for symbol_info in exchange_info['symbols']:
            if is_same_symbol(symbol_info['symbol'], symbol):
                # Get the minimum order size from the symbol info
                min_qty = float(symbol_info['filters'][1]['minQty'])
                return min_qty
                
        logger.warning(f"No min order size found for {symbol}, using default")
        return 0.001  # Default minimum
        
    except Exception as e:
        logger.error(f"Error getting min order size for {symbol}: {e}")
        return 0.001  # Safe default

def send_telegram_message(message: str) -> bool:
    """
    Send message to Telegram with duplicate prevention and message chunking
    
    Args:
        message (str): The message to send
        
    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.error("Telegram credentials not configured")
            return False
            
        current_time = time.time()
        message_key = hash(message)  # Create unique key for message
        
        # Check if message was recently sent
        if message_key in message_history:
            last_sent_time = message_history[message_key]
            if current_time - last_sent_time < MESSAGE_COOLDOWN:
                logger.debug("Message was recently sent, skipping")
                return True
                
        # Check for duplicate content
        global last_message_content, last_message_time
        if message == last_message_content and current_time - last_message_time < MESSAGE_COOLDOWN:
            logger.debug("Duplicate message content, skipping")
            return True
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Split message into chunks if too long
        max_length = 4000  # Leave room for special characters
        message_chunks = [message[i:i+max_length] for i in range(0, len(message), max_length)]
        
        success = True
        for chunk in message_chunks:
            payload = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML"
            }
            
            try:
                response = requests.post(url, json=payload, timeout=10)
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Error sending Telegram message chunk: {e}")
                success = False
                break
                
        if success:
            # Update message history
            message_history[message_key] = current_time
            last_message_content = message
            last_message_time = current_time
            
        return success
        
    except Exception as e:
        logger.error(f"Error in send_telegram_message: {e}")
        return False

def save_open_positions(positions: Dict[str, Any]) -> bool:
    """Save open positions to file"""
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save positions with timestamp
        positions_data = {
            'timestamp': datetime.now().isoformat(),
            'positions': positions
        }
        
        file_path = os.path.join(data_dir, 'open_positions.json')
        with open(file_path, 'w') as f:
            json.dump(positions_data, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error saving open positions: {e}")
        return False

def load_open_positions() -> Dict[str, Any]:
    """Load open positions from file"""
    try:
        file_path = os.path.join(os.getcwd(), 'data', 'open_positions.json')
        
        if not os.path.exists(file_path):
            logger.warning("No open positions file found")
            return {}
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check if positions are too old (older than 24 hours)
        timestamp = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - timestamp > timedelta(hours=24):
            logger.warning("Open positions data is older than 24 hours")
            return {}
            
        return data.get('positions', {})
    except Exception as e:
        logger.error(f"Error loading open positions: {e}")
        return {}

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

