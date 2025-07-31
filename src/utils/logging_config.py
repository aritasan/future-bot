import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup enhanced logging configuration."""
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/trading_bot_quantitative_20250731.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Disable noisy logs completely
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('dash').setLevel(logging.ERROR)
    logging.getLogger('dash.dash').setLevel(logging.ERROR)
    logging.getLogger('flask').setLevel(logging.ERROR)
    logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
    logging.getLogger('websockets.server').setLevel(logging.WARNING)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
    logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.ERROR)
    
    # Keep important logs at INFO level
    logging.getLogger('src.strategies.enhanced_trading_strategy_with_quantitative').setLevel(logging.INFO)
    logging.getLogger('main_with_quantitative').setLevel(logging.INFO)
    logging.getLogger('src.quantitative').setLevel(logging.INFO)
    logging.getLogger('src.services').setLevel(logging.INFO)
    logging.getLogger('src.core').setLevel(logging.INFO)
    logging.getLogger('src.utils').setLevel(logging.INFO)
    
    # Set specific loggers to WARNING to reduce noise
    logging.getLogger('src.quantitative.performance_tracker').setLevel(logging.WARNING)
    logging.getLogger('src.quantitative.real_time_performance_monitor').setLevel(logging.WARNING)
    
    print("Enhanced logging configured successfully")

if __name__ == "__main__":
    setup_logging()
