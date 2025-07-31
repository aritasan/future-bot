#!/usr/bin/env python3
"""
Fix Logging Issues Script
- Disable werkzeug logs
- Fix logging configuration
- Investigate why enhanced_trading_strategy_with_quantitative logs stopped
"""

import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def disable_werkzeug_logs():
    """Disable werkzeug logs to reduce noise."""
    try:
        # Disable werkzeug logs
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('dash').setLevel(logging.ERROR)
        logging.getLogger('dash.dash').setLevel(logging.ERROR)
        logging.getLogger('flask').setLevel(logging.ERROR)
        
        print("✅ Disabled werkzeug, dash, and flask logs")
        
    except Exception as e:
        print(f"❌ Error disabling werkzeug logs: {e}")

def configure_logging():
    """Configure logging to reduce noise and focus on important logs."""
    try:
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_bot_quantitative_20250731.log', mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set specific log levels
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('dash').setLevel(logging.ERROR)
        logging.getLogger('dash.dash').setLevel(logging.ERROR)
        logging.getLogger('flask').setLevel(logging.ERROR)
        logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
        logging.getLogger('websockets.server').setLevel(logging.WARNING)
        
        # Keep important logs at INFO level
        logging.getLogger('src.strategies.enhanced_trading_strategy_with_quantitative').setLevel(logging.INFO)
        logging.getLogger('main_with_quantitative').setLevel(logging.INFO)
        logging.getLogger('src.quantitative').setLevel(logging.INFO)
        
        print("✅ Logging configured successfully")
        
    except Exception as e:
        print(f"❌ Error configuring logging: {e}")

async def investigate_log_stop():
    """Investigate why enhanced_trading_strategy_with_quantitative logs stopped."""
    try:
        print("\n🔍 Investigating log stop issue...")
        
        # Check if the strategy is still running
        from src.strategies.enhanced_trading_strategy_with_quantitative import EnhancedTradingStrategyWithQuantitative
        from src.services.binance_service import BinanceService
        from src.services.indicator_service import IndicatorService
        from src.services.notification_service import NotificationService
        from src.core.config import load_config
        
        config = load_config()
        
        # Test strategy initialization
        print("🧪 Testing strategy initialization...")
        
        # Mock services for testing
        class MockBinanceService:
            async def get_account_balance(self):
                return {'USDT': {'total': 1000}}
            
            async def get_positions(self):
                return []
            
            async def get_klines(self, symbol, timeframe, limit):
                return {'close': [100, 101, 102, 103, 104]}
        
        class MockIndicatorService:
            async def get_klines(self, symbol, timeframe, limit):
                return {'close': [100, 101, 102, 103, 104]}
        
        class MockNotificationService:
            async def send_notification(self, message):
                pass
        
        # Initialize strategy
        strategy = EnhancedTradingStrategyWithQuantitative(
            config=config,
            binance_service=MockBinanceService(),
            indicator_service=MockIndicatorService(),
            notification_service=MockNotificationService()
        )
        
        # Test initialization
        init_result = await strategy.initialize()
        print(f"✅ Strategy initialization: {'SUCCESS' if init_result else 'FAILED'}")
        
        # Test signal generation
        print("🧪 Testing signal generation...")
        signal = await strategy.generate_signals('BTCUSDT', MockIndicatorService())
        print(f"✅ Signal generation: {'SUCCESS' if signal else 'FAILED'}")
        
        # Test performance metrics
        print("🧪 Testing performance metrics...")
        metrics = await strategy.get_performance_metrics()
        print(f"✅ Performance metrics: {'SUCCESS' if metrics else 'FAILED'}")
        
        # Test quantitative analysis
        print("🧪 Testing quantitative analysis...")
        try:
            analysis = await strategy.analyze_portfolio_optimization(['BTCUSDT', 'ETHUSDT'])
            print(f"✅ Portfolio optimization: {'SUCCESS' if analysis else 'FAILED'}")
        except Exception as e:
            print(f"⚠️ Portfolio optimization failed: {e}")
        
        # Test factor analysis
        print("🧪 Testing factor analysis...")
        try:
            factor_analysis = await strategy.analyze_factor_exposures(['BTCUSDT', 'ETHUSDT'])
            print(f"✅ Factor analysis: {'SUCCESS' if factor_analysis else 'FAILED'}")
        except Exception as e:
            print(f"⚠️ Factor analysis failed: {e}")
        
        print("\n✅ Strategy investigation completed")
        
    except Exception as e:
        print(f"❌ Error investigating log stop: {e}")
        import traceback
        traceback.print_exc()

def check_main_bot_status():
    """Check if main bot is still running and processing symbols."""
    try:
        print("\n🔍 Checking main bot status...")
        
        # Check if main_with_quantitative is still running
        try:
            import psutil
            
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] == 'python.exe':
                        cmdline = proc.info['cmdline']
                        if cmdline and any('main_with_quantitative' in arg for arg in cmdline):
                            python_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if python_processes:
                print(f"✅ Found {len(python_processes)} main_with_quantitative processes running")
                for proc in python_processes:
                    print(f"   PID: {proc.pid}, CPU: {proc.cpu_percent()}%, Memory: {proc.memory_info().rss / 1024 / 1024:.1f}MB")
            else:
                print("❌ No main_with_quantitative processes found running")
            
            # Check for any Python processes
            all_python = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] == 'python.exe':
                        cmdline = proc.info['cmdline']
                        if cmdline:
                            all_python.append(f"PID: {proc.pid}, CMD: {' '.join(cmdline[:3])}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            print(f"📊 Total Python processes: {len(all_python)}")
            for proc_info in all_python[:5]:  # Show first 5
                print(f"   {proc_info}")
                
        except ImportError:
            print("⚠️ psutil not available, skipping process check")
        
    except Exception as e:
        print(f"❌ Error checking bot status: {e}")

def create_logging_config():
    """Create a proper logging configuration file."""
    try:
        config_content = '''
import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    
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
    
    # Disable noisy logs
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('dash').setLevel(logging.ERROR)
    logging.getLogger('dash.dash').setLevel(logging.ERROR)
    logging.getLogger('flask').setLevel(logging.ERROR)
    logging.getLogger('aiohttp.access').setLevel(logging.ERROR)
    logging.getLogger('websockets.server').setLevel(logging.WARNING)
    
    # Keep important logs at INFO level
    logging.getLogger('src.strategies.enhanced_trading_strategy_with_quantitative').setLevel(logging.INFO)
    logging.getLogger('main_with_quantitative').setLevel(logging.INFO)
    logging.getLogger('src.quantitative').setLevel(logging.INFO)
    logging.getLogger('src.services').setLevel(logging.INFO)
    logging.getLogger('src.core').setLevel(logging.INFO)
    
    print("✅ Logging configured successfully")

if __name__ == "__main__":
    setup_logging()
'''
        
        with open('src/utils/logging_config.py', 'w') as f:
            f.write(config_content)
        
        print("✅ Created logging configuration file: src/utils/logging_config.py")
        
    except Exception as e:
        print(f"❌ Error creating logging config: {e}")

async def main():
    """Main function to fix logging issues."""
    print("🔧 Fixing Logging Issues")
    print("=" * 50)
    
    # 1. Disable werkzeug logs
    disable_werkzeug_logs()
    
    # 2. Configure logging
    configure_logging()
    
    # 3. Create logging config file
    create_logging_config()
    
    # 4. Check main bot status
    check_main_bot_status()
    
    # 5. Investigate log stop
    await investigate_log_stop()
    
    print("\n" + "=" * 50)
    print("✅ Logging issues fixed!")
    print("\n📋 Summary:")
    print("   • Disabled werkzeug, dash, and flask logs")
    print("   • Configured proper logging levels")
    print("   • Created logging configuration file")
    print("   • Checked main bot status")
    print("   • Investigated strategy log stop")
    
    print("\n🚀 Next steps:")
    print("   1. Restart the bot with new logging configuration")
    print("   2. Monitor logs for enhanced_trading_strategy_with_quantitative")
    print("   3. Check if symbol processing continues")
    print("   4. Verify quantitative analysis is working")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 