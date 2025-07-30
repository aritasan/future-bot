import asyncio
import sys
import os

# Set event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.core.config import load_config
from src.services.binance_service import BinanceService

async def test_api_connection():
    """Test the API connection and credentials."""
    try:
        # Load configuration
        config = load_config()
        print("Configuration loaded successfully")
        
        # Check API credentials
        use_testnet = config['api']['binance']['use_testnet']
        api_config = config['api']['binance']['testnet' if use_testnet else 'mainnet']
        
        print(f"Using {'testnet' if use_testnet else 'mainnet'} mode")
        print(f"API Key: {'***' + api_config['api_key'][-4:] if api_config['api_key'] else 'NOT SET'}")
        print(f"API Secret: {'***' + api_config['api_secret'][-4:] if api_config['api_secret'] else 'NOT SET'}")
        
        if not api_config['api_key'] or not api_config['api_secret']:
            print("❌ API credentials are not set!")
            print("Please set the following environment variables:")
            if use_testnet:
                print("  BINANCE_TESTNET_API_KEY")
                print("  BINANCE_TESTNET_API_SECRET")
            else:
                print("  BINANCE_MAINNET_API_KEY")
                print("  BINANCE_MAINNET_API_SECRET")
            return False
        
        # Initialize Binance service
        binance_service = BinanceService(config)
        print("Initializing Binance service...")
        
        success = await binance_service.initialize()
        if success:
            print("✅ Binance service initialized successfully")
            
            # Test getting account balance
            print("Testing account balance...")
            balance = await binance_service.get_account_balance()
            if balance:
                print("✅ Account balance retrieved successfully")
                print(f"Total Balance: {balance.get('total', 'N/A')}")
            else:
                print("❌ Failed to get account balance")
            
            # Test getting positions
            print("Testing positions...")
            positions = await binance_service.get_positions()
            if positions:
                print("✅ Positions retrieved successfully")
                print(f"Number of positions: {len(positions)}")
            else:
                print("❌ Failed to get positions")
            
            await binance_service.close()
            return True
        else:
            print("❌ Failed to initialize Binance service")
            return False
            
    except Exception as e:
        print(f"❌ Error during API connection test: {str(e)}")
        return False

async def main():
    print("Testing API Connection...")
    print("=" * 50)
    
    success = await test_api_connection()
    
    print("=" * 50)
    if success:
        print("✅ API connection test completed successfully")
    else:
        print("❌ API connection test failed")
        print("\nTroubleshooting tips:")
        print("1. Check if your API credentials are set correctly")
        print("2. Verify your API key has the correct permissions")
        print("3. Check if your system time is synchronized")
        print("4. Ensure you have a stable internet connection")

if __name__ == "__main__":
    asyncio.run(main()) 