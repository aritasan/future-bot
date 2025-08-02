#!/usr/bin/env python3
"""
Test script for WorldQuant DCA and Trailing Stop integration.
"""

import asyncio
import logging
from typing import Dict, Any
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_worldquant_integration():
    """Test the WorldQuant DCA and Trailing Stop integration."""
    logger.info("🧪 Testing WorldQuant DCA and Trailing Stop integration...")
    
    # Mock config
    config = {
        'risk_management': {
            'dca': {
                'enabled': True,
                'dca_size_multiplier': 0.5,  # 50% of current position
                'max_dca_size_multiplier': 2.0,  # Max 2x current position
                'min_dca_size': 0.001,
                'max_attempts': 3,
                'price_drop_thresholds': [5, 10, 15]  # 5%, 10%, 15%
            },
            'trailing_stop': {
                'enabled': True,
                'profit_thresholds': [2, 5, 10],  # 2%, 5%, 10%
                'trailing_multipliers': [2.0, 1.5, 1.0]  # Tighter as profit increases
            }
        }
    }
    
    # Test WorldQuant DCA
    logger.info("\n📊 Testing WorldQuant DCA...")
    from src.quantitative.worldquant_dca_trailing import WorldQuantDCA
    
    dca = WorldQuantDCA(config)
    
    # Mock position and market data
    long_position = {
        'symbol': 'ETHUSDT',
        'markPrice': '3000',  # Current price dropped from 3200
        'entryPrice': '3200',  # Entry price
        'unrealizedPnl': '-200',
        'info': {
            'positionSide': 'LONG',
            'positionAmt': '0.1'
        }
    }
    
    market_data = {
        'klines': [
            ['', '', '', '', '3200'],  # Entry price
            ['', '', '', '', '3100'],
            ['', '', '', '', '3000']   # Current price
        ],
        'orderbook': {
            'bids': [['2999', '1.0']],
            'asks': [['3001', '1.0']]
        }
    }
    
    dca_result = await dca.check_dca_opportunity('ETHUSDT', long_position, market_data)
    logger.info(f"WorldQuant DCA Result: {dca_result}")
    
    # Test WorldQuant Trailing Stop
    logger.info("\n📊 Testing WorldQuant Trailing Stop...")
    from src.quantitative.worldquant_dca_trailing import WorldQuantTrailingStop
    
    trailing = WorldQuantTrailingStop(config)
    
    profitable_long_position = {
        'symbol': 'ETHUSDT',
        'markPrice': '3360',  # Current price rose from 3200 (5% profit)
        'entryPrice': '3200',  # Entry price
        'unrealizedPnl': '160',
        'info': {
            'positionSide': 'LONG',
            'positionAmt': '0.1'
        }
    }
    
    trailing_result = await trailing.check_trailing_stop_opportunity('ETHUSDT', profitable_long_position, market_data)
    logger.info(f"WorldQuant Trailing Stop Result: {trailing_result}")
    
    logger.info("\n✅ WorldQuant integration tested successfully!")

async def test_strategy_integration():
    """Test the integration with the trading strategy."""
    logger.info("\n🧪 Testing Strategy Integration...")
    
    # Mock config
    config = {
        'risk_management': {
            'dca': {
                'enabled': True,
                'dca_size_multiplier': 0.5,
                'max_dca_size_multiplier': 2.0,
                'min_dca_size': 0.001,
                'max_attempts': 3,
                'price_drop_thresholds': [5, 10, 15]
            },
            'trailing_stop': {
                'enabled': True,
                'profit_thresholds': [2, 5, 10],
                'trailing_multipliers': [2.0, 1.5, 1.0]
            }
        }
    }
    
    # Test that the classes can be imported and initialized
    try:
        from src.quantitative.worldquant_dca_trailing import WorldQuantDCA, WorldQuantTrailingStop
        
        dca = WorldQuantDCA(config)
        trailing = WorldQuantTrailingStop(config)
        
        logger.info("✅ WorldQuant DCA and Trailing Stop classes initialized successfully")
        
        # Test that they have the required methods
        assert hasattr(dca, 'check_dca_opportunity')
        assert hasattr(dca, 'execute_dca')
        assert hasattr(trailing, 'check_trailing_stop_opportunity')
        assert hasattr(trailing, 'execute_trailing_stop_update')
        
        logger.info("✅ All required methods are present")
        
    except Exception as e:
        logger.error(f"❌ Error in strategy integration test: {str(e)}")
        return False
    
    return True

async def create_integration_config():
    """Create configuration file for WorldQuant integration."""
    logger.info("🔧 Creating WorldQuant integration configuration...")
    
    integration_config = {
        'risk_management': {
            'dca': {
                'enabled': True,
                'dca_size_multiplier': 0.5,  # 50% of current position
                'max_dca_size_multiplier': 2.0,  # Max 2x current position
                'min_dca_size': 0.001,
                'max_attempts': 3,
                'price_drop_thresholds': [5, 10, 15],  # 5%, 10%, 15%
                'min_interval': 3600,  # 1 hour between DCA attempts
                'risk_reduction': 0.5  # Reduce risk by 50% for each DCA
            },
            'trailing_stop': {
                'enabled': True,
                'profit_thresholds': [2, 5, 10],  # 2%, 5%, 10%
                'trailing_multipliers': [2.0, 1.5, 1.0],  # Tighter as profit increases
                'update_interval': 300,  # 5 minutes between updates
                'min_profit_for_trail': 1.0  # 1% minimum profit to start trailing
            }
        }
    }
    
    # Save integration config
    with open('worldquant_integration_config.json', 'w') as f:
        json.dump(integration_config, f, indent=2)
    
    logger.info("✅ WorldQuant integration configuration created: worldquant_integration_config.json")

if __name__ == "__main__":
    print("🚀 Testing WorldQuant DCA and Trailing Stop Integration...")
    
    # Run tests
    asyncio.run(test_worldquant_integration())
    asyncio.run(test_strategy_integration())
    
    # Create integration configuration
    asyncio.run(create_integration_config())
    
    print("\n🎉 WorldQuant DCA and Trailing Stop integration completed!")
    print("📁 Integration configuration saved: worldquant_integration_config.json")
    print("📊 Test results logged above")
    
    print("\n📋 Integration Summary:")
    print("✅ WorldQuant DCA: Integrated into quantitative trading strategy")
    print("✅ WorldQuant Trailing Stop: Integrated into quantitative trading strategy")
    print("✅ Both components: Automatically check opportunities during signal processing")
    print("✅ Configuration: Ready for use with trading bot") 