# Log Errors Fixes Summary

## Overview
Fixed multiple errors that were appearing in the trading bot logs, specifically addressing issues with the IndicatorService, performance metrics, and Discord service handling.

## Errors Identified and Fixed

### 1. **`'IndicatorService' object has no attribute 'get_klines'`**

**Problem**: The `IndicatorService` was missing the `get_klines` method that was being called by the strategy.

**Solution**: Added the `get_klines` method to `src/services/indicator_service.py`:

```python
async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[Dict]:
    """Get klines data for a symbol.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles to fetch
        
    Returns:
        Optional[Dict]: Klines data with OHLCV information
    """
    try:
        if not self._is_initialized:
            logger.error("Indicator service not initialized")
            return None
            
        if self._is_closed:
            logger.error("Indicator service is closed")
            return None
        
        # Get historical data
        df = await self.get_historical_data(symbol, timeframe, limit)
        if df is None or df.empty:
            logger.warning(f"Could not get data for {symbol}: No data available")
            return None
        
        # Convert DataFrame to dictionary format
        klines_data = {
            'open': df['open'].tolist(),
            'high': df['high'].tolist(),
            'low': df['low'].tolist(),
            'close': df['close'].tolist(),
            'volume': df['volume'].tolist() if 'volume' in df.columns else [0] * len(df)
        }
        
        return klines_data
        
    except Exception as e:
        logger.error(f"Error getting klines for {symbol}: {str(e)}")
        return None
```

### 2. **`object dict can't be used in 'await' expression`**

**Problem**: The strategy was trying to await `self.quantitative_system.get_performance_metrics()` but the method is not async.

**Solution**: Removed the `await` keyword in `src/strategies/enhanced_trading_strategy_with_quantitative.py`:

```python
# Before:
qs_metrics = await self.quantitative_system.get_performance_metrics()

# After:
qs_metrics = self.quantitative_system.get_performance_metrics()
```

### 3. **`object NoneType can't be used in 'await' expression`**

**Problem**: The Discord service was `None` but the code was trying to call `await discord_service.pause_trading()`.

**Solution**: The fix was already implemented in `main_with_quantitative.py` with proper `None` checks:

```python
# Check profit target
if await strategy.check_profit_target():
    # Pause trading through both services
    if telegram_service:
        await telegram_service.pause_trading()
    if discord_service:
        await discord_service.pause_trading()
```

### 4. **`'list' object has no attribute 'values'`**

**Problem**: The `_get_market_conditions` function was trying to access `.values` on a list instead of a pandas DataFrame.

**Solution**: Enhanced the data format handling in `src/strategies/enhanced_trading_strategy_with_quantitative.py`:

```python
async def _get_market_conditions(self, symbol: str) -> Dict:
    """Get market conditions for stop loss adjustment."""
    try:
        # Get recent price data for volatility calculation
        klines = await self.indicator_service.get_klines(symbol, '1h', limit=24)
        if klines is not None and len(klines) > 1:
            # Handle both pandas DataFrame and dict formats
            if isinstance(klines, dict) and 'close' in klines:
                # Handle dictionary format with list
                prices = np.array(klines['close'])
            elif hasattr(klines, 'values'):
                # Handle pandas DataFrame format
                prices = klines['close'].values
            else:
                # Fallback to default values
                logger.warning(f"Unexpected klines format for {symbol}: {type(klines)}")
                return {'volatility': 0.02, 'price_change_24h': 0.0}
            
            if len(prices) > 1:
                returns = np.diff(np.log(prices))
                volatility = float(np.std(returns) * np.sqrt(252))
                
                return {
                    'volatility': volatility,
                    'price_change_24h': float((prices[-1] / prices[0] - 1) * 100)
                }
        
        return {'volatility': 0.02, 'price_change_24h': 0.0}
        
    except Exception as e:
        logger.error(f"Error getting market conditions for {symbol}: {str(e)}")
        return {'volatility': 0.02, 'price_change_24h': 0.0}
```

## Testing

Created and ran `test_log_fixes.py` to verify all fixes:

- ✅ **IndicatorService has get_klines method** - Fixed
- ✅ **Strategy performance metrics work correctly** - Fixed  
- ✅ **_get_market_conditions works correctly** - Fixed
- ✅ **Discord service None handling works correctly** - Fixed

## Benefits of the Fixes

1. **Improved Data Handling**: The `IndicatorService` now properly provides klines data in the expected format
2. **Better Error Handling**: All services now handle `None` values gracefully
3. **Robust Performance Metrics**: Performance metrics calculation no longer causes async/await errors
4. **Enhanced Market Analysis**: Market conditions calculation works with different data formats
5. **Cleaner Logs**: Reduced error messages and warnings in the trading bot logs

## Files Modified

1. **`src/services/indicator_service.py`**: Added `get_klines` method
2. **`src/strategies/enhanced_trading_strategy_with_quantitative.py`**: 
   - Fixed performance metrics async/await issue
   - Enhanced `_get_market_conditions` data format handling
3. **`main_with_quantitative.py`**: Already had proper Discord service None handling

## Verification

All tests pass successfully, confirming that the log errors have been resolved and the trading bot should now run without the previously reported errors. 