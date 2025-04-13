"""
Technical analysis indicators and utilities.
"""
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int): Period for moving average
        std_dev (int): Number of standard deviations
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands added
    """
    try:
        if len(df) < period:
            period = len(df)
            
        df["BB_middle"] = df["close"].rolling(window=period, min_periods=1).mean()
        rolling_std = df["close"].rolling(window=period, min_periods=1).std()
        df["BB_upper"] = df["BB_middle"] + (rolling_std * std_dev)
        df["BB_lower"] = df["BB_middle"] - (rolling_std * std_dev)
        return df
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return df

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int): Period for ADX calculation
        
    Returns:
        pd.Series: ADX values
    """
    try:
        if len(df) < period:
            period = len(df)
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=1).mean() / tr.ewm(alpha=1/period, min_periods=1).mean())
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=1).mean() / tr.ewm(alpha=1/period, min_periods=1).mean())
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, min_periods=1).mean()
        
        return adx
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        return pd.Series(0, index=df.index)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int): Period for ATR calculation
        
    Returns:
        pd.Series: ATR values
    """
    try:
        if len(df) < period:
            period = len(df)
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=1).mean()
        
        return atr
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        return pd.Series(0, index=df.index)

def calculate_indicators(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate all technical indicators for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        params (Dict, optional): Custom parameters for indicators
        
    Returns:
        pd.DataFrame: DataFrame with all indicators added
    """
    try:
        if params is None:
            params = {
                "EMA_FAST": 9,
                "EMA_SLOW": 21,
                "RSI_PERIOD": 14,
                "MACD_FAST": 12,
                "MACD_SLOW": 26,
                "MACD_SIGNAL": 9
            }
        
        # Validate data
        if df.empty or len(df) < 2:
            logger.warning("Not enough data points to calculate indicators")
            return df
            
        # Check for missing values in required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}, Found: {df.columns.tolist()}")
            return df
            
        # Check for missing values and fill them
        if df[required_columns].isnull().any().any():
            logger.warning("Missing values detected in price data, filling with forward/backward fill")
            df[required_columns] = df[required_columns].ffill().bfill()
            
        # Ensure we have enough data points for calculations
        min_periods = max(params.values())
        if len(df) < min_periods:
            logger.warning(f"Not enough data points for calculations. Need at least {min_periods}, got {len(df)}")
            return df
            
        # Calculate RSI with proper handling of initial values
        delta = df["close"].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=params["RSI_PERIOD"], min_periods=params["RSI_PERIOD"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params["RSI_PERIOD"], min_periods=params["RSI_PERIOD"]).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Calculate ADX and ATR with proper min_periods
        df["ADX"] = calculate_adx(df)
        df["ATR"] = calculate_atr(df)
        
        # Calculate EMAs with proper min_periods
        df["EMA_FAST"] = df["close"].ewm(span=params["EMA_FAST"], adjust=False, min_periods=params["EMA_FAST"]).mean()
        df["EMA_SLOW"] = df["close"].ewm(span=params["EMA_SLOW"], adjust=False, min_periods=params["EMA_SLOW"]).mean()
        
        # Calculate MACD with proper min_periods
        fast_ema = df["close"].ewm(span=params["MACD_FAST"], adjust=False, min_periods=params["MACD_FAST"]).mean()
        slow_ema = df["close"].ewm(span=params["MACD_SLOW"], adjust=False, min_periods=params["MACD_SLOW"]).mean()
        df["MACD"] = fast_ema - slow_ema
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=params["MACD_SIGNAL"], adjust=False, min_periods=params["MACD_SIGNAL"]).mean()
        
        # Calculate Bollinger Bands
        df = calculate_bollinger_bands(df)
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
        
        # Fill any remaining NaN values with 0 for indicators
        indicator_columns = ['RSI', 'ADX', 'ATR', 'EMA_FAST', 'EMA_SLOW', 'MACD', 'MACD_SIGNAL', 
                           'BB_middle', 'BB_upper', 'BB_lower', 'BB_width']
        df[indicator_columns] = df[indicator_columns].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df 