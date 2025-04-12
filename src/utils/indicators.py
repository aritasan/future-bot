"""
Technical analysis indicators and utilities.
"""
import pandas as pd
from typing import Dict, Optional

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
    df["BB_middle"] = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()
    df["BB_upper"] = df["BB_middle"] + (rolling_std * std_dev)
    df["BB_lower"] = df["BB_middle"] - (rolling_std * std_dev)
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
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean())
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period).mean()
    
    return adx

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        period (int): Period for ATR calculation
        
    Returns:
        pd.Series: ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period).mean()
    
    return atr

def calculate_indicators(df: pd.DataFrame, params: Optional[Dict] = None) -> pd.DataFrame:
    """
    Calculate all technical indicators for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        params (Dict, optional): Custom parameters for indicators
        
    Returns:
        pd.DataFrame: DataFrame with all indicators added
    """
    if params is None:
        params = {
            "EMA_FAST": 9,
            "EMA_SLOW": 21,
            "RSI_PERIOD": 14,
            "MACD_FAST": 12,
            "MACD_SLOW": 26,
            "MACD_SIGNAL": 9
        }
    
    # Calculate RSI
    delta = df["close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=params["RSI_PERIOD"]).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=params["RSI_PERIOD"]).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    
    # Calculate ADX and ATR
    df["ADX"] = calculate_adx(df)
    df["ATR"] = calculate_atr(df)
    
    # Calculate EMAs
    df["EMA_FAST"] = df["close"].ewm(span=params["EMA_FAST"], adjust=False).mean()
    df["EMA_SLOW"] = df["close"].ewm(span=params["EMA_SLOW"], adjust=False).mean()
    
    # Calculate MACD
    df["MACD"] = df["close"].ewm(span=params["MACD_FAST"], adjust=False).mean() - df["close"].ewm(span=params["MACD_SLOW"], adjust=False).mean()
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=params["MACD_SIGNAL"], adjust=False).mean()
    
    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
    
    return df 