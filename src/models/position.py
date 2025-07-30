from dataclasses import dataclass
from typing import Optional

@dataclass
class Position:
    """Class representing a trading position."""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    amount: float
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: float
    dca_attempts: int = 0
    last_dca_time: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'cancelled'
    close_price: Optional[float] = None
    close_time: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    fees: Optional[float] = None
    notes: Optional[str] = None 