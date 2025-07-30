# Stop Loss và Take Profit Implementation

## Tổng quan
Đã thêm các hàm tính toán stop loss và take profit vào chiến lược trading quantitative, dựa trên logic từ `enhanced_trading_strategy.py` cũ.

## Các hàm đã được thêm

### 1. `_calculate_stop_loss()` - Dòng 708
**Chức năng:** Tính toán giá stop loss dựa trên ATR và điều kiện thị trường

**Tham số:**
- `symbol`: Ký hiệu giao dịch
- `position_type`: Loại vị thế ("LONG" hoặc "SHORT")
- `current_price`: Giá hiện tại
- `atr`: Average True Range

**Logic tính toán:**
- **LONG position**: `stop_loss = current_price - (atr * multiplier)`
- **SHORT position**: `stop_loss = current_price + (atr * multiplier/2)`
- Điều chỉnh dựa trên volatility
- Đảm bảo khoảng cách tối thiểu từ giá hiện tại

**Cấu hình từ config:**
```python
stop_loss_multiplier = config['risk_management']['stop_loss_atr_multiplier']
min_distance = config['risk_management']['min_stop_distance']
```

### 2. `_calculate_take_profit()` - Dòng 766
**Chức năng:** Tính toán giá take profit dựa trên risk-reward ratio

**Tham số:**
- `symbol`: Ký hiệu giao dịch
- `position_type`: Loại vị thế ("LONG" hoặc "SHORT")
- `current_price`: Giá hiện tại
- `stop_loss`: Giá stop loss đã tính

**Logic tính toán:**
- **LONG position**: `take_profit = current_price + (price_diff * risk_reward_ratio)`
- **SHORT position**: `take_profit = current_price - (price_diff * risk_reward_ratio/8)`
- Đảm bảo khoảng cách tối thiểu từ giá hiện tại

**Cấu hình từ config:**
```python
risk_reward_ratio = config['risk_management']['take_profit_multiplier']
min_distance = config['risk_management']['min_tp_distance']
```

### 3. `_get_market_conditions()` - Dòng 805
**Chức năng:** Lấy điều kiện thị trường để điều chỉnh stop loss

**Trả về:**
- `volatility`: Độ biến động (tính từ returns 24h)
- `price_change_24h`: Thay đổi giá 24h (%)

## Cập nhật các hàm Execute

### 1. `_execute_buy_order()` - Dòng 598
**Cải tiến:**
- Tính toán stop loss và take profit trước khi đặt lệnh
- Đặt lệnh chính (MARKET BUY)
- Đặt lệnh stop loss (STOP_MARKET SELL)
- Đặt lệnh take profit (LIMIT SELL)

**Flow:**
```python
# 1. Tính toán stop loss và take profit
atr = signals.get('atr', current_price * 0.02)
stop_loss = await self._calculate_stop_loss(symbol, "LONG", current_price, atr)
take_profit = await self._calculate_take_profit(symbol, "LONG", current_price, stop_loss)

# 2. Đặt lệnh chính
order = await self.binance_service.place_order(symbol, 'BUY', 'MARKET', quantity)

# 3. Đặt stop loss
if stop_loss and stop_loss > 0:
    stop_order = await self.binance_service.place_order(
        symbol, 'SELL', 'STOP_MARKET', quantity, stopPrice=stop_loss
    )

# 4. Đặt take profit
if take_profit and take_profit > current_price:
    tp_order = await self.binance_service.place_order(
        symbol, 'SELL', 'LIMIT', quantity, price=take_profit
    )
```

### 2. `_execute_sell_order()` - Dòng 650
**Cải tiến:**
- Tính toán stop loss và take profit cho short position
- Đặt lệnh chính (MARKET SELL)
- Đặt lệnh stop loss (STOP_MARKET BUY để cover)
- Đặt lệnh take profit (LIMIT BUY để cover)

**Flow:**
```python
# 1. Tính toán stop loss và take profit cho SHORT
atr = signals.get('atr', current_price * 0.02)
stop_loss = await self._calculate_stop_loss(symbol, "SHORT", current_price, atr)
take_profit = await self._calculate_take_profit(symbol, "SHORT", current_price, stop_loss)

# 2. Đặt lệnh chính
order = await self.binance_service.place_order(symbol, 'SELL', 'MARKET', quantity)

# 3. Đặt stop loss (buy to cover)
if stop_loss and stop_loss > current_price:
    stop_order = await self.binance_service.place_order(
        symbol, 'BUY', 'STOP_MARKET', quantity, stopPrice=stop_loss
    )

# 4. Đặt take profit (buy to cover)
if take_profit and take_profit < current_price:
    tp_order = await self.binance_service.place_order(
        symbol, 'BUY', 'LIMIT', quantity, price=take_profit
    )
```

## Cấu hình cần thiết

### Trong config file:
```json
{
  "risk_management": {
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_multiplier": 2.0,
    "min_stop_distance": 0.01,
    "min_tp_distance": 0.01
  }
}
```

### Giải thích các tham số:
- `stop_loss_atr_multiplier`: Hệ số nhân ATR cho stop loss (mặc định: 2.0)
- `take_profit_multiplier`: Hệ số risk-reward cho take profit (mặc định: 2.0)
- `min_stop_distance`: Khoảng cách tối thiểu từ giá hiện tại đến stop loss (1%)
- `min_tp_distance`: Khoảng cách tối thiểu từ giá hiện tại đến take profit (1%)

## Tính năng bảo vệ

### 1. Validation cho LONG position:
- Đảm bảo stop loss > 0
- Đảm bảo stop loss < current_price
- Đảm bảo take profit > current_price

### 2. Validation cho SHORT position:
- Đảm bảo stop loss > current_price
- Đảm bảo take profit < current_price
- Xử lý trường hợp take profit <= 0

### 3. Điều chỉnh theo volatility:
- Tăng khoảng cách stop loss khi volatility cao (>2%)
- Giảm khoảng cách khi volatility thấp

## Logging và Monitoring

### Log messages:
```python
logger.info(f"Calculated stop loss for {symbol} {position_type.lower()}: {stop_loss}")
logger.info(f"Calculated take profit for {symbol} {position_type.lower()}: {take_profit}")
logger.info(f"Stop loss order placed for {symbol}: {stop_order}")
logger.info(f"Take profit order placed for {symbol}: {tp_order}")
```

## Lợi ích

### 1. Quản lý rủi ro tự động:
- Stop loss tự động bảo vệ khỏi thua lỗ lớn
- Take profit tự động thu lợi nhuận khi đạt mục tiêu

### 2. Tối ưu hóa risk-reward:
- Risk-reward ratio có thể cấu hình
- Điều chỉnh theo điều kiện thị trường

### 3. Tính linh hoạt:
- Hỗ trợ cả LONG và SHORT position
- Điều chỉnh theo volatility
- Cấu hình dễ dàng qua config

## Status: ✅ HOÀN THÀNH
Các hàm stop loss và take profit đã được implement đầy đủ và tích hợp vào chiến lược trading quantitative. 