# 🚀 Hướng dẫn Setup và Chạy Trading Bot

## 1. Cấu hình API Keys

Tạo file `.env` trong thư mục gốc với nội dung sau:

```bash
# Binance API Configuration
BINANCE_MAINNET_API_KEY=your_binance_api_key_here
BINANCE_MAINNET_API_SECRET=your_binance_api_secret_here
USE_TESTNET=false

# Telegram Configuration (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
IS_TELEGRAM_ENABLED=false

# Discord Configuration (Optional)
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here
DISCORD_ENABLED=true

# Trading Parameters
MAX_DRAWDOWN=10.0
RISK_PER_TRADE=0.005
ATR_PERIOD=14
MAX_ORDERS_PER_SYMBOL=3

# Risk Management
BASE_STOP_DISTANCE=0.02
VOLATILITY_MULTIPLIER=1.5
TREND_MULTIPLIER=1.2
TAKE_PROFIT_MULTIPLIER=2.0
DCA_MULTIPLIER=0.5
ATR_MULTIPLIER=1.5
STOP_LOSS_ATR_MULTIPLIER=1.5

# Cache Configuration
CACHE_ENABLED=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Performance Monitoring
PERFORMANCE_MONITORING_ENABLED=true
WEBSOCKET_PORT=8765
DASHBOARD_PORT=8050
```

## 2. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

## 3. Chạy Bot với Real-time Performance Tracking

### Cách 1: Chạy trực tiếp
```bash
python main_with_quantitative.py
```

### Cách 2: Chạy với Performance Dashboard
```bash
# Terminal 1: Chạy bot chính
python main_with_quantitative.py

# Terminal 2: Chạy Performance Dashboard
python performance_dashboard_enhanced.py
```

### Cách 3: Chạy với Docker
```bash
docker-compose up -d
```

## 4. Truy cập Performance Dashboard

Sau khi chạy, bạn có thể truy cập:

- **Performance Dashboard**: http://localhost:8050
- **WebSocket Server**: ws://localhost:8765

## 5. Monitoring Real-time Performance

### Các metrics được track:
- **Financial Metrics**: P&L, Sharpe Ratio, VaR, Drawdown
- **System Metrics**: CPU, Memory, API Response Time
- **Trading Metrics**: Win Rate, Average Trade Duration
- **Risk Metrics**: Position Sizing, Correlation Analysis

### Alerts được gửi khi:
- Drawdown > 10%
- System CPU > 80%
- API Response Time > 100ms
- Error Rate > 5%

## 6. Logs và Debugging

Logs được lưu trong thư mục `logs/`:
- `trading_bot.log`: Logs chính của bot
- `performance_monitor.log`: Logs performance monitoring
- `error.log`: Logs lỗi

## 7. Testing Performance Monitoring

```bash
python test_real_time_performance_monitoring.py
```

## 8. Troubleshooting

### Nếu bot không khởi động:
1. Kiểm tra API keys trong file .env
2. Kiểm tra kết nối internet
3. Kiểm tra logs trong thư mục logs/

### Nếu Performance Dashboard không hiển thị:
1. Kiểm tra port 8050 có bị chiếm không
2. Kiểm tra WebSocket connection
3. Refresh browser và clear cache

### Nếu không nhận được alerts:
1. Kiểm tra cấu hình Discord/Telegram
2. Kiểm tra webhook URLs
3. Kiểm tra bot permissions 