# 🚀 Hướng dẫn Chạy Trading Bot với Real-time Performance Tracking

## 📋 Mục lục
1. [Chuẩn bị môi trường](#1-chuẩn-bị-môi-trường)
2. [Cấu hình API Keys](#2-cấu-hình-api-keys)
3. [Chạy Bot](#3-chạy-bot)
4. [Performance Dashboard](#4-performance-dashboard)
5. [Monitoring Real-time Performance](#5-monitoring-real-time-performance)
6. [Testing](#6-testing)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Chuẩn bị môi trường

### 1.1 Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 1.2 Tạo thư mục logs
```bash
mkdir logs
```

---

## 2. Cấu hình API Keys

### 2.1 Tạo file .env
Tạo file `.env` trong thư mục gốc với nội dung:

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

### 2.2 Lấy API Keys

#### Binance API:
1. Đăng nhập vào Binance
2. Vào API Management
3. Tạo API Key mới
4. Copy API Key và Secret vào file .env

#### Discord (Optional):
1. Tạo Discord Server
2. Tạo Webhook URL
3. Copy Webhook URL vào file .env

#### Telegram (Optional):
1. Tạo bot với @BotFather
2. Lấy Bot Token
3. Copy Token vào file .env

---

## 3. Chạy Bot

### 3.1 Cách 1: Chạy với script tự động
```bash
python run_bot.py
```

### 3.2 Cách 2: Chạy trực tiếp
```bash
python main_with_quantitative.py
```

### 3.3 Cách 3: Chạy với Docker
```bash
docker-compose up -d
```

### 3.4 Cách 4: Chạy từng component riêng biệt

#### Terminal 1: Chạy bot chính
```bash
python main_with_quantitative.py
```

#### Terminal 2: Chạy Performance Dashboard
```bash
python run_dashboard.py
```

---

## 4. Performance Dashboard

### 4.1 Truy cập Dashboard
- **URL**: http://localhost:8050
- **WebSocket**: ws://localhost:8765

### 4.2 Features của Dashboard
- 📊 **Real-time Financial Metrics**
  - P&L tracking
  - Sharpe Ratio
  - VaR calculation
  - Drawdown monitoring

- 🖥️ **System Health Monitoring**
  - CPU usage
  - Memory usage
  - API response time
  - Error rate

- 📈 **Performance Charts**
  - Portfolio performance
  - Risk metrics
  - Trading activity

- ⚠️ **Alert Notifications**
  - Performance alerts
  - System alerts
  - Risk alerts

- 🎯 **Risk Analysis**
  - Position sizing
  - Correlation analysis
  - Volatility tracking

---

## 5. Monitoring Real-time Performance

### 5.1 Các Metrics được Track

#### Financial Metrics:
- **Total Return**: Tổng lợi nhuận
- **Sharpe Ratio**: Tỷ lệ Sharpe
- **VaR (Value at Risk)**: Rủi ro giá trị
- **Maximum Drawdown**: Mức sụt giảm tối đa
- **Win Rate**: Tỷ lệ thắng
- **Average Trade Duration**: Thời gian giao dịch trung bình

#### System Metrics:
- **CPU Usage**: Sử dụng CPU
- **Memory Usage**: Sử dụng bộ nhớ
- **API Response Time**: Thời gian phản hồi API
- **Error Rate**: Tỷ lệ lỗi
- **Cache Hit Rate**: Tỷ lệ cache hit

#### Trading Metrics:
- **Position Count**: Số lượng vị thế
- **Active Orders**: Lệnh đang hoạt động
- **Risk per Trade**: Rủi ro mỗi giao dịch
- **Portfolio Correlation**: Tương quan danh mục

### 5.2 Alerts System

#### Performance Alerts:
- Drawdown > 10%
- Sharpe Ratio < 0.5
- VaR > 5%
- Win Rate < 40%

#### System Alerts:
- CPU Usage > 80%
- Memory Usage > 80%
- API Response Time > 100ms
- Error Rate > 5%

#### Trading Alerts:
- Position Size > 20% of portfolio
- Correlation > 0.7
- Volatility > 5%

### 5.3 Real-time Data Flow

```
Trading Bot → Performance Monitor → WebSocket → Dashboard
     ↓              ↓                    ↓           ↓
  Signals    →  Metrics    →    Real-time    →  Visualization
     ↓              ↓                    ↓           ↓
  Analysis   →  Alerts    →    Broadcasting →  User Interface
```

---

## 6. Testing

### 6.1 Test Performance Monitoring
```bash
python test_performance_monitoring.py
```

### 6.2 Test Real-time Performance
```bash
python test_real_time_performance_monitoring.py
```

### 6.3 Test Dashboard
```bash
python run_dashboard.py
```

---

## 7. Troubleshooting

### 7.1 Bot không khởi động

#### Kiểm tra:
1. **API Keys**: Đảm bảo API keys đúng trong file .env
2. **Dependencies**: Chạy `pip install -r requirements.txt`
3. **Logs**: Kiểm tra logs trong thư mục `logs/`

#### Lỗi thường gặp:
```bash
# Lỗi API key
❌ Error: Invalid API key
✅ Fix: Kiểm tra lại API key trong file .env

# Lỗi dependencies
❌ Error: ModuleNotFoundError
✅ Fix: pip install -r requirements.txt

# Lỗi kết nối
❌ Error: Connection timeout
✅ Fix: Kiểm tra internet connection
```

### 7.2 Dashboard không hiển thị

#### Kiểm tra:
1. **Port 8050**: Đảm bảo port không bị chiếm
2. **WebSocket**: Kiểm tra kết nối WebSocket
3. **Browser**: Refresh và clear cache

#### Lệnh kiểm tra:
```bash
# Kiểm tra port
netstat -an | grep 8050

# Kiểm tra WebSocket
curl -I http://localhost:8050
```

### 7.3 Không nhận được alerts

#### Kiểm tra:
1. **Discord/Telegram**: Kiểm tra webhook URLs
2. **Bot permissions**: Đảm bảo bot có quyền gửi message
3. **Alert thresholds**: Kiểm tra cấu hình alerts

### 7.4 Performance monitoring không hoạt động

#### Kiểm tra:
1. **WebSocket connection**: ws://localhost:8765
2. **Monitor service**: Kiểm tra service đang chạy
3. **Data flow**: Kiểm tra luồng dữ liệu

---

## 8. Monitoring Commands

### 8.1 Kiểm tra logs
```bash
# Logs chính
tail -f logs/trading_bot.log

# Performance logs
tail -f logs/performance_monitor.log

# Dashboard logs
tail -f logs/dashboard.log
```

### 8.2 Kiểm tra processes
```bash
# Kiểm tra Python processes
ps aux | grep python

# Kiểm tra ports
netstat -tulpn | grep :8050
netstat -tulpn | grep :8765
```

### 8.3 Restart services
```bash
# Kill processes
pkill -f "python.*main_with_quantitative"
pkill -f "python.*run_dashboard"

# Restart
python run_bot.py
```

---

## 9. Performance Optimization

### 9.1 Tối ưu hóa hệ thống
- **CPU**: Giới hạn 80% usage
- **Memory**: Giới hạn 80% usage
- **Network**: Tối ưu API calls
- **Storage**: Rotate logs

### 9.2 Tối ưu hóa trading
- **Risk Management**: Strict position sizing
- **Correlation**: Monitor portfolio correlation
- **Volatility**: Adjust to market conditions
- **Timing**: Optimize entry/exit timing

---

## 10. Security Best Practices

### 10.1 API Security
- Sử dụng API keys với quyền hạn tối thiểu
- Không chia sẻ API keys
- Rotate keys định kỳ

### 10.2 System Security
- Firewall configuration
- Regular updates
- Backup strategies
- Monitoring alerts

---

## 🎯 Kết luận

Trading bot với real-time performance tracking đã sẵn sàng để chạy. Hệ thống bao gồm:

✅ **WorldQuant-level quantitative analysis**  
✅ **Real-time performance monitoring**  
✅ **Interactive dashboard**  
✅ **Automated alerts**  
✅ **Risk management**  
✅ **Comprehensive testing**  

Bắt đầu với `python run_bot.py` và truy cập dashboard tại http://localhost:8050! 