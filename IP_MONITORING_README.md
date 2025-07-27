# IP Monitoring System

## Tổng quan

Hệ thống IP Monitoring được thiết kế để tự động phát hiện và thông báo khi địa chỉ IP thay đổi, giúp giải quyết vấn đề whitelist IP trên Binance API.

## Tính năng chính

### 1. Tự động phát hiện thay đổi IP
- Monitor địa chỉ IP công khai theo định kỳ
- Phát hiện khi IP thay đổi
- Sử dụng nhiều dịch vụ IP check để đảm bảo độ tin cậy

### 2. Phát hiện lỗi IP từ Binance API
- Tự động phát hiện các lỗi liên quan đến IP whitelist
- Phân tích mã lỗi Binance API
- Xử lý thông minh để tránh spam notification

### 3. Thông báo tự động
- Gửi thông báo qua Telegram/Discord khi phát hiện thay đổi IP
- Cung cấp thông tin chi tiết về IP cũ và mới
- Hướng dẫn cách thêm IP vào whitelist

## Cấu hình

### Environment Variables

Thêm các biến môi trường sau vào file `.env`:

```env
# IP Monitoring Configuration
IP_MONITOR_ENABLED=true
IP_MONITOR_CHECK_INTERVAL=300
IP_MONITOR_NOTIFICATION_COOLDOWN=300

# Notification Services
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

DISCORD_ENABLED=true
DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

### Cấu hình trong config

```python
'ip_monitor': {
    'enabled': True,  # Bật/tắt IP monitoring
    'check_interval': 300,  # Kiểm tra IP mỗi 5 phút
    'notification_cooldown': 300,  # Cooldown giữa các notification (5 phút)
    'ip_services': [
        'https://api.ipify.org',
        'https://httpbin.org/ip',
        'https://ipinfo.io/ip',
        'https://icanhazip.com'
    ]
}
```

## Cách hoạt động

### 1. Khởi tạo
- Bot khởi động và lấy IP hiện tại
- Bắt đầu monitoring trong background
- Kiểm tra IP theo interval đã cấu hình

### 2. Phát hiện thay đổi IP
- So sánh IP hiện tại với IP trước đó
- Nếu khác nhau, gửi notification
- Cập nhật IP hiện tại

### 3. Phát hiện lỗi API
- Khi gọi Binance API, kiểm tra response
- Nếu có lỗi IP-related, gửi notification
- Bao gồm IP hiện tại trong thông báo

### 4. Thông báo
- Gửi message chi tiết qua Telegram/Discord
- Bao gồm thời gian, IP cũ, IP mới
- Hướng dẫn cách thêm IP vào whitelist

## Ví dụ thông báo

### Thay đổi IP
```
🚨 **IP ADDRESS CHANGE DETECTED** 🚨

**Time:** 2024-01-15 14:30:25
**Old IP:** 192.168.1.100
**New IP:** 192.168.1.101

⚠️ **Action Required:**
1. Log into your Binance account
2. Go to API Management
3. Add the new IP address to whitelist: **192.168.1.101**
4. Remove old IP if no longer needed: **192.168.1.100**

🔗 **Quick Links:**
• Binance API Management: https://www.binance.com/en/my/settings/api-management
• IP Check: https://whatismyipaddress.com/

The bot will resume trading once the new IP is whitelisted.
```

### Lỗi API
```
🚨 **BINANCE API IP ERROR DETECTED** 🚨

**Time:** 2024-01-15 14:30:25
**Error:** Invalid API-key, IP, or permissions for action
**Current IP:** 192.168.1.101

⚠️ **Action Required:**
1. Log into your Binance account
2. Go to API Management
3. Add this IP address to whitelist: **192.168.1.101**
4. Ensure API key has proper permissions

🔗 **Quick Links:**
• Binance API Management: https://www.binance.com/en/my/settings/api-management
• IP Check: https://whatismyipaddress.com/

The bot will resume trading once the IP is whitelisted.
```

## Testing

Chạy script test để kiểm tra tính năng:

```bash
python test_ip_monitor.py
```

Script này sẽ:
1. Test IP monitoring
2. Test IP error detection
3. Gửi notification test

## Troubleshooting

### IP không được phát hiện
- Kiểm tra kết nối internet
- Kiểm tra các dịch vụ IP check có hoạt động không
- Xem log để debug

### Notification không gửi được
- Kiểm tra cấu hình Telegram/Discord
- Kiểm tra bot token và webhook URL
- Xem log để debug

### Bot không khởi động
- Kiểm tra cấu hình IP monitoring
- Kiểm tra các dependency
- Xem log để debug

## Logic chống spam notification

### IP Change Notification Logic:
1. **Lần đầu phát hiện IP mới**: Luôn gửi notification
2. **IP không thay đổi**: Không gửi notification (tránh spam)
3. **IP thay đổi sang IP khác**: Gửi notification mới
4. **Sau cooldown period**: Có thể gửi lại notification cho cùng IP

### IP Error Notification Logic:
1. **Lần đầu gặp lỗi IP**: Luôn gửi notification
2. **Cùng IP gặp lỗi lại**: Không gửi notification (tránh spam)
3. **IP khác gặp lỗi**: Gửi notification mới
4. **Sau cooldown period**: Có thể gửi lại notification cho cùng IP

### Ví dụ thực tế:
```
IP A (192.168.1.100) -> Notification ✅ (lần đầu)
IP A (192.168.1.100) -> Không gửi ❌ (tránh spam)
IP B (192.168.1.101) -> Notification ✅ (IP mới)
IP A (192.168.1.100) -> Notification ✅ (sau cooldown)
```

## Lưu ý quan trọng

1. **Smart Cooldown**: Hệ thống có cooldown thông minh để tránh spam notification
2. **IP-based Tracking**: Theo dõi IP đã thông báo để tránh spam
3. **Multiple IP Services**: Sử dụng nhiều dịch vụ để đảm bảo độ tin cậy
4. **Error Handling**: Xử lý lỗi gracefully để không ảnh hưởng đến trading
5. **Background Monitoring**: IP monitoring chạy trong background
6. **Automatic Cleanup**: Tự động cleanup khi bot shutdown

## Cập nhật

Để cập nhật IP monitoring:

1. Cập nhật code
2. Restart bot
3. Kiểm tra log để đảm bảo hoạt động bình thường

## Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra log file
2. Chạy test script
3. Kiểm tra cấu hình
4. Liên hệ support nếu cần 