import requests
import time
import telegram

# Replace with your actual Telegram Bot token and chat ID
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'

bot = telegram.Bot(token=TELEGRAM_TOKEN)

def get_public_ip():
    return requests.get('https://api.ipify.org').text

last_ip = None

while True:
    current_ip = get_public_ip()
    if current_ip != last_ip:
        message = f'🔔 IP của bạn đã thay đổi!\nIP mới: {current_ip}'
        bot.send_message(chat_id=CHAT_ID, text=message)
        last_ip = current_ip
    time.sleep(600)  # Check mỗi 10 phút
