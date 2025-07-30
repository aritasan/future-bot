import time
import requests
from datetime import datetime

def check_system_time():
    """Check system time against online time services."""
    print("Checking system time...")
    print("=" * 50)
    
    # Get local time
    local_time = time.time()
    local_datetime = datetime.fromtimestamp(local_time)
    print(f"Local time: {local_datetime}")
    print(f"Local timestamp: {local_time}")
    
    # Try to get time from multiple online services
    time_services = [
        "https://worldtimeapi.org/api/timezone/Etc/UTC",
        "https://httpbin.org/headers",
        "https://api.ipify.org"
    ]
    
    for service in time_services:
        try:
            response = requests.get(service, timeout=10)
            if response.status_code == 200:
                print(f"✅ Successfully connected to {service}")
                # Get server time from response headers
                server_time = response.headers.get('date')
                if server_time:
                    print(f"Server time header: {server_time}")
            else:
                print(f"❌ Failed to connect to {service}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error connecting to {service}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Recommendations:")
    print("1. If your system time is significantly off, sync it with an NTP server")
    print("2. On Windows, you can sync time by:")
    print("   - Right-click on the clock in taskbar")
    print("   - Select 'Adjust date/time'")
    print("   - Click 'Sync now' under 'Synchronize your clock'")
    print("3. On Linux, use: sudo ntpdate -s time.nist.gov")

if __name__ == "__main__":
    check_system_time() 