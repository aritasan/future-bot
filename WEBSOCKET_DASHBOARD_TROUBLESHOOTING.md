# 🔧 WebSocket & Dashboard Troubleshooting Guide

## 📊 **Current Status**

### ✅ **Working Components:**
- **Dashboard Server**: ✅ Running on http://localhost:8050
- **Trading Bot**: ✅ Running and processing symbols
- **WebSocket Server**: ✅ Running on ws://localhost:8765

### ❌ **Issues Identified:**
- **WebSocket Internal Error**: `received 1011 (internal error)`
- **Dashboard Connection**: Shows "Disconnected - Retrying..."

## 🔍 **Root Cause Analysis**

### **Issue 1: WebSocket Internal Error (1011)**
**Problem**: WebSocket server is running but sending malformed data or causing internal errors.

**Causes**:
1. **Complex Data Structure**: `get_real_time_summary()` returns complex nested objects
2. **JSON Serialization Issues**: Some objects can't be serialized to JSON
3. **Async Function Calls**: Calling async functions in sync context

### **Issue 2: Dashboard Connection Failure**
**Problem**: Dashboard can't establish stable WebSocket connection.

**Causes**:
1. **Data Format Mismatch**: Dashboard expects specific data format
2. **Connection Timeout**: WebSocket connection times out
3. **Error Handling**: Poor error handling in WebSocket client

## 🛠️ **Solutions**

### **Solution 1: Fix WebSocket Data Format**

The issue is in `get_real_time_summary()` method. Let's create a simplified version:

```python
async def get_real_time_summary(self) -> Dict[str, Any]:
    """Get simplified real-time summary for WebSocket."""
    try:
        # Create simple, serializable data structure
        summary = {
            'performance_score': float(self.monitoring_state.get('performance_score', 0.0)),
            'risk_score': float(self.monitoring_state.get('risk_score', 0.0)),
            'stability_score': float(self.monitoring_state.get('stability_score', 0.0)),
            'timestamp': datetime.now().isoformat(),
            'alerts_count': len(self.monitoring_state.get('alerts', [])),
            'system_status': 'active'
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting real-time summary: {str(e)}")
        return {
            'performance_score': 0.0,
            'risk_score': 0.0,
            'stability_score': 0.0,
            'timestamp': datetime.now().isoformat(),
            'alerts_count': 0,
            'system_status': 'error'
        }
```

### **Solution 2: Improve WebSocket Handler**

```python
async def websocket_handler(websocket, path):
    """Simplified WebSocket handler."""
    try:
        self.monitoring_state['websocket_clients'].add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        # Send initial data
        try:
            initial_data = await self.get_real_time_summary()
            await websocket.send(json.dumps(initial_data, default=str))
            logger.info("Sent initial data successfully")
        except Exception as e:
            logger.error(f"Error sending initial data: {str(e)}")
            # Send fallback data
            fallback_data = {
                'performance_score': 0.0,
                'risk_score': 0.0,
                'stability_score': 0.0,
                'timestamp': datetime.now().isoformat(),
                'status': 'fallback'
            }
            await websocket.send(json.dumps(fallback_data))
        
        # Keep connection alive
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                # Send performance data
                data = await self.get_real_time_summary()
                await websocket.send(json.dumps(data, default=str))
                
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket connection closed by client")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                break
                
    except Exception as e:
        logger.error(f"WebSocket handler error: {str(e)}")
    finally:
        self.monitoring_state['websocket_clients'].discard(websocket)
        logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
```

### **Solution 3: Dashboard Connection Retry Logic**

```python
def start_websocket_client(self):
    """Start WebSocket client with retry logic."""
    def websocket_client():
        async def connect():
            retry_count = 0
            max_retries = 10
            
            while retry_count < max_retries:
                try:
                    async with websockets.connect(self.websocket_url) as websocket:
                        print(f"✅ Connected to WebSocket server: {self.websocket_url}")
                        self.websocket_connected = True
                        retry_count = 0  # Reset retry count on success
                        
                        async for message in websocket:
                            try:
                                data = json.loads(message)
                                print(f"📊 Received data: {data}")
                                self.latest_data = data
                            except json.JSONDecodeError as e:
                                print(f"❌ Error parsing JSON: {e}")
                                
                except Exception as e:
                    print(f"❌ WebSocket error: {e}")
                    self.websocket_connected = False
                    retry_count += 1
                    wait_time = min(5 * retry_count, 30)  # Exponential backoff
                    print(f"🔄 Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
            
            print("❌ Max retries exceeded. WebSocket connection failed.")
        
        asyncio.run(connect())
    
    # Start WebSocket client in a separate thread
    thread = threading.Thread(target=websocket_client, daemon=True)
    thread.start()
```

## 🚀 **Quick Fix Steps**

### **Step 1: Restart the System**
```bash
# Stop all running processes
pkill -f "python run_bot.py"
pkill -f "python run_dashboard.py"

# Wait a moment
sleep 2

# Start bot first
python run_bot.py &

# Wait for bot to initialize
sleep 5

# Start dashboard
python run_dashboard.py &
```

### **Step 2: Test Connection**
```bash
# Test WebSocket
python check_websocket_port.py

# Test dashboard
curl http://localhost:8050
```

### **Step 3: Monitor Logs**
```bash
# Monitor bot logs
tail -f logs/trading_bot_quantitative_20250731.log

# Monitor dashboard logs (if any)
tail -f logs/dashboard.log
```

## 📋 **Expected Results After Fix**

### **✅ Success Indicators:**
- Dashboard shows "Connected" status
- Performance metrics display real values (not 0)
- WebSocket connection established without errors
- Real-time data updates every 5 seconds

### **📊 Dashboard Should Show:**
- **Performance Score**: 15-25 (real values)
- **Risk Score**: 10-30 (real values)
- **Stability Score**: 80-100 (real values)
- **Connection Status**: "Connected" (green)

## 🔧 **Alternative Solutions**

### **Option 1: Use Simple WebSocket Server**
If the main WebSocket continues to have issues, we can use a simplified version:

```bash
# Start simple WebSocket server
python test_simple_websocket.py &

# Start simple dashboard
python test_simple_dashboard.py &
```

### **Option 2: HTTP Polling Instead of WebSocket**
If WebSocket continues to fail, we can implement HTTP polling:

```python
# Dashboard polls for data every 5 seconds
@app.callback(
    Output("data-store", "data"),
    [Input("interval-component", "n_intervals")]
)
def update_data_store(n):
    # Make HTTP request to get latest data
    try:
        response = requests.get("http://localhost:8000/api/performance")
        return response.json()
    except:
        return {"error": "No data available"}
```

## 🎯 **Next Steps**

1. **Implement the fixes above**
2. **Test the WebSocket connection**
3. **Verify dashboard displays real data**
4. **Monitor system stability**

## 📞 **Support**

If issues persist after implementing these fixes:
1. Check the logs for specific error messages
2. Verify all dependencies are installed
3. Ensure ports 8765 and 8050 are available
4. Consider using the simple WebSocket server as a fallback 