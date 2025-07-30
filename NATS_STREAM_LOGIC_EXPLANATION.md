# NATS Stream Logic - Đọc/Ghi Events

## Tổng Quan Hệ Thống NATS

NATS (Neural Autonomic Transport System) được sử dụng trong trading bot như một **message broker** để đồng bộ hóa cache và monitoring giữa các services. Đây là một hệ thống **publish-subscribe** phân tán.

## Kiến Trúc NATS Trong Trading Bot

### 1. **NATS Server Setup**
```yaml
# docker-compose.yml
nats:
  image: nats:2.9-alpine
  ports:
    - "4223:4222"      # NATS client port
    - "8223:8222"      # NATS HTTP monitoring
  command: -js -m 8222  # JetStream + monitoring
```

### 2. **NATS UI (Management Interface)**
```yaml
nui:
  image: ghcr.io/nats-nui/nui:latest
  ports:
    - "8081:31311"
  environment:
    - NATS_URL=nats://nats:4222
```

## Logic Đọc/Ghi NATS Stream

### **1. Kết Nối NATS**

#### **Khởi Tạo Connection**
```python
# src/utils/advanced_cache_manager.py
import nats
from nats.aio.client import Client as NATS

class AdvancedCacheManager:
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.nats_client = None
    
    async def initialize(self):
        # Tạo NATS client
        self.nats_client = NATS()
        # Kết nối đến NATS server
        await self.nats_client.connect(self.nats_url)
        logger.info("NATS connection established")
```

### **2. Subscribe (Đọc Events)**

#### **A. Subscribe Cache Events**
```python
async def _subscribe_to_cache_events(self):
    """Subscribe to NATS cache events."""
    try:
        # Subscribe cache invalidation events
        await self.nats_client.subscribe(
            "cache.invalidate",           # Topic name
            cb=self._handle_cache_invalidation  # Callback function
        )
        
        # Subscribe cache statistics events  
        await self.nats_client.subscribe(
            "cache.stats",
            cb=self._handle_cache_stats
        )
        
        # Subscribe performance events
        await self.nats_client.subscribe(
            "cache.performance", 
            cb=self._handle_performance_event
        )
        
    except Exception as e:
        logger.error(f"Error subscribing to cache events: {str(e)}")
```

#### **B. Handle Incoming Messages**
```python
async def _handle_cache_invalidation(self, msg: Msg):
    """Handle cache invalidation events."""
    try:
        # Decode JSON message
        data = json.loads(msg.data.decode())
        pattern = data.get('pattern')
        keys = data.get('keys', [])
        
        if pattern:
            await self._invalidate_by_pattern(pattern)
        elif keys:
            await self._invalidate_keys(keys)
            
    except Exception as e:
        logger.error(f"Error handling cache invalidation: {str(e)}")

async def _handle_cache_stats(self, msg: Msg):
    """Handle cache statistics events."""
    try:
        data = json.loads(msg.data.decode())
        # Update local stats with distributed stats
        for key, value in data.items():
            if key in self.cache_stats:
                self.cache_stats[key] += value
                
    except Exception as e:
        logger.error(f"Error handling cache stats: {str(e)}")
```

### **3. Publish (Ghi Events)**

#### **A. Publish Cache Updates**
```python
async def _publish_cache_update(self, key: str, value: Any, ttl: int):
    """Publish cache update to NATS."""
    try:
        # Tạo message JSON
        message = {
            'action': 'update',
            'key': key,
            'value': value,
            'ttl': ttl,
            'timestamp': datetime.now().isoformat()
        }
        
        # Publish đến topic "cache.update"
        await self.nats_client.publish(
            "cache.update",                           # Topic
            json.dumps(message, default=str).encode() # Message data
        )
        
    except Exception as e:
        logger.error(f"Error publishing cache update: {str(e)}")
```

#### **B. Publish Cache Invalidation**
```python
async def _publish_cache_invalidation(self, key: str):
    """Publish cache invalidation to NATS."""
    try:
        message = {
            'action': 'invalidate',
            'key': key,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.nats_client.publish(
            "cache.invalidate",
            json.dumps(message, default=str).encode()
        )
        
    except Exception as e:
        logger.error(f"Error publishing cache invalidation: {str(e)}")
```

#### **C. Publish Pattern Invalidation**
```python
async def _publish_pattern_invalidation(self, pattern: str):
    """Publish pattern invalidation to NATS."""
    try:
        message = {
            'action': 'invalidate_pattern',
            'pattern': pattern,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.nats_client.publish(
            "cache.invalidate",
            json.dumps(message, default=str).encode()
        )
        
    except Exception as e:
        logger.error(f"Error publishing pattern invalidation: {str(e)}")
```

### **4. Cache Monitor Service**

#### **A. Subscribe Monitoring Events**
```python
# src/services/cache_monitor_service.py
async def _subscribe_to_monitoring_events(self):
    """Subscribe to monitoring events."""
    try:
        # Subscribe performance events
        await self.nats_client.subscribe(
            "cache.performance",
            cb=self._handle_performance_event
        )
        
        # Subscribe statistics events
        await self.nats_client.subscribe(
            "cache.stats", 
            cb=self._handle_stats_event
        )
        
        # Subscribe error events
        await self.nats_client.subscribe(
            "cache.error",
            cb=self._handle_error_event
        )
        
    except Exception as e:
        logger.error(f"Error subscribing to monitoring events: {str(e)}")
```

#### **B. Handle Performance Events**
```python
async def _handle_performance_event(self, msg: Msg):
    """Handle performance monitoring events."""
    try:
        data = json.loads(msg.data.decode())
        
        # Store performance data
        self.monitoring_data['performance_metrics'].append({
            'timestamp': datetime.now().isoformat(),
            'avg_response_time': data.get('avg_response_time', 0),
            'cache_hit_rate': data.get('cache_hit_rate', 0),
            'memory_usage': data.get('memory_usage', 0)
        })
        
        # Update performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'response_time': data.get('avg_response_time', 0),
            'hit_rate': data.get('cache_hit_rate', 0),
            'memory_usage': data.get('memory_usage', 0)
        })
        
    except Exception as e:
        logger.error(f"Error handling performance event: {str(e)}")
```

## Flow Hoạt Động NATS

### **1. Cache Update Flow**
```
Trading Bot → Cache Update → NATS Publish → Other Services Subscribe
     ↓              ↓              ↓                    ↓
Set Cache → _publish_cache_update → "cache.update" → Handle Update
```

### **2. Cache Invalidation Flow**
```
Cache Invalidation → NATS Publish → Distributed Invalidation
        ↓                ↓                ↓
Delete Cache → "cache.invalidate" → All Services Clear Cache
```

### **3. Performance Monitoring Flow**
```
Performance Data → NATS Publish → Monitor Service → Dashboard
       ↓              ↓              ↓              ↓
Cache Stats → "cache.performance" → Store Data → Display Metrics
```

## Topics NATS Được Sử Dụng

### **1. Cache Management Topics**
- `cache.update` - Thông báo cache được cập nhật
- `cache.invalidate` - Thông báo cache bị xóa
- `cache.stats` - Thống kê cache performance

### **2. Monitoring Topics**
- `cache.performance` - Metrics performance
- `cache.error` - Error events
- `cache.alert` - Alert notifications

## Message Format

### **Cache Update Message**
```json
{
  "action": "update",
  "key": "trading_signal_BTCUSDT",
  "value": {
    "signal": "buy",
    "confidence": 0.85,
    "timestamp": "2025-07-30T08:45:23"
  },
  "ttl": 3600,
  "timestamp": "2025-07-30T08:45:23.123456"
}
```

### **Cache Invalidation Message**
```json
{
  "action": "invalidate",
  "key": "trading_signal_BTCUSDT",
  "timestamp": "2025-07-30T08:45:23.123456"
}
```

### **Performance Message**
```json
{
  "avg_response_time": 0.0023,
  "cache_hit_rate": 0.85,
  "memory_usage": 45,
  "timestamp": "2025-07-30T08:45:23.123456"
}
```

## Lợi Ích Của NATS Integration

### **1. Distributed Cache Synchronization**
- Tất cả services được đồng bộ cache
- Invalidation được broadcast toàn bộ hệ thống
- Consistency across multiple instances

### **2. Real-time Monitoring**
- Performance metrics được publish real-time
- Centralized monitoring dashboard
- Alert system for cache issues

### **3. Scalability**
- Horizontal scaling với multiple cache instances
- Load balancing tự động
- Fault tolerance với NATS clustering

### **4. Performance**
- Low latency message delivery
- High throughput cho cache operations
- Efficient memory usage

## Error Handling

### **1. Connection Errors**
```python
try:
    await self.nats_client.connect(self.nats_url)
except Exception as e:
    logger.error(f"NATS connection failed: {str(e)}")
    # Fallback to local cache only
```

### **2. Message Processing Errors**
```python
async def _handle_cache_invalidation(self, msg: Msg):
    try:
        data = json.loads(msg.data.decode())
        # Process message
    except Exception as e:
        logger.error(f"Error handling cache invalidation: {str(e)}")
        # Continue processing other messages
```

### **3. Publish Errors**
```python
async def _publish_cache_update(self, key: str, value: Any, ttl: int):
    try:
        await self.nats_client.publish("cache.update", message)
    except Exception as e:
        logger.error(f"Error publishing cache update: {str(e)}")
        # Continue with local cache operations
```

## Monitoring & Debugging

### **1. NATS UI Dashboard**
- URL: `http://localhost:8081`
- Monitor message flow
- View connection status
- Debug message content

### **2. Log Monitoring**
```bash
# View NATS server logs
docker logs nats-server-quantitative

# View cache manager logs
docker logs trading-bot-quantitative
```

### **3. Performance Metrics**
- Message throughput
- Response times
- Error rates
- Cache hit rates

## Kết Luận

NATS stream logic trong trading bot cung cấp:

1. **Distributed Cache Management** - Đồng bộ cache across services
2. **Real-time Monitoring** - Performance tracking và alerting
3. **Scalable Architecture** - Support multiple instances
4. **Fault Tolerance** - Graceful error handling
5. **High Performance** - Low latency message delivery

Hệ thống này đảm bảo trading bot có thể scale horizontally và maintain cache consistency across tất cả services. 