# 🔧 Troubleshooting Guide

## 🚨 Common Issues & Solutions

### **1. Docker Desktop Connection Error**

**Error:** `unable to get image 'ghcr.io/nats-nui/nui:latest': error during connect`

**Solution:**
```bash
# 1. Start Docker Desktop
# 2. Wait for Docker to fully initialize
# 3. Run the startup script
./start-services.sh
```

### **2. Environment Variable Warnings**

**Error:** `The "PYTHONPATH" variable is not set`

**Solution:** ✅ Fixed in docker-compose.yml
- Added `PYTHONPATH=/app` to environment variables

### **3. Docker Compose Command Issues**

**Error:** `docker-compose: command not found`

**Solution:**
```bash
# Use the new Docker Compose syntax
docker compose up -d

# Or use the startup script
./start-services.sh
```

### **4. Image Pull Failures**

**Error:** `unable to pull image`

**Solution:**
```bash
# Pull images manually
docker pull nats:2.9-alpine
docker pull redis:7-alpine
docker pull ghcr.io/nats-nui/nui:latest
docker pull rediscommander/redis-commander:latest

# Then start services
./start-services.sh
```

### **5. Port Conflicts**

**Error:** `port is already allocated`

**Solution:**
```bash
# Check what's using the ports
netstat -ano | findstr :8081
netstat -ano | findstr :8082
netstat -ano | findstr :8083
netstat -ano | findstr :8000

# Stop conflicting services or change ports in docker-compose.yml
```

## 🛠️ Quick Fixes

### **Start Services Properly:**
```bash
# Method 1: Use startup script
./start-services.sh

# Method 2: Manual commands
docker compose pull
docker compose build --no-cache
docker compose up -d
```

### **Check Service Status:**
```bash
# View all services
docker compose ps

# View logs
docker compose logs -f trading-bot
docker compose logs -f cache-monitor
```

### **Restart Services:**
```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart trading-bot
```

### **Clean Up:**
```bash
# Stop all services
docker compose down

# Remove volumes (WARNING: This will delete data)
docker compose down -v

# Remove images and rebuild
docker compose down --rmi all
docker compose build --no-cache
```

## 🔍 Debug Commands

### **Check Docker Status:**
```bash
# Verify Docker is running
docker info

# Check Docker Compose version
docker compose version
```

### **Check Network:**
```bash
# List networks
docker network ls

# Inspect network
docker network inspect trading_bot_new_nats-network
```

### **Check Containers:**
```bash
# List all containers
docker ps -a

# Inspect specific container
docker inspect trading-bot
```

## 📊 Service URLs

Once services are running:

- **NATS UI**: http://localhost:8081
- **Redis Commander**: http://localhost:8082  
- **Cache Monitor**: http://localhost:8083
- **Trading Bot**: http://localhost:8000

## 🚀 Performance Monitoring

### **Check Cache Performance:**
```bash
# View cache monitor logs
docker compose logs -f cache-monitor

# Check Redis stats
docker exec redis-cache redis-cli info
```

### **Monitor System Resources:**
```bash
# Check container resource usage
docker stats

# Check disk usage
docker system df
```

## 🔒 Security Notes

- ✅ Using non-root user in containers
- ✅ Updated base images with security patches
- ✅ Removed unnecessary packages and caches
- ✅ Set proper file permissions

## 📞 Support

If issues persist:

1. **Check Docker Desktop** is running and updated
2. **Restart Docker Desktop** completely
3. **Use the startup script**: `./start-services.sh`
4. **Check logs**: `docker compose logs -f`
5. **Verify ports** are not in use by other applications 