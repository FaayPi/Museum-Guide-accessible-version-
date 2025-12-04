# Deployment Guide

Production deployment guide for the Museum Guide App with scalability and monitoring.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Local Development](#local-development)
4. [Production Deployment](#production-deployment)
5. [Monitoring & Logging](#monitoring--logging)
6. [Health Checks](#health-checks)
7. [Scaling](#scaling)

---

## Prerequisites

### Required Services

- **OpenAI API** - GPT-4 Vision, Chat, TTS
- **Pinecone** - Vector database for RAG
- **Python 3.10+**

### System Requirements

**Minimum:**
- 2 CPU cores
- 4GB RAM
- 10GB disk space

**Recommended (Production):**
- 4+ CPU cores
- 8GB+ RAM
- 50GB disk space (for logs and cache)

---

## Environment Configuration

### 1. Copy Environment Template

```bash
cp .env.example .env
```

### 2. Configure Variables

Edit `.env` with your values:

```bash
# API Keys (Required)
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...

# Environment
ENVIRONMENT=production  # Options: development, production, testing

# Logging
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=logs/app.log

# Server
HOST=0.0.0.0  # Bind to all interfaces for Docker/Cloud
PORT=7860

# Performance
MAX_WORKERS=4
REQUEST_TIMEOUT=30

# Features
ENABLE_RAG=true
ENABLE_CACHE=true
ENABLE_METRICS=true

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60
MAX_REQUESTS_PER_HOUR=1000
```

### 3. Validate Configuration

```bash
python -c "import config; print(f'Environment: {config.ENVIRONMENT}')"
```

---

## Local Development

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application

```bash
# Development mode (with auto-reload)
ENVIRONMENT=development python app.py

# Access at http://localhost:7860
```

### 3. View Logs

```bash
# Follow logs in real-time
tail -f logs/app.log

# View metrics
cat logs/metrics.jsonl | jq
```

---

## Production Deployment

### Option 1: Docker (Recommended)

#### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.core.health_check import liveness_probe; liveness_probe()"

# Run application
CMD ["python", "app.py"]
```

#### Build and Run

```bash
# Build image
docker build -t museum-guide:latest .

# Run container
docker run -d \
  --name museum-guide \
  -p 7860:7860 \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  museum-guide:latest

# View logs
docker logs -f museum-guide
```

### Option 2: Systemd Service (Linux)

Create `/etc/systemd/system/museum-guide.service`:

```ini
[Unit]
Description=Museum Guide App
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/museum-guide
Environment="PATH=/opt/museum-guide/.venv/bin"
EnvironmentFile=/opt/museum-guide/.env
ExecStart=/opt/museum-guide/.venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable museum-guide
sudo systemctl start museum-guide
sudo systemctl status museum-guide
```

### Option 3: Cloud Deployment

#### Heroku

```bash
# Install Heroku CLI
heroku login

# Create app
heroku create museum-guide-app

# Set environment variables
heroku config:set OPENAI_API_KEY=sk-...
heroku config:set PINECONE_API_KEY=pcsk-...
heroku config:set ENVIRONMENT=production

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

#### AWS EC2

```bash
# Connect to instance
ssh -i key.pem ubuntu@ec2-xxx.compute.amazonaws.com

# Setup
git clone https://github.com/your-repo/museum_guide_app.git
cd museum_guide_app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Edit with your keys

# Run with nohup
nohup python app.py > logs/app.log 2>&1 &
```

---

## Monitoring & Logging

### Log Files

```bash
logs/
├── app.log          # Main application log (JSON in production)
├── audit.log        # Audit trail (daily rotation)
└── metrics.jsonl    # Performance metrics (newline-delimited JSON)
```

### Log Format (Production)

JSON structured logs for easy parsing:

```json
{
  "timestamp": "2024-12-04T15:30:45.123Z",
  "level": "INFO",
  "logger": "src.core.analyze",
  "message": "Performance: artwork_analysis",
  "module": "analyze",
  "function": "analyze_artwork",
  "line": 145,
  "execution_time": 6.8
}
```

### Metrics Analysis

```bash
# Average execution time
cat logs/metrics.jsonl | jq -s 'map(.execution_time) | add/length'

# Success rate
cat logs/metrics.jsonl | jq -s 'map(.success) | map(if . then 1 else 0 end) | add/length'

# Errors in last hour
cat logs/app.log | jq 'select(.level=="ERROR" and .timestamp > (now - 3600))'
```

### Integration with Monitoring Tools

#### Prometheus

```python
# Add to app.py for metrics endpoint
from prometheus_client import start_http_server, Counter, Histogram

request_counter = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

# Start metrics server on port 9090
start_http_server(9090)
```

#### CloudWatch (AWS)

```python
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_data(
    Namespace='MuseumGuide',
    MetricData=[{
        'MetricName': 'AnalysisTime',
        'Value': execution_time,
        'Unit': 'Seconds'
    }]
)
```

---

## Health Checks

### Endpoints

```python
from src.core.health_check import health_status, liveness_probe, readiness_probe

# Full health status
status = health_status()
# Returns: {'status': 'healthy', 'checks': {...}, 'statistics': {...}}

# Liveness probe (Kubernetes)
liveness = liveness_probe()
# Returns: {'status': 'alive', 'timestamp': '...'}

# Readiness probe (Kubernetes)
readiness = readiness_probe()
# Returns: {'status': 'ready', 'details': {...}}
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: museum-guide
spec:
  containers:
  - name: app
    image: museum-guide:latest
    ports:
    - containerPort: 7860
    livenessProbe:
      httpGet:
        path: /health/liveness
        port: 7860
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health/readiness
        port: 7860
      initialDelaySeconds: 5
      periodSeconds: 5
```

---

## Scaling

### Horizontal Scaling

#### Load Balancer Configuration (Nginx)

```nginx
upstream museum_guide {
    least_conn;
    server 10.0.1.10:7860;
    server 10.0.1.11:7860;
    server 10.0.1.12:7860;
}

server {
    listen 80;
    server_name museum-guide.example.com;

    location / {
        proxy_pass http://museum_guide;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://museum_guide/health;
        access_log off;
    }
}
```

### Vertical Scaling

Adjust in `.env`:

```bash
# Increase workers for CPU-bound operations
MAX_WORKERS=8

# Increase timeout for slow API calls
REQUEST_TIMEOUT=60
```

### Auto-Scaling (AWS)

```bash
# Create Auto Scaling Group
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name museum-guide-asg \
    --launch-configuration-name museum-guide-lc \
    --min-size 2 \
    --max-size 10 \
    --desired-capacity 2 \
    --health-check-type ELB \
    --health-check-grace-period 300 \
    --target-group-arns arn:aws:elasticloadbalancing:...

# Scale based on CPU
aws autoscaling put-scaling-policy \
    --auto-scaling-group-name museum-guide-asg \
    --policy-name cpu-scale-up \
    --scaling-adjustment 1 \
    --adjustment-type ChangeInCapacity \
    --cooldown 300
```

---

## Troubleshooting

### Common Issues

**Issue: High Memory Usage**
```bash
# Check memory
cat logs/app.log | jq 'select(.memory_percent > 80)'

# Solution: Increase RAM or reduce MAX_WORKERS
```

**Issue: Slow Response Times**
```bash
# Check performance metrics
cat logs/metrics.jsonl | jq 'select(.execution_time > 10)'

# Solution: Enable caching, increase timeout
ENABLE_CACHE=true
REQUEST_TIMEOUT=60
```

**Issue: API Rate Limits**
```bash
# Check error logs
cat logs/app.log | jq 'select(.level=="ERROR" and .message | contains("rate limit"))'

# Solution: Implement request queuing or reduce rate limits
MAX_REQUESTS_PER_MINUTE=30
```

---

## Security Best Practices

1. **Never commit `.env` files**
   ```bash
   # Verify .env is in .gitignore
   git check-ignore .env
   ```

2. **Rotate API keys regularly**
   ```bash
   # Update .env with new keys
   # Restart application
   sudo systemctl restart museum-guide
   ```

3. **Use HTTPS in production**
   ```bash
   # With Caddy (automatic HTTPS)
   caddy reverse-proxy --from museum-guide.example.com --to localhost:7860
   ```

4. **Implement rate limiting**
   ```bash
   # Already configured in .env
   MAX_REQUESTS_PER_MINUTE=60
   MAX_REQUESTS_PER_HOUR=1000
   ```

---

## Performance Benchmarks

| Metric | Development | Production |
|--------|-------------|------------|
| Artwork Analysis | 6-8s | 6-8s |
| Chat Response | 1-2s | 1-2s |
| Memory Usage | ~500MB | ~800MB |
| CPU Usage | 20-40% | 30-50% |

---

## Support

For deployment issues:
1. Check logs: `tail -f logs/app.log`
2. Verify health: `python -c "from src.core.health_check import health_status; print(health_status())"`
3. Review configuration: `python -c "import config; print(vars(config))"`
