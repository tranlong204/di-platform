# Production Deployment Guide
# Document Intelligence Platform - Real-World Application Setup

## 🚀 Overview

This guide will help you transform your Document Intelligence Platform into a production-ready, real-world application that can handle enterprise workloads, scale to thousands of users, and maintain high availability.

## 📋 Prerequisites

### Infrastructure Requirements
- **Cloud Provider**: AWS, Google Cloud, or Azure
- **Compute**: Minimum 4 CPU cores, 8GB RAM (recommended: 8+ cores, 16GB+ RAM)
- **Storage**: SSD storage for database and vector indices
- **Network**: Load balancer, CDN for static assets
- **Domain**: Custom domain with SSL certificate

### Software Requirements
- Docker & Docker Compose
- Kubernetes (for advanced deployments)
- CI/CD pipeline (GitHub Actions, GitLab CI, etc.)
- Monitoring tools (Prometheus, Grafana, Sentry)

## 🔧 Phase 1: Security & Authentication

### 1.1 User Authentication System

```python
# Add to app/core/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os

SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
```

### 1.2 Rate Limiting

```python
# Add to app/core/rate_limiting.py
from fastapi import Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

redis_client = redis.Redis(host="redis", port=6379, db=0)
limiter = Limiter(key_func=get_remote_address, storage_uri="redis://redis:6379")

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

## 📊 Phase 2: Monitoring & Observability

### 2.1 Application Metrics

```python
# Add to app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
DOCUMENT_PROCESSING_TIME = Histogram('document_processing_seconds', 'Document processing time')
QUERY_RESPONSE_TIME = Histogram('query_response_seconds', 'Query response time')

def track_request(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__).inc()
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    return wrapper
```

### 2.2 Logging Configuration

```python
# Add to app/core/logging.py
import logging
import sys
from loguru import logger
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure loguru
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")
logger.add("/app/logs/app.log", rotation="1 day", retention="30 days", compression="zip")
```

## 🔄 Phase 3: CI/CD Pipeline

### 3.1 GitHub Actions Workflow

```yaml
# Create .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      - name: Run tests
        run: pytest tests/

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security scan
        uses: securecodewarrior/github-action-add-sarif@v1
        with:
          sarif-file: 'security-scan-results.sarif'

  deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          # Add your deployment commands here
          echo "Deploying to production..."
```

## 🏗️ Phase 4: Scalability & Performance

### 4.1 Database Optimization

```sql
-- Add to scripts/optimize_db.sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_documents_processed ON documents(processed);
CREATE INDEX CONCURRENTLY idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX CONCURRENTLY idx_queries_created_at ON queries(created_at);
CREATE INDEX CONCURRENTLY idx_queries_query_hash ON queries(query_hash);

-- Partition large tables
CREATE TABLE queries_2024 PARTITION OF queries
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Enable connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

### 4.2 Caching Strategy

```python
# Add to app/core/cache.py
import redis
from functools import wraps
import json
import hashlib

redis_client = redis.Redis(host="redis", port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

## 🔒 Phase 5: Security Hardening

### 5.1 Input Validation & Sanitization

```python
# Add to app/core/validation.py
from pydantic import BaseModel, validator
import re
from typing import Optional

class DocumentUploadRequest(BaseModel):
    filename: str
    content_type: str
    file_size: int
    
    @validator('filename')
    def validate_filename(cls, v):
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Invalid filename')
        if len(v) > 255:
            raise ValueError('Filename too long')
        return v
    
    @validator('file_size')
    def validate_file_size(cls, v):
        if v > 50 * 1024 * 1024:  # 50MB limit
            raise ValueError('File too large')
        return v
```

### 5.2 API Security Headers

```python
# Add to app/core/security.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

def add_security_headers(app: FastAPI):
    @app.middleware("http")
    async def add_security_headers_middleware(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
```

## 📈 Phase 6: Business Features

### 6.1 Multi-Tenancy

```python
# Add to app/models/tenant.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class Tenant(Base):
    __tablename__ = "tenants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    domain = Column(String(255), unique=True, nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

### 6.2 Usage Analytics

```python
# Add to app/services/analytics.py
from sqlalchemy.orm import Session
from app.models import Query, Document
from datetime import datetime, timedelta
import pandas as pd

class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_usage_stats(self, tenant_id: int, days: int = 30):
        start_date = datetime.utcnow() - timedelta(days=days)
        
        queries = self.db.query(Query).filter(
            Query.created_at >= start_date,
            Query.tenant_id == tenant_id
        ).all()
        
        documents = self.db.query(Document).filter(
            Document.created_at >= start_date,
            Document.tenant_id == tenant_id
        ).all()
        
        return {
            "total_queries": len(queries),
            "total_documents": len(documents),
            "avg_response_time": sum(q.processing_time for q in queries) / len(queries) if queries else 0,
            "popular_topics": self._extract_topics(queries)
        }
```

## 🚀 Deployment Options

### Option 1: Docker Compose (Simple)
```bash
# Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### Option 2: Kubernetes (Scalable)
```yaml
# Create k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: di-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: di-platform
  template:
    metadata:
      labels:
        app: di-platform
    spec:
      containers:
      - name: di-platform
        image: your-registry/di-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: di-platform-secrets
              key: database-url
```

### Option 3: AWS ECS/Fargate (Managed)
```bash
# Deploy to AWS ECS
aws ecs create-service \
  --cluster di-platform-cluster \
  --service-name di-platform-service \
  --task-definition di-platform-task \
  --desired-count 3
```

## 📊 Monitoring & Alerting

### Key Metrics to Monitor
- **Application**: Response time, error rate, throughput
- **Database**: Connection pool, query performance, storage usage
- **Infrastructure**: CPU, memory, disk, network
- **Business**: User activity, document processing, query patterns

### Alerting Rules
```yaml
# Add to monitoring/alerts.yml
groups:
- name: di-platform
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
  
  - alert: DatabaseConnectionsHigh
    expr: postgresql_connections_active > 80
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Database connection pool nearly full"
```

## 🔧 Maintenance & Operations

### Backup Strategy
```bash
#!/bin/bash
# scripts/backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$DATE"

# Database backup
pg_dump $DATABASE_URL > "$BACKUP_DIR/database.sql"

# Vector index backup
cp -r /app/data/faiss_index "$BACKUP_DIR/"

# Upload to S3
aws s3 sync "$BACKUP_DIR" "s3://your-backup-bucket/$DATE/"
```

### Health Checks
```python
# Add to app/api/routes/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
import redis

router = APIRouter()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    checks = {
        "database": await check_database(db),
        "redis": await check_redis(),
        "vector_store": await check_vector_store(),
        "openai": await check_openai()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return {"status": "healthy" if all_healthy else "unhealthy", "checks": checks}
```

## 💰 Cost Optimization

### Resource Optimization
- **Auto-scaling**: Scale based on CPU/memory usage
- **Spot instances**: Use for non-critical workloads
- **Reserved instances**: For predictable workloads
- **Storage optimization**: Use appropriate storage classes

### Monitoring Costs
```python
# Add cost tracking
import boto3

def track_costs():
    client = boto3.client('ce')
    response = client.get_cost_and_usage(
        TimePeriod={
            'Start': '2024-01-01',
            'End': '2024-01-31'
        },
        Granularity='MONTHLY',
        Metrics=['BlendedCost']
    )
    return response
```

## 🎯 Next Steps

1. **Start with Phase 1**: Implement security and authentication
2. **Set up monitoring**: Deploy Prometheus and Grafana
3. **Create CI/CD pipeline**: Automate testing and deployment
4. **Implement caching**: Add Redis caching for better performance
5. **Add user management**: Implement multi-tenancy
6. **Scale gradually**: Start with single instance, scale as needed
7. **Monitor and optimize**: Continuously monitor and improve

## 📞 Support & Maintenance

- **Documentation**: Keep README and API docs updated
- **Monitoring**: Set up alerts for critical issues
- **Backups**: Regular automated backups
- **Updates**: Keep dependencies updated
- **Security**: Regular security audits

This guide provides a comprehensive roadmap to transform your Document Intelligence Platform into a production-ready, enterprise-grade application. Start with the phases that are most critical for your use case and gradually implement the remaining features.
