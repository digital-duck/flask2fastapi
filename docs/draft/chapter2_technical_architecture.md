# Flask to FastAPI Migration Guide
## Chapter 2: Technical Architecture & Design

### Overview

This chapter covers the technical foundation for migrating from Flask to FastAPI, focusing on AWS services integration, infrastructure considerations, and architectural decisions specific to our chatbot implementation.

---

## AWS Services Strategy

### aioboto3 vs boto3: The Critical Difference

**Current Flask Implementation (Blocking)**
```python
import boto3

# Synchronous client - blocks entire FastAPI process
bedrock_client = boto3.client('bedrock-agent', region_name='us-east-1')
response = bedrock_client.invoke_agent(...)  # Blocks thread
```

**FastAPI Implementation (Non-blocking)**
```python
import aioboto3

# Asynchronous client - non-blocking
session = aioboto3.Session()
async with session.client('bedrock-agent') as client:
    response = await client.invoke_agent(...)  # Non-blocking
```

### AWS Services Migration Matrix

| Service | Current (boto3) | Migration Required | FastAPI (aioboto3) |
|---------|----------------|-------------------|-------------------|
| **AWS Bedrock** | ✅ In use | 🔄 **Critical** | async client required |
| **DynamoDB** | ✅ Session storage | 🔄 **High priority** | async operations |
| **CloudWatch** | ✅ Logging | 🔄 **Medium** | async metrics |
| **S3** | ❓ File storage | 🔄 **If used** | async file ops |
| **API Gateway** | ✅ Entry point | ✅ **No change** | Works identically |
| **ECS** | ✅ Container hosting | ✅ **No change** | Same infrastructure |
| **Load Balancer** | ✅ Traffic distribution | ✅ **No change** | Same configuration |

### Core Service Implementations

#### 1. AWS Bedrock Integration
```python
# services/bedrock_service.py
import asyncio
import aioboto3
from typing import Dict, Optional
import structlog

logger = structlog.get_logger()

class BedrockService:
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.session = aioboto3.Session()
        
    async def __aenter__(self):
        self.client = await self.session.client(
            'bedrock-agent', 
            region_name=self.region_name
        ).__aenter__()
        return self
    
    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)
    
    async def invoke_agent(
        self, 
        message: str, 
        session_id: str,
        agent_id: str
    ) -> Dict:
        """Invoke Bedrock agent with async support"""
        try:
            response = await self.client.invoke_agent(
                agentId=agent_id,
                sessionId=session_id,
                inputText=message
            )
            
            return {
                'response': response['output']['text'],
                'session_id': session_id,
                'tokens_used': response.get('usage', {}).get('totalTokens', 0)
            }
            
        except Exception as e:
            logger.error("Bedrock invocation failed", 
                        error=str(e), 
                        session_id=session_id)
            raise
    
    async def invoke_with_timeout(
        self, 
        message: str, 
        session_id: str,
        agent_id: str,
        timeout: float = 30.0
    ) -> Dict:
        """Invoke with timeout handling"""
        try:
            return await asyncio.wait_for(
                self.invoke_agent(message, session_id, agent_id),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Bedrock timeout", session_id=session_id)
            return {"error": "Request timeout", "session_id": session_id}
```

#### 2. DynamoDB Session Management
```python
# services/session_service.py
import aioboto3
from datetime import datetime, timedelta
from typing import Dict, Optional

class SessionService:
    def __init__(self, table_name: str = "chat_sessions"):
        self.table_name = table_name
        self.session = aioboto3.Session()
    
    async def save_session(
        self, 
        session_id: str, 
        user_data: Dict,
        ttl_hours: int = 24
    ) -> bool:
        """Save session data with TTL"""
        try:
            async with self.session.resource('dynamodb') as dynamodb:
                table = await dynamodb.Table(self.table_name)
                
                # Calculate TTL
                ttl = datetime.utcnow() + timedelta(hours=ttl_hours)
                
                await table.put_item(
                    Item={
                        'session_id': session_id,
                        'user_data': user_data,
                        'created_at': datetime.utcnow().isoformat(),
                        'ttl': int(ttl.timestamp())
                    }
                )
                return True
        except Exception as e:
            logger.error("Session save failed", error=str(e))
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve session data"""
        try:
            async with self.session.resource('dynamodb') as dynamodb:
                table = await dynamodb.Table(self.table_name)
                
                response = await table.get_item(
                    Key={'session_id': session_id}
                )
                
                return response.get('Item')
        except Exception as e:
            logger.error("Session retrieval failed", error=str(e))
            return None
```

#### 3. CloudWatch Metrics (Async)
```python
# services/metrics_service.py
import aioboto3
import asyncio
from datetime import datetime
from typing import Dict, List

class MetricsService:
    def __init__(self, namespace: str = "ChatBot/FastAPI"):
        self.namespace = namespace
        self.session = aioboto3.Session()
        self._metrics_buffer = []
        self._buffer_lock = asyncio.Lock()
    
    async def put_metric(
        self, 
        metric_name: str, 
        value: float,
        unit: str = "Count",
        dimensions: Dict[str, str] = None
    ):
        """Buffer metrics for batch sending"""
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.utcnow()
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
        
        async with self._buffer_lock:
            self._metrics_buffer.append(metric_data)
            
        # Auto-flush when buffer is full
        if len(self._metrics_buffer) >= 20:
            await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Send buffered metrics to CloudWatch"""
        if not self._metrics_buffer:
            return
            
        async with self._buffer_lock:
            metrics_to_send = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
        
        try:
            async with self.session.client('cloudwatch') as cw:
                await cw.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=metrics_to_send
                )
        except Exception as e:
            logger.error("Metrics flush failed", error=str(e))
```

---

## Infrastructure Architecture

### Current Flask Architecture
```
API Gateway → Lambda → ECS (4 instances) → Flask App
                                ↓
                           AWS Bedrock
                                ↓
                           DynamoDB (sessions)
```

### Target FastAPI Architecture
```
API Gateway → Lambda → ECS (1-2 instances) → FastAPI App
                                ↓
                           AWS Bedrock (async)
                                ↓
                           DynamoDB (async)
```

### Key Infrastructure Changes

#### 1. ECS Task Definition Updates
```json
{
  "family": "chatbot-fastapi",
  "taskRoleArn": "arn:aws:iam::account:role/ECSTaskRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "chatbot-api",
      "image": "your-repo/chatbot-fastapi:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "command": [
        "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "1",
        "--loop", "uvloop"
      ],
      "environment": [
        {
          "name": "BEDROCK_AGENT_ID",
          "value": "your-agent-id"
        },
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-east-1"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/aws/ecs/chatbot-fastapi",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 2. Dockerfile Optimization
```dockerfile
# Dockerfile.fastapi
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 3. Requirements.txt Updates
```txt
# Core FastAPI stack
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# AWS async support
aioboto3==12.0.0
aiobotocore==2.7.0

# Database and caching
asyncpg==0.29.0  # if using PostgreSQL
aioredis==2.0.1  # if using Redis

# Monitoring and logging
structlog==23.2.0
prometheus-client==0.19.0

# Development and testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2  # for async HTTP testing

# Production
gunicorn==21.2.0  # if needed for fallback
```

---

## API Design Patterns

### 1. Request/Response Models with Pydantic
```python
# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can you help me?",
                "session_id": "sess_123",
                "user_id": "user_456"
            }
        }

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    processing_time_ms: float

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    timestamp: datetime
    request_id: Optional[str] = None
```

### 2. Dependency Injection Pattern
```python
# dependencies.py
import asyncio
from fastapi import Depends, HTTPException
from services.bedrock_service import BedrockService
from services.session_service import SessionService
from services.metrics_service import MetricsService

# Global service instances
_bedrock_service = None
_session_service = None
_metrics_service = None

async def get_bedrock_service() -> BedrockService:
    global _bedrock_service
    if _bedrock_service is None:
        _bedrock_service = BedrockService()
    return _bedrock_service

async def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service

async def get_metrics_service() -> MetricsService:
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service

# Rate limiting dependency
from collections import defaultdict
from time import time

request_counts = defaultdict(list)

async def rate_limit_check(client_ip: str = None):
    """Simple rate limiting - 100 requests per hour"""
    if not client_ip:
        return
        
    now = time()
    hour_ago = now - 3600
    
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if req_time > hour_ago
    ]
    
    # Check limit
    if len(request_counts[client_ip]) >= 100:
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded"
        )
    
    # Record this request
    request_counts[client_ip].append(now)
```

### 3. Main FastAPI Application
```python
# main.py
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import uuid

from models.schemas import ChatRequest, ChatResponse, ErrorResponse
from dependencies import (
    get_bedrock_service, 
    get_session_service, 
    get_metrics_service,
    rate_limit_check
)
from services.bedrock_service import BedrockService
from services.session_service import SessionService
from services.metrics_service import MetricsService

# Initialize FastAPI app
app = FastAPI(
    title="AI Chatbot API",
    description="High-performance chatbot API built with FastAPI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client_request: Request,
    bedrock_service: BedrockService = Depends(get_bedrock_service),
    session_service: SessionService = Depends(get_session_service),
    metrics_service: MetricsService = Depends(get_metrics_service),
    _rate_limit: None = Depends(rate_limit_check)
):
    """Process chat message and return AI response"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get session context (optional)
        session_data = await session_service.get_session(session_id)
        
        # Call Bedrock agent
        async with bedrock_service as bedrock:
            result = await bedrock.invoke_with_timeout(
                message=request.message,
                session_id=session_id,
                agent_id="your-agent-id",
                timeout=30.0
            )
        
        # Handle timeout or error
        if "error" in result:
            raise HTTPException(status_code=408, detail=result["error"])
        
        # Save session data
        if request.user_id:
            await session_service.save_session(
                session_id, 
                {"user_id": request.user_id, "last_message": request.message}
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        await metrics_service.put_metric(
            "ChatRequest", 1, "Count",
            {"Endpoint": "/chat", "Status": "Success"}
        )
        await metrics_service.put_metric(
            "ProcessingTime", processing_time, "Milliseconds"
        )
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            timestamp=time.time(),
            tokens_used=result.get("tokens_used"),
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Record error metrics
        await metrics_service.put_metric(
            "ChatRequest", 1, "Count",
            {"Endpoint": "/chat", "Status": "Error"}
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Pre-warm service connections
    await get_bedrock_service()
    await get_session_service()
    await get_metrics_service()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Flush any remaining metrics
    metrics_service = await get_metrics_service()
    await metrics_service._flush_metrics()
```

---

## Connection Management

### 1. Connection Pooling Strategy
```python
# config/connections.py
import aioboto3
from typing import Optional

class AWSConnectionManager:
    def __init__(self):
        self.session: Optional[aioboto3.Session] = None
        self._clients = {}
    
    async def initialize(self):
        """Initialize AWS session"""
        self.session = aioboto3.Session()
    
    async def get_client(self, service_name: str):
        """Get or create AWS client with connection reuse"""
        if service_name not in self._clients:
            self._clients[service_name] = await self.session.client(
                service_name
            ).__aenter__()
        return self._clients[service_name]
    
    async def cleanup(self):
        """Close all connections"""
        for client in self._clients.values():
            await client.__aexit__(None, None, None)
        self._clients.clear()

# Global connection manager
connection_manager = AWSConnectionManager()
```

### 2. Resource Management Best Practices
```python
# Context managers for proper resource cleanup
async def safe_bedrock_call(message: str, session_id: str):
    """Example of proper resource management"""
    session = aioboto3.Session()
    
    try:
        async with session.client('bedrock-agent') as client:
            response = await client.invoke_agent(
                agentId='your-agent-id',
                sessionId=session_id,
                inputText=message
            )
            return response
    except Exception as e:
        logger.error("Bedrock call failed", error=str(e))
        raise
    # Client automatically closed when exiting context
```

---

## Migration Checklist

### Pre-Migration Validation
- [ ] Verify aioboto3 compatibility with current AWS SDK version
- [ ] Test async AWS service calls in development environment
- [ ] Validate DynamoDB table permissions for async operations
- [ ] Confirm CloudWatch metrics namespace and permissions

### Infrastructure Preparation
- [ ] Update ECS task definition with FastAPI configuration
- [ ] Configure new CloudWatch log groups
- [ ] Test Docker container with FastAPI + uvicorn
- [ ] Validate health check endpoints

### Service Integration Testing
- [ ] Test Bedrock agent async invocation
- [ ] Verify DynamoDB session storage/retrieval
- [ ] Confirm CloudWatch metrics publishing
- [ ] Load test async endpoint performance

---

*Next: Chapter 3 - Team Upskilling & Training*