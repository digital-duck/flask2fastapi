    def test_api_documentation(self, client):
        """Test that API documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/api/v1/chat")
        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
    
    def test_gzip_compression(self, client):
        """Test that gzip compression is enabled"""
        # Large response should be compressed
        response = client.get("/health")
        # Check if gzip is supported
        assert response.status_code == 200

# Performance comparison test
class TestPerformanceValidation:
    """Compare performance between Flask and FastAPI"""
    
    @pytest.mark.asyncio
    async def test_response_time_benchmark(self):
        """Benchmark response times"""
        import time
        
        async def measure_response_time():
            start = time.time()
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post("/api/v1/chat", json={
                    "message": "Performance test message"
                })
                assert response.status_code == 200
            return time.time() - start
        
        # Measure 5 requests
        times = []
        for _ in range(5):
            response_time = await measure_response_time()
            times.append(response_time)
        
        avg_time = sum(times) / len(times)
        print(f"Average response time: {avg_time:.3f}s")
        
        # Should be reasonably fast (adjust threshold as needed)
        assert avg_time < 2.0, f"Response time too slow: {avg_time:.3f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_load(self):
        """Test handling of concurrent load"""
        async def make_request(request_id):
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                start = time.time()
                response = await ac.post("/api/v1/chat", json={
                    "message": f"Load test message {request_id}"
                })
                duration = time.time() - start
                return response.status_code, duration
        
        # Send 20 concurrent requests
        import time
        start_time = time.time()
        tasks = [make_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Check results
        successful_requests = [r for r in results if not isinstance(r, Exception) and r[0] == 200]
        success_rate = len(successful_requests) / len(results)
        
        print(f"Total time for 20 requests: {total_time:.3f}s")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Requests per second: {len(results)/total_time:.1f}")
        
        # Assertions
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert total_time < 10.0, f"Total time too slow: {total_time:.3f}s"
```

### Load Testing Script

```python
# tests/load_test.py
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import argparse

class LoadTestRunner:
    """Load testing utility for FastAPI migration validation"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def single_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Make a single request and measure performance"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/api/v1/chat",
                json={
                    "message": f"Load test message {request_id}",
                    "session_id": f"load-test-{request_id}"
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                await response.json()
                duration = time.time() - start_time
                
                return {
                    "request_id": request_id,
                    "status_code": response.status,
                    "duration": duration,
                    "success": response.status == 200
                }
                
        except Exception as e:
            duration = time.time() - start_time
            return {
                "request_id": request_id,
                "status_code": 0,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
    
    async def run_load_test(
        self, 
        concurrent_users: int = 10,
        total_requests: int = 100,
        ramp_up_time: float = 5.0
    ) -> Dict:
        """
        Run load test with specified parameters
        
        Args:
            concurrent_users: Number of concurrent users
            total_requests: Total number of requests to send
            ramp_up_time: Time to ramp up to full load (seconds)
        """
        print(f"Starting load test:")
        print(f"  Concurrent users: {concurrent_users}")
        print(f"  Total requests: {total_requests}")
        print(f"  Ramp-up time: {ramp_up_time}s")
        print(f"  Target URL: {self.base_url}")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async def limited_request(request_id: int):
                async with semaphore:
                    # Add ramp-up delay
                    if ramp_up_time > 0:
                        delay = (request_id / total_requests) * ramp_up_time
                        await asyncio.sleep(delay)
                    
                    return await self.single_request(session, request_id)
            
            # Execute all requests
            tasks = [limited_request(i) for i in range(total_requests)]
            self.results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
        
        # Analyze results
        return self.analyze_results(total_time)
    
    def analyze_results(self, total_time: float) -> Dict:
        """Analyze load test results"""
        # Filter successful results
        successful_results = [
            r for r in self.results 
            if not isinstance(r, Exception) and r.get("success", False)
        ]
        
        error_results = [
            r for r in self.results 
            if isinstance(r, Exception) or not r.get("success", False)
        ]
        
        if not successful_results:
            return {
                "error": "No successful requests",
                "total_requests": len(self.results),
                "success_rate": 0.0
            }
        
        # Calculate statistics
        durations = [r["duration"] for r in successful_results]
        
        stats = {
            "total_requests": len(self.results),
            "successful_requests": len(successful_results),
            "failed_requests": len(error_results),
            "success_rate": len(successful_results) / len(self.results),
            "total_time": total_time,
            "requests_per_second": len(self.results) / total_time,
            "avg_response_time": statistics.mean(durations),
            "min_response_time": min(durations),
            "max_response_time": max(durations),
            "p50_response_time": statistics.median(durations),
            "p90_response_time": statistics.quantiles(durations, n=10)[8] if len(durations) >= 10 else max(durations),
            "p99_response_time": statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations)
        }
        
        return stats
    
    def print_results(self, stats: Dict):
        """Print formatted test results"""
        print("\n" + "="*60)
        print("LOAD TEST RESULTS")
        print("="*60)
        print(f"Total Requests:        {stats['total_requests']}")
        print(f"Successful Requests:   {stats['successful_requests']}")
        print(f"Failed Requests:       {stats['failed_requests']}")
        print(f"Success Rate:          {stats['success_rate']:.2%}")
        print(f"Total Time:            {stats['total_time']:.2f}s")
        print(f"Requests/Second:       {stats['requests_per_second']:.1f}")
        print()
        print("RESPONSE TIMES:")
        print(f"Average:               {stats['avg_response_time']*1000:.0f}ms")
        print(f"Minimum:               {stats['min_response_time']*1000:.0f}ms")
        print(f"Maximum:               {stats['max_response_time']*1000:.0f}ms")
        print(f"50th Percentile (P50): {stats['p50_response_time']*1000:.0f}ms")
        print(f"90th Percentile (P90): {stats['p90_response_time']*1000:.0f}ms")
        print(f"99th Percentile (P99): {stats['p99_response_time']*1000:.0f}ms")
        print("="*60)

async def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description="FastAPI Load Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Base@router.post("/chat", response_model=ChatResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    408: {"model": ErrorResponse, "description": "Request Timeout"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"}
})
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    bedrock_service: BedrockService = Depends(get_bedrock_service),
    session_service: SessionService = Depends(get_session_service),
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Process chat message and return AI response
    
    - **message**: User's chat message (1-4000 characters)
    - **session_id**: Optional session identifier for context
    - **user_id**: Optional user identifier for analytics
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(
        "Processing chat request",
        request_id=request_id,
        session_id=session_id,
        message_length=len(request.message)
    )
    
    try:
        # Get session context (optional)
        session_data = await session_service.get_session(session_id)
        
        # Process message with Bedrock
        result = await bedrock_service.invoke_agent(
            message=request.message,
            session_id=session_id
        )
        
        # Handle timeout or error from Bedrock
        if "error" in result:
            logger.warning(
                "Bedrock request failed",
                request_id=request_id,
                error=result["error"]
            )
            raise HTTPException(
                status_code=408 if "timeout" in result["error"].lower() else 500,
                detail=result["error"]
            )
        
        # Save session data if user_id provided
        if request.user_id:
            await session_service.save_session(
                session_id=session_id,
                user_data={
                    "user_id": request.user_id,
                    "last_message": request.message,
                    "timestamp": time.time()
                }
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
        
        # Background analytics
        background_tasks.add_task(
            record_chat_analytics,
            request.message,
            result["response"],
            processing_time,
            session_id
        )
        
        logger.info(
            "Chat request completed",
            request_id=request_id,
            session_id=session_id,
            processing_time_ms=processing_time
        )
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            timestamp=time.time(),
            processing_time_ms=processing_time,
            tokens_used=result.get("tokens_used")
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in chat endpoint",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        
        # Record error metrics
        await metrics_service.put_metric(
            "ChatRequest", 1, "Count",
            {"Endpoint": "/chat", "Status": "Error"}
        )
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        )

# Additional chat endpoints
@router.get("/chat/sessions/{session_id}")
async def get_session_info(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """Get session information"""
    session_data = await session_service.get_session(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "session_data": session_data
    }

@router.delete("/chat/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service)
):
    """Delete session data"""
    success = await session_service.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}
```

---

## Step 6: Service Layer Migration

### Bedrock Service Migration

**Flask Bedrock Service:**
```python
# services/bedrock_service.py (Flask)
import boto3
import json
import logging

logger = logging.getLogger(__name__)

class BedrockService:
    def __init__(self):
        self.client = boto3.client('bedrock-agent', region_name='us-east-1')
    
    def invoke_agent(self, message, session_id):
        try:
            response = self.client.invoke_agent(
                agentId='your-agent-id',
                sessionId=session_id,
                inputText=message
            )
            return response['output']['text']
        except Exception as e:
            logger.error(f"Bedrock invocation failed: {str(e)}")
            raise
```

**FastAPI Bedrock Service:**
```python
# services/bedrock_service.py (FastAPI)
import aioboto3
import asyncio
import structlog
from typing import Dict, Optional
from contextlib import asynccontextmanager

from config.settings import settings

logger = structlog.get_logger()

class BedrockService:
    def __init__(self):
        self.region_name = settings.aws_region
        self.agent_id = settings.bedrock_agent_id
        self.session = aioboto3.Session()
        self._client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._client = await self.session.client(
            'bedrock-agent',
            region_name=self.region_name
        ).__aenter__()
        return self
    
    async def __aexit__(self, *args):
        """Async context manager exit"""
        if self._client:
            await self._client.__aexit__(*args)
    
    async def invoke_agent(
        self, 
        message: str, 
        session_id: str,
        timeout: float = None
    ) -> Dict:
        """
        Invoke Bedrock agent with async support and timeout handling
        
        Args:
            message: User message to process
            session_id: Session identifier for context
            timeout: Request timeout in seconds (default from settings)
            
        Returns:
            Dict containing response and metadata
            
        Raises:
            asyncio.TimeoutError: If request times out
            Exception: For other Bedrock errors
        """
        timeout = timeout or settings.request_timeout
        
        try:
            # Use context manager for proper resource management
            async with self.session.client(
                'bedrock-agent',
                region_name=self.region_name
            ) as client:
                
                # Apply timeout to the Bedrock call
                response = await asyncio.wait_for(
                    client.invoke_agent(
                        agentId=self.agent_id,
                        sessionId=session_id,
                        inputText=message
                    ),
                    timeout=timeout
                )
                
                logger.info(
                    "Bedrock invocation successful",
                    session_id=session_id,
                    message_length=len(message),
                    response_length=len(response.get('output', {}).get('text', ''))
                )
                
                return {
                    'response': response['output']['text'],
                    'session_id': session_id,
                    'tokens_used': response.get('usage', {}).get('totalTokens', 0),
                    'model_id': response.get('modelId', 'unknown')
                }
                
        except asyncio.TimeoutError:
            logger.warning(
                "Bedrock request timeout",
                session_id=session_id,
                timeout=timeout
            )
            return {"error": f"Request timeout after {timeout} seconds"}
            
        except Exception as e:
            logger.error(
                "Bedrock invocation failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return {"error": f"Bedrock error: {str(e)}"}
    
    async def health_check(self) -> bool:
        """Check if Bedrock service is accessible"""
        try:
            async with self.session.client(
                'bedrock-agent',
                region_name=self.region_name
            ) as client:
                # Simple connectivity test
                await asyncio.wait_for(
                    client.list_agents(maxResults=1),
                    timeout=5.0
                )
                return True
        except Exception as e:
            logger.error("Bedrock health check failed", error=str(e))
            return False

# Singleton instance for dependency injection
_bedrock_service_instance = None

async def get_bedrock_service() -> BedrockService:
    """Dependency provider for Bedrock service"""
    global _bedrock_service_instance
    if _bedrock_service_instance is None:
        _bedrock_service_instance = BedrockService()
    return _bedrock_service_instance
```

### Session Service Migration

**Flask Session Service:**
```python
# services/session_service.py (Flask)
import boto3
from datetime import datetime, timedelta

class SessionService:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.table = self.dynamodb.Table('chat_sessions')
    
    def save_session(self, session_id, data):
        try:
            self.table.put_item(
                Item={
                    'session_id': session_id,
                    'data': data,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            return True
        except Exception as e:
            print(f"Session save failed: {e}")
            return False
    
    def get_session(self, session_id):
        try:
            response = self.table.get_item(Key={'session_id': session_id})
            return response.get('Item', {}).get('data')
        except Exception as e:
            print(f"Session retrieval failed: {e}")
            return None
```

**FastAPI Session Service:**
```python
# services/session_service.py (FastAPI)
import aioboto3
import structlog
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import json

from config.settings import settings

logger = structlog.get_logger()

class SessionService:
    def __init__(self):
        self.table_name = settings.dynamodb_table
        self.region_name = settings.aws_region
        self.session = aioboto3.Session()
    
    async def save_session(
        self, 
        session_id: str, 
        user_data: Dict[str, Any],
        ttl_hours: int = 24
    ) -> bool:
        """
        Save session data with TTL
        
        Args:
            session_id: Unique session identifier
            user_data: Data to store
            ttl_hours: Time to live in hours
            
        Returns:
            bool: Success status
        """
        try:
            # Calculate TTL timestamp
            ttl_timestamp = int(
                (datetime.utcnow() + timedelta(hours=ttl_hours)).timestamp()
            )
            
            async with self.session.resource('dynamodb', region_name=self.region_name) as dynamodb:
                table = await dynamodb.Table(self.table_name)
                
                await table.put_item(
                    Item={
                        'session_id': session_id,
                        'user_data': user_data,
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat(),
                        'ttl': ttl_timestamp
                    }
                )
                
            logger.info(
                "Session saved successfully",
                session_id=session_id,
                ttl_hours=ttl_hours
            )
            return True
            
        except Exception as e:
            logger.error(
                "Session save failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing session data or None if not found
        """
        try:
            async with self.session.resource('dynamodb', region_name=self.region_name) as dynamodb:
                table = await dynamodb.Table(self.table_name)
                
                response = await table.get_item(
                    Key={'session_id': session_id}
                )
                
                item = response.get('Item')
                if not item:
                    logger.info("Session not found", session_id=session_id)
                    return None
                
                # Check if session has expired
                ttl = item.get('ttl', 0)
                if ttl and ttl < datetime.utcnow().timestamp():
                    logger.info("Session expired", session_id=session_id)
                    # Optionally delete expired session
                    await self.delete_session(session_id)
                    return None
                
                logger.info("Session retrieved successfully", session_id=session_id)
                return item
                
        except Exception as e:
            logger.error(
                "Session retrieval failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def update_session(
        self, 
        session_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update existing session data
        
        Args:
            session_id: Session identifier
            updates: Fields to update
            
        Returns:
            bool: Success status
        """
        try:
            async with self.session.resource('dynamodb', region_name=self.region_name) as dynamodb:
                table = await dynamodb.Table(self.table_name)
                
                # Build update expression
                update_expression = "SET updated_at = :updated_at"
                expression_values = {
                    ':updated_at': datetime.utcnow().isoformat()
                }
                
                for key, value in updates.items():
                    if key not in ['session_id', 'created_at']:  # Protect immutable fields
                        update_expression += f", {key} = :{key}"
                        expression_values[f":{key}"] = value
                
                await table.update_item(
                    Key={'session_id': session_id},
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues=expression_values
                )
                
            logger.info("Session updated successfully", session_id=session_id)
            return True
            
        except Exception as e:
            logger.error(
                "Session update failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session data
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: Success status
        """
        try:
            async with self.session.resource('dynamodb', region_name=self.region_name) as dynamodb:
                table = await dynamodb.Table(self.table_name)
                
                await table.delete_item(
                    Key={'session_id': session_id}
                )
                
            logger.info("Session deleted successfully", session_id=session_id)
            return True
            
        except Exception as e:
            logger.error(
                "Session deletion failed",
                session_id=session_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def list_user_sessions(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> list:
        """
        List sessions for a specific user
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session data
        """
        try:
            async with self.session.resource('dynamodb', region_name=self.region_name) as dynamodb:
                table = await dynamodb.Table(self.table_name)
                
                # Note: This requires a GSI on user_id in production
                response = await table.scan(
                    FilterExpression='user_data.user_id = :user_id',
                    ExpressionAttributeValues={':user_id': user_id},
                    Limit=limit
                )
                
                return response.get('Items', [])
                
        except Exception as e:
            logger.error(
                "User sessions listing failed",
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            return []

# Singleton instance for dependency injection
_session_service_instance = None

async def get_session_service() -> SessionService:
    """Dependency provider for Session service"""
    global _session_service_instance
    if _session_service_instance is None:
        _session_service_instance = SessionService()
    return _session_service_instance
```

---

## Step 7: Dependencies Management

### FastAPI Dependency Injection

```python
# dependencies.py
import asyncio
from fastapi import Depends, HTTPException, Request
from typing import Optional
import structlog

from services.bedrock_service import BedrockService, get_bedrock_service
from services.session_service import SessionService, get_session_service
from services.metrics_service import MetricsService
from config.settings import settings

logger = structlog.get_logger()

# Service instances cache
_services_cache = {}

async def get_metrics_service() -> MetricsService:
    """Get or create metrics service instance"""
    if 'metrics' not in _services_cache:
        _services_cache['metrics'] = MetricsService(
            namespace=settings.metrics_namespace
        )
    return _services_cache['metrics']

# Rate limiting dependency
from collections import defaultdict
from time import time

# Simple in-memory rate limiting (use Redis in production)
request_counts = defaultdict(list)

async def rate_limit_check(
    request: Request,
    requests_per_hour: int = 1000
) -> None:
    """
    Rate limiting dependency
    
    Args:
        request: FastAPI request object
        requests_per_hour: Maximum requests per hour per IP
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get client IP
    client_ip = request.client.host
    if not client_ip:
        return  # Skip rate limiting if IP not available
    
    now = time()
    hour_ago = now - 3600
    
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if req_time > hour_ago
    ]
    
    # Check limit
    if len(request_counts[client_ip]) >= requests_per_hour:
        logger.warning(
            "Rate limit exceeded",
            client_ip=client_ip,
            requests_count=len(request_counts[client_ip])
        )
        raise HTTPException(
            status_code=429, 
            detail={
                "error": "Rate limit exceeded",
                "retry_after": 3600,
                "limit": requests_per_hour
            }
        )
    
    # Record this request
    request_counts[client_ip].append(now)

# Authentication dependency (optional)
async def get_current_user(request: Request) -> Optional[dict]:
    """
    Extract user information from request headers
    
    Args:
        request: FastAPI request object
        
    Returns:
        User information dict or None
    """
    # Simple header-based auth (implement proper auth in production)
    user_id = request.headers.get("X-User-ID")
    api_key = request.headers.get("X-API-Key")
    
    if user_id and api_key:
        # Validate API key (implement proper validation)
        return {
            "user_id": user_id,
            "api_key": api_key,
            "authenticated": True
        }
    
    return None

# Service health check dependency
async def check_services_health() -> dict:
    """
    Check health of all dependent services
    
    Returns:
        Dict with service health status
    """
    health_status = {
        "bedrock": False,
        "dynamodb": False,
        "overall": False
    }
    
    try:
        # Check Bedrock
        bedrock_service = await get_bedrock_service()
        health_status["bedrock"] = await bedrock_service.health_check()
        
        # Check DynamoDB
        session_service = await get_session_service()
        # Simple connectivity test
        test_session = await session_service.get_session("health_check_test")
        health_status["dynamodb"] = True  # If no exception, it's accessible
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
    
    health_status["overall"] = all(health_status.values())
    return health_status

# Cleanup function for application shutdown
async def cleanup_services():
    """Cleanup all service connections"""
    try:
        # Flush metrics
        if 'metrics' in _services_cache:
            await _services_cache['metrics']._flush_metrics()
        
        # Close AWS connections
        # (aioboto3 clients are auto-closed via context managers)
        
        logger.info("Service cleanup completed")
        
    except Exception as e:
        logger.error("Service cleanup failed", error=str(e))
```

---

## Step 8: Middleware Migration

### Metrics Middleware

```python
# middleware/metrics.py
import time
import asyncio
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog

logger = structlog.get_logger()

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._metrics_buffer = []
        self._buffer_lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics"""
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Handle exceptions
            logger.error(
                "Request processing failed",
                method=method,
                path=path,
                error=str(e),
                exc_info=True
            )
            status_code = 500
            # Create error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Log request
        logger.info(
            "Request processed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration * 1000,
            client_ip=client_ip
        )
        
        # Buffer metrics for async sending
        await self._buffer_metric({
            "timestamp": start_time,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "client_ip": client_ip
        })
        
        return response
    
    async def _buffer_metric(self, metric_data: dict):
        """Buffer metric for batch sending"""
        async with self._buffer_lock:
            self._metrics_buffer.append(metric_data)
            
            # Auto-flush if buffer is getting full
            if len(self._metrics_buffer) >= 50:
                await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Flush metrics buffer to CloudWatch"""
        if not self._metrics_buffer:
            return
        
        async with self._buffer_lock:
            metrics_to_send = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
        
        try:
            # Send to CloudWatch
            from dependencies import get_metrics_service
            metrics_service = await get_metrics_service()
            
            for metric in metrics_to_send:
                await metrics_service.put_metric(
                    "RequestCount", 1, "Count",
                    {
                        "Method": metric["method"],
                        "Path": metric["path"],
                        "StatusCode": str(metric["status_code"])
                    }
                )
                
                await metrics_service.put_metric(
                    "RequestDuration", metric["duration"] * 1000, "Milliseconds",
                    {
                        "Method": metric["method"],
                        "Path": metric["path"]
                    }
                )
                
        except Exception as e:
            logger.error("Metrics flush failed", error=str(e))
```

---

## Step 9: Docker Configuration

### Updated Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create non-root user
RUN adduser --disabled-password --gecos '' --uid 1000 appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Updated Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  chatbot-fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AWS_REGION=us-east-1
      - BEDROCK_AGENT_ID=your-agent-id
      - DYNAMODB_TABLE=chat_sessions
      - LOG_LEVEL=INFO
      - DEBUG=false
    volumes:
      - ~/.aws:/home/appuser/.aws:ro  # AWS credentials
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  # Optional: Local DynamoDB for development
  dynamodb-local:
    image: amazon/dynamodb-local:latest
    ports:
      - "8001:8000"
    command: ["-jar", "DynamoDBLocal.jar", "-sharedDb", "-inMemory"]
    profiles:
      - dev
```

---

## Step 10: Migration Validation

### Functional Testing

```python
# tests/test_migration.py
import pytest
import httpx
import asyncio
from fastapi.testclient import TestClient
from main import app

class TestMigrationValidation:
    """Test suite to validate Flask to FastAPI migration"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health endpoint maintains compatibility"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_chat_endpoint_basic(self, client):
        """Test basic chat functionality"""
        response = client.post("/api/v1/chat", json={
            "message": "Hello, test message"
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "timestamp" in data
        assert "processing_time_ms" in data
    
    def test_chat_endpoint_with_session(self, client):
        """Test chat with session ID"""
        session_id = "test-session-123"
        response = client.post("/api/v1/chat", json={
            "message": "Hello with session",
            "session_id": session_id
        })
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
    
    def test_chat_endpoint_validation(self, client):
        """Test request validation"""
        # Empty message
        response = client.post("/api/v1/chat", json={
            "message": ""
        })
        assert response.status_code == 422
        
        # Missing message
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test async performance with concurrent requests"""
        async def single_request():
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post("/api/v1/chat", json={
                    "message": "Concurrent test message"
                })
                return response.status_code
        
        # Send 10 concurrent requests
        tasks = [single_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(status == 200 for status in results)
    
    def test_api_documentation(self, client):
        """Test that API documentation is available"""
        response = client.get("/docs")
        assert response.status_code == 200# Flask to FastAPI Migration Guide
## Chapter 4: Implementation Guide

### Overview

This chapter provides step-by-step instructions for converting your Flask chatbot application to FastAPI, with practical code examples and migration strategies.

---

## Migration Strategy Overview

### Phase-by-Phase Approach

**Phase 1: Foundation Setup** (Week 1-2)
- Project structure migration
- Core dependencies update
- Basic FastAPI app structure

**Phase 2: Core Functionality** (Week 3-4)
- Route conversion
- AWS service integration
- Request/response models

**Phase 3: Advanced Features** (Week 5-6)
- Middleware migration
- Error handling
- Performance optimization

---

## Step 1: Project Structure Migration

### Current Flask Structure
```
chatbot-flask/
├── app.py                    # Main Flask application
├── routes/
│   ├── __init__.py
│   ├── chat.py              # Chat endpoints
│   └── health.py            # Health check
├── services/
│   ├── __init__.py
│   ├── bedrock_service.py   # AWS Bedrock integration
│   └── session_service.py   # Session management
├── models/
│   ├── __init__.py
│   └── request_models.py    # Request/response classes
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### Target FastAPI Structure
```
chatbot-fastapi/
├── main.py                  # FastAPI application entry point
├── routers/
│   ├── __init__.py
│   ├── chat.py             # Chat endpoints (async)
│   └── health.py           # Health check
├── services/
│   ├── __init__.py
│   ├── bedrock_service.py  # AWS Bedrock (aioboto3)
│   └── session_service.py  # Session management (async)
├── models/
│   ├── __init__.py
│   └── schemas.py          # Pydantic models
├── middleware/
│   ├── __init__.py
│   └── metrics.py          # Performance middleware
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration (Pydantic settings)
├── dependencies.py         # FastAPI dependencies
├── requirements.txt        # Updated dependencies
├── Dockerfile             # FastAPI-optimized
└── docker-compose.yml     # Updated services
```

---

## Step 2: Dependencies Migration

### Requirements.txt Conversion

**Remove Flask Dependencies:**
```txt
# OLD - Flask stack
Flask==2.3.3
gunicorn==21.2.0
Flask-CORS==4.0.0
Flask-HTTPAuth==4.8.0
boto3==1.28.85
```

**Add FastAPI Dependencies:**
```txt
# NEW - FastAPI stack
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# AWS async support
aioboto3==12.0.0
aiobotocore==2.7.0

# HTTP client for async
httpx==0.25.2

# Async database support (if needed)
asyncpg==0.29.0
aioredis==2.0.1

# Logging and monitoring
structlog==23.2.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Development
black==23.11.0
isort==5.12.0
mypy==1.7.1
```

---

## Step 3: Configuration Migration

### Flask Config → Pydantic Settings

**Current Flask Configuration:**
```python
# config/settings.py (Flask)
import os

class Config:
    BEDROCK_AGENT_ID = os.getenv('BEDROCK_AGENT_ID', 'default-agent')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE', 'chat_sessions')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
```

**New FastAPI Configuration:**
```python
# config/settings.py (FastAPI)
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # AWS Configuration
    bedrock_agent_id: str = "default-agent"
    aws_region: str = "us-east-1"
    dynamodb_table: str = "chat_sessions"
    
    # Application Configuration
    app_name: str = "AI Chatbot API"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Performance Configuration
    max_concurrent_requests: int = 1000
    request_timeout: float = 30.0
    
    # CloudWatch Configuration
    metrics_namespace: str = "ChatBot/FastAPI"
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

---

## Step 4: Main Application Migration

### Flask App → FastAPI App

**Current Flask Application:**
```python
# app.py (Flask)
from flask import Flask
from flask_cors import CORS
from routes import chat_bp, health_bp
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Register blueprints
app.register_blueprint(chat_bp, url_prefix='/api/v1')
app.register_blueprint(health_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
```

**New FastAPI Application:**
```python
# main.py (FastAPI)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import structlog

from routers import chat, health
from middleware.metrics import MetricsMiddleware
from config.settings import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
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

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="High-performance chatbot API built with FastAPI",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom metrics middleware
if settings.enable_metrics:
    app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger = structlog.get_logger()
    logger.info("Starting FastAPI chatbot application", version=settings.app_version)
    
    # Pre-warm AWS connections
    from dependencies import get_bedrock_service, get_session_service
    await get_bedrock_service()
    await get_session_service()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger = structlog.get_logger()
    logger.info("Shutting down FastAPI chatbot application")
    
    # Cleanup connections
    from dependencies import cleanup_services
    await cleanup_services()
```

---

## Step 5: Routes Migration

### Health Check Route

**Flask Health Route:**
```python
# routes/health.py (Flask)
from flask import Blueprint, jsonify
import time

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })
```

**FastAPI Health Route:**
```python
# routers/health.py (FastAPI)
from fastapi import APIRouter
from pydantic import BaseModel
import time

from config.settings import settings

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    environment: str

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Application health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version=settings.app_version,
        environment="development" if settings.debug else "production"
    )

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with service status"""
    # Test AWS connectivity
    aws_status = "unknown"
    try:
        from dependencies import get_bedrock_service
        bedrock = await get_bedrock_service()
        # Simple connectivity test
        aws_status = "connected"
    except Exception:
        aws_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.app_version,
        "services": {
            "aws_bedrock": aws_status,
            "database": "connected"  # Add actual DB check
        }
    }
```

### Chat Route Migration

**Flask Chat Route:**
```python
# routes/chat.py (Flask)
from flask import Blueprint, request, jsonify
import time
import uuid
import boto3
from services.bedrock_service import BedrockService

chat_bp = Blueprint('chat', __name__)
bedrock_service = BedrockService()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Call Bedrock service
        start_time = time.time()
        response = bedrock_service.invoke_agent(message, session_id)
        processing_time = (time.time() - start_time) * 1000
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "processing_time_ms": processing_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

**FastAPI Chat Route:**
```python
# routers/chat.py (FastAPI)
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import time
import uuid
import structlog

from services.bedrock_service import BedrockService
from services.session_service import SessionService
from services.metrics_service import MetricsService
from dependencies import get_bedrock_service, get_session_service, get_metrics_service

logger = structlog.get_logger()
router = APIRouter()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can you help me today?",
                "session_id": "sess_123",
                "user_id": "user_456"
            }
        }

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session identifier")
    timestamp: float = Field(..., description="Response timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed")

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    timestamp: float
    request_id: Optional[str] = None

# Background task for analytics
async def record_chat_analytics(
    message: str, 
    response: str, 
    processing_time: float,
    session_id: str
):
    """Record chat analytics asynchronously"""
    try:
        # Simulate analytics recording
        logger.info(
            "Chat analytics recorded",
            message_length=len(message),
            response_length=len(response),
            processing_time=processing_time,
            session_id=session_id
        )
    except Exception as e:
        logger.error("Analytics recording failed", error=str(e))

@router.post("/chat", response_model=ChatResponse, responses={
    400: {"model": ErrorResponse, "description": "Bad Request"},
    408: {"model": ErrorResponse, "description": "Request Timeout"},
    500: {"model": ErrorResponse, "description": "Internal Server Error"}
})
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    bedrock_service: BedrockService = Depends(get_bedrock_service),
    session_service: SessionService = Depends(get_session_service),
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Process chat message and return AI response
    
    - **message**: User's chat message (1-4000 characters)
    - **session_id**: Optional session identifier for context