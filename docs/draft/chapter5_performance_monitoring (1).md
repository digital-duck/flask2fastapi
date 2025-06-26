# Flask to FastAPI Migration Guide
## Chapter 5: Performance Monitoring & Metrics

### Overview

This chapter focuses on implementing comprehensive performance monitoring to validate the migration success and provide ongoing visibility into system performance. The metrics will provide concrete data to demonstrate the business value of the FastAPI migration.

---

## Key Performance Indicators (KPIs)

### Primary Metrics to Track

**Performance Metrics:**
- **Queries Per Second (QPS)** - Request throughput
- **Latency P50/P90/P99** - Response time percentiles
- **Error Rate** - Percentage of failed requests
- **Concurrent User Capacity** - Maximum simultaneous users

**Business Metrics:**
- **Infrastructure Cost** - Monthly AWS spend
- **User Experience** - Average response times
- **System Reliability** - Uptime percentage
- **Resource Utilization** - CPU, Memory, Network usage

---

## CloudWatch Metrics Implementation

### 1. Core Metrics Service

```python
# services/metrics_service.py
import aioboto3
import asyncio
import time
import structlog
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, deque
import statistics

from config.settings import settings

logger = structlog.get_logger()

class MetricsService:
    """Async CloudWatch metrics service with intelligent batching"""
    
    def __init__(self, namespace: str = None):
        self.namespace = namespace or settings.metrics_namespace
        self.session = aioboto3.Session()
        self._metrics_buffer = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task = None
        
        # In-memory metrics for real-time dashboard
        self._request_times = deque(maxlen=1000)
        self._error_count = 0
        self._total_requests = 0
        self._start_time = time.time()
    
    async def start_background_flush(self):
        """Start background task to flush metrics periodically"""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._periodic_flush())
    
    async def stop_background_flush(self):
        """Stop background flush task"""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            finally:
                await self._flush_metrics()  # Final flush
    
    async def _periodic_flush(self):
        """Periodically flush metrics buffer"""
        while True:
            try:
                await asyncio.sleep(60)  # Flush every minute
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Periodic metrics flush failed", error=str(e))
    
    async def put_metric(
        self, 
        metric_name: str, 
        value: float,
        unit: str = "Count",
        dimensions: Dict[str, str] = None,
        timestamp: datetime = None
    ):
        """
        Put single metric to CloudWatch buffer
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement (Count, Milliseconds, etc.)
            dimensions: Additional dimensions for the metric
            timestamp: Metric timestamp (defaults to now)
        """
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': timestamp or datetime.utcnow()
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
        
        async with self._buffer_lock:
            self._metrics_buffer.append(metric_data)
            
            # Auto-flush if buffer is getting full
            if len(self._metrics_buffer) >= 20:
                await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Flush metrics buffer to CloudWatch"""
        if not self._metrics_buffer:
            return
            
        async with self._buffer_lock:
            metrics_to_send = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
        
        if not metrics_to_send:
            return
        
        try:
            async with self.session.client('cloudwatch') as cw:
                # CloudWatch accepts max 20 metrics per call
                for i in range(0, len(metrics_to_send), 20):
                    batch = metrics_to_send[i:i+20]
                    await cw.put_metric_data(
                        Namespace=self.namespace,
                        MetricData=batch
                    )
                    
            logger.debug(
                "Metrics flushed to CloudWatch",
                metrics_count=len(metrics_to_send)
            )
            
        except Exception as e:
            logger.error(
                "Failed to flush metrics to CloudWatch",
                error=str(e),
                metrics_count=len(metrics_to_send)
            )
    
    async def record_request_metrics(
        self, 
        endpoint: str, 
        method: str,
        status_code: int, 
        duration: float,
        user_agent: str = None
    ):
        """Record comprehensive request metrics"""
        # Update in-memory counters
        self._request_times.append(duration)
        self._total_requests += 1
        if status_code >= 400:
            self._error_count += 1
        
        dimensions = {
            'Endpoint': endpoint,
            'Method': method,
            'StatusCode': str(status_code)
        }
        
        # Add user agent dimension if available
        if user_agent:
            # Simplify user agent for grouping
            if 'mobile' in user_agent.lower():
                dimensions['ClientType'] = 'Mobile'
            elif 'bot' in user_agent.lower():
                dimensions['ClientType'] = 'Bot'
            else:
                dimensions['ClientType'] = 'Desktop'
        
        # Send multiple metrics
        await asyncio.gather(
            self.put_metric('RequestCount', 1, 'Count', dimensions),
            self.put_metric('RequestLatency', duration * 1000, 'Milliseconds', dimensions),
            self.put_metric(
                'ErrorCount' if status_code >= 400 else 'SuccessCount', 
                1, 'Count', dimensions
            )
        )
    
    async def record_bedrock_metrics(
        self, 
        duration: float, 
        success: bool,
        tokens_used: int = None,
        model_id: str = None
    ):
        """Record Bedrock-specific metrics"""
        dimensions = {'Service': 'Bedrock'}
        if model_id:
            dimensions['ModelId'] = model_id
        
        metrics = [
            ('BedrockLatency', duration * 1000, 'Milliseconds'),
            ('BedrockRequestCount', 1, 'Count'),
        ]
        
        if success:
            metrics.append(('BedrockSuccessCount', 1, 'Count'))
            if tokens_used:
                metrics.append(('BedrockTokensUsed', tokens_used, 'Count'))
        else:
            metrics.append(('BedrockErrorCount', 1, 'Count'))
        
        # Send all metrics
        for metric_name, value, unit in metrics:
            await self.put_metric(metric_name, value, unit, dimensions)
    
    async def record_custom_metric(
        self, 
        metric_name: str, 
        value: float,
        dimensions: Dict[str, str] = None
    ):
        """Record custom business metrics"""
        await self.put_metric(
            metric_name, 
            value, 
            'Count', 
            dimensions or {}
        )
    
    def get_realtime_metrics(self) -> Dict:
        """Get real-time metrics from in-memory data"""
        if not self._request_times:
            return {
                "qps": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p90_latency_ms": 0,
                "p99_latency_ms": 0,
                "error_rate": 0,
                "total_requests": 0,
                "uptime_seconds": time.time() - self._start_time
            }
        
        uptime = time.time() - self._start_time
        request_times = list(self._request_times)
        
        return {
            "qps": self._total_requests / uptime if uptime > 0 else 0,
            "avg_latency_ms": statistics.mean(request_times) * 1000,
            "p50_latency_ms": statistics.median(request_times) * 1000,
            "p90_latency_ms": statistics.quantiles(request_times, n=10)[8] * 1000 if len(request_times) >= 10 else max(request_times) * 1000,
            "p99_latency_ms": statistics.quantiles(request_times, n=100)[98] * 1000 if len(request_times) >= 100 else max(request_times) * 1000,
            "error_rate": (self._error_count / self._total_requests) * 100 if self._total_requests > 0 else 0,
            "total_requests": self._total_requests,
            "uptime_seconds": uptime
        }

# Global metrics instance
metrics_service = MetricsService()
```

### 2. Automatic Metrics Middleware

```python
# middleware/metrics_middleware.py
import time
import asyncio
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog

from services.metrics_service import metrics_service

logger = structlog.get_logger()

class PerformanceMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically collect performance metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.active_requests = 0
        self.peak_concurrent_requests = 0
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect comprehensive metrics"""
        start_time = time.time()
        
        # Track concurrent requests
        self.active_requests += 1
        if self.active_requests > self.peak_concurrent_requests:
            self.peak_concurrent_requests = self.active_requests
        
        # Extract request information
        method = request.method
        path = self._normalize_path(request.url.path)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Add request ID for tracing
        request_id = request.headers.get("x-request-id", f"req-{int(time.time()*1000)}")
        
        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Handle exceptions
            logger.error(
                "Request processing failed",
                method=method,
                path=path,
                error=str(e),
                request_id=request_id,
                exc_info=True
            )
            status_code = 500
            
            # Create error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id
                }
            )
        finally:
            # Always decrement active requests
            self.active_requests -= 1
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Log request details
        logger.info(
            "Request completed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration * 1000,
            client_ip=client_ip,
            request_id=request_id,
            active_requests=self.active_requests
        )
        
        # Record metrics asynchronously
        asyncio.create_task(
            metrics_service.record_request_metrics(
                endpoint=path,
                method=method,
                status_code=status_code,
                duration=duration,
                user_agent=user_agent
            )
        )
        
        # Record concurrent requests metric
        if self.active_requests > 0:
            asyncio.create_task(
                metrics_service.put_metric(
                    'ConcurrentRequests', 
                    self.active_requests, 
                    'Count'
                )
            )
        
        # Add response headers for debugging
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics grouping"""
        # Replace dynamic path parameters with placeholders
        # e.g., /users/123 -> /users/{id}
        import re
        
        # Common patterns
        patterns = [
            (r'/\d+', '/{id}'),  # Numbers
            (r'/[0-9a-fA-F-]{36}', '/{uuid}'),  # UUIDs
            (r'/sess_[a-zA-Z0-9]+', '/{session_id}'),  # Session IDs
        ]
        
        normalized = path
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
```

### 3. Enhanced Bedrock Service with Metrics

```python
# services/bedrock_service_with_metrics.py
import aioboto3
import asyncio
import time
import structlog
from typing import Dict, Optional

from config.settings import settings
from services.metrics_service import metrics_service

logger = structlog.get_logger()

class EnhancedBedrockService:
    """Bedrock service with comprehensive metrics collection"""
    
    def __init__(self):
        self.region_name = settings.aws_region
        self.agent_id = settings.bedrock_agent_id
        self.session = aioboto3.Session()
        
        # Performance tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.total_tokens = 0
    
    async def invoke_agent(
        self, 
        message: str, 
        session_id: str,
        timeout: float = None
    ) -> Dict:
        """Enhanced invoke_agent with detailed metrics"""
        start_time = time.time()
        timeout = timeout or settings.request_timeout
        success = False
        tokens_used = 0
        model_id = "unknown"
        
        # Increment call counter
        self.total_calls += 1
        
        try:
            async with self.session.client(
                'bedrock-agent',
                region_name=self.region_name
            ) as client:
                
                # Make the Bedrock call with timeout
                response = await asyncio.wait_for(
                    client.invoke_agent(
                        agentId=self.agent_id,
                        sessionId=session_id,
                        inputText=message
                    ),
                    timeout=timeout
                )
                
                # Extract response data
                success = True
                self.successful_calls += 1
                tokens_used = response.get('usage', {}).get('totalTokens', 0)
                model_id = response.get('modelId', 'bedrock-agent')
                self.total_tokens += tokens_used
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Log successful call
                logger.info(
                    "Bedrock invocation successful",
                    session_id=session_id,
                    duration_ms=duration * 1000,
                    tokens_used=tokens_used,
                    message_length=len(message),
                    response_length=len(response.get('output', {}).get('text', ''))
                )
                
                return {
                    'response': response['output']['text'],
                    'session_id': session_id,
                    'tokens_used': tokens_used,
                    'model_id': model_id,
                    'processing_time': duration
                }
                
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.warning(
                "Bedrock request timeout",
                session_id=session_id,
                timeout=timeout,
                duration_ms=duration * 1000
            )
            return {
                "error": f"Request timeout after {timeout} seconds",
                "session_id": session_id
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Bedrock invocation failed",
                session_id=session_id,
                error=str(e),
                duration_ms=duration * 1000,
                exc_info=True
            )
            return {
                "error": f"Bedrock error: {str(e)}",
                "session_id": session_id
            }
            
        finally:
            # Always record metrics
            duration = time.time() - start_time
            await metrics_service.record_bedrock_metrics(
                duration=duration,
                success=success,
                tokens_used=tokens_used,
                model_id=model_id
            )
            
            # Record business metrics
            await metrics_service.record_custom_metric(
                "BedrockAPICallsTotal",
                self.total_calls,
                {"Agent": self.agent_id}
            )
            
            if success:
                await metrics_service.record_custom_metric(
                    "BedrockTokensTotal",
                    self.total_tokens,
                    {"Agent": self.agent_id}
                )
    
    def get_service_stats(self) -> Dict:
        """Get service-level statistics"""
        success_rate = (self.successful_calls / self.total_calls) * 100 if self.total_calls > 0 else 0
        avg_tokens = self.total_tokens / self.successful_calls if self.successful_calls > 0 else 0
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "success_rate": success_rate,
            "total_tokens_used": self.total_tokens,
            "avg_tokens_per_call": avg_tokens
        }
```

---

## Real-Time Performance Dashboard

### 1. Metrics Endpoint

```python
# routers/metrics.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Optional
import time

from services.metrics_service import metrics_service
from dependencies import get_bedrock_service
from middleware.metrics_middleware import PerformanceMetricsMiddleware

router = APIRouter()

class PerformanceMetrics(BaseModel):
    """Real-time performance metrics model"""
    qps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    total_requests: int
    uptime_seconds: float
    concurrent_requests: int

class SystemHealth(BaseModel):
    """System health status model"""
    status: str
    timestamp: float
    performance: PerformanceMetrics
    services: Dict[str, str]
    version: str

@router.get("/metrics/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get real-time performance metrics"""
    return PerformanceMetrics(**metrics_service.get_realtime_metrics())

@router.get("/metrics/health", response_model=SystemHealth)
async def get_system_health(
    bedrock_service = Depends(get_bedrock_service)
):
    """Get comprehensive system health"""
    # Get performance metrics
    perf_metrics = metrics_service.get_realtime_metrics()
    
    # Check service health
    services_status = {
        "bedrock": "healthy",  # Add actual health check
        "dynamodb": "healthy",  # Add actual health check
        "cloudwatch": "healthy"
    }
    
    # Determine overall status
    overall_status = "healthy"
    if perf_metrics["error_rate"] > 5.0:
        overall_status = "degraded"
    elif perf_metrics["error_rate"] > 10.0:
        overall_status = "unhealthy"
    
    return SystemHealth(
        status=overall_status,
        timestamp=time.time(),
        performance=PerformanceMetrics(**perf_metrics),
        services=services_status,
        version="2.0.0"
    )

@router.get("/metrics/bedrock")
async def get_bedrock_metrics(
    bedrock_service = Depends(get_bedrock_service)
):
    """Get Bedrock service metrics"""
    if hasattr(bedrock_service, 'get_service_stats'):
        return bedrock_service.get_service_stats()
    return {"message": "Bedrock metrics not available"}

@router.get("/metrics/comparison")
async def get_comparison_metrics():
    """Get Flask vs FastAPI comparison metrics"""
    current_metrics = metrics_service.get_realtime_metrics()
    
    # Simulated Flask baseline (replace with actual historical data)
    flask_baseline = {
        "qps": 250,
        "avg_latency_ms": 450,
        "p90_latency_ms": 800,
        "p99_latency_ms": 1200,
        "error_rate": 2.1,
        "concurrent_capacity": 700
    }
    
    # Calculate improvements
    improvements = {}
    for metric, flask_value in flask_baseline.items():
        if metric in current_metrics:
            current_value = current_metrics[metric]
            if metric == "error_rate":
                # Lower is better for error rate
                improvement = ((flask_value - current_value) / flask_value) * 100
            else:
                # Higher is better for other metrics
                improvement = ((current_value - flask_value) / flask_value) * 100
            
            improvements[f"{metric}_improvement_percent"] = improvement
    
    return {
        "flask_baseline": flask_baseline,
        "fastapi_current": current_metrics,
        "improvements": improvements
    }
```

### 2. CloudWatch Dashboard Configuration

```python
# monitoring/dashboard_config.py
import json
from typing import Dict

def create_cloudwatch_dashboard_config(region: str = "us-east-1") -> Dict:
    """Generate CloudWatch dashboard configuration"""
    
    dashboard_config = {
        "widgets": [
            # Queries Per Second Widget
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "RequestCount", "Method", "POST", "Endpoint", "/api/v1/chat"],
                        [".", ".", ".", "GET", ".", "/health"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Queries Per Second",
                    "period": 60,
                    "stat": "Sum",
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            },
            
            # Latency Percentiles Widget
            {
                "type": "metric",
                "x": 12, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "RequestLatency", "Endpoint", "/api/v1/chat", {"stat": "Average"}],
                        ["...", {"stat": "p50"}],
                        ["...", {"stat": "p90"}],
                        ["...", {"stat": "p99"}]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Response Time Percentiles (P50, P90, P99)",
                    "period": 300,
                    "yAxis": {
                        "left": {
                            "min": 0,
                            "max": 5000
                        }
                    }
                }
            },
            
            # Error Rate Widget
            {
                "type": "metric",
                "x": 0, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "ErrorCount", "Endpoint", "/api/v1/chat"],
                        [".", "SuccessCount", ".", "."]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Success vs Error Rate",
                    "period": 300,
                    "stat": "Sum"
                }
            },
            
            # Bedrock Performance Widget
            {
                "type": "metric",
                "x": 12, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "BedrockLatency", "Service", "Bedrock"],
                        [".", "BedrockTokensUsed", ".", "."]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Bedrock Performance",
                    "period": 300,
                    "stat": "Average"
                }
            },
            
            # Concurrent Requests Widget
            {
                "type": "metric",
                "x": 0, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "ConcurrentRequests"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Concurrent Requests",
                    "period": 60,
                    "stat": "Maximum"
                }
            },
            
            # Cost Optimization Widget
            {
                "type": "metric",
                "x": 12, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "view": "singleValue",
                    "metrics": [
                        ["AWS/ECS", "CPUUtilization", "ServiceName", "chatbot-fastapi"],
                        [".", "MemoryUtilization", ".", "."]
                    ],
                    "region": region,
                    "title": "Resource Utilization",
                    "period": 300,
                    "stat": "Average"
                }
            }
        ]
    }
    
    return dashboard_config

def deploy_dashboard(dashboard_name: str = "ChatBot-FastAPI-Performance"):
    """Deploy CloudWatch dashboard"""
    import boto3
    
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_config = create_cloudwatch_dashboard_config()
    
    try:
        cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_config)
        )
        print(f"Dashboard '{dashboard_name}' deployed successfully")
        
        # Return dashboard URL
        region = boto3.Session().region_name or "us-east-1"
        dashboard_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={dashboard_name}"
        print(f"Dashboard URL: {dashboard_url}")
        
        return dashboard_url
        
    except Exception as e:
        print(f"Failed to deploy dashboard: {e}")
        return None
```

---

## Load Testing & Benchmarking

### 1. Automated Load Testing

```python
# tests/performance_benchmark.py
import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_time: float
    requests_per_second: float
    avg_response_time: float
    p50_response_time: float
    p90_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def run_benchmark_suite(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        print("Starting FastAPI Performance Benchmark Suite")
        print("=" * 60)
        
        # Test scenarios
        scenarios = [
            ("baseline_10_users", 10, 100, 0),
            ("moderate_50_users", 50, 500, 5),
            ("high_load_100_users", 100, 1000, 10),
            ("stress_200_users", 200, 2000, 15),
        ]
        
        for name, concurrent_users, total_requests, ramp_up in scenarios:
            print(f"\nRunning {name}...")
            result = await self.run_load_test(
                test_name=name,
                concurrent_users=concurrent_users,
                total_requests=total_requests,
                ramp_up_time=ramp_up
            )
            self.results.append(result)
            self.print_result(result)
            
            # Cool down between tests
            await asyncio.sleep(5)
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    async def run_load_test(
        self,
        test_name: str,
        concurrent_users: int,
        total_requests: int,
        ramp_up_time: float
    ) -> BenchmarkResult:
        """Run individual load test"""
        
        async def make_request(session: aiohttp.ClientSession, request_id: int):
            start_time = time.time()
            
            try:
                # Add ramp-up delay
                if ramp_up_time > 0:
                    delay = (request_id / total_requests) * ramp_up_time
                    await asyncio.sleep(delay)
                
                async with session.post(
                    f"{self.base_url}/api/v1/chat",
                    json={
                        "message": f"Benchmark test message {request_id}",
                        "session_id": f"bench-{test_name}-{request_id}"
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    await response.json()
                    duration = time.time() - start_time
                    
                    return {
                        "success": response.status == 200,
                        "status_code": response.status,
                        