# Chapter 9, Section 4.3c: Complete Integration and Production Examples

## Overview

This final subsection demonstrates the complete integration of the AI router with caching, provides production-ready examples, and shows how to implement monitoring, scaling, and operational best practices.

## Complete Integration Example

### Production-Ready AI Service

```python
# services/production_ai_service.py
import asyncio
import structlog
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
import time
import json

from services.ai_router_service import AIServiceRouter, RoutingStrategy, TaskType
from services.ai_cache_service import AdvancedCacheService, CacheStrategy

logger = structlog.get_logger()

@dataclass
class RequestConfig:
    """Configuration for AI requests"""
    max_retries: int = 3
    timeout: float = 30.0
    cache_enabled: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    routing_strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT
    fallback_enabled: bool = True
    cost_limit: Optional[float] = None
    speed_priority: bool = False
    reliability_priority: bool = False

class ProductionAIService:
    """Production-ready AI service with routing, caching, and monitoring"""
    
    def __init__(self, 
                 cache_size: int = 2000,
                 default_ttl: float = 3600,
                 enable_metrics: bool = True):
        self.router = AIServiceRouter()
        self.cache = AdvancedCacheService(max_l1_size=cache_size, default_ttl=default_ttl)
        self.enable_metrics = enable_metrics
        
        # Service metrics
        self.service_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'provider_requests': 0,
            'errors': 0,
            'total_response_time': 0.0,
            'cost_savings_from_cache': 0.0
        }
        
        # Circuit breaker for providers
        self.circuit_breakers = {}
        
        # Request queue for rate limiting
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.processing_requests = False
    
    async def initialize(self):
        """Initialize the service"""
        logger.info("Initializing Production AI Service")
        
        # Initialize router
        await self.router.initialize()
        
        # Start background tasks
        if self.enable_metrics:
            asyncio.create_task(self._metrics_cleanup_task())
            asyncio.create_task(self._health_monitoring_task())
        
        # Start request processor
        asyncio.create_task(self._process_request_queue())
        
        logger.info("Production AI Service initialized successfully")
    
    async def process_request(
        self,
        prompt: str,
        config: Optional[RequestConfig] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process AI request with full production features"""
        
        start_time = time.time()
        config = config or RequestConfig()
        request_id = request_id or f"req_{int(time.time() * 1000)}"
        
        # Update service stats
        self.service_stats['total_requests'] += 1
        
        logger.info(
            "Processing AI request",
            request_id=request_id,
            user_id=user_id,
            prompt_length=len(prompt),
            config=config.__dict__
        )
        
        try:
            # 1. Check cache first if enabled
            cached_response = None
            if config.cache_enabled:
                cached_response = await self._get_cached_response(
                    prompt, config, request_id
                )
            
            if cached_response:
                # Cache hit - return immediately
                response_time = time.time() - start_time
                
                result = {
                    'request_id': request_id,
                    'response': cached_response['response'],
                    'cache_hit': True,
                    'cache_age': cached_response.get('cache_age', 0),
                    'response_time': response_time,
                    'provider_used': cached_response.get('provider', 'cache'),
                    'cost_estimate': 0.0,  # No cost for cached responses
                    'metadata': metadata or {}
                }
                
                self.service_stats['cache_hits'] += 1
                logger.info(
                    "Request served from cache",
                    request_id=request_id,
                    response_time=response_time,
                    cache_age=cached_response.get('cache_age', 0)
                )
                
                return result
            
            # 2. Cache miss - route to provider
            self.service_stats['cache_misses'] += 1
            
            # Classify task type
            task_type = await self.router._classify_task_type(prompt)
            
            # Route request
            provider_response = await self.router.route_request(
                prompt=prompt,
                strategy=config.routing_strategy,
                task_type=task_type,
                max_cost=config.cost_limit,
                fallback_enabled=config.fallback_enabled,
                max_retries=config.max_retries,
                timeout=config.timeout
            )
            
            self.service_stats['provider_requests'] += 1
            
            # 3. Store in cache if enabled
            if config.cache_enabled:
                await self._store_in_cache(
                    prompt, provider_response, config, task_type
                )
            
            # 4. Prepare final response
            response_time = time.time() - start_time
            
            result = {
                'request_id': request_id,
                'response': provider_response.get('response', ''),
                'cache_hit': False,
                'response_time': response_time,
                'provider_used': provider_response.get('provider_used'),
                'model_used': provider_response.get('model_used'),
                'task_type': task_type.value,
                'routing_strategy': config.routing_strategy.value,
                'fallback_used': provider_response.get('fallback_used', False),
                'attempt_number': provider_response.get('attempt_number', 1),
                'cost_estimate': self._estimate_cost(provider_response),
                'metadata': metadata or {}
            }
            
            # Update service stats
            self.service_stats['total_response_time'] += response_time
            
            logger.info(
                "Request processed successfully",
                request_id=request_id,
                provider=result['provider_used'],
                response_time=response_time,
                cost_estimate=result['cost_estimate']
            )
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            self.service_stats['errors'] += 1
            error_response_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                response_time=error_response_time
            )
            
            return {
                'request_id': request_id,
                'error': str(e),
                'cache_hit': False,
                'response_time': error_response_time,
                'success': False,
                'metadata': metadata or {}
            }
    
    async def _get_cached_response(
        self,
        prompt: str,
        config: RequestConfig,
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get response from cache"""
        
        try:
            return await self.cache.get(
                prompt=prompt,
                strategy=config.cache_strategy
            )
        except Exception as e:
            logger.warning(
                "Cache lookup failed",
                request_id=request_id,
                error=str(e)
            )
            return None
    
    async def _store_in_cache(
        self,
        prompt: str,
        provider_response: Dict[str, Any],
        config: RequestConfig,
        task_type: TaskType
    ):
        """Store response in cache"""
        
        try:
            await self.cache.set(
                prompt=prompt,
                response=provider_response,
                provider=provider_response.get('provider_used'),
                model=provider_response.get('model_used'),
                task_type=task_type.value,
                response_time=provider_response.get('response_time'),
                cost_estimate=self._estimate_cost(provider_response),
                strategy=config.cache_strategy
            )
        except Exception as e:
            logger.warning("Cache storage failed", error=str(e))
    
    def _estimate_cost(self, provider_response: Dict[str, Any]) -> float:
        """Estimate cost of provider response"""
        
        provider = provider_response.get('provider_used')
        if not provider:
            return 0.0
        
        # Simple cost estimation (would be more sophisticated in production)
        provider_enum = None
        for p in self.router.providers.keys():
            if p.value == provider:
                provider_enum = p
                break
        
        if provider_enum:
            return self.router._get_avg_cost(provider_enum) * 0.5  # Rough estimate
        
        return 0.001  # Default estimate
    
    async def batch_process(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 10,
        config: Optional[RequestConfig] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple requests concurrently"""
        
        config = config or RequestConfig()
        
        logger.info(
            "Starting batch processing",
            batch_size=len(requests),
            max_concurrent=max_concurrent
        )
        
        async def process_single_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_request(
                    prompt=request_data.get('prompt', ''),
                    config=config,
                    user_id=request_data.get('user_id'),
                    request_id=request_data.get('request_id'),
                    metadata=request_data.get('metadata')
                )
        
        # Process all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(
            *[process_single_request(req) for req in requests],
            return_exceptions=True
        )
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'request_id': requests[i].get('request_id', f'batch_{i}'),
                    'error': str(result),
                    'success': False
                })
            else:
                processed_results.append(result)
        
        batch_time = time.time() - start_time
        successful_count = sum(1 for r in processed_results if r.get('success', True))
        
        logger.info(
            "Batch processing completed",
            total_requests=len(requests),
            successful=successful_count,
            failed=len(requests) - successful_count,
            total_time=batch_time,
            avg_time_per_request=batch_time / len(requests)
        )
        
        return processed_results
    
    async def _process_request_queue(self):
        """Background task to process queued requests"""
        self.processing_requests = True
        
        while self.processing_requests:
            try:
                # Get request from queue (wait up to 1 second)
                request_data = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=1.0
                )
                
                # Process the request
                await self.process_request(**request_data)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except asyncio.TimeoutError:
                # No requests in queue, continue
                continue
            except Exception as e:
                logger.error("Error processing queued request", error=str(e))
    
    async def _metrics_cleanup_task(self):
        """Background task for metrics cleanup and cache maintenance"""
        
        while True:
            try:
                # Clean expired cache entries every 5 minutes
                expired_count = self.cache.clear_expired()
                if expired_count > 0:
                    logger.info("Cleaned expired cache entries", count=expired_count)
                
                # Log service metrics every 10 minutes
                if self.service_stats['total_requests'] > 0:
                    avg_response_time = (
                        self.service_stats['total_response_time'] / 
                        self.service_stats['total_requests']
                    )
                    
                    cache_hit_rate = (
                        self.service_stats['cache_hits'] / 
                        (self.service_stats['cache_hits'] + self.service_stats['cache_misses']) * 100
                        if (self.service_stats['cache_hits'] + self.service_stats['cache_misses']) > 0 else 0
                    )
                    
                    logger.info(
                        "Service metrics summary",
                        total_requests=self.service_stats['total_requests'],
                        cache_hit_rate=f"{cache_hit_rate:.1f}%",
                        avg_response_time=f"{avg_response_time:.2f}s",
                        error_rate=f"{self.service_stats['errors'] / self.service_stats['total_requests'] * 100:.1f}%"
                    )
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error("Error in metrics cleanup task", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _health_monitoring_task(self):
        """Background task for health monitoring and alerting"""
        
        while True:
            try:
                # Check provider health
                health_summary = self.router.get_provider_health_summary()
                
                # Alert on unhealthy providers
                for provider_info in health_summary['unhealthy_providers']:
                    logger.warning(
                        "Unhealthy provider detected",
                        provider=provider_info['provider'],
                        success_rate=provider_info['success_rate'],
                        avg_response_time=provider_info['avg_response_time']
                    )
                
                # Alert on low overall health
                if health_summary['overall_health_percentage'] < 50:
                    logger.error(
                        "Critical: Low overall provider health",
                        health_percentage=health_summary['overall_health_percentage']
                    )
                
                await asyncio.sleep(180)  # 3 minutes
                
            except Exception as e:
                logger.error("Error in health monitoring task", error=str(e))
                await asyncio.sleep(60)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        
        # Get router analytics
        router_analytics = await self.router.get_provider_analytics()
        
        # Get cache statistics
        cache_stats = self.cache.get_cache_stats()
        
        # Calculate service metrics
        total_requests = self.service_stats['total_requests']
        cache_hit_rate = 0
        avg_response_time = 0
        error_rate = 0
        
        if total_requests > 0:
            cache_requests = self.service_stats['cache_hits'] + self.service_stats['cache_misses']
            cache_hit_rate = (self.service_stats['cache_hits'] / cache_requests * 100) if cache_requests > 0 else 0
            avg_response_time = self.service_stats['total_response_time'] / total_requests
            error_rate = self.service_stats['errors'] / total_requests * 100
        
        return {
            'service_metrics': {
                'total_requests': total_requests,
                'cache_hit_rate': cache_hit_rate,
                'avg_response_time': avg_response_time,
                'error_rate': error_rate,
                'provider_requests': self.service_stats['provider_requests']
            },
            'cache_status': cache_stats,
            'provider_analytics': router_analytics,
            'health_summary': self.router.get_provider_health_summary(),
            'queue_status': {
                'queue_size': self.request_queue.qsize(),
                'max_queue_size': self.request_queue.maxsize,
                'processing_active': self.processing_requests
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of the service"""
        logger.info("Shutting down Production AI Service")
        
        # Stop processing new requests
        self.processing_requests = False
        
        # Wait for existing requests to complete (with timeout)
        try:
            await asyncio.wait_for(self.request_queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for request queue to empty")
        
        logger.info("Production AI Service shutdown complete")

# Context manager for service lifecycle
@asynccontextmanager
async def ai_service_context(config: Optional[Dict[str, Any]] = None):
    """Context manager for AI service lifecycle"""
    
    service_config = config or {}
    service = ProductionAIService(**service_config)
    
    try:
        await service.initialize()
        yield service
    finally:
        await service.shutdown()
```

## Production Usage Examples

### FastAPI Integration

```python
# api/ai_endpoints.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
from contextlib import asynccontextmanager

from services.production_ai_service import ProductionAIService, RequestConfig
from services.ai_router_service import RoutingStrategy, TaskType
from services.ai_cache_service import CacheStrategy

# Global service instance
ai_service: Optional[ProductionAIService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global ai_service
    
    # Startup
    ai_service = ProductionAIService(cache_size=5000, enable_metrics=True)
    await ai_service.initialize()
    
    yield
    
    # Shutdown
    if ai_service:
        await ai_service.shutdown()

app = FastAPI(
    title="AI Service API",
    description="Production AI service with intelligent routing and caching",
    version="1.0.0",
    lifespan=lifespan
)

# Request models
class AIRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for the AI model")
    task_type: Optional[str] = Field(None, description="Type of task (auto-detected if not provided)")
    routing_strategy: Optional[str] = Field("intelligent", description="Routing strategy to use")
    cache_enabled: bool = Field(True, description="Whether to use caching")
    max_retries: int = Field(3, description="Maximum number of retries")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    cost_limit: Optional[float] = Field(None, description="Maximum cost per 1K tokens")
    speed_priority: bool = Field(False, description="Prioritize speed over other factors")
    reliability_priority: bool = Field(False, description="Prioritize reliability over other factors")
    user_id: Optional[str] = Field(None, description="User identifier for tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class BatchRequest(BaseModel):
    requests: List[AIRequest] = Field(..., description="List of AI requests to process")
    max_concurrent: int = Field(10, description="Maximum concurrent requests")

class AIResponse(BaseModel):
    request_id: str
    response: Optional[str] = None
    error: Optional[str] = None
    cache_hit: bool
    response_time: float
    provider_used: Optional[str] = None
    cost_estimate: float
    success: bool = True

# API endpoints
@app.post("/ai/generate", response_model=AIResponse)
async def generate_response(request: AIRequest):
    """Generate AI response with intelligent routing and caching"""
    
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        # Convert request to config
        config = RequestConfig(
            max_retries=request.max_retries,
            timeout=request.timeout,
            cache_enabled=request.cache_enabled,
            cache_strategy=CacheStrategy.ADAPTIVE,
            routing_strategy=RoutingStrategy(request.routing_strategy),
            cost_limit=request.cost_limit,
            speed_priority=request.speed_priority,
            reliability_priority=request.reliability_priority
        )
        
        # Process request
        result = await ai_service.process_request(
            prompt=request.prompt,
            config=config,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        # Convert to response model
        return AIResponse(
            request_id=result['request_id'],
            response=result.get('response'),
            error=result.get('error'),
            cache_hit=result.get('cache_hit', False),
            response_time=result['response_time'],
            provider_used=result.get('provider_used'),
            cost_estimate=result.get('cost_estimate', 0.0),
            success=result.get('success', True)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")

@app.post("/ai/batch", response_model=List[AIResponse])
async def batch_generate(request: BatchRequest):
    """Process multiple AI requests concurrently"""
    
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        # Convert requests to dict format
        batch_requests = []
        for ai_req in request.requests:
            config = RequestConfig(
                max_retries=ai_req.max_retries,
                timeout=ai_req.timeout,
                cache_enabled=ai_req.cache_enabled,
                cache_strategy=CacheStrategy.ADAPTIVE,
                routing_strategy=RoutingStrategy(ai_req.routing_strategy),
                cost_limit=ai_req.cost_limit,
                speed_priority=ai_req.speed_priority,
                reliability_priority=ai_req.reliability_priority
            )
            
            batch_requests.append({
                'prompt': ai_req.prompt,
                'config': config,
                'user_id': ai_req.user_id,
                'metadata': ai_req.metadata
            })
        
        # Process batch
        results = await ai_service.batch_process(
            batch_requests,
            max_concurrent=request.max_concurrent
        )
        
        # Convert to response models
        return [
            AIResponse(
                request_id=result['request_id'],
                response=result.get('response'),
                error=result.get('error'),
                cache_hit=result.get('cache_hit', False),
                response_time=result['response_time'],
                provider_used=result.get('provider_used'),
                cost_estimate=result.get('cost_estimate', 0.0),
                success=result.get('success', True)
            )
            for result in results
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/ai/status")
async def get_service_status():
    """Get comprehensive service status and analytics"""
    
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    return await ai_service.get_service_status()

@app.get("/ai/recommendations")
async def get_recommendations(
    prompt: str,
    speed_priority: bool = False,
    reliability_priority: bool = False,
    budget_constraint: Optional[float] = None
):
    """Get provider recommendations for a specific prompt"""
    
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    try:
        return await ai_service.router.recommend_provider(
            prompt=prompt,
            speed_priority=speed_priority,
            reliability_priority=reliability_priority,
            budget_constraint=budget_constraint,
            detailed_analysis=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/ai/cache/clear")
async def clear_cache(tags: Optional[List[str]] = None):
    """Clear cache entries, optionally by tags"""
    
    if not ai_service:
        raise HTTPException(status_code=503, detail="AI service not available")
    
    if tags:
        cleared = ai_service.cache.clear_by_tags(tags)
    else:
        cleared = ai_service.cache.clear_expired()
    
    return {"cleared_entries": cleared}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if not ai_service:
        return {"status": "unhealthy", "message": "AI service not available"}
    
    health_summary = ai_service.router.get_provider_health_summary()
    
    return {
        "status": "healthy" if health_summary['overall_health_percentage'] > 50 else "degraded",
        "healthy_providers": health_summary['healthy_providers'],
        "overall_health": f"{health_summary['overall_health_percentage']:.1f}%"
    }
```

## Complete Production Demo

### End-to-End Example

```python
# demo/production_demo.py
import asyncio
import json
from typing import List, Dict, Any

from services.production_ai_service import ProductionAIService, RequestConfig, ai_service_context
from services.ai_router_service import RoutingStrategy
from services.ai_cache_service import CacheStrategy

async def run_production_demo():
    """Comprehensive production demo"""
    
    print("🚀 Starting Production AI Service Demo")
    print("=" * 60)
    
    # Use context manager for proper lifecycle management
    async with ai_service_context({'cache_size': 1000, 'enable_metrics': True}) as service:
        
        # Demo 1: Single requests with different configurations
        print("\n📝 Demo 1: Single Requests with Different Configurations")
        
        test_requests = [
            {
                'prompt': "Analyze the performance of our Q3 sales data",
                'config': RequestConfig(
                    routing_strategy=RoutingStrategy.INTELLIGENT,
                    cache_enabled=True,
                    speed_priority=False,
                    reliability_priority=True
                ),
                'description': "Analysis task with reliability priority"
            },
            {
                'prompt': "Write a Python function to sort a list efficiently",
                'config': RequestConfig(
                    routing_strategy=RoutingStrategy.PERFORMANCE_BASED,
                    cache_enabled=True,
                    speed_priority=True,
                    cost_limit=0.005
                ),
                'description': "Code generation with speed priority and cost limit"
            },
            {
                'prompt': "Create a creative story about artificial intelligence",
                'config': RequestConfig(
                    routing_strategy=RoutingStrategy.COST_OPTIMIZED,
                    cache_enabled=False,  # Creative content shouldn't be cached
                    speed_priority=False
                ),
                'description': "Creative writing with cost optimization (no cache)"
            }
        ]
        
        for i, test_case in enumerate(test_requests, 1):
            print(f"\n--- Request {i}: {test_case['description']} ---")
            
            result = await service.process_request(
                prompt=test_case['prompt'],
                config=test_case['config'],
                user_id=f"demo_user_{i}",
                request_id=f"demo_{i}",
                metadata={'demo': True, 'test_case': i}
            )
            
            print(f"✅ Provider: {result.get('provider_used', 'unknown')}")
            print(f"📊 Cache Hit: {result.get('cache_hit', False)}")
            print(f"⏱️  Response Time: {result.get('response_time', 0):.2f}s")
            print(f"💰 Cost Estimate: ${result.get('cost_estimate', 0):.4f}")
            print(f"🔄 Fallback Used: {result.get('fallback_used', False)}")
        
        # Demo 2: Batch processing
        print(f"\n📦 Demo 2: Batch Processing")
        
        batch_requests = [
            {'prompt': f"Summarize document #{i}", 'user_id': 'batch_user'}
            for i in range(1, 6)
        ]
        
        batch_config = RequestConfig(
            routing_strategy=RoutingStrategy.COST_OPTIMIZED,
            cache_enabled=True,
            max_retries=2
        )
        
        print(f"Processing {len(batch_requests)} requests concurrently...")
        
        batch_results = await service.batch_process(
            batch_requests,
            max_concurrent=3,
            config=batch_config
        )
        
        successful = sum(1 for r in batch_results if r.get('success', True))
        total_time = sum(r.get('response_time', 0) for r in batch_results)
        
        print(f"✅ Batch completed: {successful}/{len(batch_requests)} successful")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Average time per request: {total_time/len(batch_requests):.2f}s")
        
        # Demo 3: Cache effectiveness test
        print(f"\n🗄️  Demo 3: Cache Effectiveness Test")
        
        # Make the same request multiple times to test caching
        cache_test_prompt = "What are the benefits of artificial intelligence?"
        cache_test_config = RequestConfig(
            cache_enabled=True,
            cache_strategy=CacheStrategy.SEMANTIC
        )
        
        print("Making identical requests to test cache performance...")
        
        cache_test_results = []
        for i in range(3):
            result = await service.process_request(
                prompt=cache_test_prompt,
                config=cache_test_config,
                request_id=f"cache_test_{i}"
            )
            cache_test_results.append(result)
            
            print(f"Request {i+1}: "
                  f"Cache Hit: {result.get('cache_hit', False)}, "
                  f"Time: {result.get('response_time', 0):.2f}s")
        
        # Demo 4: Semantic cache test
        print(f"\n🔍 Demo 4: Semantic Cache Test")
        
        similar_prompts = [
            "What are the benefits of AI?",
            "Can you explain the advantages of artificial intelligence?",
            "Tell me about the benefits of artificial intelligence"
        ]
        
        print("Testing semantic similarity caching with similar prompts...")
        
        for i, prompt in enumerate(similar_prompts):
            result = await service.process_request(
                prompt=prompt,
                config=RequestConfig(cache_enabled=True, cache_strategy=CacheStrategy.SEMANTIC),
                request_id=f"semantic_test_{i}"
            )
            
            print(f"'{prompt[:30]}...': "
                  f"Cache Hit: {result.get('cache_hit', False)}, "
                  f"Similarity: {result.get('cache_similarity', 0):.2f}")
        
        # Demo 5: Service analytics and monitoring
        print(f"\n📈 Demo 5: Service Analytics and Monitoring")
        
        # Get comprehensive service status
        status = await service.get_service_status()
        
        print(f"Service Metrics:")
        metrics = status['service_metrics']
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"  Average Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"  Error Rate: {metrics['error_rate']:.1f}%")
        
        print(f"\nProvider Health:")
        health = status['health_summary']
        print(f"  Overall Health: {health['overall_health_percentage']:.1f}%")
        print(f"  Healthy Providers: {len(health['healthy_providers'])}")
        print(f"  Unhealthy Providers: {len(health['unhealthy_providers'])}")
        
        print(f"\nCache Performance:")
        cache_stats = status['cache_status']
        print(f"  Cache Size: {cache_stats['cache_size']}/{cache_stats['max_size']}")
        print(f"  Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"  Estimated Cost Savings: ${cache_stats['estimated_cost_savings']:.4f}")
        print(f"  Estimated Time Savings: {cache_stats['estimated_time_savings']:.2f}s")
        
        # Demo 6: Provider recommendations
        print(f"\n💡 Demo 6: Provider Recommendations")
        
        recommendation_prompts = [
            ("Generate Python code for data analysis", True, False),  # Speed priority
            ("Analyze complex business metrics", False, True),        # Reliability priority
            ("Write a simple summary", False, False)                 # Balanced
        ]
        
        for prompt, speed_priority, reliability_priority in recommendation_prompts:
            recommendations = await service.router.recommend_provider(
                prompt=prompt,
                speed_priority=speed_priority,
                reliability_priority=reliability_priority,
                detailed_analysis=False
            )
            
            print(f"\nPrompt: '{prompt[:40]}...'")
            print(f"Priority: {'Speed' if speed_priority else 'Reliability' if reliability_priority else 'Balanced'}")
            
            if recommendations['recommendations']:
                top_rec = recommendations['recommendations'][0]
                print(f"Recommended: {top_rec['provider']} (score: {top_rec['total_score']:.1f})")
                print(f"Reasons: {', '.join(top_rec['reasons'][:2])}")
    
    print(f"\n🎉 Production Demo Completed Successfully!")
    print("The AI service has been properly shut down.")

if __name__ == "__main__":
    asyncio.run(run_production_demo())
```

## Key Production Features

This complete production system provides:

1. **Intelligent Routing**: Multi-factor provider selection with real-time adaptation
2. **Advanced Caching**: Semantic similarity matching with provider-aware TTLs
3. **Comprehensive Monitoring**: Real-time metrics, health checks, and alerting
4. **Graceful Error Handling**: Circuit breakers, retries, and fallback mechanisms
5. **Scalable Architecture**: Async processing, batch operations, and queue management
6. **Production APIs**: FastAPI integration with proper lifecycle management
7. **Cost Optimization**: Intelligent cost analysis and optimization recommendations
8. **Performance Analytics**: Detailed insights into provider performance and trends

The system is designed for enterprise use with proper logging, monitoring, and operational best practices.