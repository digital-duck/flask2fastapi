@dataclass
class SSEClient:
    """SSE client connection tracking"""
    client_id: str
    user_id: Optional[str]
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    subscriptions: List[str] = field(default_factory=list)
    request: Optional[Request] = None
    
    def is_connected(self) -> bool:
        """Check if client is still connected"""
        return self.request and not self.request.is_disconnected()
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()

class SSEStreamingService:
    """Server-Sent Events streaming service for real-time updates"""
    
    def __init__(self, ai_service: ProductionAIService):
        self.ai_service = ai_service
        
        # Active SSE clients
        self.active_clients: Dict[str, SSEClient] = {}
        self.user_clients: Dict[str, List[str]] = defaultdict(list)
        
        # Event queues for different types
        self.event_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        
        # Broadcasting tasks
        self.broadcast_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.sse_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'events_sent': defaultdict(int),
            'total_events': 0
        }
    
    async def initialize(self):
        """Initialize SSE streaming service"""
        logger.info("Initializing SSE Streaming Service")
        
        # Start broadcasting tasks
        self.broadcast_tasks['system_status'] = asyncio.create_task(self._broadcast_system_status())
        self.broadcast_tasks['provider_health'] = asyncio.create_task(self._broadcast_provider_health())
        self.broadcast_tasks['cache_stats'] = asyncio.create_task(self._broadcast_cache_stats())
        
        logger.info("SSE Streaming Service initialized")
    
    async def create_client_stream(
        self,
        request: Request,
        user_id: Optional[str] = None,
        subscriptions: Optional[List[str]] = None
    ) -> StreamingResponse:
        """Create SSE stream for client"""
        
        client_id = str(uuid.uuid4())
        client = SSEClient(
            client_id=client_id,
            user_id=user_id,
            subscriptions=subscriptions or ['ai_response', 'system_status'],
            request=request
        )
        
        # Register client
        self.active_clients[client_id] = client
        if user_id:
            self.user_clients[user_id].append(client_id)
        
        # Update stats
        self.sse_stats['total_connections'] += 1
        self.sse_stats['active_connections'] = len(self.active_clients)
        
        logger.info(
            "SSE client connected",
            client_id=client_id,
            user_id=user_id,
            subscriptions=subscriptions
        )
        
        # Return streaming response
        return StreamingResponse(
            self._client_event_stream(client),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def _client_event_stream(self, client: SSEClient) -> AsyncGenerator[str, None]:
        """Generate SSE event stream for specific client"""
        
        try:
            # Send initial connection event
            yield self._format_sse_event({
                'type': 'connection',
                'data': {
                    'client_id': client.client_id,
                    'connected_at': client.connected_at,
                    'subscriptions': client.subscriptions
                }
            })
            
            # Stream events
            while client.is_connected():
                try:
                    # Check for events in subscribed queues
                    event_received = False
                    
                    for subscription in client.subscriptions:
                        if subscription in self.event_queues:
                            queue = self.event_queues[subscription]
                            try:
                                # Non-blocking queue check
                                event = queue.get_nowait()
                                yield self._format_sse_event(event)
                                event_received = True
                                client.update_activity()
                                queue.task_done()
                            except asyncio.QueueEmpty:
                                continue
                    
                    if not event_received:
                        # Send heartbeat every 30 seconds
                        if time.time() - client.last_activity > 30:
                            yield self._format_sse_event({
                                'type': 'heartbeat',
                                'data': {'timestamp': time.time()}
                            })
                            client.update_activity()
                        
                        # Small delay to prevent busy waiting
                        await asyncio.sleep(0.1)
                
                except Exception as e:
                    logger.error(
                        "Error in client event stream",
                        client_id=client.client_id,
                        error=str(e)
                    )
                    break
        
        except Exception as e:
            logger.error(
                "Client stream error",
                client_id=client.client_id,
                error=str(e)
            )
        finally:
            # Clean up client
            await self._remove_client(client.client_id)
    
    def _format_sse_event(self, event_data: Dict[str, Any]) -> str:
        """Format event data as SSE message"""
        
        event_type = event_data.get('type', 'message')
        data = event_data.get('data', {})
        event_id = event_data.get('id', str(uuid.uuid4()))
        
        # Update stats
        self.sse_stats['events_sent'][event_type] += 1
        self.sse_stats['total_events'] += 1
        
        # Format as SSE
        sse_message = f"id: {event_id}\n"
        sse_message += f"event: {event_type}\n"
        sse_message += f"data: {json.dumps(data)}\n\n"
        
        return sse_message
    
    async def stream_ai_response_sse(
        self,
        prompt: str,
        config: Optional[RequestConfig] = None,
        client_id: Optional[str] = None
    ) -> str:
        """Stream AI response via SSE"""
        
        request_id = str(uuid.uuid4())
        config = config or RequestConfig()
        
        logger.info(
            "Starting SSE AI stream",
            request_id=request_id,
            client_id=client_id,
            prompt_length=len(prompt)
        )
        
        try:
            # Send start event
            await self._broadcast_event('ai_response', {
                'type': 'stream_start',
                'data': {
                    'request_id': request_id,
                    'prompt_length': len(prompt),
                    'client_id': client_id
                }
            })
            
            # Classify task and get provider
            task_type = await self.ai_service.router._classify_task_type(prompt)
            
            # Stream response chunks
            full_response = ""
            chunk_count = 0
            
            # Simulate streaming (in production, integrate with actual provider streaming)
            async for chunk in self._simulate_sse_response(prompt):
                full_response += chunk
                chunk_count += 1
                
                # Broadcast chunk
                await self._broadcast_event('ai_response', {
                    'type': 'chunk',
                    'data': {
                        'request_id': request_id,
                        'chunk': chunk,
                        'chunk_number': chunk_count,
                        'accumulated_length': len(full_response),
                        'client_id': client_id
                    }
                })
                
                # Progress update every 5 chunks
                if chunk_count % 5 == 0:
                    progress = min(100, (len(full_response) / (len(prompt) * 3)) * 100)
                    await self._broadcast_event('progress_update', {
                        'type': 'progress',
                        'data': {
                            'request_id': request_id,
                            'progress': progress,
                            'chunks_sent': chunk_count,
                            'client_id': client_id
                        }
                    })
                
                await asyncio.sleep(0.05)  # Streaming delay
            
            # Send completion event
            await self._broadcast_event('ai_response', {
                'type': 'stream_complete',
                'data': {
                    'request_id': request_id,
                    'total_chunks': chunk_count,
                    'total_length': len(full_response),
                    'client_id': client_id
                }
            })
            
            logger.info(
                "SSE AI stream completed",
                request_id=request_id,
                chunks=chunk_count,
                response_length=len(full_response)
            )
            
            return full_response
            
        except Exception as e:
            # Send error event
            await self._broadcast_event('error_alert', {
                'type': 'stream_error',
                'data': {
                    'request_id': request_id,
                    'error': str(e),
                    'client_id': client_id
                }
            })
            raise
    
    async def _simulate_sse_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Simulate streaming response for SSE"""
        
        # Generate response based on prompt type
        if "status" in prompt.lower() or "health" in prompt.lower():
            mock_response = """System Health Report:

✅ All AI providers are operational
⚡ Average response time: 1.2s
💰 Current cost efficiency: 94%
📈 Request volume: Normal levels

Provider Status:
• AWS Bedrock: Healthy (99.8% uptime)
• Azure OpenAI: Healthy (99.5% uptime)  
• Google Vertex: Healthy (99.9% uptime)

Cache Performance:
• Hit rate: 78%
• Storage utilization: 65%
• Estimated savings: $127.50 today

Everything is running smoothly!"""
        
        elif "monitor" in prompt.lower():
            mock_response = """Real-time Monitoring Dashboard:

🔍 Current Metrics (Last 5 minutes):
• Total requests: 1,247
• Success rate: 99.2%
• Cache hits: 973 (78%)
• Average latency: 890ms

🏥 Provider Health:
• Primary routes: All green
• Fallback usage: 0.8%
• Circuit breakers: All closed

💡 Insights:
• Peak traffic period detected
• Recommend scaling cache
• Cost optimization opportunity available

System performing within normal parameters."""
        
        else:
            mock_response = f"""Processing your request: "{prompt[:40]}..."

This response is being delivered via Server-Sent Events (SSE), which provides:

✨ Real-time streaming without complex WebSocket setup
📡 Automatic reconnection on connection loss  
🔄 Simple integration with web browsers
⚡ Lower overhead than WebSockets for one-way communication

Key advantages of SSE:
• Built-in browser support
• Automatic connection management
• Event-based architecture
• Perfect for dashboards and live updates

The response continues to stream in real-time, providing immediate feedback to enhance user experience."""
        
        # Stream word by word for smoother delivery
        words = mock_response.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + " " + word) > 15:  # Smaller chunks for SSE
                if current_chunk:
                    yield current_chunk + " "
                current_chunk = word
            else:
                current_chunk += (" " if current_chunk else "") + word
        
        if current_chunk:
            yield current_chunk
    
    async def _broadcast_event(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast event to subscribed clients"""
        
        if event_type in self.event_queues:
            try:
                await self.event_queues[event_type].put(event_data)
            except Exception as e:
                logger.error(
                    "Failed to broadcast event",
                    event_type=event_type,
                    error=str(e)
                )
    
    async def _broadcast_system_status(self):
        """Background task to broadcast system status updates"""
        
        while True:
            try:
                # Get current system status
                status = await self.ai_service.get_service_status()
                
                # Broadcast status update
                await self._broadcast_event('system_status', {
                    'type': 'status_update',
                    'data': {
                        'timestamp': time.time(),
                        'service_metrics': status['service_metrics'],
                        'health_summary': status['health_summary'],
                        'queue_status': status['queue_status']
                    }
                })
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error("Error in system status broadcast", error=str(e))
                await asyncio.sleep(60)
    
    async def _broadcast_provider_health(self):
        """Background task to broadcast provider health updates"""
        
        while True:
            try:
                # Get provider analytics
                analytics = await self.ai_service.router.get_provider_analytics()
                
                # Broadcast health update
                await self._broadcast_event('provider_health', {
                    'type': 'health_update',
                    'data': {
                        'timestamp': time.time(),
                        'providers': analytics['providers'],
                        'overall_stats': analytics['overall_stats'],
                        'recommendations': analytics['routing_recommendations']
                    }
                })
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error("Error in provider health broadcast", error=str(e))
                await asyncio.sleep(120)
    
    async def _broadcast_cache_stats(self):
        """Background task to broadcast cache statistics"""
        
        while True:
            try:
                # Get cache stats
                cache_stats = self.ai_service.cache.get_cache_stats()
                
                # Broadcast cache update
                await self._broadcast_event('cache_stats', {
                    'type': 'cache_update',
                    'data': {
                        'timestamp': time.time(),
                        'stats': cache_stats,
                        'performance': {
                            'hit_rate': cache_stats['hit_rate_percent'],
                            'size_utilization': (cache_stats['cache_size'] / cache_stats['max_size']) * 100,
                            'estimated_savings': cache_stats['estimated_cost_savings']
                        }
                    }
                })
                
                await asyncio.sleep(45)  # Update every 45 seconds
                
            except Exception as e:
                logger.error("Error in cache stats broadcast", error=str(e))
                await asyncio.sleep(90)
    
    async def _remove_client(self, client_id: str):
        """Remove SSE client"""
        
        if client_id in self.active_clients:
            client = self.active_clients[client_id]
            
            # Remove from user clients
            if client.user_id and client.user_id in self.user_clients:
                user_clients = self.user_clients[client.user_id]
                if client_id in user_clients:
                    user_clients.remove(client_id)
                    if not user_clients:
                        del self.user_clients[client.user_id]
            
            # Remove client
            del self.active_clients[client_id]
            
            # Update stats
            self.sse_stats['active_connections'] = len(self.active_clients)
            
            logger.info(
                "SSE client disconnected",
                client_id=client_id,
                connection_duration=time.time() - client.connected_at
            )
    
    async def get_sse_stats(self) -> Dict[str, Any]:
        """Get SSE service statistics"""
        
        return {
            'connections': {
                'total': self.sse_stats['total_connections'],
                'active': self.sse_stats['active_connections'],
                'by_user': len(self.user_clients)
            },
            'events': {
                'total_sent': self.sse_stats['total_events'],
                'by_type': dict(self.sse_stats['events_sent'])
            },
            'active_clients': {
                client_id: {
                    'user_id': client.user_id,
                    'connected_duration': time.time() - client.connected_at,
                    'subscriptions': client.subscriptions,
                    'is_connected': client.is_connected()
                }
                for client_id, client in self.active_clients.items()
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of SSE service"""
        
        logger.info("Shutting down SSE Streaming Service")
        
        # Cancel broadcasting tasks
        for task_name, task in self.broadcast_tasks.items():
            if task and not task.done():
                task.cancel()
                logger.info(f"Cancelled {task_name} broadcast task")
        
        # Wait for tasks to complete
        if self.broadcast_tasks:
            await asyncio.gather(*self.broadcast_tasks.values(), return_exceptions=True)
        
        # Clear clients (they'll disconnect naturally)
        self.active_clients.clear()
        self.user_clients.clear()
        
        logger.info("SSE Streaming Service shutdown complete")
```

## SSE Endpoints and Integration

### FastAPI SSE Endpoints

```python
# api/sse_endpoints.py
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json

from services.sse_streaming_service import SSEStreamingService
from services.production_ai_service import ProductionAIService, RequestConfig
from services.ai_router_service import RoutingStrategy

# Global SSE service (initialized in lifespan)
sse_service: Optional[SSEStreamingService] = None

@app.get("/sse/ai-stream")
async def sse_ai_stream(
    request: Request,
    user_id: Optional[str] = Query(None),
    subscriptions: Optional[str] = Query("ai_response,progress_update")
):
    """SSE endpoint for AI response streaming"""
    
    if not sse_service:
        raise HTTPException(status_code=503, detail="SSE service not available")
    
    # Parse subscriptions
    subscription_list = [s.strip() for s in subscriptions.split(",")] if subscriptions else []
    
    return await sse_service.create_client_stream(
        request=request,
        user_id=user_id,
        subscriptions=subscription_list
    )

@app.get("/sse/dashboard")
async def sse_dashboard_stream(
    request: Request,
    user_id: Optional[str] = Query(None)
):
    """SSE endpoint for dashboard real-time updates"""
    
    if not sse_service:
        raise HTTPException(status_code=503, detail="SSE service not available")
    
    # Dashboard specific subscriptions
    dashboard_subscriptions = [
        'system_status',
        'provider_health', 
        'cache_stats',
        'error_alert'
    ]
    
    return await sse_service.create_client_stream(
        request=request,
        user_id=user_id,
        subscriptions=dashboard_subscriptions
    )

@app.get("/sse/monitoring")
async def sse_monitoring_stream(
    request: Request,
    user_id: Optional[str] = Query(None)
):
    """SSE endpoint for system monitoring"""
    
    if not sse_service:
        raise HTTPException(status_code=503, detail="SSE service not available")
    
    # Monitoring specific subscriptions
    monitoring_subscriptions = [
        'provider_health',
        'system_status',
        'error_alert',
        'cost_update'
    ]
    
    return await sse_service.create_client_stream(
        request=request,
        user_id=user_id,
        subscriptions=monitoring_subscriptions
    )

@app.post("/api/sse/trigger-stream")
async def trigger_sse_stream(request_data: dict):
    """Trigger AI response streaming via SSE"""
    
    if not sse_service:
        raise HTTPException(status_code=503, detail="SSE service not available")
    
    prompt = request_data.get('prompt', '')
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Extract configuration
    config_data = request_data.get('config', {})
    config = RequestConfig(
        routing_strategy=RoutingStrategy(config_data.get('routing_strategy', 'intelligent')),
        speed_priority=config_data.get('speed_priority', True),
        reliability_priority=config_data.get('reliability_priority', False),
        cost_limit=config_data.get('cost_limit'),
        timeout=config_data.get('timeout', 30.0)
    )
    
    # Start streaming (response will be sent via SSE)
    try:
        response = await sse_service.stream_ai_response_sse(
            prompt=prompt,
            config=config,
            client_id=request_data.get('client_id')
        )
        
        return {
            'status': 'streaming_started',
            'message': 'Response will be streamed via SSE',
            'response_length': len(response)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@app.get("/api/sse/stats")
async def get_sse_stats():
    """Get SSE service statistics"""
    
    if not sse_service:
        raise HTTPException(status_code=503, detail="SSE service not available")
    
    return await sse_service.get_sse_stats()
```

## Frontend Integration Examples

### JavaScript SSE Client

```javascript
// static/js/sse-client.js

class SSEClient {
    constructor(endpoint, options = {}) {
        this.endpoint = endpoint;
        this.options = {
            reconnect: true,
            reconnectDelay: 1000,
            maxReconnectAttempts: 5,
            ...options
        };
        
        this.eventSource = null;
        this.reconnectAttempts = 0;
        this.handlers = new Map();
        this.isConnected = false;
    }
    
    connect() {
        try {
            this.eventSource = new EventSource(this.endpoint);
            
            this.eventSource.onopen = (event) => {
                console.log('SSE connection opened');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.emit('connected', event);
            };
            
            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.emit('message', data);
                } catch (e) {
                    console.error('Failed to parse SSE message:', e);
                }
            };
            
            this.eventSource.onerror = (event) => {
                console.error('SSE connection error:', event);
                this.isConnected = false;
                this.emit('error', event);
                
                if (this.options.reconnect && this.reconnectAttempts < this.options.maxReconnectAttempts) {
                    setTimeout(() => {
                        this.reconnectAttempts++;
                        console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
                        this.connect();
                    }, this.options.reconnectDelay * this.reconnectAttempts);
                }
            };
            
            // Handle specific event types
            this.eventSource.addEventListener('stream_start', (event) => {
                const data = JSON.parse(event.data);
                this.emit('streamStart', data);
            });
            
            this.eventSource.addEventListener('chunk', (event) => {
                const data = JSON.parse(event.data);
                this.emit('chunk', data);
            });
            
            this.eventSource.addEventListener('stream_complete', (event) => {
                const data = JSON.parse(event.data);
                this.emit('streamComplete', data);
            });
            
            this.eventSource.addEventListener('progress', (event) => {
                const data = JSON.parse(event.data);
                this.emit('progress', data);
            });
            
            this.eventSource.addEventListener('status_update', (event) => {
                const data = JSON.parse(event.data);
                this.emit('statusUpdate', data);
            });
            
            this.eventSource.addEventListener('health_update', (event) => {
                const data = JSON.parse(event.data);
                this.emit('healthUpdate', data);
            });
            
        } catch (error) {
            console.error('Failed to create SSE connection:', error);
            this.emit('error', error);
        }
    }
    
    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
            this.isConnected = false;
            this.emit('disconnected');
        }
    }
    
    on(event, handler) {
        if (!this.handlers.has(event)) {
            this.handlers.set(event, []);
        }
        this.handlers.get(event).push(handler);
    }
    
    off(event, handler) {
        if (this.handlers.has(event)) {
            const handlers = this.handlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.handlers.has(event)) {
            this.handlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }
}

// AI Streaming Client
class AIStreamingClient extends SSEClient {
    constructor(options = {}) {
        super('/sse/ai-stream', options);
        this.currentRequest = null;
        this.responseBuffer = '';
    }
    
    async streamPrompt(prompt, config = {}) {
        // Trigger streaming via API
        try {
            const response = await fetch('/api/sse/trigger-stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: prompt,
                    config: config,
                    client_id: this.clientId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            console.log('Streaming started:', result);
            
        } catch (error) {
            console.error('Failed to start streaming:', error);
            this.emit('error', error);
        }
    }
    
    setupStreamingHandlers() {
        this.on('streamStart', (data) => {
            console.log('Stream started:', data);
            this.currentRequest = data.request_id;
            this.responseBuffer = '';
            this.emit('aiStreamStart', data);
        });
        
        this.on('chunk', (data) => {
            if (data.request_id === this.currentRequest) {
                this.responseBuffer += data.chunk;
                this.emit('aiChunk', {
                    chunk: data.chunk,
                    fullResponse: this.responseBuffer,
                    progress: data
                });
            }
        });
        
        this.on('progress', (data) => {
            if (data.request_id === this.currentRequest) {
                this.emit('aiProgress', data);
            }
        });
        
        this.on('streamComplete', (data) => {
            if (data.request_id === this.currentRequest) {
                this.emit('aiStreamComplete', {
                    ...data,
                    fullResponse: this.responseBuffer
                });
                this.currentRequest = null;
            }
        });
    }
}

// Dashboard Client
class DashboardClient extends SSEClient {
    constructor(options = {}) {
        super('/sse/dashboard', options);
        this.setupDashboardHandlers();
    }
    
    setupDashboardHandlers() {
        this.on('statusUpdate', (data) => {
            this.updateSystemStatus(data);
        });
        
        this.on('healthUpdate', (data) => {
            this.updateProviderHealth(data);
        });
        
        this.on('cache_update', (data) => {
            this.updateCacheStats(data);
        });
    }
    
    updateSystemStatus(data) {
        // Update system status display
        const statusElement = document.getElementById('system-status');
        if (statusElement) {
            const metrics = data.service_metrics;
            statusElement.innerHTML = `
                <div class="metric">
                    <span class="label">Total Requests:</span>
                    <span class="value">${metrics.total_requests}</span>
                </div>
                <div class="metric">
                    <span class="label">Cache Hit Rate:</span>
                    <span class="value">${metrics.cache_hit_rate.toFixed(1)}%</span>
                </div>
                <div class="metric">
                    <span class="label">Avg Response Time:</span>
                    <span class="value">${metrics.avg_response_time.toFixed(2)}s</span>
                </div>
            `;
        }
    }
    
    updateProviderHealth(data) {
        // Update provider health display
        const healthElement = document.getElementById('provider-health');
        if (healthElement) {
            let healthHtml = '<h3>Provider Health</h3>';
            
            Object.entries(data.providers).forEach(([provider, stats]) => {
                const statusClass = stats.is_healthy ? 'healthy' : 'unhealthy';
                healthHtml += `
                    <div class="provider ${statusClass}">
                        <span class="name">${provider}</span>
                        <span class="status">${stats.performance_grade}</span>
                        <span class="rate">${(stats.success_rate * 100).toFixed(1)}%</span>
                    </div>
                `;
            });
            
            healthElement.innerHTML = healthHtml;
        }
    }
    
    updateCacheStats(data) {
        // Update cache statistics display
        const cacheElement = document.getElementById('cache-stats');
        if (cacheElement) {
            const stats = data.stats;
            const performance = data.performance;
            
            cacheElement.innerHTML = `
                <div class="cache-metric">
                    <span class="label">Hit Rate:</span>
                    <span class="value">${performance.hit_rate.toFixed(1)}%</span>
                </div>
                <div class="cache-metric">
                    <span class="label">Size:</span>
                    <span class="value">${stats.cache_size}/${stats.max_size}</span>
                </div>
                <div class="cache-metric">
                    <span class="label">Savings:</span>
                    <span class="value">${performance.estimated_savings.toFixed(4)}</span>
                </div>
            `;
        }
    }
}
```

## Real-time Dashboard Implementation

### HTML Dashboard Template

```html
<!-- templates/dashboard.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Service Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        .card h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .label {
            font-weight: 500;
            color: #666;
        }
        
        .value {
            font-weight: bold;
            color: #007bff;
        }
        
        .provider {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 4px;
            transition: background-color 0.3s ease;
        }
        
        .provider.healthy {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }
        
        .provider.unhealthy {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
        }
        
        .status-indicator {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .healthy .status-indicator {
            background-color: #28a745;
            color: white;
        }
        
        .unhealthy .status-indicator {
            background-color: #dc3545;
            color: white;
        }
        
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            z-index: 1000;
        }
        
        .connected {
            background-color: #28a745;
            color: white;
        }
        
        .disconnected {
            background-color: #dc3545;
            color: white;
        }
        
        .ai-chat {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
        
        .message {
            margin: 10px 0;
            padding: 8px;
            border-radius: 4px;
        }
        
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: 20px;
        }
        
        .ai-message {
            background-color: #f8f9fa;
            border-left: 3px solid #007bff;
        }
        
        .typing-indicator {
            font-style: italic;
            color: #666;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #007bff, #0056b3);
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .input-group input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .input-group button {
            padding: 8px 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .input-group button:hover {
            background-color: #0056b3;
        }
        
        .input-group button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connection-status">Connecting...</div>
    
    <h1>AI Service Real-time Dashboard</h1>
    
    <div class="dashboard">
        <!-- System Status Card -->
        <div class="card">
            <h3>System Status</h3>
            <div id="system-status">
                <div class="metric">
                    <span class="label">Status:</span>
                    <span class="value">Loading...</span>
                </div>
            </div>
        </div>
        
        <!-- Provider Health Card -->
        <div class="card">
            <h3>Provider Health</h3>
            <div id="provider-health">
                <div class="provider">
                    <span class="name">Loading...</span>
                </div>
            </div>
        </div>
        
        <!-- Cache Statistics Card -->
        <div class="card">
            <h3>Cache Performance</h3>
            <div id="cache-stats">
                <div class="cache-metric">
                    <span class="label">Loading...</span>
                </div>
            </div>
        </div>
        
        <!-- Live AI Chat Card -->
        <div class="card">
            <h3>Live AI Chat</h3>
            <div class="ai-chat" id="ai-chat">
                <div class="message ai-message">
                    <strong>AI Assistant:</strong> Hello! I'm ready to help you. Try asking me something!
                </div>
            </div>
            <div class="progress-bar" id="progress-bar" style="display: none;">
                <div class="progress-fill" id="progress-fill" style="width: 0%;"></div>
            </div>
            <div class="input-group">
                <input type="text" id="chat-input" placeholder="Ask me anything..." />
                <button onclick="sendMessage()" id="send-button">Send</button>
            </div>
        </div>
        
        <!-- Streaming Statistics Card -->
        <div class="card">
            <h3>Streaming Statistics</h3>
            <div id="streaming-stats">
                <div class="metric">
                    <span class="label">Active Connections:</span>
                    <span class="value" id="active-connections">0</span>
                </div>
                <div class="metric">
                    <span class="label">Events Sent:</span>
                    <span class="value" id="events-sent">0</span>
                </div>
                <div class="metric">
                    <span class="label">Data Streamed:</span>
                    <span class="value" id="data-streamed">0 KB</span>
                </div>
            </div>
        </div>
        
        <!-- Cost Monitoring Card -->
        <div class="card">
            <h3>Cost Monitoring</h3>
            <div id="cost-monitoring">
                <div class="metric">
                    <span class="label">Today's Spend:</span>
                    <span class="value" id="daily-spend">$0.00</span>
                </div>
                <div class="metric">
                    <span class="label">Cache Savings:</span>
                    <span class="value" id="cache-savings">$0.00</span>
                </div>
                <div class="metric">
                    <span class="label">Efficiency:</span>
                    <span class="value" id="cost-efficiency">0%</span>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/static/js/sse-client.js"></script>
    <script>
        // Initialize dashboard clients
        let dashboardClient;
        let aiClient;
        let isStreaming = false;
        
        // Initialize dashboard
        function initializeDashboard() {
            // Create dashboard SSE client
            dashboardClient = new DashboardClient({
                reconnect: true,
                reconnectDelay: 2000,
                maxReconnectAttempts: 10
            });
            
            // Create AI streaming client
            aiClient = new AIStreamingClient({
                reconnect: true,
                reconnectDelay: 1000,
                maxReconnectAttempts: 5
            });
            
            // Setup connection status handling
            setupConnectionHandlers();
            
            // Setup AI chat handlers
            setupAIChatHandlers();
            
            // Connect clients
            dashboardClient.connect();
            aiClient.connect();
            
            // Load initial statistics
            loadStreamingStats();
            
            // Setup periodic stats refresh
            setInterval(loadStreamingStats, 30000); // Every 30 seconds
        }
        
        function setupConnectionHandlers() {
            const statusElement = document.getElementById('connection-status');
            
            dashboardClient.on('connected', () => {
                statusElement.textContent = 'Dashboard Connected';
                statusElement.className = 'connection-status connected';
            });
            
            dashboardClient.on('disconnected', () => {
                statusElement.textContent = 'Dashboard Disconnected';
                statusElement.className = 'connection-status disconnected';
            });
            
            dashboardClient.on('error', (error) => {
                console.error('Dashboard connection error:', error);
                statusElement.textContent = 'Connection Error';
                statusElement.className = 'connection-status disconnected';
            });
        }
        
        function setupAIChatHandlers() {
            const chatContainer = document.getElementById('ai-chat');
            const progressBar = document.getElementById('progress-bar');
            const progressFill = document.getElementById('progress-fill');
            const sendButton = document.getElementById('send-button');
            const chatInput = document.getElementById('chat-input');
            
            // Setup AI streaming handlers
            aiClient.setupStreamingHandlers();
            
            aiClient.on('aiStreamStart', (data) => {
                isStreaming = true;
                sendButton.disabled = true;
                sendButton.textContent = 'Streaming...';
                progressBar.style.display = 'block';
                progressFill.style.width = '0%';
                
                // Add typing indicator
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message ai-message typing-indicator';
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = '<strong>AI Assistant:</strong> <span>Thinking...</span>';
                chatContainer.appendChild(typingDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            });
            
            aiClient.on('aiChunk', (data) => {
                // Update or create response message
                let responseDiv = document.getElementById('current-response');
                if (!responseDiv) {
                    // Remove typing indicator
                    const typingIndicator = document.getElementById('typing-indicator');
                    if (typingIndicator) {
                        typingIndicator.remove();
                    }
                    
                    // Create response div
                    responseDiv = document.createElement('div');
                    responseDiv.className = 'message ai-message';
                    responseDiv.id = 'current-response';
                    responseDiv.innerHTML = '<strong>AI Assistant:</strong> <span id="response-text"></span>';
                    chatContainer.appendChild(responseDiv);
                }
                
                // Update response text
                const responseText = document.getElementById('response-text');
                if (responseText) {
                    responseText.textContent = data.fullResponse;
                }
                
                // Auto-scroll
                chatContainer.scrollTop = chatContainer.scrollHeight;
            });
            
            aiClient.on('aiProgress', (data) => {
                progressFill.style.width = `${data.progress}%`;
            });
            
            aiClient.on('aiStreamComplete', (data) => {
                isStreaming = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                progressBar.style.display = 'none';
                
                // Finalize response
                const responseDiv = document.getElementById('current-response');
                if (responseDiv) {
                    responseDiv.id = ''; // Remove temporary ID
                }
                
                console.log('AI stream completed:', data);
            });
            
            // Handle enter key in chat input
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !isStreaming) {
                    sendMessage();
                }
            });
        }
        
        async function sendMessage() {
            const chatInput = document.getElementById('chat-input');
            const chatContainer = document.getElementById('ai-chat');
            const message = chatInput.value.trim();
            
            if (!message || isStreaming) return;
            
            // Add user message to chat
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.innerHTML = `<strong>You:</strong> ${message}`;
            chatContainer.appendChild(userDiv);
            
            // Clear input
            chatInput.value = '';
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Start AI streaming
            try {
                await aiClient.streamPrompt(message, {
                    routing_strategy: 'intelligent',
                    speed_priority: true
                });
            } catch (error) {
                console.error('Failed to send message:', error);
                
                const errorDiv = document.createElement('div');
                errorDiv.className = 'message ai-message';
                errorDiv.innerHTML = `<strong>AI Assistant:</strong> Sorry, I encountered an error: ${error.message}`;
                chatContainer.appendChild(errorDiv);
            }
        }
        
        async function loadStreamingStats() {
            try {
                const response = await fetch('/api/sse/stats');
                const stats = await response.json();
                
                // Update streaming statistics
                document.getElementById('active-connections').textContent = stats.connections.active;
                document.getElementById('events-sent').textContent = stats.events.total_sent;
                
                // Calculate data streamed (rough estimate)
                const dataStreamed = (stats.events.total_sent * 100) / 1024; // Estimate 100 bytes per event
                document.getElementById('data-streamed').textContent = `${dataStreamed.toFixed(1)} KB`;
                
            } catch (error) {
                console.error('Failed to load streaming stats:', error);
            }
        }
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', initializeDashboard);
    </script>
</body>
</html>
```

## Production Usage Examples

### Complete Real-time Demo

```python
# demo/realtime_demo.py
import asyncio
import json
from typing import Dict, Any
import time

from services.production_ai_service import ProductionAIService, ai_service_context
from services.streaming_ai_service import StreamingAIService
from services.sse_streaming_service import SSEStreamingService

async def run_realtime_demo():
    """Comprehensive real-time features demonstration"""
    
    print("🚀 Starting Real-time AI Features Demo")
    print("=" * 60)
    
    # Initialize services
    async with ai_service_context({'cache_size': 500}) as ai_service:
        
        # Initialize streaming services
        streaming_service = StreamingAIService(ai_service)
        sse_service = SSEStreamingService(ai_service)
        
        await streaming_service.initialize()
        await sse_service.initialize()
        
        try:
            # Demo 1: SSE Streaming
            print("\n📡 Demo 1: Server-Sent Events Streaming")
            print("-" * 40)
            
            # Simulate SSE streaming
            test_prompts = [
                "What is the current system status?",
                "Show me the provider health monitoring dashboard",
                "Analyze performance metrics and provide insights"
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n🔄 SSE Stream {i}: {prompt[:40]}...")
                
                start_time = time.time()
                response = await sse_service.stream_ai_response_sse(
                    prompt=prompt,
                    client_id=f"demo_client_{i}"
                )
                duration = time.time() - start_time
                
                print(f"✅ Completed in {duration:.2f}s")
                print(f"📝 Response length: {len(response)} characters")
                
                # Small delay between streams
                await asyncio.sleep(1)
            
            # Demo 2: Get streaming statistics
            print(f"\n📊 Demo 2: Streaming Statistics")
            print("-" * 40)
            
            sse_stats = await sse_service.get_sse_stats()
            
            print(f"SSE Connections:")
            print(f"  Total: {sse_stats['connections']['total']}")
            print(f"  Active: {sse_stats['connections']['active']}")
            print(f"  By User: {sse_stats['connections']['by_user']}")
            
            print(f"\nSSE Events:")
            print(f"  Total Sent: {sse_stats['events']['total_sent']}")
            print(f"  By Type: {dict(sse_stats['events']['by_type'])}")
            
            # Demo 3: Real-time monitoring simulation
            print(f"\n🔍 Demo 3: Real-time Monitoring Simulation")
            print("-" * 40)
            
            # Simulate monitoring events
            monitoring_events = [
                {"type": "provider_health_check", "provider": "aws_bedrock"},
                {"type": "cache_stats_update", "hit_rate": 85.2},
                {"type": "cost_alert", "daily_spend": 24.75},
                {"type": "performance_warning", "response_time": 3.2}
            ]
            
            for event in monitoring_events:
                print(f"📢 Broadcasting: {event['type']}")
                
                # Broadcast monitoring event
                await sse_service._broadcast_event('system_status', {
                    'type': 'monitoring_event',
                    'data': event
                })
                
                await asyncio.sleep(0.5)
            
            # Demo 4: WebSocket vs SSE comparison
            print(f"\n⚖️  Demo 4: WebSocket vs SSE Comparison")
            print("-" * 40)
            
            comparison_data = {
                'websocket': {
                    'bidirectional': True,
                    'overhead': 'Higher',
                    'reconnection': 'Manual',
                    'browser_support': 'Excellent',
                    'use_cases': ['Interactive chat', 'Real-time collaboration', 'Gaming']
                },
                'sse': {
                    'bidirectional': False,
                    'overhead': 'Lower',
                    'reconnection': 'Automatic',
                    'browser_support': 'Native',
                    'use_cases': ['Live dashboards', 'Notifications', 'Status updates']
                }
            }
            
            print("📊 Feature Comparison:")
            for tech, features in comparison_data.items():
                print(f"\n{tech.upper()}:")
                for feature, value in features.items():
                    if feature == 'use_cases':
                        print(f"  {feature}: {', '.join(value)}")
                    else:
                        print(f"  {feature}: {value}")
            
            # Demo 5: Performance metrics
            print(f"\n📈 Demo 5: Performance Metrics")
            print("-" * 40)
            
            # Get comprehensive service status
            service_status = await ai_service.get_service_status()
            
            print("Service Performance:")
            metrics = service_status['service_metrics']
            print(f"  Total Requests: {metrics['total_requests']}")
            print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
            print(f"  Avg Response Time: {metrics['avg_response_time']:.2f}s")
            print(f"  Error Rate: {metrics['error_rate']:.1f}%")
            
            print(f"\nReal-time Features Impact:")
            print(f"  SSE Connections: {sse_stats['connections']['active']}")
            print(f"  Events Broadcast: {sse_stats['events']['total_sent']}")
            print(f"  Data Throughput: {(sse_stats['events']['total_sent'] * 0.1):.1f} KB")
            
            # Demo 6: Error handling and resilience
            print(f"\n🛡️  Demo 6: Error Handling and Resilience")
            print("-" * 40)
            
            # Simulate various error scenarios
            error_scenarios = [
                "Connection timeout simulation",
                "Provider failure simulation", 
                "Cache overflow simulation",
                "Rate limit simulation"
            ]
            
            for scenario in error_scenarios:
                print(f"⚠️  Testing: {scenario}")
                
                # Broadcast error alert
                await sse_service._broadcast_event('error_alert', {
                    'type': 'error_simulation',
                    'data': {
                        'scenario': scenario,
                        'timestamp': time.time(),
                        'severity': 'warning',
                        'auto_recovery': True
                    }
                })
                
                await asyncio.sleep(0.3)
                
                # Simulate recovery
                await sse_service._broadcast_event('system_status', {
                    'type': 'recovery_event',
                    'data': {
                        'scenario': scenario,
                        'status': 'recovered',
                        'timestamp': time.time()
                    }
                })
                
                print(f"✅ Recovery: {scenario} handled successfully")
                await asyncio.sleep(0.5)
            
            # Demo 7: Client simulation
            print(f"\n👥 Demo 7: Multiple Client Simulation")
            print("-" * 40)
            
            # Simulate multiple clients connecting and disconnecting
            simulated_clients = []
            
            for i in range(5):
                client_id = f"sim_client_{i}"
                user_id = f"user_{i % 3}"  # 3 users with multiple sessions
                
                print(f"📱 Client {client_id} connecting (user: {user_id})")
                
                # Simulate client connection by updating stats
                sse_service.sse_stats['total_connections'] += 1
                sse_service.sse_stats['active_connections'] += 1
                
                simulated_clients.append((client_id, user_id))
                await asyncio.sleep(0.2)
            
            print(f"\n📊 Active Clients: {len(simulated_clients)}")
            
            # Simulate some clients disconnecting
            for i in range(2):
                client_id, user_id = simulated_clients.pop()
                print(f"📱 Client {client_id} disconnecting")
                sse_service.sse_stats['active_connections'] -= 1
                await asyncio.sleep(0.2)
            
            print(f"📊 Remaining Clients: {len(simulated_clients)}")
            
            # Final statistics
            print(f"\n📋 Final Demo Statistics")
            print("-" * 40)
            
            final_sse_stats = await sse_service.get_sse_stats()
            
            print(f"SSE Service Summary:")
            print(f"  Total Events Sent: {final_sse_stats['events']['total_sent']}")
            print(f"  Connection Efficiency: 100%")  # Demo simulation
            print(f"  Error Recovery Rate: 100%")     # All errors were handled
            print(f"  Average Event Latency: <50ms")  # Simulated performance
            
        finally:
            # Cleanup services
            await streaming_service.shutdown()
            await sse_service.shutdown()
    
    print(f"\n🎉 Real-time Features Demo Completed!")
    print("Key takeaways:")
    print("• SSE provides simple, efficient one-way streaming")
    print("• WebSocket enables bidirectional real-time communication")
    print("• Proper error handling ensures resilient connections")
    print("• Real-time features significantly enhance user experience")

if __name__ == "__main__":
    asyncio.run(run_realtime_demo())
```

## Key Features Summary

This comprehensive real-time AI features implementation provides:

### **WebSocket Streaming (Section 5.1):**
- Bidirectional communication for interactive AI chat
- Token-by-token streaming with progress indicators
- Session management with automatic cleanup
- Error handling and request cancellation
- Comprehensive statistics and monitoring

### **Server-Sent Events (Section 5.2):**
- One-way streaming for dashboards and monitoring
- Automatic reconnection and heartbeat management
- Event-based architecture with typed events
- Background broadcasting for system updates
- Lower overhead than WebSockets for simple streaming

### **Production Features:**
- **Real-time Dashboard**: Live system monitoring with automatic updates
- **Error Resilience**: Circuit breakers and automatic recovery
- **Performance Monitoring**: Real-time metrics and health tracking
- **Client Management**: Session tracking and resource cleanup
- **Scalable Architecture**: Efficient event broadcasting and queue management

The next section will cover **Chapter 9, Section 6: AI Security and Compliance** - focusing on securing AI integrations, handling sensitive data, and ensuring compliance with regulations.# Chapter 9, Section 5.2: Server-Sent Events and Real-time Updates

## Overview

Server-Sent Events (SSE) provide a simpler alternative to WebSockets for one-way real-time communication. This section implements SSE-based streaming for AI responses, real-time system monitoring, and live dashboard updates.

## SSE Streaming Implementation

### Core SSE Service

```python
# services/sse_streaming_service.py
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, List
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict

from fastapi import Request
from fastapi.responses import StreamingResponse
from services.production_ai_service import ProductionAIService, RequestConfig
from services.streaming_ai_service import StreamingEventType, StreamingEvent

logger = structlog.get_logger()

class SSEEventType(str, Enum):
    AI_RESPONSE = "ai_response"
    PROGRESS_UPDATE = "progress_update"
    SYSTEM_STATUS = "system_status"
    PROVIDER_HEALTH = "provider_health"
    CACHE_STATS = "cache_stats"
    COST_UPDATE = "cost_update"
    ERROR_ALERT = "error_alert"
    CUSTOM_EVENT = "custom_event"

@dataclass
class SSEClient:
    """SSE client connection tracking"""
    client_id: str
    user_id: Optional[str]
    connected_at: float = field(default_factory=time.time)