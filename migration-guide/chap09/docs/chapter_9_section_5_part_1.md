# Chapter 9, Section 5.1: WebSocket-Based Streaming AI Responses

## Overview

Real-time AI features are essential for modern applications, providing users with immediate feedback and creating engaging interactive experiences. This section implements streaming AI responses using WebSockets, allowing for real-time token-by-token streaming, progress updates, and interactive conversations.

## WebSocket Streaming Architecture

### Core Streaming Service

```python
# services/streaming_ai_service.py
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
from services.production_ai_service import ProductionAIService, RequestConfig
from services.ai_router_service import RoutingStrategy, TaskType

logger = structlog.get_logger()

class StreamingEventType(str, Enum):
    STREAM_START = "stream_start"
    TOKEN = "token"
    CHUNK = "chunk"
    PROGRESS = "progress"
    METADATA = "metadata"
    ERROR = "error"
    STREAM_END = "stream_end"
    CONNECTION_STATUS = "connection_status"
    TYPING_INDICATOR = "typing_indicator"

class ConnectionState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PROCESSING = "processing"
    IDLE = "idle"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"

@dataclass
class StreamingEvent:
    """Structured streaming event"""
    event_type: StreamingEventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'request_id': self.request_id,
            'session_id': self.session_id,
            'metadata': self.metadata
        }

@dataclass
class StreamingSession:
    """WebSocket session management"""
    session_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    state: ConnectionState = ConnectionState.CONNECTING
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    active_requests: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        self.last_activity = time.time()
    
    def get_session_duration(self) -> float:
        return time.time() - self.created_at

class StreamingAIService:
    """Real-time AI service with WebSocket streaming capabilities"""
    
    def __init__(self, ai_service: ProductionAIService):
        self.ai_service = ai_service
        
        # Active sessions management
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        
        # Streaming statistics
        self.streaming_stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_streams': 0,
            'active_streams': 0,
            'bytes_streamed': 0,
            'avg_stream_duration': 0.0
        }
        
        # Session cleanup task
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize streaming service"""
        logger.info("Initializing Streaming AI Service")
        
        # Start session cleanup task
        self._cleanup_task = asyncio.create_task(self._session_cleanup_task())
        
        logger.info("Streaming AI Service initialized")
    
    async def create_session(
        self, 
        websocket: WebSocket, 
        user_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> StreamingSession:
        """Create new streaming session"""
        
        session_id = str(uuid.uuid4())
        session = StreamingSession(
            session_id=session_id,
            websocket=websocket,
            user_id=user_id,
            preferences=preferences or {}
        )
        
        # Store session
        self.active_sessions[session_id] = session
        if user_id:
            self.user_sessions[user_id].append(session_id)
        
        # Update stats
        self.streaming_stats['total_sessions'] += 1
        self.streaming_stats['active_sessions'] = len(self.active_sessions)
        
        logger.info(
            "Session created",
            session_id=session_id,
            user_id=user_id,
            active_sessions=self.streaming_stats['active_sessions']
        )
        
        # Send connection confirmation
        await self._send_event(session, StreamingEvent(
            event_type=StreamingEventType.CONNECTION_STATUS,
            data={'status': 'connected', 'session_id': session_id},
            session_id=session_id
        ))
        
        session.state = ConnectionState.CONNECTED
        return session
    
    async def remove_session(self, session_id: str):
        """Remove streaming session"""
        
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Cancel any active requests
        for request_id in list(session.active_requests.keys()):
            await self._cancel_request(session, request_id)
        
        # Remove from user sessions
        if session.user_id:
            user_sessions = self.user_sessions.get(session.user_id, [])
            if session_id in user_sessions:
                user_sessions.remove(session_id)
                if not user_sessions:
                    del self.user_sessions[session.user_id]
        
        # Remove session
        del self.active_sessions[session_id]
        
        # Update stats
        self.streaming_stats['active_sessions'] = len(self.active_sessions)
        
        logger.info(
            "Session removed",
            session_id=session_id,
            duration=session.get_session_duration(),
            active_sessions=self.streaming_stats['active_sessions']
        )
    
    async def stream_ai_response(
        self,
        session: StreamingSession,
        prompt: str,
        config: Optional[RequestConfig] = None,
        stream_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Stream AI response in real-time"""
        
        request_id = str(uuid.uuid4())
        config = config or RequestConfig()
        stream_config = stream_config or {}
        
        session.update_activity()
        session.state = ConnectionState.PROCESSING
        
        # Store active request
        session.active_requests[request_id] = {
            'prompt': prompt,
            'config': config,
            'start_time': time.time(),
            'status': 'active'
        }
        
        # Update stats
        self.streaming_stats['total_streams'] += 1
        self.streaming_stats['active_streams'] += 1
        
        logger.info(
            "Starting AI stream",
            session_id=session.session_id,
            request_id=request_id,
            prompt_length=len(prompt)
        )
        
        try:
            # Send stream start event
            await self._send_event(session, StreamingEvent(
                event_type=StreamingEventType.STREAM_START,
                data={
                    'request_id': request_id,
                    'prompt_length': len(prompt),
                    'estimated_duration': self._estimate_stream_duration(prompt)
                },
                request_id=request_id,
                session_id=session.session_id
            ))
            
            # Show typing indicator
            await self._send_typing_indicator(session, True, request_id)
            
            # Classify task type for routing
            task_type = await self.ai_service.router._classify_task_type(prompt)
            
            # Get provider recommendation for streaming
            provider_recommendation = await self._get_streaming_provider(prompt, task_type, config)
            
            # Send metadata about routing decision
            await self._send_event(session, StreamingEvent(
                event_type=StreamingEventType.METADATA,
                data={
                    'task_type': task_type.value,
                    'recommended_provider': provider_recommendation,
                    'routing_strategy': config.routing_strategy.value
                },
                request_id=request_id,
                session_id=session.session_id
            ))
            
            # Stream the response
            full_response = ""
            chunk_count = 0
            
            # Simulate streaming response (in production, this would be actual provider streaming)
            async for chunk in self._simulate_streaming_response(prompt, provider_recommendation):
                if request_id not in session.active_requests:
                    # Request was cancelled
                    break
                
                full_response += chunk
                chunk_count += 1
                
                # Send chunk
                await self._send_event(session, StreamingEvent(
                    event_type=StreamingEventType.CHUNK,
                    data={
                        'chunk': chunk,
                        'chunk_number': chunk_count,
                        'accumulated_length': len(full_response)
                    },
                    request_id=request_id,
                    session_id=session.session_id
                ))
                
                # Send progress updates every 10 chunks
                if chunk_count % 10 == 0:
                    progress = min(100, (len(full_response) / self._estimate_response_length(prompt)) * 100)
                    await self._send_event(session, StreamingEvent(
                        event_type=StreamingEventType.PROGRESS,
                        data={'progress': progress, 'chunks_sent': chunk_count},
                        request_id=request_id,
                        session_id=session.session_id
                    ))
                
                # Update streaming stats
                self.streaming_stats['bytes_streamed'] += len(chunk.encode('utf-8'))
                
                # Small delay to simulate real streaming
                await asyncio.sleep(0.02)
            
            # Hide typing indicator
            await self._send_typing_indicator(session, False, request_id)
            
            # Calculate final metrics
            duration = time.time() - session.active_requests[request_id]['start_time']
            
            # Send stream end event
            await self._send_event(session, StreamingEvent(
                event_type=StreamingEventType.STREAM_END,
                data={
                    'request_id': request_id,
                    'total_chunks': chunk_count,
                    'total_length': len(full_response),
                    'duration': duration,
                    'provider_used': provider_recommendation,
                    'cache_hit': False  # Streaming typically bypasses cache
                },
                request_id=request_id,
                session_id=session.session_id
            ))
            
            # Clean up active request
            if request_id in session.active_requests:
                del session.active_requests[request_id]
            
            # Update stats
            self.streaming_stats['active_streams'] -= 1
            if self.streaming_stats['avg_stream_duration'] == 0:
                self.streaming_stats['avg_stream_duration'] = duration
            else:
                # Exponential moving average
                self.streaming_stats['avg_stream_duration'] = (
                    0.9 * self.streaming_stats['avg_stream_duration'] + 0.1 * duration
                )
            
            session.state = ConnectionState.IDLE
            
            logger.info(
                "AI stream completed",
                session_id=session.session_id,
                request_id=request_id,
                duration=duration,
                chunks=chunk_count,
                response_length=len(full_response)
            )
            
            return full_response
            
        except Exception as e:
            # Handle streaming errors
            await self._handle_streaming_error(session, request_id, e)
            raise
    
    async def _simulate_streaming_response(
        self, 
        prompt: str, 
        provider: str
    ) -> AsyncGenerator[str, None]:
        """Simulate streaming response from AI provider"""
        
        # In production, this would connect to actual provider streaming APIs
        # For demo purposes, we'll simulate token-by-token streaming
        
        # Generate a mock response based on prompt
        if "code" in prompt.lower():
            mock_response = """```python
def example_function(data):
    # Process the data
    result = []
    for item in data:
        if validate_item(item):
            processed = transform_item(item)
            result.append(processed)
    return result

def validate_item(item):
    return item is not None and len(str(item)) > 0

def transform_item(item):
    return str(item).upper()
```

This function demonstrates data processing with validation and transformation."""
        
        elif "analyze" in prompt.lower():
            mock_response = """Based on the analysis request, here are the key findings:

1. **Performance Metrics**: The system shows strong performance indicators with 95% uptime and average response times under 200ms.

2. **Usage Patterns**: Peak usage occurs between 9 AM and 5 PM EST, with a secondary peak around 8 PM.

3. **Key Insights**:
   - User engagement is highest on weekdays
   - Mobile traffic represents 60% of total usage
   - Feature adoption rate is accelerating

4. **Recommendations**:
   - Scale infrastructure during peak hours
   - Optimize mobile experience further
   - Implement progressive feature rollouts

This analysis provides actionable insights for optimization."""
        
        else:
            mock_response = f"""Thank you for your question about: "{prompt[:50]}..."

This is a comprehensive response that addresses your query. The response is being streamed in real-time to provide immediate feedback and create an engaging user experience.

Key points to consider:
• Real-time streaming improves user engagement
• Token-by-token delivery feels more natural
• Progress indicators help manage expectations
• Error handling ensures reliability

The streaming system supports various AI providers and can adapt to different response patterns and requirements."""
        
        # Split response into chunks and stream them
        words = mock_response.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > 20:  # Chunk size
                if current_chunk:
                    yield current_chunk + " "
                    current_chunk = word
                else:
                    yield word + " "
            else:
                current_chunk += (" " if current_chunk else "") + word
        
        # Send remaining chunk
        if current_chunk:
            yield current_chunk
    
    async def _get_streaming_provider(
        self, 
        prompt: str, 
        task_type: TaskType, 
        config: RequestConfig
    ) -> str:
        """Get best provider for streaming based on speed priority"""
        
        # For streaming, prioritize speed over cost
        streaming_config = RequestConfig(
            routing_strategy=RoutingStrategy.PERFORMANCE_BASED,
            speed_priority=True,
            reliability_priority=config.reliability_priority,
            cost_limit=config.cost_limit,
            timeout=config.timeout
        )
        
        # Get recommendation from router
        recommendation = await self.ai_service.router.recommend_provider(
            prompt=prompt,
            task_type=task_type,
            speed_priority=True,
            detailed_analysis=False
        )
        
        if recommendation['recommendations']:
            return recommendation['recommendations'][0]['provider']
        
        return 'aws_bedrock'  # Default fallback
    
    async def _send_event(self, session: StreamingSession, event: StreamingEvent):
        """Send event to WebSocket client"""
        
        try:
            await session.websocket.send_text(json.dumps(event.to_dict()))
            session.update_activity()
            
        except Exception as e:
            logger.error(
                "Failed to send WebSocket event",
                session_id=session.session_id,
                event_type=event.event_type.value,
                error=str(e)
            )
            # Mark session for removal
            session.state = ConnectionState.DISCONNECTED
    
    async def _send_typing_indicator(
        self, 
        session: StreamingSession, 
        is_typing: bool, 
        request_id: Optional[str] = None
    ):
        """Send typing indicator to client"""
        
        await self._send_event(session, StreamingEvent(
            event_type=StreamingEventType.TYPING_INDICATOR,
            data={'is_typing': is_typing},
            request_id=request_id,
            session_id=session.session_id
        ))
    
    async def _handle_streaming_error(
        self, 
        session: StreamingSession, 
        request_id: str, 
        error: Exception
    ):
        """Handle streaming errors gracefully"""
        
        logger.error(
            "Streaming error occurred",
            session_id=session.session_id,
            request_id=request_id,
            error=str(error)
        )
        
        # Send error event to client
        await self._send_event(session, StreamingEvent(
            event_type=StreamingEventType.ERROR,
            data={
                'error': str(error),
                'error_type': type(error).__name__,
                'request_id': request_id
            },
            request_id=request_id,
            session_id=session.session_id
        ))
        
        # Clean up request
        if request_id in session.active_requests:
            session.active_requests[request_id]['status'] = 'error'
        
        # Hide typing indicator
        await self._send_typing_indicator(session, False, request_id)
        
        # Update stats
        self.streaming_stats['active_streams'] -= 1
        session.state = ConnectionState.IDLE
    
    async def _cancel_request(self, session: StreamingSession, request_id: str):
        """Cancel active streaming request"""
        
        if request_id in session.active_requests:
            session.active_requests[request_id]['status'] = 'cancelled'
            
            # Send cancellation event
            await self._send_event(session, StreamingEvent(
                event_type=StreamingEventType.STREAM_END,
                data={
                    'request_id': request_id,
                    'status': 'cancelled',
                    'reason': 'Request cancelled by system'
                },
                request_id=request_id,
                session_id=session.session_id
            ))
            
            logger.info(
                "Request cancelled",
                session_id=session.session_id,
                request_id=request_id
            )
    
    def _estimate_stream_duration(self, prompt: str) -> float:
        """Estimate streaming duration based on prompt"""
        # Simple heuristic: 1 second per 50 characters
        return max(2.0, len(prompt) / 50.0)
    
    def _estimate_response_length(self, prompt: str) -> int:
        """Estimate response length for progress calculation"""
        # Simple heuristic: response is usually 2-5x prompt length
        return len(prompt) * 3
    
    async def _session_cleanup_task(self):
        """Background task to clean up inactive sessions"""
        
        while True:
            try:
                current_time = time.time()
                cleanup_threshold = 3600  # 1 hour of inactivity
                
                sessions_to_remove = []
                
                for session_id, session in self.active_sessions.items():
                    if (current_time - session.last_activity) > cleanup_threshold:
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    logger.info(
                        "Cleaning up inactive session",
                        session_id=session_id,
                        inactive_duration=current_time - self.active_sessions[session_id].last_activity
                    )
                    await self.remove_session(session_id)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error("Error in session cleanup task", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        
        return {
            'sessions': {
                'total_sessions': self.streaming_stats['total_sessions'],
                'active_sessions': self.streaming_stats['active_sessions'],
                'active_session_details': {
                    session_id: {
                        'user_id': session.user_id,
                        'state': session.state.value,
                        'duration': session.get_session_duration(),
                        'active_requests': len(session.active_requests)
                    }
                    for session_id, session in self.active_sessions.items()
                }
            },
            'streams': {
                'total_streams': self.streaming_stats['total_streams'],
                'active_streams': self.streaming_stats['active_streams'],
                'avg_stream_duration': self.streaming_stats['avg_stream_duration'],
                'bytes_streamed': self.streaming_stats['bytes_streamed']
            },
            'performance': {
                'avg_session_duration': sum(
                    session.get_session_duration() 
                    for session in self.active_sessions.values()
                ) / len(self.active_sessions) if self.active_sessions else 0,
                'sessions_per_user': len(self.user_sessions)
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of streaming service"""
        
        logger.info("Shutting down Streaming AI Service")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Close all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.remove_session(session_id)
        
        logger.info("Streaming AI Service shutdown complete")
```

## WebSocket Endpoint Implementation

### FastAPI WebSocket Integration

```python
# api/streaming_endpoints.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional
import json
import asyncio
import structlog

from services.streaming_ai_service import StreamingAIService, RequestConfig
from services.production_ai_service import ProductionAIService
from services.ai_router_service import RoutingStrategy

logger = structlog.get_logger()

# Global streaming service (initialized in lifespan)
streaming_service: Optional[StreamingAIService] = None

class WebSocketManager:
    """Enhanced WebSocket connection manager"""
    
    def __init__(self):
        self.connection_handlers = {}
    
    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        preferences: Optional[dict] = None
    ):
        """Handle new WebSocket connection"""
        
        await websocket.accept()
        
        try:
            # Create streaming session
            session = await streaming_service.create_session(
                websocket=websocket,
                user_id=user_id,
                preferences=preferences
            )
            
            logger.info(
                "WebSocket connection established",
                session_id=session.session_id,
                user_id=user_id
            )
            
            # Handle messages
            await self._handle_messages(session)
            
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected normally")
        except Exception as e:
            logger.error("WebSocket error", error=str(e))
        finally:
            if 'session' in locals():
                await streaming_service.remove_session(session.session_id)
    
    async def _handle_messages(self, session):
        """Handle incoming WebSocket messages"""
        
        while True:
            try:
                # Receive message from client
                message = await session.websocket.receive_text()
                data = json.loads(message)
                
                message_type = data.get('type')
                
                if message_type == 'ai_request':
                    await self._handle_ai_request(session, data)
                elif message_type == 'cancel_request':
                    await self._handle_cancel_request(session, data)
                elif message_type == 'update_preferences':
                    await self._handle_update_preferences(session, data)
                elif message_type == 'ping':
                    await self._handle_ping(session, data)
                else:
                    logger.warning(
                        "Unknown message type",
                        type=message_type,
                        session_id=session.session_id
                    )
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid JSON received",
                    session_id=session.session_id
                )
            except Exception as e:
                logger.error(
                    "Error handling message",
                    session_id=session.session_id,
                    error=str(e)
                )
    
    async def _handle_ai_request(self, session, data):
        """Handle AI streaming request"""
        
        prompt = data.get('prompt', '')
        if not prompt:
            return
        
        # Extract configuration
        config_data = data.get('config', {})
        config = RequestConfig(
            routing_strategy=RoutingStrategy(config_data.get('routing_strategy', 'intelligent')),
            speed_priority=config_data.get('speed_priority', True),  # Default to speed for streaming
            reliability_priority=config_data.get('reliability_priority', False),
            cost_limit=config_data.get('cost_limit'),
            timeout=config_data.get('timeout', 30.0)
        )
        
        # Start streaming response
        try:
            await streaming_service.stream_ai_response(
                session=session,
                prompt=prompt,
                config=config,
                stream_config=data.get('stream_config', {})
            )
        except Exception as e:
            logger.error(
                "AI streaming request failed",
                session_id=session.session_id,
                error=str(e)
            )
    
    async def _handle_cancel_request(self, session, data):
        """Handle request cancellation"""
        
        request_id = data.get('request_id')
        if request_id and request_id in session.active_requests:
            await streaming_service._cancel_request(session, request_id)
    
    async def _handle_update_preferences(self, session, data):
        """Handle preference updates"""
        
        new_preferences = data.get('preferences', {})
        session.preferences.update(new_preferences)
        
        logger.info(
            "Preferences updated",
            session_id=session.session_id,
            preferences=session.preferences
        )
    
    async def _handle_ping(self, session, data):
        """Handle ping/keepalive messages"""
        
        await session.websocket.send_text(json.dumps({
            'type': 'pong',
            'timestamp': data.get('timestamp'),
            'server_time': time.time()
        }))

# Global WebSocket manager
ws_manager = WebSocketManager()

# WebSocket endpoint
@app.websocket("/ws/ai-stream")
async def websocket_ai_stream(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None),
    model_preference: Optional[str] = Query(None),
    language: Optional[str] = Query("en")
):
    """WebSocket endpoint for AI streaming"""
    
    if not streaming_service:
        await websocket.close(code=1011, reason="Streaming service not available")
        return
    
    preferences = {
        'model_preference': model_preference,
        'language': language
    }
    
    await ws_manager.handle_connection(
        websocket=websocket,
        user_id=user_id,
        preferences=preferences
    )

# REST endpoints for streaming management
@app.get("/api/streaming/stats")
async def get_streaming_stats():
    """Get streaming service statistics"""
    
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    return await streaming_service.get_streaming_stats()

@app.get("/api/streaming/sessions")
async def get_active_sessions():
    """Get information about active streaming sessions"""
    
    if not streaming_service:
        raise HTTPException(status_code=503, detail="Streaming service not available")
    
    stats = await streaming_service.get_streaming_stats()
    return stats['sessions']
```

This section establishes the foundation for real-time AI streaming with WebSocket support. The next part will cover Server-Sent Events (SSE) for one-way streaming and advanced real-time features.