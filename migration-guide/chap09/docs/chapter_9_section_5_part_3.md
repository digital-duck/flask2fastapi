@dataclass
class BackgroundTask:
    """Background task tracking"""
    task_id: str
    user_id: Optional[str]
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration(self) -> Optional[float]:
        """Get task duration if completed"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    def is_active(self) -> bool:
        """Check if task is still active"""
        return self.status in [TaskStatus.PENDING, TaskStatus.STARTED, TaskStatus.PROCESSING, TaskStatus.STREAMING]

class BackgroundAIService:
    """Background task service for long-running AI operations"""
    
    def __init__(self, ai_service: ProductionAIService, sse_service: SSEStreamingService):
        self.ai_service = ai_service
        self.sse_service = sse_service
        self.redis_client = Redis(host='localhost', port=6379, db=1, decode_responses=True)
        
        # Task tracking
        self.active_tasks: Dict[str, BackgroundTask] = {}
        self.task_history: List[BackgroundTask] = []
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_duration': 0.0,
            'tasks_by_type': {},
            'tasks_by_priority': {}
        }
    
    async def submit_ai_task(
        self,
        prompt: str,
        user_id: Optional[str] = None,
        config: Optional[RequestConfig] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_type: str = "ai_processing",
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit AI processing task to background queue"""
        
        task_id = str(uuid.uuid4())
        
        # Create task record
        task = BackgroundTask(
            task_id=task_id,
            user_id=user_id,
            task_type=task_type,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store task
        self.active_tasks[task_id] = task
        await self._store_task_state(task)
        
        # Submit to Celery
        celery_task = process_ai_request.apply_async(
            args=[prompt, config.__dict__ if config else None],
            kwargs={
                'task_id': task_id,
                'user_id': user_id,
                'callback_url': callback_url,
                'metadata': metadata
            },
            task_id=task_id,
            priority=self._get_celery_priority(priority)
        )
        
        # Update stats
        self.stats['total_tasks'] += 1
        self.stats['tasks_by_type'][task_type] = self.stats['tasks_by_type'].get(task_type, 0) + 1
        self.stats['tasks_by_priority'][priority.value] = self.stats['tasks_by_priority'].get(priority.value, 0) + 1
        
        logger.info(
            "Background AI task submitted",
            task_id=task_id,
            task_type=task_type,
            priority=priority.value,
            user_id=user_id
        )
        
        return task_id
    
    async def submit_batch_task(
        self,
        prompts: List[str],
        user_id: Optional[str] = None,
        config: Optional[RequestConfig] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        batch_size: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit batch AI processing task"""
        
        task_id = str(uuid.uuid4())
        
        # Create batch task record
        task = BackgroundTask(
            task_id=task_id,
            user_id=user_id,
            task_type="batch_processing",
            priority=priority,
            metadata={
                'batch_size': len(prompts),
                'chunk_size': batch_size,
                **(metadata or {})
            }
        )
        
        self.active_tasks[task_id] = task
        await self._store_task_state(task)
        
        # Submit batch task to Celery
        celery_task = batch_ai_processing.apply_async(
            args=[prompts, config.__dict__ if config else None],
            kwargs={
                'task_id': task_id,
                'user_id': user_id,
                'batch_size': batch_size,
                'metadata': metadata
            },
            task_id=task_id,
            priority=self._get_celery_priority(priority)
        )
        
        logger.info(
            "Background batch task submitted",
            task_id=task_id,
            batch_size=len(prompts),
            chunk_size=batch_size,
            user_id=user_id
        )
        
        return task_id
    
    async def submit_streaming_task(
        self,
        prompt: str,
        user_id: Optional[str] = None,
        config: Optional[RequestConfig] = None,
        stream_to_sse: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit streaming AI task"""
        
        task_id = str(uuid.uuid4())
        
        task = BackgroundTask(
            task_id=task_id,
            user_id=user_id,
            task_type="streaming",
            priority=TaskPriority.HIGH,  # Streaming tasks get high priority
            metadata={
                'stream_to_sse': stream_to_sse,
                **(metadata or {})
            }
        )
        
        self.active_tasks[task_id] = task
        await self._store_task_state(task)
        
        # Submit streaming task
        celery_task = stream_ai_response.apply_async(
            args=[prompt, config.__dict__ if config else None],
            kwargs={
                'task_id': task_id,
                'user_id': user_id,
                'stream_to_sse': stream_to_sse,
                'metadata': metadata
            },
            task_id=task_id,
            priority=9  # High priority for streaming
        )
        
        logger.info(
            "Background streaming task submitted",
            task_id=task_id,
            stream_to_sse=stream_to_sse,
            user_id=user_id
        )
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """Get current task status"""
        
        # Check active tasks first
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            # Update with Celery status
            celery_result = AsyncResult(task_id, app=celery_app)
            if celery_result.state == 'PENDING':
                task.status = TaskStatus.PENDING
            elif celery_result.state == 'STARTED':
                task.status = TaskStatus.STARTED
                if not task.started_at:
                    task.started_at = time.time()
            elif celery_result.state == 'SUCCESS':
                task.status = TaskStatus.SUCCESS
                task.result = celery_result.result
                if not task.completed_at:
                    task.completed_at = time.time()
                    self.stats['completed_tasks'] += 1
            elif celery_result.state == 'FAILURE':
                task.status = TaskStatus.FAILURE
                task.error = str(celery_result.info)
                if not task.completed_at:
                    task.completed_at = time.time()
                    self.stats['failed_tasks'] += 1
            
            # Update progress if available
            if hasattr(celery_result, 'info') and isinstance(celery_result.info, dict):
                task.progress = celery_result.info.get('progress', task.progress)
            
            await self._store_task_state(task)
            return task
        
        # Check stored state
        return await self._load_task_state(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel background task"""
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            # Revoke Celery task
            celery_app.control.revoke(task_id, terminate=True)
            
            # Update task status
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            await self._store_task_state(task)
            
            logger.info("Background task cancelled", task_id=task_id)
            return True
        
        return False
    
    async def get_user_tasks(self, user_id: str, limit: int = 50) -> List[BackgroundTask]:
        """Get tasks for specific user"""
        
        user_tasks = []
        
        # Get active tasks
        for task in self.active_tasks.values():
            if task.user_id == user_id:
                await self.get_task_status(task.task_id)  # Update status
                user_tasks.append(task)
        
        # Get recent completed tasks from history
        for task in reversed(self.task_history[-100:]):  # Last 100 tasks
            if task.user_id == user_id and len(user_tasks) < limit:
                user_tasks.append(task)
        
        # Sort by creation time (newest first)
        user_tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return user_tasks[:limit]
    
    def _get_celery_priority(self, priority: TaskPriority) -> int:
        """Convert task priority to Celery priority (0-9)"""
        priority_map = {
            TaskPriority.LOW: 1,
            TaskPriority.NORMAL: 5,
            TaskPriority.HIGH: 8,
            TaskPriority.URGENT: 9
        }
        return priority_map.get(priority, 5)
    
    async def _store_task_state(self, task: BackgroundTask):
        """Store task state in Redis"""
        try:
            task_data = {
                'task_id': task.task_id,
                'user_id': task.user_id,
                'task_type': task.task_type,
                'status': task.status.value,
                'priority': task.priority.value,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'progress': task.progress,
                'result': json.dumps(task.result) if task.result else None,
                'error': task.error,
                'metadata': json.dumps(task.metadata)
            }
            
            # Store with TTL of 24 hours
            self.redis_client.hset(f"task:{task.task_id}", mapping=task_data)
            self.redis_client.expire(f"task:{task.task_id}", 86400)
            
        except Exception as e:
            logger.error("Failed to store task state", task_id=task.task_id, error=str(e))
    
    async def _load_task_state(self, task_id: str) -> Optional[BackgroundTask]:
        """Load task state from Redis"""
        try:
            task_data = self.redis_client.hgetall(f"task:{task_id}")
            
            if not task_data:
                return None
            
            return BackgroundTask(
                task_id=task_data['task_id'],
                user_id=task_data.get('user_id'),
                task_type=task_data['task_type'],
                status=TaskStatus(task_data['status']),
                priority=TaskPriority(task_data['priority']),
                created_at=float(task_data['created_at']),
                started_at=float(task_data['started_at']) if task_data.get('started_at') else None,
                completed_at=float(task_data['completed_at']) if task_data.get('completed_at') else None,
                progress=float(task_data.get('progress', 0)),
                result=json.loads(task_data['result']) if task_data.get('result') else None,
                error=task_data.get('error'),
                metadata=json.loads(task_data.get('metadata', '{}'))
            )
            
        except Exception as e:
            logger.error("Failed to load task state", task_id=task_id, error=str(e))
            return None
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        # Clean active tasks that are completed and old
        for task_id in list(self.active_tasks.keys()):
            task = self.active_tasks[task_id]
            
            if (not task.is_active() and 
                task.completed_at and 
                task.completed_at < cutoff_time):
                
                # Move to history
                self.task_history.append(task)
                del self.active_tasks[task_id]
                cleaned_count += 1
        
        # Limit history size
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]
        
        if cleaned_count > 0:
            logger.info("Cleaned up completed tasks", count=cleaned_count)
        
        return cleaned_count
    
    async def get_background_stats(self) -> Dict[str, Any]:
        """Get background task statistics"""
        
        # Count active tasks by status
        active_by_status = {}
        for task in self.active_tasks.values():
            status = task.status.value
            active_by_status[status] = active_by_status.get(status, 0) + 1
        
        # Calculate average duration
        completed_durations = [
            task.get_duration() for task in self.active_tasks.values()
            if task.get_duration() is not None
        ] + [
            task.get_duration() for task in self.task_history[-100:]
            if task.get_duration() is not None
        ]
        
        avg_duration = sum(completed_durations) / len(completed_durations) if completed_durations else 0
        
        return {
            'summary': {
                'total_tasks': self.stats['total_tasks'],
                'active_tasks': len(self.active_tasks),
                'completed_tasks': self.stats['completed_tasks'],
                'failed_tasks': self.stats['failed_tasks'],
                'success_rate': (self.stats['completed_tasks'] / max(1, self.stats['total_tasks'])) * 100
            },
            'active_tasks': {
                'by_status': active_by_status,
                'by_type': self.stats['tasks_by_type'],
                'by_priority': self.stats['tasks_by_priority']
            },
            'performance': {
                'avg_duration': avg_duration,
                'avg_duration_formatted': f"{avg_duration:.2f}s" if avg_duration > 0 else "N/A"
            },
            'queue_health': await self._get_queue_health()
        }
    
    async def _get_queue_health(self) -> Dict[str, Any]:
        """Get Celery queue health information"""
        
        try:
            # Get queue lengths (requires Celery management commands)
            inspect = celery_app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active() or {}
            
            # Get queue lengths (simplified)
            queue_info = {
                'ai_processing': len([t for worker_tasks in active_tasks.values() for t in worker_tasks if 'process_ai_request' in t.get('name', '')]),
                'ai_batch': len([t for worker_tasks in active_tasks.values() for t in worker_tasks if 'batch_ai_processing' in t.get('name', '')]),
                'ai_streaming': len([t for worker_tasks in active_tasks.values() for t in worker_tasks if 'stream_ai_response' in t.get('name', '')]),
                'ai_collaboration': len([t for worker_tasks in active_tasks.values() for t in worker_tasks if 'collaborative_ai_task' in t.get('name', '')])
            }
            
            return {
                'status': 'healthy',
                'workers_online': len(active_tasks),
                'queues': queue_info,
                'total_active': sum(queue_info.values())
            }
            
        except Exception as e:
            logger.error("Failed to get queue health", error=str(e))
            return {
                'status': 'unknown',
                'error': str(e)
            }

# Celery Tasks
@celery_app.task(bind=True, name='ai_background_tasks.process_ai_request')
def process_ai_request(
    self,
    prompt: str,
    config_dict: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Celery task for processing AI requests"""
    
    try:
        # Update task state
        self.update_state(
            state='STARTED',
            meta={'progress': 0, 'status': 'Initializing AI request'}
        )
        
        # Create AI service (in production, this would be dependency injected)
        # For demo purposes, we'll simulate the processing
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 25, 'status': 'Routing to AI provider'}
        )
        
        # Simulate AI processing time
        import time
        time.sleep(2)
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 75, 'status': 'Generating response'}
        )
        
        # Simulate response generation
        time.sleep(1)
        
        # Mock response
        response = {
            'response': f"AI response to: {prompt[:50]}...",
            'provider_used': 'aws_bedrock',
            'processing_time': 3.0,
            'tokens_used': len(prompt.split()) * 2,
            'task_id': task_id,
            'user_id': user_id
        }
        
        self.update_state(
            state='SUCCESS',
            meta={'progress': 100, 'status': 'Completed', 'result': response}
        )
        
        return response
        
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'progress': 0}
        )
        raise

@celery_app.task(bind=True, name='ai_background_tasks.batch_ai_processing')
def batch_ai_processing(
    self,
    prompts: List[str],
    config_dict: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    batch_size: int = 10,
    metadata: Optional[Dict[str, Any]] = None
):
    """Celery task for batch AI processing"""
    
    try:
        total_prompts = len(prompts)
        processed_count = 0
        results = []
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'status': f'Processing batch of {total_prompts} prompts'}
        )
        
        # Process in chunks
        for i in range(0, total_prompts, batch_size):
            chunk = prompts[i:i + batch_size]
            
            # Process chunk (simulate)
            import time
            time.sleep(1)  # Simulate processing time
            
            chunk_results = []
            for prompt in chunk:
                result = {
                    'prompt': prompt,
                    'response': f"Batch response to: {prompt[:30]}...",
                    'processed_at': time.time()
                }
                chunk_results.append(result)
                processed_count += 1
                
                # Update progress
                progress = (processed_count / total_prompts) * 100
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'progress': progress,
                        'status': f'Processed {processed_count}/{total_prompts} prompts',
                        'partial_results': len(results) + len(chunk_results)
                    }
                )
            
            results.extend(chunk_results)
        
        # Final result
        final_result = {
            'total_processed': processed_count,
            'results': results,
            'batch_stats': {
                'success_rate': 100,  # Simplified for demo
                'avg_processing_time': 1.0,
                'total_time': processed_count * 1.0 / batch_size
            },
            'task_id': task_id,
            'user_id': user_id
        }
        
        return final_result
        
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'progress': 0}
        )
        raise

@celery_app.task(bind=True, name='ai_background_tasks.stream_ai_response')
def stream_ai_response(
    self,
    prompt: str,
    config_dict: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    stream_to_sse: bool = True,
    metadata: Optional[Dict[str, Any]] = None
):
    """Celery task for streaming AI responses"""
    
    try:
        self.update_state(
            state='STREAMING',
            meta={'progress': 0, 'status': 'Starting stream'}
        )
        
        # Simulate streaming response
        chunks = []
        total_chunks = 20  # Simulate 20 chunks
        
        for i in range(total_chunks):
            chunk = f"Chunk {i+1} of streaming response to: {prompt[:20]}... "
            chunks.append(chunk)
            
            progress = ((i + 1) / total_chunks) * 100
            
            self.update_state(
                state='STREAMING',
                meta={
                    'progress': progress,
                    'status': f'Streaming chunk {i+1}/{total_chunks}',
                    'chunks_sent': i + 1,
                    'current_chunk': chunk
                }
            )
            
            # Simulate streaming delay
            import time
            time.sleep(0.1)
        
        # Complete streaming
        full_response = ''.join(chunks)
        
        result = {
            'response': full_response,
            'chunks_sent': len(chunks),
            'stream_duration': total_chunks * 0.1,
            'task_id': task_id,
            'user_id': user_id
        }
        
        return result
        
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'progress': 0}
        )
        raise
```

## Real-time Collaboration Features

### Collaborative AI Sessions

```python
# services/collaborative_ai_service.py
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import defaultdict

from fastapi import WebSocket
from services.production_ai_service import ProductionAIService
from services.streaming_ai_service import StreamingAIService, StreamingSession

logger = structlog.get_logger()

class CollaborationEventType(str, Enum):
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    MESSAGE_SENT = "message_sent"
    AI_RESPONSE = "ai_response"
    TYPING_START = "typing_start"
    TYPING_STOP = "typing_stop"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    DOCUMENT_EDIT = "document_edit"
    REACTION_ADDED = "reaction_added"
    SESSION_STATE_SYNC = "session_state_sync"

class UserRole(str, Enum):
    OWNER = "owner"
    MODERATOR = "moderator"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"

@dataclass
class CollaborativeUser:
    """User in collaborative session"""
    user_id: str
    username: str
    role: UserRole
    joined_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    websocket: Optional[WebSocket] = None
    cursor_position: Optional[Dict[str, Any]] = None
    is_typing: bool = False
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        self.last_activity = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'username': self.username,
            'role': self.role.value,
            'joined_at': self.joined_at,
            'last_activity': self.last_activity,
            'cursor_position': self.cursor_position,
            'is_typing': self.is_typing,
            'is_online': self.websocket is not None
        }

@dataclass
class CollaborativeMessage:
    """Message in collaborative session"""
    message_id: str
    user_id: str
    username: str
    content: str
    message_type: str = "user"  # user, ai, system
    timestamp: float = field(default_factory=time.time)
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji -> [user_ids]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_reaction(self, emoji: str, user_id: str):
        if emoji not in self.reactions:
            self.reactions[emoji] = []
        if user_id not in self.reactions[emoji]:
            self.reactions[emoji].append(user_id)
    
    def remove_reaction(self, emoji: str, user_id: str):
        if emoji in self.reactions and user_id in self.reactions[emoji]:
            self.reactions[emoji].remove(user_id)
            if not self.reactions[emoji]:
                del self.reactions[emoji]

@dataclass
class CollaborativeSession:
    """Collaborative AI session"""
    session_id: str
    title: str
    created_by: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    users: Dict[str, CollaborativeUser] = field(default_factory=dict)
    messages: List[CollaborativeMessage] = field(default_factory=list)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    permissions: Dict[str, Any] = field(default_factory=dict)
    ai_config: Dict[str, Any] = field(default_factory=dict)
    
    def add_user(self, user: CollaborativeUser):
        self.users[user.user_id] = user
        self.last_activity = time.time()
    
    def remove_user(self, user_id: str):
        if user_id in self.users:
            del self.users[user_id]
            self.last_activity = time.time()
    
    def add_message(self, message: CollaborativeMessage):
        self.messages.append(message)
        self.last_activity = time.time()
        
        # Keep only last 1000 messages
        if len(self.messages) > 1000:
            self.messages = self.messages[-1000:]
    
    def get_active_users(self) -> List[CollaborativeUser]:
        return [user for user in self.users.values() if user.websocket is not None]
    
    def can_user_edit(self, user_id: str) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        return user.role in [UserRole.OWNER, UserRole.MODERATOR, UserRole.COLLABORATOR]
    
    def can_user_moderate(self, user_id: str) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        return user.role in [UserRole.OWNER, UserRole.MODERATOR]

class CollaborativeAIService:
    """Real-time collaborative AI service"""
    
    def __init__(self, ai_service: ProductionAIService, streaming_service: StreamingAIService):
        self.ai_service = ai_service
        self.streaming_service = streaming_service
        
        # Active sessions
        self.sessions: Dict[str, CollaborativeSession] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        
        # Statistics
        self.stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_users': 0,
            'active_users': 0,
            'messages_sent': 0,
            'ai_interactions': 0
        }
    
    async def create_session(
        self,
        title: str,
        created_by: str,
        creator_username: str,
        ai_config: Optional[Dict[str, Any]] = None,
        permissions: Optional[Dict[str, Any]] = None
    ) -> CollaborativeSession:
        """Create new collaborative session"""
        
        session_id = str(uuid.uuid4())
        
        # Create owner user
        owner = CollaborativeUser(
            user_id=created_by,
            username=creator_username,
            role=UserRole.OWNER
        )
        
        # Create session
        session = CollaborativeSession(
            session_id=session_id,
            title=title,
            created_by=created_by,
            ai_config=ai_config or {},
            permissions=permissions or {}
        )
        
        session.add_user(owner)
        
        # Store session
        self.sessions[session_id] = session
        self.user_sessions[created_by].add(session_id)
        
        # Update stats
        self.stats['total_sessions'] += 1
        self.stats['active_sessions'] = len(self.sessions)
        
        logger.info(
            "Collaborative session created",
            session_id=session_id,
            title=title,
            created_by=created_by
        )
        
        return session
    
    async def join_session(
        self,
        session_id: str,
        user_id: str,
        username: str,
        websocket: WebSocket,
        role: UserRole = UserRole.COLLABORATOR
    ) -> bool:
        """Join collaborative session"""
        
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Create or update user
        if user_id in session.users:
            user = session.users[user_id]
            user.websocket = websocket
            user.update_activity()
        else:
            user = CollaborativeUser(
                user_id=user_id,
                username=username,
                role=role,
                websocket=websocket
            )
            session.add_user(user)
            self.user_sessions[user_id].add(session_id)
        
        # Broadcast user joined event
        await self._broadcast_to_session(session, {
            'type': CollaborationEventType.USER_JOINED.value,
            'user': user.to_dict(),
            'session_stats': {
                'active_users': len(session.get_active_users()),
                'total_users': len(session.users)
            }
        }, exclude_user=user_id)
        
        # Send session state to new user
        await self._send_to_user(session, user_id, {
            'type': CollaborationEventType.SESSION_STATE_SYNC.value,
            'session': {
                'session_id': session.session_id,
                'title': session.title,
                'users': [u.to_dict() for u in session.users.values()],
                'recent_messages': session.messages[-50:],  # Last 50 messages
                'shared_state': session.shared_state,
                'ai_config': session.ai_config
            }
        })
        
        # Update stats
        self.stats['active_users'] = sum(len(s.get_active_users()) for s in self.sessions.values())
        
        logger.info(
            "User joined collaborative session",
            session_id=session_id,
            user_id=user_id,
            username=username,
            active_users=len(session.get_active_users())
        )
        
        return True
    
    async def leave_session(self, session_id: str, user_id: str):
        """Leave collaborative session"""
        
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id in session.users:
            user = session.users[user_id]
            user.websocket = None  # Mark as offline but keep in session
            
            # Broadcast user left event
            await self._broadcast_to_session(session, {
                'type': CollaborationEventType.USER_LEFT.value,
                'user_id': user_id,
                'username': user.username,
                'session_stats': {
                    'active_users': len(session.get_active_users()),
                    'total_users': len(session.users)
                }
            }, exclude_user=user_id)
            
            # Update stats
            self.stats['active_users'] = sum(len(s.get_active_users()) for s in self.sessions.values())
            
            logger.info(
                "User left collaborative session",
                session_id=session_id,
                user_id=user_id,
                active_users=len(session.get_active_users())
            )
    
    async def send_message(
        self,
        session_id: str,
        user_id: str,
        content: str,
        message_type: str = "user"
    ) -> Optional[str]:
        """Send message to collaborative session"""
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if user_id not in session.users:
            return None
        
        user = session.users[user_id]
        message_id = str(uuid.uuid4())
        
        # Create message
        message = CollaborativeMessage(
            message_id=message_id,
            user_id=user_id,
            username=user.username,
            content=content,
            message_type=message_type
        )
        
        session.add_message(message)
        
        # Broadcast message to all users
        await self._broadcast_to_session(session, {
            'type': CollaborationEventType.MESSAGE_SENT.value,
            'message': {
                'message_id': message.message_id,
                'user_id': message.user_id,
                'username': message.username,
                'content': message.content,
                'message_type': message.message_type,
                'timestamp': message.timestamp,
                'reactions': message.reactions
            }
        })
        
        # Update stats
        self.stats['messages_sent'] += 1
        
        logger.info(
            "Message sent in collaborative session",
            session_id=session_id,
            user_id=user_id,
            message_id=message_id,
            content_length=len(content)
        )
        
        return message_id
    
    async def send_ai_request(
        self,
        session_id: str,
        user_id: str,
        prompt: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Send AI request in collaborative session"""
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if user_id not in session.users or not session.can_user_edit(user_id):
            return None
        
        # Send user message first
        await self.send_message(session_id, user_id, prompt, "user")
        
        # Broadcast typing indicator for AI
        await self._broadcast_to_session(session, {
            'type': CollaborationEventType.TYPING_START.value,
            'user_id': 'ai_assistant',
            'username': 'AI Assistant'
        })
        
        try:
            # Process AI request
            ai_config = {**session.ai_config, **(config or {})}
            response = await self.ai_service.process_request(
                prompt=prompt,
                config=self.ai_service.RequestConfig(**ai_config) if ai_config else None,
                user_id=user_id,
                metadata={'session_id': session_id, 'collaborative': True}
            )
            
            # Send AI response message
            ai_message_id = await self.send_message(
                session_id, 
                'ai_assistant', 
                response.get('response', 'Sorry, I encountered an error.'),
                "ai"
            )
            
            # Broadcast AI response event with metadata
            await self._broadcast_to_session(session, {
                'type': CollaborationEventType.AI_RESPONSE.value,
                'message_id': ai_message_id,
                'response_metadata': {
                    'provider_used': response.get('provider_used'),
                    'response_time': response.get('response_time'),
                    'cost_estimate': response.get('cost_estimate'),
                    'cache_hit': response.get('cache_hit')
                }
            })
            
            # Update stats
            self.stats['ai_interactions'] += 1
            
            logger.info(
                "AI request processed in collaborative session",
                session_id=session_id,
                user_id=user_id,
                provider=response.get('provider_used'),
                response_time=response.get('response_time')
            )
            
            return ai_message_id
            
        except Exception as e:
            # Send error message
            await self.send_message(
                session_id,
                'ai_assistant',
                f"I apologize, but I encountered an error: {str(e)}",
                "ai"
            )
            
            logger.error(
                "AI request failed in collaborative session",
                session_id=session_id,
                user_id=user_id,
                error=str(e)
            )
            
        finally:
            # Stop AI typing indicator
            await self._broadcast_to_session(session, {
                'type': CollaborationEventType.TYPING_STOP.value,
                'user_id': 'ai_assistant',
                'username': 'AI Assistant'
            })
        
        return None
    
    async def update_typing_status(self, session_id: str, user_id: str, is_typing: bool):
        """Update user typing status"""
        
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id in session.users:
            user = session.users[user_id]
            user.is_typing = is_typing
            user.update_activity()
            
            # Broadcast typing status
            event_type = CollaborationEventType.TYPING_START if is_typing else CollaborationEventType.TYPING_STOP
            await self._broadcast_to_session(session, {
                'type': event_type.value,
                'user_id': user_id,
                'username': user.username
            }, exclude_user=user_id)
    
    async def update_cursor_position(
        self,
        session_id: str,
        user_id: str,
        position: Dict[str, Any]
    ):
        """Update user cursor position"""
        
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        if user_id in session.users:
            user = session.users[user_id]
            user.cursor_position = position
            user.update_activity()
            
            # Broadcast cursor move
            await self._broadcast_to_session(session, {
                'type': CollaborationEventType.CURSOR_MOVE.value,
                'user_id': user_id,
                'username': user.username,
                'position': position
            }, exclude_user=user_id)
    
    async def add_reaction(
        self,
        session_id: str,
        user_id: str,
        message_id: str,
        emoji: str
    ) -> bool:
        """Add reaction to message"""
        
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Find message
        message = None
        for msg in session.messages:
            if msg.message_id == message_id:
                message = msg
                break
        
        if not message:
            return False
        
        # Add reaction
        message.add_reaction(emoji, user_id)
        
        # Broadcast reaction
        await self._broadcast_to_session(session, {
            'type': CollaborationEventType.REACTION_ADDED.value,
            'message_id': message_id,
            'emoji': emoji,
            'user_id': user_id,
            'reactions': message.reactions
        })
        
        return True
    
    async def _broadcast_to_session(
        self,
        session: CollaborativeSession,
        event: Dict[str, Any],
        exclude_user: Optional[str] = None
    ):
        """Broadcast event to all active users in session"""
        
        event_json = json.dumps(event)
        
        for user in session.get_active_users():
            if exclude_user and user.user_id == exclude_user:
                continue
            
            try:
                await user.websocket.send_text(event_json)
                user.update_activity()
            except Exception as e:
                logger.error(
                    "Failed to send event to user",
                    session_id=session.session_id,
                    user_id=user.user_id,
                    error=str(e)
                )
                # Mark user as disconnected
                user.websocket = None
    
    async def _send_to_user(
        self,
        session: CollaborativeSession,
        user_id: str,
        event: Dict[str, Any]
    ):
        """Send event to specific user"""
        
        if user_id in session.users:
            user = session.users[user_id]
            if user.websocket:
                try:
                    await user.websocket.send_text(json.dumps(event))
                    user.update_activity()
                except Exception as e:
                    logger.error(
                        "Failed to send event to user",
                        session_id=session.session_id,
                        user_id=user_id,
                        error=str(e)
                    )
                    user.websocket = None
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        return {
            'session_id': session.session_id,
            'title': session.title,
            'created_by': session.created_by,
            'created_at': session.created_at,
            'last_activity': session.last_activity,
            'users': [user.to_dict() for user in session.users.values()],
            'active_users': len(session.get_active_users()),
            'total_messages': len(session.messages),
            'shared_state': session.shared_state,
            'ai_config': session.ai_config
        }
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get sessions for user"""
        
        user_session_ids = self.user_sessions.get(user_id, set())
        sessions_info = []
        
        for session_id in user_session_ids:
            if session_id in self.sessions:
                session_info = await self.get_session_info(session_id)
                if session_info:
                    sessions_info.append(session_info)
        
        return sessions_info
    
    async def cleanup_inactive_sessions(self, max_idle_hours: int = 24):
        """Clean up inactive sessions"""
        
        cutoff_time = time.time() - (max_idle_hours * 3600)
        cleaned_sessions = []
        
        for session_id, session in list(self.sessions.items()):
            # Check if session has been idle too long
            if (session.last_activity < cutoff_time and 
                len(session.get_active_users()) == 0):
                
                # Remove session
                del self.sessions[session_id]
                
                # Remove from user sessions
                for user_id in session.users.keys():
                    if user_id in self.user_sessions:
                        self.user_sessions[user_id].discard(session_id)
                
                cleaned_sessions.append(session_id)
        
        # Update stats
        self.stats['active_sessions'] = len(self.sessions)
        
        if cleaned_sessions:
            logger.info(
                "Cleaned up inactive collaborative sessions",
                count=len(cleaned_sessions),
                session_ids=cleaned_sessions[:5]  # Log first 5
            )
        
        return len(cleaned_sessions)
    
    async def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration service statistics"""
        
        # Calculate real-time stats
        active_users = sum(len(s.get_active_users()) for s in self.sessions.values())
        total_messages = sum(len(s.messages) for s in self.sessions.values())
        
        session_sizes = [len(s.get_active_users()) for s in self.sessions.values()]
        avg_session_size = sum(session_sizes) / len(session_sizes) if session_sizes else 0
        
        return {
            'sessions': {
                'total': self.stats['total_sessions'],
                'active': len(self.sessions),
                'avg_size': avg_session_size
            },
            'users': {
                'total_registered': self.stats['total_users'],
                'currently_active': active_users,
                'unique_users_today': len(self.user_sessions)
            },
            'activity': {
                'messages_sent': self.stats['messages_sent'],
                'ai_interactions': self.stats['ai_interactions'],
                'total_messages_all_sessions': total_messages
            },
            'engagement': {
                'avg_messages_per_session': total_messages / len(self.sessions) if self.sessions else 0,
                'ai_interaction_rate': (self.stats['ai_interactions'] / max(1, self.stats['messages_sent'])) * 100
            }
        }
```

## Complete Integration Example

### Production Demo with All Features

```python
# demo/complete_realtime_demo.py
import asyncio
import json
import time
from typing import Dict, Any

from services.production_ai_service import ProductionAIService, ai_service_context
from services.streaming_ai_service import StreamingAIService
from services.sse_streaming_service import SSEStreamingService
from services.background_ai_service import BackgroundAIService, TaskPriority
from services.collaborative_ai_service import CollaborativeAIService, UserRole

async def run_complete_realtime_demo():
    """Complete demonstration of all real-time AI features"""
    
    print("🚀 Complete Real-time AI Features Demo")
    print("=" * 70)
    
    # Initialize all services
    async with ai_service_context({'cache_size': 1000}) as ai_service:
        
        # Initialize real-time services
        streaming_service = StreamingAIService(ai_service)
        sse_service = SSEStreamingService(ai_service)
        background_service = BackgroundAIService(ai_service, sse_service)
        collaborative_service = CollaborativeAIService(ai_service, streaming_service)
        
        await streaming_service.initialize()
        await sse_service.initialize()
        
        try:
            # Demo 1: Background Task Processing
            print("\n⚙️  Demo 1: Background Task Processing")
            print("-" * 50)
            
            # Submit various background tasks
            task_configs = [
                ("Analyze quarterly sales data with detailed insights", TaskPriority.HIGH, "analysis"),
                ("Generate code documentation for Python project", TaskPriority.NORMAL, "documentation"),
                ("Create marketing copy for new product launch", TaskPriority.LOW, "content_creation")
            ]
            
            submitted_tasks = []
            
            for prompt, priority, task_type in task_configs:
                task_id = await background_service.submit_ai_task(
                    prompt=prompt,
                    user_id="demo_user",
                    priority=priority,
                    task_type=task_type,
                    metadata={'demo': True, 'batch': 'demo_1'}
                )
                submitted_tasks.append((task_id, prompt[:30]))
                print(f"📋 Submitted {priority.value} priority task: {prompt[:40]}...")
            
            # Submit batch task
            batch_prompts = [
                "Summarize this document",
                "Extract key insights",
                "Generate action items",
                "Create executive summary"
            ]
            
            batch_task_id = await background_service.submit_batch_task(
                prompts=batch_prompts,
                user_id="demo_user",
                priority=TaskPriority.NORMAL,
                batch_size=2,
                metadata={'demo_batch': True}
            )
            
            print(f"📦 Submitted batch task with {len(batch_prompts)} prompts")
            
            # Monitor task progress
            print(f"\n📊 Monitoring task progress...")
            for i in range(10):  # Monitor for 10 seconds
                for task_id, description in submitted_tasks:
                    task_status = await background_service.get_task_status(task_id)
                    if task_status:
                        print(f"  {description}: {task_status.status.value} ({task_status.progress:.0f}%)")
                
                # Check batch task
                batch_status = await background_service.get_task_status(batch_task_id)
                if batch_status:
                    print(f"  Batch task: {batch_status.status.value} ({batch_status.progress:.0f}%)")
                
                await asyncio.sleep(1)
                print()  # Clear line for next update
            
            # Demo 2: Collaborative AI Session
            print("\n👥 Demo 2: Collaborative AI Session")
            print("-" * 50)
            
            # Create collaborative session
            collab_session = await collaborative_service.create_session(
                title="AI Strategy Planning Session",
                created_by="user1",
                creator_username="Alice",
                ai_config={'routing_strategy': 'intelligent', 'speed_priority': True}
            )
            
            print(f"🏢 Created collaborative session: {collab_session.title}")
            
            # Simulate multiple users joining
            users = [
                ("user2", "Bob", UserRole.COLLABORATOR),
                ("user3", "Charlie", UserRole.VIEWER),
                ("user4", "Diana", UserRole.MODERATOR)
            ]
            
            # Note: In real implementation, these would be actual WebSocket connections
            # For demo, we'll simulate the join process
            for user_id, username, role in users:
                print(f"👤 {username} ({role.value}) joining session...")
                # In real implementation: await collaborative_service.join_session(...)
                
                # Simulate sending messages
                await collaborative_service.send_message(
                    session_id=collab_session.session_id,
                    user_id=user_id,
                    content=f"Hello everyone! I'm {username} and ready to collaborate."
                )
            
            # Simulate AI interactions
            ai_prompts = [
                "What are the key trends in AI adoption for 2025?",
                "How can we improve our AI strategy based on current market conditions?",
                "What are the potential risks and mitigation strategies for our AI implementation?"
            ]
            
            for prompt in ai_prompts:
                print(f"🤖 Processing AI request: {prompt[:50]}...")
                message_id = await collaborative_service.send_ai_request(
                    session_id=collab_session.session_id,
                    user_id="user1",  # Alice (owner) sends AI requests
                    prompt=prompt
                )
                
                if message_id:
                    print(f"✅ AI response generated (Message ID: {message_id[:8]})")
                    
                    # Simulate reactions
                    await collaborative_service.add_reaction(
                        session_id=collab_session.session_id,
                        user_id="user2",
                        message_id=message_id,
                        emoji="👍"
                    )
                
                await asyncio.sleep(1)
            
            # Demo 3: SSE Dashboard Updates
            print(f"\n📡 Demo 3: SSE Dashboard Updates")
            print("-" * 50)
            
            # Simulate dashboard events
            dashboard_events = [
                {"metric": "active_sessions", "value": len(collaborative_service.sessions)},
                {"metric": "background_tasks", "value": len(background_service.active_tasks)},
                {"metric": "cache_hit_rate", "value": 78.5},
                {"metric": "avg_response_time", "value": 1.2}
            ]
            
            for event in dashboard_events:
                print(f"📊 Broadcasting {event['metric']}: {event['value']}")
                await sse_service._broadcast_event('system_status', {
                    'type': 'metric_update',
                    'data': event
                })
                await asyncio.sleep(0.5)
            
            # Demo 4: Real-time Monitoring
            print(f"\n🔍 Demo 4: Real-time Monitoring")
            print("-" * 50)
            
            # Get comprehensive statistics
            bg_stats = await background_service.get_background_stats()
            collab_stats = await collaborative_service.get_collaboration_stats()
            sse_stats = await sse_service.get_sse_stats()
            
            print(f"Background Tasks:")
            print(f"  Total: {bg_stats['summary']['total_tasks']}")
            print(f"  Active: {bg_stats['summary']['active_tasks']}")
            print(f"  Success Rate: {bg_stats['summary']['success_rate']:.1f}%")
            
            print(f"\nCollaborative Sessions:")
            print(f"  Active Sessions: {collab_stats['sessions']['active']}")
            print(f"  Active Users: {collab_stats['users']['currently_active']}")
            print(f"  Messages Sent: {collab_stats['activity']['messages_sent']}")
            print(f"  AI Interactions: {collab_stats['activity']['ai_interactions']}")
            
            print(f"\nSSE Streaming:")
            print(f"  Active Connections: {sse_stats['connections']['active']}")
            print(f"  Events Sent: {sse_stats['events']['total_sent']}")
            
            # Demo 5: Error Handling and Recovery
            print(f"\n🛡️  Demo 5: Error Handling and Recovery")
            print("-" * 50)
            
            # Simulate various error scenarios
            error_scenarios = [
                "Provider timeout simulation",
                "Rate limit exceeded simulation",
                "WebSocket connection failure",
                "Background task failure"
            ]
            
            for scenario in error_scenarios:
                print(f"⚠️  Simulating: {scenario}")
                
                # Broadcast error alert
                await sse_service._broadcast_event('error_alert', {
                    'type': 'error_simulation',
                    'data': {
                        'scenario': scenario,
                        'severity': 'warning',
                        'timestamp': time.time(),
                        'auto_recovery': True
                    }
                })
                
                await asyncio.sleep(0.5)
                
                # Simulate recovery
                await sse_service._broadcast_event('system_status', {
                    'type': 'recovery_complete',
                    'data': {
                        'scenario': scenario,
                        'recovery_time': '0.5s',
                        'status': 'healthy'
                    }
                })
                
                print(f"✅ Recovery: {scenario} resolved")
                await asyncio.sleep(0.3)
            
            # Demo 6: Performance Metrics
            print(f"\n📈 Demo 6: Performance Metrics Summary")
            print("-" * 50)
            
            # Calculate performance metrics
            total_operations = (
                bg_stats['summary']['total_tasks'] +
                collab_stats['activity']['messages_sent'] +
                sse_stats['events']['total_sent']
            )
            
            system_health = {
                'total_operations': total_operations,
                'background_task_success_rate': bg_stats['summary']['success_rate'],
                'collaboration_engagement': collab_stats['engagement']['ai_interaction_rate'],
                'streaming_efficiency': 100,  # Simulated
                'overall_system_health': 95.5  # Composite score
            }
            
            print(f"System Performance Summary:")
            for metric, value in system_health.items():
                if isinstance(value, float):
                    print(f"  {metric.replace('_', ' ').title()}: {value:.1f}%")
                else:
                    print(f"  {metric.replace('_', ' ').title()}: {value}")
            
            # Demo 7: Integration Benefits
            print(f"\n🎯 Demo 7: Integration Benefits")
            print("-" * 50)
            
            benefits = {
                'Real-time Collaboration': [
                    'Multiple users can interact with AI simultaneously',
                    'Shared context and conversation history',
                    'Live cursor tracking and typing indicators',
                    'Instant reactions and feedback'
                ],
                'Background Processing': [
                    'Long-running tasks don\'t block user interface',
                    'Priority-based task scheduling',
                    'Batch processing for efficiency',
                    'Progress tracking and monitoring'
                ],
                'SSE Streaming': [
                    'Real-time dashboard updates',
                    'Live system monitoring',
                    'Automatic reconnection handling',
                    'Low-latency event broadcasting'
                ],
                'Error Resilience': [
                    'Automatic failover between providers',
                    'Circuit breaker patterns',
                    'Graceful degradation',
                    'Real-time health monitoring'
                ]
            }
            
            for feature, benefit_list in benefits.items():
                print(f"\n{feature}:")
                for benefit in benefit_list:
                    print(f"  • {benefit}")
            
        finally:
            # Cleanup all services
            print(f"\n🧹 Cleaning up services...")
            
            await streaming_service.shutdown()
            await sse_service.shutdown()
            
            # Clean up collaborative sessions
            await collaborative_service.cleanup_inactive_sessions(max_idle_hours=0)
            
            # Clean up background tasks
            await background_service.cleanup_completed_tasks(max_age_hours=0)
    
    print(f"\n🎉 Complete Real-time Features Demo Finished!")
    print("\nKey Achievements:")
    print("✅ Background task processing with priority queues")
    print("✅ Real-time collaborative AI sessions")
    print("✅ SSE-based live dashboard updates")
    print("✅ Comprehensive error handling and recovery")
    print("✅ Performance monitoring and analytics")
    print("✅ Seamless integration of all components")

if __name__ == "__main__":
    asyncio.run(run_complete_realtime_demo())
```

## Chapter 9, Section 5 Summary

This comprehensive real-time AI features implementation provides:

### **Section 5.1: WebSocket Streaming**
- Bidirectional real-time communication
- Token-by-token AI response streaming
- Session management with automatic cleanup
- Progress indicators and typing indicators
- Error handling and request cancellation

### **Section 5.2: Server-Sent Events (SSE)**
- One-way streaming for dashboards
- Real-time system monitoring
- Automatic reconnection and heartbeat
- Event-based architecture with typed events
- Background broadcasting tasks

### **Section 5.3: Background Tasks & Collaboration**
- Celery integration for long-running AI tasks
- Priority-based task scheduling
- Real-time collaborative AI sessions
- Multi-user chat with AI integration
- Comprehensive monitoring and analytics

### **Production-Ready Features:**
- **Scalable Architecture**: Async processing with proper resource management
- **Error Resilience**: Circuit breakers, retries, and automatic recovery
- **Real-time Monitoring**: Live metrics and health tracking
- **User Experience**: Smooth, responsive interfaces with immediate feedback
- **Performance Optimization**: Efficient streaming and caching strategies

The next section will be **Chapter 9, Section 6: AI Security and Compliance** - focusing on securing AI integrations, handling sensitive data, and ensuring regulatory compliance.# Chapter 9, Section 5.3: Background Tasks and Real-time Collaboration

## Overview

This final subsection of real-time AI features focuses on background task processing, real-time collaboration features, and advanced streaming patterns. We'll implement Celery integration, collaborative AI sessions, and sophisticated real-time workflows.

## Background Task Processing with Celery

### Celery Integration for AI Tasks

```python
# services/background_ai_service.py
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from datetime import datetime, timedelta

from celery import Celery, Task
from celery.result import AsyncResult
from redis import Redis

from services.production_ai_service import ProductionAIService, RequestConfig
from services.streaming_ai_service import StreamingAIService
from services.sse_streaming_service import SSEStreamingService

logger = structlog.get_logger()

# Celery configuration
celery_app = Celery(
    'ai_background_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    task_routes={
        'ai_background_tasks.process_ai_request': {'queue': 'ai_processing'},
        'ai_background_tasks.batch_ai_processing': {'queue': 'ai_batch'},
        'ai_background_tasks.stream_ai_response': {'queue': 'ai_streaming'},
        'ai_background_tasks.collaborative_ai_task': {'queue': 'ai_collaboration'}
    }
)

class TaskStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROCESSING = "PROCESSING"
    STREAMING = "STREAMING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"

class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class BackgroundTask:
    """Background task tracking"""
    task_id: str
    user_id: Optional[str]
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    