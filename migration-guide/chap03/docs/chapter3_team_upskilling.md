# Flask to FastAPI Migration Guide

## Chapter 3: Team Upskilling & Training

### Overview

The biggest challenge in migrating from Flask to FastAPI isn't technical—it's the mindset shift to asynchronous programming. This chapter provides a structured approach to upskill your team efficiently within the 12-week migration timeline.

---

## The Async Mindset Shift

### Understanding the Fundamental Difference

The transition from Flask's synchronous model to FastAPI's async model requires understanding these core concepts:

#### 1. Blocking vs Non-Blocking Operations
```python
# Flask mindset (blocking)
def process_request():
    data = database.fetch_user()      # Thread waits here
    result = api_call_to_bedrock()    # Thread waits here
    return result

# FastAPI mindset (non-blocking)
async def process_request():
    data = await database.fetch_user()      # Other requests can be handled
    result = await api_call_to_bedrock()    # while waiting for this
    return result
```

#### 2. When to Use async/await

**Use async/await for:**
- Database queries
- HTTP requests (AWS Bedrock calls)
- File I/O operations
- Any network operations
- Time.sleep() → asyncio.sleep()

**Don't use async/await for:**
- CPU-intensive calculations
- Simple data processing
- Synchronous library calls
- Mathematical operations

#### 3. Common Async Pitfalls & Solutions

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Mixing sync/async | `def` function calling `await` | Use `async def` consistently |
| Blocking calls in async | `requests.get()` in async function | Use `aiohttp` or `httpx` |
| Not awaiting | `async_function()` without `await` | Always `await` async functions |
| Sync database calls | `sqlite3.execute()` in async | Use `aiosqlite` or `asyncpg` |
| Blocking sleep | `time.sleep()` in async | Use `asyncio.sleep()` |

---

## Compressed Learning Path (2-Week Intensive)

### Week 1: Foundation Bootcamp

#### Day 1: Async Fundamentals (8 hours)
**Morning Session (4 hours): Concepts**
- Event loop basics and how async works
- async/await syntax and patterns
- Difference between concurrency and parallelism
- Common async patterns in Python

**Afternoon Session (4 hours): Hands-on Practice**
```python
# Exercise 1: Basic async function
import asyncio

# Convert this synchronous function
def fetch_data():
    time.sleep(2)  # Simulate API call
    return "data"

def process_multiple():
    results = []
    for i in range(3):
        results.append(fetch_data())
    return results

# To this async version
async def fetch_data_async():
    await asyncio.sleep(2)  # Non-blocking
    return "data"

async def process_multiple_async():
    tasks = [fetch_data_async() for i in range(3)]
    results = await asyncio.gather(*tasks)
    return results

# This runs in 2 seconds instead of 6!
```

**Practice Exercises:**
1. Convert 5 synchronous functions to async
2. Use `asyncio.gather()` for concurrent operations
3. Handle exceptions in async functions
4. Implement async context managers

#### Day 2: FastAPI Crash Course (8 hours)
**Morning Session (4 hours): FastAPI Basics**
- FastAPI app structure and routing
- Request/response models with Pydantic
- Dependency injection system
- Automatic API documentation

**Afternoon Session (4 hours): Build Complete App**
```python
# Build a simple FastAPI app together
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import asyncio

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

# Simulate database
fake_db = []

async def get_db():
    # Simulate async database connection
    await asyncio.sleep(0.1)
    return fake_db

@app.post("/items/")
async def create_item(item: Item, db=Depends(get_db)):
    db.append(item.dict())
    return {"message": "Item created", "item": item}

@app.get("/items/")
async def list_items(db=Depends(get_db)):
    return {"items": db}
```

#### Day 3: AWS Async Integration (4 hours)
**Focus: aioboto3 and AWS patterns**
```python
# Exercise: Convert boto3 to aioboto3
import boto3
import aioboto3

# Synchronous AWS call (Flask style)
def get_s3_object_sync(bucket, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()

# Asynchronous AWS call (FastAPI style)
async def get_s3_object_async(bucket, key):
    session = aioboto3.Session()
    async with session.client('s3') as s3:
        response = await s3.get_object(Bucket=bucket, Key=key)
        return await response['Body'].read()

# Bedrock integration practice
async def call_bedrock_async(message):
    session = aioboto3.Session()
    async with session.client('bedrock-agent') as client:
        response = await client.invoke_agent(
            agentId='agent-123',
            inputText=message
        )
        return response['output']['text']
```

### Week 2: Advanced Patterns & Production Readiness

#### Day 4: Error Handling & Testing (4 hours)
```python
# Async error handling patterns
import asyncio
from fastapi import HTTPException

async def safe_bedrock_call(message: str):
    try:
        # Timeout handling
        return await asyncio.wait_for(
            call_bedrock_async(message),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Testing async functions
import pytest
import httpx

@pytest.mark.asyncio
async def test_chat_endpoint():
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
```

#### Day 5: Performance & Production (4 hours)
```python
# Connection pooling and resource management
class AsyncResourceManager:
    def __init__(self):
        self.session = aioboto3.Session()
        self._clients = {}
    
    async def get_client(self, service_name):
        if service_name not in self._clients:
            self._clients[service_name] = await self.session.client(
                service_name
            ).__aenter__()
        return self._clients[service_name]
    
    async def cleanup(self):
        for client in self._clients.values():
            await client.__aexit__(None, None, None)

# Background tasks
from fastapi import BackgroundTasks

@app.post("/process/")
async def process_data(background_tasks: BackgroundTasks):
    background_tasks.add_task(expensive_computation)
    return {"message": "Processing started"}

async def expensive_computation():
    # Long-running task that doesn't block response
    await asyncio.sleep(10)
    print("Task completed")
```

---

## Common Async Patterns for Chatbots

### 1. Concurrent Request Processing
```python
async def handle_multiple_chats(messages: List[str], session_id: str):
    """Process multiple user requests simultaneously"""
    tasks = [
        process_single_chat(msg, session_id) 
        for msg in messages
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            processed_results.append({"error": str(result)})
        else:
            processed_results.append(result)
    
    return processed_results
```

### 2. Connection Pooling for AWS Services
```python
# Reuse connections efficiently
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_bedrock_client():
    session = aioboto3.Session()
    async with session.client('bedrock-agent') as client:
        yield client
        # Automatically cleaned up

# Usage in FastAPI
async def chat_endpoint(message: str):
    async with get_bedrock_client() as bedrock:
        response = await bedrock.invoke_agent(...)
        return response
```

### 3. Timeout and Retry Patterns
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def resilient_bedrock_call(message: str):
    """Bedrock call with retry logic"""
    try:
        async with get_bedrock_client() as client:
            return await asyncio.wait_for(
                client.invoke_agent(...),
                timeout=30.0
            )
    except asyncio.TimeoutError:
        raise Exception("Bedrock timeout")
```

### 4. Sync-Async Interoperability Patterns

#### Technique 1: Calling Sync Functions from Async Code

**When You Need This:**
- Testing async code with sync test utilities
- Using legacy sync libraries in async applications
- Quick debugging or verification during development
- Integrating with sync-only third-party APIs

**Method 1: Using asyncio.to_thread() (Python 3.9+)**
```python
import asyncio
import time
import requests  # Sync library

# Sync function that we need to call
def sync_api_call(url: str) -> dict:
    """Legacy sync function that makes HTTP request"""
    response = requests.get(url)
    return response.json()

def sync_data_processing(data: list) -> dict:
    """CPU-intensive sync function"""
    # Simulate heavy processing
    time.sleep(2)
    return {"processed_count": len(data), "sum": sum(data)}

# Async function calling sync functions
async def async_workflow():
    """Demonstrates calling sync functions from async context"""
    
    # Method 1: Use asyncio.to_thread for I/O-bound sync calls
    print("Starting async workflow...")
    
    # Call sync API function without blocking event loop
    result1 = await asyncio.to_thread(
        sync_api_call, 
        "https://jsonplaceholder.typicode.com/posts/1"
    )
    print(f"API result: {result1['title']}")
    
    # Call sync processing function
    data = [1, 2, 3, 4, 5] * 1000  # Large dataset
    result2 = await asyncio.to_thread(
        sync_data_processing,
        data
    )
    print(f"Processing result: {result2}")
    
    # Multiple sync calls concurrently
    tasks = [
        asyncio.to_thread(sync_api_call, f"https://jsonplaceholder.typicode.com/posts/{i}")
        for i in range(1, 4)
    ]
    results = await asyncio.gather(*tasks)
    print(f"Concurrent results: {len(results)} API calls completed")

# Run the async workflow
async def main():
    await async_workflow()

# Execute
if __name__ == "__main__":
    asyncio.run(main())
```

**Method 2: Using loop.run_in_executor() (More Control)**
```python
import asyncio
import concurrent.futures
import functools

async def async_with_executor():
    """Using executor for more control over sync function execution"""
    loop = asyncio.get_event_loop()
    
    # Create thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        
        # Simple sync function call
        result1 = await loop.run_in_executor(
            executor,
            sync_data_processing,
            [1, 2, 3, 4, 5]
        )
        
        # Sync function with multiple arguments using functools.partial
        sync_function_with_args = functools.partial(
            sync_api_call,
            "https://jsonplaceholder.typicode.com/posts/1"
        )
        result2 = await loop.run_in_executor(executor, sync_function_with_args)
        
        print(f"Executor results: {result1}, {result2['id']}")

# Testing pattern for async functions
async def test_async_function():
    """Example of testing async code with sync assertions"""
    
    # Your async function under test
    async def fetch_user_data(user_id: int):
        # Simulate async database call
        await asyncio.sleep(0.1)
        return {"id": user_id, "name": f"User {user_id}"}
    
    # Test it by calling sync assertion functions
    result = await fetch_user_data(123)
    
    # Call sync test utilities
    assert_result = await asyncio.to_thread(
        lambda data: data["id"] == 123 and "name" in data,
        result
    )
    
    print(f"Test passed: {assert_result}")
```

**Quick Testing Shortcuts:**
```python
# Quick and dirty sync call in async context (development only!)
import asyncio
import nest_asyncio  # pip install nest-asyncio

# For Jupyter notebooks or testing environments
nest_asyncio.apply()

async def quick_test():
    # This allows running async code in sync context for testing
    result = await some_async_function()
    return result

# Can now call from sync context during development
# result = asyncio.run(quick_test())  # This works with nest_asyncio
```

#### Technique 2: Calling Async Functions from Sync Code

**When You Need This:**
- Integrating async FastAPI code with sync Flask code during migration
- Testing async functions from sync test frameworks
- Quick verification or debugging from sync contexts
- Legacy sync applications calling new async services

**Method 1: Using asyncio.run() (Preferred)**
```python
import asyncio
import aiohttp

# Async function that we need to call from sync code
async def async_api_call(url: str) -> dict:
    """Async function using aiohttp"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def async_bedrock_call(message: str) -> str:
    """Simulate async Bedrock call"""
    await asyncio.sleep(1)  # Simulate API delay
    return f"AI response to: {message}"

# Sync function calling async functions
def sync_function_calling_async():
    """Sync function that needs to call async functions"""
    
    print("Starting sync function...")
    
    # Method 1: Simple async call
    result1 = asyncio.run(async_api_call("https://jsonplaceholder.typicode.com/posts/1"))
    print(f"Sync got async result: {result1['title']}")
    
    # Method 2: Multiple async calls
    async def multiple_async_calls():
        tasks = [
            async_api_call("https://jsonplaceholder.typicode.com/posts/1"),
            async_api_call("https://jsonplaceholder.typicode.com/posts/2"),
            async_bedrock_call("Hello from sync world")
        ]
        return await asyncio.gather(*tasks)
    
    results = asyncio.run(multiple_async_calls())
    print(f"Multiple async results: {len(results)} calls completed")
    
    return results

# Flask-style sync endpoint calling FastAPI async logic
def flask_endpoint_calling_fastapi_logic():
    """Example: Flask endpoint calling FastAPI async business logic"""
    
    # Your async business logic (from FastAPI migration)
    async def process_chat_message(message: str) -> dict:
        # Simulate async processing
        await asyncio.sleep(0.5)
        processed = await async_bedrock_call(message)
        return {
            "original": message,
            "processed": processed,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    # Flask route can call this
    user_message = "Hello from Flask"
    result = asyncio.run(process_chat_message(user_message))
    
    return result  # Return to Flask's jsonify()

# Testing async functions from sync test frameworks
def test_async_function_from_sync():
    """Testing async functions using sync test frameworks like pytest"""
    
    async def async_function_to_test(x: int, y: int) -> int:
        await asyncio.sleep(0.1)  # Simulate async work
        return x + y
    
    # Test from sync context
    result = asyncio.run(async_function_to_test(2, 3))
    assert result == 5
    print("Test passed!")
    
    # Test error handling
    async def async_function_that_raises():
        await asyncio.sleep(0.1)
        raise ValueError("Test error")
    
    try:
        asyncio.run(async_function_that_raises())
    except ValueError as e:
        print(f"Correctly caught error: {e}")

if __name__ == "__main__":
    # Run sync functions that call async code
    sync_function_calling_async()
    flask_result = flask_endpoint_calling_fastapi_logic()
    print(f"Flask result: {flask_result}")
    test_async_function_from_sync()
```

**Method 2: Using asyncio.get_event_loop() (Advanced)**
```python
import asyncio
import threading

def sync_function_with_event_loop():
    """More advanced pattern with event loop management"""
    
    async def complex_async_workflow():
        # Multiple async operations
        results = []
        
        for i in range(3):
            result = await async_bedrock_call(f"Message {i}")
            results.append(result)
            await asyncio.sleep(0.1)
        
        return results
    
    # Method 1: Simple run
    results = asyncio.run(complex_async_workflow())
    print(f"Complex workflow results: {results}")
    
    # Method 2: Using existing event loop (if available)
    try:
        loop = asyncio.get_running_loop()
        # If there's already a running loop, we need to use run_in_executor
        # This is common in Jupyter notebooks
        print("Found running loop, using executor...")
        
        def run_async():
            return asyncio.run(complex_async_workflow())
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async)
            results = future.result()
            print(f"Executor results: {results}")
            
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        results = asyncio.run(complex_async_workflow())
        print(f"Direct run results: {results}")

# Practical migration pattern: Gradual async adoption
class HybridService:
    """Service that supports both sync and async interfaces during migration"""
    
    async def _async_process(self, data: str) -> str:
        """Core async implementation"""
        await asyncio.sleep(0.1)
        return f"Processed: {data}"
    
    def sync_process(self, data: str) -> str:
        """Sync interface for legacy code"""
        return asyncio.run(self._async_process(data))
    
    async def async_process(self, data: str) -> str:
        """Async interface for new code"""
        return await self._async_process(data)

# Usage example
def demonstrate_hybrid_service():
    service = HybridService()
    
    # Legacy sync code can still work
    sync_result = service.sync_process("sync data")
    print(f"Sync interface result: {sync_result}")
    
    # New async code can use async interface
    async def new_async_code():
        async_result = await service.async_process("async data")
        print(f"Async interface result: {async_result}")
    
    asyncio.run(new_async_code())
```

**Quick Testing and Verification Patterns:**
```python
# Pattern 1: Quick async verification in sync context
def quick_verify_async_function():
    """Quick verification pattern for development"""
    
    async def verify_bedrock_integration():
        # Your async function to verify
        result = await async_bedrock_call("test message")
        return len(result) > 0
    
    # Quick verification
    is_working = asyncio.run(verify_bedrock_integration())
    print(f"Bedrock integration working: {is_working}")

# Pattern 2: Sync wrapper for repeated testing
def create_sync_wrapper(async_func):
    """Create sync wrapper for async function"""
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# Usage
sync_bedrock_call = create_sync_wrapper(async_bedrock_call)
result = sync_bedrock_call("test from sync wrapper")
print(f"Wrapper result: {result}")

# Pattern 3: Testing async generators from sync code
async def async_data_generator():
    """Async generator for testing"""
    for i in range(3):
        await asyncio.sleep(0.1)
        yield f"data_{i}"

def test_async_generator():
    """Test async generator from sync code"""
    async def collect_data():
        results = []
        async for item in async_data_generator():
            results.append(item)
        return results
    
    data = asyncio.run(collect_data())
    print(f"Generator data: {data}")

if __name__ == "__main__":
    sync_function_with_event_loop()
    demonstrate_hybrid_service()
    quick_verify_async_function()
    test_async_generator()
```

**Development Shortcuts and Best Practices:**

```python
# Development utility: Async function runner for interactive testing
class AsyncRunner:
    """Utility class for running async functions in sync contexts"""
    
    @staticmethod
    def run(async_func, *args, **kwargs):
        """Run async function with args"""
        return asyncio.run(async_func(*args, **kwargs))
    
    @staticmethod
    def run_multiple(*async_funcs):
        """Run multiple async functions concurrently"""
        async def runner():
            return await asyncio.gather(*async_funcs)
        return asyncio.run(runner())

# Usage for quick testing
runner = AsyncRunner()

# Quick test individual async functions
result1 = runner.run(async_bedrock_call, "test message 1")
print(f"Quick test 1: {result1}")

# Quick test multiple functions
async def test_func2():
    await asyncio.sleep(0.1)
    return "test result 2"

results = runner.run_multiple(
    async_bedrock_call("test message A"),
    test_func2()
)
print(f"Multiple quick tests: {results}")
```

**Common Pitfalls and Solutions:**

```python
# Pitfall 1: RuntimeError: asyncio.run() cannot be called from a running event loop
def handle_nested_event_loop():
    """How to handle nested event loop scenarios"""
    
    try:
        # This will fail if called from within an async context
        result = asyncio.run(async_bedrock_call("test"))
        return result
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # Solution: Use nest_asyncio or run in thread
            import nest_asyncio
            nest_asyncio.apply()
            result = asyncio.run(async_bedrock_call("test"))
            return result
        else:
            raise

# Pitfall 2: Blocking event loop with sync calls in async functions
async def bad_async_function():
    """DON'T DO THIS - blocks event loop"""
    time.sleep(2)  # Blocks entire event loop!
    return "bad result"

async def good_async_function():
    """DO THIS - use asyncio.to_thread for blocking calls"""
    result = await asyncio.to_thread(time.sleep, 2)
    return "good result"

# Testing both patterns
def compare_blocking_patterns():
    import time
    
    start = time.time()
    
    # Bad pattern - everything runs sequentially
    async def bad_concurrent():
        tasks = [bad_async_function() for _ in range(3)]
        return await asyncio.gather(*tasks)
    
    # This will take ~6 seconds (sequential)
    # asyncio.run(bad_concurrent())
    
    # Good pattern - truly concurrent
    async def good_concurrent():
        tasks = [good_async_function() for _ in range(3)]
        return await asyncio.gather(*tasks)
    
    # This will take ~2 seconds (concurrent)
    asyncio.run(good_concurrent())
    
    print(f"Good pattern completed in: {time.time() - start:.2f} seconds")
```

These techniques are essential for:
- **Migration periods** when you have both sync and async code
- **Testing and debugging** async functions quickly
- **Gradual adoption** of async patterns in existing codebases
- **Integration** with legacy systems during FastAPI migration

The key is knowing when to use each pattern and understanding the performance implications!
```python
class BedrockService:
    def __init__(self):
        self.session = aioboto3.Session()
        self.client = None
    
    async def __aenter__(self):
        self.client = await self.session.client('bedrock-agent').__aenter__()
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.__aexit__(*args)
    
    async def invoke_agent(self, message: str):
        if not self.client:
            raise RuntimeError("Service not initialized")
        return await self.client.invoke_agent(...)

# Usage
async with BedrockService() as bedrock:
    result = await bedrock.invoke_agent("Hello")
```

---

## Knowledge Transfer Sessions

### Session 1: "Async Fundamentals Workshop" (2 hours)
**Objectives:** Build async intuition through live coding

**Format:**
- 30 min: Concept explanation with visual diagrams
- 60 min: Live coding exercises (convert Flask routes)
- 30 min: Q&A and common pitfalls discussion

**Live Coding Exercise:**
```python
# Start with this Flask code
from flask import Flask, jsonify
import time
import requests

app = Flask(__name__)

@app.route('/weather/<city>')
def get_weather(city):
    # Simulate slow API calls
    api_response = requests.get(f'http://api.weather.com/{city}')
    time.sleep(1)  # Database lookup
    return jsonify({"city": city, "weather": api_response.json()})

@app.route('/multi-weather')
def get_multiple_weather():
    cities = ['London', 'Tokyo', 'New York']
    results = []
    for city in cities:
        # This takes 3+ seconds total
        weather = get_weather_data(city)
        results.append(weather)
    return jsonify(results)

# Convert to FastAPI async version live
from fastapi import FastAPI
import asyncio
import httpx

app = FastAPI()

@app.get('/weather/{city}')
async def get_weather(city: str):
    async with httpx.AsyncClient() as client:
        api_response = await client.get(f'http://api.weather.com/{city}')
        await asyncio.sleep(1)  # Async database lookup
        return {"city": city, "weather": api_response.json()}

@app.get('/multi-weather')
async def get_multiple_weather():
    cities = ['London', 'Tokyo', 'New York']
    # This takes ~1 second total (concurrent!)
    tasks = [get_weather_async(city) for city in cities]
    results = await asyncio.gather(*tasks)
    return results
```

### Session 2: "FastAPI Deep Dive Workshop" (2 hours)
**Objectives:** Master FastAPI patterns and dependency injection

**Hands-on Lab:**
```python
# Build a complete chatbot API together
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import uuid

app = FastAPI(title="Team Chatbot", version="1.0.0")

# Models
class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    processing_time: float

# Mock services for practice
class MockBedrockService:
    async def process_message(self, message: str) -> str:
        await asyncio.sleep(0.5)  # Simulate API delay
        return f"AI Response to: {message}"

class MockSessionService:
    def __init__(self):
        self.sessions = {}
    
    async def save_session(self, session_id: str, data: dict):
        self.sessions[session_id] = data
    
    async def get_session(self, session_id: str):
        return self.sessions.get(session_id, {})

# Dependency injection
async def get_bedrock_service():
    return MockBedrockService()

async def get_session_service():
    return MockSessionService()

# Analytics background task
async def record_analytics(message: str, response: str):
    await asyncio.sleep(0.1)  # Simulate analytics API
    print(f"Analytics: {len(message)} chars in, {len(response)} chars out")

# Main endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    background_tasks: BackgroundTasks,
    bedrock: MockBedrockService = Depends(get_bedrock_service),
    sessions: MockSessionService = Depends(get_session_service)
):
    start_time = time.time()
    
    # Generate session if needed
    session_id = message.session_id or str(uuid.uuid4())
    
    # Get session context
    session_data = await sessions.get_session(session_id)
    
    # Process with AI
    ai_response = await bedrock.process_message(message.content)
    
    # Save session
    await sessions.save_session(session_id, {
        "last_message": message.content,
        "message_count": session_data.get("message_count", 0) + 1
    })
    
    # Background analytics
    background_tasks.add_task(
        record_analytics, 
        message.content, 
        ai_response
    )
    
    processing_time = time.time() - start_time
    
    return ChatResponse(
        response=ai_response,
        session_id=session_id,
        processing_time=processing_time
    )

# Team builds this together in 2 hours
```

### Session 3: "AWS Async Integration Workshop" (2 hours)
**Objectives:** Master aioboto3 and AWS service integration

**Lab Exercise:**
```python
# Practice converting current AWS integrations
import aioboto3
import asyncio
from contextlib import asynccontextmanager

# Exercise 1: Basic aioboto3 usage
async def exercise_bedrock_basic():
    session = aioboto3.Session()
    
    async with session.client('bedrock-agent') as client:
        response = await client.invoke_agent(
            agentId='your-agent-id',
            sessionId='test-session',
            inputText='Hello from async!'
        )
        return response

# Exercise 2: Error handling and timeouts
async def exercise_bedrock_robust():
    session = aioboto3.Session()
    
    try:
        async with session.client('bedrock-agent') as client:
            response = await asyncio.wait_for(
                client.invoke_agent(
                    agentId='your-agent-id',
                    sessionId='test-session',
                    inputText='Hello with timeout!'
                ),
                timeout=30.0
            )
            return response
    except asyncio.TimeoutError:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": f"Bedrock error: {str(e)}"}

# Exercise 3: Connection reuse pattern
class AWSServiceManager:
    def __init__(self):
        self.session = aioboto3.Session()
        self._clients = {}
    
    @asynccontextmanager
    async def get_client(self, service_name: str):
        if service_name not in self._clients:
            client = await self.session.client(service_name).__aenter__()
            self._clients[service_name] = client
        
        try:
            yield self._clients[service_name]
        finally:
            # Keep connection open for reuse
            pass
    
    async def cleanup(self):
        for client in self._clients.values():
            await client.__aexit__(None, None, None)

# Global manager instance
aws_manager = AWSServiceManager()

# Usage in FastAPI
async def chat_with_connection_reuse(message: str):
    async with aws_manager.get_client('bedrock-agent') as bedrock:
        response = await bedrock.invoke_agent(...)
        return response
```

### Session 4: "Testing & Production Patterns" (2 hours)
**Objectives:** Learn async testing and production best practices

**Testing Workshop:**
```python
# pytest-asyncio patterns
import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Testing async endpoints
@pytest.mark.asyncio
async def test_chat_endpoint_async():
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/chat", json={
            "content": "test message"
        })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "session_id" in data

# Mocking async services
@pytest.mark.asyncio
async def test_chat_with_mock():
    with patch('services.bedrock_service.BedrockService') as mock_bedrock:
        # Setup async mock
        mock_instance = AsyncMock()
        mock_instance.process_message.return_value = "Mocked response"
        mock_bedrock.return_value = mock_instance
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/chat", json={
                "content": "test message"
            })
        
        assert response.status_code == 200
        mock_instance.process_message.assert_called_once()

# Load testing async endpoints
import asyncio
import aiohttp
import time

async def load_test_chat():
    """Simulate 100 concurrent requests"""
    async def single_request(session, request_id):
        async with session.post(
            'http://localhost:8000/chat',
            json={"content": f"Test message {request_id}"}
        ) as response:
            return await response.json()
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            single_request(session, i) 
            for i in range(100)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    duration = time.time() - start_time
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    
    print(f"Completed 100 requests in {duration:.2f}s")
    print(f"Success rate: {success_count}/100")
    print(f"Requests per second: {100/duration:.2f}")

# Run: asyncio.run(load_test_chat())
```

---

## Development Environment Setup

### 1. Local Development Environment
```bash
# Setup script for team members
#!/bin/bash
# setup_fastapi_dev.sh

echo "Setting up FastAPI development environment..."

# Create virtual environment
python -m venv fastapi_env
source fastapi_env/bin/activate  # On Windows: fastapi_env\Scripts\activate

# Install development dependencies
pip install fastapi uvicorn[standard] aioboto3 pytest pytest-asyncio httpx

# Install development tools
pip install black isort mypy pre-commit

# Setup pre-commit hooks
pre-commit install

echo "Environment ready! Start with: uvicorn main:app --reload"
```

### 2. VS Code Configuration
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./fastapi_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "files.associations": {
        "*.py": "python"
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "-v"
    ]
}
```

### 3. Debugging Async Code
```python
# debug_helpers.py
import asyncio
import functools
import time

def async_timer(func):
    """Decorator to time async functions"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

def debug_async_calls(func):
    """Decorator to debug async function calls"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"Calling async function: {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            print(f"✓ {func.__name__} completed successfully")
            return result
        except Exception as e:
            print(f"✗ {func.__name__} failed: {str(e)}")
            raise
    return wrapper

# Usage
@async_timer
@debug_async_calls
async def slow_operation():
    await asyncio.sleep(2)
    return "completed"
```

---

## Team Readiness Assessment

### Pre-Migration Checklist
Before starting the actual migration, ensure each team member can:

**Week 1 Competencies:**
- [ ] Explain the difference between sync and async execution
- [ ] Convert a simple Flask route to FastAPI equivalent
- [ ] Use async/await with `asyncio.sleep()` and `asyncio.gather()`
- [ ] Handle basic exceptions in async functions
- [ ] Create Pydantic models for request/response

**Week 2 Competencies:**
- [ ] Use aioboto3 for AWS service calls
- [ ] Implement proper async context managers
- [ ] Write basic async tests with pytest-asyncio
- [ ] Use FastAPI dependency injection
- [ ] Handle timeouts and retries in async code

### Skills Validation Exercises

**Exercise 1: Basic Async Conversion**
```python
# Given this Flask code, convert to FastAPI async
from flask import Flask, request, jsonify
import requests
import time

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    
    # Simulate database save
    time.sleep(0.5)
    
    # Call external API
    response = requests.post('http://external-api.com', json=data)
    
    # Return result
    return jsonify({"status": "processed", "result": response.json()})

# Team member should produce FastAPI equivalent
```

**Exercise 2: AWS Integration**
```python
# Convert this boto3 code to aioboto3
import boto3

def call_bedrock(message):
    client = boto3.client('bedrock-agent')
    response = client.invoke_agent(
        agentId='agent-123',
        sessionId='session-456',
        inputText=message
    )
    return response['output']['text']

# Expected async version with proper error handling
```

### Knowledge Gaps & Mitigation

**Common Struggle Areas:**
1. **Event Loop Confusion** → Use visual diagrams and step-through debugging
2. **When to Use Async** → Provide clear decision tree and examples
3. **AWS Async Patterns** → Hands-on labs with real AWS services
4. **Testing Async Code** → Pair programming with async testing patterns

**Support Strategies:**
- **Buddy System**: Pair async-experienced dev with Flask developer
- **Daily Standups**: Share async learning challenges and solutions
- **Code Reviews**: Focus on async patterns and best practices
- **Reference Guide**: Quick async patterns cheat sheet

---

## Resources & References

### Essential Documentation
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [aioboto3 Documentation](https://aioboto3.readthedocs.io/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

### Recommended Reading
- "Using Asyncio in Python" by Caleb Hattingh
- "Effective Python" by Brett Slatkin (Async chapters)
- FastAPI tutorial series on Real Python

### Practice Repositories
1. **Simple FastAPI Examples**: Basic CRUD operations
2. **AWS Integration Examples**: aioboto3 patterns
3. **Testing Examples**: pytest-asyncio patterns
4. **Production Patterns**: Connection pooling, error handling

### Team Slack Channels
- `#fastapi-migration`: Migration-specific discussions
- `#async-help`: Async programming questions
- `#code-reviews`: Async code review requests

---

*Next: Chapter 4 - Implementation Guide*