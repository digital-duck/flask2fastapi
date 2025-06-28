"""
Async Programming Patterns Example
Chapter 2-3: Demonstrates fundamental async concepts for Flask to FastAPI migration

This comprehensive example covers:
- Basic async/await patterns
- Concurrent vs sequential execution
- Async context managers
- Error handling in async code
- Real-world async patterns

Usage:
    python src/async_patterns/async_basics.py
"""

import asyncio
import aiohttp
import time
import random
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 1. Basic Async Patterns
async def basic_async_example():
    """Demonstrate basic async/await patterns"""
    print("\n" + "="*50)
    print("1. BASIC ASYNC PATTERNS")
    print("="*50)
    
    async def async_operation(name: str, delay: float) -> str:
        """Simulate an async operation"""
        print(f"  🚀 Starting {name}...")
        await asyncio.sleep(delay)
        print(f"  ✅ Completed {name}")
        return f"Result from {name}"
    
    # Sequential execution (like synchronous code)
    print("\n📊 Sequential execution:")
    start_time = time.time()
    result1 = await async_operation("Operation 1", 1.0)
    result2 = await async_operation("Operation 2", 0.5)
    sequential_time = time.time() - start_time
    print(f"  ⏱️  Sequential took: {sequential_time:.2f}s")
    
    # Concurrent execution (async advantage)
    print("\n🚀 Concurrent execution:")
    start_time = time.time()
    result1, result2 = await asyncio.gather(
        async_operation("Operation A", 1.0),
        async_operation("Operation B", 0.5)
    )
    concurrent_time = time.time() - start_time
    print(f"  ⏱️  Concurrent took: {concurrent_time:.2f}s")
    print(f"  📈 Speedup: {sequential_time/concurrent_time:.2f}x")


# 2. HTTP Requests Pattern
async def async_http_requests():
    """Demonstrate async HTTP requests vs sync approach"""
    print("\n" + "="*50)
    print("2. ASYNC HTTP REQUESTS")
    print("="*50)
    
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/1", 
        "https://httpbin.org/delay/1"
    ]
    
    async def fetch_url(session: aiohttp.ClientSession, url: str, request_id: int) -> Dict[str, Any]:
        """Fetch single URL asynchronously"""
        try:
            print(f"  🌐 Starting request {request_id}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                data = await response.json()
                print(f"  ✅ Completed request {request_id}")
                return {
                    "request_id": request_id,
                    "url": url, 
                    "status": response.status, 
                    "data": data
                }
        except Exception as e:
            print(f"  ❌ Failed request {request_id}: {e}")
            return {"request_id": request_id, "url": url, "error": str(e)}
    
    # Simulate synchronous approach timing
    sync_time = len(urls) * 1  # Each request takes ~1 second
    print(f"📊 Synchronous approach would take: ~{sync_time}s")
    
    # Async approach
    print("\n🚀 Making concurrent async requests...")
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_url(session, url, i+1) for i, url in enumerate(urls)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        async_time = time.time() - start_time
        print(f"  ⏱️  Async total: {async_time:.2f}s")
        print(f"  📈 Speedup: {sync_time/async_time:.2f}x")
        
    except Exception as e:
        print(f"  ❌ HTTP requests failed: {e}")
        print("  💡 This might be due to network connectivity")


# 3. Database Simulation Pattern
async def async_database_pattern():
    """Simulate async database operations"""
    print("\n" + "="*50)
    print("3. ASYNC DATABASE PATTERNS")
    print("="*50)
    
    async def simulate_db_query(query: str, delay: float) -> Dict[str, Any]:
        """Simulate async database query"""
        print(f"  🗄️  Executing: {query}")
        await asyncio.sleep(delay)  # Simulate I/O wait
        return {
            "query": query,
            "results": f"Mock results for {query}",
            "execution_time": delay,
            "rows_affected": random.randint(1, 100)
        }
    
    # Simulate multiple database operations
    queries = [
        ("SELECT * FROM users WHERE active = true", 0.3),
        ("SELECT * FROM orders WHERE date > '2024-01-01'", 0.5),
        ("SELECT * FROM products WHERE category = 'electronics'", 0.2),
    ]
    
    print("📊 Sequential database queries:")
    start_time = time.time()
    for query, delay in queries:
        result = await simulate_db_query(query, delay)
        print(f"    ✅ {result['rows_affected']} rows")
    sequential_db_time = time.time() - start_time
    
    print("\n🚀 Concurrent database queries:")
    start_time = time.time()
    
    # Execute all queries concurrently
    tasks = [simulate_db_query(query, delay) for query, delay in queries]
    results = await asyncio.gather(*tasks)
    
    concurrent_db_time = time.time() - start_time
    theoretical_sync_time = sum(delay for _, delay in queries)
    
    total_rows = sum(r['rows_affected'] for r in results)
    print(f"    ✅ Total rows processed: {total_rows}")
    print(f"  ⏱️  Concurrent execution: {concurrent_db_time:.2f}s")
    print(f"  📊 Sequential would take: {theoretical_sync_time:.2f}s")
    print(f"  📈 Efficiency gain: {theoretical_sync_time/concurrent_db_time:.2f}x")


# 4. Async Context Managers
async def async_context_manager_example():
    """Demonstrate async context managers"""
    print("\n" + "="*50)
    print("4. ASYNC CONTEXT MANAGERS")
    print("="*50)
    
    class AsyncDatabaseConnection:
        """Example async context manager for database connections"""
        
        def __init__(self, connection_string: str):
            self.connection_string = connection_string
            self.connected = False
            self.connection_id = random.randint(1000, 9999)
        
        async def __aenter__(self):
            print(f"  🔌 Opening connection {self.connection_id}...")
            await asyncio.sleep(0.1)  # Simulate connection time
            self.connected = True
            print(f"  ✅ Connection {self.connection_id} established")
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            print(f"  🔒 Closing connection {self.connection_id}...")
            await asyncio.sleep(0.05)  # Simulate cleanup time
            self.connected = False
            print(f"  ✅ Connection {self.connection_id} closed")
            
            if exc_type:
                print(f"  ⚠️  Exception occurred: {exc_type.__name__}")
        
        async def execute(self, query: str):
            """Execute a query"""
            if not self.connected:
                raise RuntimeError("Database not connected")
            print(f"    📝 Executing: {query}")
            await asyncio.sleep(0.1)
            return f"Query result: {query} completed"
    
    # Use async context manager
    print("🗄️  Using async context manager for database operations:")
    
    try:
        async with AsyncDatabaseConnection("postgresql://localhost/mydb") as db:
            result1 = await db.execute("INSERT INTO users (name) VALUES ('Alice')")
            result2 = await db.execute("SELECT * FROM users WHERE id = 1")
            print(f"    📊 {result1}")
            print(f"    📊 {result2}")
    except Exception as e:
        print(f"    ❌ Database operation failed: {e}")
    
    print("  💡 Context manager automatically handled connection lifecycle")


# 5. Error Handling Patterns
async def async_error_handling():
    """Demonstrate async error handling patterns"""
    print("\n" + "="*50)
    print("5. ASYNC ERROR HANDLING")
    print("="*50)
    
    async def unreliable_operation(name: str, should_fail: bool = False, delay: float = 0.1):
        """Operation that might fail"""
        await asyncio.sleep(delay)
        if should_fail:
            raise ValueError(f"Operation {name} failed as expected")
        return f"Success: {name}"
    
    operations = [
        ("Operation 1", False, 0.1),
        ("Operation 2", True, 0.2),   # This will fail
        ("Operation 3", False, 0.15),
        ("Operation 4", True, 0.05),  # This will also fail
    ]
    
    # Handle errors gracefully with asyncio.gather
    print("🛡️  Graceful error handling with return_exceptions=True:")
    
    tasks = [unreliable_operation(name, should_fail, delay) 
             for name, should_fail, delay in operations]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        name = operations[i][0]
        if isinstance(result, Exception):
            print(f"  ❌ {name}: {result}")
        else:
            print(f"  ✅ {name}: {result}")
    
    # Alternative: Individual try-catch for each operation
    print("\n🔄 Individual error handling:")
    
    async def safe_operation(name: str, should_fail: bool, delay: float):
        try:
            result = await unreliable_operation(name, should_fail, delay)
            return {"name": name, "status": "success", "result": result}
        except Exception as e:
            return {"name": name, "status": "error", "error": str(e)}
    
    safe_tasks = [safe_operation(name, should_fail, delay) 
                  for name, should_fail, delay in operations]
    safe_results = await asyncio.gather(*safe_tasks)
    
    for result in safe_results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        content = result.get("result", result.get("error"))
        print(f"  {status_icon} {result['name']}: {content}")


# 6. Producer-Consumer Pattern
async def producer_consumer_pattern():
    """Demonstrate producer-consumer pattern with async queues"""
    print("\n" + "="*50)
    print("6. PRODUCER-CONSUMER PATTERN")
    print("="*50)
    
    async def producer(queue: asyncio.Queue, producer_id: int, item_count: int):
        """Produce items and put them in queue"""
        print(f"  🏭 Producer {producer_id} starting...")
        
        for i in range(item_count):
            item = f"Item-{producer_id}-{i+1}"
            await queue.put(item)
            print(f"    📦 Producer {producer_id} created: {item}")
            await asyncio.sleep(0.1)  # Simulate production time
        
        print(f"  🏁 Producer {producer_id} finished")
    
    async def consumer(queue: asyncio.Queue, consumer_id: int):
        """Consume items from queue"""
        print(f"  🤖 Consumer {consumer_id} starting...")
        processed_count = 0
        
        while True:
            try:
                # Wait for item with timeout
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
                print(f"    ⚙️  Consumer {consumer_id} processing: {item}")
                await asyncio.sleep(0.2)  # Simulate processing time
                queue.task_done()
                processed_count += 1
                
            except asyncio.TimeoutError:
                print(f"  🏁 Consumer {consumer_id} finished (timeout), processed {processed_count} items")
                break
    
    # Create queue and start producers/consumers
    queue = asyncio.Queue(maxsize=5)
    
    print("🚀 Starting producer-consumer simulation...")
    
    # Start producers and consumers concurrently
    await asyncio.gather(
        producer(queue, 1, 3),
        producer(queue, 2, 2),
        consumer(queue, 1),
        consumer(queue, 2)
    )
    
    print("  ✅ Producer-consumer pattern completed")


# 7. Real-World Migration Example
async def migration_example():
    """Demonstrate a realistic Flask-to-FastAPI migration scenario"""
    print("\n" + "="*50)
    print("7. REAL-WORLD MIGRATION EXAMPLE")
    print("="*50)
    
    # Simulate Flask-style synchronous user service
    class FlaskUserService:
        """Traditional Flask-style service (synchronous)"""
        
        def get_user_profile(self, user_id: int) -> Dict[str, Any]:
            """Get user profile (blocks thread)"""
            time.sleep(0.2)  # Simulate database query
            return {"id": user_id, "name": f"User {user_id}", "profile": "basic"}
        
        def get_user_orders(self, user_id: int) -> List[Dict[str, Any]]:
            """Get user orders (blocks thread)"""
            time.sleep(0.3)  # Simulate database query
            return [{"id": i, "user_id": user_id, "amount": 100 * i} for i in range(1, 4)]
        
        def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
            """Get user preferences (blocks thread)"""
            time.sleep(0.1)  # Simulate database query
            return {"theme": "dark", "notifications": True}
    
    # FastAPI-style asynchronous user service
    class FastAPIUserService:
        """Modern FastAPI-style service (asynchronous)"""
        
        async def get_user_profile(self, user_id: int) -> Dict[str, Any]:
            """Get user profile (non-blocking)"""
            await asyncio.sleep(0.2)  # Simulate async database query
            return {"id": user_id, "name": f"User {user_id}", "profile": "premium"}
        
        async def get_user_orders(self, user_id: int) -> List[Dict[str, Any]]:
            """Get user orders (non-blocking)"""
            await asyncio.sleep(0.3)  # Simulate async database query
            return [{"id": i, "user_id": user_id, "amount": 150 * i} for i in range(1, 4)]
        
        async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
            """Get user preferences (non-blocking)"""
            await asyncio.sleep(0.1)  # Simulate async database query
            return {"theme": "light", "notifications": True, "language": "en"}
    
    # Demonstrate the difference
    user_id = 123
    
    print("📊 Flask-style sequential processing:")
    flask_service = FlaskUserService()
    start_time = time.time()
    
    profile = flask_service.get_user_profile(user_id)
    orders = flask_service.get_user_orders(user_id)
    preferences = flask_service.get_user_preferences(user_id)
    
    flask_time = time.time() - start_time
    print(f"  ⏱️  Flask approach took: {flask_time:.2f}s")
    print(f"  📊 Profile: {profile['name']}")
    print(f"  📊 Orders count: {len(orders)}")
    print(f"  📊 Theme: {preferences['theme']}")
    
    print("\n🚀 FastAPI-style concurrent processing:")
    fastapi_service = FastAPIUserService()
    start_time = time.time()
    
    # All operations run concurrently
    profile, orders, preferences = await asyncio.gather(
        fastapi_service.get_user_profile(user_id),
        fastapi_service.get_user_orders(user_id),
        fastapi_service.get_user_preferences(user_id)
    )
    
    fastapi_time = time.time() - start_time
    print(f"  ⏱️  FastAPI approach took: {fastapi_time:.2f}s")
    print(f"  📊 Profile: {profile['name']}")
    print(f"  📊 Orders count: {len(orders)}")
    print(f"  📊 Theme: {preferences['theme']}")
    print(f"  📈 Speedup: {flask_time/fastapi_time:.2f}x")
    
    print("\n💡 Key migration benefits:")
    print("  • Non-blocking I/O operations")
    print("  • Concurrent request processing")
    print("  • Better resource utilization")
    print("  • Improved application responsiveness")


# Main execution function
async def main():
    """Run all async pattern examples"""
    print("🚀 ASYNC PROGRAMMING PATTERNS")
    print("From Flask to FastAPI Migration Guide")
    print("=" * 60)
    
    examples = [
        ("Basic Async Patterns", basic_async_example),
        ("HTTP Requests", async_http_requests),
        ("Database Patterns", async_database_pattern),
        ("Context Managers", async_context_manager_example),
        ("Error Handling", async_error_handling),
        ("Producer-Consumer", producer_consumer_pattern),
        ("Migration Example", migration_example)
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
            await asyncio.sleep(0.5)  # Brief pause between examples
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            logger.exception(f"Exception in {name}")
    
    print("\n" + "="*60)
    print("✅ ALL ASYNC EXAMPLES COMPLETED")
    print("="*60)
    print("\n📚 Key Takeaways:")
    print("  1. async/await enables concurrent execution")
    print("  2. I/O-bound operations benefit most from async")
    print("  3. Proper error handling is crucial in async code")
    print("  4. Context managers help manage resources")
    print("  5. Producer-consumer patterns handle data flows")
    print("  6. Migration from Flask to FastAPI unlocks concurrency")
    print("\n🎯 Next Steps:")
    print("  • Study the FastAPI example application")
    print("  • Practice converting Flask routes to FastAPI")
    print("  • Implement async database operations")
    print("  • Add proper error handling and logging")


if __name__ == "__main__":
    # Configure event loop for Windows compatibility
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the async main function
    asyncio.run(main())
