"""


Async Exception Handling and Context Managers 

This module demonstrates techniques
- advanced async exception handling (Example 1)
- context management (Example 2)

```bash
cd flask2fastapi/book/async-book/src
python -m utils.exception-handling-context-manager.demos
```

"""

# Example 1: Handle exceptions in async functions
import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp

from utils import *

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@timer
@debug_async_calls
async def fetch_user_data(
    user_id: int, session: aiohttp.ClientSession
) -> Optional[Dict]:
    """
    Simulate fetching user data from an API with exception handling
    """
    url = f"https://jsonplaceholder.typicode.com/users/{user_id}"

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
            if response.status == 404:
                logger.warning(f"User {user_id} not found")
                return None

            response.raise_for_status()  # Raises for 4xx/5xx status codes
            data = await response.json()
            logger.info(f"Successfully fetched user {user_id}: {data['name']}")
            return data

    except asyncio.TimeoutError:
        logger.error(f"Timeout while fetching user {user_id}")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error for user {user_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error for user {user_id}: {e}")
        return None


@timer
async def fetch_multiple_users_with_error_handling(user_ids: List[int]) -> List[Dict]:
    """
    Fetch multiple users concurrently with proper exception handling
    """
    async with aiohttp.ClientSession() as session:
        # Use asyncio.gather with return_exceptions=True to handle partial failures
        tasks = [fetch_user_data(user_id, session) for user_id in user_ids]

        # This won't stop on first exception - collects all results/exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch user {user_ids[i]}: {result}")
            elif result is not None:
                successful_results.append(result)

        return successful_results


# Example 2: Implement async context managers
class AsyncDatabaseConnection:
    """
    Async context manager for database connections
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
        self.transaction = None

    async def __aenter__(self):
        """Enter the async context - establish connection"""
        logger.info(f"🔌 Connecting to database: {self.connection_string}")
        # Simulate async connection setup
        await asyncio.sleep(0.1)
        self.connection = f"Connected to {self.connection_string}"
        logger.info("✅ Database connection established")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context - clean up connection"""
        if exc_type:
            logger.error(f"❌ Exception occurred: {exc_val}")
            if self.transaction:
                logger.info("🔄 Rolling back transaction")
                await asyncio.sleep(0.05)  # Simulate rollback

        logger.info("🔌 Closing database connection")
        await asyncio.sleep(0.05)  # Simulate cleanup
        self.connection = None
        logger.info("✅ Database connection closed")

        # Return False to propagate exceptions, True to suppress them
        return False

    async def begin_transaction(self):
        """Start a database transaction"""
        logger.info("📝 Starting transaction")
        await asyncio.sleep(0.02)
        self.transaction = "active"
        return self.transaction

    async def execute_query(self, query: str):
        """Execute a database query"""
        if not self.connection:
            raise RuntimeError("No database connection")

        logger.info(f"🔍 Executing query: {query}")
        await asyncio.sleep(0.1)  # Simulate query execution

        # Simulate occasional query failures
        if "FAIL" in query.upper():
            raise RuntimeError(f"Query failed: {query}")

        return f"Result for: {query}"


@timer
@debug_async_calls
async def database_operations_with_context_manager():
    """
    Demonstrate async context manager usage with proper exception handling
    """
    # Example 1: Successful operations
    async with AsyncDatabaseConnection("postgresql://localhost:5432/mydb") as db:
        await db.begin_transaction()
        result1 = await db.execute_query("SELECT * FROM users")
        result2 = await db.execute_query("SELECT * FROM posts")
        logger.info(f"Query results: {result1}, {result2}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Operations with failure (demonstrates cleanup)
    try:
        async with AsyncDatabaseConnection("postgresql://localhost:5432/mydb") as db:
            await db.begin_transaction()
            result1 = await db.execute_query("SELECT * FROM users")
            # This will fail and trigger cleanup
            result2 = await db.execute_query("SELECT * FROM invalid_table FAIL")

    except RuntimeError as e:
        logger.error(f"Database operation failed: {e}")


# Bonus: Async context manager for resource pooling
class AsyncResourcePool:
    """
    More advanced async context manager for managing resource pools
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.available_resources = asyncio.Queue(maxsize=max_size)
        self.total_created = 0

    async def __aenter__(self):
        # Try to get an existing resource or create a new one
        try:
            resource = self.available_resources.get_nowait()
            logger.info(f"♻️  Reusing existing resource: {resource}")
        except asyncio.QueueEmpty:
            if self.total_created < self.max_size:
                resource = f"Resource-{self.total_created}"
                self.total_created += 1
                logger.info(f"🆕 Created new resource: {resource}")
            else:
                logger.info("⏳ Waiting for available resource...")
                resource = await self.available_resources.get()
                logger.info(f"♻️  Got resource from pool: {resource}")

        self.current_resource = resource
        return resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Return resource to pool
        await self.available_resources.put(self.current_resource)
        logger.info(f"🔄 Returned resource to pool: {self.current_resource}")


@timer
async def demonstrate_resource_pool():
    """
    Show how async context managers can manage resource pools
    """
    pool = AsyncResourcePool(max_size=2)

    async def worker(worker_id: int):
        async with pool as resource:
            logger.info(f"Worker {worker_id} using {resource}")
            await asyncio.sleep(1)  # Simulate work
            logger.info(f"Worker {worker_id} finished with {resource}")

    # Start 4 workers with only 2 resources - shows pooling in action
    tasks = [worker(i) for i in range(4)]
    await asyncio.gather(*tasks)


# Main execution
async def main():
    print("🔥 Example 1: Exception Handling in Async Functions")
    print("=" * 60)

    # Test with mix of valid and invalid user IDs
    user_ids = [1, 2, 999, 3]  # 999 should return 404
    users = await fetch_multiple_users_with_error_handling(user_ids)
    print(
        f"\n📊 Successfully fetched {len(users)} users out of {len(user_ids)} requested"
    )

    print("\n\n🏗️  Example 2: Async Context Managers")
    print("=" * 60)

    await database_operations_with_context_manager()

    print("\n\n🏊 Bonus: Resource Pool Context Manager")
    print("=" * 60)

    await demonstrate_resource_pool()


# Run the examples
if __name__ == "__main__":
    # Use your async_run_safe helper!
    asyncio.run(main())
