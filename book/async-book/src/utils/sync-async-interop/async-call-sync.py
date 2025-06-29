"""
Example of calling synchronous functions from an asynchronous context using asyncio.to_thread

```bash
cd flask2fastapi/book/async-book/src
python -m utils.sync-async-interop.async-call-sync
```

"""

import asyncio
import concurrent.futures
import functools
import time

import requests  # Sync library

from utils import debug_async_calls, timer


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
@timer
async def async_workflow():
    """Demonstrates calling sync functions from async context"""

    # Method 1: Use asyncio.to_thread for I/O-bound sync calls
    print("Starting async workflow...")

    # Call sync API function without blocking event loop
    result1 = await asyncio.to_thread(
        sync_api_call, "https://jsonplaceholder.typicode.com/posts/1"
    )
    print(f"API result: {result1['title']}")

    # Call sync processing function
    data = [1, 2, 3, 4, 5] * 1000  # Large dataset
    result2 = await asyncio.to_thread(sync_data_processing, data)
    print(f"Processing result: {result2}")

    # Multiple sync calls concurrently
    tasks = [
        asyncio.to_thread(
            sync_api_call, f"https://jsonplaceholder.typicode.com/posts/{i}"
        )
        for i in range(1, 4)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Concurrent results: {len(results)} API calls completed")


@timer
async def async_with_executor():
    """Using executor for more control over sync function execution"""
    loop = asyncio.get_event_loop()

    # Create thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Simple sync function call
        result1 = await loop.run_in_executor(
            executor, sync_data_processing, [1, 2, 3, 4, 5]
        )

        # Sync function with multiple arguments using functools.partial
        sync_function_with_args = functools.partial(
            sync_api_call, "https://jsonplaceholder.typicode.com/posts/1"
        )
        result2 = await loop.run_in_executor(executor, sync_function_with_args)

        print(f"Executor results: {result1}, {result2['id']}")


# Testing pattern for async functions
@timer
async def test_async_function():
    """Example of testing async code with sync assertions"""

    # Your async function under test
    async def fetch_user_data(user_id: int):
        # Simulate async database call
        await asyncio.sleep(1)
        return {"id": user_id, "name": f"User {user_id}"}

    # Test it by calling sync assertion functions
    result = await fetch_user_data(123)

    # Call sync test utilities
    assert_result = await asyncio.to_thread(
        lambda data: data["id"] == 123 and "name" in data, result
    )

    print(f"Test passed: {assert_result}")


# Run the async workflow
async def main():
    sep = "=" * 60
    print(f"\n{sep}\n🔥 Starting async call to sync functions ...")
    await async_workflow()

    print(f"\n{sep}\n🔥 Starting async_with_executor() ...")
    await async_with_executor()

    print(f"\n{sep}\n🔥 Starting test_async_function() ...")
    await test_async_function()


# Execute
if __name__ == "__main__":
    asyncio.run(main())
