"""
Sync code that calls async functions using asyncio.run() and aiohttp for HTTP requests 

```bash
cd flask2fastapi/book/async-book/src
python -m utils.sync-call-async.async-call-sync
```

"""

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
    result1 = asyncio.run(
        async_api_call("https://jsonplaceholder.typicode.com/posts/1")
    )
    print(f"Sync got async result: {result1['title']}")

    # Method 2: Multiple async calls
    async def multiple_async_calls():
        tasks = [
            async_api_call("https://jsonplaceholder.typicode.com/posts/1"),
            async_api_call("https://jsonplaceholder.typicode.com/posts/2"),
            async_bedrock_call("Hello from sync world"),
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
            "timestamp": asyncio.get_event_loop().time(),
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
