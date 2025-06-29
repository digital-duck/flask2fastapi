import asyncio


async def async_bedrock_call(message: str) -> str:
    """Simulate async Bedrock call"""
    await asyncio.sleep(1)  # Simulate API delay
    return f"AI response to: {message}"


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
    # Usage
    sync_bedrock_call = create_sync_wrapper(async_bedrock_call)
    result = sync_bedrock_call("test from sync wrapper")
    print(f"\n\nWrapper result: {result}")

    print("\n\nTesting quick async verification...")
    quick_verify_async_function()

    print("\n\nTesting async generator...")
    test_async_generator()
