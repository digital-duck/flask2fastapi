"""
Utility functions for managing event loops in sync-async interop scenarios

```bash
cd flask2fastapi/book/async-book/src
python -m utils.sync-async-interop.get_event_loop
```


"""


import asyncio
import threading

from utils import timer


async def async_bedrock_call(message: str) -> str:
    """Simulate async Bedrock call"""
    await asyncio.sleep(1)  # Simulate API delay
    return f"AI response to: {message}"


@timer
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
@timer
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


if __name__ == "__main__":
    print("Running sync function with event loop management...")
    sync_function_with_event_loop()

    print("\nDemonstrating hybrid service...")
    demonstrate_hybrid_service()
