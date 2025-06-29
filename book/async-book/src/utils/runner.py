"""
Utility functions for running async code in a synchronous context

```bash
cd flask2fastapi/book/async-book/src
python -m utils.runner
```

"""

import asyncio


async def async_bedrock_call(message: str) -> str:
    """Simulate async Bedrock call"""
    await asyncio.sleep(1)  # Simulate API delay
    return f"AI response to: {message}"


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


# Quick test multiple functions
async def test_func2():
    await asyncio.sleep(0.1)
    return "test result 2"


# Usage for quick testing
runner = AsyncRunner()

# Quick test individual async functions
result1 = runner.run(async_bedrock_call, "test message 1")
print(f"\nQuick test 1: {result1}")

results = runner.run_multiple(async_bedrock_call("test message A"), test_func2())
print(f"\nMultiple quick tests: {results}")
