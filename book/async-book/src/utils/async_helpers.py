# async_helpers.py
import asyncio
import functools
import time
from typing import Any, Callable, Coroutine

# Timing Decorator


def timer(func: Callable) -> Callable:
    """Simple timing decorator for development"""

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"⏱️  {func.__name__} took {duration:.3f} seconds")
        return result

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = time.perf_counter() - start
        print(f"⏱️  {func.__name__} took {duration:.3f} seconds")
        return result

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# def timer(func):
#     """Decorator to time function"""
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         print(f"{func.__name__} took {end - start:.3f} seconds")
#         return result
#     return wrapper

# def async_timer(func):
#     """Decorator to time async functions"""
#     @functools.wraps(func)
#     async def wrapper(*args, **kwargs):
#         start = time.time()
#         result = await func(*args, **kwargs)
#         end = time.time()
#         print(f"{func.__name__} took {end - start:.3f} seconds")
#         return result
#     return wrapper


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


async def async_run_safe(coro: Coroutine) -> Any:
    """
    Safely run async code in both Jupyter notebooks and regular Python scripts.

    In Jupyter: Uses await (since event loop is already running)
    In scripts: Uses asyncio.run() to create new event loop
    """
    try:
        loop = asyncio.get_running_loop()
        print("📓 Running in existing event loop (likely Jupyter)")
        return await coro
    except RuntimeError:
        print("🐍 No event loop found, creating new one (likely script)")
        return asyncio.run(coro)
