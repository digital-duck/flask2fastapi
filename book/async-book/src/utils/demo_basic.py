#!/usr/bin/env python
# coding: utf-8
""" 

Basic async function example for demonstration purposes. 

```bash
cd flask2fastapi/book/async-book/src
python -m utils.demo_basic
```
Output:
```
⏱️  process_multiple took 6.000 seconds
Sync results: ['completed in 0', 'completed in 1', 'completed in 2', 'completed in 3']
⏱️  process_multiple_async took 3.002 seconds
Async results: ['completed in 0', 'completed in 1', 'completed in 2', 'completed in 3']
```

"""


import asyncio
import time

from utils import *

# Exercise 1: Basic async function


# Convert this synchronous function
def fetch_data(wait_sec: int):
    time.sleep(wait_sec)  # Simulate API call
    return f"completed in {wait_sec}"


@timer
def process_multiple(num_calls: int):
    results = []
    for i in range(num_calls):
        results.append(fetch_data(i))
    return results


# To this async version
async def fetch_data_async(wait_sec):
    await asyncio.sleep(wait_sec)  # Non-blocking
    return f"completed in {wait_sec}"


@timer
async def process_multiple_async(num_calls: int):
    tasks = [fetch_data_async(i) for i in range(num_calls)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    res = process_multiple(4)
    print("Sync results:", res)

    async_res = asyncio.run(process_multiple_async(4))
    print("Async results:", async_res)
