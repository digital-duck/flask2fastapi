"""
Code Example Template
Chapter X: Chapter Title

Description: Brief description of what this code demonstrates

Learning objectives:
- Objective 1
- Objective 2

Usage:
    python code_example_template.py
"""

import asyncio
from typing import List, Optional, Dict, Any


class ExampleClass:
    """Example class demonstrating patterns"""
    
    def __init__(self, name: str):
        self.name = name
        self.data: Dict[str, Any] = {}
    
    async def async_method(self) -> str:
        """Example async method"""
        await asyncio.sleep(0.1)  # Simulate async operation
        return f"Hello from {self.name}"
    
    def sync_method(self) -> str:
        """Example synchronous method for comparison"""
        return f"Sync hello from {self.name}"


async def demonstrate_async_patterns():
    """Demonstrate async programming patterns"""
    print("Async Patterns Demonstration")
    print("=" * 30)
    
    # Create example instances
    examples = [
        ExampleClass("FastAPI"),
        ExampleClass("AsyncIO"),
        ExampleClass("Pydantic")
    ]
    
    # Sequential execution
    print("\nSequential execution:")
    for example in examples:
        result = await example.async_method()
        print(f"  {result}")
    
    # Concurrent execution
    print("\nConcurrent execution:")
    tasks = [example.async_method() for example in examples]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"  {result}")


def demonstrate_sync_patterns():
    """Demonstrate synchronous patterns for comparison"""
    print("\nSync Patterns Demonstration")
    print("=" * 30)
    
    examples = [
        ExampleClass("Flask"),
        ExampleClass("Django"),
        ExampleClass("Traditional")
    ]
    
    for example in examples:
        result = example.sync_method()
        print(f"  {result}")


async def main():
    """Main function demonstrating usage"""
    print("Code Example: Async vs Sync Patterns")
    print("=" * 40)
    
    # Demonstrate both patterns
    await demonstrate_async_patterns()
    demonstrate_sync_patterns()
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
