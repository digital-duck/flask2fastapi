# Chapter 9, Section 4.3a: Advanced Caching System

## Overview

This subsection implements a sophisticated multi-level caching system with semantic similarity matching, provider-aware caching, and intelligent TTL management to optimize AI service performance and reduce costs.

## Smart Cache Implementation

### Core Cache Components

```python
# services/ai_cache_service.py
import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import structlog
from collections import defaultdict, OrderedDict

logger = structlog.get_logger()

class CacheStrategy(str, Enum):
    NONE = "none"
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"
    PROVIDER_AWARE = "provider_aware"

class CacheLevel(str, Enum):
    L1_MEMORY = "l1_memory"        # In-memory cache (fastest)
    L2_REDIS = "l2_redis"          # Redis cache (fast)
    L3_DATABASE = "l3_database"    # Database cache (persistent)

@dataclass
class CacheEntry:
    """Enhanced cache entry with comprehensive metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_hash: Optional[str] = None
    response_time: Optional[float] = None
    cost_estimate: Optional[float] = None
    task_type: Optional[str] = None
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at

class AdvancedCacheService:
    """Multi-level caching with semantic similarity and provider awareness"""
    
    def __init__(self, max_l1_size: int = 1000, default_ttl: float = 3600):
        self.max_l1_size = max_l1_size
        self.default_ttl = default_ttl
        
        # L1 Cache: In-memory with LRU eviction
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Cache statistics
        self.stats = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
            'evictions': defaultdict(int),
            'total_requests': 0,
            'total_savings_estimate': 0.0,
            'time_savings_estimate': 0.0
        }
        
        # Semantic similarity threshold for fuzzy matching
        self.similarity_threshold = 0.85
        
        # Provider-specific cache settings
        self.provider_cache_config = {
            'aws_bedrock': {'ttl': 7200, 'enabled': True},
            'azure_openai': {'ttl': 3600, 'enabled': True},
            'gcp_vertex': {'ttl': 1800, 'enabled': True}
        }
        
        # Task-specific cache settings
        self.task_cache_config = {
            'chat': {'ttl': 1800, 'enabled': True},
            'code_generation': {'ttl': 7200, 'enabled': True},
            'analysis': {'ttl': 14400, 'enabled': True},
            'creative_writing': {'ttl': 3600, 'enabled': False},  # Less cacheable
            'multimodal': {'ttl': 3600, 'enabled': True},
            'embeddings': {'ttl': 86400, 'enabled': True},  # Very cacheable
            'summarization': {'ttl': 7200, 'enabled': True}
        }
    
    def _generate_cache_key(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        strategy: CacheStrategy = CacheStrategy.SIMPLE
    ) -> str:
        """Generate cache key based on strategy"""
        
        if strategy == CacheStrategy.SIMPLE:
            # Simple hash of prompt
            return hashlib.md5(prompt.encode()).hexdigest()
        
        elif strategy == CacheStrategy.PROVIDER_AWARE:
            # Include provider and model in key
            key_components = [prompt, provider or "", model or ""]
            combined = "|".join(key_components)
            return hashlib.md5(combined.encode()).hexdigest()
        
        elif strategy == CacheStrategy.SEMANTIC:
            # Normalize prompt for semantic matching
            normalized = self._normalize_prompt(prompt)
            return hashlib.md5(normalized.encode()).hexdigest()
        
        elif strategy == CacheStrategy.ADAPTIVE:
            # Adaptive key based on prompt characteristics
            if len(prompt) < 100:
                # Short prompts: exact matching
                return hashlib.md5(prompt.encode()).hexdigest()
            else:
                # Long prompts: semantic matching
                normalized = self._normalize_prompt(prompt)
                return hashlib.md5(normalized.encode()).hexdigest()
        
        else:
            return hashlib.md5(prompt.encode()).hexdigest()
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for semantic caching"""
        import re
        
        # Convert to lowercase
        normalized = prompt.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove common variations that don't affect meaning
        replacements = [
            (r'\bplease\b', ''),
            (r'\bcan you\b', ''),
            (r'\bcould you\b', ''),
            (r'\bwould you\b', ''),
            (r'\bthanks?\b', ''),
            (r'\bthank you\b', ''),
        ]
        
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove extra spaces again
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Jaccard similarity"""
        
        if not text1 or not text2:
            return 0.0
        
        # Jaccard similarity on word level
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
```

## Cache Operations

### Get and Set Operations

```python
    async def get(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ) -> Optional[Dict[str, Any]]:
        """Get cached response with advanced matching"""
        
        self.stats['total_requests'] += 1
        
        # Check if caching is enabled for this configuration
        if not self._is_cache_enabled(provider, task_type):
            self.stats['misses']['disabled'] += 1
            return None
        
        # Try exact key match first
        cache_key = self._generate_cache_key(prompt, provider, model, strategy)
        
        # L1 Cache lookup
        if cache_key in self.l1_cache:
            entry = self.l1_cache[cache_key]
            
            if entry.is_expired():
                # Remove expired entry
                del self.l1_cache[cache_key]
                self.stats['misses']['expired'] += 1
                logger.debug("Cache entry expired", key=cache_key[:8])
            else:
                # Move to end (LRU)
                self.l1_cache.move_to_end(cache_key)
                entry.update_access()
                
                self.stats['hits']['l1'] += 1
                self.stats['total_savings_estimate'] += entry.cost_estimate or 0
                self.stats['time_savings_estimate'] += entry.response_time or 0
                
                logger.debug(
                    "L1 cache hit",
                    key=cache_key[:8],
                    access_count=entry.access_count,
                    age=entry.get_age()
                )
                
                # Add cache metadata to response
                response = entry.value.copy() if isinstance(entry.value, dict) else {'response': entry.value}
                response.update({
                    'cache_hit': True,
                    'cache_level': entry.cache_level.value,
                    'cache_age': entry.get_age(),
                    'cache_access_count': entry.access_count
                })
                
                return response
        
        # Try semantic/fuzzy matching if enabled
        if strategy in [CacheStrategy.SEMANTIC, CacheStrategy.ADAPTIVE]:
            fuzzy_result = await self._fuzzy_cache_lookup(
                prompt, provider, model, task_type
            )
            if fuzzy_result:
                self.stats['hits']['fuzzy'] += 1
                return fuzzy_result
        
        # Cache miss
        self.stats['misses']['not_found'] += 1
        logger.debug("Cache miss", key=cache_key[:8], strategy=strategy.value)
        
        return None
    
    async def set(
        self,
        prompt: str,
        response: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        response_time: Optional[float] = None,
        cost_estimate: Optional[float] = None,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        custom_ttl: Optional[float] = None
    ) -> bool:
        """Store response in cache with metadata"""
        
        # Check if caching is enabled
        if not self._is_cache_enabled(provider, task_type):
            return False
        
        cache_key = self._generate_cache_key(prompt, provider, model, strategy)
        
        # Determine TTL
        ttl = custom_ttl or self._get_ttl(provider, task_type)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=response,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            provider=provider,
            model=model,
            prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
            response_time=response_time,
            cost_estimate=cost_estimate,
            task_type=task_type,
            cache_level=CacheLevel.L1_MEMORY,
            ttl=ttl,
            tags=self._generate_tags(prompt, provider, model, task_type)
        )
        
        # Store normalized prompt for fuzzy matching
        if strategy in [CacheStrategy.SEMANTIC, CacheStrategy.ADAPTIVE]:
            entry.normalized_prompt = self._normalize_prompt(prompt)
            entry.original_prompt = prompt
        
        # Add to L1 cache
        self.l1_cache[cache_key] = entry
        
        # Enforce size limit with LRU eviction
        while len(self.l1_cache) > self.max_l1_size:
            oldest_key = next(iter(self.l1_cache))
            evicted_entry = self.l1_cache.pop(oldest_key)
            self.stats['evictions']['l1_size'] += 1
            
            logger.debug(
                "Cache eviction (size limit)",
                evicted_key=oldest_key[:8],
                cache_size=len(self.l1_cache),
                evicted_age=evicted_entry.get_age()
            )
        
        logger.debug(
            "Cache entry stored",
            key=cache_key[:8],
            provider=provider,
            task_type=task_type,
            ttl=ttl,
            cache_size=len(self.l1_cache)
        )
        
        return True
```

## Cache Management

### Helper Methods and Configuration

```python
    async def _fuzzy_cache_lookup(
        self,
        prompt: str,
        provider: Optional[str],
        model: Optional[str],
        task_type: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Perform fuzzy/semantic cache lookup"""
        
        normalized_prompt = self._normalize_prompt(prompt)
        best_match = None
        best_score = 0
        
        for cache_key, entry in self.l1_cache.items():
            if entry.is_expired():
                continue
            
            # Check provider compatibility
            if provider and entry.provider and entry.provider != provider:
                continue
            
            # Calculate similarity if normalized prompt exists
            if hasattr(entry, 'normalized_prompt'):
                score = self._calculate_similarity(normalized_prompt, entry.normalized_prompt)
                if score > best_score and score >= self.similarity_threshold:
                    best_score = score
                    best_match = entry
        
        if best_match:
            best_match.update_access()
            logger.debug(
                "Fuzzy cache hit",
                similarity_score=best_score,
                original_prompt_length=len(prompt),
                cached_prompt_length=len(getattr(best_match, 'original_prompt', ''))
            )
            
            response = best_match.value.copy() if isinstance(best_match.value, dict) else {'response': best_match.value}
            response.update({
                'cache_hit': True,
                'cache_level': 'l1_fuzzy',
                'cache_similarity': best_score,
                'cache_age': best_match.get_age()
            })
            
            return response
        
        return None
    
    def _is_cache_enabled(self, provider: Optional[str], task_type: Optional[str]) -> bool:
        """Check if caching is enabled for given provider and task type"""
        
        # Check provider-specific settings
        if provider:
            provider_config = self.provider_cache_config.get(provider, {})
            if not provider_config.get('enabled', True):
                return False
        
        # Check task-specific settings
        if task_type:
            task_config = self.task_cache_config.get(task_type, {})
            if not task_config.get('enabled', True):
                return False
        
        return True
    
    def _get_ttl(self, provider: Optional[str], task_type: Optional[str]) -> float:
        """Get TTL based on provider and task type"""
        
        ttl = self.default_ttl
        
        # Apply provider-specific TTL
        if provider:
            provider_config = self.provider_cache_config.get(provider, {})
            ttl = provider_config.get('ttl', ttl)
        
        # Apply task-specific TTL (takes precedence)
        if task_type:
            task_config = self.task_cache_config.get(task_type, {})
            ttl = task_config.get('ttl', ttl)
        
        return ttl
    
    def _generate_tags(
        self,
        prompt: str,
        provider: Optional[str],
        model: Optional[str],
        task_type: Optional[str]
    ) -> List[str]:
        """Generate tags for cache entry organization"""
        
        tags = []
        
        if provider:
            tags.append(f"provider:{provider}")
        if model:
            tags.append(f"model:{model}")
        if task_type:
            tags.append(f"task:{task_type}")
        
        # Add prompt-based tags
        prompt_lower = prompt.lower()
        if len(prompt) < 100:
            tags.append("short_prompt")
        elif len(prompt) > 1000:
            tags.append("long_prompt")
        
        if any(word in prompt_lower for word in ['analyze', 'analysis']):
            tags.append("analytical")
        if any(word in prompt_lower for word in ['create', 'generate', 'write']):
            tags.append("generative")
        
        return tags
    
    def clear_expired(self) -> int:
        """Remove expired entries and return count"""
        
        expired_keys = []
        
        for key, entry in self.l1_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.l1_cache[key]
            self.stats['evictions']['expired'] += 1
        
        if expired_keys:
            logger.info(
                "Cleared expired cache entries",
                count=len(expired_keys),
                remaining_entries=len(self.l1_cache)
            )
        
        return len(expired_keys)
    
    def clear_by_tags(self, tags: List[str]) -> int:
        """Clear cache entries matching any of the provided tags"""
        
        cleared_keys = []
        
        for key, entry in self.l1_cache.items():
            if any(tag in entry.tags for tag in tags):
                cleared_keys.append(key)
        
        for key in cleared_keys:
            del self.l1_cache[key]
            self.stats['evictions']['manual'] += 1
        
        logger.info(
            "Cleared cache entries by tags",
            tags=tags,
            count=len(cleared_keys)
        )
        
        return len(cleared_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        total_hits = sum(self.stats['hits'].values())
        total_misses = sum(self.stats['misses'].values())
        total_requests = total_hits + total_misses
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate cache efficiency metrics
        current_time = time.time()
        cache_ages = [entry.get_age() for entry in self.l1_cache.values()]
        avg_age = sum(cache_ages) / len(cache_ages) if cache_ages else 0
        
        access_counts = [entry.access_count for entry in self.l1_cache.values()]
        avg_access_count = sum(access_counts) / len(access_counts) if access_counts else 0
        
        return {
            'cache_size': len(self.l1_cache),
            'max_size': self.max_l1_size,
            'hit_rate_percent': hit_rate,
            'total_requests': total_requests,
            'hits_by_type': dict(self.stats['hits']),
            'misses_by_type': dict(self.stats['misses']),
            'evictions_by_type': dict(self.stats['evictions']),
            'estimated_cost_savings': self.stats['total_savings_estimate'],
            'estimated_time_savings': self.stats['time_savings_estimate'],
            'average_entry_age': avg_age,
            'average_access_count': avg_access_count,
            'memory_usage_estimate': len(self.l1_cache) * 1024,  # Rough estimate
        }
```

## Usage Examples

### Basic Cache Usage

```python
async def demonstrate_caching():
    """Demonstrate cache functionality"""
    
    cache = AdvancedCacheService(max_l1_size=100)
    
    # Test different caching strategies
    prompts = [
        "What is machine learning?",
        "Can you explain machine learning?",  # Similar to above
        "Write Python code for sorting",
        "Create a sorting algorithm in Python"  # Similar to above
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Request {i+1}: {prompt} ---")
        
        # Try to get from cache
        cached = await cache.get(
            prompt=prompt,
            provider="aws_bedrock",
            task_type="chat",
            strategy=CacheStrategy.SEMANTIC
        )
        
        if cached:
            print(f"✅ Cache hit! Age: {cached.get('cache_age', 0):.1f}s")
        else:
            print("❌ Cache miss - making API call")
            
            # Simulate API response
            response = {"response": f"Mock response for: {prompt}"}
            
            # Store in cache
            await cache.set(
                prompt=prompt,
                response=response,
                provider="aws_bedrock",
                task_type="chat",
                response_time=1.5,
                cost_estimate=0.001,
                strategy=CacheStrategy.SEMANTIC
            )
            
            print("💾 Stored in cache")
    
    # Display cache statistics
    stats = cache.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"Hit Rate: {stats['hit_rate_percent']:.1f}%")
    print(f"Cache Size: {stats['cache_size']}/{stats['max_size']}")
    print(f"Estimated Savings: ${stats['estimated_cost_savings']:.4f}")

if __name__ == "__main__":
    asyncio.run(demonstrate_caching())
```

This caching system provides intelligent response caching with semantic similarity matching, provider-aware configurations, and comprehensive statistics tracking. The next sub-section will cover the analytics and recommendation system.