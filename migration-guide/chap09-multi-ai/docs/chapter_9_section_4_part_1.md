# Chapter 9, Section 4.1: Core Router Architecture and Provider Management

## Overview

This subsection establishes the foundation for intelligent AI provider orchestration. We'll implement the core router architecture, provider statistics tracking, and basic routing strategies that form the backbone of our multi-provider system.

## Core Components

### Provider Definitions and Statistics

```python
# services/ai_router_service.py
from typing import Dict, Any, Optional, List, Tuple
import structlog
from enum import Enum
import asyncio
import time
import random
from dataclasses import dataclass, field
from collections import defaultdict
import json
import hashlib

from services.aws_bedrock_service import BedrockService
from services.azure_openai_service import AzureOpenAIService
from services.gcp_vertex_service import VertexAIService

logger = structlog.get_logger()

class AIProvider(str, Enum):
    AWS_BEDROCK = "aws_bedrock"
    AZURE_OPENAI = "azure_openai"
    GCP_VERTEX = "gcp_vertex"

class RoutingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    PERFORMANCE_BASED = "performance_based"
    COST_OPTIMIZED = "cost_optimized"
    AVAILABILITY_BASED = "availability_based"
    MODEL_SPECIFIC = "model_specific"
    INTELLIGENT = "intelligent"
    WEIGHTED_RANDOM = "weighted_random"

class TaskType(str, Enum):
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATIVE_WRITING = "creative_writing"
    MULTIMODAL = "multimodal"
    EMBEDDINGS = "embeddings"
    SUMMARIZATION = "summarization"

@dataclass
class ProviderStats:
    """Comprehensive provider statistics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    total_failures: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0
    is_healthy: bool = True
    cost_per_token: float = 0.0
    success_rate: float = 1.0
    response_times: List[float] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)
    
    def add_request_result(self, success: bool, response_time: float, error: Optional[str] = None):
        """Add a request result to the stats with exponential moving averages"""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.response_times.append(response_time)
            
            # Keep only last 100 response times for percentile calculations
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            # Update moving average with exponential decay
            if self.avg_response_time == 0:
                self.avg_response_time = response_time
            else:
                # Exponential moving average (alpha = 0.1)
                self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time
        else:
            self.total_failures += 1
            if error:
                self.recent_errors.append(f"{time.time()}: {error}")
                # Keep only last 10 errors with timestamps
                if len(self.recent_errors) > 10:
                    self.recent_errors = self.recent_errors[-10:]
        
        # Update success rate
        self.success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 1.0
    
    def get_p95_response_time(self) -> float:
        """Get 95th percentile response time"""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    def get_health_score(self) -> float:
        """Calculate composite health score (0-1)"""
        if self.total_requests == 0:
            return 0.8  # Neutral score for untested providers
        
        # Base score from success rate
        base_score = self.success_rate
        
        # Penalty for high response times (normalize to 10 seconds max)
        response_penalty = min(0.3, self.avg_response_time / 10.0 * 0.3)
        
        # Recent activity bonus (more recent requests = more reliable data)
        hours_since_last = (time.time() - self.last_health_check) / 3600
        staleness_penalty = min(0.2, hours_since_last / 24 * 0.2)
        
        return max(0, base_score - response_penalty - staleness_penalty)

class AIServiceRouter:
    """Advanced AI service router with intelligent provider selection"""
    
    def __init__(self):
        # Initialize provider services
        self.providers = {
            AIProvider.AWS_BEDROCK: BedrockService(),
            AIProvider.AZURE_OPENAI: AzureOpenAIService(),
            AIProvider.GCP_VERTEX: VertexAIService()
        }
        
        # Provider statistics tracking
        self.provider_stats = {
            provider: ProviderStats()
            for provider in AIProvider
        }
        
        # Round-robin state
        self.round_robin_index = 0
        
        # Provider capabilities and strengths
        self.model_capabilities = {
            AIProvider.AWS_BEDROCK: {
                'models': ['claude-3-5-sonnet', 'claude-3-haiku', 'llama-3-1', 'amazon-titan'],
                'strengths': ['reasoning', 'analysis', 'coding', 'rag'],
                'max_context': 200000,
                'multimodal': False,
                'enterprise_ready': True
            },
            AIProvider.AZURE_OPENAI: {
                'models': ['gpt-4-turbo', 'gpt-4', 'gpt-35-turbo', 'gpt-4-vision'],
                'strengths': ['creativity', 'function_calling', 'multimodal', 'enterprise'],
                'max_context': 128000,
                'multimodal': True,
                'enterprise_ready': True
            },
            AIProvider.GCP_VERTEX: {
                'models': ['gemini-1.0-pro', 'gemini-1.0-pro-vision', 'text-bison@001', 'code-bison@001'],
                'strengths': ['cost_effective', 'multimodal', 'coding', 'fast'],
                'max_context': 32000,
                'multimodal': True,
                'enterprise_ready': False
            }
        }
        
        # Task-specific provider preferences (ordered by preference)
        self.task_preferences = {
            TaskType.CHAT: [AIProvider.AZURE_OPENAI, AIProvider.GCP_VERTEX, AIProvider.AWS_BEDROCK],
            TaskType.CODE_GENERATION: [AIProvider.AWS_BEDROCK, AIProvider.GCP_VERTEX, AIProvider.AZURE_OPENAI],
            TaskType.ANALYSIS: [AIProvider.AWS_BEDROCK, AIProvider.AZURE_OPENAI, AIProvider.GCP_VERTEX],
            TaskType.CREATIVE_WRITING: [AIProvider.AZURE_OPENAI, AIProvider.AWS_BEDROCK, AIProvider.GCP_VERTEX],
            TaskType.MULTIMODAL: [AIProvider.GCP_VERTEX, AIProvider.AZURE_OPENAI],
            TaskType.EMBEDDINGS: [AIProvider.GCP_VERTEX, AIProvider.AZURE_OPENAI],
            TaskType.SUMMARIZATION: [AIProvider.AWS_BEDROCK, AIProvider.AZURE_OPENAI, AIProvider.GCP_VERTEX]
        }
        
        # Pricing information (per 1K tokens) - updated regularly
        self.pricing = {
            AIProvider.AWS_BEDROCK: {
                'claude-3-5-sonnet': {'input': 0.003, 'output': 0.015},
                'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
                'llama-3-1': {'input': 0.00065, 'output': 0.00065},
                'amazon-titan': {'input': 0.0008, 'output': 0.0016}
            },
            AIProvider.AZURE_OPENAI: {
                'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
                'gpt-4': {'input': 0.03, 'output': 0.06},
                'gpt-35-turbo': {'input': 0.0015, 'output': 0.002},
                'gpt-4-vision': {'input': 0.01, 'output': 0.03}
            },
            AIProvider.GCP_VERTEX: {
                'gemini-1.0-pro': {'input': 0.00025, 'output': 0.0005},
                'gemini-1.0-pro-vision': {'input': 0.00025, 'output': 0.0005},
                'text-bison@001': {'input': 0.0005, 'output': 0.0005},
                'code-bison@001': {'input': 0.0005, 'output': 0.0005}
            }
        }
        
        # Dynamic provider weights for weighted random selection
        self.provider_weights = {
            AIProvider.AWS_BEDROCK: 1.0,
            AIProvider.AZURE_OPENAI: 1.0,
            AIProvider.GCP_VERTEX: 1.0
        }
    
    async def initialize(self):
        """Initialize router with health checks"""
        logger.info("Initializing AI Service Router")
        
        # Perform initial health checks
        await self._update_health_status()
        
        # Log initialization status
        healthy_providers = [
            provider for provider in AIProvider
            if self.provider_stats[provider].is_healthy
        ]
        
        logger.info(
            "Router initialized",
            total_providers=len(AIProvider),
            healthy_providers=len(healthy_providers),
            provider_status={
                provider.value: self.provider_stats[provider].is_healthy
                for provider in AIProvider
            }
        )
```

## Basic Routing Strategies

### Simple Strategy Implementations

```python
    def _round_robin_selection(self) -> AIProvider:
        """Round-robin provider selection with health checking"""
        providers = [p for p in AIProvider if self.provider_stats[p].is_healthy]
        
        if not providers:
            # Fallback to all providers if none are healthy
            providers = list(AIProvider)
        
        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider
    
    def _weighted_random_selection(self) -> AIProvider:
        """Select provider using weighted random selection based on performance"""
        
        # Calculate adjusted weights based on recent performance
        adjusted_weights = {}
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            base_weight = self.provider_weights[provider]
            
            # Adjust weight based on health score
            health_score = stats.get_health_score()
            adjusted_weights[provider] = base_weight * health_score
        
        # Select based on weights
        total_weight = sum(adjusted_weights.values())
        if total_weight == 0:
            return random.choice(list(AIProvider))
        
        rand_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for provider, weight in adjusted_weights.items():
            cumulative_weight += weight
            if rand_value <= cumulative_weight:
                return provider
        
        return list(AIProvider)[0]  # Fallback
    
    def _cost_optimized_selection(self, max_cost: Optional[float]) -> List[AIProvider]:
        """Select providers based on cost optimization"""
        
        cost_scores = {}
        
        for provider in AIProvider:
            if provider not in self.pricing:
                continue
            
            # Calculate average cost across all models for this provider
            provider_models = self.pricing[provider]
            avg_input_cost = sum(model['input'] for model in provider_models.values()) / len(provider_models)
            avg_output_cost = sum(model['output'] for model in provider_models.values()) / len(provider_models)
            avg_total_cost = (avg_input_cost + avg_output_cost) / 2
            
            # Filter by max cost if specified
            if max_cost and avg_total_cost > max_cost:
                continue
            
            # Only consider healthy providers
            if not self.provider_stats[provider].is_healthy:
                continue
            
            cost_scores[provider] = avg_total_cost
        
        # Sort by cost (ascending - lower cost first)
        sorted_providers = sorted(cost_scores.keys(), key=lambda p: cost_scores[p])
        return sorted_providers or [AIProvider.GCP_VERTEX]  # Default to most cost-effective
    
    def _model_specific_selection(self, task_type: TaskType) -> List[AIProvider]:
        """Select providers based on model capabilities for specific tasks"""
        
        if task_type in self.task_preferences:
            # Filter by health status
            healthy_preferences = [
                provider for provider in self.task_preferences[task_type]
                if self.provider_stats[provider].is_healthy
            ]
            
            if healthy_preferences:
                return healthy_preferences
        
        # Fallback to all healthy providers
        return [p for p in AIProvider if self.provider_stats[p].is_healthy] or list(AIProvider)
```

## Health Monitoring

### Provider Health Management

```python
    async def _update_health_status(self):
        """Update health status for all providers with parallel checks"""
        
        current_time = time.time()
        
        # Determine which providers need health checks (every 5 minutes)
        health_check_tasks = []
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            if current_time - stats.last_health_check > 300:  # 5 minutes
                health_check_tasks.append(self._check_provider_health(provider))
        
        if health_check_tasks:
            # Run health checks in parallel
            results = await asyncio.gather(*health_check_tasks, return_exceptions=True)
            
            # Log any health check failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    provider = list(AIProvider)[i]
                    logger.error(
                        "Health check exception",
                        provider=provider,
                        error=str(result)
                    )
    
    async def _check_provider_health(self, provider: AIProvider):
        """Check health of a specific provider with timeout"""
        
        try:
            service = self.providers[provider]
            
            # Health check with timeout
            health_result = await asyncio.wait_for(
                service.health_check(), 
                timeout=10.0  # 10 second timeout
            )
            
            stats = self.provider_stats[provider]
            stats.is_healthy = health_result.get('status') == 'healthy'
            stats.last_health_check = time.time()
            
            # Extract additional metrics if available
            if 'response_time' in health_result:
                stats.add_request_result(True, health_result['response_time'])
            
            logger.debug(
                "Provider health updated",
                provider=provider,
                healthy=stats.is_healthy,
                response_time=health_result.get('response_time')
            )
            
        except asyncio.TimeoutError:
            logger.warning(
                "Health check timeout",
                provider=provider,
                timeout=10.0
            )
            self.provider_stats[provider].is_healthy = False
            self.provider_stats[provider].last_health_check = time.time()
            
        except Exception as e:
            logger.error(
                "Health check failed",
                provider=provider,
                error=str(e)
            )
            self.provider_stats[provider].is_healthy = False
            self.provider_stats[provider].last_health_check = time.time()
    
    async def force_health_check(self, provider: Optional[AIProvider] = None):
        """Force immediate health check for specific provider or all providers"""
        
        if provider:
            await self._check_provider_health(provider)
        else:
            # Check all providers
            tasks = [self._check_provider_health(p) for p in AIProvider]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_provider_health_summary(self) -> Dict[str, Any]:
        """Get summary of all provider health statuses"""
        
        summary = {
            'healthy_providers': [],
            'unhealthy_providers': [],
            'total_providers': len(AIProvider),
            'overall_health_percentage': 0
        }
        
        healthy_count = 0
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            provider_info = {
                'provider': provider.value,
                'is_healthy': stats.is_healthy,
                'health_score': stats.get_health_score(),
                'last_check': stats.last_health_check,
                'success_rate': stats.success_rate,
                'avg_response_time': stats.avg_response_time
            }
            
            if stats.is_healthy:
                summary['healthy_providers'].append(provider_info)
                healthy_count += 1
            else:
                summary['unhealthy_providers'].append(provider_info)
        
        summary['overall_health_percentage'] = (healthy_count / len(AIProvider)) * 100
        
        return summary
```

## Usage Examples

### Basic Router Usage

```python
# Example usage of the core router functionality

async def demonstrate_core_router():
    """Demonstrate core router functionality"""
    
    # Initialize router
    router = AIServiceRouter()
    await router.initialize()
    
    # Check provider health
    health_summary = router.get_provider_health_summary()
    print(f"Router Health: {health_summary['overall_health_percentage']:.1f}% healthy")
    
    # Test different routing strategies
    strategies = [
        RoutingStrategy.ROUND_ROBIN,
        RoutingStrategy.COST_OPTIMIZED,
        RoutingStrategy.WEIGHTED_RANDOM
    ]
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy:")
        
        # Select provider using strategy
        if strategy == RoutingStrategy.ROUND_ROBIN:
            provider = router._round_robin_selection()
            print(f"Selected: {provider.value}")
        
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            providers = router._cost_optimized_selection(max_cost=None)
            print(f"Cost-ordered providers: {[p.value for p in providers]}")
        
        elif strategy == RoutingStrategy.WEIGHTED_RANDOM:
            provider = router._weighted_random_selection()
            print(f"Randomly selected: {provider.value}")

if __name__ == "__main__":
    asyncio.run(demonstrate_core_router())
```

## Key Features

This core router architecture provides:

1. **Comprehensive Statistics**: Detailed tracking of provider performance, health, and costs
2. **Health Monitoring**: Automated health checks with timeout handling and parallel execution
3. **Basic Routing Strategies**: Round-robin, cost-optimized, and weighted random selection
4. **Failure Handling**: Graceful degradation when providers are unhealthy
5. **Performance Tracking**: Moving averages and percentile calculations for response times

The next subsection will cover advanced routing strategies and intelligent provider selection algorithms.