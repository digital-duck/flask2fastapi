# Chapter 9, Section 4: Multi-Provider Orchestration and Advanced Patterns

## Overview

This section demonstrates how to orchestrate multiple AI providers intelligently, implement advanced caching strategies, create sophisticated routing algorithms, and build resilient AI workflows. We'll create a unified AI service that can automatically select the best provider based on cost, performance, capabilities, and availability while maintaining high reliability and optimal user experience.

## Intelligent AI Service Router

### Advanced Router Implementation

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
        """Add a request result to the stats"""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.response_times.append(response_time)
            
            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
            # Update moving average
            if self.avg_response_time == 0:
                self.avg_response_time = response_time
            else:
                # Exponential moving average
                self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time
        else:
            self.total_failures += 1
            if error:
                self.recent_errors.append(error)
                # Keep only last 10 errors
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

class AIServiceRouter:
    """Advanced AI service router with intelligent provider selection"""
    
    def __init__(self):
        self.providers = {
            AIProvider.AWS_BEDROCK: BedrockService(),
            AIProvider.AZURE_OPENAI: AzureOpenAIService(),
            AIProvider.GCP_VERTEX: VertexAIService()
        }
        
        self.provider_stats = {
            provider: ProviderStats()
            for provider in AIProvider
        }
        
        self.round_robin_index = 0
        
        # Model capabilities mapping
        self.model_capabilities = {
            AIProvider.AWS_BEDROCK: {
                'models': ['claude-3-5-sonnet', 'claude-3-haiku', 'llama-3-1', 'amazon-titan'],
                'strengths': ['reasoning', 'analysis', 'coding', 'rag'],
                'max_context': 200000,
                'multimodal': False
            },
            AIProvider.AZURE_OPENAI: {
                'models': ['gpt-4-turbo', 'gpt-4', 'gpt-35-turbo', 'gpt-4-vision'],
                'strengths': ['creativity', 'function_calling', 'multimodal', 'enterprise'],
                'max_context': 128000,
                'multimodal': True
            },
            AIProvider.GCP_VERTEX: {
                'models': ['gemini-1.0-pro', 'gemini-1.0-pro-vision', 'text-bison@001', 'code-bison@001'],
                'strengths': ['cost_effective', 'multimodal', 'coding', 'fast'],
                'max_context': 32000,
                'multimodal': True
            }
        }
        
        # Task-specific provider preferences
        self.task_preferences = {
            TaskType.CHAT: [AIProvider.AZURE_OPENAI, AIProvider.GCP_VERTEX, AIProvider.AWS_BEDROCK],
            TaskType.CODE_GENERATION: [AIProvider.AWS_BEDROCK, AIProvider.GCP_VERTEX, AIProvider.AZURE_OPENAI],
            TaskType.ANALYSIS: [AIProvider.AWS_BEDROCK, AIProvider.AZURE_OPENAI, AIProvider.GCP_VERTEX],
            TaskType.CREATIVE_WRITING: [AIProvider.AZURE_OPENAI, AIProvider.AWS_BEDROCK, AIProvider.GCP_VERTEX],
            TaskType.MULTIMODAL: [AIProvider.GCP_VERTEX, AIProvider.AZURE_OPENAI],
            TaskType.EMBEDDINGS: [AIProvider.GCP_VERTEX, AIProvider.AZURE_OPENAI],
            TaskType.SUMMARIZATION: [AIProvider.AWS_BEDROCK, AIProvider.AZURE_OPENAI, AIProvider.GCP_VERTEX]
        }
        
        # Pricing information (per 1K tokens)
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
        
        # Provider weights for weighted random selection
        self.provider_weights = {
            AIProvider.AWS_BEDROCK: 1.0,
            AIProvider.AZURE_OPENAI: 1.0,
            AIProvider.GCP_VERTEX: 1.0
        }
    
    async def route_request(
        self,
        prompt: str,
        strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT,
        task_type: Optional[TaskType] = None,
        preferred_provider: Optional[AIProvider] = None,
        model_preference: Optional[str] = None,
        max_cost: Optional[float] = None,
        max_response_time: Optional[float] = None,
        fallback_enabled: bool = True,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Route AI request with intelligent provider selection"""
        
        # Classify task type if not provided
        if not task_type:
            task_type = await self._classify_task_type(prompt)
        
        # Use preferred provider if specified and available
        if preferred_provider and self.provider_stats[preferred_provider].is_healthy:
            try:
                result = await self._make_request_with_retries(
                    preferred_provider, prompt, model_preference, max_retries
                )
                return result
            except Exception as e:
                logger.warning(
                    "Preferred provider failed",
                    provider=preferred_provider,
                    error=str(e)
                )
                if not fallback_enabled:
                    raise
        
        # Select provider based on strategy
        provider_candidates = await self._select_provider_candidates(
            strategy, task_type, max_cost, max_response_time
        )
        
        # Execute request with fallback
        return await self._execute_with_fallback(
            provider_candidates, prompt, model_preference, max_retries
        )
    
    async def _classify_task_type(self, prompt: str) -> TaskType:
        """Classify task type based on prompt content"""
        
        prompt_lower = prompt.lower()
        
        # Simple keyword-based classification
        if any(word in prompt_lower for word in ['code', 'function', 'program', 'debug', 'script']):
            return TaskType.CODE_GENERATION
        elif any(word in prompt_lower for word in ['analyze', 'analysis', 'data', 'research', 'study']):
            return TaskType.ANALYSIS
        elif any(word in prompt_lower for word in ['story', 'poem', 'creative', 'write', 'imagine']):
            return TaskType.CREATIVE_WRITING
        elif any(word in prompt_lower for word in ['image', 'picture', 'photo', 'visual', 'video']):
            return TaskType.MULTIMODAL
        elif any(word in prompt_lower for word in ['summarize', 'summary', 'tldr', 'brief']):
            return TaskType.SUMMARIZATION
        elif any(word in prompt_lower for word in ['embed', 'similarity', 'vector', 'search']):
            return TaskType.EMBEDDINGS
        else:
            return TaskType.CHAT
    
    async def _select_provider_candidates(
        self,
        strategy: RoutingStrategy,
        task_type: TaskType,
        max_cost: Optional[float],
        max_response_time: Optional[float]
    ) -> List[AIProvider]:
        """Select provider candidates based on strategy and constraints"""
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            return [self._round_robin_selection()]
        
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return await self._performance_based_selection(max_response_time)
        
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_selection(max_cost)
        
        elif strategy == RoutingStrategy.AVAILABILITY_BASED:
            return await self._availability_based_selection()
        
        elif strategy == RoutingStrategy.MODEL_SPECIFIC:
            return self._model_specific_selection(task_type)
        
        elif strategy == RoutingStrategy.WEIGHTED_RANDOM:
            return [self._weighted_random_selection()]
        
        elif strategy == RoutingStrategy.INTELLIGENT:
            return await self._intelligent_selection(task_type, max_cost, max_response_time)
        
        else:
            return [AIProvider.AWS_BEDROCK]  # Default fallback
    
    def _round_robin_selection(self) -> AIProvider:
        """Round-robin provider selection"""
        providers = list(AIProvider)
        provider = providers[self.round_robin_index % len(providers)]
        self.round_robin_index += 1
        return provider
    
    async def _performance_based_selection(
        self, 
        max_response_time: Optional[float]
    ) -> List[AIProvider]:
        """Select providers based on performance metrics"""
        
        await self._update_health_status()
        
        # Filter healthy providers
        candidates = [
            provider for provider in AIProvider
            if self.provider_stats[provider].is_healthy
        ]
        
        # Filter by response time if specified
        if max_response_time:
            candidates = [
                provider for provider in candidates
                if self.provider_stats[provider].avg_response_time <= max_response_time
            ]
        
        # Sort by performance score (lower is better)
        def performance_score(provider: AIProvider) -> float:
            stats = self.provider_stats[provider]
            response_time_score = stats.avg_response_time or 10.0
            failure_penalty = (1.0 - stats.success_rate) * 100
            return response_time_score + failure_penalty
        
        candidates.sort(key=performance_score)
        return candidates or [AIProvider.AWS_BEDROCK]
    
    def _cost_optimized_selection(self, max_cost: Optional[float]) -> List[AIProvider]:
        """Select providers based on cost optimization"""
        
        cost_scores = {}
        
        for provider in AIProvider:
            if provider not in self.pricing:
                continue
            
            # Calculate average cost across all models
            provider_models = self.pricing[provider]
            avg_cost = sum(
                (model['input'] + model['output']) / 2
                for model in provider_models.values()
            ) / len(provider_models)
            
            # Filter by max cost if specified
            if max_cost and avg_cost > max_cost:
                continue
            
            cost_scores[provider] = avg_cost
        
        # Sort by cost (ascending)
        sorted_providers = sorted(cost_scores.keys(), key=lambda p: cost_scores[p])
        return sorted_providers or [AIProvider.GCP_VERTEX]  # Default to most cost-effective
    
    async def _availability_based_selection(self) -> List[AIProvider]:
        """Select providers based on availability and success rates"""
        
        await self._update_health_status()
        
        # Calculate availability scores
        availability_scores = {}
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            # Base score from success rate
            base_score = stats.success_rate
            
            # Health bonus
            health_bonus = 0.2 if stats.is_healthy else -0.5
            
            # Recent activity bonus
            activity_bonus = 0.1 if stats.total_requests > 10 else 0
            
            availability_scores[provider] = base_score + health_bonus + activity_bonus
        
        # Sort by availability score (descending)
        sorted_providers = sorted(
            availability_scores.keys(), 
            key=lambda p: availability_scores[p], 
            reverse=True
        )
        
        return sorted_providers
    
    def _model_specific_selection(self, task_type: TaskType) -> List[AIProvider]:
        """Select providers based on model capabilities for specific tasks"""
        
        if task_type in self.task_preferences:
            return self.task_preferences[task_type]
        
        # Fallback to general capabilities
        return list(AIProvider)
    
    def _weighted_random_selection(self) -> AIProvider:
        """Select provider using weighted random selection"""
        
        # Adjust weights based on recent performance
        adjusted_weights = {}
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            base_weight = self.provider_weights[provider]
            
            # Adjust weight based on success rate and health
            health_multiplier = 1.0 if stats.is_healthy else 0.1
            success_multiplier = stats.success_rate
            
            adjusted_weights[provider] = base_weight * health_multiplier * success_multiplier
        
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
    
    async def _intelligent_selection(
        self,
        task_type: TaskType,
        max_cost: Optional[float],
        max_response_time: Optional[float]
    ) -> List[AIProvider]:
        """Intelligent provider selection combining multiple factors"""
        
        await self._update_health_status()
        
        # Calculate composite scores for each provider
        provider_scores = {}
        
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            # Skip unhealthy providers
            if not stats.is_healthy:
                continue
            
            # Task suitability score (0-1)
            task_score = 0.8 if provider in self.task_preferences.get(task_type, []) else 0.4
            task_rank = self.task_preferences.get(task_type, []).index(provider) if provider in self.task_preferences.get(task_type, []) else 99
            task_score += (1.0 - task_rank / 10) * 0.3  # Bonus for higher rank
            
            # Performance score (0-1)
            performance_score = min(1.0, stats.success_rate * (1.0 - min(stats.avg_response_time / 10.0, 0.5)))
            
            # Cost score (0-1) - lower cost = higher score
            if provider in self.pricing:
                avg_cost = sum(
                    (model['input'] + model['output']) / 2
                    for model in self.pricing[provider].values()
                ) / len(self.pricing[provider])
                cost_score = max(0, 1.0 - avg_cost / 0.01)  # Normalize to $0.01 max
            else:
                cost_score = 0.5
            
            # Apply constraints
            if max_cost and avg_cost > max_cost:
                continue
            if max_response_time and stats.avg_response_time > max_response_time:
                continue
            
            # Weighted composite score
            composite_score = (
                task_score * 0.4 +
                performance_score * 0.4 +
                cost_score * 0.2
            )
            
            provider_scores[provider] = composite_score
        
        # Sort by composite score (descending)
        sorted_providers = sorted(
            provider_scores.keys(),
            key=lambda p: provider_scores[p],
            reverse=True
        )
        
        logger.info(
            "Intelligent provider selection",
            task_type=task_type,
            scores=provider_scores,
            selected_order=[p.value for p in sorted_providers]
        )
        
        return sorted_providers or [AIProvider.AWS_BEDROCK]
    
    async def _execute_with_fallback(
        self,
        provider_candidates: List[AIProvider],
        prompt: str,
        model_preference: Optional[str],
        max_retries: int
    ) -> Dict[str, Any]:
        """Execute request with fallback across providers"""
        
        last_error = None
        
        for attempt, provider in enumerate(provider_candidates):
            try:
                result = await self._make_request_with_retries(
                    provider, prompt, model_preference, max_retries
                )
                
                # Add routing metadata
                result['provider_used'] = provider.value
                result['fallback_used'] = attempt > 0
                result['attempt_number'] = attempt + 1
                result['total_candidates'] = len(provider_candidates)
                
                return result
                
            except Exception as e:
                logger.warning(
                    "Provider request failed",
                    provider=provider,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                # Update failure stats
                self.provider_stats[provider].add_request_result(
                    success=False,
                    response_time=0,
                    error=str(e)
                )
                
                last_error = e
                continue
        
        # All providers failed
        logger.error("All provider candidates failed", error=str(last_error))
        raise Exception(f"All AI providers failed. Last error: {str(last_error)}")
    
    async def _make_request_with_retries(
        self,
        provider: AIProvider,
        prompt: str,
        model_preference: Optional[str],
        max_retries: int
    ) -> Dict[str, Any]:
        """Make request with retry logic"""
        
        for retry in range(max_retries + 1):
            try:
                start_time = time.time()
                result = await self._make_request(provider, prompt, model_preference)
                response_time = time.time() - start_time
                
                # Update success stats
                self.provider_stats[provider].add_request_result(
                    success=True,
                    response_time=response_time
                )
                
                result['response_time'] = response_time
                return result
                
            except Exception as e:
                if retry == max_retries:
                    raise
                
                # Exponential backoff with jitter
                wait_time = (2 ** retry) * 0.5 + random.uniform(0, 0.5)
                logger.info(
                    "Retrying request",
                    provider=provider,
                    retry=retry + 1,
                    wait_time=wait_time,
                    error=str(e)
                )
                await asyncio.sleep(wait_time)
    
    async def _make_request(
        self,
        provider: AIProvider,
        prompt: str,
        model_preference: Optional[str]
    ) -> Dict[str, Any]:
        """Make request to specific provider"""
        
        service = self.providers[provider]
        
        if provider == AIProvider.AWS_BEDROCK:
            model = model_preference or 'claude-3-5-sonnet'
            return await service.generate_response(prompt, model)
            
        elif provider == AIProvider.AZURE_OPENAI:
            model = model_preference or 'gpt-4-turbo'
            return await service.generate_response(prompt, model)
            
        elif provider == AIProvider.GCP_VERTEX:
            model = model_preference or 'gemini-1.0-pro'
            return await service.generate_response(prompt, model)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _update_health_status(self):
        """Update health status for all providers"""
        
        current_time = time.time()
        
        # Check health every 5 minutes
        health_check_tasks = []
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            if current_time - stats.last_health_check > 300:  # 5 minutes
                health_check_tasks.append(self._check_provider_health(provider))
        
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_provider_health(self, provider: AIProvider):
        """Check health of a specific provider"""
        
        try:
            service = self.providers[provider]
            health_result = await service.health_check()
            
            stats = self.provider_stats[provider]
            stats.is_healthy = health_result.get('status') == 'healthy'
            stats.last_health_check = time.time()
            
            logger.info(
                "Provider health updated",
                provider=provider,
                healthy=stats.is_healthy
            )
            
        except Exception as e:
            logger.error(
                "Health check failed",
                provider=provider,
                error=str(e)
            )
            self.provider_stats[provider].is_healthy = False
            self.provider_stats[provider].last_health_check = time.time()
    
    async def get_provider_analytics(self) -> Dict[str, Any]:
        """Get comprehensive provider analytics"""
        
        await self._update_health_status()
        
        analytics = {
            'providers': {},
            'overall_stats': {
                'total_requests': sum(stats.total_requests for stats in self.provider_stats.values()),
                'overall_success_rate': 0,
                'avg_response_time': 0
            },
            'routing_recommendations': {}
        }
        
        total_requests = 0
        total_successful = 0
        weighted_response_time = 0
        
        for provider, stats in self.provider_stats.items():
            provider_analytics = {
                'total_requests': stats.total_requests,
                'successful_requests': stats.successful_requests,
                'success_rate': stats.success_rate,
                'avg_response_time': stats.avg_response_time,
                'p95_response_time': stats.get_p95_response_time(),
                'is_healthy': stats.is_healthy,
                'recent_errors': stats.recent_errors[-5:],  # Last 5 errors
                'capabilities': self.model_capabilities.get(provider, {}),
                'estimated_cost_per_1k_tokens': self._get_avg_cost(provider)
            }
            
            analytics['providers'][provider.value] = provider_analytics
            
            # Update overall stats
            total_requests += stats.total_requests
            total_successful += stats.successful_requests
            weighted_response_time += stats.avg_response_time * stats.total_requests
        
        # Calculate overall metrics
        if total_requests > 0:
            analytics['overall_stats']['overall_success_rate'] = total_successful / total_requests
            analytics['overall_stats']['avg_response_time'] = weighted_response_time / total_requests
        
        # Routing recommendations
        analytics['routing_recommendations'] = {
            'fastest_provider': min(self.provider_stats.keys(), key=lambda p: self.provider_stats[p].avg_response_time).value,
            'most_reliable': max(self.provider_stats.keys(), key=lambda p: self.provider_stats[p].success_rate).value,
            'most_cost_effective': min(self.pricing.keys(), key=lambda p: self._get_avg_cost(p)).value,
            'best_for_multimodal': 'gcp_vertex',  # Based on capabilities
            'best_for_coding': 'aws_bedrock',    # Based on capabilities
            'best_for_enterprise': 'azure_openai'  # Based on capabilities
        }
        
        return analytics
    
    def _get_avg_cost(self, provider: AIProvider) -> float:
        """Get average cost per 1K tokens for a provider"""
        if provider not in self.pricing:
            return 0.0
        
        models = self.pricing[provider]
        avg_cost = sum(
            (model['input'] + model['output']) / 2
            for model in models.values()
        ) / len(models)
        
        return avg_cost
    
    async def recommend_provider(
        self,
        prompt: str,
        task_type: Optional[TaskType] = None,
        budget_constraint: Optional[float] = None,
        speed_priority: bool = False,
        reliability_priority: bool = False
    ) -> Dict[str, Any]:
        """Get provider recommendations based on requirements"""
        
        if not task_type:
            task_type = await self._classify_task_type(prompt)
        
        recommendations = []
        
        # Score each provider
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            if not stats.is_healthy:
                continue
            
            score = 0
            reasons = []
            
            # Task suitability
            if provider in self.task_preferences.get(task_type, []):
                task_rank = self.task_preferences[task_type].index(provider)
                task_score = max(0, 10 - task_rank * 2)
                score += task_score
                reasons.append(f"Well-suited for {task_type.value} (+{task_score})")
            
            # Speed priority
            if speed_priority:
                speed_score = max(0, 10 - stats.avg_response_time * 2)
                score += speed_score
                reasons.append(f"Fast response time (+{speed_score:.1f})")
            
            # Reliability priority
            if reliability_priority:
                reliability_score = stats.success_rate * 10
                score += reliability_score
                reasons.append(f"High reliability (+{reliability_score:.1f})")
            
            # Budget constraint
            avg_cost = self._get_avg_cost(provider)
            if budget_constraint:
                if avg_cost <= budget_constraint:
                    cost_score = 5
                    reasons.append(f"Within budget (+{cost_score})")
                else:
                    cost_score = -10
                    reasons.append(f"Over budget ({cost_score})")
                score += cost_score
            else:
                # Reward lower cost
                cost_score = max(0, 5 - avg_cost * 1000)
                score += cost_score
                reasons.append(f"Cost effective (+{cost_score:.1f})")
            
            recommendations.append({
                'provider': provider.value,
                'score': score,
                'reasons': reasons,
                'stats': {
                    'success_rate': stats.success_rate,
                    'avg_response_time': stats.avg_response_time,
                    'estimated_cost_per_1k_tokens': avg_cost
                },
                'capabilities': self.model_capabilities.get(provider, {})
            })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'task_type': task_type.value,
            'requirements': {
                'budget_constraint': budget_constraint,
                'speed_priority': speed_priority,
                'reliability_priority': reliability_priority
            },
            'recommendations': recommendations[:3],  # Top 3
            'prompt_analysis': {
                'length': len(prompt),