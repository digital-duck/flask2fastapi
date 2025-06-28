# Chapter 9, Section 4.2: Advanced Routing Strategies and Request Execution

## Overview

Building on the core router architecture, this subsection implements sophisticated routing algorithms including performance-based selection, intelligent multi-factor decision making, and robust request execution with fallback mechanisms.

## Task Classification and Intelligent Routing

### Task Type Classification

```python
    async def _classify_task_type(self, prompt: str) -> TaskType:
        """Classify task type based on prompt content with enhanced analysis"""
        
        prompt_lower = prompt.lower()
        
        # Enhanced keyword-based classification with weights
        task_scores = {
            TaskType.CODE_GENERATION: 0,
            TaskType.ANALYSIS: 0,
            TaskType.CREATIVE_WRITING: 0,
            TaskType.MULTIMODAL: 0,
            TaskType.SUMMARIZATION: 0,
            TaskType.EMBEDDINGS: 0,
            TaskType.CHAT: 0
        }
        
        # Code generation indicators
        code_keywords = ['code', 'function', 'program', 'debug', 'script', 'algorithm', 
                        'python', 'javascript', 'java', 'c++', 'sql', 'api', 'class',
                        'method', 'variable', 'loop', 'if', 'else', 'import', 'library']
        task_scores[TaskType.CODE_GENERATION] = sum(2 if word in prompt_lower else 0 for word in code_keywords)
        
        # Analysis indicators  
        analysis_keywords = ['analyze', 'analysis', 'data', 'research', 'study', 'compare',
                           'evaluate', 'assess', 'examine', 'investigate', 'statistics',
                           'trends', 'insights', 'metrics', 'performance', 'report']
        task_scores[TaskType.ANALYSIS] = sum(2 if word in prompt_lower else 0 for word in analysis_keywords)
        
        # Creative writing indicators
        creative_keywords = ['story', 'poem', 'creative', 'write', 'imagine', 'character',
                           'plot', 'narrative', 'fiction', 'novel', 'essay', 'blog',
                           'article', 'content', 'marketing', 'copy']
        task_scores[TaskType.CREATIVE_WRITING] = sum(2 if word in prompt_lower else 0 for word in creative_keywords)
        
        # Multimodal indicators
        multimodal_keywords = ['image', 'picture', 'photo', 'visual', 'video', 'audio',
                             'diagram', 'chart', 'graph', 'screenshot', 'drawing']
        task_scores[TaskType.MULTIMODAL] = sum(3 if word in prompt_lower else 0 for word in multimodal_keywords)
        
        # Summarization indicators
        summary_keywords = ['summarize', 'summary', 'tldr', 'brief', 'overview',
                          'key points', 'main ideas', 'condensed', 'abstract']
        task_scores[TaskType.SUMMARIZATION] = sum(3 if word in prompt_lower else 0 for word in summary_keywords)
        
        # Embeddings indicators
        embedding_keywords = ['embed', 'similarity', 'vector', 'search', 'semantic',
                            'clustering', 'classification', 'recommendation']
        task_scores[TaskType.EMBEDDINGS] = sum(3 if word in prompt_lower else 0 for word in embedding_keywords)
        
        # Length-based heuristics
        if len(prompt) > 1000:
            task_scores[TaskType.ANALYSIS] += 1
            task_scores[TaskType.SUMMARIZATION] += 2
        
        # Question-based heuristics
        if prompt.strip().endswith('?'):
            task_scores[TaskType.CHAT] += 1
        
        # Default chat score for conversational patterns
        chat_patterns = ['how', 'what', 'why', 'when', 'where', 'can you', 'please', 'help']
        task_scores[TaskType.CHAT] = sum(1 if word in prompt_lower else 0 for word in chat_patterns)
        
        # Select task type with highest score
        best_task = max(task_scores.keys(), key=lambda k: task_scores[k])
        
        # If no clear winner, default to CHAT
        if task_scores[best_task] == 0:
            return TaskType.CHAT
        
        logger.debug(
            "Task classification",
            prompt_length=len(prompt),
            scores=task_scores,
            classified_as=best_task.value
        )
        
        return best_task

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
```

## Advanced Selection Algorithms

### Performance-Based Selection

```python
    async def _performance_based_selection(
        self, 
        max_response_time: Optional[float]
    ) -> List[AIProvider]:
        """Select providers based on comprehensive performance metrics"""
        
        await self._update_health_status()
        
        # Start with healthy providers only
        candidates = [
            provider for provider in AIProvider
            if self.provider_stats[provider].is_healthy
        ]
        
        # Apply response time constraint if specified
        if max_response_time:
            candidates = [
                provider for provider in candidates
                if self.provider_stats[provider].avg_response_time <= max_response_time
            ]
        
        if not candidates:
            # Relaxed fallback - use all providers if none meet criteria
            candidates = list(AIProvider)
        
        # Calculate comprehensive performance scores
        def performance_score(provider: AIProvider) -> float:
            stats = self.provider_stats[provider]
            
            # Response time component (normalized to 0-10 seconds)
            response_time_score = min(10.0, stats.avg_response_time or 10.0)
            
            # Success rate component (inverted - lower is better)
            failure_penalty = (1.0 - stats.success_rate) * 50
            
            # P95 response time penalty (for consistency)
            p95_penalty = min(5.0, stats.get_p95_response_time() / 10.0 * 5.0)
            
            # Recent activity bonus (providers with more data are more reliable)
            activity_bonus = -min(2.0, stats.total_requests / 100 * 2.0)
            
            # Health score component
            health_penalty = (1.0 - stats.get_health_score()) * 10
            
            total_score = (response_time_score + failure_penalty + 
                          p95_penalty + activity_bonus + health_penalty)
            
            return total_score
        
        # Sort by performance score (lower is better)
        candidates.sort(key=performance_score)
        
        logger.info(
            "Performance-based selection",
            candidates=[p.value for p in candidates],
            scores={p.value: performance_score(p) for p in candidates[:3]}
        )
        
        return candidates

    async def _availability_based_selection(self) -> List[AIProvider]:
        """Select providers based on availability and reliability metrics"""
        
        await self._update_health_status()
        
        # Calculate comprehensive availability scores
        availability_scores = {}
        current_time = time.time()
        
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            # Base score from success rate (0-1)
            base_score = stats.success_rate
            
            # Health status impact
            health_multiplier = 1.0 if stats.is_healthy else 0.1
            
            # Recent activity bonus (more recent = more reliable score)
            hours_since_check = (current_time - stats.last_health_check) / 3600
            staleness_penalty = min(0.3, hours_since_check / 24 * 0.3)
            
            # Request volume confidence (more requests = more confidence)
            volume_bonus = min(0.2, stats.total_requests / 100 * 0.2)
            
            # Error rate impact (recent errors are weighted more heavily)
            recent_error_penalty = min(0.2, len(stats.recent_errors) / 10 * 0.2)
            
            final_score = (base_score * health_multiplier + volume_bonus - 
                          staleness_penalty - recent_error_penalty)
            
            availability_scores[provider] = max(0, final_score)
        
        # Sort by availability score (descending - higher is better)
        sorted_providers = sorted(
            availability_scores.keys(), 
            key=lambda p: availability_scores[p], 
            reverse=True
        )
        
        logger.info(
            "Availability-based selection",
            scores=availability_scores,
            selected_order=[p.value for p in sorted_providers]
        )
        
        return sorted_providers

    async def _intelligent_selection(
        self,
        task_type: TaskType,
        max_cost: Optional[float],
        max_response_time: Optional[float]
    ) -> List[AIProvider]:
        """Intelligent provider selection combining multiple factors with ML-like scoring"""
        
        await self._update_health_status()
        
        # Calculate composite scores for each provider
        provider_scores = {}
        
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            # Skip unhealthy providers for now, but don't eliminate completely
            health_multiplier = 1.0 if stats.is_healthy else 0.1
            
            # 1. Task suitability score (0-1)
            task_score = 0.5  # Base score
            if provider in self.task_preferences.get(task_type, []):
                task_rank = self.task_preferences[task_type].index(provider)
                # Higher preference = higher score (1.0 for first choice, 0.8 for second, etc.)
                task_score = max(0.5, 1.0 - (task_rank * 0.2))
            
            # 2. Performance score (0-1)
            if stats.total_requests > 0:
                # Success rate component
                success_component = stats.success_rate
                
                # Response time component (normalize to reasonable range)
                response_component = max(0, 1.0 - (stats.avg_response_time / 10.0))
                
                # Consistency component (based on P95 vs average)
                p95_time = stats.get_p95_response_time()
                consistency_component = max(0, 1.0 - abs(p95_time - stats.avg_response_time) / 10.0)
                
                performance_score = (success_component * 0.5 + 
                                   response_component * 0.3 + 
                                   consistency_component * 0.2)
            else:
                performance_score = 0.6  # Neutral score for untested providers
            
            # 3. Cost score (0-1) - lower cost = higher score
            if provider in self.pricing:
                avg_cost = sum(
                    (model['input'] + model['output']) / 2
                    for model in self.pricing[provider].values()
                ) / len(self.pricing[provider])
                
                # Normalize cost score (assuming $0.01 per 1K tokens as expensive)
                cost_score = max(0, 1.0 - (avg_cost / 0.01))
            else:
                cost_score = 0.5
            
            # Apply cost constraint
            if max_cost and avg_cost > max_cost:
                continue  # Skip providers that exceed cost limit
            
            # Apply response time constraint
            if max_response_time and stats.avg_response_time > max_response_time:
                continue  # Skip providers that are too slow
            
            # 4. Capability score (0-1) based on model features
            capability_score = 0.5
            capabilities = self.model_capabilities.get(provider, {})
            
            # Bonus for multimodal if needed
            if task_type == TaskType.MULTIMODAL and capabilities.get('multimodal', False):
                capability_score += 0.3
            
            # Bonus for large context if prompt is long
            # (This would need prompt length passed in - simplified for now)
            if capabilities.get('max_context', 0) > 100000:
                capability_score += 0.1
            
            # Bonus for enterprise readiness
            if capabilities.get('enterprise_ready', False):
                capability_score += 0.1
            
            capability_score = min(1.0, capability_score)
            
            # 5. Historical performance trend (0-1)
            trend_score = 0.5
            if len(stats.response_times) >= 10:
                # Calculate trend from recent response times
                recent_times = stats.response_times[-10:]
                older_times = stats.response_times[-20:-10] if len(stats.response_times) >= 20 else recent_times
                
                recent_avg = sum(recent_times) / len(recent_times)
                older_avg = sum(older_times) / len(older_times)
                
                # Positive trend if recent times are better (lower)
                if older_avg > 0:
                    improvement_ratio = (older_avg - recent_avg) / older_avg
                    trend_score = 0.5 + (improvement_ratio * 0.5)  # Scale to 0-1
                    trend_score = max(0, min(1.0, trend_score))
            
            # Weighted composite score with strategic weightings
            weights = {
                'task': 0.25,      # Task suitability
                'performance': 0.30, # Historical performance
                'cost': 0.15,       # Cost efficiency
                'capability': 0.20,  # Technical capabilities
                'trend': 0.10       # Performance trend
            }
            
            composite_score = (
                task_score * weights['task'] +
                performance_score * weights['performance'] +
                cost_score * weights['cost'] +
                capability_score * weights['capability'] +
                trend_score * weights['trend']
            )
            
            # Apply health multiplier
            final_score = composite_score * health_multiplier
            
            provider_scores[provider] = {
                'final_score': final_score,
                'components': {
                    'task': task_score,
                    'performance': performance_score,
                    'cost': cost_score,
                    'capability': capability_score,
                    'trend': trend_score
                },
                'health_multiplier': health_multiplier
            }
        
        # Sort by final score (descending - higher is better)
        sorted_providers = sorted(
            provider_scores.keys(),
            key=lambda p: provider_scores[p]['final_score'],
            reverse=True
        )
        
        logger.info(
            "Intelligent provider selection",
            task_type=task_type.value,
            constraints={
                'max_cost': max_cost,
                'max_response_time': max_response_time
            },
            top_scores={
                p.value: provider_scores[p]['final_score'] 
                for p in sorted_providers[:3]
            },
            detailed_scores={
                p.value: provider_scores[p]['components'] 
                for p in sorted_providers[:3]
            }
        )
        
        return sorted_providers or [AIProvider.AWS_BEDROCK]
```

## Request Execution with Fallback

### Robust Request Handling

```python
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
        max_retries: int = 3,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Route AI request with intelligent provider selection and comprehensive error handling
        
        Args:
            prompt: The input prompt for the AI model
            strategy: Routing strategy to use
            task_type: Type of task (auto-detected if None)
            preferred_provider: Preferred provider to try first
            model_preference: Specific model to prefer
            max_cost: Maximum cost per 1K tokens
            max_response_time: Maximum acceptable response time
            fallback_enabled: Whether to fall back to other providers on failure
            max_retries: Maximum retries per provider
            timeout: Overall request timeout
            
        Returns:
            Dict containing response and metadata
        """
        
        start_time = time.time()
        
        # Classify task type if not provided
        if not task_type:
            task_type = await self._classify_task_type(prompt)
        
        # Try preferred provider first if specified and available
        if preferred_provider and self.provider_stats[preferred_provider].is_healthy:
            try:
                logger.info(
                    "Trying preferred provider",
                    provider=preferred_provider.value,
                    task_type=task_type.value
                )
                
                result = await self._make_request_with_retries(
                    preferred_provider, prompt, model_preference, max_retries, timeout
                )
                
                result.update({
                    'provider_used': preferred_provider.value,
                    'strategy_used': 'preferred_provider',
                    'fallback_used': False,
                    'total_time': time.time() - start_time
                })
                
                return result
                
            except Exception as e:
                logger.warning(
                    "Preferred provider failed",
                    provider=preferred_provider.value,
                    error=str(e),
                    fallback_enabled=fallback_enabled
                )
                
                if not fallback_enabled:
                    raise
        
        # Select provider candidates based on strategy
        provider_candidates = await self._select_provider_candidates(
            strategy, task_type, max_cost, max_response_time
        )
        
        # Execute request with fallback across providers
        result = await self._execute_with_fallback(
            provider_candidates, prompt, model_preference, max_retries, timeout, start_time
        )
        
        result.update({
            'strategy_used': strategy.value,
            'task_type': task_type.value,
            'total_candidates': len(provider_candidates)
        })
        
        return result

    async def _execute_with_fallback(
        self,
        provider_candidates: List[AIProvider],
        prompt: str,
        model_preference: Optional[str],
        max_retries: int,
        timeout: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """Execute request with intelligent fallback across providers"""
        
        last_error = None
        attempts_log = []
        
        for attempt, provider in enumerate(provider_candidates):
            attempt_start = time.time()
            
            try:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    raise TimeoutError(f"Overall request timeout of {timeout}s exceeded")
                
                logger.info(
                    "Attempting provider",
                    provider=provider.value,
                    attempt=attempt + 1,
                    total_candidates=len(provider_candidates)
                )
                
                result = await self._make_request_with_retries(
                    provider, prompt, model_preference, max_retries, timeout
                )
                
                # Calculate attempt time
                attempt_time = time.time() - attempt_start
                
                # Add comprehensive routing metadata
                result.update({
                    'provider_used': provider.value,
                    'fallback_used': attempt > 0,
                    'attempt_number': attempt + 1,
                    'total_candidates': len(provider_candidates),
                    'attempt_time': attempt_time,
                    'total_time': time.time() - start_time,
                    'attempts_log': attempts_log + [{
                        'provider': provider.value,
                        'success': True,
                        'time': attempt_time
                    }]
                })
                
                logger.info(
                    "Request successful",
                    provider=provider.value,
                    attempt=attempt + 1,
                    response_time=attempt_time,
                    total_time=result['total_time']
                )
                
                return result
                
            except Exception as e:
                attempt_time = time.time() - attempt_start
                
                logger.warning(
                    "Provider attempt failed",
                    provider=provider.value,
                    attempt=attempt + 1,
                    error=str(e),
                    attempt_time=attempt_time
                )
                
                # Log this attempt
                attempts_log.append({
                    'provider': provider.value,
                    'success': False,
                    'error': str(e),
                    'time': attempt_time
                })
                
                # Update failure stats
                self.provider_stats[provider].add_request_result(
                    success=False,
                    response_time=attempt_time,
                    error=str(e)
                )
                
                last_error = e
                continue
        
        # All providers failed
        total_time = time.time() - start_time
        error_summary = {
            'message': 'All AI providers failed',
            'last_error': str(last_error),
            'total_attempts': len(provider_candidates),
            'total_time': total_time,
            'attempts_log': attempts_log,
            'provider_candidates': [p.value for p in provider_candidates]
        }
        
        logger.error("All providers failed", **error_summary)
        raise Exception(f"All AI providers failed. Attempts: {len(provider_candidates)}. Last error: {str(last_error)}")

    async def _make_request_with_retries(
        self,
        provider: AIProvider,
        prompt: str,
        model_preference: Optional[str],
        max_retries: int,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make request with exponential backoff retry logic"""
        
        for retry in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Calculate remaining timeout
                remaining_timeout = None
                if timeout:
                    remaining_timeout = max(1.0, timeout - (time.time() - start_time))
                
                result = await self._make_request(provider, prompt, model_preference, remaining_timeout)
                response_time = time.time() - start_time
                
                # Update success stats
                self.provider_stats[provider].add_request_result(
                    success=True,
                    response_time=response_time
                )
                
                result['response_time'] = response_time
                result['retry_count'] = retry
                
                if retry > 0:
                    logger.info(
                        "Request succeeded after retries",
                        provider=provider.value,
                        retry_count=retry,
                        response_time=response_time
                    )
                
                return result
                
            except Exception as e:
                if retry == max_retries:
                    logger.error(
                        "Request failed after all retries",
                        provider=provider.value,
                        retry_count=retry,
                        error=str(e)
                    )
                    raise
                
                # Exponential backoff with jitter
                base_wait = min(2 ** retry, 8)  # Cap at 8 seconds
                jitter = random.uniform(0, 0.5)
                wait_time = base_wait * 0.5 + jitter
                
                logger.info(
                    "Retrying request",
                    provider=provider.value,
                    retry=retry + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e)
                )
                
                await asyncio.sleep(wait_time)

    async def _make_request(
        self,
        provider: AIProvider,
        prompt: str,
        model_preference: Optional[str],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make request to specific provider with timeout"""
        
        service = self.providers[provider]
        
        # Set default timeout if not specified
        if timeout is None:
            timeout = 30.0  # 30 second default
        
        try:
            if provider == AIProvider.AWS_BEDROCK:
                model = model_preference or 'claude-3-5-sonnet'
                result = await asyncio.wait_for(
                    service.generate_response(prompt, model),
                    timeout=timeout
                )
                
            elif provider == AIProvider.AZURE_OPENAI:
                model = model_preference or 'gpt-4-turbo'
                result = await asyncio.wait_for(
                    service.generate_response(prompt, model),
                    timeout=timeout
                )
                
            elif provider == AIProvider.GCP_VERTEX:
                model = model_preference or 'gemini-1.0-pro'
                result = await asyncio.wait_for(
                    service.generate_response(prompt, model),
                    timeout=timeout
                )
                
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Ensure result has required fields
            if not isinstance(result, dict):
                result = {'response': str(result)}
            
            result.setdefault('model_used', model)
            result.setdefault('provider', provider.value)
            
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {provider.value} timed out after {timeout}s")
        except Exception as e:
            # Re-raise with provider context
            raise Exception(f"{provider.value} error: {str(e)}")
```

## Usage Examples

### Advanced Routing in Practice

```python
async def demonstrate_advanced_routing():
    """Demonstrate advanced routing capabilities"""
    
    router = AIServiceRouter()
    await router.initialize()
    
    # Example 1: Intelligent routing with constraints
    try:
        result = await router.route_request(
            prompt="Analyze this sales data and provide insights with recommendations",
            strategy=RoutingStrategy.INTELLIGENT,
            max_cost=0.005,  # Max $0.005 per 1K tokens
            max_response_time=5.0,  # Max 5 seconds
            fallback_enabled=True,
            timeout=15.0
        )
        
        print(f"Analysis completed by {result['provider_used']}")
        print(f"Total time: {result['total_time']:.2f}s")
        print(f"Used fallback: {result['fallback_used']}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    # Example 2: Performance-based routing for speed-critical tasks
    try:
        result = await router.route_request(
            prompt="Quick summary of this document please",
            strategy=RoutingStrategy.PERFORMANCE_BASED,
            max_response_time=2.0,
            max_retries=1  # Quick fail for speed
        )
        
        print(f"Quick summary by {result['provider_used']}")
        print(f"Response time: {result['response_time']:.2f}s")
        
    except Exception as e:
        print(f"Quick summary failed: {e}")
    
    # Example 3: Cost-optimized routing for batch operations
    results = []
    prompts = [
        "Translate this to Spanish: Hello world",
        "What is 2+2?",
        "Write a haiku about programming"
    ]
    
    for prompt in prompts:
        try:
            result = await router.route_request(
                prompt=prompt,
                strategy=RoutingStrategy.COST_OPTIMIZED,
                max_cost=0.001,  # Very low cost limit
                fallback_enabled=True
            )
            results.append(result)
            
        except Exception as e:
            print(f"Cost-optimized request failed: {e}")
    
    # Analyze cost optimization results
    total_cost = sum(r.get('estimated_cost', 0) for r in results)
    providers_used = [r['provider_used'] for r in results]
    
    print(f"Batch completed with total estimated cost: ${total_cost:.4f}")
    print(f"Providers used: {providers_used}")

if __name__ == "__main__":
    asyncio.run(demonstrate_advanced_routing())
```

## Key Features of Advanced Routing

This advanced routing system provides:

1. **Multi-Factor Decision Making**: Combines task suitability, performance, cost, capabilities, and trends
2. **Intelligent Fallback**: Graceful degradation across multiple providers with detailed logging
3. **Constraint Handling**: Respects cost and performance limits with hard constraints
4. **Retry Logic**: Exponential backoff with jitter for transient failures
5. **Comprehensive Monitoring**: Detailed metrics and attempt logging for debugging
6. **Timeout Management**: Global and per-request timeout handling
7. **Performance Trends**: Historical analysis for improving future selections

The next subsection will cover advanced caching strategies and analytics.