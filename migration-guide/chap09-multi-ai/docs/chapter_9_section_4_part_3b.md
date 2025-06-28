# Chapter 9, Section 4.3b: Provider Analytics and Performance Monitoring

## Overview

This subsection implements comprehensive analytics, performance monitoring, and intelligent recommendation systems for the AI service router. It provides deep insights into provider performance, cost analysis, and automated optimization recommendations.

## Enhanced Analytics Methods for AIServiceRouter

### Comprehensive Provider Analytics

```python
# Add these methods to the AIServiceRouter class

    def _calculate_performance_grade(self, stats: ProviderStats) -> str:
        """Calculate a letter grade for provider performance"""
        
        if stats.total_requests < 5:
            return "N/A"  # Not enough data
        
        score = 0
        
        # Success rate component (0-40 points)
        score += stats.success_rate * 40
        
        # Response time component (0-30 points)
        # Good: < 2s, Average: 2-5s, Poor: > 5s
        if stats.avg_response_time < 2.0:
            score += 30
        elif stats.avg_response_time < 5.0:
            score += 20
        else:
            score += 10
        
        # Consistency component (0-20 points)
        p95_time = stats.get_p95_response_time()
        if stats.avg_response_time > 0:
            consistency_ratio = p95_time / stats.avg_response_time
            if consistency_ratio < 1.5:
                score += 20
            elif consistency_ratio < 2.0:
                score += 15
            else:
                score += 10
        else:
            score += 10  # Default for no data
        
        # Health component (0-10 points)
        score += stats.get_health_score() * 10
        
        # Convert to letter grade
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    async def get_provider_analytics(self) -> Dict[str, Any]:
        """Get comprehensive provider analytics with trends and insights"""
        
        await self._update_health_status()
        
        analytics = {
            'providers': {},
            'overall_stats': {
                'total_requests': sum(stats.total_requests for stats in self.provider_stats.values()),
                'overall_success_rate': 0,
                'avg_response_time': 0,
                'total_estimated_cost': 0
            },
            'routing_recommendations': {},
            'performance_trends': {},
            'cost_analysis': {},
            'health_summary': {}
        }
        
        total_requests = 0
        total_successful = 0
        weighted_response_time = 0
        total_cost = 0
        
        # Detailed provider analytics
        for provider, stats in self.provider_stats.items():
            provider_cost = self._get_avg_cost(provider)
            estimated_provider_cost = stats.total_requests * provider_cost * 0.5  # Rough estimate
            
            provider_analytics = {
                'total_requests': stats.total_requests,
                'successful_requests': stats.successful_requests,
                'success_rate': stats.success_rate,
                'avg_response_time': stats.avg_response_time,
                'p95_response_time': stats.get_p95_response_time(),
                'is_healthy': stats.is_healthy,
                'health_score': stats.get_health_score(),
                'recent_errors': stats.recent_errors[-3:],  # Last 3 errors
                'capabilities': self.model_capabilities.get(provider, {}),
                'estimated_cost_per_1k_tokens': provider_cost,
                'estimated_total_cost': estimated_provider_cost,
                'last_health_check': stats.last_health_check,
                'performance_grade': self._calculate_performance_grade(stats)
            }
            
            analytics['providers'][provider.value] = provider_analytics
            
            # Update overall stats
            total_requests += stats.total_requests
            total_successful += stats.successful_requests
            weighted_response_time += stats.avg_response_time * stats.total_requests
            total_cost += estimated_provider_cost
        
        # Calculate overall metrics
        if total_requests > 0:
            analytics['overall_stats']['overall_success_rate'] = total_successful / total_requests
            analytics['overall_stats']['avg_response_time'] = weighted_response_time / total_requests
        analytics['overall_stats']['total_estimated_cost'] = total_cost
        
        # Generate routing recommendations
        analytics['routing_recommendations'] = await self._generate_routing_recommendations()
        
        # Performance trends analysis
        analytics['performance_trends'] = self._analyze_performance_trends()
        
        # Cost analysis
        analytics['cost_analysis'] = self._analyze_costs()
        
        # Health summary
        analytics['health_summary'] = self.get_provider_health_summary()
        
        return analytics
    
    async def _generate_routing_recommendations(self) -> Dict[str, Any]:
        """Generate intelligent routing recommendations"""
        
        recommendations = {
            'fastest_provider': None,
            'most_reliable': None,
            'most_cost_effective': None,
            'best_overall': None,
            'task_specific': {},
            'general_advice': []
        }
        
        # Find fastest provider (healthy providers only)
        healthy_providers = [p for p in AIProvider if self.provider_stats[p].is_healthy]
        
        if healthy_providers:
            fastest = min(healthy_providers, key=lambda p: self.provider_stats[p].avg_response_time)
            recommendations['fastest_provider'] = {
                'provider': fastest.value,
                'avg_response_time': self.provider_stats[fastest].avg_response_time,
                'reliability': self.provider_stats[fastest].success_rate
            }
            
            # Most reliable
            most_reliable = max(healthy_providers, key=lambda p: self.provider_stats[p].success_rate)
            recommendations['most_reliable'] = {
                'provider': most_reliable.value,
                'success_rate': self.provider_stats[most_reliable].success_rate,
                'avg_response_time': self.provider_stats[most_reliable].avg_response_time
            }
            
            # Most cost effective
            cost_effective = min(self.pricing.keys(), key=lambda p: self._get_avg_cost(p))
            recommendations['most_cost_effective'] = {
                'provider': cost_effective.value,
                'avg_cost_per_1k_tokens': self._get_avg_cost(cost_effective),
                'reliability': self.provider_stats[cost_effective].success_rate
            }
            
            # Best overall (using composite scoring)
            best_overall = max(healthy_providers, key=lambda p: self.provider_stats[p].get_health_score())
            recommendations['best_overall'] = {
                'provider': best_overall.value,
                'health_score': self.provider_stats[best_overall].get_health_score(),
                'performance_grade': self._calculate_performance_grade(self.provider_stats[best_overall])
            }
        
        # Task-specific recommendations
        for task_type, preferred_providers in self.task_preferences.items():
            healthy_preferred = [p for p in preferred_providers if self.provider_stats[p].is_healthy]
            if healthy_preferred:
                recommendations['task_specific'][task_type.value] = {
                    'primary': healthy_preferred[0].value,
                    'alternatives': [p.value for p in healthy_preferred[1:3]]
                }
        
        # Generate general advice
        advice = []
        
        # Check for underperforming providers
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            if stats.total_requests > 10:
                if stats.success_rate < 0.9:
                    advice.append(f"Consider avoiding {provider.value} due to low success rate ({stats.success_rate:.1%})")
                if stats.avg_response_time > 10.0:
                    advice.append(f"{provider.value} has high response times ({stats.avg_response_time:.1f}s)")
        
        # Check for cost optimization opportunities
        if self.pricing:
            cheapest_provider = min(self.pricing.keys(), key=lambda p: self._get_avg_cost(p))
            most_used_provider = max(self.provider_stats.keys(), key=lambda p: self.provider_stats[p].total_requests)
            
            if cheapest_provider != most_used_provider:
                cost_savings = (self._get_avg_cost(most_used_provider) - self._get_avg_cost(cheapest_provider)) * 1000
                if cost_savings > 0.001:  # More than $0.001 difference
                    advice.append(f"Consider using {cheapest_provider.value} more often for cost savings (${cost_savings:.3f} per 1K tokens)")
        
        recommendations['general_advice'] = advice
        
        return recommendations
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across providers"""
        
        trends = {
            'response_time_trends': {},
            'success_rate_trends': {},
            'overall_trend': 'stable'
        }
        
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            if len(stats.response_times) >= 10:
                # Simple trend analysis on recent response times
                recent_times = stats.response_times[-10:]
                older_times = stats.response_times[-20:-10] if len(stats.response_times) >= 20 else recent_times
                
                recent_avg = sum(recent_times) / len(recent_times)
                older_avg = sum(older_times) / len(older_times)
                
                trend = "improving" if recent_avg < older_avg else "degrading" if recent_avg > older_avg else "stable"
                
                trends['response_time_trends'][provider.value] = {
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'older_avg': older_avg,
                    'change_percent': ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                }
            
            # Success rate trends (simplified)
            trends['success_rate_trends'][provider.value] = {
                'current_rate': stats.success_rate,
                'trend': 'stable'  # Would need historical data for real trend analysis
            }
        
        return trends
    
    def _analyze_costs(self) -> Dict[str, Any]:
        """Analyze cost patterns and optimization opportunities"""
        
        cost_analysis = {
            'provider_costs': {},
            'total_estimated_spend': 0,
            'cost_distribution': {},
            'optimization_opportunities': []
        }
        
        total_requests = sum(stats.total_requests for stats in self.provider_stats.values())
        total_cost = 0
        
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            avg_cost = self._get_avg_cost(provider)
            estimated_cost = stats.total_requests * avg_cost * 0.5  # Rough estimate
            
            cost_analysis['provider_costs'][provider.value] = {
                'avg_cost_per_1k_tokens': avg_cost,
                'estimated_total_cost': estimated_cost,
                'request_count': stats.total_requests,
                'cost_per_request': estimated_cost / stats.total_requests if stats.total_requests > 0 else 0
            }
            
            total_cost += estimated_cost
        
        cost_analysis['total_estimated_spend'] = total_cost
        
        # Cost distribution
        if total_cost > 0:
            for provider in AIProvider:
                provider_cost = cost_analysis['provider_costs'][provider.value]['estimated_total_cost']
                cost_analysis['cost_distribution'][provider.value] = (provider_cost / total_cost) * 100
        
        # Optimization opportunities
        if total_requests > 100:  # Only analyze if we have enough data
            cheapest_provider = min(self.pricing.keys(), key=lambda p: self._get_avg_cost(p))
            most_expensive = max(self.pricing.keys(), key=lambda p: self._get_avg_cost(p))
            
            savings_potential = (self._get_avg_cost(most_expensive) - self._get_avg_cost(cheapest_provider)) * total_requests * 0.5
            
            if savings_potential > 1.0:  # More than $1 potential savings
                cost_analysis['optimization_opportunities'].append({
                    'type': 'provider_optimization',
                    'description': f"Switching more requests to {cheapest_provider.value} could save ~${savings_potential:.2f}",
                    'potential_savings': savings_potential
                })
        
        return cost_analysis
    
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
```

## Intelligent Recommendation System

### Advanced Provider Recommendations

```python
    async def recommend_provider(
        self,
        prompt: str,
        task_type: Optional[TaskType] = None,
        budget_constraint: Optional[float] = None,
        speed_priority: bool = False,
        reliability_priority: bool = False,
        detailed_analysis: bool = False
    ) -> Dict[str, Any]:
        """Get comprehensive provider recommendations with detailed reasoning"""
        
        if not task_type:
            task_type = await self._classify_task_type(prompt)
        
        recommendations = []
        
        # Score each provider
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            
            if not stats.is_healthy:
                continue
            
            score_components = {
                'task_suitability': 0,
                'performance': 0,
                'cost': 0,
                'reliability': 0,
                'capability': 0,
                'trend': 0
            }
            reasons = []
            
            # 1. Task suitability (0-10)
            if provider in self.task_preferences.get(task_type, []):
                task_rank = self.task_preferences[task_type].index(provider)
                task_score = max(0, 10 - task_rank * 2)
                score_components['task_suitability'] = task_score
                reasons.append(f"Well-suited for {task_type.value} (rank #{task_rank + 1})")
            else:
                score_components['task_suitability'] = 3
                reasons.append(f"General capability for {task_type.value}")
            
            # 2. Performance scoring
            if speed_priority:
                speed_score = max(0, 10 - stats.avg_response_time * 2)
                score_components['performance'] = speed_score
                reasons.append(f"Response time: {stats.avg_response_time:.2f}s")
            else:
                perf_score = min(10, stats.success_rate * 10)
                score_components['performance'] = perf_score
                reasons.append(f"Success rate: {stats.success_rate:.1%}")
            
            # 3. Reliability scoring
            if reliability_priority:
                reliability_score = stats.success_rate * 10
                score_components['reliability'] = reliability_score
                reasons.append(f"High reliability: {stats.success_rate:.1%}")
            else:
                reliability_score = stats.get_health_score() * 10
                score_components['reliability'] = reliability_score
                reasons.append(f"Health score: {stats.get_health_score():.2f}")
            
            # 4. Cost scoring
            avg_cost = self._get_avg_cost(provider)
            if budget_constraint:
                if avg_cost <= budget_constraint:
                    cost_score = 10
                    reasons.append(f"Within budget (${avg_cost:.4f} ≤ ${budget_constraint:.4f})")
                else:
                    cost_score = 0
                    reasons.append(f"Over budget (${avg_cost:.4f} > ${budget_constraint:.4f})")
                    continue  # Skip if over budget
            else:
                # Reward lower cost
                max_cost = max(self._get_avg_cost(p) for p in self.pricing.keys()) if self.pricing else 0.01
                cost_score = 10 * (1 - avg_cost / max_cost) if max_cost > 0 else 5
                reasons.append(f"Cost: ${avg_cost:.4f}/1K tokens")
            
            score_components['cost'] = cost_score
            
            # 5. Capability scoring
            capabilities = self.model_capabilities.get(provider, {})
            capability_score = 5  # Base score
            
            if task_type == TaskType.MULTIMODAL and capabilities.get('multimodal', False):
                capability_score += 3
                reasons.append("Multimodal support")
            
            if len(prompt) > 5000 and capabilities.get('max_context', 0) > 100000:
                capability_score += 2
                reasons.append("Large context support")
            
            score_components['capability'] = min(10, capability_score)
            
            # 6. Trend scoring (simplified)
            if len(stats.response_times) >= 10:
                recent_avg = sum(stats.response_times[-5:]) / 5
                older_avg = sum(stats.response_times[-10:-5]) / 5
                if recent_avg < older_avg:
                    trend_score = 8
                    reasons.append("Improving performance trend")
                else:
                    trend_score = 5
                    reasons.append("Stable performance")
            else:
                trend_score = 5
                reasons.append("Limited trend data")
            
            score_components['trend'] = trend_score
            
            # Calculate weighted total score
            weights = {
                'task_suitability': 0.25,
                'performance': 0.25,
                'cost': 0.20,
                'reliability': 0.15,
                'capability': 0.10,
                'trend': 0.05
            }
            
            # Adjust weights based on priorities
            if speed_priority:
                weights['performance'] = 0.4
                weights['task_suitability'] = 0.2
            if reliability_priority:
                weights['reliability'] = 0.3
                weights['performance'] = 0.2
            
            total_score = sum(
                score_components[component] * weight
                for component, weight in weights.items()
            )
            
            recommendations.append({
                'provider': provider.value,
                'total_score': total_score,
                'score_components': score_components,
                'reasons': reasons,
                'stats': {
                    'success_rate': stats.success_rate,
                    'avg_response_time': stats.avg_response_time,
                    'health_score': stats.get_health_score(),
                    'estimated_cost_per_1k_tokens': avg_cost,
                    'performance_grade': self._calculate_performance_grade(stats)
                },
                'capabilities': capabilities
            })
        
        # Sort by total score (descending)
        recommendations.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Prepare response
        response = {
            'task_type': task_type.value,
            'prompt_analysis': {
                'length': len(prompt),
                'estimated_tokens': len(prompt.split()) * 1.3,  # Rough estimate
                'complexity': 'high' if len(prompt) > 1000 else 'medium' if len(prompt) > 200 else 'low'
            },
            'requirements': {
                'budget_constraint': budget_constraint,
                'speed_priority': speed_priority,
                'reliability_priority': reliability_priority
            },
            'recommendations': recommendations[:3] if not detailed_analysis else recommendations,
            'summary': {
                'top_choice': recommendations[0]['provider'] if recommendations else None,
                'reasoning': recommendations[0]['reasons'][:3] if recommendations else [],
                'confidence': 'high' if recommendations and recommendations[0]['total_score'] > 7 else 'medium'
            }
        }
        
        return response
```

## Real-time Monitoring Dashboard Data

### Dashboard Metrics Generation

```python
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Generate real-time dashboard metrics"""
        
        current_time = time.time()
        
        # Basic metrics
        total_requests = sum(stats.total_requests for stats in self.provider_stats.values())
        total_successful = sum(stats.successful_requests for stats in self.provider_stats.values())
        overall_success_rate = total_successful / total_requests if total_requests > 0 else 0
        
        # Calculate weighted average response time
        weighted_response_time = 0
        if total_requests > 0:
            for stats in self.provider_stats.values():
                weighted_response_time += stats.avg_response_time * stats.total_requests
            weighted_response_time /= total_requests
        
        # Health status
        healthy_providers = sum(1 for stats in self.provider_stats.values() if stats.is_healthy)
        total_providers = len(self.provider_stats)
        
        # Recent activity (last hour)
        recent_requests = 0
        recent_errors = 0
        
        for stats in self.provider_stats.values():
            # Count recent errors (simplified - in production, you'd track timestamps)
            recent_errors += len([e for e in stats.recent_errors if 'error' in e.lower()])
        
        dashboard_data = {
            'overview': {
                'total_requests': total_requests,
                'success_rate': overall_success_rate,
                'avg_response_time': weighted_response_time,
                'healthy_providers': f"{healthy_providers}/{total_providers}",
                'recent_requests_1h': recent_requests,  # Would need time-based tracking
                'recent_errors_1h': recent_errors
            },
            'provider_status': {},
            'performance_chart_data': [],
            'cost_breakdown': {},
            'alerts': []
        }
        
        # Provider status for dashboard
        for provider, stats in self.provider_stats.items():
            status = "healthy" if stats.is_healthy else "unhealthy"
            if stats.avg_response_time > 10.0:
                status = "slow"
            elif stats.success_rate < 0.9:
                status = "unreliable"
            
            dashboard_data['provider_status'][provider.value] = {
                'status': status,
                'requests': stats.total_requests,
                'success_rate': stats.success_rate,
                'avg_response_time': stats.avg_response_time,
                'grade': self._calculate_performance_grade(stats)
            }
        
        # Performance chart data (last 24 hours simulation)
        for i in range(24):
            hour_ago = current_time - (i * 3600)
            dashboard_data['performance_chart_data'].append({
                'timestamp': hour_ago,
                'requests': max(0, total_requests // 24 + random.randint(-5, 5)),
                'avg_response_time': weighted_response_time + random.uniform(-0.5, 0.5),
                'success_rate': min(1.0, overall_success_rate + random.uniform(-0.05, 0.05))
            })
        
        # Cost breakdown
        total_cost = 0
        for provider in AIProvider:
            stats = self.provider_stats[provider]
            cost = stats.total_requests * self._get_avg_cost(provider) * 0.5
            total_cost += cost
            dashboard_data['cost_breakdown'][provider.value] = cost
        
        # Generate alerts
        alerts = []
        for provider, stats in self.provider_stats.items():
            if not stats.is_healthy:
                alerts.append({
                    'type': 'error',
                    'message': f"{provider.value} is unhealthy",
                    'timestamp': current_time
                })
            elif stats.avg_response_time > 10.0:
                alerts.append({
                    'type': 'warning',
                    'message': f"{provider.value} has high response times ({stats.avg_response_time:.1f}s)",
                    'timestamp': current_time
                })
            elif stats.success_rate < 0.9:
                alerts.append({
                    'type': 'warning',
                    'message': f"{provider.value} has low success rate ({stats.success_rate:.1%})",
                    'timestamp': current_time
                })
        
        dashboard_data['alerts'] = alerts
        
        return dashboard_data
```

## Usage Examples

### Analytics Demo

```python
async def demonstrate_analytics():
    """Demonstrate comprehensive analytics capabilities"""
    
    router = AIServiceRouter()
    await router.initialize()
    
    # Simulate some requests to generate data
    test_scenarios = [
        ("Analyze sales performance", TaskType.ANALYSIS, True, False),
        ("Write Python code", TaskType.CODE_GENERATION, False, True),
        ("Create a story", TaskType.CREATIVE_WRITING, False, False),
        ("Summarize document", TaskType.SUMMARIZATION, True, True)
    ]
    
    print("🔄 Simulating requests to generate analytics data...")
    
    for prompt, task_type, speed_priority, reliability_priority in test_scenarios:
        try:
            result = await router.route_request(
                prompt=prompt,
                task_type=task_type,
                strategy=RoutingStrategy.INTELLIGENT
            )
            print(f"✅ {prompt[:20]}... -> {result.get('provider_used')}")
        except Exception as e:
            print(f"❌ {prompt[:20]}... -> Failed: {e}")
    
    # Get comprehensive analytics
    print("\n📊 PROVIDER ANALYTICS")
    print("=" * 50)
    
    analytics = await router.get_provider_analytics()
    
    # Overall stats
    overall = analytics['overall_stats']
    print(f"Total Requests: {overall['total_requests']}")
    print(f"Success Rate: {overall['overall_success_rate']:.1%}")
    print(f"Avg Response Time: {overall['avg_response_time']:.2f}s")
    print(f"Estimated Cost: ${overall['total_estimated_cost']:.4f}")
    
    # Provider breakdown
    print(f"\n🏥 PROVIDER HEALTH")
    for provider, data in analytics['providers'].items():
        print(f"{provider}: {data['performance_grade']} "
              f"({data['success_rate']:.1%} success, "
              f"{data['avg_response_time']:.2f}s avg)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    recs = analytics['routing_recommendations']
    if recs['fastest_provider']:
        print(f"Fastest: {recs['fastest_provider']['provider']} "
              f"({recs['fastest_provider']['avg_response_time']:.2f}s)")
    if recs['most_reliable']:
        print(f"Most Reliable: {recs['most_reliable']['provider']} "
              f"({recs['most_reliable']['success_rate']:.1%})")
    
    # Cost analysis
    print(f"\n💰 COST ANALYSIS")
    cost_analysis = analytics['cost_analysis']
    print(f"Total Spend: ${cost_analysis['total_estimated_spend']:.4f}")
    
    for opportunity in cost_analysis['optimization_opportunities']:
        print(f"💡 {opportunity['description']}")
    
    # Get specific recommendation
    print(f"\n🎯 RECOMMENDATION FOR CODING TASK")
    recommendation = await router.recommend_provider(
        prompt="Write a Python function to sort a list",
        task_type=TaskType.CODE_GENERATION,
        speed_priority=True,
        detailed_analysis=True
    )
    
    if recommendation['recommendations']:
        top_rec = recommendation['recommendations'][0]
        print(f"Recommended: {top_rec['provider']} (score: {top_rec['total_score']:.1f})")
        print(f"Reasons: {', '.join(top_rec['reasons'][:3])}")
    
    # Dashboard metrics
    print(f"\n📈 DASHBOARD METRICS")
    dashboard = await router.get_dashboard_metrics()
    
    print(f"System Health: {dashboard['overview']['healthy_providers']}")
    print(f"Recent Errors: {dashboard['overview']['recent_errors_1h']}")
    
    if dashboard['alerts']:
        print(f"\n⚠️  ACTIVE ALERTS")
        for alert in dashboard['alerts'][:3]:
            print(f"{alert['type'].upper()}: {alert['message']}")

if __name__ == "__main__":
    import random  # For dashboard simulation
    asyncio.run(demonstrate_analytics())
```

This analytics system provides comprehensive monitoring, intelligent recommendations, and real-time dashboard metrics for optimizing AI provider usage. The next sub-section will cover the final integration and usage examples.