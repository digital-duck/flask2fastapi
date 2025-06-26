                    return {
                        "success": response.status == 200,
                        "status_code": response.status,
                        "duration": duration
                    }
                    
            except Exception as e:
                duration = time.time() - start_time
                return {
                    "success": False,
                    "status_code": 0,
                    "duration": duration,
                    "error": str(e)
                }
        
        # Execute test
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def limited_request(request_id: int):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    return await make_request(session, request_id)
        
        tasks = [limited_request(i) for i in range(total_requests)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in raw_results if not isinstance(r, Exception) and r.get("success", False)]
        failed_results = [r for r in raw_results if isinstance(r, Exception) or not r.get("success", False)]
        
        if successful_results:
            durations = [r["duration"] for r in successful_results]
            
            return BenchmarkResult(
                test_name=test_name,
                total_requests=total_requests,
                successful_requests=len(successful_results),
                failed_requests=len(failed_results),
                success_rate=len(successful_results) / total_requests,
                total_time=total_time,
                requests_per_second=total_requests / total_time,
                avg_response_time=statistics.mean(durations),
                p50_response_time=statistics.median(durations),
                p90_response_time=statistics.quantiles(durations, n=10)[8] if len(durations) >= 10 else max(durations),
                p99_response_time=statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations),
                min_response_time=min(durations),
                max_response_time=max(durations)
            )
        else:
            return BenchmarkResult(
                test_name=test_name,
                total_requests=total_requests,
                successful_requests=0,
                failed_requests=total_requests,
                success_rate=0.0,
                total_time=total_time,
                requests_per_second=0.0,
                avg_response_time=0.0,
                p50_response_time=0.0,
                p90_response_time=0.0,
                p99_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0
            )
    
    def print_result(self, result: BenchmarkResult):
        """Print formatted benchmark result"""
        print(f"\nTest: {result.test_name}")
        print(f"Success Rate: {result.success_rate:.2%}")
        print(f"Requests/Second: {result.requests_per_second:.1f}")
        print(f"Avg Response Time: {result.avg_response_time*1000:.0f}ms")
        print(f"P90 Response Time: {result.p90_response_time*1000:.0f}ms")
        print(f"P99 Response Time: {result.p99_response_time*1000:.0f}ms")
    
    def generate_comparison_report(self):
        """Generate comparison report with Flask baseline"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        # Flask baseline (historical data)
        flask_baseline = {
            "max_concurrent_users": 700,
            "max_rps": 250,
            "avg_response_time_ms": 450,
            "p90_response_time_ms": 800,
            "p99_response_time_ms": 1200
        }
        
        # Get FastAPI results
        if self.results:
            best_result = max(self.results, key=lambda x: x.requests_per_second)
            
            fastapi_metrics = {
                "max_concurrent_users": 200,  # From stress test
                "max_rps": best_result.requests_per_second,
                "avg_response_time_ms": best_result.avg_response_time * 1000,
                "p90_response_time_ms": best_result.p90_response_time * 1000,
                "p99_response_time_ms": best_result.p99_response_time * 1000
            }
            
            # Calculate improvements
            print(f"{'Metric':<25} {'Flask':<15} {'FastAPI':<15} {'Improvement':<15}")
            print("-" * 70)
            
            for metric in flask_baseline:
                flask_val = flask_baseline[metric]
                fastapi_val = fastapi_metrics[metric]
                
                if metric == "max_concurrent_users":
                    improvement = ((fastapi_val - flask_val) / flask_val) * 100
                elif "response_time" in metric:
                    # Lower is better for response time
                    improvement = ((flask_val - fastapi_val) / flask_val) * 100
                else:
                    # Higher is better for RPS
                    improvement = ((fastapi_val - flask_val) / flask_val) * 100
                
                print(f"{metric:<25} {flask_val:<15} {fastapi_val:<15.1f} {improvement:>+.1f}%")
        
        # Export results to JSON
        self.export_results()
    
    def export_results(self):
        """Export results to JSON file"""
        export_data = {
            "timestamp": time.time(),
            "test_suite": "FastAPI Performance Benchmark",
            "results": [asdict(result) for result in self.results]
        }
        
        filename = f"benchmark_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nResults exported to: {filename}")

# Command-line execution
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI Performance Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL to test")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(args.url)
    
    if args.quick:
        # Quick test
        result = await benchmark.run_load_test(
            test_name="quick_test",
            concurrent_users=10,
            total_requests=50,
            ramp_up_time=0
        )
        benchmark.print_result(result)
    else:
        # Full benchmark suite
        await benchmark.run_benchmark_suite()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Alerting & Monitoring

### 1. CloudWatch Alarms Configuration

```python
# monitoring/alerts.py
import boto3
from typing import List, Dict
import json

class AlertManager:
    """Manage CloudWatch alarms for FastAPI application"""
    
    def __init__(self, region: str = "us-east-1"):
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sns_topic_arn = None  # Set your SNS topic ARN
    
    def create_performance_alarms(self) -> List[str]:
        """Create all performance-related alarms"""
        alarms_created = []
        
        # High error rate alarm
        alarms_created.append(self._create_error_rate_alarm())
        
        # High latency alarm
        alarms_created.append(self._create_latency_alarm())
        
        # Low throughput alarm
        alarms_created.append(self._create_throughput_alarm())
        
        # Bedrock timeout alarm
        alarms_created.append(self._create_bedrock_timeout_alarm())
        
        # Resource utilization alarms
        alarms_created.extend(self._create_resource_alarms())
        
        return alarms_created
    
    def _create_error_rate_alarm(self) -> str:
        """Create alarm for high error rate"""
        alarm_name = "ChatBot-FastAPI-HighErrorRate"
        
        self.cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='ErrorCount',
            Namespace='ChatBot/FastAPI',
            Period=300,
            Statistic='Sum',
            Threshold=10.0,
            ActionsEnabled=True,
            AlarmActions=[
                self.sns_topic_arn
            ] if self.sns_topic_arn else [],
            AlarmDescription='Alert when error rate is high',
            Dimensions=[
                {
                    'Name': 'Endpoint',
                    'Value': '/api/v1/chat'
                }
            ]
        )
        
        return alarm_name
    
    def _create_latency_alarm(self) -> str:
        """Create alarm for high response latency"""
        alarm_name = "ChatBot-FastAPI-HighLatency"
        
        self.cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=3,
            MetricName='RequestLatency',
            Namespace='ChatBot/FastAPI',
            Period=300,
            Statistic='Average',
            Threshold=2000.0,  # 2 seconds
            ActionsEnabled=True,
            AlarmActions=[
                self.sns_topic_arn
            ] if self.sns_topic_arn else [],
            AlarmDescription='Alert when average response time exceeds 2 seconds',
            Dimensions=[
                {
                    'Name': 'Endpoint',
                    'Value': '/api/v1/chat'
                }
            ]
        )
        
        return alarm_name
    
    def _create_throughput_alarm(self) -> str:
        """Create alarm for low throughput"""
        alarm_name = "ChatBot-FastAPI-LowThroughput"
        
        self.cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='LessThanThreshold',
            EvaluationPeriods=3,
            MetricName='RequestCount',
            Namespace='ChatBot/FastAPI',
            Period=300,
            Statistic='Sum',
            Threshold=10.0,  # Less than 10 requests in 5 minutes
            ActionsEnabled=True,
            AlarmActions=[
                self.sns_topic_arn
            ] if self.sns_topic_arn else [],
            AlarmDescription='Alert when request throughput is unusually low'
        )
        
        return alarm_name
    
    def _create_bedrock_timeout_alarm(self) -> str:
        """Create alarm for Bedrock timeouts"""
        alarm_name = "ChatBot-FastAPI-BedrockTimeouts"
        
        self.cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='BedrockErrorCount',
            Namespace='ChatBot/FastAPI',
            Period=300,
            Statistic='Sum',
            Threshold=5.0,
            ActionsEnabled=True,
            AlarmActions=[
                self.sns_topic_arn
            ] if self.sns_topic_arn else [],
            AlarmDescription='Alert when Bedrock errors are frequent',
            Dimensions=[
                {
                    'Name': 'Service',
                    'Value': 'Bedrock'
                }
            ]
        )
        
        return alarm_name
    
    def _create_resource_alarms(self) -> List[str]:
        """Create resource utilization alarms"""
        alarms = []
        
        # CPU utilization alarm
        cpu_alarm = "ChatBot-FastAPI-HighCPU"
        self.cloudwatch.put_metric_alarm(
            AlarmName=cpu_alarm,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=3,
            MetricName='CPUUtilization',
            Namespace='AWS/ECS',
            Period=300,
            Statistic='Average',
            Threshold=80.0,
            ActionsEnabled=True,
            AlarmActions=[
                self.sns_topic_arn
            ] if self.sns_topic_arn else [],
            AlarmDescription='Alert when CPU utilization is high',
            Dimensions=[
                {
                    'Name': 'ServiceName',
                    'Value': 'chatbot-fastapi'
                }
            ]
        )
        alarms.append(cpu_alarm)
        
        # Memory utilization alarm
        memory_alarm = "ChatBot-FastAPI-HighMemory"
        self.cloudwatch.put_metric_alarm(
            AlarmName=memory_alarm,
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=3,
            MetricName='MemoryUtilization',
            Namespace='AWS/ECS',
            Period=300,
            Statistic='Average',
            Threshold=85.0,
            ActionsEnabled=True,
            AlarmActions=[
                self.sns_topic_arn
            ] if self.sns_topic_arn else [],
            AlarmDescription='Alert when memory utilization is high',
            Dimensions=[
                {
                    'Name': 'ServiceName',
                    'Value': 'chatbot-fastapi'
                }
            ]
        )
        alarms.append(memory_alarm)
        
        return alarms

def setup_monitoring():
    """Setup complete monitoring stack"""
    alert_manager = AlertManager()
    
    print("Creating CloudWatch alarms...")
    alarms = alert_manager.create_performance_alarms()
    
    for alarm in alarms:
        print(f"Created alarm: {alarm}")
    
    print("Deploying CloudWatch dashboard...")
    from monitoring.dashboard_config import deploy_dashboard
    dashboard_url = deploy_dashboard()
    
    if dashboard_url:
        print(f"Dashboard available at: {dashboard_url}")
    
    print("Monitoring setup complete!")

if __name__ == "__main__":
    setup_monitoring()
```

### 2. Custom Health Checks

```python
# monitoring/health_checks.py
import asyncio
import aiohttp
import time
from typing import Dict, List
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class HealthCheckResult:
    """Health check result"""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float
    details: Dict
    timestamp: float

class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.checks = []
    
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks"""
        results = await asyncio.gather(
            self.check_api_health(),
            self.check_chat_functionality(),
            self.check_metrics_endpoint(),
            self.check_documentation(),
            return_exceptions=True
        )
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, HealthCheckResult)]
        
        return valid_results
    
    async def check_api_health(self) -> HealthCheckResult:
        """Check basic API health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    data = await response.json()
                    
                    if response.status == 200 and data.get("status") == "healthy":
                        status = "healthy"
                    else:
                        status = "unhealthy"
                    
                    return HealthCheckResult(
                        name="api_health",
                        status=status,
                        response_time_ms=response_time,
                        details={
                            "status_code": response.status,
                            "response": data
                        },
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="api_health",
                status="unhealthy",
                response_time_ms=response_time,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    async def check_chat_functionality(self) -> HealthCheckResult:
        """Check chat endpoint functionality"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/chat",
                    json={
                        "message": "Health check test message",
                        "session_id": f"health-check-{int(time.time())}"
                    },
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        if "response" in data and "session_id" in data:
                            if response_time < 5000:  # Less than 5 seconds
                                status = "healthy"
                            else:
                                status = "degraded"
                        else:
                            status = "unhealthy"
                    else:
                        status = "unhealthy"
                        data = {"status_code": response.status}
                    
                    return HealthCheckResult(
                        name="chat_functionality",
                        status=status,
                        response_time_ms=response_time,
                        details={
                            "status_code": response.status,
                            "has_response": "response" in data if isinstance(data, dict) else False
                        },
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="chat_functionality",
                status="unhealthy",
                response_time_ms=response_time,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    async def check_metrics_endpoint(self) -> HealthCheckResult:
        """Check metrics endpoint"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/metrics/performance",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        required_fields = ["qps", "avg_latency_ms", "error_rate"]
                        has_all_fields = all(field in data for field in required_fields)
                        
                        status = "healthy" if has_all_fields else "degraded"
                    else:
                        status = "unhealthy"
                        data = {}
                    
                    return HealthCheckResult(
                        name="metrics_endpoint",
                        status=status,
                        response_time_ms=response_time,
                        details={
                            "status_code": response.status,
                            "metrics_available": bool(data)
                        },
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="metrics_endpoint",
                status="unhealthy",
                response_time_ms=response_time,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    async def check_documentation(self) -> HealthCheckResult:
        """Check API documentation availability"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/docs",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    status = "healthy" if response.status == 200 else "unhealthy"
                    
                    return HealthCheckResult(
                        name="api_documentation",
                        status=status,
                        response_time_ms=response_time,
                        details={"status_code": response.status},
                        timestamp=time.time()
                    )
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="api_documentation",
                status="unhealthy",
                response_time_ms=response_time,
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def generate_health_report(self, results: List[HealthCheckResult]) -> Dict:
        """Generate comprehensive health report"""
        total_checks = len(results)
        healthy_checks = len([r for r in results if r.status == "healthy"])
        degraded_checks = len([r for r in results if r.status == "degraded"])
        unhealthy_checks = len([r for r in results if r.status == "unhealthy"])
        
        # Determine overall status
        if unhealthy_checks > 0:
            overall_status = "unhealthy"
        elif degraded_checks > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        avg_response_time = sum(r.response_time_ms for r in results) / len(results) if results else 0
        
        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "summary": {
                "total_checks": total_checks,
                "healthy": healthy_checks,
                "degraded": degraded_checks,
                "unhealthy": unhealthy_checks,
                "avg_response_time_ms": avg_response_time
            },
            "details": [
                {
                    "name": r.name,
                    "status": r.status,
                    "response_time_ms": r.response_time_ms,
                    "details": r.details
                } for r in results
            ]
        }

# Standalone health check script
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI Health Check")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL to check")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.url)
    results = await checker.run_all_checks()
    report = checker.generate_health_report(results)
    
    if args.json:
        import json
        print(json.dumps(report, indent=2))
    else:
        print(f"Overall Status: {report['overall_status'].upper()}")
        print(f"Healthy Checks: {report['summary']['healthy']}/{report['summary']['total_checks']}")
        print(f"Average Response Time: {report['summary']['avg_response_time_ms']:.0f}ms")
        
        for detail in report['details']:
            status_icon = "✅" if detail['status'] == "healthy" else "⚠️" if detail['status'] == "degraded" else "❌"
            print(f"{status_icon} {detail['name']}: {detail['status']} ({detail['response_time_ms']:.0f}ms)")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Migration Success Validation

### KPI Dashboard Template

```python
# monitoring/migration_kpis.py
from typing import Dict, List
import time
import json

class MigrationKPITracker:
    """Track and validate migration success metrics"""
    
    def __init__(self):
        self.flask_baseline = {
            "concurrent_users": 700,
            "avg_qps": 250,
            "avg_latency_ms": 450,
            "p90_latency_ms": 800,
            "p99_latency_ms": 1200,
            "error_rate_percent": 2.1,
            "monthly_cost": 2000,
            "ecs_instances": 4
        }
        
        self.success_targets = {
            "concurrent_users": 2000,  # 3x improvement
            "avg_qps": 750,           # 3x improvement
            "avg_latency_ms": 300,    # 33% improvement
            "p90_latency_ms": 500,    # 37% improvement
            "p99_latency_ms": 800,    # 33% improvement
            "error_rate_percent": 1.0, # 50% improvement
            "monthly_cost": 1000,     # 50% reduction
            "ecs_instances": 2        # 50% reduction
        }
    
    def calculate_improvements(self, current_metrics: Dict) -> Dict:
        """Calculate improvement percentages vs Flask baseline"""
        improvements = {}
        
        for metric, flask_value in self.flask_baseline.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                
                if metric in ["avg_latency_ms", "p90_latency_ms", "p99_latency_ms", "error_rate_percent", "monthly_cost", "ecs_instances"]:
                    # Lower is better
                    improvement = ((flask_value - current_value) / flask_value) * 100
                else:
                    # Higher is better
                    improvement = ((current_value - flask_value) / flask_value) * 100
                
                improvements[metric] = {
                    "flask_baseline": flask_value,
                    "fastapi_current": current_value,
                    "improvement_percent": improvement,
                    "target": self.success_targets.get(metric),
                    "target_met": self._check_target_met(metric, current_value)
                }
        
        return improvements
    
    def _check_target_met(self, metric: str, current_value: float) -> bool:
        """Check if target is met for given metric"""
        target = self.success_targets.get(metric)
        if target is None:
            return False
        
        if metric in ["avg_latency_ms", "p90_latency_ms", "p99_latency_ms", "error_rate_percent", "monthly_cost", "ecs_instances"]:
            # Lower is better
            return current_value <= target
        else:
            # Higher is better
            return current_value >= target
    
    def generate_migration_report(self, current_metrics: Dict) -> Dict:
        """Generate comprehensive migration success report"""
        improvements = self.calculate_improvements(current_metrics)
        
        # Calculate overall success score
        targets_met = sum(1 for imp in improvements.values() if imp["target_met"])
        total_targets = len(improvements)
        success_score = (targets_met / total_targets) * 100
        
        # Determine migration status
        if success_score >= 80:
            migration_status = "SUCCESSFUL"
        elif success_score >= 60:
            migration_status = "PARTIALLY_SUCCESSFUL"
        else:
            migration_status = "NEEDS_IMPROVEMENT"
        
        # Calculate cost savings
        cost_savings_annual = (self.flask_baseline["monthly_cost"] - current_metrics.get("monthly_cost", self.flask_baseline["monthly_cost"])) * 12
        
        return {
            "migration_status": migration_status,
            "success_score": success_score,
            "targets_met": targets_met,
            "total_targets": total_targets,
            "timestamp": time.time(),
            "cost_savings": {
                "monthly": self.flask_baseline["monthly_cost"] - current_metrics.get("monthly_cost", self.flask_baseline["monthly_cost"]),
                "annual": cost_savings_annual
            },
            "performance_improvements": improvements,
            "summary": self._generate_summary(improvements)
        }
    
    def _generate_summary(self, improvements: Dict) -> List[str]:
        """Generate human-readable summary"""
        summary = []
        
        for metric, data in improvements.items():
            improvement_pct = data["improvement_percent"]
            target_met = data["target_met"]
            
            metric_name = metric.replace("_", " ").title()
            
            if improvement_pct > 0:
                summary.append(f"✅ {metric_name}: {improvement_pct:+.1f}% improvement (Target {'✓' if target_met else '✗'})")
            else:
                summary.append(f"❌ {metric_name}: {improvement_pct:+.1f}% change (Target {'✓' if target_met else '✗'})")
        
        return summary

# Example usage
def create_sample_report():
    """Create sample migration report"""
    tracker = MigrationKPITracker()
    
    # Sample current metrics (replace with actual data from monitoring)
    current_metrics = {
        "concurrent_users": 2100,
        "avg_qps": 850,
        "avg_latency_ms": 280,
        "p90_latency_ms": 420,
        "p99_latency_ms": 650,
        "error_rate_percent": 0.8,
        "monthly_cost": 800,
        "ecs_instances": 1
    }
    
    report = tracker.generate_migration_report(current_metrics)
    
    print("FastAPI Migration Success Report")
    print("=" * 50)
    print(f"Migration Status: {report['migration_status']}")
    print(f"Success Score: {report['success_score']:.1f}%")
    print(f"Targets Met: {report['targets_met']}/{report['total_targets']}")
    print(f"Annual Cost Savings: ${report['cost_savings']['annual']:,}")
    print("\nDetailed Improvements:")
    for summary_line in report['summary']:
        print(f"  {summary_line}")
    
    return report

if __name__ == "__main__":
    create_sample_report()
```

---

*Next: Chapter 6 - Deployment & Operations*# Flask to FastAPI Migration Guide
## Chapter 5: Performance Monitoring & Metrics

### Overview

This chapter focuses on implementing comprehensive performance monitoring to validate the migration success and provide ongoing visibility into system performance. The metrics will provide concrete data to demonstrate the business value of the FastAPI migration.

---

## Key Performance Indicators (KPIs)

### Primary Metrics to Track

**Performance Metrics:**
- **Queries Per Second (QPS)** - Request throughput
- **Latency P50/P90/P99** - Response time percentiles
- **Error Rate** - Percentage of failed requests
- **Concurrent User Capacity** - Maximum simultaneous users

**Business Metrics:**
- **Infrastructure Cost** - Monthly AWS spend
- **User Experience** - Average response times
- **System Reliability** - Uptime percentage
- **Resource Utilization** - CPU, Memory, Network usage

---

## CloudWatch Metrics Implementation

### 1. Core Metrics Service

```python
# services/metrics_service.py
import aioboto3
import asyncio
import time
import structlog
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, deque
import statistics

from config.settings import settings

logger = structlog.get_logger()

class MetricsService:
    """Async CloudWatch metrics service with intelligent batching"""
    
    def __init__(self, namespace: str = None):
        self.namespace = namespace or settings.metrics_namespace
        self.session = aioboto3.Session()
        self._metrics_buffer = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task = None
        
        # In-memory metrics for real-time dashboard
        self._request_times = deque(maxlen=1000)
        self._error_count = 0
        self._total_requests = 0
        self._start_time = time.time()
    
    async def start_background_flush(self):
        """Start background task to flush metrics periodically"""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._periodic_flush())
    
    async def stop_background_flush(self):
        """Stop background flush task"""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            finally:
                await self._flush_metrics()  # Final flush
    
    async def _periodic_flush(self):
        """Periodically flush metrics buffer"""
        while True:
            try:
                await asyncio.sleep(60)  # Flush every minute
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Periodic metrics flush failed", error=str(e))
    
    async def put_metric(
        self, 
        metric_name: str, 
        value: float,
        unit: str = "Count",
        dimensions: Dict[str, str] = None,
        timestamp: datetime = None
    ):
        """
        Put single metric to CloudWatch buffer
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement (Count, Milliseconds, etc.)
            dimensions: Additional dimensions for the metric
            timestamp: Metric timestamp (defaults to now)
        """
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': timestamp or datetime.utcnow()
        }
        
        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
        
        async with self._buffer_lock:
            self._metrics_buffer.append(metric_data)
            
            # Auto-flush if buffer is getting full
            if len(self._metrics_buffer) >= 20:
                await self._flush_metrics()
    
    async def _flush_metrics(self):
        """Flush metrics buffer to CloudWatch"""
        if not self._metrics_buffer:
            return
            
        async with self._buffer_lock:
            metrics_to_send = self._metrics_buffer.copy()
            self._metrics_buffer.clear()
        
        if not metrics_to_send:
            return
        
        try:
            async with self.session.client('cloudwatch') as cw:
                # CloudWatch accepts max 20 metrics per call
                for i in range(0, len(metrics_to_send), 20):
                    batch = metrics_to_send[i:i+20]
                    await cw.put_metric_data(
                        Namespace=self.namespace,
                        MetricData=batch
                    )
                    
            logger.debug(
                "Metrics flushed to CloudWatch",
                metrics_count=len(metrics_to_send)
            )
            
        except Exception as e:
            logger.error(
                "Failed to flush metrics to CloudWatch",
                error=str(e),
                metrics_count=len(metrics_to_send)
            )
    
    async def record_request_metrics(
        self, 
        endpoint: str, 
        method: str,
        status_code: int, 
        duration: float,
        user_agent: str = None
    ):
        """Record comprehensive request metrics"""
        # Update in-memory counters
        self._request_times.append(duration)
        self._total_requests += 1
        if status_code >= 400:
            self._error_count += 1
        
        dimensions = {
            'Endpoint': endpoint,
            'Method': method,
            'StatusCode': str(status_code)
        }
        
        # Add user agent dimension if available
        if user_agent:
            # Simplify user agent for grouping
            if 'mobile' in user_agent.lower():
                dimensions['ClientType'] = 'Mobile'
            elif 'bot' in user_agent.lower():
                dimensions['ClientType'] = 'Bot'
            else:
                dimensions['ClientType'] = 'Desktop'
        
        # Send multiple metrics
        await asyncio.gather(
            self.put_metric('RequestCount', 1, 'Count', dimensions),
            self.put_metric('RequestLatency', duration * 1000, 'Milliseconds', dimensions),
            self.put_metric(
                'ErrorCount' if status_code >= 400 else 'SuccessCount', 
                1, 'Count', dimensions
            )
        )
    
    async def record_bedrock_metrics(
        self, 
        duration: float, 
        success: bool,
        tokens_used: int = None,
        model_id: str = None
    ):
        """Record Bedrock-specific metrics"""
        dimensions = {'Service': 'Bedrock'}
        if model_id:
            dimensions['ModelId'] = model_id
        
        metrics = [
            ('BedrockLatency', duration * 1000, 'Milliseconds'),
            ('BedrockRequestCount', 1, 'Count'),
        ]
        
        if success:
            metrics.append(('BedrockSuccessCount', 1, 'Count'))
            if tokens_used:
                metrics.append(('BedrockTokensUsed', tokens_used, 'Count'))
        else:
            metrics.append(('BedrockErrorCount', 1, 'Count'))
        
        # Send all metrics
        for metric_name, value, unit in metrics:
            await self.put_metric(metric_name, value, unit, dimensions)
    
    async def record_custom_metric(
        self, 
        metric_name: str, 
        value: float,
        dimensions: Dict[str, str] = None
    ):
        """Record custom business metrics"""
        await self.put_metric(
            metric_name, 
            value, 
            'Count', 
            dimensions or {}
        )
    
    def get_realtime_metrics(self) -> Dict:
        """Get real-time metrics from in-memory data"""
        if not self._request_times:
            return {
                "qps": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p90_latency_ms": 0,
                "p99_latency_ms": 0,
                "error_rate": 0,
                "total_requests": 0,
                "uptime_seconds": time.time() - self._start_time
            }
        
        uptime = time.time() - self._start_time
        request_times = list(self._request_times)
        
        return {
            "qps": self._total_requests / uptime if uptime > 0 else 0,
            "avg_latency_ms": statistics.mean(request_times) * 1000,
            "p50_latency_ms": statistics.median(request_times) * 1000,
            "p90_latency_ms": statistics.quantiles(request_times, n=10)[8] * 1000 if len(request_times) >= 10 else max(request_times) * 1000,
            "p99_latency_ms": statistics.quantiles(request_times, n=100)[98] * 1000 if len(request_times) >= 100 else max(request_times) * 1000,
            "error_rate": (self._error_count / self._total_requests) * 100 if self._total_requests > 0 else 0,
            "total_requests": self._total_requests,
            "uptime_seconds": uptime
        }

# Global metrics instance
metrics_service = MetricsService()
```

### 2. Automatic Metrics Middleware

```python
# middleware/metrics_middleware.py
import time
import asyncio
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog

from services.metrics_service import metrics_service

logger = structlog.get_logger()

class PerformanceMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically collect performance metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.active_requests = 0
        self.peak_concurrent_requests = 0
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect comprehensive metrics"""
        start_time = time.time()
        
        # Track concurrent requests
        self.active_requests += 1
        if self.active_requests > self.peak_concurrent_requests:
            self.peak_concurrent_requests = self.active_requests
        
        # Extract request information
        method = request.method
        path = self._normalize_path(request.url.path)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Add request ID for tracing
        request_id = request.headers.get("x-request-id", f"req-{int(time.time()*1000)}")
        
        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Handle exceptions
            logger.error(
                "Request processing failed",
                method=method,
                path=path,
                error=str(e),
                request_id=request_id,
                exc_info=True
            )
            status_code = 500
            
            # Create error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id
                }
            )
        finally:
            # Always decrement active requests
            self.active_requests -= 1
        
        # Calculate metrics
        duration = time.time() - start_time
        
        # Log request details
        logger.info(
            "Request completed",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration * 1000,
            client_ip=client_ip,
            request_id=request_id,
            active_requests=self.active_requests
        )
        
        # Record metrics asynchronously
        asyncio.create_task(
            metrics_service.record_request_metrics(
                endpoint=path,
                method=method,
                status_code=status_code,
                duration=duration,
                user_agent=user_agent
            )
        )
        
        # Record concurrent requests metric
        if self.active_requests > 0:
            asyncio.create_task(
                metrics_service.put_metric(
                    'ConcurrentRequests', 
                    self.active_requests, 
                    'Count'
                )
            )
        
        # Add response headers for debugging
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics grouping"""
        # Replace dynamic path parameters with placeholders
        # e.g., /users/123 -> /users/{id}
        import re
        
        # Common patterns
        patterns = [
            (r'/\d+', '/{id}'),  # Numbers
            (r'/[0-9a-fA-F-]{36}', '/{uuid}'),  # UUIDs
            (r'/sess_[a-zA-Z0-9]+', '/{session_id}'),  # Session IDs
        ]
        
        normalized = path
        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        return normalized
```

### 3. Enhanced Bedrock Service with Metrics

```python
# services/bedrock_service_with_metrics.py
import aioboto3
import asyncio
import time
import structlog
from typing import Dict, Optional

from config.settings import settings
from services.metrics_service import metrics_service

logger = structlog.get_logger()

class EnhancedBedrockService:
    """Bedrock service with comprehensive metrics collection"""
    
    def __init__(self):
        self.region_name = settings.aws_region
        self.agent_id = settings.bedrock_agent_id
        self.session = aioboto3.Session()
        
        # Performance tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.total_tokens = 0
    
    async def invoke_agent(
        self, 
        message: str, 
        session_id: str,
        timeout: float = None
    ) -> Dict:
        """Enhanced invoke_agent with detailed metrics"""
        start_time = time.time()
        timeout = timeout or settings.request_timeout
        success = False
        tokens_used = 0
        model_id = "unknown"
        
        # Increment call counter
        self.total_calls += 1
        
        try:
            async with self.session.client(
                'bedrock-agent',
                region_name=self.region_name
            ) as client:
                
                # Make the Bedrock call with timeout
                response = await asyncio.wait_for(
                    client.invoke_agent(
                        agentId=self.agent_id,
                        sessionId=session_id,
                        inputText=message
                    ),
                    timeout=timeout
                )
                
                # Extract response data
                success = True
                self.successful_calls += 1
                tokens_used = response.get('usage', {}).get('totalTokens', 0)
                model_id = response.get('modelId', 'bedrock-agent')
                self.total_tokens += tokens_used
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Log successful call
                logger.info(
                    "Bedrock invocation successful",
                    session_id=session_id,
                    duration_ms=duration * 1000,
                    tokens_used=tokens_used,
                    message_length=len(message),
                    response_length=len(response.get('output', {}).get('text', ''))
                )
                
                return {
                    'response': response['output']['text'],
                    'session_id': session_id,
                    'tokens_used': tokens_used,
                    'model_id': model_id,
                    'processing_time': duration
                }
                
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.warning(
                "Bedrock request timeout",
                session_id=session_id,
                timeout=timeout,
                duration_ms=duration * 1000
            )
            return {
                "error": f"Request timeout after {timeout} seconds",
                "session_id": session_id
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Bedrock invocation failed",
                session_id=session_id,
                error=str(e),
                duration_ms=duration * 1000,
                exc_info=True
            )
            return {
                "error": f"Bedrock error: {str(e)}",
                "session_id": session_id
            }
            
        finally:
            # Always record metrics
            duration = time.time() - start_time
            await metrics_service.record_bedrock_metrics(
                duration=duration,
                success=success,
                tokens_used=tokens_used,
                model_id=model_id
            )
            
            # Record business metrics
            await metrics_service.record_custom_metric(
                "BedrockAPICallsTotal",
                self.total_calls,
                {"Agent": self.agent_id}
            )
            
            if success:
                await metrics_service.record_custom_metric(
                    "BedrockTokensTotal",
                    self.total_tokens,
                    {"Agent": self.agent_id}
                )
    
    def get_service_stats(self) -> Dict:
        """Get service-level statistics"""
        success_rate = (self.successful_calls / self.total_calls) * 100 if self.total_calls > 0 else 0
        avg_tokens = self.total_tokens / self.successful_calls if self.successful_calls > 0 else 0
        
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "success_rate": success_rate,
            "total_tokens_used": self.total_tokens,
            "avg_tokens_per_call": avg_tokens
        }
```

---

## Real-Time Performance Dashboard

### 1. Metrics Endpoint

```python
# routers/metrics.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Optional
import time

from services.metrics_service import metrics_service
from dependencies import get_bedrock_service
from middleware.metrics_middleware import PerformanceMetricsMiddleware

router = APIRouter()

class PerformanceMetrics(BaseModel):
    """Real-time performance metrics model"""
    qps: float
    avg_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    total_requests: int
    uptime_seconds: float
    concurrent_requests: int

class SystemHealth(BaseModel):
    """System health status model"""
    status: str
    timestamp: float
    performance: PerformanceMetrics
    services: Dict[str, str]
    version: str

@router.get("/metrics/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get real-time performance metrics"""
    return PerformanceMetrics(**metrics_service.get_realtime_metrics())

@router.get("/metrics/health", response_model=SystemHealth)
async def get_system_health(
    bedrock_service = Depends(get_bedrock_service)
):
    """Get comprehensive system health"""
    # Get performance metrics
    perf_metrics = metrics_service.get_realtime_metrics()
    
    # Check service health
    services_status = {
        "bedrock": "healthy",  # Add actual health check
        "dynamodb": "healthy",  # Add actual health check
        "cloudwatch": "healthy"
    }
    
    # Determine overall status
    overall_status = "healthy"
    if perf_metrics["error_rate"] > 5.0:
        overall_status = "degraded"
    elif perf_metrics["error_rate"] > 10.0:
        overall_status = "unhealthy"
    
    return SystemHealth(
        status=overall_status,
        timestamp=time.time(),
        performance=PerformanceMetrics(**perf_metrics),
        services=services_status,
        version="2.0.0"
    )

@router.get("/metrics/bedrock")
async def get_bedrock_metrics(
    bedrock_service = Depends(get_bedrock_service)
):
    """Get Bedrock service metrics"""
    if hasattr(bedrock_service, 'get_service_stats'):
        return bedrock_service.get_service_stats()
    return {"message": "Bedrock metrics not available"}

@router.get("/metrics/comparison")
async def get_comparison_metrics():
    """Get Flask vs FastAPI comparison metrics"""
    current_metrics = metrics_service.get_realtime_metrics()
    
    # Simulated Flask baseline (replace with actual historical data)
    flask_baseline = {
        "qps": 250,
        "avg_latency_ms": 450,
        "p90_latency_ms": 800,
        "p99_latency_ms": 1200,
        "error_rate": 2.1,
        "concurrent_capacity": 700
    }
    
    # Calculate improvements
    improvements = {}
    for metric, flask_value in flask_baseline.items():
        if metric in current_metrics:
            current_value = current_metrics[metric]
            if metric == "error_rate":
                # Lower is better for error rate
                improvement = ((flask_value - current_value) / flask_value) * 100
            else:
                # Higher is better for other metrics
                improvement = ((current_value - flask_value) / flask_value) * 100
            
            improvements[f"{metric}_improvement_percent"] = improvement
    
    return {
        "flask_baseline": flask_baseline,
        "fastapi_current": current_metrics,
        "improvements": improvements
    }
```

### 2. CloudWatch Dashboard Configuration

```python
# monitoring/dashboard_config.py
import json
from typing import Dict

def create_cloudwatch_dashboard_config(region: str = "us-east-1") -> Dict:
    """Generate CloudWatch dashboard configuration"""
    
    dashboard_config = {
        "widgets": [
            # Queries Per Second Widget
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "RequestCount", "Method", "POST", "Endpoint", "/api/v1/chat"],
                        [".", ".", ".", "GET", ".", "/health"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Queries Per Second",
                    "period": 60,
                    "stat": "Sum",
                    "yAxis": {
                        "left": {
                            "min": 0
                        }
                    }
                }
            },
            
            # Latency Percentiles Widget
            {
                "type": "metric",
                "x": 12, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "RequestLatency", "Endpoint", "/api/v1/chat", {"stat": "Average"}],
                        ["...", {"stat": "p50"}],
                        ["...", {"stat": "p90"}],
                        ["...", {"stat": "p99"}]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Response Time Percentiles (P50, P90, P99)",
                    "period": 300,
                    "yAxis": {
                        "left": {
                            "min": 0,
                            "max": 5000
                        }
                    }
                }
            },
            
            # Error Rate Widget
            {
                "type": "metric",
                "x": 0, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "ErrorCount", "Endpoint", "/api/v1/chat"],
                        [".", "SuccessCount", ".", "."]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Success vs Error Rate",
                    "period": 300,
                    "stat": "Sum"
                }
            },
            
            # Bedrock Performance Widget
            {
                "type": "metric",
                "x": 12, "y": 6, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "BedrockLatency", "Service", "Bedrock"],
                        [".", "BedrockTokensUsed", ".", "."]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Bedrock Performance",
                    "period": 300,
                    "stat": "Average"
                }
            },
            
            # Concurrent Requests Widget
            {
                "type": "metric",
                "x": 0, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "metrics": [
                        ["ChatBot/FastAPI", "ConcurrentRequests"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": region,
                    "title": "Concurrent Requests",
                    "period": 60,
                    "stat": "Maximum"
                }
            },
            
            # Cost Optimization Widget
            {
                "type": "metric",
                "x": 12, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "view": "singleValue",
                    "metrics": [
                        ["AWS/ECS", "CPUUtilization", "ServiceName", "chatbot-fastapi"],
                        [".", "MemoryUtilization", ".", "."]
                    ],
                    "region": region,
                    "title": "Resource Utilization",
                    "period": 300,
                    "stat": "Average"
                }
            }
        ]
    }
    
    return dashboard_config

def deploy_dashboard(dashboard_name: str = "ChatBot-FastAPI-Performance"):
    """Deploy CloudWatch dashboard"""
    import boto3
    
    cloudwatch = boto3.client('cloudwatch')
    
    dashboard_config = create_cloudwatch_dashboard_config()
    
    try:
        cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_config)
        )
        print(f"Dashboard '{dashboard_name}' deployed successfully")
        
        # Return dashboard URL
        region = boto3.Session().region_name or "us-east-1"
        dashboard_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={dashboard_name}"
        print(f"Dashboard URL: {dashboard_url}")
        
        return dashboard_url
        
    except Exception as e:
        print(f"Failed to deploy dashboard: {e}")
        return None
```

---

## Load Testing & Benchmarking

### 1. Automated Load Testing

```python
# tests/performance_benchmark.py
import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    total_time: float
    requests_per_second: float
    avg_response_time: float
    p50_response_time: float
    p90_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def run_benchmark_suite(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        print("Starting FastAPI Performance Benchmark Suite")
        print("=" * 60)
        
        # Test scenarios
        scenarios = [
            ("baseline_10_users", 10, 100, 0),
            ("moderate_50_users", 50, 500, 5),
            ("high_load_100_users", 100, 1000, 10),
            ("stress_200_users", 200, 2000, 15),
        ]
        
        for name, concurrent_users, total_requests, ramp_up in scenarios:
            print(f"\nRunning {name}...")
            result = await self.run_load_test(
                test_name=name,
                concurrent_users=concurrent_users,
                total_requests=total_requests,
                ramp_up_time=ramp_up
            )
            self.results.append(result)
            self.print_result(result)
            
            # Cool down between tests
            await asyncio.sleep(5)
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    async def run_load_test(
        self,
        test_name: str,
        concurrent_users: int,
        total_requests: int,
        ramp_up_time: float
    ) -> BenchmarkResult:
        """Run individual load test"""
        
        async def make_request(session: aiohttp.ClientSession, request_id: int):
            start_time = time.time()
            
            try:
                # Add ramp-up delay
                if ramp_up_time > 0:
                    delay = (request_id / total_requests) * ramp_up_time
                    await asyncio.sleep(delay)
                
                async with session.post(
                    f"{self.base_url}/api/v1/chat",
                    json={
                        "message": f"Benchmark test message {request_id}",
                        "session_id": f"bench-{test_name}-{request_id}"
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    await response.json()
                    duration = time.time() - start_time
                    
                    return {
                        "success": response.status == 200,
                        "status_code": response.status,
                        