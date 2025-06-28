# Flask to FastAPI Migration Guide

## Chapter 6: Deployment & Operations

### Overview

This chapter covers the deployment strategy, infrastructure configuration, and operational procedures for safely migrating from Flask to FastAPI in production. We'll use a blue-green deployment approach to minimize risk and ensure zero downtime.

---

## Deployment Strategy

### Blue-Green Deployment Approach

**Blue Environment (Current Flask)**
- Existing 4 ECS instances running Flask
- Current traffic routing through API Gateway
- Existing monitoring and alerting

**Green Environment (New FastAPI)**
- New ECS service with FastAPI containers
- Parallel infrastructure setup
- Independent monitoring during validation

**Migration Phases:**
1. **Preparation**: Deploy green environment alongside blue
2. **Validation**: Test green environment with synthetic traffic
3. **Gradual Migration**: Route 10% → 50% → 100% traffic to green
4. **Cleanup**: Decommission blue environment

---

## Infrastructure Configuration

### 1. ECS Service Definition (FastAPI)

```json
{
  "family": "chatbot-fastapi",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/chatbot-task-role",
  "containerDefinitions": [
    {
      "name": "chatbot-fastapi",
      "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/chatbot-fastapi:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-east-1"
        },
        {
          "name": "BEDROCK_AGENT_ID",
          "value": "YOUR_AGENT_ID"
        },
        {
          "name": "DYNAMODB_TABLE",
          "value": "chat_sessions"
        },
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        },
        {
          "name": "METRICS_NAMESPACE",
          "value": "ChatBot/FastAPI"
        }
      ],
      "secrets": [
        {
          "name": "API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:chatbot/api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/aws/ecs/chatbot-fastapi",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs",
          "awslogs-create-group": "true"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "command": [
        "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "1",
        "--access-log",
        "--loop", "uvloop"
      ]
    }
  ]
}
```

### 2. ECS Service Configuration

```yaml
# deploy/ecs-service.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ecs-service-config
data:
  service-definition: |
    {
      "serviceName": "chatbot-fastapi",
      "cluster": "chatbot-cluster",
      "taskDefinition": "chatbot-fastapi:LATEST",
      "desiredCount": 1,
      "launchType": "FARGATE",
      "networkConfiguration": {
        "awsvpcConfiguration": {
          "subnets": [
            "subnet-12345678",
            "subnet-87654321"
          ],
          "securityGroups": [
            "sg-chatbot-fastapi"
          ],
          "assignPublicIp": "DISABLED"
        }
      },
      "loadBalancers": [
        {
          "targetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT:targetgroup/chatbot-fastapi-tg/1234567890123456",
          "containerName": "chatbot-fastapi",
          "containerPort": 8000
        }
      ],
      "serviceRegistries": [
        {
          "registryArn": "arn:aws:servicediscovery:us-east-1:ACCOUNT:service/srv-chatbot-fastapi"
        }
      ],
      "deploymentConfiguration": {
        "maximumPercent": 200,
        "minimumHealthyPercent": 50,
        "deploymentCircuitBreaker": {
          "enable": true,
          "rollback": true
        }
      },
      "healthCheckGracePeriodSeconds": 60,
      "enableExecuteCommand": true,
      "tags": [
        {
          "key": "Environment",
          "value": "production"
        },
        {
          "key": "Application",
          "value": "chatbot"
        },
        {
          "key": "Framework",
          "value": "fastapi"
        }
      ]
    }
```

### 3. Load Balancer Configuration

```yaml
# deploy/alb-config.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: alb-config
data:
  target-group: |
    {
      "Name": "chatbot-fastapi-tg",
      "Protocol": "HTTP",
      "Port": 8000,
      "VpcId": "vpc-12345678",
      "HealthCheckProtocol": "HTTP",
      "HealthCheckPath": "/health",
      "HealthCheckIntervalSeconds": 30,
      "HealthCheckTimeoutSeconds": 5,
      "HealthyThresholdCount": 2,
      "UnhealthyThresholdCount": 3,
      "TargetType": "ip",
      "Matcher": {
        "HttpCode": "200"
      },
      "HealthCheckGracePeriodSeconds": 60,
      "Tags": [
        {
          "Key": "Name",
          "Value": "chatbot-fastapi-targets"
        },
        {
          "Key": "Environment",
          "Value": "production"
        }
      ]
    }
  
  listener-rule: |
    {
      "Priority": 100,
      "Conditions": [
        {
          "Field": "host-header",
          "Values": ["api.chatbot.company.com"]
        },
        {
          "Field": "path-pattern",
          "Values": ["/api/v1/*", "/health", "/metrics/*"]
        }
      ],
      "Actions": [
        {
          "Type": "forward",
          "TargetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT:targetgroup/chatbot-fastapi-tg/1234567890123456"
        }
      ]
    }
```

---

## Deployment Automation

### 1. Deployment Script

```bash
#!/bin/bash
# deploy/deploy.sh

set -e

# Configuration
CLUSTER_NAME="chatbot-cluster"
SERVICE_NAME="chatbot-fastapi"
TASK_FAMILY="chatbot-fastapi"
ECR_REPOSITORY="chatbot-fastapi"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    echo_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo_error "Docker is not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo_error "AWS credentials not configured"
        exit 1
    fi
    
    echo_success "Prerequisites check passed"
}

# Function to build and push Docker image
build_and_push_image() {
    echo_info "Building Docker image..."
    
    # Build image with git commit as tag
    GIT_COMMIT=$(git rev-parse --short HEAD)
    IMAGE_TAG="${GIT_COMMIT}"
    FULL_IMAGE_NAME="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"
    LATEST_IMAGE_NAME="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:latest"
    
    # Build image
    docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
    docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${FULL_IMAGE_NAME}
    docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${LATEST_IMAGE_NAME}
    
    # Login to ECR
    echo_info "Logging in to ECR..."
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    # Push images
    echo_info "Pushing image to ECR..."
    docker push ${FULL_IMAGE_NAME}
    docker push ${LATEST_IMAGE_NAME}
    
    echo_success "Image pushed: ${FULL_IMAGE_NAME}"
    echo "IMAGE_URI=${FULL_IMAGE_NAME}" > .env.deploy
}

# Function to update task definition
update_task_definition() {
    echo_info "Updating task definition..."
    
    # Get current task definition
    CURRENT_TASK_DEF=$(aws ecs describe-task-definition --task-definition ${TASK_FAMILY} --region ${AWS_REGION})
    
    # Extract the task definition (without revision info)
    NEW_TASK_DEF=$(echo ${CURRENT_TASK_DEF} | jq '.taskDefinition | del(.taskDefinitionArn, .revision, .status, .requiresAttributes, .placementConstraints, .compatibilities, .registeredAt, .registeredBy)')
    
    # Update image URI
    source .env.deploy
    NEW_TASK_DEF=$(echo ${NEW_TASK_DEF} | jq --arg IMAGE "${IMAGE_URI}" '.containerDefinitions[0].image = $IMAGE')
    
    # Register new task definition
    NEW_TASK_INFO=$(aws ecs register-task-definition --region ${AWS_REGION} --cli-input-json "${NEW_TASK_DEF}")
    NEW_REVISION=$(echo ${NEW_TASK_INFO} | jq '.taskDefinition.revision')
    
    echo_success "New task definition registered: ${TASK_FAMILY}:${NEW_REVISION}"
    echo "TASK_DEFINITION_ARN=${TASK_FAMILY}:${NEW_REVISION}" >> .env.deploy
}

# Function to deploy to ECS
deploy_to_ecs() {
    echo_info "Deploying to ECS..."
    
    source .env.deploy
    
    # Update service
    aws ecs update-service \
        --cluster ${CLUSTER_NAME} \
        --service ${SERVICE_NAME} \
        --task-definition ${TASK_DEFINITION_ARN} \
        --region ${AWS_REGION} > /dev/null
    
    echo_info "Waiting for deployment to complete..."
    
    # Wait for deployment to stabilize
    aws ecs wait services-stable \
        --cluster ${CLUSTER_NAME} \
        --services ${SERVICE_NAME} \
        --region ${AWS_REGION}
    
    echo_success "Deployment completed successfully"
}

# Function to run health checks
run_health_checks() {
    echo_info "Running health checks..."
    
    # Get service endpoint
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --names chatbot-alb \
        --query 'LoadBalancers[0].DNSName' \
        --output text \
        --region ${AWS_REGION})
    
    HEALTH_URL="http://${ALB_DNS}/health"
    
    # Wait for service to be healthy
    for i in {1..30}; do
        if curl -f -s ${HEALTH_URL} > /dev/null; then
            echo_success "Health check passed"
            return 0
        fi
        echo_info "Health check attempt ${i}/30 failed, retrying in 10 seconds..."
        sleep 10
    done
    
    echo_error "Health checks failed after 5 minutes"
    return 1
}

# Function to run smoke tests
run_smoke_tests() {
    echo_info "Running smoke tests..."
    
    ALB_DNS=$(aws elbv2 describe-load-balancers \
        --names chatbot-alb \
        --query 'LoadBalancers[0].DNSName' \
        --output text \
        --region ${AWS_REGION})
    
    # Test chat endpoint
    CHAT_RESPONSE=$(curl -s -X POST "http://${ALB_DNS}/api/v1/chat" \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello, deployment test"}')
    
    if echo ${CHAT_RESPONSE} | jq -e '.response' > /dev/null; then
        echo_success "Chat endpoint smoke test passed"
    else
        echo_error "Chat endpoint smoke test failed"
        echo "Response: ${CHAT_RESPONSE}"
        return 1
    fi
    
    # Test metrics endpoint
    METRICS_RESPONSE=$(curl -s "http://${ALB_DNS}/metrics/performance")
    
    if echo ${METRICS_RESPONSE} | jq -e '.qps' > /dev/null; then
        echo_success "Metrics endpoint smoke test passed"
    else
        echo_error "Metrics endpoint smoke test failed"
        return 1
    fi
    
    echo_success "All smoke tests passed"
}

# Function to rollback deployment
rollback_deployment() {
    echo_warning "Rolling back deployment..."
    
    # Get previous task definition
    PREVIOUS_TASK_DEF=$(aws ecs describe-services \
        --cluster ${CLUSTER_NAME} \
        --services ${SERVICE_NAME} \
        --region ${AWS_REGION} \
        --query 'services[0].deployments[1].taskDefinition' \
        --output text)
    
    if [ "${PREVIOUS_TASK_DEF}" != "None" ]; then
        # Rollback to previous version
        aws ecs update-service \
            --cluster ${CLUSTER_NAME} \
            --service ${SERVICE_NAME} \
            --task-definition ${PREVIOUS_TASK_DEF} \
            --region ${AWS_REGION} > /dev/null
        
        echo_info "Waiting for rollback to complete..."
        aws ecs wait services-stable \
            --cluster ${CLUSTER_NAME} \
            --services ${SERVICE_NAME} \
            --region ${AWS_REGION}
        
        echo_success "Rollback completed"
    else
        echo_error "No previous deployment found for rollback"
    fi
}

# Main deployment function
main() {
    echo_info "Starting FastAPI deployment..."
    
    check_prerequisites
    
    # Build and push image
    if ! build_and_push_image; then
        echo_error "Image build/push failed"
        exit 1
    fi
    
    # Update task definition
    if ! update_task_definition; then
        echo_error "Task definition update failed"
        exit 1
    fi
    
    # Deploy to ECS
    if ! deploy_to_ecs; then
        echo_error "ECS deployment failed"
        rollback_deployment
        exit 1
    fi
    
    # Run health checks
    if ! run_health_checks; then
        echo_error "Health checks failed"
        rollback_deployment
        exit 1
    fi
    
    # Run smoke tests
    if ! run_smoke_tests; then
        echo_error "Smoke tests failed"
        rollback_deployment
        exit 1
    fi
    
    echo_success "FastAPI deployment completed successfully!"
    
    # Clean up
    rm -f .env.deploy
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "health")
        run_health_checks
        ;;
    "smoke")
        run_smoke_tests
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|health|smoke]"
        exit 1
        ;;
esac
```

### 2. CI/CD Pipeline Configuration

```yaml
# .github/workflows/deploy.yml
name: FastAPI Deployment Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: chatbot-fastapi
  ECS_CLUSTER: chatbot-cluster
  ECS_SERVICE: chatbot-fastapi

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio httpx
    
    - name: Run tests
      run: |
        pytest tests/ -v --tb=short
    
    - name: Run linting
      run: |
        pip install black isort mypy
        black --check .
        isort --check-only .
        mypy .
  
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  build-and-deploy:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build, tag, and push image to Amazon ECR
      id: build-image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
        echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT
    
    - name: Deploy to Amazon ECS
      run: |
        chmod +x deploy/deploy.sh
        deploy/deploy.sh deploy
    
    - name: Run integration tests
      run: |
        python tests/integration_tests.py
    
    - name: Notify deployment success
      if: success()
      run: |
        curl -X POST -H 'Content-type: application/json' \
          --data '{"text":"FastAPI deployment successful! :rocket:"}' \
          ${{ secrets.SLACK_WEBHOOK_URL }}
    
    - name: Notify deployment failure
      if: failure()
      run: |
        curl -X POST -H 'Content-type: application/json' \
          --data '{"text":"FastAPI deployment failed! :warning: Please check the logs."}' \
          ${{ secrets.SLACK_WEBHOOK_URL }}
```

---

## Blue-Green Migration Process

### 1. Traffic Routing Strategy

```python
# deploy/traffic_router.py
import boto3
import time
import json
from typing import Dict, List

class TrafficRouter:
    """Manage traffic routing for blue-green deployment"""
    
    def __init__(self, region: str = "us-east-1"):
        self.elbv2 = boto3.client('elbv2', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # Configuration
        self.listener_arn = "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT:listener/app/chatbot-alb/1234567890123456/1234567890123456"
        self.blue_target_group = "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT:targetgroup/chatbot-flask-tg/1234567890123456"
        self.green_target_group = "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT:targetgroup/chatbot-fastapi-tg/1234567890123456"
    
    def get_current_traffic_weights(self) -> Dict[str, int]:
        """Get current traffic distribution"""
        response = self.elbv2.describe_rules(ListenerArn=self.listener_arn)
        
        for rule in response['Rules']:
            if rule.get('Priority') == '100':  # Our main routing rule
                for action in rule['Actions']:
                    if action['Type'] == 'forward':
                        target_groups = action['ForwardConfig']['TargetGroups']
                        weights = {}
                        for tg in target_groups:
                            if tg['TargetGroupArn'] == self.blue_target_group:
                                weights['blue'] = tg['Weight']
                            elif tg['TargetGroupArn'] == self.green_target_group:
                                weights['green'] = tg['Weight']
                        return weights
        
        return {'blue': 100, 'green': 0}
    
    def update_traffic_weights(self, blue_weight: int, green_weight: int) -> bool:
        """Update traffic distribution weights"""
        try:
            # Validate weights
            if blue_weight + green_weight != 100:
                raise ValueError("Weights must sum to 100")
            
            # Update listener rule
            self.elbv2.modify_rule(
                RuleArn=self._get_main_rule_arn(),
                Actions=[
                    {
                        'Type': 'forward',
                        'ForwardConfig': {
                            'TargetGroups': [
                                {
                                    'TargetGroupArn': self.blue_target_group,
                                    'Weight': blue_weight
                                },
                                {
                                    'TargetGroupArn': self.green_target_group,
                                    'Weight': green_weight
                                }
                            ]
                        }
                    }
                ]
            )
            
            print(f"Traffic updated: Blue {blue_weight}%, Green {green_weight}%")
            return True
            
        except Exception as e:
            print(f"Failed to update traffic weights: {e}")
            return False
    
    def _get_main_rule_arn(self) -> str:
        """Get ARN of main routing rule"""
        response = self.elbv2.describe_rules(ListenerArn=self.listener_arn)
        
        for rule in response['Rules']:
            if rule.get('Priority') == '100':
                return rule['RuleArn']
        
        raise Exception("Main routing rule not found")
    
    def gradual_migration(self, steps: List[int], wait_minutes: int = 10) -> bool:
        """Perform gradual traffic migration"""
        print("Starting gradual traffic migration...")
        
        for step, green_percentage in enumerate(steps):
            blue_percentage = 100 - green_percentage
            
            print(f"\nStep {step + 1}: Routing {green_percentage}% to FastAPI")
            
            # Update traffic weights
            if not self.update_traffic_weights(blue_percentage, green_percentage):
                print("Failed to update traffic weights, aborting migration")
                return False
            
            # Wait for traffic to stabilize
            print(f"Waiting {wait_minutes} minutes for traffic to stabilize...")
            time.sleep(wait_minutes * 60)
            
            # Check health metrics
            if not self._check_health_metrics(green_percentage):
                print("Health metrics check failed, rolling back...")
                self.update_traffic_weights(100, 0)  # Rollback to Flask
                return False
        
        print("\nGradual migration completed successfully!")
        return True
    
    def _check_health_metrics(self, green_percentage: int) -> bool:
        """Check health metrics during migration"""
        # Define thresholds based on traffic percentage
        error_rate_threshold = 5.0  # 5% max error rate
        latency_threshold = 3000    # 3 second max average latency
        
        try:
            # Get error rate metrics
            end_time = time.time()
            start_time = end_time - 300  # Last 5 minutes
            
            error_response = self.cloudwatch.get_metric_statistics(
                Namespace='ChatBot/FastAPI',
                MetricName='ErrorCount',
                Dimensions=[
                    {'Name': 'Endpoint', 'Value': '/api/v1/chat'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum']
            )
            
            success_response = self.cloudwatch.get_metric_statistics(
                Namespace='ChatBot/FastAPI',
                MetricName='SuccessCount',
                Dimensions=[
                    {'Name': 'Endpoint', 'Value': '/api/v1/chat'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum']
            )
            
            # Calculate error rate
            total_errors = sum(point['Sum'] for point in error_response['Datapoints'])
            total_success = sum(point['Sum'] for point in success_response['Datapoints'])
            total_requests = total_errors + total_success
            
            if total_requests > 0:
                error_rate = (total_errors / total_requests) * 100
                print(f"Current error rate: {error_rate:.2f}%")
                
                if error_rate > error_rate_threshold:
                    print(f"Error rate {error_rate:.2f}% exceeds threshold {error_rate_threshold}%")
                    return False
            
            # Get latency metrics
            latency_response = self.cloudwatch.get_metric_statistics(
                Namespace='ChatBot/FastAPI',
                MetricName='RequestLatency',
                Dimensions=[
                    {'Name': 'Endpoint', 'Value': '/api/v1/chat'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            
            if latency_response['Datapoints']:
                avg_latency = latency_response['Datapoints'][-1]['Average']
                print(f"Current average latency: {avg_latency:.0f}ms")
                
                if avg_latency > latency_threshold:
                    print(f"Latency {avg_latency:.0f}ms exceeds threshold {latency_threshold}ms")
                    return False
            
            print("Health metrics check passed")
            return True
            
        except Exception as e:
            print(f"Error checking health metrics: {e}")
            return False
    
    def complete_migration(self) -> bool:
        """Complete migration to FastAPI (100% traffic)"""
        return self.update_traffic_weights(0, 100)
    
    def rollback_migration(self) -> bool:
        """Rollback to Flask (100% traffic)"""
        return self.update_traffic_weights(100, 0)

# Migration execution script
def execute_migration():
    """Execute the complete blue-green migration"""
    router = TrafficRouter()
    
    # Migration steps: 10% -> 50% -> 100%
    migration_steps = [10, 50, 100]
    
    print("Current traffic distribution:")
    current_weights = router.get_current_traffic_weights()
    print(f"Blue (Flask): {current_weights.get('blue', 0)}%")
    print(f"Green (FastAPI): {current_weights.get('green', 0)}%")
    
    # Perform gradual migration
    if router.gradual_migration(migration_steps, wait_minutes=10):
        print("\n🎉 Migration completed successfully!")
        print("FastAPI is now handling 100% of traffic")
        
        # Optional: Clean up Flask resources after successful migration
        print("\nYou can now safely decommission the Flask environment")
    else:
        print("\n❌ Migration failed and was rolled back")
        print("Flask is still handling 100% of traffic")

if __name__ == "__main__":
    execute_migration()
```

### 2. Migration Monitoring Dashboard

```python
# deploy/migration_monitor.py
import boto3
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import json


class MigrationMonitor:
    """Monitor migration metrics in real-time"""
    
    def __init__(self, region: str = "us-east-1"):
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.metrics_data = []
    
    def collect_metrics(self, duration_minutes: int = 60) -> Dict:
        """Collect metrics for specified duration"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        metrics = {
            'timestamps': [],
            'flask_qps': [],
            'fastapi_qps': [],
            'flask_latency': [],
            'fastapi_latency': [],
            'flask_errors': [],
            'fastapi_errors': []
        }
        
        # Collect Flask metrics
        flask_qps = self._get_metric_data(
            'ChatBot/Flask', 'RequestCount', start_time, end_time
        )
        flask_latency = self._get_metric_data(
            'ChatBot/Flask', 'RequestLatency', start_time, end_time, 'Average'
        )
        flask_errors = self._get_metric_data(
            'ChatBot/Flask', 'ErrorCount', start_time, end_time
        )
        
        # Collect FastAPI metrics
        fastapi_qps = self._get_metric_data(
            'ChatBot/FastAPI', 'RequestCount', start_time, end_time
        )
        fastapi_latency = self._get_metric_data(
            'ChatBot/FastAPI', 'RequestLatency', start_time, end_time, 'Average'
        )
        fastapi_errors = self._get_metric_data(
            'ChatBot/FastAPI', 'ErrorCount', start_time, end_time
        )
        
        # Combine data points
        all_timestamps = set()
        for data in [flask_qps, fastapi_qps, flask_latency, fastapi_latency, flask_errors, fastapi_errors]:
            all_timestamps.update(point['Timestamp'] for point in data['Datapoints'])
        
        sorted_timestamps = sorted(all_timestamps)
        
        for ts in sorted_timestamps:
            metrics['timestamps'].append(ts)
            metrics['flask_qps'].append(self._get_value_at_timestamp(flask_qps, ts))
            metrics['fastapi_qps'].append(self._get_value_at_timestamp(fastapi_qps, ts))
            metrics['flask_latency'].append(self._get_value_at_timestamp(flask_latency, ts))
            metrics['fastapi_latency'].append(self._get_value_at_timestamp(fastapi_latency, ts))
            metrics['flask_errors'].append(self._get_value_at_timestamp(flask_errors, ts))
            metrics['fastapi_errors'].append(self._get_value_at_timestamp(fastapi_errors, ts))
        
        return metrics
    
    def _get_metric_data(self, namespace: str, metric_name: str, start_time: datetime, end_time: datetime, statistic: str = 'Sum'):
        """Get metric data from CloudWatch"""
        return self.cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5 minutes
            Statistics=[statistic]
        )
    
    def _get_value_at_timestamp(self, metric_data: Dict, timestamp: datetime) -> float:
        """Get metric value at specific timestamp"""
        for point in metric_data['Datapoints']:
            if point['Timestamp'] == timestamp:
                return point.get('Sum', point.get('Average', 0))
        return 0
    
    def generate_migration_report(self, metrics: Dict) -> str:
        """Generate migration performance report"""
        if not metrics['timestamps']:
            return "No metrics data available"
        
        # Calculate summary statistics
        total_flask_requests = sum(metrics['flask_qps'])
        total_fastapi_requests = sum(metrics['fastapi_qps'])
        total_requests = total_flask_requests + total_fastapi_requests
        
        avg_flask_latency = sum(metrics['flask_latency']) / len([x for x in metrics['flask_latency'] if x > 0]) if any(metrics['flask_latency']) else 0
        avg_fastapi_latency = sum(metrics['fastapi_latency']) / len([x for x in metrics['fastapi_latency'] if x > 0]) if any(metrics['fastapi_latency']) else 0
        
        total_flask_errors = sum(metrics['flask_errors'])
        total_fastapi_errors = sum(metrics['fastapi_errors'])
        
        flask_error_rate = (total_flask_errors / total_flask_requests * 100) if total_flask_requests > 0 else 0
        fastapi_error_rate = (total_fastapi_errors / total_fastapi_requests * 100) if total_fastapi_requests > 0 else 0
        
        # Generate report
        report = f"""
MIGRATION PERFORMANCE REPORT
Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
{'='*60}

TRAFFIC DISTRIBUTION:
  Flask Requests:    {total_flask_requests:,} ({total_flask_requests/total_requests*100:.1f}%)
  FastAPI Requests:  {total_fastapi_requests:,} ({total_fastapi_requests/total_requests*100:.1f}%)
  Total Requests:    {total_requests:,}

PERFORMANCE COMPARISON:
  Average Latency:
    Flask:    {avg_flask_latency:.0f}ms
    FastAPI:  {avg_fastapi_latency:.0f}ms
    Improvement: {((avg_flask_latency - avg_fastapi_latency) / avg_flask_latency * 100) if avg_flask_latency > 0 else 0:.1f}%

  Error Rates:
    Flask:    {flask_error_rate:.2f}%
    FastAPI:  {fastapi_error_rate:.2f}%
    Improvement: {flask_error_rate - fastapi_error_rate:.2f} percentage points

MIGRATION STATUS:
  {'✅ SUCCESSFUL' if fastapi_error_rate < 1.0 and avg_fastapi_latency < avg_flask_latency else '⚠️  NEEDS ATTENTION'}
        """
        
        return report
    
    def create_visualization(self, metrics: Dict, output_file: str = "migration_metrics.png"):
        """Create visualization of migration metrics"""
        if not metrics['timestamps']:
            print("No data to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        timestamps = metrics['timestamps']
        
        # QPS Comparison
        ax1.plot(timestamps, metrics['flask_qps'], label='Flask QPS', color='blue', linewidth=2)
        ax1.plot(timestamps, metrics['fastapi_qps'], label='FastAPI QPS', color='green', linewidth=2)
        ax1.set_title('Queries Per Second Comparison')
        ax1.set_ylabel('Requests/5min')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Latency Comparison
        ax2.plot(timestamps, metrics['flask_latency'], label='Flask Latency', color='blue', linewidth=2)
        ax2.plot(timestamps, metrics['fastapi_latency'], label='FastAPI Latency', color='green', linewidth=2)
        ax2.set_title('Response Latency Comparison')
        ax2.set_ylabel('Latency (ms)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Error Rate Comparison
        total_flask = [f + s for f, s in zip(metrics['flask_qps'], metrics['flask_errors'])]
        total_fastapi = [f + s for f, s in zip(metrics['fastapi_qps'], metrics['fastapi_errors'])]
        
        flask_error_rates = [(e/t*100) if t > 0 else 0 for e, t in zip(metrics['flask_errors'], total_flask)]
        fastapi_error_rates = [(e/t*100) if t > 0 else 0 for e, t in zip(metrics['fastapi_errors'], total_fastapi)]
        
        ax3.plot(timestamps, flask_error_rates, label='Flask Error Rate', color='blue', linewidth=2)
        ax3.plot(timestamps, fastapi_error_rates, label='FastAPI Error Rate', color='green', linewidth=2)
        ax3.set_title('Error Rate Comparison')
        ax3.set_ylabel('Error Rate (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Traffic Distribution
        traffic_distribution = []
        for i in range(len(timestamps)):
            total = metrics['flask_qps'][i] + metrics['fastapi_qps'][i]
            if total > 0:
                fastapi_percent = (metrics['fastapi_qps'][i] / total) * 100
            else:
                fastapi_percent = 0
            traffic_distribution.append(fastapi_percent)
        
        ax4.plot(timestamps, traffic_distribution, label='FastAPI Traffic %', color='purple', linewidth=3)
        ax4.fill_between(timestamps, traffic_distribution, alpha=0.3, color='purple')
        ax4.set_title('Traffic Distribution Over Time')
        ax4.set_ylabel('FastAPI Traffic (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Migration metrics visualization saved to {output_file}")

def monitor_migration():
    """Monitor migration in real-time"""
    monitor = MigrationMonitor()
    
    print("Starting migration monitoring...")
    print("Collecting metrics every 5 minutes...")
    
    while True:
        try:
            # Collect last hour of metrics
            metrics = monitor.collect_metrics(duration_minutes=60)
            
            # Generate report
            report = monitor.generate_migration_report(metrics)
            print("\n" + report)
            
            # Create visualization
            monitor.create_visualization(metrics)
            
            # Wait 5 minutes before next collection
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            time.sleep(60)  # Wait 1 minute before retry

if __name__ == "__main__":
    monitor_migration()
```

---

## Post-Deployment Operations

### 1. Operational Runbook

```markdown
# FastAPI Operations Runbook

## Daily Operations Checklist

### Morning Checks (9 AM)
- [ ] Check CloudWatch dashboard for overnight metrics
- [ ] Verify error rate is < 1%
- [ ] Confirm average latency is < 500ms
- [ ] Check ECS service health (desired vs running tasks)
- [ ] Review CloudWatch logs for any ERROR level messages

### Health Check Commands
```bash
# Check service status
aws ecs describe-services --cluster chatbot-cluster --services chatbot-fastapi

# Check task health
aws ecs list-tasks --cluster chatbot-cluster --service-name chatbot-fastapi

# Get recent logs
aws logs tail /aws/ecs/chatbot-fastapi --since 1h

# Test endpoints
curl -f http://your-alb-dns/health
curl -X POST http://your-alb-dns/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Health check test"}'
```

### Weekly Operations Tasks
- [ ] Review performance trends and capacity planning
- [ ] Check for security updates in base Docker image
- [ ] Validate backup and recovery procedures
- [ ] Review and rotate logs if needed
- [ ] Update CloudWatch dashboard with new metrics if required

## Troubleshooting Guide

### High Latency Issues
1. Check Bedrock service status and regional latency
2. Review ECS task CPU/Memory utilization
3. Check for DynamoDB throttling
4. Verify network connectivity between services
5. Scale ECS service if CPU > 70%

### High Error Rate Issues
1. Check application logs for specific error patterns
2. Verify Bedrock agent ID and permissions
3. Test DynamoDB connectivity and permissions
4. Check API Gateway throttling limits
5. Validate input data format and size

### Service Unavailable
1. Check ECS service status and task count
2. Verify target group health in Load Balancer
3. Check security group rules
4. Validate task definition and image availability
5. Review CloudWatch alarms and notifications

### Scaling Procedures
```bash
# Scale up ECS service
aws ecs update-service \
  --cluster chatbot-cluster \
  --service chatbot-fastapi \
  --desired-count 3

# Scale down ECS service
aws ecs update-service \
  --cluster chatbot-cluster \
  --service chatbot-fastapi \
  --desired-count 1
```

## Emergency Procedures

### Rollback to Flask (Emergency)
```bash
# Quick rollback using traffic router
python deploy/traffic_router.py rollback

# Or direct ALB rule modification
aws elbv2 modify-rule \
  --rule-arn YOUR_RULE_ARN \
  --actions Type=forward,TargetGroupArn=FLASK_TARGET_GROUP_ARN
```

### Complete Service Restart
```bash
# Force new deployment
aws ecs update-service \
  --cluster chatbot-cluster \
  --service chatbot-fastapi \
  --force-new-deployment

# Wait for stabilization
aws ecs wait services-stable \
  --cluster chatbot-cluster \
  --services chatbot-fastapi
```

## Performance Optimization

### Auto Scaling Configuration
```json
{
  "service_name": "chatbot-fastapi",
  "auto_scaling": {
    "min_capacity": 1,
    "max_capacity": 10,
    "target_cpu": 70,
    "target_memory": 80,
    "scale_out_cooldown": 300,
    "scale_in_cooldown": 300
  }
}
```

### Cost Optimization Checklist
- [ ] Right-size ECS task CPU/Memory based on actual usage
- [ ] Implement scheduled scaling for predictable traffic patterns
- [ ] Use Spot instances for non-critical environments
- [ ] Regular review of CloudWatch logs retention settings
- [ ] Monitor Bedrock token usage and optimize prompts
```

### 2. Monitoring and Alerting Setup

```python
# ops/monitoring_setup.py
import boto3
import json
from typing import List, Dict

class ProductionMonitoring:
    """Setup production monitoring and alerting"""
    
    def __init__(self, region: str = "us-east-1"):
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sns = boto3.client('sns', region_name=region)
        self.region = region
    
    def setup_complete_monitoring(self) -> Dict[str, List[str]]:
        """Setup complete monitoring stack"""
        results = {
            "alarms_created": [],
            "dashboards_created": [],
            "sns_topics_created": []
        }
        
        # Create SNS topics for alerts
        topics = self.create_sns_topics()
        results["sns_topics_created"] = topics
        
        # Create CloudWatch alarms
        alarms = self.create_production_alarms(topics["critical"], topics["warning"])
        results["alarms_created"] = alarms
        
        # Create dashboards
        dashboards = self.create_monitoring_dashboards()
        results["dashboards_created"] = dashboards
        
        return results
    
    def create_sns_topics(self) -> Dict[str, str]:
        """Create SNS topics for different alert levels"""
        topics = {}
        
        # Critical alerts topic
        critical_response = self.sns.create_topic(
            Name='chatbot-fastapi-critical-alerts',
            Attributes={
                'DisplayName': 'ChatBot FastAPI Critical Alerts'
            },
            Tags=[
                {'Key': 'Application', 'Value': 'chatbot'},
                {'Key': 'AlertLevel', 'Value': 'critical'}
            ]
        )
        topics["critical"] = critical_response['TopicArn']
        
        # Warning alerts topic
        warning_response = self.sns.create_topic(
            Name='chatbot-fastapi-warning-alerts',
            Attributes={
                'DisplayName': 'ChatBot FastAPI Warning Alerts'
            },
            Tags=[
                {'Key': 'Application', 'Value': 'chatbot'},
                {'Key': 'AlertLevel', 'Value': 'warning'}
            ]
        )
        topics["warning"] = warning_response['TopicArn']
        
        print(f"Created SNS topics:")
        print(f"  Critical: {topics['critical']}")
        print(f"  Warning: {topics['warning']}")
        
        return topics
    
    def create_production_alarms(self, critical_topic: str, warning_topic: str) -> List[str]:
        """Create comprehensive production alarms"""
        alarms = []
        
        # Critical Alarms
        critical_alarms = [
            {
                "name": "FastAPI-CriticalErrorRate",
                "metric": "ErrorCount",
                "threshold": 50,
                "comparison": "GreaterThanThreshold",
                "evaluation_periods": 2,
                "period": 300,
                "description": "Critical: High error count detected"
            },
            {
                "name": "FastAPI-ServiceDown",
                "metric": "RequestCount", 
                "threshold": 1,
                "comparison": "LessThanThreshold",
                "evaluation_periods": 3,
                "period": 300,
                "description": "Critical: Service appears to be down"
            },
            {
                "name": "FastAPI-ExtremeLatency",
                "metric": "RequestLatency",
                "threshold": 10000,  # 10 seconds
                "comparison": "GreaterThanThreshold",
                "evaluation_periods": 2,
                "period": 300,
                "statistic": "Average",
                "description": "Critical: Extreme response latency"
            }
        ]
        
        # Warning Alarms
        warning_alarms = [
            {
                "name": "FastAPI-HighErrorRate",
                "metric": "ErrorCount",
                "threshold": 10,
                "comparison": "GreaterThanThreshold",
                "evaluation_periods": 3,
                "period": 300,
                "description": "Warning: Elevated error rate"
            },
            {
                "name": "FastAPI-HighLatency",
                "metric": "RequestLatency",
                "threshold": 2000,  # 2 seconds
                "comparison": "GreaterThanThreshold",
                "evaluation_periods": 3,
                "period": 300,
                "statistic": "Average",
                "description": "Warning: High response latency"
            },
            {
                "name": "FastAPI-LowThroughput",
                "metric": "RequestCount",
                "threshold": 50,
                "comparison": "LessThanThreshold",
                "evaluation_periods": 2,
                "period": 900,  # 15 minutes
                "description": "Warning: Low request throughput"
            }
        ]
        
        # Create critical alarms
        for alarm_config in critical_alarms:
            alarm_name = self._create_alarm(alarm_config, critical_topic)
            alarms.append(alarm_name)
        
        # Create warning alarms
        for alarm_config in warning_alarms:
            alarm_name = self._create_alarm(alarm_config, warning_topic)
            alarms.append(alarm_name)
        
        # ECS specific alarms
        ecs_alarms = self._create_ecs_alarms(critical_topic, warning_topic)
        alarms.extend(ecs_alarms)
        
        return alarms
    
    def _create_alarm(self, config: Dict, topic_arn: str) -> str:
        """Create individual CloudWatch alarm"""
        self.cloudwatch.put_metric_alarm(
            AlarmName=config["name"],
            ComparisonOperator=config["comparison"],
            EvaluationPeriods=config["evaluation_periods"],
            MetricName=config["metric"],
            Namespace='ChatBot/FastAPI',
            Period=config["period"],
            Statistic=config.get("statistic", "Sum"),
            Threshold=config["threshold"],
            ActionsEnabled=True,
            AlarmActions=[topic_arn],
            AlarmDescription=config["description"],
            Dimensions=[
                {
                    'Name': 'Endpoint',
                    'Value': '/api/v1/chat'
                }
            ] if config["metric"] in ["RequestCount", "ErrorCount", "RequestLatency"] else [],
            Unit='Count' if config["metric"] in ["RequestCount", "ErrorCount"] else 'Milliseconds'
        )
        
        print(f"Created alarm: {config['name']}")
        return config["name"]
    
    def _create_ecs_alarms(self, critical_topic: str, warning_topic: str) -> List[str]:
        """Create ECS-specific alarms"""
        alarms = []
        
        # High CPU utilization
        self.cloudwatch.put_metric_alarm(
            AlarmName="FastAPI-ECS-HighCPU",
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=3,
            MetricName='CPUUtilization',
            Namespace='AWS/ECS',
            Period=300,
            Statistic='Average',
            Threshold=80.0,
            ActionsEnabled=True,
            AlarmActions=[warning_topic],
            AlarmDescription='Warning: High CPU utilization',
            Dimensions=[
                {'Name': 'ServiceName', 'Value': 'chatbot-fastapi'},
                {'Name': 'ClusterName', 'Value': 'chatbot-cluster'}
            ]
        )
        alarms.append("FastAPI-ECS-HighCPU")
        
        # High Memory utilization
        self.cloudwatch.put_metric_alarm(
            AlarmName="FastAPI-ECS-HighMemory",
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=3,
            MetricName='MemoryUtilization',
            Namespace='AWS/ECS',
            Period=300,
            Statistic='Average',
            Threshold=85.0,
            ActionsEnabled=True,
            AlarmActions=[critical_topic],
            AlarmDescription='Critical: High memory utilization',
            Dimensions=[
                {'Name': 'ServiceName', 'Value': 'chatbot-fastapi'},
                {'Name': 'ClusterName', 'Value': 'chatbot-cluster'}
            ]
        )
        alarms.append("FastAPI-ECS-HighMemory")
        
        print(f"Created ECS alarms: {alarms}")
        return alarms
    
    def create_monitoring_dashboards(self) -> List[str]:
        """Create monitoring dashboards"""
        dashboards = []
        
        # Main operational dashboard
        main_dashboard = self._create_operational_dashboard()
        dashboards.append(main_dashboard)
        
        # Performance comparison dashboard
        comparison_dashboard = self._create_comparison_dashboard()
        dashboards.append(comparison_dashboard)
        
        return dashboards
    
    def _create_operational_dashboard(self) -> str:
        """Create main operational dashboard"""
        dashboard_name = "ChatBot-FastAPI-Operations"
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["ChatBot/FastAPI", "RequestCount", "Endpoint", "/api/v1/chat"],
                            [".", "ErrorCount", ".", "."]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Request Volume and Errors",
                        "period": 300,
                        "stat": "Sum"
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["ChatBot/FastAPI", "RequestLatency", "Endpoint", "/api/v1/chat", {"stat": "Average"}],
                            ["...", {"stat": "p90"}],
                            ["...", {"stat": "p99"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Response Latency (Avg, P90, P99)",
                        "period": 300
                    }
                },
                {
                    "type": "metric",
                    "x": 0, "y": 6, "width": 24, "height": 6,
                    "properties": {
                        "metrics": [
                            ["AWS/ECS", "CPUUtilization", "ServiceName", "chatbot-fastapi", "ClusterName", "chatbot-cluster"],
                            [".", "MemoryUtilization", ".", ".", ".", "."]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "ECS Resource Utilization",
                        "period": 300,
                        "stat": "Average"
                    }
                }
            ]
        }
        
        self.cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body)
        )
        
        print(f"Created dashboard: {dashboard_name}")
        return dashboard_name
    
    def _create_comparison_dashboard(self) -> str:
        """Create Flask vs FastAPI comparison dashboard"""
        dashboard_name = "ChatBot-Migration-Comparison"
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["ChatBot/Flask", "RequestCount", {"label": "Flask QPS"}],
                            ["ChatBot/FastAPI", "RequestCount", {"label": "FastAPI QPS"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "QPS Comparison: Flask vs FastAPI",
                        "period": 300,
                        "stat": "Sum"
                    }
                },
                {
                    "type": "metric",
                    "x": 12, "y": 0, "width": 12, "height": 6,
                    "properties": {
                        "metrics": [
                            ["ChatBot/Flask", "RequestLatency", {"label": "Flask Latency"}],
                            ["ChatBot/FastAPI", "RequestLatency", {"label": "FastAPI Latency"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Latency Comparison: Flask vs FastAPI",
                        "period": 300,
                        "stat": "Average"
                    }
                }
            ]
        }
        
        self.cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body)
        )
        
        print(f"Created dashboard: {dashboard_name}")
        return dashboard_name

def setup_production_monitoring():
    """Setup complete production monitoring"""
    monitor = ProductionMonitoring()
    
    print("Setting up production monitoring for FastAPI...")
    results = monitor.setup_complete_monitoring()
    
    print("\nMonitoring setup completed!")
    print(f"Alarms created: {len(results['alarms_created'])}")
    print(f"Dashboards created: {len(results['dashboards_created'])}")
    print(f"SNS topics created: {len(results['sns_topics_created'])}")
    
    print("\nNext steps:")
    print("1. Subscribe to SNS topics for email/Slack notifications")
    print("2. Configure dashboard access permissions")
    print("3. Test alarm notifications")
    print("4. Schedule regular monitoring reviews")

if __name__ == "__main__":
    setup_production_monitoring()
```

---

## Final Migration Checklist

### Pre-Migration Validation
- [ ] FastAPI application deployed and tested in staging
- [ ] Load testing completed with satisfactory results
- [ ] All AWS services (Bedrock, DynamoDB) accessible from FastAPI
- [ ] Monitoring and alerting configured
- [ ] Blue-green deployment infrastructure ready
- [ ] Rollback procedures tested and documented

### Migration Execution
- [ ] Announce maintenance window to stakeholders
- [ ] Deploy FastAPI to green environment
- [ ] Run health checks and smoke tests
- [ ] Begin gradual traffic migration (10% → 50% → 100%)
- [ ] Monitor metrics at each migration step
- [ ] Validate performance improvements
- [ ] Complete traffic migration to FastAPI

### Post-Migration Tasks
- [ ] Verify all functionality working correctly
- [ ] Confirm performance improvements achieved
- [ ] Update documentation and runbooks
- [ ] Train operations team on new monitoring
- [ ] Schedule Flask environment decommissioning
- [ ] Conduct migration retrospective
- [ ] Document lessons learned

### Success Criteria Validation
- [ ] **Performance**: 3x improvement in concurrent users (700 → 2,100+)
- [ ] **Latency**: P90 latency < 500ms (down from 800ms)
- [ ] **Error Rate**: < 1% (down from 2.1%)
- [ ] **Cost**: 50%+ reduction in infrastructure costs
- [ ] **Reliability**: Zero unplanned downtime during migration

---

*Next: Chapter 7 - Project Timeline & Execution*# Flask to FastAPI Migration Guide


