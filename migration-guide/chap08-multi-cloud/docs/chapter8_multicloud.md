# Chapter 8: Multi-Cloud Deployment

## Overview

This chapter explores deploying FastAPI applications across multiple cloud platforms, providing comprehensive guidance on platform-specific optimizations, cost considerations, and migration strategies. We'll implement our MedAssistant application on AWS, Azure, and Google Cloud Platform (GCP), documenting real-world experiences, performance comparisons, and lessons learned.

**Key Topics Covered:**
- Platform-specific deployment strategies
- Cost-performance analysis across clouds
- Migration patterns between platforms
- Disaster recovery and high availability
- Infrastructure as Code (IaC) approaches

---

## AWS Deployment Strategy

### Amazon Web Services Architecture

**Core Services Stack:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │    Database     │    │   AI Services   │
│                 │    │                 │    │                 │
│ ECS Fargate     │◄──►│ RDS PostgreSQL  │    │ Amazon Bedrock  │
│ Application     │    │ DynamoDB        │    │ Claude/Titan    │
│ Load Balancer   │    │ ElastiCache     │    │ Knowledge Bases │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Networking     │    │   Monitoring    │    │    Storage      │
│                 │    │                 │    │                 │
│ VPC             │    │ CloudWatch      │    │ S3 Buckets      │
│ Route 53        │    │ X-Ray Tracing   │    │ EFS Volumes     │
│ CloudFront CDN  │    │ AWS Config      │    │ Parameter Store │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### ECS Fargate Deployment

**Task Definition Configuration:**
```json
{
  "family": "medassistant-fastapi",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/medassistant-task-role",
  "containerDefinitions": [
    {
      "name": "medassistant-api",
      "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/medassistant:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        },
        {
          "name": "BEDROCK_MODEL_ID",
          "value": "anthropic.claude-3-sonnet-20240229-v1:0"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:ssm:us-east-1:ACCOUNT:parameter/medassistant/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/medassistant",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### Infrastructure as Code (Terraform)

**AWS Main Configuration:**
```hcl
# aws/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "medassistant-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = true
  
  tags = {
    Terraform = "true"
    Environment = var.environment
    Project = "medassistant"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "medassistant-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Environment = var.environment
    Project = "medassistant"
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "medassistant-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
  
  enable_deletion_protection = false
  
  tags = {
    Environment = var.environment
    Project = "medassistant"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier     = "medassistant-db"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_encrypted     = true
  
  db_name  = "medassistant"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  
  tags = {
    Environment = var.environment
    Project = "medassistant"
  }
}

# DynamoDB for Sessions
resource "aws_dynamodb_table" "sessions" {
  name           = "medassistant-sessions"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "session_id"
  
  attribute {
    name = "session_id"
    type = "S"
  }
  
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
  
  tags = {
    Environment = var.environment
    Project = "medassistant"
  }
}
```

### AWS-Specific Optimizations

**Performance Optimizations:**
- **ECS Service Auto Scaling:** Scale based on CPU/memory utilization
- **ALB Target Group Health Checks:** Custom health endpoints
- **CloudFront CDN:** Cache static assets and API responses where appropriate
- **ElastiCache:** Redis for session caching and frequently accessed data

**Cost Optimizations:**
- **Spot Instances:** Use for non-critical workloads
- **Reserved Instances:** Commit to steady-state capacity
- **S3 Intelligent Tiering:** Automatic cost optimization for document storage
- **Lambda for Background Tasks:** Pay-per-execution for async operations

### Deployment Pipeline

**AWS CodePipeline Configuration:**
```yaml
# aws/codepipeline.yml
version: 0.2
phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
  build:
    commands:
      - echo Build started on `date`
      - echo Building the Docker image...
      - docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .
      - docker tag $IMAGE_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG
  post_build:
    commands:
      - echo Build completed on `date`
      - echo Pushing the Docker image...
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG
      - echo Writing image definitions file...
      - printf '[{"name":"medassistant-api","imageUri":"%s"}]' $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/$IMAGE_REPO_NAME:$IMAGE_TAG > imagedefinitions.json
artifacts:
  files: imagedefinitions.json
```

**Monitoring and Logging:**
```python
# aws/monitoring.py
import boto3
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class AWSMonitoring:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.xray = boto3.client('xray')
    
    async def put_custom_metric(
        self, 
        metric_name: str, 
        value: float, 
        unit: str = 'Count',
        dimensions: Dict[str, str] = None
    ):
        """Put custom metric to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='MedAssistant/FastAPI',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Dimensions': [
                            {'Name': k, 'Value': v} 
                            for k, v in (dimensions or {}).items()
                        ]
                    }
                ]
            )
        except Exception as e:
            logger.error("Failed to put CloudWatch metric", error=str(e))
    
    async def create_alarm(
        self,
        alarm_name: str,
        metric_name: str,
        threshold: float,
        comparison_operator: str = 'GreaterThanThreshold'
    ):
        """Create CloudWatch alarm"""
        self.cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator=comparison_operator,
            EvaluationPeriods=2,
            MetricName=metric_name,
            Namespace='MedAssistant/FastAPI',
            Period=300,
            Statistic='Average',
            Threshold=threshold,
            ActionsEnabled=True,
            AlarmActions=[
                'arn:aws:sns:us-east-1:ACCOUNT:medassistant-alerts'
            ],
            AlarmDescription=f'Alarm for {metric_name}',
            Unit='Count'
        )
```

---

## Azure Deployment Strategy

### Microsoft Azure Architecture

**Core Services Stack:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │    Database     │    │   AI Services   │
│                 │    │                 │    │                 │
│ Container Apps  │◄──►│ PostgreSQL      │    │ Azure OpenAI    │
│ App Gateway     │    │ Cosmos DB       │    │ GPT-4, GPT-3.5  │
│ Load Balancer   │    │ Redis Cache     │    │ Cognitive Search│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Networking     │    │   Monitoring    │    │    Storage      │
│                 │    │                 │    │                 │
│ Virtual Network │    │ App Insights    │    │ Blob Storage    │
│ Traffic Manager │    │ Log Analytics   │    │ Key Vault       │
│ Front Door      │    │ Azure Monitor   │    │ File Shares     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Container Apps Deployment

**Container Apps Configuration:**
```yaml
# azure/container-app.yml
apiVersion: 2022-10-01
kind: ContainerApp
metadata:
  name: medassistant-app
  location: East US
properties:
  resourceGroup: medassistant-rg
  environmentId: /subscriptions/SUBSCRIPTION_ID/resourceGroups/medassistant-rg/providers/Microsoft.App/managedEnvironments/medassistant-env
  configuration:
    secrets:
      - name: database-connection-string
        value: "[DATABASE_CONNECTION_STRING]"
      - name: openai-api-key
        value: "[OPENAI_API_KEY]"
    ingress:
      external: true
      targetPort: 8000
      traffic:
        - weight: 100
          latestRevision: true
    dapr:
      enabled: true
      appId: medassistant-api
      appProtocol: http
      appPort: 8000
  template:
    containers:
      - image: medassistant.azurecr.io/medassistant:latest
        name: medassistant-api
        env:
          - name: AZURE_OPENAI_ENDPOINT
            value: "https://medassistant-openai.openai.azure.com/"
          - name: AZURE_OPENAI_API_VERSION
            value: "2024-02-15-preview"
          - name: DATABASE_URL
            secretRef: database-connection-string
        resources:
          cpu: 1.0
          memory: 2Gi
        probes:
          - type: Liveness
            httpGet:
              path: "/health"
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          - type: Readiness
            httpGet:
              path: "/health"
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
    scale:
      minReplicas: 1
      maxReplicas: 10
      rules:
        - name: http-rule
          http:
            metadata:
              concurrentRequests: "30"
```

### Infrastructure as Code (Bicep)

**Azure Resource Deployment:**
```bicep
// azure/main.bicep
param location string = resourceGroup().location
param environment string = 'dev'
param appName string = 'medassistant'

// Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2022-10-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// Log Analytics Workspace
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Application Insights
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${appName}-insights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
}

// PostgreSQL Flexible Server
resource postgresqlServer 'Microsoft.DBforPostgreSQL/flexibleServers@2022-12-01' = {
  name: '${appName}-postgres'
  location: location
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    version: '15'
    administratorLogin: 'medassistant_admin'
    administratorLoginPassword: 'PLACEHOLDER_PASSWORD'
    storage: {
      storageSizeGB: 32
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
}

// Cosmos DB for Sessions
resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: '${appName}-cosmos'
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    databaseAccountOfferType: 'Standard'
    capabilities: [
      {
        name: 'EnableServerless'
      }
    ]
  }
}

// Azure OpenAI Service
resource openAIService 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${appName}-openai'
  location: 'East US'  // OpenAI service availability
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: '${appName}-openai'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// Key Vault for Secrets
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: '${appName}-kv'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    enableRbacAuthorization: true
  }
}

// Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: '${appName}acr'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}
```

### Azure-Specific Optimizations

**Performance Optimizations:**
- **KEDA Scaling:** Kubernetes Event-Driven Autoscaling for Container Apps
- **Azure Front Door:** Global load balancing and CDN
- **Azure Cache for Redis:** Session and data caching
- **Application Insights:** Deep performance monitoring

**Cost Optimizations:**
- **Container Apps Consumption Plan:** Pay for actual usage
- **Cosmos DB Serverless:** Pay per request unit
- **Azure Reserved Instances:** Long-term commitments for predictable workloads
- **Azure Hybrid Benefit:** Use existing licenses

---

## Google Cloud Platform (GCP) Deployment

### Google Cloud Architecture

**Core Services Stack:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │    Database     │    │   AI Services   │
│                 │    │                 │    │                 │
│ Cloud Run       │◄──►│ Cloud SQL       │    │ Vertex AI       │
│ Load Balancer   │    │ Firestore       │    │ PaLM, Gemini    │
│ Cloud Armor     │    │ Memorystore     │    │ Document AI     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Networking     │    │   Monitoring    │    │    Storage      │
│                 │    │                 │    │                 │
│ VPC Network     │    │ Cloud Monitoring│    │ Cloud Storage   │
│ Cloud DNS       │    │ Cloud Logging   │    │ Secret Manager  │
│ Cloud CDN       │    │ Error Reporting │    │ Persistent Disk │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Cloud Run Deployment

**Cloud Run Service Configuration:**
```yaml
# gcp/service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: medassistant-service
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      serviceAccountName: medassistant-sa@PROJECT_ID.iam.gserviceaccount.com
      containers:
      - image: gcr.io/PROJECT_ID/medassistant:latest
        ports:
        - name: http1
          containerPort: 8000
        env:
        - name: GCP_PROJECT_ID
          value: "PROJECT_ID"
        - name: VERTEX_AI_LOCATION
          value: "us-central1"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-url
              key: latest
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Infrastructure as Code (Terraform for GCP)

**GCP Resource Configuration:**
```hcl
# gcp/main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "monitoring.googleapis.com"
  ])
  
  service = each.value
  disable_on_destroy = false
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "main" {
  name             = "medassistant-db"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier = "db-f1-micro"
    
    backup_configuration {
      enabled = true
      start_time = "03:00"
    }
    
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"  # Restrict in production
      }
    }
  }
  
  deletion_protection = false
}

resource "google_sql_database" "database" {
  name     = "medassistant"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "user" {
  name     = var.db_user
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

# Firestore for Sessions
resource "google_firestore_database" "database" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
}

# Service Account for Cloud Run
resource "google_service_account" "medassistant" {
  account_id   = "medassistant-sa"
  display_name = "MedAssistant Service Account"
}

# IAM Roles
resource "google_project_iam_member" "medassistant_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
    "roles/cloudsql.client",
    "roles/datastore.user"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.medassistant.email}"
}

# Cloud Run Service
resource "google_cloud_run_service" "medassistant" {
  name     = "medassistant-service"
  location = var.region
  
  template {
    spec {
      service_account_name = google_service_account.medassistant.email
      containers {
        image = "gcr.io/${var.project_id}/medassistant:latest"
        
        ports {
          container_port = 8000
        }
        
        env {
          name  = "GCP_PROJECT_ID"
          value = var.project_id
        }
        
        env {
          name = "DATABASE_URL"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.database_url.secret_id
              key  = "latest"
            }
          }
        }
        
        resources {
          limits = {
            cpu    = "2"
            memory = "2Gi"
          }
        }
      }
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "100"
        "autoscaling.knative.dev/minScale" = "1"
        "run.googleapis.com/cpu-throttling" = "false"
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
  
  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access
resource "google_cloud_run_service_iam_member" "allUsers" {
  service  = google_cloud_run_service.medassistant.name
  location = google_cloud_run_service.medassistant.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Secret Manager
resource "google_secret_manager_secret" "database_url" {
  secret_id = "database-url"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret_version" "database_url" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = "postgresql://${var.db_user}:${var.db_password}@${google_sql_database_instance.main.connection_name}/${google_sql_database.database.name}"
}
```

### GCP-Specific Optimizations

**Performance Optimizations:**
- **Cloud Run Concurrency:** Optimize container concurrency settings
- **Cloud CDN:** Cache responses globally
- **Memorystore:** Redis for high-performance caching
- **VPC Connector:** Direct VPC access for databases

**Cost Optimizations:**
- **Cloud Run Pay-per-use:** No idle costs
- **Committed Use Discounts:** For predictable workloads
- **Preemptible Instances:** For batch processing
- **Cloud Storage Lifecycle:** Automatic data archiving

---

## Cost-Performance Comparison

### Monthly Cost Analysis

**Baseline Configuration:**
- **Application:** 2 vCPU, 4GB RAM, 10 requests/second average
- **Database:** Small instance with 100GB storage
- **AI Services:** 1M tokens per month
- **Storage:** 500GB documents and logs

| Service Category | AWS (Monthly) | Azure (Monthly) | GCP (Monthly) |
|------------------|---------------|-----------------|---------------|
| **Compute** | $73 (ECS Fargate) | $65 (Container Apps) | $45 (Cloud Run) |
| **Database** | $25 (RDS t3.micro) | $30 (PostgreSQL Flexible) | $20 (Cloud SQL f1-micro) |
| **AI Services** | $35 (Bedrock Claude) | $40 (OpenAI GPT-4) | $30 (Vertex AI PaLM) |
| **Storage** | $15 (S3 + EFS) | $18 (Blob + Files) | $12 (Cloud Storage) |
| **Networking** | $10 (ALB + data transfer) | $12 (App Gateway) | $8 (Load Balancer) |
| **Monitoring** | $8 (CloudWatch) | $10 (App Insights) | $5 (Cloud Monitoring) |
| **Other Services** | $12 (DynamoDB, etc.) | $15 (Cosmos DB, etc.) | $10 (Firestore, etc.) |
| **TOTAL** | **$178/month** | **$190/month** | **$130/month** |

*Note: Prices are estimates and vary by region, usage patterns, and commitment levels.*

### Performance Benchmarks

**Load Testing Results (1000 concurrent users):**

| Metric | AWS ECS | Azure Container Apps | GCP Cloud Run |
|--------|---------|---------------------|---------------|
| **Avg Response Time** | [PLACEHOLDER: 245ms] | [PLACEHOLDER: 267ms] | [PLACEHOLDER: 198ms] |
| **95th Percentile** | [PLACEHOLDER: 890ms] | [PLACEHOLDER: 945ms] | [PLACEHOLDER: 712ms] |
| **Throughput (RPS)** | [PLACEHOLDER: 2,847] | [PLACEHOLDER: 2,634] | [PLACEHOLDER: 3,156] |
| **Cold Start Time** | [PLACEHOLDER: 2.1s] | [PLACEHOLDER: 3.4s] | [PLACEHOLDER: 1.8s] |
| **Memory Usage** | [PLACEHOLDER: 1.2GB] | [PLACEHOLDER: 1.4GB] | [PLACEHOLDER: 1.1GB] |
| **Error Rate** | [PLACEHOLDER: 0.02%] | [PLACEHOLDER: 0.03%] | [PLACEHOLDER: 0.01%] |

*Benchmarks will be updated with actual test results during implementation.*

---

## Migration Strategies Between Clouds

### Application Portability Design

**Multi-Cloud Abstract Layer:**
```python
# common/cloud_abstractions.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()

class CloudStorageInterface(ABC):
    """Abstract interface for cloud storage operations"""
    
    @abstractmethod
    async def upload_file(self, file_path: str, bucket: str, key: str) -> bool:
        pass
    
    @abstractmethod
    async def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        pass
    
    @abstractmethod
    async def delete_file(self, bucket: str, key: str) -> bool:
        pass

class CloudDatabaseInterface(ABC):
    """Abstract interface for cloud database operations"""
    
    @abstractmethod
    async def execute_query(self, query: str, params: Dict[str, Any]) -> Any:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass

class CloudSecretsInterface(ABC):
    """Abstract interface for secrets management"""
    
    @abstractmethod
    async def get_secret(self, secret_name: str) -> Optional[str]:
        pass
    
    @abstractmethod
    async def set_secret(self, secret_name: str, value: str) -> bool:
        pass

# AWS Implementation
class AWSStorageService(CloudStorageInterface):
    def __init__(self):
        import boto3
        self.s3_client = boto3.client('s3')
    
    async def upload_file(self, file_path: str, bucket: str, key: str) -> bool:
        try:
            self.s3_client.upload_file(file_path, bucket, key)
            logger.info("File uploaded to S3", bucket=bucket, key=key)
            return True
        except Exception as e:
            logger.error("S3 upload failed", error=str(e))
            return False

class AWSSecretsService(CloudSecretsInterface):
    def __init__(self):
        import boto3
        self.secrets_client = boto3.client('secretsmanager')
    
    async def get_secret(self, secret_name: str) -> Optional[str]:
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        except Exception as e:
            logger.error("Failed to get secret from AWS", secret=secret_name, error=str(e))
            return None

# Azure Implementation
class AzureStorageService(CloudStorageInterface):
    def __init__(self):
        from azure.storage.blob.aio import BlobServiceClient
        self.blob_client = BlobServiceClient(account_url="https://ACCOUNT.blob.core.windows.net")
    
    async def upload_file(self, file_path: str, container: str, blob_name: str) -> bool:
        try:
            async with self.blob_client.get_blob_client(
                container=container, blob=blob_name
            ) as blob_client:
                with open(file_path, 'rb') as data:
                    await blob_client.upload_blob(data, overwrite=True)
            logger.info("File uploaded to Azure Blob", container=container, blob=blob_name)
            return True
        except Exception as e:
            logger.error("Azure Blob upload failed", error=str(e))
            return False

class AzureSecretsService(CloudSecretsInterface):
    def __init__(self):
        from azure.keyvault.secrets.aio import SecretClient
        from azure.identity.aio import DefaultAzureCredential
        
        credential = DefaultAzureCredential()
        self.secret_client = SecretClient(
            vault_url="https://VAULT_NAME.vault.azure.net/", 
            credential=credential
        )
    
    async def get_secret(self, secret_name: str) -> Optional[str]:
        try:
            secret = await self.secret_client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error("Failed to get secret from Azure", secret=secret_name, error=str(e))
            return None

# GCP Implementation
class GCPStorageService(CloudStorageInterface):
    def __init__(self):
        from google.cloud import storage
        self.client = storage.Client()
    
    async def upload_file(self, file_path: str, bucket_name: str, blob_name: str) -> bool:
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(file_path)
            logger.info("File uploaded to GCS", bucket=bucket_name, blob=blob_name)
            return True
        except Exception as e:
            logger.error("GCS upload failed", error=str(e))
            return False

class GCPSecretsService(CloudSecretsInterface):
    def __init__(self):
        from google.cloud import secretmanager
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = "PROJECT_ID"
    
    async def get_secret(self, secret_name: str) -> Optional[str]:
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error("Failed to get secret from GCP", secret=secret_name, error=str(e))
            return None

# Cloud Factory
class CloudServiceFactory:
    """Factory to create cloud-specific service implementations"""
    
    @staticmethod
    def create_storage_service(cloud_provider: str) -> CloudStorageInterface:
        if cloud_provider.lower() == 'aws':
            return AWSStorageService()
        elif cloud_provider.lower() == 'azure':
            return AzureStorageService()
        elif cloud_provider.lower() == 'gcp':
            return GCPStorageService()
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")
    
    @staticmethod
    def create_secrets_service(cloud_provider: str) -> CloudSecretsInterface:
        if cloud_provider.lower() == 'aws':
            return AWSSecretsService()
        elif cloud_provider.lower() == 'azure':
            return AzureSecretsService()
        elif cloud_provider.lower() == 'gcp':
            return GCPSecretsService()
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider}")
```

### Configuration-Driven Deployment

**Environment-Specific Configuration:**
```python
# config/cloud_settings.py
from pydantic import BaseModel
from typing import Dict, Any, Optional
from enum import Enum

class CloudProvider(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"

class CloudConfig(BaseModel):
    provider: CloudProvider
    region: str
    project_id: Optional[str] = None  # GCP
    resource_group: Optional[str] = None  # Azure
    
    # Database settings
    database_type: str  # postgresql, mysql, etc.
    database_host: str
    database_port: int = 5432
    database_name: str
    
    # AI Service settings
    ai_service_endpoint: str
    ai_model_name: str
    ai_api_version: Optional[str] = None
    
    # Storage settings
    storage_bucket: str
    storage_region: Optional[str] = None
    
    # Monitoring settings
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Environment-specific configurations
CLOUD_CONFIGS = {
    "aws": CloudConfig(
        provider=CloudProvider.AWS,
        region="us-east-1",
        database_type="postgresql",
        database_host="medassistant-db.cluster-xyz.us-east-1.rds.amazonaws.com",
        database_name="medassistant",
        ai_service_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
        ai_model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        storage_bucket="medassistant-documents",
    ),
    
    "azure": CloudConfig(
        provider=CloudProvider.AZURE,
        region="eastus",
        resource_group="medassistant-rg",
        database_type="postgresql",
        database_host="medassistant-postgres.postgres.database.azure.com",
        database_name="medassistant",
        ai_service_endpoint="https://medassistant-openai.openai.azure.com",
        ai_model_name="gpt-4",
        ai_api_version="2024-02-15-preview",
        storage_bucket="medassistant-storage",
    ),
    
    "gcp": CloudConfig(
        provider=CloudProvider.GCP,
        region="us-central1",
        project_id="medassistant-project",
        database_type="postgresql",
        database_host="127.0.0.1:5432",  # Via Cloud SQL Proxy
        database_name="medassistant",
        ai_service_endpoint="https://us-central1-aiplatform.googleapis.com",
        ai_model_name="text-bison@001",
        storage_bucket="medassistant-documents-gcp",
    )
}

def get_cloud_config(provider: str) -> CloudConfig:
    """Get cloud configuration for specified provider"""
    if provider not in CLOUD_CONFIGS:
        raise ValueError(f"Unsupported cloud provider: {provider}")
    return CLOUD_CONFIGS[provider]
```

### Migration Automation Scripts

**Cloud Migration Utility:**
```python
# scripts/cloud_migrator.py
import asyncio
import argparse
from typing import Dict, Any
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()

@dataclass
class MigrationPlan:
    source_cloud: str
    target_cloud: str
    components: list
    estimated_downtime: str
    rollback_plan: str

class CloudMigrator:
    """Utility for migrating between cloud providers"""
    
    def __init__(self, source_cloud: str, target_cloud: str):
        self.source_cloud = source_cloud
        self.target_cloud = target_cloud
        self.migration_steps = []
    
    async def create_migration_plan(self) -> MigrationPlan:
        """Create detailed migration plan"""
        logger.info("Creating migration plan", 
                   source=self.source_cloud, 
                   target=self.target_cloud)
        
        # Analyze current deployment
        current_resources = await self._analyze_current_resources()
        
        # Map to target cloud resources
        target_mapping = await self._map_target_resources(current_resources)
        
        # Estimate migration complexity
        complexity = self._estimate_complexity(current_resources, target_mapping)
        
        return MigrationPlan(
            source_cloud=self.source_cloud,
            target_cloud=self.target_cloud,
            components=target_mapping,
            estimated_downtime=complexity['downtime'],
            rollback_plan=complexity['rollback']
        )
    
    async def _analyze_current_resources(self) -> Dict[str, Any]:
        """Analyze current cloud resources"""
        resources = {
            'compute': [],
            'database': [],
            'storage': [],
            'networking': [],
            'monitoring': []
        }
        
        if self.source_cloud == 'aws':
            # Use boto3 to analyze AWS resources
            resources = await self._analyze_aws_resources()
        elif self.source_cloud == 'azure':
            # Use Azure SDK to analyze resources
            resources = await self._analyze_azure_resources()
        elif self.source_cloud == 'gcp':
            # Use GCP client libraries
            resources = await self._analyze_gcp_resources()
        
        return resources
    
    async def _analyze_aws_resources(self) -> Dict[str, Any]:
        """Analyze AWS resources"""
        # Placeholder - implement actual AWS resource analysis
        return {
            'compute': [{'type': 'ECS', 'name': 'medassistant-service'}],
            'database': [{'type': 'RDS', 'name': 'medassistant-db'}],
            'storage': [{'type': 'S3', 'name': 'medassistant-documents'}],
        }
    
    async def _map_target_resources(self, current_resources: Dict[str, Any]) -> list:
        """Map current resources to target cloud equivalents"""
        mapping = []
        
        for category, resources in current_resources.items():
            for resource in resources:
                target_resource = self._get_target_equivalent(
                    category, resource, self.target_cloud
                )
                mapping.append({
                    'source': resource,
                    'target': target_resource,
                    'migration_complexity': self._assess_migration_complexity(resource, target_resource)
                })
        
        return mapping
    
    def _get_target_equivalent(self, category: str, source_resource: Dict, target_cloud: str) -> Dict:
        """Get equivalent resource in target cloud"""
        # Resource mapping logic
        equivalents = {
            'aws_to_azure': {
                'ECS': 'Container Apps',
                'RDS': 'PostgreSQL Flexible Server',
                'S3': 'Blob Storage',
                'DynamoDB': 'Cosmos DB'
            },
            'aws_to_gcp': {
                'ECS': 'Cloud Run',
                'RDS': 'Cloud SQL',
                'S3': 'Cloud Storage',
                'DynamoDB': 'Firestore'
            },
            'azure_to_gcp': {
                'Container Apps': 'Cloud Run',
                'PostgreSQL Flexible Server': 'Cloud SQL',
                'Blob Storage': 'Cloud Storage',
                'Cosmos DB': 'Firestore'
            }
        }
        
        mapping_key = f"{self.source_cloud}_to_{target_cloud}"
        if mapping_key in equivalents:
            target_type = equivalents[mapping_key].get(source_resource['type'], 'Unknown')
            return {
                'type': target_type,
                'name': source_resource['name'].replace(self.source_cloud, target_cloud),
                'configuration': self._generate_target_config(source_resource, target_type)
            }
        
        return {'type': 'Unknown', 'name': 'Unknown'}
    
    def _generate_target_config(self, source_resource: Dict, target_type: str) -> Dict:
        """Generate configuration for target resource"""
        # Placeholder - implement actual configuration generation
        return {
            'cpu': '2 vCPU',
            'memory': '4GB',
            'scaling': 'auto',
            'region': 'primary'
        }
    
    async def execute_migration(self, migration_plan: MigrationPlan, dry_run: bool = True) -> bool:
        """Execute the migration plan"""
        logger.info("Starting migration execution", dry_run=dry_run)
        
        if dry_run:
            logger.info("DRY RUN MODE - No actual changes will be made")
        
        try:
            # Pre-migration validation
            await self._validate_migration_prerequisites()
            
            # Execute migration steps
            for step in migration_plan.components:
                await self._execute_migration_step(step, dry_run)
            
            # Post-migration validation
            if not dry_run:
                await self._validate_migration_success()
            
            logger.info("Migration completed successfully")
            return True
            
        except Exception as e:
            logger.error("Migration failed", error=str(e))
            if not dry_run:
                await self._execute_rollback(migration_plan)
            return False
    
    async def _execute_migration_step(self, step: Dict, dry_run: bool):
        """Execute individual migration step"""
        logger.info("Executing migration step", 
                   source=step['source']['name'],
                   target=step['target']['name'],
                   dry_run=dry_run)
        
        if dry_run:
            await asyncio.sleep(0.1)  # Simulate work
            return
        
        # Actual migration logic would go here
        # This would involve:
        # 1. Creating target resources
        # 2. Migrating data
        # 3. Updating DNS/routing
        # 4. Validating functionality
        pass

async def main():
    """Main migration CLI"""
    parser = argparse.ArgumentParser(description="Cloud Migration Utility")
    parser.add_argument("--source", required=True, choices=['aws', 'azure', 'gcp'])
    parser.add_argument("--target", required=True, choices=['aws', 'azure', 'gcp'])
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="Actually execute migration")
    
    args = parser.parse_args()
    
    if args.source == args.target:
        print("Source and target clouds cannot be the same")
        return
    
    migrator = CloudMigrator(args.source, args.target)
    
    # Create migration plan
    plan = await migrator.create_migration_plan()
    print(f"Migration Plan: {args.source} -> {args.target}")
    print(f"Components to migrate: {len(plan.components)}")
    print(f"Estimated downtime: {plan.estimated_downtime}")
    
    # Execute if requested
    if args.execute:
        success = await migrator.execute_migration(plan, dry_run=not args.execute)
        if success:
            print("Migration completed successfully!")
        else:
            print("Migration failed - check logs for details")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Disaster Recovery and High Availability

### Multi-Region Deployment Strategy

**Active-Passive Configuration:**
```yaml
# disaster-recovery/active-passive.yml
# Primary Region (Active)
primary_region:
  aws:
    region: us-east-1
    services:
      - ecs_cluster: medassistant-primary
      - rds_cluster: medassistant-db-primary
      - s3_bucket: medassistant-docs-primary
    
  azure:
    region: eastus
    services:
      - container_apps: medassistant-primary
      - postgresql: medassistant-db-primary
      - storage_account: medassistantprimary
    
  gcp:
    region: us-central1
    services:
      - cloud_run: medassistant-primary
      - cloud_sql: medassistant-db-primary
      - storage_bucket: medassistant-docs-primary

# Secondary Region (Passive/Standby)
secondary_region:
  aws:
    region: us-west-2
    services:
      - ecs_cluster: medassistant-secondary
      - rds_replica: medassistant-db-replica
      - s3_bucket: medassistant-docs-replica
    
  azure:
    region: westus2  
    services:
      - container_apps: medassistant-secondary
      - postgresql_replica: medassistant-db-replica
      - storage_account: medassistantsecondary
    
  gcp:
    region: us-west1
    services:
      - cloud_run: medassistant-secondary
      - cloud_sql_replica: medassistant-db-replica
      - storage_bucket: medassistant-docs-replica

# Failover Configuration
failover:
  rto: "15 minutes"  # Recovery Time Objective
  rpo: "5 minutes"   # Recovery Point Objective
  
  triggers:
    - health_check_failures: 3
    - response_time_threshold: "5 seconds"
    - error_rate_threshold: "5%"
  
  automation:
    dns_failover: true
    database_promotion: true
    storage_sync: true
```

### Cross-Cloud Backup Strategy

**Backup and Sync Utility:**
```python
# disaster-recovery/backup_sync.py
import asyncio
from typing import Dict, List
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

class CrossCloudBackupManager:
    """Manages backups and synchronization across cloud providers"""
    
    def __init__(self, primary_cloud: str, backup_clouds: List[str]):
        self.primary_cloud = primary_cloud
        self.backup_clouds = backup_clouds
        self.backup_schedule = {
            'database': '0 2 * * *',  # Daily at 2 AM
            'storage': '0 4 * * *',   # Daily at 4 AM
            'config': '0 6 * * 0'     # Weekly on Sunday at 6 AM
        }
    
    async def create_cross_cloud_backup(self, backup_type: str) -> Dict:
        """Create backup across all configured clouds"""
        logger.info("Starting cross-cloud backup", type=backup_type)
        
        backup_results = {}
        backup_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        try:
            if backup_type == 'database':
                backup_results = await self._backup_databases(backup_timestamp)
            elif backup_type == 'storage':
                backup_results = await self._backup_storage(backup_timestamp)
            elif backup_type == 'config':
                backup_results = await self._backup_configurations(backup_timestamp)
            
            # Verify backup integrity
            verification_results = await self._verify_backups(backup_results)
            
            logger.info("Cross-cloud backup completed", 
                       type=backup_type,
                       timestamp=backup_timestamp,
                       results=backup_results)
            
            return {
                'timestamp': backup_timestamp,
                'type': backup_type,
                'backups': backup_results,
                'verification': verification_results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error("Cross-cloud backup failed", 
                        type=backup_type, 
                        error=str(e))
            return {
                'timestamp': backup_timestamp,
                'type': backup_type,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _backup_databases(self, timestamp: str) -> Dict:
        """Backup databases across clouds"""
        results = {}
        
        # Primary cloud database backup
        primary_backup = await self._create_database_backup(
            self.primary_cloud, timestamp, is_primary=True
        )
        results[self.primary_cloud] = primary_backup
        
        # Replicate to backup clouds
        for cloud in self.backup_clouds:
            if cloud != self.primary_cloud:
                backup_result = await self._replicate_database_backup(
                    primary_backup, cloud, timestamp
                )
                results[cloud] = backup_result
        
        return results
    
    async def _create_database_backup(self, cloud: str, timestamp: str, is_primary: bool = False) -> Dict:
        """Create database backup for specific cloud"""
        if cloud == 'aws':
            return await self._aws_database_backup(timestamp, is_primary)
        elif cloud == 'azure':
            return await self._azure_database_backup(timestamp, is_primary)
        elif cloud == 'gcp':
            return await self._gcp_database_backup(timestamp, is_primary)
    
    async def _aws_database_backup(self, timestamp: str, is_primary: bool) -> Dict:
        """Create AWS RDS backup"""
        import boto3
        
        rds_client = boto3.client('rds')
        
        try:
            if is_primary:
                # Create manual snapshot
                snapshot_id = f"medassistant-backup-{timestamp}"
                response = rds_client.create_db_snapshot(
                    DBSnapshotIdentifier=snapshot_id,
                    DBInstanceIdentifier='medassistant-db'
                )
                
                # Wait for snapshot completion
                waiter = rds_client.get_waiter('db_snapshot_completed')
                waiter.wait(DBSnapshotIdentifier=snapshot_id)
                
            # Export to S3 for cross-cloud access
            export_task = rds_client.start_export_task(
                ExportTaskIdentifier=f"export-{timestamp}",
                SourceArn=f"arn:aws:rds:us-east-1:ACCOUNT:snapshot:{snapshot_id}",
                S3BucketName='medassistant-backups',
                S3Prefix=f'database-exports/{timestamp}/',
                IamRoleArn='arn:aws:iam::ACCOUNT:role/rds-export-role'
            )
            
            return {
                'status': 'success',
                'snapshot_id': snapshot_id,
                'export_task_id': export_task['ExportTaskIdentifier'],
                'location': f's3://medassistant-backups/database-exports/{timestamp}/'
            }
            
        except Exception as e:
            logger.error("AWS database backup failed", error=str(e))
            return {'status': 'failed', 'error': str(e)}
    
    async def restore_from_backup(self, backup_info: Dict, target_cloud: str) -> bool:
        """Restore service from backup to target cloud"""
        logger.info("Starting restore operation", 
                   target_cloud=target_cloud,
                   backup_timestamp=backup_info['timestamp'])
        
        try:
            # Restore database
            db_restore = await self._restore_database(backup_info, target_cloud)
            
            # Restore storage
            storage_restore = await self._restore_storage(backup_info, target_cloud)
            
            # Restore configuration
            config_restore = await self._restore_configuration(backup_info, target_cloud)
            
            # Validate restore
            validation = await self._validate_restore(target_cloud)
            
            if all([db_restore, storage_restore, config_restore, validation]):
                logger.info("Restore completed successfully", target_cloud=target_cloud)
                return True
            else:
                logger.error("Restore validation failed", target_cloud=target_cloud)
                return False
                
        except Exception as e:
            logger.error("Restore operation failed", 
                        target_cloud=target_cloud, 
                        error=str(e))
            return False
    
    async def test_disaster_recovery(self) -> Dict:
        """Test disaster recovery procedures"""
        logger.info("Starting disaster recovery test")
        
        test_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'tests': [],
            'overall_status': 'unknown'
        }
        
        try:
            # Test 1: Database failover
            db_test = await self._test_database_failover()
            test_results['tests'].append(db_test)
            
            # Test 2: Application failover
            app_test = await self._test_application_failover()
            test_results['tests'].append(app_test)
            
            # Test 3: Storage sync
            storage_test = await self._test_storage_sync()
            test_results['tests'].append(storage_test)
            
            # Test 4: End-to-end functionality
            e2e_test = await self._test_end_to_end_functionality()
            test_results['tests'].append(e2e_test)
            
            # Determine overall status
            all_passed = all(test['status'] == 'passed' for test in test_results['tests'])
            test_results['overall_status'] = 'passed' if all_passed else 'failed'
            
            logger.info("Disaster recovery test completed", 
                       status=test_results['overall_status'])
            
            return test_results
            
        except Exception as e:
            logger.error("Disaster recovery test failed", error=str(e))
            test_results['overall_status'] = 'error'
            test_results['error'] = str(e)
            return test_results

# Monitoring and Alerting for DR
class DisasterRecoveryMonitor:
    """Monitor disaster recovery readiness and trigger failover"""
    
    def __init__(self):
        self.health_checks = {}
        self.failover_triggered = False
    
    async def monitor_primary_health(self) -> Dict:
        """Monitor primary region health"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'primary_region': 'healthy',
            'services': {}
        }
        
        # Check application health
        app_health = await self._check_application_health()
        health_status['services']['application'] = app_health
        
        # Check database health
        db_health = await self._check_database_health()
        health_status['services']['database'] = db_health
        
        # Check storage health
        storage_health = await self._check_storage_health()
        health_status['services']['storage'] = storage_health
        
        # Determine overall health
        all_healthy = all(
            service['status'] == 'healthy' 
            for service in health_status['services'].values()
        )
        
        health_status['primary_region'] = 'healthy' if all_healthy else 'unhealthy'
        
        # Trigger failover if needed
        if not all_healthy and not self.failover_triggered:
            await self._trigger_failover(health_status)
        
        return health_status
    
    async def _trigger_failover(self, health_status: Dict):
        """Trigger automatic failover to secondary region"""
        logger.critical("Triggering automatic failover", 
                       health_status=health_status)
        
        self.failover_triggered = True
        
        # Implementation would include:
        # 1. Update DNS to point to secondary region
        # 2. Promote read replica to primary
        # 3. Start services in secondary region
        # 4. Notify operations team
        
        # Placeholder for actual failover logic
        await asyncio.sleep(1)
        
        logger.info("Failover procedure initiated")
```

---

## Platform-Specific Optimizations Summary

### AWS Optimizations
- **ECS Service Auto Scaling** with custom metrics
- **CloudFront CDN** for global content delivery
- **ElastiCache Redis** for session management
- **X-Ray tracing** for performance monitoring
- **Parameter Store** for secure configuration management

### Azure Optimizations
- **Container Apps KEDA scaling** with multiple triggers
- **Azure Front Door** for intelligent routing
- **Azure Cache for Redis** with clustering
- **Application Insights** for comprehensive monitoring
- **Key Vault** for secrets and certificate management

### GCP Optimizations
- **Cloud Run concurrency** tuning for optimal performance
- **Cloud CDN** with custom caching policies
- **Memorystore Redis** with high availability
- **Cloud Monitoring** with custom dashboards
- **Secret Manager** with automatic rotation

---

## Migration Checklist

### Pre-Migration Planning
- [ ] **Resource Inventory**: Complete audit of current cloud resources
- [ ] **Dependency Mapping**: Identify all service dependencies
- [ ] **Performance Baseline**: Establish current performance metrics
- [ ] **Cost Analysis**: Compare pricing across target clouds
- [ ] **Compliance Review**: Ensure regulatory requirements are met

### Migration Execution
- [ ] **Backup Creation**: Full backup of all data and configurations
- [ ] **Target Environment Setup**: Deploy infrastructure in target cloud
- [ ] **Data Migration**: Transfer databases and storage with validation
- [ ] **Application Deployment**: Deploy and configure application services
- [ ] **DNS/Traffic Routing**: Update routing to new environment

### Post-Migration Validation
- [ ] **Functionality Testing**: Verify all features work correctly
- [ ] **Performance Testing**: Confirm performance meets requirements
- [ ] **Security Audit**: Validate security configurations
- [ ] **Monitoring Setup**: Ensure all monitoring and alerting is operational
- [ ] **Documentation Update**: Update deployment and operational docs

### Rollback Preparation
- [ ] **Rollback Plan**: Detailed procedures for reverting changes
- [ ] **Rollback Testing**: Test rollback procedures in non-production
- [ ] **Communication Plan**: Stakeholder notification procedures
- [ ] **Data Sync Strategy**: Plan for data consistency during rollback

---

## Summary

This chapter demonstrated comprehensive multi-cloud deployment strategies for FastAPI applications, covering AWS, Azure, and GCP implementations. Key takeaways include:

**Strategic Benefits:**
- **Vendor Independence**: Avoid lock-in with cloud-agnostic architectures
- **Cost Optimization**: Leverage best pricing across different providers
- **Performance**: Choose optimal regions and services for your use case
- **Resilience**: Multi-cloud disaster recovery capabilities

**Implementation Approach:**
- **Abstraction Layers**: Use interfaces to abstract cloud-specific services
- **Infrastructure as Code**: Terraform/Bicep for reproducible deployments
- **Configuration-Driven**: Environment-specific settings for easy migration
- **Automation**: Scripts and tools for migration and disaster recovery

**Lessons Learned:**
- Each cloud has unique strengths and pricing models
- Migration complexity varies significantly by service type
- Monitoring and observability require cloud-specific configurations
- Disaster recovery testing is essential for production readiness

The multi-cloud approach provides flexibility and resilience but requires careful architecture planning and operational overhead. Choose the strategy that best fits your organization's risk tolerance and technical capabilities.

---

*Next: Chapter 9 - AI Service Integration*