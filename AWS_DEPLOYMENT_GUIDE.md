# AWS Deployment Guide
# Document Intelligence Platform - Complete AWS Setup

## 🚀 Overview

This guide will walk you through deploying your Document Intelligence Platform to AWS using modern cloud services for scalability, reliability, and cost optimization.

## 📋 Prerequisites

### AWS Account Setup
- AWS Account with billing enabled
- AWS CLI installed and configured
- Docker installed locally
- Domain name (optional but recommended)

### Required AWS Services
- **ECS (Elastic Container Service)** - Container orchestration
- **RDS (Relational Database Service)** - PostgreSQL database
- **ElastiCache** - Redis caching
- **S3** - File storage
- **Application Load Balancer** - Traffic distribution
- **CloudFront** - CDN for static assets
- **Route 53** - DNS management
- **Certificate Manager** - SSL certificates
- **CloudWatch** - Monitoring and logging
- **IAM** - Security and permissions

## 🔧 Step 1: Infrastructure Setup

### 1.1 Create AWS Infrastructure Stack

```bash
# Create infrastructure directory
mkdir -p aws/infrastructure
cd aws/infrastructure
```

Create `main.tf` for Terraform infrastructure:

```hcl
# aws/infrastructure/main.tf
terraform {
  required_version = ">= 1.0"
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

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "di-platform-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = {
    Environment = var.environment
    Project     = "di-platform"
  }
}

# RDS PostgreSQL Database
resource "aws_db_subnet_group" "di_platform" {
  name       = "di-platform-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Name = "DI Platform DB subnet group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "di-platform-rds-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "DI Platform RDS Security Group"
  }
}

resource "aws_db_instance" "di_platform" {
  identifier = "di-platform-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "di_platform"
  username = "di_user"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.di_platform.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "di-platform-final-snapshot"
  
  tags = {
    Name = "DI Platform Database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "di_platform" {
  name       = "di-platform-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "di-platform-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  tags = {
    Name = "DI Platform Redis Security Group"
  }
}

resource "aws_elasticache_replication_group" "di_platform" {
  replication_group_id       = "di-platform-redis"
  description                = "DI Platform Redis Cluster"
  
  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  
  subnet_group_name          = aws_elasticache_subnet_group.di_platform.name
  security_group_ids         = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "DI Platform Redis"
  }
}

# S3 Bucket for file storage
resource "aws_s3_bucket" "di_platform_files" {
  bucket = "di-platform-files-${random_string.bucket_suffix.result}"
  
  tags = {
    Name = "DI Platform File Storage"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

resource "aws_s3_bucket_versioning" "di_platform_files" {
  bucket = aws_s3_bucket.di_platform_files.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "di_platform_files" {
  bucket = aws_s3_bucket.di_platform_files.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "di_platform" {
  name = "di-platform-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Name = "DI Platform ECS Cluster"
  }
}

# Application Load Balancer
resource "aws_lb" "di_platform" {
  name               = "di-platform-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
  
  enable_deletion_protection = false
  
  tags = {
    Name = "DI Platform Load Balancer"
  }
}

resource "aws_security_group" "alb" {
  name_prefix = "di-platform-alb-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "DI Platform ALB Security Group"
  }
}

# SSL Certificate (if domain is provided)
resource "aws_acm_certificate" "di_platform" {
  count = var.domain_name != "" ? 1 : 0
  
  domain_name       = var.domain_name
  validation_method = "DNS"
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = {
    Name = "DI Platform SSL Certificate"
  }
}

# Route 53 Hosted Zone (if domain is provided)
resource "aws_route53_zone" "di_platform" {
  count = var.domain_name != "" ? 1 : 0
  
  name = var.domain_name
  
  tags = {
    Name = "DI Platform Hosted Zone"
  }
}

# Outputs
output "vpc_id" {
  value = module.vpc.vpc_id
}

output "database_endpoint" {
  value = aws_db_instance.di_platform.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.di_platform.primary_endpoint_address
}

output "s3_bucket" {
  value = aws_s3_bucket.di_platform_files.bucket
}

output "alb_dns_name" {
  value = aws_lb.di_platform.dns_name
}

output "alb_zone_id" {
  value = aws_lb.di_platform.zone_id
}
```

### 1.2 Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Create terraform.tfvars file
cat > terraform.tfvars << EOF
aws_region = "us-east-1"
environment = "production"
domain_name = "your-domain.com"  # Optional
db_password = "your-secure-password"
EOF

# Plan deployment
terraform plan

# Deploy infrastructure
terraform apply
```

## 🐳 Step 2: Container Setup

### 2.1 Create ECR Repository

```bash
# Create ECR repository
aws ecr create-repository --repository-name di-platform --region us-east-1

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# Build and push image
docker build -f Dockerfile.prod -t di-platform .
docker tag di-platform:latest $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com/di-platform:latest
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com/di-platform:latest
```

### 2.2 Create ECS Task Definition

Create `aws/ecs-task-definition.json`:

```json
{
  "family": "di-platform",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "di-platform",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/di-platform:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "DATABASE_URL",
          "value": "postgresql://di_user:PASSWORD@DB_ENDPOINT:5432/di_platform"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://REDIS_ENDPOINT:6379/0"
        },
        {
          "name": "OPENAI_API_KEY",
          "value": "your-openai-api-key"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:ssm:us-east-1:ACCOUNT_ID:parameter/di-platform/secret-key"
        },
        {
          "name": "JWT_SECRET",
          "valueFrom": "arn:aws:ssm:us-east-1:ACCOUNT_ID:parameter/di-platform/jwt-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/di-platform",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 2.3 Create CloudWatch Log Group

```bash
# Create log group
aws logs create-log-group --log-group-name /ecs/di-platform --region us-east-1
```

## 🔐 Step 3: Security Setup

### 3.1 Create IAM Roles

Create `aws/iam-roles.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:us-east-1:ACCOUNT_ID:log-group:/ecs/di-platform:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter",
        "ssm:GetParameters"
      ],
      "Resource": "arn:aws:ssm:us-east-1:ACCOUNT_ID:parameter/di-platform/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::di-platform-files-*/*"
    }
  ]
}
```

### 3.2 Store Secrets in Parameter Store

```bash
# Generate secrets
SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 32)

# Store in Parameter Store
aws ssm put-parameter --name "/di-platform/secret-key" --value "$SECRET_KEY" --type "SecureString"
aws ssm put-parameter --name "/di-platform/jwt-secret" --value "$JWT_SECRET" --type "SecureString"
```

## 🚀 Step 4: ECS Service Deployment

### 4.1 Create ECS Service

Create `aws/ecs-service.json`:

```json
{
  "serviceName": "di-platform-service",
  "cluster": "di-platform-cluster",
  "taskDefinition": "di-platform",
  "desiredCount": 2,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": ["subnet-12345", "subnet-67890"],
      "securityGroups": ["sg-12345"],
      "assignPublicIp": "ENABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT_ID:targetgroup/di-platform-tg/12345",
      "containerName": "di-platform",
      "containerPort": 8000
    }
  ],
  "healthCheckGracePeriodSeconds": 300,
  "deploymentConfiguration": {
    "maximumPercent": 200,
    "minimumHealthyPercent": 50
  }
}
```

### 4.2 Create Target Group

```bash
# Create target group
aws elbv2 create-target-group \
  --name di-platform-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-12345 \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3
```

### 4.3 Create Load Balancer Listener

```bash
# Get target group ARN
TARGET_GROUP_ARN=$(aws elbv2 describe-target-groups --names di-platform-tg --query 'TargetGroups[0].TargetGroupArn' --output text)

# Create HTTP listener
aws elbv2 create-listener \
  --load-balancer-arn $(aws elbv2 describe-load-balancers --names di-platform-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text) \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN
```

## 📊 Step 5: Monitoring Setup

### 5.1 Create CloudWatch Dashboard

Create `aws/cloudwatch-dashboard.json`:

```json
{
  "widgets": [
    {
      "type": "metric",
      "x": 0,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", "ServiceName", "di-platform-service", "ClusterName", "di-platform-cluster"],
          [".", "MemoryUtilization", ".", ".", ".", "."]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "us-east-1",
        "title": "ECS Service Metrics",
        "period": 300
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "metrics": [
          ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", "di-platform-db"],
          [".", "DatabaseConnections", ".", "."]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "us-east-1",
        "title": "RDS Database Metrics",
        "period": 300
      }
    }
  ]
}
```

### 5.2 Create CloudWatch Alarms

```bash
# CPU utilization alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "di-platform-high-cpu" \
  --alarm-description "High CPU utilization" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:di-platform-alerts

# Memory utilization alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "di-platform-high-memory" \
  --alarm-description "High memory utilization" \
  --metric-name MemoryUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:ACCOUNT_ID:di-platform-alerts
```

## 🌐 Step 6: Domain and SSL Setup

### 6.1 Configure Route 53 (if using custom domain)

```bash
# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers --names di-platform-alb --query 'LoadBalancers[0].DNSName' --output text)

# Create A record
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456789 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.your-domain.com",
        "Type": "A",
        "AliasTarget": {
          "DNSName": "'$ALB_DNS'",
          "EvaluateTargetHealth": true,
          "HostedZoneId": "Z35SXDOTRQ7X7K"
        }
      }
    }]
  }'
```

### 6.2 Add HTTPS Listener

```bash
# Get certificate ARN
CERT_ARN=$(aws acm list-certificates --query 'CertificateSummaryList[?DomainName==`your-domain.com`].CertificateArn' --output text)

# Create HTTPS listener
aws elbv2 create-listener \
  --load-balancer-arn $(aws elbv2 describe-load-balancers --names di-platform-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text) \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=$CERT_ARN \
  --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN
```

## 🔄 Step 7: CI/CD Pipeline

### 7.1 Create GitHub Actions Workflow

Create `.github/workflows/aws-deploy.yml`:

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: di-platform
  ECS_SERVICE: di-platform-service
  ECS_CLUSTER: di-platform-cluster
  ECS_TASK_DEFINITION: aws/ecs-task-definition.json
  CONTAINER_NAME: di-platform

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
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      - name: Run tests
        run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout
        uses: actions/checkout@v3
        
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
          docker build -f Dockerfile.prod -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT
          
      - name: Fill in the new image ID in the Amazon ECS task definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: ${{ env.ECS_TASK_DEFINITION }}
          container-name: ${{ env.CONTAINER_NAME }}
          image: ${{ steps.build-image.outputs.image }}
          
      - name: Deploy Amazon ECS task definition
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
```

## 📋 Step 8: Deployment Script

Create `aws/deploy.sh`:

```bash
#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="di-platform"
ECS_CLUSTER="di-platform-cluster"
ECS_SERVICE="di-platform-service"
TASK_DEFINITION="aws/ecs-task-definition.json"

echo "🚀 Starting AWS deployment..."

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Login to ECR
echo "📦 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# Build and push image
echo "🔨 Building and pushing Docker image..."
docker build -f Dockerfile.prod -t $ECR_REPOSITORY .
docker tag $ECR_REPOSITORY:latest $ECR_REGISTRY/$ECR_REPOSITORY:latest
docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

# Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service \
  --cluster $ECS_CLUSTER \
  --service $ECS_SERVICE \
  --force-new-deployment \
  --region $AWS_REGION

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
aws ecs wait services-stable \
  --cluster $ECS_CLUSTER \
  --services $ECS_SERVICE \
  --region $AWS_REGION

echo "✅ Deployment completed successfully!"

# Get service URL
ALB_DNS=$(aws elbv2 describe-load-balancers --names di-platform-alb --query 'LoadBalancers[0].DNSName' --output text)
echo "🌐 Service URL: http://$ALB_DNS"
```

Make it executable:

```bash
chmod +x aws/deploy.sh
```

## 🎯 Step 9: Complete Deployment

### 9.1 Run Complete Deployment

```bash
# 1. Deploy infrastructure
cd aws/infrastructure
terraform init
terraform apply

# 2. Get infrastructure outputs
DB_ENDPOINT=$(terraform output -raw database_endpoint)
REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)
S3_BUCKET=$(terraform output -raw s3_bucket)
ALB_DNS=$(terraform output -raw alb_dns_name)

# 3. Update task definition with real values
sed -i "s/DB_ENDPOINT/$DB_ENDPOINT/g" aws/ecs-task-definition.json
sed -i "s/REDIS_ENDPOINT/$REDIS_ENDPOINT/g" aws/ecs-task-definition.json
sed -i "s/ACCOUNT_ID/$ACCOUNT_ID/g" aws/ecs-task-definition.json

# 4. Register task definition
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json

# 5. Create ECS service
aws ecs create-service --cli-input-json file://aws/ecs-service.json

# 6. Deploy application
./aws/deploy.sh
```

### 9.2 Verify Deployment

```bash
# Check ECS service status
aws ecs describe-services --cluster di-platform-cluster --services di-platform-service

# Check ALB health
aws elbv2 describe-target-health --target-group-arn $(aws elbv2 describe-target-groups --names di-platform-tg --query 'TargetGroups[0].TargetGroupArn' --output text)

# Test application
curl http://$ALB_DNS/health
```

## 💰 Cost Optimization

### Estimated Monthly Costs (us-east-1)

- **ECS Fargate (2 tasks)**: ~$60/month
- **RDS PostgreSQL (db.t3.medium)**: ~$50/month
- **ElastiCache Redis (cache.t3.micro)**: ~$15/month
- **Application Load Balancer**: ~$20/month
- **S3 Storage (100GB)**: ~$3/month
- **CloudWatch Logs**: ~$10/month
- **Data Transfer**: ~$5/month

**Total: ~$163/month**

### Cost Optimization Tips

1. **Use Spot Instances**: For non-critical workloads
2. **Reserved Instances**: For predictable workloads
3. **Auto Scaling**: Scale based on demand
4. **S3 Lifecycle**: Move old files to cheaper storage
5. **CloudWatch**: Set up billing alerts

## 🔧 Troubleshooting

### Common Issues

1. **ECS Service Won't Start**
   ```bash
   # Check task definition
   aws ecs describe-task-definition --task-definition di-platform
   
   # Check service events
   aws ecs describe-services --cluster di-platform-cluster --services di-platform-service
   ```

2. **Database Connection Issues**
   ```bash
   # Check security groups
   aws ec2 describe-security-groups --group-ids sg-12345
   
   # Test connection
   aws rds describe-db-instances --db-instance-identifier di-platform-db
   ```

3. **Load Balancer Health Check Failures**
   ```bash
   # Check target health
   aws elbv2 describe-target-health --target-group-arn $TARGET_GROUP_ARN
   
   # Check security groups
   aws ec2 describe-security-groups --group-ids sg-12345
   ```

## 📊 Monitoring and Maintenance

### Health Checks

- **Application**: `/health` endpoint
- **Database**: Connection pool monitoring
- **Redis**: Memory usage and connections
- **Load Balancer**: Target group health

### Backup Strategy

```bash
# Database backup
aws rds create-db-snapshot --db-instance-identifier di-platform-db --db-snapshot-identifier di-platform-backup-$(date +%Y%m%d)

# S3 backup (if needed)
aws s3 sync s3://di-platform-files-12345 s3://di-platform-backup-12345
```

## 🎉 Success!

Your Document Intelligence Platform is now deployed on AWS with:

- ✅ **Scalable ECS Fargate** containers
- ✅ **Managed RDS PostgreSQL** database
- ✅ **ElastiCache Redis** for caching
- ✅ **Application Load Balancer** for traffic distribution
- ✅ **S3** for file storage
- ✅ **CloudWatch** monitoring
- ✅ **CI/CD pipeline** for automated deployments
- ✅ **SSL/TLS** encryption
- ✅ **Auto-scaling** capabilities

Your application is now production-ready and can handle enterprise workloads! 🚀
