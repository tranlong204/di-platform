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
