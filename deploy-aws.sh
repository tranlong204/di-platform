#!/bin/bash
set -e

echo "🚀 Document Intelligence Platform - AWS Deployment"
echo "=================================================="

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    echo "   Visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "❌ Terraform is not installed. Please install it first."
    echo "   Visit: https://learn.hashicorp.com/tutorials/terraform/install-cli"
    exit 1
fi

echo "✅ All prerequisites are installed!"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "📋 AWS Account ID: $ACCOUNT_ID"

# Get user input
echo ""
echo "📝 Please provide the following information:"
read -p "AWS Region (default: us-east-1): " AWS_REGION
AWS_REGION=${AWS_REGION:-us-east-1}

read -p "Database password: " DB_PASSWORD
if [ -z "$DB_PASSWORD" ]; then
    echo "❌ Database password is required!"
    exit 1
fi

read -p "OpenAI API Key: " OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OpenAI API Key is required!"
    exit 1
fi

read -p "Domain name (optional): " DOMAIN_NAME

echo ""
echo "🚀 Starting deployment..."

# Step 1: Deploy Infrastructure
echo "📦 Step 1: Deploying AWS Infrastructure..."
cd aws/infrastructure

# Create terraform.tfvars
cat > terraform.tfvars << EOF
aws_region = "$AWS_REGION"
environment = "production"
domain_name = "$DOMAIN_NAME"
db_password = "$DB_PASSWORD"
EOF

# Initialize and deploy Terraform
terraform init
terraform plan
terraform apply -auto-approve

# Get infrastructure outputs
DB_ENDPOINT=$(terraform output -raw database_endpoint)
REDIS_ENDPOINT=$(terraform output -raw redis_endpoint)
S3_BUCKET=$(terraform output -raw s3_bucket)
ALB_DNS=$(terraform output -raw alb_dns_name)

echo "✅ Infrastructure deployed successfully!"
echo "   Database: $DB_ENDPOINT"
echo "   Redis: $REDIS_ENDPOINT"
echo "   S3 Bucket: $S3_BUCKET"
echo "   Load Balancer: $ALB_DNS"

cd ../..

# Step 2: Create ECR Repository
echo "📦 Step 2: Creating ECR Repository..."
aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION 2>/dev/null || echo "Repository already exists"

# Step 3: Store Secrets
echo "🔐 Step 3: Storing secrets..."
SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 32)

aws ssm put-parameter --name "/di-platform/secret-key" --value "$SECRET_KEY" --type "SecureString" --overwrite
aws ssm put-parameter --name "/di-platform/jwt-secret" --value "$JWT_SECRET" --type "SecureString" --overwrite

# Step 4: Create CloudWatch Log Group
echo "📊 Step 4: Creating CloudWatch Log Group..."
aws logs create-log-group --log-group-name /ecs/di-platform --region $AWS_REGION 2>/dev/null || echo "Log group already exists"

# Step 5: Update Task Definition
echo "📝 Step 5: Updating ECS Task Definition..."
sed -i.bak "s/ACCOUNT_ID/$ACCOUNT_ID/g" aws/ecs-task-definition.json
sed -i.bak "s/DB_ENDPOINT/$DB_ENDPOINT/g" aws/ecs-task-definition.json
sed -i.bak "s/REDIS_ENDPOINT/$REDIS_ENDPOINT/g" aws/ecs-task-definition.json
sed -i.bak "s/S3_BUCKET/$S3_BUCKET/g" aws/ecs-task-definition.json
sed -i.bak "s/PASSWORD/$DB_PASSWORD/g" aws/ecs-task-definition.json
sed -i.bak "s/your-openai-api-key/$OPENAI_API_KEY/g" aws/ecs-task-definition.json

# Step 6: Register Task Definition
echo "📋 Step 6: Registering ECS Task Definition..."
aws ecs register-task-definition --cli-input-json file://aws/ecs-task-definition.json

# Step 7: Create Target Group
echo "🎯 Step 7: Creating Target Group..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=di-platform-vpc" --query 'Vpcs[0].VpcId' --output text)
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
  --name di-platform-tg \
  --protocol HTTP \
  --port 8000 \
  --vpc-id $VPC_ID \
  --target-type ip \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

# Step 8: Create Load Balancer Listener
echo "🌐 Step 8: Creating Load Balancer Listener..."
ALB_ARN=$(aws elbv2 describe-load-balancers --names di-platform-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text)
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN

# Step 9: Create ECS Service
echo "🚀 Step 9: Creating ECS Service..."
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:Name,Values=*public*" --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
SECURITY_GROUPS=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=*ecs*" --query 'SecurityGroups[0].GroupId' --output text)

aws ecs create-service \
  --cluster di-platform-cluster \
  --service-name di-platform-service \
  --task-definition di-platform \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUPS],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=$TARGET_GROUP_ARN,containerName=di-platform,containerPort=8000" \
  --health-check-grace-period-seconds 300

# Step 10: Deploy Application
echo "🐳 Step 10: Deploying Application..."
./aws/deploy.sh

echo ""
echo "🎉 Deployment completed successfully!"
echo "=================================="
echo "🌐 Application URL: http://$ALB_DNS"
echo "📊 CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group/%2Fecs%2Fdi-platform"
echo "🚀 ECS Console: https://console.aws.amazon.com/ecs/home?region=$AWS_REGION#/clusters/di-platform-cluster/services"

if [ ! -z "$DOMAIN_NAME" ]; then
    echo ""
    echo "🌍 Domain Setup:"
    echo "   1. Create A record for $DOMAIN_NAME pointing to $ALB_DNS"
    echo "   2. Update ALB listener to use HTTPS with SSL certificate"
fi

echo ""
echo "📋 Next Steps:"
echo "   1. Test the application: curl http://$ALB_DNS/health"
echo "   2. Upload documents via the web interface"
echo "   3. Set up monitoring alerts in CloudWatch"
echo "   4. Configure custom domain and SSL certificate"
echo "   5. Set up CI/CD pipeline with GitHub Actions"

echo ""
echo "💰 Estimated Monthly Cost: ~$163"
echo "   - ECS Fargate: ~$60"
echo "   - RDS PostgreSQL: ~$50"
echo "   - ElastiCache Redis: ~$15"
echo "   - Application Load Balancer: ~$20"
echo "   - S3 Storage: ~$3"
echo "   - CloudWatch: ~$10"
echo "   - Data Transfer: ~$5"
