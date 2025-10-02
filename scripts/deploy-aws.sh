#!/bin/bash

# AWS Deployment Script for Document Intelligence Platform

set -e

# Configuration
STACK_NAME="di-platform"
REGION="us-east-1"
KEY_PAIR_NAME="di-platform-key"
INSTANCE_TYPE="t3.large"
DB_INSTANCE_CLASS="db.t3.micro"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Document Intelligence Platform deployment...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

# Prompt for required parameters
echo -e "${YELLOW}📝 Please provide the following information:${NC}"
read -s -p "Database Password (min 8 characters): " DB_PASSWORD
echo
read -s -p "OpenAI API Key: " OPENAI_API_KEY
echo

# Validate inputs
if [ ${#DB_PASSWORD} -lt 8 ]; then
    echo -e "${RED}❌ Database password must be at least 8 characters long.${NC}"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ OpenAI API Key is required.${NC}"
    exit 1
fi

# Create S3 bucket for CloudFormation templates
BUCKET_NAME="di-platform-templates-$(date +%s)"
echo -e "${YELLOW}📦 Creating S3 bucket for templates: $BUCKET_NAME${NC}"
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Upload CloudFormation template
echo -e "${YELLOW}📤 Uploading CloudFormation template...${NC}"
aws s3 cp aws/cloudformation-template.yaml s3://$BUCKET_NAME/ --region $REGION

# Deploy CloudFormation stack
echo -e "${YELLOW}🏗️  Deploying CloudFormation stack...${NC}"
aws cloudformation deploy \
    --template-file aws/cloudformation-template.yaml \
    --stack-name $STACK_NAME \
    --parameter-overrides \
        KeyPairName=$KEY_PAIR_NAME \
        InstanceType=$INSTANCE_TYPE \
        DBInstanceClass=$DB_INSTANCE_CLASS \
        DatabasePassword="$DB_PASSWORD" \
        OpenAIAPIKey="$OPENAI_API_KEY" \
    --capabilities CAPABILITY_IAM \
    --region $REGION

# Get stack outputs
echo -e "${YELLOW}📋 Getting deployment information...${NC}"
WEBSITE_URL=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`WebsiteURL`].OutputValue' \
    --output text)

DATABASE_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
    --output text)

REDIS_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`RedisEndpoint`].OutputValue' \
    --output text)

S3_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --region $REGION \
    --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
    --output text)

# Wait for EC2 instance to be ready
echo -e "${YELLOW}⏳ Waiting for EC2 instance to be ready...${NC}"
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=DI-Platform-Instance" \
    --region $REGION \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo
echo -e "${GREEN}🌐 Application Information:${NC}"
echo -e "   Website URL: http://$PUBLIC_IP"
echo -e "   Database Endpoint: $DATABASE_ENDPOINT"
echo -e "   Redis Endpoint: $REDIS_ENDPOINT"
echo -e "   S3 Bucket: $S3_BUCKET"
echo
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo -e "   1. Wait 5-10 minutes for the application to fully start"
echo -e "   2. Visit http://$PUBLIC_IP to access the web interface"
echo -e "   3. Upload documents and start asking questions!"
echo
echo -e "${YELLOW}🔧 Monitoring:${NC}"
echo -e "   Prometheus: http://$PUBLIC_IP:9090"
echo -e "   Grafana: http://$PUBLIC_IP:3000 (admin/admin)"
echo
echo -e "${YELLOW}📊 To view logs:${NC}"
echo -e "   ssh -i ~/.ssh/$KEY_PAIR_NAME.pem ec2-user@$PUBLIC_IP"
echo -e "   cd di-platform && docker-compose logs -f"
echo
echo -e "${GREEN}🎉 Document Intelligence Platform is now running on AWS!${NC}"
