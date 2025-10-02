# AWS Infrastructure as Code
# Document Intelligence Platform - Terraform Configuration

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

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
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
