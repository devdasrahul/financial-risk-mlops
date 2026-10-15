# terraform/main.tf - Core Cloud Provider, S3, ECR & Logging Resources

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      Platform    = "FinancialRiskMLOps"
    }
  }
}

# Unique suffix for S3 bucket global naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# ------------------------------------------------------------------------------
# Amazon S3 Bucket: Datasets, Model Artifacts, Model Cards, Audit Logs
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "mlops_artifacts" {
  bucket        = "${var.project_name}-artifacts-${random_id.bucket_suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "artifacts_versioning" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts_encryption" {
  bucket = aws_s3_bucket.mlops_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts_block_public" {
  bucket                  = aws_s3_bucket.mlops_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ------------------------------------------------------------------------------
# Amazon ECR Repository: Container Registry with Vulnerability Scanning
# ------------------------------------------------------------------------------
resource "aws_ecr_repository" "inference_service" {
  name                 = "${var.project_name}-inference"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# Lifecycle policy: retain 10 most recent images to conserve cloud cost
resource "aws_ecr_lifecycle_policy" "retention" {
  repository = aws_ecr_repository.inference_service.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 production images"
        selection = {
          tagStatus   = "any"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 30
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# CloudWatch Log Groups
# ------------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "ecs_logs" {
  name              = "/ecs/${var.project_name}-inference"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "step_functions_logs" {
  name              = "/aws/vendedlogs/states/${var.project_name}-retraining"
  retention_in_days = 14
}
