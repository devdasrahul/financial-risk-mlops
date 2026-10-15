# terraform/variables.tf - Input Variables for MLOps Infrastructure

variable "aws_region" {
  description = "Target AWS deployment region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "project_name" {
  description = "Unique identifier prefix for all MLOps platform resources"
  type        = string
  default     = "fraud-risk-mlops"
}

variable "container_port" {
  description = "Port exposed by the FastAPI inference service container"
  type        = number
  default     = 8000
}

variable "fargate_cpu" {
  description = "CPU units allocated for ECS Fargate task (1024 = 1 vCPU)"
  type        = number
  default     = 1024
}

variable "fargate_memory" {
  description = "Memory allocated for ECS Fargate task in MB"
  type        = number
  default     = 2048
}

variable "desired_task_count" {
  description = "Desired number of active container replicas behind ALB"
  type        = number
  default     = 2
}
