# terraform/outputs.tf - Infrastructure Outputs

output "alb_dns_name" {
  description = "Public DNS name of the Application Load Balancer serving the FastAPI inference service"
  value       = aws_lb.api_alb.dns_name
}

output "ecr_repository_url" {
  description = "Amazon ECR repository URL for Docker image push"
  value       = aws_ecr_repository.inference_service.repository_url
}

output "s3_bucket_name" {
  description = "Amazon S3 bucket name storing model artifacts, datasets, and Model Cards"
  value       = aws_s3_bucket.mlops_artifacts.id
}

output "step_functions_state_machine_arn" {
  description = "ARN of the automated retraining Step Functions state machine"
  value       = aws_sfn_state_machine.mlops_retraining_pipeline.arn
}

output "ecs_cluster_name" {
  description = "Amazon ECS Fargate cluster name"
  value       = aws_ecs_cluster.mlops_cluster.name
}

output "sns_topic_arn" {
  description = "Amazon SNS Topic ARN for MLOps retraining alerts"
  value       = aws_sns_topic.retraining_alerts.arn
}
