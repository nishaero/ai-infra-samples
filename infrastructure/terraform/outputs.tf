# Outputs for Climate Prediction Infrastructure

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "api_url" {
  description = "URL of the Climate Prediction API"
  value       = "http://${aws_lb.main.dns_name}"
}

output "s3_data_bucket" {
  description = "Name of the S3 data bucket"
  value       = aws_s3_bucket.data.bucket
}

output "s3_models_bucket" {
  description = "Name of the S3 models bucket"
  value       = aws_s3_bucket.models.bucket
}

output "s3_artifacts_bucket" {
  description = "Name of the S3 artifacts bucket"
  value       = aws_s3_bucket.artifacts.bucket
}

output "ecr_api_repository_url" {
  description = "URL of the API ECR repository"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_training_repository_url" {
  description = "URL of the training ECR repository"
  value       = aws_ecr_repository.training.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = var.enable_rds ? aws_db_instance.main[0].endpoint : "N/A"
}

output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  value       = aws_iam_role.ecs_task_execution_role.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task_role.arn
}

# Security group outputs
output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
}

output "ecs_security_group_id" {
  description = "ID of the ECS tasks security group"
  value       = aws_security_group.ecs_tasks.id
}

output "rds_security_group_id" {
  description = "ID of the RDS security group"
  value       = aws_security_group.rds.id
}

# Configuration for application deployment
output "deployment_config" {
  description = "Configuration values for application deployment"
  value = {
    aws_region          = var.aws_region
    vpc_id              = aws_vpc.main.id
    private_subnet_ids  = aws_subnet.private[*].id
    security_group_id   = aws_security_group.ecs_tasks.id
    target_group_arn    = aws_lb_target_group.api.arn
    cluster_name        = aws_ecs_cluster.main.name
    execution_role_arn  = aws_iam_role.ecs_task_execution_role.arn
    task_role_arn       = aws_iam_role.ecs_task_role.arn
    log_group_name      = aws_cloudwatch_log_group.ecs.name
  }
  sensitive = false
}

# Environment variables for the application
output "environment_variables" {
  description = "Environment variables for the application"
  value = {
    AWS_REGION                = var.aws_region
    S3_DATA_BUCKET           = aws_s3_bucket.data.bucket
    S3_MODELS_BUCKET         = aws_s3_bucket.models.bucket
    S3_ARTIFACTS_BUCKET      = aws_s3_bucket.artifacts.bucket
    DATABASE_URL             = var.enable_rds ? "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.main[0].endpoint}/airflow" : ""
    MLFLOW_BACKEND_STORE_URI = var.enable_rds ? "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.main[0].endpoint}/mlflow" : "file:///mlflow/mlruns"
    ENVIRONMENT              = var.environment
  }
  sensitive = true
}