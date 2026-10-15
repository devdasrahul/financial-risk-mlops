# terraform/step_functions.tf - AWS Step Functions Retraining Pipeline & SNS Alerts

# ------------------------------------------------------------------------------
# SNS Topic for Pipeline Notifications
# ------------------------------------------------------------------------------
resource "aws_sns_topic" "retraining_alerts" {
  name              = "${var.project_name}-retraining-alerts"
  kms_master_key_id = "alias/aws/sns"
}

# ------------------------------------------------------------------------------
# AWS Step Functions State Machine
# ------------------------------------------------------------------------------
resource "aws_sfn_state_machine" "mlops_retraining_pipeline" {
  name     = "${var.project_name}-automated-retraining"
  role_arn = aws_iam_role.step_functions_role.arn

  definition = templatefile("${path.module}/../statemachine/mlops_retraining_pipeline.asl.json", {})

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.step_functions_logs.arn}:*"
    include_execution_data = true
    level                  = "ALL"
  }
}
