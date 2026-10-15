# terraform/eventbridge.tf - Amazon EventBridge Rule & Step Functions Target

# ------------------------------------------------------------------------------
# EventBridge Rule: Triggers on Evidently AI 'FraudModelDriftDetected' Event
# ------------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "drift_detected_rule" {
  name        = "${var.project_name}-drift-detected-rule"
  description = "Captures Evidently AI covariate drift alerts and triggers Step Functions retraining"

  event_pattern = jsonencode({
    source      = ["custom.fraud.mlops"]
    detail-type = ["FraudModelDriftDetected"]
  })
}

# ------------------------------------------------------------------------------
# EventBridge Target: Step Functions State Machine
# ------------------------------------------------------------------------------
resource "aws_cloudwatch_event_target" "trigger_step_functions" {
  rule      = aws_cloudwatch_event_rule.drift_detected_rule.name
  target_id = "TriggerStepFunctionsRetrainingPipeline"
  arn       = aws_sfn_state_machine.mlops_retraining_pipeline.arn
  role_arn  = aws_iam_role.eventbridge_invoke_sfn_role.arn

  input_transformer {
    input_paths = {
      drift_share       = "$.detail.drift_share"
      drifted_features  = "$.detail.drifted_features"
      event_timestamp   = "$.detail.timestamp"
    }
    input_template = jsonencode({
      DatasetS3Uri      = "s3://${aws_s3_bucket.mlops_artifacts.id}/datasets/latest/"
      TrainingDataUri   = "s3://${aws_s3_bucket.mlops_artifacts.id}/datasets/latest/train.parquet"
      ValidationDataUri = "s3://${aws_s3_bucket.mlops_artifacts.id}/datasets/latest/validation.parquet"
      TestDataUri       = "s3://${aws_s3_bucket.mlops_artifacts.id}/datasets/latest/test.parquet"
      TriggerReason     = "Automated Evidently AI Drift Detection (PSI Breach)"
      DriftShare        = "<drift_share>"
      DriftedFeatures   = "<drifted_features>"
      TriggeredAt       = "<event_timestamp>"
    })
  }
}
