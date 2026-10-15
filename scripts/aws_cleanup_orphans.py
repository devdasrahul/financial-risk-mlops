"""Automated AWS Resource Cleanup Script for Financial Risk MLOps.

Safely discovers and tears down any half-created or orphaned resources
matching the 'fraud-risk-mlops' project prefix to ensure a pristine 100% clean
slate for Terraform provisioning.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aws_cleanup")

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
PROJECT_NAME = os.getenv("PROJECT_NAME", "fraud-risk-mlops")


def cleanup_ecs(session: boto3.Session) -> None:
    logger.info("--- Cleaning up Amazon ECS ---")
    ecs = session.client("ecs", region_name=REGION)
    cluster_name = f"{PROJECT_NAME}-cluster"
    service_name = f"{PROJECT_NAME}-service"

    try:
        # Scale down and delete service
        services = ecs.describe_services(cluster=cluster_name, services=[service_name])
        active_services = [s for s in services.get("services", []) if s.get("status") != "INACTIVE"]
        if active_services:
            logger.info(f"Scaling down and deleting ECS service: {service_name}")
            ecs.update_service(cluster=cluster_name, service=service_name, desiredCount=0)
            ecs.delete_service(cluster=cluster_name, service=service_name, force=True)
            time.sleep(2)
    except ClientError as e:
        logger.info(f"ECS service cleanup note: {e}")

    try:
        logger.info(f"Deleting ECS cluster: {cluster_name}")
        ecs.delete_cluster(cluster=cluster_name)
    except ClientError as e:
        logger.info(f"ECS cluster cleanup note: {e}")


def cleanup_load_balancers(session: boto3.Session) -> None:
    logger.info("--- Cleaning up ALB & Target Groups ---")
    elbv2 = session.client("elbv2", region_name=REGION)

    # Delete Load Balancers
    alb_arn = None
    try:
        lbs = elbv2.describe_load_balancers().get("LoadBalancers", [])
        for lb in lbs:
            if lb.get("LoadBalancerName", "").startswith(f"{PROJECT_NAME}-alb"):
                alb_arn = lb["LoadBalancerArn"]
                logger.info(f"Deleting ALB: {lb['LoadBalancerName']} ({alb_arn})")
                elbv2.delete_load_balancer(LoadBalancerArn=alb_arn)
    except ClientError as e:
        logger.info(f"ALB cleanup note: {e}")

    if alb_arn:
        try:
            logger.info("Waiting for ALB deletion to complete and ENIs to detach...")
            waiter = elbv2.get_waiter("load_balancers_deleted")
            waiter.wait(LoadBalancerArns=[alb_arn], WaiterConfig={"Delay": 5, "MaxAttempts": 20})
            logger.info("ALB successfully deleted and ENIs detached.")
        except Exception as e:
            logger.info(f"ALB waiter note: {e}")
            time.sleep(15)

    # Delete Target Groups
    try:
        tgs = elbv2.describe_target_groups().get("TargetGroups", [])
        for tg in tgs:
            if tg.get("TargetGroupName", "").startswith(f"{PROJECT_NAME}-tg"):
                tg_arn = tg["TargetGroupArn"]
                logger.info(f"Deleting Target Group: {tg['TargetGroupName']} ({tg_arn})")
                elbv2.delete_target_group(TargetGroupArn=tg_arn)
    except ClientError as e:
        logger.info(f"Target group cleanup note: {e}")


def cleanup_security_groups(session: boto3.Session) -> None:
    logger.info("--- Cleaning up Security Groups ---")
    ec2 = session.client("ec2", region_name=REGION)

    try:
        all_sgs = ec2.describe_security_groups().get("SecurityGroups", [])
        matching_sgs = [sg for sg in all_sgs if sg.get("GroupName", "").startswith(PROJECT_NAME)]
    except ClientError as e:
        logger.info(f"Describe SG note: {e}")
        return

    # Step 1: Strip all ingress and egress rules to break cross-group dependencies
    for sg in matching_sgs:
        sg_id = sg["GroupId"]
        if sg.get("IpPermissions"):
            try:
                logger.info(f"Revoking all ingress rules on {sg['GroupName']} ({sg_id})")
                ec2.revoke_security_group_ingress(GroupId=sg_id, IpPermissions=sg["IpPermissions"])
            except ClientError as e:
                logger.info(f"Revoke ingress note: {e}")
        if sg.get("IpPermissionsEgress"):
            try:
                logger.info(f"Revoking all egress rules on {sg['GroupName']} ({sg_id})")
                ec2.revoke_security_group_egress(GroupId=sg_id, IpPermissions=sg["IpPermissionsEgress"])
            except ClientError as e:
                logger.info(f"Revoke egress note: {e}")

    # Step 2: Delete security groups with retry backoff in case ENIs are still freeing
    for sg in matching_sgs:
        sg_id = sg["GroupId"]
        sg_name = sg.get("GroupName", sg_id)
        for attempt in range(6):
            try:
                logger.info(f"Deleting Security Group: {sg_name} ({sg_id})")
                ec2.delete_security_group(GroupId=sg_id)
                logger.info(f"Successfully deleted {sg_name}")
                break
            except ClientError as e:
                logger.info(f"Attempt {attempt+1}/6 to delete {sg_name}: {e}")
                time.sleep(5)


def cleanup_step_functions_and_eventbridge(session: boto3.Session) -> None:
    logger.info("--- Cleaning up Step Functions & EventBridge ---")
    sfn = session.client("stepfunctions", region_name=REGION)
    events = session.client("events", region_name=REGION)
    sts = session.client("sts", region_name=REGION)

    try:
        account_id = sts.get_caller_identity()["Account"]
        sfn_arn = f"arn:aws:states:{REGION}:{account_id}:stateMachine:{PROJECT_NAME}-automated-retraining"
        logger.info(f"Deleting Step Functions State Machine: {sfn_arn}")
        sfn.delete_state_machine(stateMachineArn=sfn_arn)
    except ClientError as e:
        logger.info(f"Step Functions cleanup note: {e}")

    rule_name = f"{PROJECT_NAME}-drift-detected-rule"
    try:
        targets = events.list_targets_by_rule(Rule=rule_name).get("Targets", [])
        if targets:
            target_ids = [t["Id"] for t in targets]
            logger.info(f"Removing targets {target_ids} from EventBridge rule: {rule_name}")
            events.remove_targets(Rule=rule_name, Ids=target_ids)
        logger.info(f"Deleting EventBridge rule: {rule_name}")
        events.delete_rule(Name=rule_name)
    except ClientError as e:
        logger.info(f"EventBridge cleanup note: {e}")


def cleanup_cloudwatch_and_sns(session: boto3.Session) -> None:
    logger.info("--- Cleaning up CloudWatch Logs & SNS ---")
    logs = session.client("logs", region_name=REGION)
    sns = session.client("sns", region_name=REGION)
    sts = session.client("sts", region_name=REGION)

    for log_group in [
        f"/ecs/{PROJECT_NAME}-inference",
        f"/aws/vendedlogs/states/{PROJECT_NAME}-retraining",
    ]:
        try:
            logger.info(f"Deleting CloudWatch Log Group: {log_group}")
            logs.delete_log_group(logGroupName=log_group)
        except ClientError as e:
            logger.info(f"Log group {log_group} cleanup note: {e}")

    try:
        account_id = sts.get_caller_identity()["Account"]
        topic_arn = f"arn:aws:sns:{REGION}:{account_id}:{PROJECT_NAME}-retraining-alerts"
        logger.info(f"Deleting SNS Topic: {topic_arn}")
        sns.delete_topic(TopicArn=topic_arn)
    except ClientError as e:
        logger.info(f"SNS topic cleanup note: {e}")


def cleanup_ecr(session: boto3.Session) -> None:
    logger.info("--- Cleaning up Amazon ECR ---")
    ecr = session.client("ecr", region_name=REGION)
    repo_name = f"{PROJECT_NAME}-inference"
    try:
        logger.info(f"Deleting ECR Repository: {repo_name}")
        ecr.delete_repository(repositoryName=repo_name, force=True)
    except ClientError as e:
        logger.info(f"ECR cleanup note: {e}")


def cleanup_iam(session: boto3.Session) -> None:
    logger.info("--- Cleaning up IAM Roles & Policies ---")
    iam = session.client("iam", region_name=REGION)
    sts = session.client("sts", region_name=REGION)
    account_id = sts.get_caller_identity()["Account"]

    roles = [
        f"{PROJECT_NAME}-ecs-task-execution-role",
        f"{PROJECT_NAME}-ecs-task-role",
        f"{PROJECT_NAME}-step-functions-role",
        f"{PROJECT_NAME}-eventbridge-sfn-role",
    ]

    for role_name in roles:
        try:
            # Detach managed policies
            attached = iam.list_attached_role_policies(RoleName=role_name).get("AttachedPolicies", [])
            for pol in attached:
                logger.info(f"Detaching policy {pol['PolicyArn']} from {role_name}")
                iam.detach_role_policy(RoleName=role_name, PolicyArn=pol["PolicyArn"])

            # Delete inline policies
            inline = iam.list_role_policies(RoleName=role_name).get("PolicyNames", [])
            for pol_name in inline:
                logger.info(f"Deleting inline policy {pol_name} from {role_name}")
                iam.delete_role_policy(RoleName=role_name, PolicyName=pol_name)

            logger.info(f"Deleting IAM Role: {role_name}")
            iam.delete_role(RoleName=role_name)
        except ClientError as e:
            logger.info(f"Role {role_name} cleanup note: {e}")

    policies = [
        f"{PROJECT_NAME}-ecs-task-runtime-policy",
        f"{PROJECT_NAME}-step-functions-policy",
        f"{PROJECT_NAME}-eventbridge-sfn-policy",
    ]

    for pol_name in policies:
        try:
            pol_arn = f"arn:aws:iam::{account_id}:policy/{pol_name}"
            logger.info(f"Deleting IAM Policy: {pol_arn}")
            iam.delete_policy(PolicyArn=pol_arn)
        except ClientError as e:
            logger.info(f"Policy {pol_name} cleanup note: {e}")


def main() -> None:
    logger.info(f"Starting automated cleanup for project: {PROJECT_NAME} in region {REGION}")
    session = boto3.Session(region_name=REGION)

    cleanup_ecs(session)
    cleanup_load_balancers(session)
    cleanup_security_groups(session)
    cleanup_step_functions_and_eventbridge(session)
    cleanup_cloudwatch_and_sns(session)
    cleanup_ecr(session)
    cleanup_iam(session)

    logger.info("=== Cleanup Complete: AWS environment is clean and ready for Terraform ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error(f"Cleanup encountered an unhandled error: {exc}")
        sys.exit(1)
