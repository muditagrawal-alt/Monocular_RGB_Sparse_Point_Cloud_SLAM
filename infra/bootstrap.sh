#!/usr/bin/env bash
# SUPERSEDED. The service now deploys to Modal; see infra/modal_app.py.
# Kept as the record of the original AWS deployment, which shipped and ran
# before the account's credits were exhausted. The resources it creates were
# all torn down, so this runs from scratch against an empty account.
#
# One-time AWS setup: ECR repository, IAM roles and the log group.
# Run once per account, then use deploy.sh for every release.
set -euo pipefail

: "${AWS_PROFILE:=ohive}"
: "${AWS_REGION:=us-east-1}"
export AWS_PROFILE AWS_DEFAULT_REGION="$AWS_REGION"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"

echo "==> ECR repository"
aws ecr create-repository --repository-name monocular-slam \
  --image-scanning-configuration scanOnPush=true >/dev/null 2>&1 || echo "    exists"

echo "==> log group"
aws logs create-log-group --log-group-name /ecs/monocular-slam >/dev/null 2>&1 || echo "    exists"
aws logs put-retention-policy --log-group-name /ecs/monocular-slam --retention-in-days 7

echo "==> service-linked roles"
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com >/dev/null 2>&1 || echo "    ecs exists"
aws iam create-service-linked-role --aws-service-name ecs.application-autoscaling.amazonaws.com >/dev/null 2>&1 || echo "    autoscaling exists"

echo "==> task execution role (pull from ECR, write logs)"
aws iam create-role --role-name slamTaskExecutionRole --assume-role-policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null 2>&1 || echo "    exists"
aws iam attach-role-policy --role-name slamTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

echo "==> infrastructure role (manages ALB, target groups, scaling)"
aws iam create-role --role-name slamExpressInfraRole --assume-role-policy-document '{
  "Version":"2012-10-17","Statement":[{"Effect":"Allow",
  "Principal":{"Service":"ecs.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null 2>&1 || echo "    exists"
aws iam attach-role-policy --role-name slamExpressInfraRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices

echo "==> creating the Express Gateway service (4 vCPU / 8 GB, fits the default 8 vCPU Fargate quota)"
cat > /tmp/slam-express.json <<JSON
{
  "serviceName": "monocular-slam-mudit-agrawal",
  "executionRoleArn": "arn:aws:iam::${ACCOUNT}:role/slamTaskExecutionRole",
  "infrastructureRoleArn": "arn:aws:iam::${ACCOUNT}:role/slamExpressInfraRole",
  "healthCheckPath": "/healthz",
  "primaryContainer": {
    "image": "${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/monocular-slam:latest",
    "containerPort": 8000,
    "awsLogsConfiguration": {"logGroup": "/ecs/monocular-slam", "logStreamPrefix": "slam"}
  },
  "cpu": "4096",
  "memory": "8192",
  "cpuArchitecture": "X86_64"
}
JSON
aws ecs create-express-gateway-service --cli-input-json file:///tmp/slam-express.json \
  --query 'service.serviceName' --output text 2>/dev/null || echo "    service exists"

echo "==> done. now run ./infra/deploy.sh"
