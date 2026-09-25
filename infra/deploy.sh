#!/usr/bin/env bash
# SUPERSEDED. The service now deploys to Modal; see infra/modal_app.py.
# Kept as the record of the original AWS deployment. Requires bootstrap.sh
# to have been run first, since the AWS resources were torn down.
#
# Build, push and deploy the service to AWS ECS Express Mode.
#
# Express Mode provisions the Fargate service, Application Load Balancer, TLS
# certificate, auto-scaling and an AWS-provided HTTPS hostname from a container
# image. AWS App Runner would have been the obvious choice but stopped
# accepting new customers on 2026-04-30.
#
#   ./infra/deploy.sh                 build, push and roll out
#   ./infra/deploy.sh --config-only   change env vars without rebuilding
set -euo pipefail

: "${AWS_PROFILE:=ohive}"
: "${AWS_REGION:=us-east-1}"
export AWS_PROFILE AWS_DEFAULT_REGION="$AWS_REGION"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPO="monocular-slam"
SERVICE_ARN="arn:aws:ecs:${AWS_REGION}:${ACCOUNT}:service/default/${REPO}"
LOG_GROUP="/ecs/${REPO}"
TAG="$(git rev-parse --short HEAD)"

# Runtime tuning. A Fargate vCPU is roughly 2.5x slower than an Apple Silicon
# core, so the deployed settings are more conservative than the development
# defaults. Every value here is reported back in each result, so a run is never
# ambiguous about what produced it.
: "${SLAM_TARGET_WIDTH:=448}"
: "${SLAM_MAX_FEATURES:=560}"
: "${SLAM_LOOP_VERIFICATIONS:=100}"
: "${SLAM_VERIFY_COST_S:=0.030}"
: "${SLAM_RTF_TARGET:=0.80}"

container_spec() {
  cat <<JSON
{
  "image": "${REGISTRY}/${REPO}:${1}",
  "containerPort": 8000,
  "awsLogsConfiguration": {"logGroup": "${LOG_GROUP}", "logStreamPrefix": "slam"},
  "environment": [
    {"name": "SLAM_TARGET_WIDTH",       "value": "${SLAM_TARGET_WIDTH}"},
    {"name": "SLAM_MAX_FEATURES",       "value": "${SLAM_MAX_FEATURES}"},
    {"name": "SLAM_LOOP_VERIFICATIONS", "value": "${SLAM_LOOP_VERIFICATIONS}"},
    {"name": "SLAM_VERIFY_COST_S",      "value": "${SLAM_VERIFY_COST_S}"},
    {"name": "SLAM_RTF_TARGET",         "value": "${SLAM_RTF_TARGET}"}
  ]
}
JSON
}

if [[ "${1:-}" != "--config-only" ]]; then
  echo "==> building ${REPO}:${TAG} for linux/amd64"
  # --provenance=false keeps the result a plain manifest. Buildx attestation
  # produces an OCI image index that Fargate fails to pull.
  docker build --platform linux/amd64 --provenance=false --sbom=false \
    -f docker/Dockerfile -t "${REGISTRY}/${REPO}:${TAG}" .

  echo "==> pushing to ECR"
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
  docker push "${REGISTRY}/${REPO}:${TAG}"
else
  TAG="$(aws ecs describe-express-gateway-service --service "$SERVICE_ARN" \
        --query 'service.primaryContainer.image' --output text 2>/dev/null \
        | awk -F: '{print $NF}')"
  echo "==> config-only update, keeping image tag ${TAG}"
fi

echo "==> updating the service"
container_spec "$TAG" > /tmp/slam-container.json
aws ecs update-express-gateway-service \
  --service-arn "$SERVICE_ARN" \
  --primary-container "file:///tmp/slam-container.json" \
  --query 'service.status.statusCode' --output text

echo "==> waiting for rollout"
until [ "$(aws ecs describe-services --cluster default --services "$REPO" \
          --query 'services[0].deployments[0].rolloutState' --output text)" = "COMPLETED" ]; do
  sleep 15
done

URL="$(aws elbv2 describe-rules \
  --listener-arn "$(aws elbv2 describe-listeners \
      --load-balancer-arn "$(aws elbv2 describe-load-balancers \
          --query 'LoadBalancers[0].LoadBalancerArn' --output text)" \
      --query 'Listeners[0].ListenerArn' --output text)" \
  --query 'Rules[?Priority!=`default`].Conditions[0].Values[0]' --output text | head -1)"

echo "==> deployed: https://${URL}"
curl -fsS "https://${URL}/healthz" && echo
