#!/usr/bin/env bash
set -Eeuo pipefail

AWS_REGION="${AWS_REGION:-ap-south-1}"
SSM_PARAMETER_PATH="${SSM_PARAMETER_PATH:-/agentic-rag/prod}"
EC2_ROLE_NAME="${EC2_ROLE_NAME:-AgenticRagEc2Role}"

failures=0
warnings=0

pass() {
  printf 'PASS: %s\n' "$*"
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  failures=$((failures + 1))
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
  warnings=$((warnings + 1))
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'ERROR: required command %s is unavailable.\n' "$1" >&2
    exit 1
  }
}

resource_count() {
  local label="$1"
  local count="$2"
  if [[ "${count}" == "0" ]]; then
    pass "no existing ${label} found in ${AWS_REGION}"
  else
    warn "found ${count} existing ${label} in ${AWS_REGION}; review them for ongoing cost"
  fi
}

require_command aws
require_command jq

identity_json="$(aws sts get-caller-identity --output json)"
account_id="$(jq -r '.Account' <<<"${identity_json}")"
caller_arn="$(jq -r '.Arn' <<<"${identity_json}")"
pass "authenticated as ${caller_arn} in account ${account_id}"

region_status="$(
  aws ec2 describe-regions \
    --all-regions \
    --region-names "${AWS_REGION}" \
    --query 'Regions[0].OptInStatus' \
    --output text
)"
if [[ "${region_status}" == "opt-in-not-required" || "${region_status}" == "opted-in" ]]; then
  pass "region ${AWS_REGION} is enabled"
else
  fail "region ${AWS_REGION} is not enabled (status=${region_status})"
fi

if role_json="$(aws iam get-role --role-name "${EC2_ROLE_NAME}" --output json 2>/dev/null)"; then
  if jq -e '
    [
      .Role.AssumeRolePolicyDocument.Statement[]?
      | select(.Effect == "Allow")
      | .Principal.Service as $service
      | select(([$service] | flatten | index("ec2.amazonaws.com")) != null)
    ] | length > 0
  ' >/dev/null <<<"${role_json}"; then
    pass "${EC2_ROLE_NAME} trusts EC2"
  else
    fail "${EC2_ROLE_NAME} does not contain an EC2 trust relationship"
  fi
else
  fail "IAM role ${EC2_ROLE_NAME} was not found"
  role_json=""
fi

if [[ -n "${role_json}" ]]; then
  attached_json="$(
    aws iam list-attached-role-policies \
      --role-name "${EC2_ROLE_NAME}" \
      --output json
  )"
  if jq -e '
    [.AttachedPolicies[].PolicyArn]
    | index("arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore") != null
  ' >/dev/null <<<"${attached_json}"; then
    pass "${EC2_ROLE_NAME} has AmazonSSMManagedInstanceCore"
  else
    fail "${EC2_ROLE_NAME} is missing AmazonSSMManagedInstanceCore"
  fi

  profiles_json="$(
    aws iam list-instance-profiles-for-role \
      --role-name "${EC2_ROLE_NAME}" \
      --output json
  )"
  if (( $(jq '.InstanceProfiles | length' <<<"${profiles_json}") > 0 )); then
    profile_names="$(jq -r '[.InstanceProfiles[].InstanceProfileName] | join(", ")' <<<"${profiles_json}")"
    pass "instance profile exists (${profile_names})"
  else
    fail "${EC2_ROLE_NAME} has no EC2 instance profile"
  fi

  expected_parameter_arn="arn:aws:ssm:${AWS_REGION}:${account_id}:parameter${SSM_PARAMETER_PATH}/*"
  matching_inline_policy=false
  inline_policy_names="$(
    aws iam list-role-policies \
      --role-name "${EC2_ROLE_NAME}" \
      --query 'PolicyNames[]' \
      --output text
  )"
  for policy_name in ${inline_policy_names}; do
    policy_json="$(
      aws iam get-role-policy \
        --role-name "${EC2_ROLE_NAME}" \
        --policy-name "${policy_name}" \
        --output json
    )"
    if jq -e --arg expected "${expected_parameter_arn}" '
      [
        .PolicyDocument.Statement[]?
        | select(.Effect == "Allow")
        | .Action as $action
        | .Resource as $resource
        | select(
            (([$action] | flatten | index("ssm:GetParametersByPath")) != null)
            and (([$resource] | flatten | index($expected)) != null)
          )
      ] | length > 0
    ' >/dev/null <<<"${policy_json}"; then
      matching_inline_policy=true
      break
    fi
  done
  if [[ "${matching_inline_policy}" == "true" ]]; then
    pass "role can read only the expected Parameter Store path"
  else
    fail "no inline role policy allows ssm:GetParametersByPath on ${expected_parameter_arn}"
  fi
fi

parameters_json="$(
  aws ssm describe-parameters \
    --region "${AWS_REGION}" \
    --parameter-filters "Key=Name,Option=BeginsWith,Values=${SSM_PARAMETER_PATH}/" \
    --output json
)"

required_parameters=(
  "GROQ_API_KEY:SecureString"
  "OPENROUTER_API_KEY:SecureString"
  "PINECONE_API_KEY:SecureString"
  "POSTGRES_PASSWORD:SecureString"
  "PROVIDER_ORDER:String"
  "PINECONE_INDEX_NAME:String"
  "V3_INDEX_NAME:String"
)

for specification in "${required_parameters[@]}"; do
  key="${specification%%:*}"
  expected_type="${specification##*:}"
  full_name="${SSM_PARAMETER_PATH}/${key}"
  actual_type="$(
    jq -r --arg name "${full_name}" \
      '.Parameters[] | select(.Name == $name) | .Type' \
      <<<"${parameters_json}"
  )"
  if [[ "${actual_type}" == "${expected_type}" ]]; then
    pass "${full_name} exists as ${expected_type}"
  elif [[ -z "${actual_type}" ]]; then
    fail "${full_name} is missing"
  else
    fail "${full_name} is ${actual_type}; expected ${expected_type}"
  fi
done

if budgets_json="$(
  aws budgets describe-budgets \
    --account-id "${account_id}" \
    --region us-east-1 \
    --output json 2>/dev/null
)"; then
  budget_count="$(jq '.Budgets | length' <<<"${budgets_json}")"
  if (( budget_count >= 2 )); then
    budget_names="$(jq -r '[.Budgets[].BudgetName] | join(", ")' <<<"${budgets_json}")"
    pass "${budget_count} budget(s) found (${budget_names})"
  elif (( budget_count == 1 )); then
    fail "only one AWS budget was found; configure both zero-spend and monthly-cost budgets"
  else
    fail "no AWS budget was found"
  fi

  notification_count=0
  while IFS= read -r budget_name; do
    [[ -n "${budget_name}" ]] || continue
    notifications_json="$(
      aws budgets describe-notifications-for-budget \
        --account-id "${account_id}" \
        --budget-name "${budget_name}" \
        --region us-east-1 \
        --output json
    )"
    notification_count=$((
      notification_count
      + $(jq '.Notifications | length' <<<"${notifications_json}")
    ))
  done < <(jq -r '.Budgets[].BudgetName' <<<"${budgets_json}")
  if (( notification_count > 0 )); then
    pass "${notification_count} budget notification rule(s) found"
  else
    fail "budgets exist but no notification rule was found"
  fi
else
  warn "budget API could not be read; verify budget alerts manually in Billing"
fi

instance_count="$(
  aws ec2 describe-instances \
    --region "${AWS_REGION}" \
    --filters 'Name=instance-state-name,Values=pending,running,stopping,stopped' \
    --output json \
    | jq '[.Reservations[].Instances[]] | length'
)"
resource_count "non-terminated EC2 instance(s)" "${instance_count}"

volume_count="$(
  aws ec2 describe-volumes --region "${AWS_REGION}" --output json \
    | jq '.Volumes | length'
)"
resource_count "EBS volume(s)" "${volume_count}"

snapshot_count="$(
  aws ec2 describe-snapshots --region "${AWS_REGION}" --owner-ids self --output json \
    | jq '.Snapshots | length'
)"
resource_count "account-owned EBS snapshot(s)" "${snapshot_count}"

address_count="$(
  aws ec2 describe-addresses --region "${AWS_REGION}" --output json \
    | jq '.Addresses | length'
)"
resource_count "Elastic IP address(es)" "${address_count}"

nat_count="$(
  aws ec2 describe-nat-gateways --region "${AWS_REGION}" --output json \
    | jq '[.NatGateways[] | select(.State != "deleted")] | length'
)"
resource_count "NAT gateway(s)" "${nat_count}"

rds_count="$(
  aws rds describe-db-instances --region "${AWS_REGION}" --output json \
    | jq '.DBInstances | length'
)"
resource_count "RDS database instance(s)" "${rds_count}"

load_balancer_count="$(
  aws elbv2 describe-load-balancers --region "${AWS_REGION}" --output json \
    | jq '.LoadBalancers | length'
)"
resource_count "load balancer(s)" "${load_balancer_count}"

ecr_count="$(
  aws ecr describe-repositories --region "${AWS_REGION}" --output json 2>/dev/null \
    | jq '.repositories | length'
)" || ecr_count=0
resource_count "private ECR repository/repositories" "${ecr_count}"

printf '\nPreflight summary: %d failure(s), %d warning(s).\n' "${failures}" "${warnings}"
if (( failures > 0 )); then
  exit 1
fi
if (( warnings > 0 )); then
  exit 2
fi

printf 'The inspected AWS control-plane configuration is ready for EC2 launch.\n'
