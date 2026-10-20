#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  printf 'Run this script with sudo.\n' >&2
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  printf 'Unable to identify the operating system.\n' >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID}" != "ubuntu" || "${VERSION_ID}" != "24.04" ]]; then
  printf 'This bootstrap is pinned to Ubuntu Server 24.04; found %s %s.\n' "${ID}" "${VERSION_ID}" >&2
  exit 1
fi

architecture="$(dpkg --print-architecture)"
if [[ "${architecture}" != "amd64" ]]; then
  printf 'The current CPU image is verified for amd64; found %s.\n' "${architecture}" >&2
  exit 1
fi

memory_kib="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
if (( memory_kib < 3500000 )); then
  printf 'At least 4 GiB RAM is required for the API, local models, parser, and PostgreSQL.\n' >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes --no-install-recommends \
  awscli \
  ca-certificates \
  docker.io \
  docker-compose-v2 \
  git \
  jq

systemctl enable --now docker

docker --version
docker compose version
aws --version
jq --version

cat <<'EOF'
Host prerequisites are ready.

Next, attach the AgenticRagEc2Role instance profile, create the documented SSM
parameters, and run:

  sudo env AWS_REGION=ap-south-1 SSM_PARAMETER_PATH=/agentic-rag/prod \
    bash deploy/aws/manage.sh deploy
EOF
