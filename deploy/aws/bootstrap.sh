#!/usr/bin/env bash
set -Eeuo pipefail

AWS_CLI_VERSION="2.38.3"
AWS_CLI_INSTALL_DIR="/usr/local/aws-cli"
AWS_CLI_BIN_DIR="/usr/local/bin"
aws_cli_tmp_dir=""

cleanup() {
  if [[ -n "${aws_cli_tmp_dir}" && -d "${aws_cli_tmp_dir}" ]]; then
    rm -rf "${aws_cli_tmp_dir}"
  fi
}

trap cleanup EXIT

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
  ca-certificates \
  curl \
  docker.io \
  docker-compose-v2 \
  git \
  jq \
  unzip

expected_aws_version="aws-cli/${AWS_CLI_VERSION}"
installed_aws_version=""
if command -v aws >/dev/null 2>&1; then
  installed_aws_version="$(aws --version 2>&1 || true)"
fi

if [[ "${installed_aws_version}" != "${expected_aws_version}"* ]]; then
  aws_cli_tmp_dir="$(mktemp -d)"
  curl --fail --silent --show-error --location \
    "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-${AWS_CLI_VERSION}.zip" \
    --output "${aws_cli_tmp_dir}/awscliv2.zip"
  unzip -q "${aws_cli_tmp_dir}/awscliv2.zip" -d "${aws_cli_tmp_dir}"

  aws_install_args=(
    --install-dir "${AWS_CLI_INSTALL_DIR}"
    --bin-dir "${AWS_CLI_BIN_DIR}"
  )
  if [[ -d "${AWS_CLI_INSTALL_DIR}/v2" ]]; then
    aws_install_args+=(--update)
  fi
  "${aws_cli_tmp_dir}/aws/install" "${aws_install_args[@]}"

  rm -rf "${aws_cli_tmp_dir}"
  aws_cli_tmp_dir=""
fi

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
