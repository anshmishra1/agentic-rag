#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/compose.yml"
AWS_REGION="${AWS_REGION:-ap-south-1}"
SSM_PARAMETER_PATH="${SSM_PARAMETER_PATH:-/agentic-rag/prod}"
RUNTIME_DIR="/run/agentic-rag"
RUNTIME_ENV_FILE="${RUNTIME_DIR}/runtime.env"
PARAMETER_JSON="${RUNTIME_DIR}/parameters.json"
APP_IMAGE="${APP_IMAGE:-agentic-rag-app:aws-cpu}"

required_parameters=(
  GROQ_API_KEY
  OPENROUTER_API_KEY
  PINECONE_API_KEY
  POSTGRES_PASSWORD
  PROVIDER_ORDER
  PINECONE_INDEX_NAME
  V3_INDEX_NAME
)

cleanup() {
  rm -f -- "${RUNTIME_ENV_FILE}" "${PARAMETER_JSON}"
}
trap cleanup EXIT

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command '$1' is not installed"
}

require_root() {
  [[ "${EUID}" -eq 0 ]] || fail "run this command with sudo"
}

prepare_runtime_env() {
  require_root
  require_command aws
  require_command jq

  install -d -m 0700 "${RUNTIME_DIR}"
  umask 077

  aws ssm get-parameters-by-path \
    --region "${AWS_REGION}" \
    --path "${SSM_PARAMETER_PATH}" \
    --recursive \
    --with-decryption \
    --output json >"${PARAMETER_JSON}"

  jq -r '
    .Parameters
    | sort_by(.Name)[]
    | (.Name | split("/") | last) as $key
    | select($key | test("^[A-Z][A-Z0-9_]*$"))
    | $key + "=" + (.Value | @json)
  ' "${PARAMETER_JSON}" >"${RUNTIME_ENV_FILE}"
  chmod 0600 "${RUNTIME_ENV_FILE}"

  local key
  for key in "${required_parameters[@]}"; do
    grep -q "^${key}=" "${RUNTIME_ENV_FILE}" \
      || fail "SSM path ${SSM_PARAMETER_PATH} is missing ${key}"
  done

  export APP_IMAGE
  export APP_RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE}"
}

compose() {
  docker compose \
    --project-directory "${PROJECT_ROOT}" \
    --env-file "${RUNTIME_ENV_FILE}" \
    -f "${COMPOSE_FILE}" \
    "$@"
}

wait_for_health() {
  local deadline=$((SECONDS + 600))
  local api_id frontend_id api_status frontend_status

  while (( SECONDS < deadline )); do
    api_id="$(compose ps -q api)"
    frontend_id="$(compose ps -q frontend)"
    api_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${api_id}" 2>/dev/null || true)"
    frontend_status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${frontend_id}" 2>/dev/null || true)"

    if [[ "${api_status}" == "healthy" && "${frontend_status}" == "healthy" ]]; then
      printf 'API and frontend are healthy.\n'
      return 0
    fi

    if [[ "${api_status}" == "unhealthy" || "${frontend_status}" == "unhealthy" ]]; then
      compose ps
      compose logs --tail 120 api frontend postgres
      fail "a service became unhealthy"
    fi

    sleep 10
  done

  compose ps
  compose logs --tail 120 api frontend postgres
  fail "services did not become healthy within 10 minutes"
}

usage() {
  cat <<'EOF'
Usage: sudo env AWS_REGION=<region> SSM_PARAMETER_PATH=<path> bash deploy/aws/manage.sh <command>

Commands:
  deploy   Build the shared CPU image, start services, and wait for health.
  build    Build only the shared CPU image.
  start    Start the existing image without rebuilding it, then wait for health.
  stop     Stop services while retaining containers and persistent volumes.
  status   Show container and image status.
  logs     Follow bounded application logs until interrupted.
  health   Check API and Streamlit from inside their containers.
EOF
}

main() {
  require_root
  require_command docker
  prepare_runtime_env

  case "${1:-}" in
    deploy)
      compose build api
      compose up -d --no-build
      wait_for_health
      compose ps
      ;;
    build)
      compose build api
      ;;
    start)
      compose up -d --no-build
      wait_for_health
      compose ps
      ;;
    stop)
      compose stop
      ;;
    status)
      compose ps
      docker image ls "${APP_IMAGE}"
      ;;
    logs)
      compose logs --follow --tail 150 api frontend postgres
      ;;
    health)
      compose exec -T api python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health', timeout=5).read().decode())"
      compose exec -T frontend python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5).read().decode())"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
