import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_uses_locked_project_dependencies() -> None:
    dockerfile = _read("Dockerfile")

    assert "FROM python:3.12-slim" in dockerfile
    assert "COPY --chown=appuser:appuser pyproject.toml uv.lock ./" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "requirements.txt" not in dockerfile
    assert "USER appuser" in dockerfile
    assert dockerfile.index("USER appuser") < dockerfile.index("uv sync --frozen --no-dev")


def test_docker_context_excludes_local_and_sensitive_artifacts() -> None:
    dockerignore = set(_read(".dockerignore").splitlines())

    assert {".env", ".venv", ".git", "PDF/", "logs/"} <= dockerignore
    assert "!.env.example" in dockerignore


def test_index_entry_points_cannot_create_cosine_indexes() -> None:
    canonical = _read("scripts/create_hybrid_index.py")
    compatibility = _read("scripts/create_index.py")
    misspelled_compatibility = _read("scripts/create_hybird_index.py")

    assert 'metric="dotproduct"' in canonical
    assert 'metric="cosine"' not in canonical
    assert "from create_hybrid_index import main" in compatibility
    assert "from create_hybrid_index import main" in misspelled_compatibility


def test_windows_launcher_uses_project_root_and_virtual_environment() -> None:
    launcher = _read("scripts/run_server.ps1")

    assert "$projectRoot = Split-Path -Parent $PSScriptRoot" in launcher
    assert '.venv\\Scripts\\python.exe' in launcher
    assert '"agentic_rag.api.main:app"' in launcher
    assert '"src.agentic_rag.api.main:app"' not in launcher


def test_documented_setup_files_exist() -> None:
    assert (ROOT / ".env.example").is_file()
    assert (ROOT / "scripts/create_hybrid_index.py").is_file()
    assert (ROOT / "src/agentic_rag/policies/calibrate_retrieval.py").is_file()
    assert (ROOT / "uv.lock").is_file()


def test_aws_bootstrap_uses_pinned_official_cli_v2_bundle() -> None:
    bootstrap = _read("deploy/aws/bootstrap.sh")
    apt_packages = bootstrap.split("apt-get install", 1)[1].split(
        "expected_aws_version", 1
    )[0]

    assert re.search(r'^AWS_CLI_VERSION="\d+\.\d+\.\d+"$', bootstrap, re.MULTILINE)
    assert "awscli" not in apt_packages
    assert "curl" in apt_packages
    assert "unzip" in apt_packages
    assert (
        "https://awscli.amazonaws.com/"
        "awscli-exe-linux-x86_64-${AWS_CLI_VERSION}.zip"
    ) in bootstrap
    assert '--install-dir "${AWS_CLI_INSTALL_DIR}"' in bootstrap
    assert "aws_install_args+=(--update)" in bootstrap


def test_aws_compose_build_context_matches_runtime_project_directory() -> None:
    compose = _read("deploy/aws/compose.yml")
    manager = _read("deploy/aws/manage.sh")
    workflow = _read(".github/workflows/offline.yml")

    assert "      context: .\n" in compose
    assert '--project-directory "${PROJECT_ROOT}"' in manager
    assert '--project-directory "$GITHUB_WORKSPACE"' in workflow
    assert 'test -f "${resolved_build_context}/Dockerfile"' in workflow


def test_selected_v3_parser_dependencies_are_exactly_pinned() -> None:
    project = tomllib.loads(_read("pyproject.toml"))
    dependencies = set(project["project"]["dependencies"])

    assert {
        "pymupdf==1.28.2",
        "pymupdf4llm==1.28.2",
        "pymupdf-layout==1.28.2",
        "rapidocr==3.9.2",
    } <= dependencies


def test_gpu_runtime_provisions_triton_compiler_as_root_then_drops_privileges() -> None:
    dockerfile = _read("Dockerfile")
    gpu_runtime = dockerfile.split("FROM application-base AS gpu-runtime", 1)[1]

    root_index = gpu_runtime.index("USER root")
    compiler_index = gpu_runtime.index("gcc")
    appuser_index = gpu_runtime.index("USER appuser")
    cuda_install_index = gpu_runtime.index("uv pip install")
    source_copy_index = gpu_runtime.index("COPY --chown=appuser:appuser src/ src/")
    project_install_index = gpu_runtime.index("--no-deps --no-build-isolation .")

    assert "libc6-dev" in gpu_runtime
    assert root_index < compiler_index < appuser_index < cuda_install_index
    assert cuda_install_index < source_copy_index < project_install_index
    assert "uv sync" not in gpu_runtime
    assert "--mount=type=cache" not in gpu_runtime[source_copy_index:]
    assert "final_torch_cuda_build=" in gpu_runtime


def test_v3_assets_have_an_isolated_index_and_durable_api_volume() -> None:
    config = _read("src/agentic_rag/config.py")
    compose = _read("docker-compose.yml")
    template = _read(".env.example")
    dockerfile = _read("Dockerfile")

    assert 'v3_index_name: str = "agentic-rag-hybrid-v3"' in config
    assert 'v3_asset_root: Path = Path("/app/data/v3-assets")' in config
    assert "V3_INDEX_NAME=agentic-rag-hybrid-v3" in template
    assert "V3_ASSET_ROOT=/app/data/v3-assets" in template
    assert "v3_assets:/app/data/v3-assets" in compose
    assert "mkdir -p /app/data/v3-assets" in dockerfile
