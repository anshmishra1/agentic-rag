from pathlib import Path


ROOT = Path(__file__).parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_uses_locked_project_dependencies() -> None:
    dockerfile = _read("Dockerfile")

    assert "FROM python:3.12-slim" in dockerfile
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "requirements.txt" not in dockerfile
    assert "USER appuser" in dockerfile


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
