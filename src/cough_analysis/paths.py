from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward from a start path."""
    current = Path.cwd() if start is None else Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        has_repo_marker = (candidate / ".git").exists() or (candidate / "pyproject.toml").exists()
        has_project_package = (candidate / "src" / "cough_analysis").exists()
        if has_repo_marker and has_project_package and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find project root containing pyproject.toml, src/cough_analysis, and data/."
    )


def project_path(*parts: str, root: str | Path | None = None) -> Path:
    """Build an absolute path inside the project root."""
    return find_project_root(root) / Path(*parts)
