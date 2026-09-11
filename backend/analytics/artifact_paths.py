"""Portable display paths for generated analytics metadata."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_relative_path(path: Path) -> str:
    """Return a repository-relative POSIX path when the artifact is inside FootyScout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
