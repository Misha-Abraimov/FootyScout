from pathlib import Path

from analytics.artifact_paths import REPOSITORY_ROOT, repository_relative_path


def test_repository_artifact_paths_are_portable() -> None:
    path = REPOSITORY_ROOT / "data" / "processed" / "passes.parquet"
    assert repository_relative_path(path) == "data/processed/passes.parquet"


def test_external_artifact_paths_are_left_as_supplied(tmp_path: Path) -> None:
    path = tmp_path / "external-output.json"
    assert repository_relative_path(path) == path.as_posix()
