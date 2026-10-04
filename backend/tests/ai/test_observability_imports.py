"""Isolated import regressions for the observability dependency boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
SITE_PACKAGES = BACKEND_ROOT / ".venv" / "Lib" / "site-packages"


@pytest.mark.parametrize(
    "module_name",
    (
        "app.ai.synthesis",
        "app.ai.executor",
        "app.ai.observability.telemetry",
        "app.ai.evaluation.cases",
        "scripts.run_ai_scout_evals",
    ),
)
def test_production_modules_import_independently(module_name: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(BACKEND_ROOT),
            str(SITE_PACKAGES),
            environment.get("PYTHONPATH", ""),
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=BACKEND_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_telemetry_import_does_not_initialize_domain_or_trace_modules() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(BACKEND_ROOT), str(SITE_PACKAGES)))
    code = (
        "import sys; import app.ai.observability.telemetry; "
        "assert 'app.ai.observability.tracing' not in sys.modules; "
        "assert 'app.ai.synthesis' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
