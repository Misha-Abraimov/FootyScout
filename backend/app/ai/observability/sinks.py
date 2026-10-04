"""Provider-neutral destinations for completed AI Scout traces."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Protocol

from app.ai.diagnostics import sanitize_error_message
from app.ai.observability.models import AIScoutRunTrace

logger = logging.getLogger(__name__)


class TraceSink(Protocol):
    def record(self, trace: AIScoutRunTrace) -> None:
        """Persist or retain one completed trace."""


class NullTraceSink:
    def record(self, trace: AIScoutRunTrace) -> None:
        del trace


class InMemoryTraceSink:
    def __init__(self) -> None:
        self.traces: list[AIScoutRunTrace] = []

    def record(self, trace: AIScoutRunTrace) -> None:
        self.traces.append(trace)


class JsonlTraceSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def record(self, trace: AIScoutRunTrace) -> None:
        line = trace.model_dump_json(exclude_none=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(f"{line}\n")


def record_trace_safely(sink: TraceSink, trace: AIScoutRunTrace) -> bool:
    """Isolate trace-persistence failures from the completed product response."""
    try:
        sink.record(trace)
    except Exception as exc:  # noqa: BLE001 - the sink is an isolated I/O boundary
        logger.warning(
            "AI Scout observability error trace_write_error (%s): %s",
            type(exc).__name__,
            sanitize_error_message(str(exc)),
        )
        return False
    return True
