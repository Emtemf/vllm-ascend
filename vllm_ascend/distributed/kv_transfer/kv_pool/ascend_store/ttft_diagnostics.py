"""Compact Store diagnostics; payloads and transfer bookkeeping stay caller-owned."""

import threading
import time
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from vllm.v1.metrics import ttft_diagnostics as diag


@dataclass(frozen=True)
class QueuedDiagnostic:
    payload: Any
    fields: tuple[tuple[str, Any], ...]
    enqueued_ns: int


_task_fields: ContextVar[dict[str, Any] | None] = ContextVar("store_task_fields", default=None)
_CHECK_REASONS = frozenset((
    "empty_copy", "request_not_tracked", "pooled_chunks_skipped", "store_mask_empty",
    "candidate_keys", "all_hit", "no_keys", "empty_tasks", "no_shared_blocks", "no_blocks",
))


def capture(thread: Any, payload: Any) -> QueuedDiagnostic:
    """Snapshot only immutable association fields at the actual queue boundary."""
    tasks = payload if isinstance(payload, list) else getattr(payload, "transfer_tasks", ())
    request_ids = []
    layer_ids = []
    if hasattr(payload, "req_id"):
        request_ids.append(payload.req_id)
    if hasattr(payload, "layer_id"):
        layer_ids.append(payload.layer_id)
    for task in tasks:
        layer_ids.append(task.layer_id)
        for block_range in task.block_ranges:
            request_ids.append(block_range.request.req_id)
        shared = task.shared_block_data
        if shared is not None:
            request_ids.extend(shared.req_ids)
    fields = (
        ("request_ids", tuple(dict.fromkeys(request_ids))),
        ("task_id", diag.new_id("store_task")),
        ("execution_id", getattr(payload, "execution_id", "")
         or diag.current().get("execution_id", "")),
        ("tp_rank", thread.tp_rank),
        ("layer_ids", tuple(dict.fromkeys(layer_ids))),
        ("direction", getattr(thread, "diagnostic_direction", "unknown")),
    )
    return QueuedDiagnostic(payload, fields, time.monotonic_ns())


def execute(thread: Any, queued: QueuedDiagnostic) -> None:
    """Aggregate bounded counters only while executing this queue-owned payload."""
    dequeued_ns = time.monotonic_ns()
    with (
        diag.scope(**dict(queued.fields), native_tid=threading.get_native_id()),
        diag.span(
            "store_task", enqueued_ns=queued.enqueued_ns, dequeued_ns=dequeued_ns,
            queue_wait_ns=dequeued_ns - queued.enqueued_ns,
        ) as fields,
    ):
        token = _task_fields.set(fields)
        try:
            thread._handle_request(queued.payload)
        finally:
            _task_fields.reset(token)


def _add(**counts: int) -> None:
    """Ignore diagnostics without a task and never interfere with transfer work."""
    try:
        fields = _task_fields.get()
        if fields is not None:
            for name, count in counts.items():
                fields[name] = fields.get(name, 0) + count
    except Exception:
        pass


def decision(reason: str, *, direction: str = "save", **fields: Any) -> None:
    """Count a fixed vocabulary of checks, never per-group labels or arrays."""
    if reason not in _CHECK_REASONS:
        return
    _add(**{f"check_{reason}_count": 1})
    if reason == "candidate_keys":
        _add(candidate_key_count=fields.get("count", 0))
    elif reason == "all_hit":
        _add(all_hit_key_count=fields.get("count", 0))
    elif reason == "pooled_chunks_skipped":
        _add(pooled_chunks_skipped_count=fields.get("count", 0))


def byte_count(sizes: Any) -> int | None:
    """Return diagnostic bytes when measurable, leaving unknown sizes missing."""
    try:
        return sum(sum(size) if hasattr(size, "__iter__") else int(size) for size in sizes)
    except Exception:
        return None


def submitted(sizes: Any, count: int, *, direction: str = "save", operation: str = "put") -> None:
    """Accumulate submitted volume without claiming successful persistence."""
    if not diag.enabled() or _task_fields.get() is None:
        return
    prefix = {"put": "submitted", "get": "get_submitted", "batch_copy": "copy_submitted"}.get(operation)
    if prefix is None:
        return
    _add(**{f"{prefix}_count": 1, f"{prefix}_key_count": count})
    size = byte_count(sizes)
    if size is not None:
        _add(**{f"{prefix}_bytes": size})


def lookup_result(keys: Any, result: Any = None, *, error: Exception | None = None) -> None:
    """Only observed lookups create lookup fields; errors retain explicit events."""
    try:
        if not diag.enabled():
            return
        _add(lookup_count=1, lookup_key_count=len(keys))
        if error is not None:
            _add(lookup_exception_count=1)
            diag.emit("store_lookup", status="error", count=len(keys),
                      error_type=type(error).__name__, fallback="all_missing")
            return
        errors = sum(value not in (0, 1) for value in result)
        _add(lookup_returned_count=len(result), lookup_hit_count=sum(value == 1 for value in result),
             lookup_miss_count=sum(value == 0 for value in result), lookup_error_count=errors)
        if errors:
            diag.emit("store_lookup", status="error", count=len(keys), error_count=errors,
                      fallback="invalid_results_missing")
    except Exception:
        pass


def backend_result(fields: Any, result: Any) -> None:
    """Attach observed native results without turning diagnostic errors into failures."""
    with suppress(Exception):
        fields.update(returned_count=len(result), failed_count=sum(value < 0 for value in result))


def failure(event: str, **fields: Any) -> None:
    """Keep exceptional outcomes visible in both the task and its error record."""
    _add(failure_count=1)
    if diag.enabled():
        diag.emit(event, **fields)
