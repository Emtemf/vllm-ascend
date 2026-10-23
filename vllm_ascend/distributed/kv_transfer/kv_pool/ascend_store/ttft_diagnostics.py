"""Opt-in Store diagnostics; payloads and transfer bookkeeping stay caller-owned."""

import time
from dataclasses import dataclass
from typing import Any

from vllm.v1.metrics import ttft_diagnostics as diag


@dataclass(frozen=True)
class QueuedDiagnostic:
    payload: Any
    fields: tuple[tuple[str, Any], ...]
    enqueued_ns: int


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
        ("tp_rank", thread.tp_rank),
        ("layer_ids", tuple(dict.fromkeys(layer_ids))),
        ("direction", getattr(thread, "diagnostic_direction", "unknown")),
    )
    enqueued_ns = time.monotonic_ns()
    with diag.scope(**dict(fields)):
        diag.emit("store_enqueue")
    return QueuedDiagnostic(payload, fields, enqueued_ns)


def execute(thread: Any, queued: QueuedDiagnostic) -> None:
    with diag.scope(**dict(queued.fields)):
        diag.emit("store_dequeue", queue_wait_ns=time.monotonic_ns() - queued.enqueued_ns)
        with diag.span("store_task"):
            thread._handle_request(queued.payload)


def decision(reason: str, *, direction: str = "save", **fields: Any) -> None:
    if diag.enabled():
        diag.emit("store_check", direction=direction, reason=reason, **fields)


def submitted(sizes: Any, count: int, *, direction: str = "save", operation: str = "put") -> None:
    if diag.enabled():
        byte_count = sum(sum(size) if isinstance(size, (list, tuple)) else int(size) for size in sizes)
        diag.emit("store_submit", direction=direction, operation=operation, count=count, bytes=byte_count)
