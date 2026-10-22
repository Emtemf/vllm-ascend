# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only store task handoffs; never owns transfer bookkeeping."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any

from vllm.v1.metrics import ttft_diagnostics as ttft


@dataclass(frozen=True)
class CapturedTask:
    payload: Any
    association: tuple[tuple[str, Any], ...]
    event: str
    queued_ns: int
    queue_span_id: str


@dataclass
class SaveCheck:
    checked_keys: int = 0
    existing_keys: int = 0
    missing_keys: int = 0
    lookup_errors: int = 0
    submitted_keys: int = 0
    submitted_bytes: int = 0
    reason: str | None = None
    failed: bool = False


_CHECK: ContextVar[SaveCheck | None] = ContextVar("ascend_store_save_check", default=None)
_OPERATION: ContextVar[dict[str, Any] | None] = ContextVar("ascend_store_operation", default=None)


def _requests(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, list):
        tasks = payload
    elif hasattr(payload, "transfer_tasks"):
        tasks = payload.transfer_tasks
    else:
        req_id = getattr(payload, "req_id", None)
        return (req_id,) if req_id is not None else ()
    return tuple(dict.fromkeys(br.request.req_id for task in tasks for br in task.block_ranges))


def capture(payload: Any, operation: str, rank: int) -> CapturedTask:
    # Only immutable identifiers cross the handoff, never mutable request metadata.
    context = ttft.current()
    request_ids = _requests(payload)
    context.update(task_id=ttft.new_id("task"), request_ids=request_ids,
                   request_id=request_ids[0] if len(request_ids) == 1 else None,
                   rank=rank)
    layer_id = getattr(payload, "layer_id", None)
    if isinstance(payload, list) and payload:
        layer_id = payload[0].layer_id
    if layer_id is not None:
        context["layer_id"] = layer_id
    # Shared worker contexts contain scalar IDs and may include a mutable list
    # of request IDs. Snapshot that list too, without retaining the worker list.
    association = tuple((key, tuple(value) if isinstance(value, list) else value)
                        for key, value in context.items())
    event = "store_" + operation
    queued_ns = time.perf_counter_ns()
    queue_span_id = ttft.new_id("span")
    with ttft.scope(**dict(association)):
        ttft.emit(event + "_queue_start", span_id=queue_span_id, start_ns=queued_ns)
    return CapturedTask(payload, association, event, queued_ns, queue_span_id)


@contextmanager
def execute(task: CapturedTask, *, queued: bool = True):
    with ttft.scope(**dict(task.association)):
        if queued:
            end_ns = time.perf_counter_ns()
            ttft.emit(task.event + "_queue", span_id=task.queue_span_id,
                      start_ns=task.queued_ns, end_ns=end_ns,
                      duration_ns=end_ns - task.queued_ns, status="ok")
        check = SaveCheck() if task.event == "store_save" else None
        token = _CHECK.set(check)
        try:
            with ttft.span(task.event + "_task", execution="thread" if queued else "sync",
                           unmeasured_subphases=["build_keys", "prepare", "device_wait", "result_bookkeeping"]):
                try:
                    yield task.payload
                except BaseException:
                    if check is not None:
                        check.failed = True
                    raise
        finally:
            if check is not None:
                reason = check.reason or (
                    "error" if check.failed else
                    "submitted" if check.submitted_keys else
                    "query_error" if check.lookup_errors else
                    "all_hit" if check.checked_keys and not check.missing_keys else
                    "not_submitted" if check.missing_keys else "no_eligible_keys"
                )
                ttft.emit("store_save_check", checked_keys=check.checked_keys,
                          existing_keys=check.existing_keys, missing_keys=check.missing_keys,
                          lookup_errors=check.lookup_errors, submitted_keys=check.submitted_keys,
                          submitted_bytes=check.submitted_bytes, reason=reason,
                          result="error" if check.failed else "not_inferred")
            _CHECK.reset(token)


@contextmanager
def request_load(request: Any, *, asynchronous: bool, rank: int):
    if not ttft.enabled():
        yield
        return
    with ttft.scope(request_id=request.req_id, request_ids=(request.req_id,), rank=rank):
        spec = request.load_spec
        ttft.emit("request_match", source="connector_load_spec",
                  local_tokens=spec.vllm_cached_tokens if spec is not None else None,
                  external_tokens=spec.kvpool_cached_tokens if spec is not None else None,
                  can_load=spec.can_load if spec is not None else None)
        if asynchronous:
            yield
        else:
            task = capture(request, "load", rank)
            # A synchronous operation has no thread queue, so do not invent a
            # zero queue duration. The capture start is explicitly not queued.
            ttft.emit("store_load_queue", span_id=task.queue_span_id,
                      start_ns=task.queued_ns, status="skipped", reason="synchronous")
            with execute(task, queued=False):
                yield


def lookup_result(values: list[int] | None, count: int, *, error: bool = False) -> None:
    check = _CHECK.get()
    errors = count if error else sum(value not in (0, 1) for value in values or ())
    existing = sum(value == 1 for value in values or ())
    if check is not None:
        check.checked_keys += count
        check.existing_keys += existing
        # Preserve the original lookup's non-1 -> False behavior, but report
        # negative/error states separately rather than labelling them misses.
        check.missing_keys += sum(value == 0 for value in values or ())
        check.lookup_errors += errors
    ttft.emit("store_lookup", checked_keys=count, existing_keys=existing,
              missing_keys=sum(value == 0 for value in values or ()),
              lookup_errors=errors, result="query_error" if errors else "ok")


def task_error(reason: str) -> None:
    check = _CHECK.get()
    if check is not None:
        check.failed = True
        check.reason = reason
    ttft.emit("store_task_result", status="error", reason=reason)


def task_skip(reason: str) -> None:
    check = _CHECK.get()
    if check is not None:
        check.reason = reason
    ttft.emit("store_task_result", status="skipped", reason=reason)


def save_skip(request: Any, rank: int) -> None:
    with ttft.scope(request_id=request.req_id, request_ids=(request.req_id,),
                    rank=rank, task_id=ttft.new_id("task")):
        ttft.emit("store_save_check", checked_keys=0, existing_keys=0,
                  missing_keys=0, lookup_errors=0, submitted_keys=0,
                  submitted_bytes=0, reason="can_save_false" if request.can_save is False else "can_save_none",
                  status="skipped")


def operation_result(result: str, **fields: Any) -> None:
    state = _OPERATION.get()
    if state is not None:
        state.update(result=result, **fields)
    if result in ("error", "exception", "not_executed"):
        check = _CHECK.get()
        if check is not None:
            check.failed = True


def backend_operation(operation: str):
    def decorate(method):
        @wraps(method)
        def wrapped(self, keys, addrs, sizes):
            if not ttft.enabled():
                return method(self, keys, addrs, sizes)
            trace_id = ttft.new_id("put" if operation == "put" else "get")
            logical_bytes = sum(sum(s) if isinstance(s, list) else s for s in sizes)
            state: dict[str, Any] = {"result": "not_reported"}
            token = _OPERATION.set(state)
            event = "store_" + operation
            with ttft.scope(trace_id=trace_id, keys=len(keys), bytes=logical_bytes,
                            backend=type(self).__name__):
                start = time.perf_counter_ns()
                ttft.emit(event + "_start", start_ns=start, span_id=trace_id)
                try:
                    result = method(self, keys, addrs, sizes)
                    return result
                except BaseException as exc:
                    operation_result("exception", error_type=type(exc).__name__)
                    raise
                finally:
                    end = time.perf_counter_ns()
                    ttft.emit(event, start_ns=start, end_ns=end, duration_ns=end - start,
                              span_id=trace_id, status="error" if state["result"] in
                              ("error", "exception", "not_executed") else "ok", **state)
                    _OPERATION.reset(token)
        return wrapped
    return decorate


def submitted(keys: list[str], sizes: list[list[int]]) -> None:
    check = _CHECK.get()
    if check is not None:
        check.submitted_keys += len(keys)
        check.submitted_bytes += sum(sum(s) if isinstance(s, list) else s for s in sizes)
