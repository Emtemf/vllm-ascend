"""Host-only, task-scoped observations for the existing Pool transfer work."""

from __future__ import annotations

import contextvars
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any

from vllm.v1.metrics import pd_diagnostics as diag

MAX_MEMBERS = 128
MAX_NATIVE_CALLS = 64
MAX_COMPLETIONS = 128
_RETAINED_STAGES = frozenset(("build_key", "lookup", "prepare", "sync", "put"))
_ID_FIELDS = (
    "execution_id",
    "observation_id",
    "remote_request_id",
    "remote_engine_id",
    "dp_rank",
    "pp_rank",
    "pcp_rank",
    "dcp_rank",
)
_active: contextvars.ContextVar[Task | None] = contextvars.ContextVar("pool_diagnostic_task", default=None)


@dataclass(frozen=True)
class Snapshot:
    fields: tuple[tuple[str, Any], ...]
    members: tuple[tuple[tuple[str, Any], ...], ...]
    queued_ns: int | None
    members_dropped: int = 0


def capture(request: Any, operation: str, tp_rank: int | None = None, *, queued: bool = True) -> Snapshot:
    context = diag.current()
    fields = {name: context.get(name) for name in _ID_FIELDS}
    fields["request_id"] = None
    fields.update(task_id=diag.new_id("pool"), operation=operation)
    if tp_rank is not None:
        fields["tp_rank"] = tp_rank
    members: dict[tuple[Any, ...], tuple[tuple[str, Any], ...]] = {}
    members_dropped = 0

    def visit(item: Any) -> None:
        nonlocal members_dropped
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "req_id"):
            identity = {"request_id": item.req_id, "execution_id": getattr(item, "pd_execution_id", None)}
            for name in ("observation_id", "request_generation"):
                value = getattr(item, "pd_" + name, None)
                if value is None:
                    value = getattr(item, name, None)
                if value is not None:
                    identity[name] = value
            job = getattr(item, "store_job_id", None)
            if job is not None and job != -1:
                identity["store_job_id"] = job
            frozen = tuple(identity.items())
            key = tuple(identity.values())
            if key in members or len(members) < MAX_MEMBERS:
                members[key] = frozen
            else:
                members_dropped += 1
        elif hasattr(item, "block_ranges"):
            for block_range in item.block_ranges:
                visit(block_range.request)
        elif hasattr(item, "transfer_tasks"):
            visit(item.transfer_tasks)
        elif hasattr(item, "requests"):
            visit(item.requests)

    visit(request)
    if len(members) == 1 and not members_dropped:
        fields.update(dict(next(iter(members.values()))))
    elif members or members_dropped:
        fields["execution_id"] = None
    return Snapshot(
        tuple(fields.items()), tuple(members.values()), time.monotonic_ns() if queued else None, members_dropped
    )


class Task:
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.started_ns = time.monotonic_ns()
        self.counters: dict[str, int] = defaultdict(int)
        self.stages: dict[str, dict[str, Any]] = {}
        self.associated_stages: dict[str, dict[str, Any]] = {}
        self.members = dict.fromkeys(snapshot.members)
        self.members_dropped = snapshot.members_dropped
        self.native_calls: list[dict[str, Any]] = []
        self.native_calls_dropped = 0
        self.completions: list[dict[str, Any]] = []
        self.completions_dropped = 0
        self.status = "unknown"
        self.finished = False

    def merge_members(self, snapshot: Snapshot) -> None:
        if snapshot is self.snapshot:
            return
        self.members_dropped += snapshot.members_dropped
        for member in snapshot.members:
            if member in self.members or len(self.members) < MAX_MEMBERS:
                self.members[member] = None
            else:
                self.members_dropped += 1

    def finish(self, incomplete: bool = False) -> None:
        if self.finished:
            return
        self.finished = True
        ended = time.monotonic_ns()
        fields = dict(self.snapshot.fields)
        if self.status == "unknown":
            submitted = sum(value for key, value in self.counters.items() if key.endswith("_submitted_keys"))
            failures = sum(
                value
                for key, value in self.counters.items()
                if key.endswith("_failed_keys") or key.endswith("_exceptions") or key.endswith("_batch_failed")
            )
            missing = sum(
                value
                for key, value in self.counters.items()
                if key.endswith("_unknown_result_keys")
                or key.endswith("_unreported_keys")
                or key.endswith("_extra_results")
            )
            if failures:
                self.status = "backend_failure"
            elif submitted:
                self.status = "unknown_result" if missing else "succeeded"
            elif self.counters.get("all_hit_groups"):
                self.status = "all_hit"
            elif "candidate_keys" in self.counters and self.counters["candidate_keys"] == 0:
                self.status = "no_task"
        if len(self.members) > 1 or self.members_dropped:
            fields["request_id"] = None
            executions = {dict(member).get("execution_id") for member in self.members}
            fields["execution_id"] = (
                next(iter(executions)) if len(executions) == 1 and not self.members_dropped else None
            )
            fields.pop("store_job_id", None)
            for name in ("observation_id", "request_generation"):
                values = {dict(member).get(name) for member in self.members}
                fields[name] = next(iter(values)) if len(values) == 1 and not self.members_dropped else None
        incomplete |= bool(self.members_dropped or self.native_calls_dropped or self.completions_dropped)
        for name in _RETAINED_STAGES:
            if name not in self.stages:
                self.stages[name] = {
                    "calls": 0,
                    "status": ("skipped" if self.status in ("skipped", "no_task") else "unmeasured"),
                    "exceptions": 0,
                }
        diag.emit(
            "pool_task",
            **fields,
            scope="task",
            start_ns=self.started_ns,
            end_ns=ended,
            duration_ns=ended - self.started_ns,
            queue_enter_ns=self.snapshot.queued_ns,
            status=self.status,
            incomplete=incomplete,
            complete=not incomplete,
            member_count=len(self.members),
            members=[dict(member) for member in self.members],
            members_limit=MAX_MEMBERS,
            members_dropped=self.members_dropped,
            members_count_semantics="retained_unique; dropped_occurrences",
            native_calls=self.native_calls,
            native_calls_limit=MAX_NATIVE_CALLS,
            native_call_count=len(self.native_calls) + self.native_calls_dropped,
            native_calls_dropped=self.native_calls_dropped,
            completions=self.completions,
            completions_limit=MAX_COMPLETIONS,
            completions_dropped=self.completions_dropped,
            counters=dict(self.counters),
            stages=self.stages,
            associated_stages=self.associated_stages,
            timing_scope="task_processing_envelope",
            timing_domain="host_monotonic",
        )


@contextmanager
def processing(snapshot: Snapshot, task: Task | None = None, final: bool = True):
    started = time.monotonic_ns()
    if task is None:
        task = Task(snapshot)
    task.merge_members(snapshot)
    token = _active.set(task)
    fields = {"request_id": None, "execution_id": None, **dict(task.snapshot.fields)}
    if snapshot.queued_ns is not None:
        task.counters["queue_entries"] += 1
        task.counters["queue_wait_ns"] += started - snapshot.queued_ns
    # Never inherit a later execution from the thread dispatching deferred work.
    try:
        with diag.scope(**fields):
            try:
                with stage("processing"):
                    yield task
            except BaseException:
                task.status = "exception"
                final = True
                raise
            finally:
                if final:
                    task.finish()
    finally:
        _active.reset(token)


def add(**counts: int) -> None:
    task = _active.get()
    if task is not None:
        for key, value in counts.items():
            task.counters[key] += value


def outcome(status: str) -> None:
    task = _active.get()
    if task is not None:
        task.status = status


@contextmanager
def stage(name: str):
    task = _active.get()
    if task is None:
        yield
        return
    started = time.monotonic_ns()
    failed = False
    try:
        yield
    except StopIteration:
        raise
    except BaseException:
        failed = True
        raise
    finally:
        ended = time.monotonic_ns()
        stages = task.stages if name in _RETAINED_STAGES else task.associated_stages
        metric = stages.setdefault(
            name, {"calls": 0, "duration_ns": 0, "exceptions": 0, "start_ns": started, "end_ns": ended, "status": "ok"}
        )
        metric["calls"] += 1
        metric["duration_ns"] += ended - started
        metric["end_ns"] = ended
        metric["exceptions"] += int(failed)
        if failed:
            metric["status"] = "exception"


def timed(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def timed_iterator(name: str):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            iterator = iter(function(*args, **kwargs))
            while True:
                try:
                    with stage(name):
                        value = next(iterator)
                except StopIteration:
                    return
                yield value

        return wrapped

    return decorate


class BackendCall:
    def __init__(self, operation: str, key_count: int, sizes: Any, mode: str):
        self.operation = operation
        self.key_count = key_count
        self.byte_sizes = (
            None if sizes is None else [sum(size) if isinstance(size, (list, tuple)) else int(size) for size in sizes]
        )
        self.mode = mode
        self.recorded = False
        self.result_status = "unknown_result"

    def result(self, values: Any) -> None:
        if values is None:
            return
        if isinstance(values, bool) and self.mode == "boolean":
            add(
                **{
                    self.operation + ("_succeeded_keys" if values else "_failed_keys"): self.key_count,
                    self.operation + "_batch_result": 1,
                }
            )
            if self.byte_sizes is not None:
                add(**{self.operation + ("_succeeded_bytes" if values else "_failed_bytes"): sum(self.byte_sizes)})
            self.recorded = True
            self.result_status = "succeeded" if values else "failed"
            return
        if not isinstance(values, (list, tuple)):
            return
        succeeded = 0
        failed = 0
        succeeded_bytes = 0
        failed_bytes = 0
        for index, value in enumerate(values):
            if index >= self.key_count:
                break
            ok = bool(value) if self.mode == "boolean" else (value >= 0 if self.mode == "nonnegative" else value == 0)
            succeeded += int(ok)
            failed += int(not ok)
            if self.byte_sizes is not None and index < len(self.byte_sizes):
                if ok:
                    succeeded_bytes += self.byte_sizes[index]
                else:
                    failed_bytes += self.byte_sizes[index]
        prefix = self.operation
        add(
            **{
                prefix + "_result_count": len(values),
                prefix + "_succeeded_keys": succeeded,
                prefix + "_failed_keys": failed,
                prefix + "_unreported_keys": max(0, self.key_count - len(values)),
                prefix + "_extra_results": max(0, len(values) - self.key_count),
            }
        )
        if self.byte_sizes is not None:
            add(**{prefix + "_succeeded_bytes": succeeded_bytes, prefix + "_failed_bytes": failed_bytes})
        self.recorded = True
        self.result_status = (
            "failed" if failed else ("unknown_result" if len(values) != self.key_count else "succeeded")
        )

    def batch_result(self, value: int) -> None:
        add(
            **{
                self.operation + "_batch_succeeded": int(value == 0),
                self.operation + "_batch_failed": int(value != 0),
                self.operation + "_unknown_result_keys": self.key_count,
            }
        )
        self.recorded = True
        self.result_status = "failed" if value != 0 else "unknown_result"


@contextmanager
def backend(operation: str, key_count: int, sizes: Any = None, mode: str = "zero"):
    if _active.get() is None:
        with processing(capture(None, operation)), backend(operation, key_count, sizes, mode) as call:
            yield call
        return
    call = BackendCall(operation, key_count, sizes, mode)
    counts = {operation + "_calls": 1, operation + "_submitted_keys": key_count}
    if call.byte_sizes is not None:
        counts[operation + "_submitted_bytes"] = sum(call.byte_sizes)
    add(**counts)
    task = _active.get()
    start_ns = time.monotonic_ns()
    start_wall_ns = time.time_ns()
    native_tid = threading.get_native_id()
    failed = False
    try:
        with stage("put" if operation in ("put", "batch_put_from_multi_buffer_ranges") else "backend_" + operation):
            yield call
    except BaseException:
        failed = True
        add(**{operation + "_exceptions": 1})
        raise
    finally:
        end_ns = time.monotonic_ns()
        end_wall_ns = time.time_ns()
        if not call.recorded:
            add(**{operation + "_unknown_result_keys": key_count})
        # put/get are Python wrappers; native evidence belongs to the actual
        # Mooncake call nested inside them, never to a duplicate outer range.
        if operation not in ("put", "get"):
            if len(task.native_calls) < MAX_NATIVE_CALLS:
                local = time.localtime(start_wall_ns // 1_000_000_000)
                offset = getattr(local, "tm_gmtoff", -(time.altzone if local.tm_isdst > 0 else time.timezone))
                task.native_calls.append(
                    {
                        "call_id": diag.new_id("pool_call"),
                        "operation": operation,
                        "native_tid": native_tid,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ns": end_ns - start_ns,
                        "start_wall_ns": start_wall_ns,
                        "end_wall_ns": end_wall_ns,
                        "wall_utc_offset_seconds": offset,
                        "submitted_keys": key_count,
                        "submitted_bytes": counts.get(operation + "_submitted_bytes"),
                        "status": "exception" if failed else call.result_status,
                        "result_recorded": call.recorded,
                    }
                )
            else:
                task.native_calls_dropped += 1


def completion(**fields: Any) -> None:
    """Record bookkeeping completion separately from backend success."""
    task = _active.get()
    if fields.get("store_job_id") == -1:
        fields.pop("store_job_id")
    fields["completion_semantics"] = "bookkeeping_only"
    if task is not None:
        if len(task.completions) < MAX_COMPLETIONS:
            task.completions.append(fields)
        else:
            task.completions_dropped += 1
    else:
        diag.emit(
            "pool_task",
            task_id=diag.new_id("pool"),
            operation="store_completion",
            scope="bookkeeping",
            status="observed",
            complete=True,
            incomplete=False,
            members=[
                {
                    key: fields[key]
                    for key in ("request_id", "execution_id", "store_job_id", "observation_id", "request_generation")
                    if key in fields
                }
            ],
            completions=[fields],
            **fields,
        )


def task(operation: str):
    """Scope synchronous worker work; queued work is scoped by the dispatcher."""

    def decorate(function):
        self_name, request_name = tuple(signature(function).parameters)[:2]

        @wraps(function)
        def wrapped(*args, **kwargs):
            instance = args[0] if args else kwargs.get(self_name)
            request = args[1] if len(args) > 1 else kwargs.get(request_name)
            snapshot = capture(request, operation, getattr(instance, "tp_rank", None), queued=False)
            with processing(snapshot):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def put(store: Any, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
    with backend("put", len(keys), sizes, mode="boolean") as call:
        result = store.put(keys, addrs, sizes)
        call.result(result)
        return result


def get(store: Any, keys: list[str], addrs: list[list[int]], sizes: list[list[int]]):
    with backend("get", len(keys), sizes) as call:
        result = store.get(keys, addrs, sizes)
        call.result(result)
        return result
