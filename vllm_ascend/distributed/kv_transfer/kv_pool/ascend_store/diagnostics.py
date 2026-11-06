"""Host-only, task-scoped observations for the existing Pool transfer work."""
from __future__ import annotations

import contextvars
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from typing import Any

from vllm.v1.metrics import pd_diagnostics as diag

_ID_FIELDS = ("execution_id", "observation_id", "remote_request_id", "remote_engine_id", "dp_rank", "pp_rank", "pcp_rank", "dcp_rank")
_active: contextvars.ContextVar[Task | None] = contextvars.ContextVar("pool_diagnostic_task", default=None)


@dataclass(frozen=True)
class Snapshot:
    fields: tuple[tuple[str, Any], ...]
    members: tuple[tuple[tuple[str, Any], ...], ...]
    queued_ns: int | None


def capture(request: Any, operation: str, tp_rank: int | None = None, *, queued: bool = True) -> Snapshot:
    context = diag.current()
    fields = {name: context.get(name) for name in _ID_FIELDS}
    fields["request_id"] = None
    fields.update(task_id=diag.new_id("pool"), operation=operation)
    if tp_rank is not None:
        fields["tp_rank"] = tp_rank
    members: dict[tuple[Any, ...], tuple[tuple[str, Any], ...]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "req_id"):
            identity = {"request_id": item.req_id, "execution_id": getattr(item, "pd_execution_id", None)}
            job = getattr(item, "store_job_id", None)
            if job is not None:
                identity["store_job_id"] = job
            frozen = tuple(identity.items())
            members[tuple(identity.values())] = frozen
        elif hasattr(item, "block_ranges"):
            for block_range in item.block_ranges:
                visit(block_range.request)
        elif hasattr(item, "transfer_tasks"):
            visit(item.transfer_tasks)
        elif hasattr(item, "requests"):
            visit(item.requests)

    visit(request)
    if len(members) == 1:
        fields.update(dict(next(iter(members.values()))))
    elif members:
        fields["execution_id"] = None
    return Snapshot(tuple(fields.items()), tuple(members.values()), time.monotonic_ns() if queued else None)


class Task:
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.started_ns = time.monotonic_ns()
        self.counters: dict[str, int] = defaultdict(int)
        self.stages: dict[str, dict[str, int]] = {}
        self.members = dict.fromkeys(snapshot.members)
        self.status = "unknown"

    def finish(self, incomplete: bool = False) -> None:
        ended = time.monotonic_ns()
        fields = dict(self.snapshot.fields)
        if self.status == "unknown":
            submitted = sum(value for key, value in self.counters.items() if key.endswith("_submitted_keys"))
            failures = sum(value for key, value in self.counters.items() if key.endswith("_failed_keys") or key.endswith("_exceptions") or key.endswith("_batch_failed"))
            missing = sum(value for key, value in self.counters.items() if key.endswith("_unknown_result_keys") or key.endswith("_unreported_keys") or key.endswith("_extra_results"))
            if failures:
                self.status = "backend_failure"
            elif submitted:
                self.status = "unknown_result" if missing else "succeeded"
            elif self.counters.get("all_hit_groups"):
                self.status = "all_hit"
            elif "candidate_keys" in self.counters and self.counters["candidate_keys"] == 0:
                self.status = "no_task"
        if len(self.members) > 1:
            fields["request_id"] = None
            executions = {dict(member).get("execution_id") for member in self.members}
            fields["execution_id"] = next(iter(executions)) if len(executions) == 1 else None
            fields["request_ids"] = [dict(member)["request_id"] for member in self.members]
            fields.pop("store_job_id", None)
        diag.emit("pool_task", **fields, start_ns=self.started_ns, end_ns=ended,
                  duration_ns=ended - self.started_ns, queue_enter_ns=self.snapshot.queued_ns,
                  status=self.status, incomplete=incomplete, member_count=len(self.members),
                  counters=dict(self.counters), stages=self.stages, timing_domain="host_monotonic")
        if len(self.members) > 1:
            for member in self.members:
                diag.emit("pool_task_member", task_id=fields["task_id"],
                          operation=fields["operation"], **dict(member))


@contextmanager
def processing(snapshot: Snapshot, task: Task | None = None, final: bool = True):
    started = time.monotonic_ns()
    if task is None:
        task = Task(snapshot)
    task.members.update(dict.fromkeys(snapshot.members))
    token = _active.set(task)
    fields = {"request_id": None, "execution_id": None, **dict(task.snapshot.fields)}
    if snapshot.queued_ns is not None:
        task.counters["queue_entries"] += 1
        task.counters["queue_wait_ns"] += started - snapshot.queued_ns
    # Never inherit a later execution from the thread dispatching deferred work.
    try:
        with diag.scope(**fields):
            try:
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
        metric = task.stages.setdefault(name, {"calls": 0, "duration_ns": 0, "exceptions": 0,
                                               "start_ns": started, "end_ns": ended})
        metric["calls"] += 1
        metric["duration_ns"] += ended - started
        metric["end_ns"] = ended
        metric["exceptions"] += int(failed)


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
        self.byte_sizes = None if sizes is None else [sum(size) if isinstance(size, (list, tuple)) else int(size) for size in sizes]
        self.mode = mode
        self.recorded = False

    def result(self, values: Any) -> None:
        if values is None:
            return
        if isinstance(values, bool) and self.mode == "boolean":
            add(**{self.operation + ("_succeeded_keys" if values else "_failed_keys"): self.key_count,
                   self.operation + "_batch_result": 1})
            if self.byte_sizes is not None:
                add(**{self.operation + ("_succeeded_bytes" if values else "_failed_bytes"): sum(self.byte_sizes)})
            self.recorded = True
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
        add(**{prefix + "_result_count": len(values), prefix + "_succeeded_keys": succeeded,
               prefix + "_failed_keys": failed,
               prefix + "_unreported_keys": max(0, self.key_count - len(values)),
               prefix + "_extra_results": max(0, len(values) - self.key_count)})
        if self.byte_sizes is not None:
            add(**{prefix + "_succeeded_bytes": succeeded_bytes, prefix + "_failed_bytes": failed_bytes})
        self.recorded = True

    def batch_result(self, value: int) -> None:
        add(**{self.operation + "_batch_succeeded": int(value == 0),
               self.operation + "_batch_failed": int(value != 0),
               self.operation + "_unknown_result_keys": self.key_count})
        self.recorded = True


@contextmanager
def backend(operation: str, key_count: int, sizes: Any = None, mode: str = "zero"):
    if _active.get() is None:
        with processing(capture(None, operation)):
            with backend(operation, key_count, sizes, mode) as call:
                yield call
        return
    call = BackendCall(operation, key_count, sizes, mode)
    counts = {operation + "_calls": 1, operation + "_submitted_keys": key_count}
    if call.byte_sizes is not None:
        counts[operation + "_submitted_bytes"] = sum(call.byte_sizes)
    add(**counts)
    start_wall_ns = time.time_ns() if operation == "mooncake_put" else None
    try:
        with stage("backend_" + operation):
            yield call
    except BaseException:
        add(**{operation + "_exceptions": 1})
        raise
    finally:
        if not call.recorded:
            add(**{operation + "_unknown_result_keys": key_count})
        if start_wall_ns is not None:
            end_wall_ns = time.time_ns()
            local = time.localtime(start_wall_ns // 1_000_000_000)
            offset = getattr(local, "tm_gmtoff", -(time.altzone if local.tm_isdst > 0 else time.timezone))
            diag.emit(
                "pool_native_call",
                operation=operation,
                start_wall_ns=start_wall_ns,
                end_wall_ns=end_wall_ns,
                wall_utc_offset_seconds=offset,
                submitted_keys=key_count,
                submitted_bytes=counts.get(operation + "_submitted_bytes"),
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
