"""Behavioral coverage for compact Store diagnostics, without NPU work."""

import itertools
import queue
import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer, ttft_diagnostics
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_backend


class Recorder:
    def __init__(self, enabled=True):
        self.active = enabled
        self.fields = {}
        self.events = []
        self.ids = itertools.count()

    def enabled(self):
        return self.active

    def new_id(self, kind):
        return f"{kind}-{next(self.ids)}"

    def current(self):
        return dict(self.fields)

    @contextmanager
    def scope(self, **fields):
        previous = self.fields
        self.fields = {**previous, **fields}
        try:
            yield
        finally:
            self.fields = previous

    @contextmanager
    def span(self, event, **fields):
        try:
            yield fields
        except BaseException:
            self.emit(event, status="error", **fields)
            raise
        else:
            self.emit(event, status="ok", **fields)

    def emit(self, event, **fields):
        self.events.append({**self.fields, **fields, "event": event})


class TestStoreDiagnostics(unittest.TestCase):
    def setUp(self):
        self.diag = Recorder()
        for module in (kv_transfer, ttft_diagnostics, mooncake_backend):
            self.enterContext(patch.object(module, "diag", self.diag))

    def thread(self, sending=True):
        parent = kv_transfer.KVCacheStoreSendingThread if sending else kv_transfer.KVCacheStoreRecvingThread
        cls = type("UnrelatedWorker", (parent,), {})
        thread = object.__new__(cls)
        thread.tp_rank = 3
        thread.request_queue = queue.Queue()
        return thread

    def backend(self):
        backend = object.__new__(mooncake_backend.MooncakeBackend)
        backend.store = MagicMock()
        backend._store_initialized = True
        backend._lazy_init = False
        backend.config = SimpleNamespace(preferred_segment=True, prefer_alloc_in_same_node=True)
        backend.local_seg = "existing-segment"
        return backend

    def run_task(self, thread, action):
        result = []
        thread._handle_request = lambda payload: result.append(action())
        queued = ttft_diagnostics.capture(thread, SimpleNamespace(req_id="r"))
        ttft_diagnostics.execute(thread, queued)
        return result[0], self.diag.events[-1]

    def test_disabled_and_shutdown_preserve_payload_identity_without_capture(self):
        thread = self.thread()
        payload = object()
        self.diag.active = False
        with patch.object(ttft_diagnostics, "capture", side_effect=AssertionError("captured")):
            kv_transfer.KVTransferThread.add_request(thread, payload)
            self.assertIs(thread.request_queue.get_nowait(), payload)
            self.diag.active = True
            kv_transfer.KVTransferThread.add_request(thread, None)
            self.assertIsNone(thread.request_queue.get_nowait())
        self.assertEqual(self.diag.events, [])

    def test_snapshot_survives_mutation_and_multiple_directions(self):
        task_ids = []
        for sending in (True, False):
            thread = self.thread(sending)
            request = SimpleNamespace(req_id="request-before")
            task = SimpleNamespace(layer_id=2, shared_block_data=None, block_ranges=[SimpleNamespace(request=request)])
            payload = [task] if sending else SimpleNamespace(transfer_tasks=[task], layer_id=2)
            kv_transfer.KVTransferThread.add_request(thread, payload)
            queued = thread.request_queue.get_nowait()
            request.req_id = "request-after"
            task.layer_id = 9
            contexts = []
            thread._handle_request = lambda original, contexts=contexts: contexts.append(
                (original, dict(self.diag.fields))
            )
            ttft_diagnostics.execute(thread, queued)
            self.assertIs(contexts[0][0], payload)
            fields = contexts[0][1]
            self.assertEqual(fields["request_ids"], ("request-before",))
            self.assertEqual(fields["layer_ids"], (2,))
            self.assertEqual(fields["direction"], "save" if sending else "load")
            task_ids.append(fields["task_id"])
            self.assertEqual(self.diag.fields, {})
        self.assertNotEqual(*task_ids)

    def test_worker_exception_is_not_suppressed_and_context_is_restored(self):
        thread = self.thread()

        def fail(payload):
            ttft_diagnostics.decision("all_hit", count=5)
            raise ValueError("worker failed")

        thread._handle_request = fail
        with self.diag.scope(execution_id="execution-1", request_ids=("outer",)):
            queued = ttft_diagnostics.capture(thread, SimpleNamespace(req_id="r"))
            with self.assertRaisesRegex(ValueError, "worker failed"):
                ttft_diagnostics.execute(thread, queued)
            self.assertEqual(self.diag.fields, {"execution_id": "execution-1", "request_ids": ("outer",)})
        self.assertEqual(self.diag.fields, {})
        summary = self.diag.events[-1]
        self.assertEqual(summary["status"], "error")
        self.assertEqual(summary["error_type"], "ValueError")
        self.assertEqual(summary["all_hit_key_count"], 5)
        self.assertEqual(summary["execution_id"], "execution-1")
        self.assertEqual(summary["task_id"], dict(queued.fields)["task_id"])
        _, empty = self.run_task(thread, lambda: None)
        self.assertNotIn("all_hit_key_count", empty)

    def test_lookup_error_preserves_all_missing_fallback(self):
        thread = self.thread()
        thread.m_store = MagicMock()
        thread.m_store.exists.side_effect = OSError("query failed")
        self.assertEqual(kv_transfer.KVTransferThread.lookup(thread, ["a", "b"]), [False, False])
        self.assertEqual(self.diag.events[-1]["fallback"], "all_missing")
        result, task = self.run_task(thread, lambda: kv_transfer.KVTransferThread.lookup(thread, ["a", "b"]))
        self.assertEqual(result, [False, False])
        self.assertEqual(task["lookup_exception_count"], 1)
        self.assertNotIn("lookup_miss_count", task)
        self.assertNotIn("lookup_hit_count", task)
        thread.m_store.exists.side_effect = None
        thread.m_store.exists.return_value = [1, 1]
        before = len(self.diag.events)
        self.assertEqual(kv_transfer.KVTransferThread.lookup(thread, ["a", "b"]), [True, True])
        self.assertEqual(len(self.diag.events), before)
        result, task = self.run_task(thread, lambda: kv_transfer.KVTransferThread.lookup(thread, ["a", "b"]))
        self.assertEqual(result, [True, True])
        self.assertEqual(task["lookup_hit_count"], 2)

    def test_lookup_reports_negative_states_without_reclassifying_as_misses(self):
        thread = self.thread()
        thread.m_store = MagicMock()
        thread.m_store.exists.return_value = [-1, 1, 0]
        result, task = self.run_task(thread, lambda: kv_transfer.KVTransferThread.lookup(thread, ["a", "b", "c"]))
        self.assertEqual(result, [False, True, False])
        self.assertEqual(task["lookup_key_count"], 3)
        self.assertEqual(task["lookup_returned_count"], 3)
        self.assertEqual(task["lookup_hit_count"], 1)
        self.assertEqual(task["lookup_miss_count"], 1)
        self.assertEqual(task["lookup_error_count"], 1)
        error = next(event for event in self.diag.events if event["event"] == "store_lookup")
        self.assertEqual(error["status"], "error")

    def test_lookup_short_result_preserves_baseline_false_padding_without_claiming_misses(self):
        thread = self.thread()
        thread.m_store = MagicMock()
        thread.m_store.exists.return_value = [1]
        result, task = self.run_task(thread, lambda: kv_transfer.KVTransferThread.lookup(thread, ["a", "b", "c"]))
        self.assertEqual(result, [True, False, False])
        self.assertEqual(task["lookup_key_count"], 3)
        self.assertEqual(task["lookup_returned_count"], 1)
        self.assertEqual(task["lookup_hit_count"], 1)
        self.assertEqual(task["lookup_miss_count"], 0)
        self.assertEqual(task["lookup_error_count"], 0)

    def test_disabled_put_keeps_original_call_and_caught_error(self):
        backend = self.backend()
        self.diag.active = False
        backend.store.batch_put_from_multi_buffers.side_effect = OSError("original failure")
        self.assertIsNone(backend.put(["k"], [[123]], [[16]]))
        backend.store.batch_put_from_multi_buffers.assert_called_once()
        backend.store.batch_put_from_multi_buffers_with_trace.assert_not_called()
        self.assertEqual(self.diag.events, [])

    def test_put_uses_original_call_once_and_preserves_failure_outcome(self):
        # The traced put API was removed: diagnostics must record the same
        # original batch_put_from_multi_buffers call without altering it.
        backend = self.backend()
        backend.store.batch_put_from_multi_buffers.return_value = [0, -1]
        keys, addrs, sizes = ["k1", "k2"], [[1], [2]], [[8], [8]]
        backend.put(keys, addrs, sizes)
        call = backend.store.batch_put_from_multi_buffers.call_args
        self.assertEqual(backend.store.batch_put_from_multi_buffers.call_count, 1)
        self.assertIs(call.args[0], keys)
        self.assertIs(call.args[1], addrs)
        self.assertIs(call.args[2], sizes)
        self.assertTrue(call.args[3].prefer_alloc_in_same_node)
        result = next(event for event in self.diag.events if event["event"] == "store_backend_put_result")
        self.assertEqual(result["failed_count"], 1)
        completion = next(event for event in self.diag.events if event["event"] == "store_backend_put")
        self.assertEqual(completion["count"], 2)
        self.assertEqual(completion["bytes"], 16)
        self.assertEqual(completion["returned_count"], 2)
        self.assertEqual(completion["failed_count"], 1)
        self.assertNotIn("trace_id", result)
        self.assertNotIn("k1", repr(self.diag.events))

    def test_successful_put_emits_one_completion_without_result_event(self):
        backend = self.backend()
        backend.store.batch_put_from_multi_buffers.return_value = [0]
        backend.put(["k"], [[1]], [[8]])
        backend.store.batch_put_from_multi_buffers.assert_called_once()
        self.assertEqual([event["event"] for event in self.diag.events], ["store_backend_put"])
        self.assertEqual(self.diag.events[0]["returned_count"], 1)
        self.assertEqual(self.diag.events[0]["failed_count"], 0)

    def test_task_aggregates_groups_and_resets_without_detached_success_events(self):
        thread = self.thread()
        thread.m_store = MagicMock()
        thread.m_store.exists.return_value = [1, 0]

        def action():
            for group_id in range(3):
                ttft_diagnostics.decision("candidate_keys", group_id=group_id, count=2)
                kv_transfer.KVTransferThread.lookup(thread, ["a", "b"])
                ttft_diagnostics.submitted([[8, 16]], 1)

        _, task = self.run_task(thread, action)
        self.assertEqual(len(self.diag.events), 1)
        self.assertEqual(task["candidate_key_count"], 6)
        self.assertEqual(task["lookup_count"], 3)
        self.assertEqual(task["lookup_hit_count"], 3)
        self.assertEqual(task["lookup_miss_count"], 3)
        self.assertEqual(task["submitted_key_count"], 3)
        self.assertEqual(task["submitted_bytes"], 72)
        self.assertEqual(task["queue_wait_ns"], task["dequeued_ns"] - task["enqueued_ns"])
        self.assertNotIn("group_id", task)
        before = len(self.diag.events)
        ttft_diagnostics.decision("all_hit", count=999)
        ttft_diagnostics.submitted([[999]], 1)
        self.assertEqual(len(self.diag.events), before)
        _, empty = self.run_task(thread, lambda: None)
        self.assertFalse(any(key.startswith(("lookup_", "submitted_", "check_")) for key in empty))

    def test_native_put_failures_preserve_submission_and_identity(self):
        thread = self.thread()
        backend = self.backend()
        backend.store.batch_put_from_multi_buffers.side_effect = [[0, -1], OSError("native failure")]

        def action():
            ttft_diagnostics.decision("candidate_keys", count=2)
            ttft_diagnostics.decision("all_hit", count=1)
            for _ in range(2):
                ttft_diagnostics.submitted([[8], [16]], 2)
                backend.put(["a", "b"], [[1], [2]], [[8], [16]])

        with self.diag.scope(execution_id="execution-2"):
            _, summary = self.run_task(thread, action)
            self.assertEqual(self.diag.fields, {"execution_id": "execution-2"})
        self.assertEqual(backend.store.batch_put_from_multi_buffers.call_count, 2)
        self.assertEqual(summary["candidate_key_count"], 2)
        self.assertEqual(summary["all_hit_key_count"], 1)
        self.assertEqual(summary["submitted_key_count"], 4)
        self.assertEqual(summary["submitted_bytes"], 48)
        puts = [event for event in self.diag.events if event["event"] == "store_backend_put"]
        self.assertEqual(len(puts), 2)
        self.assertEqual(puts[0]["failed_count"], 1)
        self.assertEqual(puts[1]["status"], "error")
        for event in self.diag.events:
            self.assertEqual(event["task_id"], summary["task_id"])
            self.assertEqual(event["execution_id"], "execution-2")
            self.assertEqual(event["tp_rank"], 3)
        _, empty = self.run_task(thread, lambda: None)
        self.assertNotIn("failure_count", empty)

    def test_get_reports_not_executed_and_suppressed_exception(self):
        backend = self.backend()
        backend._lazy_init = True
        backend._store_initialized = False
        self.assertIsNone(backend.get(["k"], [[123]], [[16]]))
        backend.store.batch_get_into_multi_buffers.assert_not_called()
        self.assertEqual(self.diag.events[-1]["status"], "not_executed")
        backend._store_initialized = True
        backend.store.batch_get_into_multi_buffers.side_effect = OSError("native failure")
        self.assertIsNone(backend.get(["k"], [[123]], [[16]]))
        self.assertTrue(self.diag.events[-1]["suppressed"])
        self.assertTrue(self.diag.events[-1]["executed"])

    def test_lazy_init_succeeds_without_native_trace_probe(self):
        # Startup no longer probes optional traced-put capability; lazy init
        # must succeed with the plain Mooncake store.
        backend = self.backend()
        backend._store_initialized = False
        backend._lazy_init = True
        backend._use_fabric_mem = True
        backend._contribute_memory = False
        backend._store_init_lock = threading.Lock()
        backend.config = SimpleNamespace(
            enable_ssd_offload=False,
            tenant_id="default",
            protocol="ascend",
            device_name="",
            metadata_server="metadata",
            master_server_address="master",
        )
        store = MagicMock()
        store.setup.return_value = 0
        with (
            patch("mooncake.store.MooncakeDistributedStore", return_value=store),
            patch.object(mooncake_backend, "get_ip", return_value="host"),
        ):
            backend.ensure_initialized()
        self.assertTrue(backend._store_initialized)
