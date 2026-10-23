"""Behavioral coverage for optional Store diagnostics, without NPU work."""

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
        self.emit(event + "_start", **fields)
        try:
            yield
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

    def backend(self, native_trace=True):
        backend = object.__new__(mooncake_backend.MooncakeBackend)
        backend.store = MagicMock()
        backend._store_initialized = True
        backend._lazy_init = False
        backend._native_trace_supported = native_trace
        backend.config = SimpleNamespace(preferred_segment=True, prefer_alloc_in_same_node=True)
        backend.local_seg = "existing-segment"
        return backend

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
            task = SimpleNamespace(layer_id=2, shared_block_data=None,
                                   block_ranges=[SimpleNamespace(request=request)])
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
        thread._handle_request = MagicMock(side_effect=ValueError("worker failed"))
        queued = ttft_diagnostics.capture(thread, SimpleNamespace(req_id="r"))
        with self.assertRaisesRegex(ValueError, "worker failed"):
            ttft_diagnostics.execute(thread, queued)
        self.assertEqual(self.diag.fields, {})
        self.assertEqual(self.diag.events[-1]["status"], "error")

    def test_lookup_error_preserves_all_missing_fallback(self):
        thread = self.thread()
        thread.m_store = MagicMock()
        thread.m_store.exists.side_effect = OSError("query failed")
        self.assertEqual(kv_transfer.KVTransferThread.lookup(thread, ["a", "b"]), [False, False])
        self.assertEqual(self.diag.events[-1]["fallback"], "all_missing")
        thread.m_store.exists.side_effect = None
        thread.m_store.exists.return_value = [1, 1]
        self.assertEqual(kv_transfer.KVTransferThread.lookup(thread, ["a", "b"]), [True, True])
        self.assertEqual(self.diag.events[-1]["hit_count"], 2)

    def test_lookup_reports_negative_states_without_reclassifying_as_misses(self):
        thread = self.thread()
        thread.m_store = MagicMock()
        thread.m_store.exists.return_value = [-1, 1, 0]
        self.assertEqual(kv_transfer.KVTransferThread.lookup(thread, ["a", "b", "c"]),
                         [False, True, False])
        result = self.diag.events[-1]
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["returned_count"], 3)
        self.assertEqual(result["hit_count"], 1)
        self.assertEqual(result["miss_count"], 1)
        self.assertEqual(result["error_count"], 1)

    def test_lookup_short_result_preserves_baseline_false_padding_without_claiming_misses(self):
        thread = self.thread()
        thread.m_store = MagicMock()
        thread.m_store.exists.return_value = [1]
        self.assertEqual(kv_transfer.KVTransferThread.lookup(thread, ["a", "b", "c"]),
                         [True, False, False])
        result = self.diag.events[-1]
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["hit_count"], 1)
        self.assertEqual(result["miss_count"], 0)
        self.assertEqual(result["error_count"], 0)

    def test_put_uses_one_traced_call_and_links_native_id(self):
        backend = self.backend()
        backend.store.batch_put_from_multi_buffers_with_trace.return_value = [-1]
        keys, addrs, sizes = ["private-key"], [[123]], [[16]]
        backend.put(keys, addrs, sizes)
        backend.store.batch_put_from_multi_buffers.assert_not_called()
        call = backend.store.batch_put_from_multi_buffers_with_trace.call_args
        self.assertEqual(backend.store.batch_put_from_multi_buffers_with_trace.call_count, 1)
        self.assertIs(call.args[0], keys)
        self.assertIs(call.args[1], addrs)
        self.assertIs(call.args[2], sizes)
        self.assertTrue(call.args[3].prefer_alloc_in_same_node)
        result = next(event for event in self.diag.events if event["event"] == "store_backend_put_result")
        self.assertEqual(call.args[4], result["trace_id"])
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["native_trace"], "linked")
        self.assertNotIn("private-key", repr(self.diag.events))

    def test_put_degrades_to_original_api_when_native_trace_unavailable(self):
        # Older/unrecompiled Mooncake (or a read-only D node) lacks the traced
        # put API: diagnostics stay on, put uses the original call, and records
        # are tagged native_trace=unavailable so offline analysis does not try
        # to join native BatchPutTiming lines on a trace_id that never existed.
        backend = self.backend(native_trace=False)
        backend.store.batch_put_from_multi_buffers.return_value = [0, -1]
        keys, addrs, sizes = ["k1", "k2"], [[1], [2]], [[8], [8]]
        backend.put(keys, addrs, sizes)
        backend.store.batch_put_from_multi_buffers_with_trace.assert_not_called()
        call = backend.store.batch_put_from_multi_buffers.call_args
        self.assertEqual(backend.store.batch_put_from_multi_buffers.call_count, 1)
        self.assertIs(call.args[0], keys)
        result = next(event for event in self.diag.events if event["event"] == "store_backend_put_result")
        self.assertEqual(result["native_trace"], "unavailable")
        self.assertEqual(result["failed_count"], 1)
        self.assertNotIn("trace_id", result)

    def test_disabled_put_keeps_original_call_and_caught_error(self):
        backend = self.backend()
        self.diag.active = False
        backend.store.batch_put_from_multi_buffers.side_effect = OSError("original failure")
        self.assertIsNone(backend.put(["k"], [[123]], [[16]]))
        backend.store.batch_put_from_multi_buffers.assert_called_once()
        backend.store.batch_put_from_multi_buffers_with_trace.assert_not_called()
        self.assertEqual(self.diag.events, [])

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

    def test_lazy_init_degrades_when_native_trace_unavailable(self):
        # Diagnostics on + a Mooncake client without the traced put API must NOT
        # fail startup (that broke read-only D nodes / unrecompiled Mooncake).
        # It initializes normally and records the degraded capability flag.
        backend = self.backend(native_trace=True)  # flag is recomputed by _setup_store
        backend._store_initialized = False
        backend._lazy_init = True
        backend._use_fabric_mem = True
        backend._contribute_memory = False
        backend._store_init_lock = threading.Lock()
        backend.config = SimpleNamespace(enable_ssd_offload=False, tenant_id="default", protocol="ascend",
                                         device_name="", metadata_server="metadata", master_server_address="master")
        store = MagicMock()
        store.setup.return_value = 0
        store.supports_batch_put_trace.return_value = False
        with (
            patch("mooncake.store.MooncakeDistributedStore", return_value=store),
            patch.object(mooncake_backend, "get_ip", return_value="host"),
        ):
            backend.ensure_initialized()
        self.assertTrue(backend._store_initialized)
        self.assertFalse(backend._native_trace_supported)
