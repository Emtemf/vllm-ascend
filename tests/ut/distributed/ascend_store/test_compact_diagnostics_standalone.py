"""Direct stdlib producer tests, without importing torch or NPU packages.

Set VLLM_COMPACT_ROOT if the matching vLLM checkout is not the sibling
vllm-pd-ttft-compact-032 directory. Compile actual receive methods from source
AST to isolate hardware imports; only transport/device boundaries are fakes.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import logging
import os
import queue
import sys
import threading
import time
import types
import unittest
from collections import OrderedDict, defaultdict
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
VLLM_ROOT = Path(os.environ.get("VLLM_COMPACT_ROOT", ROOT.parent / "vllm-pd-ttft-compact-032"))


def load_source(name, path, injected=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if injected:
        with patch.dict(sys.modules, injected):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


diag = load_source("compact_contract_helper", VLLM_ROOT / "vllm/v1/metrics/pd_diagnostics.py")
metrics = types.ModuleType("vllm.v1.metrics")
metrics.pd_diagnostics = diag
pool = load_source(
    "compact_pool_producer",
    ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/diagnostics.py",
    {"vllm": types.ModuleType("vllm"), "vllm.v1": types.ModuleType("vllm.v1"), "vllm.v1.metrics": metrics},
)
source_path = ROOT / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py"
source = ast.parse(source_path.read_text(encoding="utf-8"))
method_names = {
    "add_request",
    "_mark_request_task_submitted",
    "_mark_request_task_done",
    "_handle_request",
    "_pd_finish_task",
    "_transfer_batch",
    "_transfer_kv_cache_all_groups",
}
selected = []
for node in source.body:
    if (
        isinstance(node, ast.FunctionDef)
        and node.name == "_pd_transfer_stage"
        or isinstance(node, ast.ClassDef)
        and node.name == "KVCacheTaskTracker"
    ):
        selected.append(node)
    elif isinstance(node, ast.ClassDef) and node.name == "KVCacheRecvingThread":
        node.bases = []
        node.body = [
            method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in method_names
        ]
        selected.append(node)
namespace = dict(
    diag=diag,
    time=time,
    threading=threading,
    contextlib=contextlib,
    Any=Any,
    CancelledError=CancelledError,
    OrderedDict=OrderedDict,
    BlockIds=list,
    RemotePortInfo=dict,
    MAX_TRANSFER_NATIVE_CALLS=64,
    MambaSpec=type("MambaSpec", (), {}),
    logger=logging.getLogger("compact_receive_test"),
)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
        str(source_path),
        "exec",
        flags=__import__("__future__").annotations.compiler_flag,
    ),
    namespace,
)
Receiver = namespace["KVCacheRecvingThread"]


class CompactProducerTests(unittest.TestCase):
    def records(self, captured):
        records = [json.loads(r.getMessage().split("[PD_DIAG] ", 1)[1]) for r in captured.records]
        for record in records:
            self.assertEqual(record["version"], 2)
            for key in ("host", "pid", "native_tid", "run_id", "timestamp_ns", "record_id"):
                self.assertIn(key, record)
        return records

    def receiver(self):
        receiver = Receiver()
        receiver.tp_rank = 1
        receiver.local_engine_id = "D-engine"
        receiver.pd_connector_id = "connector"
        receiver.vllm_config = types.SimpleNamespace(parallel_config=types.SimpleNamespace(data_parallel_rank=0))
        receiver.pd_request_tasks = {}
        receiver.request_task_counts_lock = threading.Lock()
        receiver.request_task_counts = defaultdict(int)
        receiver.finished_request_markers = set()
        receiver.request_queue = queue.Queue()
        receiver.task_tracker = namespace["KVCacheTaskTracker"]()
        receiver.proc_not_transfer_request = {}
        receiver.proc_not_transfer_request_lock = threading.Lock()
        receiver.use_hybrid = True
        receiver._send_done_signal_to_free_remote_port = lambda *args: None
        receiver._send_done_recv_signal = lambda *args: None
        return receiver

    def enqueue(self, receiver, request="D", remote="P", final=True):
        with diag.scope(execution_id="scheduled", observation_id="observation"):
            receiver.add_request(request, remote, [[1]], [[2]], "P-engine", "peer", 31000, 0, 1, all_task_done=final)
        task = receiver.request_queue.get_nowait()
        receiver._mark_request_task_submitted(task)
        receiver.task_tracker.add_req_to_process(request)
        return task

    def test_pool_bounds_frozen_members_and_native_provenance(self):
        requests = [
            types.SimpleNamespace(
                req_id=str(i), pd_execution_id="E", pd_observation_id="O", request_generation="G", store_job_id=-1
            )
            for i in range(130)
        ]
        snapshot = pool.capture(requests, "save", queued=False)
        requests[0].pd_execution_id = "later"
        with self.assertLogs(diag._LOG, level="INFO") as captured, pool.processing(snapshot) as task:
            for _ in range(66):
                with pool.backend("batch_put_from_multi_buffer_ranges", 1, [[8]], "nonnegative") as call:
                    call.result([8])
            with pool.stage("save_wait"):
                pass
            task.finish()
        (record,) = self.records(captured)
        self.assertEqual(record["event"], "pool_task")
        self.assertEqual(len(record["members"]), 128)
        self.assertEqual(record["members_dropped"], 2)
        self.assertEqual(
            record["members"][0], dict(request_id="0", execution_id="E", observation_id="O", request_generation="G")
        )
        self.assertEqual(len(record["native_calls"]), 64)
        self.assertEqual(record["native_calls_dropped"], 2)
        self.assertEqual(record["native_call_count"], 66)
        self.assertTrue(record["incomplete"])
        self.assertFalse(record["complete"])
        self.assertEqual(set(record["stages"]), {"build_key", "lookup", "prepare", "sync", "put"})
        self.assertEqual(record["stages"]["put"]["calls"], 66)
        self.assertIn("save_wait", record["associated_stages"])
        for call in record["native_calls"]:
            self.assertEqual(call["native_tid"], threading.get_native_id())
            self.assertLessEqual(call["start_wall_ns"], call["end_wall_ns"])
            self.assertIsInstance(call["wall_utc_offset_seconds"], int)
            self.assertEqual(call["submitted_keys"], 1)
            self.assertEqual(call["submitted_bytes"], 8)

    def test_pool_zero_repetition_exception_and_member_merge(self):
        first = pool.capture(types.SimpleNamespace(req_id="A", pd_execution_id="E"), "save", queued=False)
        second = pool.capture(types.SimpleNamespace(req_id="B", pd_execution_id="E"), "save", queued=False)
        failure = RuntimeError("original")
        with (
            self.assertLogs(diag._LOG, level="INFO") as captured,
            patch.object(pool.time, "monotonic_ns", return_value=100),
        ):
            with pool.processing(first, final=False) as task:
                for _ in range(2):
                    with pool.stage("build_key"):
                        pass
            with (
                self.assertRaises(RuntimeError) as raised,
                pool.processing(second, task, final=False),
                pool.stage("prepare"),
            ):
                raise failure
            task.finish()
        self.assertIs(raised.exception, failure)
        (record,) = self.records(captured)
        self.assertEqual({m["request_id"] for m in record["members"]}, {"A", "B"})
        self.assertIsNone(record["request_id"])
        self.assertEqual(record["stages"]["build_key"]["calls"], 2)
        self.assertEqual(record["stages"]["build_key"]["duration_ns"], 0)
        self.assertEqual(record["stages"]["prepare"]["exceptions"], 1)
        self.assertEqual(record["status"], "exception")
        self.assertNotIn("duration_ns", record["stages"]["put"])

    def test_receive_actual_read_sum_generation_and_truncation(self):
        receiver = self.receiver()
        first = self.enqueue(receiver, remote="P-a", final=False)
        second = self.enqueue(receiver, remote="P-b")

        def read(*args):
            return 0

        receiver.engine = types.SimpleNamespace(batch_transfer_sync_read=read)

        def transfer(meta):
            for _ in range(3):
                receiver._transfer_batch(meta, "peer:1", [1], [2], [8], time.perf_counter())

        receiver._transfer_kv_cache_all_groups = transfer
        with self.assertLogs(diag._LOG, level="INFO") as captured, patch.dict(namespace, MAX_TRANSFER_NATIVE_CALLS=2):
            receiver._handle_request(first)
            receiver._handle_request(second)
        records = self.records(captured)
        self.assertEqual([r["scope"] for r in records], ["task", "task", "request_generation"])
        self.assertEqual(len({r["request_generation"] for r in records}), 1)
        for record in records[:2]:
            self.assertEqual(record["stages"]["wait_for_transfer"]["calls"], 3)
            self.assertEqual(record["native_calls_dropped"], 1)
            self.assertEqual(record["submitted_bytes"], 24)
            self.assertLessEqual(record["stages"]["wait_for_transfer"]["duration_ns"], record["transfer_envelope_ns"])
            self.assertFalse(record["complete"])
            self.assertEqual(record["execution_id"], "scheduled")
            self.assertEqual(record["links"][0]["kind"], "producer")
        self.assertEqual(records[2]["native_calls_dropped"], 2)
        self.assertFalse(records[2]["complete"])
        self.assertIsNone(records[2]["execution_id"])
        self.assertFalse(receiver.pd_request_tasks)
        self.assertEqual(receiver.request_queue.unfinished_tasks, 0)

    def test_metadata_remains_auxiliary_and_read_retains_transfer_timing(self):
        receiver = self.receiver()
        receiver._prefill_pp_size = 1
        receiver.hma_group_size = 1
        receiver.kv_cache_specs = [namespace["MambaSpec"]()]
        receiver.local_handshake_port = 32000
        receiver.remote_metadata_lock = threading.Lock()
        receiver.kv_caches_base_addr = {"D-engine": {32000: [100]}}
        receiver.remote_te_port = {}
        receiver.block_len_per_addr = [8]
        receiver.block_stride_per_addr = [8]
        receiver.addr_group_idx = [[0]]

        def metadata(*args):
            receiver.kv_caches_base_addr["P-engine"] = {31000: [200]}
            receiver.remote_te_port["P-engine"] = {31000: 7777}

        receiver._get_remote_metadata = metadata
        receiver.engine = types.SimpleNamespace(batch_transfer_sync_read=lambda *args: 0)
        # Mamba's single-block grouping is an external array helper, outside
        # diagnostic timing. This boundary supplies its original one-block form.
        with patch.dict(namespace, group_concurrent_contiguous=lambda a, b: ([a], [b])):
            for cached in (False, True):
                task = self.enqueue(receiver)
                with self.assertLogs(diag._LOG, level="INFO") as captured:
                    receiver._handle_request(task)
                record, generation = self.records(captured)
                self.assertEqual(record["status"], "succeeded")
                self.assertNotIn("submit", record["stages"])
                self.assertEqual("metadata_request" in record["associated_stages"], not cached)
                self.assertEqual(record["stages"]["wait_for_transfer"]["calls"], 1)
                self.assertEqual(record["transfer_envelope_ns"], record["native_calls"][0]["duration_ns"])
                self.assertEqual(generation["scope"], "request_generation")

    def test_read_duration_is_call_sum_instead_of_envelope(self):
        receiver = self.receiver()
        task = self.enqueue(receiver)
        receiver.engine = types.SimpleNamespace(batch_transfer_sync_read=lambda *args: 0)
        with patch.object(time, "monotonic_ns", side_effect=[100, 110, 1000, 1020]):
            receiver._transfer_batch(task, "peer:1", [1], [2], [8], 0)
            receiver._transfer_batch(task, "peer:1", [1], [2], [8], 0)
        receiver._transfer_kv_cache_all_groups = lambda meta: None
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            receiver._handle_request(task)
        record, generation = self.records(captured)
        self.assertEqual(record["stages"]["wait_for_transfer"]["duration_ns"], 30)
        self.assertEqual(record["stages"]["wait_for_transfer"]["calls"], 2)
        self.assertEqual(record["transfer_envelope_ns"], 920)
        self.assertEqual(generation["transfer_envelope_ns"], 920)
        self.assertEqual([c["duration_ns"] for c in record["native_calls"]], [10, 20])

    def test_receive_cancelled_read_preserves_exception_identity_and_closes_task(self):
        receiver = self.receiver()
        task = self.enqueue(receiver)
        failure = CancelledError("cancelled")

        def read(*args):
            raise failure

        receiver.engine = types.SimpleNamespace(batch_transfer_sync_read=read)
        with self.assertRaises(CancelledError) as raised:
            receiver._transfer_batch(task, "peer:1", [1], [2], [8], time.perf_counter())
        self.assertIs(raised.exception, failure)
        self.assertEqual(task["_pd_diag"]["native_calls"][0]["status"], "cancelled")
        receiver._transfer_kv_cache_all_groups = lambda meta: receiver._transfer_batch(
            meta, "peer:1", [1], [2], [8], time.perf_counter()
        )
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            receiver._handle_request(task)
        (record,) = [r for r in self.records(captured) if r["scope"] == "task"]
        self.assertEqual(record["stages"]["wait_for_transfer"]["status"], "cancelled")
        self.assertEqual(record["failed_bytes"], 16)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["stages"]["wait_for_transfer"]["calls"], 2)
        self.assertEqual(record["native_calls"][0]["error_type"], "CancelledError")

    def test_actual_skipped_receive_has_no_invented_measurement(self):
        receiver = self.receiver()
        receiver._prefill_pp_size = 1
        task = self.enqueue(receiver)
        task["local_block_ids"] = [[]]
        with self.assertLogs(diag._LOG, level="INFO") as captured:
            receiver._handle_request(task)
        record, generation = self.records(captured)
        self.assertEqual(record["status"], "skipped")
        self.assertNotIn("submit", record["stages"])
        self.assertEqual(record["stages"]["wait_for_transfer"]["status"], "skipped")
        self.assertIsNone(record["stages"]["wait_for_transfer"].get("duration_ns"))
        self.assertEqual(record["native_calls"], [])
        self.assertIsNone(generation["transfer_envelope_ns"])


if __name__ == "__main__":
    unittest.main()
