# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU contracts for main adaptation; no hardware claims."""

import ast
import importlib.util
import logging
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "vllm_ascend/distributed/kv_transfer"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def method(path, cls, name, namespace):
    """Execute a production method unchanged with explicit dependency fixtures."""
    tree = ast.parse(path.read_text())
    node = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls)
    fn = next(f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == name)
    unit = ast.Module(body=[fn], type_ignores=[])
    exec(
        compile(
            ast.fix_missing_locations(unit), str(path), "exec", flags=__import__("__future__").annotations.compiler_flag
        ),
        namespace,
    )
    return namespace[name]


class MainContracts(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        self.ai = load(ROOT / "vllm_ascend/ai_qos.py", "vllm_ascend.ai_qos")
        self.policy = load(
            BASE / "kv_pool/ascend_store/qos.py", "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos"
        )
        self.layer = load(BASE / "kv_p2p/layerwise_qos.py", "qos_test_layer")
        self.life = load(BASE / "qos_lifecycle.py", "qos_test_life")
        self.cfg = dict(priority_to_qos={0: 0, 3: 3, 7: 7}, level_names=True)
        self.sender = NS(
            qos_policy=self.policy.KvQosPolicy.from_config(self.cfg),
            qos_pool=NS(write=Mock(return_value=0)),
            qos_error=None,
            qos_terminal_reqs=set(),
            failed_reqs=set(),
            total_layers=3,
            layer_metadata={"a": NS(tensor_group_idx=[0]), "b": NS(tensor_group_idx=[0])},
            callback_func=Mock(),
            reuse_completion_callback=Mock(),
            send_queue=queue.Queue(),
            _transfer_kv_cache=Mock(return_value=None),
        )

    def task(self, idx=2, finish=True, empty=False):
        return NS(
            layer_idx=idx,
            layer_name="a",
            layer_names=["a"],
            failed_requests={},
            send_request={}
            if empty
            else {
                "r": NS(
                    kv_priority=7,
                    chunk_finish=finish,
                    remote_qos_te_rpc_ports={0: 20001, 3: 20003, 7: 20007},
                    remote_host="peer",
                )
            },
        )

    def run_task(self, t):
        self.sender.send_queue.put(t)
        actual = self.sender.send_queue.get()
        # Call the actual connector entrypoint, not only the helper.
        fn = method(
            BASE / "kv_p2p/mooncake_layerwise_connector.py",
            "KVCacheSendingLayerThread",
            "_handle_request",
            {"layerwise_qos": self.layer, "logger": logging.getLogger(__name__)},
        )
        fn(self.sender, actual)
        self.assertEqual(self.sender.send_queue.unfinished_tasks, 0)

    def test_explicit_static_conflict_even_zero_none_bool(self):
        for val in [0, 3, None, False]:
            with self.subTest(val=val), self.assertRaisesRegex(ValueError, "conflict"):
                self.policy.validate_request_qos_policy({"kv_qos": self.cfg, "qos_priority": val})
        self.assertIsNone(self.policy.validate_request_qos_policy({"qos_priority": 3}))

    def test_reserved_names_and_manual_fail(self):
        for config in [
            {"collective_communication": {}},
            {"collective": {}},
            {"op_submit": {"manual": {}}},
            {"kv_transfer": {"manual": {}}},
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.ai.validate_ai_qos(config)

    def test_default_does_not_validate_supplied_value_but_validates_object(self):
        p = self.policy.KvQosPolicy.from_config(dict(self.cfg, request_priority=False, default_priority=3))
        for value in [None, {}, True, "not-a-level", 7]:
            self.assertEqual(p.resolve_priority({"kv_priority": value}), 3)
        with self.assertRaises(ValueError):
            p.resolve_priority("bad")

    def test_layerwise_multicomponent_groups_and_descriptors(self):
        t = self.task()
        t.layer_names = ["a", "b"]
        t.send_request["s"] = NS(**vars(t.send_request["r"]))
        t.send_request["s"].kv_priority = 3
        self.sender.get_transfer_meta = lambda task, rid, meta, layer, group: (
            [100 if layer == "a" else 110],
            [200 if layer == "a" else 210],
            [10],
        )
        batches = self.layer.group_batches(self.sender, t, 0)
        self.assertEqual(set(batches), {(7, "peer:20007"), (3, "peer:20003")})
        for b in batches.values():
            self.assertEqual((b.src, b.dst, b.length), ([100, 110], [200, 210], [10, 10]))
        self.layer.send_batches(self.sender, t, batches, 0)
        self.assertEqual(self.sender.qos_pool.write.call_count, 2)
        self.sender.callback_func.assert_not_called()

    def test_empty_batch_no_write_but_last_layer_still_notifies(self):
        t = self.task()
        self.sender.get_transfer_meta = lambda *args: ([], [], [])
        self.layer.send_batches(self.sender, t, self.layer.group_batches(self.sender, t, 0), 0)
        self.sender.qos_pool.write.assert_not_called()
        self.run_task(t)
        self.sender.callback_func.assert_called_once_with("r", t.send_request["r"], 0, trans_flag=True)
        self.sender.reuse_completion_callback.assert_called_once_with(2, None)

    def test_intermediate_and_incomplete_chunk_do_not_notify(self):
        self.run_task(self.task(idx=0))
        self.run_task(self.task(finish=False))
        self.sender.callback_func.assert_not_called()
        self.assertEqual(self.sender.reuse_completion_callback.call_count, 2)

    def test_early_failure_one_terminal_until_last_layer(self):
        self.sender._transfer_kv_cache.side_effect = RuntimeError("write failed")
        for idx in [0, 1, 2]:
            self.run_task(self.task(idx))
        self.assertEqual(self.sender.callback_func.call_count, 1)
        self.assertFalse(self.sender.callback_func.call_args.kwargs["trans_flag"])
        self.assertEqual(self.sender._transfer_kv_cache.call_count, 1)
        self.assertFalse(self.sender.qos_terminal_reqs)
        for call in self.sender.reuse_completion_callback.call_args_list:
            self.assertIn("write failed", call.args[1])

    def test_notification_exception_does_not_leak_queue_or_reuse(self):
        self.sender.callback_func.side_effect = RuntimeError("ack failed")
        self.run_task(self.task())
        self.assertIn("ack failed", self.sender.qos_error)
        self.assertIn("ack failed", self.sender.reuse_completion_callback.call_args.args[1])

    def test_reuse_callback_exception_visible_to_drain(self):
        self.sender.reuse_completion_callback.side_effect = RuntimeError("gate failed")
        self.run_task(self.task())
        with self.assertRaisesRegex(RuntimeError, "gate failed"):
            self.layer.drain(self.sender)

    def test_failed_metadata_only_notifies_without_write(self):
        t = self.task(empty=True)
        t.failed_requests = self.task().send_request
        self.run_task(t)
        self.assertFalse(self.sender.callback_func.call_args.kwargs["trans_flag"])
        self.assertIsNotNone(self.sender.qos_error)

    def test_missing_lane_or_mixed_group_rejected(self):
        t = self.task()
        t.send_request["r"].remote_qos_te_rpc_ports = {0: 20001}
        with self.assertRaisesRegex(ValueError, "lane"):
            self.layer.group_batches(self.sender, t, 0)
        t = self.task()
        t.layer_names = ["a", "b"]
        self.sender.layer_metadata["b"].tensor_group_idx = [1]
        self.sender.get_transfer_meta = lambda *args: ([], [], [])
        with self.assertRaises(AssertionError):
            self.layer.group_batches(self.sender, t, 0)
        self.sender.qos_pool.write.assert_not_called()

    def test_drains_dead_or_stuck_sender_are_bounded(self):
        self.sender.send_queue.put(self.task())
        self.sender.is_alive = lambda: False
        with self.assertRaisesRegex(RuntimeError, "exited"):
            self.layer.drain(self.sender, timeout=0)
        self.sender.is_alive = lambda: True
        with self.assertRaises(TimeoutError):
            self.layer.drain(self.sender, timeout=0)
        with self.assertRaises(TimeoutError):
            self.life.drain_queue(self.sender, self.sender.send_queue, timeout=0)

    def test_stop_queue_counts_sentinel_once(self):
        q = queue.Queue()

        def consume():
            while True:
                x = q.get()
                q.task_done()
                if x is None:
                    return

        t = threading.Thread(target=consume)
        t.start()
        q.put(1)
        self.life.stop_queue(t, q, timeout=1)
        self.assertFalse(t.is_alive())
        self.assertEqual(q.unfinished_tasks, 0)

    def test_pd_port_validation_before_native_read(self):
        fn = method(
            BASE / "kv_p2p/mooncake_connector.py",
            "KVCacheRecvingThread",
            "_submit_kv_read",
            {"logger": logging.getLogger(__name__)},
        )
        obj = NS(
            qos_policy=self.sender.qos_policy,
            remote_metadata_lock=threading.Lock(),
            remote_qos_te_ports={"e": {5000: {0: 20001, 3: 20003, 7: 20007}}},
            qos_pool=NS(read=Mock(return_value=0)),
        )
        req = dict(kv_priority=7, remote_engine_id="e", remote_handshake_port=5000, remote_host="h")
        self.assertEqual(fn(obj, req, "unused", [100], [200], [10]), 0)
        obj.qos_pool.read.assert_called_once_with(7, "h:20007", [100], [200], [10])
        for ports in [{0: 20001, 3: 20003}, {0: 20001, 3: 20001, 7: 20007}, {0: True, 3: 20003, 7: 20007}]:
            obj.remote_qos_te_ports["e"][5000] = ports
            with self.assertRaises(RuntimeError):
                fn(obj, req, "unused", [100], [200], [10])
        self.assertEqual(obj.qos_pool.read.call_count, 1)

    def test_layerwise_peer_metadata_strict(self):
        for ports, version in [
            ({0: 1, 3: 2, 7: 3}, True),
            ({0: 1, 3: 1, 7: 3}, 1),
            ({0: 1, 7: 3}, 1),
            ({0: 1, 3: 2, 7: False}, 1),
        ]:
            with self.assertRaises(ValueError):
                self.layer.peer_ports(self.sender.qos_policy, NS(qos_te_rpc_ports=ports, layerwise_qos_version=version))

    def test_no_qos_layerwise_actual_handler_protocol_preserved(self):
        self.sender.qos_pool = None
        self.run_task(self.task())
        self.sender.callback_func.assert_called_once()
        self.sender.reuse_completion_callback.assert_called_once_with(2, None)


if __name__ == "__main__":
    unittest.main()
