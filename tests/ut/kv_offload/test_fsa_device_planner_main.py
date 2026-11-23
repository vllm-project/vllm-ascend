# SPDX-License-Identifier: Apache-2.0
"""Exercise the main planner integration on CPU with mocked NPU tasks."""

import __future__

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

ROOT = Path(__file__).resolve().parents[3]
MANAGER = "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py"
GRAPH = "vllm_ascend/worker/v2/aclgraph_utils.py"


def load_methods(file, class_name, names, namespace):
    # Execute the production methods without importing the platform and CANN.
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    code = compile(ast.Module(body=methods, type_ignores=[]), file, "exec", __future__.annotations.compiler_flag)
    exec(code, namespace)
    return namespace


class Stream:
    def __init__(self, trace):
        self.trace = trace
        self.npu_stream = 17

    def record_event(self):
        self.trace.append("record")
        return object()

    def wait_event(self, _):
        self.trace.append("wait input")

    def wait_stream(self, _):
        self.trace.append("wait KV")


class TestFsaPlannerMain(unittest.TestCase):
    def make_submitter(self, rank=0, publish_status=0, notify=True):
        trace = []
        compute = Stream(trace)
        plan = Stream(trace)
        ctx = SimpleNamespace(_fsa_plan_ipc_join=None)
        npu = SimpleNamespace(current_stream=lambda: compute, stream=lambda _: contextlib.nullcontext())
        namespace = load_methods(
            MANAGER,
            "SparseKVOffloadManager",
            {"_submit_fused_device_plan"},
            {"torch_npu": SimpleNamespace(npu=npu), "get_forward_context": lambda: ctx},
        )
        ipc = SimpleNamespace(
            payload_ptr=32,
            ensure_usable=lambda: trace.append("usable"),
            before_publish=lambda _: trace.append("ready"),
            after_publish=lambda _: trace.append("ack"),
            mark_failed=Mock(),
        )
        mgr = SimpleNamespace(
            _fused_device_plan_library=SimpleNamespace(
                fsa_lru_publish_async_v4=lambda *_: trace.append("publish") or publish_status,
            ),
            fsa_plan_ipc=ipc if notify else None,
            fused_device_planner_payload=[Mock()],
            fused_plan_stream=plan,
            current_kv_save_stream=object(),
            tp_rank=rank,
            tp_size=2,
            tp_group=SimpleNamespace(broadcast=lambda *_, **__: trace.append("broadcast")),
            _launch_fused_device_plan=lambda *_, **__: trace.append("plan"),
            _get_fused_device_planner_layer_state=lambda *_: object(),
            _join_fsa_plan_ipc=Mock(),
            fused_plan_metadata_npu=SimpleNamespace(data_ptr=lambda: 16),
            fused_device_planner_vector_cores=24,
            topk=16,
        )
        submit = namespace["_submit_fused_device_plan"]
        return mgr, ipc, ctx, trace, submit

    def test_notify_orders_plan_publish_event_and_ack(self):
        for rank, expected in (
            (0, ["usable", "record", "wait input", "plan", "wait KV", "ready", "publish", "record", "ack"]),
            (1, ["usable", "record", "wait input", "ready", "publish", "record", "ack"]),
        ):
            with self.subTest(rank=rank):
                mgr, _, ctx, trace, submit = self.make_submitter(rank)
                submit(mgr, 0, 1, (), SimpleNamespace(data_ptr=lambda: 64), True)
                self.assertEqual(trace, expected)
                self.assertIs(ctx._fsa_plan_ipc_join, mgr._join_fsa_plan_ipc)
                self.assertIsNotNone(mgr._fsa_plan_publish_done)

    def test_hccl_comparison_publishes_after_broadcast(self):
        mgr, _, ctx, trace, submit = self.make_submitter(notify=False)
        # The real HCCL tensor exposes data_ptr on a slice.

        class Payload:
            def data_ptr(self):
                return 32

            def __getitem__(self, _):
                return self

        mgr.fused_device_planner_payload = Payload()
        submit(mgr, 0, 1, (), SimpleNamespace(data_ptr=lambda: 64), False)
        self.assertEqual(trace, ["plan", "broadcast", "publish"])
        self.assertIsNone(ctx._fsa_plan_ipc_join)

    def test_failed_publish_does_not_acknowledge_payload(self):
        mgr, ipc, _, trace, submit = self.make_submitter(publish_status=17)
        with self.assertRaisesRegex(RuntimeError, "ACL status 17"):
            submit(mgr, 0, 1, (), SimpleNamespace(data_ptr=lambda: 64), False)
        self.assertNotIn("ack", trace)
        self.assertFalse(hasattr(mgr, "_fsa_plan_publish_done"))
        ipc.mark_failed.assert_called_once()

    def test_failed_plan_does_not_signal_ready(self):
        mgr, ipc, _, trace, submit = self.make_submitter()
        mgr._launch_fused_device_plan = Mock(side_effect=RuntimeError("planner launch failed"))
        with self.assertRaisesRegex(RuntimeError, "planner launch failed"):
            submit(mgr, 0, 1, (), SimpleNamespace(data_ptr=lambda: 64), True)
        self.assertEqual(trace, ["usable", "record", "wait input"])
        ipc.mark_failed.assert_called_once()

    def test_reused_layer_waits_for_shared_kv_only_for_multi_token_decode(self):
        torch_npu = Mock()
        stream = torch_npu.npu.current_stream.return_value
        inject = load_methods(
            MANAGER, "SparseKVOffloadManager", {"inject_current_kv_into_selection"}, {"torch_npu": torch_npu}
        )["inject_current_kv_into_selection"]
        tensor = MagicMock(shape=(4, 2))
        mgr = SimpleNamespace(
            _get_offload_layer_id=lambda _: 1,
            current_kv_by_layer={1: (tensor, tensor)},
            fsa_plan_ipc=object(),
            _fsa_plan_publish_done=object(),
            _fused_device_plan_library=object(),
            current_kv_save_stream=object(),
            tp_group=Mock(),
            tp_rank=0,
            tp_size=2,
            fused_plan_status_npu=object(),
            fused_plan_current_linear_slots_npu=tensor,
        )
        for speculative, capturing, tokens, requests, owner, broadcast in (
            (False, True, 2, 2, 0, False),
            (False, True, 2, 4, 0, False),  # single-token graph with padded requests
            (True, True, 2, 2, 0, True),  # one-token dummy rows can replay wider verification rows
            (True, False, 3, 2, 0, True),
            (True, False, 2, 2, 0, False),
            (True, True, 3, 2, 1, False),  # owner already ordered publication
        ):
            with self.subTest(
                speculative=speculative, capturing=capturing, tokens=tokens, requests=requests, owner=owner
            ):
                mgr.fused_plan_multitoken_capture = speculative
                mgr.fused_overlap_plan_owner_layer_id = owner
                mgr.tp_group.reset_mock()
                torch_npu.reset_mock()
                inject(mgr, "layer", tokens, tensor, tensor, capturing, multi_token_decode=tokens != requests)
                stream.wait_event.assert_called_once_with(mgr._fsa_plan_publish_done)
                if broadcast:
                    mgr.tp_group.broadcast.assert_called_once_with(mgr.fused_plan_status_npu, src=0)
                else:
                    mgr.tp_group.broadcast.assert_not_called()
                if capturing and broadcast:
                    stream.wait_stream.assert_called_once_with(mgr.current_kv_save_stream)
                else:
                    stream.wait_stream.assert_not_called()
                self.assertEqual(torch_npu.npu_scatter_nd_update_.call_count, 2)

    def test_cpu_owner_capture_orders_writeback_before_plan_broadcast(self):
        stream = Mock()
        npu = SimpleNamespace(current_stream=Mock(), stream=lambda _: contextlib.nullcontext())
        prepare = load_methods(
            MANAGER,
            "SparseKVOffloadManager",
            {"prepare_fused_overlap_external_plan"},
            dict(
                torch_npu=SimpleNamespace(npu=npu),
                _sparse_kv_ops=Mock(),
                FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT=32,
                FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS=64,
            ),
        )["prepare_fused_overlap_external_plan"]
        tensor = MagicMock()
        for rank in (0, 1):
            manager = MagicMock(
                tp_rank=rank,
                tp_size=2,
                topk=8,
                _fused_device_plan_library=None,
                fused_plan_stream=stream,
                fused_overlap_plan_owner_layer_id=None,
            )
            order = Mock()
            order.attach_mock(stream.wait_stream, "writeback")
            order.attach_mock(manager.tp_group.broadcast, "publish")
            prepare(manager, "layer", 2, tensor, tensor, tensor, tensor, tensor, capturing=True)
            expected = (["writeback"] if rank == 0 else []) + ["publish"]
            self.assertEqual([call[0] for call in order.mock_calls], expected)
            if rank == 0:
                stream.wait_stream.assert_called_once_with(manager.current_kv_save_stream)

    def test_cpu_reuse_orders_both_ranks_before_attention(self):
        torch_npu = Mock()
        inject = load_methods(
            MANAGER, "SparseKVOffloadManager", {"inject_current_kv_into_selection"}, {"torch_npu": torch_npu}
        )["inject_current_kv_into_selection"]
        tensor = MagicMock(shape=(4, 2))
        for rank in (0, 1):
            for capturing, multi, owner, sync in (
                (True, True, 0, True),
                (False, True, 0, True),
                (False, False, 0, False),
                (True, True, 1, False),
            ):
                with self.subTest(rank=rank, capturing=capturing, multi=multi, owner=owner):
                    manager = MagicMock(
                        tp_rank=rank,
                        tp_size=2,
                        fsa_plan_ipc=None,
                        _fused_device_plan_library=None,
                        fused_overlap_plan_owner_layer_id=owner,
                        fused_plan_multitoken_capture=True,
                    )
                    manager._get_offload_layer_id.return_value = 1
                    manager.current_kv_by_layer.get.return_value = (tensor, tensor)
                    order = Mock()
                    stream = torch_npu.npu.current_stream.return_value
                    order.attach_mock(stream.wait_stream, "wait")
                    order.attach_mock(manager.tp_group.broadcast, "broadcast")
                    order.attach_mock(torch_npu.npu_scatter_nd_update_, "scatter")
                    inject(manager, "layer", 2, tensor, tensor, capturing, multi_token_decode=multi)
                    expected = [("wait", manager.fused_plan_stream)] if capturing else []
                    if sync and capturing and rank == 0:
                        expected.append(("wait", manager.current_kv_save_stream))
                    if sync:
                        expected.append(("broadcast", manager.fused_plan_status_npu))
                    self.assertEqual([(c[0], c.args[0]) for c in order.mock_calls[:-2]], expected)
                    self.assertEqual([c[0] for c in order.mock_calls[-2:]], ["scatter", "scatter"])

    def test_capture_joins_after_forward_and_before_context_exit(self):
        trace = []
        extra = SimpleNamespace(capturing=False)
        ctx = SimpleNamespace(cudagraph_runtime_mode="full")
        namespace = load_methods(
            GRAPH,
            "ModelWithContext",
            {"forward"},
            {
                "get_forward_context": lambda: ctx,
                "torch": SimpleNamespace(npu=SimpleNamespace(is_current_stream_capturing=lambda: True)),
                "_EXTRA_CTX": extra,
                "CUDAGraphMode": SimpleNamespace(PIECEWISE="piecewise"),
                "patch": patch,
                "nullcontext": contextlib.nullcontext,
            },
        )

        def forward():
            trace.append("attention")
            ctx._fsa_plan_ipc_join = lambda: trace.append("join ACK")
            return 42

        wrapper = SimpleNamespace(
            original_model=forward,
            is_draft_model=False,
            is_draft_model_prefill=False,
            fsa_plan_ipc=object(),
            device_metadata=SimpleNamespace(
                begin_forward=lambda: trace.append("metadata begin"),
                finish=lambda: trace.append("metadata end"),
            ),
        )
        self.assertEqual(namespace["forward"](wrapper), 42)
        self.assertEqual(trace, ["metadata begin", "attention", "join ACK", "metadata end"])
        self.assertFalse(hasattr(ctx, "_fsa_plan_ipc_join"))

    def test_preflight_rejects_mixed_modes_and_remote_notify_peers(self):
        remote = [None]

        def gather(reports, local, *, group):
            reports[:] = [local, remote[0]]

        notice = SimpleNamespace(read_text=lambda **_: "local-boot")
        namespace = load_methods(
            MANAGER,
            "SparseKVOffloadManager",
            {"_preflight_device_planner"},
            {
                "Path": lambda _: notice,
                "torch": SimpleNamespace(distributed=SimpleNamespace(all_gather_object=gather)),
            },
        )
        mgr = SimpleNamespace(
            use_fused_overlap=True,
            tp_size=2,
            tp_group=SimpleNamespace(cpu_group=object()),
            topk=64,
            topk_buffer_size=1024,
            max_num_topk_rows=7,
            max_model_len=16384,
            _load_fused_device_planner=Mock(),
        )
        preflight = namespace["_preflight_device_planner"]
        remote[0] = ("notify", (64, 1024, 7, 16384), "other-boot", None)
        with self.assertRaisesRegex(RuntimeError, "mode must match"):
            preflight(mgr, "cpu")
        mgr._load_fused_device_planner.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "host-local"):
            preflight(mgr, "notify")

    def test_retirement_clears_target_and_all_draft_graph_managers(self):
        namespace = load_methods(MANAGER, "SparseKVOffloadManager", {"_release_fsa_planner_graphs"}, {})
        target = SimpleNamespace(release_graphs=Mock())
        draft = SimpleNamespace(release_graphs=Mock())
        old_draft = SimpleNamespace(graphs={1: object()}, _graphs_captured=True, breakable_cg_runner=Mock())
        runner = SimpleNamespace(
            cudagraph_manager=target,
            speculator=SimpleNamespace(
                cudagraph_manager=draft,
                prefill_cudagraph_manager=old_draft,
                decode_cudagraph_manager=draft,
            ),
        )
        namespace["_release_fsa_planner_graphs"](runner)
        target.release_graphs.assert_called_once()
        draft.release_graphs.assert_called_once()
        old_draft.breakable_cg_runner.clear_graphs.assert_called_once()
        self.assertEqual(old_draft.graphs, {})
        self.assertFalse(old_draft._graphs_captured)

    def test_failed_drain_retains_notify_resources(self):
        namespace = load_methods(MANAGER, "SparseKVOffloadManager", {"close_fsa_plan_ipc"}, {})
        ipc = SimpleNamespace(imported={1: object()}, owned={}, owner_payload=None, mark_failed=Mock(), close=Mock())
        manager = SimpleNamespace(
            fsa_plan_ipc=ipc,
            _retire_fused_device_plan_graphs=Mock(side_effect=RuntimeError("stream failed")),
        )
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            namespace["close_fsa_plan_ipc"](manager, object())
        ipc.mark_failed.assert_called_once()
        ipc.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
