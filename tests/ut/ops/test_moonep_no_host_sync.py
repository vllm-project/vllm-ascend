# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for device-resident dispatch counts and reusable outputs."""
import ast
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_moonep_real_tokens import Metadata, dispatch_test_modules, dispatch_test_quant, load_method


class NoHostCountSyncTests(unittest.TestCase):
    def make_dispatch(self, transport):
        tokens, capacity, hidden, experts, topk, rows = 2, 4, 1024, 4, 2, 8
        hidden_dtype = torch.float8_e4m3fn if transport == "mxfp8" else torch.bfloat16
        shard = torch.zeros(rows, hidden, dtype=hidden_dtype)
        weights = torch.arange(rows, dtype=torch.float32)
        scales = torch.zeros(rows, hidden // 64, 2, dtype=torch.uint8)
        cu = torch.tensor([2, 3, 3, 3, 4, 4], dtype=torch.int32)
        calls = []

        class Buffer:
            def _require_ctx(self):
                return {"B": 2}

            def dispatch(self, *args, **kwargs):
                calls.append(kwargs)
                return shard, weights, cu, SimpleNamespace(experts_to_copy=torch.tensor([[2, -1]]))

            def dispatch_mxfp8(self, *args, **kwargs):
                calls.append(kwargs)
                return shard, scales, weights, cu, SimpleNamespace(experts_to_copy=torch.tensor([[2, -1]]))

        buffer = Buffer()
        scratch = {
            "tpe_compat": torch.zeros(experts, dtype=torch.int32),
            "cu_pad": torch.zeros(7, dtype=torch.int64),
            "counts": torch.zeros(6, dtype=torch.int64),
            "own": torch.empty(2, dtype=torch.int64),
            "w": torch.empty(rows, dtype=torch.float32),
        }
        pads = {
            "hidden": torch.empty(capacity, hidden, dtype=torch.bfloat16),
            "route": torch.empty(capacity, topk),
            "topk": torch.empty(capacity, topk, dtype=torch.int32),
            "combine_route": torch.empty(capacity, 8),
        }
        namespace = {
            "torch": torch, "time": time, "os": os,
            "_EXTRA_CTX": SimpleNamespace(capturing=False),
            "_ENV_PACE_SYNC": False,
            "_SLOT_TLS": SimpleNamespace(state=object()),
            "get_ep_group": lambda: SimpleNamespace(world_size=2, rank_in_group=0),
            "get_moonep_tokens_capacity": lambda: capacity,
            "_select_buffer_for_step": lambda *args: (buffer, capacity),
            "_pads_for": lambda *args: pads,
            "_scratch_for": lambda *args: scratch,
            "_MOONEP_RUNTIME": SimpleNamespace(_buffers={(experts, topk, hidden): {capacity: buffer}}),
            "MoETokenDispatchOutput": SimpleNamespace,
            "MoonEPCombineMetadata": Metadata,
        }
        dispatch = load_method("token_dispatch", namespace)
        dispatcher = SimpleNamespace(num_experts=experts, top_k=topk,
                                     _logged_transports=set(), _tstats=lambda *args: None)
        modules = dispatch_test_modules()
        inputs = SimpleNamespace(
            hidden_states=torch.ones(tokens, hidden, dtype=torch.bfloat16),
            topk_weights=torch.full((tokens, topk), 0.5),
            topk_ids=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            quant=dispatch_test_quant(modules, transport),
        )
        return dispatch, dispatcher, inputs, shard, weights, scales, calls, modules

    def invoke(self, case, old_switch):
        dispatch, dispatcher, inputs, _, _, _, _, modules = case
        environment = {"MOONEP_NO_SYNC": old_switch, "MOONEP_REUSE_OUT": old_switch,
                       "MOONEP_DISPATCH_IN_SHARD": old_switch,
                       "MOONEP_SLOT_MODE": old_switch,
                       "MOONEP_SLOT_STATS": "0"}
        with patch.dict(os.environ, environment), patch.dict(sys.modules, modules), patch.object(
            torch.Tensor, "item", side_effect=AssertionError("Dispatch must not read device counts to host")
        ):
            return dispatch(dispatcher, inputs)

    def test_all_dispatch_paths_keep_full_shard_and_device_counts(self):
        for transport in ("bf16", "mxfp8"):
            for old_switch in ("0", "1"):
                with self.subTest(transport=transport, old_switch=old_switch):
                    case = self.make_dispatch(transport)
                    result = self.invoke(case, old_switch)
                    _, _, inputs, shard, weights, scales, calls, _ = case
                    self.assertEqual(tuple(result.hidden_states.shape), tuple(shard.shape))
                    self.assertEqual(result.hidden_states.data_ptr(), shard.data_ptr())
                    self.assertIsNot(result.hidden_states, shard)
                    self.assertEqual(result.group_list.tolist(), [2, 1, 1, 0])
                    self.assertEqual(result.group_list_type, 1)
                    metadata = result.combine_metadata
                    self.assertTrue(metadata.tail_masked)
                    self.assertIs(metadata.group_list, result.group_list)
                    self.assertIsNotNone(metadata.state)
                    self.assertIsNotNone(metadata.source_topk_weights_sk)
                    self.assertEqual(metadata.num_tokens, inputs.hidden_states.shape[0])
                    self.assertEqual(calls[0]["num_tokens"], metadata.num_tokens)
                    if transport == "bf16":
                        self.assertTrue(calls[0]["inter_rank_sync"])
                        self.assertTrue(calls[0]["zero_copy"])
                        self.assertTrue(calls[0]["reuse_outputs"])
                        self.assertIsNone(result.dynamic_scale)
                    else:
                        self.assertEqual(result.dynamic_scale.data_ptr(), scales.data_ptr())
                    self.assertTrue(torch.equal(metadata.w_trim, weights))
                    expected_weights = weights.clone()
                    weights.fill_(99)
                    self.assertTrue(torch.equal(metadata.w_trim, expected_weights))

    def test_disposing_view_does_not_damage_reused_sdk_output(self):
        for transport in ("bf16", "mxfp8"):
            with self.subTest(transport=transport):
                case = self.make_dispatch(transport)
                first = self.invoke(case, "0")
                shard = case[3]
                first.hidden_states.set_(torch.empty(0, dtype=shard.dtype))
                self.assertEqual(tuple(shard.shape), (8, case[2].hidden_states.shape[1]))
                second = self.invoke(case, "0")
                self.assertEqual(tuple(second.hidden_states.shape), (8, case[2].hidden_states.shape[1]))
                self.assertEqual(second.hidden_states.data_ptr(), shard.data_ptr())
                if transport == "mxfp8":
                    self.assertEqual(first.dynamic_scale.data_ptr(), second.dynamic_scale.data_ptr())


class BufferOutputReuseTests(unittest.TestCase):
    """Exercise SDK cache/dispatch methods with CPU tensors and mocked kernels."""

    def setUp(self):
        source = Path(__file__).resolve().parents[4] / "ascend-moonep/ascend_moonep/api.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Buffer")
        names = {"_reused_cache", "_reused_output", "_reused_plan_outputs",
                 "dispatch", "_run_dispatch_on_current_stream"}
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.planning_allocations = []

        def allocate_planning_outputs(ctx):
            self.planning_allocations.append(ctx)
            return SimpleNamespace(dst=torch.empty(ctx["NvS"], 2, dtype=torch.int32)), torch.empty(6, dtype=torch.int32)

        self.events = []
        namespace = {
            "torch": torch, "MoonEPCommPlan": SimpleNamespace,
            "allocate_planning_outputs": allocate_planning_outputs,
            "_TSTATS_ON": False, "_ts": lambda *args: 0,
            "cross_rank_barrier": lambda *args: self.events.append("barrier"),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        self.buffer_type = type("BufferReuseHarness", (), {name: namespace[name] for name in names})

    def make_sdk_dispatch(self):
        hidden = torch.zeros(8, 16, dtype=torch.bfloat16)
        raw_weights = torch.zeros(8, dtype=torch.float32).view(torch.int32)

        def dispatch_kernel(*args, **kwargs):
            self.events.append("dispatch")
            hidden.fill_(3)
            raw_weights.view(torch.float32).fill_(0.5)

        ctx = {
            "N": 8, "E": 4, "NvS": 8, "B": 2,
            "hidden_buf_local": hidden, "weights_buf_local": raw_weights,
            "weights_buf": raw_weights,
            "after_planning_hook": lambda plan: self.events.append("prefetch_hook"),
            "kernels": {
                "planning": SimpleNamespace(compute=lambda *args, **kwargs: self.events.append("planning")),
                "dispatch": SimpleNamespace(compute=dispatch_kernel),
            },
        }
        buffer = self.buffer_type()
        buffer._require_ctx = lambda: ctx
        buffer._ensure_shmem_views = lambda: None
        inputs = (torch.ones(4, 16, dtype=torch.bfloat16), torch.ones(4, 2),
                  torch.zeros(4, 2, dtype=torch.int32), torch.zeros(4, dtype=torch.int32))
        return buffer, ctx, inputs

    def test_sdk_zero_copy_returns_shmem_views_without_boundary_copies(self):
        buffer, ctx, inputs = self.make_sdk_dispatch()
        with patch.dict(os.environ, {"ASCEND_MOONEP_TRACE": "0"}), patch.object(
            torch.Tensor, "copy_", side_effect=AssertionError("Unexpected full-shard copy")
        ):
            hidden, weights, _, _ = buffer.dispatch(
                *inputs, zero_copy=True, reuse_outputs=True, num_tokens=2)
        self.assertIs(hidden, ctx["hidden_buf_local"])
        self.assertEqual(weights.data_ptr(), ctx["weights_buf_local"].data_ptr())
        self.assertTrue(torch.all(hidden == 3))
        self.assertTrue(torch.all(weights == 0.5))
        self.assertEqual(ctx["num_valid_tokens"], 2)
        self.assertEqual(self.events, ["barrier", "planning", "prefetch_hook", "dispatch"])

    def test_public_sdk_still_supports_independent_copy_outputs(self):
        buffer, ctx, inputs = self.make_sdk_dispatch()
        with patch.dict(os.environ, {"ASCEND_MOONEP_TRACE": "0"}):
            hidden, weights, _, _ = buffer.dispatch(
                *inputs, zero_copy=False, reuse_outputs=True, num_tokens=2)
        self.assertNotEqual(hidden.data_ptr(), ctx["hidden_buf_local"].data_ptr())
        self.assertNotEqual(weights.data_ptr(), ctx["weights_buf_local"].data_ptr())
        self.assertTrue(torch.equal(hidden, ctx["hidden_buf_local"]))
        self.assertTrue(torch.equal(weights, ctx["weights_buf_local"].view(torch.float32)))
        ctx["hidden_buf_local"].zero_()
        ctx["weights_buf_local"].zero_()
        self.assertTrue(torch.all(hidden == 3))
        self.assertTrue(torch.all(weights == 0.5))

    def test_consecutive_outputs_keep_storage_and_avoid_reallocation(self):
        buffer = self.buffer_type()
        template = torch.empty(8, 16, dtype=torch.bfloat16)
        with patch.object(torch, "empty_like", wraps=torch.empty_like) as allocate:
            first = buffer._reused_output("hidden", template)
            first.fill_(3)
            second = buffer._reused_output("hidden", template)
        self.assertIs(first, second)
        self.assertEqual(allocate.call_count, 1)
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.all(second == 3))
        first_weights = buffer._reused_output("weights", shape=(8,), dtype=torch.float32, device="cpu")
        self.assertIs(first_weights, buffer._reused_output("weights", shape=(8,), dtype=torch.float32, device="cpu"))

    def test_distinct_capacity_buffers_have_independent_outputs(self):
        small, large = self.buffer_type(), self.buffer_type()
        a = small._reused_output("hidden", torch.empty(8, 16))
        b = large._reused_output("hidden", torch.empty(16, 16))
        a.zero_()
        b.fill_(1)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertEqual(tuple(a.shape), (8, 16))
        self.assertEqual(tuple(b.shape), (16, 16))
        self.assertTrue(torch.all(a == 0))

    def test_planning_tables_reuse_per_buffer_without_cross_capacity_alias(self):
        small, large = self.buffer_type(), self.buffer_type()
        small_ctx, large_ctx = {"NvS": 8}, {"NvS": 16}
        first = small._reused_plan_outputs(small_ctx)
        self.assertIs(first, small._reused_plan_outputs(small_ctx))
        other = large._reused_plan_outputs(large_ctx)
        self.assertEqual(len(self.planning_allocations), 2)
        self.assertNotEqual(first[0].dst.data_ptr(), other[0].dst.data_ptr())
        self.assertNotEqual(first[1].data_ptr(), other[1].data_ptr())


if __name__ == "__main__":
    unittest.main()
