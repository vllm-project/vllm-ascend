"""CPU regressions for permanently disabled vLLM execution diagnostics.

Execute real dispatcher, prefetch and MLP functions with CPU tensors / fake
device submissions. No NPU initialization or device operators are required.
"""
import ast
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Generic, TypeVar
import unittest
from unittest.mock import Mock, patch

import torch

import test_moonep_no_host_sync as counts_cases
import test_moonep_pool as pool_cases
import test_moonep_slot_cache_stream as cache_cases


ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "vllm_ascend/ops/fused_moe"
REMOVED = (
    "MOONEP_TIME_STATS", "MOONEP_TIME_STATS_EVERY", "MOONEP_MLP_SYNC",
    "MOONEP_PF_TIMING", "MOONEP_PACE_SYNC", "MOONEP_CAP_TRACE",
)


def environment(value):
    return {**dict.fromkeys(REMOVED, value), "MOONEP_SLOT_STATS": "0"}


def load_functions(relative, names, namespace):
    source = OPS / relative
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)


class NoExecutionDiagnosticsTests(unittest.TestCase):
    def test_removed_keys_have_no_runtime_string_references(self):
        found = []
        for source in (ROOT / "vllm_ascend").rglob("*.py"):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and node.value in REMOVED:
                    found.append((str(source), node.lineno, node.value))
        self.assertEqual(found, [])

    def test_bf16_and_mxfp8_dispatch_ignore_old_values_without_clock_or_device_sync(self):
        for value in ("0", "1", "not-a-number"):
            for transport in ("bf16", "mxfp8"):
                with self.subTest(value=value, transport=transport):
                    case = counts_cases.NoHostCountSyncTests().make_dispatch(transport)
                    fn, dispatcher, inputs, shard, weights, scales, calls, modules = case
                    fn.__globals__.pop("_ENV_PACE_SYNC", None)
                    del dispatcher._tstats
                    forbidden = Mock(side_effect=AssertionError("Diagnostic device sync is forbidden"))
                    with patch.dict(os.environ, environment(value)), patch.dict(sys.modules, modules), patch.object(
                        torch, "npu", SimpleNamespace(synchronize=forbidden), create=True
                    ), patch("time.perf_counter", side_effect=AssertionError("Diagnostic clock is forbidden")), patch.object(
                        torch.Tensor, "item", side_effect=AssertionError("No count D2H")
                    ):
                        result = fn(dispatcher, inputs)
                    forbidden.assert_not_called()
                    self.assertEqual(result.hidden_states.data_ptr(), shard.data_ptr())
                    self.assertEqual(result.group_list.tolist(), [2, 1, 1, 0])
                    self.assertEqual(result.group_list_type, 1)
                    self.assertIs(result.combine_metadata.group_list, result.group_list)
                    self.assertIsNotNone(result.combine_metadata.state)
                    self.assertTrue(result.combine_metadata.tail_masked)
                    self.assertTrue(torch.equal(result.combine_metadata.w_trim, weights))
                    self.assertEqual(calls[0]["num_tokens"], 2)
                    if transport == "bf16":
                        self.assertTrue(calls[0]["zero_copy"])
                        self.assertTrue(calls[0]["reuse_outputs"])
                        self.assertTrue(calls[0]["inter_rank_sync"])
                    else:
                        self.assertEqual(result.dynamic_scale.data_ptr(), scales.data_ptr())

    def test_capacity_selection_still_tracks_dp_metadata_without_trace_output(self):
        case = pool_cases.AdaptivePoolTests()
        case.setUp()
        case.namespace.pop("_ENV_CAP_TRACE", None)
        case.namespace.pop("_CAP_TRACE", None)
        case.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = 16
        case.install_pool([16, 64, 128, 4096])
        for value in ("0", "1", "invalid"):
            output = io.StringIO()
            with patch.dict(os.environ, environment(value)), redirect_stdout(output):
                case.context.max_tokens_across_dp = 33
                self.assertEqual(case.select(8), 64)
                case.context.max_tokens_across_dp = 129
                self.assertEqual(case.select(8), 4096)
            self.assertEqual(output.getvalue(), "")
        self.assertIn((256, 6, 4096), case.namespace["_GEO_CACHE"])

    def test_native_import_does_not_replace_mc2_ops_even_with_old_flag_one(self):
        from abc import ABC, abstractmethod
        source = OPS / "token_dispatcher.py"
        tree = ast.parse(source.read_text())
        body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *body
        ], type_ignores=[])
        def operator(*args, **kwargs):
            raise AssertionError("An operator must not execute during module setup")
        names = ("npu_moe_distribute_dispatch_v2", "npu_moe_distribute_dispatch",
                 "npu_moe_distribute_combine_v2", "npu_moe_distribute_combine")
        native = SimpleNamespace(**dict.fromkeys(names, operator))
        namespace = dict(ABC=ABC, abstractmethod=abstractmethod, Generic=Generic,
                         TMoECombineMetadata=TypeVar("TMoECombineMetadata"),
                         torch=torch, torch_npu=native)
        for name in ("MoEMC2CombineMetadata", "MoEAllGatherCombineMetadata", "MoEAllToAllCombineMetadata"):
            namespace[name] = type(name, (), {})
        with patch.dict(os.environ, environment("1")):
            exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        for name in names:
            self.assertIs(getattr(native, name), operator)
        for name in ("TokenDispatcherWithAllGather", "TokenDispatcherWithAll2AllV"):
            self.assertFalse(getattr(namespace[name].token_dispatch, "_ts_wrapped", False))
            self.assertFalse(getattr(namespace[name].token_combine, "_ts_wrapped", False))



class NoTimingPrefetchTests(unittest.TestCase):
    def fixture(self):
        case = cache_cases.CacheStreamTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def forbid(self):
        return patch.object(
            torch, "npu",
            SimpleNamespace(synchronize=Mock(side_effect=AssertionError("No full device sync"))),
            create=True,
        )

    def test_serial_prefetch_and_cache_fill_ignore_old_timing_flags(self):
        for value in ("0", "1", "invalid"):
            case = self.fixture()
            case.ns["_PREFETCH_OVERLAP"] = "none"
            plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
            with patch.dict(os.environ, environment(value)), self.forbid(), patch(
                "time.perf_counter", side_effect=AssertionError("No diagnostic clocks")
            ):
                output = case.forward(plan)
            self.assertEqual(output.flatten().tolist(), [11.0, 22.0])
            self.assertEqual(case.state.cache_stats, [1, 1])
            self.assertEqual(case.events, ["prefetch", "gemm"])

    def test_dispatch_overlap_keeps_event_join_before_gemm(self):
        case = self.fixture()
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        with patch.dict(os.environ, environment("1")), self.forbid():
            case.hook(plan)
            case.forward(plan)
        self.assertEqual(len(case.calls), 1)
        self.assertLess(max(i for i, event in enumerate(case.events) if event == "wait"),
                        case.events.index("gemm"))
        self.assertNotIn("slot_prefetch_ready", case.ctx)

    def test_home_gemm_overlap_receives_original_prefetch_completion_event(self):
        case = self.fixture()
        case.ns.update(_PREFETCH_OVERLAP="dispatch|home_gemm1", _CACHED_EXPERT_COUNT=0)
        case.state.epn, case.state.F, case.state.B = 2, 1, 2
        runner = SimpleNamespace(activation_logged=True)
        case.state.home_gmm_runner = lambda: runner
        received = []
        def home(**kwargs):
            received.append(kwargs)
            return case.state.slots["w13"].clone(), None
        case.ns["home_gemm_apply_mlp"] = home
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        with patch.dict(os.environ, environment("1")), self.forbid():
            case.hook(plan)
            event = case.ctx["slot_prefetch_ready"][-1]
            case.forward(plan)
        self.assertIs(received[0]["prefetch_ready"], event)
        self.assertIs(received[0]["range_kernel"], runner)
        self.assertEqual(received[0]["home_experts"], 2)
        self.assertNotIn("gemm", case.events)
        self.assertNotIn("slot_prefetch_ready", case.ctx)

    def test_udma_independent_qp_guard_remains_before_submission(self):
        case = self.fixture()
        case.ctx["_prefetch_use_udma"] = True
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        with patch.dict(os.environ, environment("1")), self.forbid():
            with self.assertRaisesRegex(RuntimeError, "dedicated QP"):
                case.hook(plan)
        self.assertEqual(case.calls, [])
        self.assertEqual(case.events, [])


class NoTimingMlpTests(unittest.TestCase):
    def test_native_mlp_preserves_quant_payload_and_required_event(self):
        calls = []
        output, event = object(), object()
        payload = SimpleNamespace(quant=SimpleNamespace(is_quant=True), lora_context=None)
        hidden, scale = object(), object()
        def gmm1(value):
            self.assertIs(value, payload)
            calls.append("gmm1")
            return hidden, scale
        def gmm2(value, incoming, incoming_scale):
            self.assertIs(value, payload)
            self.assertIs(incoming, hidden)
            self.assertIs(incoming_scale, scale)
            calls.append("gmm2")
            return output
        payload.activation = "silu"
        quant = SimpleNamespace(supports_fused_activation=lambda _: True,
                                apply_gmm1_act_quant=gmm1, apply_gmm2=gmm2)
        ns = dict(torch=SimpleNamespace(npu=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(record_event=lambda: event))))
        load_functions("moe_mlp.py", {"apply_moe_mlp"}, ns)
        for value in ("0", "1", "invalid"):
            calls.clear()
            with patch.dict(os.environ, environment(value)):
                actual = ns["apply_moe_mlp"](payload, quant)
            self.assertEqual(actual, (output, event))
            self.assertEqual(calls, ["gmm1", "gmm2"])


if __name__ == "__main__":
    unittest.main()
