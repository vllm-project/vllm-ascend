"""CPU checks for named prefetch overlap modes and the changed default."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import test_moonep_no_early_prefetch as early_cases
import test_moonep_slot_cache_stream as cache_cases


ROOT = Path(__file__).resolve().parents[3]
NEW = "MOONEP_PREFETCH_SIDE_STREAM_OVERLAP"
OLD = "MOONEP_PF_STREAM"


class PrefetchOverlapTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("overlap_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        self.source = ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py"
        self.tree = ast.parse(self.source.read_text())

    def snapshot(self):
        assignment = next(n for n in self.tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "_PREFETCH_OVERLAP"
                                  for t in n.targets))
        ns = {"envs": self.envs}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(self.source), "exec"), ns)
        return ns["_PREFETCH_OVERLAP"]

    def case(self, mode, cache=0):
        case = cache_cases.CacheStreamTests(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.ns.update(_PREFETCH_OVERLAP=mode, _CACHED_EXPERT_COUNT=cache)
        return case

    def installer(self, mode="dispatch"):
        case = early_cases.NoEarlyPrefetchTests(methodName="runTest")
        return case.installer(mode)

    def test_central_registration_and_default_dispatch(self):
        self.assertIn(NEW, self.envs.env_variables)
        self.assertNotIn(OLD, self.envs.env_variables)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.snapshot(), "dispatch")

    def test_three_exact_values_reach_runtime_snapshot(self):
        for mode in ("none", "dispatch", "dispatch|home_gemm1"):
            with self.subTest(mode=mode), patch.dict(os.environ, {NEW: mode}, clear=True):
                self.assertEqual(self.snapshot(), mode)

    def test_old_values_are_ignored_including_disable_and_home(self):
        for value in ("0", "1", "2", "3", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {OLD: value}, clear=True):
                self.assertEqual(self.snapshot(), "dispatch")

    def test_new_value_wins_over_old(self):
        for mode in ("none", "dispatch", "dispatch|home_gemm1"):
            with self.subTest(mode=mode), patch.dict(os.environ, {NEW: mode, OLD: "3"}, clear=True):
                self.assertEqual(self.snapshot(), mode)

    def test_invalid_new_values_never_fall_back(self):
        for value in ("", "0", "1", "2", "3", "DISPATCH", " dispatch",
                      "dispatch ", "home_gemm1", "dispatch|home gemm1",
                      "home_gemm1|dispatch", "none|dispatch", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {NEW: value, OLD: "2"}, clear=True):
                with self.assertRaisesRegex(ValueError, NEW):
                    self.snapshot()

    def test_configuration_is_snapshotted_before_forward(self):
        with patch.dict(os.environ, {NEW: "none"}, clear=True):
            mode = self.snapshot()
            os.environ[NEW] = "dispatch"
            self.assertEqual(mode, "none")
            self.assertEqual(self.snapshot(), "dispatch")
        for function_name in ("run_slot_forward", "_pf_stream_hook", "udma_prefetch_overlap_requested"):
            function = next(n for n in self.tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == function_name)
            self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == NEW
                                 for n in ast.walk(function)))

    def test_none_preserves_main_stream_prefetch_on_each_layer_call(self):
        case = self.case("none")
        case.ctx["_prefetch_use_udma"] = True
        # With no concurrent dispatch, no dedicated-overlap QP capability is needed.
        for experts in ([1, 2], [3, 4]):
            output = case.forward(cache_cases.Plan(torch.tensor([experts])))
            self.assertEqual(output.flatten().tolist(), [11. * e for e in experts])
        self.assertEqual(case.events, ["prefetch", "gemm", "prefetch", "gemm"])
        self.assertNotIn("after_planning_hook", case.ctx)
        self.assertNotIn("slot_prefetch_ready", case.ctx)
        self.assertIsNone(case.ns["_PF"]["stream"])

    def test_none_preserves_cache_hits_misses_and_refill(self):
        case = self.case("none", cache=2)
        # Replace the unused cache entry, retaining the expert hit by this plan.
        case.state.cache_next = 1
        plan = cache_cases.Plan(torch.tensor([[1, 2]]))
        self.assertEqual(case.forward(plan).flatten().tolist(), [11., 22.])
        self.assertEqual(case.calls[0].tolist(), [[-1, 2]])
        self.assertEqual(case.forward(plan).flatten().tolist(), [11., 22.])
        self.assertEqual(len(case.calls), 1)
        self.assertEqual(case.state.cache_stats, [3, 1])
        self.assertNotIn("after_planning_hook", case.ctx)

    def test_dispatch_joins_after_dispatch_before_gemm(self):
        case = self.case("dispatch")
        plan = cache_cases.Plan(torch.tensor([[1, 2]]))
        case.hook(plan)
        case.events.append("dispatch")
        case.forward(plan)
        self.assertEqual(case.events[-3:], ["dispatch", "wait", "gemm"])
        self.assertEqual(len(case.calls), 1)

    def test_both_overlap_modes_keep_udma_dedicated_qp_guard(self):
        for mode in ("dispatch", "dispatch|home_gemm1"):
            with self.subTest(mode=mode):
                case = self.case(mode)
                case.ctx["_prefetch_use_udma"] = True
                with self.assertRaisesRegex(RuntimeError, "dedicated QP"):
                    case.hook(cache_cases.Plan(torch.tensor([[1, 2]])))
                self.assertEqual(case.events, [])

    def test_udma_overlap_request_matches_modes_and_transport(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "udma_prefetch_overlap_requested")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(self.source), "exec"), ns)
        for mode in ("none", "dispatch", "dispatch|home_gemm1"):
            ns["_PREFETCH_OVERLAP"] = mode
            for udma in (False, True):
                for slot in (False, True):
                    self.assertEqual(ns[fn.name](udma, slot), udma and slot and mode != "none")

    def test_native_off_does_not_install_hooks_or_query_ep(self):
        for mode in ("dispatch", "dispatch|home_gemm1"):
            ns = self.installer(mode)
            group = Mock(side_effect=AssertionError("Off must not query EP"))
            with patch.dict(sys.modules, {
                "vllm_ascend.ascend_config": SimpleNamespace(
                    get_ascend_config=lambda: SimpleNamespace(enable_moonep=False)),
                "vllm.distributed.parallel_state": SimpleNamespace(get_ep_group=group),
            }):
                ns["maybe_install_pf_stream"](object())
            group.assert_not_called()
            self.assertFalse(ns["_PF"]["done"])

    def test_ep_one_does_not_install_hooks_or_enumerate_layers(self):
        ns = self.installer()
        layers = Mock(side_effect=AssertionError("EP1 must not enumerate layers"))
        with patch.dict(sys.modules, {
            "vllm_ascend.ascend_config": SimpleNamespace(
                get_ascend_config=lambda: SimpleNamespace(enable_moonep=True)),
            "vllm.distributed.parallel_state": SimpleNamespace(
                get_ep_group=lambda: SimpleNamespace(world_size=1)),
            "vllm_ascend.ops.fused_moe": SimpleNamespace(moonep_comm=SimpleNamespace(_early_layers=layers)),
        }):
            ns["maybe_install_pf_stream"](object())
        layers.assert_not_called()
        self.assertFalse(ns["_PF"]["done"])

    def test_none_does_not_query_config_or_install_hooks(self):
        ns = self.installer("none")
        config = Mock(side_effect=AssertionError("Disabled overlap must return immediately"))
        with patch.dict(sys.modules, {
            "vllm_ascend.ascend_config": SimpleNamespace(get_ascend_config=config),
        }):
            ns["maybe_install_pf_stream"](object())
        config.assert_not_called()
        self.assertFalse(ns["_PF"]["done"])

    def test_both_overlap_modes_install_hook_once(self):
        for mode in ("dispatch", "dispatch|home_gemm1"):
            ns = self.installer(mode)
            hooks = []
            weight = object()
            layer = SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace(w13_weight=weight)),
                                    register_forward_pre_hook=lambda hook, **kw: hooks.append(hook))
            with patch.dict(sys.modules, {
                "vllm_ascend.ascend_config": SimpleNamespace(
                    get_ascend_config=lambda: SimpleNamespace(enable_moonep=True)),
                "vllm.distributed.parallel_state": SimpleNamespace(
                    get_ep_group=lambda: SimpleNamespace(world_size=8)),
                "vllm_ascend.ops.fused_moe": SimpleNamespace(
                    moonep_comm=SimpleNamespace(_early_layers=lambda model: [layer])),
            }):
                ns["maybe_install_pf_stream"](object())
                ns["maybe_install_pf_stream"](object())
            self.assertEqual(len(hooks), 1)
            hooks[0](layer, (), {})
            self.assertEqual(ns["_PF"]["cur_key"], id(weight))

    def test_no_old_runtime_key_or_numeric_mode(self):
        for path in [ROOT / "vllm_ascend/envs.py", self.source,
                     ROOT / "vllm_ascend/worker/v2/model_runner.py"]:
            tree = ast.parse(path.read_text())
            self.assertFalse(any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                 and OLD in n.value for n in ast.walk(tree)), str(path))
            self.assertFalse(any(isinstance(n, ast.Name) and n.id == "_PF_STREAM"
                                 for n in ast.walk(tree)), str(path))


if __name__ == "__main__":
    unittest.main()
