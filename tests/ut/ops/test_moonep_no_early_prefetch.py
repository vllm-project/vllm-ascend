"""CPU regressions for removal of speculative pre-attention prefetch."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

import test_moonep_slot_cache_stream as cache_cases


ROOT = Path(__file__).resolve().parents[3]
BALANCE = ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py"


class NoEarlyPrefetchTests(unittest.TestCase):
    def harness(self, mode="dispatch", cache=0):
        case = cache_cases.CacheStreamTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.ns.update(_PREFETCH_OVERLAP=mode, _CACHED_EXPERT_COUNT=cache)
        return case

    def check_old_value(self, value):
        environment = {"MOONEP_EARLY_PREFETCH": value, "MOONEP_EARLY_PF_DEV": value}
        with patch.dict(os.environ, environment):
            case = self.harness()
            plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
            case.hook(plan)
            result = case.forward(plan)
        self.assertEqual(result.flatten().tolist(), [11., 22.])
        self.assertEqual(len(case.calls), 1)
        self.assertEqual(case.calls[0].tolist(), [[1, 2]])
        self.assertNotIn("slot_prefetch_ready", case.ctx)
        waits = [i for i, event in enumerate(case.events) if event == "wait"]
        self.assertEqual(len(waits), 2)
        self.assertLess(waits[0], case.events.index("prefetch"))
        self.assertLess(case.events.index("prefetch"), waits[1])
        self.assertLess(waits[1], case.events.index("gemm"))

    def test_old_zero_does_not_change_real_plan_prefetch(self):
        self.check_old_value("0")

    def test_old_one_does_not_enable_speculation(self):
        self.check_old_value("1")

    def test_old_invalid_value_is_ignored(self):
        self.check_old_value("invalid")

    def test_unset_variables_use_real_plan(self):
        with patch.dict(os.environ, clear=False):
            os.environ.pop("MOONEP_EARLY_PREFETCH", None)
            os.environ.pop("MOONEP_EARLY_PF_DEV", None)
            case = self.harness()
            plan = cache_cases.Plan(torch.tensor([[2, 4]], dtype=torch.int32))
            case.hook(plan)
            self.assertEqual(case.forward(plan).flatten().tolist(), [22., 44.])
            self.assertEqual(len(case.calls), 1)

    def test_main_stream_mode_fetches_without_side_event(self):
        case = self.harness(mode="none")
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        case.forward(plan)
        self.assertEqual(case.events, ["prefetch", "gemm"])
        self.assertNotIn("after_planning_hook", case.ctx)

    def test_home_mode_passes_completion_event_to_home_runner(self):
        case = self.harness(mode="dispatch|home_gemm1")
        runner = types.SimpleNamespace(activation_logged=True)
        case.state.home_gmm_runner = lambda: runner
        case.state.epn = 1
        case.state.F = 1
        observed = []

        def home(**kwargs):
            observed.append(kwargs)
            case.events.append("home_gemm")
            return case.state.slots["w13"].clone(), None

        case.ns["home_gemm_apply_mlp"] = home
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        case.hook(plan)
        ready = case.ctx["slot_prefetch_ready"][-1]
        case.forward(plan)
        self.assertIs(observed[0]["prefetch_ready"], ready)
        self.assertIs(observed[0]["range_kernel"], runner)
        self.assertNotIn("slot_prefetch_ready", case.ctx)
        self.assertEqual(case.events.count("wait"), 1)
        self.assertLess(case.events.index("wait"), case.events.index("prefetch"))
        self.assertEqual(len(case.calls), 1)

    def test_home_mode_still_rejects_persistent_cache(self):
        case = self.harness(mode="dispatch|home_gemm1", cache=2)
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError, "requires cache disabled"):
            case.forward(plan)
        self.assertEqual(case.events, [])

    def installer(self, mode):
        tree = ast.parse(BALANCE.read_text())
        names = {"maybe_install_pf_stream"}
        statements = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                statements.append(node)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id in {"_PF", "_PREFETCH_OVERLAP"} for t in targets):
                    statements.append(node)
        spec = importlib.util.spec_from_file_location("prefetch_envs", ROOT / "vllm_ascend/envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        ns = {"envs": envs}
        with patch.dict(os.environ, {"MOONEP_PREFETCH_SIDE_STREAM_OVERLAP": mode,
                                    "MOONEP_EARLY_PREFETCH": "1", "MOONEP_EARLY_PF_DEV": "1"}):
            exec(compile(ast.Module(body=statements, type_ignores=[]), str(BALANCE), "exec"), ns)
        return ns

    def test_shared_state_retains_only_real_plan_stream_fields(self):
        ns = self.installer("dispatch")
        self.assertEqual(ns["_PF"], {"stream": None, "done": False, "fail": 0, "cur_key": None})

    def test_real_plan_layer_hook_installs_once_and_sets_key(self):
        ns = self.installer("dispatch")
        hooks = []
        weight = object()
        layer = types.SimpleNamespace(mlp=types.SimpleNamespace(
            experts=types.SimpleNamespace(w13_weight=weight)),
            register_forward_pre_hook=lambda hook, **kwargs: hooks.append(hook))
        mc = types.SimpleNamespace(_early_layers=lambda model: [layer])
        with patch.dict(sys.modules, {
            "vllm_ascend.ops.fused_moe": types.SimpleNamespace(moonep_comm=mc),
            "vllm_ascend.ascend_config": types.SimpleNamespace(
                get_ascend_config=lambda: types.SimpleNamespace(enable_moonep=True)),
            "vllm.distributed.parallel_state": types.SimpleNamespace(
                get_ep_group=lambda: types.SimpleNamespace(world_size=8)),
        }):
            ns["maybe_install_pf_stream"](object())
            ns["maybe_install_pf_stream"](object())
        self.assertEqual(len(hooks), 1)
        hooks[0](layer, (), {})
        self.assertEqual(ns["_PF"]["cur_key"], id(weight))
        self.assertTrue(ns["_PF"]["done"])

    def test_off_mode_does_not_install_layer_hook(self):
        ns = self.installer("none")
        ns["maybe_install_pf_stream"](object())
        self.assertFalse(ns["_PF"]["done"])

    def test_runtime_no_longer_references_removed_variables_or_helpers(self):
        removed = {"MOONEP_EARLY_PREFETCH", "MOONEP_EARLY_PF_DEV"}
        helpers = {"maybe_install_early_prefetch", "stash_plan_row", "note_current_layer",
                   "take_predicted_plan", "finish_speculative_prefetch", "_pf_shell_plan",
                   "_pf_stats_init", "_dump_early_pf_stats", "_EARLY_PF", "_EARLY_PF_DEV"}
        for relative in ["ops/fused_moe/moonep_balance.py", "ops/fused_moe/moonep_comm.py",
                         "worker/model_runner_v1.py", "envs.py"]:
            tree = ast.parse((ROOT / "vllm_ascend" / relative).read_text())
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                       and isinstance(n.value, str)}
            symbols = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            definitions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            self.assertFalse(removed & strings, relative)
            self.assertFalse(helpers & (symbols | definitions), relative)


if __name__ == "__main__":
    unittest.main()
