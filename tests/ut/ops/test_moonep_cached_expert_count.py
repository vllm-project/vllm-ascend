"""CPU checks for the renamed, nonnegative persistent expert cache capacity."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch

import test_moonep_actual_cache_layers as layer_cases
import test_moonep_slot_cache_stream as stream_cases


ROOT = Path(__file__).resolve().parents[3]
BALANCE = ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py"
NEW = "MOONEP_CACHED_REDUNDANT_EXPERT_COUNT"
OLD = "MOONEP_SLOT_CACHE"


class CachedExpertCountTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("cached_expert_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)

    def snapshot(self):
        tree = ast.parse(BALANCE.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_CACHED_EXPERT_COUNT" for t in n.targets))
        ns = {"envs": self.envs}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(BALANCE), "exec"), ns)
        return ns["_CACHED_EXPERT_COUNT"]

    def allocation_case(self, value):
        case = layer_cases.ActualCacheLayersTests(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        with patch.dict(os.environ, {NEW: value}, clear=True):
            case.ns["_CACHED_EXPERT_COUNT"] = self.snapshot()
        return case

    def test_default_zero_and_new_central_registration(self):
        self.assertIn(NEW, self.envs.env_variables)
        self.assertNotIn(OLD, self.envs.env_variables)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.snapshot(), 0)

    def test_nonnegative_integer_parsing_preserves_whitespace_and_plus(self):
        for raw, expected in (("0", 0), ("8", 8), (" +08 ", 8), ("32", 32), ("100000", 100000)):
            with self.subTest(raw=raw), patch.dict(os.environ, {NEW: raw}, clear=True):
                self.assertEqual(self.snapshot(), expected)

    def test_negative_values_rejected(self):
        for raw in ("-1", "-8"):
            with self.subTest(raw=raw), patch.dict(os.environ, {NEW: raw}, clear=True):
                with self.assertRaisesRegex(ValueError, NEW + " must be >= 0"):
                    self.snapshot()

    def test_noninteger_values_rejected(self):
        for raw in ("", "invalid", "1.5", "none"):
            with self.subTest(raw=raw), patch.dict(os.environ, {NEW: raw}, clear=True):
                with self.assertRaises(ValueError):
                    self.snapshot()

    def test_old_positive_negative_invalid_values_all_ignored(self):
        for raw in ("0", "8", "-1", "invalid"):
            with self.subTest(raw=raw), patch.dict(os.environ, {OLD: raw}, clear=True):
                self.assertEqual(self.snapshot(), 0)

    def test_new_zero_and_positive_values_override_old(self):
        for raw, expected in (("0", 0), ("3", 3)):
            with self.subTest(raw=raw), patch.dict(os.environ, {NEW: raw, OLD: "8"}, clear=True):
                self.assertEqual(self.snapshot(), expected)

    def test_invalid_new_value_never_falls_back_to_old(self):
        for raw in ("-1", "invalid"):
            with self.subTest(raw=raw), patch.dict(os.environ, {NEW: raw, OLD: "8"}, clear=True):
                with self.assertRaises(ValueError):
                    self.snapshot()

    def test_runtime_snapshots_one_central_read(self):
        getter = Mock(return_value=8)
        with patch.dict(self.envs.env_variables, {NEW: getter}):
            value = self.snapshot()
        self.assertEqual(value, 8)
        getter.assert_called_once_with()
        with patch.dict(os.environ, {NEW: "0"}, clear=True):
            self.assertEqual(value, 8)
        tree = ast.parse(BALANCE.read_text())
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == NEW for n in ast.walk(node)))

    def test_zero_still_allocates_no_persistent_cache(self):
        case = self.allocation_case("0")
        states = case.build([case.layer(), case.layer()])
        self.assertEqual(case.pools()[0]["cache"], {})
        self.assertEqual(states[0].caches, {})
        self.assertTrue(all(len(shape) == 3 for shape in case.allocations))

    def test_capacity_may_exceed_local_experts_and_temporary_slots(self):
        case = self.allocation_case("5")
        state = case.build([case.layer()])[0]
        self.assertEqual(state.epn, 2)
        self.assertEqual(state.B, 2)
        self.assertEqual(state.caches["w13"].shape, (1, 5, 2, 4))
        self.assertEqual(len(state.cache_experts), 5)
        self.assertGreater(5, state.epn)
        self.assertGreater(5, state.B)

    def test_different_layer_expert_counts_do_not_introduce_an_upper_bound(self):
        case = self.allocation_case("5")
        first, second = case.layer(), case.layer()
        second.w13_weight = torch.zeros(4, 2, 4)
        second.w2_weight = torch.zeros(4, 2, 2)
        states = case.build([first, second])
        self.assertEqual([s.epn for s in states], [2, 4])
        self.assertEqual([s.caches["w13"].shape[1] for s in states], [5, 5])

    def test_home_gemm_overlap_still_requires_cache_zero(self):
        for mode in ("none", "dispatch", "dispatch|home_gemm1"):
            with self.subTest(mode=mode):
                case = stream_cases.CacheStreamTests(methodName="runTest")
                case.setUp()
                self.addCleanup(case.doCleanups)
                case.ns["_PREFETCH_OVERLAP"] = mode
                case.ns["_CACHED_EXPERT_COUNT"] = 2
                plan = stream_cases.Plan(torch.tensor([[1, 2]]))
                if mode == "dispatch|home_gemm1":
                    with self.assertRaisesRegex(RuntimeError, NEW + "=0"):
                        case.forward(plan)
                    self.assertEqual(case.events, [])
                else:
                    self.assertEqual(case.forward(plan).flatten().tolist(), [11., 22.])

    def test_cyclic_replacement_can_use_cache_entries_beyond_slot_count(self):
        case = stream_cases.CacheStreamTests(methodName="runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.ns.update(_PREFETCH_OVERLAP="none", _CACHED_EXPERT_COUNT=5)
        case.state.caches = {name: torch.zeros(1, 5, 1, 1) for name in case.state.caches}
        case.state.cache_experts = [-1] * 5
        for experts in ([1, 2], [3, 4], [5, -1]):
            case.forward(stream_cases.Plan(torch.tensor([experts])))
        self.assertEqual(case.state.cache_experts, [1, 2, 3, 4, 5])
        self.assertEqual(case.state.cache_next, 0)
        before = len(case.calls)
        self.assertEqual(case.forward(stream_cases.Plan(torch.tensor([[5, 1]]))).flatten().tolist(), [55., 11.])
        self.assertEqual(len(case.calls), before)

    def test_no_old_runtime_name_or_alias_remains(self):
        for rel in ("envs.py", "ops/fused_moe/moonep_balance.py", "ops/fused_moe/moonep_comm.py"):
            tree = ast.parse((ROOT / "vllm_ascend" / rel).read_text())
            self.assertFalse(any(isinstance(n, ast.Name) and n.id == OLD for n in ast.walk(tree)))
            self.assertFalse(any(isinstance(n, ast.Constant) and n.value == OLD for n in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
