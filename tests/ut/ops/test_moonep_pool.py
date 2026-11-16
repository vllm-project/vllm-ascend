# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for adaptive capacity selection and collective-safe fallbacks."""
import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/moonep_comm.py"


class AdaptivePoolTests(unittest.TestCase):
    def setUp(self):
        self.context = SimpleNamespace(dp_metadata=object(), max_tokens_across_dp=1)
        self.dp_world_size = 8
        spec = importlib.util.spec_from_file_location(
            "pool_memory_envs", SOURCE.parents[2] / "envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        self.namespace = {
            "envs": self.envs,
            "os": os,
            "get_forward_context": self.get_context,
            "get_dp_group": lambda: SimpleNamespace(world_size=self.dp_world_size),
            "get_ep_group": lambda: SimpleNamespace(world_size=8, rank_in_group=0),
            "MOONEP_MIN_DISPATCH_TOKENS": 128,
            "_ENV_CAP_TRACE": False, "_CAP_TRACE": {"capacity": None},
            "_GEO_CACHE": {},
        }
        names = {"_min_dispatch_tokens", "_moonep_pool_capacities", "_step_token_count",
                 "_geometry", "_select_buffer_for_step"}
        tree = ast.parse(SOURCE.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), self.namespace)
        with patch.dict(os.environ, {}, clear=True):
            caps = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.install_pool(caps)

    def get_context(self):
        if self.context is None:
            raise RuntimeError("No forward context")
        return self.context

    def install_pool(self, caps):
        self.pool = {cap: SimpleNamespace(_require_ctx=lambda: {"B": 32}) for cap in caps}
        self.namespace["_MOONEP_RUNTIME"] = SimpleNamespace(
            _ensure_initialized=Mock(), _buffers={(256, 6, 4096): self.pool})
        self.namespace["_GEO_CACHE"].clear()

    def select(self, tokens):
        buffer, capacity = self.namespace["_select_buffer_for_step"](256, 6, 4096, tokens)
        self.assertIs(buffer, self.pool[capacity])
        return capacity

    def test_old_pool_environment_cannot_disable_adaptive_selection(self):
        self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = 16
        self.context.max_tokens_across_dp = 33
        for value in ("0", "1"):
            with self.subTest(old_switch=value), patch.dict(os.environ, {"MOONEP_POOL": value}):
                self.assertEqual(self.select(8), 64)

    def test_default_minimum_stays_128(self):
        with patch.dict(os.environ, {}, clear=True):
            minimum = self.namespace["_min_dispatch_tokens"]()
        self.assertEqual(minimum, 128)
        self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = minimum
        self.assertEqual(self.select(1), 128)

    def test_explicit_experiment_minimum_16_still_applies(self):
        with patch.dict(os.environ, {"MOONEP_MIN_DISPATCH_TOKENS": "16"}, clear=True):
            self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = self.namespace["_min_dispatch_tokens"]()
        self.assertEqual(self.select(1), 16)

    def test_removed_minimum_alias_does_not_override_default(self):
        for old_value in ("16", "0", "not-an-integer"):
            with self.subTest(old_value=old_value), patch.dict(
                os.environ, {"MOONEP_MIN_CAP": old_value}, clear=True
            ):
                minimum = self.namespace["_min_dispatch_tokens"]()
            self.assertEqual(minimum, 128)
            self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = minimum
            self.assertEqual(self.select(1), 128)

    def test_primary_minimum_still_applies_with_removed_alias_present(self):
        for new_value, old_value in (("16", "32"), ("32", "16"), ("16", "not-an-integer")):
            with self.subTest(new_value=new_value, old_value=old_value), patch.dict(
                os.environ, {"MOONEP_MIN_DISPATCH_TOKENS": new_value, "MOONEP_MIN_CAP": old_value},
                clear=True,
            ):
                minimum = self.namespace["_min_dispatch_tokens"]()
            self.assertEqual(minimum, int(new_value))
            self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = minimum
            self.assertEqual(self.select(1), int(new_value))

    def test_invalid_minimum_is_still_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"MOONEP_MIN_DISPATCH_TOKENS": value, "MOONEP_MIN_CAP": "16"}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "must be >= 1"):
                    self.namespace["_min_dispatch_tokens"]()

    def test_global_dp_maximum_gives_all_ranks_the_same_capacity(self):
        self.context.max_tokens_across_dp = 513
        for local_tokens in (0, 1, 128, 512, 513):
            with self.subTest(local_tokens=local_tokens):
                self.assertEqual(self.select(local_tokens), 768)

    def test_single_dp_can_select_without_dp_metadata(self):
        self.dp_world_size = 1
        self.context = SimpleNamespace(dp_metadata=None, max_tokens_across_dp=33)
        self.namespace["MOONEP_MIN_DISPATCH_TOKENS"] = 16
        self.assertEqual(self.select(33), 64)

    def test_multi_dp_missing_metadata_uses_full_capacity(self):
        self.context.dp_metadata = None
        self.assertEqual(self.select(1), 4096)

    def test_missing_forward_context_uses_full_capacity(self):
        self.context = None
        self.assertEqual(self.select(1), 4096)

    def test_missing_global_token_count_uses_full_capacity(self):
        self.context = SimpleNamespace(dp_metadata=object())
        self.assertEqual(self.select(1), 4096)

    def test_oversized_global_count_keeps_existing_full_capacity_fallback(self):
        self.context.max_tokens_across_dp = 8192
        self.assertEqual(self.select(1), 4096)

    def test_local_tokens_beyond_full_capacity_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "selected capacity"):
            self.select(4097)

    def test_custom_capacity_list_still_controls_selection(self):
        with patch.dict(os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": "1024,256,256"}, clear=True):
            caps = self.namespace["_moonep_pool_capacities"](4096)
        self.assertEqual(caps, [256, 1024, 4096])
        self.install_pool(caps)
        self.context.max_tokens_across_dp = 300
        self.assertEqual(self.select(64), 1024)

    def test_historical_4k_capacity_ladder_is_preserved(self):
        expected = [16, 32, 64, 128, 256, 512, *range(768, 4097, 256)]
        self.assertEqual(sorted(self.pool), expected)
        with patch.dict(os.environ, {"MOONEP_POOL_MAX_TIER_COUNT": "32",
                                     "MOONEP_POOL_MEMORY_BUDGET_MB": "2048"}, clear=True):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](
                4096, cost_per_token=6 * 4096 * 2), expected)

    def test_rung_limit_and_budget_still_change_ladder_density(self):
        with patch.dict(os.environ, {"MOONEP_POOL_MAX_TIER_COUNT": "8"}, clear=True):
            sparse = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertLessEqual(len(sparse), 8)
        self.assertEqual(sparse[-1], 4096)
        with patch.dict(os.environ, {"MOONEP_POOL_MEMORY_BUDGET_MB": "1"}, clear=True):
            low_budget = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertLess(len(low_budget), len(self.pool))
        self.assertEqual(low_budget[-1], 4096)

    def test_memory_budget_registration_and_default(self):
        self.assertIn("MOONEP_POOL_MEMORY_BUDGET_MB", self.envs.env_variables)
        self.assertNotIn("MOONEP_POOL_BUDGET_MB", self.envs.env_variables)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.envs.MOONEP_POOL_MEMORY_BUDGET_MB, 2048)

    def test_memory_budget_preserves_integer_parsing(self):
        for raw, expected in (("2048", 2048), (" +01024 ", 1024), ("0", 0), ("-1", -1)):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"MOONEP_POOL_MEMORY_BUDGET_MB": raw}, clear=True
            ):
                self.assertEqual(self.envs.MOONEP_POOL_MEMORY_BUDGET_MB, expected)

    def test_old_budget_values_are_ignored(self):
        for old in ("1", "0", "-1", "invalid"):
            with self.subTest(old=old), patch.dict(
                os.environ, {"MOONEP_POOL_BUDGET_MB": old}, clear=True
            ):
                caps = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
                self.assertEqual(caps, sorted(self.pool))

    def test_new_budget_wins_over_old(self):
        with patch.dict(os.environ, {
            "MOONEP_POOL_MEMORY_BUDGET_MB": "1", "MOONEP_POOL_BUDGET_MB": "2048",
        }, clear=True):
            caps = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertEqual(caps, [16, 32, 64, 128, 256, 512, 4096])

    def test_invalid_new_budget_does_not_fall_back(self):
        for value in ("", "invalid", "1.5"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "MOONEP_POOL_MEMORY_BUDGET_MB": value, "MOONEP_POOL_BUDGET_MB": "2048",
            }, clear=True):
                with self.assertRaises(ValueError):
                    self.namespace["_moonep_pool_capacities"](4096)

    def test_nonpositive_budget_disables_only_the_budget_constraint(self):
        for value in ("0", "-1", "-2048"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "MOONEP_POOL_MEMORY_BUDGET_MB": value, "MOONEP_POOL_MAX_TIER_COUNT": "100",
            }, clear=True):
                caps = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
                self.assertEqual(caps, [16, 32, 64, 128, 256, 512, *range(640, 4097, 128)])

    def test_manual_caps_bypass_even_invalid_budget_without_reading_it(self):
        getter = Mock(side_effect=AssertionError("Manual caps must not read the budget"))
        with patch.dict(self.envs.env_variables, {"MOONEP_POOL_MEMORY_BUDGET_MB": getter}), patch.dict(
            os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": "1024,256", "MOONEP_POOL_MEMORY_BUDGET_MB": "invalid"},
            clear=True,
        ):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](4096), [256, 1024, 4096])
        getter.assert_not_called()

    def test_budget_is_read_at_ladder_generation_without_new_cache(self):
        with patch.dict(os.environ, {"MOONEP_POOL_MEMORY_BUDGET_MB": "1"}, clear=True):
            sparse = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
            os.environ["MOONEP_POOL_MEMORY_BUDGET_MB"] = "2048"
            dense = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertLess(len(sparse), len(dense))

    def test_budget_remains_soft_and_keeps_maximum_capacity(self):
        with patch.dict(os.environ, {"MOONEP_POOL_MEMORY_BUDGET_MB": "1"}, clear=True):
            caps = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertEqual(caps[-1], 4096)
        self.assertGreater(sum(caps) * 6 * 4096 * 2, 2**20)

    def test_old_budget_key_has_no_runtime_read_or_registration(self):
        for relative in ("envs.py", "ops/fused_moe/moonep_comm.py"):
            tree = ast.parse((SOURCE.parents[2] / relative).read_text())
            self.assertFalse(any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                 and n.value == "MOONEP_POOL_BUDGET_MB" for n in ast.walk(tree)))


    def test_tier_parameters_are_centralized_with_original_defaults(self):
        self.assertIn("MOONEP_POOL_TOKEN_CAPACITY_TIERS", self.envs.env_variables)
        self.assertIn("MOONEP_POOL_MAX_TIER_COUNT", self.envs.env_variables)
        self.assertNotIn("MOONEP_POOL_CAPS", self.envs.env_variables)
        self.assertNotIn("MOONEP_POOL_MAX_RUNGS", self.envs.env_variables)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.envs.MOONEP_POOL_TOKEN_CAPACITY_TIERS)
            self.assertEqual(self.envs.MOONEP_POOL_MAX_TIER_COUNT, 32)

    def test_empty_tier_string_preserves_automatic_generation(self):
        with patch.dict(os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": ""}, clear=True):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](
                4096, cost_per_token=6 * 4096 * 2), sorted(self.pool))

    def test_manual_tiers_preserve_sort_dedup_and_empty_item_parsing(self):
        with patch.dict(os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": " 1024,256, ,256,+0128, "}, clear=True):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](4096), [128, 256, 1024, 4096])

    def test_manual_tiers_filter_oversized_values_and_keep_maximum(self):
        for raw in ("8192", "128,4096,8192", "4096", " , , "):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": raw}, clear=True
            ):
                tiers = self.namespace["_moonep_pool_capacities"](4096)
                self.assertEqual(tiers[-1], 4096)
                self.assertTrue(all(tier <= 4096 for tier in tiers))
                self.assertEqual(tiers, [128, 4096] if "128," in raw else [4096])

    def test_manual_tiers_do_not_add_a_new_positive_value_validator(self):
        # Preserve the existing parsing seam; invalid Buffer capacities are still
        # rejected by the existing initialization checks, not silently changed here.
        with patch.dict(os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": "-1,0,128"}, clear=True):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](4096), [-1, 0, 128, 4096])

    def test_old_tier_and_count_names_are_ignored(self):
        for raw in ("128,1024", "invalid", "", " , "):
            with self.subTest(raw=raw), patch.dict(os.environ, {
                "MOONEP_POOL_CAPS": raw, "MOONEP_POOL_MAX_RUNGS": "invalid",
            }, clear=True):
                self.assertEqual(self.namespace["_moonep_pool_capacities"](
                    4096, cost_per_token=6 * 4096 * 2), sorted(self.pool))

    def test_new_manual_tiers_override_old_name(self):
        with patch.dict(os.environ, {
            "MOONEP_POOL_TOKEN_CAPACITY_TIERS": "256,1024",
            "MOONEP_POOL_CAPS": "128,512", "MOONEP_POOL_MAX_RUNGS": "invalid",
        }, clear=True):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](4096), [256, 1024, 4096])

    def test_new_maximum_count_overrides_old_name(self):
        with patch.dict(os.environ, {
            "MOONEP_POOL_MAX_TIER_COUNT": "8", "MOONEP_POOL_MAX_RUNGS": "100",
        }, clear=True):
            tiers = self.namespace["_moonep_pool_capacities"](4096, cost_per_token=6 * 4096 * 2)
        self.assertEqual(tiers, [16, 32, 64, 128, 256, 512, 2560, 4096])

    def test_invalid_new_names_do_not_fall_back(self):
        for values in (
            {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": "bad", "MOONEP_POOL_CAPS": "128"},
            {"MOONEP_POOL_MAX_TIER_COUNT": "bad", "MOONEP_POOL_MAX_RUNGS": "32"},
        ):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True):
                with self.assertRaises(ValueError):
                    self.namespace["_moonep_pool_capacities"](4096)

    def test_manual_tiers_do_not_read_count_or_budget(self):
        count = Mock(side_effect=AssertionError("Manual tiers must skip the tier-count target"))
        budget = Mock(side_effect=AssertionError("Manual tiers must skip the budget"))
        with patch.dict(self.envs.env_variables, {
            "MOONEP_POOL_MAX_TIER_COUNT": count, "MOONEP_POOL_MEMORY_BUDGET_MB": budget,
        }), patch.dict(os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": "128,1024"}, clear=True):
            self.assertEqual(self.namespace["_moonep_pool_capacities"](4096), [128, 1024, 4096])
        count.assert_not_called()
        budget.assert_not_called()

    def test_tier_count_keeps_integer_parsing_and_no_new_range_validation(self):
        for raw, expected in ((" +032 ", 32), ("8", 8), ("0", 0), ("-1", -1)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"MOONEP_POOL_MAX_TIER_COUNT": raw}, clear=True):
                self.assertEqual(self.envs.MOONEP_POOL_MAX_TIER_COUNT, expected)

    def test_maximum_tier_count_is_still_a_generation_target(self):
        with patch.dict(os.environ, {"MOONEP_POOL_MAX_TIER_COUNT": "1"}, clear=True):
            tiers = self.namespace["_moonep_pool_capacities"](4096)
        self.assertGreater(len(tiers), 1)
        self.assertEqual(tiers[:6], [16, 32, 64, 128, 256, 512])
        self.assertEqual(tiers[-1], 4096)

    def test_tier_configuration_is_read_when_generating_without_a_new_cache(self):
        with patch.dict(os.environ, {"MOONEP_POOL_TOKEN_CAPACITY_TIERS": "128"}, clear=True):
            first = self.namespace["_moonep_pool_capacities"](4096)
            os.environ["MOONEP_POOL_TOKEN_CAPACITY_TIERS"] = "512"
            second = self.namespace["_moonep_pool_capacities"](4096)
        self.assertEqual(first, [128, 4096])
        self.assertEqual(second, [512, 4096])

    def test_old_tier_keys_have_no_runtime_reference(self):
        for relative in ("envs.py", "ops/fused_moe/moonep_comm.py"):
            tree = ast.parse((SOURCE.parents[2] / relative).read_text())
            self.assertFalse(any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                 and any(key in n.value for key in ("MOONEP_POOL_CAPS", "MOONEP_POOL_MAX_RUNGS"))
                                 for n in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
