"""CPU regressions for native routing after synthetic-route removal."""
import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
SELECTOR = ROOT / "vllm_ascend/ops/fused_moe/experts_selector.py"
REMOVED = (
    "VLLM_ASCEND_MOE_SIM_ROUTE", "VLLM_ASCEND_MOE_SIM_ROUTE_SEED",
    "VLLM_ASCEND_MOE_SIM_ROUTE_EVERY", "VLLM_ASCEND_MOE_SIM_ROUTE_TABLE",
)


def functions(path, names=None, **namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
             and (names is None or n.name in names)]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[future, *nodes], type_ignores=[])), str(path), "exec"), namespace)
    return namespace


class NoSimRouteTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("native_route_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        self.ns = functions(SELECTOR, torch=torch, F=F, get_weight_prefetch_method=lambda: None)
        self.hidden = torch.zeros(2, 8)
        self.logits = torch.tensor([[5., 1., -2., 3.], [-3., 2., 6., 0.]])
        self.params = dict(hidden_states=self.hidden, router_logits=self.logits,
                           top_k=2, use_grouped_topk=False, renormalize=True)

    def native(self, **overrides):
        self.ns["check_npu_moe_gating_top_k"] = Mock(return_value=False)
        return self.ns["select_experts"](**{**self.params, **overrides})


    def test_all_four_keys_removed_even_when_invalid_values_are_present(self):
        with patch.dict(os.environ, {key: "invalid" for key in REMOVED}):
            for key in REMOVED:
                self.assertNotIn(key, self.envs.env_variables)
                with self.assertRaises(AttributeError):
                    getattr(self.envs, key)

    def test_unset_simulation_uses_real_softmax_probabilities(self):
        weights, ids = self.native()
        expected, expected_ids = self.logits.softmax(-1).topk(2, dim=-1)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(ids, expected_ids.to(torch.int32))

    def test_old_simulation_specs_and_log_settings_cannot_inject_routes(self):
        expected = self.native()
        for profile in ("", "0", "@/does/not/exist", "0:99,1:1", "invalid"):
            with self.subTest(profile=profile), patch.dict(os.environ, {
                REMOVED[0]: profile, REMOVED[1]: "bad", REMOVED[2]: "bad", REMOVED[3]: "bad",
            }):
                actual = self.native()
                torch.testing.assert_close(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1])

    def test_real_router_logits_still_change_expert_selection(self):
        first = self.native()
        second = self.native(router_logits=-self.logits)
        self.assertFalse(torch.equal(first[1], second[1]))
        self.assertFalse(torch.all(first[0] == 0.5).item())

    def test_sigmoid_nonrenormalized_route_is_preserved(self):
        weights, ids = self.native(scoring_func="sigmoid", renormalize=False)
        expected, expected_ids = self.logits.sigmoid().topk(2, dim=-1)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(ids, expected_ids.to(torch.int32))

    def test_sqrtsoftplus_native_route_is_preserved(self):
        weights, ids = self.native(scoring_func="sqrtsoftplus")
        expected, expected_ids = F.softplus(self.logits).sqrt().topk(2, dim=-1)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected)
        torch.testing.assert_close(ids, expected_ids.to(torch.int32))

    def test_grouped_expert_selection_is_preserved(self):
        weights, ids = self.native(use_grouped_topk=True, topk_group=1, num_expert_group=2)
        self.assertEqual(set(ids[0].tolist()), {0, 1})
        self.assertEqual(set(ids[1].tolist()), {2, 3})
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))

    def test_custom_router_is_called_with_original_inputs(self):
        expected_weights = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
        expected_ids = torch.tensor([[3, 0], [1, 2]], dtype=torch.int64)
        custom = Mock(return_value=(expected_weights, expected_ids))
        weights, ids = self.native(custom_routing_function=custom)
        custom.assert_called_once()
        self.assertIs(custom.call_args.kwargs["hidden_states"], self.hidden)
        self.assertIs(custom.call_args.kwargs["gating_output"], self.logits)
        self.assertEqual(custom.call_args.kwargs["topk"], 2)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(ids, expected_ids.to(torch.int32))

    def test_fused_backend_remains_selected_and_receives_original_parameters(self):
        expected = (torch.tensor([[0.8, 0.2], [0.6, 0.4]]),
                    torch.tensor([[0, 3], [2, 1]], dtype=torch.int32))
        fused = Mock(return_value=expected)
        self.ns["_select_experts_with_fusion_ops"] = fused
        with patch.dict(os.environ, {key: "invalid" for key in REMOVED}):
            result = self.ns["select_experts"](**self.params)
        self.assertIs(result[0], expected[0])
        self.assertIs(result[1], expected[1])
        self.assertIs(fused.call_args.kwargs["router_logits"], self.logits)
        self.assertIs(fused.call_args.kwargs["hidden_states"], self.hidden)
        self.assertEqual(fused.call_args.kwargs["top_k"], 2)

    def test_native_weight_prefetch_preprocess_remains_before_routing(self):
        events = []
        prefetch = Mock(side_effect=lambda *args: events.append("prefetch"))
        original = self.ns["_native_select_experts"]
        def route(**kwargs):
            events.append("route")
            return original(**kwargs)
        self.ns["get_weight_prefetch_method"] = lambda: type(
            "WeightPrefetch", (), {"maybe_prefetch_moe_weight_preprocess": staticmethod(prefetch)})()
        self.ns["_native_select_experts"] = route
        self.native()
        self.assertEqual(events, ["prefetch", "route"])
        self.assertIs(prefetch.call_args.args[0], self.hidden)
        self.assertEqual(prefetch.call_args.args[1], "gate_up")

    def test_mix_placement_shared_experts_still_append_after_native_routing(self):
        weights, ids = self.native(mix_placement=True, num_logical_experts=4, num_shared_experts=1)
        self.assertEqual(tuple(ids.shape), (2, 3))
        torch.testing.assert_close(ids[:, -1], torch.full((2,), 4, dtype=torch.int32))
        torch.testing.assert_close(weights[:, -1], torch.ones(2))


    def test_runtime_and_generic_launcher_have_no_simulation_entry_points(self):
        self.assertFalse((ROOT / "vllm_ascend/ops/fused_moe/moe_sim_route.py").exists())
        for rel in ("envs.py", "ops/fused_moe/experts_selector.py", "worker/worker.py"):
            text = (ROOT / "vllm_ascend" / rel).read_text()
            self.assertNotIn("moe_sim_route", text)
            for key in REMOVED:
                self.assertNotIn(key, text)
        launcher = (ROOT / "scripts/dsv4flash_service/service/start_service_mode.sh").read_text()
        self.assertNotIn("SIM_ROUTE", launcher)
        self.assertNotIn("仿真路由", launcher)


if __name__ == "__main__":
    unittest.main()
