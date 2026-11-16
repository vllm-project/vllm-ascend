"""CPU regressions for v0.30 native routers after synthetic-route removal."""

import ast
import importlib.util
import os
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
ROUTERS = ROOT / "vllm_ascend/ops/fused_moe/router"
REMOVED = (
    "VLLM_ASCEND_MOE_SIM_ROUTE",
    "VLLM_ASCEND_MOE_SIM_ROUTE_SEED",
    "VLLM_ASCEND_MOE_SIM_ROUTE_EVERY",
    "VLLM_ASCEND_MOE_SIM_ROUTE_TABLE",
)


def functions(path, names=None, **namespace):
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and (names is None or n.name in names)
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace


class NoSimRouteTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("native_route_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        path = ROUTERS / "grouped_topk_router.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_compute_routing", "_native_grouped_topk", "_renormalize_topk_weights")
        ]
        ns = {"torch": torch, "F": F}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])), str(path), "exec"),
            ns,
        )
        self.methods = ns
        self.hidden = torch.zeros(2, 8)
        self.logits = torch.tensor([[5.0, 1.0, -2.0, 3.0], [-3.0, 2.0, 6.0, 0.0]])

    def native(
        self,
        *,
        router_logits=None,
        scoring_func="softmax",
        renormalize=True,
        use_grouped_topk=False,
        routed_scaling_factor=1.0,
    ):
        router = SimpleNamespace(
            top_k=2,
            scoring_func=scoring_func,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=1,
            num_expert_group=2,
            e_score_correction_bias=None,
            routed_scaling_factor=routed_scaling_factor,
        )
        for name in ("_native_grouped_topk", "_renormalize_topk_weights"):
            setattr(router, name, MethodType(self.methods[name], router))
        return self.methods["_compute_routing"](
            router, self.hidden, self.logits if router_logits is None else router_logits, torch.int32
        )

    def factory(self, **kwargs):
        constructors = {
            name: Mock(return_value=object())
            for name in ("CustomRoutingRouter", "AscendFusedMoERouter", "AscendGroupedTopKRouter")
        }
        ns = functions(
            ROUTERS / "router_factory.py",
            **constructors,
            get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda capability: False),
            HardwareCapability=SimpleNamespace(FUSED_MOE_COMPATIBILITY=object()),
        )
        result = ns["create_ascend_fused_moe_router"](top_k=2, global_num_experts=4, **kwargs)
        return result, constructors

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
            with (
                self.subTest(profile=profile),
                patch.dict(
                    os.environ,
                    {
                        REMOVED[0]: profile,
                        REMOVED[1]: "bad",
                        REMOVED[2]: "bad",
                        REMOVED[3]: "bad",
                    },
                ),
            ):
                actual = self.native()
                torch.testing.assert_close(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1])

    def test_real_router_logits_still_change_expert_selection(self):
        first, second = self.native(), self.native(router_logits=-self.logits)
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
        weights, ids = self.native(use_grouped_topk=True)
        self.assertEqual(set(ids[0].tolist()), {0, 1})
        self.assertEqual(set(ids[1].tolist()), {2, 3})
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))

    def test_custom_router_factory_preserves_callable(self):
        custom = Mock()
        result, constructors = self.factory(custom_routing_function=custom)
        constructor = constructors["CustomRoutingRouter"]
        self.assertIs(result, constructor.return_value)
        self.assertIs(constructor.call_args.kwargs["custom_routing_function"], custom)
        constructors["AscendFusedMoERouter"].assert_not_called()
        constructors["AscendGroupedTopKRouter"].assert_not_called()

    def test_fused_backend_remains_selected_and_receives_original_parameters(self):
        with patch.dict(os.environ, {key: "invalid" for key in REMOVED}):
            result, constructors = self.factory(scoring_func="sqrtsoftplus", routed_scaling_factor=2.0)
        constructor = constructors["AscendFusedMoERouter"]
        self.assertIs(result, constructor.return_value)
        self.assertEqual(constructor.call_args.kwargs["scoring_func"], "sqrtsoftplus")
        self.assertEqual(constructor.call_args.kwargs["routed_scaling_factor"], 2.0)

    def test_native_router_preserves_routed_scaling_factor(self):
        expected = self.native()
        actual = self.native(routed_scaling_factor=2.0)
        torch.testing.assert_close(actual[0], expected[0] * 2)
        torch.testing.assert_close(actual[1], expected[1])

    def test_shared_expert_configuration_is_forwarded_to_native_router(self):
        _, constructors = self.factory(scoring_func="sigmoid", renormalize=False, num_fused_shared_experts=1)
        self.assertEqual(constructors["AscendGroupedTopKRouter"].call_args.kwargs["num_fused_shared_experts"], 1)

    def test_runtime_and_generic_launcher_have_no_simulation_entry_points(self):
        self.assertFalse((ROOT / "vllm_ascend/ops/fused_moe/moe_sim_route.py").exists())
        for rel in (
            "envs.py",
            "ops/fused_moe/router/router_factory.py",
            "ops/fused_moe/router/fused_topk_router.py",
            "ops/fused_moe/router/grouped_topk_router.py",
            "worker/worker.py",
        ):
            text = (ROOT / "vllm_ascend" / rel).read_text()
            self.assertNotIn("moe_sim_route", text)
            for key in REMOVED:
                self.assertNotIn(key, text)
        launcher = (ROOT / "scripts/dsv4flash_service/service/start_service_mode.sh").read_text()
        self.assertNotIn("SIM_ROUTE", launcher)
        self.assertNotIn("仿真路由", launcher)


if __name__ == "__main__":
    unittest.main()
