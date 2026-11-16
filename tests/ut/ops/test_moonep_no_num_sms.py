"""Host-only regressions for removing vLLM's global MoonEP block-count override."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_moonep_transport_type as transport_cases


ROOT = Path(__file__).resolve().parents[3]
SDK = ROOT.parent / "ascend-moonep"
OLD = "MOONEP_NUM_SMS"
PLANNING = "MOONEP_PLANNING_NUM_SMS"


class NoGlobalNumSmsTests(unittest.TestCase):
    def setUp(self):
        self.sdk_tree = ast.parse((SDK / "ascend_moonep/api.py").read_text())
        self.context_fn = next(n for n in self.sdk_tree.body if isinstance(n, ast.FunctionDef)
                               and n.name == "_create_context")
        self.planning_fn = next(n for n in self.sdk_tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == "_planning_num_sms")

    def runtime(self, mode="dispatch"):
        case = transport_cases.TransportTypeTests(methodName="runTest")
        case.setUp()
        return case.harness(mode=mode)

    def planning_setup(self, tokens, topk=6, num_sms=None):
        # Execute the real SDK selection, workspace allocation and Op-construction
        # statements, with host-only allocation/size/Op doubles. No NPU imports.
        nodes = []
        selected = {"N", "NvS", "planning_num_sms", "planning_local_elems",
                    "planning_symmetric_bytes", "planning_workspace", "ws", "kernels"}
        for node in self.context_fn.body:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "num_sms is None":
                nodes.append(node)
            elif isinstance(node, ast.Assert) and "num_sms" in ast.unparse(node.test):
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                names = {n.id for target in node.targets for n in ast.walk(target)
                         if isinstance(n, ast.Name)}
                if names & selected:
                    nodes.append(node)
        sizes = Mock(return_value=(123, 456))
        empty = Mock(return_value=object())
        constructors = {name: Mock(return_value=object())
                        for name in ("PlanningOp", "DispatchOp", "CombineOp", "PrefetchOp")}
        ns = {"os": os, "S": tokens, "K": topk, "E": 256, "R": 8, "num_sms": num_sms,
              "dev": "host-only", "planning_workspace_sizes": sizes,
              "_op_ws_bytes_from_env": lambda: 64,
              "torch": SimpleNamespace(empty=empty, int32=object()), **constructors}
        exec(compile(ast.Module(body=[self.planning_fn], type_ignores=[]),
                     str(SDK / "ascend_moonep/api.py"), "exec"), ns)
        def run(environment):
            with patch.dict(os.environ, environment, clear=True):
                exec(compile(ast.Module(body=nodes, type_ignores=[]),
                             str(SDK / "ascend_moonep/api.py"), "exec"), ns)
            return ns["planning_num_sms"]
        return SimpleNamespace(run=run, ns=ns, sizes=sizes, empty=empty, ops=constructors)

    def assert_consistent(self, setup, expected):
        self.assertEqual(setup.ns["planning_num_sms"], expected)
        self.assertEqual(setup.sizes.call_args.args[-1], expected)
        self.assertEqual(setup.ops["PlanningOp"].call_args.kwargs,
                         {"ws_bytes": 456, "num_sms": expected})
        self.assertEqual(setup.empty.call_args.args, (123,))
        self.assertEqual(setup.empty.call_args.kwargs["device"], "host-only")

    def test_old_values_are_ignored_by_complete_vllm_initialization(self):
        for transport in ("mte", "udma"):
            for raw in ("8", "16", "32", "56", "0", "-1", "", "invalid"):
                with self.subTest(transport=transport, raw=raw):
                    h = self.runtime()
                    h.run({"MOONEP_TRANSPORT_TYPE": transport, OLD: raw})
                    self.assertEqual(h.constructor.call_count, 2)
                    for call in h.constructor.call_args_list:
                        self.assertNotIn("num_sms", call.kwargs)
                    h.state.publish.assert_called_once_with()

    def test_all_overlap_modes_keep_sdk_default_instead_of_old_override(self):
        for mode in ("none", "dispatch", "dispatch|home_gemm1"):
            with self.subTest(mode=mode):
                h = self.runtime(mode=mode)
                h.run({OLD: "16"})
                for call in h.constructor.call_args_list:
                    self.assertNotIn("num_sms", call.kwargs)

    def test_automatic_selection_and_workspace_match_at_route_threshold(self):
        for tokens, topk, expected in ((16, 6, 8), (1365, 6, 8), (8192, 1, 8),
                                       (8193, 1, 32), (1366, 6, 32), (4096, 6, 32)):
            with self.subTest(tokens=tokens, topk=topk):
                h = self.planning_setup(tokens, topk)
                self.assertEqual(h.run({OLD: "invalid"}), expected)
                self.assertEqual(h.ns["num_sms"], 32)
                self.assert_consistent(h, expected)

    def test_explicit_planning_override_controls_both_workspace_and_op(self):
        for tokens in (128, 4096):
            for value in (8, 16, 32, 56):
                with self.subTest(tokens=tokens, value=value):
                    h = self.planning_setup(tokens)
                    self.assertEqual(h.run({PLANNING: str(value), OLD: "invalid"}), value)
                    self.assert_consistent(h, value)

    def test_empty_planning_override_keeps_automatic_rule(self):
        for tokens, expected in ((128, 8), (4096, 32)):
            with self.subTest(tokens=tokens):
                h = self.planning_setup(tokens)
                self.assertEqual(h.run({PLANNING: "", OLD: "16"}), expected)
                self.assert_consistent(h, expected)

    def test_invalid_planning_override_rejects_before_workspace_allocation(self):
        for raw in ("7", "57", "0", "-1", "1.5", "auto", " "):
            with self.subTest(raw=raw):
                h = self.planning_setup(4096)
                with self.assertRaises(ValueError):
                    h.run({PLANNING: raw, OLD: "32"})
                h.sizes.assert_not_called()
                h.empty.assert_not_called()
                for constructor in h.ops.values():
                    constructor.assert_not_called()

    def test_existing_planning_integer_parsing_is_preserved(self):
        h = self.planning_setup(4096)
        self.assertEqual(h.run({PLANNING: " +016 "}), 16)
        self.assert_consistent(h, 16)

    def test_independent_sdk_num_sms_argument_still_changes_auto_selection(self):
        for default, expected in ((4, 8), (16, 16), (32, 32), (56, 32)):
            with self.subTest(default=default):
                h = self.planning_setup(4096, num_sms=default)
                self.assertEqual(h.run({}), expected)
                self.assert_consistent(h, expected)
        h = self.planning_setup(4096, num_sms=16)
        self.assertEqual(h.run({PLANNING: "56"}), 56)
        self.assert_consistent(h, 56)

    def test_each_capacity_pool_uses_its_own_consistent_planning_block_count(self):
        for tokens in (16, 128, 1024, 2048, 4096):
            with self.subTest(tokens=tokens):
                h = self.planning_setup(tokens)
                expected = 8 if tokens * 6 <= 8192 else 32
                h.run({})
                self.assert_consistent(h, expected)

    def test_removed_environment_has_no_runtime_or_central_alias(self):
        for path in (ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py",
                     ROOT / "vllm_ascend/envs.py"):
            strings = {n.value for n in ast.walk(ast.parse(path.read_text()))
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertNotIn(OLD, strings)
        # The surviving Planning override is implemented by the unchanged SDK.
        self.assertIn(PLANNING, {n.value for n in ast.walk(self.planning_fn)
                               if isinstance(n, ast.Constant) and isinstance(n.value, str)})


if __name__ == "__main__":
    unittest.main()
