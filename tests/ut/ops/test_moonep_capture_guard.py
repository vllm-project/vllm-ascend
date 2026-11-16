"""CPU checks that MoonEP capture guards cannot be bypassed by old flags."""
import ast
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

import test_moonep_slot_cache_stream as cache_cases


ROOT = Path(__file__).resolve().parents[3]


def load_guarded_entry(relative, name, context):
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        function,
    ], type_ignores=[])
    namespace = {"_EXTRA_CTX": context, "os": os}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


class Untouched:
    def __getattribute__(self, name):
        raise AssertionError("Capture rejection must precede input, buffer, or weight access")


class CaptureGuardTests(unittest.TestCase):
    def check_rejected(self, value):
        with patch.dict(os.environ):
            os.environ.pop("MOONEP_ALLOW_CAPTURE", None)
            if value is not None:
                os.environ["MOONEP_ALLOW_CAPTURE"] = value
            context = types.SimpleNamespace(capturing=True)
            dispatch = load_guarded_entry(
                "vllm_ascend/ops/fused_moe/moonep_comm.py", "token_dispatch", context)
            forward = load_guarded_entry(
                "vllm_ascend/ops/fused_moe/moonep_balance.py", "run_slot_forward", context)
            untouched = Untouched()
            with self.assertRaisesRegex(RuntimeError, "captured graphs.*--enforce-eager"):
                dispatch(untouched, untouched)
            with self.assertRaisesRegex(RuntimeError, "captured graphs.*--enforce-eager"):
                forward(state=untouched, buffer=untouched, plan=untouched,
                        token_dispatch_output=untouched, fused_experts_input=untouched,
                        use_fusion_ops=False)

    def test_unset_rejects_capture_before_any_work(self):
        self.check_rejected(None)

    def test_old_zero_rejects_capture_before_any_work(self):
        self.check_rejected("0")

    def test_old_one_cannot_bypass_either_guard(self):
        self.check_rejected("1")

    def test_invalid_old_value_cannot_bypass_either_guard(self):
        self.check_rejected("invalid")

    def harness(self):
        case = cache_cases.CacheStreamTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.ns["_CACHED_EXPERT_COUNT"] = 0
        return case

    def test_missing_forward_context_keeps_existing_fallback(self):
        case = self.harness()
        case.ns.pop("_EXTRA_CTX")
        case.ns["_PREFETCH_OVERLAP"] = "none"
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        self.assertEqual(case.forward(plan).flatten().tolist(), [11., 22.])
        self.assertEqual(case.events, ["prefetch", "gemm"])

    def test_forward_context_lookup_error_keeps_existing_fallback(self):
        class MissingContext:
            @property
            def capturing(self):
                raise RuntimeError("No engine forward context")

        case = self.harness()
        case.ns["_EXTRA_CTX"] = MissingContext()
        case.ns["_PREFETCH_OVERLAP"] = "none"
        plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        self.assertEqual(case.forward(plan).flatten().tolist(), [11., 22.])

    def test_eager_old_one_preserves_side_stream_prefetch_and_wait(self):
        with patch.dict(os.environ, {"MOONEP_ALLOW_CAPTURE": "1"}):
            case = self.harness()
            plan = cache_cases.Plan(torch.tensor([[1, 2]], dtype=torch.int32))
            case.hook(plan)
            self.assertEqual(case.forward(plan).flatten().tolist(), [11., 22.])
        self.assertEqual(len(case.calls), 1)
        self.assertNotIn("slot_prefetch_ready", case.ctx)
        waits = [i for i, event in enumerate(case.events) if event == "wait"]
        self.assertEqual(len(waits), 2)
        self.assertLess(waits[0], case.events.index("prefetch"))
        self.assertLess(case.events.index("prefetch"), waits[1])
        self.assertLess(waits[1], case.events.index("gemm"))

    def test_runtime_has_no_removed_flag_or_constant(self):
        for relative in ["vllm_ascend/ops/fused_moe/moonep_comm.py",
                         "vllm_ascend/ops/fused_moe/moonep_balance.py",
                         "vllm_ascend/envs.py"]:
            tree = ast.parse((ROOT / relative).read_text())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                       and isinstance(n.value, str)}
            self.assertNotIn("_ALLOW_CAPTURE", names)
            self.assertFalse(any("MOONEP_ALLOW_CAPTURE" in text for text in strings))


if __name__ == "__main__":
    unittest.main()
