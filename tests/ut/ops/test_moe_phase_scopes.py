# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual MLP function with mocked operators and scope recording."""

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/moe_mlp.py"


class PhaseScopeTest(unittest.TestCase):
    def test_phases_preserve_math_order_and_cleanup(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "apply_moe_mlp")
        for enabled in (False, True):
            for fused in (False, True):
                for failure in (None, "gmm1", "gmm2"):
                    stack: list[str] = []
                    calls: list[str] = []

                    @contextmanager
                    def scope(name, enabled=True, *, stack=stack):
                        if enabled:
                            stack.append(name)
                        try:
                            yield
                        finally:
                            if enabled:
                                self.assertEqual(stack.pop(), name)

                    def op(name, expected, result, *, calls=calls, enabled=enabled, failure=failure, stack=stack):
                        def run(*args):
                            self.assertEqual(stack, [expected] if enabled else [])
                            calls.append(name)
                            if failure == name:
                                raise ValueError(name)
                            return result

                        return run

                    def event(*, calls=calls, stack=stack):
                        self.assertEqual(stack, [])
                        calls.append("event")
                        return "event-token"

                    ns = {
                        "super_kernel_scope": scope,
                        "maybe_record_event": event,
                        "_unified_apply_activation": op("activation", "moe_routed_gmm1", "activated"),
                    }
                    module = ast.Module(
                        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
                        type_ignores=[],
                    )
                    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
                    quant = NS(
                        supports_fused_activation=lambda _, fused=fused: fused,
                        apply_gmm1_act_quant=op("gmm1", "moe_routed_gmm1", ("hidden", "scale")),
                        apply_gmm1=op("gmm1", "moe_routed_gmm1", "hidden"),
                        apply_act_quant=op("quant", "moe_routed_gmm1", ("hidden", "scale")),
                        apply_gmm2=op("gmm2", "moe_routed_gmm2", "output"),
                    )
                    inp = NS(quant=NS(is_quant=True), lora_context=None, activation="swiglu")
                    if failure:
                        with self.assertRaisesRegex(ValueError, failure):
                            ns["apply_moe_mlp"](inp, quant, enable_super_kernel_scopes=enabled)
                    else:
                        self.assertEqual(
                            ns["apply_moe_mlp"](inp, quant, enable_super_kernel_scopes=enabled),
                            ("output", "event-token"),
                        )
                        self.assertEqual(
                            calls,
                            ["gmm1", "event", "gmm2"] if fused else ["gmm1", "activation", "quant", "event", "gmm2"],
                        )
                    self.assertEqual(stack, [])


if __name__ == "__main__":
    unittest.main()
