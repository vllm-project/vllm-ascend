# SPDX-License-Identifier: Apache-2.0
"""Exercise the adapter's launch/sync contract without importing an NPU runtime."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock


class TestCatccosAsyncContract(TestCase):
    def test_sync_is_opt_in_and_only_after_launch(self):
        source = Path(__file__).parents[3] / "vllm_ascend/ops/fused_moe/catccos_adapter.py"
        tree = ast.parse(source.read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_catccos"
        )
        for sync_enabled in (False, True):
            with self.subTest(sync_enabled=sync_enabled):
                events = []
                output = object()
                tensor = Mock()
                tensor.contiguous.return_value = tensor
                tensor.to.return_value = tensor

                def launch(*args, events=events, tensor=tensor, output=output):
                    events.append("launch")
                    self.assertEqual(args, (tensor,) * 7)
                    return output

                torch = SimpleNamespace(
                    Tensor=object,
                    int32="int32",
                    float32="float32",
                    npu=SimpleNamespace(synchronize=lambda events=events: events.append("sync")),
                    ops=SimpleNamespace(catccos=SimpleNamespace(ascend950_dispatch_ffn_combine=launch)),
                )
                namespace = {
                    "torch": torch,
                    "initialize_catccos": lambda events=events: events.append("init"),
                    "moe_profile_range": lambda *args: nullcontext(),
                    "get_ascend_config": lambda sync_enabled=sync_enabled: SimpleNamespace(
                        catccos_sync_after_launch=sync_enabled
                    ),
                }
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
                self.assertIs(namespace["apply_catccos"](*([tensor] * 7)), output)
                self.assertEqual(events, ["init", "launch", "sync"] if sync_enabled else ["init", "launch"])


if __name__ == "__main__":
    main()
