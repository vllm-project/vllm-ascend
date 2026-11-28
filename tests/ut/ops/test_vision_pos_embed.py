# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU dispatch/contract checks. Also runnable directly without UT conftest.

AST extraction isolates the device/model import boundary; this does not execute
or validate the Triton kernel. NPU coverage lives in the nightly operator test.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

_ROOT = Path(__file__).resolve().parents[3]


def _load_function(path, name, namespace):
    tree = ast.parse((_ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestVisionPositionDispatch(unittest.TestCase):
    def test_dispatch_and_concat(self):
        for is_310p, supported in ((False, True), (True, True), (False, False)):
            with self.subTest(is_310p=is_310p, supported=supported):
                grids = [[1, 2, 4], [3, 4, 2]]
                native = Mock(side_effect=lambda weight, t, h, w, *args: torch.full((t * h * w, 3), h))
                fused = Mock(side_effect=native.side_effect)
                ns = dict(
                    torch=torch,
                    is_310p=Mock(return_value=is_310p),
                    supports_vision_pos_embed=Mock(return_value=supported),
                    vision_pos_embed_interpolate=fused,
                    pos_embed_interpolate_native=native,
                )
                dispatch = _load_function(
                    "vllm_ascend/patch/worker/patch_qwen3vl.py", "_fast_pos_embed_interpolate", ns
                )
                tower = SimpleNamespace(
                    pos_embed=SimpleNamespace(weight=torch.randn(16, 3)),
                    num_grid_per_side=4,
                    spatial_merge_size=2,
                    dtype=torch.float32,
                )
                output = dispatch(tower, grids)
                selected, unused = (fused, native) if supported and not is_310p else (native, fused)
                self.assertEqual(selected.call_count, 2)
                unused.assert_not_called()
                self.assertTrue(torch.equal(output[:8], torch.full((8, 3), 2)))
                self.assertTrue(torch.equal(output[8:], torch.full((24, 3), 4)))
                self.assertIs(selected.call_args_list[0].args[0], tower.pos_embed.weight)

    def test_kernel_failure_is_visible(self):
        native = Mock()
        ns = dict(
            torch=torch,
            is_310p=Mock(return_value=False),
            supports_vision_pos_embed=Mock(return_value=True),
            vision_pos_embed_interpolate=Mock(side_effect=RuntimeError("compilation failed")),
            pos_embed_interpolate_native=native,
        )
        dispatch = _load_function("vllm_ascend/patch/worker/patch_qwen3vl.py", "_fast_pos_embed_interpolate", ns)
        tower = SimpleNamespace(
            pos_embed=SimpleNamespace(weight=torch.empty(16, 7)),
            num_grid_per_side=4,
            spatial_merge_size=2,
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "compilation failed"):
            dispatch(tower, [[1, 2, 2]])
        native.assert_not_called()

    def test_support_gate(self):
        ns = dict(
            torch=torch, HAS_TRITON=True, torch_npu=SimpleNamespace(get_npu_format=Mock(return_value=2)), _ND_FORMAT=2
        )
        supports = _load_function("vllm_ascend/ops/triton/vision_pos_embed.py", "supports_vision_pos_embed", ns)
        weight = SimpleNamespace(
            device=SimpleNamespace(type="npu"), ndim=2, shape=(16, 7), dtype=torch.bfloat16, is_contiguous=lambda: True
        )
        self.assertTrue(supports(weight, torch.bfloat16))
        for attribute, value in (("device", SimpleNamespace(type="cpu")), ("ndim", 3), ("dtype", torch.int32)):
            old = getattr(weight, attribute)
            setattr(weight, attribute, value)
            self.assertFalse(supports(weight, torch.bfloat16))
            setattr(weight, attribute, old)
        self.assertFalse(supports(weight, torch.float32))
        weight.is_contiguous = lambda: False
        self.assertFalse(supports(weight, torch.bfloat16))
        weight.is_contiguous = lambda: True
        ns["torch_npu"].get_npu_format.return_value = 29
        self.assertFalse(supports(weight, torch.bfloat16))
        ns["torch_npu"].get_npu_format.return_value = 2
        ns["HAS_TRITON"] = False
        self.assertFalse(supports(weight, torch.bfloat16))

    def test_wrapper_rejects_invalid_grids(self):
        ns = dict(torch=torch, supports_vision_pos_embed=lambda *args: True, _BLOCK_SIZE=1024)
        wrapper = _load_function("vllm_ascend/ops/triton/vision_pos_embed.py", "vision_pos_embed_interpolate", ns)
        weight = torch.empty(16, 7)
        for grid in ((0, 2, 2, 4, 2), (1, 3, 2, 4, 2), (1, 2, 2, 5, 2), (1, 2, 2, 4, 0)):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                wrapper(weight, *grid, torch.float32)


if __name__ == "__main__":
    unittest.main()
