# SPDX-License-Identifier: Apache-2.0
"""CPU regression for RoPE return values in DSpark context KV insertion."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


class TestContextRopeReturn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_qwen3_dflash.py"
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "precompute_and_store_context_kv"
        )
        namespace = {"torch": torch, "F": F}
        # Exercise the real function without initializing the NPU platform or
        # installing its process-wide model patches in this CPU-only test.
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        cls.precompute = staticmethod(namespace[function.name])

    def _check_rope(self, inplace, returns_none=False):
        writes = []
        states = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)
        slots = torch.arange(3)

        def rotate(positions, query, key):
            torch.testing.assert_close(positions, slots.repeat(2))
            rotated_query = query.neg_() if inplace else query.neg()
            if returns_none:
                return None
            return rotated_query, key.neg()

        def store(attn, keys, values, cache, mapping):
            writes.append((keys.clone(), values.clone(), mapping.clone()))

        attention = SimpleNamespace(
            k_norm=torch.nn.Identity(),
            rotary_emb=rotate,
            kv_cache=None,
            impl=SimpleNamespace(do_kv_cache_update=store),
        )
        model = SimpleNamespace(
            _num_attn_layers=2,
            _kv_size=2,
            _head_dim=2,
            _num_kv_heads=1,
            hidden_norm=torch.nn.Identity(),
            _fused_kv_weight=torch.cat([torch.eye(2) * scale for scale in (1, 2, 3, 4)]),
            _fused_kv_bias=None,
            layers=[SimpleNamespace(self_attn=attention)] * 2,
            _attn_layers=[attention] * 2,
        )
        self.precompute(model, states, slots, slots)
        self.assertEqual(len(writes), 2)
        for layer, (keys, values, actual_slots) in enumerate(writes):
            torch.testing.assert_close(keys, -states[:, None, :] * (2 * layer + 1))
            torch.testing.assert_close(values, states[:, None, :] * (2 * layer + 2))
            torch.testing.assert_close(actual_slots, slots)

    def test_out_of_place_rope(self):
        self._check_rope(inplace=False)

    def test_in_place_rope(self):
        self._check_rope(inplace=True)

    def test_in_place_rope_returning_none(self):
        self._check_rope(inplace=True, returns_none=True)


if __name__ == "__main__":
    unittest.main()
