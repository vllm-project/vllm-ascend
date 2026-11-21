# SPDX-License-Identifier: Apache-2.0
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SOURCE_PATH = "vllm_ascend/models/minimax_m3/minimax_m3.py"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / SOURCE_PATH).is_file())
TREE = ast.parse((ROOT / SOURCE_PATH).read_text(encoding="utf-8"))

NODE = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3RMSNorm")
SCOPE = {"torch": torch, "nn": nn}
exec(compile(ast.Module(body=[NODE], type_ignores=[]), SOURCE_PATH, "exec"), SCOPE)
Norm = SCOPE["MiniMaxM3RMSNorm"]


class TestM3Norm(unittest.TestCase):
    def make_norm(self, dtype):
        norm = Norm(64).to(dtype=dtype)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(-0.1, 0.1, 64).to(dtype))
        return norm

    def reference(self, x, weight, eps):
        # Independent PyTorch RMSNorm implementation plus the FP32 offset.
        return torch.nn.functional.rms_norm(x.float(), (64,), 1 + weight.float(), eps).to(x.dtype)

    def test_forward_and_state_dict(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                norm = self.make_norm(dtype)
                x = torch.randn(3, 64, generator=torch.Generator().manual_seed(12)).to(dtype)
                before = x.clone()
                output = norm(x)
                torch.testing.assert_close(output, self.reference(x, norm.weight, norm.variance_epsilon))
                self.assertEqual(output.dtype, dtype)
                self.assertTrue(torch.equal(x, before))
                self.assertEqual(list(norm.state_dict()), ["weight"])

    def test_residual_rounding_and_inputs_unchanged(self):
        norm = self.make_norm(torch.bfloat16)
        x = torch.randn(5, 64, generator=torch.Generator().manual_seed(42)).bfloat16()
        residual = torch.randn(5, 64, generator=torch.Generator().manual_seed(17)).bfloat16()
        x_before, residual_before = x.clone(), residual.clone()
        output, combined = norm(x, residual)
        expected_sum = (x.float() + residual.float()).bfloat16()
        self.assertTrue(torch.equal(combined, expected_sum))
        torch.testing.assert_close(output, self.reference(expected_sum, norm.weight, norm.variance_epsilon))
        self.assertTrue(torch.equal(x, x_before))
        self.assertTrue(torch.equal(residual, residual_before))
        wrong = self.reference(x.float() + residual.float(), norm.weight, norm.variance_epsilon).bfloat16()
        self.assertFalse(torch.equal(output, wrong), "fixture must distinguish unrounded residual fusion")

    def test_fp32_offset_detects_old_bf16_rounding(self):
        norm = self.make_norm(torch.bfloat16)
        x = torch.linspace(-2, 2, 64).repeat(3, 1).bfloat16()
        actual = norm(x)
        old_gamma = (1 + norm.weight).float()
        wrong = torch.nn.functional.rms_norm(x.float(), (64,), old_gamma, norm.variance_epsilon).bfloat16()
        self.assertFalse(torch.equal(actual, wrong), "fixture must expose low-precision gamma rounding")
        torch.testing.assert_close(actual, self.reference(x, norm.weight, norm.variance_epsilon))

    def test_all_text_norm_sites_and_fused_gamma(self):
        norm_calls = [
            n
            for n in ast.walk(TREE)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "MiniMaxM3RMSNorm"
        ]
        self.assertEqual(len(norm_calls), 9)
        self.assertFalse(any(isinstance(n, ast.Name) and n.id == "GemmaRMSNorm" for n in ast.walk(TREE)))
        weights = torch.tensor([0.003, -0.003], dtype=torch.bfloat16)
        owner = SimpleNamespace(q_norm=SimpleNamespace(weight=weights), k_norm=SimpleNamespace(weight=weights))
        keywords = [n for n in ast.walk(TREE) if isinstance(n, ast.keyword) and n.arg in ("q_weight", "k_weight")]
        self.assertEqual(len(keywords), 2)
        for keyword in keywords:
            value = eval(compile(ast.Expression(keyword.value), SOURCE_PATH, "eval"), {"self": owner})
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.equal(value, 1 + weights.float()))


if __name__ == "__main__":
    unittest.main()
