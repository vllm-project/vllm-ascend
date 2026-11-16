"""CPU regressions for unconditional support of validated Slot payloads."""
import ast
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[3]
BALANCE = ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py"


def load_payload_functions():
    tree = ast.parse(BALANCE.read_text())
    names = {"_PayloadSpec", "_carrier_shape", "_make_spec", "_infer_F",
             "_payload_specs", "slot_quant_enabled"}
    nodes = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "_MAX_CARRIER_ROW_ELEMS"
                      for t in n.targets)]
    namespace = {"__name__": __name__, "torch": torch, "dataclass": dataclass}
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(BALANCE), "exec"), namespace)
    return namespace


def quantized_layer():
    # CPU views reproduce dense per-expert strides; torch_npu format queries
    # are mocked. These are byte-layout checks, not hardware NZ validation.
    return types.SimpleNamespace(
        w13_weight=torch.arange(32, dtype=torch.uint8).reshape(2, 8, 2).transpose(1, 2),
        w2_weight=torch.arange(16, dtype=torch.uint8).reshape(2, 4, 2).transpose(1, 2),
        w13_weight_scale=torch.arange(32, dtype=torch.uint8).reshape(2, 1, 8, 2),
        w2_weight_scale=torch.arange(16, dtype=torch.uint8).reshape(2, 1, 4, 2),
    )


class SlotQuantPayloadTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_payload_functions()
        self.patch_npu = patch.dict(sys.modules, {
            "torch_npu": types.SimpleNamespace(get_npu_format=lambda ref: 2),
        })
        self.patch_npu.start()
        self.addCleanup(self.patch_npu.stop)

    def check_old_value(self, value):
        with patch.dict(os.environ):
            os.environ.pop("MOONEP_SLOT_QUANT", None)
            if value is not None:
                os.environ["MOONEP_SLOT_QUANT"] = value
            spec = importlib.util.spec_from_file_location("slot_payload_envs", ROOT / "vllm_ascend/envs.py")
            envs = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(envs)
            self.assertNotIn("MOONEP_SLOT_QUANT", envs.env_variables)
            self.assertFalse(hasattr(envs, "MOONEP_SLOT_QUANT"))
            self.assertTrue(self.ns["slot_quant_enabled"]())
            layer = quantized_layer()
            originals = {name: ref.clone() for name, ref in vars(layer).items()}
            self.assertEqual(self.ns["_infer_F"](layer, 4), 4)
            specs = self.ns["_payload_specs"](layer, 4, 4)
        attributes = ["w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"]
        self.assertEqual([s.name for s in specs], ["w13", "w13_scale", "w2", "w2_scale"])
        self.assertEqual([s.is_nz for s in specs], [True, False, True, False])
        for payload, attribute in zip(specs, attributes):
            ref = getattr(layer, attribute)
            self.assertIs(payload.ref, ref)
            self.assertEqual(ref.dtype, torch.uint8)
            self.assertTrue(torch.equal(ref, originals[attribute]))
            self.assertEqual(payload.row_bytes * 2, payload.nbytes)
            self.assertEqual(payload.carrier[0] * payload.carrier[1] * 2, payload.row_bytes)

    def test_unset_preserves_quantized_weights_and_scales(self):
        self.check_old_value(None)

    def test_old_zero_no_longer_rejects_quantized_payloads(self):
        self.check_old_value("0")

    def test_old_one_preserves_quantized_weights_and_scales(self):
        self.check_old_value("1")

    def test_invalid_old_value_is_ignored(self):
        self.check_old_value("invalid")

    def test_bf16_keeps_two_original_payloads_without_scales(self):
        layer = types.SimpleNamespace(
            w13_weight=torch.arange(64).reshape(2, 4, 8).bfloat16(),
            w2_weight=torch.arange(32).reshape(2, 4, 4).bfloat16())
        with patch.dict(os.environ, {"MOONEP_SLOT_QUANT": "0"}):
            self.assertEqual(self.ns["_infer_F"](layer, 4), 4)
            specs = self.ns["_payload_specs"](layer, 4, 4)
        self.assertEqual([s.name for s in specs], ["w13", "w2"])
        self.assertEqual([s.carrier for s in specs], [(4, 8), (4, 4)])
        for spec, ref in zip(specs, [layer.w13_weight, layer.w2_weight]):
            self.assertIs(spec.ref, ref)
            self.assertFalse(spec.is_nz)
            self.assertEqual(spec.ref.dtype, torch.bfloat16)

    def test_missing_scales_are_still_rejected(self):
        for attribute in ["w13_weight_scale", "w2_weight_scale"]:
            with self.subTest(attribute=attribute):
                layer = quantized_layer()
                delattr(layer, attribute)
                with self.assertRaisesRegex(NotImplementedError, "requires both"):
                    self.ns["_payload_specs"](layer, 4, 4)
                if attribute == "w13_weight_scale":
                    with self.assertRaisesRegex(NotImplementedError, "requires w13_weight_scale"):
                        self.ns["_infer_F"](layer, 4)

    def test_unsupported_weight_element_size_is_still_rejected(self):
        for attribute in ["w13_weight", "w2_weight"]:
            with self.subTest(attribute=attribute):
                layer = quantized_layer()
                setattr(layer, attribute, getattr(layer, attribute).float())
                with self.assertRaisesRegex(NotImplementedError, "unsupported .* layout dtype"):
                    self.ns["_payload_specs"](layer, 4, 4)

    def test_per_expert_padding_is_still_rejected(self):
        layer = quantized_layer()
        layer.w13_weight = torch.zeros(2, 3, 8, dtype=torch.uint8)[:, :2, :]
        with self.assertRaisesRegex(NotImplementedError, "per-expert padding"):
            self.ns["_payload_specs"](layer, 4, 4)

    def test_storage_boundary_is_still_checked(self):
        # Invalid storage metadata cannot be constructed as a real CPU tensor.
        ref = types.SimpleNamespace(
            shape=(2, 2, 8), stride=lambda dim: 16, element_size=lambda: 1,
            numel=lambda: 32, storage_offset=lambda: 1,
            untyped_storage=lambda: types.SimpleNamespace(nbytes=lambda: 32))
        with self.assertRaisesRegex(NotImplementedError, "payload runs past its storage"):
            self.ns["_make_spec"]("w13", ref, is_nz=True)

    def test_carrier_byte_count_and_ub_bound_are_preserved(self):
        rows, cols = self.ns["_carrier_shape"](3 * 32768 * 2)
        self.assertEqual(rows * cols * 2, 3 * 32768 * 2)
        self.assertLessEqual(cols, 32768)
        for invalid in [0, 3]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "positive even"):
                self.ns["_carrier_shape"](invalid)

    def test_shape_and_scale_consistency_checks_remain(self):
        cases = [
            ("unexpected w13 shape", {"w13_weight": torch.zeros(2, 8, dtype=torch.uint8)}),
            ("bf16 w13 should", {"w13_weight": torch.zeros(2, 5, 8, dtype=torch.bfloat16)}),
            ("bf16 w2 should", {"w13_weight": torch.zeros(2, 4, 8, dtype=torch.bfloat16)}),
            ("quantized w13 should", {"w13_weight": torch.zeros(2, 3, 8, dtype=torch.uint8)}),
            ("disagrees", {"w13_weight_scale": torch.zeros(2, 1, 6, 2, dtype=torch.uint8)}),
        ]
        for expected, replacements in cases:
            with self.subTest(expected=expected):
                layer = quantized_layer()
                vars(layer).update(replacements)
                with self.assertRaisesRegex(NotImplementedError, expected):
                    self.ns["_infer_F"](layer, 4)

    def test_removed_runtime_registration_and_references(self):
        for relative in ["vllm_ascend/envs.py", "vllm_ascend/ops/fused_moe/moonep_balance.py",
                         "vllm_ascend/ops/fused_moe/moonep_comm.py"]:
            tree = ast.parse((ROOT / relative).read_text())
            self.assertFalse(any(isinstance(n, ast.Name) and n.id == "_slot_quant" for n in ast.walk(tree)))
            self.assertFalse(any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                 and "MOONEP_SLOT_QUANT" in n.value for n in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
