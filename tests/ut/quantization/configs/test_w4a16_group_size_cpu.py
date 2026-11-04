# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-only regression: compressed-tensors W4A16 retains per-layer group_size.

Run directly with Python 3.10+; torch, torch_npu and ACL are not required.
The AST loader executes production methods unchanged with synthetic tensors.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]


def load_class(relative_path, name, methods, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class QuantizationArgs:
    @staticmethod
    def model_validate(value):
        return SimpleNamespace(**value)


class GroupSizeRegression(unittest.TestCase):
    def test_checkpoint_group_size_controls_all_scale_shapes(self):
        tensor_api = mock.MagicMock(name="torch")
        tensor_api.empty.side_effect = lambda *shape, **kwargs: SimpleNamespace(shape=shape)
        tensor_api.zeros.side_effect = tensor_api.empty.side_effect
        current = SimpleNamespace(quant_config=None, use_v2_model_runner=True)
        ns = {
            "__package__": "vllm_ascend.quantization.configs",
            "torch": tensor_api,
            "get_current_vllm_config": lambda: current,
            "cast": cast,
            "QuantizationArgs": QuantizationArgs,
            "is_activation_quantization_format": lambda value: False,
            "QuantizationType": SimpleNamespace(INT="int"),
            "QuantizationStrategy": SimpleNamespace(GROUP=SimpleNamespace(value="group")),
            "CompressionFormat": SimpleNamespace(
                mxfp8_quantized=SimpleNamespace(value="mxfp8-quantized"),
                mxfp4_pack_quantized=SimpleNamespace(value="mxfp4-pack-quantized"),
                mixed_precision=SimpleNamespace(value="mixed-precision"),
            ),
        }
        scheme_cls = load_class(
            "vllm_ascend/quantization/methods/wna16/w4a16.py",
            "AscendW4A16FusedMoEMethod",
            {"__init__", "get_dynamic_quant_param"},
            ns,
        )
        config_cls = load_class(
            "vllm_ascend/quantization/configs/compressed_tensors_config.py",
            "AscendCompressedTensorsConfig",
            {
                "__init__",
                "from_config",
                "_quantization_scheme_map_from_config",
                "_create_scheme_for_layer_type",
                "_detect_quant_type",
                "_is_w4a16",
            },
            ns,
        )
        methods_module = ModuleType("vllm_ascend.quantization.methods")
        methods_module.get_scheme_class = lambda quant_type, layer_type: scheme_cls
        modules = {"torch_npu": mock.MagicMock(), "acl": mock.MagicMock(), methods_module.__name__: methods_module}
        with mock.patch.dict(sys.modules, modules):
            for group_size in (32, 64, 128):
                with self.subTest(group_size=group_size):
                    config = config_cls.from_config(
                        {
                            "format": "pack-quantized",
                            "config_groups": {
                                "group_0": {
                                    "targets": ["Linear"],
                                    "weights": {
                                        "num_bits": 4,
                                        "type": "int",
                                        "strategy": "group",
                                        "group_size": group_size,
                                        "dynamic": False,
                                        "symmetric": True,
                                    },
                                }
                            },
                        }
                    )
                    current.quant_config = config
                    weight_quant = config.target_scheme_map["Linear"]["weights"]
                    scheme = config._create_scheme_for_layer_type(weight_quant, None, None, "moe")
                    tensors = scheme.get_dynamic_quant_param(2, 256, 512, "float16")
                    self.assertEqual(scheme.group_size, group_size)
                    self.assertEqual(tensors["w13_weight_scale"].shape, (2, 512, 512 // group_size))
                    self.assertEqual(tensors["w2_weight_scale"].shape, (2, 512, 256 // group_size))
                    self.assertEqual(tensors["w13_weight_offset"].shape, tensors["w13_weight_scale"].shape)

    def test_modelslim_constructor_keeps_legacy_group_size(self):
        ns = {
            "get_current_vllm_config": lambda: SimpleNamespace(
                quant_config=SimpleNamespace(quant_description={"group_size": 64}), use_v2_model_runner=True
            )
        }
        cls = load_class(
            "vllm_ascend/quantization/methods/wna16/w4a16.py", "AscendW4A16FusedMoEMethod", {"__init__"}, ns
        )
        self.assertEqual(cls().group_size, 64)


if __name__ == "__main__":
    unittest.main()
