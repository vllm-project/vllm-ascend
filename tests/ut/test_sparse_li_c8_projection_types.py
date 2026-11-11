# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run sparse indexer type-selection contracts without loading model weights.

The real AscendConfig derivation assignments and parser methods are extracted
so this regression also runs on development hosts without PyTorch or vLLM.
Only vLLM's layer-index utility is substituted when vLLM is unavailable.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from vllm.model_executor.models.utils import extract_layer_index
except ImportError:

    def extract_layer_index(name):
        return int(name.split("layers.", 1)[1].split(".", 1)[0])


class _RemoveLayerIndexImport(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == "vllm.model_executor.models.utils":
            assert [alias.name for alias in node.names] == ["extract_layer_index"]
            return None
        return node


def load_sparse_config_type(use_int8_adaptor=True):
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ascend_config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendConfig")
    derive = next(
        node for node in config.body if isinstance(node, ast.FunctionDef) and node.name == "derive_and_validate"
    )
    fields = {
        "enable_sparse_li_c8",
        "enable_sparse_li_c4",
        "_sparse_li_c8_layer_ids",
        "_sparse_li_c4_layer_ids",
        "_sparse_li_layer_filter_enabled",
    }
    assignments = [
        node
        for node in derive.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr in fields for target in ast.walk(node.targets[0]))
    ]
    assert len(assignments) == 5, "Sparse LI derivation changed; update the focused harness"
    configure = ast.parse(
        "def configure(self, vllm_config, use_sparse=True):\n    vc = vllm_config\n    quant_config = vc.quant_config\n"
    ).body[0]
    assert isinstance(configure, ast.FunctionDef)
    configure.body.extend(assignments)
    methods = {
        "_has_sparse_li_layer_config",
        "_parse_sparse_li_layers_from_quant_config",
        "_is_sparse_li_layer",
        "is_sparse_li_c8_layer",
        "is_sparse_li_c4_layer",
    }
    kept_methods = [node for node in config.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    config.body = [configure, *kept_methods]
    config.decorator_list = []
    module = _RemoveLayerIndexImport().visit(ast.Module(body=[config], type_ignores=[]))
    namespace = {
        "Any": Any,
        "extract_layer_index": extract_layer_index,
        "HardwareCapability": SimpleNamespace(NATIVE_INT8_SPARSE=object()),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda capability: use_int8_adaptor),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["AscendConfig"]


def make_config(description, *, dtype="int8", sparse=True, use_int8_adaptor=True):
    config = load_sparse_config_type(use_int8_adaptor)()
    config.configure(
        SimpleNamespace(
            quant_config=SimpleNamespace(quant_description=description),
            attention_config=SimpleNamespace(indexer_kv_dtype=dtype),
        ),
        use_sparse=sparse,
    )
    return config


class TestSparseLIC8ProjectionTypes(unittest.TestCase):
    def test_a5_keeps_original_projection_and_explicit_type_resolution(self):
        prefix = "model.layers.3.self_attn.indexer"
        config = make_config({f"{prefix}.wq_b.weight": "W8A8_DYNAMIC"}, use_int8_adaptor=False)
        self.assertFalse(config.is_sparse_li_c8_layer(f"{prefix}.k_cache"))
        config = make_config(
            {f"{prefix}.quant_type": "FLOAT", f"{prefix}.wq_b.weight": "W8A8_MXFP8"}, use_int8_adaptor=False
        )
        self.assertTrue(config.is_sparse_li_c8_layer(f"{prefix}.k_cache"))

    def test_delivered_a3_checkpoint_selects_all_24_dynamic_int8_indexers(self):
        layers = [*range(3, 92, 4), 92]
        description = {
            f"language_model.model.layers.{layer}.self_attn.indexer.wq_b.weight": "W8A8_DYNAMIC" for layer in layers
        }
        config = make_config(description)
        selected = [
            layer
            for layer in layers
            if config.is_sparse_li_c8_layer(f"language_model.model.layers.{layer}.self_attn.indexer.k_cache")
        ]
        self.assertEqual(selected, layers)
        self.assertEqual(len(config._sparse_li_c8_layer_ids), 24)
        self.assertEqual(config._sparse_li_c4_layer_ids, set())

    def test_float_rollbacks_and_unlisted_layers_remain_unquantized(self):
        config = make_config(
            {
                "model.layers.3.self_attn.indexer.wq_b.weight": "W8A8_DYNAMIC",
                "model.layers.7.self_attn.indexer.wq_b.weight": "FLOAT",
                "model.layers.11.self_attn.indexer.quant_type": "BF16",
            }
        )
        self.assertTrue(config.is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache"))
        for layer in (7, 11, 15):
            with self.subTest(layer=layer):
                self.assertFalse(config.is_sparse_li_c8_layer(f"model.layers.{layer}.self_attn.indexer.k_cache"))

    def test_existing_c8_aliases_remain_supported(self):
        for suffix, quant_type in (("quant_type", "INT8_DYNAMIC"), ("wq_b.weight", "W8A8_MXFP8")):
            for dtype in ("int8", "fp8"):
                with self.subTest(quant_type=quant_type, dtype=dtype):
                    config = make_config({f"model.layers.3.self_attn.indexer.{suffix}": quant_type}, dtype=dtype)
                    self.assertTrue(config.is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache"))

    def test_explicit_indexer_type_takes_precedence_over_projection_type(self):
        prefix = "model.layers.3.self_attn.indexer"
        for explicit_type in ("FLOAT", "BF16"):
            for projection_type, dtype in (("W8A8_DYNAMIC", "int8"), ("W8A8_MXFP8", "mxfp4")):
                for explicit_first in (False, True):
                    with self.subTest(explicit_type=explicit_type, dtype=dtype, explicit_first=explicit_first):
                        entries = [(f"{prefix}.quant_type", explicit_type), (f"{prefix}.wq_b.weight", projection_type)]
                        description = dict(entries if explicit_first else reversed(entries))
                        config = make_config(description, dtype=dtype)
                        self.assertFalse(config.is_sparse_li_c8_layer(f"{prefix}.k_cache"))
                        self.assertFalse(config.is_sparse_li_c4_layer(f"{prefix}.k_cache"))
        config = make_config({f"{prefix}.quant_type": "INT8_DYNAMIC", f"{prefix}.wq_b.weight": "FLOAT"})
        self.assertTrue(config.is_sparse_li_c8_layer(f"{prefix}.k_cache"))

    def test_projection_type_does_not_enable_an_unrequested_cache_dtype(self):
        description = {"model.layers.3.self_attn.indexer.wq_b.weight": "W8A8_DYNAMIC"}
        for dtype in ("auto", "bfloat16", "fp8", "mxfp4"):
            with self.subTest(dtype=dtype):
                config = make_config(description, dtype=dtype)
                self.assertFalse(config.is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache"))
                self.assertFalse(config.is_sparse_li_c4_layer("model.layers.3.self_attn.indexer.k_cache"))
        self.assertFalse(
            make_config(description, sparse=False).is_sparse_li_c8_layer("model.layers.3.self_attn.indexer")
        )

    def test_c4_filter_and_metadata_absent_behavior_are_unchanged(self):
        config = make_config({"model.layers.3.self_attn.indexer.wq_b.weight": "W8A8_MXFP8"}, dtype="mxfp4")
        self.assertTrue(config.is_sparse_li_c4_layer("model.layers.3.self_attn.indexer.k_cache"))
        self.assertFalse(config.is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache"))
        for description in ({}, None, {"indexer_quant_type": "INT8_DYNAMIC"}):
            with self.subTest(description=description):
                self.assertTrue(
                    make_config(description).is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache")
                )


if __name__ == "__main__":
    unittest.main()
