# SPDX-License-Identifier: Apache-2.0
"""The ND-only policy must not select an NZ-only fused weight-preparation path."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_fused_preprocess_honors_existing_weight_layout_policy(mode):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_v1.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_fused_preprocess_type")
    quant = type("DynamicQuant", (), {})
    preprocess = SimpleNamespace(PROLOG_V3=object(), MLAPO=object())
    namespace = {
        "get_ascend_config": lambda: SimpleNamespace(weight_nz_mode=mode),
        "PreprocessType": preprocess,
        "AscendW8A8DynamicLinearMethod": quant,
        "AscendW8A8MXFP8DynamicLinearMethod": type("MXQuant", (), {}),
        "AscendW8A8LinearMethod": type("StaticQuant", (), {}),
    }
    method.returns = None
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    owner = SimpleNamespace(
        fused_qkv_a_proj=object(),
        _get_layer_quant_method=lambda layer: quant(),
        _get_fused_type_unsupported_reasons=lambda kind: [],
    )
    result = namespace[method.name](owner)
    assert result is (None if mode == 0 else preprocess.PROLOG_V3)
