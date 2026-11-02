# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reject the observed INT8/MXFP checkpoint mismatch before loading weights."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("quant_type", ["W4A8_MXFP", "W4A8", None])
def test_reference_checkpoint_quantization_contract(tmp_path, quant_type):
    root = Path(__file__).resolve().parents[2]
    validate = runpy.run_path(str(root / "tools/hisparse/serve_reproduction.py"))["validate_checkpoint"]
    contract = json.loads((root / "tools/hisparse/reference_configuration.json").read_text())["checkpoint_contract"]
    if quant_type is not None:
        data = {
            "model_quant_type": quant_type,
            "group_size": 32,
            "model.layers.0.self_attn.q_a_proj.weight": "W8A8_MXFP8" if quant_type == "W4A8_MXFP" else "W8A8_DYNAMIC",
        }
        (tmp_path / "quant_model_description.json").write_text(json.dumps(data))
    if quant_type == "W4A8_MXFP":
        validate(str(tmp_path), contract)
    else:
        with pytest.raises(ValueError):
            validate(str(tmp_path), contract)


@pytest.mark.parametrize("arm", ["on", "off"])
@pytest.mark.parametrize("profile", ["a5_c8_62592", "a5_c8_main"])
def test_profile_arguments_preserve_reference_and_current_checkpoint_policy(arm, profile):
    root = Path(__file__).resolve().parents[2]
    build = runpy.run_path(str(root / "tools/hisparse/serve_reproduction.py"))["arguments_for_profile"]
    config = json.loads((root / "tools/hisparse/reference_configuration.json").read_text())
    saved = json.dumps(config, sort_keys=True)
    argv = build(config, arm, profile)
    assert json.dumps(config, sort_keys=True) == saved
    additional = json.loads(argv[argv.index("--additional-config") + 1])
    if profile == "a5_c8_main":
        assert "sparse_li_c8_layer_policy" not in additional
        assert additional["weight_nz_mode"] == 1
    else:
        assert additional == {"sparse_li_c8_layer_policy": "all"}
    assert argv[argv.index("--max-num-seqs") + 1] == "32"
    assert argv[argv.index("--speculative-config") + 1] == config[arm][config[arm].index("--speculative-config") + 1]
