# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the real Flash checkpoint preflight."""

import importlib.util
import json
from pathlib import Path


def test_real_checkpoint_preflight(tmp_path):
    # Import the lightweight checks without loading the NPU E2E conftest.
    path = Path(__file__).resolve().parents[2] / "e2e/common/glm53flash_pd_checks.py"
    spec = importlib.util.spec_from_file_location("flash_pd_checks", path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    config = {
        "text_config": {
            "num_hidden_layers": 9,
            "num_nextn_predict_layers": 1,
            "hidden_size": 4096,
            "n_routed_experts": 288,
            "vocab_size": 154880,
            "linear_attn_config": {"kda_layers": [0, 1, 2, 4, 5, 6, 8], "full_attn_layers": [3, 7]},
        }
    }
    files = {
        "config.json": config,
        "quant_model_description.json": {"model.layers.0.weight": "W8A8_MXFP8"},
        "model.safetensors.index.json": {
            "metadata": {"total_size": 2},
            "weight_map": {"tensor_a": "part-1.safetensors", "tensor_b": "part-2.safetensors"},
        },
        "tokenizer.json": {},
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    for name in ("part-1.safetensors", "part-2.safetensors"):
        (tmp_path / name).write_bytes(b"x")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    checks.check_checkpoint(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
