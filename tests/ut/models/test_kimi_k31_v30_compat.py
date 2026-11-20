# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for interfaces that differ between main and v30."""

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_ascend.models.kimi_k3 import kimi_k3
from vllm_ascend.patch.platform import patch_kv_cache_utils
from vllm_ascend.utils import vllm_version_is

ROOT = Path(__file__).resolve().parents[3]


def test_dsa_gate_registers_warmup_after_backend_construction(monkeypatch):
    impl = SimpleNamespace(g_proj=None, local_num_heads=2)

    def construct_backend(self, **kwargs):
        nn.Module.__init__(self)
        self.mla_attn = SimpleNamespace(mla_attn=SimpleNamespace(impl=impl))

    monkeypatch.setattr(kimi_k3.DeepseekV2MLAAttention, "__init__", construct_backend)
    monkeypatch.setattr(kimi_k3, "ColumnParallelLinear", lambda n, m, **kwargs: nn.Linear(n, m, bias=False))
    register = Mock()
    monkeypatch.setattr(kimi_k3, "register_flash_attention_gate_warmup", register)
    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        max_position_embeddings=128,
        mla_use_output_gate=True,
    )
    runtime = SimpleNamespace(cache_config=None, quant_config=None, model_config=SimpleNamespace(dtype=torch.bfloat16))
    attention = kimi_k3.AscendKimiDSAAttention(config, runtime, "model.layers.0.self_attn", None)
    assert attention.impl.g_proj is attention.g_proj
    register.assert_called_once_with(torch.bfloat16, 8)


@pytest.mark.skipif(
    not (vllm_version_is("0.30.0") or vllm_version_is("0.30.0")), reason="v30's two-argument admission API"
)
def test_v30_k31_admission_calls_paired_accounting(monkeypatch):
    groups = [object()]
    runtime = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k31")))
    paired = Mock(return_value=123456)
    monkeypatch.setattr(patch_kv_cache_utils, "paired_sfa_max_memory", paired)
    assert (
        patch_kv_cache_utils.vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups(runtime, groups) == 123456
    )
    paired.assert_called_once_with(runtime, groups)


def test_launcher_arguments_parse_with_real_v30_engine(tmp_path):
    launcher = runpy.run_path(str(ROOT / "examples/kimi_k31_950.py"))
    args = argparse.Namespace(
        model=tmp_path,
        tp=8,
        host="127.0.0.1",
        port=8000,
        max_model_len=4096,
        memory_utilization=0.9,
        multimodal=False,
    )
    command = launcher["build_command"](args)
    parser = make_arg_parser(FlexibleArgumentParser())
    parsed = parser.parse_args(["--model", command[2], *command[3:]])
    engine = EngineArgs.from_cli_args(parsed)
    assert engine.attention_config.indexer_kv_dtype == "fp8"
    assert engine.enforce_eager
    assert engine.tensor_parallel_size == 8
    assert engine.language_model_only
    assert not engine.mm_device_do_normalize


def test_launcher_rejects_incomplete_sharded_checkpoint(tmp_path):
    launcher = runpy.run_path(str(ROOT / "examples/kimi_k31_950.py"))
    config = {
        "model_type": "kimi_k31",
        "architectures": ["KimiK31ForConditionalGeneration"],
        "text_config": {
            "num_attention_heads": 16,
            "num_hidden_layers": 4,
            "linear_attn_config": {"num_heads": 16, "kda_layers": [1, 2, 3], "full_attn_layers": [4]},
            "dsa_config": {"dsa_layers": [4]},
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "quant_model_description.json").write_text(json.dumps({"weight": "W4A8_MXFP"}))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="missing weight shards"):
        launcher["validate_checkpoint"](tmp_path, 8)
