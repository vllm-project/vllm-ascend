# Copyright (c) 2026 ouyangqing. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Qwen3.5 MoE hybrid xlite adapter (CPU-only)."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

# The real xlite._C probes the NPU runtime at import time and *aborts the
# process* (not an ImportError) on CPU-only runners, so it must only be
# imported when a real NPU is present. Otherwise install the stub below.
_HAVE_XLITE = False
try:
    import subprocess as _sp

    _sp.run(["npu-smi", "info"], capture_output=True, check=True)
    import xlite._C  # noqa: F401

    _HAVE_XLITE = True
except Exception:  # pragma: no cover - CPU-only runners take the stub path
    _HAVE_XLITE = False
    import enum

    class _AttnKind(enum.IntEnum):
        MHA = 0
        MLA = 1
        HYBRID = 2
        DSA = 3

    xlite_mod = types.ModuleType("xlite")
    xlite_c = types.ModuleType("xlite._C")
    # setattr keeps the dynamic ModuleType attributes invisible to mypy.
    for _name, _value in {
        "AttnMHA": _AttnKind.MHA,
        "AttnMLA": _AttnKind.MLA,
        "AttnHybrid": _AttnKind.HYBRID,
        "AttnDSA": _AttnKind.DSA,
        "AttnMeta": type("AttnMeta", (), {}),
        "AttnMetaV2": type("AttnMetaV2", (), {}),
        # required by vllm_ascend/xlite/utils.py
        "Model": type("Model", (), {}),
        "ModelConfig": type("ModelConfig", (), {}),
        "Runtime": MagicMock,
        "ScoringFuncSoftmax": 0,
        "ScoringFuncSigmoid": 1,
    }.items():
        setattr(xlite_c, _name, _value)
    for _name in (
        "graph_capture_begin",
        "graph_capture_end",
        "graph_destroy",
        "graph_exec",
        "set_gdn_chunk_callback",
    ):
        setattr(xlite_c, _name, MagicMock())
    sys.modules.setdefault("xlite", xlite_mod)
    sys.modules.setdefault("xlite._C", xlite_c)

import vllm_ascend.xlite.xlite as xlite_mod_under_test  # noqa: E402
from vllm_ascend.xlite.xlite import (  # noqa: E402
    Qwen3_5MoeXliteModel,
    _architecture_strategy_map,
)


def _hf_text_config(**overrides):
    base = dict(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        intermediate_size=128,
        rope_theta=10000.0,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        # MoE
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _adapter_self(hf):
    """Real instance (bypassing __init__) so zero-arg super() works."""
    self = object.__new__(Qwen3_5MoeXliteModel)
    self.xlite_config = SimpleNamespace(
        hidden_size=64, moe_ep_size=1, moe_tp_size=1, moe_intermediate_size=64, experts_weight_transpose=True
    )
    self.xlite_model = SimpleNamespace()

    self.rottary_embed = None
    self._decoder_layer_mlp_module = "mlp"
    self.is_tensor_nz = lambda t: False
    self.quantization = False
    self.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512, max_num_seqs=8),
        parallel_config=SimpleNamespace(data_parallel_size=1, enable_expert_parallel=False),
        model_config=SimpleNamespace(
            max_model_len=1024, dtype=torch.bfloat16, hf_text_config=hf, hf_config=SimpleNamespace()
        ),
        cache_config=SimpleNamespace(block_size=128),
        speculative_config=None,
        quant_config=None,
    )
    return self


def test_qwen35_moe_architectures_registered():
    assert _architecture_strategy_map.get("Qwen3_5MoeForCausalLM") is Qwen3_5MoeXliteModel
    assert _architecture_strategy_map.get("Qwen3_5MoeForConditionalGeneration") is Qwen3_5MoeXliteModel
    # MoE adapter reuses the dense Qwen3.5 hybrid attention path.
    from vllm_ascend.xlite.xlite import Qwen3_5XliteModel

    assert issubclass(Qwen3_5MoeXliteModel, Qwen3_5XliteModel)


def test_build_model_config_moe_fields(monkeypatch):
    monkeypatch.setattr(xlite_mod_under_test, "get_tensor_model_parallel_world_size", lambda: 1)
    ascend_cfg = SimpleNamespace(
        weight_nz_mode=0,
        xlite_graph_config=SimpleNamespace(full_mode=True),
    )
    monkeypatch.setattr(xlite_mod_under_test, "get_ascend_config", lambda: ascend_cfg)

    self = _adapter_self(_hf_text_config())
    Qwen3_5MoeXliteModel._build_model_config(self)

    cfg = self.xlite_config
    # Sparse MoE: dense layers are replaced by routed experts.
    assert cfg.n_dense_layers == 0
    assert cfg.n_routed_experts == 8
    assert cfg.n_act_experts == 2
    assert cfg.moe_intermediate_size == 64
    assert cfg.n_shared_experts == 1  # shared_expert_intermediate_size > 0
    assert cfg.attn_type == xlite_mod_under_test.AttnHybrid


def test_build_model_config_without_shared_experts(monkeypatch):
    monkeypatch.setattr(xlite_mod_under_test, "get_tensor_model_parallel_world_size", lambda: 1)
    ascend_cfg = SimpleNamespace(
        weight_nz_mode=0,
        xlite_graph_config=SimpleNamespace(full_mode=True),
    )
    monkeypatch.setattr(xlite_mod_under_test, "get_ascend_config", lambda: ascend_cfg)

    self = _adapter_self(_hf_text_config(shared_expert_intermediate_size=0))
    Qwen3_5MoeXliteModel._build_model_config(self)

    assert self.xlite_config.n_shared_experts == 0


def test_load_moe_weights_binds_shared_expert_gate(monkeypatch):
    """y = routed + sigmoid(se_gate(x)) * shared(x): the gate must be bound."""
    requested: list[str] = []

    def fake_get_layer_weights(layers, *paths, **kwargs):
        for p in paths:
            requested.append(p)
        return [torch.zeros(2, 4)]

    monkeypatch.setattr(xlite_mod_under_test, "get_layer_weights", fake_get_layer_weights)
    matmul_requests: list[tuple[str, str]] = []

    def fake_init_matmul_weights(layers, name, path):
        matmul_requests.append((name, path))

    self = _adapter_self(_hf_text_config())
    self.init_matmul_weights = fake_init_matmul_weights
    Qwen3_5MoeXliteModel._load_moe_weights(self, layers=[SimpleNamespace()])

    assert "mlp.shared_expert_gate.weight" in requested
    assert ("se_up_gate", "mlp.shared_expert.gate_up_proj") in matmul_requests
    assert ("se_down", "mlp.shared_expert.down_proj") in matmul_requests
    assert "mlp.gate.weight" in requested
