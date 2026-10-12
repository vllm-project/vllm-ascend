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
"""Unit tests for the Qwen3.5 hybrid (MHA+GDN) xlite adapter.

All tests run on CPU. The xlite runtime extension (``xlite._C``) is not part
of the CI dependencies, so a minimal stub is installed when it is missing;
only pure-Python adapter logic is exercised.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
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

from vllm_ascend.xlite.xlite import (  # noqa: E402
    Qwen3_5XliteModel,
    _architecture_strategy_map,
    _gemma_norm_to_xlite,
    _pack_mha_qkv_bias,
    _pack_mha_qkv_weight,
    _resolve_full_attention_interval,
)


def test_gemma_norm_to_xlite_adds_one():
    weight = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = _gemma_norm_to_xlite(weight)
    assert torch.allclose(out, torch.tensor([2.0, 3.0]))
    assert out.dtype == weight.dtype
    assert out.is_contiguous()


def test_pack_mha_qkv_weight_layout():
    head_dim, num_heads, num_kv_heads = 4, 2, 1
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    # vLLM interleaved layout: per-head [Q | Gate] for Q rows, then K, then V.
    qkv = torch.arange(q_size * 2 + kv_size * 2, dtype=torch.float32).unsqueeze(1)

    packed = _pack_mha_qkv_weight(qkv, head_dim, num_heads, num_kv_heads)
    # xlite layout: [Q | K | V | Gate]
    q_w, k_w, v_w, g_w = (
        packed[:q_size],
        packed[q_size : q_size + kv_size],
        packed[q_size + kv_size : q_size + 2 * kv_size],
        packed[q_size + 2 * kv_size :],
    )
    q_gate = qkv[: q_size * 2].view(num_heads, 2, head_dim, -1)
    assert torch.equal(q_w, q_gate[:, 0].reshape(q_size, -1))
    assert torch.equal(k_w, qkv[q_size * 2 : q_size * 2 + kv_size])
    assert torch.equal(v_w, qkv[q_size * 2 + kv_size :])
    assert torch.equal(g_w, q_gate[:, 1].reshape(q_size, -1))


def test_pack_mha_qkv_bias_reordered():
    head_dim, num_heads, num_kv_heads = 2, 2, 1
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    bias = torch.randn(q_size * 2 + kv_size * 2)

    packed_bias = _pack_mha_qkv_bias(bias, head_dim, num_heads, num_kv_heads)

    assert packed_bias.shape[0] == q_size + 2 * kv_size + q_size
    q_gate_b = bias[: q_size * 2].view(num_heads, 2, head_dim)
    assert torch.equal(packed_bias[:q_size], q_gate_b[:, 0].reshape(-1))
    assert torch.equal(packed_bias[q_size : q_size + kv_size], bias[q_size * 2 : q_size * 2 + kv_size])
    assert torch.equal(packed_bias[-q_size:], q_gate_b[:, 1].reshape(-1))


def test_pack_mha_qkv_weight_rejects_unexpected_rows():
    with pytest.raises(ValueError, match="Unexpected fused QKV rows"):
        _pack_mha_qkv_weight(torch.zeros(7, 4), head_dim=4, num_heads=1, num_kv_heads=1)


def test_resolve_full_attention_interval_prefers_explicit_field():
    hf = SimpleNamespace(full_attention_interval=6)
    assert _resolve_full_attention_interval(hf) == 6


def test_resolve_full_attention_interval_derived_from_layer_types():
    hf = SimpleNamespace(
        full_attention_interval=None,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
    )
    # Single full-attention layer at index 1 -> every (1+1)-th layer is MHA.
    assert _resolve_full_attention_interval(hf) == 2


def test_resolve_full_attention_interval_uses_gap_between_full_layers():
    hf = SimpleNamespace(
        full_attention_interval=None,
        layer_types=[
            "full_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
    )
    # A model may start with a full layer (index 0); the period must come from
    # the distance between the first two full layers (3), not idx+1 (which
    # would wrongly yield 1).
    assert _resolve_full_attention_interval(hf) == 3


def test_resolve_full_attention_interval_defaults_to_four():
    assert _resolve_full_attention_interval(SimpleNamespace()) == 4


def test_qwen35_hybrid_architectures_registered():
    assert _architecture_strategy_map.get("Qwen3_5ForCausalLM") is Qwen3_5XliteModel
    assert _architecture_strategy_map.get("Qwen3_5ForConditionalGeneration") is Qwen3_5XliteModel
    assert Qwen3_5XliteModel._attn_metadata_type is not None


def _hybrid_self():
    """Build a minimal `self` for _adapt_hybrid_kv_caches (batch-major)."""
    cfg = SimpleNamespace(
        n_layers=2,
        full_attention_interval=2,
        max_batch_size=4,
        def_tp_size=1,
        linear_num_v_heads=2,
        linear_num_k_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        block_size=16,
        head_dim=8,
        n_kv_heads=2,
    )
    # The wrapper reads the role mask from the adapter (built in
    # `_build_model_config`), not from its own attributes.
    adapter = SimpleNamespace(xlite_config=cfg, _full_attention_mask=[False, True])  # layer 0 GDN, layer 1 MHA
    return SimpleNamespace(
        adapter_xlite_model=adapter,
        hidden_states=torch.zeros(1, 16, dtype=torch.float32),
        device="cpu",
        _xlite_hybrid_state_refs=[],
        # referenced by the slot-major adaptation log
        _xlite_native_gdn_groups=[],
    )


def test_adapt_hybrid_kv_caches_layout():
    from vllm_ascend.xlite.xlite import XliteWrapper

    self = _hybrid_self()
    cfg = self.adapter_xlite_model.xlite_config
    conv_dim = cfg.linear_num_k_heads * cfg.linear_key_head_dim * 2 + cfg.linear_num_v_heads * cfg.linear_value_head_dim
    # Layer 0 is GDN (interval=2), layer 1 is full attention.
    full_kv = [
        torch.zeros(3, cfg.block_size, cfg.n_kv_heads, cfg.head_dim),
        torch.zeros(3, cfg.block_size, cfg.n_kv_heads, cfg.head_dim),
    ]
    caches = [
        [],  # unused for the GDN layer in the batch-major layout
        full_kv,
    ]
    adapted = XliteWrapper._adapt_hybrid_kv_caches(self, caches)

    assert len(adapted) == 2
    conv, ssm = adapted[0]
    assert conv.shape == (cfg.max_batch_size, conv_dim, cfg.linear_conv_kernel_dim)
    assert ssm.shape == (cfg.max_batch_size, cfg.linear_num_v_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    assert adapted[1][0].data_ptr() == caches[1][0].data_ptr()  # full layers pass through
    # slot-major: each GDN layer keeps persistent (conv, ssm) + batch (conv, ssm)
    assert len(self._xlite_hybrid_state_refs) == 4
    conv_p, ssm_p = self._xlite_hybrid_persistent[0]
    assert conv_p.shape[0] == cfg.max_batch_size + 1  # slot 0 is NULL


def test_adapt_hybrid_kv_caches_follows_role_mask_for_noncanonical_layout():
    """Cache roles must follow _full_attention_mask, not the inferred interval:
    for full layers at indices 0 and 3 (interval resolved to 3), the modulo
    predicate would mark index 2 as full and hand the GDN cache to as_kv()."""
    from vllm_ascend.xlite.xlite import XliteWrapper

    self = _hybrid_self()
    cfg = self.adapter_xlite_model.xlite_config
    cfg.n_layers = 4
    # Non-canonical arrangement: full at 0 and 3, GDN elsewhere.
    self.adapter_xlite_model._full_attention_mask = [True, False, False, True]
    full_kv = [
        torch.zeros(3, cfg.block_size, cfg.n_kv_heads, cfg.head_dim),
        torch.zeros(3, cfg.block_size, cfg.n_kv_heads, cfg.head_dim),
    ]
    caches = [full_kv, [], [], full_kv]
    adapted = XliteWrapper._adapt_hybrid_kv_caches(self, caches)

    assert len(adapted) == 4
    # Full layers pass through as paged K/V.
    assert adapted[0][0].data_ptr() == caches[0][0].data_ptr()
    assert adapted[3][0].data_ptr() == caches[3][0].data_ptr()
    # GDN layers get recurrent-state buffers, not paged K/V.
    for i in (1, 2):
        conv, ssm = adapted[i]
        assert conv.shape[0] == cfg.max_batch_size
        assert ssm.shape[0] == cfg.max_batch_size
