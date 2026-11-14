# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm_ascend.models.glm5next.attention as model_attention
from vllm_ascend.ops.mla import IndexerWrapper


class _FakeKPoolBackend(nn.Module):
    def __init__(self, source, rope_dim):
        super().__init__()
        del rope_dim
        self.k_cache = source.k_cache
        self.head_dim = source.head_dim
        self.enable_sparse_li_c8 = False
        self.num_cache_tensors = 1

    def process_weights_after_loading(self):
        return None


class _FakeIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 2
        self.head_dim = 4
        self.topk_tokens = 8
        self.q_lora_rank = 3
        self.wq_b = nn.Linear(3, 8, bias=False)
        self.wk_weights_proj = nn.Linear(5, 6, bias=False)
        self.k_norm = nn.LayerNorm(4)
        self.softmax_scale = 0.5
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(4, 4))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(4, 5))
        self.k_cache = SimpleNamespace(prefix="indexer.k_cache")

    def get_ascend_indexer_backend_cls(self):
        return _FakeKPoolBackend


def test_wrapper_selects_model_backend_and_preserves_weight_names() -> None:
    wrapper = IndexerWrapper(_FakeIndexer(), qk_rope_head_dim=0)

    assert isinstance(wrapper.impl, _FakeKPoolBackend)
    names = set(dict(wrapper.named_parameters()))
    assert {
        "wq_b.weight",
        "wk_weights_proj.weight",
        "k_norm.weight",
        "k_norm.bias",
        "index_kpool_compress_ape",
        "index_kpool_compress_gate",
    }.issubset(names)
    assert not any(name.startswith("impl.") for name in names)


@pytest.mark.parametrize("skip_rope", [False, True])
def test_nope_skips_rotary_construction_but_keeps_the_indexer(monkeypatch, skip_rope):
    # Construct the real attention module; device-dependent projections and
    # kernels are irrelevant to its RoPE/indexer initialization contract.
    for name in ("DeepSeekV2FusedQkvAProjLinear", "ColumnParallelLinear", "RowParallelLinear", "RMSNorm"):
        monkeypatch.setattr(model_attention, name, lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(model_attention, "get_tensor_model_parallel_world_size", lambda: 1)
    get_rope = Mock(side_effect=lambda *args, **kwargs: nn.Identity())
    indexer = Mock(return_value=nn.Identity())
    wrapper = Mock(return_value=SimpleNamespace(mla_attn=SimpleNamespace()))
    monkeypatch.setattr(model_attention, "get_rope", get_rope)
    monkeypatch.setattr(model_attention, "Indexer", indexer)
    monkeypatch.setattr(model_attention, "MultiHeadLatentAttentionWrapper", wrapper)
    config = SimpleNamespace(
        rms_norm_eps=1e-6,
        index_topk=4,
        indexer_rope_interleave=True,
    )
    # A NoPE config deliberately has no rope_parameters, as in GLM-5.3-Flash.
    if not skip_rope:
        config.rope_parameters = {"rope_type": "default"}
    layer = model_attention.Glm5NextMLAAttention(
        vllm_config=SimpleNamespace(),
        config=config,
        hidden_size=8,
        num_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0 if skip_rope else 4,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        skip_rope=skip_rope,
    )
    modules = next(arg for arg in wrapper.call_args.args if isinstance(arg, model_attention.MLAModules))
    assert modules.indexer_rotary_emb is layer.indexer_rope_emb
    assert modules.indexer is layer.indexer
    indexer.assert_called_once()
    assert layer.indexer is indexer.return_value
    assert get_rope.call_count == (0 if skip_rope else 2)
    assert (layer.rotary_emb is None) == skip_rope
    assert (layer.indexer_rope_emb is None) == skip_rope
