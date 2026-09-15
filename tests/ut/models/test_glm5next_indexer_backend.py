# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

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
