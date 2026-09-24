# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for V4.1 index selection and compressor scheduling."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl, DeepseekV41PreparedIndexer

TOKENS, TOPK = 4, 512


def _impl(role):
    impl = AscendDSAV41Impl.__new__(AscendDSAV41Impl)
    impl.role = role
    impl.index_k_source_prefix = "model.layers.2.attn"
    impl.topology = SimpleNamespace(candidate_topk_blocks=8, candidate_block_size=8)
    return impl


def _attn(selected):
    shared = SimpleNamespace(
        topk_indices=torch.full((TOKENS, TOPK), 7, dtype=torch.int32),
        candidates=torch.full((TOKENS, 1, 8), 7, dtype=torch.int32),
    )
    indexer = SimpleNamespace(select_projected=lambda *args, **kwargs: (selected, None))
    return SimpleNamespace(shared_state=shared, indexer=indexer), shared


def test_empty_cache_selection_publishes_no_slot(monkeypatch):
    prefix = "model.layers.2.attn"
    monkeypatch.setattr(
        dsa_v41,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={prefix: SimpleNamespace(kv_cache=[None])}),
    )
    selected = torch.full((TOKENS, 0), -1, dtype=torch.int32)
    attn, shared = _attn(selected)
    impl = _impl(
        SimpleNamespace(
            has_long_context=True,
            is_index_source=True,
            is_candidate_source=False,
            uses_candidate_filter=False,
        )
    )

    out = impl._select_sparse_indices(
        attn,
        torch.zeros(TOKENS, 8),
        torch.zeros(TOKENS, 8),
        torch.arange(TOKENS),
        None,
        None,
        SimpleNamespace(
            swa=SimpleNamespace(num_actual_tokens=TOKENS),
            indexer=SimpleNamespace(cache=object()),
        ),
        DeepseekV41PreparedIndexer(
            query=torch.zeros(TOKENS, 1, 8),
            weights=torch.zeros(TOKENS, 1),
        ),
    )

    assert shared.topk_indices.shape == (TOKENS, TOPK)
    assert torch.all(shared.topk_indices == -1)
    assert out is not None and out.shape == (TOKENS, TOPK)


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_input_ready_before_query_quantization(monkeypatch, ratio):
    hidden_states = torch.zeros(TOKENS, 8, dtype=torch.bfloat16)
    latent = torch.zeros(TOKENS, 8, dtype=torch.bfloat16)
    positions = torch.arange(TOKENS)
    cos, sin = torch.ones(TOKENS, 4), torch.zeros(TOKENS, 4)
    calls = []
    original_float = torch.Tensor.float

    def cast(value, *args, **kwargs):
        result = original_float(value, *args, **kwargs)
        if value is hidden_states:
            calls.append("input_cast")
        return result

    def quantize_query(*args):
        calls.append("query_quantization")

    def project_kv(value):
        calls.append("wkv")
        assert value.dtype == (torch.float32 if ratio == 2 else torch.bfloat16)
        return latent

    monkeypatch.setattr(torch.Tensor, "float", cast)
    monkeypatch.setattr(AscendDSAV41Impl, "_quantize_indexer_query", quantize_query)
    monkeypatch.setattr(dsa_v41, "wait_for_device_metadata", lambda *args: None)
    monkeypatch.setattr(dsa_v41, "scatter_cache_sk", lambda *args: None)
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *args, **kwargs: None, raising=False)
    cache = SimpleNamespace(slot_mapping=positions)
    metadata = SimpleNamespace(
        indexer=SimpleNamespace(cache=cache),
        compressor=SimpleNamespace(
            cache=cache,
            state=SimpleNamespace(
                c2_metadata_group_id=0,
                c2_source_cos=cos,
                c2_source_sin=sin,
            ),
        ),
    )
    attn = SimpleNamespace(
        compressor=SimpleNamespace(
            wkv=project_kv,
            wgate=lambda value: torch.zeros_like(value[:, :8]),
            norm=lambda value: value,
            pool_projected=lambda *args: latent,
        ),
        indexer=SimpleNamespace(update_keys=lambda *args: None),
        long_kv_cache=SimpleNamespace(kv_cache=[None]),
        head_dim=8,
        nope_head_dim=4,
    )
    impl = _impl(SimpleNamespace(compress_ratio=ratio))
    impl._write_compressed_source(attn, hidden_states, positions, cos, sin, metadata, prepared_indexer=object())

    assert calls.count("query_quantization") == calls.count("wkv") == 1
    assert calls.index("query_quantization") < calls.index("wkv")
    if ratio == 2:
        assert calls.index("input_cast") < calls.index("query_quantization")
    else:
        assert "input_cast" not in calls
