# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for V4.1 index selection and compressor scheduling."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import dsa_v41_cp
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl, DeepseekV41PreparedIndexer
from vllm_ascend.ops.dsv41_a5 import indexer as a5_indexer

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
        candidate_lengths=None,
        topk_lengths=None,
    )

    def select_projected(*args, indices_output, **kwargs):
        if selected.shape[1] != 0:
            indices_output.copy_(selected)
            return indices_output, None
        return selected, None

    indexer = SimpleNamespace(select_projected=select_projected)
    return SimpleNamespace(shared_state=shared, indexer=indexer), shared


@pytest.mark.parametrize("empty_cache", [True, False])
def test_index_selection_publishes_shared_output(monkeypatch, empty_cache):
    prefix = "model.layers.2.attn"
    monkeypatch.setattr(
        dsa_v41,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={prefix: SimpleNamespace(kv_cache=[None])}),
    )
    selected = torch.full((TOKENS, 0 if empty_cache else TOPK), 3, dtype=torch.int32)
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
    assert torch.all(shared.topk_indices == (-1 if empty_cache else 3))
    assert out is not None and out.shape == (TOKENS, TOPK)


@pytest.mark.parametrize(
    ("a5", "prefills", "overlap_enabled", "expected"),
    [
        (False, 1, True, "multistream"),
        (True, 0, True, "multistream"),
        (True, 1, True, "multistream"),
        (True, 2, True, "multistream"),
        (True, 1, False, "serial"),
    ],
)
def test_qkv_projection_stream_choice(monkeypatch, a5, prefills, overlap_enabled, expected):
    impl = _impl(SimpleNamespace(is_kv_source=False))
    calls = []
    monkeypatch.setattr(impl, "preprocess", lambda *args: calls.append("serial") or (None, None))
    monkeypatch.setattr(impl, "multistream_preprocess", lambda *args: calls.append("multistream") or (None, None))
    attn = SimpleNamespace(
        dsv41_backend=object() if a5 else None,
        dsa_attn=SimpleNamespace(
            dsa_attn=SimpleNamespace(impl=SimpleNamespace(multistream_dsv4_dsa_overlap=overlap_enabled))
        ),
    )
    metadata = SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=TOKENS, num_prefills=prefills))
    impl._prepare_queries(attn, torch.zeros(TOKENS, 8), None, None, None, metadata)
    assert calls == [expected]


@pytest.mark.parametrize(
    ("prefills", "decodes", "requests", "write_on_main"),
    [(0, 2, 2, False), (1, 1, 2, True), (0, 0, 2, True)],
)
def test_a5_cache_writer_stream_choice(prefills, decodes, requests, write_on_main):
    metadata = SimpleNamespace(num_prefills=prefills, num_decodes=decodes, num_reqs=requests)
    attn = SimpleNamespace(dsv41_backend=object())
    assert AscendDSAV41Impl._write_swa_cache_on_main_stream(attn, metadata) is write_on_main


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

    def project_kv(value):
        calls.append("wkv")
        assert value.dtype == (torch.float32 if ratio == 2 else torch.bfloat16)
        return latent

    monkeypatch.setattr(torch.Tensor, "float", cast)
    monkeypatch.setattr(AscendDSAV41Impl, "_quantize_indexer_query", lambda *args: calls.append("query_quantization"))
    monkeypatch.setattr(dsa_v41, "wait_for_device_metadata", lambda *args: None)
    monkeypatch.setattr(dsa_v41, "scatter_cache_sk", lambda *args: None)
    monkeypatch.setattr(AscendDSAV41Impl, "_apply_rotary", lambda *args, **kwargs: None)
    cache = SimpleNamespace(slot_mapping=positions)
    metadata = SimpleNamespace(
        indexer=SimpleNamespace(cache=cache),
        compressor=SimpleNamespace(
            cache=cache,
            state=SimpleNamespace(c2_metadata_group_id=0, c2_source_cos=cos, c2_source_sin=sin),
        ),
    )

    class Compressor:
        def __call__(self, value):
            return project_kv(value)

        wkv = staticmethod(project_kv)
        wgate = staticmethod(lambda value: torch.zeros_like(value[:, :8]))
        pool_projected = staticmethod(lambda *args: latent)

    attn = SimpleNamespace(
        compressor=Compressor(),
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


def test_cp_keeps_local_query_and_global_compressor_inputs(monkeypatch):
    impl = dsa_v41_cp.AscendDSAV41CPImpl.__new__(dsa_v41_cp.AscendDSAV41CPImpl)
    hidden_states = torch.arange(32).view(4, 8)
    local_metadata = SimpleNamespace(swa=SimpleNamespace(cp_token_range=(2, 4, 2, 0), num_actual_tokens=2))
    global_metadata = SimpleNamespace(
        swa=SimpleNamespace(num_actual_tokens=4),
        positions=torch.arange(4),
        rope=lambda name, count: (torch.ones(count, 4), torch.zeros(count, 4)),
    )
    calls = []
    monkeypatch.setattr(impl, "_global_layer_metadata", lambda metadata: global_metadata)
    monkeypatch.setattr(dsa_v41_cp, "get_forward_context", lambda: SimpleNamespace(attn_metadata=object()))
    monkeypatch.setattr(impl, "_write_compressed_source", lambda *args, **kwargs: calls.append((args, kwargs)))

    assert torch.equal(impl._indexer_hidden_states(hidden_states, local_metadata), hidden_states[2:4])
    impl._write_forward_compressed_source(
        SimpleNamespace(rotary_emb=SimpleNamespace(layername="attn")),
        hidden_states,
        None,
        None,
        None,
        local_metadata,
        prepared_indexer=object(),
    )
    assert len(calls) == 1
    assert torch.equal(calls[0][0][1], hidden_states)
    assert calls[0][0][5] is global_metadata


def test_a5_indexer_uses_prequantized_query(monkeypatch):
    monkeypatch.setattr(a5_indexer, "wait_for_device_metadata", lambda *args: None)
    monkeypatch.setattr(
        a5_indexer,
        "mxfp4_quantize_e8m0",
        lambda query: pytest.fail("query must not be quantized twice"),
    )
    query = torch.zeros(2, 1, 128)
    quantized = torch.zeros(2, 1, 64, dtype=torch.uint8)
    scale = torch.zeros(2, 1, 4, dtype=torch.uint8)
    source_metadata = SimpleNamespace(
        qli_metadata=torch.zeros(1),
        query_start_loc=torch.zeros(1),
        cache_seq_lens=torch.zeros(1),
        cmp_residual=None,
        block_table=torch.zeros(1),
    )
    result = a5_indexer._common(
        query,
        torch.ones(2, 1),
        (torch.zeros(1), torch.zeros(2, 1, 4)),
        source_metadata,
        1,
        quantized,
        scale,
    )
    assert result[0] is quantized
    assert result[1].shape == (2, 1, 2, 2)
