# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.models.qwen4_exp import qsa
from vllm_ascend.utils import vllm_version_is


def test_main_and_draft_qkv_projection_use_upstream_path() -> None:
    assert "_project_qkv_gate" not in qsa.AscendQwen4ExpQSAAttention.__dict__
    assert qsa.AscendQwen4ExpQSAAttention._project_qkv_gate is qsa.upstream_qsa.Qwen4ExpQSAAttention._project_qkv_gate


def test_main_falls_back_when_fused_projection_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value=("q", "k", "v", "gate"))
    monkeypatch.setattr(
        qsa.upstream_qsa.Qwen4ExpQSAAttention,
        "_project_qkv_gate",
        upstream_projection,
    )
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(is_draft_model=False),
    )
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer._main_fused_norm_rope_eligible = Mock(return_value=False)
    qkv = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer._project_qkv_gate(qkv, positions) == ("q", "k", "v", "gate")
    upstream_projection.assert_called_once_with(qkv, positions)


def test_indexer_query_falls_back_when_fused_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value=("query", "key"))
    monkeypatch.setattr(
        qsa.upstream_indexer.QSAIndexer,
        "project_qk",
        upstream_projection,
    )
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    layer.index_n_heads = 2
    layer.index_head_dim = 4
    layer.index_kv_heads = 1
    layer.index_qk_proj = Mock(return_value=(torch.empty((2, 12)), None))
    layer._fused_norm_rope_eligible = Mock(return_value=False)
    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)

    assert layer.project_qk(hidden_states, positions) == ("query", "key")
    upstream_projection.assert_called_once_with(hidden_states, positions)


def test_indexer_compressed_key_falls_back_when_fused_abi_is_not_satisfied(monkeypatch) -> None:
    upstream_projection = Mock(return_value="keys")
    monkeypatch.setattr(
        qsa.upstream_indexer.QSAIndexer,
        "normalize_compressed_keys",
        upstream_projection,
    )
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    layer.index_head_dim = 4
    layer.rotary_emb = SimpleNamespace(mrope_section=[1, 1, 0])
    layer._fused_norm_rope_eligible = Mock(return_value=False)
    pooled = torch.empty((2, 4))
    first_positions = torch.arange(6).reshape(2, 3)

    assert layer.normalize_compressed_keys(pooled, first_positions) == "keys"
    upstream_projection.assert_called_once_with(pooled, first_positions)


def test_indexer_compression_falls_back_when_graph_shape_misses_fused_abi(monkeypatch) -> None:
    layer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(layer)
    raw_cache = torch.empty((4, 1, 4))
    compressed_cache = torch.empty_like(raw_cache)
    layer.raw_key_cache = SimpleNamespace(key_cache=raw_cache, rope_position_cache=None)
    layer.compressed_key_cache = SimpleNamespace(kv_cache=compressed_cache)
    layer.compress_ratio = 4
    layer._fused_norm_rope_eligible = Mock(return_value=False)

    pooled = torch.empty((1, 4))
    first_positions = torch.zeros((1, 3), dtype=torch.long)
    compress = Mock(return_value=(pooled, first_positions))
    normalize = Mock(return_value=torch.ones_like(pooled))
    store = Mock()
    fused_compress = Mock(side_effect=AssertionError("fused path must not run"))
    monkeypatch.setattr(qsa, "_compress_qsa_groups", compress)
    monkeypatch.setattr(qsa, "_store_qsa_cache_rows", store)
    monkeypatch.setattr(qsa, "qsa_compress_groups_with_ratio_fused", fused_compress)
    layer.normalize_compressed_keys = normalize

    raw_metadata = SimpleNamespace(
        num_actual_tokens=2,
        logical_positions=torch.tensor([0, 1]),
        block_table=torch.empty((1, 1), dtype=torch.int32),
        token_to_req=torch.zeros(2, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1]),
    )
    compressed_metadata = SimpleNamespace(slot_mapping=torch.tensor([0]))
    token_k = torch.empty((2, 1, 4))
    positions = torch.arange(2)

    layer._update_and_compress(token_k, positions, raw_metadata, compressed_metadata)

    compress.assert_called_once()
    normalize.assert_called_once_with(pooled, first_positions)
    fused_compress.assert_not_called()
    assert store.call_count == 2
    assert store.call_args_list[0].args[0] is compressed_cache
    assert store.call_args_list[1].args[0] is raw_cache


def test_text_only_indexer_uses_fused_rope_without_mrope_sections(monkeypatch) -> None:
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_n_heads = 4
    indexer.index_kv_heads = 1
    indexer.index_head_dim = 128
    norm = SimpleNamespace(weight=torch.zeros(128, dtype=torch.bfloat16), variance_epsilon=1e-6)
    indexer.q_layernorm = norm
    indexer.k_layernorm = norm
    cache = torch.arange(256, dtype=torch.bfloat16).reshape(4, 64)
    indexer.rotary_emb = SimpleNamespace(rotary_dim=64, is_neox_style=True, _match_cos_sin_cache_dtype=lambda _: cache)
    projection = torch.ones(2, 5 * 128, dtype=torch.bfloat16)
    indexer.index_qk_proj = lambda _: (projection, None)
    fused = Mock(side_effect=lambda **kwargs: kwargs["tensor"])
    monkeypatch.setattr(torch.ops.vllm, "triton_qsa_rmsnorm_mrope", fused, raising=False)
    positions = torch.tensor([1, 3])

    queries, keys = indexer.project_qk(torch.empty(2, 16), positions)
    indexer.normalize_compressed_keys(keys, positions[:, None])

    assert queries.shape == (2, 4, 128)
    assert keys.shape == (2, 1, 128)
    assert fused.call_count == 2
    for call in fused.call_args_list:
        assert call.kwargs["mrope_section"] == []
        torch.testing.assert_close(call.kwargs["cos_sin"], cache)
        torch.testing.assert_close(call.kwargs["positions"], positions)


@pytest.mark.parametrize("pcp", [False, True])
def test_fused_prepare_upstream_abi_materializes_qkv_and_gate(monkeypatch, pcp) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0.self_attn.attn"
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.head_dim = 4
    layer.kv_cache = (torch.empty(1), torch.empty(1))
    layer.topk_indices_buffer = torch.empty((2, 3), dtype=torch.int32)

    main_metadata = SimpleNamespace(
        num_actual_tokens=2,
        slot_mapping=torch.tensor([0, 1]),
    )
    side_prefix = "model.layers.0.self_attn.indexer.raw_key_cache"
    side_metadata = SimpleNamespace(
        num_actual_tokens=2,
        token_to_req=torch.tensor([0, 0]),
        logical_positions=torch.tensor([0, 1]),
        seq_lens=torch.tensor([2]),
    )
    if pcp:
        side_metadata.pcp_context = SimpleNamespace(gather_tokens=lambda rows: torch.cat((rows, rows + 100)))
        main_metadata.slot_mapping = torch.tensor([4, 5, 6, 7])
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={
                layer.layer_name: main_metadata,
                side_prefix: side_metadata,
            }
        ),
    )

    selected = torch.zeros((2, 3), dtype=torch.int32)
    # Main's indexer returns (selection, prepared main outputs).
    indexer = Mock(return_value=selected if vllm_version_is("0.33.0") else (selected, None))
    indexer.output_width = 3
    indexer.raw_key_cache = SimpleNamespace(prefix=side_prefix)
    layer.indexer = indexer
    layer.impl = Mock()

    projected_q = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    projected_k = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    projected_v = projected_k + 10
    projected_gate = projected_q + 20
    layer._project_qkv_gate = Mock(return_value=(projected_q, projected_k, projected_v, projected_gate))

    hidden_states = torch.empty((2, 8))
    positions = torch.arange(2)
    qkv = torch.empty((2, 24))
    output = torch.empty((2, 2, 4))
    gate = torch.empty_like(output)

    layer._run_qsa(
        hidden_states,
        positions,
        None,
        None,
        None,
        output,
        qkv=qkv,
        gate=gate,
    )

    layer._project_qkv_gate.assert_called_once_with(qkv, positions)
    torch.testing.assert_close(gate, projected_gate.view_as(gate))
    layer.impl.do_kv_cache_update.assert_called_once()
    update_args = layer.impl.do_kv_cache_update.call_args.args
    expected_key = projected_k.view(2, 1, 4)
    expected_value = projected_v.view(2, 1, 4)
    if pcp:
        expected_key = torch.cat((expected_key, expected_key + 100))
        expected_value = torch.cat((expected_value, expected_value + 100))
    torch.testing.assert_close(update_args[1], expected_key)
    torch.testing.assert_close(update_args[2], expected_value)
    torch.testing.assert_close(update_args[4], main_metadata.slot_mapping)
    assert layer.impl.forward_qsa.call_count + layer.impl.forward.call_count == 1
    forward_call = layer.impl.forward_qsa if vllm_version_is("0.33.0") else layer.impl.forward
    forward_args = forward_call.call_args.args
    torch.testing.assert_close(forward_args[1], projected_q.view(2, 2, 4))


def test_fused_prepare_zeros_gate_without_runtime_metadata(monkeypatch) -> None:
    layer = qsa.AscendQwen4ExpQSAAttention.__new__(qsa.AscendQwen4ExpQSAAttention)
    torch.nn.Module.__init__(layer)
    monkeypatch.setattr(
        qsa,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    output = torch.ones((2, 2, 4))
    gate = torch.ones_like(output)

    layer._run_qsa(
        torch.empty((2, 8)),
        torch.arange(2),
        None,
        None,
        None,
        output,
        qkv=torch.empty((2, 24)),
        gate=gate,
    )

    assert not output.any()
    assert not gate.any()


def test_pcp_indexer_compresses_complete_groups_across_ranks(monkeypatch) -> None:
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.compress_ratio = 4
    indexer.raw_key_cache = SimpleNamespace(key_cache=torch.empty(1), rope_position_cache=None)
    indexer.compressed_key_cache = SimpleNamespace(kv_cache=torch.empty(1))
    indexer.normalize_compressed_keys = lambda keys, positions: keys
    global_keys = torch.tensor([10, 11, 12, 13]).view(4, 1, 1)
    raw_write = SimpleNamespace(
        num_actual_tokens=4,
        block_table=torch.tensor([[1]]),
        token_to_req=torch.tensor([0, 0, 0, 0]),
        query_start_loc=torch.tensor([0, 4]),
        logical_positions=torch.arange(4),
        slot_mapping=torch.tensor([4, 5, 6, 7]),
    )
    compressed_write = SimpleNamespace(slot_mapping=torch.tensor([-1, -1, -1, 8]))
    local_keys = global_keys[[0, 3]]
    context = SimpleNamespace(gather_tokens=Mock(return_value=global_keys))
    raw = SimpleNamespace(pcp_context=context, write_metadata=raw_write)
    compressed = SimpleNamespace(write_metadata=compressed_write)
    compress = Mock(return_value=(torch.empty(1), torch.empty(1)))
    store = Mock()
    monkeypatch.setattr(qsa, "_compress_qsa_groups", compress)
    monkeypatch.setattr(qsa, "_store_qsa_cache_rows", store)

    indexer._update_and_compress(local_keys, torch.tensor([0, 3]), raw, compressed)

    context.gather_tokens.assert_called_once_with(local_keys)
    compression_args = compress.call_args.args
    torch.testing.assert_close(compression_args[0], global_keys)
    torch.testing.assert_close(compression_args[6], raw_write.logical_positions)
    torch.testing.assert_close(compression_args[7], compressed_write.slot_mapping)
    torch.testing.assert_close(store.call_args_list[1].args[1], raw_write.slot_mapping)


def test_pcp_indexer_replaces_registered_cache_owners() -> None:
    static_context = {}
    config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=static_context))
    cache_args = dict(
        vllm_config=config,
        cache_config=SimpleNamespace(block_size=128),
        dtype=torch.bfloat16,
        head_size=128,
        compress_ratio=4,
    )
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.raw_key_cache = qsa.QSARawKeyCache(**cache_args, prefix="raw", cache_rope_positions=True)
    indexer.compressed_key_cache = qsa.qsa_indexer.QSACompressedKeyCache(**cache_args, prefix="compressed")
    raw, compressed = indexer.raw_key_cache, indexer.compressed_key_cache

    indexer.configure_pcp_cache_backends(config)

    assert static_context == {"raw": indexer.raw_key_cache, "compressed": indexer.compressed_key_cache}
    assert indexer.raw_key_cache is not raw
    assert indexer.compressed_key_cache is not compressed
    assert indexer.raw_key_cache.cache_rope_positions
    for original, cache in ((raw, indexer.raw_key_cache), (compressed, indexer.compressed_key_cache)):
        assert cache.head_size == original.head_size
        assert cache.compress_ratio == original.compress_ratio
        assert cache.get_attn_backend().supports_pcp()
        assert cache.get_attn_backend().get_builder_cls() is qsa.AscendPCPQSAIndexerMetadataBuilder


def test_pcp_indexer_global_write_invalidates_the_local_layout(monkeypatch) -> None:
    local_starts = torch.tensor([0, 2], dtype=torch.int32)
    global_starts = torch.tensor([0, 5], dtype=torch.int32)
    table = torch.tensor([[0]], dtype=torch.int32)
    context = SimpleNamespace(
        local_token_indices=torch.arange(2),
        global_batch=SimpleNamespace(
            num_reqs=1,
            num_tokens=5,
            query_start_loc=global_starts,
            query_start_loc_np=global_starts.numpy(),
            seq_lens=global_starts[1:],
            positions=torch.arange(5),
            num_scheduled_tokens=global_starts[1:].numpy(),
        ),
        global_block_tables=table.unsqueeze(0),
        global_slot_mappings=torch.arange(5).unsqueeze(0),
    )
    common = AscendCommonAttentionMetadata(
        query_start_loc=local_starts,
        query_start_loc_cpu=local_starts,
        seq_lens=local_starts[1:],
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=5,
        block_table_tensor=table,
        slot_mapping=torch.arange(2),
        pcp_context=context,
        pcp_cache_group_idx=0,
    )
    local_mapping = torch.zeros(2, dtype=torch.int32)
    common._token_to_req_indices_cache = local_mapping
    common._num_computed_tokens_cache = torch.tensor([0])
    monkeypatch.setattr(qsa.QSAIndexerMetadataBuilder, "build", lambda *_: SimpleNamespace())
    builder = qsa.AscendPCPQSAIndexerMetadataBuilder.__new__(qsa.AscendPCPQSAIndexerMetadataBuilder)
    builder.write_builder = Mock()

    builder.build(0, common)

    global_common = builder.write_builder.build.call_args.args[1]
    assert global_common.num_actual_tokens == 5
    assert global_common._token_to_req_indices_cache is None
    assert global_common._num_computed_tokens_cache is None
    assert common._token_to_req_indices_cache is local_mapping


@pytest.mark.parametrize("num_local_tokens", [0, 2])
def test_pcp_indexer_projects_padding_for_collective_but_selects_only_live_queries(num_local_tokens) -> None:
    indexer = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.skip_topk = False
    raw = SimpleNamespace(num_actual_tokens=num_local_tokens, pcp_context=object())
    indexer._metadata = lambda: (raw, raw)
    indexer.project_qk = lambda hidden, positions: (hidden, hidden)
    indexer._update_and_compress = Mock()
    indexer._select = lambda query, metadata, out: query
    hidden = torch.arange(3).view(3, 1)
    positions = torch.arange(3)

    if vllm_version_is("0.33.0"):
        selected = indexer(hidden, positions)
    else:
        selected, _ = indexer(hidden, positions, attn=object())

    key_rows, write_positions = indexer._update_and_compress.call_args.args[:2]
    torch.testing.assert_close(key_rows, hidden)
    torch.testing.assert_close(write_positions, positions)
    torch.testing.assert_close(selected, hidden[:num_local_tokens])


def test_qsa_kv_store_keeps_page_stride_with_multiple_heads(monkeypatch) -> None:
    backing = torch.zeros(2, 2, 4, 2, 3)
    cache = backing[:, 0]
    slots = torch.tensor([5])
    rows = torch.arange(1, 7, dtype=cache.dtype).view(1, 2, 3)

    def store(cache_view, slot_mapping, flat_rows):
        cache_view[slot_mapping // 4, slot_mapping % 4, 0] = flat_rows

    monkeypatch.setattr(qsa, "qsa_store_cache_rows_fused", store)
    qsa._store_qsa_cache_rows(cache, slots, rows)

    torch.testing.assert_close(cache[1, 1], rows[0])
    assert not backing[:, 1].any()
