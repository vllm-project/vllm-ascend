# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

upstream = pytest.importorskip("vllm.v1.attention.ops.shardq")

# The matched release can predate ShardQ; skip it before importing the bridge.
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadata  # noqa: E402
from vllm_ascend.attention.indexer_kpool import (  # noqa: E402
    AscendIndexerKPoolMetadata,
    AscendIndexerKPoolTailMetadata,
    Glm5NextKPoolIndexerBackend,
)
from vllm_ascend.attention.sfa_shardq import build_sfa_shardq_metadata  # noqa: E402
from vllm_ascend.attention.sfa_v1 import (  # noqa: E402
    AscendSFAImpl,
    AscendSFAMetadata,
    PreprocessType,
    SparseMLAMetadataState,
)


def make_metadata(lens=(8,), histories=(100,), num_spec=0):
    starts = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    seq_lens = torch.tensor([history + length for history, length in zip(histories, lens)], dtype=torch.int32)
    positions = torch.cat([torch.arange(history, history + length) for history, length in zip(histories, lens)])
    common = SimpleNamespace(
        num_actual_tokens=sum(lens),
        num_reqs=len(lens),
        max_query_len=max(lens),
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        seq_lens_cpu_upper_bound=seq_lens,
    )
    metadata = AscendSFAMetadata(
        num_actual_tokens=sum(lens),
        num_input_tokens=sum(lens),
        slot_mapping=torch.arange(sum(lens)),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        cum_query_lens=starts[1:],
        query_start_loc=starts,
        block_table=torch.arange(2 * len(lens), dtype=torch.int32).view(len(lens), 2),
        positions=positions,
        sin=None,
        cos=None,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(sparse_mla_shard_q_min_tokens=1),
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec) if num_spec else None,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2)),
    )
    return common, metadata, config


def build_for_rank(common, metadata, config, rank, world):
    with patch.object(upstream, "get_tp_group", return_value=SimpleNamespace(rank_in_group=rank, world_size=world)):
        return build_sfa_shardq_metadata(common, metadata, config)


def test_eight_tokens_keep_separate_causal_ends():
    common, metadata, config = make_metadata()
    common.seq_lens_cpu_upper_bound = metadata.seq_lens + 3
    shardq = build_for_rank(common, metadata, config, 0, 2)
    assert shardq.query_indices.tolist() == [0, 1, 6, 7]
    assert shardq.attn_metadata.cum_query_lens.tolist() == [2, 4]
    assert shardq.attn_metadata.seq_lens.tolist() == [102, 108]
    assert shardq.attn_metadata.block_table.tolist() == [[0, 1], [0, 1]]
    assert shardq.attn_metadata.shardq is None
    assert shardq.decode is None
    assert metadata.seq_lens.tolist() == [108]


def test_uneven_requests_preserve_positions_and_speculative_decode_prefix():
    world, num_spec = 4, 2
    common, metadata, config = make_metadata((1 + num_spec, 31, 19), (53, 101, 211), num_spec)
    owned_rows = []
    for rank in range(world):
        shardq = build_for_rank(common, metadata, config, rank, world)
        assert shardq.num_decode_tokens == 1 + num_spec
        assert shardq.decode.seq_lens.tolist() == [54 + num_spec]
        assert shardq.query_indices[: 1 + num_spec].tolist() == list(range(1 + num_spec))
        owned_rows.extend(shardq.query_indices[1 + num_spec :].tolist())
        view = shardq.attn_metadata
        starts = view.query_start_loc.tolist()
        for fragment, (start, stop) in enumerate(zip(starts, starts[1:])):
            visible_lengths = view.seq_lens[fragment] - (stop - start) + torch.arange(1, stop - start + 1)
            torch.testing.assert_close(visible_lengths.long(), view.positions[start:stop] + 1)
    assert sorted(owned_rows) == list(range(1 + num_spec, sum((1 + num_spec, 31, 19))))


def test_empty_step_does_not_build_shardq():
    assert build_sfa_shardq_metadata(SimpleNamespace(num_actual_tokens=0), None, None) is None


def test_nope_sfa_restores_local_heads_and_padded_rows():
    lens, padding = (1, 8), 3
    common, metadata, config = make_metadata(lens, (50, 100))
    shardq = build_for_rank(common, metadata, config, 0, 2)
    metadata.shardq = shardq
    tokens = sum(lens)
    local_query = torch.ones(tokens + padding, 2, 3)
    prefill_query = torch.ones(shardq.attn_metadata.num_actual_tokens, 4, 3)
    topk = torch.arange(shardq.query_indices.numel(), dtype=torch.int32).view(-1, 1, 1)
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.use_shardq = True
    impl.kv_lora_rank = 3
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_turboquant = False
    impl.enable_sparse_sfa_c8 = False
    impl.scale = 1.0
    with (
        patch.object(upstream, "shard_q", return_value=(local_query[:1], prefill_query)),
        patch.object(upstream, "unshard_out", return_value=local_query[:tokens] * 2) as restore,
        patch("vllm_ascend.attention.sfa_v1.sparse_mla", side_effect=lambda q, *_: q * 2) as attend,
    ):
        result = impl._execute_sparse_flash_attention_process(
            local_query,
            None,
            (torch.zeros(1),),
            topk,
            metadata,
            metadata.cum_query_lens,
            metadata.seq_lens,
        )
    decode, prefill = (call.args for call in attend.call_args_list)
    assert (decode[0].shape[1], prefill[0].shape[1]) == (2, 4)
    torch.testing.assert_close(decode[2], topk[:1])
    torch.testing.assert_close(prefill[2], topk[1:])
    torch.testing.assert_close(restore.call_args.args[0], prefill_query * 2)
    torch.testing.assert_close(restore.call_args.args[2], local_query[:1] * 2)
    assert result.shape == local_query.shape
    torch.testing.assert_close(result[:tokens], local_query[:tokens] * 2)
    assert torch.count_nonzero(result[tokens:]) == 0


def test_nope_shardq_binds_local_and_full_head_metadata():
    common, metadata, config = make_metadata((1, 8), (2, 2048))
    state = SparseMLAMetadataState.__new__(SparseMLAMetadataState)
    state.use_smla = True
    state.sinks = torch.ones(32)
    state.shardq_sinks = None
    state.length_buffer = metadata.smla_topk_length = torch.zeros(9, 1, dtype=torch.int32)
    state.valid_mask_buffer = metadata.smla_valid_mask = torch.ones(9, 1, dtype=torch.bool)
    metadata.smla_metadata = torch.zeros(1024, dtype=torch.int32)
    metadata.smla_sinks = state.sinks
    shardq = build_for_rank(common, metadata, config, 0, 2)

    state.prepare_shardq(shardq, 64)

    assert shardq.decode.smla_sinks.shape == (32,)
    assert shardq.attn_metadata.smla_sinks.shape == (64,)
    assert shardq.decode.smla_metadata is shardq.attn_metadata.smla_metadata is metadata.smla_metadata
    assert shardq.decode.smla_topk_length.shape == (1, 1)
    assert shardq.attn_metadata.smla_topk_length.shape == (4, 1)
    assert shardq.decode.max_query_len == 1
    assert shardq.attn_metadata.max_query_len == 2


def test_kpool_shardq_preserves_full_writes_and_fragment_pool_lengths():
    lens, histories = (1, 8), (2, 2048)
    common, sfa, config = make_metadata(lens, histories)
    shardq = build_for_rank(common, sfa, config, 0, 2)
    tokens = sum(lens)
    positions = sfa.positions
    table = sfa.block_table + 20
    metadata = AscendIndexerKPoolMetadata(
        block_table=table,
        slot_mapping=sfa.slot_mapping,
        seq_lens=sfa.seq_lens.long() // 4,
        positions=positions,
        block_size=128,
        compress_ratio=4,
        cum_query_lens=sfa.cum_query_lens.long(),
        num_actual_tokens=tokens,
        query_start_loc=sfa.query_start_loc,
        start_pos=torch.tensor(histories),
        pool_tail=sfa.seq_lens.long() % 4,
        pooled_slot_mapping=torch.arange(2),
        shardq=shardq,
        indexer_row_shard=SimpleNamespace(start=1, stop=2),
    )
    op = Glm5NextKPoolIndexerBackend.__new__(Glm5NextKPoolIndexerBackend)
    torch.nn.Module.__init__(op)
    op.n_head, op.head_dim, op.index_kpool, op.topk_tokens = 1, 2, 4, 2048
    op.wq_b = MagicMock(side_effect=lambda q: (q,))
    op._head_weight_f32 = torch.ones(1, 2)
    op.wk_weights_proj = SimpleNamespace(weight=torch.empty(3, 2))
    op.index_kpool_compress_gate = torch.empty(2, 2)
    op.softmax_scale = 1.0
    op.k_norm = SimpleNamespace(eps=1e-6, weight=None, bias=None)
    op.index_kpool_compress_ape = torch.empty(4, 2)
    op.k_cache = SimpleNamespace(kv_cache=torch.empty(2, 2, 1, 2, dtype=torch.bfloat16))
    op.tail_cache = SimpleNamespace(prefix="tail", kv_cache=torch.empty(2, 4, 4))
    op.topk_indices_buffer = torch.empty(tokens, op.topk_output_width, dtype=torch.int32)
    op._prepare_keypool_state = MagicMock()
    op._key_pool = MagicMock()
    op._write_keypool_caches = MagicMock()
    op._cann_pool_key_indexer = MagicMock()
    op._pool_key_indexer = MagicMock(return_value=(torch.empty(5, op.topk_output_width), None))
    op._pack_keypool_indices = MagicMock(
        side_effect=lambda indices, scoring, k, p, buffer: buffer[: indices.shape[0]].unsqueeze(1)
    )
    tail_metadata = AscendIndexerKPoolTailMetadata(
        block_table=table,
        slot_mapping=sfa.slot_mapping,
        block_size=4,
        cann_inputs=SimpleNamespace(retained_indices=None),
    )
    ctx = SimpleNamespace(attn_metadata={"tail": tail_metadata}, virtual_engine=0)
    hidden = torch.arange(tokens * 2, dtype=torch.bfloat16).reshape(tokens, 2)
    expected_rows = [0, 1, 2, 7, 8]
    with (
        patch("vllm_ascend.attention.indexer_kpool.get_forward_context", return_value=ctx),
        patch("vllm_ascend.attention.indexer_kpool._exchange_row_shard_topk") as gather,
    ):
        result = op.forward(hidden, hidden, hidden, metadata)

    gather.assert_not_called()
    torch.testing.assert_close(op._key_pool.call_args.args[0], hidden)
    assert op._write_keypool_caches.call_args.args[5] is metadata
    torch.testing.assert_close(op.wq_b.call_args.args[0], hidden[expected_rows])
    args, kwargs = op._pool_key_indexer.call_args
    assert kwargs["actual_seq_q"].tolist() == [1, 3, 5]
    assert kwargs["actual_seq_k"].tolist() == [0, 512, 514]
    assert args[4].tolist() == [3, 2, 0]
    torch.testing.assert_close(kwargs["block_table"], table[[0, 1, 1]])
    torch.testing.assert_close(op._pack_keypool_indices.call_args.args[1].positions, positions[expected_rows])
    assert result.shape == (len(expected_rows), 1, 2051)


@pytest.mark.parametrize("quant", ["bf16", "c8", "c4"])
def test_indexer_scores_shard_rows_after_writing_all_keys(quant):
    common, metadata, config = make_metadata() if quant == "bf16" else make_metadata((1, 8), (50, 100))
    shardq = build_for_rank(common, metadata, config, 0, 2)
    rows = shardq.query_indices
    tokens, heads, dim = common.num_actual_tokens, 2, 4
    hidden = torch.randn(tokens, dim)
    query_input = torch.arange(tokens * dim).view(tokens, dim)
    keys = torch.randn(tokens, 1, dim)
    weights = torch.randn(tokens, heads)
    cos, sin = torch.rand(tokens, 1, dim), torch.rand(tokens, 1, dim)
    query = torch.arange(tokens * heads * dim).view(tokens, heads, dim)
    scale = None
    if quant == "c8":
        scale = torch.arange(tokens * heads).view(tokens, heads)
    elif quant == "c4":
        scale = torch.arange(tokens * heads * 4).view(tokens, heads, 2, 2)
        query = query[..., : dim // 2].contiguous()

    indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(indexer)
    indexer.n_head, indexer.head_dim = heads, dim
    indexer.enable_sparse_li_c8 = quant == "c8"
    indexer.enable_sparse_li_c4 = quant == "c4"
    indexer.use_torch_npu_lightning_indexer = False
    indexer.wq_b = indexer.wk_weights_proj = object()
    indexer.k_cache = SimpleNamespace(kv_cache=(torch.zeros_like(keys),))
    indexer._select_execution_paths = lambda *_: (None, False, False)
    indexer._gather_cache_inputs = lambda k, s, m: (k, s, m.slot_mapping)

    def forward_k(k_hidden, k_cos, k_sin):
        assert k_hidden is hidden and k_cos is cos and k_sin is sin
        return keys, None, weights

    def forward_q(_, q, q_cos, q_sin):
        if isinstance(q, tuple):
            q, input_scale = q
            assert input_scale.dtype == torch.float8_e8m0fnu
            torch.testing.assert_close(input_scale.view(torch.uint8), rows.to(torch.uint8))
            q = q.view(torch.uint8).long()
        torch.testing.assert_close(q, query_input[rows])
        torch.testing.assert_close(q_cos, cos[rows])
        torch.testing.assert_close(q_sin, sin[rows])
        if scale is None:
            return query[rows], None, None
        local_scale = scale[rows].flatten() if quant == "c8" else scale[rows]
        return query[rows].flatten(0, 1), local_scale, (rows.numel(), heads, dim)

    def write(k, _scale, slots, _scatter):
        indexer.k_cache.kv_cache[0].copy_(k)
        torch.testing.assert_close(slots, metadata.slot_mapping)

    def select(q, _scale, _shape, _weights, kv_cache, *_args):
        assert kv_cache is indexer.k_cache.kv_cache
        torch.testing.assert_close(kv_cache[0], keys)
        return q[:, :1, :1].int()

    indexer.forward_k = forward_k
    indexer._forward_q = forward_q
    indexer.write_cache = write
    index_metadata = AscendSFAIndexerMetadata(
        num_actual_tokens=tokens,
        slot_mapping=metadata.slot_mapping,
        seq_lens=metadata.seq_lens,
        cum_query_lens=metadata.cum_query_lens,
        block_table=metadata.block_table,
        sin=sin,
        cos=cos,
        block_size=128,
        actual_seq_lengths_query=metadata.cum_query_lens,
        actual_seq_lengths_key=metadata.seq_lens,
        shardq=shardq,
    )
    q_c = query_input
    if quant == "c8":
        q_c = (
            query_input.to(torch.uint8).view(torch.float8_e4m3fn),
            torch.arange(tokens, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        )
    with patch(
        "vllm_ascend.attention.sfa_shardq.DeviceOperator.indexer_select_post_process", side_effect=select
    ) as select_topk:
        result = indexer.forward(hidden, q_c, hidden, index_metadata)

    torch.testing.assert_close(result, query[rows, :1, :1].int())
    assert select_topk.call_count == 1 + (shardq.decode is not None)
    prefill_rows = rows[shardq.num_decode_tokens :]
    prefill_q, prefill_scale, prefill_shape, prefill_weights = select_topk.call_args.args[:4]
    torch.testing.assert_close(prefill_q, query[prefill_rows])
    torch.testing.assert_close(prefill_weights, weights[prefill_rows])
    if scale is not None:
        torch.testing.assert_close(prefill_scale, scale[prefill_rows])
        assert prefill_shape == (prefill_rows.numel(), heads, dim)


def test_profile_accounts_for_exchange_memory():
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_turboquant = False
    impl.use_shardq = True
    impl.tp_size, impl.local_num_heads, impl.kv_lora_rank, impl.qk_rope_head_dim = 2, 4, 8, 2
    impl.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))
    hidden = MagicMock(shape=(16, 4))
    output = torch.empty(16, 4)
    result = impl._forward_impl("layer", hidden, (), None, output)
    hidden.new_empty.assert_called_once()
    assert hidden.new_empty.call_args.args[0][0] >= 2 * 16
    assert result is output
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize(
    ("pp_size", "stage_start", "compatible", "enabled"),
    [(2, 2, True, True), (2, 1, True, False), (1, 0, False, False)],
)
def test_indexcache_shardq_requires_one_layout_for_the_shared_buffer(pp_size, stage_start, compatible, enabled):
    shared = torch.empty(8, 2, dtype=torch.int32)
    owner, consumer = [AscendSFAImpl.__new__(AscendSFAImpl) for _ in range(2)]
    owner._skip_topk, consumer._skip_topk = False, True
    owner.use_shardq, consumer.use_shardq = True, compatible
    for layer in (owner, consumer):
        layer.use_index_cache = True
        layer.topk_indices_buffer = shared
    owner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=4, indexer_types=["full", "shared", "full", "shared"])
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={"owner": SimpleNamespace(impl=owner), "consumer": SimpleNamespace(impl=consumer)}
        ),
    )
    with patch("vllm_ascend.attention.sfa_v1.get_pp_indices", return_value=(stage_start, 4)):
        owner._configure_indexcache_shardq()
    assert owner.use_shardq is consumer.use_shardq is enabled


def test_indexcache_forward_reuses_only_the_current_query_rows():
    rank, local_rows = 1, [0, 2, 3, 4, 5]
    buffer = torch.full((8, 2), -99, dtype=torch.int32)
    layers = [AscendSFAImpl.__new__(AscendSFAImpl) for _ in range(2)]
    for index, layer in enumerate(layers):
        layer.layer_name = f"model.layers.{index}.self_attn.attn"
        layer._skip_topk = layer is not layers[0]
        layer.has_indexer, layer.skip_indexer_pre_process = not layer.skip_topk, layer.skip_topk
        layer.use_index_cache = layer.use_shardq = True
        layer.topk_indices_buffer = buffer
        layer.enable_sparse_sfa_turboquant = layer.pcp_enabled = False
        layer.enable_sparse_sfa_c8 = False
        layer.qk_rope_head_dim = 1
        layer.g_proj = layer.layerwise_kv_cache_hook = None
        layer.preprocess_type = PreprocessType.MLAPO
        layer._get_indexer_attn_metadata = MagicMock(return_value=None)
        layer._start_o_proj_weight_gather = MagicMock()
        layer._v_up_proj = MagicMock(side_effect=lambda x: x)
        layer._finalize_o_proj = MagicMock()
        layer._prepare_indexer_metadata = MagicMock()

    steps = [((1, 7), True, local_rows), ((3,), False, [0, 1, 2]), ((1, 7), True, local_rows)]
    for step, (lens, use_shardq, query_rows) in enumerate(steps):
        common, metadata, config = make_metadata(lens, (100,) * len(lens))
        hidden = torch.zeros(sum(lens), 2)
        expected = (torch.tensor(query_rows, dtype=torch.int32) + 10 * step).view(-1, 1, 1).expand(-1, 1, 2)
        config.attention_config.sparse_mla_shard_q_min_tokens = 4
        metadata.shardq = build_for_rank(common, metadata, config, rank, 2)
        assert (metadata.shardq is not None) == use_shardq
        indexer_metadata = AscendSFAIndexerMetadata(
            num_actual_tokens=sum(lens),
            slot_mapping=metadata.slot_mapping,
            seq_lens=metadata.seq_lens,
            cum_query_lens=metadata.cum_query_lens,
            block_table=metadata.block_table,
            sin=None,
            cos=None,
            block_size=128,
        )
        owner = layers[0]
        owner._get_indexer_attn_metadata.return_value = indexer_metadata
        cache = (torch.zeros(1),)
        owner.indexer = MagicMock(return_value=expected, k_cache=SimpleNamespace(kv_cache=cache), num_cache_tensors=1)
        tail_before = buffer[len(query_rows) :].clone()
        for layer in layers:
            layer._sfa_preprocess_mlapo = MagicMock(return_value=(hidden, hidden, hidden, hidden))
            layer._execute_sparse_flash_attention_process = MagicMock(return_value=hidden)
            layer._forward_impl("layer", hidden, cache * 2, metadata, torch.empty_like(hidden))
            actual = layer._execute_sparse_flash_attention_process.call_args.args[3]
            torch.testing.assert_close(actual, expected)
        owner.indexer.assert_called_once()
        assert (owner.indexer.call_args.args[3].shardq is not None) == use_shardq
        torch.testing.assert_close(buffer[len(query_rows) :], tail_before)
