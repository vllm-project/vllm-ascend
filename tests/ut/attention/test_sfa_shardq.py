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
from vllm_ascend.attention.sfa_shardq import build_sfa_shardq_metadata, select_shardq_topk  # noqa: E402
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
    shardq = build_for_rank(common, metadata, config, 0, 2)
    assert shardq.query_indices.tolist() == [0, 1, 6, 7]
    assert shardq.attn_metadata.cum_query_lens.tolist() == [2, 4]
    assert shardq.attn_metadata.seq_lens.tolist() == [102, 108]
    assert shardq.attn_metadata.block_table.tolist() == [[0, 1], [0, 1]]
    assert shardq.attn_metadata.shardq is None
    assert shardq.decode is None
    assert metadata.seq_lens.tolist() == [108]


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("num_spec", [0, 2])
def test_uneven_requests_preserve_positions_and_decode_prefix(world, num_spec):
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


def test_planner_declines_empty_short_and_dense_steps():
    common, metadata, config = make_metadata()
    config.attention_config.sparse_mla_shard_q_min_tokens = 9
    assert build_for_rank(common, metadata, config, 0, 2) is None
    config.attention_config.sparse_mla_shard_q_min_tokens = 1
    config.model_config.hf_text_config.index_topk = 108
    assert build_for_rank(common, metadata, config, 0, 2) is None
    common.num_actual_tokens = 0
    assert build_for_rank(common, metadata, config, 0, 2) is None


def test_planner_uses_exact_device_lengths_with_cpu_upper_bounds():
    common, metadata, config = make_metadata()
    common.seq_lens_cpu_upper_bound = metadata.seq_lens + 3
    shardq = build_for_rank(common, metadata, config, 0, 2)
    assert shardq.attn_metadata.seq_lens.tolist() == [102, 108]


@pytest.mark.parametrize("quant", ["bf16", "c8", "c4"])
def test_selector_splits_compact_queries_and_scales_with_full_cache(quant):
    common, metadata, config = make_metadata((1, 8), (50, 100))
    shardq = build_for_rank(common, metadata, config, 0, 2)
    tokens, heads, dim = 9, 2, 4
    query = torch.arange(tokens * heads * dim).view(tokens, heads, dim)
    weights = torch.arange(tokens * heads).view(tokens, heads)
    scale = None
    if quant == "c8":
        scale = torch.arange(tokens * heads)
    elif quant == "c4":
        scale = torch.arange(tokens * heads * 4).view(tokens, heads, 2, 2)
        query = query[..., : dim // 2].contiguous()
    indexer = SimpleNamespace(
        n_head=heads,
        head_dim=dim,
        enable_sparse_li_quant=quant != "bf16",
        enable_sparse_li_c8=quant == "c8",
        enable_sparse_li_c4=quant == "c4",
        use_torch_npu_lightning_indexer=False,
        k_cache=SimpleNamespace(kv_cache=(torch.zeros(1), torch.zeros(1))),
    )
    indexer_metadata = AscendSFAIndexerMetadata(
        num_actual_tokens=tokens,
        slot_mapping=metadata.slot_mapping,
        seq_lens=metadata.seq_lens,
        cum_query_lens=metadata.cum_query_lens,
        block_table=metadata.block_table,
        sin=None,
        cos=None,
        block_size=128,
    )
    rows = shardq.query_indices
    query_input = query[rows]
    if quant != "bf16":
        query_input = query_input.flatten(0, 1)
        scale = scale.view(tokens, heads, *scale.shape[2:])[rows]
    with patch("vllm_ascend.attention.sfa_shardq.DeviceOperator.indexer_select_post_process") as select:
        select.side_effect = lambda q, *args: q[:, :1, :1]
        result = select_shardq_topk(query_input, weights[rows], indexer, indexer_metadata, scale, shardq=shardq)
    torch.testing.assert_close(result, query[rows, :1, :1])
    prefill_call = select.call_args_list[1].args
    torch.testing.assert_close(prefill_call[0], query[rows[1:]])
    torch.testing.assert_close(prefill_call[3], weights[rows[1:]])
    assert prefill_call[4] is indexer.k_cache.kv_cache
    if scale is not None:
        torch.testing.assert_close(prefill_call[1], scale[1:])


@pytest.mark.parametrize("decode_tokens", [0, 1])
@pytest.mark.parametrize("padding", [0, 3])
@pytest.mark.parametrize("rope", [0, 1])
def test_sfa_uses_upstream_exchanges_and_restores_local_heads(decode_tokens, padding, rope):
    lens = (1, 8) if decode_tokens else (8,)
    histories = (50, 100) if decode_tokens else (100,)
    common, metadata, config = make_metadata(lens, histories)
    world, rank, heads, latent = 2, 0, 4, 3
    shardq = build_for_rank(common, metadata, config, rank, world)
    metadata.shardq = shardq
    tokens = sum(lens)
    full_query = torch.arange((tokens + padding) * heads * (latent + rope), dtype=torch.float32).view(
        tokens + padding, heads, -1
    )
    local_query = full_query[:, : heads // world].contiguous()
    expected = full_query[..., :latent] * 2
    expected[tokens:] = 0
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.use_shardq = True
    impl.kv_lora_rank = latent
    impl.qk_rope_head_dim = rope
    impl.enable_sparse_sfa_turboquant = False
    impl.enable_sparse_sfa_c8 = False
    impl.scale = 1.0
    exchanges = []

    def exchange(recv, send, **kwargs):
        exchanges.append(kwargs)
        if len(exchanges) == 1:
            received = torch.stack(full_query[shardq.query_indices[decode_tokens:]].chunk(world, dim=1))
        else:
            all_rows = shardq.token_ids + decode_tokens
            received = expected[all_rows, : heads // world].contiguous()
        recv.copy_(received.view(torch.uint8))

    def attend(_impl, q, _q_pe, cache, topk, view, query_lens, seq_lens, **kwargs):
        assert view.shardq is None
        assert len(seq_lens) == len(query_lens)
        assert topk.shape[0] == q.shape[0]
        return q * 2

    attention_heads = []

    def attend_nope(q, cache, topk, view, scale):
        assert view.positions.shape[0] == q.shape[0] == topk.shape[0]
        attention_heads.append(q.shape[1])
        return q * 2

    group = SimpleNamespace(rank_in_group=rank, world_size=world, device_group=object())
    with (
        patch.object(upstream, "get_tp_group", return_value=group),
        patch.object(upstream, "_workspace", None),
        patch.object(upstream.dist, "all_to_all_single", side_effect=exchange),
        patch("vllm_ascend.attention.sfa_v1.DeviceOperator.execute_sparse_flash_attention_process", side_effect=attend),
        patch("vllm_ascend.attention.sfa_v1.sparse_mla", side_effect=attend_nope),
    ):
        result = impl._execute_sparse_flash_attention_process(
            local_query[..., :latent],
            local_query[..., latent:],
            (torch.zeros(1),),
            torch.zeros(len(shardq.query_indices), 1, 2, dtype=torch.int32),
            metadata,
            metadata.cum_query_lens,
            metadata.seq_lens,
        )
    torch.testing.assert_close(result, expected[:, : heads // world])
    assert len(exchanges) == 2
    if rope == 0:
        assert attention_heads == ([heads // world, heads] if decode_tokens else [heads])


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


@pytest.mark.parametrize("mixed", [False, True])
def test_kpool_shardq_preserves_full_writes_and_fragment_pool_lengths(mixed):
    lens, histories = ((1, 8), (2, 2048)) if mixed else ((8,), (2048,))
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
    op._key_weight = op._gate_weight = torch.empty(2, 2)
    op._norm_weight_f32 = op._norm_bias_f32 = None
    op.softmax_scale = 1.0
    op.k_norm = SimpleNamespace(eps=1e-6)
    op.index_kpool_compress_ape = torch.empty(4, 2)
    op.k_cache = SimpleNamespace(kv_cache=torch.empty(2, 2, 1, 2, dtype=torch.bfloat16))
    op.tail_cache = SimpleNamespace(prefix="tail", kv_cache=torch.empty(2, 4, 4))
    op.topk_indices_buffer = torch.empty(tokens, op.topk_output_width, dtype=torch.int32)
    op._prepare_keypool_state = MagicMock()
    op._key_pool = MagicMock()
    op._write_keypool_caches = MagicMock()
    op._cann_pool_key_indexer = MagicMock()
    op._pool_key_indexer = MagicMock(return_value=(torch.empty(4 + mixed, op.topk_output_width), None))
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
    expected_rows = [0, 1, 2, 7, 8] if mixed else [0, 1, 6, 7]
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
    assert kwargs["actual_seq_q"].tolist() == ([1, 3, 5] if mixed else [2, 4])
    assert kwargs["actual_seq_k"].tolist() == ([0, 512, 514] if mixed else [512, 514])
    assert args[4].tolist() == ([3, 2, 0] if mixed else [2, 0])
    torch.testing.assert_close(kwargs["block_table"], table[[0, 1, 1] if mixed else [0, 0]])
    torch.testing.assert_close(op._pack_keypool_indices.call_args.args[1].positions, positions[expected_rows])
    assert result.shape == (len(expected_rows), 1, 2051)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("packed_q", [False, True])
def test_indexer_projects_shard_rows_and_writes_all_keys(mixed, packed_q):
    common, metadata, config = make_metadata((1, 8), (50, 100)) if mixed else make_metadata()
    shardq = build_for_rank(common, metadata, config, 0, 2)
    rows = shardq.query_indices
    tokens = common.num_actual_tokens
    hidden = torch.randn(tokens, 4)
    query_input = torch.arange(tokens * 4).reshape(tokens, 4)
    keys = torch.randn(tokens, 1, 4)
    weights = torch.randn(tokens, 2)
    cos, sin = torch.rand(tokens, 1, 4), torch.rand(tokens, 1, 4)
    events = []
    indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(indexer)
    indexer.n_head, indexer.head_dim = 2, 4
    indexer.enable_sparse_li_c8 = indexer.enable_sparse_li_c4 = False
    indexer.use_torch_npu_lightning_indexer = False
    indexer.wq_b = indexer.wk_weights_proj = object()
    indexer.k_cache = SimpleNamespace(kv_cache=(torch.zeros_like(keys),))
    indexer._select_execution_paths = lambda *_: (None, False, False)
    indexer._gather_cache_inputs = lambda k, s, m: (k, s, m.slot_mapping)

    def forward_k(k_hidden, k_cos, k_sin):
        assert k_hidden is hidden and k_cos is cos and k_sin is sin
        return keys, None, weights

    def forward_q(_, q, q_cos, q_sin):
        if packed_q:
            q, scale = q
            torch.testing.assert_close(scale.view(torch.uint8), rows.to(torch.uint8))
            q = q.view(torch.uint8).long()
        torch.testing.assert_close(q, query_input[rows])
        torch.testing.assert_close(q_cos, cos[rows])
        torch.testing.assert_close(q_sin, sin[rows])
        events.append("project")
        return query_input[rows, None, :].expand(-1, 2, -1), None, None

    def write(k, scale, slots, scatter):
        indexer.k_cache.kv_cache[0].copy_(k)
        assert slots.numel() == tokens
        events.append("write")

    def select(q, *args):
        torch.testing.assert_close(indexer.k_cache.kv_cache[0], keys)
        events.append("select")
        return torch.zeros(q.shape[0], 1, 2, dtype=torch.int32)

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
    if packed_q:
        q_c = (
            query_input.to(torch.uint8).view(torch.float8_e4m3fn),
            torch.arange(tokens, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        )
    with patch("vllm_ascend.attention.sfa_shardq.DeviceOperator.indexer_select_post_process", side_effect=select):
        result = indexer.forward(hidden, q_c, hidden, index_metadata)
    assert result.shape[0] == rows.numel()
    assert events == ["write", "project"] + ["select"] * (1 + mixed)


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
    [(1, 0, True, True), (2, 2, True, True), (2, 1, True, False), (1, 0, False, False)],
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


@pytest.mark.parametrize(("rank", "local_rows"), [(0, [0, 1, 6, 7]), (1, [0, 2, 3, 4, 5])])
def test_indexcache_forward_reuses_only_the_current_query_rows(rank, local_rows):
    buffer = torch.full((8, 2), -99, dtype=torch.int32)
    layers = [AscendSFAImpl.__new__(AscendSFAImpl) for _ in range(3)]
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
