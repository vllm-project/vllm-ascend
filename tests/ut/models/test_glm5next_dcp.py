# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.attention.context_parallel.sfa_cp as cp
import vllm_ascend.attention.indexer_kpool as kpool
import vllm_ascend.attention.sfa_v1 as sfa
from vllm_ascend.attention.indexer_kpool import AscendIndexerKPoolBackend, AscendIndexerKPoolMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolSpec, AscendIndexerKPoolTailSpec
from vllm_ascend.models.glm5next.cache_views import view_compressed_indexer_cache
from vllm_ascend.utils import enable_sfa_dcp_replicated_indexer


def _config(block_size, dcp_size):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(
            max_model_len=2 * block_size + 1,
            hf_text_config=SimpleNamespace(index_kpool=4, index_topk=2048),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=2),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size, prefill_context_parallel_size=1),
    )


def _spec(block_size, dcp_size, pool):
    return AscendIndexerKPoolSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        model_version="glm5_next",
        tokens_per_state=pool,
        dcp_replication_size=dcp_size,
        indexes_kv_by_block_stride=True,
    )


@pytest.mark.parametrize("dcp_size", [1, 8])
@pytest.mark.parametrize("dsa_cp", [False, True])
def test_kpool_backend_does_not_gate_dcp_on_dsa_cp(monkeypatch, dcp_size, dsa_cp):
    monkeypatch.setattr(kpool, "enable_dsa_cp", lambda: dsa_cp)
    monkeypatch.setattr(
        kpool, "get_current_hardware_profile", lambda: SimpleNamespace(attention_backend_family=object())
    )
    monkeypatch.setattr(kpool.Glm5NextKPoolIndexerBackend, "_init_cann_ops", lambda self: None)
    source = SimpleNamespace(
        vllm_config=_config(128, dcp_size),
        n_head=2,
        head_dim=4,
        topk_tokens=2048,
        q_lora_rank=4,
        index_kpool=4,
        wq_b=torch.nn.Linear(4, 8, bias=False),
        wk_weights_proj=torch.nn.Linear(8, 6, bias=False),
        k_norm=torch.nn.LayerNorm(4),
        softmax_scale=0.5,
        index_kpool_compress_ape=torch.nn.Parameter(torch.zeros(4, 4)),
        index_kpool_compress_gate=torch.nn.Parameter(torch.zeros(4, 8)),
        k_cache=SimpleNamespace(),
        tail_cache=SimpleNamespace(),
        topk_indices_buffer=None,
    )
    backend = kpool.Glm5NextKPoolIndexerBackend(source, qk_rope_head_dim=0)
    assert backend.k_cache is source.k_cache
    assert backend.topk_output_width == 2051


@pytest.mark.parametrize("dcp_size", [1, 2, 4])
@pytest.mark.parametrize("block_size", [128, 512, 640])
@pytest.mark.parametrize("pool", [4, 16])
def test_kpool_dcp_cache_pages_cover_replicated_history(dcp_size, block_size, pool):
    spec = _spec(block_size, dcp_size, pool)
    assert spec.page_size_bytes == block_size // pool * 128 * 2 * dcp_size
    assert AscendIndexerKPoolSpec.merge([spec, spec]) == spec
    config = _config(block_size, dcp_size)
    num_pages = math.ceil(config.model_config.max_model_len / (block_size * dcp_size))
    assert spec.max_memory_usage_bytes(config) == num_pages * spec.page_size_bytes
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.uint8)
    (cache,) = view_compressed_indexer_cache("indexer", spec, raw, AscendIndexerKPoolBackend, 128)
    assert cache.shape == (3 * dcp_size * (block_size // 128), 128 // pool, 1, 128)
    assert cache.data_ptr() == raw.data_ptr()
    # Adjacent scheduler pages must not alias, including their replicated rows.
    rows_per_page = dcp_size * (block_size // pool)
    flat = cache.view(-1, 128)
    flat[rows_per_page - 1].fill_(1)
    flat[rows_per_page].fill_(2)
    assert flat[rows_per_page - 1].eq(1).all()
    assert flat[rows_per_page].eq(2).all()
    assert enable_sfa_dcp_replicated_indexer(config) == (dcp_size > 1)


def test_kpool_dcp_rejects_inconsistent_replication_and_keeps_tail_replicated():
    spec = _spec(128, 2, 4)
    with pytest.raises(AssertionError, match="replication size"):
        AscendIndexerKPoolSpec.merge([spec, replace(spec, dcp_replication_size=4)])
    tail = AscendIndexerKPoolTailSpec(
        block_size=4, sliding_window=4, compress_ratio=4, num_kv_heads=1, head_size=128, dtype=torch.float32
    )
    assert not tail.dcp_sharded


@pytest.mark.parametrize(
    ("pool_size", "buffer_width", "expected_width"),
    [(None, None, 2048), (1, None, 2048), (4, None, 2051), (4, 2048, 2051), (4, 2176, 2176), (16, None, 2063)],
)
@pytest.mark.parametrize("config_field", ["hf_text_config", "hf_config"])
def test_sfa_dcp_indexer_width_includes_tail_and_buffer_padding(pool_size, buffer_width, expected_width, config_field):
    config = SimpleNamespace(index_topk=2048)
    if pool_size is not None:
        config.index_kpool = pool_size
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(**{config_field: config}))
    buffer = torch.empty(1, buffer_width, dtype=torch.int32) if buffer_width is not None else None
    assert cp._get_sfa_indexer_output_width(vllm_config, buffer) == expected_width


@pytest.mark.parametrize("index_topk", [None, 0, -1])
def test_sfa_dcp_indexer_width_rejects_missing_or_invalid_topk(index_topk):
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=index_topk)))
    with pytest.raises(RuntimeError, match="index_topk must be set"):
        cp._get_sfa_indexer_output_width(config)


@pytest.mark.parametrize("shared_indexer", [False, True])
@pytest.mark.parametrize("buffer_width", [None, 2051, 2176])
def test_kpool_dcp_remap_capacity_includes_unpooled_tail(monkeypatch, shared_indexer, buffer_width):
    def base_init(self, *args, **kwargs):
        self.vllm_config = _config(128, 2)
        self.vllm_config.parallel_config.cp_kv_cache_interleave_size = 128
        self.q_proj = SimpleNamespace(weight=torch.empty(1))
        self.indexer = None if shared_indexer else SimpleNamespace(topk_output_width=2051)
        self.topk_indices_buffer = torch.empty(1, buffer_width, dtype=torch.int32) if buffer_width is not None else None
        self.dcp_size = 2

    monkeypatch.setattr(cp.DCPImplMixin, "__init__", base_init)
    impl = cp.AscendSFADCPImpl(4, 128, 1.0, 1, None, None, "auto", None, "decoder", None)
    expected_width = max(2051, buffer_width or 0)
    assert impl._remap_order.numel() == expected_width
    empty_queries = torch.empty(0, 1, expected_width, dtype=torch.int32)
    assert impl._remap_sparse_indices(empty_queries) is empty_queries
    with pytest.raises(RuntimeError, match="exceeds the configured indexer output width"):
        impl._remap_sparse_indices(torch.empty(0, 1, expected_width + 1, dtype=torch.int32))


@pytest.mark.parametrize("dcp_size", [2, 8])
@pytest.mark.parametrize("interleave_size", [128, 640])
@pytest.mark.parametrize("dcp_rank", [0, 1])
def test_kpool_dcp_remaps_shared_mtp_buffer_and_preserves_padding(monkeypatch, dcp_size, interleave_size, dcp_rank):
    logical_width, buffer_width = 2051, 2176
    indices = (torch.arange(logical_width, dtype=torch.int32) * 3 + interleave_size - 1).repeat(2, 1)
    indices[1] = indices[1].flip(0)
    indices[:, ::7] = -1

    def base_init(self, *args, **kwargs):
        self.vllm_config = _config(interleave_size, dcp_size)
        self.vllm_config.parallel_config.cp_kv_cache_interleave_size = interleave_size
        self.q_proj = SimpleNamespace(weight=torch.empty(1))
        self.indexer = SimpleNamespace(topk_output_width=logical_width)
        self.topk_indices_buffer = torch.full((2, buffer_width), -1, dtype=torch.int32)
        self.dcp_size, self.dcp_rank = dcp_size, dcp_rank

    monkeypatch.setattr(cp.DCPImplMixin, "__init__", base_init)
    monkeypatch.setattr(cp, "HAS_TRITON", False)
    # The NPU fallback uses int32 gather indices; CPU PyTorch requires int64.
    gather = torch.gather
    monkeypatch.setattr(cp.torch, "gather", lambda tensor, dim, index: gather(tensor, dim, index.to(torch.int64)))
    impl = cp.AscendSFADCPImpl(4, 128, 1.0, 1, None, None, "auto", None, "decoder", None)
    impl._update_indexcache_topk_indices(indices.unsqueeze(1))
    shared_indices = impl._get_indexcache_topk_indices(indices.shape[0])
    original = shared_indices.clone()
    remapped = impl._remap_sparse_indices(shared_indices)
    unpadded = impl._remap_sparse_indices(indices.unsqueeze(1))

    assert remapped.shape == (2, 1, buffer_width)
    torch.testing.assert_close(remapped[..., :logical_width], unpadded)
    assert remapped[..., logical_width:].eq(-1).all()
    torch.testing.assert_close(shared_indices, original)
    for row, actual in zip(indices.tolist(), remapped[:, 0].tolist()):
        expected = [
            index // (dcp_size * interleave_size) * interleave_size + index % interleave_size
            for index in row
            if index >= 0 and index // interleave_size % dcp_size == dcp_rank
        ]
        assert actual == expected + [-1] * (buffer_width - len(expected))


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("block_size", [128, 512, 640])
def test_kpool_dcp_metadata_uses_global_slots_and_persistent_buffers(monkeypatch, dcp_size, block_size):
    monkeypatch.setattr(kpool, "enable_dsa_cp", lambda: False)
    config = _config(block_size, dcp_size)
    builder = AscendIndexerKPoolMetadataBuilder(
        _spec(block_size, dcp_size, 4), ["model.layers.0.indexer.k_cache"], config, torch.device("cpu")
    )
    pages = torch.tensor([[2, 5], [7, 9]], dtype=torch.int32)
    subpages = block_size // 128
    local_table = (pages.unsqueeze(-1) * subpages + torch.arange(subpages)).reshape(2, -1)
    positions = torch.tensor([block_size - 1, block_size, 2 * block_size - 1, 2 * block_size, 0, 99999])
    query_start = torch.tensor([0, 4, 5], dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=query_start,
        query_start_loc_cpu=query_start.clone(),
        seq_lens=torch.tensor([2 * block_size + 1, 1], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        num_input_tokens=6,
        max_query_len=4,
        max_seq_len=2 * block_size + 1,
        positions=positions,
        block_table_tensor=local_table,
        slot_mapping=torch.full((6,), -1, dtype=torch.int64),
    )
    buffers = builder._get_metadata_buffers(common)
    replicated = builder._replicated_cache_metadata(common, buffers)
    expected = []
    for req, pos in zip([0, 0, 0, 0, 1], positions[:5].tolist()):
        page = pages[req, pos // (block_size * dcp_size)].item()
        expected.append(page * block_size * dcp_size + pos % (block_size * dcp_size))
    assert replicated.slot_mapping.tolist() == [*expected, -1]
    assert common.slot_mapping.eq(-1).all()
    assert replicated.seq_lens is common.seq_lens
    again = builder._replicated_cache_metadata(common, builder._get_metadata_buffers(common))
    assert again.block_table_tensor.data_ptr() == replicated.block_table_tensor.data_ptr()
    assert again.slot_mapping.data_ptr() == replicated.slot_mapping.data_ptr()
    # PCP's global write view must own different buffers from local queries.
    write = builder._replicated_cache_metadata(common, builder._get_metadata_buffers(common, write=True))
    torch.testing.assert_close(write.slot_mapping, replicated.slot_mapping)
    assert write.slot_mapping.data_ptr() != replicated.slot_mapping.data_ptr()


@pytest.mark.parametrize("pcp_size", [1, 2])
@pytest.mark.parametrize("capture_sizes", [[], [8, 16, 192]])
def test_kpool_dcp_block_table_capacity_uses_scheduler_pages_and_request_buckets(monkeypatch, pcp_size, capture_sizes):
    monkeypatch.setattr(kpool, "enable_dsa_cp", lambda: False)
    config = _config(640, 8)
    config.model_config.max_model_len = 1048576
    config.scheduler_config.max_num_batched_tokens = 8192
    config.scheduler_config.max_num_seqs = 32
    config.parallel_config.prefill_context_parallel_size = pcp_size
    config.compilation_config = SimpleNamespace(cudagraph_capture_sizes=capture_sizes)
    # V2 replaces the scheduler spec's block size with the kernel block size.
    builder = AscendIndexerKPoolMetadataBuilder(
        _spec(128, 8, 4), ["model.layers.0.indexer.k_cache"], config, torch.device("meta")
    )
    common = SimpleNamespace(slot_mapping=torch.zeros(6, dtype=torch.int64))
    buffers = builder._get_metadata_buffers(common)
    expected_rows = max([32 * (2 if pcp_size > 1 else 1), *capture_sizes]) + 1
    assert buffers.replicated_block_table.shape == (expected_rows, 8200)
    assert buffers.replicated_block_table.numel() * buffers.replicated_block_table.element_size() < 8 * 1024**2
    assert builder._get_metadata_buffers(common) is buffers
    write = builder._get_metadata_buffers(common, write=True)
    assert write.replicated_block_table.shape == buffers.replicated_block_table.shape
    assert write.replicated_block_table is not buffers.replicated_block_table
    with pytest.raises(RuntimeError, match="replicated block-table capacity"):
        builder._replicated_cache_metadata(SimpleNamespace(num_reqs=expected_rows + 1), buffers)


@pytest.mark.parametrize("pcp", [False, True])
def test_nope_dcp_prefill_keeps_scheduler_subpages_together(pcp):
    cls = cp.AscendSFAPCPDCPMetadataBuilder if pcp else cp.AscendSFADCPMetadataBuilder
    builder = cls.__new__(cls)
    builder.nope, builder.blocks_per_phys_block, builder.dcp_size = True, 4, 2
    builder.device = torch.device("cpu")
    builder.arange_buffer = torch.arange(4, dtype=torch.int32)
    builder.dcp_collective_rank_order = torch.tensor([1, 0], dtype=torch.int32)
    table = torch.tensor([[8, 9, 10, 11]], dtype=torch.int32)
    kwargs = dict(global_dcp_block_table=table, global_dcp_num_blocks=torch.tensor([4])) if pcp else {}
    selected, gathered_table = builder._build_compact_kv_gather_metadata(table, **kwargs)
    torch.testing.assert_close(selected, table[0])
    assert gathered_table.tolist() == [[4, 5, 6, 7, 0, 1, 2, 3]]


@pytest.mark.parametrize("pcp_size", [1, 2])
@pytest.mark.parametrize("block_size", [128, 640])
def test_nope_dcp_metadata_capacity_covers_partial_pages_and_pcp_rows(monkeypatch, pcp_size, block_size):
    monkeypatch.setattr(sfa, "get_current_hardware_profile", lambda: SimpleNamespace(device_adaptor_family=None))
    config = _config(block_size, 2)
    config.parallel_config.prefill_context_parallel_size = pcp_size
    config.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[])
    state = sfa.SparseMLAMetadataState(_spec(block_size, 2, 4), config, torch.device("cpu"))
    assert state.block_table_buffer.shape == (2 * pcp_size + 1, 4 * (block_size // 128))


@pytest.mark.parametrize("prefill", [False, True])
def test_nope_dcp_uses_single_cache_and_the_correct_attention_view(monkeypatch, prefill):
    impl = cp.AscendSFADCPImpl.__new__(cp.AscendSFADCPImpl)
    impl.qk_rope_head_dim, impl.scale = 0, 0.125
    impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.dcp_group = object()
    impl._has_prefill = lambda _: prefill
    query = torch.randn(2, 4, 128)
    rope = query[..., :0]
    cache = torch.randn(3, 128, 1, 128)
    gathered_cache = torch.cat([cache, cache])
    indices = torch.zeros(2, 1, 2051, dtype=torch.int32)
    context = cp.DCPContext(
        slot_mapping=torch.tensor([0, -1]),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        kv_gather_block_ids=torch.tensor([0, 2]),
        kv_gather_block_table=torch.tensor([[0, 3]], dtype=torch.int32),
        smla_sinks=torch.full((4,), 1.0 - math.log(2)),
    )
    metadata = cp.AscendSFADCPMetadata.__new__(cp.AscendSFADCPMetadata)
    # Populate all dataclass fields so replace() retains the metadata type.
    for field in fields(metadata):
        if not hasattr(metadata, field.name):
            setattr(metadata, field.name, None)
    metadata.dcp_context = context
    metadata.num_prefills = int(prefill)
    impl._start_dcp_gather = Mock(return_value=object())
    impl._record_dcp_kv_gather_context((cache,), metadata)
    if prefill:
        torch.testing.assert_close(impl._start_dcp_gather.call_args.args[0], cache[[0, 2]])
    else:
        context.gather_context = object()
    impl._finish_dcp_gather = Mock(return_value=(gathered_cache,) if prefill else (query, rope))
    impl._remap_sparse_indices = Mock(return_value=indices)
    output, lse = torch.randn_like(query), torch.zeros(2, 4, 1)
    operator = Mock(return_value=output if prefill else (output, lse))
    monkeypatch.setattr(cp, "sparse_mla", operator)
    monkeypatch.setattr(cp, "enable_sfa_dcp_force_tmajor_restore", lambda: False)
    impl._merge_dcp_outputs = Mock(return_value=output)
    actual = impl._execute_sparse_flash_attention_process(query, rope, (cache,), indices, metadata, None, None)
    torch.testing.assert_close(actual, output)
    assert operator.call_args.args[1] is (gathered_cache if prefill else cache)
    assert operator.call_args.args[3].block_table is (context.kv_gather_block_table if prefill else context.block_table)
    assert impl._merge_dcp_outputs.call_count == int(not prefill)
    if not prefill:
        assert operator.call_args.kwargs["sparse_mode"] == 0
        assert operator.call_args.kwargs["return_lse"]
        torch.testing.assert_close(operator.call_args.kwargs["sinks"], context.smla_sinks)


@pytest.mark.parametrize("a5", [False, True])
def test_nope_dcp_lse_merge_matches_unsharded_attention(monkeypatch, a5):
    # An empty local selection must contribute zero output, not NaNs. On A5
    # each shard contributes only 1/D of the original sink's denominator.
    scores = torch.tensor([0.3, -0.8, 1.2])
    values = torch.randn(3, 128)
    query = torch.zeros(2, 2, 128)
    indices = torch.tensor([[[0, 1, -1]], [[-1, -1, -1]]], dtype=torch.int32)
    if a5:
        # Stale captured indices do not make a padded query valid.
        indices[1].fill_(0)
    sinks = torch.full((2,), sfa.SMLA_DEFAULT_SINK_VALUE - math.log(2))
    metadata = SimpleNamespace(
        block_size=128,
        block_table=torch.zeros(1, 2, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        smla_metadata=torch.zeros(1024, dtype=torch.int32) if a5 else None,
        smla_topk_length=torch.zeros(2, 1, dtype=torch.int32),
        smla_valid_mask=torch.tensor([[True], [False]]),
        smla_sinks=torch.ones(2),
    )
    monkeypatch.setattr(sfa, "build_smla_metadata", lambda *args: None)
    partial_outputs, partial_lses = [], []
    for selected in ([0, 1], [2]):
        logits, vals = scores[selected], values[selected]
        mass = logits.exp().sum() + (sinks[0].exp() if a5 else 0)
        out = (logits.exp()[:, None] * vals).sum(0) / mass
        raw_output = torch.full_like(query, torch.nan)
        raw_output[0] = out
        raw_lse = torch.full((1, 2, 2), torch.nan)
        raw_lse[0, 0] = mass.log()
        if a5:

            def smla_operator(q, raw_output=raw_output, raw_lse=raw_lse, **kwargs):
                assert kwargs["return_softmax_lse"]
                torch.testing.assert_close(kwargs["sinks"], sinks)
                return raw_output, raw_lse

            monkeypatch.setattr(sfa, "sparse_flash_mla", smla_operator)
        else:

            def sfa_operator(raw_output=raw_output, raw_lse=raw_lse, **kwargs):
                assert kwargs["sparse_mode"] == 0 and kwargs["return_softmax_lse"]
                return raw_output, raw_lse.permute(2, 1, 0), torch.ones(2, 2, 1)

            monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention", sfa_operator, raising=False)
        out, lse = sfa.sparse_mla(
            query,
            torch.zeros(1, 128, 1, 128),
            indices,
            metadata,
            1.0,
            return_lse=True,
            sparse_mode=0,
            sinks=sinks,
        )
        assert out[1].eq(0).all()
        if a5:
            torch.testing.assert_close(lse[1, :, 0], sinks)
        else:
            assert lse[1].isneginf().all()
        partial_outputs.append(out[0])
        partial_lses.append(lse[0])
    weights = torch.stack(partial_lses).softmax(0)
    merged = (torch.stack(partial_outputs) * weights).sum(0)
    denominator = scores.exp().sum() + (math.exp(sfa.SMLA_DEFAULT_SINK_VALUE) if a5 else 0)
    expected = (scores.exp()[:, None] * values).sum(0) / denominator
    torch.testing.assert_close(merged, expected.expand_as(merged))
