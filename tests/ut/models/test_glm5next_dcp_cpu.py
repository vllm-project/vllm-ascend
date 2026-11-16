# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts; run with pytest --noconftest on machines without CANN.

Load production functions without importing the NPU runtime. Device kernels
are covered separately by test_glm5_next_dcp_indexer on A3/A5.
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names, **scope):
    source = ROOT / "vllm_ascend" / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    scope["torch"] = torch
    exec(compile(module, str(source), "exec"), scope)
    return SimpleNamespace(**scope)


def load_method(path, class_name, method, **scope):
    source = ROOT / "vllm_ascend" / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    scope["torch"] = torch
    exec(compile(module, str(source), "exec"), scope)
    return scope[method]


def config(world=2, interleave=128):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=world,
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=interleave,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_kpool=4, index_topk=2048), max_model_len=1048576
        ),
        cache_config=SimpleNamespace(block_size=512),
    )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [4, 128, 512])
def test_dcp_config(world, interleave):
    validate = load_functions("models/glm5next/dcp.py", ["validate_kpool_dcp"]).validate_kpool_dcp
    validate(config(world, interleave))


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("parallel_config", "prefill_context_parallel_size", 4, "DCP=PCP"),
        ("parallel_config", "tensor_parallel_size", 3, "must divide"),
        ("parallel_config", "cp_kv_cache_interleave_size", 1, "divisible"),
        ("parallel_config", "cp_kv_cache_interleave_size", 192, "dividing"),
        ("model_config", "max_model_len", (1 << 24) + 1, "2\\*\\*24"),
    ],
)
def test_dcp_config_rejects_invalid_layout(section, field, value, message):
    cfg = config()
    setattr(getattr(cfg, section), field, value)
    validate = load_functions("models/glm5next/dcp.py", ["validate_kpool_dcp"]).validate_kpool_dcp
    with pytest.raises((ValueError, NotImplementedError), match=message):
        validate(cfg)


def test_early_recoverssm_validation_leaves_layout_to_platform():
    validate = load_functions("models/glm5next/dcp.py", ["validate_kpool_dcp"]).validate_kpool_dcp
    cfg = config(interleave=1)
    cfg.cache_config.block_size = None
    validate(cfg, check_cache_layout=False)
    cfg.parallel_config.prefill_context_parallel_size = 2
    validate(cfg, check_cache_layout=False)
    cfg.parallel_config.decode_context_parallel_size = 4
    with pytest.raises(ValueError, match="DCP=PCP"):
        validate(cfg, check_cache_layout=False)


@pytest.mark.parametrize("tp,pcp,dcp", [(1, 2, 2), (2, 2, 2), (2, 2, 4), (4, 2, 8), (2, 4, 8)])
def test_joint_pcp_dcp_topologies(tp, pcp, dcp):
    cfg = config(dcp)
    cfg.parallel_config.tensor_parallel_size = tp
    cfg.parallel_config.prefill_context_parallel_size = pcp
    load_functions("models/glm5next/dcp.py", ["validate_kpool_dcp"]).validate_kpool_dcp(cfg)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("prefix", [False, True])
def test_recoverssm_dcp_validation_keeps_prefix_contract(monkeypatch, world, prefix):
    validator = load_functions("models/glm5next/dcp.py", ["validate_kpool_dcp"])
    monkeypatch.setitem(sys.modules, "vllm_ascend.models.glm5next.dcp", validator)
    validate = load_functions("models/glm5next/recoverssm_config.py", ["validate_glm_recoverssm"])
    cfg = config(world)
    cfg.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    cfg.mamba_config = SimpleNamespace(enable_stochastic_rounding=False)
    cfg.cache_config.enable_prefix_caching = prefix
    cfg.cache_config.mamba_cache_mode = "align" if prefix else "none"
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2", kv_role="kv_consumer")
    validate.validate_glm_recoverssm(cfg)
    cfg.cache_config.mamba_cache_mode = "none" if prefix else "align"
    with pytest.raises(ValueError, match="Mamba cache mode"):
        validate.validate_glm_recoverssm(cfg)


@pytest.mark.parametrize("kpool", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_metadata_preserves_local_cache_and_restores_common_view(kpool, fails):
    build = load_method(
        "attention/context_parallel/sfa_cp.py",
        "AscendSFADCPMetadataBuilder",
        "_build_with_metadata_view",
        enable_kpool_dcp=lambda _: kpool,
        enable_sfa_dcp_sharded_indexer=lambda _: False,
        is_pd_decode_recompute_scheduler_enabled=lambda _: False,
        split_decodes_and_prefills=lambda *args, **kwargs: (0, 1, 0, 2),
        AscendSFADCPMetadata=SimpleNamespace,
        DCPContext=SimpleNamespace,
    )
    slots = torch.tensor([-1, 130])
    full_table = torch.tensor([[1, 2, 0, 0]], dtype=torch.int32)
    local_table = full_table[:, :2].contiguous()
    replicated_table = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
    replicated_slots = torch.tensor([258, 386])
    common = SimpleNamespace(
        slot_mapping=slots,
        block_table_tensor=full_table,
        num_reqs=1,
        seq_lens=torch.tensor([260]),
        dcp_local_seq_lens=torch.tensor([132]),
    )
    builder = SimpleNamespace(
        vllm_config=config(),
        device=torch.device("cpu"),
        use_dcp=True,
        decode_threshold=1,
        dcp_local_seq_lens_buf=torch.empty(4, dtype=torch.int32),
        _get_dcp_local_block_table=Mock(return_value=local_table),
        _build_block_table_replicated_view=Mock(return_value=replicated_table),
        _build_slot_mapping_replicated_view=Mock(return_value=replicated_slots),
        _build_compact_kv_gather_metadata=Mock(return_value=(torch.tensor([1, 2]), replicated_table)),
    )

    def build_metadata():
        assert common.block_table_tensor is (local_table if kpool else replicated_table)
        assert common.slot_mapping is (slots if kpool else replicated_slots)
        if fails:
            raise RuntimeError("metadata failed")
        return SimpleNamespace(seq_lens=common.seq_lens, pcp_has_global_prefill=False)

    if fails:
        with pytest.raises(RuntimeError, match="metadata failed"):
            build(builder, common, build_metadata)
    else:
        metadata = build(builder, common, build_metadata)
        assert metadata.dcp_context.slot_mapping is slots
        assert metadata.dcp_context.block_table is local_table
        assert metadata.dcp_context.seq_lens.tolist() == [132]
        assert (metadata.dcp_context.kv_gather_block_table is None) == kpool
        assert builder._build_compact_kv_gather_metadata.call_count == (0 if kpool else 1)
    assert common.block_table_tensor is full_table
    assert common.slot_mapping is slots
    assert builder._build_block_table_replicated_view.call_count == (0 if kpool else 1)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_kpool_routing_does_not_change_kimi_layout(world):
    funcs = load_functions("utils.py", ["enable_kpool_dcp", "model_uses_kpool_indexer", "model_uses_sfa_sparse"])
    cfg = config(world)
    assert funcs.enable_kpool_dcp(cfg) == (world > 1)
    assert not funcs.model_uses_sfa_sparse(cfg.model_config)
    del cfg.model_config.hf_text_config.index_kpool
    assert not funcs.enable_kpool_dcp(cfg)
    assert funcs.model_uses_sfa_sparse(cfg.model_config)
    assert not funcs.enable_kpool_dcp(SimpleNamespace())


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [4, 128, 512])
def test_only_owner_writes_completed_pool_including_mtp_rollback(world, interleave):
    convert = load_functions("models/glm5next/kv_cache.py", ["format_indexer_kpool_slot_mapping"])
    # Visits straddle pools and owner boundaries, then roll back rejected drafts.
    visits = torch.tensor([interleave - 2, interleave - 1, interleave, interleave + 1, interleave - 1])
    local = visits // (world * interleave) * interleave + visits % interleave
    owner = visits // interleave % world
    writes = []
    for rank in range(world):
        slots = torch.where(owner == rank, local + 7 * 512, -1)
        compressed = convert.format_indexer_kpool_slot_mapping(slots, visits, 512, 4)
        writes.append(compressed >= 0)
        valid = compressed >= 0
        torch.testing.assert_close(compressed[valid], 7 * 128 + local[valid] // 4)
    torch.testing.assert_close(torch.stack(writes).sum(0), ((visits + 1) % 4 == 0).long())


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("empty_owner", [False, True])
def test_a5_sink_is_counted_once_in_global_softmax(world, empty_owner):
    add_sink = load_functions(
        "attention/context_parallel/sfa_cp.py", ["_add_nope_dcp_sink"], SMLA_DEFAULT_SINK_VALUE=1.0
    )._add_nope_dcp_sink
    generator = torch.Generator().manual_seed(20883)
    logits = torch.randn(world, 3, 5, 7, generator=generator).double() * 5
    values = torch.randn(world, 3, 5, 7, 8, generator=generator).double()
    if empty_owner:
        logits[0].fill_(-torch.inf)
    maxima = logits.amax(-1)
    exponentials = torch.nan_to_num(torch.exp(logits - maxima[..., None]), nan=0)
    sums = exponentials.sum(-1)
    partials = (exponentials[..., None] * values).sum(-2) / sums[..., None]
    outputs, stats, totals = [], [], []
    for rank in range(world):
        out, maximum, total = partials[rank], maxima[rank][None], sums[rank][None]
        if rank == 0:
            out, maximum, total = add_sink(out, maximum, total)
        outputs.append(out)
        stats.append(maximum[0])
        totals.append(total[0])
    lses = torch.stack(stats) + torch.stack(totals).log()
    combined = (torch.stack(outputs) * lses.softmax(0)[..., None]).sum(0)
    global_logits = logits.permute(1, 2, 0, 3).flatten(2)
    global_values = values.permute(1, 2, 0, 3, 4).flatten(2, 3)
    probabilities = torch.cat([global_logits, torch.ones(3, 5, 1)], -1).softmax(-1)
    expected = (probabilities[..., :-1, None] * global_values).sum(-2)
    torch.testing.assert_close(combined, expected, rtol=1e-6, atol=1e-7)


class Kernel:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        return self.function


def indexer_functions(**overrides):
    namespace = dict(
        triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b, next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        NEG_INF_SENTINEL=torch.finfo(torch.float32).min,
        QUERY_END_SENTINEL=(1 << 31) - 1,
        DCP_POOL_TILE=32,
        TRITON_MAX_PROGRAMS=256,
        TRITON_INDEX_BLOCK=256,
        DCP_SCORE_BYTES=32 << 20,
    )
    namespace.update(overrides)
    return load_functions(
        "ops/triton/glm5_next_dcp_indexer.py",
        ["select_local_pool_candidates", "glm5_next_dcp_indexer"],
        **namespace,
    )


def test_local_topk_keeps_ties_and_pads_empty_owners():
    lowest = torch.finfo(torch.float32).min
    raw = torch.tensor([[3.0, 3.0, lowest, 5.0], [lowest] * 4])

    def score(*args):
        args[7].copy_(raw)

    funcs = indexer_functions(_score_local_pools=Kernel(score))
    ids, scores = funcs.select_local_pool_candidates(
        torch.zeros(2, 8),
        torch.zeros(1, 4, 1, 8),
        SimpleNamespace(
            cum_query_lens=torch.tensor([2]),
            seq_lens=torch.tensor([4]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            positions=torch.arange(2),
        ),
        world_size=4,
        rank=3,
        interleave=1,
        pool_size=4,
        pool_topk=6,
        capacity=4,
    )
    assert ids.tolist() == [[[3, 0, 1, -1, -1, -1]], [[-1] * 6]]
    assert scores[0, 0, :3].tolist() == [5, 3, 3]
    assert scores[1].isneginf().all()


def test_full_graph_padding_uses_device_query_ends():
    merge = Mock(return_value=torch.full((3, 1, 2), -1, dtype=torch.int32))
    expand = Mock()
    funcs = indexer_functions(distributed_topk=merge, _expand_pool_indices=Kernel(expand))
    # Zero local capacity still participates in collectives on every rank.
    metadata = SimpleNamespace(
        cum_query_lens=torch.tensor([0]),
        positions=torch.arange(3),
        block_table=torch.empty((1, 0), dtype=torch.int32),
    )
    owner = torch.full((5, 15), 99, dtype=torch.int32)
    result = funcs.glm5_next_dcp_indexer(
        torch.zeros(3, 8),
        torch.zeros(1, 4, 1, 8),
        metadata,
        SimpleNamespace(world_size=4, rank_in_group=3),
        topk=8,
        pool_size=4,
        interleave=128,
        max_model_len=8192,
        output_buffer=owner[1:4],
    )
    assert result.shape == (3, 1, 11)
    assert result.data_ptr() == owner[1:].data_ptr()
    ids, scores, _, k, interleave = merge.call_args.args
    assert (k, interleave) == (2, 32)
    assert (ids == -1).all() and scores.isneginf().all()
    # Python must not skip expansion when the captured real-token count is 0.
    assert expand.call_args.args[3] is metadata.cum_query_lens
    assert expand.call_args.args[12:14] == (15, 15)
    assert (owner[[0, 4]] == 99).all()


def test_output_rows_cannot_alias():
    funcs = indexer_functions()
    with pytest.raises(ValueError, match="output must cover"):
        funcs.glm5_next_dcp_indexer(
            torch.zeros(3, 8),
            torch.zeros(1, 4, 1, 8),
            None,
            None,
            topk=8,
            pool_size=4,
            interleave=128,
            max_model_len=8192,
            output_buffer=torch.empty(1, 11, dtype=torch.int32).expand(3, 11),
        )


@pytest.mark.parametrize("qbar", [torch.zeros(3, 2, 8), torch.zeros(3, 8).bfloat16(), torch.zeros(3, 16)[:, ::2]])
def test_weighted_query_contract(qbar):
    with pytest.raises(ValueError, match="contiguous FP32 weighted query"):
        indexer_functions().glm5_next_dcp_indexer(
            qbar, None, None, None, topk=8, pool_size=4, interleave=128, max_model_len=8192
        )


def test_weighted_query_tiles_keep_global_token_offsets():
    qbar = torch.arange(5 * 8).reshape(5, 8).float()
    seen = []

    def score(*args):
        seen.append((args[0].clone(), args[9]))
        args[7].zero_()

    funcs = indexer_functions(
        DCP_SCORE_BYTES=2 * 4 * 4,
        _score_local_pools=Kernel(score),
        distributed_topk=lambda ids, *args: ids,
        _expand_pool_indices=Kernel(Mock()),
    )
    funcs.glm5_next_dcp_indexer(
        qbar,
        torch.zeros(1, 4, 1, 8),
        SimpleNamespace(
            cum_query_lens=torch.tensor([5]),
            positions=torch.arange(5),
            seq_lens=torch.tensor([16]),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
        ),
        SimpleNamespace(world_size=4, rank_in_group=0),
        topk=8,
        pool_size=4,
        interleave=4,
        max_model_len=64,
    )
    assert [offset for _, offset in seen] == [0, 2, 4]
    torch.testing.assert_close(torch.cat([tile for tile, _ in seen]), qbar, rtol=0, atol=0)


def pcp_functions():
    return load_functions(
        "attention/context_parallel/kpool_pcp.py",
        ["build_kpool_pcp_query_layout", "gather_kpool_pcp_queries", "restore_kpool_pcp_queries"],
        KPoolPCPQueryLayout=SimpleNamespace,
    )


def pcp_batch(pcp, mixed, short, remaining_prompt_tokens=None):
    # Two separated prefill chunks per rank, with an optional replicated decode.
    tokens = 1 if short else 17
    chunks = torch.tensor_split(torch.arange(int(mixed), tokens + int(mixed)), 2 * pcp)
    rank_indices = [torch.cat((chunks[r], chunks[2 * pcp - 1 - r])) for r in range(pcp)]
    if mixed:
        rank_indices = [torch.cat((torch.tensor([0]), indices)) for indices in rank_indices]
    if remaining_prompt_tokens is not None:
        assert pcp == 2 and not mixed
        tokens = remaining_prompt_tokens + 5
        rank_indices = (
            [torch.arange(6), torch.arange(6)]
            if remaining_prompt_tokens == 1
            else [torch.tensor([0, 1, 6]), torch.tensor([2, 3, 4, 5])]
        )
    width = max(map(len, rank_indices)) + 1
    padded = torch.zeros(pcp, width, dtype=torch.long)
    restore = torch.full((tokens + int(mixed),), -1, dtype=torch.long)
    for rank, indices in enumerate(rank_indices):
        padded[rank, : len(indices)] = indices
        for offset, index in enumerate(indices):
            if restore[index] < 0:
                restore[index] = rank * width + offset
    assert (restore >= 0).all()
    return SimpleNamespace(
        padded_gather_idx=padded.flatten(),
        hidden_restore_idx=restore,
        global_batch=SimpleNamespace(num_tokens=restore.numel()),
    ), rank_indices


@pytest.mark.parametrize("pcp", [2, 4])
@pytest.mark.parametrize("mixed,short", [(False, False), (True, False), (False, True), (True, True)])
def test_pcp_queries_align_and_restore_with_empty_ranks(pcp, mixed, short):
    funcs = pcp_functions()
    context, rank_indices = pcp_batch(pcp, mixed, short)
    global_query = torch.arange(context.global_batch.num_tokens * 16).reshape(-1, 2, 8).float()
    width = context.padded_gather_idx.numel() // pcp
    sends = [global_query.new_full((width, 2, 8), -999) for _ in range(pcp)]
    for send, indices in zip(sends, rank_indices):
        send[: len(indices)] = global_query[indices]
    for rank, indices in enumerate(rank_indices):
        group = SimpleNamespace(world_size=pcp, rank_in_group=rank)
        group.all_gather = Mock(return_value=torch.cat(sends))
        common = SimpleNamespace(num_input_tokens=width, num_actual_tokens=len(indices))
        layout = funcs.build_kpool_pcp_query_layout(common, context, group)
        # Exercise the short-input padding path as well as a completely empty rank.
        gathered = funcs.gather_kpool_pcp_queries(sends[rank][: len(indices)], layout, group)
        torch.testing.assert_close(gathered, global_query)
        assert group.all_gather.call_args.args[0].shape[0] == width
        local = funcs.restore_kpool_pcp_queries(gathered, layout, width, fill_value=-1)
        torch.testing.assert_close(local[: len(indices)], global_query[indices])
        assert (local[len(indices) :] == -1).all()


@pytest.mark.parametrize("global_rank", range(4))
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "fp8_dequantized"])
@pytest.mark.parametrize(
    "mixed,short,remaining_prompt_tokens",
    [
        (False, False, None),
        (True, False, None),
        (False, True, None),
        (True, True, None),
        (False, False, 1),
        (False, False, 2),
    ],
    ids=["prefill", "mixed", "empty_rank", "short_mixed", "prompt1_mtp5", "prompt2_mtp5"],
)
def test_tp2_pcp2_dcp4_reduces_query_before_single_gather(
    monkeypatch, global_rank, quantized, mixed, short, remaining_prompt_tokens
):
    funcs = pcp_functions()
    context, rank_indices = pcp_batch(2, mixed, short, remaining_prompt_tokens)
    width = context.padded_gather_idx.numel() // 2
    generator = torch.Generator().manual_seed(21835)
    global_query = torch.randn(context.global_batch.num_tokens, 64, 128, generator=generator).bfloat16()
    if quantized:
        # CPU stand-in for the already-dequantized query supplied by
        # _forward_cann. Actual npu_dynamic_quant is covered by the HCCL test.
        scale = global_query.float().abs().amax(-1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max
        global_query = (global_query.float() / scale).to(torch.float8_e4m3fn).float() * scale
    # Non-BF16 values ensure moving the reduction cannot omit weight rounding.
    global_weights = torch.randn(context.global_batch.num_tokens, 64, generator=generator)
    indices = context.padded_gather_idx
    rank = global_rank // 2
    group = SimpleNamespace(world_size=2, rank_in_group=rank)
    dcp_group = SimpleNamespace(world_size=4, rank_in_group=(global_rank % 2) * 2 + rank)
    group.all_gather = Mock(side_effect=[global_query[indices], global_weights[indices]])
    common = SimpleNamespace(num_input_tokens=width, num_actual_tokens=len(rank_indices[rank]))
    layout = funcs.build_kpool_pcp_query_layout(common, context, group)
    local_indices = indices.view(2, width)[rank]
    # Reproduce the old gather-then-reduce ordering as the numerical oracle.
    old_query = funcs.gather_kpool_pcp_queries(global_query[local_indices], layout, group)
    old_weights = funcs.gather_kpool_pcp_queries(global_weights[local_indices], layout, group).bfloat16()
    expected_qbar = (old_query.float() * old_weights.float().unsqueeze(-1)).sum(1)
    local_qbars = [
        (global_query[rows].float() * global_weights[rows].bfloat16().float().unsqueeze(-1)).sum(1)
        for rows in indices.view(2, width)
    ]
    group.all_gather = Mock(return_value=torch.cat(local_qbars))
    scoring_metadata = SimpleNamespace(positions=torch.arange(context.global_batch.num_tokens))
    metadata = SimpleNamespace(pcp_dcp_query_layout=layout, write_metadata=scoring_metadata)
    global_topk = torch.arange(context.global_batch.num_tokens * 11, dtype=torch.int32).reshape(-1, 1, 11)
    kernel = Mock(return_value=global_topk)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.triton.glm5_next_dcp_indexer",
        SimpleNamespace(glm5_next_dcp_indexer=kernel),
    )
    select = load_method(
        "attention/indexer_kpool.py",
        "Glm5NextKPoolIndexerBackend",
        "_select_dcp_candidates",
        gather_kpool_pcp_queries=funcs.gather_kpool_pcp_queries,
        restore_kpool_pcp_queries=funcs.restore_kpool_pcp_queries,
        get_pcp_group=lambda: group,
        get_dcp_group=lambda: dcp_group,
    )
    shared = torch.full((width + 1, 15), 99, dtype=torch.int32)
    backend = SimpleNamespace(
        topk_tokens=8, index_kpool=4, _dcp_interleave=128, _max_model_len=8192, topk_indices_buffer=shared
    )
    result = select(backend, global_query[local_indices], global_weights[local_indices], None, metadata, None)
    group.all_gather.assert_called_once()
    sent = group.all_gather.call_args.args[0]
    assert sent.shape == (width, 128) and sent.dtype == torch.float32
    torch.testing.assert_close(sent, local_qbars[rank], rtol=0, atol=0)
    torch.testing.assert_close(kernel.call_args.args[0], expected_qbar, rtol=0, atol=0)
    assert kernel.call_args.args[2] is scoring_metadata
    assert kernel.call_args.args[3] is dcp_group
    assert kernel.call_args.kwargs["output_buffer"] is None
    assert result.data_ptr() == shared.data_ptr()
    expected = funcs.restore_kpool_pcp_queries(global_topk, layout, width, fill_value=-1)
    torch.testing.assert_close(result, expected)
    assert (shared[-1] == 99).all() and (shared[:, 11:] == 99).all()


@pytest.mark.parametrize("query_dtype", [torch.bfloat16, torch.float32])
def test_decode_weighted_query_needs_no_pcp_gather(monkeypatch, query_dtype):
    kernel = Mock(return_value=torch.zeros(3, 1, 11, dtype=torch.int32))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ops.triton.glm5_next_dcp_indexer",
        SimpleNamespace(glm5_next_dcp_indexer=kernel),
    )
    gather = Mock(side_effect=AssertionError("decode must not gather PCP query rows"))
    group = SimpleNamespace(world_size=4, rank_in_group=0)
    select = load_method(
        "attention/indexer_kpool.py",
        "Glm5NextKPoolIndexerBackend",
        "_select_dcp_candidates",
        gather_kpool_pcp_queries=gather,
        get_dcp_group=lambda: group,
    )
    generator = torch.Generator().manual_seed(21835)
    query = torch.randn(3, 64, 128, generator=generator).to(query_dtype)
    weights = torch.randn(3, 64, generator=generator)
    metadata = SimpleNamespace(pcp_dcp_query_layout=None)
    shared = torch.empty(3, 11, dtype=torch.int32)
    backend = SimpleNamespace(
        topk_tokens=8, index_kpool=4, _dcp_interleave=128, _max_model_len=8192, topk_indices_buffer=shared
    )
    assert select(backend, query, weights, None, metadata, None) is kernel.return_value
    expected = (query.float() * weights.bfloat16().float().unsqueeze(-1)).sum(1)
    torch.testing.assert_close(kernel.call_args.args[0], expected, rtol=0, atol=0)
    assert kernel.call_args.args[2] is metadata
    assert kernel.call_args.kwargs["output_buffer"] is shared
    gather.assert_not_called()


def test_joint_metadata_keeps_local_and_global_requests_separate():
    funcs = pcp_functions()
    context, rank_indices = pcp_batch(2, True, False)
    width = context.padded_gather_idx.numel() // 2
    context.global_slot_mappings = torch.arange(context.global_batch.num_tokens).unsqueeze(0)
    common = SimpleNamespace(num_input_tokens=width, num_actual_tokens=len(rank_indices[0]))
    global_common = SimpleNamespace(
        num_reqs=2,
        num_input_tokens=context.global_batch.num_tokens,
        num_actual_tokens=context.global_batch.num_tokens,
        query_start_loc=torch.tensor([0, 1, context.global_batch.num_tokens]),
        seq_lens=torch.tensor([257, 513]),
        positions=torch.arange(context.global_batch.num_tokens),
        block_table_tensor=torch.tensor([[5, 7], [11, 13]], dtype=torch.int32),
    )
    local_lens = torch.tensor([128, 131, 255])
    metadata = SimpleNamespace(
        dcp_context=SimpleNamespace(seq_lens=local_lens),
        pcp_dcp_query_layout=None,
        pcp_dcp_global_metadata=None,
        cum_query_lens=torch.tensor([1, 5, 9]),
    )
    attach = load_method(
        "attention/context_parallel/sfa_cp.py",
        "AscendSFAPCPDCPMetadataBuilder",
        "_attach_kpool_pcp_queries",
        get_sfa_pcp_global_metadata=Mock(return_value=global_common),
        DCPContext=SimpleNamespace,
        replace=lambda obj, **kwargs: SimpleNamespace(**(vars(obj) | kwargs)),
        build_kpool_pcp_query_layout=funcs.build_kpool_pcp_query_layout,
        get_pcp_group=lambda: SimpleNamespace(world_size=2, rank_in_group=0),
    )
    builder = SimpleNamespace(
        _get_dcp_local_block_table=lambda table, count: table,
        _get_dcp_local_seq_lens=lambda seq: torch.tensor([129, 257]),
    )
    attach(builder, metadata, common, context, 0)
    assert metadata.dcp_context.seq_lens is local_lens
    assert metadata.cum_query_lens.tolist() == [1, 5, 9]
    global_metadata = metadata.pcp_dcp_global_metadata
    assert global_metadata.cum_query_lens.tolist() == [1, 18]
    assert global_metadata.dcp_context.seq_lens.tolist() == [129, 257]
    assert global_metadata.dcp_context.block_table is global_common.block_table_tensor
    assert global_metadata.pcp_dcp_query_layout is None


@pytest.mark.parametrize("rank,short", [(0, False), (1, False), (1, True)])
@pytest.mark.parametrize("quantized", [False, True])
def test_joint_sfa_uses_global_requests_and_restores_local_outputs(rank, short, quantized):
    funcs = pcp_functions()
    context, rank_indices = pcp_batch(2, not short, short)
    width = context.padded_gather_idx.numel() // 2
    rows = context.global_batch.num_tokens
    common = SimpleNamespace(num_input_tokens=width, num_actual_tokens=len(rank_indices[rank]))
    group = SimpleNamespace(world_size=2, rank_in_group=rank)
    layout = funcs.build_kpool_pcp_query_layout(common, context, group)
    query = torch.arange(rows * 16).reshape(rows, 2, 8).float()
    topk = torch.arange(rows * 11, dtype=torch.int32).reshape(rows, 1, 11)
    group.all_gather = Mock(return_value=topk[context.padded_gather_idx])
    local_query = query[layout.local_indices]
    global_metadata = SimpleNamespace(
        cum_query_lens=torch.tensor([rows]),
        seq_lens=torch.tensor([513]),
        block_size=128,
        smla_metadata=None,
        dcp_context=SimpleNamespace(
            seq_lens=torch.tensor([257]), block_table=torch.tensor([[7, 11]]), gather_context=None
        ),
    )
    metadata = SimpleNamespace(
        dcp_context=object(), pcp_dcp_query_layout=layout, pcp_dcp_global_metadata=global_metadata
    )
    # Distinct output rows make a wrong local/global permutation observable.
    global_output = query + 1000
    operator = Mock(return_value=(global_output, torch.zeros(1, rows, 2), torch.ones(1, rows, 2)))
    execute = load_method(
        "attention/context_parallel/sfa_cp.py",
        "AscendSFADCPImpl",
        "_execute_sparse_flash_attention_process",
        gather_kpool_pcp_queries=funcs.gather_kpool_pcp_queries,
        restore_kpool_pcp_queries=funcs.restore_kpool_pcp_queries,
        get_pcp_group=lambda: group,
        _view_cache_as_operator_pages=lambda cache, block_size: cache,
        enable_sfa_dcp_force_tmajor_restore=lambda: False,
        DeviceOperator=SimpleNamespace(execute_sparse_flash_attention_process=operator),
    )
    impl = SimpleNamespace(
        dcp_group=group,
        qk_rope_head_dim=0,
        enable_sparse_sfa_turboquant=False,
        _sharded_indexer=True,
        _local_topk=False,
        _kpool_dcp=True,
        _has_prefill=lambda _: True,
        _start_kpool_pcp_query_gather=Mock(return_value=query),
        _finish_dcp_gather=lambda gathered, **kw: (gathered, None),
        _finish_dcp_packed_gather=lambda gathered, **kw: gathered,
        _remap_sparse_indices=Mock(side_effect=lambda ids: ids),
        _mask_nope_dcp_padding=lambda *args: args[:1] + args[2:],
        _merge_dcp_outputs=Mock(return_value=global_output),
    )
    result = execute(
        impl,
        local_query,
        None,
        (torch.zeros(1, dtype=torch.int8 if quantized else torch.bfloat16),),
        topk[layout.local_indices],
        metadata,
        torch.tensor([width]),
        torch.tensor([999]),
    )
    args, kwargs = operator.call_args
    torch.testing.assert_close(args[4], topk)
    assert args[5] is global_metadata
    assert args[6] is global_metadata.cum_query_lens
    assert args[7] is global_metadata.dcp_context.seq_lens
    assert kwargs["block_table"] is global_metadata.dcp_context.block_table
    torch.testing.assert_close(kwargs["packed_query"] if quantized else args[1], query)
    assert global_metadata.dcp_context.gather_context is None
    torch.testing.assert_close(result[: len(rank_indices[rank])], global_output[rank_indices[rank]])
    assert (result[len(rank_indices[rank]) :] == 0).all()


def test_joint_cache_write_slots_keep_dcp_ownership():
    context, rank_indices = pcp_batch(2, True, False)
    width = context.padded_gather_idx.numel() // 2
    # Every global token has exactly one DCP cache owner. Decode replicas and
    # PCP padding must not turn a non-owner's -1 into a writable cache address.
    positions = torch.arange(context.global_batch.num_tokens)
    mask = torch.zeros(2, width, dtype=torch.bool)
    for rank, indices in enumerate(rank_indices):
        mask[rank, : len(indices)] = True
    mask[1, 0] = False
    context.gathered_kv_write_mask = mask.flatten()
    funcs = load_functions("attention/context_parallel/common_cp.py", ["build_pcp_ordered_slot_mapping"])
    build = load_method(
        "attention/context_parallel/sfa_cp.py",
        "AscendSFAPCPDCPMetadataBuilder",
        "_build_pcp_ordered_indexer_slot_mapping",
        get_sfa_pcp_global_metadata=lambda *a: None,
        enable_kpool_dcp=lambda _: True,
        build_pcp_ordered_slot_mapping=funcs.build_pcp_ordered_slot_mapping,
    )
    for owner in range(2):
        slots = torch.where(positions // 4 % 2 == owner, positions // 8 * 4 + positions % 4 + 128, -1)
        context.global_slot_mappings = slots.unsqueeze(0)
        builder = SimpleNamespace(
            vllm_config=config(), pcp_indexer_slot_mapping_buf=torch.empty(2 * width, dtype=torch.long)
        )
        actual = build(builder, None, context, 0)
        expected = slots[context.padded_gather_idx].masked_fill(~mask.flatten(), -1)
        torch.testing.assert_close(actual, expected)
