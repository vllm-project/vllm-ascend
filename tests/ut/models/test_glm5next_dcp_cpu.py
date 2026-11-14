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
        ("parallel_config", "prefill_context_parallel_size", 2, "PCP=1"),
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
    with pytest.raises(NotImplementedError, match="PCP=1"):
        validate(cfg, check_cache_layout=False)


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
        torch.zeros(2, 2, 8),
        torch.zeros(1, 4, 1, 8),
        torch.zeros(2, 2),
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
        torch.zeros(3, 2, 8),
        torch.zeros(1, 4, 1, 8),
        torch.zeros(3, 2),
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
            torch.zeros(3, 2, 8),
            torch.zeros(1, 4, 1, 8),
            torch.zeros(3, 2),
            None,
            None,
            topk=8,
            pool_size=4,
            interleave=128,
            max_model_len=8192,
            output_buffer=torch.empty(1, 11, dtype=torch.int32).expand(3, 11),
        )
