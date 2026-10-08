# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for replicated-cache V4.1 PCP.

The AST loader executes production classes while replacing import-only vLLM
interfaces and hardware operations. Run with ``pytest --noconftest`` on a CPU
host. Collective and Ascend operator stubs validate their inputs; these tests
do not establish distributed/NPU correctness or model quality.
"""

import ast
import sys
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def _definitions(module, path, names):
    """Load unchanged production definitions without importing NPU libraries."""
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(tree, str(source), "exec"), module.__dict__)


class _Common(SimpleNamespace):
    def replace(self, **kwargs):
        return type(self)(**(vars(self) | kwargs))


class _MetadataBuilder:
    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device


@dataclass(frozen=True, kw_only=True)
class _Spec:
    block_size: int = 8
    num_kv_heads: int = 1
    head_size: int = 2
    dtype: torch.dtype = torch.float32
    tokens_per_state: int = 1
    scale_dim: int = 0
    model_version: str = "deepseek_v41"
    sliding_window: int = 128

    @property
    def storage_block_size(self):
        return self.block_size // self.tokens_per_state


@dataclass(frozen=True, kw_only=True)
class _MLASpec(_Spec):
    pass


@dataclass(frozen=True, kw_only=True)
class _IndexSpec(_MLASpec):
    scale_dim: int = 1


@dataclass(frozen=True, kw_only=True)
class _SWASpec(_Spec):
    pass


@dataclass(frozen=True, kw_only=True)
class _RingSpec(_Spec):
    pass


class _Stage(IntEnum):
    COMPRESSOR = 0
    INDEXER = 1
    ATTENTION = 2


@pytest.fixture
def production(monkeypatch):
    module = ModuleType("v41_pcp_cpu_contract")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        torch=torch,
        F=F,
        dsa_v1=SimpleNamespace(AscendDSAMetadataBuilder=_MetadataBuilder),
        np=np,
        dataclass=dataclass,
        replace=replace,
        AttentionMetadata=object,
        AttentionMetadataBuilder=_MetadataBuilder,
        AscendCommonAttentionMetadata=_Common,
        AscendMLAAttentionSpec=_MLASpec,
        MLAAttentionSpec=_MLASpec,
        UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}),
        vllm_version_is=lambda _: False,
        AscendSlidingWindowMLASpec=_SWASpec,
        CircularBufferSpec=_RingSpec,
        DeviceMetadataStage=_Stage,
        V41_METADATA_BUFFER_SIZE=1024,
        STATE_RING_ROWS=32,
        PCPManager=object,
        get_pcp_group=lambda: SimpleNamespace(rank_in_group=0),
        wait_for_device_metadata=lambda *args: None,
    )
    _definitions(
        module,
        "vllm_ascend/core/kv_cache_interface.py",
        ("get_kv_cache_compression_ratio", "get_storage_block_size"),
    )
    _definitions(module, "vllm_ascend/ops/rope_dsv4.py", ("RopeGlobalState", "RopeDataProxy"))
    module._ROPE_STATE = module.RopeGlobalState()
    module._ROPE_STATE.layer_info = {"rope": ("first", ["default"]), "other": ("second", ["default"])}
    module.rope_calls = []

    def rope(positions, **kwargs):
        module.rope_calls.append(positions.clone())
        positions = positions.float().reshape(-1, 1, 1, 1)
        values = {
            "first": {"default": (positions + 1, positions + 2)},
            "second": {"default": (positions + 11, positions + 12)},
        }
        return module.RopeDataProxy(values), module.RopeDataProxy(values, is_cos=False)

    module.get_cos_and_sin_dsa = rope
    _definitions(
        module,
        "vllm_ascend/attention/dsa_v41.py",
        (
            "_config_value",
            "AscendDSAV41Metadata",
            "DeepseekV41CompressorMetadata",
            "DeepseekV41IndexerMetadata",
            "DeepseekV41LayerMetadata",
            "compressed_slot_mapping",
            "_request_counts",
            "scatter_cache_sk",
            "pad_sparse_indices",
            "AscendDSAV41Impl",
            "AscendDSAV41MetadataBuilder",
        ),
    )
    _definitions(
        module,
        "vllm_ascend/attention/context_parallel/dsa_cp.py",
        ("AscendDSAPCPMetadataBuilder",),
    )
    _definitions(
        module,
        "vllm_ascend/attention/context_parallel/dsa_v41_cp.py",
        (
            "gather_and_restore_hidden_states",
            "_ReplicatedCacheMetadataBuilder",
            "AscendDSAV41PCPMetadataBuilder",
            "AscendDSAV41PCPImpl",
        ),
    )
    _definitions(
        module,
        "vllm_ascend/worker/v2/pcp_manager.py",
        (
            "AscendPCPAttentionContext",
            "AscendPCPManager",
        ),
    )
    return module


def _config(*, pcp=2, tp=1, heads=1):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=8),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, tensor_parallel_size=tp),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                sliding_window=128,
                num_attention_heads=heads,
                head_dim=2,
                index_topk=512,
                index_n_heads=32,
                index_head_dim=128,
            )
        ),
    )


def _context(production, lengths, offsets, rank_indices, rank_segments, rank, *, is_prefilling=None):
    """Explicit scheduler fixture: local segments carry their causal end."""
    boundaries = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    positions = torch.cat([torch.arange(offset, offset + length) for length, offset in zip(lengths, offsets)])
    seq_lens = torch.tensor(np.asarray(lengths) + offsets, dtype=torch.int32)
    num_blocks = max(2, (int(seq_lens.max()) + 7) // 8)
    block_stride = max(4, num_blocks)
    block_tables = torch.tensor(
        [list(range(2 + index * block_stride, 2 + index * block_stride + num_blocks)) for index in range(len(lengths))],
        dtype=torch.int32,
    )
    slots = torch.cat(
        [
            block_tables[request, torch.arange(offset, offset + length) // 8].long() * 8
            + torch.arange(offset, offset + length) % 8
            for request, (length, offset) in enumerate(zip(lengths, offsets))
        ]
    )
    gather = torch.tensor(rank_indices, dtype=torch.long)
    flat_gather = gather.flatten()
    gathered_slots = slots[flat_gather]
    if is_prefilling is None:
        is_prefilling = [length > 1 for length in lengths]
    write_mask = torch.zeros_like(gather, dtype=torch.bool)
    for owner, segments in enumerate(rank_segments):
        start = 0
        for request, size, _ in segments:
            if owner == 0 or is_prefilling[request]:
                write_mask[owner, start : start + size] = True
            start += size
    gathered_slots[~write_mask.flatten()] = -1
    # Decode tokens can occur on every rank, but their global cache write occurs once.
    restore = torch.tensor([flat_gather.tolist().index(index) for index in range(sum(lengths))])
    global_batch = SimpleNamespace(
        num_reqs=len(lengths),
        num_reqs_after_padding=len(lengths),
        num_tokens=sum(lengths),
        num_tokens_after_padding=sum(lengths),
        query_start_loc=torch.from_numpy(boundaries),
        query_start_loc_np=boundaries,
        seq_lens=seq_lens,
        seq_lens_np=seq_lens.numpy(),
        seq_lens_cpu_upper_bound=seq_lens,
        num_computed_tokens_np=np.asarray(offsets, dtype=np.int32),
        num_scheduled_tokens=np.asarray(lengths),
        is_prefilling_np=np.asarray(is_prefilling),
        dcp_local_seq_lens=None,
        positions=positions,
        attn_state=None,
        is_dummy=False,
    )
    context = production.AscendPCPAttentionContext(
        global_batch,
        (block_tables,),
        slots[None],
        restore,
        flat_gather,
        write_mask.flatten(),
    )
    segments = rank_segments[rank]
    local_boundaries = torch.tensor([0, *np.cumsum([size for _, size, _ in segments])], dtype=torch.int32)
    local_lengths = torch.tensor([end for _, _, end in segments], dtype=torch.int32)
    local_positions = positions[gather[rank]].clone()
    local_positions[int(local_boundaries[-1]) :].zero_()
    local = _Common(
        num_reqs=len(segments),
        num_actual_tokens=sum(size for _, size, _ in segments),
        num_input_tokens=gather.shape[1],
        query_start_loc=local_boundaries,
        query_start_loc_cpu=local_boundaries,
        seq_lens=local_lengths,
        seq_lens_cpu=local_lengths,
        max_query_len=max((size for _, size, _ in segments), default=0),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=block_tables[[request for request, _, _ in segments]],
        slot_mapping=gathered_slots,
        positions=local_positions,
        causal=True,
        is_prefilling=torch.tensor([is_prefilling[request] for request, _, _ in segments]),
    )
    return context, local


def _two_requests(production, rank):
    # vLLM ced6857 orders prefills by request then segment start, heads first.
    return _context(
        production,
        [8, 4],
        [0, 0],
        [[0, 1, 6, 7, 8, 11], [2, 3, 4, 5, 9, 10]],
        [[(0, 2, 2), (0, 2, 8), (1, 1, 1), (1, 1, 4)], [(0, 2, 4), (0, 2, 6), (1, 1, 2), (1, 1, 3)]],
        rank,
    )


def _builder(production, spec, rank=0, *, config=None):
    production.get_pcp_group = lambda: SimpleNamespace(rank_in_group=rank)
    return production.AscendDSAV41PCPMetadataBuilder(
        spec, ["layer.compressor.state_cache"], _config() if config is None else config, torch.device("cpu")
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("kind,ratio", [("swa", 1), ("long", 1), ("index", 1), ("long", 2), ("index", 2)])
def test_global_cache_and_local_causal_metadata_are_separate(production, rank, kind, ratio):
    context, local = _two_requests(production, rank)
    spec_type = {
        "swa": _SWASpec,
        "long": _MLASpec,
        "index": _IndexSpec,
    }[kind]
    builder = _builder(production, spec_type(tokens_per_state=ratio), rank)
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0, num_actual_reqs=4)
    global_metadata = metadata.global_metadata
    assert global_metadata.num_reqs == global_metadata.num_actual_reqs == 2
    assert global_metadata.num_actual_tokens == 12
    assert global_metadata.query_start_loc.tolist() == [0, 8, 12]
    assert global_metadata.seq_lens.tolist() == [8, 4]
    assert metadata.num_actual_tokens == 6 and metadata.num_reqs == 4
    assert metadata.seq_lens.tolist() == local.seq_lens.tolist()
    assert torch.equal(metadata.query_start_loc, local.query_start_loc)
    assert metadata.seq_lens.data_ptr() != global_metadata.seq_lens.data_ptr()
    assert metadata.slot_mapping.shape == (6, 2)
    assert global_metadata.slot_mapping.shape == (12, 2)
    expected_global = context.global_slot_mappings[0]
    expected_local = expected_global[context.padded_gather_idx.view(2, -1)[rank]]
    for slots, original in ((global_metadata.slot_mapping, expected_global), (metadata.slot_mapping, expected_local)):
        valid = original >= 0 if ratio == 1 else (original + 1).remainder(2) == 0
        expected = torch.stack((original // 8, original.remainder(8) // ratio), dim=-1).int()
        expected[~valid] = -1
        assert torch.equal(slots, expected)
    if ratio == 2:
        assert torch.equal(metadata.cache_seq_lens, local.seq_lens // 2)
        assert torch.equal(metadata.cmp_residual, local.seq_lens % 2)
    if kind == "swa":
        assert len(production.rope_calls) == 1
        for layer, delta in (("rope", 1), ("other", 11)):
            assert torch.equal(metadata.cos[layer].flatten(), local.positions.float() + delta)
            assert torch.equal(global_metadata.cos[layer].flatten(), context.global_batch.positions.float() + delta)
            assert metadata.cos[layer].data_ptr() != global_metadata.cos[layer].data_ptr()


def _four_rank_two_requests(production, rank):
    # Explicit ced6857 PCP4 contract: eight chunks per request, with each
    # request's head before its tail on every rank.
    return _context(
        production,
        [16, 8],
        [0, 0],
        [
            [0, 1, 14, 15, 16, 23],
            [2, 3, 12, 13, 17, 22],
            [4, 5, 10, 11, 18, 21],
            [6, 7, 8, 9, 19, 20],
        ],
        [
            [(0, 2, 2), (0, 2, 16), (1, 1, 1), (1, 1, 8)],
            [(0, 2, 4), (0, 2, 14), (1, 1, 2), (1, 1, 7)],
            [(0, 2, 6), (0, 2, 12), (1, 1, 3), (1, 1, 6)],
            [(0, 2, 8), (0, 2, 10), (1, 1, 4), (1, 1, 5)],
        ],
        rank,
    )


@pytest.mark.parametrize("rank", range(4))
def test_pcp4_tp4_metadata_keeps_global_writes_and_local_queries(production, monkeypatch, rank):
    context, local = _four_rank_two_requests(production, rank)
    config = _config(pcp=4, tp=4, heads=8)
    native_calls = []

    def native_metadata(*args, **kwargs):
        # Replace only the NPU metadata constructor; build() must derive the
        # TP-local head count and PCP-local request/causal coordinates itself.
        native_calls.append((args, kwargs))
        return torch.zeros(production.V41_METADATA_BUFFER_SIZE, dtype=torch.int32)

    for op in ("npu_sparse_flash_mla_metadata", "npu_quant_lightning_indexer_v2_metadata"):
        monkeypatch.setattr(torch.ops._C_ascend, op, native_metadata, raising=False)

    for spec in (_SWASpec(), _MLASpec(), _MLASpec(tokens_per_state=2), _IndexSpec(), _IndexSpec(tokens_per_state=2)):
        builder = _builder(production, spec, rank, config=config)
        builder._supports_device_ops = True
        native_calls.clear()
        metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0, num_actual_reqs=4)
        global_metadata = metadata.global_metadata
        assert global_metadata.num_reqs == global_metadata.num_actual_reqs == 2
        assert global_metadata.num_actual_tokens == 24
        assert global_metadata.query_start_loc.tolist() == [0, 16, 24]
        assert global_metadata.seq_lens.tolist() == [16, 8]
        assert metadata.num_actual_tokens == 6 and metadata.num_reqs == 4
        assert torch.equal(metadata.query_start_loc, local.query_start_loc)
        assert torch.equal(metadata.seq_lens, local.seq_lens)
        assert metadata.seq_lens.data_ptr() != global_metadata.seq_lens.data_ptr()
        # PCP4 still gives at most TWO segments per request per rank, not four.
        assert builder._seq_lens.numel() == 2 * config.scheduler_config.max_num_seqs
        assert builder._global_builder._seq_lens.numel() == config.scheduler_config.max_num_seqs

        ratio = spec.tokens_per_state
        expected_slots = torch.full((24, 2), -1, dtype=torch.int32)
        expected_slots[ratio - 1 :: ratio] = torch.tensor(
            [(block, offset) for block in (2, 3, 6) for offset in range(8 // ratio)], dtype=torch.int32
        )
        assert torch.equal(global_metadata.slot_mapping, expected_slots)
        local_indices = context.padded_gather_idx.view(4, -1)[rank]
        assert torch.equal(metadata.slot_mapping, expected_slots[local_indices])
        assert torch.equal(metadata.cache_seq_lens, local.seq_lens // ratio)
        if ratio == 2:
            assert torch.equal(metadata.cmp_residual, local.seq_lens % 2)
        if isinstance(spec, _SWASpec):
            for layer, delta in (("rope", 1), ("other", 11)):
                assert torch.equal(metadata.cos[layer].flatten(), local.positions.float() + delta)
                assert torch.equal(global_metadata.cos[layer].flatten(), context.global_batch.positions.float() + delta)
                assert metadata.cos[layer].data_ptr() != global_metadata.cos[layer].data_ptr()

        assert len(native_calls) == 1  # The global cache view must not build query metadata.
        args, kwargs = native_calls[0]
        assert args[0] == (32 if isinstance(spec, _IndexSpec) else 2)
        assert kwargs["batch_size"] == 4
        assert torch.equal(kwargs["cu_seqlens_q"], local.query_start_loc)
        if isinstance(spec, _IndexSpec):
            assert torch.equal(kwargs["seqused_k"], local.seq_lens // ratio)
        else:
            assert torch.equal(kwargs["seqused_ori_kv"], local.seq_lens)


def test_pcp_requires_canonical_context_and_token_map(production):
    context, local = _two_requests(production, 0)
    builder = _builder(production, _SWASpec())
    with pytest.raises(ValueError, match="canonical batch context"):
        builder.build(0, local)
    with pytest.raises(ValueError, match="local-to-global token map"):
        builder.build(0, local, pcp_context=replace(context, padded_gather_idx=None), pcp_cache_group_idx=0)


def test_runner_dispatch_capacity_and_source_rope_initialization(production):
    builder = _builder(production, _RingSpec(block_size=32))
    assert builder.consumes_pcp_context
    assert builder._seq_lens.numel() == 2 * _config().scheduler_config.max_num_seqs
    assert builder._global_builder._seq_lens.numel() == _config().scheduler_config.max_num_seqs
    tables = (torch.ones(16, 1, 1, 2), torch.zeros(16, 1, 1, 2))
    calls = []

    def source_rope(layer_name):
        calls.append(layer_name)
        return tables

    production.get_full_cos_and_sin_dsa_for_layer = source_rope
    builder.prepare_source_rope()
    assert calls == ["layer.attn"]
    assert builder._global_builder._c2_full_source_rope is tables
    assert builder._c2_full_source_rope is None
    assert not builder._device_metadata_enabled
    assert not builder._global_builder._device_metadata_enabled


def test_compressed_spec_normalizes_to_logical_block_geometry(production):
    # MRV2 can hand a builder the kernel block view. The authored base builder
    # converts it to logical scheduler geometry before deriving C2 slots.
    spec = _MLASpec(block_size=4, tokens_per_state=2)
    builder = _builder(production, spec)
    assert builder.kv_cache_spec.block_size == builder._global_builder.kv_cache_spec.block_size == 8
    context, local = _two_requests(production, 0)
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0)
    assert metadata.storage_block_size == metadata.global_metadata.storage_block_size == 4
    assert metadata.compress_ratio == metadata.global_metadata.compress_ratio == 2


def test_odd_prefill_discards_rank_padding_before_global_cache_writes(production):
    # Seven tokens leave rank 0 with one padding row after its head/tail rows.
    context, local = _context(
        production,
        [7],
        [0],
        [[0, 1, 6, 0], [2, 3, 4, 5]],
        [[(0, 2, 2), (0, 1, 7)], [(0, 2, 4), (0, 2, 6)]],
        0,
    )
    local = local.replace(num_actual_tokens=3)
    local.slot_mapping[3] = -1
    builder = _builder(production, _MLASpec(tokens_per_state=2))
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0)
    assert metadata.num_actual_tokens == 3 and metadata.num_input_tokens == 4
    assert metadata.slot_mapping[-1].tolist() == [-1, -1]
    assert metadata.hidden_restore_idx.numel() == 7
    global_hidden = torch.arange(7).float()[:, None]
    padded_hidden = global_hidden[context.padded_gather_idx]
    padded_hidden[3] = 999
    actual = production.gather_and_restore_hidden_states(
        padded_hidden[:4],
        metadata.hidden_restore_idx,
        SimpleNamespace(all_gather=lambda value, dim: padded_hidden),
    )
    assert torch.equal(actual, global_hidden)
    assert int((metadata.global_metadata.slot_mapping[:, 0] >= 0).sum()) == 3


@pytest.mark.parametrize("rank", [0, 1])
def test_c2_ring_uses_global_pairs_and_retains_odd_chunk_offset(production, rank):
    # At position 3, the ring must use the residual token at position 2 from
    # the preceding chunk. A second request must have independent ring state.
    context, local = _context(
        production,
        [5, 3],
        [3, 1],
        [[0, 1, 5, 0, 0], [2, 3, 4, 6, 7]],
        [[(0, 2, 5), (1, 1, 2)], [(0, 2, 7), (0, 1, 8), (1, 1, 3), (1, 1, 4)]],
        rank,
    )
    builder = _builder(production, _RingSpec(block_size=32), rank)
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0, num_actual_reqs=local.num_reqs)
    assert metadata.c2_ring_metadata is None
    state = metadata.global_metadata
    assert state.c2_ring_metadata.tolist() == [[3, 1], [5, 3], [0, 5], [0, 5], [2, 6]]
    assert state.c2_complete_mask.tolist() == [True, False, True, False, True, True, False, True]
    assert state.c2_source_positions.tolist() == [2, 0, 4, 0, 6, 0, 0, 2]
    assert builder._c2_ring_metadata.numel() == 0
    hidden = torch.arange(8).float()[:, None]
    gathered = hidden[context.padded_gather_idx]
    collective = SimpleNamespace(all_gather=lambda value, dim: gathered)
    restored = production.gather_and_restore_hidden_states(
        gathered.view(2, 5, 1)[rank], context.hidden_restore_idx, collective
    )
    assert torch.equal(restored, hidden)
    # The pair at positions 4 and 5 is split between ranks 0 and 1. Restore
    # its two rows before the real kernel consumes completion position 5.
    assert 1 in context.padded_gather_idx.view(2, -1)[0]
    assert 2 in context.padded_gather_idx.view(2, -1)[1]
    assert restored[1:3, 0].tolist() == [1, 2]


@pytest.mark.parametrize("rank", range(4))
def test_pcp4_c2_restores_cross_rank_pairs_and_next_chunk_residual(production, rank):
    builder = _builder(production, _RingSpec(block_size=32), rank, config=_config(pcp=4, tp=4, heads=8))
    # A 23-token chunk has eight chunks of up to three tokens. Pairs (2, 3)
    # and (20, 21) straddle ranks 0/1; position 22 must remain incomplete.
    context, local = _context(
        production,
        [23],
        [0],
        [
            [0, 1, 2, 21, 22, 0],
            [3, 4, 5, 18, 19, 20],
            [6, 7, 8, 15, 16, 17],
            [9, 10, 11, 12, 13, 14],
        ],
        [
            [(0, 3, 3), (0, 2, 23)],
            [(0, 3, 6), (0, 3, 21)],
            [(0, 3, 9), (0, 3, 18)],
            [(0, 3, 12), (0, 3, 15)],
        ],
        rank,
    )
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0)
    state = metadata.global_metadata
    assert metadata.c2_ring_metadata is None
    assert state.c2_ring_metadata.tolist() == [[0], [23], [0], [0], [2]]
    assert state.c2_complete_mask.tolist() == [False, True] * 11 + [False]
    assert state.c2_source_positions[state.c2_complete_mask].tolist() == list(range(0, 22, 2))
    assert metadata.num_actual_tokens == (5 if rank == 0 else 6)
    global_hidden = torch.arange(23).float()[:, None]
    gathered = global_hidden[context.padded_gather_idx]
    gathered[local.slot_mapping < 0] = -999
    restored = production.gather_and_restore_hidden_states(
        gathered.view(4, 6, 1)[rank],
        metadata.hidden_restore_idx,
        SimpleNamespace(all_gather=lambda value, dim: gathered),
    )
    assert torch.equal(restored, global_hidden)
    assert restored[2:4, 0].tolist() == [2, 3]
    assert restored[20:23, 0].tolist() == [20, 21, 22]

    # The next chunk begins at 23: its first completed pair must reference
    # the earlier token 22. This checks real compressor controls, not a CPU
    # reimplementation of the NPU pooling kernel or its persistent state.
    context, local = _context(
        production,
        [3],
        [23],
        [[0], [1], [2], [0]],
        [[(0, 1, 24)], [(0, 1, 25)], [(0, 1, 26)], [(0, 0, 0)]],
        rank,
    )
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0)
    state = metadata.global_metadata
    assert state.c2_ring_metadata.tolist() == [[23], [3], [0], [0], [2]]
    assert state.c2_complete_mask.tolist() == [True, False, True]
    assert state.c2_source_positions.tolist() == [22, 0, 24]
    assert metadata.num_actual_tokens == (0 if rank == 3 else 1)
    assert state.num_actual_tokens == 3 and state.num_actual_reqs == 1


def test_empty_local_batch_does_not_zero_global_ring_request_count(production):
    context, local = _two_requests(production, 1)
    local = local.replace(
        num_reqs=0,
        num_actual_tokens=0,
        query_start_loc=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32),
        seq_lens_cpu=torch.empty(0, dtype=torch.int32),
        block_table_tensor=torch.empty((0, 2), dtype=torch.int32),
        max_query_len=0,
        is_prefilling=torch.empty(0, dtype=torch.bool),
    )
    builder = _builder(production, _RingSpec(block_size=32), 1)
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0, num_actual_reqs=0)
    assert metadata.num_actual_reqs == metadata.num_actual_tokens == 0
    assert metadata.global_metadata.num_actual_reqs == 2
    assert metadata.global_metadata.c2_ring_metadata[1].tolist() == [8, 4]


@pytest.mark.parametrize("kind", ["swa", "long", "index"])
def test_empty_rank_does_not_submit_native_query_metadata(production, monkeypatch, kind):
    context, local = _two_requests(production, 1)
    local = local.replace(
        num_reqs=0,
        num_actual_tokens=0,
        query_start_loc=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32),
        seq_lens_cpu=torch.empty(0, dtype=torch.int32),
        block_table_tensor=torch.empty((0, 2), dtype=torch.int32),
        max_query_len=0,
        is_prefilling=torch.empty(0, dtype=torch.bool),
    )
    spec = {"swa": _SWASpec(), "long": _MLASpec(tokens_per_state=2), "index": _IndexSpec(tokens_per_state=2)}[kind]
    builder = _builder(production, spec, 1)
    # Exercise the production device-metadata path using CPU buffers. Empty
    # local ranks must not call either native metadata constructor.
    builder._supports_device_ops = True
    builder._smla_metadata.fill_(7)
    builder._qli_metadata.fill_(7)

    def native_metadata(*args, **kwargs):
        pytest.fail("empty PCP rank submitted native query metadata")

    for op in ("npu_sparse_flash_mla_metadata", "npu_quant_lightning_indexer_v2_metadata"):
        monkeypatch.setattr(torch.ops._C_ascend, op, native_metadata, raising=False)
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0, num_actual_reqs=0)
    query_metadata = metadata.qli_metadata if kind == "index" else metadata.smla_metadata
    assert query_metadata is not None and torch.count_nonzero(query_metadata) == 0
    assert metadata.global_metadata.num_actual_tokens == 12


@pytest.mark.parametrize("decode_only", [False, True])
def test_replicated_decode_is_removed_before_global_cache_write(production, decode_only):
    if decode_only:
        context, local = _context(production, [1, 1], [5, 7], [[0, 1], [0, 1]], [[(0, 1, 6), (1, 1, 8)]] * 2, 0)
    else:
        context, local = _context(
            production,
            [1, 8],
            [5, 0],
            [[0, 1, 2, 7, 8], [0, 3, 4, 5, 6]],
            [[(0, 1, 6), (1, 2, 2), (1, 2, 8)], [(0, 1, 6), (1, 2, 4), (1, 2, 6)]],
            0,
        )
    metadata = _builder(production, _SWASpec()).build(
        0,
        local,
        pcp_context=context,
        pcp_cache_group_idx=0,
    )
    hidden = torch.arange(context.global_batch.num_tokens).float()[:, None]
    gathered = hidden[context.padded_gather_idx]
    calls = []

    def all_gather(value, dim):
        calls.append(value.clone())
        return gathered

    production.get_pcp_group = lambda: SimpleNamespace(all_gather=all_gather)
    impl = production.AscendDSAV41PCPImpl("layer", _role(), SimpleNamespace(), None, None)
    writes = []
    impl._update_caches = lambda attn, value, meta: writes.append((value.clone(), meta))
    by_prefix = {impl.swa_prefix: metadata}
    bundle = impl._get_layer_metadata(by_prefix)
    local_hidden = hidden[context.padded_gather_idx.view(2, -1)[0]]
    impl._prepare_inputs_and_caches(None, local_hidden, bundle, by_prefix)
    assert len(calls) == len(writes) == 1
    assert torch.equal(writes[0][0], hidden)
    assert writes[0][1].swa is metadata.global_metadata


def test_pcp4_decode_restoration_writes_each_request_once_on_every_rank(production):
    config = _config(pcp=4, tp=4, heads=8)
    canonical = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    # Deliberately distinguish redundant copies: restoration must select the
    # canonical rank-0 rows instead of averaging or inserting all eight rows.
    gathered = torch.cat([canonical + 100 * rank for rank in range(4)])
    for rank in range(4):
        context, local = _context(
            production,
            [1, 1],
            [5, 7],
            [[0, 1]] * 4,
            [[(0, 1, 6), (1, 1, 8)]] * 4,
            rank,
        )
        metadata = _builder(production, _SWASpec(), rank, config=config).build(
            0, local, pcp_context=context, pcp_cache_group_idx=0
        )
        assert metadata.num_decode_tokens == 2 and metadata.global_metadata.num_actual_tokens == 2
        # Upstream gives each decode rank its complete local SWA mapping.
        # V4.1 consumes only global metadata for writes, verified below.
        assert torch.equal(metadata.slot_mapping, metadata.global_metadata.slot_mapping)
        assert torch.all(metadata.global_metadata.slot_mapping >= 0)
        calls, writes = [], []

        def all_gather(value, dim, calls=calls):
            calls.append(value.clone())
            assert dim == 0
            return gathered

        production.get_pcp_group = lambda all_gather=all_gather: SimpleNamespace(all_gather=all_gather)
        impl = production.AscendDSAV41PCPImpl("layer", _role(), SimpleNamespace(), None, None)
        impl._update_caches = lambda attn, value, meta, writes=writes: writes.append((value.clone(), meta))
        by_prefix = {impl.swa_prefix: metadata}
        local_hidden = gathered.view(4, 2, 2)[rank]
        impl._prepare_inputs_and_caches(None, local_hidden, impl._get_layer_metadata(by_prefix), by_prefix)
        assert len(calls) == len(writes) == 1
        assert torch.equal(calls[0], local_hidden)
        assert torch.equal(writes[0][0], canonical)
        assert writes[0][1].swa is metadata.global_metadata
        assert writes[0][1].positions.tolist() == [5, 7]


def _role(**overrides):
    return SimpleNamespace(
        **(
            dict(
                is_kv_source=False,
                has_long_context=False,
                compress_ratio=0,
                is_index_source=False,
                is_candidate_source=False,
                uses_candidate_filter=False,
            )
            | overrides
        )
    )


def test_empty_query_rank_still_collects_and_updates_cache(production):
    context, local = _two_requests(production, 0)
    metadata = _builder(production, _SWASpec()).build(
        0,
        local,
        pcp_context=context,
        pcp_cache_group_idx=0,
    )
    metadata.num_actual_tokens = 0
    impl = production.AscendDSAV41PCPImpl("layer", _role(), SimpleNamespace(), None, None)
    events = []
    global_hidden = torch.arange(12).float().reshape(-1, 1)

    def gather(value, dim):
        events.append("gather")
        assert value.shape == (6, 1)
        return global_hidden[context.padded_gather_idx]

    production.get_pcp_group = lambda: SimpleNamespace(all_gather=gather)
    production.get_forward_context = lambda: SimpleNamespace(attn_metadata={impl.swa_prefix: metadata})
    impl._update_caches = lambda attn, hidden, meta: events.append(("write", hidden.clone()))
    impl._prepare_queries = lambda *args: pytest.fail("empty rank must not project queries")

    def project_output(attn, attention_output, hidden, metadata, *, projected):
        events.append("output")
        assert attention_output.shape[0] == 0
        return projected.zero_()

    impl._project_output = project_output
    output = impl.forward(SimpleNamespace(n_heads=1, n_local_heads=1, head_dim=1), None, torch.zeros(6, 1))
    assert events[0] == "gather" and events[-1] == "output"
    assert torch.equal(events[1][1], global_hidden)
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_pcp4_one_token_prefill_empty_ranks_still_prepare_cache(production, monkeypatch, rank):
    # A one-token PREFILL belongs only to rank 0. The three empty ranks have
    # a padded token and the upstream manager's zero-length placeholder row.
    context, local = _context(
        production,
        [1],
        [0],
        [[0]] * 4,
        [[(0, 1, 1)], [(0, 0, 0)], [(0, 0, 0)], [(0, 0, 0)]],
        rank,
        is_prefilling=[True],
    )
    builder = _builder(production, _SWASpec(), rank, config=_config(pcp=4, tp=4, heads=8))
    builder._supports_device_ops = True

    def native_metadata(*args, **kwargs):
        pytest.fail("empty PCP4 rank submitted native query metadata")

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla_metadata", native_metadata, raising=False)
    metadata = builder.build(0, local, pcp_context=context, pcp_cache_group_idx=0)
    assert metadata.num_actual_tokens == 0 and metadata.num_input_tokens == 1
    assert metadata.num_reqs == 1 and metadata.query_start_loc.tolist() == [0, 0]
    assert torch.all(metadata.slot_mapping == -1)
    assert torch.count_nonzero(metadata.smla_metadata) == 0
    assert metadata.global_metadata.is_prefilling.tolist() == [True]
    assert metadata.global_metadata.num_actual_tokens == 1
    impl = production.AscendDSAV41PCPImpl("layer", _role(), SimpleNamespace(), None, None)
    canonical = torch.tensor([[11.0, 12.0]])
    gathered = torch.cat((canonical, torch.full((3, 2), -999.0)))
    events = []

    def all_gather(value, dim):
        events.append("gather")
        assert dim == 0 and value.shape == (1, 2)
        return gathered

    production.get_pcp_group = lambda: SimpleNamespace(all_gather=all_gather)
    production.get_forward_context = lambda: SimpleNamespace(attn_metadata={impl.swa_prefix: metadata})
    impl._update_caches = lambda attn, hidden, meta: events.append(("write", hidden.clone()))
    impl._prepare_queries = lambda *args: pytest.fail("empty PCP4 rank projected a query")

    def project_output(attn, attention_output, hidden, metadata, *, projected):
        events.append("output")
        assert attention_output.shape == (0, 2, 2)
        return projected.zero_()

    impl._project_output = project_output
    output = impl.forward(SimpleNamespace(n_local_heads=2, head_dim=2), None, gathered[rank : rank + 1])
    assert len(events) == 3 and events[0] == "gather" and events[-1] == "output"
    assert torch.equal(events[1][1], canonical)
    assert torch.count_nonzero(output) == 0


def test_query_projection_does_not_write_caches_again(production):
    impl = production.AscendDSAV41PCPImpl(
        "layer", _role(is_kv_source=True, compress_ratio=2), SimpleNamespace(), "long", "index"
    )
    calls = []
    impl._project_q = lambda attn, hidden, cos, sin: calls.append(hidden.clone()) or (hidden, hidden)
    impl._update_caches = lambda *args: pytest.fail("query projection rewrote cache")
    impl._write_compressed_source = lambda *args: pytest.fail("query projection consumed C2 residual twice")
    impl.preprocess = lambda *args: pytest.fail("ordinary preprocessing writes local cache")
    hidden = torch.arange(8).reshape(4, 2)
    q, qr = impl._prepare_queries(
        None, hidden, None, None, None, SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=3))
    )
    assert len(calls) == 1
    assert torch.equal(q, hidden[:3]) and torch.equal(qr, hidden[:3])


def test_query_projection_uses_local_heads_and_model_linear_modules(production, monkeypatch):
    hidden = torch.arange(12).reshape(3, 4).float()
    calls = []

    def wq_a(value):
        calls.append("wq_a")
        assert torch.equal(value, hidden)
        return value + 1

    def q_norm(value):
        calls.append("q_norm")
        return value * 2

    def wq_b(value):
        calls.append("wq_b")
        return torch.cat((value, value + 1), dim=-1)

    cos, sin = torch.ones(3, 1, 1, 2), torch.zeros(3, 1, 1, 2)

    def apply_rope(q, selected_cos, selected_sin, **kwargs):
        calls.append("rope")
        assert q.shape == (3, 1, 2, 4)
        assert selected_cos is cos and selected_sin is sin
        assert kwargs == {"rotary_mode": "interleave", "partial_slice": [2, 4]}

    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", apply_rope, raising=False)
    attn = SimpleNamespace(wq_a=wq_a, q_norm=q_norm, wq_b=wq_b, n_local_heads=2, head_dim=4, nope_head_dim=2)
    q, qr = production.AscendDSAV41PCPImpl._project_q(attn, hidden, cos, sin)
    assert calls == ["wq_a", "q_norm", "wq_b", "rope"]
    assert q.shape == (3, 2, 4) and q.dtype == hidden.dtype
    assert torch.equal(qr, (hidden + 1) * 2)


@pytest.mark.parametrize("ratio", [1, 2])
def test_cache_write_pipeline_projects_global_tokens_once(production, monkeypatch, ratio):
    context, local = _two_requests(production, 0)
    impl = production.AscendDSAV41PCPImpl(
        "layer", _role(is_kv_source=True, compress_ratio=ratio), SimpleNamespace(), "long", "index"
    )
    specs = {
        impl.swa_prefix: _SWASpec(),
        "long": _MLASpec(tokens_per_state=ratio),
        "index": _IndexSpec(tokens_per_state=ratio),
    }
    if ratio == 2:
        specs[impl.compressor_state_prefix] = _RingSpec(block_size=32)
    by_prefix = {
        prefix: _builder(production, spec).build(0, local, pcp_context=context, pcp_cache_group_idx=0)
        for prefix, spec in specs.items()
    }
    bundle = impl._get_layer_metadata(by_prefix)
    global_hidden = torch.arange(24).float().reshape(12, 2)
    gathered = global_hidden[context.padded_gather_idx]
    events = []

    def gather(value, dim):
        events.append("gather")
        return gathered

    production.get_pcp_group = lambda: SimpleNamespace(all_gather=gather)

    def project_kv(attn, hidden, cos, sin):
        events.append("kv")
        assert torch.equal(hidden, global_hidden)
        assert torch.equal(cos.flatten(), context.global_batch.positions.float() + 1)
        return hidden

    impl._project_kv = project_kv

    class Compressor:
        def __call__(self, hidden):
            events.append("c1")
            assert torch.equal(hidden, global_hidden)
            return hidden.clone()

        def wkv(self, hidden):
            assert torch.equal(hidden, global_hidden)
            return hidden.clone()

        def wgate(self, hidden):
            return torch.zeros_like(hidden)

        def pool_projected(self, kv, scores, controls):
            # Replace only the hardware compressor. Verify that its real
            # caller supplies global token order and independent request state.
            events.append("c2")
            assert controls is by_prefix[impl.compressor_state_prefix].global_metadata
            assert controls.c2_ring_metadata.tolist() == [[0, 0], [8, 4], [0, 8], [0, 8], [2, 6]]
            assert torch.equal(kv, global_hidden)
            assert controls.c2_complete_mask.tolist() == [False, True] * 6
            return kv

    swa_cache = torch.zeros(12, 8, 1, 2)
    long_cache = torch.zeros(12, 8 // ratio, 1, 2)
    index_writes, stores = [], []

    def update_keys(latent, slots, cos, sin):
        index_writes.append((latent.clone(), slots.clone()))
        assert latent.shape == (12, 2)
        assert torch.equal(slots, by_prefix["index"].global_metadata.slot_mapping)

    def scatter(cache, slots, updates):
        stores.append((cache.data_ptr(), slots.clone(), updates.clone()))

    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", scatter, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *args, **kwargs: None, raising=False)
    attn = SimpleNamespace(
        rotary_emb=SimpleNamespace(layername="rope"),
        head_dim=2,
        nope_head_dim=0,
        compressor=Compressor(),
        indexer=SimpleNamespace(update_keys=update_keys),
        long_kv_cache=SimpleNamespace(kv_cache=[long_cache]),
        dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=[swa_cache])),
    )
    impl._prepare_inputs_and_caches(attn, gathered[:6], bundle, by_prefix)
    assert events == ["gather", "kv", f"c{ratio}"]
    assert len(index_writes) == 1 and len(stores) == 2
    assert stores[0][0] == swa_cache.data_ptr() and stores[1][0] == long_cache.data_ptr()
    assert torch.equal(stores[0][1], by_prefix[impl.swa_prefix].global_metadata.slot_mapping)
    assert torch.equal(stores[1][1], by_prefix["long"].global_metadata.slot_mapping)
    assert stores[0][2].shape[0] == stores[1][2].shape[0] == 12
    assert int((stores[1][1][:, 0] >= 0).sum()) == 12 // ratio


@pytest.mark.parametrize("kind", ["swa", "long", "state"])
def test_dummy_metadata_suppresses_replicated_writes(production, kind):
    context, local = _two_requests(production, 0)
    context.global_batch.is_dummy = True
    spec = {
        "swa": _SWASpec(),
        "long": _MLASpec(tokens_per_state=2),
        "state": _RingSpec(block_size=32),
    }[kind]
    metadata = _builder(production, spec).build(0, local, pcp_context=context, pcp_cache_group_idx=0)
    assert torch.all(metadata.slot_mapping == -1)
    assert torch.all(metadata.global_metadata.slot_mapping == -1)
    if kind == "state":
        assert metadata.c2_ring_metadata is None
        assert torch.count_nonzero(metadata.global_metadata.c2_ring_metadata[1]) == 0
        assert torch.count_nonzero(metadata.global_metadata.c2_complete_mask) == 0


@pytest.mark.parametrize(
    "mode,empty_cache", [("full", False), ("reindex", False), ("reuse", False), ("full", True), ("reindex", True)]
)
def test_sparse_selection_stays_in_local_query_order(production, mode, empty_cache):
    context, local = _two_requests(production, 1)
    metadata = _builder(production, _IndexSpec(), 1).build(
        0,
        local,
        pcp_context=context,
        pcp_cache_group_idx=0,
    )
    role = _role(
        has_long_context=True,
        compress_ratio=1,
        is_index_source=mode != "reuse",
        is_candidate_source=mode == "full",
        uses_candidate_filter=mode == "reindex",
    )
    topology = SimpleNamespace(candidate_topk_blocks=2, candidate_block_size=2)
    impl = production.AscendDSAV41PCPImpl("layer", role, topology, "long", "index")
    shared = SimpleNamespace(
        topk_indices=torch.arange(16).reshape(8, 2).int(), candidates=torch.arange(16).reshape(8, 2).int() + 100
    )
    before_topk, before_candidates = shared.topk_indices.clone(), shared.candidates.clone()
    cache = object()
    production.get_forward_context = lambda: SimpleNamespace(
        no_compile_layers={"index": SimpleNamespace(kv_cache=[cache])}
    )
    hidden = torch.arange(16).reshape(8, 2).float()
    calls = []

    def select(selected_hidden, qr, positions, cos, sin, source, controls, **kwargs):
        calls.append(kwargs)
        assert torch.equal(selected_hidden, hidden[:6])
        assert torch.equal(positions, local.positions)
        assert source is cache and controls is metadata
        assert torch.equal(controls.cache_seq_lens, local.seq_lens)
        assert torch.equal(kwargs["candidates"], before_candidates[:6])
        output_indices = kwargs["output_indices"]
        assert output_indices.shape == (6, 2)
        assert output_indices.data_ptr() == shared.topk_indices.data_ptr()
        candidates = torch.full((6, 2), -1 if empty_cache else 9, dtype=torch.int32)
        if empty_cache:
            # The new indexer leaves output_indices untouched when there is
            # no long KV. The production caller must clear stale TopK rows.
            return torch.empty((6, 0), dtype=torch.int32), candidates
        # PR16899 writes selections directly into the supplied local view.
        output_indices.fill_(7)
        return output_indices, candidates

    attn = SimpleNamespace(shared_state=shared, indexer=SimpleNamespace(select=select))
    bundle = SimpleNamespace(swa=metadata, indexer=SimpleNamespace(cache=metadata))
    actual = impl._select_sparse_indices(attn, hidden, None, local.positions, None, None, bundle)
    assert actual.shape == (6, 2)
    assert torch.equal(shared.topk_indices[6:], before_topk[6:])
    if mode == "reuse":
        assert not calls and torch.equal(actual, before_topk[:6])
    else:
        assert len(calls) == 1 and torch.all(actual == (-1 if empty_cache else 7))
        assert calls[0]["is_candidate_source"] == (mode == "full")
        assert calls[0]["uses_candidate_filter"] == (mode == "reindex")
    if mode == "full":
        assert torch.all(shared.candidates[:6] == (-1 if empty_cache else 9))
        assert torch.equal(shared.candidates[6:], before_candidates[6:])
    else:
        assert torch.equal(shared.candidates, before_candidates)


@pytest.mark.parametrize("ratio", [1, 2])
def test_attention_operator_receives_local_causal_lengths(production, monkeypatch, ratio):
    context, local = _two_requests(production, 0)
    swa = _builder(production, _SWASpec()).build(
        0,
        local,
        pcp_context=context,
        pcp_cache_group_idx=0,
    )
    long = _builder(production, _MLASpec(tokens_per_state=ratio)).build(
        0,
        local,
        pcp_context=context,
        pcp_cache_group_idx=0,
    )
    long.smla_metadata = torch.zeros(16, dtype=torch.int32)
    impl = production.AscendDSAV41PCPImpl(
        "layer", _role(compress_ratio=ratio), SimpleNamespace(index_topk=512), "long", "index"
    )
    captured = []

    def attention(q, **kwargs):
        captured.append(kwargs)
        return q, None

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", attention, raising=False)
    attn = SimpleNamespace(
        head_dim=512,
        window_size=128,
        attn_sink=None,
        softmax_scale=1,
        dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=[object()])),
    )
    q = torch.zeros(6, 1, 512)
    result = impl._forward_attention(
        attn,
        q,
        SimpleNamespace(swa=swa, attention=long),
        source_cache=object(),
        compressed_indices=torch.zeros(6, 2, dtype=torch.int32),
    )
    assert result is q and len(captured) == 1
    actual = captured[0]
    assert actual["cu_seqlens_q"].tolist() == [0, 2, 4, 5, 6]
    assert actual["seqused_ori_kv"].tolist() == [2, 8, 1, 4]
    assert torch.equal(actual["seqused_cmp_kv"], local.seq_lens // ratio)
    assert actual["cmp_sparse_indices"].shape == (6, 1, 512)
    assert torch.all(actual["cmp_sparse_indices"][:, :, 2:] == -1)
    assert actual["ori_mask_mode"] == 4 and actual["ori_win_left"] == 127
    assert actual["cmp_mask_mode"] == 3


@pytest.mark.parametrize("pcp_size,rank", [(2, 0), (2, 1), *((4, rank) for rank in range(4))])
def test_dummy_context_never_uses_previous_live_batch(production, pcp_size, rank):
    manager = production.AscendPCPManager.__new__(production.AscendPCPManager)
    manager.pcp_rank, manager.pcp_world_size, manager.device = rank, pcp_size, torch.device("cpu")
    manager._global_batch = object()
    manager._hidden_restore_idx = torch.tensor([99])
    manager._global_batch_slot_mappings = torch.tensor([[99]])
    manager._block_tables = SimpleNamespace(gather_block_tables=lambda *args: pytest.fail("reused live block tables"))
    batch = SimpleNamespace(is_dummy=True, num_tokens_after_padding=3, num_reqs=1)
    tables = (torch.zeros(1, 1, dtype=torch.int32),)
    slots = torch.full((1, 3 * pcp_size), -1)
    context = manager.build_attention_context(batch, tables, slots)
    assert context.global_batch is batch and context.global_block_tables is tables
    assert context.global_slot_mappings.shape == (1, 3)
    assert context.hidden_restore_idx.tolist() == list(range(rank * 3, (rank + 1) * 3))
    assert context.padded_gather_idx is None
    assert manager._hidden_restore_idx.tolist() == [99]
    with pytest.raises(AssertionError):
        manager.build_attention_context(batch)
