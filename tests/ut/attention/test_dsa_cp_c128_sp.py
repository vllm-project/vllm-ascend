# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import random
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.attention.dsa_compressor import (
    CompressorExecutor,
    CompressorSPMetadataBuilder,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import AscendDeviceType


@dataclass
class _KernelMetadata:
    query_start_loc: torch.Tensor
    start_pos: torch.Tensor
    num_compressed_tokens: int
    cache_group_key: str


# DSA-CP cache-group classification and resource ownership.

def _build_cpu_dsa_cp_builder(
    cache_group_key: str,
    *,
    compress_ratio: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    head_size: int = 4,
    device_type: AscendDeviceType = AscendDeviceType.A3,
    tp_size: int = 2,
    enable_compressor_sp: bool = True,
):
    """Construct a small CPU builder for cache-group role tests."""
    hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        num_attention_heads=2,
        index_topk=2,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            qk_rope_head_dim=2,
            index_head_dim=3,
            head_dim=4,
        ),
        hf_config=hf_config,
        dtype=torch.bfloat16,
        enable_sleep_mode=False,
        get_hidden_size=lambda: 8,
        get_head_size=lambda: 4,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16,
            max_num_seqs=4,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        speculative_config=None,
        additional_config={"enable_compressor_sp": enable_compressor_sp},
    )
    kv_cache_spec = SimpleNamespace(
        compress_ratio=compress_ratio,
        dtype=dtype,
        head_size=head_size,
    )
    with (
        patch.object(dsa_cp, "get_ascend_device_type", return_value=device_type),
        patch.object(AscendDSACPMetadataBuilder, "hadamard", torch.empty(0)),
    ):
        return AscendDSACPMetadataBuilder(
            kv_cache_spec=kv_cache_spec,
            layer_names=[cache_group_key],
            vllm_config=vllm_config,
            device=torch.device("cpu"),
        )


def test_dsa_cp_builder_initializes_only_resources_used_by_each_group() -> None:
    main_output = _build_cpu_dsa_cp_builder("model.layers.0.attn", compress_ratio=4)
    assert main_output.is_compressor_output
    assert not main_output.is_indexer_compressor_output
    assert main_output.compressor_sp_output_key == ("main", 4)
    assert main_output.compressor_sp_metadata_builder is not None
    assert main_output.req_sas_metadata is not None
    assert main_output.req_qli_metadata is None

    indexer_output = _build_cpu_dsa_cp_builder("model.layers.0.indexer.k_cache", compress_ratio=4)
    assert indexer_output.is_compressor_output
    assert indexer_output.is_indexer_compressor_output
    assert indexer_output.compressor_sp_output_key == ("indexer", 4)
    assert indexer_output.compressor_sp_metadata_builder is not None
    assert indexer_output.req_sas_metadata is None
    assert indexer_output.req_qli_metadata is not None

    main_state = _build_cpu_dsa_cp_builder(
        "model.layers.0.compressor.state_cache",
        dtype=torch.float32,
        head_size=16,
    )
    assert main_state.is_compressor_state
    assert main_state.compressor_sp_output_key is None
    assert main_state.compressor_sp_state_key == ("main", 4)
    assert main_state.compressor_sp_metadata_builder is None
    assert main_state.req_sas_metadata is None
    assert main_state.req_qli_metadata is None

    indexer_state = _build_cpu_dsa_cp_builder(
        "model.layers.0.indexer.compressor.state_cache",
        dtype=torch.float32,
        head_size=12,
    )
    assert indexer_state.is_compressor_state
    assert indexer_state.compressor_sp_output_key is None
    assert indexer_state.compressor_sp_state_key == ("indexer", 4)
    assert indexer_state.compressor_sp_metadata_builder is None
    assert indexer_state.req_sas_metadata is None
    assert indexer_state.req_qli_metadata is None

    non_a3_output = _build_cpu_dsa_cp_builder(
        "model.layers.0.attn",
        compress_ratio=128,
        device_type=AscendDeviceType.A2,
    )
    assert non_a3_output.compressor_sp_output_key == ("main", 128)
    assert non_a3_output.compressor_sp_metadata_builder is None

    tp1_output = _build_cpu_dsa_cp_builder(
        "model.layers.0.attn",
        compress_ratio=128,
        tp_size=1,
    )
    assert tp1_output.compressor_sp_metadata_builder is None

    disabled_output = _build_cpu_dsa_cp_builder(
        "model.layers.0.attn",
        compress_ratio=128,
        enable_compressor_sp=False,
    )
    assert disabled_output.compressor_sp_metadata_builder is None


def test_dsa_cp_builder_skips_slot_mapping_format_for_compressor_output() -> None:
    builder = _build_cpu_dsa_cp_builder("model.layers.0.attn", compress_ratio=128)
    shared_metadata = {
        "num_decodes": 0,
        "num_prefills": 1,
        "num_decode_tokens": 0,
        "num_prefill_tokens": 4,
        "input_positions": torch.arange(4),
        "cos": torch.empty(0),
        "sin": torch.empty(0),
        "seq_lens": torch.tensor([4], dtype=torch.int32),
        "seq_lens_cpu": torch.tensor([4], dtype=torch.int32),
    }
    common_attn_metadata = SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        num_actual_tokens=4,
        num_input_tokens=4,
        attn_state=None,
        slot_mapping=torch.zeros((4, 2), dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
    )
    req_metadata = object()

    with (
        patch.object(builder, "build_req_metadata", return_value=req_metadata),
        patch.object(
            DeviceOperator,
            "format_dsa_slot_mapping",
            side_effect=AssertionError("compressed output must not format ordinary slots"),
        ),
    ):
        metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            common_ratio_to_sas_metadata=shared_metadata,
        )

    assert metadata.req_metadata is req_metadata


# Compressor SP plan reference helpers.

def _build_cpu_compressor_sp_metadata(
    *,
    compress_ratio: int = 128,
    coff: int = 1,
    **kwargs,
):
    """Build one small CPU SP plan with production metadata code."""
    device = kwargs.pop("device")
    tp_size = kwargs.pop("tp_size")
    builder = CompressorSPMetadataBuilder(
        max_num_batched_tokens=kwargs["num_input_tokens"],
        max_num_seqs=len(kwargs["seq_lens"]),
        tp_size=tp_size,
        compress_ratio=compress_ratio,
        coff=coff,
        hidden_dim=1,
        output_dim=2,
        dtype=torch.float32,
        device=device,
    )
    return builder.build_sp(**kwargs)


def _materialize_metadata_input(
    metadata,
    *,
    num_actual_tokens: int,
    num_input_tokens: int,
    tp_size: int,
    tp_rank: int,
) -> list[int]:
    """Resolve pack indices back to global token positions for assertions."""
    tokens_per_rank = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size // tp_size
    suffix_size = metadata.suffix_buffer.shape[0]
    gathered_suffixes: list[int] = []
    for rank in range(tp_size):
        rank_start = rank * tokens_per_rank
        rank_valid = max(0, min(tokens_per_rank, num_actual_tokens - rank_start))
        suffix_valid = min(rank_valid, suffix_size)
        suffix_start = rank_start + rank_valid - suffix_valid
        gathered_suffixes.extend(
            [-1] * (suffix_size - suffix_valid)
            + list(range(suffix_start, suffix_start + suffix_valid))
        )
    local_start = tp_rank * tokens_per_rank
    local_valid = max(0, min(tokens_per_rank, num_actual_tokens - local_start))
    local_hidden = list(range(local_start, local_start + local_valid)) + [-1] * (
        tokens_per_rank - local_valid
    )
    source = gathered_suffixes + local_hidden
    return [source[index] for index in metadata.pack_indices.tolist()]


def _metadata_output_counts(metadata, compress_ratio: int = 128) -> list[int]:
    """Return the raw Compressor output rows produced for each request."""
    query_start_loc = metadata.packed_query_start_loc.tolist()
    start_pos = metadata.packed_start_pos.tolist()
    return [
        (start + query_end - query_start) // compress_ratio - start // compress_ratio
        for query_start, query_end, start in zip(
            query_start_loc[:-1], query_start_loc[1:], start_pos
        )
    ]


# C128 input planning, KV aggregation, and fixed workspaces.


def test_c128_sp_metadata_handles_cross_rank_and_multiple_local_requests() -> None:
    query_start_loc = [0, 230, 270, 320, 570, 640]
    seq_lens = [267, 104, 300, 250, 190]
    expected = {
        0: (
            [0, 160, 160, 160, 160, 160],
            [37, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            160,
            range(0, 160),
        ),
        1: (
            [0, 139, 179, 229, 229, 229],
            [128, 64, 250, 0, 0],
            [1, 0, 1, 0, 0],
            229,
            range(91, 320),
        ),
        2: (
            [0, 0, 0, 0, 160, 160],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            160,
            range(320, 480),
        ),
        3: (
            [0, 0, 0, 0, 122, 192],
            [0, 0, 0, 128, 120],
            [0, 0, 0, 0, 1],
            192,
            range(448, 640),
        ),
    }

    total_outputs = 0
    total_inputs = 0
    reference_reorder = None
    for rank in range(4):
        metadata = _build_cpu_compressor_sp_metadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_actual_tokens=640,
            num_input_tokens=640,
            tp_size=4,
            tp_rank=rank,
            num_reqs_actual=5,
            device=torch.device("cpu"),
        )
        query_locs, start_pos, output_counts, input_count, packed_positions = expected[
            rank
        ]
        assert metadata.packed_query_start_loc.tolist() == query_locs
        assert metadata.packed_start_pos.tolist() == start_pos
        assert _metadata_output_counts(metadata) == output_counts
        assert metadata.input_count == input_count
        assert _materialize_metadata_input(
            metadata,
            num_actual_tokens=640,
            num_input_tokens=640,
            tp_size=4,
            tp_rank=rank,
        ) == list(packed_positions)
        total_outputs += sum(_metadata_output_counts(metadata))
        total_inputs += metadata.input_count
        assert metadata.gathered_compressed_tokens == 6
        assert metadata.global_num_compressed_tokens == 10
        if reference_reorder is None:
            reference_reorder = metadata.gathered_kv_reorder_indices
        else:
            torch.testing.assert_close(
                metadata.gathered_kv_reorder_indices,
                reference_reorder,
            )

    assert total_outputs == 5
    assert total_inputs == 741
    assert reference_reorder is not None
    assert reference_reorder.tolist() == [0, 6, 7, 12, 18, 0, 0, 0, 0, 0]


def test_c128_sp_metadata_reuses_cpu_gpu_buffers_across_batches() -> None:
    builder = CompressorSPMetadataBuilder(
        max_num_batched_tokens=640,
        max_num_seqs=5,
        tp_size=4,
        compress_ratio=128,
        coff=1,
        hidden_dim=1,
        output_dim=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    first = builder.build_sp(
        query_start_loc=[0, 230, 270, 320, 570, 640],
        seq_lens=[267, 104, 300, 250, 190],
        num_actual_tokens=640,
        num_input_tokens=640,
        tp_rank=1,
        num_reqs_actual=5,
    )
    tensor_names = (
        "packed_query_start_loc",
        "packed_start_pos",
        "suffix_buffer",
        "pack_indices",
        "gathered_kv_reorder_indices",
        "compressed_kv_send_buffer",
        "gathered_compressed_kv_buffer",
        "state_send_buffer",
        "gathered_state_buffer",
        "local_block_indices",
        "local_offset_indices",
        "valid_slots",
        "scatter_slot_mapping",
    )
    first_ptrs = {
        name: getattr(first, name).data_ptr()
        for name in tensor_names
    }

    second = builder.build_sp(
        query_start_loc=[0, 100],
        seq_lens=[100],
        num_actual_tokens=100,
        num_input_tokens=100,
        tp_rank=3,
        num_reqs_actual=1,
    )

    assert {
        name: getattr(second, name).data_ptr()
        for name in tensor_names
    } == first_ptrs
    assert second.packed_query_start_loc.tolist() == [0, 100]
    assert second.packed_start_pos.tolist() == [0]
    assert second.pack_indices.shape == (100,)
    assert second.gathered_kv_reorder_indices.tolist() == [0]
    assert first.compressed_kv_send_buffer.shape == (6, 2)
    assert first.gathered_compressed_kv_buffer.shape == (24, 2)
    assert first.state_send_buffer.shape == (160, 4)
    assert first.gathered_state_buffer.shape == (640, 4)
    assert second.compressed_kv_send_buffer.shape == (1, 2)
    assert second.gathered_compressed_kv_buffer.shape == (4, 2)
    assert second.state_send_buffer.shape == (25, 4)
    assert second.gathered_state_buffer.shape == (100, 4)


def test_compressor_metadata_builder_derives_and_reuses_safe_state_slots() -> None:
    builder = CompressorSPMetadataBuilder(
        max_num_batched_tokens=3,
        max_num_seqs=1,
        tp_size=2,
        compress_ratio=128,
        coff=1,
        hidden_dim=1,
        output_dim=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    first = builder.build_sp(
        query_start_loc=[0, 3],
        seq_lens=[3],
        num_actual_tokens=3,
        num_input_tokens=3,
        tp_rank=0,
        num_reqs_actual=1,
    )
    builder.bind_state_slots(
        metadata=first,
        state_slot_mapping=torch.tensor(
            [[1, 2], [0, 3], [2, 1]],
            dtype=torch.int32,
        ),
        local_token_start=0,
        tokens_per_rank=2,
        num_tokens_pad=4,
    )

    assert first.local_block_indices.tolist() == [1, 0]
    assert first.local_offset_indices.tolist() == [2, 0]
    assert first.valid_slots.tolist() == [True, False, True, False]
    torch.testing.assert_close(
        first.scatter_slot_mapping,
        torch.tensor([[1, 2], [0, 0], [2, 1], [0, 0]], dtype=torch.int32),
    )
    buffer_ptr = first.scatter_slot_mapping.data_ptr()

    second = builder.build_sp(
        query_start_loc=[0, 1],
        seq_lens=[1],
        num_actual_tokens=1,
        num_input_tokens=3,
        tp_rank=1,
        num_reqs_actual=1,
    )
    builder.bind_state_slots(
        metadata=second,
        state_slot_mapping=torch.tensor([[3, 4]], dtype=torch.int32),
        local_token_start=2,
        tokens_per_rank=2,
        num_tokens_pad=4,
    )

    assert second.scatter_slot_mapping.data_ptr() == buffer_ptr
    assert second.local_block_indices.tolist() == [0, 0]
    assert second.local_offset_indices.tolist() == [0, 0]
    assert second.valid_slots.tolist() == [True, False, False, False]
    torch.testing.assert_close(
        second.scatter_slot_mapping,
        torch.tensor([[3, 4], [0, 0], [0, 0], [0, 0]], dtype=torch.int32),
    )


def test_c128_sp_metadata_replays_boundary_from_multiple_upstream_ranks() -> None:
    metadata = _build_cpu_compressor_sp_metadata(
        query_start_loc=[0, 100],
        seq_lens=[100],
        num_actual_tokens=100,
        num_input_tokens=100,
        tp_size=4,
        tp_rank=3,
        num_reqs_actual=1,
        device=torch.device("cpu"),
    )

    assert metadata.input_count == 100
    assert metadata.packed_query_start_loc.tolist() == [0, 100]
    assert metadata.packed_start_pos.tolist() == [0]
    assert _materialize_metadata_input(
        metadata,
        num_actual_tokens=100,
        num_input_tokens=100,
        tp_size=4,
        tp_rank=3,
    ) == list(range(100))


def test_c128_input_pack_uses_blocking_fixed_shape_all_gather() -> None:
    sp_metadata = _build_cpu_compressor_sp_metadata(
        query_start_loc=[0, 230, 270, 320, 570, 640],
        seq_lens=[267, 104, 300, 250, 190],
        num_actual_tokens=640,
        num_input_tokens=640,
        tp_size=4,
        tp_rank=1,
        num_reqs_actual=5,
        device=torch.device("cpu"),
    )
    tp_group = SimpleNamespace(world_size=4, device_group=object())
    executor = CompressorExecutor(
        SimpleNamespace(compress_ratio=128),
        rope_head_dim=64,
        tp_group=tp_group,
    )
    hidden_states_local = torch.arange(160, 320, dtype=torch.float32).unsqueeze(-1)
    gathered_suffixes = torch.cat(
        [
            torch.arange(rank * 160 + 32, (rank + 1) * 160, dtype=torch.float32)
            for rank in range(4)
        ]
    ).unsqueeze(-1)

    def fake_all_gather(output, local_suffix, *, group, async_op):
        assert output.shape == (4 * 128, 1)
        assert local_suffix.shape == (128, 1)
        torch.testing.assert_close(
            local_suffix[:, 0],
            torch.arange(192, 320, dtype=torch.float32),
        )
        assert group is tp_group.device_group
        assert async_op is False
        output.copy_(gathered_suffixes)

    with (
        patch.object(
            torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather
        ) as gather,
        patch.object(torch, "cat", wraps=torch.cat) as cat,
    ):
        packed = executor.prepare_sp_input(hidden_states_local, sp_metadata)

    gather.assert_called_once()
    cat.assert_called_once()
    torch.testing.assert_close(
        packed[:229, 0], torch.arange(91, 320, dtype=torch.float32)
    )
    assert packed.shape == (229, 1)


def test_c128_compressed_gather_uses_one_collective_and_preallocated_buffers() -> None:
    sp_metadata = _build_cpu_compressor_sp_metadata(
        query_start_loc=[0, 300],
        seq_lens=[300],
        num_actual_tokens=300,
        num_input_tokens=300,
        tp_size=4,
        tp_rank=0,
        num_reqs_actual=1,
        device=torch.device("cpu"),
    )
    tp_group = SimpleNamespace(world_size=4, device_group=object())
    executor = CompressorExecutor(
        SimpleNamespace(compress_ratio=128),
        rope_head_dim=64,
        tp_group=tp_group,
    )
    compressed_kv = torch.tensor([[1.0, 2.0]])
    calls = 0

    def fake_all_gather(output, local, *, group, async_op):
        nonlocal calls
        calls += 1
        assert group is tp_group.device_group
        assert async_op is False
        assert local.shape == (2, 2)
        torch.testing.assert_close(local[0], compressed_kv[0])
        output.copy_(local.repeat(4, 1))

    with patch.object(
        torch.distributed, "all_gather_into_tensor", side_effect=fake_all_gather
    ):
        gathered_kv = executor._gather_sp_output(
            compressed_kv,
            sp_metadata,
        )
        send_buffer = sp_metadata.compressed_kv_send_buffer
        gather_buffer = sp_metadata.gathered_compressed_kv_buffer
        gathered_kv_again = executor._gather_sp_output(
            compressed_kv,
            sp_metadata,
        )

    assert calls == 2  # One compressed-KV collective per invocation; no slot collective.
    assert gathered_kv.shape == (8, 2)
    assert gathered_kv_again.data_ptr() == gathered_kv.data_ptr()
    assert sp_metadata.compressed_kv_send_buffer is send_buffer
    assert sp_metadata.gathered_compressed_kv_buffer is gather_buffer
    assert sp_metadata.gathered_kv_reorder_indices.tolist() == [2, 6, 0]


# C4 overlap planning and shared main/LI input semantics.


def test_c4_sp_metadata_replays_overlap_and_reorders_raw_outputs() -> None:
    expected_inputs = {
        0: list(range(0, 4)),
        1: list(range(0, 8)),
        2: list(range(4, 12)),
        3: list(range(8, 16)),
    }
    reference_reorder = None
    for rank in range(4):
        metadata = _build_cpu_compressor_sp_metadata(
            query_start_loc=[0, 16],
            seq_lens=[16],
            num_actual_tokens=16,
            num_input_tokens=16,
            tp_size=4,
            tp_rank=rank,
            num_reqs_actual=1,
            compress_ratio=4,
            coff=2,
            device=torch.device("cpu"),
        )

        assert metadata.suffix_buffer.shape == (8, 1)
        assert metadata.input_capacity == 11
        assert _materialize_metadata_input(
            metadata,
            num_actual_tokens=16,
            num_input_tokens=16,
            tp_size=4,
            tp_rank=rank,
        ) == expected_inputs[rank]
        assert metadata.gathered_compressed_tokens == 3
        assert metadata.global_num_compressed_tokens == 5
        if reference_reorder is None:
            reference_reorder = metadata.gathered_kv_reorder_indices
        else:
            torch.testing.assert_close(metadata.gathered_kv_reorder_indices, reference_reorder)

    assert reference_reorder is not None
    assert reference_reorder.tolist() == [0, 4, 7, 10, 0]


def test_c4_main_and_indexer_metadata_share_plan_but_own_workspaces() -> None:
    builders = [
        CompressorSPMetadataBuilder(
            max_num_batched_tokens=16,
            max_num_seqs=1,
            tp_size=4,
            compress_ratio=4,
            coff=2,
            hidden_dim=1,
            output_dim=output_dim,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        for output_dim in (4, 1)
    ]
    metadata = [
        builder.build_sp(
            query_start_loc=[0, 16],
            seq_lens=[16],
            num_actual_tokens=16,
            num_input_tokens=16,
            tp_rank=1,
            num_reqs_actual=1,
        )
        for builder in builders
    ]
    main_metadata, indexer_metadata = metadata

    for field in (
        "packed_query_start_loc",
        "packed_start_pos",
        "pack_indices",
        "gathered_kv_reorder_indices",
    ):
        torch.testing.assert_close(
            getattr(main_metadata, field),
            getattr(indexer_metadata, field),
        )

    assert main_metadata.local_suffix_start == indexer_metadata.local_suffix_start
    assert main_metadata.local_suffix_valid_len == indexer_metadata.local_suffix_valid_len
    assert main_metadata.suffix_buffer.data_ptr() != indexer_metadata.suffix_buffer.data_ptr()

    assert main_metadata.compressed_kv_send_buffer.shape == (3, 4)
    assert indexer_metadata.compressed_kv_send_buffer.shape == (3, 1)
    assert main_metadata.state_send_buffer.shape == (4, 16)
    assert indexer_metadata.state_send_buffer.shape == (4, 4)

    main_slots = torch.tensor([[1, offset % 4] for offset in range(16)], dtype=torch.int32)
    indexer_slots = torch.tensor([[2, offset % 4] for offset in range(16)], dtype=torch.int32)
    builders[0].bind_state_slots(main_metadata, main_slots, 4, 4, 16)
    builders[1].bind_state_slots(indexer_metadata, indexer_slots, 4, 4, 16)

    assert main_metadata.scatter_slot_mapping.data_ptr() != indexer_metadata.scatter_slot_mapping.data_ptr()
    assert main_metadata.local_block_indices.tolist() == [1, 1, 1, 1]
    assert indexer_metadata.local_block_indices.tolist() == [2, 2, 2, 2]


def test_c4_sp_metadata_handles_unaligned_rank_boundary_and_multiple_requests() -> None:
    query_start_loc = [0, 9, 19, 28]
    seq_lens = [9, 14, 11]
    max_replayed_tokens = 0
    reference_reorder = None

    for rank in range(4):
        metadata = _build_cpu_compressor_sp_metadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_actual_tokens=28,
            num_input_tokens=28,
            tp_size=4,
            tp_rank=rank,
            num_reqs_actual=3,
            compress_ratio=4,
            coff=2,
            device=torch.device("cpu"),
        )
        packed_query_start_loc = metadata.packed_query_start_loc.tolist()
        for req_idx, (query_start, query_end) in enumerate(
            zip(query_start_loc[:-1], query_start_loc[1:])
        ):
            local_rows = max(
                0,
                min(query_end, (rank + 1) * 7) - max(query_start, rank * 7),
            )
            packed_rows = packed_query_start_loc[req_idx + 1] - packed_query_start_loc[req_idx]
            replayed_rows = packed_rows - local_rows
            assert 0 <= replayed_rows <= 7
            max_replayed_tokens = max(max_replayed_tokens, replayed_rows)
        assert all(
            query_end >= query_start
            for query_start, query_end in zip(
                packed_query_start_loc[:-1], packed_query_start_loc[1:]
            )
        )
        assert metadata.input_count <= metadata.input_capacity == 14
        if reference_reorder is None:
            reference_reorder = metadata.gathered_kv_reorder_indices
        else:
            torch.testing.assert_close(metadata.gathered_kv_reorder_indices, reference_reorder)

    expected_outputs = sum(
        seq_len // 4 - (seq_len - (query_end - query_start)) // 4
        for query_start, query_end, seq_len in zip(
            query_start_loc[:-1], query_start_loc[1:], seq_lens
        )
    )
    assert expected_outputs == 6
    assert max_replayed_tokens == 7
    assert reference_reorder is not None
    assert len(set(reference_reorder.tolist()[:expected_outputs])) == expected_outputs


def test_compressor_sp_metadata_randomized_boundaries_cover_each_output_once() -> None:
    rng = random.Random(20260829)

    for compress_ratio, coff in ((4, 2), (128, 1)):
        for _ in range(100):
            tp_size = rng.choice((2, 4, 8))
            num_actual_tokens = rng.randint(1, 384)
            num_reqs = rng.randint(1, min(8, num_actual_tokens))
            cuts = sorted(rng.sample(range(1, num_actual_tokens), num_reqs - 1))
            query_start_loc = [0, *cuts, num_actual_tokens]
            query_lens = [
                query_end - query_start
                for query_start, query_end in zip(
                    query_start_loc[:-1], query_start_loc[1:]
                )
            ]
            prefix_lens = [rng.randint(0, 192) for _ in range(num_reqs)]
            seq_lens = [
                prefix_len + query_len
                for prefix_len, query_len in zip(prefix_lens, query_lens)
            ]
            num_input_tokens = num_actual_tokens + rng.randint(0, tp_size * 3)
            tokens_per_rank = (
                (num_input_tokens + tp_size - 1) // tp_size
            )
            expected_outputs = sum(
                seq_len // compress_ratio - prefix_len // compress_ratio
                for seq_len, prefix_len in zip(seq_lens, prefix_lens)
            )

            reference_reorder = None
            for tp_rank in range(tp_size):
                metadata = _build_cpu_compressor_sp_metadata(
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    num_actual_tokens=num_actual_tokens,
                    num_input_tokens=num_input_tokens,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    num_reqs_actual=num_reqs,
                    compress_ratio=compress_ratio,
                    coff=coff,
                    device=torch.device("cpu"),
                )

                assert metadata.suffix_buffer.shape == (compress_ratio * coff, 1)
                assert metadata.input_count <= metadata.input_capacity
                packed_positions = _materialize_metadata_input(
                    metadata,
                    num_actual_tokens=num_actual_tokens,
                    num_input_tokens=num_input_tokens,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
                packed_query_start_loc = metadata.packed_query_start_loc.tolist()
                local_start = tp_rank * tokens_per_rank
                local_end = local_start + tokens_per_rank
                overlap_tokens = compress_ratio * (coff - 1)
                for req_idx, (query_start, query_end, prefix_len) in enumerate(
                    zip(
                        query_start_loc[:-1],
                        query_start_loc[1:],
                        prefix_lens,
                    )
                ):
                    flat_start = max(query_start, local_start)
                    flat_end = min(query_end, local_end, num_actual_tokens)
                    segment_start = packed_query_start_loc[req_idx]
                    segment_end = packed_query_start_loc[req_idx + 1]
                    if flat_end <= flat_start:
                        assert segment_start == segment_end
                        continue
                    absolute_start = prefix_len + flat_start - query_start
                    group_start = absolute_start // compress_ratio * compress_ratio
                    compute_start = max(prefix_len, group_start - overlap_tokens)
                    boundary_flat_start = query_start + compute_start - prefix_len
                    assert metadata.packed_start_pos[req_idx].item() == compute_start
                    assert packed_positions[segment_start:segment_end] == list(
                        range(boundary_flat_start, flat_end)
                    )

                reorder = metadata.gathered_kv_reorder_indices.tolist()
                if reference_reorder is None:
                    reference_reorder = reorder
                else:
                    assert reorder == reference_reorder

            assert reference_reorder is not None
            assert len(set(reference_reorder[:expected_outputs])) == expected_outputs


def test_c4_input_pack_uses_blocking_eight_token_suffix_all_gather() -> None:
    sp_metadata = _build_cpu_compressor_sp_metadata(
        query_start_loc=[0, 16],
        seq_lens=[16],
        num_actual_tokens=16,
        num_input_tokens=16,
        tp_size=4,
        tp_rank=1,
        num_reqs_actual=1,
        compress_ratio=4,
        coff=2,
        device=torch.device("cpu"),
    )
    tp_group = SimpleNamespace(world_size=4, device_group=object())
    executor = CompressorExecutor(
        SimpleNamespace(compress_ratio=4),
        rope_head_dim=64,
        tp_group=tp_group,
    )
    hidden_states_local = torch.arange(4, 8, dtype=torch.float32).unsqueeze(-1)
    gathered_suffixes = torch.cat(
        [
            torch.cat(
                [
                    torch.full((4,), -1.0),
                    torch.arange(rank * 4, (rank + 1) * 4, dtype=torch.float32),
                ]
            )
            for rank in range(4)
        ]
    ).unsqueeze(-1)

    def fake_all_gather(output, local_suffix, *, group, async_op):
        assert output.shape == (4 * 8, 1)
        assert local_suffix.shape == (8, 1)
        torch.testing.assert_close(
            local_suffix[:, 0],
            torch.tensor([0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0]),
        )
        assert group is tp_group.device_group
        assert async_op is False
        output.copy_(gathered_suffixes)

    with patch.object(
        torch.distributed,
        "all_gather_into_tensor",
        side_effect=fake_all_gather,
    ) as gather:
        packed = executor.prepare_sp_input(hidden_states_local, sp_metadata)

    gather.assert_called_once()
    torch.testing.assert_close(packed[:, 0], torch.arange(0, 8, dtype=torch.float32))


def test_empty_rank_suffix_buffer_is_zero() -> None:
    sp_metadata = _build_cpu_compressor_sp_metadata(
        query_start_loc=[0, 4],
        seq_lens=[4],
        num_actual_tokens=4,
        num_input_tokens=16,
        tp_size=4,
        tp_rank=3,
        num_reqs_actual=1,
        compress_ratio=4,
        coff=2,
        device=torch.device("cpu"),
    )
    tp_group = SimpleNamespace(world_size=4, device_group=object())
    executor = CompressorExecutor(
        SimpleNamespace(compress_ratio=4),
        rope_head_dim=64,
        tp_group=tp_group,
    )

    def fake_all_gather(output, local_suffix, *, group, async_op):
        torch.testing.assert_close(local_suffix, torch.zeros((8, 1)))
        assert group is tp_group.device_group
        assert async_op is False
        output.zero_()

    with patch.object(
        torch.distributed,
        "all_gather_into_tensor",
        side_effect=fake_all_gather,
    ):
        packed = executor.prepare_sp_input(torch.ones((4, 1)), sp_metadata)

    assert sp_metadata.local_suffix_start == 0
    assert sp_metadata.local_suffix_valid_len == 0
    assert packed.shape == (0, 1)


def test_c4_sp_executor_gathers_raw_output_before_global_reorder() -> None:
    sp_metadata = _build_cpu_compressor_sp_metadata(
        query_start_loc=[0, 16],
        seq_lens=[16],
        num_actual_tokens=16,
        num_input_tokens=16,
        tp_size=4,
        tp_rank=1,
        num_reqs_actual=1,
        compress_ratio=4,
        coff=2,
        device=torch.device("cpu"),
    )
    tp_group = SimpleNamespace(world_size=4, device_group=object())
    executor = CompressorExecutor(
        SimpleNamespace(
            compress_ratio=4,
            coff=2,
            norm=SimpleNamespace(weight=torch.ones(2)),
        ),
        rope_head_dim=1,
        tp_group=tp_group,
    )
    raw_compressed_kv = torch.tensor([[10.0], [11.0], [99.0]])
    gathered_kv = torch.arange(12, dtype=torch.float32).unsqueeze(-1)
    global_slot_mapping = torch.zeros((5, 2), dtype=torch.int32)
    metadata = _KernelMetadata(
        query_start_loc=torch.tensor([0, 16], dtype=torch.int32),
        start_pos=torch.tensor([0], dtype=torch.int32),
        num_compressed_tokens=5,
        cache_group_key="c4",
    )
    state_cache = torch.empty((1, 1, 1, 2))

    def fake_gather(compressed_kv, metadata_arg):
        assert metadata_arg is sp_metadata
        torch.testing.assert_close(compressed_kv, raw_compressed_kv)
        return gathered_kv

    with (
        patch.object(
            executor,
            "_run_kernel",
            return_value=(raw_compressed_kv, torch.empty((3, 2), dtype=torch.int32)),
        ),
        patch.object(executor, "_gather_sp_output", side_effect=fake_gather),
        patch.object(executor, "_write_cache") as write_cache,
        patch.object(executor, "_sync_sp_state") as sync_state,
        patch(
            "vllm_ascend.attention.dsa_compressor.get_or_compute_compressor_metadata",
            return_value=(torch.empty(0), torch.empty(0), global_slot_mapping),
        ),
    ):
        executor.run(
            torch.ones((8, 2)),
            state_cache,
            torch.empty((1, 1, 2)),
            metadata=metadata,
            state_block_table=torch.zeros((1, 1), dtype=torch.int32),
            sp_metadata=sp_metadata,
        )

    write_cache.assert_called_once()
    torch.testing.assert_close(
        write_cache.call_args.args[0],
        torch.tensor([[0.0], [4.0], [7.0], [10.0], [0.0]]),
    )
    assert write_cache.call_args.args[1] is global_slot_mapping
    sync_state.assert_called_once()
    assert sync_state.call_args.args[0] is state_cache
    assert sync_state.call_args.args[1] is sp_metadata


# Full state write-set synchronization required by prefix caching.


def test_state_sync_all_gathers_full_write_set_from_padded_cache() -> None:
    tp_group = SimpleNamespace(world_size=2, device_group=object())
    executor = CompressorExecutor(
        SimpleNamespace(compress_ratio=128),
        rope_head_dim=64,
        tp_group=tp_group,
    )
    builder = CompressorSPMetadataBuilder(
        max_num_batched_tokens=4,
        max_num_seqs=1,
        tp_size=2,
        compress_ratio=128,
        coff=1,
        hidden_dim=1,
        output_dim=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    sp_metadata = builder.build_sp(
        query_start_loc=[0, 3],
        seq_lens=[3],
        num_actual_tokens=3,
        num_input_tokens=4,
        tp_rank=0,
        num_reqs_actual=1,
    )
    # Row 0 is an internal cacheable checkpoint, row 2 is the current tail.
    # Block 0 and the padded row are outside the non-SP state write-set.
    state_slot_mapping = torch.tensor(
        [[1, 0], [0, 1], [1, 3]],
        dtype=torch.int32,
    )
    builder.bind_state_slots(
        metadata=sp_metadata,
        state_slot_mapping=state_slot_mapping,
        local_token_start=0,
        tokens_per_rank=2,
        num_tokens_pad=4,
    )
    state_cache_base = torch.arange(36, dtype=torch.float32).reshape(3, 6, 2)
    state_cache = state_cache_base[:2, :4].unsqueeze(-2)
    assert not state_cache.squeeze(-2).is_contiguous()

    def fake_all_gather(output, local, *, group, async_op):
        assert group is tp_group.device_group
        assert async_op is False
        torch.testing.assert_close(local, torch.tensor([[12.0, 13.0], [0.0, 1.0]]))
        output.copy_(
            torch.tensor(
                [[12.0, 13.0], [2.0, 3.0], [30.0, 31.0], [40.0, 41.0]]
            )
        )

    with (
        patch.object(
            torch.distributed,
            "all_gather_into_tensor",
            side_effect=fake_all_gather,
        ),
        patch.object(DeviceOperator, "dsa_kv_compress_scatter") as scatter,
    ):
        executor._sync_sp_state(
            state_cache,
            sp_metadata,
        )

    scatter.assert_called_once()
    torch.testing.assert_close(
        scatter.call_args.args[1],
        torch.tensor([[12.0, 13.0], [0.0, 1.0], [30.0, 31.0], [0.0, 1.0]]),
    )
    torch.testing.assert_close(
        scatter.call_args.args[2],
        torch.tensor([[1, 0], [0, 0], [1, 3], [0, 0]], dtype=torch.int32),
    )


def test_state_sync_supports_partial_row_page_padding() -> None:
    tp_group = SimpleNamespace(world_size=1, device_group=object())
    executor = CompressorExecutor(
        SimpleNamespace(compress_ratio=4),
        rope_head_dim=1,
        tp_group=tp_group,
    )
    builder = CompressorSPMetadataBuilder(
        max_num_batched_tokens=2,
        max_num_seqs=1,
        tp_size=1,
        compress_ratio=4,
        coff=2,
        hidden_dim=1,
        output_dim=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    sp_metadata = builder.build_sp(
        query_start_loc=[0, 2],
        seq_lens=[2],
        num_actual_tokens=2,
        num_input_tokens=2,
        tp_rank=0,
        num_reqs_actual=1,
    )
    builder.bind_state_slots(
        metadata=sp_metadata,
        state_slot_mapping=torch.tensor([[1, 0], [1, 1]], dtype=torch.int32),
        local_token_start=0,
        tokens_per_rank=2,
        num_tokens_pad=2,
    )

    # Each logical page has 2 * 4 elements plus 2 padding elements. This is
    # the same layout class as LI state cache's 2 * 512 + 16 elements.
    state_storage = torch.arange(30, dtype=torch.float32)
    state_cache = torch.as_strided(
        state_storage,
        size=(2, 2, 1, 4),
        stride=(10, 4, 4, 1),
    )

    def fake_all_gather(output, local, *, group, async_op):
        assert group is tp_group.device_group
        assert async_op is False
        torch.testing.assert_close(
            local,
            torch.tensor([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]]),
        )
        output.copy_(local)

    with (
        patch.object(
            torch.distributed,
            "all_gather_into_tensor",
            side_effect=fake_all_gather,
        ),
        patch.object(DeviceOperator, "dsa_kv_compress_scatter") as scatter,
    ):
        executor._sync_sp_state(state_cache, sp_metadata)

    scatter.assert_called_once()
    torch.testing.assert_close(
        scatter.call_args.args[1],
        torch.tensor([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]]),
    )
