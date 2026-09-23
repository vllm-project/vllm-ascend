# SPDX-License-Identifier: Apache-2.0
"""CPU metadata contracts for MLA DCP producers and graph updates."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.mla_cp import AscendMLADCPDecodeMetadata
from vllm_ascend.worker.dcp_utils import DCPManager
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


def _local_lengths(lengths, size, rank, interleave):
    # Independent oracle: count the token positions owned by this rank.
    return [sum((pos // interleave) % size == rank for pos in range(length)) for length in lengths]


def _manager(size, rank, interleave):
    manager = object.__new__(DCPManager)
    manager.dcp_world_size = size
    manager.dcp_world_rank = rank
    manager.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave)
    )
    return manager


@pytest.mark.parametrize("size,interleave", [(2, 1), (8, 4), (16, 128)])
@pytest.mark.parametrize("use_internal_cpu", [False, True])
def test_common_lengths_cover_all_ranks_and_preserve_global(size, interleave, use_internal_cpu):
    cycle = size * interleave
    expected_global = [0, 1, interleave - 1, interleave, interleave + 1, cycle - 1, cycle, cycle + 1]
    lengths = torch.tensor(expected_global, dtype=torch.int32)
    rank_lengths = []
    for rank in range(size):
        common = SimpleNamespace(
            _seq_lens_cpu=lengths if use_internal_cpu else None,
            seq_lens_cpu=lengths + 100 if use_internal_cpu else lengths,
            num_reqs=len(lengths),
            seq_lens=lengths + 3,
            dcp_local_seq_lens=torch.full_like(lengths, -2),
            dcp_local_seq_lens_cpu=torch.full_like(lengths, -1),
        )
        _manager(size, rank, interleave).prepare_common_attn_metadata(common)
        assert common.dcp_local_seq_lens.tolist() == _local_lengths(
            (lengths + 3).tolist(), size, rank, interleave
        )
        local = common.dcp_local_seq_lens_cpu
        assert local.tolist() == _local_lengths(lengths.tolist(), size, rank, interleave)
        rank_lengths.append(local)
        assert lengths.tolist() == expected_global
        assert common.seq_lens.tolist() == [length + 3 for length in expected_global]
    assert torch.equal(torch.stack(rank_lengths).sum(0), lengths)


@pytest.mark.parametrize("for_capture", [False, True])
@pytest.mark.parametrize("use_dcp", [False, True])
def test_v2_common_lengths_are_shared_by_groups(monkeypatch, for_capture, use_dcp):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8 if use_dcp else 1, cp_kv_cache_interleave_size=4
        )
    )
    monkeypatch.setattr(attn_utils, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1))

    class Builder:
        def build(self, common_prefix_len, common_attn_metadata):
            return common_attn_metadata

        def build_for_cudagraph_capture(self, common_attn_metadata):
            return common_attn_metadata

    groups = [
        [SimpleNamespace(get_metadata_builder=lambda _: Builder(), layer_names=[name])]
        for name in ("layer0", "layer1")
    ]
    lengths = torch.tensor([37, 128, 0, 0], dtype=torch.int32)
    device_local = torch.tensor([5, 16, 0, 0], dtype=torch.int32) if use_dcp else None
    with patch.object(attn_utils, "get_dcp_local_seq_lens", wraps=attn_utils.get_dcp_local_seq_lens) as partition:
        result = attn_utils.build_attn_metadata(
            attn_groups=groups,
            num_reqs=4,
            num_actual_reqs=2,
            num_tokens=4,
            query_start_loc_gpu=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
            max_query_len=1,
            seq_lens=lengths,
            seq_lens_np=lengths.numpy(),
            max_seq_len=4096,
            block_tables=[torch.zeros((4, 2), dtype=torch.int32)] * 2,
            slot_mappings=torch.zeros((2, 4), dtype=torch.int64),
            kv_cache_config=SimpleNamespace(kv_cache_groups=[None, None]),
            dcp_local_seq_lens=device_local,
            parallel_config=config.parallel_config,
            for_cudagraph_capture=for_capture,
        )
    assert partition.call_count == int(use_dcp)
    common0, common1 = result["layer0"], result["layer1"]
    assert common0.dcp_local_seq_lens is device_local
    assert common0.dcp_local_seq_lens_cpu is common1.dcp_local_seq_lens_cpu
    if use_dcp:
        assert common0.dcp_local_seq_lens_cpu.tolist() == [5, 16, 0, 0]
    else:
        assert common0.dcp_local_seq_lens_cpu is None
    assert common0.seq_lens_cpu.tolist() == lengths.tolist()


@pytest.mark.parametrize("size,rank,interleave", [(2, 0, 4), (2, 1, 4), (8, 7, 1)])
def test_full_draft_steps_refresh_local_lengths_without_aliasing(size, rank, interleave):
    speculator = object.__new__(AscendAutoRegressiveSpeculator)
    speculator.attn_architecture = "MLA"
    speculator.use_dcp = True
    speculator.max_model_len = 128
    speculator.dcp_manager = _manager(size, rank, interleave)
    speculator.draft_vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=size, cp_kv_cache_interleave_size=interleave
        )
    )
    target = torch.tensor([7, 127, 0, 0], dtype=torch.int32)
    speculator._get_seq_lens_cpu = lambda _: target
    metadata_steps = []
    for step in (1, 2, 3):
        decode = object.__new__(AscendMLADCPDecodeMetadata)
        metadata = SimpleNamespace(decode=decode, seq_lens_cpu=torch.zeros_like(target))
        with patch.object(
            speculator.dcp_manager,
            "prepare_dcp_local_seq_lens_cpu",
            wraps=speculator.dcp_manager.prepare_dcp_local_seq_lens_cpu,
        ) as partition:
            speculator._update_decode_attn_metadata(
                {"layer0": metadata, "layer1": metadata}, step, num_reqs=2
            )
        partition.assert_called_once()
        metadata_steps.append(metadata)

    for step, metadata in enumerate(metadata_steps, start=1):
        lengths = [7 + step, 128, 0, 0]
        assert metadata.seq_lens_cpu.tolist() == lengths
        assert metadata.decode.cp_seq_len == _local_lengths(lengths, size, rank, interleave)
        assert metadata.decode.cp_history_seq_len == _local_lengths(
            [6 + step, 127, 0, 0], size, rank, interleave
        )
    assert target.tolist() == [7, 127, 0, 0]




def test_draft_factory_forwards_cpu_lengths_and_explicit_config(monkeypatch):
    module = SimpleNamespace(build_attn_metadata=lambda **kwargs: kwargs)
    original = module.build_attn_metadata
    monkeypatch.setattr(attn_utils, "_BUILD_ATTN_METADATA_MODULE", module)
    lengths = torch.tensor([32, 128, 0, 0], dtype=torch.int32)
    config = SimpleNamespace(decode_context_parallel_size=8, cp_kv_cache_interleave_size=4)
    with attn_utils.build_draft_attn_metadata_factory(
        torch.arange(8),
        4,
        torch.zeros(4, dtype=torch.bool),
        seq_lens_cpu=lengths,
        parallel_config=config,
    ):
        forwarded = module.build_attn_metadata()
    assert module.build_attn_metadata is original
    assert forwarded["parallel_config"] is config
    assert forwarded["seq_lens_np"].tolist() == [32, 128, 0, 0]
    assert forwarded["positions"].tolist() == [0, 1, 2, 3]
