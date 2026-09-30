"""Regression tests for Ascend QSA metadata with DP padding."""

from types import SimpleNamespace

import torch

from vllm_ascend.models.qwen4_exp.qsa import _ascend_build_qsa_metadata_torch


def test_phantom_query_tokens_do_not_get_cache_slots() -> None:
    # FIA can append a synthetic request to query_start_loc after DP padding.
    # Only the first four tokens belong to the real scheduler batch.
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        query_start_loc_cpu=torch.tensor([0, 4, 8], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        seq_lens=torch.tensor([4, 4], dtype=torch.int32),
        slot_mapping=torch.arange(4, dtype=torch.int64),
        token_to_req_indices=lambda _buffer: torch.zeros(4, dtype=torch.int32),
    )
    positions = torch.empty(4, dtype=torch.int32)
    visible_blocks = torch.empty(4, dtype=torch.int32)
    _, actual_positions, actual_visible_blocks, slots = _ascend_build_qsa_metadata_torch(
        metadata,
        torch.empty(4, dtype=torch.int32),
        positions,
        visible_blocks,
        torch.empty(4, dtype=torch.int64),
        storage_block_size=16,
        compress_ratio=1,
    )

    assert actual_positions.tolist() == [0, 1, 2, 3]
    assert actual_visible_blocks.tolist() == [1, 2, 3, 4]
    assert slots.tolist() == [0, 1, 2, 3]
