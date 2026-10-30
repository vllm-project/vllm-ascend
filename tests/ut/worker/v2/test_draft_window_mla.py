# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real MLA metadata and the actual persistent RoPE producer."""

import pytest
import torch

from vllm_ascend.attention.mla_v1 import AscendMLADecodeMetadata, AscendMLAMetadata
from vllm_ascend.ops import rotary_embedding
from vllm_ascend.worker.v2.spec_decode.draft_window_mla import freeze_mla_read_metadata


def metadata_for_decode(positions, cos, sin):
    table = torch.tensor([[10, 11]], dtype=torch.int32)
    lengths = torch.tensor([129], dtype=torch.int32)
    return AscendMLAMetadata(
        num_actual_tokens=positions.numel(),
        slot_mapping=torch.tensor([1552, 1553]),
        query_start_loc=torch.tensor([0, positions.numel()], dtype=torch.int32),
        seq_lens=lengths,
        seq_lens_cpu=lengths,
        block_tables=table,
        num_decodes=1,
        num_decode_tokens=positions.numel(),
        num_prefills=0,
        decode=AscendMLADecodeMetadata(
            input_positions=positions,
            block_table=table,
            seq_lens=lengths,
            max_seq_lens=129,
            seq_lens_list=[129],
            cos=cos,
            sin=sin,
        ),
    )


@pytest.mark.parametrize("owned_submission", [False, True])
def test_later_real_rope_build_cannot_overwrite_inflight_window_metadata(monkeypatch, owned_submission):
    cos_cache = torch.arange(64, dtype=torch.float32).view(16, 4)
    sin_cache = -cos_cache
    for name, value in {
        "_cos_cache": cos_cache,
        "_sin_cache": sin_cache,
        "_cos_mla": torch.zeros(4, 1, 1, 4),
        "_sin_mla": torch.zeros(4, 1, 1, 4),
    }.items():
        monkeypatch.setattr(rotary_embedding, name, value)
    positions = torch.tensor([9, 10])
    cos, sin = rotary_embedding.get_cos_and_sin_mla(positions, use_cache=True)
    expected_cos, expected_sin = cos.clone(), sin.clone()
    metadata = metadata_for_decode(positions, cos, sin)
    full_slots, query, table, decode = (
        metadata.slot_mapping,
        metadata.query_start_loc,
        metadata.block_tables,
        metadata.decode,
    )
    layers = {"draft.a": metadata, "draft.b": metadata}
    submitted = []
    retain = lambda *objects: submitted.extend(objects)
    retained = freeze_mla_read_metadata(layers, tuple(layers), retain=retain if owned_submission else None)
    if owned_submission:
        assert any(value is cos for value in submitted)
        assert any(value is sin for value in submitted)
        assert all(any(value is output for value in submitted) for output in retained)
    rotary_embedding.get_cos_and_sin_mla(torch.tensor([2, 3]), use_cache=True)
    # The original producer is demonstrably mutable; the Window read isn't.
    assert not torch.equal(cos, expected_cos)
    assert torch.equal(metadata.decode.cos, expected_cos)
    assert torch.equal(metadata.decode.sin, expected_sin)
    assert len(retained) == 2
    assert metadata.slot_mapping is full_slots
    assert metadata.query_start_loc is query
    assert metadata.block_tables is table
    assert metadata.decode is decode
    assert metadata.decode.input_positions is positions
    assert layers["draft.a"] is layers["draft.b"]


def test_nope_and_prefill_do_not_copy_full_query_or_metadata():
    metadata = metadata_for_decode(torch.tensor([9, 10]), None, None)
    assert freeze_mla_read_metadata({"draft": metadata}, ["draft"]) == ()
    metadata.decode = None
    assert freeze_mla_read_metadata({"draft": metadata}, ["draft"]) == ()


def test_flashmla_is_not_silently_treated_as_fia():
    metadata = metadata_for_decode(torch.tensor([9, 10]), None, None)
    metadata.external_flashmla = object()
    with pytest.raises(NotImplementedError, match="FlashMLA"):
        freeze_mla_read_metadata({"draft": metadata}, ["draft"])
