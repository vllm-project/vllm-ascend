# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs

from vllm_ascend.attention.context_parallel.sfa_dcp_utils import get_sfa_pcp_prefill_metadata
from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata, AscendSFAMetadataBuilder


def make_common(query_lens, local_flags, global_flags, local_rows):
    tokens = sum(query_lens)
    num_reqs = len(query_lens)
    offsets = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0)
    return SimpleNamespace(
        pcp_context=SimpleNamespace(
            global_batch=SimpleNamespace(is_prefilling_np=np.array(global_flags)),
            local_to_global_req_indices=local_rows,
        ),
        pcp_cache_group_idx=0,
        context_parallel_metadata=None,
        num_reqs=num_reqs,
        num_actual_tokens=tokens,
        num_input_tokens=max(tokens, 1),
        slot_mapping=torch.arange(8 * max(tokens, 1)),
        positions=torch.zeros(max(tokens, 1), dtype=torch.int64),
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.full((num_reqs,), 1537, dtype=torch.int32),
        seq_lens_cpu=torch.full((num_reqs,), 1537, dtype=torch.int32),
        _seq_lens_cpu=None,
        max_query_len=max(query_lens, default=0),
        max_seq_len=1537,
        causal=True,
        attn_state=None,
        is_prefilling=torch.tensor(local_flags, dtype=torch.bool),
    )


def build_metadata(kind, common, rank=0):
    cls = AscendSFAMetadataBuilder if kind == "sfa" else AscendSFAIndexerMetadataBuilder
    builder = cls.__new__(cls)
    builder.use_pcp = True
    builder.use_dcp = False
    builder.is_pcp_decode_sharded = False
    builder.decode_threshold = 6
    builder.speculative_config = None
    builder.kernel_block_size = 128
    builder.nope = False
    builder.metadata_cls = AscendSFAMetadata
    builder.model_config = SimpleNamespace(get_head_size=lambda: 64)
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *_: None)
    builder._mask_lim_slot_mapping = MagicMock()
    rope = torch.zeros(common.num_input_tokens, 64)
    module = "vllm_ascend.attention.sfa_v1" if kind == "sfa" else "vllm_ascend.attention.indexer"
    with (
        patch(f"{module}.get_cos_and_sin_mla", return_value=(rope, rope)),
        patch(
            "vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=SimpleNamespace(world_size=8, rank_in_group=rank)
        ),
    ):
        return builder.build(0, common)


@pytest.mark.parametrize("kind", ["sfa", "indexer"])
@pytest.mark.parametrize("global_prefill", [False, True])
@pytest.mark.parametrize("rank", range(8))
def test_sfa_lookahead_shards_join_the_same_kv_collectives(kind, global_prefill, rank):
    # MTP5: rank 0 holds the real token, ranks 1..5 hold lookahead,
    # and ranks 6,7 hold an empty shard with one padded input row.
    actual = int(not global_prefill or rank < 6)
    local_prefill = global_prefill and rank not in range(1, 6)
    common = make_common([actual], [local_prefill], [global_prefill], (0,))
    original_flags = common.is_prefilling.clone()
    metadata = build_metadata(kind, common, rank)
    assert metadata.num_decode_tokens == (0 if global_prefill else 1)
    group = SimpleNamespace(world_size=8, all_gather=MagicMock(side_effect=lambda value, dim=0: value.repeat(8, 1)))
    with patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group):
        _gather_prefill_cache_inputs((torch.zeros(1, 4),), common.slot_mapping, metadata.num_decode_tokens)
    assert group.all_gather.call_count == int(global_prefill)
    if kind == "sfa" and global_prefill:
        assert metadata.pcp_prolog_global_slots.numel() == 8
    torch.testing.assert_close(common.is_prefilling, original_flags)


@pytest.mark.parametrize("kind", ["sfa", "indexer"])
def test_sfa_mixed_batch_maps_global_flags_and_preserves_padding(kind):
    common = make_common([1, 1, 1, 0], [False, False, False, False], [True, False], (1, 0, 0))
    metadata = build_metadata(kind, common)
    assert metadata.num_decode_tokens == 1
    view = get_sfa_pcp_prefill_metadata(common, common.pcp_context)
    assert view.is_prefilling.tolist() == [False, True, True, False]
    assert not common.is_prefilling.any()


@pytest.mark.parametrize("context", [None, SimpleNamespace(local_to_global_req_indices=None)])
def test_sfa_without_pcp_request_mapping_keeps_original_metadata(context):
    common = SimpleNamespace(is_prefilling=torch.tensor([False]))
    assert get_sfa_pcp_prefill_metadata(common, context) is common
