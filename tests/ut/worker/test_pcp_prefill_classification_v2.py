# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs

from vllm_ascend.attention.utils import split_decodes_and_prefills
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def finalize_batch(global_batch, local_batch, *, is_decode_sharded=False, hf_text_config=None):
    manager = AscendPCPManager.__new__(AscendPCPManager)
    if hf_text_config is None:
        hf_text_config = SimpleNamespace(index_topk=2048)
    manager.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf_text_config))
    manager.kv_cache_config = None
    manager._input_buffers = SimpleNamespace(seq_lens_np=np.zeros(local_batch.num_reqs_after_padding, dtype=np.int32))
    with (
        patch.object(AscendPCPManager, "is_decode_sharded", new_callable=PropertyMock, return_value=is_decode_sharded),
        patch.object(manager, "_pad_hidden_restore_idx"),
        patch.object(manager, "_pad_replicated_decode_graph", return_value=local_batch),
        patch("vllm_ascend.worker.v2.pcp_manager.build_attn_state"),
    ):
        return manager._finalize_partition(global_batch, local_batch, None)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("is_global_prefill", [True, False])
@pytest.mark.parametrize("is_decode_sharded", [True, False])
def test_short_prefill_lookahead_keeps_pcp_collective_participation(rank, is_global_prefill, is_decode_sharded):
    # One real prompt token plus MTP5 splits into six one-token shards.
    # Ranks 1..5 begin beyond the prompt; ranks 6,7 carry an empty placeholder.
    num_tokens = int(not is_global_prefill or rank < 6)
    local_start = 1536 + (rank if is_global_prefill and rank < 6 else 0)
    local_prefill = is_global_prefill and local_start < 1537
    global_batch = SimpleNamespace(req_ids=["request"], is_prefilling_np=np.array([is_global_prefill]))
    local_batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        num_reqs_after_padding=2,
        num_tokens=num_tokens,
        num_computed_tokens_np=np.array([local_start]),
        num_scheduled_tokens=np.array([num_tokens]),
        seq_lens_cpu_upper_bound=torch.zeros(1, dtype=torch.int32),
        is_prefilling_np=np.array([local_prefill, False]),
        has_prefill=local_prefill,
        decode_graph_eligible=not local_prefill,
    )
    result = finalize_batch(global_batch, local_batch, is_decode_sharded=is_decode_sharded)
    common = SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=num_tokens,
        num_reqs=1,
        num_actual_tokens=num_tokens,
        query_start_loc_cpu=torch.tensor([0, num_tokens], dtype=torch.int32),
        is_prefilling=torch.from_numpy(result.is_prefilling_np),
    )
    with patch("vllm_ascend.attention.utils.is_pd_decode_recompute_scheduler_enabled", return_value=False):
        _, _, num_decode_tokens, _ = split_decodes_and_prefills(
            common, decode_threshold=6, treat_short_extends_as_decodes=False
        )
    group = SimpleNamespace(
        world_size=8,
        all_gather=MagicMock(side_effect=lambda value, dim=0: torch.cat([value] * 8, dim=dim)),
    )
    with patch("vllm.v1.attention.ops.pcp.get_pcp_group", return_value=group):
        _gather_prefill_cache_inputs((torch.zeros(1, 4),), torch.arange(8), num_decode_tokens)

    assert group.all_gather.call_count == int(is_global_prefill)
    assert result.has_prefill == is_global_prefill
    assert result.decode_graph_eligible != is_global_prefill
    assert not result.is_prefilling_np[1]
    assert global_batch.is_prefilling_np.tolist() == [is_global_prefill]


def test_mixed_pcp_batch_restores_flags_by_request_identity():
    global_batch = SimpleNamespace(req_ids=["decode", "prefill"], is_prefilling_np=np.array([False, True]))
    local_batch = SimpleNamespace(
        req_ids=["decode", "prefill", "prefill"],
        num_reqs=3,
        num_reqs_after_padding=4,
        num_tokens=3,
        num_computed_tokens_np=np.array([2000, 1537, 1536]),
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.zeros(3, dtype=torch.int32),
        is_prefilling_np=np.array([False, False, True, False]),
        has_prefill=True,
        decode_graph_eligible=False,
    )
    result = finalize_batch(global_batch, local_batch)
    assert result.is_prefilling_np.tolist() == [False, True, True, False]
    assert result.has_prefill
    assert not result.decode_graph_eligible


@pytest.mark.parametrize("has_graph_eligibility", [True, False])
@pytest.mark.parametrize("is_local_prefill", [True, False])
def test_mixed_batch_shard_preserves_graph_restrictions(has_graph_eligibility, is_local_prefill):
    global_batch = SimpleNamespace(req_ids=["decode", "prefill"], is_prefilling_np=np.array([False, True]))
    local_batch = SimpleNamespace(
        req_ids=["prefill" if is_local_prefill else "decode"],
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=1,
        num_computed_tokens_np=np.array([2000]),
        num_scheduled_tokens=np.ones(1, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.zeros(1, dtype=torch.int32),
        is_prefilling_np=np.array([False]),
        has_prefill=False,
    )
    if has_graph_eligibility:
        local_batch.decode_graph_eligible = False

    result = finalize_batch(global_batch, local_batch)

    assert result.has_prefill == is_local_prefill
    assert result.is_prefilling_np.tolist() == [is_local_prefill]
    if has_graph_eligibility:
        assert not result.decode_graph_eligible
    else:
        assert not hasattr(result, "decode_graph_eligible")


@pytest.mark.parametrize(
    "hf_text_config,expected_prefill",
    [
        (SimpleNamespace(), False),
        (SimpleNamespace(index_topk=2048, compress_ratios=[4]), False),
        (SimpleNamespace(index_topk=2048), True),
        (SimpleNamespace(index_topk=2048, index_kpool=4), True),
    ],
    ids=["dense", "compressed-dsa", "sfa", "glm-kpool-sfa"],
)
def test_prefill_restoration_is_restricted_to_sfa(hf_text_config, expected_prefill):
    global_batch = SimpleNamespace(req_ids=["request"], is_prefilling_np=np.array([True]))
    local_batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=1,
        num_computed_tokens_np=np.array([1537]),
        num_scheduled_tokens=np.ones(1, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.zeros(1, dtype=torch.int32),
        is_prefilling_np=np.array([False]),
        has_prefill=False,
        decode_graph_eligible=True,
    )

    result = finalize_batch(global_batch, local_batch, hf_text_config=hf_text_config)

    assert result.is_prefilling_np.tolist() == [expected_prefill]
    assert result.has_prefill == expected_prefill
    assert result.decode_graph_eligible != expected_prefill
