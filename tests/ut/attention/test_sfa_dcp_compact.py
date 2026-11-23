# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFAPCPDCPMetadataBuilder,
)
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    _index_select_kv_blocks,
    build_sfa_dcp_compact_kv_metadata,
)
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, AscendSFAIndexerMetadata


@pytest.mark.parametrize("interleave", [128, 768])
@pytest.mark.parametrize("world", [2, 8])
def test_sfa_pcp_dcp_compact_kv_respects_interleave(interleave, world):
    builder = AscendSFAPCPDCPMetadataBuilder.__new__(AscendSFAPCPDCPMetadataBuilder)
    builder.kernel_block_size = 128
    builder.cp_kv_cache_interleave_size = interleave
    builder.arange_buffer = torch.arange(12 * world, dtype=torch.int32)
    builder.dcp_size = world
    order = [0, 1] if world == 2 else [0, 4, 1, 5, 2, 6, 3, 7]
    builder.dcp_collective_rank_order = torch.tensor(order, dtype=torch.int32)
    # Requests share a physical prefix page, but have different history and
    # stale tails. Each PCP receiver can have a different request-row order.
    global_table = torch.tensor(
        [list(range(18, 24)) + list(range(42, 48)) + [999], list(range(18, 24)) + list(range(60, 66)) + [998]],
        dtype=torch.int32,
    )
    local_table = global_table[[1, 0], :12]
    ids, table = builder._build_compact_kv_gather_metadata(
        local_table,
        global_dcp_block_table=global_table,
        global_dcp_num_blocks=torch.tensor([12, 12], dtype=torch.int32),
    )
    assert ids.tolist() == list(range(18, 24)) + list(range(42, 48)) + list(range(60, 66))
    for request in range(2):
        expected = []
        for global_block in range(12 * world):
            position = global_block * 128
            owner = (position // interleave) % world
            local_position = position // (world * interleave) * interleave + position % interleave
            block_id = local_table[request, local_position // 128].item()
            expected.append(order[owner] * len(ids) + ids.tolist().index(block_id))
        assert table[request].tolist() == expected


@pytest.mark.parametrize("interleave", [1, 192])
def test_sfa_compact_history_rejects_interleave_inside_kernel_blocks(interleave):
    with pytest.raises(NotImplementedError, match="multiple of the kernel block size"):
        build_sfa_dcp_compact_kv_metadata(
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
            torch.arange(2, dtype=torch.int32),
            128,
            interleave,
        )


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("interleave", [128, 768])
def test_pcp_dcp_history_view_preserves_causal_topk_and_attention(world, interleave):
    torch.manual_seed(41)
    block_size, local_blocks, width = 128, 12, 8
    length = local_blocks * block_size * world
    order = [0, 1] if world == 2 else [0, 4, 1, 5, 2, 6, 3, 7]
    # Two requests share a prefix page. Payloads differ across KV owners;
    # the receiver's query rows are deliberately not in global request order.
    global_table = torch.tensor([list(range(12)), list(range(6)) + list(range(12, 18))], dtype=torch.int32)
    requests = [1, 0, 1]
    ids, table = build_sfa_dcp_compact_kv_metadata(
        global_table[requests],
        torch.arange(local_blocks * world, dtype=torch.int32),
        torch.tensor(order, dtype=torch.int32),
        block_size,
        interleave,
        global_dcp_block_table=global_table,
        global_dcp_num_blocks=torch.tensor([12, 12], dtype=torch.int32),
    )
    keys = torch.randn(world, 18, block_size, 1, width).to(torch.float8_e4m3fn)
    scales = torch.rand(world, 18, block_size, 1, 1) + 0.25
    # AllGather emits physical ProcessGroup order, not logical PCP-first order.
    gathered_key = torch.cat(
        [_index_select_kv_blocks(keys[order.index(rank)], ids).view(torch.uint8) for rank in range(world)]
    ).view(keys.dtype)
    gathered_scale = torch.cat([scales[order.index(rank)].index_select(0, ids) for rank in range(world)])
    positions = [world * interleave + 17, 127, world * interleave + 20]
    for row, request in enumerate(requests):
        # Build the oracle by assigning each owner's contiguous local token
        # stream to its global positions, independently of compact addresses.
        dense = torch.empty(length, width)
        all_positions = torch.arange(length)
        for owner in range(world):
            mask = all_positions // interleave % world == owner
            local_key = _index_select_kv_blocks(keys[owner], global_table[request]).float().reshape(-1, width)
            local_scale = scales[owner].index_select(0, global_table[request]).reshape(-1, 1)
            dense[mask] = local_key * local_scale
        restored_key = _index_select_kv_blocks(gathered_key, table[row]).float().reshape(-1, width)
        restored_scale = gathered_scale.index_select(0, table[row]).reshape(-1, 1)
        restored = restored_key * restored_scale
        torch.testing.assert_close(restored, dense, rtol=0, atol=0)
        query = torch.randn(width)
        outputs = []
        selections = []
        for history in (dense, restored):
            scores = history @ query
            scores[positions[row] + 1 :] = -torch.inf
            selected = scores.topk(16).indices
            selections.append(selected)
            outputs.append(scores[selected].softmax(-1) @ history[selected])
        torch.testing.assert_close(selections[0], selections[1], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("history_gather", [False, True])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("pcp", [False, True])
def test_pcp_indexer_waits_for_history_before_cache_collectives(history_gather, pending, pcp):
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.pcp_enabled = pcp
    handle = Mock()
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.dcp_context = SimpleNamespace(
        kv_gather_block_ids=torch.arange(2) if history_gather else None,
        gather_context=SimpleNamespace(handle=handle) if pending else None,
    )
    impl._before_indexer(metadata)
    assert handle.wait.call_count == int(pcp and history_gather and pending)


@pytest.mark.parametrize(
    "dtype,local_tokens,compute_topk,shared_hidden,qk_overlap",
    [
        pytest.param(torch.int8, 4, False, False, False, id="cache-write-only"),
        pytest.param(torch.int8, 0, True, False, False, id="empty-receiver"),
        pytest.param(torch.float8_e4m3fn, 4, True, False, False, id="fp8-independent-projections"),
        pytest.param(torch.int8, 4, True, True, True, id="int8-shared-hidden-qk-overlap"),
    ],
)
def test_indexer_history_collectives_follow_writes_and_include_empty_receivers(
    dtype, local_tokens, compute_topk, shared_hidden, qk_overlap
):
    model = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(model)
    caches = (torch.arange(12).reshape(3, 2, 1, 2).to(dtype), torch.ones(3, 2, 1, 1))
    model.k_cache = SimpleNamespace(kv_cache=caches)
    model._sharded_dcp = True
    model.topk_tokens = 1024
    model.enable_sparse_li_c8 = True
    model.enable_sparse_li_c4 = model.use_torch_npu_lightning_indexer = False
    model._can_fuse_k_cache = Mock(return_value=False)
    model._q_stream = object() if qk_overlap else None
    model.head_dim = 2
    model.wq_b = Mock()
    hidden = torch.ones(4, 2)
    k_hidden = hidden if shared_hidden else hidden.clone()
    keys, scales, weights = hidden, torch.ones(4, 1), torch.ones(4, 1)
    model.forward_k = Mock(return_value=(keys, scales, weights))
    slots = torch.arange(4)
    model._gather_cache_inputs = Mock(return_value=(keys, scales, slots))
    query = (torch.ones(4, 1, 2), torch.ones(4, 1), (4, 1, 2))
    events = []

    def project_weights(*args):
        events.append("weights")
        return torch.cat((hidden, weights), dim=-1), None

    def project_q(*args):
        events.append("q")
        return query

    model.wk_weights_proj = Mock(side_effect=project_weights)
    model._forward_q = Mock(side_effect=project_q)
    model._forward_qk = Mock(return_value=(query, keys, scales, weights))

    def write_cache(*args):
        events.append("write")
        caches[0][2].copy_(torch.full_like(caches[0][2], 7))

    model.write_cache = Mock(side_effect=write_cache)
    metadata = AscendSFAIndexerMetadata(
        num_actual_tokens=local_tokens,
        slot_mapping=slots,
        seq_lens=torch.tensor([4]),
        cum_query_lens=torch.tensor([4]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        sin=torch.empty(4, 0),
        cos=torch.empty(4, 0),
        actual_seq_lengths_query=torch.tensor([4]),
        actual_seq_lengths_key=torch.tensor([8]),
        kv_gather_block_ids=torch.tensor([2, 0]),
    )
    handles = []

    def gather(blocks, group):
        assert events[0] == "write"
        events.append("gather")
        expected = torch.cat((blocks, blocks))
        gathered = torch.empty_like(expected)

        def complete():
            events.append("wait")
            gathered.copy_(expected)

        handle = Mock()
        handle.wait.side_effect = complete
        handles.append(handle)
        return gathered, handle

    def select_topk(**kwargs):
        # Payloads become visible only at wait(), as with an asynchronous
        # collective. Checking at consumption catches a missing/late join.
        events.append("topk")
        for source, gathered in zip(caches, (kwargs["key"], kwargs["key_dequant_scale"].unsqueeze(2))):
            expected = torch.cat([_index_select_kv_blocks(source, metadata.kv_gather_block_ids).view(torch.uint8)] * 2)
            assert torch.equal(gathered.view(torch.uint8), expected)
        torch.testing.assert_close(kwargs["weights"], weights.to(kwargs["weights"].dtype))
        torch.testing.assert_close(kwargs["query"], query[0])
        torch.testing.assert_close(kwargs["query_dequant_scale"], query[1])
        return torch.arange(kwargs["sparse_count"], dtype=torch.int32).expand(local_tokens, 1, -1)

    with (
        patch("vllm_ascend.attention.indexer.get_dcp_group", return_value=object()),
        patch("vllm_ascend.attention.indexer.all_gather_async", side_effect=gather) as collective,
        patch(
            "vllm_ascend.device.device_op.torch_npu.npu_quant_lightning_indexer", side_effect=select_topk, create=True
        ) as select,
    ):
        actual = model(hidden, hidden, k_hidden, metadata, compute_topk=compute_topk)
    assert collective.call_count == (2 if compute_topk else 0)
    assert select.call_count == int(compute_topk and local_tokens > 0)
    assert model._forward_q.call_count == int(select.called and not qk_overlap)
    assert model._forward_qk.call_count == int(select.called and qk_overlap)
    assert model.wk_weights_proj.call_count == int(select.called and not shared_hidden)
    for handle in handles:
        handle.wait.assert_called_once()
    if select.called:
        impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
        impl.topk_indices_buffer = torch.full((local_tokens, model.topk_tokens), -1, dtype=torch.int32)
        impl._update_indexcache_topk_indices(actual)
        torch.testing.assert_close(
            impl.topk_indices_buffer, torch.arange(model.topk_tokens, dtype=torch.int32).expand(local_tokens, -1)
        )
    else:
        assert actual is None
    expected_events = ["write"]
    if compute_topk:
        expected_events += ["gather", "gather"]
        if local_tokens > 0:
            if not shared_hidden:
                expected_events.append("weights")
            if not qk_overlap:
                expected_events.append("q")
        expected_events += ["wait", "wait"]
        if local_tokens > 0:
            expected_events.append("topk")
    assert events == expected_events
