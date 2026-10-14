# SPDX-License-Identifier: Apache-2.0
"""A5 graph replay of metadata with changing requests and persistent tables.

This exercises device graph semantics without loading model weights. Full
MTP/P-D output and acceptance validation is a separate model-level check.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import AscendMultiKVCacheGroupMTPProposer


@pytest.mark.skipif("950" not in torch.npu.get_device_name(0), reason="A5 draft graph metadata")
@torch.inference_mode()
def test_mtp_graph_metadata_replays_dynamic_active_count_and_positions():
    proposer = AscendMultiKVCacheGroupMTPProposer.__new__(AscendMultiKVCacheGroupMTPProposer)
    proposer.arange = torch.arange(3, device="npu")
    proposer.token_arange_np = np.arange(3, dtype=np.int32)
    proposer._mtp_num_actual_reqs = torch.zeros((), dtype=torch.int32, device="npu")
    proposer._mtp_graph_common_metadata = SimpleNamespace()
    proposer.block_size = 128
    proposer.max_model_len = 1024
    proposer.seq_lens_group = [torch.zeros(2, dtype=torch.int32, device="npu") for _ in range(5)]
    proposer.query_start_loc_group = [torch.zeros(3, dtype=torch.int32, device="npu") for _ in range(5)]
    proposer.slot_mapping_group = [torch.full((2,), -1, dtype=torch.int32, device="npu") for _ in range(5)]
    table = torch.zeros(2, 4, dtype=torch.int32, device="npu")
    positions = torch.zeros(2, dtype=torch.int32, device="npu")
    builder = SimpleNamespace(build_for_drafting=lambda common, step: common)
    group = SimpleNamespace(kv_cache_group_id=0, layer_names=["main"], get_metadata_builder=lambda: builder)
    proposer.draft_attn_groups = [group]
    proposer._get_primary_draft_attn_group = lambda: group
    proposer._common_attn_metadata_for_draft_group = lambda common, *args: common
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table=[SimpleNamespace(get_device_tensor=lambda: table)])
    )

    def prepare():
        return [
            proposer._build_in_graph_draft_metadata(step, 2, positions + step, None)["main"] for step in range(1, 5)
        ]

    prepare()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = prepare()
    for actual, base_position, page_bias in [(2, 100, 2), (0, 200, 7), (1, 105, 3), (2, 201, 4)]:
        host_table = torch.arange(8, dtype=torch.int32).view(2, 4) + page_bias
        table.copy_(host_table)
        positions.copy_(torch.tensor([base_position, base_position + 30], dtype=torch.int32))
        proposer._mtp_num_actual_reqs.fill_(actual)
        graph.replay()
        torch.npu.synchronize()
        for step, metadata in enumerate(output, start=1):
            pos = torch.tensor([base_position + step, base_position + 30 + step])
            expected_lengths = torch.where(torch.arange(2) < actual, pos + 1, 0).int()
            expected_slots = host_table.gather(1, (pos // 128).view(-1, 1)).view(-1) * 128 + pos % 128
            expected_slots[actual:] = -1
            torch.testing.assert_close(metadata.seq_lens.cpu(), expected_lengths)
            torch.testing.assert_close(metadata.slot_mapping.cpu(), expected_slots.int())
            assert metadata.query_start_loc.cpu().tolist() == [0, min(actual, 1), actual]
    assert len({metadata.seq_lens.data_ptr() for metadata in output}) == 4
