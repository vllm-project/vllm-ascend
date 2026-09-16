# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device metadata, ownership transitions and padded graph replay without a model."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.sfa_kv_offload import (
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
)

MODULE = "vllm_ascend.attention.sfa_kv_offload"


def make_builder():
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = True
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        model_config=SimpleNamespace(
            max_model_len=16384, hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        ),
    )
    with patch(
        MODULE + ".get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(topk_buffer_size=8192)),
    ):
        builder._init_nano_metadata_buffers(config, torch.device("npu"))
    return builder


def common(ends, lengths, pools=(1, 0), generations=(11, 12)):
    count = len(lengths)
    return SimpleNamespace(
        query_start_loc=torch.tensor([0, *ends], dtype=torch.int32, device="npu"),
        query_start_loc_cpu=torch.tensor([0, *ends], dtype=torch.int32),
        # Intentionally no CPU sequence-length attribute: lengths must stay on device.
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
        req_topk_buffer_slots=torch.tensor(pools, dtype=torch.int32, device="npu"),
        req_topk_buffer_generations=torch.tensor(generations, dtype=torch.int64, device="npu"),
        block_table_tensor=torch.arange(count * 128, dtype=torch.int32, device="npu").reshape(count, 128),
        req_ids_tensor=None,
        token_to_req=None,
        nano_eligible=True,
        offload_dummy=False,
        max_query_len=4,
        num_reqs=count,
        num_input_tokens=ends[-1],
    )


def populate(builder, cm, draft_index=None):
    metadata = SimpleNamespace()
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(cm.num_reqs, 0, cm.num_input_tokens, 0)):
        if draft_index is None:
            builder._populate_offload_metadata(metadata, cm)
        else:
            with patch(MODULE + ".AscendSFAMetadataBuilder.build_for_drafting", return_value=metadata):
                metadata = builder.build_for_drafting(cm, draft_index=draft_index)
    return metadata


def make_impl():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.nano_states = torch.empty(4, dtype=torch.int32, device="npu")
    impl.nano_last_generation = torch.full((8,), -1, dtype=torch.int64, device="npu")
    impl.nano_last_prefix = torch.zeros(8, dtype=torch.int32, device="npu")
    impl.nano_last_cache = torch.zeros(8, dtype=torch.int32, device="npu")
    return impl


def test_device_lengths_tail_geometry_and_rejection():
    builder = make_builder()
    cm = common([4, 5], [10371, 8321])
    metadata = populate(builder, cm)
    # [S-Q] are 10367 and 8320. Prefix rounds down to complete 128-token blocks.
    assert metadata.nano_prefix_lens.cpu().tolist() == [10240, 8320]
    assert metadata.nano_cache_tokens.cpu().tolist() == [8192, 8192]
    assert metadata.nano_logical_lens.cpu().tolist() == [8323, 8193]
    # Current query KV is scattered locally: H2D must restore only prior KV.
    assert metadata.nano_tail_lengths.cpu().tolist() == [[127, 0], [0, 0]]
    assert metadata.nano_copy_count.item() == 8
    assert metadata.nano_copy_lengths.cpu().tolist() == [127 * 1024, 0, 0, 0, 127 * 128, 0, 0, 0]
    assert metadata.nano_tail_src.cpu().tolist() == [[10240, 10368], [24704, 24832]]
    assert metadata.nano_hbm_block_table[:, 64:66].cpu().tolist() == [[130, 131], [65, 64]]
    stride = 8192 + 256
    assert metadata.nano_device_slots.cpu().tolist() == [stride + 8192 + pos % 256 for pos in range(10367, 10371)] + [
        8192 + 8320 % 256
    ]
    address = metadata.nano_prefix_lens.data_ptr()
    cm.seq_lens.copy_(torch.tensor([10243, 8321], dtype=torch.int32, device="npu"))
    revised = populate(builder, cm)
    assert revised.nano_prefix_lens.data_ptr() == address
    assert revised.nano_prefix_lens.cpu().tolist() == [10112, 8320]


def test_generation_compaction_and_prefix_rollback_reset():
    builder, impl = make_builder(), make_impl()
    cm = common([4, 8], [10371, 8324])
    metadata = populate(builder, cm)
    impl._prepare_nano_lim_state(metadata)
    assert impl.nano_states[:2].cpu().tolist() == [-2, -2]
    impl._prepare_nano_lim_state(metadata)
    assert impl.nano_states[:2].cpu().tolist() == [-1, -1]
    # Swap batch order, keeping request-owned pool and generation together.
    cm = common([4, 8], [8324, 10371], pools=(0, 1), generations=(12, 11))
    metadata = populate(builder, cm)
    impl._prepare_nano_lim_state(metadata)
    assert impl.nano_states[:2].cpu().tolist() == [-1, -1]
    # New generation and rollback independently force cold fill.
    cm.req_topk_buffer_generations[0] = 13
    cm.seq_lens[1] = 10243
    impl._prepare_nano_lim_state(populate(builder, cm))
    assert impl.nano_states[:2].cpu().tolist() == [-2, -2]


def test_inactive_capture_becomes_active_on_graph_replay():
    builder, impl = make_builder(), make_impl()
    cm = common([4, 8], [0, 0], pools=(0, 0), generations=(-1, -1))
    metadata = populate(builder, cm)
    # Private pools 4 and 5; positive cache budgets avoid copy-SFA's cold-fill predicate.
    assert metadata.nano_pool_entries.cpu().tolist() == [4, 5]
    assert metadata.nano_cache_tokens.cpu().tolist() == [2048, 2048]
    assert metadata.nano_tail_lengths.count_nonzero().item() == 0
    assert metadata.nano_device_slots.min().item() >= 4 * (8192 + 256)
    for _ in range(3):
        impl._prepare_nano_lim_state(metadata)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl._prepare_nano_lim_state(metadata)
    assert impl.nano_states[:2].cpu().tolist() == [-3, -3]
    cm.seq_lens.copy_(torch.tensor([10371, 0], dtype=torch.int32, device="npu"))
    cm.req_topk_buffer_generations[0] = 11
    cm.req_topk_buffer_slots[0] = 1
    populate(builder, cm)
    graph.replay()
    assert impl.nano_states[:2].cpu().tolist() == [-2, -3]
    graph.replay()
    assert impl.nano_states[:2].cpu().tolist() == [-1, -3]
    # Dummy execution must not change real request 1's residency.
    cm.req_topk_buffer_generations.fill_(-1)
    populate(builder, cm)
    graph.replay()
    assert impl.nano_states[:2].cpu().tolist() == [-3, -3]
    assert impl.nano_last_generation[1].item() == 11


def test_eager_sp_padding_uses_private_tail_and_exact_query_count():
    builder = make_builder()
    cm = common([4, 5], [10371, 8321])
    cm.num_input_tokens = 8  # padded for TP8, only five actual query rows
    metadata = populate(builder, cm)
    assert metadata.num_decode_tokens == 5
    assert metadata.nano_token_active.cpu().tolist() == [True] * 5 + [False] * 3
    private_start = builder.nano_pool_capacity * (8192 + 256)
    assert metadata.nano_device_slots[5:].min().item() >= private_start
    assert metadata.nano_device_slots[:5].max().item() < private_start


def test_runner_pool_ownership_survives_compaction_and_dummy_run():
    import numpy as np

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.max_num_reqs = 2
    runner._offload_pool_slots = SimpleNamespace(np=np.zeros(4, dtype=np.int32), copy_to_gpu=lambda n: None)
    runner._offload_pool_generations = SimpleNamespace(np=np.zeros(4, dtype=np.int64), copy_to_gpu=lambda n: None)
    runner._offload_request_slots = {}
    runner._offload_slot_generation = 0
    runner._offload_slot_generations = {}
    runner.input_batch = SimpleNamespace(req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1})
    runner._prepare_nano_request_slots(2, 3, dummy=False)
    assert runner._offload_pool_slots.np[:3].tolist() == [0, 1, 6]
    assert runner._offload_pool_generations.np[:3].tolist() == [1, 2, -1]
    # Removing a compacts b; new c may reuse a's slot, with a new generation.
    runner.input_batch = SimpleNamespace(req_ids=["b", "c"], req_id_to_index={"b": 0, "c": 1})
    runner._prepare_nano_request_slots(2, 3, dummy=False)
    assert runner._offload_pool_slots.np[:3].tolist() == [1, 0, 6]
    assert runner._offload_pool_generations.np[:3].tolist() == [2, 3, -1]
    runner._prepare_nano_request_slots(2, 3, dummy=True)
    assert runner._offload_pool_slots.np[:3].tolist() == [4, 5, 6]
    assert runner._offload_pool_generations.np[:3].tolist() == [-1, -1, -1]
    assert runner._offload_request_slots == {"b": 1, "c": 0}
    runner._prepare_nano_request_slots(2, 3, dummy=False)
    assert runner._offload_pool_generations.np[:3].tolist() == [2, 3, -1]


def test_draft_metadata_remains_valid_until_its_step_executes():
    builder = make_builder()
    first = populate(builder, common([4, 8], [10371, 8324]))
    saved = {
        name: value.clone()
        for name, value in vars(first).items()
        if name.startswith("nano_") and isinstance(value, torch.Tensor)
    }
    # The proposer builds both subsequent Q1 steps before executing the Q4
    # first step. Neither query prefixes nor tail/copy geometry may alias.
    second = populate(builder, common([1, 2], [10372, 8325]), draft_index=1)
    third = populate(builder, common([1, 2], [10373, 8326]), draft_index=2)
    torch.npu.synchronize()
    for name, expected in saved.items():
        torch.testing.assert_close(getattr(first, name), expected)
        addresses = {getattr(md, name).data_ptr() for md in (first, second, third)}
        assert len(addresses) == 3, name
    assert first.nano_query_ends.cpu().tolist() == [4, 8]
    assert second.nano_seq_lens.cpu().tolist() == [10372, 8325]
    assert third.nano_seq_lens.cpu().tolist() == [10373, 8326]

    # A captured consumer must continue reading its own stable step address.
    observed = torch.empty_like(first.nano_query_ends)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        observed.copy_(first.nano_query_ends)
    for ends, lengths in (([4, 5], [10374, 8327]), ([4, 8], [10375, 8328])):
        populate(builder, common(ends, lengths))
        populate(builder, common([1, 2], [10376, 8329]), draft_index=1)
        populate(builder, common([1, 2], [10377, 8330]), draft_index=2)
        graph.replay()
        assert observed.cpu().tolist() == ends
