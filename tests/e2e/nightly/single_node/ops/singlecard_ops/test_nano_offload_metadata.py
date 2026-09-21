# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device metadata, ownership transitions and padded graph replay without a model."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl, AscendSFAKVOffloadMetadataBuilder

MODULE = "vllm_ascend.attention.sfa_kv_offload"


def make_builder(num_mtp_layers=1):
    builder = AscendSFAKVOffloadMetadataBuilder.__new__(AscendSFAKVOffloadMetadataBuilder)
    builder.use_nano = True
    builder.decode_threshold = 4
    builder.is_pd_decode_consumer = True
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_nextn_predict_layers=num_mtp_layers, index_share_for_mtp_iteration=True)
            ),
        ),
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
        # Deliberately stale mirrors must never affect nano geometry.
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
        seq_lens_cpu=torch.full((count,), 777777, dtype=torch.int32),
        _seq_lens_cpu=torch.full((count,), 777777, dtype=torch.int32),
        nano_draft_index=None,
        # Most geometry tests exercise the optional restore descriptors too.
        nano_restore_tails=True,
        req_topk_buffer_slots=torch.tensor(pools, dtype=torch.int32),
        req_topk_buffer_generations=torch.tensor(generations, dtype=torch.int64),
        block_table_tensor=torch.arange(count * 128, dtype=torch.int32, device="npu").reshape(count, 128),
        slot_mapping=torch.arange(16, dtype=torch.int64, device="npu") + 128,
        req_ids_tensor=None,
        token_to_req=None,
        offload_dummy=False,
        max_query_len=4,
        num_reqs=count,
        num_input_tokens=ends[-1],
    )


def populate(builder, cm, draft_index=None, *, reuse_topk=False):
    cm.nano_draft_index = draft_index
    builder.nano_reuse_topk = reuse_topk
    metadata = SimpleNamespace(slot_mapping=cm.slot_mapping[: cm.num_input_tokens])
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(cm.num_reqs, 0, cm.num_input_tokens, 0)):
        if draft_index in (None, 0):
            # The proposer uses ordinary build() for draft step 0 too.
            with patch(MODULE + ".AscendSFAMetadataBuilder.build", return_value=metadata):
                metadata = builder.build(0, cm)
        else:
            with patch(MODULE + ".AscendSFAMetadataBuilder.build_for_drafting", return_value=metadata):
                metadata = builder.build_for_drafting(cm, draft_index=draft_index)
    return metadata


def test_prefill_batches_carry_pool_slots():
    builder = make_builder()
    # Prefill batch (num_prefills > 0): nano_enabled is False, but the row
    # slots must still ride on the metadata for the exec_kv prefill-end D2D.
    cm = common([4], [10371])
    metadata = SimpleNamespace()
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(0, cm.num_reqs, cm.num_input_tokens, 0)):
        builder._populate_offload_metadata(metadata, cm)
    assert metadata.nano_enabled is False
    assert metadata.nano_prefill_pool_slots.cpu().tolist() == [1]

    # Decode batch: the per-step tail restore is always skipped now (initial
    # KV arrives via the PD D2D or the colocate prefill-end D2D).
    colocate_builder = make_builder()
    colocate_builder.is_pd_decode_consumer = False
    metadata = populate(colocate_builder, common([4], [10371]))
    assert metadata.nano_enabled is True
    assert metadata.nano_prefill_pool_slots.cpu().tolist() == [1]


def test_device_lengths_tail_geometry_and_rejection():
    builder = make_builder()
    cm = common([4, 5], [10371, 8321])
    metadata = populate(builder, cm)
    # [S-Q] are 10367 and 8320. Prefix rounds down to complete 128-token blocks.
    assert metadata.nano_prefix_lens.cpu().tolist() == [10240, 8320]
    assert metadata.nano_cache_tokens.cpu().tolist() == [8192, 8192]
    assert metadata.nano_logical_lens.cpu().tolist() == [8323, 8193]
    # Current query KV is scattered locally: descriptors still describe prior KV
    # so prefix rollback can eager-restore. PD decode skips the graph H2D.
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
    builder = make_builder()
    cm = common([4, 8], [10371, 8324])
    metadata = populate(builder, cm)
    assert metadata.nano_request_state.cpu().tolist() == [-2, -2]
    assert populate(builder, cm).nano_request_state.cpu().tolist() == [-1, -1]
    # Swap batch order, keeping request-owned pool and generation together.
    cm = common([4, 8], [8324, 10371], pools=(0, 1), generations=(12, 11))
    metadata = populate(builder, cm)
    assert metadata.nano_request_state.cpu().tolist() == [-1, -1]
    # New generation and rollback independently force cold fill.
    cm.req_topk_buffer_generations[0] = 13
    cm.seq_lens[1] = 10243
    assert populate(builder, cm).nano_request_state.cpu().tolist() == [-2, -2]


def test_builder_histories_are_independent():
    target_builder = make_builder()
    draft_builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    first = populate(target_builder, cm)
    draft = populate(draft_builder, cm, draft_index=0)
    assert first.nano_request_state.cpu().tolist() == [-2]
    assert draft.nano_request_state.cpu().tolist() == [-2]
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(0, 1, 0, 4)):
        target_builder._populate_offload_metadata(SimpleNamespace(), common([4], [10371]))
    assert populate(target_builder, cm).nano_request_state.cpu().tolist() == [-2]
    assert populate(draft_builder, cm, draft_index=0).nano_request_state.cpu().tolist() == [-1]


def test_target_and_draft_history_are_separate_and_reuse_does_not_advance_it():
    builder = make_builder()
    target = populate(builder, common([4], [10371], pools=(1,), generations=(11,)))
    draft = populate(builder, common([4], [10371], pools=(1,), generations=(11,)), draft_index=0)
    assert target.nano_request_state.cpu().tolist() == [-2]
    assert draft.nano_request_state.cpu().tolist() == [-2]
    assert target.nano_request_state.data_ptr() != draft.nano_request_state.data_ptr()
    # Later Q1 steps cross a block boundary, but skip LIM entirely.
    populate(builder, common([1], [10380], pools=(1,), generations=(11,)), draft_index=1, reuse_topk=True)
    populate(builder, common([1], [10381], pools=(1,), generations=(11,)), draft_index=2, reuse_topk=True)
    assert builder.nano_last_prefix[1, 1].item() == 10240
    assert draft.nano_request_state.cpu().tolist() == [-2]
    next_draft = populate(builder, common([4], [10371], pools=(1,), generations=(11,)), draft_index=0)
    assert next_draft.nano_request_state.cpu().tolist() == [-1]


def test_fallback_resets_only_the_affected_model_history():
    builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    populate(builder, cm)
    populate(builder, cm, draft_index=0)
    cm.nano_draft_index = None
    with patch(MODULE + ".split_decodes_and_prefills", return_value=(0, 1, 0, 4)):
        builder._populate_offload_metadata(SimpleNamespace(), cm)
    assert populate(builder, cm).nano_request_state.cpu().tolist() == [-2]
    assert populate(builder, cm, draft_index=0).nano_request_state.cpu().tolist() == [-1]


def test_physical_mtp_layers_have_independent_history():
    builder = make_builder(num_mtp_layers=2)
    for step, expected in ((0, -2), (1, -2), (2, -1)):
        cm = common([1], [10371], pools=(1,), generations=(11,))
        md = populate(builder, cm, draft_index=step, reuse_topk=False)
        assert md.nano_request_state.cpu().tolist() == [expected]


@pytest.mark.parametrize("draft_index", [None, 0, 1, 2], ids=["target", "draft0", "draft1", "draft2"])
def test_lim_consumes_shared_state_without_modifying_it(draft_index):
    builder = make_builder()
    cm = common([4, 8, 12], [10371, 500, 0], pools=(1, 0, 0), generations=(11, 12, -1))
    populate(builder, cm, draft_index=draft_index)
    metadata = populate(builder, cm, draft_index=draft_index)
    assert metadata.nano_request_state.cpu().tolist() == [-1, -3, -3]
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl._nano_metadata = metadata
    impl.nano_slot_map = torch.empty((8, 16384), dtype=torch.int32, device="npu")
    for name, shape in (
        ("nano_topk_src", (12, 1, 2048)),
        ("nano_topk_dst", (12, 1, 2048)),
        ("nano_topk_misses", (12,)),
        ("nano_miss_src", (3, 32768)),
        ("nano_miss_dst", (3, 32768)),
        ("nano_misses", (3,)),
        ("nano_reuse_logical_lens", (3,)),
        ("nano_reuse_cache_tokens", (3,)),
    ):
        setattr(impl, name, torch.full(shape, -999, dtype=torch.int32, device="npu"))
    impl.nano_reuse_request_count = 0
    impl.nano_key_scale = None
    indexer = SimpleNamespace(
        k_cache=SimpleNamespace(kv_cache=[torch.empty((1, 128, 1, 128), dtype=torch.bfloat16, device="npu")])
    )
    query = torch.empty((12, 32, 128), dtype=torch.bfloat16, device="npu")
    weights = torch.empty((12, 32), dtype=torch.bfloat16, device="npu")
    with patch(MODULE + ".torch.ops._C_ascend.npu_fused_li_manage_mtp") as lim:
        impl._nano_select(query, weights, indexer, SimpleNamespace(block_table=cm.block_table_tensor))
        assert lim.call_args.args[10].data_ptr() == metadata.nano_request_state.data_ptr()
        assert metadata.nano_request_state.cpu().tolist() == [-1, -3, -3]
    bank = 0 if draft_index is None else 1
    assert builder.nano_last_generation[bank, 1].item() == 11
    if draft_index == 0:
        assert impl.nano_reuse_request_count == 3
        assert impl.nano_reuse_logical_lens.cpu().tolist() == [8192, 500, 0]
        assert impl.nano_reuse_cache_tokens.cpu().tolist() == [8192, 0, 2048]
    else:
        assert impl.nano_reuse_request_count == 0
        assert impl.nano_reuse_logical_lens.cpu().tolist() == [-999] * 3
        assert impl.nano_reuse_cache_tokens.cpu().tolist() == [-999] * 3


def test_reuse_extent_is_prepared_once_per_draft_round():
    builder = make_builder()
    storage = builder.nano_vectors["reuse_logical_lens"]
    storage.fill_(-999)
    cm = common([4, 8, 12], [500, 10371, 0], pools=(1, 0, 0), generations=(11, 12, -1))
    target = populate(builder, cm)
    assert target.nano_reuse_logical_lens is None
    assert (storage == -999).all().item()

    first = populate(builder, cm, draft_index=0)
    assert first.nano_reuse_logical_lens.cpu().tolist() == [500, 8192, 0]
    address = first.nano_reuse_logical_lens.data_ptr()
    saved = storage.clone()
    for step in (1, 2):
        later = populate(
            builder,
            common([1, 2, 3], [500 + step, 10371 + step, 0], pools=(1, 0, 0), generations=(11, 12, -1)),
            draft_index=step,
            reuse_topk=True,
        )
        assert later.nano_reuse_logical_lens is None
        torch.testing.assert_close(storage, saved)

    # A new round refreshes step 0 in place, including changed activity.
    next_cm = common([4, 8, 12], [504, 10375, 0], pools=(1, 0, 0), generations=(11, 12, -1))
    assert populate(builder, next_cm).nano_reuse_logical_lens is None
    torch.testing.assert_close(storage, saved)
    next_cm.req_topk_buffer_generations[1] = -1
    next_first = populate(builder, next_cm, draft_index=0)
    assert next_first.nano_reuse_logical_lens.data_ptr() == address
    assert next_first.nano_reuse_logical_lens.cpu().tolist() == [504, 0, 0]
    assert (storage[0] == -999).all().item()
    assert (storage[2:] == -999).all().item()


def test_short_row_dense_geometry_in_mixed_batch():
    builder = make_builder()
    # Row 0: short (aligned prefix 4992 < hot 8192), row 1: long.
    cm = common([4, 8], [5000, 10371])
    metadata = populate(builder, cm)
    assert metadata.nano_prefix_lens.cpu().tolist() == [4992, 10240]
    # Short rows run copy-SFA's dense mode (C == 0) and see the whole
    # sequence; long rows keep the full hot budget.
    assert metadata.nano_cache_tokens.cpu().tolist() == [0, 8192]
    assert metadata.nano_logical_lens.cpu().tolist() == [5000, 8323]
    assert metadata.nano_reuse_logical_lens is None
    # Short row: identity block table for every stride block, zero tail, and
    # front-to-back device slots (token p -> row slot p).
    stride_blocks = 8192 // 128 + 2
    assert metadata.nano_hbm_block_table[0].cpu().tolist() == [1 * stride_blocks + b for b in range(stride_blocks)]
    assert metadata.nano_tail_lengths[0].cpu().tolist() == [0, 0]
    assert metadata.nano_device_slots[:4].cpu().tolist() == [1 * (8192 + 256) + 4996 + p for p in range(4)]
    # Long row keeps the ring-tail geometry.
    long_ring = metadata.nano_hbm_block_table[1, 64:66].cpu().tolist()
    assert long_ring == [
        stride_blocks - 2 + (10240 // 128 + 64 - 64) % 2,
        stride_blocks - 2 + (10240 // 128 + 65 - 64) % 2,
    ]
    assert metadata.nano_tail_lengths[1].cpu().tolist() == [127, 0]
    assert metadata.nano_device_slots[4:].cpu().tolist() == [8192 + (10367 + p) % 256 for p in range(4)]


def test_short_lifecycle_minus3_minus2_minus1():
    builder = make_builder()
    # Short rows always run -3 (dense, C == 0): the row content is provided
    # by the PD dense D2D / the eager full-row fill, not by a -2 rebuild.
    cm = common([4], [5000], pools=(1,), generations=(11,))
    for _ in range(3):
        assert populate(builder, cm).nano_request_state.cpu().tolist() == [-3]
    # Growth past the hot budget: the cache flip (0 -> hot) forces the -2
    # offload init, then -1 steady state.
    cm.seq_lens[0] = 8196
    assert populate(builder, cm).nano_request_state.cpu().tolist() == [-2]
    assert populate(builder, cm).nano_request_state.cpu().tolist() == [-1]
    # Rollback below the budget: directly back to -3; the runner separately
    # refills the row because the sparse layout is unreadable in dense mode.
    cm.seq_lens[0] = 5000
    assert populate(builder, cm).nano_request_state.cpu().tolist() == [-3]


def test_tiny_row_stays_minus3_without_legal_init():
    builder = make_builder()
    # L < 2048 has no legal -2 (operator contract), so the row must stay -3;
    # in PD the read thread has already populated the dense content.
    cm = common([4], [500], pools=(1,), generations=(11,))
    for _ in range(3):
        assert populate(builder, cm).nano_request_state.cpu().tolist() == [-3]


@pytest.mark.parametrize("draft_index", [None, 0], ids=["target", "draft0"])
def test_inactive_capture_becomes_active_on_graph_replay(draft_index):
    builder = make_builder()
    cm = common([4, 8], [0, 0], pools=(0, 0), generations=(-1, -1))
    metadata = populate(builder, cm, draft_index=draft_index)
    # Private pools 4 and 5; positive cache budgets avoid copy-SFA's cold-fill predicate.
    assert metadata.nano_pool_entries.cpu().tolist() == [4, 5]
    assert metadata.nano_cache_tokens.cpu().tolist() == [2048, 2048]
    assert metadata.nano_logical_lens.cpu().tolist() == [0, 0]
    if draft_index == 0:
        assert metadata.nano_reuse_logical_lens.cpu().tolist() == [0, 0]
    else:
        assert metadata.nano_reuse_logical_lens is None
    assert metadata.nano_tail_lengths.count_nonzero().item() == 0
    assert metadata.nano_device_slots.min().item() >= 4 * (8192 + 256)
    # Model capture only consumes the persistent state address. Metadata
    # preparation updates its contents before every replay, including padding.
    observed = torch.empty((3, 2), dtype=torch.int32, device="npu")
    observed_slots = torch.empty_like(metadata.slot_mapping)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        observed[0].copy_(metadata.nano_request_state)
        observed[1].copy_(metadata.nano_logical_lens)
        if metadata.nano_reuse_logical_lens is not None:
            observed[2].copy_(metadata.nano_reuse_logical_lens)
        else:
            observed[2].zero_()
        observed_slots.copy_(metadata.slot_mapping)
    graph.replay()
    assert observed.cpu().tolist() == [[-3, -3], [0, 0], [0, 0]]
    assert observed_slots.cpu().tolist() == [-1] * 8
    cm.seq_lens.copy_(torch.tensor([10371, 0], dtype=torch.int32, device="npu"))
    cm.req_topk_buffer_generations[0] = 11
    cm.req_topk_buffer_slots[0] = 1
    # Real scheduling regenerates mappings before metadata preparation.
    cm.slot_mapping.copy_(torch.arange(16, dtype=torch.int64, device="npu") + 256)
    populate(builder, cm, draft_index=draft_index)
    graph.replay()
    reuse = [8192, 0] if draft_index == 0 else [0, 0]
    assert observed.cpu().tolist() == [[-2, -3], [8323, 0], reuse]
    assert observed_slots.cpu().tolist() == [256, 257, 258, 259, -1, -1, -1, -1]
    populate(builder, cm, draft_index=draft_index)
    graph.replay()
    assert observed.cpu().tolist() == [[-1, -3], [8323, 0], reuse]
    cm.req_topk_buffer_generations.fill_(-1)
    populate(builder, cm, draft_index=draft_index)
    graph.replay()
    assert observed.cpu().tolist() == [[-3, -3], [0, 0], [0, 0]]
    assert observed_slots.cpu().tolist() == [-1] * 8
    bank = 0 if draft_index is None else 1
    assert builder.nano_last_generation[bank, 1].item() == 11


def test_eager_sp_padding_uses_private_tail_and_exact_query_count():
    builder = make_builder()
    cm = common([4, 5], [10371, 8321])
    cm.num_input_tokens = 8  # padded for TP8, only five actual query rows
    metadata = populate(builder, cm)
    assert metadata.num_decode_tokens == 5
    assert metadata.slot_mapping.cpu().tolist() == [128, 129, 130, 131, 132, -1, -1, -1]
    private_start = builder.nano_pool_capacity * (8192 + 256)
    assert metadata.nano_device_slots[5:].min().item() >= private_start
    assert metadata.nano_device_slots[:5].max().item() < private_start


def test_main_and_indexer_slots_preserve_independent_layouts():
    builder = make_builder()
    # An inactive row between two real rows plus SP padding: validity must
    # follow request ownership, not a contiguous real-token prefix.
    cm = common([2, 4, 5], [10371, 0, 500], pools=(1, 0, 0), generations=(11, -1, 12))
    cm.num_input_tokens = 8
    index_slots = torch.arange(8, dtype=torch.int64, device="npu") + 1024
    index_address = index_slots.data_ptr()
    index_builder = AscendSFAIndexerMetadataBuilder.__new__(AscendSFAIndexerMetadataBuilder)
    index_builder._slot_capacity = 16
    index_builder._nano_token_masks = {}
    metadata = populate(builder, cm)
    with patch(
        "vllm_ascend.attention.indexer.get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(use_nano=True)),
    ):
        index_builder._mask_nano_slot_mapping(cm, index_slots, "test")
    assert metadata.slot_mapping.cpu().tolist() == [128, 129, -1, -1, 132, -1, -1, -1]
    assert index_slots.cpu().tolist() == [1024, 1025, -1, -1, 1028, -1, -1, -1]
    assert index_slots.data_ptr() == index_address
    assert metadata.nano_logical_lens.cpu().tolist() == [8195, 0, 500]
    assert metadata.nano_reuse_logical_lens is None
    assert metadata.nano_tail_lengths[1].count_nonzero().item() == 0

    # Without nano request ownership the normal indexer is unchanged.
    cm.req_topk_buffer_generations = None
    index_slots.fill_(7)
    with patch(
        "vllm_ascend.attention.indexer.get_ascend_config",
        return_value=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(use_nano=True)),
    ):
        index_builder._mask_nano_slot_mapping(cm, index_slots, "test")
    assert index_slots.cpu().tolist() == [7] * 8


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
    first = populate(builder, common([4, 8], [10371, 8324]), draft_index=0)
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
        if name == "nano_reuse_logical_lens":
            assert second.nano_reuse_logical_lens is None
            assert third.nano_reuse_logical_lens is None
            continue
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
        populate(builder, common(ends, lengths), draft_index=0)
        populate(builder, common([1, 2], [10376, 8329]), draft_index=1)
        populate(builder, common([1, 2], [10377, 8330]), draft_index=2)
        graph.replay()
        assert observed.cpu().tolist() == ends


def test_normal_decode_skips_restore_descriptors_and_reuses_block_table():
    builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    cm.nano_restore_tails = False
    md = populate(builder, cm)
    assert md.nano_copy_src_offsets is None
    assert md.nano_source_block_table.data_ptr() == cm.block_table_tensor.data_ptr()
    assert md.nano_logical_lens.cpu().tolist() == [8323]
    cm.nano_restore_tails = True
    restored = populate(builder, cm)
    assert restored.nano_copy_src_offsets is not None
    assert restored.nano_tail_lengths.cpu().tolist() == [[127, 0]]
    cm.nano_restore_tails = False
    assert populate(builder, cm).nano_copy_src_offsets is None


def test_host_buffers_are_persistent_and_device_lengths_are_authoritative():
    builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    md = populate(builder, cm)
    addresses = {
        name: (buffers[0].cpu.data_ptr(), buffers[0].gpu.data_ptr()) for name, buffers in builder.nano_host.items()
    }
    cm.seq_lens_cpu.fill_(1)
    cm._seq_lens_cpu.fill_(1048576)
    cm.seq_lens.fill_(10368)
    changed = populate(builder, cm)
    assert changed.nano_seq_lens.cpu().tolist() == [10368]
    assert changed.nano_prefix_lens.cpu().tolist() == [10240]
    assert changed.nano_logical_lens.cpu().tolist() == [8320]
    assert changed.nano_device_slots.cpu().tolist() == [8448 + 8192 + p % 256 for p in range(10364, 10368)]
    assert changed.nano_query_ends.data_ptr() == md.nano_query_ends.data_ptr()
    assert addresses == {
        name: (buffers[0].cpu.data_ptr(), buffers[0].gpu.data_ptr()) for name, buffers in builder.nano_host.items()
    }


def test_nano_host_inputs_never_fall_back_to_device_to_host_copy():
    builder = make_builder()
    cm = common([4], [10371], pools=(1,), generations=(11,))
    cm.req_topk_buffer_slots = cm.req_topk_buffer_slots.to("npu")
    with pytest.raises(RuntimeError, match="must be CPU tensors"):
        populate(builder, cm)
