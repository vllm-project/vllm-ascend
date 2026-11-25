# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.core.gdn_checkpoint_policy import GDNCheckpointPolicy
from vllm_ascend.ops import gdn_checkpoint, native_gdn_checkpoint
from vllm_ascend.ops.gdn_checkpoint import build_gdn_checkpoint_metadata
from vllm_ascend.ops.native_gdn_checkpoint import plan_native_checkpoint


def test_mixed_batch_restores_original_token_order():
    prefix, tail, tail_rows, restore, prefix_limits, tail_limits = plan_native_checkpoint([2, 0, 1], [4, 2, 3])
    packed = prefix + tail
    assert prefix == [0, 1, 4, 5, 6]
    assert tail == [2, 3, 7, 8]
    assert tail_rows == [0, 2]
    assert prefix_limits == [0, 2, 4, 5]
    assert tail_limits == [0, 2, 4]
    assert [packed[index] for index in restore] == list(range(9))


@pytest.mark.parametrize(("offsets", "lengths"), [([4], [4]), ([0], [4])])
def test_plan_requires_a_checkpoint_with_a_nonempty_tail(offsets, lengths):
    with pytest.raises(AssertionError):
        plan_native_checkpoint(offsets, lengths)


@pytest.mark.parametrize("joint_pack", [False, True])
def test_contiguous_checkpoint_keeps_token_order_and_packs_heads_once(monkeypatch, joint_pack):
    head_first = torch.arange(56).reshape(4, 7, 2)
    tensors = tuple(part.transpose(0, 1).unsqueeze(0) for part in head_first.split((1, 1, 2)))
    checkpoint = SimpleNamespace(query_lens=[7], offsets=[4])
    metadata = SimpleNamespace(checkpoint=checkpoint)
    initial_state = torch.zeros(1, 2, 2, 2)
    gate = torch.ones(1, 7, 2)
    calls = []

    def recur(inputs, state, partition, scale, fused, fla_fwd, *, save_training_intermediates):
        calls.append((inputs, state, partition, save_training_intermediates))
        return inputs[2], state + 1

    export = Mock()
    monkeypatch.setattr(native_gdn_checkpoint, "_apply_recurrence", recur)
    monkeypatch.setattr(native_gdn_checkpoint, "_export_checkpoint", export)
    layer = SimpleNamespace()
    output, state = native_gdn_checkpoint.run_gdn_native_checkpoint_prefill(
        layer,
        metadata,
        None,
        *tensors,
        gate,
        gate,
        initial_state,
        fused=None,
        fla_fwd=object(),
        head_first_qkv=head_first if joint_pack else None,
    )

    assert len(calls) == 2
    for (inputs, _, _, saves), tokens in zip(calls, (slice(0, 4), slice(4, None)), strict=True):
        assert saves is not joint_pack
        for actual, expected in zip(inputs[:3], tensors, strict=True):
            torch.testing.assert_close(actual, expected[:, tokens])
        if joint_pack:
            assert all(part.transpose(1, 2).is_contiguous() for part in inputs[:3])
            assert len({part.untyped_storage().data_ptr() for part in inputs[:3]}) == 1
    torch.testing.assert_close(calls[1][1], initial_state + 1)
    torch.testing.assert_close(export.call_args.args[-1], initial_state + 1)
    torch.testing.assert_close(output, tensors[2])
    torch.testing.assert_close(state, initial_state + 2)


def test_host_block_ids_match_device_null_mask():
    cache = SimpleNamespace(hash_block_size=64, cache_hit_alignment_tokens=64)
    config = SimpleNamespace(cache_config=cache, num_hash_lookahead_tokens=0)
    spec = SimpleNamespace(block_size=1152, prefill_checkpoint_alignment=64)
    builder = SimpleNamespace(vllm_config=config, kv_cache_spec=spec, drop_eagle_block=False)
    query_starts = torch.tensor([0, 3400, 6800, 10200], dtype=torch.int32)
    blocks = torch.tensor([[0, 17, 19], [0, 0, 20], [0, 0, 0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([3400, 3400, 3400], dtype=torch.int32),
        query_start_loc_cpu=query_starts,
        query_start_loc=query_starts,
        block_table_tensor=blocks,
    )
    policy = GDNCheckpointPolicy(8192, True, True)
    rows = [2, 0, 1]
    fallback = build_gdn_checkpoint_metadata(builder, metadata, rows, policy)
    fallback.checkpoint_offsets.masked_fill_(fallback.state_indices == 0, 0)

    host_blocks = ([0, 17, 19], [0, 0, 20])
    optimized = build_gdn_checkpoint_metadata(builder, metadata, rows, policy, host_blocks=host_blocks)
    assert optimized.offsets == [0, 2304, 0] == fallback.checkpoint_offsets.tolist()
    assert optimized.state_indices.tolist() == fallback.state_indices.tolist() == [0, 17, 0]
    assert optimized.query_lens == fallback.query_lens
    assert build_gdn_checkpoint_metadata(builder, metadata, [1, 2], policy, host_blocks=host_blocks) is None


def test_dedicated_page_coexists_with_another_rows_positional_checkpoint():
    cache = SimpleNamespace(hash_block_size=128, cache_hit_alignment_tokens=128)
    config = SimpleNamespace(cache_config=cache, num_hash_lookahead_tokens=1)
    spec = SimpleNamespace(block_size=2048, prefill_checkpoint_alignment=64)
    builder = SimpleNamespace(vllm_config=config, kv_cache_spec=spec, drop_eagle_block=False)
    query_starts = torch.tensor([0, 1452, 4852], dtype=torch.int32)
    metadata = SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([3500, 3400], dtype=torch.int32),
        query_start_loc_cpu=query_starts,
        query_start_loc=query_starts,
        block_table_tensor=torch.tensor([[31, 40], [22, 33]], dtype=torch.int32),
        gdn_dedicated_checkpoint_rows={0: ("req", (1, 3456, 77))},
    )
    checkpoint = build_gdn_checkpoint_metadata(
        builder, metadata, [0, 1], GDNCheckpointPolicy(8192, False, False), host_blocks=([31, 40], [22, 33])
    )
    assert checkpoint.offsets == [1408, 3328]
    assert checkpoint.state_indices.tolist() == [77, 22]
    assert checkpoint.dedicated_ack.writes == {"req": ((1, 3456, 77),)}
    assert checkpoint.dedicated_ack.layers == set()


def test_dedicated_only_batch_uses_one_host_transfer(monkeypatch):
    cache = SimpleNamespace(hash_block_size=128, cache_hit_alignment_tokens=128)
    config = SimpleNamespace(cache_config=cache, num_hash_lookahead_tokens=1)
    spec = SimpleNamespace(block_size=2048, prefill_checkpoint_alignment=64)
    builder = SimpleNamespace(vllm_config=config, kv_cache_spec=spec, drop_eagle_block=False)
    starts = torch.tensor([0, 1452, 2308], dtype=torch.int32)
    metadata = SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([3500, 7000], dtype=torch.int32),
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        # The dedicated-only planner needs the dtype but never indexes this table.
        block_table_tensor=torch.empty((0, 0), dtype=torch.int32),
        gdn_dedicated_checkpoint_rows={0: ("a", (1, 3456, 77)), 1: ("b", (1, 6912, 78))},
    )
    upload = Mock(side_effect=torch.tensor)
    monkeypatch.setattr(gdn_checkpoint, "async_tensor_h2d", upload)
    checkpoint = build_gdn_checkpoint_metadata(builder, metadata, [0, 1], GDNCheckpointPolicy(65536, True, False))
    upload.assert_called_once()
    assert checkpoint.offsets == checkpoint.checkpoint_offsets.tolist() == [1408, 768]
    assert checkpoint.state_indices.tolist() == [77, 78]
    assert checkpoint.request_rows is None and checkpoint.block_cols is None
    assert checkpoint.dedicated_ack.writes == {"a": ((1, 3456, 77),), "b": ((1, 6912, 78),)}
