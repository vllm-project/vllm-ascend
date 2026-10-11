import inspect
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from vllm_ascend.ascend_forward_context import get_pcp_profile_num_tokens
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMC2
from vllm_ascend.platform import NPUPlatform


@pytest.mark.parametrize("pcp_size", [1, 2, 4, 16])
@pytest.mark.parametrize("decode_query_len", [1, 2, 4])
def test_profile_bound_covers_mixed_batches(pcp_size, decode_query_len):
    manager = PCPManager.__new__(PCPManager)
    manager.pcp_world_size = pcp_size
    manager.dcp_world_size = 1
    manager._reorder_segments = lambda segments, *_: segments
    uses_computed_tokens = "num_computed_tokens" in inspect.signature(manager._get_rank_segments).parameters
    for batch_lengths in product(range(1, 8), repeat=3):
        lengths = np.array(batch_lengths)
        starts = np.concatenate(([0], np.cumsum(lengths)))
        bound = get_pcp_profile_num_tokens(int(lengths.sum()), pcp_size, 3, decode_query_len)
        for prefilling in product([False, True], repeat=3):
            if any(not prefill and length > decode_query_len for length, prefill in zip(lengths, prefilling)):
                continue
            for rank in range(pcp_size):
                args = (
                    (rank, lengths, np.zeros(3), np.array(prefilling), starts)
                    if uses_computed_tokens
                    else (rank, lengths, np.array(prefilling), starts)
                )
                segments = manager._get_rank_segments(*args)
                assert sum(segment.num_tokens for segment in segments) <= bound


def test_profile_bound_retains_decode_and_rounding_budget():
    assert get_pcp_profile_num_tokens(16384, 4, 4, 1) == 4107
    assert get_pcp_profile_num_tokens(16384, 4, 4, 2) == 4110
    assert get_pcp_profile_num_tokens(128, 4, 128, 2) == 128
    assert get_pcp_profile_num_tokens(16384, 1, 4, 1) == 16384


@pytest.mark.parametrize("replace_allreduce", [False, True])
def test_profile_shards_moe_and_restores_global_rows(replace_allreduce):
    hidden = torch.arange(26, dtype=torch.float32).reshape(13, 2)
    logits = hidden + 100
    ids = torch.arange(13)
    context = SimpleNamespace(pcp_profile_num_tokens=5, padded_num_tokens=13, mc2_mask=torch.ones(13, dtype=torch.bool))
    group = SimpleNamespace(world_size=4, rank_in_group=0, all_gather=Mock())
    gathered = torch.nn.functional.pad(hidden * 2, (0, 0, 0, 7))
    group.all_gather.return_value = gathered
    with (
        patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", context),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_pcp_group", return_value=group),
    ):
        for rank in range(4):
            group.rank_in_group = rank
            layer = PrepareAndFinalizeWithMC2.__new__(PrepareAndFinalizeWithMC2)
            layer.tp_size = 1
            layer.tp_rank = 0
            prepared = layer.prepare(hidden, logits, replace_allreduce=replace_allreduce)
            start = rank * 5
            expected_hidden = torch.nn.functional.pad(hidden, (0, 0, 0, 7))[start : start + 5]
            torch.testing.assert_close(prepared.hidden_states, expected_hidden)
            torch.testing.assert_close(
                prepared.router_logits, torch.nn.functional.pad(logits, (0, 0, 0, 7))[start : start + 5]
            )
            assert prepared.mc2_mask.tolist() == [row < 13 for row in range(start, start + 5)]
            torch.testing.assert_close(
                layer.pad_and_split_input_ids(ids), torch.nn.functional.pad(ids, (0, 7))[start : start + 5]
            )
            restored = layer.finalize(prepared.hidden_states * 2, False, prepared.padded_hidden_states_shape)
            torch.testing.assert_close(restored, hidden * 2)

        # The same layer must stop sharding as soon as profiling ends.
        context.pcp_profile_num_tokens = None
        prepared = layer.prepare(hidden, logits, replace_allreduce=replace_allreduce)
        assert prepared.hidden_states.shape == hidden.shape
        group.all_gather.reset_mock()
        torch.testing.assert_close(layer.finalize(hidden, False, prepared.padded_hidden_states_shape), hidden)
        group.all_gather.assert_not_called()


def test_small_profile_keeps_full_decode_rows():
    layer = PrepareAndFinalizeWithMC2.__new__(PrepareAndFinalizeWithMC2)
    layer.tp_size = 1
    layer.tp_rank = 0
    hidden = torch.ones(4, 2)
    context = SimpleNamespace(
        pcp_profile_num_tokens=4107, padded_num_tokens=4, mc2_mask=torch.ones(4, dtype=torch.bool)
    )
    with patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", context):
        prepared = layer.prepare(hidden, hidden)
        assert prepared.hidden_states.shape[0] == 4
        assert layer._pcp_profile_num_tokens is None


@pytest.mark.parametrize("replace_allreduce", [False, True])
def test_profile_sharding_composes_with_tp(replace_allreduce):
    hidden = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    context = SimpleNamespace(
        pcp_profile_num_tokens=10, padded_num_tokens=16, mc2_mask=torch.ones(16, dtype=torch.bool)
    )
    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=Mock())
    layer = PrepareAndFinalizeWithMC2.__new__(PrepareAndFinalizeWithMC2)
    layer.tp_size = 2
    layer.tp_rank = 1
    layer.moe_config = SimpleNamespace(tp_group=SimpleNamespace(device_group=None))
    tp_hidden = hidden[8:]
    group.all_gather.return_value = torch.nn.functional.pad(tp_hidden * 2, (0, 0, 0, 2))

    def gather_tp(outputs, *_):
        for output, expected in zip(outputs, hidden.tensor_split(2)):
            output.copy_(expected * 2)

    with (
        patch("vllm_ascend.ops.fused_moe.prepare_finalize._EXTRA_CTX", context),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.get_pcp_group", return_value=group),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.dist.all_gather", side_effect=gather_tp),
        patch("vllm_ascend.ops.fused_moe.prepare_finalize.sequence_parallel_chunk", side_effect=lambda x: x[8:]),
    ):
        for rank in range(2):
            group.rank_in_group = rank
            inputs = tp_hidden if replace_allreduce else hidden
            prepared = layer.prepare(inputs, inputs, replace_allreduce=replace_allreduce)
            start = rank * 5
            torch.testing.assert_close(
                prepared.hidden_states, torch.nn.functional.pad(tp_hidden, (0, 0, 0, 2))[start : start + 5]
            )
            torch.testing.assert_close(
                layer.pad_and_split_input_ids(torch.arange(16)),
                torch.nn.functional.pad(torch.arange(8, 16), (0, 2))[start : start + 5],
            )
            restored = layer.finalize(prepared.hidden_states * 2, False, prepared.padded_hidden_states_shape)
            torch.testing.assert_close(restored, (tp_hidden if replace_allreduce else hidden) * 2)


@pytest.mark.parametrize(
    ("profile", "metadata", "pcp_size", "dcp_size", "expected"),
    [
        (True, None, 4, 1, 4107),
        (False, None, 4, 1, None),
        (True, {}, 4, 1, None),
        (True, None, 1, 1, None),
        (True, None, 4, 2, None),
    ],
)
def test_platform_enables_sharding_only_for_pcp_memory_profile(profile, metadata, pcp_size, dcp_size, expected):
    config = MagicMock()
    config.use_v2_model_runner = True
    config.parallel_config.prefill_context_parallel_size = pcp_size
    config.parallel_config.decode_context_parallel_size = dcp_size
    config.scheduler_config.max_num_batched_tokens = 16384
    config.scheduler_config.max_num_seqs = 4
    config.speculative_config = None
    with (
        patch("vllm_ascend.ascend_forward_context.get_mrv2_in_profile_run", return_value=profile),
        patch("vllm_ascend.ascend_forward_context.get_mc2_mask", return_value=None),
        patch("vllm_ascend.ascend_forward_context.select_moe_comm_method"),
        patch("vllm_ascend.ascend_forward_context.use_cann_megamoe", return_value=False),
        patch("vllm_ascend.ascend_forward_context._is_decode_only_node", return_value=False),
        patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method"),
        patch("vllm_ascend.quantization.utils.get_dynamic_mx_quant_scale_alg"),
        patch("vllm.distributed.get_dp_group", return_value=SimpleNamespace(world_size=1)),
        patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=1),
        patch("vllm_ascend.platform.is_moe_model", return_value=True),
    ):
        result = NPUPlatform.set_additional_forward_context(metadata, config, None, num_tokens=16384)
        assert result["pcp_profile_num_tokens"] == expected
