from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithMC2
from vllm_ascend.platform import NPUPlatform


@pytest.mark.parametrize("replace_allreduce", [False, True])
def test_profile_shards_moe_and_restores_global_rows(replace_allreduce):
    hidden = torch.arange(26, dtype=torch.float32).reshape(13, 2)
    logits = hidden + 100
    ids = torch.arange(13)
    context = SimpleNamespace(pcp_profile=True, padded_num_tokens=13, mc2_mask=torch.ones(13, dtype=torch.bool))
    group = SimpleNamespace(world_size=4, rank_in_group=0, all_gather=Mock())
    gathered = torch.nn.functional.pad(hidden * 2, (0, 0, 0, 3))
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
            start = rank * 4
            expected_hidden = torch.nn.functional.pad(hidden, (0, 0, 0, 3))[start : start + 4]
            torch.testing.assert_close(prepared.hidden_states, expected_hidden)
            torch.testing.assert_close(
                prepared.router_logits, torch.nn.functional.pad(logits, (0, 0, 0, 3))[start : start + 4]
            )
            assert prepared.mc2_mask.tolist() == [row < 13 for row in range(start, start + 4)]
            torch.testing.assert_close(
                layer.pad_and_split_input_ids(ids), torch.nn.functional.pad(ids, (0, 3))[start : start + 4]
            )
            restored = layer.finalize(prepared.hidden_states * 2, False, prepared.padded_hidden_states_shape)
            torch.testing.assert_close(restored, hidden * 2)

        # The same layer must stop sharding as soon as profiling ends.
        context.pcp_profile = False
        prepared = layer.prepare(hidden, logits, replace_allreduce=replace_allreduce)
        assert prepared.hidden_states.shape == hidden.shape
        group.all_gather.reset_mock()
        torch.testing.assert_close(layer.finalize(hidden, False, prepared.padded_hidden_states_shape), hidden)
        group.all_gather.assert_not_called()


@pytest.mark.parametrize("replace_allreduce", [False, True])
def test_profile_sharding_composes_with_tp(replace_allreduce):
    hidden = torch.arange(32, dtype=torch.float32).reshape(16, 2)
    context = SimpleNamespace(pcp_profile=True, padded_num_tokens=16, mc2_mask=torch.ones(16, dtype=torch.bool))
    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=Mock())
    layer = PrepareAndFinalizeWithMC2.__new__(PrepareAndFinalizeWithMC2)
    layer.tp_size = 2
    layer.tp_rank = 1
    layer.moe_config = SimpleNamespace(tp_group=SimpleNamespace(device_group=None))
    tp_hidden = hidden[8:]
    group.all_gather.return_value = tp_hidden * 2

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
            start = rank * 4
            torch.testing.assert_close(prepared.hidden_states, tp_hidden[start : start + 4])
            torch.testing.assert_close(
                layer.pad_and_split_input_ids(torch.arange(16)),
                torch.arange(8, 16)[start : start + 4],
            )
            restored = layer.finalize(prepared.hidden_states * 2, False, prepared.padded_hidden_states_shape)
            torch.testing.assert_close(restored, (tp_hidden if replace_allreduce else hidden) * 2)


@pytest.mark.parametrize(
    ("profile", "metadata", "pcp_size", "dcp_size", "expected"),
    [
        (True, None, 4, 1, True),
        (False, None, 4, 1, False),
        (True, {}, 4, 1, False),
        (True, None, 1, 1, False),
        (True, None, 4, 2, False),
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
        assert result["pcp_profile"] == expected
