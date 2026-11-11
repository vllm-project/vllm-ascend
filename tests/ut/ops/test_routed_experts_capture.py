# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import MixtralConfig
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import BatchDescriptor, DPMetadata, ForwardContext, override_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.platforms.cpu import CpuPlatform

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe import fused_moe, prepare_finalize, routed_experts_capture
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAll2All, PrepareAndFinalizeWithMC2
from vllm_ascend.quantization.quant_type import QuantType


class _Router(BaseRouter):
    @property
    def routing_method_type(self):
        return RoutingMethodType.TopK

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        return torch.ones_like(router_logits), router_logits.to(torch.int32)


@pytest.fixture
def capture_runtime(tmp_path, monkeypatch):
    MixtralConfig(
        architectures=["MixtralForCausalLM"], num_hidden_layers=2, num_local_experts=64, num_experts_per_tok=2
    ).save_pretrained(tmp_path)
    monkeypatch.setattr("vllm.platforms.current_platform", CpuPlatform())
    monkeypatch.setattr("vllm.model_executor.layers.fused_moe.routed_experts_capturer.current_platform", CpuPlatform())
    config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), dtype="float32", max_model_len=32, skip_tokenizer_init=True),
        device_config=DeviceConfig(device="cpu"),
    )
    moe_config = FusedMoEConfig(
        num_experts=64,
        experts_per_token=2,
        hidden_dim=4,
        intermediate_size=4,
        num_local_experts=64,
        num_logical_experts=64,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.float32,
    )
    tp_group = MagicMock(world_size=2, rank_in_group=0)
    dp_group = MagicMock(world_size=1, rank_in_group=0)
    monkeypatch.setattr(fused_moe, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(fused_moe, "get_dp_group", lambda: dp_group)
    monkeypatch.setattr(routed_experts_capture, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_world_size", lambda: tp_group.world_size)
    monkeypatch.setattr(prepare_finalize, "get_tensor_model_parallel_rank", lambda: tp_group.rank_in_group)
    # No expert kernels or NPU communication objects are needed for capture.
    monkeypatch.setattr(fused_moe, "setup_moe_comm_method", lambda _: None)
    monkeypatch.setattr(fused_moe, "get_moe_comm_method", lambda *args: None)
    experts = SimpleNamespace(quant_type=QuantType.NONE, quant_method=SimpleNamespace(is_monolithic=False))
    with set_current_vllm_config(config):
        runners = torch.nn.ModuleList(
            [
                AscendMoERunner(f"model.layers.{i}.mlp", moe_config, _Router(top_k=2, global_num_experts=64), experts)
                for i in range(2)
            ]
        )
        capturer = RoutedExpertsCapturer(16, config)
        bind_routed_experts_capturer(runners, capturer)
        assert capturer.restore_rows.__self__ is runners[0]
        yield runners, capturer, tp_group, dp_group


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize(
    "comm,sp,counts,tp_size,dp_rank,input_padding",
    [
        (MoECommType.ALLTOALL, False, [5], 2, 0, 0),  # uneven 3+2
        (MoECommType.ALLTOALL, False, [1], 4, 0, 0),
        (MoECommType.ALLTOALL, False, [0], 2, 0, 0),
        (MoECommType.ALLTOALL, False, [5, 13], 2, 0, 0),
        (MoECommType.ALLTOALL, True, [5, 13], 2, 0, 0),
        (MoECommType.MC2, False, [5], 2, 0, 0),
        (MoECommType.MC2, False, [5, 13], 2, 0, 0),
        (MoECommType.MC2, True, [5, 13], 2, 0, 0),  # padding BETWEEN SP shards
        (MoECommType.MC2, True, [5, 13], 2, 0, 3),  # input padded BEFORE SP
        (MoECommType.MC2, True, [0, 13], 2, 0, 0),
        (MoECommType.FUSED_MC2, True, [13, 5], 2, 1, 0),
        (MoECommType.MC2, False, [5], 1, 0, 0),
    ],
)
def test_prepare_router_capture_restores_exact_rows(
    capture_runtime, monkeypatch, v2, comm, sp, counts, tp_size, dp_rank, input_padding
):
    """Run real prepare -> router -> upstream capture on every simulated TP rank."""
    runners, capturer, tp_group, dp_group = capture_runtime
    monkeypatch.setattr("vllm_ascend.ascend_forward_context._USE_V2_EXTRA_KWARGS", v2)
    tp_group.world_size = tp_size
    dp_group.rank_in_group = dp_rank
    rows = counts[dp_rank]
    expected = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2) + 1
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=DPMetadata(torch.tensor(counts)) if len(counts) > 1 else None,
        batch_descriptor=BatchDescriptor(num_tokens=rows),
    )
    with override_forward_context(ctx):
        _EXTRA_CTX.num_tokens = rows
        _EXTRA_CTX.padded_num_tokens = ((max(counts) + tp_size - 1) // tp_size) * tp_size
        _EXTRA_CTX.moe_comm_type = comm
        _EXTRA_CTX.mc2_mask = torch.arange(_EXTRA_CTX.padded_num_tokens) < rows
        inputs = []
        prepares = []
        for rank in range(tp_size):
            tp_group.rank_in_group = rank
            cls = PrepareAndFinalizeWithAll2All if comm == MoECommType.ALLTOALL else PrepareAndFinalizeWithMC2
            prepare = cls(runners[0].moe_config)
            states = torch.nn.functional.pad(expected.float(), (0, 0, 0, input_padding))
            if sp:
                shard_rows = (states.shape[0] + tp_size - 1) // tp_size
                states = torch.nn.functional.pad(states, (0, 0, 0, shard_rows * tp_size - states.shape[0]))
                states = states[rank * shard_rows : (rank + 1) * shard_rows]
            output = prepare.prepare(states, states, replace_allreduce=sp)
            inputs.append(output.router_logits.to(torch.int32))
            prepares.append(prepare)

        def gather(outputs, local, group):
            assert group is tp_group.device_group
            torch.testing.assert_close(local, inputs[tp_group.rank_in_group])
            for dest, source in zip(outputs, inputs, strict=True):
                assert dest.shape == source.shape
                dest.copy_(source)

        with patch.object(routed_experts_capture.dist, "all_gather", side_effect=gather) as collective:
            for rank, ids in enumerate(inputs):
                tp_group.rank_in_group = rank
                _EXTRA_CTX.moe_comm_method = SimpleNamespace(prepare_finalize=prepares[rank])
                capturer.device_buffer.fill_(-1)
                original = ids.clone()
                for runner in runners:
                    runner.router.select_experts(torch.empty_like(ids), ids)
                wanted = torch.full_like(capturer.device_buffer, -1)
                wanted[:rows] = expected[:, None, :].expand(-1, 2, -1)
                torch.testing.assert_close(capturer.device_buffer, wanted)
                torch.testing.assert_close(capturer.snapshot_routing_data(rows).to(torch.int32), wanted[:rows])
                torch.testing.assert_close(ids, original)
            assert collective.call_count == (tp_size * 2 if tp_size > 1 and inputs[0].shape[0] > 0 else 0)


@pytest.mark.parametrize(
    "counts,dp_rank,physical,local_sizes,start",
    [
        ([5], 0, 5, None, 0),
        ([5, 7], 1, 12, None, 5),  # unpadded DP concatenation
        ([5, 7], 1, 7, None, 0),  # local batch
        ([5, 7], 1, 14, None, 7),  # padded DP concatenation
        ([5, 9], 1, 16, [3, 3, 5, 5], 6),  # EP gather retaining each DP's SP padding
    ],
)
def test_allgather_capture_selects_dp_rows(capture_runtime, counts, dp_rank, physical, local_sizes, start):
    runners, capturer, _, dp_group = capture_runtime
    dp_group.rank_in_group = dp_rank
    rows = counts[dp_rank]
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=DPMetadata(torch.tensor(counts), local_sizes=local_sizes) if len(counts) > 1 else None,
    )
    ids = torch.arange(physical * 2, dtype=torch.int32).reshape(physical, 2)
    with override_forward_context(ctx), patch.object(routed_experts_capture.dist, "all_gather") as collective:
        _EXTRA_CTX.num_tokens = rows
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        capturer.device_buffer.fill_(-1)
        runners[1].router.select_experts(torch.empty_like(ids), ids)
        wanted = torch.full_like(capturer.device_buffer, -1)
        wanted[:rows, 1] = ids[start : start + rows]
        torch.testing.assert_close(capturer.device_buffer, wanted)
        collective.assert_not_called()


def test_capture_rejects_unknown_rows(capture_runtime):
    _, capturer, _, _ = capture_runtime
    ctx = ForwardContext(
        no_compile_layers={}, attn_metadata={}, slot_mapping={}, dp_metadata=DPMetadata(torch.tensor([5, 7]))
    )
    with override_forward_context(ctx):
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        with pytest.raises(AssertionError, match="Unexpected Ascend routing rows"):
            capturer.capture(0, torch.zeros(10, 2, dtype=torch.int32))
