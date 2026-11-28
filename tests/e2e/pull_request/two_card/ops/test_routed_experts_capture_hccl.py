# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Check exact routed-expert capture rows with real NPU TP collectives."""

from functools import partial
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import DPMetadata, create_forward_context, override_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.utils.network_utils import get_open_port

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.routed_experts_capture import collect_local_routing_rows


def _check_capture(rank, config, comm_type, expected, shards, *, sp=False, pre_mc2_rows=None, dp_counts=None):
    dp_metadata = DPMetadata(torch.tensor(dp_counts)) if dp_counts is not None else None
    context = create_forward_context(None, config, dp_metadata=dp_metadata)
    with override_forward_context(context):
        _EXTRA_CTX.moe_comm_type = comm_type
        _EXTRA_CTX.num_tokens = expected.shape[0]
        _EXTRA_CTX.padded_num_tokens = sum(shard.shape[0] for shard in shards)
        _EXTRA_CTX.moe_comm_method = SimpleNamespace(
            prepare_finalize=SimpleNamespace(
                replace_allreduce=sp,
                num_tokens=pre_mc2_rows if sp else expected.shape[0],
            )
        )
        capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
        capturer.device_buffer = torch.full((16, 1, 2), -1, dtype=torch.int32, device="npu")
        capturer.output_dtype = torch.uint8
        capturer.restore_rows = partial(
            collect_local_routing_rows, dp_rank=rank if dp_counts is not None else 0, tp_size=2
        )
        capturer.capture(0, shards[rank])
        torch.testing.assert_close(capturer.device_buffer[: len(expected), 0], expected)
        assert torch.all(capturer.device_buffer[len(expected) :, 0] == -1)
        torch.testing.assert_close(capturer.snapshot_routing_data(len(expected))[:, 0], expected.to(torch.uint8))


def _worker(rank, port):
    torch.npu.set_device(rank)
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_ascend_config(config)
        init_distributed_environment(2, rank, f"tcp://127.0.0.1:{port}", rank, backend="hccl")
        try:
            initialize_model_parallel(tensor_model_parallel_size=2, backend="hccl")
            expected = (torch.arange(10, dtype=torch.int32, device="npu") + 1).reshape(5, 2)

            # ALLTOALL uses unequal TP shards when tokens are not divisible by TP.
            _check_capture(rank, config, MoECommType.ALLTOALL, expected, torch.tensor_split(expected, 2))
            small = torch.cat((expected[:1], torch.zeros((1, 2), dtype=torch.int32, device="npu")))
            _check_capture(rank, config, MoECommType.ALLTOALL, expected[:1], torch.tensor_split(small, 2))

            # MC2 pads the full batch; SP pads each TP shard separately.
            padded = torch.cat((expected, torch.zeros((1, 2), dtype=torch.int32, device="npu")))
            _check_capture(rank, config, MoECommType.MC2, expected, torch.tensor_split(padded, 2))
            sp_shards = tuple(
                torch.cat((shard, torch.zeros((1, 2), dtype=torch.int32, device="npu")))
                for shard in torch.tensor_split(padded, 2)
            )
            for comm_type in (MoECommType.MC2, MoECommType.FUSED_MC2):
                _check_capture(rank, config, comm_type, expected, sp_shards, sp=True, pre_mc2_rows=3)

            # Input padding before SP must survive the per-shard MC2 unpadding.
            input_padded = torch.cat((expected, torch.zeros((3, 2), dtype=torch.int32, device="npu")))
            sp_shards = tuple(
                torch.cat((shard, torch.zeros((3, 2), dtype=torch.int32, device="npu")))
                for shard in torch.tensor_split(input_padded, 2)
            )
            _check_capture(rank, config, MoECommType.MC2, expected, sp_shards, sp=True, pre_mc2_rows=4)

            # ALLGATHER receives the cross-DP batch and must select this rank's rows.
            physical = torch.arange(24, dtype=torch.int32, device="npu").reshape(12, 2)
            local = physical[:5] if rank == 0 else physical[5:]
            _check_capture(rank, config, MoECommType.ALLGATHER, local, (physical, physical), dp_counts=[5, 7])
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def test_routed_experts_capture_rows_with_hccl():
    mp.spawn(_worker, args=(get_open_port(),), nprocs=2)
