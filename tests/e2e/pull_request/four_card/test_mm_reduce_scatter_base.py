#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.ops.register_custom_ops import DeviceOperator  # noqa: E402

WORLD_SIZE = 4
GLOBAL_M = 32
K_SIZE = 256
N_SIZE = 256
TOLERANCE = 1e-2
HCCL_SOCKET_PORT_RANGE_SIZE = 32


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_hccl_socket_port_range() -> str:
    start = min(_find_free_port(), 65535 - HCCL_SOCKET_PORT_RANGE_SIZE)
    end = start + HCCL_SOCKET_PORT_RANGE_SIZE - 1
    return f"{start}-{end}"


def _get_hccl_comm_name(rank: int) -> str:
    from torch.distributed.distributed_c10d import _get_default_group

    default_pg = _get_default_group()
    return default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)


def _expected_reduce_scatter(mm_output: torch.Tensor, rank: int) -> torch.Tensor:
    outputs = [torch.empty_like(mm_output) for _ in range(WORLD_SIZE)]
    dist.all_gather(outputs, mm_output)
    reduced_output = torch.stack(outputs).sum(dim=0)
    return torch.chunk(reduced_output, WORLD_SIZE, dim=0)[rank].contiguous()


def _run_mm_reduce_scatter_base_unquantized(rank: int, master_port: int) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )

    hcom = _get_hccl_comm_name(rank)
    x1 = torch.full(
        (GLOBAL_M, K_SIZE),
        rank + 1,
        device="npu",
        dtype=torch.bfloat16,
    )
    x2 = torch.ones((K_SIZE, N_SIZE), device="npu", dtype=torch.bfloat16)

    output = DeviceOperator.npu_mm_reduce_scatter_base(
        x1,
        x2,
        hcom,
        WORLD_SIZE,
        reduce_op="sum",
        bias=None,
        comm_turn=0,
        comm_mode="aiv",
    )
    torch.npu.synchronize()

    expected = _expected_reduce_scatter(torch.matmul(x1, x2), rank)
    torch.npu.synchronize()

    assert output.shape == (GLOBAL_M // WORLD_SIZE, N_SIZE)
    torch.testing.assert_close(
        output.float().cpu(),
        expected.float().cpu(),
        atol=TOLERANCE,
        rtol=TOLERANCE,
    )

    dist.barrier()
    dist.destroy_process_group()


def _run_mm_reduce_scatter_base_quantized(rank: int, master_port: int) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )

    hcom = _get_hccl_comm_name(rank)
    x1 = torch.full(
        (GLOBAL_M, K_SIZE),
        rank + 1,
        device="npu",
        dtype=torch.int8,
    )
    x1_scale = torch.ones((GLOBAL_M, 1), device="npu", dtype=torch.float32)
    x2 = torch.ones(
        (K_SIZE, N_SIZE),
        device="npu",
        dtype=torch.int8,
    )
    x2_scale = torch.ones((1, N_SIZE), device="npu", dtype=torch.float32)

    output = DeviceOperator.npu_mm_reduce_scatter_base(
        x1,
        x2,
        hcom,
        WORLD_SIZE,
        reduce_op="sum",
        bias=None,
        comm_turn=0,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        output_dtype=torch.bfloat16,
        comm_mode="aiv",
    )
    torch.npu.synchronize()

    local_mm = torch.matmul(x1.float(), x2.float())
    local_mm = (local_mm * x1_scale * x2_scale).to(torch.bfloat16)
    expected = _expected_reduce_scatter(local_mm, rank)
    torch.npu.synchronize()

    assert output.shape == (GLOBAL_M // WORLD_SIZE, N_SIZE)
    torch.testing.assert_close(
        output.float().cpu(),
        expected.float().cpu(),
        atol=TOLERANCE,
        rtol=TOLERANCE,
    )

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required for MmReduceScatterBase e2e test.",
)
@pytest.mark.skipif(
    hasattr(torch, "npu") and torch.npu.device_count() < WORLD_SIZE,
    reason="MmReduceScatterBase e2e test requires 4 visible NPUs.",
)
def test_mm_reduce_scatter_base_unquantized() -> None:
    os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", _make_hccl_socket_port_range())
    mp.spawn(
        _run_mm_reduce_scatter_base_unquantized,
        args=(_find_free_port(),),
        nprocs=WORLD_SIZE,
    )


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required for MmReduceScatterBase e2e test.",
)
@pytest.mark.skipif(
    hasattr(torch, "npu") and torch.npu.device_count() < WORLD_SIZE,
    reason="MmReduceScatterBase e2e test requires 4 visible NPUs.",
)
def test_mm_reduce_scatter_base_quantized() -> None:
    os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", _make_hccl_socket_port_range())
    mp.spawn(
        _run_mm_reduce_scatter_base_quantized,
        args=(_find_free_port(),),
        nprocs=WORLD_SIZE,
    )
