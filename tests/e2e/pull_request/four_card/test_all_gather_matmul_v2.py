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

WORLD_SIZE = 4
LOCAL_M = 32
K_SIZE = 6144
N_SIZE = 1024
UNQUANTIZED_N_SIZE = 256
SEED = 2026
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


def _all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    outputs = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(outputs, tensor)
    return torch.cat(outputs, dim=0)


def _run_all_gather_matmul_v2_quant(rank: int, master_port: int) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )

    hcom = _get_hccl_comm_name(rank)
    torch.manual_seed(SEED + rank)
    input_fp = torch.randn((LOCAL_M, K_SIZE), device="npu", dtype=torch.bfloat16)
    quantized_x, x1_scale = torch_npu.npu_dynamic_quant(input_fp, dst_type=torch.int8)
    quantized_x = quantized_x.contiguous()
    x1_scale = x1_scale.reshape(-1, 1).contiguous()

    torch.manual_seed(SEED)
    weight = torch.randint(
        -16,
        16,
        (K_SIZE, N_SIZE),
        device="npu",
        dtype=torch.int8,
    )
    x2_scale = torch.ones((1, N_SIZE), device="npu", dtype=torch.float32)

    output, gather_out = torch_npu.npu_all_gather_base_mm(
        quantized_x,
        weight,
        hcom,
        WORLD_SIZE,
        bias=None,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        gather_index=0,
        gather_output=True,
        comm_turn=0,
        output_dtype=torch.bfloat16,
        comm_mode="aiv",
    )
    torch.npu.synchronize()

    gathered_x = _all_gather_tensor(quantized_x)
    gathered_x1_scale = _all_gather_tensor(x1_scale)
    expected = torch_npu.npu_quant_matmul(
        gathered_x,
        weight,
        x2_scale.squeeze(0),
        bias=None,
        output_dtype=torch.bfloat16,
    )
    expected = (expected.float() * gathered_x1_scale).to(torch.bfloat16)
    torch.npu.synchronize()

    assert output.shape == (LOCAL_M * WORLD_SIZE, N_SIZE)
    assert gather_out.shape == (LOCAL_M * WORLD_SIZE, K_SIZE)
    torch.testing.assert_close(
        output.float().cpu(),
        expected.float().cpu(),
        atol=TOLERANCE,
        rtol=TOLERANCE,
    )
    torch.testing.assert_close(
        gather_out.cpu(),
        gathered_x.cpu(),
        atol=0,
        rtol=0,
    )

    dist.barrier()
    dist.destroy_process_group()


def _run_all_gather_matmul_v2_unquantized(rank: int, master_port: int) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=WORLD_SIZE,
        init_method=f"tcp://127.0.0.1:{master_port}",
    )

    hcom = _get_hccl_comm_name(rank)
    torch.manual_seed(SEED + rank)
    input_fp = torch.randn((LOCAL_M, K_SIZE), device="npu", dtype=torch.bfloat16)

    torch.manual_seed(SEED)
    linear_weight = torch.randn(
        (UNQUANTIZED_N_SIZE, K_SIZE),
        device="npu",
        dtype=torch.bfloat16,
    )

    output, gather_out = torch_npu.npu_all_gather_base_mm(
        input_fp,
        linear_weight.t(),
        hcom,
        WORLD_SIZE,
        bias=None,
        x1_scale=None,
        x2_scale=None,
        gather_index=0,
        gather_output=True,
        comm_turn=0,
        output_dtype=torch.bfloat16,
        comm_mode="aiv",
    )
    torch.npu.synchronize()

    gathered_x = _all_gather_tensor(input_fp)
    expected = torch.nn.functional.linear(gathered_x, linear_weight, bias=None)
    torch.npu.synchronize()

    assert output.shape == (LOCAL_M * WORLD_SIZE, UNQUANTIZED_N_SIZE)
    assert gather_out.shape == (LOCAL_M * WORLD_SIZE, K_SIZE)
    torch.testing.assert_close(
        output.float().cpu(),
        expected.float().cpu(),
        atol=TOLERANCE,
        rtol=TOLERANCE,
    )
    torch.testing.assert_close(
        gather_out.cpu(),
        gathered_x.cpu(),
        atol=0,
        rtol=0,
    )

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required for AllGatherMatmulV2 e2e test.",
)
@pytest.mark.skipif(
    hasattr(torch, "npu") and torch.npu.device_count() < WORLD_SIZE,
    reason="AllGatherMatmulV2 e2e test requires 4 visible NPUs.",
)
def test_all_gather_matmul_v2_dynamic_quant() -> None:
    os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", _make_hccl_socket_port_range())
    mp.spawn(_run_all_gather_matmul_v2_quant, args=(_find_free_port(),), nprocs=WORLD_SIZE)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is required for AllGatherMatmulV2 e2e test.",
)
@pytest.mark.skipif(
    hasattr(torch, "npu") and torch.npu.device_count() < WORLD_SIZE,
    reason="AllGatherMatmulV2 e2e test requires 4 visible NPUs.",
)
def test_all_gather_matmul_v2_unquantized() -> None:
    os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", _make_hccl_socket_port_range())
    mp.spawn(_run_all_gather_matmul_v2_unquantized, args=(_find_free_port(),), nprocs=WORLD_SIZE)
