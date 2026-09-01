# SPDX-License-Identifier: Apache-2.0

import random
import traceback

import torch
import torch.multiprocessing as mp
import torch_npu
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

import vllm_ascend.ops.triton.sfa_cp  # noqa: F401
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _reference_merge(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(lse)
    safe_lse = lse.masked_fill(~finite, float("-inf"))
    weights = torch.nan_to_num(torch.softmax(safe_lse, dim=0), nan=0.0)
    safe_output = torch.where(finite.unsqueeze(-1), output.float(), 0.0)
    return (safe_output * weights.unsqueeze(-1)).sum(0).to(output.dtype)


@torch.inference_mode()
def _worker(rank: int, world_size: int, port: int, result_queue: mp.SimpleQueue) -> None:
    dcp_group = None
    try:
        torch_npu.npu.set_device(rank)
        init_device_properties_triton()
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="hccl",
        )
        dcp_group = init_model_parallel_group(
            [list(range(world_size))],
            local_rank=rank,
            backend="hccl",
            group_name="sfa_dcp_a2a_test",
            use_device_communicator=False,
        )

        for scatter_dim in (0, 1):
            torch.manual_seed(2026 + scatter_dim)
            num_tokens, num_heads, head_dim = (4, 4, 96) if scatter_dim == 0 else (3, 8, 128)
            sender_outputs = torch.randn(
                world_size,
                num_tokens,
                num_heads,
                head_dim,
                dtype=torch.bfloat16,
                device="npu",
            )
            sender_lses = torch.randn(
                world_size,
                num_tokens,
                num_heads,
                1,
                dtype=torch.float32,
                device="npu",
            )
            sender_lses += torch.arange(world_size, dtype=torch.float32, device="npu").view(-1, 1, 1, 1)

            actual = torch.ops.vllm.sfa_dcp_a2a_fused(
                sender_outputs[rank].contiguous(),
                sender_lses[rank].contiguous(),
                world_size,
                scatter_dim,
                dcp_group.unique_name,
            )

            if scatter_dim == 0:
                local_tokens = num_tokens // world_size
                token_slice = slice(rank * local_tokens, (rank + 1) * local_tokens)
                expected = _reference_merge(
                    sender_outputs[:, token_slice],
                    sender_lses[:, token_slice, :, 0],
                )
            else:
                local_heads = num_heads // world_size
                head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
                expected = _reference_merge(
                    sender_outputs[:, :, head_slice],
                    sender_lses[:, :, head_slice, 0],
                )

            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            torch.distributed.barrier(group=dcp_group.device_group)

        result_queue.put(None)
    except Exception:
        result_queue.put(traceback.format_exc())
    finally:
        if dcp_group is not None:
            dcp_group.destroy()
        destroy_distributed_environment()


def test_registered_sfa_dcp_a2a_fused_multi_rank() -> None:
    world_size = 2
    mp.set_start_method("fork", force=True)
    result_queue = mp.SimpleQueue()
    port = 29_501 + random.randint(0, 10_000)
    processes = [
        mp.Process(
            target=_worker,
            args=(rank, world_size, port, result_queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()
    results = [result_queue.get() for _ in processes]
    for process in processes:
        process.join()

    assert all(process.exitcode == 0 for process in processes)
    assert results == [None] * world_size, "\n".join(result for result in results if result is not None)
