# SPDX-License-Identifier: Apache-2.0

import os
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
    mp.set_start_method("spawn", force=True)
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


@torch.inference_mode()
def _max_sum_graph_worker(rank: int, world_size: int, port: int, result_queue: mp.SimpleQueue) -> None:
    dcp_group = None
    try:
        os.environ["TRITON_CACHE_DIR"] = f"/tmp/sfa_dcp_max_sum_graph_{port}_rank{rank}"
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
            group_name="sfa_dcp_max_sum_graph_test",
            use_device_communicator=False,
        )

        # Keep input/output storage addresses fixed across capture and both
        # replays so the test distinguishes real graph reuse from eager reruns.
        for scatter_dim in (0, 1):
            num_tokens, num_heads, head_dim = (4, 8, 512) if scatter_dim == 0 else (5, 8, 512)

            def make_inputs(
                seed: int,
                tokens: int,
                heads: int,
                dimension: int,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                torch.manual_seed(seed)
                outputs = torch.randn(
                    world_size,
                    tokens,
                    heads,
                    dimension,
                    dtype=torch.bfloat16,
                    device="npu",
                )
                maximum = torch.randn(
                    world_size,
                    1,
                    tokens,
                    heads,
                    dtype=torch.float32,
                    device="npu",
                )
                summation = torch.rand_like(maximum) + 0.125
                return outputs, maximum, summation

            sender_outputs, sender_max, sender_sum = make_inputs(
                20260918 + scatter_dim,
                num_tokens,
                num_heads,
                head_dim,
            )
            static_output = sender_outputs[rank].contiguous()
            static_max = sender_max[rank].contiguous()
            static_sum = sender_sum[rank].contiguous()
            input_pointers = (static_output.data_ptr(), static_max.data_ptr(), static_sum.data_ptr())

            def expected_output(
                outputs: torch.Tensor,
                maximum: torch.Tensor,
                summation: torch.Tensor,
                scatter: int,
                tokens: int,
                heads: int,
            ) -> torch.Tensor:
                sender_lse = maximum[:, 0] + torch.log(summation[:, 0])
                if scatter == 0:
                    local_tokens = tokens // world_size
                    token_slice = slice(rank * local_tokens, (rank + 1) * local_tokens)
                    return _reference_merge(outputs[:, token_slice], sender_lse[:, token_slice])
                local_heads = heads // world_size
                head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
                return _reference_merge(outputs[:, :, head_slice], sender_lse[:, :, head_slice])

            eager = torch.ops.vllm.sfa_dcp_a2a_fused_max_sum(
                static_output,
                static_max,
                static_sum,
                world_size,
                scatter_dim,
                dcp_group.unique_name,
            )
            torch_npu.npu.synchronize()
            torch.testing.assert_close(
                eager,
                expected_output(sender_outputs, sender_max, sender_sum, scatter_dim, num_tokens, num_heads),
                atol=2e-2,
                rtol=2e-2,
            )

            for _ in range(3):
                torch.ops.vllm.sfa_dcp_a2a_fused_max_sum(
                    static_output,
                    static_max,
                    static_sum,
                    world_size,
                    scatter_dim,
                    dcp_group.unique_name,
                )
            torch_npu.npu.synchronize()
            torch.distributed.barrier(group=dcp_group.device_group)

            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph):
                graph_output = torch.ops.vllm.sfa_dcp_a2a_fused_max_sum(
                    static_output,
                    static_max,
                    static_sum,
                    world_size,
                    scatter_dim,
                    dcp_group.unique_name,
                )
            graph.replay()
            torch_npu.npu.synchronize()
            torch.distributed.barrier(group=dcp_group.device_group)
            torch.testing.assert_close(
                graph_output,
                expected_output(sender_outputs, sender_max, sender_sum, scatter_dim, num_tokens, num_heads),
                atol=2e-2,
                rtol=2e-2,
            )
            first_output = graph_output.clone()
            graph_output_pointer = graph_output.data_ptr()

            changed_outputs, changed_max, changed_sum = make_inputs(
                20260919 + scatter_dim,
                num_tokens,
                num_heads,
                head_dim,
            )
            static_output.copy_(changed_outputs[rank])
            static_max.copy_(changed_max[rank])
            static_sum.copy_(changed_sum[rank])
            sender_outputs = changed_outputs
            sender_max = changed_max
            sender_sum = changed_sum
            torch_npu.npu.synchronize()
            assert input_pointers == (static_output.data_ptr(), static_max.data_ptr(), static_sum.data_ptr())

            graph.replay()
            torch_npu.npu.synchronize()
            torch.distributed.barrier(group=dcp_group.device_group)
            assert graph_output.data_ptr() == graph_output_pointer
            torch.testing.assert_close(
                graph_output,
                expected_output(sender_outputs, sender_max, sender_sum, scatter_dim, num_tokens, num_heads),
                atol=2e-2,
                rtol=2e-2,
            )
            assert not torch.equal(graph_output, first_output)
            torch.distributed.barrier(group=dcp_group.device_group)

        result_queue.put(None)
    except Exception:
        result_queue.put(traceback.format_exc())
    finally:
        if dcp_group is not None:
            dcp_group.destroy()
        destroy_distributed_environment()


def test_registered_sfa_dcp_a2a_fused_max_sum_aclgraph_multi_rank() -> None:
    world_size = 2
    mp.set_start_method("spawn", force=True)
    result_queue = mp.SimpleQueue()
    port = 39_501 + random.randint(0, 10_000)
    processes = [
        mp.Process(
            target=_max_sum_graph_worker,
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
