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

from vllm_ascend.spec_decode.mtp import compact_mtp_topk_indices


@torch.inference_mode()
def _worker(rank, port, result_queue):
    group = None
    try:
        torch_npu.npu.set_device(rank)
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="hccl",
        )
        group = init_model_parallel_group(
            [[0, 1]],
            local_rank=rank,
            backend="hccl",
            group_name="mtp_index_cache_test",
            use_device_communicator=True,
        )
        for num_tokens, sample_ids in [(8, [3, 7]), (7, [0, 2, 6]), (4, [0, 1, 2, 3])]:
            local_tokens = (num_tokens + 1) // 2
            start = rank * local_tokens
            global_rows = torch.arange(local_tokens * 2 * 2048, dtype=torch.int32, device="npu").reshape(-1, 2048)
            global_rows[:, -1] = -1
            indices = torch.tensor(sample_ids, dtype=torch.int32, device="npu")
            model = torch.nn.Module()
            model.topk_indices_buffer = torch.full_like(global_rows, -99)
            model.add_module("attention", torch.nn.Module())
            model.attention.topk_indices_buffer = model.topk_indices_buffer
            buffer = model.topk_indices_buffer
            count = max(0, min(local_tokens, len(sample_ids) - start))

            for _ in range(3):
                buffer[:local_tokens].copy_(global_rows[start : start + local_tokens])
                compact_mtp_topk_indices(model, indices, num_tokens, group)
            expected = global_rows[indices][start : start + count]
            torch.testing.assert_close(buffer[:count], expected, atol=0, rtol=0)

            # Capture the real HCCL operation and replay with changed source
            # values and sampling positions, preserving all input addresses.
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            buffer[:local_tokens].copy_(global_rows[start : start + local_tokens])
            with torch.npu.graph(graph):
                compact_mtp_topk_indices(model, indices, num_tokens, group)
            for offset in (100, 200):
                global_rows.add_(offset)
                indices.copy_(torch.tensor(list(reversed(sample_ids)), dtype=torch.int32, device="npu"))
                buffer[:local_tokens].copy_(global_rows[start : start + local_tokens])
                graph.replay()
                expected = global_rows[indices][start : start + count]
                torch.testing.assert_close(buffer[:count], expected, atol=0, rtol=0)
        result_queue.put(None)
    except Exception:
        result_queue.put(traceback.format_exc())
    finally:
        if group is not None:
            group.destroy()
        destroy_distributed_environment()


def test_mtp_index_cache_dsa_cp_eager_and_graph():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = 29_501 + random.randint(0, 10_000)
    processes = [context.Process(target=_worker, args=(rank, port, result_queue)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=300) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        assert results == [None, None], "\n".join(result for result in results if result is not None)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
