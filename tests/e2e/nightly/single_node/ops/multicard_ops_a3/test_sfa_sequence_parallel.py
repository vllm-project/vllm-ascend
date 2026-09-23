# SPDX-License-Identifier: Apache-2.0

import random
import traceback
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
import torch_npu
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl


def _check_sequence_parallel_projection(impl, group, rank, world_size):
    impl.enable_dsa_cp_full_o_proj = True
    impl.o_proj = SimpleNamespace(reduce_results=False)
    impl._use_full_o_proj_weights = nullcontext
    weight = (torch.arange(256, device="npu").reshape(16, 16) % 13).to(torch.bfloat16) / 16
    impl._apply_o_proj_full_weight = lambda hidden: hidden @ weight
    for tokens in (1, 7, 8, 9, 33, 8192):
        local_tokens = (tokens + world_size - 1) // world_size
        padded_tokens = local_tokens * world_size
        hidden = torch.arange(padded_tokens * 16, device="npu").reshape(padded_tokens, 16)
        hidden = (hidden % 31).to(torch.bfloat16) / 32
        hidden[tokens:].zero_()
        local = hidden[rank * local_tokens : (rank + 1) * local_tokens].contiguous()
        # Existing decoder/attention contract: gather, slice, project into a
        # sparse replicated buffer, then pad and reduce-scatter.
        replicated = group.all_gather(local, dim=0)[:tokens]
        prepared = torch.nn.functional.pad(replicated, (0, 0, 0, padded_tokens - tokens))
        prepared = prepared[rank * local_tokens : (rank + 1) * local_tokens]
        reference_buffer = torch.empty((tokens, 16), device="npu", dtype=torch.bfloat16)
        impl._finalize_o_proj(prepared, reference_buffer, True)
        reference_buffer = torch.nn.functional.pad(reference_buffer, (0, 0, 0, padded_tokens - tokens))
        expected = group.reduce_scatter(reference_buffer, dim=0)
        actual = torch.empty_like(local)
        impl._finalize_o_proj(local, actual, True, output_is_sequence_parallel=True)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def _worker(rank, world_size, port, result_queue):
    group = None
    try:
        torch_npu.npu.set_device(rank)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="hccl",
        )
        group = init_model_parallel_group(
            [list(range(world_size))],
            local_rank=rank,
            backend="hccl",
            group_name="sfa_sequence_parallel_test",
            use_device_communicator=True,
        )
        impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
        with patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=group):
            _check_sequence_parallel_projection(impl, group, rank, world_size)
        result_queue.put(None)
    except Exception:
        result_queue.put(traceback.format_exc())
    finally:
        if group is not None:
            group.destroy()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 8])
def test_sfa_sequence_parallel_projection(world_size):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = 29_501 + random.randint(0, 10_000)
    processes = [
        context.Process(target=_worker, args=(rank, world_size, port, result_queue)) for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=300) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        assert results == [None] * world_size, "\n".join(result for result in results if result is not None)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
