# SPDX-License-Identifier: Apache-2.0

import random
import traceback
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
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend, IndexerCacheInputs


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
            group_name="sfa_cache_gather_test",
            use_device_communicator=True,
        )
        impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
        impl.enable_sparse_sfa_c8 = True
        impl._all_gather_o_proj_full_weight = lambda: None
        indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        torch.nn.Module.__init__(indexer)
        indexer.enable_sparse_li_c8 = True
        indexer._use_c8_reshape_optim = lambda: False

        for num_tokens in (1, 5, 33):
            # Include ranks containing only padding and a partial final shard.
            local_tokens = (num_tokens + world_size - 1) // world_size
            padded_tokens = local_tokens * world_size
            data = torch.arange(padded_tokens * 784, device="npu", dtype=torch.int32).reshape(padded_tokens, 784)
            data = data.to(torch.int8)
            scales = (torch.arange(padded_tokens, device="npu", dtype=torch.float32) / 7 + 0.01).half().unsqueeze(-1)
            local_data = data[rank * local_tokens : (rank + 1) * local_tokens]
            local_scale = scales[rank * local_tokens : (rank + 1) * local_tokens]
            main_cache = torch.empty((128, 656), device="npu", dtype=torch.int8)
            key_cache = torch.empty((128, 128), device="npu", dtype=torch.int8)
            scale_cache = torch.empty((128, 1), device="npu", dtype=torch.float16)
            indexer.k_cache = SimpleNamespace(kv_cache=(key_cache, scale_cache))
            slots = torch.arange(padded_tokens, device="npu", dtype=torch.int32)
            slots[num_tokens:] = -1
            metadata = SimpleNamespace(num_actual_tokens=num_tokens)

            def forward(
                local_data=local_data,
                local_scale=local_scale,
                main_cache=main_cache,
                slots=slots,
                metadata=metadata,
            ):
                cache_inputs = IndexerCacheInputs(local_data[:, 656:], local_scale, local_scale)
                pe, nope, main_scale = local_data[:, 512:640], local_data[:, :512], local_data[:, 640:656]
                packed, handles = impl._prepare_kv_for_parallel(pe, nope, main_scale, True, cache_inputs)
                impl._store_parallel_kv(
                    pe,
                    nope,
                    main_scale,
                    packed,
                    handles,
                    (main_cache,),
                    slots,
                    metadata,
                    True,
                    cache_inputs,
                )
                indexer.write_cache(cache_inputs.key, cache_inputs.scale, slots)

            def check(
                main_cache=main_cache,
                num_tokens=num_tokens,
                data=data,
                key_cache=key_cache,
                padded_tokens=padded_tokens,
                scale_cache=scale_cache,
                scales=scales,
            ):
                torch.testing.assert_close(main_cache[:num_tokens], data[:num_tokens, :656], atol=0, rtol=0)
                torch.testing.assert_close(key_cache[:num_tokens], data[:num_tokens, 656:], atol=0, rtol=0)
                torch.testing.assert_close(scale_cache[:num_tokens], scales[:num_tokens], atol=0, rtol=0)

            with patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=group):
                for _ in range(3):
                    forward()
                check()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    forward()
                # Keep captured addresses fixed while changing every rank's data.
                for increment in (1, 3):
                    data.add_(increment)
                    scales.add_(increment)
                    graph.replay()
                    check()
        result_queue.put(None)
    except Exception:
        result_queue.put(traceback.format_exc())
    finally:
        if group is not None:
            group.destroy()
        destroy_distributed_environment()


@pytest.mark.parametrize("world_size", [2, 8])
def test_sfa_fused_cache_gather_eager_and_graph(world_size):
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
