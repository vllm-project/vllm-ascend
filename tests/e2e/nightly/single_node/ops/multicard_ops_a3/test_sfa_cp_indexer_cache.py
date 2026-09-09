# SPDX-License-Identifier: Apache-2.0

import itertools
import random
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
import torch_npu
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend


@torch.inference_mode()
def _check_cache(rank, sfa_c8, li_c8, is_mtp):
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.has_indexer = True
    impl._is_mtp_layer = is_mtp
    impl.skip_topk = True
    impl.enable_sparse_sfa_c8 = sfa_c8
    impl.enable_sparse_li_c8 = li_c8
    impl.qk_rope_head_dim = 64
    impl.kv_lora_rank = 512
    impl.head_dim = 128
    dtype = torch.int8 if sfa_c8 else torch.bfloat16
    rows = torch.arange(4, device="npu").to(dtype)[:, None]
    k_pe = rows.expand(4, 64).contiguous()
    k_nope = (rows + 4).expand(4, 512).contiguous()
    scale = torch.ones((4, 8), device="npu", dtype=dtype)
    indexer_dtype = torch.int8 if li_c8 else torch.bfloat16
    k_li = torch.arange(4, device="npu").to(indexer_dtype)[:, None].expand(4, 128).contiguous()
    k_li_scale = torch.ones((4, 1), device="npu", dtype=torch.float16)
    local = slice(rank * 2, rank * 2 + 2)
    fused_kv, handles = impl._prepare_kv_for_parallel(
        k_pe[local],
        k_nope[local],
        scale[local],
        False,
    )
    cache: tuple[torch.Tensor, ...]
    if sfa_c8:
        cache = (torch.zeros((1, 128, 1, fused_kv.shape[-1]), device="npu", dtype=dtype),)
    else:
        cache = (
            torch.zeros((1, 128, 1, 512), device="npu", dtype=dtype),
            torch.zeros((1, 128, 1, 64), device="npu", dtype=dtype),
        )
    slots = torch.tensor([1, 3, 5, 7], device="npu", dtype=torch.int64)
    impl._store_parallel_kv(
        k_pe[local],
        k_nope[local],
        scale[local],
        fused_kv,
        handles,
        cache,
        slots,
        SimpleNamespace(num_actual_tokens=4),
        False,
    )
    expected_parts = [k_nope, k_pe, scale] if sfa_c8 else [k_pe, k_nope]
    torch.testing.assert_close(fused_kv, torch.cat(expected_parts, dim=-1), atol=0, rtol=0)
    if sfa_c8:
        torch.testing.assert_close(cache[0].view(128, -1)[slots], fused_kv, atol=0, rtol=0)
    else:
        torch.testing.assert_close(cache[0].view(128, 512)[slots], k_nope, atol=0, rtol=0)
        torch.testing.assert_close(cache[1].view(128, 64)[slots], k_pe, atol=0, rtol=0)
    # Indexer communication and cache writes now belong to the indexer backend.
    # Run its real gather/write path for MTP, including compute_topk=False.
    if impl.runtime_has_indexer:
        indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        torch.nn.Module.__init__(indexer)
        indexer.enable_sparse_li_c8 = li_c8
        indexer._pcp_active = False
        indexer._dsa_cp_active = True
        indexer_cache: tuple[torch.Tensor, ...] = (torch.zeros((1, 128, 1, 128), device="npu", dtype=indexer_dtype),)
        if li_c8:
            indexer_cache += (torch.zeros((1, 128, 1, 1), device="npu", dtype=torch.float16),)
        indexer.k_cache = SimpleNamespace(kv_cache=indexer_cache)
        with (
            patch.object(indexer, "forward_k", return_value=(k_li[local], k_li_scale[local] if li_c8 else None)),
            patch.object(indexer, "_use_c8_reshape_optim", return_value=False),
        ):
            result = indexer(
                k_li[local],
                k_li[local],
                k_pe[local],
                k_pe[local],
                k_li[local],
                SimpleNamespace(slot_mapping=slots),
                compute_topk=False,
            )
        assert result is None
        torch.testing.assert_close(indexer_cache[0].view(128, 128)[slots], k_li, atol=0, rtol=0)
        if li_c8:
            torch.testing.assert_close(indexer_cache[1].view(128, 1)[slots], k_li_scale, atol=0, rtol=0)
    else:
        assert not is_mtp


def _worker(rank, port):
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
            group_name="sfa_cp_indexer_test",
        )
        # Only select the test group; collectives and NPU cache writes are real.
        with (
            patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=group),
            patch("vllm_ascend.attention.indexer.get_tp_group", return_value=group),
        ):
            for sfa_c8, li_c8, is_mtp in itertools.product((False, True), repeat=3):
                _check_cache(rank, sfa_c8, li_c8, is_mtp)
    finally:
        if group is not None:
            group.destroy()
        destroy_distributed_environment()


def test_dsa_cp_runtime_indexer_cache_on_npu():
    port = 29_501 + random.randint(0, 10_000)
    processes = mp.spawn(_worker, args=(port,), nprocs=2, join=False)
    deadline = time.monotonic() + 300
    try:
        while not processes.join(timeout=30):
            assert time.monotonic() < deadline, "DSA-CP cache test timed out"
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
