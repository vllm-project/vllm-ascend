#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Multicard smoke test for the zero-buffer SHMEM runtime wrapper.

Verifies that on a real A3 box with Ascend SHMEM available:
  - ``ZbMoERuntime.init`` brings up ``aclshmemx_init_attr`` on every rank.
  - ``alloc`` returns a non-zero ``ext_info`` metadata pointer.
  - ``allocate_low_latency_tensors`` lays out the SHMEM tensor bundle for both
    the non-quant (BF16) and quant (BF16 base + INT8 alias + FP32 scales) paths,
    matching deepep_standalone's ``preallocate_lowlatency_shmem_tensors``.
  - ``alias_tensor`` actually points at the base tensor's data buffer.
  - ``finalize`` tears the runtime down without errors.

The test is intentionally HCCL-free; SHMEM is the only inter-rank dependency.
Requires:
  - Ascend SHMEM installed at /usr/local/Ascend/shmem/latest so the
    zero-buffer ops and SHMEM runtime bindings are present.
  - a reachable SHMEM control endpoint exported via
    ``VLLM_ASCEND_ZB_SHMEM_URI`` (default ``tcp://127.0.0.1:29555``).
"""

from __future__ import annotations

import os

import torch
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.ops.fused_moe.zb_runtime import (
    LowLatencyZbTensors,
    ZbMoERuntime,
    estimate_local_mem_size,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _shmem_server_ipport() -> str:
    raw = os.environ.get("VLLM_ASCEND_ZB_SHMEM_URI", "tcp://127.0.0.1:29555")
    # Ascend SHMEM expects the transport prefix in aclshmemx_init_attr.
    return raw if "://" in raw else f"tcp://{raw}"


def _worker(rank: int, world_size: int, results) -> None:
    try:
        torch_npu.npu.set_device(rank)

        max_recv_tokens = 1024
        hidden_size = 2048
        local_mem_size = estimate_local_mem_size(
            max_recv_tokens,
            hidden_size,
            use_quant=True,
        )
        runtime = ZbMoERuntime(
            rank=rank,
            world_size=world_size,
            server_ip_port=_shmem_server_ipport(),
            local_mem_size=local_mem_size,
        )

        actual_rank = runtime.init()
        assert actual_rank == rank, f"rank mismatch from aclshmem_my_pe: expected {rank}, got {actual_rank}"
        assert runtime.is_initialized(), "SHMEM runtime reports uninitialised"

        ext_info = runtime.alloc_ext_info()
        assert ext_info != 0, "ext_info metadata buffer must be non-zero"
        assert runtime.get_ext_info() == ext_info, "ext_info inconsistent across getter"

        device = f"npu:{rank}"

        # Non-quant path: combine_x and expand_x_out are independent BF16 SHMEM
        # tensors; dynamic_scales_out is absent.
        bundle = runtime.allocate_low_latency_tensors(
            max_recv_tokens=max_recv_tokens,
            hidden_size=hidden_size,
            device=device,
            use_quant=False,
        )
        assert isinstance(bundle, LowLatencyZbTensors)
        assert bundle.combine_x.dtype == torch.bfloat16
        assert bundle.combine_x.shape == (1024, 2048)
        assert bundle.expand_x_out.dtype == torch.bfloat16
        assert bundle.expand_x_out.shape == (1024, 2048)
        assert bundle.dynamic_scales_out is None
        assert bundle.combine_x.data_ptr() != bundle.expand_x_out.data_ptr(), (
            "non-quant expand_x_out must own its own SHMEM allocation"
        )

        # Quant path: expand_x_out aliases combine_x as INT8 over the same SHMEM
        # buffer; dynamic_scales_out is a separate FP32 SHMEM allocation.
        qbundle = runtime.allocate_low_latency_tensors(
            max_recv_tokens=max_recv_tokens,
            hidden_size=hidden_size,
            device=device,
            use_quant=True,
        )
        assert qbundle.combine_x.dtype == torch.bfloat16
        assert qbundle.expand_x_out.dtype == torch.int8
        assert qbundle.expand_x_out.shape == (1024, 2048)
        assert qbundle.dynamic_scales_out is not None
        assert qbundle.dynamic_scales_out.dtype == torch.float32
        assert qbundle.dynamic_scales_out.shape == (1024,)
        assert qbundle.combine_x.data_ptr() == qbundle.expand_x_out.data_ptr(), (
            "quant expand_x_out must alias combine_x's SHMEM buffer"
        )

        # Standalone alias check: same data_ptr, smaller logical shape and a
        # different dtype must be accepted as long as alias bytes <= base bytes.
        alias = runtime.alias_tensor(qbundle.combine_x, [1024, 1024], torch.int8)
        assert alias.data_ptr() == qbundle.combine_x.data_ptr()
        assert alias.dtype == torch.int8
        assert alias.shape == (1024, 1024)

        runtime.finalize()
        assert not runtime.is_initialized(), "runtime not reset by finalize"

        results.put((rank, True, None))
    except Exception as exc:  # pragma: no cover - reported through queue
        results.put((rank, False, repr(exc)))


@torch.inference_mode()
def test_zb_runtime_smoke() -> None:
    # zb_* symbols must exist. Without Ascend SHMEM at build time the
    # dispatch/combine ops would be missing, but the runtime bindings themselves
    # should still be there when SHMEM is linked.
    for op_name in ("zb_init", "zb_alloc", "zb_alloc_tensor", "zb_alias_tensor", "zb_finalize", "zb_is_initialized"):
        if not hasattr(torch.ops._C_ascend, op_name):
            raise AssertionError(f"torch.ops._C_ascend.{op_name} missing; rebuild vllm_ascend_C against Ascend SHMEM")

    world_size = int(os.environ.get("VLLM_ASCEND_ZB_TEST_WORLD_SIZE", "8"))
    mp.set_start_method("fork", force=True)

    results: mp.SimpleQueue = mp.SimpleQueue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_worker, args=(rank, world_size, results))
        p.start()
        processes.append(p)

    statuses = [results.get() for _ in range(world_size)]
    for p in processes:
        p.join()

    failures = [(r, msg) for r, ok, msg in statuses if not ok]
    assert not failures, f"SHMEM runtime smoke failures: {failures}"
