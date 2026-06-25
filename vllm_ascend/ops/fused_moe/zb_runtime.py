# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_ascend.utils import enable_custom_op

logger = logging.getLogger(__name__)

# Per-rank SHMEM heap passed to aclshmemx_init_attr(local_mem_size). All subsequent
# aclshmem_malloc / aclshmemx_calloc allocations are carved from this pool.
DEFAULT_LOCAL_MEM_SIZE = 4 * 1024 * 1024 * 1024

# Control/metadata buffer (ext_info / gva_ptr) size. Matches deepep_standalone's
# SHMEM_META_DATA_SIZE = 1 MiB. This is NOT tensor payload; kernels only need a
# small metainfo region, typically well under 2 MiB.
DEFAULT_EXT_INFO_BYTES = 1 * 1024 * 1024

# Small fixed bookkeeping inside the SHMEM pool (see deepep_standalone fixed_bytes).
DEFAULT_ZB_FIXED_OVERHEAD_BYTES = 64 * 1024

# Extra headroom for allocator alignment / runtime metadata beyond tensor payloads.
DEFAULT_ZB_POOL_SLACK_BYTES = 32 * 1024 * 1024

ZB_POOL_ALIGN_BYTES = 2 * 1024 * 1024

DEFAULT_COMM_ALG = "fullmesh_v1"

# Process-wide runtime created by ensure_zb_process_initialized().
_ZB_PROCESS_RUNTIME: ZbMoERuntime | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _round_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def estimate_low_latency_tensor_bytes(
    max_recv_tokens: int,
    hidden_size: int,
    *,
    use_quant: bool = False,
) -> int:
    """Bytes required by SHMEM-backed low-latency data tensors only."""
    if max_recv_tokens <= 0 or hidden_size <= 0:
        raise ValueError("max_recv_tokens and hidden_size must be positive")

    combine_x_bytes = max_recv_tokens * hidden_size * 2  # BF16
    if use_quant:
        # expand_x_out aliases combine_x; only dynamic_scales_out is extra.
        return combine_x_bytes + max_recv_tokens * 4
    # Non-quant path owns separate BF16 combine_x and expand_x_out buffers.
    return combine_x_bytes * 2


def compute_low_latency_max_recv_tokens(
    num_tokens_per_rank: int,
    ep_world_size: int,
    num_local_experts: int,
    *,
    max_tokens_per_rank: int | None = None,
    min_recv_tokens: int = 1024,
) -> int:
    """Mirror deepep_standalone's low-latency max_recv_tokens sizing."""
    per_rank = max(num_tokens_per_rank, max_tokens_per_rank or 0)
    calculated = per_rank * ep_world_size * num_local_experts
    return max(calculated, min_recv_tokens)


def estimate_local_mem_size(
    max_recv_tokens: int,
    hidden_size: int,
    *,
    use_quant: bool = False,
    ext_info_bytes: int | None = None,
    moe_expert_num: int | None = None,
    ep_world_size: int | None = None,
    fixed_overhead_bytes: int | None = None,
    pool_slack_bytes: int | None = None,
) -> int:
    """Estimate aclshmemx_init_attr(local_mem_size) for the ZB low-latency path.

    Components:
      - ``ext_info`` metainfo (default 1 MiB, configurable)
      - ``combine_x`` / ``expand_x_out`` / optional ``dynamic_scales_out``
      - small fixed bookkeeping (optional, scales with expert layout)
      - pool slack for alignment/runtime overhead
    """
    ext_bytes = (
        ext_info_bytes
        if ext_info_bytes is not None
        else _env_int("VLLM_ASCEND_ZB_EXT_INFO_BYTES", DEFAULT_EXT_INFO_BYTES)
    )
    if ext_bytes <= 0:
        raise ValueError("ext_info_bytes must be positive")

    data_bytes = estimate_low_latency_tensor_bytes(
        max_recv_tokens,
        hidden_size,
        use_quant=use_quant,
    )

    fixed_bytes = fixed_overhead_bytes
    if fixed_bytes is None:
        if moe_expert_num is not None and ep_world_size is not None:
            # deepep_standalone: E * 4 + R * E * 4
            fixed_bytes = moe_expert_num * 4 + ep_world_size * moe_expert_num * 4
        else:
            fixed_bytes = DEFAULT_ZB_FIXED_OVERHEAD_BYTES

    slack_bytes = (
        pool_slack_bytes
        if pool_slack_bytes is not None
        else _env_int("VLLM_ASCEND_ZB_POOL_SLACK_BYTES", DEFAULT_ZB_POOL_SLACK_BYTES)
    )

    override = os.getenv("VLLM_ASCEND_ZB_LOCAL_MEM_SIZE")
    if override:
        return int(override)

    total = ext_bytes + data_bytes + fixed_bytes + slack_bytes
    return _round_up(total, ZB_POOL_ALIGN_BYTES)


def estimate_zb_early_local_mem_size(
    *,
    max_tokens_per_rank: int,
    ep_world_size: int,
    hidden_size: int,
    moe_expert_num: int,
    use_quant: bool = False,
) -> tuple[int, int]:
    """Conservative aclshmem pool sizing for early (startup) init.

    Returns ``(local_mem_size, max_recv_tokens)``.
    """
    if moe_expert_num <= 0 or hidden_size <= 0:
        raise ValueError("moe_expert_num and hidden_size must be positive")
    if moe_expert_num % ep_world_size != 0:
        raise ValueError(f"moe_expert_num={moe_expert_num} must be divisible by ep_world_size={ep_world_size}")
    num_local_experts = moe_expert_num // ep_world_size
    max_recv_tokens = compute_low_latency_max_recv_tokens(
        num_tokens_per_rank=max_tokens_per_rank,
        ep_world_size=ep_world_size,
        num_local_experts=num_local_experts,
    )
    local_mem_size = estimate_local_mem_size(
        max_recv_tokens,
        hidden_size,
        use_quant=use_quant,
        moe_expert_num=moe_expert_num,
        ep_world_size=ep_world_size,
    )
    return local_mem_size, max_recv_tokens


def get_zb_process_runtime() -> ZbMoERuntime | None:
    return _ZB_PROCESS_RUNTIME


def ensure_zb_process_initialized(
    rank: int,
    world_size: int,
    local_mem_size: int,
    server_ip_port: str,
) -> ZbMoERuntime:
    """Bring up aclshmem once per process (mirrors MC2 HCCL init at worker startup)."""
    global _ZB_PROCESS_RUNTIME
    if _ZB_PROCESS_RUNTIME is not None:
        if not _ZB_PROCESS_RUNTIME.is_initialized():
            raise RuntimeError("ZB SHMEM process runtime exists but aclshmem is not initialized.")
        return _ZB_PROCESS_RUNTIME

    if not server_ip_port:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_ZB=1 but VLLM_ASCEND_ZB_SHMEM_URI is unset. "
            "Set it to e.g. tcp://<host>:<port> (identical across all EP ranks)."
        )

    runtime = ZbMoERuntime(
        rank=rank,
        world_size=world_size,
        server_ip_port=server_ip_port,
        local_mem_size=local_mem_size,
    )
    runtime.init()
    _ZB_PROCESS_RUNTIME = runtime
    return runtime


@dataclass
class LowLatencyZbTensors:
    combine_x: torch.Tensor
    expand_x_out: torch.Tensor
    dynamic_scales_out: torch.Tensor | None


def _ensure_custom_op_loaded() -> None:
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C custom ops are not available; cannot use zero-buffer SHMEM runtime")


def _ensure_zb_op_available(op_name: str) -> None:
    _ensure_custom_op_loaded()
    if not hasattr(torch.ops._C_ascend, op_name):
        raise RuntimeError(
            f"torch.ops._C_ascend.{op_name} is not registered. Install Ascend SHMEM at "
            "/usr/local/Ascend/shmem/latest and rebuild vllm_ascend."
        )


@dataclass
class ZbMoERuntime:
    rank: int
    world_size: int
    server_ip_port: str | None = None
    local_mem_size: int = DEFAULT_LOCAL_MEM_SIZE
    ext_info: int = 0

    def __post_init__(self) -> None:
        if self.server_ip_port is None:
            self.server_ip_port = os.getenv("VLLM_ASCEND_ZB_SHMEM_URI", "")

    def init(self) -> int:
        _ensure_custom_op_loaded()
        actual_rank = torch.ops._C_ascend.zb_init(
            self.rank,
            self.world_size,
            self.local_mem_size,
            self.server_ip_port,
        )
        self.rank = int(actual_rank)
        return self.rank

    def alloc(self, element_count: int, element_size: int = 1) -> int:
        _ensure_custom_op_loaded()
        self.ext_info = int(torch.ops._C_ascend.zb_alloc(element_count, element_size))
        return self.ext_info

    def alloc_ext_info(self, nbytes: int | None = None) -> int:
        """Allocate the ZB metainfo/control buffer (``ext_info`` / ``gva_ptr``).

        deepep_standalone uses ``SHMEM_META_DATA_SIZE = 1 MiB`` via
        ``aclshmemx_calloc(SHMEM_META_DATA_SIZE / 4, 4)``.
        """
        raw_bytes = nbytes if nbytes is not None else _env_int("VLLM_ASCEND_ZB_EXT_INFO_BYTES", DEFAULT_EXT_INFO_BYTES)
        if raw_bytes <= 0:
            raise ValueError("ext_info nbytes must be positive")
        if raw_bytes % 4 != 0:
            raise ValueError("ext_info nbytes must be a multiple of 4")
        return self.alloc(raw_bytes // 4, 4)

    def alloc_tensor(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Allocate a SHMEM-backed NPU tensor.

        Tensor data buffers are independent from ``ext_info``. The latter is
        the metadata/control buffer allocated by :meth:`alloc` and passed to
        zero-buffer kernels.
        """
        _ensure_custom_op_loaded()
        return torch.ops._C_ascend.zb_alloc_tensor(list(shape), dtype, str(device))

    def alias_tensor(self, base: torch.Tensor, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Create a tensor view with a different dtype over a SHMEM tensor buffer."""
        _ensure_custom_op_loaded()
        return torch.ops._C_ascend.zb_alias_tensor(base, list(shape), dtype)

    def allocate_low_latency_tensors(
        self,
        max_recv_tokens: int,
        hidden_size: int,
        device: torch.device | str,
        *,
        use_quant: bool = False,
    ) -> LowLatencyZbTensors:
        """Allocate the SHMEM tensors required by the zero-buffer low-latency path.

        This mirrors deepep_standalone's ``preallocate_lowlatency_shmem_tensors``:
        ``combine_x`` is always BF16; ``expand_x_out`` aliases it as INT8 when
        quantization is enabled, otherwise it owns a separate BF16 SHMEM buffer.
        ``dynamic_scales_out`` exists only for the quantized path.
        """
        combine_x = self.alloc_tensor([max_recv_tokens, hidden_size], torch.bfloat16, device)
        if use_quant:
            expand_x_out = self.alias_tensor(combine_x, [max_recv_tokens, hidden_size], torch.int8)
            dynamic_scales_out = self.alloc_tensor([max_recv_tokens], torch.float32, device)
        else:
            expand_x_out = self.alloc_tensor([max_recv_tokens, hidden_size], torch.bfloat16, device)
            dynamic_scales_out = None
        return LowLatencyZbTensors(
            combine_x=combine_x,
            expand_x_out=expand_x_out,
            dynamic_scales_out=dynamic_scales_out,
        )

    def get_ext_info(self) -> int:
        _ensure_custom_op_loaded()
        self.ext_info = int(torch.ops._C_ascend.zb_get_ext_info())
        return self.ext_info

    def free(self, ptr: int | None = None) -> None:
        _ensure_custom_op_loaded()
        raw_ptr = self.ext_info if ptr is None else ptr
        if raw_ptr:
            torch.ops._C_ascend.zb_free(raw_ptr)
        if raw_ptr == self.ext_info:
            self.ext_info = 0

    def finalize(self) -> None:
        _ensure_custom_op_loaded()
        torch.ops._C_ascend.zb_finalize()
        self.ext_info = 0

    def is_initialized(self) -> bool:
        _ensure_custom_op_loaded()
        return bool(torch.ops._C_ascend.zb_is_initialized())

    def __enter__(self) -> ZbMoERuntime:
        self.init()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finalize()


def zb_moe_distribute_dispatch_zero_buffer(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expand_x_out: torch.Tensor,
    dynamic_scales_out: torch.Tensor,
    assist_info_for_combine_out: torch.Tensor,
    expert_token_nums_out: torch.Tensor,
    ep_recv_count_out: torch.Tensor,
    tp_recv_count_out: torch.Tensor,
    *,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    ext_info: int,
    scales: torch.Tensor | None = None,
    x_active_mask: torch.Tensor | None = None,
    elastic_info: torch.Tensor | None = None,
    tp_world_size: int = 1,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 0,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    global_bs: int = 0,
    expert_token_nums_type: int = 1,
    comm_alg: str = DEFAULT_COMM_ALG,
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin Python wrapper around torch.ops._C_ascend.zb_moe_distribute_dispatch_zero_buffer.

    All output tensors must be pre-allocated by the caller. ``ext_info`` is the
    SHMEM global virtual address returned by :py:meth:`ZbMoERuntime.alloc` /
    :py:meth:`ZbMoERuntime.get_ext_info`.
    """
    _ensure_zb_op_available("zb_moe_distribute_dispatch_zero_buffer")
    return torch.ops._C_ascend.zb_moe_distribute_dispatch_zero_buffer(
        x,
        expert_ids,
        scales,
        x_active_mask,
        elastic_info,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        expert_token_nums_type,
        ext_info,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
        expand_x_out,
        dynamic_scales_out,
        assist_info_for_combine_out,
        expert_token_nums_out,
        ep_recv_count_out,
        tp_recv_count_out,
    )


def zb_moe_distribute_combine_zero_buffer(
    expand_x: torch.Tensor,
    expert_ids: torch.Tensor,
    assist_info_for_combine: torch.Tensor,
    ep_send_count: torch.Tensor,
    expert_scales: torch.Tensor,
    combined_x: torch.Tensor,
    *,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    ext_info: int,
    tp_send_count: torch.Tensor | None = None,
    x_active_mask: torch.Tensor | None = None,
    activation_scale: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    group_list: torch.Tensor | None = None,
    expand_scales: torch.Tensor | None = None,
    shared_expert_x: torch.Tensor | None = None,
    elastic_info: torch.Tensor | None = None,
    ori_x: torch.Tensor | None = None,
    const_expert_alpha1: torch.Tensor | None = None,
    const_expert_alpha2: torch.Tensor | None = None,
    const_expert_v: torch.Tensor | None = None,
    tp_world_size: int = 1,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 0,
    shared_expert_rank_num: int = 0,
    global_bs: int = 0,
    out_dtype: int = 0,
    comm_quant_mode: int = 0,
    group_list_type: int = 0,
    comm_alg: str = DEFAULT_COMM_ALG,
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
) -> torch.Tensor:
    """Thin Python wrapper around torch.ops._C_ascend.zb_moe_distribute_combine_zero_buffer.

    ``combined_x`` must be pre-allocated by the caller. ``ext_info`` is the
    SHMEM global virtual address used as the combine source buffer pointer
    (typically the same value used by ``zb_moe_distribute_dispatch_zero_buffer``).
    """
    _ensure_zb_op_available("zb_moe_distribute_combine_zero_buffer")
    if tp_send_count is None:
        # The combine tiling marks tp_send_count optional, but still validates
        # its shape and dtype. deepep_standalone always passes an int32 tensor
        # with one entry per TP rank even when tp_world_size == 1.
        tp_send_count = torch.empty((tp_world_size,), dtype=torch.int32, device=expand_x.device)
    return torch.ops._C_ascend.zb_moe_distribute_combine_zero_buffer(
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_count,
        expert_scales,
        tp_send_count,
        x_active_mask,
        activation_scale,
        weight_scale,
        group_list,
        expand_scales,
        shared_expert_x,
        elastic_info,
        ori_x,
        const_expert_alpha1,
        const_expert_alpha2,
        const_expert_v,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        global_bs,
        out_dtype,
        comm_quant_mode,
        ext_info,
        group_list_type,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
        combined_x,
    )


def zb_moe_grouped_matmul_gmm2_out(
    x: torch.Tensor,
    weight: list[torch.Tensor],
    group_list: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: list[torch.Tensor] | None = None,
    per_token_scale: list[torch.Tensor] | None = None,
    bias: list[torch.Tensor] | None = None,
    split_item: int = 2,
    group_type: int = 0,
    group_list_type: int = 0,
    act_type: int = 0,
) -> torch.Tensor:
    """ZB-only gmm2: write grouped matmul output into preallocated SHMEM ``out``."""
    _ensure_zb_op_available("zb_moe_grouped_matmul_gmm2_out")
    return torch.ops._C_ascend.zb_moe_grouped_matmul_gmm2_out(
        x,
        weight,
        scale,
        per_token_scale,
        bias,
        group_list,
        out,
        split_item,
        group_type,
        group_list_type,
        act_type,
    )
