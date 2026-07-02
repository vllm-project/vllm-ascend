# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_ascend.ascend_config import get_ascend_config
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

# aclshmem conf-store URI reserved at worker startup (one free port per MC2 group).
_ZB_SHMEM_CONF_STORE_URI: str | None = None


def set_zb_shmem_conf_store_uri(uri: str | None) -> None:
    """Record the aclshmem conf-store URI reserved for this worker/MC2 group."""
    global _ZB_SHMEM_CONF_STORE_URI
    _ZB_SHMEM_CONF_STORE_URI = uri


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
    experts_per_token: int,
    max_tokens_per_rank: int | None = None,
    min_recv_tokens: int = 1024,
) -> int:
    """Mirror deepep_standalone's low-latency max_recv_tokens sizing, aligned with MC2 tiling."""
    per_rank = max(num_tokens_per_rank, max_tokens_per_rank or 0)
    global_bs = per_rank * ep_world_size
    calculated = global_bs * min(num_local_experts, experts_per_token)
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
    experts_per_token: int,
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
        experts_per_token=experts_per_token,
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


def _format_tcp_uri(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"tcp://[{host}]:{port}"
    return f"tcp://{host}:{port}"


def _parse_tcp_host_port(init_method: str) -> tuple[str, int]:
    normalized = init_method.strip()
    if "://" not in normalized:
        normalized = f"tcp://{normalized}"
    scheme, _, remainder = normalized.partition("://")
    if scheme != "tcp":
        raise RuntimeError(f"ZB SHMEM requires a tcp:// HCCL rendezvous, got: {init_method!r}")
    if remainder.startswith("["):
        end = remainder.index("]")
        host = remainder[1:end]
        _, _, port_str = remainder[end + 1 :].partition(":")
    else:
        host, _, port_str = remainder.rpartition(":")
    if not host or not port_str:
        raise RuntimeError(f"cannot parse host/port from HCCL rendezvous: {init_method!r}")
    return host, int(port_str)


def reserve_zb_shmem_conf_store_uri(hccl_init_method: str) -> str:
    """Reserve a free TCP port for aclshmem conf-store (shared within each MC2 group).

    Reuses the HCCL rendezvous host but picks a dedicated port via ``get_open_port()``,
    broadcast from the MC2 group leader so every rank in the group shares the same URI
    without colliding with the HCCL TCPStore port.
    """
    from vllm.utils.network_utils import get_open_port

    from vllm_ascend.distributed.parallel_state import get_mc2_group

    host, _ = _parse_tcp_host_port(hccl_init_method)
    mc2 = get_mc2_group()
    port_tensor = torch.tensor([0], dtype=torch.int64)

    if mc2.rank_in_group == 0:
        port_tensor[0] = get_open_port()

    torch.distributed.broadcast(
        port_tensor,
        src=mc2.ranks[0],
        group=mc2.cpu_group,
    )
    shmem_port = int(port_tensor.item())
    if shmem_port <= 0 or shmem_port > 65535:
        raise RuntimeError(f"invalid SHMEM conf-store port broadcast result: {shmem_port}")

    uri = _format_tcp_uri(host, shmem_port)
    set_zb_shmem_conf_store_uri(uri)
    logger.info(
        "Reserved ZB SHMEM conf-store URI %s (HCCL rendezvous %s)",
        uri,
        hccl_init_method,
    )
    return uri


def _iter_zb_hf_configs(vllm_config):
    model_config = vllm_config.model_config
    for cfg in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        if cfg is not None:
            yield cfg


def resolve_zb_moe_expert_num(vllm_config) -> int:
    get_num_experts = getattr(vllm_config.model_config, "get_num_experts", None)
    if callable(get_num_experts):
        num_experts = int(get_num_experts() or 0)
        if num_experts > 0:
            return num_experts

    for cfg in _iter_zb_hf_configs(vllm_config):
        num_experts = int(getattr(cfg, "num_experts", 0) or 0)
        if num_experts > 0:
            return num_experts
    return 0


def resolve_zb_experts_per_token(vllm_config) -> int:
    for cfg in _iter_zb_hf_configs(vllm_config):
        for attr in ("num_experts_per_tok", "moe_topk", "top_k"):
            top_k = int(getattr(cfg, attr, 0) or 0)
            if top_k > 0:
                return top_k
    return 1


def resolve_zb_shmem_uri() -> str:
    """Resolve aclshmem conf-store URI for this worker.

    Serving path (no extra config):
      1. ``NPUWorker._init_worker_distributed_environment`` calls
         ``reserve_zb_shmem_conf_store_uri`` after MC2 groups are created.
      2. ``ensure_zb_process_initialized`` reads the reserved URI here.

    Fallback order: e2e override env → reserved conf-store URI.
    """
    override = os.getenv("VLLM_ASCEND_ZB_SHMEM_URI") or os.getenv("VLLM_ASCEND_ZB_URI")
    if override:
        return override if "://" in override else f"tcp://{override}"

    if _ZB_SHMEM_CONF_STORE_URI:
        return _ZB_SHMEM_CONF_STORE_URI

    raise RuntimeError(
        "additional_config.enable_mc2_zb=true but ZB SHMEM conf-store URI is unavailable. "
        "Serving workers must call reserve_zb_shmem_conf_store_uri() after MC2 init. "
        "For standalone e2e tests only, set VLLM_ASCEND_ZB_SHMEM_URI."
    )


def _synchronize_zb_init_peers() -> None:
    """Barrier EP peers before aclshmem conf-store rendezvous."""
    try:
        from vllm_ascend.distributed.parallel_state import get_mc2_group

        mc2 = get_mc2_group()
        if mc2.world_size > 1:
            # Use the CPU gloo group: HCCL device barrier here can deadlock with
            # aclshmem's TCP conf-store rendezvous under DP>1.
            torch.distributed.barrier(group=mc2.cpu_group)
    except AssertionError:
        pass


def validate_zb_serving_parallel_config(parallel_config) -> None:
    """Fail fast when ZB is enabled with unsupported parallel layout."""
    if not get_ascend_config().enable_mc2_zb:
        return


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
        server_ip_port = resolve_zb_shmem_uri()
    logger.info(
        "Initializing ZB aclshmem (rank=%s/%s, local_mem_size=%s, uri=%s)",
        rank,
        world_size,
        local_mem_size,
        server_ip_port,
    )

    runtime = ZbMoERuntime(
        rank=rank,
        world_size=world_size,
        server_ip_port=server_ip_port,
        local_mem_size=local_mem_size,
    )
    _synchronize_zb_init_peers()
    try:
        runtime.init()
    except RuntimeError as exc:
        raise RuntimeError(
            "aclshmem process init failed for additional_config.enable_mc2_zb=true. "
            f"uri={server_ip_port!r}, rank={rank}, world_size={world_size}, "
            f"local_mem_size={local_mem_size}. SHMEM conf-store uses a dedicated TCP "
            "port reserved at worker startup (separate from the HCCL TCPStore port). "
            "Check that the reserved port is free, all MC2 ranks enter init together, "
            "and host DRAM can satisfy local_mem_size. "
            "Set VLLM_ASCEND_ZB_DEBUG=1 for C++ init logs."
        ) from exc
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
            self.server_ip_port = resolve_zb_shmem_uri()

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


def zb_moe_distribute_dispatch(
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
    """Thin Python wrapper around torch.ops._C_ascend.zb_moe_distribute_dispatch.

    All output tensors must be pre-allocated by the caller. ``ext_info`` is the
    SHMEM global virtual address returned by :py:meth:`ZbMoERuntime.alloc` /
    :py:meth:`ZbMoERuntime.get_ext_info`.
    """
    _ensure_zb_op_available("zb_moe_distribute_dispatch")
    return torch.ops._C_ascend.zb_moe_distribute_dispatch(
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


def zb_moe_distribute_combine(
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
    """Thin Python wrapper around torch.ops._C_ascend.zb_moe_distribute_combine.

    ``combined_x`` must be pre-allocated by the caller. ``ext_info`` is the
    SHMEM global virtual address used as the combine source buffer pointer
    (typically the same value used by ``zb_moe_distribute_dispatch``).
    """
    _ensure_zb_op_available("zb_moe_distribute_combine")
    if tp_send_count is None:
        # The combine tiling marks tp_send_count optional, but still validates
        # its shape and dtype. deepep_standalone always passes an int32 tensor
        # with one entry per TP rank even when tp_world_size == 1.
        tp_send_count = torch.empty((tp_world_size,), dtype=torch.int32, device=expand_x.device)
    return torch.ops._C_ascend.zb_moe_distribute_combine(
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
