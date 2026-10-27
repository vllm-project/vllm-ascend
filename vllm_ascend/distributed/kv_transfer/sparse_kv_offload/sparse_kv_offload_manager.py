import contextlib
import typing
from dataclasses import dataclass
from zlib import adler32

import numpy as np
import torch
import torch_npu

with contextlib.suppress(Exception):
    # we should remove this after memfabric.offload is merged to master and available in ci machine.
    # Importing the package provisions an AICPU kernel on the fly, which raises its own
    # ProvisioningError rather than ImportError when the CANN build toolchain is incomplete.
    # Sparse KV offload is opt-in, so nothing here may abort worker startup.
    from memfabric_hybrid import offload  # type: ignore
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.ascend_config import DUMMY_POOL_ROW_COUNT, SparseKVOffloadConfig, get_ascend_config
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import copy_sfa_pool_capacity
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

# Main BF16 cache:
# [k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v].
# Sparse LI C8 indexer caches are separate and
# remain device-resident, so they are not registered with this manager.
OFFLOAD_KV_CACHE_TUPLE_LEN = 6
OFFLOAD_K_CACHE_NPU_INDEX = 0
OFFLOAD_V_CACHE_NPU_INDEX = 1
OFFLOAD_K_CACHE_CPU_INDEX = 2
OFFLOAD_V_CACHE_CPU_INDEX = 3
OFFLOAD_TOPK_BUFFER_K_INDEX = 4
OFFLOAD_TOPK_BUFFER_V_INDEX = 5
OFFLOAD_STORE_PORT_BASE = 8500
# Per-request miss-plan capacity for lim_miss_src/lim_miss_dst. MUST equal
# LIMConfig::MISS_CAPACITY in
# csrc/attention/fused_lightning_indexer_manage/op_kernel/..._constants.h;
# a mismatch causes silent buffer overrun in the LIM operator.
LIM_MISS_PLAN_MAX_WIDTH = 32768
# UB staging capacity of one KV row inside npu_lim_copy_descriptors. MUST match
# LCD_KV_DIM_MAX / LCD_ROPE_DIM_MAX in
# csrc/attention/lim_copy_descriptors/op_kernel/lim_copy_descriptors.cpp. The
# per-model dims travel to the kernel through params and are validated against
# these bounds here, so an unsupported model fails loudly instead of writing
# past the UB staging buffer.
LIM_KV_DIM_MAX = 512
LIM_ROPE_DIM_MAX = 64


@dataclass
class _LIMGroupRuntime:
    """Side-stream onboarding state for one LIM selection group.

    ``plan_event`` orders the side stream after the owner's LIM operator on
    the main stream; ``done_events`` order each member's resident SFA after
    the group copy. All objects are created once (outside graph capture) and
    reused every step. ``request_slots`` bounds the per-step request count;
    onboarding always goes through the npu_lim_copy_descriptors operator.
    ``last_launch_token`` holds the ForwardContext object of the forward
    pass that last launched this group's onboarding. Members compare it
    against their own pass before taking the all-zero-miss-plan
    (direct-addressed) path, so a pass whose owner skipped the LIM (e.g. a
    draft reuse step) falls back to inline miss resolution. The forward
    context — not the attention metadata — is the step token here: owners
    and members live in different KV cache groups under
    offload_shared_indexer_only, so their builders hand out distinct
    metadata objects for the same step.
    """

    member_names: list[str]
    plan_event: object
    done_events: dict[str, object]
    request_slots: int
    last_launch_token: object = None


FSA_EXTERNAL_PLAN_READY_MARKER = 0x5A45
FSA_PAIRED_SELECTION_COPY_MARKER = 0x5A56
FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT = 16376
FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT = 16
FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT = 8
FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT = (
    (FSA_SELECTION_MEMBERSHIP_MAP_INT16_COUNT + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1)
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT = (
    (
        FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
        + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
        + FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
        - 1
    )
    // FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
    * FSA_SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT
)
FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS = (
    FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT
)


_SPARSE_KV_OFFLOAD_OPS = None
_SPARSE_KV_OFFLOAD_OP_NAMES = (
    "sparse_kv_warmup_lru_resident_threads",
    "sparse_kv_lru_resident_compact_with_plan_stable_rows",
    "sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows",
    "sparse_kv_restore_bfloat16_tensor",
    "sparse_kv_restore_int16_tensor",
)


def _sparse_kv_ops():
    global _SPARSE_KV_OFFLOAD_OPS
    if _SPARSE_KV_OFFLOAD_OPS is not None:
        return _SPARSE_KV_OFFLOAD_OPS
    if not enable_custom_op():
        raise RuntimeError("Sparse KV offload requires custom ops to be enabled.")

    missing_ops = [name for name in _SPARSE_KV_OFFLOAD_OP_NAMES if not hasattr(torch.ops._C_ascend, name)]
    if missing_ops:
        raise RuntimeError(
            "Sparse KV offload requires _C_ascend sparse KV ops, "
            f"missing={missing_ops}. Rebuild vllm_ascend_C with Atlas A3 support."
        )
    _SPARSE_KV_OFFLOAD_OPS = torch.ops._C_ascend
    return _SPARSE_KV_OFFLOAD_OPS


def allocate_kv_offload_topk_buffer_pair(
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_width = 1
    if vllm_config.speculative_config is not None:
        decode_width += vllm_config.speculative_config.num_speculative_tokens

    topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
    num_kv_heads = 1  # sparse kv offload only support sfa(mla) now.
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    max_num_topk_rows = min(
        vllm_config.scheduler_config.max_num_batched_tokens,
        vllm_config.scheduler_config.max_num_seqs * decode_width,
    )
    if sparse_kv_offload_config.use_fused_copy_sfa:
        # The backend imports this manager; defer its layout constant import.
        from vllm_ascend.attention.sfa_kv_offload import COPY_SFA_TAIL_BLOCKS

        # LIM merges all query selections into one request-owned hot row.
        # Graph padding uses a second, private arena of the same capacity.
        # Query-row metadata still needs max_num_seqs * decode_width, but
        # multiplying the large hot K/V allocation by that width wastes HBM.
        # Inactive request slots share a small dummy-row ring
        # (DUMMY_POOL_ROW_COUNT) instead of each consuming a dedicated hot
        # row: one row per live request slot plus the ring.
        request_rows = copy_sfa_pool_capacity(vllm_config.scheduler_config.max_num_seqs) + DUMMY_POOL_ROW_COUNT
        max_num_topk_rows = request_rows
        topk_buffer_size += COPY_SFA_TAIL_BLOCKS * vllm_config.cache_config.block_size
    topk_buffer_k_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * k_dim * torch.bfloat16.itemsize
    topk_buffer_v_size_bytes = max_num_topk_rows * topk_buffer_size * num_kv_heads * v_dim * torch.bfloat16.itemsize
    # NOTE make sure to allocate k+v together and split them after allocate.
    # Refer to the comment in empty_aligned_int8_cpu_tensors for reason.
    topk_buffer_raw = torch.empty([topk_buffer_k_size_bytes + topk_buffer_v_size_bytes], dtype=torch.int8, device="npu")
    topk_buffer_k = (
        topk_buffer_raw[:topk_buffer_k_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, k_dim])
    )
    topk_buffer_v = (
        topk_buffer_raw[topk_buffer_k_size_bytes : topk_buffer_k_size_bytes + topk_buffer_v_size_bytes]
        .view(torch.bfloat16)
        .view([max_num_topk_rows, topk_buffer_size, num_kv_heads, v_dim])
    )
    if sparse_kv_offload_config.use_fused_copy_sfa:
        # Dummy copy-SFA/LIM rows read their private hot rows during capture/replay.
        topk_buffer_raw.zero_()
    return (topk_buffer_k, topk_buffer_v)


def allocate_kv_offload_topk_profile_buffers(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    num_offload_layers = sum(bool(getattr(spec, "store_on_host", False)) for spec in kv_cache_spec.values())
    if num_offload_layers == 0:
        raise RuntimeError("Sparse KV offload profile did not find any host-resident SFA KV cache layers.")

    buffers = [
        allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config) for _ in range(num_offload_layers)
    ]
    buffer_bytes = sum(tensor.numel() * tensor.element_size() for buffer_pair in buffers for tensor in buffer_pair)
    logger.info_once(
        "Sparse KV offload reserved %.2f GiB of KV offload topk buffers across %d layers for profile run.",
        buffer_bytes / (1 << 30),
        num_offload_layers,
    )
    return buffers


_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
# Worst-case 2 MiB-alignment waste per host layer (see empty_aligned_int8_cpu_tensors):
# 1x for rounding the raw buffer base address up to the alignment boundary, plus 1x tail
# padding for each of the two aligned tensors (k_cache_cpu, v_cache_cpu) whose sizes are
# rounded up to whole alignment chunks.
_CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER = 3 * _CPU_CACHE_ALIGNMENT
_VLLM_NULL_BLOCK_COUNT = 1


def empty_aligned_int8_cpu_tensors(
    sizes: list[int],
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> list[torch.Tensor]:
    """
    Allocate multiple int8 tensors with specified sizes,
    each aligned to specified alignment,
    and minimize the gap between each tensor's data_ptr.
    Keep paired K/V allocations aligned within the shared host pool.
    """
    chunk_nums = [cdiv(size, alignment) for size in sizes]
    total_chunk_num = 1 + sum(chunk_nums)
    raw_tensor = offload.empty([total_chunk_num * alignment], dtype=torch.int8, pin_memory=True)
    base_addr = raw_tensor.data_ptr()
    if base_addr % alignment:
        base_addr = (base_addr // alignment + 1) * alignment
    base_offset = base_addr - raw_tensor.data_ptr()
    allocate_tensors = []
    for size, chunk_num in zip(sizes, chunk_nums):
        allocate_tensors.append(raw_tensor[base_offset : base_offset + size])
        base_offset += chunk_num * alignment
    return allocate_tensors


@dataclass(frozen=True)
class SparseKVOffloadMemoryBudget:
    npu_limit_blocks: int
    dram_limit_blocks: int
    workload_limit_blocks: int
    final_num_blocks: int
    final_planner_bytes: int
    planned_host_bytes: int
    planned_device_bytes: int
    host_alignment_reserve_bytes: int
    limiting_factor: str


def _split_host_device_kv_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[list[KVCacheSpec], list[KVCacheSpec]]:
    if not kv_cache_spec:
        raise ValueError("Sparse KV offload requires a non-empty kv_cache_spec")
    host_specs: list[KVCacheSpec] = []
    device_specs: list[KVCacheSpec] = []
    for spec in kv_cache_spec.values():
        if getattr(spec, "store_on_host", False):
            host_specs.append(spec)
        else:
            device_specs.append(spec)
    if not host_specs:
        raise ValueError("Sparse KV offload requires at least one host KV cache spec")
    if not device_specs:
        raise ValueError("Sparse KV offload requires at least one device KV cache spec")
    block_sizes = {spec.block_size for spec in host_specs + device_specs}
    if len(block_sizes) != 1:
        raise ValueError(f"Sparse KV offload memory planning requires one shared block size, got {sorted(block_sizes)}")
    return host_specs, device_specs


def plan_sparse_kv_offload_memory(
    kv_cache_spec: dict[str, KVCacheSpec],
    vllm_config: VllmConfig,
    available_device_memory_bytes: int,
    dram_limit_bytes: int,
    keep_device_kv_cache: bool,
) -> SparseKVOffloadMemoryBudget:
    """Bound sparse KV offload blocks by NPU, DRAM, and active demand."""
    host_specs, device_specs = _split_host_device_kv_specs(kv_cache_spec)
    host_page_size_bytes = sum(spec.page_size_bytes for spec in host_specs)
    device_page_size_bytes = sum(spec.page_size_bytes for spec in device_specs)
    total_page_size_bytes = host_page_size_bytes + device_page_size_bytes

    host_alignment_reserve_bytes = len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    usable_dram_bytes = max(dram_limit_bytes - host_alignment_reserve_bytes, 0)
    dram_limit_blocks = usable_dram_bytes // host_page_size_bytes

    npu_page_size_bytes = total_page_size_bytes if keep_device_kv_cache else device_page_size_bytes
    npu_limit_blocks = max(available_device_memory_bytes, 0) // npu_page_size_bytes

    max_blocks_per_request = max(
        cdiv(
            spec.max_memory_usage_bytes(vllm_config),
            spec.page_size_bytes,
        )
        for spec in host_specs + device_specs
    )
    workload_limit_blocks = max_blocks_per_request * vllm_config.scheduler_config.max_num_seqs + _VLLM_NULL_BLOCK_COUNT

    limits = {
        "npu": npu_limit_blocks,
        "dram": dram_limit_blocks,
        "workload": workload_limit_blocks,
    }
    if dram_limit_blocks < npu_limit_blocks:
        logger.warning_once(
            "Sparse KV offload: host DRAM budget allows only %d KV blocks while NPU "
            "memory could hold %d, so batch size / max_model_len capacity is limited "
            "by host memory. Consider increasing sparse_kv_offload_config."
            "dram_size_per_dp_GB (current budget %.2f GiB) to use more of the NPU "
            "capacity.",
            dram_limit_blocks,
            npu_limit_blocks,
            dram_limit_bytes / (1 << 30),
        )
    limiting_factor = min(limits, key=lambda name: limits[name])
    final_num_blocks = limits[limiting_factor]
    final_planner_bytes = final_num_blocks * total_page_size_bytes
    planned_host_bytes = final_num_blocks * host_page_size_bytes
    planned_device_bytes = final_num_blocks * npu_page_size_bytes
    return SparseKVOffloadMemoryBudget(
        npu_limit_blocks=npu_limit_blocks,
        dram_limit_blocks=dram_limit_blocks,
        workload_limit_blocks=workload_limit_blocks,
        final_num_blocks=final_num_blocks,
        final_planner_bytes=final_planner_bytes,
        planned_host_bytes=planned_host_bytes,
        planned_device_bytes=planned_device_bytes,
        host_alignment_reserve_bytes=host_alignment_reserve_bytes,
        limiting_factor=limiting_factor,
    )


def get_sparse_kv_offload_cpu_pool_size_bytes(
    kv_cache_config: KVCacheConfig,
) -> int:
    """Return a safe upper bound for aligned host KV allocations."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            layer_specs.update(group.kv_cache_spec.kv_cache_specs)
        else:
            layer_specs.update((layer_name, group.kv_cache_spec) for layer_name in group.layer_names)
    host_specs = [spec for spec in layer_specs.values() if getattr(spec, "store_on_host", False)]
    if not host_specs:
        raise ValueError("Sparse KV offload requires host-resident KV cache specs")
    raw_host_bytes = kv_cache_config.num_blocks * sum(spec.page_size_bytes for spec in host_specs)
    alignment_reserve_bytes = len(host_specs) * _CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return raw_host_bytes + alignment_reserve_bytes


def allocate_kv_cache_tensors_for_sparse_kv_offload(
    k_tensor_size: int,
    v_tensor_size: int,
    alignment: int,
    tp_rank: int,
    keep_device_kv_cache: bool,
    npu_kv_cache_allocate_func: typing.Callable,
):
    if tp_rank == 0:
        [k_tensor_cpu, v_tensor_cpu] = empty_aligned_int8_cpu_tensors(
            [k_tensor_size, v_tensor_size],
            alignment,
        )
    else:
        k_tensor_cpu = None
        v_tensor_cpu = None

    if keep_device_kv_cache:
        k_tensor = npu_kv_cache_allocate_func(
            k_tensor_size,
            alignment,
        )
        v_tensor = npu_kv_cache_allocate_func(
            v_tensor_size,
            alignment,
        )
    else:
        k_tensor = None
        v_tensor = None

    return (k_tensor, v_tensor, k_tensor_cpu, v_tensor_cpu, k_tensor_size + v_tensor_size)


def reshape_kv_cache_tensors_for_sparse_kv_offload(
    raw_cache_tensors: tuple,
    current_kv_cache_spec: AttentionSpec,
    attn_backend: type[AttentionBackend],
    tp_rank: int,
    vllm_config: VllmConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    raw_k_tensor, raw_v_tensor, raw_k_tensor_cpu, raw_v_tensor_cpu, sum_page_size_bytes = raw_cache_tensors
    assert sum_page_size_bytes % current_kv_cache_spec.page_size_bytes == 0
    num_blocks = sum_page_size_bytes // current_kv_cache_spec.page_size_bytes
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        num_blocks,
        current_kv_cache_spec.block_size,
        current_kv_cache_spec.num_kv_heads,
        current_kv_cache_spec.head_size,
    )
    mla_num_blocks, mla_block_size, num_kv_heads, _ = kv_cache_shape
    k_dim = vllm_config.model_config.hf_text_config.kv_lora_rank
    v_dim = vllm_config.model_config.hf_text_config.qk_rope_head_dim
    k_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        k_dim,
    )
    v_shape = (
        mla_num_blocks,
        mla_block_size,
        num_kv_heads,
        v_dim,
    )
    k_cache_dtype = v_cache_dtype = current_kv_cache_spec.dtype

    k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape) if raw_k_tensor is not None else None
    v_cache = raw_v_tensor.view(v_cache_dtype).view(v_shape) if raw_v_tensor is not None else None

    if tp_rank == 0:
        k_cache_cpu = raw_k_tensor_cpu.view(k_cache_dtype).view(k_shape)
        v_cache_cpu = raw_v_tensor_cpu.view(v_cache_dtype).view(v_shape)
    else:
        k_cache_cpu = None
        v_cache_cpu = None

    topk_buffer_k, topk_buffer_v = allocate_kv_offload_topk_buffer_pair(vllm_config, sparse_kv_offload_config)
    return (k_cache, v_cache, k_cache_cpu, v_cache_cpu, topk_buffer_k, topk_buffer_v)


def update_sparse_kv_offload_metadata(
    num_tokens: int,
    num_reqs: int,
    num_tokens_padded: int,
    num_reqs_padded: int,
    req_ids: list[str],
    query_start_loc: CpuGpuBuffer | torch.Tensor | np.ndarray,
    offload_req_ids_tensor: CpuGpuBuffer,
    offload_token_to_req: CpuGpuBuffer,
) -> None:
    """Populate per-request identity tensors for the Sparse KV offload LRU.

    Request ids are adler32 hashes of the scheduler request id,
    rows are reset whenever the occupying request changes.
    """
    offload_req_ids_tensor.np[:num_reqs_padded].fill(0)
    effective_num_reqs = min(num_reqs, len(req_ids))
    req_id_values = np.asarray(
        [
            adler32(req_id.encode("utf-8")) if isinstance(req_id, str) else row + 1
            for row, req_id in enumerate(req_ids[:effective_num_reqs])
        ],
        dtype=np.int64,
    )
    offload_req_ids_tensor.np[:effective_num_reqs] = req_id_values
    offload_req_ids_tensor.copy_to_gpu(num_reqs_padded)

    if isinstance(query_start_loc, CpuGpuBuffer):
        query_start_loc_cpu = query_start_loc.cpu[: num_reqs + 1].numpy()
    elif isinstance(query_start_loc, torch.Tensor):
        if query_start_loc.device.type != "cpu":
            raise ValueError("Sparse KV offload query boundaries must have a CPU mirror")
        query_start_loc_cpu = query_start_loc[: num_reqs + 1].numpy()
    else:
        query_start_loc_cpu = query_start_loc[: num_reqs + 1]
    query_lens = np.diff(query_start_loc_cpu).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        raise RuntimeError(
            "KV offload token_to_req metadata is shorter than the scheduled token batch: "
            f"metadata={token_to_req.shape[0]}, tokens={num_tokens}"
        )
    offload_token_to_req.np[:num_tokens] = token_to_req[:num_tokens]
    if num_tokens_padded > num_tokens:
        offload_token_to_req.np[num_tokens:num_tokens_padded].fill(0)
    offload_token_to_req.copy_to_gpu(num_tokens_padded)


def prepare_sparse_kv_offload_mtp_dummy_metadata(
    num_tokens: int,
    num_reqs: int,
    query_start_loc_cpu: torch.Tensor,
    req_ids_buffer: CpuGpuBuffer,
    token_to_req_buffer: CpuGpuBuffer,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not get_ascend_config().sparse_kv_offload_config.enabled:
        return None, None
    if req_ids_buffer is None or token_to_req_buffer is None:
        raise RuntimeError("Sparse KV offload metadata buffers are not initialized")

    query_lens = np.diff(query_start_loc_cpu[: num_reqs + 1].numpy()).astype(np.int32, copy=False)
    token_to_req = np.repeat(np.arange(num_reqs, dtype=np.int32), query_lens)
    if token_to_req.shape[0] < num_tokens:
        token_to_req = np.pad(token_to_req, (0, num_tokens - token_to_req.shape[0]))

    req_ids_buffer.np[:num_reqs] = np.arange(1, num_reqs + 1, dtype=np.int64)
    req_ids_buffer.copy_to_gpu(num_reqs)
    token_to_req_buffer.np[:num_tokens] = token_to_req[:num_tokens]
    token_to_req_buffer.copy_to_gpu(num_tokens)
    return (
        req_ids_buffer.gpu[:num_reqs],
        token_to_req_buffer.gpu[:num_tokens],
    )


# LimCopyDescriptors operator used by the side-stream group onboarding
# (miss_onboard_mode=side_stream). Resolved lazily from the registered
# ``torch.ops._C_ascend`` namespace so worker processes only touch the
# dispatcher when the side-stream path actually runs.
_LIM_DESC_OP_UNRESOLVED = object()
_LIM_DESC_OP = _LIM_DESC_OP_UNRESOLVED


def _get_lim_desc_op():
    """Resolve ``torch.ops._C_ascend.npu_lim_copy_descriptors`` once.

    The operator ships with vllm_ascend_C builds that carry
    csrc/attention/lim_copy_descriptors. Older builds do not register it;
    miss_onboard_mode=side_stream then degrades to inline miss resolution
    inside the fused attention operator (see configure_lim_groups).
    """
    global _LIM_DESC_OP
    if _LIM_DESC_OP is _LIM_DESC_OP_UNRESOLVED:
        op = getattr(torch.ops._C_ascend, "npu_lim_copy_descriptors", None)
        if op is None:
            logger.warning_once(
                "npu_lim_copy_descriptors is not registered in this vllm_ascend_C build; "
                "miss_onboard_mode=side_stream falls back to inline miss resolution."
            )
        _LIM_DESC_OP = op
    return _LIM_DESC_OP


class SparseKVOffloadManager:
    """
    A manager responsible to the Sparse KV cache Offload.
    It enlarge the available memory that scheduler can see,
    so we can schedule longer max_model_len or larger decode batch size.
    No more scheduling logic: we reuse the original block_table/slot_mapping.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        sparse_kv_offload_config: SparseKVOffloadConfig,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.sparse_kv_offload_config = sparse_kv_offload_config

        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        self.num_target_layers = model_config.get_num_layers(parallel_config)
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.block_size = self._infer_group_block_sizes(self.kv_cache_config)
        self.topk_buffer_size = sparse_kv_offload_config.topk_buffer_size
        self.topk = sparse_kv_offload_config.topk
        self.use_fused_overlap = sparse_kv_offload_config.use_fused_overlap
        self.use_fused_copy_sfa = sparse_kv_offload_config.use_fused_copy_sfa
        # LIM selection groups (offload_shared_indexer_only): empty until
        # configure_lim_groups runs, so calls from the legacy full-offload
        # path are harmless no-ops.
        self._lim_group_states: dict[str, _LIMGroupRuntime] = {}
        self._lim_side_stream = None
        self._lim_group_ksc_params = None
        self._lim_group_ksc_geometry = None
        self._lim_group_ksc_ready = False
        # Main-KV row dims, published by register_kv_caches and forwarded to
        # the onboarding operator through params.
        self.head_dim_k = 0
        self.head_dim_v = 0

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        decode_width = 1
        if vllm_config.speculative_config is not None:
            decode_width += vllm_config.speculative_config.num_speculative_tokens
        self.max_num_topk_rows = min(
            self.max_num_tokens,
            self.max_num_reqs * decode_width,
        )
        self.fused_overlap_membership_map: torch.Tensor | None = None
        self.fused_overlap_membership_map_rows = 0
        self.fused_overlap_plan_owner_layer_id: int | None = None
        self.fused_overlap_plan_topk: int | None = None
        self.fused_overlap_plan_num_tokens = 0
        self.fused_overlap_plan_membership_map: torch.Tensor | None = None
        _sparse_kv_ops()

        dram_limit_bytes = int(sparse_kv_offload_config.dram_size_per_dp_GB * (1 << 30))
        planned_pool_size_bytes = get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)
        if planned_pool_size_bytes > dram_limit_bytes:
            raise ValueError(
                "Sparse KV offload planned CPU pool exceeds DRAM limit after "
                "alignment: "
                f"planned={planned_pool_size_bytes / (1 << 30):.2f} GiB, "
                f"limit={sparse_kv_offload_config.dram_size_per_dp_GB} GiB, "
                f"num_blocks={kv_cache_config.num_blocks}"
            )
        actual_pool_size_bytes = min(planned_pool_size_bytes, dram_limit_bytes)
        logger.info(
            "SparseKVOffloadManager starts CPU KV pool initialization: "
            "planned=%.2f GiB, configured_limit=%s GiB, num_blocks=%s.",
            actual_pool_size_bytes / (1 << 30),
            sparse_kv_offload_config.dram_size_per_dp_GB,
            kv_cache_config.num_blocks,
        )
        config = offload.OffloadConfig()
        config.device_id = torch_npu.npu.current_device()
        config.reserve_size = actual_pool_size_bytes
        config.alloc_size = actual_pool_size_bytes if self.tp_rank == 0 else 0
        config.world_size = self.tp_size
        config.rank_id = self.tp_rank
        config.scene = offload.Scene.SHARED
        store_port = OFFLOAD_STORE_PORT_BASE + parallel_config.data_parallel_index
        # Older pip-packaged memfabric_hybrid 1.2.0 builds have no
        # OffloadConfig.store_url field and derive the store endpoint
        # internally; only set it when the field exists.
        if hasattr(config, "store_url"):
            config.store_url = f"tcp://127.0.0.1:{store_port}"
        assert offload.initialize(config) == 0, "Sparse KV offload offload.initialize failed."
        self.tp_group.barrier()

    def _warmup_external_lru_planner_threads(self) -> int:
        if not (self.use_fused_overlap and self.tp_rank == 0):
            return 0
        warmed_threads = _sparse_kv_ops().sparse_kv_warmup_lru_resident_threads(self.lru_workspace_threads)
        logger.info(
            "Warmed external LRU planner OpenMP team with %s threads",
            warmed_threads,
        )
        return warmed_threads

    def _infer_group_block_sizes(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> int:
        block_sizes = {group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups}
        if len(block_sizes) != 1:
            raise ValueError("Sparse KV offload requires one shared block size across target and draft groups.")
        return block_sizes.pop()

    @staticmethod
    def _as_cache_tuple(cache_or_caches) -> tuple[torch.Tensor, ...]:
        if isinstance(cache_or_caches, torch.Tensor):
            return (cache_or_caches,)
        return tuple(cache_or_caches)

    def _register_offload_layers(
        self,
        kv_caches: dict[str, torch.Tensor],
        full_resident_layer_names: list[str] | None = None,
    ) -> None:
        host_layer_names = set()
        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[layer_name]
                if getattr(spec, "store_on_host", False):
                    host_layer_names.add(layer_name)
        # offload_shared_indexer_only keeps full (indexer-owner) layers on
        # standard NPU paged caches (store_on_host=False specs); they carry
        # no hot rows / CPU pool here. The model runner passes their names.
        resident_names = set(full_resident_layer_names or ())
        self.full_resident_layer_names: list[str] = [
            layer_name
            for layer_name in kv_caches
            if "indexer" not in layer_name and layer_name in resident_names and layer_name not in host_layer_names
        ]
        self.offload_layer_names = [
            layer_name
            for layer_name in kv_caches
            if layer_name in host_layer_names and layer_name not in resident_names
        ]
        if not self.offload_layer_names:
            raise ValueError("Sparse KV offload did not find SFA KV cache layers.")

        self.num_layers = len(self.offload_layer_names)
        self.layer_name_to_offload_id = {
            layer_name: layer_id for layer_id, layer_name in enumerate(self.offload_layer_names)
        }

        logger.info_once(
            "Sparse KV offload registered %s offloaded layers and %s device-resident full layers (%s target layers).",
            self.num_layers,
            len(self.full_resident_layer_names),
            self.num_target_layers,
        )
        self.mtp_layer_id = self.num_layers - 1 if self.num_layers != self.num_target_layers else -1
        if self.tp_rank == 0:
            preview_layer_names = self.offload_layer_names[:4]
            if len(self.offload_layer_names) > 4:
                preview_layer_names += ["..."] + self.offload_layer_names[-4:]
            logger.info("Sparse KV offload layer names: %s", preview_layer_names)

    def _get_offload_layer_id(self, layer_name: str) -> int:
        layer_id = self.layer_name_to_offload_id.get(layer_name)
        if layer_id is None:
            registered_layers = ", ".join(self.offload_layer_names[:8])
            if len(self.offload_layer_names) > 8:
                registered_layers += ", ..."
            raise KeyError(
                "Sparse KV offload layer is not registered, "
                f"layer_name={layer_name}, registered_layers=[{registered_layers}]"
            )
        return layer_id

    def _restore_bfloat16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = _sparse_kv_ops().sparse_kv_restore_bfloat16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_bfloat16_tensor returned a tensor with unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                f"restore_bfloat16_tensor returned unexpected shape: expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.bfloat16:
            raise RuntimeError(f"restore_bfloat16_tensor returned unexpected dtype: {view.dtype}")
        if not view.is_contiguous():
            raise RuntimeError("restore_bfloat16_tensor requires a contiguous view")
        return view

    def _restore_int16_tensor(self, ptr: int, shape: list[int]) -> torch.Tensor:
        view = _sparse_kv_ops().sparse_kv_restore_int16_tensor(ptr, shape)
        if int(view.data_ptr()) != int(ptr):
            raise RuntimeError(
                "restore_int16_tensor returned a tensor with an unexpected data_ptr: "
                f"expected={ptr}, got={view.data_ptr()}"
            )
        if list(view.shape) != list(shape):
            raise RuntimeError(
                f"restore_int16_tensor returned an unexpected shape: expected={shape}, got={list(view.shape)}"
            )
        if view.dtype != torch.int16:
            raise RuntimeError(f"restore_int16_tensor returned an unexpected dtype: {view.dtype}")
        if not view.is_contiguous():
            raise RuntimeError("restore_int16_tensor requires a contiguous view")
        return view

    def allocate_fused_overlap_membership_map(
        self,
        row_capacity: int,
    ) -> torch.Tensor:
        if not self.use_fused_overlap:
            raise RuntimeError("mapped membership allocation requires kv_offload_decode_config.use_fused_overlap=true")
        if row_capacity <= 0:
            raise ValueError(f"mapped membership row capacity must be positive, got {row_capacity}")
        if self.topk >= FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT:
            raise ValueError(
                "SFA sparse topk exceeds external plan storage: "
                f"topk={self.topk} "
                f"control_offset={FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT}"
            )

        if getattr(self, "fused_overlap_membership_map", None) is not None:
            if row_capacity > self.fused_overlap_membership_map_rows:
                raise RuntimeError(
                    "fused_overlap membership storage was allocated for fewer "
                    f"rows: allocated={self.fused_overlap_membership_map_rows}, "
                    f"required={row_capacity}"
                )
            return self.fused_overlap_membership_map

        shape = [row_capacity, FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT]
        owner_ptr = torch.zeros(1, dtype=torch.int64, device="npu")
        membership_map = None
        if self.tp_rank == 0:
            membership_map = offload.empty(
                [row_capacity * FSA_SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT],
                dtype=torch.int16,
                pin_memory=True,
            ).view(shape)
            membership_map.fill_(-1)
            control = membership_map[
                :,
                FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT : FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT
                + FSA_SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT,
            ]
            control[:, 1] = FSA_EXTERNAL_PLAN_READY_MARKER
            control[:, 2] = self.topk
            control[:, 3] = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - self.topk
            control[:, 7] = FSA_PAIRED_SELECTION_COPY_MARKER
            owner_ptr[0] = membership_map.data_ptr()
        self.tp_group.broadcast(owner_ptr, src=0)
        shared_ptr = int(owner_ptr.item())
        if self.tp_rank != 0:
            membership_map = self._restore_int16_tensor(shared_ptr, shape)
        if membership_map is None:
            raise RuntimeError("mapped membership storage was not initialized")
        self.tp_group.barrier()
        self.fused_overlap_membership_map = membership_map
        self.fused_overlap_membership_map_rows = row_capacity
        return membership_map

    def get_fused_overlap_cpu_kv_inputs(
        self,
        layer_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU full KV tensors used by fused_overlap decode.

        On TP0 these are the owned CPU pools; on other ranks they are non-owning
        GVA views restored after broadcast.
        """
        if not self.use_fused_overlap:
            raise RuntimeError(
                "get_fused_overlap_cpu_kv_inputs requires kv_offload_decode_config.use_fused_overlap=true"
            )
        layer_id = self._get_offload_layer_id(layer_name)
        if layer_id >= len(self.k_caches_cpu) or layer_id >= len(self.v_caches_cpu):
            raise RuntimeError(
                "fused_overlap CPU KV views are not registered: "
                f"layer_id={layer_id}, k_len={len(self.k_caches_cpu)}, "
                f"v_len={len(self.v_caches_cpu)}"
            )
        return self.k_caches_cpu[layer_id], self.v_caches_cpu[layer_id]

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        full_resident_layer_names: list[str] | None = None,
    ):
        self._register_offload_layers(kv_caches, full_resident_layer_names)

        # register topk_buffer and cpu kv_cache
        self.topk_buffers_k: list[torch.Tensor] = []
        self.topk_buffers_v: list[torch.Tensor] = []
        self.k_caches_cpu: list[torch.Tensor] = []
        self.v_caches_cpu: list[torch.Tensor] = []
        for layer_name in self.offload_layer_names:
            cache_or_caches = self._as_cache_tuple(kv_caches[layer_name])
            tuple_len = len(cache_or_caches)
            if tuple_len not in [OFFLOAD_KV_CACHE_TUPLE_LEN]:
                raise ValueError(
                    f"Sparse KV offload layer {layer_name}: expected tuple length "
                    f"{OFFLOAD_KV_CACHE_TUPLE_LEN}, got {tuple_len}"
                )
            self.topk_buffers_k.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_K_INDEX])
            self.topk_buffers_v.append(cache_or_caches[OFFLOAD_TOPK_BUFFER_V_INDEX])
            if self.tp_rank == 0:
                self.k_caches_cpu.append(cache_or_caches[OFFLOAD_K_CACHE_CPU_INDEX])
                self.v_caches_cpu.append(cache_or_caches[OFFLOAD_V_CACHE_CPU_INDEX])

        # Device-resident full layers: the PD pull writes their paged caches
        # directly (same per-block layout the CPU pool mirrors), so publish
        # the rank-local HBM bases and per-block byte lengths for the
        # connector's read state.
        self.full_resident_k_bases: list[int] = []
        self.full_resident_v_bases: list[int] = []
        self.full_resident_block_lens: list[tuple[int, int]] = []
        for layer_name in self.full_resident_layer_names:
            k_cache, v_cache = self._as_cache_tuple(kv_caches[layer_name])[:2]
            if k_cache is None or v_cache is None:
                raise ValueError(f"Sparse KV offload full-resident layer {layer_name} has no NPU paged K/V caches")
            self.full_resident_k_bases.append(int(k_cache.data_ptr()))
            self.full_resident_v_bases.append(int(v_cache.data_ptr()))
            self.full_resident_block_lens.append(
                (
                    int(k_cache.element_size() * np.prod(k_cache.shape[1:])),
                    int(v_cache.element_size() * np.prod(v_cache.shape[1:])),
                )
            )
        if self.full_resident_layer_names:
            logger.info_once(
                "Sparse KV offload keeps %d full layers device-resident for the PD pull.",
                len(self.full_resident_layer_names),
            )

        kv_head_num = self.topk_buffers_k[0].size(-2)
        head_dim_k = self.topk_buffers_k[0].size(-1)
        head_dim_v = self.topk_buffers_v[0].size(-1)
        dtype = self.topk_buffers_k[0].dtype
        assert kv_head_num == 1, "Sparse KV offload only support sfa(mla)"
        if dtype != torch.bfloat16:
            raise ValueError(
                "Sparse KV offload requires a BF16 main SFA cache; sparse LI "
                "C8 is supported only for the device-resident indexer cache."
            )
        self.token_size_bytes_k = kv_head_num * head_dim_k * dtype.itemsize
        self.token_size_bytes_v = kv_head_num * head_dim_v * dtype.itemsize
        self.head_dim_k = int(head_dim_k)
        self.head_dim_v = int(head_dim_v)
        if self.topk_buffer_size % self.block_size != 0:
            raise ValueError(
                "Sparse KV offload topk_buffer_size must be divisible by "
                f"block_size, got {self.topk_buffer_size} and {self.block_size}"
            )

        # D2H descriptors commit newly produced KV into the shared host pool.
        # Both prefill (colocate debug only, gated by keep_device_kv_cache)
        # and decode can produce up to max_num_tokens rows.
        d2h_descriptor_rows = self.max_num_tokens * 2
        device = self.topk_buffers_k[0].device
        self.d2h_src_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_dst_ptrs_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int64, device=device)
        self.d2h_lengths_npu = torch.empty(d2h_descriptor_rows, dtype=torch.int32, device=device)
        if self.use_fused_overlap:
            self.current_kv_save_stream = torch_npu.npu.Stream()
            self.fused_plan_stream = torch_npu.npu.Stream()
            self.fused_plan_metadata_npu = torch.zeros(
                1 + self.max_num_topk_rows,
                dtype=torch.int32,
                device=device,
            )
            self.fused_plan_status_npu = self.fused_plan_metadata_npu[:1]
            self.fused_plan_current_linear_slots_npu = self.fused_plan_metadata_npu[1:]
            self.current_kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.d2h_size_npu = torch.empty(1, dtype=torch.int32, device=device)
        self.d2h_token_indices_npu = torch.arange(self.max_num_tokens, dtype=torch.int64, device=device)

        # Broadcast the shared host KV pool bases and shapes.
        self.gvas_k_bases: list[int] = []
        self.gvas_v_bases: list[int] = []
        # RD2H consumers need byte strides for each TP-shared host KV block.
        self.cpu_block_lens: list[tuple[int, int]] = []
        gvas_k_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        gvas_v_tensor = torch.zeros([self.num_layers], dtype=torch.int64, device="npu")
        cpu_block_lens_tensor = torch.zeros([self.num_layers, 2], dtype=torch.int64, device="npu")
        shape_k_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        shape_v_tensor = torch.zeros([4], dtype=torch.int64, device="npu")
        if self.tp_rank == 0:
            for layer_id in range(self.num_layers):
                k_cpu = self.k_caches_cpu[layer_id]
                v_cpu = self.v_caches_cpu[layer_id]
                gvas_k_tensor[layer_id] = k_cpu.data_ptr()
                gvas_v_tensor[layer_id] = v_cpu.data_ptr()
                cpu_block_lens_tensor[layer_id, 0] = (
                    k_cpu.numel() * k_cpu.element_size() // self.kv_cache_config.num_blocks
                )
                cpu_block_lens_tensor[layer_id, 1] = (
                    v_cpu.numel() * v_cpu.element_size() // self.kv_cache_config.num_blocks
                )
            shape_k_tensor.copy_(torch.tensor(self.k_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
            shape_v_tensor.copy_(torch.tensor(self.v_caches_cpu[0].shape, dtype=torch.int64, device="npu"))
        self.tp_group.broadcast(gvas_k_tensor, src=0)
        self.tp_group.broadcast(gvas_v_tensor, src=0)
        self.tp_group.broadcast(cpu_block_lens_tensor, src=0)
        self.tp_group.broadcast(shape_k_tensor, src=0)
        self.tp_group.broadcast(shape_v_tensor, src=0)
        for layer_id in range(self.num_layers):
            self.gvas_k_bases.append(gvas_k_tensor[layer_id].item())
            self.gvas_v_bases.append(gvas_v_tensor[layer_id].item())

            self.cpu_block_lens.append(
                (
                    cpu_block_lens_tensor[layer_id, 0].item(),
                    cpu_block_lens_tensor[layer_id, 1].item(),
                )
            )

        if self.tp_rank != 0:
            cpu_k_shape = [int(x) for x in shape_k_tensor.tolist()]
            cpu_v_shape = [int(x) for x in shape_v_tensor.tolist()]
            self.k_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_k_shape) for ptr in self.gvas_k_bases]
            self.v_caches_cpu = [self._restore_bfloat16_tensor(ptr, cpu_v_shape) for ptr in self.gvas_v_bases]
            logger.info(
                "[fused_overlap_offload][init] restored shared CPU KV views on "
                "tp_rank=%s layer_count=%s k_shape=%s v_shape=%s",
                self.tp_rank,
                self.num_layers,
                cpu_k_shape,
                cpu_v_shape,
            )
        if self.use_fused_copy_sfa:
            self._bind_copy_sfa_bases()

        if self.use_fused_overlap:
            self._init_fused_overlap_planner()

    def _init_fused_overlap_planner(self) -> None:
        """Allocate the CPU LRU workspace used by the overlap operator."""
        self.lru_workspace_threads = 8
        self._warmup_external_lru_planner_threads()
        self.lru_topk_indices_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_to_token_cpu_list = [
            torch.full(
                [self.max_num_topk_rows, self.topk_buffer_size],
                -1,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_slots_cpu_list = [
            torch.arange(
                self.topk_buffer_size,
                dtype=torch.int32,
                device="cpu",
            )
            .view(1, -1)
            .repeat(self.max_num_topk_rows, 1)
            .pin_memory()
            for _ in range(self.num_layers)
        ]
        self.lru_current_slots_cpu = torch.empty(
            [self.max_num_topk_rows, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_count_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_tokens_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_miss_slots_cpu_list = [
            torch.empty(
                [self.max_num_topk_rows, self.topk],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_req_ids_cpu = torch.empty([self.max_num_topk_rows], dtype=torch.int64, device="cpu", pin_memory=True)
        self.lru_stable_prefix_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_last_req_ids_cpu_list = [
            torch.full(
                [self.max_num_topk_rows],
                -1,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]
        self.lru_visible_seq_lens_cpu = torch.empty(
            [self.max_num_topk_rows],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_token_mark_workspace = torch.zeros(
            [self.lru_workspace_threads, self.max_model_len],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_token_pos_workspace = torch.full(
            [self.lru_workspace_threads, self.max_model_len],
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_slot_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk_buffer_size * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_miss_position_workspace = torch.empty(
            [self.lru_workspace_threads, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_epochs = torch.zeros(
            [self.lru_workspace_threads],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.lru_physical_row_workspace = torch.empty(
            [self.max_num_topk_rows * 3],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        self.lru_req_ids_ptr = self.lru_req_ids_cpu.data_ptr()
        self.lru_stable_prefix_lens_ptr = self.lru_stable_prefix_lens_cpu.data_ptr()
        self.lru_last_req_ids_ptrs = [
            lru_last_req_ids_cpu.data_ptr() for lru_last_req_ids_cpu in self.lru_last_req_ids_cpu_list
        ]
        self.lru_visible_seq_lens_ptr = self.lru_visible_seq_lens_cpu.data_ptr()
        self.lru_topk_indices_ptr = self.lru_topk_indices_cpu.data_ptr()
        self.lru_slot_to_token_ptrs = [
            lru_slot_to_token_cpu.data_ptr() for lru_slot_to_token_cpu in self.lru_slot_to_token_cpu_list
        ]
        self.lru_slots_ptrs = [lru_slots_cpu.data_ptr() for lru_slots_cpu in self.lru_slots_cpu_list]
        self.lru_current_slots_ptr = self.lru_current_slots_cpu.data_ptr()
        self.lru_miss_count_ptrs = [
            lru_miss_count_cpu.data_ptr() for lru_miss_count_cpu in self.lru_miss_count_cpu_list
        ]
        self.lru_miss_tokens_ptrs = [
            lru_miss_tokens_cpu.data_ptr() for lru_miss_tokens_cpu in self.lru_miss_tokens_cpu_list
        ]
        self.lru_miss_slots_ptrs = [
            lru_miss_slots_cpu.data_ptr() for lru_miss_slots_cpu in self.lru_miss_slots_cpu_list
        ]
        self.lru_token_mark_workspace_ptr = self.lru_token_mark_workspace.data_ptr()
        self.lru_token_pos_workspace_ptr = self.lru_token_pos_workspace.data_ptr()
        self.lru_slot_workspace_ptr = self.lru_slot_workspace.data_ptr()
        self.lru_miss_position_workspace_ptr = self.lru_miss_position_workspace.data_ptr()
        self.lru_epochs_ptr = self.lru_epochs.data_ptr()
        self.lru_physical_row_workspace_ptr = self.lru_physical_row_workspace.data_ptr()

    def _bind_copy_sfa_bases(self) -> None:
        """Pin host/device row bases used by eager prefix-rollback tail restore."""
        if not self.k_caches_cpu or not self.topk_buffers_k:
            raise RuntimeError("fused_copy_sfa tail restore requires host and device KV bases")
        device = self.topk_buffers_k[0].device
        descriptor_rows = 4 * (self.max_num_reqs + 2)
        self.copy_sfa_copy_src = torch.empty(descriptor_rows, dtype=torch.int64, device=device)
        self.copy_sfa_copy_dst = torch.empty_like(self.copy_sfa_copy_src)
        self.copy_sfa_host_bases = []
        self.copy_sfa_device_bases = []
        for layer_id in range(len(self.topk_buffers_k)):
            self.copy_sfa_host_bases.append(
                torch.tensor(
                    [self.k_caches_cpu[layer_id].data_ptr(), self.v_caches_cpu[layer_id].data_ptr()],
                    dtype=torch.int64,
                    device=device,
                ).view(2, 1)
            )
            self.copy_sfa_device_bases.append(
                torch.tensor(
                    [self.topk_buffers_k[layer_id].data_ptr(), self.topk_buffers_v[layer_id].data_ptr()],
                    dtype=torch.int64,
                    device=device,
                ).view(2, 1)
            )

    def dense_fill_copy_sfa_rows(
        self,
        dense_fills: dict[int, tuple[int, int]],
        *,
        block_size: int,
        block_table: np.ndarray | torch.Tensor,
    ) -> None:
        """Restore whole short rows from the host pool after prefix rollback.

        A rollback across the hot boundary leaves a sparse-layout row under a
        dense (-3) reader. Restore it using the input batch's block table
        (a CPU mirror in V1 or a device tensor in V2) and block size, outside
        graph capture at metadata-build time. Host and device bases are bound
        during cache registration.
        """
        topk_k = self.topk_buffers_k[0]
        stride_tokens = topk_k.shape[1]
        token_bytes = torch.tensor(
            [
                [topk_k.element_size() * topk_k.shape[-1]],
                [self.topk_buffers_v[0].element_size() * self.topk_buffers_v[0].shape[-1]],
            ],
            dtype=torch.int64,
            device=topk_k.device,
        )
        block_bytes = token_bytes * block_size
        for slot, (row, computed) in dense_fills.items():
            nblocks = (computed + block_size - 1) // block_size
            if isinstance(block_table, torch.Tensor):
                if block_table.device != topk_k.device:
                    raise ValueError("Tensor block table must be on the same device as the top-k buffers")
                # block_bytes is a dimensioned int64 tensor, not a Python scalar.
                # The product is int64 before adding the host base, including offsets above 2 GiB.
                src_offsets = block_table[row, :nblocks].view(1, -1) * block_bytes
            else:
                src_tokens = torch.tensor(
                    [int(block_table[row, b]) * block_size for b in range(nblocks)], dtype=torch.int64
                ).to(topk_k.device)
                src_offsets = src_tokens.view(1, -1) * token_bytes
            lengths = torch.clamp(
                torch.arange(nblocks, dtype=torch.int64, device=topk_k.device) * (-block_size) + computed,
                min=0,
                max=block_size,
            )
            dst_block_tokens = torch.arange(nblocks, dtype=torch.int64, device=topk_k.device) * block_size
            dst_offsets = (slot * stride_tokens + dst_block_tokens.view(1, -1).expand(2, -1)) * token_bytes
            lengths_bytes = lengths.view(1, -1).expand(2, -1) * token_bytes
            count = torch.full((1,), 2 * nblocks, dtype=torch.int32, device=topk_k.device)
            for host_bases, device_bases in zip(self.copy_sfa_host_bases, self.copy_sfa_device_bases):
                sources = (src_offsets + host_bases).reshape(-1)
                destinations = (dst_offsets + device_bases).reshape(-1)
                self.copy_sfa_kv(sources, destinations, lengths_bytes.reshape(-1), count)

    def restore_copy_sfa_tails(self, metadata) -> None:
        """Copy the current tail descriptors for every layer. Used on prefix rollback."""
        src_off = getattr(metadata, "copy_sfa_copy_src_offsets", None)
        dst_off = getattr(metadata, "copy_sfa_copy_dst_offsets", None)
        lengths = getattr(metadata, "copy_sfa_copy_lengths", None)
        count = getattr(metadata, "copy_sfa_copy_count", None)
        if src_off is None or dst_off is None or lengths is None or count is None:
            return
        if not getattr(self, "copy_sfa_host_bases", None):
            raise RuntimeError("fused_copy_sfa KV base addresses must be bound before tail restore")
        n = src_off.numel()
        for host_bases, device_bases in zip(self.copy_sfa_host_bases, self.copy_sfa_device_bases):
            torch.add(src_off.view(2, -1), host_bases, out=self.copy_sfa_copy_src[:n].view(2, -1))
            torch.add(dst_off.view(2, -1), device_bases, out=self.copy_sfa_copy_dst[:n].view(2, -1))
            self.copy_sfa_kv(self.copy_sfa_copy_src[:n], self.copy_sfa_copy_dst[:n], lengths, count)

    def copy_sfa_kv(self, sources, destinations, lengths, count) -> None:
        """Enqueue bounded descriptor copies on the current compute stream."""
        result = offload.sparse_copy(sources, destinations, lengths, count, sources.device)
        if result not in (None, 0):
            raise RuntimeError(f"memfabric fused_copy_sfa tail H2D failed with result={result}")

    def configure_lim_groups(self, owner_to_members: dict[str, list[str]]) -> None:
        """Register the LIM selection groups and pre-create the side stream.

        Must run before any graph capture: the stream, the per-owner /
        per-member events and the per-group scratch buffers are fixed objects
        reused by every step so the fork-join topology (main LIM -> side
        copies -> member SFA) is deterministic and capture-safe.
        """
        self._lim_group_states = {}
        self._lim_side_stream = None
        self._lim_group_ksc_params = None
        self._lim_group_ksc_geometry = None
        self._lim_group_ksc_ready = _get_lim_desc_op() is not None
        if not owner_to_members:
            return
        device = self.topk_buffers_k[0].device
        requests = self.topk_buffers_k[0].shape[0]
        # One decode-D2H staging set per LIM group, laid out
        # [token, part(=k/v), member]: the live descriptors of a step are
        # then exactly the flat prefix [0, tokens * 2 * members), so the
        # group flush hands the engine contiguous views without any
        # gather/reshape copies. Member i writes its strided slice; the
        # group's last member issues a single sparse_copy for the whole
        # group. Allocated for every offload_shared_indexer_only deployment
        # (inline mode batches the decode D2H per group too); independent of
        # the side stream.
        self._group_d2h_buffers: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for owner_name, member_names in owner_to_members.items():
            if not member_names:
                continue
            # Layout [2, members, tokens]: per-member writes are contiguous
            # rows (stride 1); the flush slices [.., :members, :tc] and
            # reshapes (one gather per group, amortized across its members).
            self._group_d2h_buffers[owner_name] = (
                torch.zeros((2, len(member_names), self.max_num_tokens), dtype=torch.int64, device=device),
                torch.zeros((2, len(member_names), self.max_num_tokens), dtype=torch.int64, device=device),
                torch.zeros((2, len(member_names), self.max_num_tokens), dtype=torch.int32, device=device),
            )
        self._group_d2h_count = torch.zeros(1, dtype=torch.int32, device=device)
        self._group_d2h_step = None
        self._group_d2h_refs = []
        if not self._lim_group_ksc_ready:
            # Documented fallback: without the npu_lim_copy_descriptors
            # binding, miss_onboard_mode=side_stream degrades to inline
            # (members resolve the owner's miss plan inside the fused
            # attention operator; no pre-copy is issued).
            logger.warning(
                "npu_lim_copy_descriptors is unavailable; miss_onboard_mode=side_stream "
                "falls back to inline miss resolution for %d LIM groups.",
                sum(1 for names in owner_to_members.values() if names),
            )
            return
        # Deployment constants, created here (before any graph capture) rather
        # than lazily during the first decode step. The per-step geometry
        # (active request count + live block-table width) travels in the
        # separate _lim_group_ksc_geometry tensor, so this tensor never needs
        # to change after capture.
        if self.head_dim_k <= 0 or self.head_dim_v <= 0:
            self._lim_group_ksc_ready = False
            raise RuntimeError(
                "Sparse KV offload LIM groups require the main KV head dims to be "
                "registered before configure_lim_groups"
            )
        if self.head_dim_k > LIM_KV_DIM_MAX or self.head_dim_v > LIM_ROPE_DIM_MAX:
            self._lim_group_ksc_ready = False
            raise RuntimeError(
                "Sparse KV offload side-stream onboarding supports kv_lora_rank <= "
                f"{LIM_KV_DIM_MAX} and qk_rope_head_dim <= {LIM_ROPE_DIM_MAX}; got "
                f"kv_lora_rank={self.head_dim_k}, qk_rope_head_dim={self.head_dim_v}. "
                "Update LCD_KV_DIM_MAX / LCD_ROPE_DIM_MAX in the C++ kernel together "
                "with LIM_KV_DIM_MAX / LIM_ROPE_DIM_MAX here."
            )
        self._lim_side_stream = torch_npu.npu.Stream(device)
        self._lim_group_ksc_params = torch.tensor(
            [
                self.block_size,
                self.topk_buffer_size,
                LIM_MISS_PLAN_MAX_WIDTH,
                self.head_dim_k,
                self.head_dim_v,
            ],
            dtype=torch.int64,
            device=device,
        )
        # [active_rows, block_table_width]; rewritten every step on the main
        # stream before the side-stream launch.
        self._lim_group_ksc_geometry = torch.zeros(2, dtype=torch.int64, device=device)
        for owner_name, member_names in owner_to_members.items():
            if not member_names:
                continue
            self._lim_group_states[owner_name] = _LIMGroupRuntime(
                member_names=list(member_names),
                plan_event=torch_npu.npu.Event(),
                done_events={name: torch_npu.npu.Event() for name in member_names},
                request_slots=requests,
            )
        logger.info(
            "Sparse KV offload LIM groups: %d owners, %d shared members on the side stream.",
            len(self._lim_group_states),
            sum(len(state.member_names) for state in self._lim_group_states.values()),
        )

    @property
    def lim_group_onboard_ready(self) -> bool:
        """True when side-stream group onboarding is actually active."""
        return self._lim_group_ksc_ready

    def _lim_group_state(self, owner_impl) -> _LIMGroupRuntime | None:
        owner_name = owner_impl.layer_name or owner_impl._current_layer_name
        if owner_name is None:
            raise RuntimeError("group onboarding requires a bound owner layer name")
        return self._lim_group_states.get(owner_name)

    def launch_lim_group_onboard(self, owner_impl, metadata) -> None:
        """Issue the shared members' miss onboarding on the side stream.

        Called right after the owner's LIM operator produced the miss plan on
        the main stream (miss_onboard_mode=side_stream). One KSC operator
        launch per member; when the extension is unavailable the members fall
        back to inline miss resolution inside the fused attention operator.
        """
        if not self._lim_group_ksc_ready:
            return  # inline fallback: the fused op resolves the miss plan itself
        state = self._lim_group_state(owner_impl)
        if state is None:
            return
        count = metadata.copy_sfa_pool_entries.numel()
        if count > state.request_slots:
            raise ValueError(f"onboard request count {count} exceeds descriptor capacity {state.request_slots}")
        # No zeroing of lim_misses beyond `count` is needed: the kernel's
        # active_geometry[0] bounds its request loop to the current batch,
        # and every Python consumer slices the counts to `count`.
        self._launch_lim_group_ksc(_get_lim_desc_op(), owner_impl, state, metadata, count)

    def _launch_lim_group_ksc(self, op, owner_impl, state, metadata, count: int) -> None:
        """KSC-style group onboarding: one operator launch per member.

        Each member's ``npu_lim_copy_descriptors`` launch resolves the owner's
        miss plan through the source block table and moves the missed rows
        from the host pool into that member's hot rows entirely in-kernel —
        no Python address computation and no engine call. Launches run back to
        back on the side stream with one done event per member, so a member
        waits exactly once and only for its own (shorter) copy.
        """
        if not (0 < count <= owner_impl.lim_miss_src.shape[0]):
            raise RuntimeError(
                f"group KSC onboarding geometry check failed (count={count}, rows={owner_impl.lim_miss_src.shape[0]})"
            )
        block_table_width = metadata.copy_sfa_source_block_table.shape[1]
        if block_table_width <= 0:
            raise RuntimeError(
                f"group KSC onboarding requires a non-empty source block table, got width={block_table_width}"
            )
        # [0] active_rows bounds the kernel loop to the current batch (so stale
        # miss counts from a larger previous step are never read); [1] is the
        # live block-table row stride, so the kernel never derives it from
        # max_model_len (which can differ from the table width). Both fills are
        # captured into the graph per capture size and replayed with it.
        self._lim_group_ksc_geometry[0].fill_(count)
        self._lim_group_ksc_geometry[1].fill_(block_table_width)
        torch_npu.npu.current_stream().record_event(state.plan_event)
        with torch_npu.npu.stream(self._lim_side_stream):
            self._lim_side_stream.wait_event(state.plan_event)
            for member_name in state.member_names:
                layer_id = self._get_offload_layer_id(member_name)
                op(
                    owner_impl.lim_miss_src,
                    owner_impl.lim_miss_dst,
                    owner_impl.lim_misses,
                    metadata.copy_sfa_source_block_table[:count],
                    metadata.copy_sfa_pool_entries,
                    self.copy_sfa_host_bases[layer_id].view(2),
                    self.copy_sfa_device_bases[layer_id].view(2),
                    self._lim_group_ksc_params,
                    self._lim_group_ksc_geometry,
                )
                state.done_events[member_name].record()
        # Remember which forward pass this launch belongs to: members of a
        # later pass verify the token before taking the all-zero-miss-plan
        # (misses_already_resident) attention path, so a pass whose owner
        # skipped the LIM (e.g. a draft reuse step) falls back to inline
        # miss resolution instead of reading un-onboarded hot rows. The
        # ForwardContext object is the step token (one instance per pass,
        # shared by every layer of the pass, including other KV cache
        # groups' metadata builders).
        state.last_launch_token = get_forward_context() if is_forward_context_available() else None

    def lim_group_launched_for(self, owner_impl) -> bool:
        """Whether the side-stream onboarding ran in the current forward pass."""
        state = self._lim_group_state(owner_impl)
        if state is None or state.last_launch_token is None:
            return False
        if not is_forward_context_available():
            return False
        return state.last_launch_token is get_forward_context()

    def wait_lim_group_onboard(self, owner_impl, member_impl) -> None:
        """Order the member's resident-SFA reads after its side-stream copy."""
        state = self._lim_group_state(owner_impl)
        if state is None:
            return
        member_name = member_impl.layer_name or member_impl._current_layer_name
        event = state.done_events.get(member_name)
        if event is not None:
            torch_npu.npu.current_stream().wait_event(event)

    def stage_group_d2h(
        self,
        owner_name: str,
        member_idx: int,
        is_last: bool,
        k: torch.Tensor,
        v: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
    ) -> None:
        """Stage one shared member's decode D2H descriptors; flush per group.

        Every member of a group writes its strided slice of the group's
        descriptor buffers (source = the member's freshly computed KV rows,
        destination = the member's CPU pool rows); the last member issues a
        single ``offload.sparse_copy`` for the whole group. The buffers are
        laid out [token, part, member], so the live descriptors are the flat
        prefix [0, tokens * 2 * members) and the flush passes contiguous
        views — no per-flush gather or reshape copies. The per-step geometry
        (pool-slot validity/offsets, lengths, token-index source offsets) is
        computed once per step, in-graph, by the first member. Pending KV
        tensors are referenced until the flush so their addresses stay valid.
        """
        if self.tp_rank != 0:
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("Sparse KV offload TP0 CPU cache is not registered")
        buffers = self._group_d2h_buffers.get(owner_name)
        if buffers is None:
            raise RuntimeError(f"Sparse KV offload D2H staging was not configured for group {owner_name}")
        src_buf, dst_buf, len_buf = buffers
        k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
        v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
        token_count = k_rows.shape[0]
        if token_count > self.max_num_tokens:
            raise ValueError(f"Sparse KV offload decode rows {token_count} exceed D2H capacity {self.max_num_tokens}")
        num_slots = k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        step = self._group_d2h_step_state(slot_mapping, token_count, num_slots, k.device)
        base_k = int(k_cache_cpu.data_ptr())
        base_v = int(v_cache_cpu.data_ptr())
        src_buf[0, member_idx, :token_count].copy_(int(k_rows.data_ptr()) + step["src_off_k"])
        src_buf[1, member_idx, :token_count].copy_(int(v_rows.data_ptr()) + step["src_off_v"])
        dst_buf[0, member_idx, :token_count].copy_(base_k + step["dst_k"])
        dst_buf[1, member_idx, :token_count].copy_(base_v + step["dst_v"])
        len_buf[0, member_idx, :token_count].copy_(step["len_k"])
        len_buf[1, member_idx, :token_count].copy_(step["len_v"])
        # Keep the member's KV allocations alive until the group flush.
        self._group_d2h_refs.append((k_rows, v_rows))
        if not is_last:
            return
        members = member_idx + 1
        self._group_d2h_count.fill_(2 * members * token_count)
        result = offload.sparse_copy(
            src_buf[:, :members, :token_count].reshape(-1),
            dst_buf[:, :members, :token_count].reshape(-1),
            len_buf[:, :members, :token_count].reshape(-1),
            self._group_d2h_count,
            k.device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric D2H sparse_copy failed with result={result}")
        self._group_d2h_refs.clear()

    def _group_d2h_step_state(
        self,
        slot_mapping: torch.Tensor,
        token_count: int,
        num_slots: int,
        device: torch.device,
    ) -> dict:
        """Per-step decode D2H geometry shared by all staged members.

        The cache holds a strong reference to the slot_mapping tensor itself
        and validates identity (``is``), never ``data_ptr()``: a bare address
        can be recycled by a later tensor after the original is freed, which
        would silently reuse stale destination offsets.
        """
        state = self._group_d2h_step
        if (
            state is not None
            and state["slot_mapping"] is slot_mapping
            and state["version"] == slot_mapping._version
            and state["token_count"] == token_count
        ):
            return state
        slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        valid = (slots >= 0) & (slots < num_slots)
        safe_slots = slots.clamp(min=0, max=num_slots - 1)
        token_indices = self.d2h_token_indices_npu[:token_count]
        state = {
            "slot_mapping": slot_mapping,
            "version": slot_mapping._version,
            "token_count": token_count,
            "src_off_k": token_indices * self.token_size_bytes_k,
            "src_off_v": token_indices * self.token_size_bytes_v,
            "dst_k": safe_slots * self.token_size_bytes_k,
            "dst_v": safe_slots * self.token_size_bytes_v,
            "len_k": (valid * self.token_size_bytes_k).to(torch.int32),
            "len_v": (valid * self.token_size_bytes_v).to(torch.int32),
        }
        self._group_d2h_step = state
        return state

    def offload_new_kv(
        self,
        layer_name: str,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,
        v_cache_npu: torch.Tensor | None,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        if not has_prefill and k is not None and v is not None and self.use_fused_overlap:
            layer_id = self._get_offload_layer_id(layer_name)
            self.current_kv_by_layer[layer_id] = (k, v)
        use_side_stream = self.tp_rank == 0 and self.use_fused_overlap and capturing and not has_prefill
        if not use_side_stream:
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
            )
            return

        current_kv_ready = torch_npu.npu.current_stream().record_event()
        with torch_npu.npu.stream(self.current_kv_save_stream):
            self.current_kv_save_stream.wait_event(current_kv_ready)
            self._offload_new_kv_on_current_stream(
                slot_mapping,
                k_cache_cpu,
                v_cache_cpu,
                k_cache_npu,
                v_cache_npu,
                k,
                v,
                has_prefill,
            )

    def _offload_new_kv_on_current_stream(
        self,
        slot_mapping: torch.Tensor,
        k_cache_cpu: torch.Tensor | None,
        v_cache_cpu: torch.Tensor | None,
        k_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        v_cache_npu: torch.Tensor | None,  # prefill (colocate debug only): cache_npu[slot] -> cache_cpu[slot]
        k: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        v: torch.Tensor | None,  # decode: k/v -> cache_cpu[slot]
        has_prefill: bool = False,
        capturing: bool = False,
    ) -> None:
        # the has_prefill path (NPU paged cache -> CPU pool D2H) only exists
        # for single-node PD-colocate debug.
        if self.tp_rank != 0:
            # Decode-produced K/V is replicated across TP ranks, so TP0 alone
            # writes new decode tokens. PD pull fills disjoint parts of this
            # shared pool from all TP ranks through the broadcast GVA.
            return
        if k_cache_cpu is None or v_cache_cpu is None:
            raise RuntimeError("Sparse KV offload TP0 CPU cache is not registered")
        if has_prefill and not self.sparse_kv_offload_config.keep_device_kv_cache:
            raise RuntimeError(
                "Sparse KV offload prefill offload requires "
                "keep_device_kv_cache=True; a PD-disaggregated decode node "
                "never stages prefill KV in an NPU paged cache"
            )

        if has_prefill:
            if k_cache_npu is None or v_cache_npu is None:
                raise ValueError("prefill offload requires NPU paged K/V caches")
            device = k_cache_npu.device
        else:
            if k is None or v is None:
                raise ValueError("decode offload requires current-token K/V")
            device = k.device

        slots = slot_mapping.reshape(-1).to(device=device, dtype=torch.int64)
        token_count = slots.numel()
        if token_count > self.max_num_tokens:
            raise ValueError(
                "Sparse KV offload rows exceed D2H descriptor capacity, "
                f"got {token_count}, capacity={self.max_num_tokens}"
            )

        num_k_slots = k_cache_cpu.numel() * k_cache_cpu.element_size() // self.token_size_bytes_k
        num_v_slots = v_cache_cpu.numel() * v_cache_cpu.element_size() // self.token_size_bytes_v
        if num_k_slots != num_v_slots or num_k_slots <= 0:
            raise ValueError(
                f"Sparse KV offload CPU K/V pools have incompatible token capacities: k={num_k_slots}, v={num_v_slots}"
            )

        valid = (slots >= 0) & (slots < num_k_slots)
        safe_slots = slots.clamp(min=0, max=num_k_slots - 1)

        if has_prefill:
            assert k_cache_npu is not None and v_cache_npu is not None
            src_k = int(k_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_k
            src_v = int(v_cache_npu.data_ptr()) + safe_slots * self.token_size_bytes_v
        else:
            assert k is not None and v is not None
            k_rows = k.reshape(-1, self.token_size_bytes_k // k.element_size())
            v_rows = v.reshape(-1, self.token_size_bytes_v // v.element_size())
            if k_rows.shape[0] != token_count or v_rows.shape[0] != token_count:
                raise ValueError("decode K/V row counts must match slot_mapping")
            if not k_rows.is_contiguous():
                k_rows = k_rows.contiguous()
            if not v_rows.is_contiguous():
                v_rows = v_rows.contiguous()
            token_indices = self.d2h_token_indices_npu[:token_count]
            src_k = int(k_rows.data_ptr()) + token_indices * self.token_size_bytes_k
            src_v = int(v_rows.data_ptr()) + token_indices * self.token_size_bytes_v

        dst_k = int(k_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_k
        dst_v = int(v_cache_cpu.data_ptr()) + safe_slots * self.token_size_bytes_v
        self.d2h_src_ptrs_npu[:token_count].copy_(src_k)
        self.d2h_src_ptrs_npu[token_count : 2 * token_count].copy_(src_v)
        self.d2h_dst_ptrs_npu[:token_count].copy_(dst_k)
        self.d2h_dst_ptrs_npu[token_count : 2 * token_count].copy_(dst_v)
        self.d2h_lengths_npu[:token_count].fill_(self.token_size_bytes_k)
        self.d2h_lengths_npu[token_count : 2 * token_count].fill_(self.token_size_bytes_v)
        self.d2h_lengths_npu[:token_count].masked_fill_(~valid, 0)
        self.d2h_lengths_npu[token_count : 2 * token_count].masked_fill_(~valid, 0)
        self.d2h_size_npu.fill_(2 * token_count)

        result = offload.sparse_copy(
            self.d2h_src_ptrs_npu,
            self.d2h_dst_ptrs_npu,
            self.d2h_lengths_npu,
            self.d2h_size_npu,
            device,
        )
        if result not in (None, 0):
            raise RuntimeError(f"memfabric D2H sparse_copy failed with result={result}")

    def prepare_fused_overlap_external_plan(
        self,
        layer_name: str,
        num_tokens: int,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        visible_seq_lens_npu: torch.Tensor,
        selection_membership_map: torch.Tensor,
        capturing: bool = False,
        skip_topk: bool = False,
    ) -> bool:
        self._validate_fused_overlap_external_plan_inputs(
            num_tokens,
            topk_indices_npu,
            req_ids_npu,
            stable_prefix_lens_npu,
            visible_seq_lens_npu,
            selection_membership_map,
        )
        layer_id = self._get_offload_layer_id(layer_name)
        plan_start = FSA_SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_CNT - self.topk
        plan_storage = selection_membership_map[
            :num_tokens,
            plan_start:FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS,
        ]
        encoded_plan_stride = selection_membership_map.stride(0)

        owner_layer_id = self.fused_overlap_plan_owner_layer_id
        owner_map = self.fused_overlap_plan_membership_map
        can_reuse_owner_plan = (
            skip_topk
            and owner_layer_id is not None
            and layer_id > owner_layer_id
            and self.fused_overlap_plan_topk == self.topk
            and self.fused_overlap_plan_num_tokens == num_tokens
            and owner_map is not None
        )
        if can_reuse_owner_plan:
            assert owner_map is not None
            if selection_membership_map.data_ptr() != owner_map.data_ptr():
                if capturing:
                    torch_npu.npu.current_stream().wait_stream(self.fused_plan_stream)
                plan_storage.copy_(
                    owner_map[:num_tokens, plan_start:FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS],
                    non_blocking=capturing,
                )
            return True

        def run_planner(enqueue: bool) -> None:
            sparse_kv_ops = _sparse_kv_ops()
            planner = (
                sparse_kv_ops.sparse_kv_enqueue_lru_resident_compact_with_plan_stable_rows
                if enqueue
                else sparse_kv_ops.sparse_kv_lru_resident_compact_with_plan_stable_rows
            )
            planner(
                self.lru_req_ids_ptr,
                self.lru_last_req_ids_ptrs[layer_id],
                self.lru_topk_indices_ptr,
                self.lru_stable_prefix_lens_ptr,
                self.lru_slot_to_token_ptrs[layer_id],
                self.lru_slots_ptrs[layer_id],
                self.lru_current_slots_ptr,
                self.lru_miss_count_ptrs[layer_id],
                self.lru_miss_tokens_ptrs[layer_id],
                self.lru_miss_slots_ptrs[layer_id],
                self.lru_token_mark_workspace_ptr,
                self.lru_token_pos_workspace_ptr,
                self.lru_slot_workspace_ptr,
                self.lru_miss_position_workspace_ptr,
                self.lru_epochs_ptr,
                self.lru_physical_row_workspace_ptr,
                self.max_num_topk_rows,
                plan_storage.data_ptr(),
                encoded_plan_stride,
                num_tokens,
                self.topk,
                self.topk_buffer_size,
                self.max_model_len,
                self.lru_workspace_threads,
                self.lru_workspace_threads,
                self.lru_visible_seq_lens_ptr,
            )

        if capturing:
            plan_inputs_ready = torch_npu.npu.current_stream().record_event()
            with torch_npu.npu.stream(self.fused_plan_stream):
                self.fused_plan_stream.wait_event(plan_inputs_ready)
                if self.tp_rank == 0:
                    self.lru_topk_indices_cpu[:num_tokens].copy_(topk_indices_npu, non_blocking=True)
                    self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu, non_blocking=True)
                    self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(stable_prefix_lens_npu, non_blocking=True)
                    self.lru_visible_seq_lens_cpu[:num_tokens].copy_(visible_seq_lens_npu, non_blocking=True)
                    run_planner(enqueue=True)
                    self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                        self.lru_physical_row_workspace[
                            self.max_num_topk_rows * 2 : self.max_num_topk_rows * 2 + num_tokens
                        ],
                        non_blocking=True,
                    )
                self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
            self.fused_overlap_plan_owner_layer_id = layer_id
            self.fused_overlap_plan_topk = self.topk
            self.fused_overlap_plan_num_tokens = num_tokens
            self.fused_overlap_plan_membership_map = selection_membership_map
            return True

        planner_error = None
        if self.tp_rank == 0:
            self.fused_plan_status_npu.zero_()
            try:
                self.lru_topk_indices_cpu[:num_tokens].copy_(topk_indices_npu)
                self.lru_req_ids_cpu[:num_tokens].copy_(req_ids_npu)
                self.lru_stable_prefix_lens_cpu[:num_tokens].copy_(stable_prefix_lens_npu)
                self.lru_visible_seq_lens_cpu[:num_tokens].copy_(visible_seq_lens_npu)
                run_planner(enqueue=False)
                self.fused_plan_current_linear_slots_npu[:num_tokens].copy_(
                    self.lru_physical_row_workspace[
                        self.max_num_topk_rows * 2 : self.max_num_topk_rows * 2 + num_tokens
                    ]
                )
            except Exception as exc:
                planner_error = exc
                self.fused_plan_status_npu.fill_(1)
        self.tp_group.broadcast(self.fused_plan_metadata_npu, src=0)
        if int(self.fused_plan_status_npu.item()) != 0:
            detail = (
                f"{type(planner_error).__name__}: {planner_error}"
                if planner_error is not None
                else "TP0 external planner failed"
            )
            raise RuntimeError(
                f"SFA fused_overlap external planner failed: layer={layer_name}, tp_rank={self.tp_rank}, {detail}"
            ) from planner_error
        self.fused_overlap_plan_owner_layer_id = layer_id
        self.fused_overlap_plan_topk = self.topk
        self.fused_overlap_plan_num_tokens = num_tokens
        self.fused_overlap_plan_membership_map = selection_membership_map
        return True

    def _validate_fused_overlap_external_plan_inputs(
        self,
        num_tokens: int,
        topk_indices_npu: torch.Tensor,
        req_ids_npu: torch.Tensor,
        stable_prefix_lens_npu: torch.Tensor,
        visible_seq_lens_npu: torch.Tensor,
        selection_membership_map: torch.Tensor,
    ) -> None:
        if not self.use_fused_overlap:
            raise RuntimeError("external FSA plan requires fused_overlap mode")
        if num_tokens <= 0 or num_tokens > self.max_num_topk_rows:
            raise ValueError(
                "external FSA plan rows exceed configured workspace: "
                f"num_tokens={num_tokens}, max_rows={self.max_num_topk_rows}"
            )
        expected_topk_shape = (num_tokens, self.topk)
        if tuple(topk_indices_npu.shape) != expected_topk_shape:
            raise ValueError(
                "external FSA plan requires flattened single-head TopK input: "
                f"expected={expected_topk_shape}, actual={tuple(topk_indices_npu.shape)}"
            )
        if tuple(req_ids_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one request id per logical row: "
                f"expected=({num_tokens},), actual={tuple(req_ids_npu.shape)}"
            )
        if tuple(stable_prefix_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one stable prefix length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(stable_prefix_lens_npu.shape)}"
            )
        if tuple(visible_seq_lens_npu.shape) != (num_tokens,):
            raise ValueError(
                "external FSA plan requires one visible KV length per logical row: "
                f"expected=({num_tokens},), "
                f"actual={tuple(visible_seq_lens_npu.shape)}"
            )
        if (
            selection_membership_map.dim() != 2
            or selection_membership_map.shape[0] < num_tokens
            or selection_membership_map.shape[1] < FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS
            or selection_membership_map.dtype != torch.int16
            or selection_membership_map.device.type != "cpu"
        ):
            raise ValueError(
                "external FSA plan requires mapped CPU int16 membership storage: "
                f"min_shape=({num_tokens}, {FSA_SELECTION_MEMBERSHIP_REQUIRER_COLUMNS}), "
                f"actual_shape={tuple(selection_membership_map.shape)}, "
                f"dtype={selection_membership_map.dtype}, "
                f"device={selection_membership_map.device}"
            )

    def inject_current_kv_into_selection(
        self,
        layer_name: str,
        num_tokens: int,
        selection_kv_cache: torch.Tensor,
        selection_k_rope: torch.Tensor,
        capturing: bool = False,
    ) -> None:
        layer_id = self._get_offload_layer_id(layer_name)
        current_kv = self.current_kv_by_layer.get(layer_id)
        if current_kv is None:
            raise RuntimeError(f"current decode K/V is unavailable for fused layer {layer_name}")
        if capturing:
            torch_npu.npu.current_stream().wait_stream(self.fused_plan_stream)
        current_k, current_rope = current_kv
        current_k = current_k.reshape(-1, selection_kv_cache.shape[-1])[:num_tokens]
        current_rope = current_rope.reshape(-1, selection_k_rope.shape[-1])[:num_tokens]
        linear_slots = self.fused_plan_current_linear_slots_npu[:num_tokens]
        indices = linear_slots.view(-1, 1)
        flat_kv = selection_kv_cache.reshape(-1, selection_kv_cache.shape[-1])
        flat_rope = selection_k_rope.reshape(-1, selection_k_rope.shape[-1])
        torch_npu.npu_scatter_nd_update_(flat_kv, indices, current_k)
        torch_npu.npu_scatter_nd_update_(flat_rope, indices, current_rope)

    def wait_for_current_kv_writeback(self, capturing: bool = False) -> None:
        if self.use_fused_overlap and capturing and self.tp_rank == 0:
            torch_npu.npu.current_stream().wait_stream(self.current_kv_save_stream)


_SPARSE_KV_OFFLOAD_MANAGER: SparseKVOffloadManager | None = None


def init_sparse_kv_offload_manager(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    sparse_kv_offload_config: SparseKVOffloadConfig,
):
    global _SPARSE_KV_OFFLOAD_MANAGER
    if _SPARSE_KV_OFFLOAD_MANAGER is None:
        if get_ascend_device_type() != AscendDeviceType.A3:
            raise RuntimeError("Sparse KV offload is only support on A3")
        _SPARSE_KV_OFFLOAD_MANAGER = SparseKVOffloadManager(
            vllm_config,
            kv_cache_config,
            sparse_kv_offload_config,
        )
    return _SPARSE_KV_OFFLOAD_MANAGER


def get_sparse_kv_offload_manager():
    assert _SPARSE_KV_OFFLOAD_MANAGER is not None, "KV offload manager is not initialized."
    return _SPARSE_KV_OFFLOAD_MANAGER
