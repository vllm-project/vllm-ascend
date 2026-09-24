# mypy: ignore-errors

import itertools
from typing import Any

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch
from vllm.v1.worker.mamba_utils import MambaCopyBuffers
from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.ops.triton.mamba.postprocess import postprocess_mamba_fused_kernel
from vllm_ascend.utils import is_310p


def _can_launch_triton_batch_memcpy() -> bool:
    return not is_310p()


def _batch_memcpy_triton(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


def _tensor_view_from_data_ptr(state: torch.Tensor, start_addr: int, num_elements: int) -> torch.Tensor:
    byte_offset = start_addr - state.data_ptr()
    element_size = state.element_size()
    if byte_offset < 0 or byte_offset % element_size != 0:
        raise RuntimeError("Invalid Mamba state copy pointer.")

    element_offset = byte_offset // element_size
    flat_state = state.view(-1)
    if element_offset + num_elements > flat_state.numel():
        raise RuntimeError("Mamba state copy range exceeds tensor storage.")
    return flat_state.narrow(0, element_offset, num_elements)


def _get_tensor_copy_pairs(copy_bufs: mamba_utils.MambaCopyBuffers) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if copy_bufs.offset == 0 or not hasattr(copy_bufs, "_tensor_copy_pairs"):
        copy_bufs._tensor_copy_pairs = []
    return copy_bufs._tensor_copy_pairs


def _collect_mamba_copy_meta_torch(
    copy_bufs: mamba_utils.MambaCopyBuffers,
    kv_cache_config,
    mamba_state_copy_funcs,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    tensor_copy_pairs = _get_tensor_copy_pairs(copy_bufs)
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)
                src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                dst_state = _tensor_view_from_data_ptr(state, state[dest_block_id].data_ptr(), copy_spec.num_elements)
                tensor_copy_pairs.append((src_state, dst_state))
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def _do_mamba_copy_block_torch(copy_bufs: mamba_utils.MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        if hasattr(copy_bufs, "_tensor_copy_pairs"):
            copy_bufs._tensor_copy_pairs = []
        return

    tensor_copy_pairs = getattr(copy_bufs, "_tensor_copy_pairs", None)
    if tensor_copy_pairs is None or len(tensor_copy_pairs) != n:
        raise RuntimeError("Mamba tensor copy metadata is incomplete.")

    for src_state, dst_state in tensor_copy_pairs:
        dst_state.copy_(src_state.clone())
    copy_bufs._tensor_copy_pairs = []


def _postprocess_mamba_align_gpu_cpu_fallback(
    *,
    bufs: "mamba_utils.MambaBuffers",
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: Any,
) -> None:
    """CPU fallback for 310P where the Triton fused postprocess is unavailable."""
    ctx = bufs.postprocess_align
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    # stage_postprocess_inputs_to_gpu has already materialized the same
    # per-request values into the CpuGpuBuffer numpy views. 310P cannot use the
    # Triton fused kernel, so reuse the CPU views to mirror its decision logic.
    mamba_state_idx = ctx.mamba_state_idx_buf.np
    num_scheduled_tokens = ctx.num_scheduled_tokens_buf.np
    num_computed_tokens = ctx.num_computed_tokens_buf.np
    num_draft_tokens = ctx.num_draft_tokens_buf.np
    block_size = ctx.block_size
    copy_funcs_by_type = _resolve_copy_funcs(mamba_state_copy_funcs, kv_cache_config, ctx)

    # Upstream initializes num_accepted_tokens_out from the real accepted-token
    # counts, then only overwrites entries where src and dest are the same
    # block. Preserve that default so the next preprocess keeps the right
    # accept_token_bias when multiple draft tokens were accepted.
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])
    num_accepted_tokens = input_batch.num_accepted_tokens_cpu
    for i in range(num_reqs):
        num_tokens_running_state = num_computed_tokens[i] + num_scheduled_tokens[i] - num_draft_tokens[i]
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens[i] - 1
        aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size
        if aligned_new_computed_tokens < num_tokens_running_state:
            continue

        src_block_idx = mamba_state_idx[i]
        dest_block_idx = aligned_new_computed_tokens // block_size - 1
        accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
        if src_block_idx == dest_block_idx:
            # Match the fused kernel: once the running state remains in the
            # same block, the next preprocess should start from token bias 0.
            num_accepted_tokens_cpu_tensor[i] = 1
            if accept_token_bias == 0:
                continue

        # The upstream fused kernel also copies Mamba state in this postprocess
        # step. Do the same with tensor views so 310P avoids Triton without
        # changing where conv/temporal state lands before the next iteration.
        for mamba_group_id in ctx.mamba_group_ids:
            block_ids = input_batch.block_table[mamba_group_id].get_numpy_array()[i]
            dest_block_id = block_ids[dest_block_idx]
            group = kv_cache_config.kv_cache_groups[mamba_group_id]
            for layer_name in group.layer_names:
                funcs = copy_funcs_by_type[_get_mamba_spec_for_layer(group, layer_name).mamba_type]
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache
                if len(kv_caches) < len(funcs):
                    raise ValueError(
                        f"Expected at least {len(funcs)} Mamba state tensors, got {len(kv_caches)} for {layer_name}"
                    )
                for state, state_copy_func in zip(kv_caches, funcs):
                    copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)
                    src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                    dst_state = _tensor_view_from_data_ptr(
                        state, state[dest_block_id].data_ptr(), copy_spec.num_elements
                    )
                    dst_state.copy_(src_state.clone())


def _batch_memcpy_unavailable(src_ptrs, dst_ptrs, sizes):
    raise RuntimeError(
        "Pointer-based Mamba batch memcpy requires Triton and is not available "
        "on 310P. Use the tensor-copy fallback path instead."
    )


if _can_launch_triton_batch_memcpy():
    mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
    mamba_utils.batch_memcpy = _batch_memcpy_triton
    mamba_utils.postprocess_mamba_fused_kernel = postprocess_mamba_fused_kernel
else:
    mamba_utils.batch_memcpy = _batch_memcpy_unavailable
    mamba_utils.collect_mamba_copy_meta = _collect_mamba_copy_meta_torch
    mamba_utils.do_mamba_copy_block = _do_mamba_copy_block_torch
    mamba_utils.postprocess_mamba_align_gpu = _postprocess_mamba_align_gpu_cpu_fallback

# Ascend NPU does not support DT_UINT64 in aclnnInplaceZero.
# MambaCopyBuffers.create() uses torch.uint64 for src_ptrs/dst_ptrs,
# which triggers a runtime error. Remap to int64 at the source.
_original_create = MambaCopyBuffers.create


@classmethod
def _patched_create(cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer):
    return _original_create(
        max_num_reqs,
        kv_cache_config,
        copy_funcs,
        lambda n, dtype: make_buffer(n, dtype=torch.int64 if dtype == torch.uint64 else dtype),
    )


MambaCopyBuffers.create = _patched_create


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    # assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size) + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            mamba_utils.collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    # do not copy here, since kv_transfer still not load
    # do_mamba_copy_block(copy_bufs)


mamba_utils.preprocess_mamba = preprocess_mamba


# Backport of heterogeneous-Mamba state-copy support from vLLM main. v0.26
# assumes every Mamba layer has an identical MambaSpec and multiplies every
# layer by one global number of copy functions. Qwen4Exp has GDN layers with
# conv+temporal state and a PLE short-conv layer with conv-only state.
MambaStateCopyFuncsByType = dict[MambaAttentionBackendEnum, tuple[MambaStateCopyFunc, ...]]


def _get_mamba_spec_for_layer(kv_cache_group, layer_name: str) -> MambaSpec:
    kv_cache_spec = kv_cache_group.kv_cache_spec
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        kv_cache_spec = kv_cache_spec.kv_cache_specs[layer_name]
    assert isinstance(kv_cache_spec, MambaSpec), f"layer {layer_name} does not have a MambaSpec: {type(kv_cache_spec)}"
    return kv_cache_spec


def get_mamba_group_ids(
    mamba_groups: dict[MambaSpec, list[int]],
) -> list[int]:
    return sorted({group_id for group_ids in mamba_groups.values() for group_id in group_ids})


def get_mamba_groups(
    kv_cache_config: KVCacheConfig,
) -> dict[MambaSpec, list[int]]:
    """Map each distinct real per-layer MambaSpec to its cache group ids."""
    mamba_groups: dict[MambaSpec, set[int]] = {}
    for group_id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            first_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        else:
            first_spec = kv_cache_spec
        if not isinstance(first_spec, MambaSpec):
            continue
        for layer_name in kv_cache_group.layer_names:
            layer_spec = _get_mamba_spec_for_layer(kv_cache_group, layer_name)
            mamba_groups.setdefault(layer_spec, set()).add(group_id)
    assert mamba_groups, "no mamba layers in the model"
    return {spec: sorted(group_ids) for spec, group_ids in mamba_groups.items()}


def validate_mamba_state_copy_funcs(
    mamba_groups: dict[MambaSpec, list[int]],
    copy_funcs: MambaStateCopyFuncsByType,
) -> None:
    for mamba_spec in mamba_groups:
        assert mamba_spec.mamba_type in copy_funcs, f"missing state copy funcs for {mamba_spec.mamba_type}"
        funcs = copy_funcs[mamba_spec.mamba_type]
        assert 0 < len(funcs) <= len(mamba_spec.shapes), (
            f"{mamba_spec.mamba_type} declares {len(mamba_spec.shapes)} states, "
            f"but provides {len(funcs)} state copy funcs; expected a non-empty "
            "copyable prefix"
        )


def _validate_scheduling_parameters(
    mamba_groups: dict[MambaSpec, list[int]],
) -> MambaSpec:
    anchor = next(iter(mamba_groups))
    assert all(
        spec.block_size == anchor.block_size
        and spec.num_speculative_blocks == anchor.num_speculative_blocks
        and spec.mamba_cache_mode == anchor.mamba_cache_mode
        for spec in mamba_groups
    ), "all mamba groups must share cache scheduling parameters"
    return anchor


def _resolve_copy_funcs(
    copy_funcs: MambaStateCopyFuncsByType | tuple[MambaStateCopyFunc, ...],
    kv_cache_config: KVCacheConfig,
    owner: object | None = None,
) -> MambaStateCopyFuncsByType:
    stored = getattr(owner, "mamba_state_copy_funcs_by_type", None)
    if stored is not None:
        return stored
    if isinstance(copy_funcs, dict):
        return copy_funcs
    mamba_groups = get_mamba_groups(kv_cache_config)
    mamba_types = {spec.mamba_type for spec in mamba_groups}
    assert len(mamba_types) == 1, "heterogeneous Mamba specs require copy funcs keyed by mamba_type"
    return {next(iter(mamba_types)): tuple(copy_funcs)}


@classmethod
def _heterogeneous_copy_buffers_create(
    cls,
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    copy_funcs: MambaStateCopyFuncsByType,
    make_buffer,
):
    mamba_groups = get_mamba_groups(kv_cache_config)
    validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
    anchor = _validate_scheduling_parameters(mamba_groups)
    group_ids = get_mamba_group_ids(mamba_groups)
    entries_per_req = sum(
        len(copy_funcs[_get_mamba_spec_for_layer(kv_cache_config.kv_cache_groups[group_id], layer_name).mamba_type])
        for group_id in group_ids
        for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names
    )
    n = max_num_reqs * entries_per_req
    result = cls(
        src_ptrs=make_buffer(n, dtype=torch.int64),
        dst_ptrs=make_buffer(n, dtype=torch.int64),
        sizes=make_buffer(n, dtype=torch.int32),
        mamba_group_ids=group_ids,
        mamba_spec=anchor,
    )
    result.mamba_state_copy_funcs_by_type = copy_funcs
    result.entries_per_req = entries_per_req
    return result


@classmethod
def _heterogeneous_decode_context_create(
    cls,
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    copy_funcs: MambaStateCopyFuncsByType,
    device: torch.device,
    make_buffer,
):
    mamba_groups = get_mamba_groups(kv_cache_config)
    validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
    anchor = _validate_scheduling_parameters(mamba_groups)
    group_ids = get_mamba_group_ids(mamba_groups)
    total_states = sum(
        len(copy_funcs[_get_mamba_spec_for_layer(kv_cache_config.kv_cache_groups[group_id], layer_name).mamba_type])
        for group_id in group_ids
        for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names
    )
    # v0.26's fused launch multiplies num_layers * num_state_types. Store the
    # already flattened real state count as N*1. Keep num_states for forward
    # compatibility with the main-line representation.
    result = cls(
        state_base_addrs=torch.zeros(total_states, dtype=torch.int64, device=device),
        state_block_strides=torch.zeros(total_states, dtype=torch.int64, device=device),
        state_elem_sizes=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_inner_sizes=torch.zeros(total_states, dtype=torch.int64, device=device),
        state_conv_widths=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_group_indices=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_dim_row_count=torch.zeros(total_states, dtype=torch.int32, device=device),
        state_dim_row_stride=torch.zeros(total_states, dtype=torch.int64, device=device),
        block_size=anchor.block_size,
        num_layers=total_states,
        num_state_types=1,
        mamba_group_ids=group_ids,
        num_groups=len(group_ids),
        num_accepted_tokens_out=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        block_table_ptrs=torch.zeros(len(group_ids), dtype=torch.int64, device=device),
        mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
        is_initialized=False,
    )
    result.num_states = total_states
    result.mamba_state_copy_funcs_by_type = copy_funcs
    return result


def _heterogeneous_initialize_from_forward_context(
    self,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs,
    block_tables: list[torch.Tensor],
) -> None:
    if self.is_initialized:
        return
    copy_funcs_by_type = _resolve_copy_funcs(mamba_state_copy_funcs, kv_cache_config, self)
    idx = 0
    for local_group_id, group_id in enumerate(self.mamba_group_ids):
        group = kv_cache_config.kv_cache_groups[group_id]
        for layer_name in group.layer_names:
            spec = _get_mamba_spec_for_layer(group, layer_name)
            funcs = copy_funcs_by_type[spec.mamba_type]
            states: list[torch.Tensor] = forward_context[layer_name].kv_cache
            if len(states) < len(funcs):
                raise ValueError(
                    f"Expected at least {len(funcs)} Mamba state tensors, got {len(states)} for {layer_name}"
                )
            for state, copy_func in zip(states, funcs):
                self.state_base_addrs[idx] = state.data_ptr()
                block_stride_elems = state.stride(0) if state.dim() > 1 else state.numel()
                self.state_block_strides[idx] = block_stride_elems * state.element_size()
                self.state_elem_sizes[idx] = state.element_size()
                assert copy_func in (get_conv_copy_spec, get_temporal_copy_spec), f"unexpected copy func: {copy_func}"
                if copy_func is get_conv_copy_spec:
                    if state.dim() != 3:
                        raise ValueError(f"Expected 3D conv state cache, got shape {tuple(state.shape)}")
                    if is_conv_state_dim_first():
                        self.state_conv_widths[idx] = state.size(2)
                        self.state_inner_sizes[idx] = 1
                        self.state_dim_row_count[idx] = state.size(1)
                        self.state_dim_row_stride[idx] = state.stride(1) * state.element_size()
                    else:
                        self.state_conv_widths[idx] = state.size(1)
                        self.state_inner_sizes[idx] = state.stride(1)
                else:
                    self.state_conv_widths[idx] = 0
                    self.state_inner_sizes[idx] = state[0].numel() if state.dim() > 1 else 1
                    if state.data_ptr() % 8 != 0:
                        raise ValueError(f"{layer_name} temporal state base is not 8B-aligned")
                    if int(self.state_block_strides[idx]) % 8 != 0:
                        raise ValueError(f"{layer_name} temporal state stride is not 8B-aligned")
                self.state_group_indices[idx] = local_group_id
                idx += 1
    assert idx == self.num_states, f"fused Mamba metadata count mismatch: expected {self.num_states}, got {idx}"
    assert len(block_tables) == self.num_groups, f"expected {self.num_groups} block tables, got {len(block_tables)}"
    strides = {table.stride(0) for table in block_tables}
    assert len(strides) == 1, f"all mamba block tables must share stride(0), got {strides}"
    self.block_table_stride_req = int(next(iter(strides)))
    for i, table in enumerate(block_tables):
        self.block_table_ptrs[i] = table.data_ptr()
    self.is_initialized = True


@classmethod
def _heterogeneous_buffers_create(
    cls,
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    copy_funcs: MambaStateCopyFuncsByType,
    make_buffer,
    device: torch.device,
    with_postprocess_align: bool,
):
    preprocess_buf = mamba_utils.MambaCopyBuffers.create(max_num_reqs, kv_cache_config, copy_funcs, make_buffer)
    postprocess_ctx = (
        mamba_utils.MambaSpecDecodeGPUContext.create(
            max_num_reqs=max_num_reqs,
            kv_cache_config=kv_cache_config,
            copy_funcs=copy_funcs,
            device=device,
            make_buffer=make_buffer,
        )
        if with_postprocess_align
        else None
    )
    result = cls(preprocess=preprocess_buf, postprocess_align=postprocess_ctx)
    result.mamba_state_copy_funcs_by_type = copy_funcs
    return result


def _heterogeneous_collect_mamba_copy_meta(
    copy_bufs,
    kv_cache_config,
    mamba_state_copy_funcs,
    mamba_group_ids,
    src_block_idx,
    dest_block_idx,
    accept_token_bias,
    req_state,
    forward_context,
):
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return
    copy_funcs_by_type = _resolve_copy_funcs(mamba_state_copy_funcs, kv_cache_config, copy_bufs)
    offset = copy_bufs.offset
    tensor_pairs = _get_tensor_copy_pairs(copy_bufs) if not _can_launch_triton_batch_memcpy() else None
    for group_id in mamba_group_ids:
        block_ids = req_state.block_ids[group_id]
        dest_block_id = block_ids[dest_block_idx]
        group = kv_cache_config.kv_cache_groups[group_id]
        for layer_name in group.layer_names:
            funcs = copy_funcs_by_type[_get_mamba_spec_for_layer(group, layer_name).mamba_type]
            states: list[torch.Tensor] = forward_context[layer_name].kv_cache
            if len(states) < len(funcs):
                raise ValueError(
                    f"Expected at least {len(funcs)} Mamba state tensors, got {len(states)} for {layer_name}"
                )
            for state, copy_func in zip(states, funcs):
                assert offset < copy_bufs.sizes.np.shape[0], "Mamba copy buffer overflow"
                copy_spec = copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)
                if tensor_pairs is not None:
                    src = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                    dst = _tensor_view_from_data_ptr(
                        state,
                        state[dest_block_id].data_ptr(),
                        copy_spec.num_elements,
                    )
                    tensor_pairs.append((src, dst))
                else:
                    copy_bufs.src_ptrs.np[offset] = copy_spec.start_addr
                    copy_bufs.dst_ptrs.np[offset] = state[dest_block_id].data_ptr()
                copy_bufs.sizes.np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1
    copy_bufs.offset = offset


def _heterogeneous_postprocess_mamba_all(
    scheduler_output,
    kv_cache_config,
    input_batch,
    requests,
    mamba_state_idx,
    num_spec_tokens,
    num_reqs,
):
    """v0.26 all-mode compatibility for the new spec-to-groups mapping."""
    if num_spec_tokens <= 0:
        return
    mamba_groups = get_mamba_groups(kv_cache_config)
    anchor = _validate_scheduling_parameters(mamba_groups)
    block_size = anchor.block_size
    full_decode_len = 1 + num_spec_tokens
    scheduled = scheduler_output.num_scheduled_tokens
    for req_id in input_batch.req_ids[:num_reqs]:
        num_query = scheduled.get(req_id, 0)
        if num_query == full_decode_len:
            req = requests[req_id]
            seq_len = req.num_computed_tokens + num_query
            mamba_state_idx[req_id] = max(0, (seq_len - 1) // block_size)
        else:
            mamba_state_idx.pop(req_id, None)


mamba_utils.MambaStateCopyFuncsByType = MambaStateCopyFuncsByType
mamba_utils._get_mamba_spec_for_layer = _get_mamba_spec_for_layer
mamba_utils.get_mamba_group_ids = get_mamba_group_ids
mamba_utils.get_mamba_groups = get_mamba_groups
mamba_utils.patch_mamba_utils_resolve_copy_funcs = _resolve_copy_funcs
mamba_utils.validate_mamba_state_copy_funcs = validate_mamba_state_copy_funcs
mamba_utils.MambaCopyBuffers.create = _heterogeneous_copy_buffers_create
mamba_utils.MambaSpecDecodeGPUContext.create = _heterogeneous_decode_context_create
mamba_utils.MambaSpecDecodeGPUContext.initialize_from_forward_context = _heterogeneous_initialize_from_forward_context
mamba_utils.MambaBuffers.create = _heterogeneous_buffers_create
mamba_utils.collect_mamba_copy_meta = _heterogeneous_collect_mamba_copy_meta
mamba_utils.postprocess_mamba_all = _heterogeneous_postprocess_mamba_all
if not _can_launch_triton_batch_memcpy():
    mamba_utils.postprocess_mamba_align_gpu = _postprocess_mamba_align_gpu_cpu_fallback
