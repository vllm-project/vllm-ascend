# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache binding and commit orchestration for GDN RecoverSSM."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from vllm_ascend.ops.triton.gdn.gdn_recoverssm_commit import gdn_recoverssm_commit
from vllm_ascend.ops.triton.gdn.gdn_recoverssm_verify import validate_gdn_state_replay
from vllm_ascend.ops.triton.mamba.compact_conv_state import compact_conv_state
from vllm_ascend.ops.triton.mamba.prepare_commit_plan import prepare_commit_plan
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

GDN_PAGE_COUNT = 3  # conv, checkpoint, packed FP32 replay


def _inner_stride_signature(tensor):
    return tuple(stride if size > 1 else 0 for size, stride in zip(tensor.shape[1:], tensor.stride()[1:]))


@dataclass(frozen=True)
class GDNRecoverSSMLayerCache:
    """Cache tensors and kernel address metadata for one GDN layer."""

    conv: torch.Tensor
    conv_base_addr: int
    conv_block_stride: int
    conv_dim_stride: int
    conv_token_stride: int
    checkpoint: torch.Tensor
    checkpoint_base_addr: int
    checkpoint_block_stride: int
    replay: torch.Tensor
    replay_base_addr: int
    replay_block_stride: int
    num_key_heads: int

    @classmethod
    def from_layer(cls, layer: Any) -> "GDNRecoverSSMLayerCache":
        if len(layer.kv_cache) != GDN_PAGE_COUNT:
            raise ValueError("GDN RecoverSSM requires conv, checkpoint and replay views")
        conv, checkpoint, replay = layer.kv_cache
        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        return cls(
            conv=conv,
            conv_base_addr=conv.data_ptr(),
            conv_block_stride=conv.stride(0),
            conv_dim_stride=conv.stride(1),
            conv_token_stride=conv.stride(2),
            checkpoint=checkpoint,
            checkpoint_base_addr=checkpoint.data_ptr(),
            checkpoint_block_stride=checkpoint.stride(0),
            replay=replay,
            replay_base_addr=replay.data_ptr(),
            replay_block_stride=replay.stride(0),
            num_key_heads=layer.num_k_heads // layer.tp_size,
        )

    def state_pages(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.checkpoint, self.replay

    def all_pages(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.conv, *self.state_pages()


def _create_state_commit_cache_address_stride_table(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> torch.Tensor:
    """Create the cache-kind-major address/stride table consumed by the kernel."""
    return (
        torch.tensor(
            [
                [
                    [cache.checkpoint_base_addr, cache.checkpoint_block_stride],
                    [cache.replay_base_addr, cache.replay_block_stride],
                ]
                for cache in layer_caches
            ],
            dtype=torch.int64,
            device=layer_caches[0].checkpoint.device,
        )
        .permute(1, 2, 0)
        .contiguous()
    )


def _create_conv_cache_address_stride_table(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> torch.Tensor:
    """Create the field-major convolution address/stride table for the kernel."""
    return torch.tensor(
        [
            [
                cache.conv_base_addr,
                cache.conv_block_stride,
                cache.conv_dim_stride,
                cache.conv_token_stride,
            ]
            for cache in layer_caches
        ],
        dtype=torch.int64,
        device=layer_caches[0].conv.device,
    ).T.contiguous()


def _get_shared_conv_dim_stride(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
) -> int | None:
    """Return the common dim stride, or None when layers use different strides."""
    reference_stride = layer_caches[0].conv_dim_stride
    if all(cache.conv_dim_stride == reference_stride for cache in layer_caches):
        return reference_stride
    return None


@dataclass(frozen=True)
class GDNRecoverSSMCommitPlan:
    """Reusable per-request buffers populated by the commit-plan kernel."""

    commit_lens: torch.Tensor
    state_commit_lens: torch.Tensor
    boundary_commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor

    @classmethod
    def allocate(cls, max_num_reqs: int, device: torch.device) -> "GDNRecoverSSMCommitPlan":
        # Allocate one reusable contiguous buffer for six per-request plan
        # arrays: commit length, non-boundary/boundary commit lengths, final and
        # boundary state indices, and boundary recovery length. The row slices
        # below are views sharing this allocation. The plan kernel overwrites
        # every row in [:batch] before use, so uninitialized capacity beyond the
        # current batch is safe and avoids an unnecessary device-side fill.
        #
        #                                        request 0   request 1   request 2   request 3
        #                                      ┌───────────┬───────────┬───────────┬───────────┐
        # storage[0] commit_lens               │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[1] state_commit_lens         │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[2] boundary_commit_lens      │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[3] final_state_indices       │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[4] boundary_state_indices    │           │           │           │           │
        #                                      ├───────────┼───────────┼───────────┼───────────┤
        # storage[5] boundary_recovery_lens    │           │           │           │           │
        #                                      └───────────┴───────────┴───────────┴───────────┘
        storage = torch.empty((6, max_num_reqs), dtype=torch.int32, device=device)
        return cls(
            commit_lens=storage[0],
            state_commit_lens=storage[1],
            boundary_commit_lens=storage[2],
            final_state_indices=storage[3],
            boundary_state_indices=storage[4],
            boundary_recovery_lens=storage[5],
        )

    @property
    def capacity(self) -> int:
        return self.commit_lens.shape[0]


def _validate_layer_caches(
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...],
    spec_query_len: int,
) -> int:
    reference = layer_caches[0]
    if reference.checkpoint.ndim != 4:
        raise ValueError("GDN RecoverSSM checkpoint must have four dimensions")
    blocks, num_value_heads, value_dim, key_dim = reference.checkpoint.shape
    for cache in layer_caches:
        validate_gdn_state_replay(cache.checkpoint, cache.replay, num_value_heads, value_dim, key_dim, spec_query_len)
        if cache.num_key_heads != reference.num_key_heads:
            raise ValueError("GDN RecoverSSM layers must share key head geometry")
    num_key_heads = reference.num_key_heads
    if num_key_heads < 1 or num_value_heads % num_key_heads:
        raise ValueError("GDN RecoverSSM requires integral GQA")

    channels = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    if (
        reference.conv.ndim != 3
        or reference.conv.shape[:2] != (blocks, channels)
        or reference.conv.device != reference.checkpoint.device
    ):
        raise ValueError("GDN RecoverSSM convolution page has incompatible layout")
    history_len = reference.conv.shape[2] - spec_query_len + 1
    if history_len < 1:
        raise ValueError("GDN RecoverSSM convolution history is too short")

    for current in layer_caches[1:]:
        if any(
            (tensor.shape, tensor.dtype, tensor.device)
            != (reference_tensor.shape, reference_tensor.dtype, reference_tensor.device)
            for tensor, reference_tensor in zip(current.all_pages(), reference.all_pages())
        ):
            raise ValueError("GDN RecoverSSM layers must share shapes, dtypes and device")
        for name, tensor, reference_tensor in zip(
            ("checkpoint", "replay"),
            current.state_pages(),
            reference.state_pages(),
        ):
            if _inner_stride_signature(tensor) != _inner_stride_signature(reference_tensor):
                raise ValueError(f"GDN RecoverSSM layers must share {name} page inner strides")
    return history_len


@dataclass
class GDNRecoverSSMCommitContext:
    layer_caches: tuple[GDNRecoverSSMLayerCache, ...]
    commit_plan: GDNRecoverSSMCommitPlan
    spec_query_len: int
    # Number of convolution-history tokens retained after committing the
    # accepted prefix: conv_window_len - spec_query_len + 1.
    conv_history_len: int
    # Common conv dim stride used as a compile-time kernel constant. None
    # makes the kernel load each layer's stride from
    # _conv_cache_address_stride_table.
    conv_shared_dim_stride: int | None
    # Contiguous device-side int64 table with shape [4, num_layers]. Column i
    # describes layer_caches[i].conv:
    #   [0, i] = conv.data_ptr(), the base address of layer i's conv cache.
    #   [1, i] = conv.stride(0), the cache-block stride in elements.
    #   [2, i] = conv.stride(1), the channel/dim stride in elements.
    #   [3, i] = conv.stride(2), the token/history stride in elements.
    # The compact-conv kernel indexes each row with its device-side layer id.
    _conv_cache_address_stride_table: torch.Tensor = field(repr=False)
    # Device-side int64 table [2, 2, num_layers]: checkpoint/replay,
    # base address/block stride in elements, layer index.
    _state_commit_cache_address_stride_table: torch.Tensor = field(repr=False)
    _conv_table_views: tuple[torch.Tensor, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Cache view objects once rather than creating four new Python Tensor
        # wrappers at every post-sampling commit (also used by merged groups).
        self._conv_table_views = tuple(self._conv_cache_address_stride_table.unbind(0))

    @classmethod
    def create(
        cls,
        layers: Sequence[Any],
        *,
        spec_query_len: int,
        max_num_reqs: int,
    ) -> "GDNRecoverSSMCommitContext":
        """Bind named layer caches and allocate reusable commit metadata."""
        if not layers or spec_query_len < 1 or max_num_reqs < 1:
            raise ValueError("GDN RecoverSSM requires layers and positive capacities")
        layer_caches = tuple(GDNRecoverSSMLayerCache.from_layer(layer) for layer in layers)
        history_len = _validate_layer_caches(layer_caches, spec_query_len)
        device = layer_caches[0].checkpoint.device
        if device.type == "npu":
            init_device_properties_triton()
        return cls(
            layer_caches=layer_caches,
            commit_plan=GDNRecoverSSMCommitPlan.allocate(max_num_reqs, device),
            spec_query_len=spec_query_len,
            conv_history_len=history_len,
            conv_shared_dim_stride=_get_shared_conv_dim_stride(layer_caches),
            _conv_cache_address_stride_table=_create_conv_cache_address_stride_table(layer_caches),
            _state_commit_cache_address_stride_table=_create_state_commit_cache_address_stride_table(layer_caches),
        )

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        commit_conv: bool = True,
    ) -> None:
        """Commit each request's accepted prefix into every bound GDN layer."""
        align_mode = block_table is not None
        if block_table is not None:
            if num_computed_tokens is None or mamba_block_size is None:
                raise ValueError("GDN RecoverSSM align metadata is incomplete")
            if mamba_block_size < self.spec_query_len:
                raise ValueError("GDN RecoverSSM align block size must cover one speculative window")
            if block_table.ndim != 2 or block_table.shape[1] < 1:
                raise ValueError("GDN RecoverSSM block table must be nonempty and two-dimensional")
            if block_table.dtype not in (torch.int32, torch.int64):
                raise ValueError("GDN RecoverSSM block table must use integer indices")
        elif num_computed_tokens is not None or mamba_block_size is not None:
            raise ValueError("GDN RecoverSSM align metadata is incomplete")
        batch = state_indices.numel()
        if batch > self.commit_plan.capacity or state_indices.ndim != 1 or query_start_loc.shape != (batch + 1,):
            raise ValueError("GDN RecoverSSM commit request metadata exceeds capacity or has incompatible shape")
        if request_indices is not None and (request_indices.ndim != 1 or request_indices.shape[0] < batch):
            raise ValueError("GDN RecoverSSM sampler mapping must be one-dimensional and cover every request")
        if request_indices is None and num_accepted_tokens.numel() < batch:
            raise ValueError("GDN RecoverSSM acceptance counts are too short")
        reference = self.layer_caches[0]
        device = reference.checkpoint.device
        if any(t.device != device for t in (num_accepted_tokens, state_indices, query_start_loc)):
            raise ValueError("GDN RecoverSSM commit tensors must share the cache device")
        if request_indices is not None and request_indices.device != device:
            raise ValueError("GDN RecoverSSM sampler mapping must be on the cache device")
        if align_mode and num_computed_tokens is not None:
            if block_table.device != device or num_computed_tokens.device != device:
                raise ValueError("GDN RecoverSSM align metadata must be on the cache device")
            if request_indices is None and (block_table.shape[0] < batch or num_computed_tokens.numel() < batch):
                raise ValueError("GDN RecoverSSM align metadata must cover the sampler rows")
        if not batch:
            return

        plan = self.commit_plan
        # 1. Resolve accepted lengths and both destinations once per request.
        prepare_commit_plan(
            num_accepted_tokens,
            state_indices,
            query_start_loc,
            plan.commit_lens,
            plan.state_commit_lens,
            plan.boundary_commit_lens,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.boundary_recovery_lens,
            spec_query_len=self.spec_query_len,
            null_block_id=0,
            request_indices=request_indices,
            block_table=block_table,
            num_computed_tokens=num_computed_tokens,
            mamba_block_size=mamba_block_size,
        )
        # 2. Copy final and optional boundary convolution histories.
        if commit_conv:
            compact_conv_state(
                reference.conv,
                *self._conv_table_views,
                state_indices,
                plan.commit_lens,
                plan.final_state_indices,
                plan.boundary_state_indices,
                plan.boundary_recovery_lens,
                conv_history_len=self.conv_history_len,
                align_mode=align_mode,
                dim_stride=self.conv_shared_dim_stride,
                null_block_id=0,
            )
        # 3. Compose accepted records backwards, then merge the checkpoint.
        gdn_recoverssm_commit(
            reference.checkpoint,
            reference.replay,
            self._state_commit_cache_address_stride_table,
            state_indices,
            plan.commit_lens,
            plan.final_state_indices,
            plan.boundary_state_indices,
            plan.boundary_recovery_lens,
            spec_query_len=self.spec_query_len,
            align_mode=align_mode,
        )
