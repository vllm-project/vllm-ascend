# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU descriptors for the upstream Store hash-boundary protocol.

Attention chunks contain exactly their tokens, never the intervening slab
padding. Recurrent chunks contain a complete checkpoint, not a fraction of a
state tensor. The upstream scheduler owns the checkpoint's token boundary.
"""

import itertools
from collections.abc import Sequence

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.layout import RankLocalStoreLayout


class AscendHashStoreLayout(RankLocalStoreLayout):
    def __init__(self, metadata, block_size: int, hash_block_size: int, *, is_state: bool, head_major: bool):
        super().__init__(metadata, block_size, hash_block_size)
        self.is_state = is_state
        self.head_major = head_major
        self._views: list[torch.Tensor] = []

    def register_kv_caches(self, kv_caches: Sequence[torch.Tensor], num_blocks: int) -> None:
        # The upstream worker already groups kernel blocks under manager dim 0.
        # Retain logical views: raw-storage spans can include other layers.
        views = [view for view in kv_caches if view.numel()]
        if not views:
            raise ValueError("Hash recovery requires every cache group to expose transferable views")
        for view in views:
            if view.shape[0] != num_blocks or any(stride <= 0 for stride in view.stride()):
                raise ValueError("Invalid block-major Ascend Store view")
            # Permit inter-layer padding, but never alias two logical entries.
            span = 1
            for stride, size in sorted(zip(view.stride(), view.shape)):
                if size > 1:
                    if stride < span:
                        raise ValueError("Overlapping Ascend Store views are unsupported")
                    span += (size - 1) * stride
            if self.is_state:
                if not view[0].is_contiguous():
                    raise ValueError("Hash recovery requires contiguous checkpoint components")
            else:
                self._attention_parts(view)
        self._views = views
        self._num_blocks = num_blocks

    def _attention_parts(self, view: torch.Tensor) -> tuple[int, int, int]:
        # NHD/HND caches are exposed separately for K/V (and MLA components).
        # group_kernel_blocks optionally inserts a kernel-block dimension.
        if view.ndim not in (4, 5):
            raise ValueError("Hash recovery supports NHD/HND attention views, not packed/NZ layouts")
        scale = view.shape[1] if view.ndim == 5 else 1
        token_axis = (3 if self.head_major else 2) if view.ndim == 5 else (2 if self.head_major else 1)
        kernel_tokens = view.shape[token_axis]
        if scale * kernel_tokens != self.block_size:
            raise ValueError("Attention token axes do not match the Core-resolved manager page")
        suffix_bytes = view.element_size()
        expected_stride = 1
        for axis in range(view.ndim - 1, token_axis, -1):
            if view.stride(axis) != expected_stride:
                raise ValueError("Hash recovery requires dense attention features after the token axis")
            expected_stride *= view.shape[axis]
            suffix_bytes *= view.shape[axis]
        if view.stride(token_axis) < expected_stride:
            raise ValueError("Overlapping attention token strides are unsupported")
        return token_axis, kernel_tokens, suffix_bytes

    def _segments(self, view: torch.Tensor, block_id: int, start: int, end: int):
        elem = view.element_size()
        base = view.data_ptr() + block_id * view.stride(0) * elem
        if self.is_state:
            yield base, view[0].numel() * elem
            return
        token_axis, kernel_tokens, suffix_bytes = self._attention_parts(view)
        # Prefix axes are kernel-block and/or head. Enumerate only these axes;
        # contiguous token ranges remain one descriptor where possible.
        axes = range(1, token_axis)
        for indices in itertools.product(*(range(view.shape[axis]) for axis in axes)):
            kernel = indices[0] if view.ndim == 5 else 0
            lo = max(start - kernel * kernel_tokens, 0)
            hi = min(end - kernel * kernel_tokens, kernel_tokens)
            if lo >= hi:
                continue
            offset = sum(index * view.stride(axis) for axis, index in zip(axes, indices)) * elem
            token_stride = view.stride(token_axis) * elem
            if token_stride == suffix_bytes:
                yield base + offset + lo * token_stride, (hi - lo) * suffix_bytes
            else:
                for token in range(lo, hi):
                    yield base + offset + token * token_stride, suffix_bytes

    def prepare_values(self, chunks, block_ids, shard_ids):
        if len(chunks) != len(shard_ids) or any(shard != 0 for shard in shard_ids):
            raise ValueError("Ascend hash Store uses rank-local objects")
        addresses, sizes, used_blocks = [], [], []
        for start, end in chunks:
            if start < 0 or end <= start or start % self.hash_block_size or end % self.hash_block_size:
                raise ValueError("Store transfer must cover positive, hash-aligned token ranges")
            block_index, offset = divmod(start, self.block_size)
            if (end - 1) // self.block_size != block_index:
                raise ValueError("A hash Store object cannot cross a manager page")
            block_id = block_ids[block_index]
            if block_id <= 0 or block_id >= self._num_blocks:
                raise ValueError("Cannot restore or publish a null checkpoint block")
            segments = [
                segment
                for view in self._views
                for segment in self._segments(view, block_id, offset, end - start + offset)
            ]
            if not segments:
                raise ValueError("Hash Store object has no payload")
            addresses.append([address for address, _ in segments])
            sizes.append([size for _, size in segments])
            used_blocks.append(block_id)
        return addresses, sizes, used_blocks
