# SPDX-License-Identifier: Apache-2.0
"""Byte plans for one accepted state component across TP layouts.

Following vLLM StoreLayout and SGLang's Mamba sub-projection slicing, model
adapters describe each component; this module only maps logical intersections
onto physical strides. No Store access, worker coordination or state update.
Callers must supply the same layer, accepted boundary and generation, validate
allocation bounds, and keep both allocations alive through transfer completion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .layout import TensorLayout

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1, "uint8": 1}


@dataclass(frozen=True)
class ComponentLayout:
    """One rank's registered tensor and its logical TP partition.

    projection_sizes are GLOBAL dimensions along shard_axis: e.g. (Q, K, V)
    for convolution or (num_heads,) for recurrent state. Each projection is
    independently partitioned. Other axes retain their semantic order; moving
    shard_axis permits SD/DS conv conversion, not arbitrary axis permutations.
    Replicated components retain every projection on every rank.
    """

    tensor: TensorLayout
    dtype: str
    projection_sizes: tuple[int, ...]
    shard_axis: int
    tp_size: int
    tp_rank: int
    replicated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "projection_sizes", tuple(self.projection_sizes))
        if type(self.tp_size) is not int or self.tp_size <= 0:
            raise ValueError("TP size must be a positive integer")
        if type(self.tp_rank) is not int or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("TP rank is outside the group")
        if type(self.shard_axis) is not int or not 0 <= self.shard_axis < len(self.tensor.shape):
            raise ValueError("shard axis is outside the tensor")
        if type(self.replicated) is not bool:
            raise ValueError("replicated must be a boolean")
        if not self.projection_sizes or any(type(n) is not int or n <= 0 for n in self.projection_sizes):
            raise ValueError("projection sizes must be positive integers")
        divisor = 1 if self.replicated else self.tp_size
        if any(n % divisor for n in self.projection_sizes):
            raise ValueError("each projection must divide evenly across TP")
        if sum(self.projection_sizes) // divisor != self.tensor.shape[self.shard_axis]:
            raise ValueError("local tensor does not match the declared projection partition")
        if _DTYPE_BYTES.get(self.dtype) != self.tensor.itemsize:
            raise ValueError("unsupported dtype or mismatched element size")

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        projection_sizes: tuple[int, ...],
        shard_axis: int,
        tp_size: int,
        tp_rank: int,
        replicated: bool = False,
    ) -> ComponentLayout:
        """Describe an existing state view without moving or materializing it.

        Projection semantics come from the model adapter, never guessed from
        shape. Copy offsets are relative to tensor.data_ptr(), which already
        includes storage_offset; consumers must not add that offset twice.
        The caller retains the tensor and owns its checkpoint/lifetime proof.
        """
        layout = TensorLayout(tuple(tensor.shape), tuple(tensor.stride()), tensor.element_size())
        return cls(
            layout,
            str(tensor.dtype).removeprefix("torch."),
            projection_sizes,
            shard_axis,
            tp_size,
            tp_rank,
            replicated,
        )

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return (self.tensor.shape[self.shard_axis],) + tuple(
            n for axis, n in enumerate(self.tensor.shape) if axis != self.shard_axis
        )

    @property
    def logical_strides(self) -> tuple[int, ...]:
        return (self.tensor.strides[self.shard_axis],) + tuple(
            stride for axis, stride in enumerate(self.tensor.strides) if axis != self.shard_axis
        )


@dataclass(frozen=True)
class ByteCopy:
    """Offsets relative to tensor data pointers, not allocation base pointers."""

    source_offset: int
    destination_offset: int
    num_bytes: int


def _append(copies: list[ByteCopy], source: int, destination: int, size: int) -> None:
    if copies:
        previous = copies[-1]
        if (
            previous.source_offset + previous.num_bytes == source
            and previous.destination_offset + previous.num_bytes == destination
        ):
            copies[-1] = ByteCopy(
                previous.source_offset,
                previous.destination_offset,
                previous.num_bytes + size,
            )
            return
    copies.append(ByteCopy(source, destination, size))


def _copy_region(
    copies: list[ByteCopy],
    shape: tuple[int, ...],
    source_strides: tuple[int, ...],
    target_strides: tuple[int, ...],
    source_base: int,
    target_base: int,
    itemsize: int,
) -> None:
    # Traverse source physical order; collapse the jointly contiguous suffix.
    # Contiguous recurrent heads become one segment, not one per element.
    axes = sorted(range(len(shape)), key=lambda axis: source_strides[axis], reverse=True)
    contiguous = 1
    while axes:
        axis = axes[-1]
        if shape[axis] != 1 and (source_strides[axis] != contiguous or target_strides[axis] != contiguous):
            break
        contiguous *= shape[axis]
        axes.pop()
    for index in product(*(range(shape[axis]) for axis in axes)):
        source = source_base + sum(i * source_strides[axis] for axis, i in zip(axes, index))
        target = target_base + sum(i * target_strides[axis] for axis, i in zip(axes, index))
        _append(copies, source * itemsize, target * itemsize, contiguous * itemsize)


def plan_component_transfer(source: ComponentLayout, target: ComponentLayout) -> tuple[ByteCopy, ...]:
    """Map one source rank to one target rank; disjoint shards return ().

    Enumerating all source ranks covers a target exactly once when source
    partitions are complete. Replicated data uses source rank zero as the
    single writer. This is byte-layout compatibility, not an assertion that
    arbitrary TP/PCP/DCP model execution or checkpoint conversion is supported.
    """
    if (
        source.dtype != target.dtype
        or source.projection_sizes != target.projection_sizes
        or source.replicated != target.replicated
        or source.logical_shape[1:] != target.logical_shape[1:]
    ):
        raise ValueError("state component schemas are incompatible")
    if source.replicated and source.tp_rank != 0:
        return ()
    source_divisor = 1 if source.replicated else source.tp_size
    target_divisor = 1 if target.replicated else target.tp_size
    source_rank = 0 if source.replicated else source.tp_rank
    target_rank = 0 if target.replicated else target.tp_rank
    copies: list[ByteCopy] = []
    source_offset = target_offset = 0
    source_strides, target_strides = source.logical_strides, target.logical_strides
    for size in source.projection_sizes:
        source_count, target_count = size // source_divisor, size // target_divisor
        source_start, target_start = (
            source_rank * source_count,
            target_rank * target_count,
        )
        start, end = (
            max(source_start, target_start),
            min(source_start + source_count, target_start + target_count),
        )
        if start < end:
            _copy_region(
                copies,
                (end - start,) + source.logical_shape[1:],
                source_strides,
                target_strides,
                (source_offset + start - source_start) * source_strides[0],
                (target_offset + start - target_start) * target_strides[0],
                source.tensor.itemsize,
            )
        source_offset += source_count
        target_offset += target_count
    return tuple(copies)
