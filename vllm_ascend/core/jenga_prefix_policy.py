# SPDX-License-Identifier: Apache-2.0
"""Reference policies for Jenga-style layerwise prefix reuse.

The allocator decides *where* a cache page lives.  This module deliberately
models the orthogonal question of *which* token prefixes remain reusable for
each KV-cache group after some pages have been evicted.  Keeping the model
pure Python makes the dependency rules executable documentation and lets the
runtime compare its decisions with a small, deterministic oracle.

``cached_blocks`` uses zero-based logical block indices.  For attention block
``i`` covers the token interval ``[i * block_size, (i + 1) * block_size)``.
For checkpointed state block ``i`` stores the state after
``(i + 1) * checkpoint_stride`` tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Mapping
from dataclasses import dataclass


def _check_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _check_prefix_limit(max_prefix_tokens: int) -> None:
    if not isinstance(max_prefix_tokens, int) or isinstance(max_prefix_tokens, bool) or max_prefix_tokens < 0:
        raise ValueError("max_prefix_tokens must be a non-negative integer")


def _normalize_cached_blocks(cached_blocks: Collection[int]) -> frozenset[int]:
    normalized = frozenset(cached_blocks)
    if any(not isinstance(block, int) or isinstance(block, bool) or block < 0 for block in normalized):
        raise ValueError("cached block indices must be non-negative integers")
    return normalized


def clip_prefill_to_state_checkpoint(
    start_tokens: int,
    num_new_tokens: int,
    prefill_end_tokens: int,
    checkpoint_interval_tokens: int,
) -> int:
    """Stop a prefill chunk at the next periodic state checkpoint.

    Recurrent kernels materialize the running state at a scheduling-step end;
    a hash mask cannot reconstruct a checkpoint that one long forward pass
    skipped over.  This pure helper makes every crossed periodic boundary an
    actual step end while leaving decode and already-short chunks unchanged.
    """

    _check_prefix_limit(start_tokens)
    _check_prefix_limit(num_new_tokens)
    _check_prefix_limit(prefill_end_tokens)
    _check_positive("checkpoint_interval_tokens", checkpoint_interval_tokens)
    if num_new_tokens == 0 or start_tokens >= prefill_end_tokens:
        return num_new_tokens

    end_tokens = start_tokens + num_new_tokens
    next_checkpoint = (start_tokens // checkpoint_interval_tokens + 1) * checkpoint_interval_tokens
    if start_tokens < next_checkpoint < end_tokens:
        return next_checkpoint - start_tokens
    return num_new_tokens


class PrefixDependencyPolicy(ABC):
    """Dependency rule for one layer type (one KV-cache group).

    Prefix length zero is always legal and touches no blocks.  Non-zero
    lengths are aligned to the cache granularity of the concrete policy.
    """

    @abstractmethod
    def possible_prefix_lengths(
        self,
        cached_blocks: Collection[int],
        max_prefix_tokens: int,
    ) -> frozenset[int]:
        """Return every reusable prefix length up to ``max_prefix_tokens``."""

    @abstractmethod
    def blocks_to_touch(
        self,
        cached_blocks: Collection[int],
        prefix_tokens: int,
    ) -> tuple[int, ...]:
        """Return cached blocks whose recency must be refreshed for a hit.

        Raises:
            ValueError: If ``prefix_tokens`` is not reusable from the supplied
                cache state.
        """


@dataclass(frozen=True, slots=True)
class FullAttentionPrefixPolicy(PrefixDependencyPolicy):
    """A full-attention prefix is reusable only up to its first missing page."""

    block_size_tokens: int

    def __post_init__(self) -> None:
        _check_positive("block_size_tokens", self.block_size_tokens)

    def possible_prefix_lengths(
        self,
        cached_blocks: Collection[int],
        max_prefix_tokens: int,
    ) -> frozenset[int]:
        cached = _normalize_cached_blocks(cached_blocks)
        _check_prefix_limit(max_prefix_tokens)

        possible = {0}
        max_blocks = max_prefix_tokens // self.block_size_tokens
        for block_index in range(max_blocks):
            if block_index not in cached:
                break
            possible.add((block_index + 1) * self.block_size_tokens)
        return frozenset(possible)

    def blocks_to_touch(
        self,
        cached_blocks: Collection[int],
        prefix_tokens: int,
    ) -> tuple[int, ...]:
        cached = _normalize_cached_blocks(cached_blocks)
        _check_prefix_limit(prefix_tokens)
        if prefix_tokens % self.block_size_tokens:
            raise ValueError("full-attention prefix must be block aligned")

        touched = tuple(range(prefix_tokens // self.block_size_tokens))
        if not set(touched).issubset(cached):
            raise ValueError("full-attention prefix crosses a missing block")
        return touched


@dataclass(frozen=True, slots=True)
class SlidingWindowPrefixPolicy(PrefixDependencyPolicy):
    """Prefix dependencies for token-based sliding-window attention.

    To resume after ``p`` tokens, every block intersecting
    ``[max(0, p - window_size_tokens), p)`` must be cached.  Earlier holes do
    not matter.  The same tail blocks, rather than the whole historical
    prefix, are touched on a cache hit.
    """

    block_size_tokens: int
    window_size_tokens: int

    def __post_init__(self) -> None:
        _check_positive("block_size_tokens", self.block_size_tokens)
        _check_positive("window_size_tokens", self.window_size_tokens)

    def _dependency_blocks(self, prefix_tokens: int) -> tuple[int, ...]:
        if prefix_tokens == 0:
            return ()
        first_token = max(0, prefix_tokens - self.window_size_tokens)
        first_block = first_token // self.block_size_tokens
        block_count = prefix_tokens // self.block_size_tokens
        return tuple(range(first_block, block_count))

    def possible_prefix_lengths(
        self,
        cached_blocks: Collection[int],
        max_prefix_tokens: int,
    ) -> frozenset[int]:
        cached = _normalize_cached_blocks(cached_blocks)
        _check_prefix_limit(max_prefix_tokens)

        possible = {0}
        max_blocks = max_prefix_tokens // self.block_size_tokens
        for block_count in range(1, max_blocks + 1):
            prefix_tokens = block_count * self.block_size_tokens
            if set(self._dependency_blocks(prefix_tokens)).issubset(cached):
                possible.add(prefix_tokens)
        return frozenset(possible)

    def blocks_to_touch(
        self,
        cached_blocks: Collection[int],
        prefix_tokens: int,
    ) -> tuple[int, ...]:
        cached = _normalize_cached_blocks(cached_blocks)
        _check_prefix_limit(prefix_tokens)
        if prefix_tokens % self.block_size_tokens:
            raise ValueError("sliding-window prefix must be block aligned")

        touched = self._dependency_blocks(prefix_tokens)
        if not set(touched).issubset(cached):
            raise ValueError("sliding-window prefix has a hole inside its window")
        return touched


@dataclass(frozen=True, slots=True)
class CheckpointStatePrefixPolicy(PrefixDependencyPolicy):
    """Prefix dependencies for periodically checkpointed recurrent state.

    Jenga applies this rule to linear-attention/Mamba state, using periodic
    checkpoints (512 tokens in the paper's evaluated design).  A configurable
    stride is retained here as an engineering abstraction so the same oracle
    can model other recurrent caches such as GDN.  That configurability does
    not imply that the Jenga paper evaluated GDN checkpoint reuse.

    A hit is legal only at a cached checkpoint, and only that most recent
    checkpoint block is touched.  Older state checkpoints are not runtime
    dependencies of the resumed request.
    """

    checkpoint_stride_tokens: int = 512

    def __post_init__(self) -> None:
        _check_positive("checkpoint_stride_tokens", self.checkpoint_stride_tokens)

    def possible_prefix_lengths(
        self,
        cached_blocks: Collection[int],
        max_prefix_tokens: int,
    ) -> frozenset[int]:
        cached = _normalize_cached_blocks(cached_blocks)
        _check_prefix_limit(max_prefix_tokens)

        possible = {0}
        for block_index in cached:
            prefix_tokens = (block_index + 1) * self.checkpoint_stride_tokens
            if prefix_tokens <= max_prefix_tokens:
                possible.add(prefix_tokens)
        return frozenset(possible)

    def blocks_to_touch(
        self,
        cached_blocks: Collection[int],
        prefix_tokens: int,
    ) -> tuple[int, ...]:
        cached = _normalize_cached_blocks(cached_blocks)
        _check_prefix_limit(prefix_tokens)
        if prefix_tokens == 0:
            return ()
        if prefix_tokens % self.checkpoint_stride_tokens:
            raise ValueError("state prefix must be checkpoint aligned")

        checkpoint_block = prefix_tokens // self.checkpoint_stride_tokens - 1
        if checkpoint_block not in cached:
            raise ValueError("state prefix checkpoint is not cached")
        return (checkpoint_block,)


@dataclass(frozen=True, slots=True)
class PrefixGroupState:
    """Policy and cache presence for one KV-cache group."""

    policy: PrefixDependencyPolicy
    cached_blocks: frozenset[int]

    def __init__(
        self,
        policy: PrefixDependencyPolicy,
        cached_blocks: Collection[int],
    ) -> None:
        if not isinstance(policy, PrefixDependencyPolicy):
            raise TypeError("policy must implement PrefixDependencyPolicy")
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "cached_blocks", _normalize_cached_blocks(cached_blocks))


@dataclass(frozen=True, slots=True)
class LayerwisePrefixMatch:
    """Longest common reusable prefix and per-group blocks to touch."""

    prefix_tokens: int
    touched_blocks: Mapping[Hashable, tuple[int, ...]]


def find_longest_common_prefix(
    groups: Mapping[Hashable, PrefixGroupState],
    max_prefix_tokens: int,
) -> LayerwisePrefixMatch:
    """Intersect all groups' legal prefix sets and select the largest value.

    This is the compatibility-layer rule: a request may resume only at a
    token position that is independently valid for every participating cache
    group.  Recency is then updated according to each group's dependencies at
    that shared position.
    """

    _check_prefix_limit(max_prefix_tokens)
    if not groups:
        raise ValueError("at least one prefix-cache group is required")

    common: set[int] | None = None
    for state in groups.values():
        possible = set(state.policy.possible_prefix_lengths(state.cached_blocks, max_prefix_tokens))
        common = possible if common is None else common.intersection(possible)

    # Every conforming policy includes zero, so ``common`` cannot be empty.
    assert common
    prefix_tokens = max(common)
    touched_blocks = {
        group_id: state.policy.blocks_to_touch(state.cached_blocks, prefix_tokens) for group_id, state in groups.items()
    }
    return LayerwisePrefixMatch(prefix_tokens, touched_blocks)
