# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ownership and residency contracts for uncompressed paged DSA Draft reads.

DSA's native window may leave null blocks in the canonical table's prefix.
Only the native-window/Draft-window intersection is read. Its residency is
inherited from the unchanged native scheduler contract; table ids and valid
column counts alone cannot establish that fact. Coordinates supplied
here are logical raw tokens and kernel-addressed block-table columns.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DSAResidentRead:
    """Native-readable interval; the caller keeps the original Window anchor."""

    first_token: int
    end_token: int
    first_column: int
    end_column: int


def validate_dsa_read_subset(
    *,
    window_starts: Sequence[int],
    exact_ends: Sequence[int],
    query_lengths: Sequence[int],
    kernel_block_span: int,
    native_window: int,
    causal: bool,
) -> tuple[DSAResidentRead, ...]:
    """Prove native reads are a subset of the baseline native read interval.

    This is not a residency scan. The registered native SWA scheduler contract
    supplies baseline residency; preserving absolute query/write coordinates
    and selecting its intersection with [S, E) cannot introduce a stale-prefix
    read. Manager eviction can leave old positive ids in worker table prefixes,
    so neither those ids nor valid-column counts certify cache residency.
    """
    if len(window_starts) != len(exact_ends) or len(query_lengths) != len(exact_ends):
        raise ValueError("DSA subset inputs do not cover the same requests")
    if kernel_block_span <= 0 or native_window <= 0:
        raise ValueError("Invalid DSA native geometry")
    reads = []
    for start, end, query in zip(window_starts, exact_ends, query_lengths):
        if not (0 <= query <= end and 0 <= start <= end - query) or start % kernel_block_span:
            raise ValueError("DSA Window must preserve the complete query at an aligned start")
        baseline_first = max(0, end - query - native_window + int(causal))
        # Native masks/physical-index generation consume E-S and the table
        # shifted by S/B. Mapping their local positions back adds exactly S.
        first = max(start, baseline_first) if query else end
        end_column = (end + kernel_block_span - 1) // kernel_block_span
        first_column = first // kernel_block_span if query else end_column
        reads.append(DSAResidentRead(first, end, first_column, end_column))
    return tuple(reads)


class DSABuilderOwners:
    """Construct complete backend owners per bounded lane, step and consumer.

    The proposal pool supplies lane ids and proves completion before reuse.
    Actual constructors allocate all persistent derived state, including future
    backend additions; no hand-maintained mutable-tensor clone list is needed.
    Configs, registered immutable Hadamard weights and canonical KV remain the
    backend's normal shared references. The returned builder must be retained
    with the proposal before its build method is invoked.
    """

    def __init__(self) -> None:
        self._builders: dict[tuple[int, int, int], Any] = {}

    def select(self, original: Any, *, layer_names: Sequence[str], lane: int, step: int) -> Any:
        if lane < 0 or step < 0:
            raise ValueError("DSA owner lane and step must be nonnegative")
        if original.compressor_ratio > 1:
            raise NotImplementedError("Compressed DSA requires a distinct Window read provider")
        if original._device_metadata_enabled:
            raise NotImplementedError("Deferred DSA producers need a proposal-owned executor completion contract")
        key = (lane, step, id(original))
        existing = self._builders.get(key)
        if existing is not None:
            return existing
        # Match AttentionGroup.create_metadata_builders with the *already
        # translated* builder spec. Do not infer geometry from model dtype.
        owned = type(original)(
            original.kv_cache_spec,
            list(layer_names),
            original.vllm_config,
            original.device,
            metadata_cls=original.metadata_cls,
        )
        if owned._device_metadata_enabled or owned.compressor_ratio != original.compressor_ratio:
            raise RuntimeError("Fresh DSA builder does not preserve the eager consumer contract")
        self._builders[key] = owned
        return owned


def freeze_dsa_rotary_reads(metadata: Any, layer_names: Sequence[str]) -> tuple[Any, ...]:
    """Snapshot only globally reusable RoPE proxies, retaining metadata identity.

    DSA builders own SAS/local/slot/index buffers through DSABuilderOwners;
    RoPE proxies instead point into module-global caches. Their existing
    copy_to_buffers API copies a typed cos/sin pair into independent storage.
    """
    seen: set[int] = set()
    retained: list[Any] = []
    for layer in layer_names:
        item = metadata.get(layer)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        req = item.req_metadata
        for target, cos_name, sin_name in (
            (req, "cos", "sin"),
            (getattr(req, "cp_metadata", None), "local_cos", "local_sin"),
        ):
            if target is None:
                continue
            cos = getattr(target, cos_name, None)
            sin = getattr(target, sin_name, None)
            if cos is None and sin is None:
                continue
            if cos is None or sin is None or not callable(getattr(cos, "copy_to_buffers", None)):
                raise TypeError("DSA rotary read must be a complete RopeDataProxy pair")
            # The proxy has no public length method. Its typed map is the
            # same map consumed by copy_to_buffers; only leading extents are
            # inspected, never device contents or pointer-derived geometry.
            capacity = max((pair[0].shape[0] for groups in cos._data.values() for pair in groups.values()), default=0)
            buffers: dict = {}
            owned_cos, owned_sin = cos.copy_to_buffers(buffers, capacity)
            setattr(target, cos_name, owned_cos)
            setattr(target, sin_name, owned_sin)
            retained.extend((owned_cos, owned_sin, buffers))
    return tuple(retained)
