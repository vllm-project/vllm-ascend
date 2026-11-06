# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank verification of the ACL graphs captured for context parallelism.

Piecewise and full graphs are captured independently on every rank, but the
captured pieces contain collectives. If one rank captures a different set of
pieces (different piece count, or a different token extent for the same piece),
the first replay step issues mismatched collectives and hangs - typically long
after the real cause.

This module builds a digest of everything captured on the local rank and
compares it across the CP group right after capture. A mismatch raises with the
per-rank details while the capture stack is still on screen.

See ``docs/source/user_guide/feature_guide/context_parallel.md``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# (mode, num_tokens, has_lora, num_active_loras)
GraphEntry = tuple[str, int, int, int]

_DIGEST_BITS = 63


@dataclass(frozen=True)
class CaptureDigest:
    """Local capture summary that must be identical on every CP rank."""

    entries: tuple[GraphEntry, ...]
    digest: int
    piece_count: int
    token_extents: tuple[int, ...]

    def describe(self) -> str:
        return f"digest=0x{self.digest:016x} pieces={self.piece_count} token_extents={list(self.token_extents)}"


def build_digest(entries: Iterable[GraphEntry]) -> CaptureDigest:
    """Hash the captured graph set in a rank-stable order."""
    ordered = tuple(sorted(entries))
    payload = ";".join(
        f"{mode}:{num_tokens}:{has_lora}:{num_active_loras}" for mode, num_tokens, has_lora, num_active_loras in ordered
    )
    digest = int.from_bytes(
        hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(),
        "big",
    ) & ((1 << _DIGEST_BITS) - 1)
    token_extents = tuple(sorted({num_tokens for _, num_tokens, _, _ in ordered}))
    return CaptureDigest(
        entries=ordered,
        digest=digest,
        piece_count=len(ordered),
        token_extents=token_extents,
    )


def _entry_from_descriptor(mode: Any, descriptor: Any) -> GraphEntry:
    mode_name = getattr(mode, "name", None) or str(mode)
    return (
        str(mode_name),
        int(getattr(descriptor, "num_tokens", -1)),
        int(bool(getattr(descriptor, "has_lora", False))),
        int(getattr(descriptor, "num_active_loras", -1)),
    )


def collect_capture_entries() -> tuple[GraphEntry, ...]:
    """Read every captured graph from the live wrappers on this rank.

    Both wrappers are already registered globally by vLLM, so no extra
    bookkeeping is needed at capture time: a wrapper that has an entry without
    a graph was not captured and is skipped.
    """
    entries: list[GraphEntry] = []

    from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers

    for wrapper in list(_acl_graph_wrappers):
        mode = getattr(wrapper, "runtime_mode", None)
        for descriptor, entry in getattr(wrapper, "concrete_aclgraph_entries", {}).items():
            if getattr(entry, "aclgraph", None) is None:
                continue
            entries.append(_entry_from_descriptor(mode, descriptor))

    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

        for wrapper in list(BreakableCUDAGraphWrapper._all_instances):
            for descriptor, entry in getattr(wrapper, "entries", {}).items():
                if getattr(entry, "capture", None) is None:
                    continue
                entries.append(_entry_from_descriptor(getattr(wrapper, "runtime_mode", None), descriptor))
    except ImportError:  # pragma: no cover - older vLLM without breakable graphs
        pass

    return tuple(entries)


def collect_capture_digest() -> CaptureDigest:
    return build_digest(collect_capture_entries())


def _verification_group() -> Any:
    """The CP group that must agree on the captured set.

    A DCP group spans the widest CPS domain in the stacked topology, and every
    rank in it captures the same pieces; with DCP disabled the PCP group is used.
    """
    from vllm.distributed import get_dcp_group, get_pcp_group

    dcp_group = get_dcp_group()
    if dcp_group.world_size > 1:
        return dcp_group
    pcp_group = get_pcp_group()
    if pcp_group.world_size > 1:
        return pcp_group
    return None


def _digest_values(digest: CaptureDigest) -> list[int]:
    token_sum = sum(digest.token_extents)
    return [digest.digest, digest.piece_count, token_sum]


def verify_cp_capture_consistency(vllm_config: Any, *, phase: str = "capture") -> None:
    """Fail fast when CP ranks captured different graph sets.

    Runs once per capture phase and costs two tiny all-reduces; it is a no-op
    when context parallelism is disabled.
    """
    from vllm.logger import logger

    parallel_config = vllm_config.parallel_config
    if parallel_config.prefill_context_parallel_size <= 1 and parallel_config.decode_context_parallel_size <= 1:
        return

    group = _verification_group()
    if group is None or group.world_size <= 1:
        return

    import torch
    import torch.distributed as dist

    digest = collect_capture_digest()
    values = _digest_values(digest)
    local = torch.tensor(values, dtype=torch.int64, device=torch.npu.current_device())

    minimum = local.clone()
    maximum = local.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group.device_group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group.device_group)
    if bool(torch.equal(minimum, maximum)):
        logger.info("CP %s consistency verified: %s", phase, digest.describe())
        return

    gathered = [torch.zeros_like(local) for _ in range(group.world_size)]
    dist.all_gather(gathered, local, group=group.device_group)
    local_values = values
    mismatched = [
        f"rank {index}: digest=0x{int(values_per_rank[0]):016x} pieces={int(values_per_rank[1])} "
        f"token_extent_sum={int(values_per_rank[2])}"
        for index, values_per_rank in enumerate(gathered)
        if [int(item) for item in values_per_rank] != local_values
    ]
    raise RuntimeError(
        f"Context-parallel {phase} captured different ACL graph sets across ranks; "
        "the first replay would issue mismatched collectives. "
        f"Local rank {group.rank_in_group}: {digest.describe()}. "
        "Differing ranks: " + "; ".join(mismatched)
    )
