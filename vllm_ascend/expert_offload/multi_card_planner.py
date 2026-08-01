#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Multi-card expert placement planner for MoE offload.

This module turns "which experts does this layer need, and how heavily" into a
deterministic expert -> rank -> slot placement plus the log2phy map that the MC2
dispatcher consumes.

Design contract
---------------
* **Pure algorithm, no NPU, no comms.** The caller gathers global expert counts
  (all_reduce across the EP group) and hands them in; this module only maps
  counts -> placement. That keeps it unit-testable on CPU.
* **Deterministic.** Every rank feeds the same ``global_counts`` and gets the
  exact same placement, so no broadcast of the placement is needed. Ties are
  broken by ``(load, rank_id)`` / ``(count, expert_id)`` — never by iteration
  order of a set.
* **Load-aware greedy bin-packing.** Experts are sorted by global load (desc)
  and placed one by one onto the rank with the lowest cumulative load that still
  has a free device slot. This is the "consider LB at load time" decision: the
  LB signal is the expert count itself.
* **Capacity.** Each rank holds at most ``num_device_experts`` slots. If a layer
  activates more experts than ``ep_size * num_device_experts``, the overflow is
  returned as ``unassigned`` — the caller must size ``num_device_experts`` so
  this is empty in normal operation (MVP invariant), otherwise token routing
  would target an expert present on no rank.

The per-layer call sequence (wired up in stage 2) is::

    local_counts  = bincount(topk_ids)              # per-rank
    global_counts = all_reduce(local_counts, EP)    #通信模式相关, uniform 可优化跳过
    placement     = plan_placement(global_counts, ep_size, num_device_experts)
    for e in placement.per_rank_experts[ep_rank]:   # 本 rank 只 H2D 自己分到的
        manager.load_expert_into_slot(layer, e, slot_of(e))
    layer.log2phy.copy_(placement.log2phy)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Placement:
    """Result of planning one layer's expert placement.

    Attributes:
        log2phy: ``[global_num_experts]`` int32. ``log2phy[e] =
            rank * num_device_experts + slot`` if expert *e* is resident on a
            device, else -1. This is the exact tensor the MC2 dispatcher reads
            (``moe_comm_method.py`` applies it to ``topk_ids`` before dispatch).
        per_rank_experts: list of length ``ep_size``; entry *r* is the ordered
            list of logical expert ids assigned to rank *r*. Rank *r* H2D-loads
            exactly these (the ones already resident are a cache hit).
        per_rank_load: list of length ``ep_size``; cumulative token load placed
            on each rank, for diagnostics / imbalance reporting.
        unassigned: logical expert ids that were active but did not fit (capacity
            exceeded). Must be empty in normal MVP operation.
    """
    log2phy: torch.Tensor
    per_rank_experts: list[list[int]] = field(default_factory=list)
    per_rank_load: list[int] = field(default_factory=list)
    unassigned: list[int] = field(default_factory=list)


def _sort_active_experts(global_counts):
    """Active experts as (load, id) sorted by (load desc, id asc) — deterministic
    regardless of the order counts were produced in."""
    active = [(int(global_counts[e]), e)
              for e in range(int(global_counts.shape[0]))
              if int(global_counts[e]) > 0]
    active.sort(key=lambda x: (-x[0], x[1]))
    return active


def _plan_single_rank(global_counts, num_device_experts):
    """ep_size==1 fallback: fill slots [0, N) with the hottest experts in load
    order; log2phy value == slot index (rank offset 0)."""
    global_num_experts = int(global_counts.shape[0])
    log2phy = torch.full((global_num_experts,), -1, dtype=torch.int32)
    per_rank_experts: list[list[int]] = [[]]
    unassigned: list[int] = []
    rank_load = [0]
    for count, e in _sort_active_experts(global_counts):
        if len(per_rank_experts[0]) >= num_device_experts:
            unassigned.append(e)
            continue
        slot = len(per_rank_experts[0])
        log2phy[e] = slot  # rank 0 => physical id == slot
        per_rank_experts[0].append(e)
        rank_load[0] += count
    return Placement(log2phy=log2phy, per_rank_experts=per_rank_experts,
                     per_rank_load=rank_load, unassigned=unassigned)


def _assign_experts_to_ranks(active, ep_size, num_device_experts,
                             prev_rank_of=None, force_shard=None):
    """Assign each expert to a rank. Residence-aware: an expert that was on a
    rank last step (``prev_rank_of``) stays there — no cross-rank move means no
    redundant re-H2D on the new rank (the expert is already physically resident
    on its home rank). Genuinely-new experts (not in prev_rank_of, or whose home
    rank is full) go to the least-loaded rank (LB). Deterministic, so every rank
    computes the same assignment. Returns (rank_of, rank_load, unassigned).

    ``force_shard`` (shard-per-rank): pin each expert to its EP-owning rank
    (``e // force_shard``) — no cross-rank placement, since each rank's CPU
    buffer only holds its own shard. Overflow (owner rank full) → unassigned."""
    prev_rank_of = prev_rank_of or {}
    rank_of = {}
    rank_load = [0] * ep_size
    rank_count = [0] * ep_size
    unassigned: list[int] = []
    for count, e in active:
        if force_shard:
            r = e // force_shard
            if rank_count[r] >= num_device_experts:
                unassigned.append(e)
                continue
        else:
            candidates = [r for r in range(ep_size)
                          if rank_count[r] < num_device_experts]
            if not candidates:
                unassigned.append(e)
                continue
            home = prev_rank_of.get(e)
            r = home if home in candidates else min(
                candidates, key=lambda r: (rank_load[r], r))
        rank_of[e] = r
        rank_load[r] += count
        rank_count[r] += 1
    return rank_of, rank_load, unassigned


def plan_placement(
    global_counts: torch.Tensor,
    ep_size: int,
    num_device_experts: int,
    prev_log2phy: torch.Tensor | None = None,
    hotness=None,
    force_shard: int | None = None,
) -> Placement:
    """Deterministically place one layer's active experts onto ranks/slots.

    Args:
        global_counts: ``[global_num_experts]`` integer tensor — the all-reduced
            number of (token, topk-slot) routes hitting each expert this layer.
        ep_size: number of EP ranks.
        num_device_experts: device slots per rank (each slot holds one expert).
        prev_log2phy: previous step's log2phy (CPU int32 tensor). If given,
            experts that stay on the same rank keep their previous slot
            (stable slot → cache hit). New experts fill freed slots.
        hotness: optional ``[global_num_experts]`` hotness scores (list/np/tensor).
            New experts (not stable-kept) are ordered by hotness desc when filling
            free slots, so hotter experts get priority slots.
        force_shard: shard-per-rank shard size. If set, each expert is pinned to
            its EP-owning rank (e // force_shard) — no cross-rank placement.

    Returns:
        Placement (see class docstring). per_rank_experts entries may contain -1
        for freed (empty) slots when stable slots leave gaps.
    """
    assert global_counts.ndim == 1, "global_counts must be 1-D"
    assert ep_size >= 1, "ep_size must be >= 1"
    assert num_device_experts >= 1, "num_device_experts must be >= 1"

    if ep_size == 1:
        return _plan_single_rank(global_counts, num_device_experts)

    global_num_experts = int(global_counts.shape[0])
    log2phy = torch.full((global_num_experts,), -1, dtype=torch.int32)
    active = _sort_active_experts(global_counts)
    # Residence map from last step's placement: expert -> the rank it was on
    # (physical id // per_rank_slots). Feeding this to _assign_experts_to_ranks
    # keeps experts on their home rank (no cross-rank move -> no redundant
    # re-H2D); only genuinely-new experts are LB-distributed.
    prev_rank_of = {}
    if prev_log2phy is not None:
        for expert, pid in enumerate(prev_log2phy.tolist()):
            pid = int(pid)
            if pid >= 0:
                prev_rank_of[expert] = pid // num_device_experts
    rank_of, rank_load, unassigned = _assign_experts_to_ranks(
        active, ep_size, num_device_experts, prev_rank_of,
        force_shard=force_shard)

    # Slot assignment (stable if prev_log2phy given): experts staying on the
    # same rank keep their prev slot (cache hit); new experts fill freed slots
    # ordered by hotness desc (hotter -> priority).
    per_rank_experts = []
    for r in range(ep_size):
        active_r = [e for _c, e in active if rank_of.get(e) == r]
        slots = _assign_slots_stable(active_r, r, num_device_experts,
                                     prev_log2phy, hotness)
        for slot, eid in enumerate(slots):
            if eid >= 0:
                log2phy[eid] = r * num_device_experts + slot
        per_rank_experts.append(slots)

    return Placement(log2phy=log2phy, per_rank_experts=per_rank_experts,
                     per_rank_load=rank_load, unassigned=unassigned)



def _assign_slots_stable(active_r, rank, num_device_experts,
                         prev_log2phy=None, hotness=None):
    """Assign slot indices on ``rank``, RETAINING the previous resident set
    (LRU-style) so recurring experts stay cached across steps — not just the
    current topk.

    1. Carry over experts that were resident on this rank last step (from
       ``prev_log2phy``) into their previous slots. This is the persistent
       hot set: an expert used a few steps ago stays in its slot instead of
       being re-H2D'd when it reappears.
    2. Place ``active_r``: already-resident -> keep slot (cache hit); new ->
       a free slot; if full, evict the coldest NON-active resident (hotness).

    The full result (retained + active) flows into ``log2phy`` so the next
    step's ``prev_log2phy`` carries retained experts forward — retention is
    self-sustaining. Returns a list of length up to ``num_device_experts``
    (expert id or -1).
    """
    result = [-1] * num_device_experts
    # 1. Retain previous residents on this rank (persistent hot set).
    if prev_log2phy is not None:
        prev_list = (prev_log2phy.tolist()
                     if hasattr(prev_log2phy, "tolist") else prev_log2phy)
        for eid, pid in enumerate(prev_list):
            pid = int(pid)
            if pid >= 0 and pid // num_device_experts == rank:
                slot = pid % num_device_experts
                if 0 <= slot < num_device_experts and result[slot] == -1:
                    result[slot] = int(eid)
    resident = {eid: s for s, eid in enumerate(result) if eid >= 0}
    active_set = {int(e) for e in active_r}
    # 2. Place active experts not already resident (resident ones are hits).
    new_active = [int(e) for e in active_r if int(e) not in resident]
    if hotness is not None:
        new_active.sort(key=lambda e: -float(hotness[e]))
    for eid in new_active:
        free = next((s for s in range(num_device_experts)
                     if result[s] == -1), None)
        if free is not None:
            result[free] = eid
            continue
        # No free slot: evict the coldest NON-active resident.
        evictees = [s for s in range(num_device_experts)
                    if result[s] >= 0 and result[s] not in active_set]
        if not evictees:
            break  # every slot holds an active expert this step; can't place
        if hotness is not None:
            slot = min(evictees, key=lambda s: float(hotness[int(result[s])]))
        else:
            slot = evictees[0]
        result[slot] = eid
    while result and result[-1] == -1:
        result.pop()
    return result


def local_expert_counts(
    topk_ids: torch.Tensor,
    global_num_experts: int,
) -> torch.Tensor:
    """Per-rank expert route counts for one layer.

    Args:
        topk_ids: ``[num_tokens, topk]`` logical expert ids selected by the
            router for this rank's tokens.
        global_num_experts: size of the expert vocabulary.

    Returns:
        ``[global_num_experts]`` int64 tensor — how many (token, slot) routes
        hit each expert on this rank. Feed this into an all_reduce to get the
        global counts the planner needs.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(global_num_experts, dtype=torch.int64,
                         device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    return counts


def gather_global_counts(
    local_counts: torch.Tensor,
    ep_group=None,
) -> torch.Tensor:
    """All-reduce per-rank expert counts into global counts.

    On CPU (no process group) this returns ``local_counts`` unchanged — used by
    unit tests. In the real runtime ``ep_group`` is the EP HCCL device group
    (``get_ep_group().device_group``); uniform EP mode (each rank sees all
    tokens) could short-circuit this, but we keep one code path that is correct
    for MC2 / flashcommV1 / DP too (decision 8).

    IMPORTANT: MC2's prepare() SPLITS tokens across EP=TP ranks, so each rank's
    ``topk_ids`` differ -> local_counts differ -> we MUST all_reduce, otherwise
    each rank plans a different placement and the MC2 dispatch all-to-all
    deadlocks. We use ``async_op=True`` + ``work.wait()`` so the HCCL op is
    guaranteed complete before any subsequent D2H ``.cpu()`` (avoids the earlier
    ACL stream-sync error 507014 that came from racing all_reduce against .cpu()).
    """
    if ep_group is None:
        return local_counts
    import torch.distributed as dist
    if not dist.is_initialized():
        return local_counts
    global_counts = local_counts.clone()
    work = dist.all_reduce(global_counts, op=dist.ReduceOp.SUM,
                           group=ep_group, async_op=True)
    work.wait()
    return global_counts


def local_expert_counts_cpu(
    topk_ids: torch.Tensor,
    global_num_experts: int,
) -> torch.Tensor:
    """CPU-side per-rank expert route counts (bincount) for one layer.

    Counterpart of :func:`local_expert_counts` but runs entirely on the CPU:
    ``topk_ids`` is a (pinned) CPU tensor (e.g. the ``topk_ids_h`` buffer the
    single-card graph path D2H-copies into). Used by the multi-card graph-mode
    host callback so the expert counts never touch the NPU stream (the HCCL
    all_reduce can't be a captured graph op, but a gloo CPU all_reduce can be
    issued from a ``_launch_host_func`` callback).
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.bincount(flat, minlength=global_num_experts)
    return counts.to(torch.int64)


def gather_global_counts_cpu(
    local_counts: torch.Tensor,
    cpu_group=None,
) -> torch.Tensor:
    """All-reduce CPU expert counts across the EP group via gloo.

    ``cpu_group`` is ``get_ep_group().cpu_group`` (a gloo ``ProcessGroup`` that
    vLLM creates alongside every device group). gloo is socket/host-based and
    **stream-independent**, so this is safe to call from inside a
    ``_launch_host_func`` host callback during cudagraph replay — unlike the
    HCCL ``all_reduce`` in :func:`gather_global_counts`, which cannot be a
    captured graph op. ``cpu_group=None`` returns ``local_counts`` unchanged
    (unit-test path).
    """
    if cpu_group is None:
        return local_counts
    import torch.distributed as dist
    if not dist.is_initialized():
        return local_counts
    global_counts = local_counts.clone()
    dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=cpu_group)
    return global_counts


def plan_for_layer(
    topk_ids: torch.Tensor,
    global_num_experts: int,
    ep_size: int,
    num_device_experts: int,
    ep_rank: int,
    ep_group=None,
) -> Placement:
    """End-to-end per-layer planning: counts -> all_reduce -> placement.

    This is the convenience entry point called from ``apply()`` once stage 2
    wires the planner in. Rank *r* then H2D-loads ``placement.per_rank_experts[r]``
    and writes ``placement.log2phy`` onto the layer.
    """
    local_counts = local_expert_counts(topk_ids, global_num_experts)
    global_counts = gather_global_counts(local_counts, ep_group)
    # plan_placement iterates expert-by-expert with int() indexing, so pull the
    # all-reduced counts back to host once (avoids per-element NPU->CPU sync).
    return plan_placement(global_counts.cpu(), ep_size, num_device_experts)


# --------------------------------------------------------------------------- #
#  Multi-card PREFILL (EP shard + All2All) — pure logic, unit-testable        #
# --------------------------------------------------------------------------- #
# MC2 dispatch caps at 512 tokens, so prefill (large batch) must use All2All
# with a per-rank contiguous EP shard. These functions compute the pure-logic
# shard state consumed by the All2All dispatcher / GMM, with NO NPU dependency,
# so they can be unit-tested on CPU.

MC2_DISPATCH_TOKEN_HARD_LIMIT = 512


def comm_method_for_multi_card(num_tokens: int, mc2_tokens_capacity: int,
                               mc2_hard_limit: int = MC2_DISPATCH_TOKEN_HARD_LIMIT) -> str:
    """Decide MC2 (decode) vs ALLTOALL (prefill) for multi-card offload.

    Decode (small batch, fits MC2 capacity AND the kernel's 512-token hard
    limit) -> MC2 with dynamic placement. Prefill (large batch) -> ALLTOALL
    with EP shard. Mirrors select_moe_comm_method's multi-card branch.

    Returns "MC2" or "ALLTOALL".
    """
    if num_tokens <= mc2_tokens_capacity and num_tokens <= mc2_hard_limit:
        return "MC2"
    return "ALLTOALL"


def shard_size(num_total_experts: int, ep_size: int) -> int:
    """Experts per rank in a standard contiguous EP shard."""
    assert num_total_experts % ep_size == 0, (
        f"num_total_experts ({num_total_experts}) must be divisible by ep_size "
        f"({ep_size}) for a contiguous shard")
    return num_total_experts // ep_size


def shard_expert_map(ep_rank: int, ep_size: int,
                     num_total_experts: int) -> torch.Tensor:
    """Standard EP shard expert_map for rank ``ep_rank``.

    ``expert_map[e] = e - ep_rank*shard`` if e is in this rank's contiguous
    shard ``[base, base+shard)``, else -1. Length = num_total_experts. The
    AllGather dispatcher masks topk_ids via ``expert_map != -1``; All2All does
    not read expert_map (it uses local_expert_indices) but this is kept for the
    AllGather fallback / completeness.
    """
    sh = shard_size(num_total_experts, ep_size)
    base = ep_rank * sh
    emap = torch.full((num_total_experts,), -1, dtype=torch.int32)
    for i in range(sh):
        emap[base + i] = i
    return emap


def all2all_local_expert_indices(ep_rank: int, shard_sz: int) -> list:
    """TokenDispatcherWithAll2AllV.local_expert_indices for a rank's shard.

    Contiguous global ids ``[ep_rank*shard, ep_rank*shard + shard)``. The
    All2All dispatcher asserts these are contiguous.
    """
    base = ep_rank * shard_sz
    return [base + i for i in range(shard_sz)]


def all2all_expert_ids_per_ep_rank(shard_sz: int,
                                   num_total_experts: int) -> list:
    """TokenDispatcherWithAll2AllV.expert_ids_per_ep_rank values for a shard.

    ``[i % shard for i in range(num_total_experts)]`` — maps each global expert
    id to its local slot within whichever rank's shard holds it. Length must
    equal num_total_experts and satisfy ``nel * ep_size == num_total_experts``
    (the dispatcher reshapes counts as (ep_size, nel)).
    """
    return [i % shard_sz for i in range(num_total_experts)]


def prefill_is_all2all(comm_type) -> bool:
    """Routing predicate: prefill regime = comm is NOT MC2 (i.e. All2All here).

    Single source of truth shared by update_weights_multi_card and apply() so
    decode (MC2) and prefill (All2All) never disagree. ``comm_type`` is the
    MoECommType enum value from _EXTRA_CTX.moe_comm_type.
    """
    return getattr(comm_type, "name", str(comm_type)) != "MC2"

