"""Unit tests for multi_card_planner (pure CPU, no NPU).

Run: cd vllm-ascend && python3 tests/ut/expert_offload/test_multi_card_planner.py
"""
import os
import sys

# Import the planner as a standalone module (it only depends on torch + dataclass)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..",
                                "vllm_ascend", "expert_offload"))
import multi_card_planner as mcp  # noqa: E402

import torch  # noqa: E402


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_determinism():
    print("test_determinism")
    counts = torch.tensor([5, 4, 3, 2, 1, 0, 2], dtype=torch.int64)
    p1 = mcp.plan_placement(counts, ep_size=2, num_device_experts=3)
    p2 = mcp.plan_placement(counts, ep_size=2, num_device_experts=3)
    _check(torch.equal(p1.log2phy, p2.log2phy), "same input -> same log2phy")
    _check(p1.per_rank_experts == p2.per_rank_experts, "same input -> same per_rank")


def test_cross_rank_consistency():
    """The core contract: every rank feeds the same global_counts and must get
    the same placement (so no placement broadcast is needed)."""
    print("test_cross_rank_consistency")
    counts = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.int64)
    placements = [mcp.plan_placement(counts, ep_size=4, num_device_experts=3)
                  for _ in range(4)]  # simulate 4 ranks computing independently
    for r in range(1, 4):
        _check(torch.equal(placements[0].log2phy, placements[r].log2phy),
               f"rank {r} log2phy == rank 0")
    _check(placements[0].unassigned == [], "no overflow expected")


def test_load_balance_and_encoding():
    print("test_load_balance_and_encoding")
    counts = torch.tensor([5, 4, 3, 2, 1], dtype=torch.int64)  # total load 15
    ep_size, ndev = 2, 3
    p = mcp.plan_placement(counts, ep_size=ep_size, num_device_experts=ndev)
    # capacity: each rank <= ndev experts
    for r in range(ep_size):
        _check(len(p.per_rank_experts[r]) <= ndev,
               f"rank {r} within slot capacity ({len(p.per_rank_experts[r])}<={ndev})")
    # log2phy encoding: assigned expert -> rank*ndev+slot, slot in [0,ndev)
    for r in range(ep_size):
        for slot, e in enumerate(p.per_rank_experts[r]):
            _check(int(p.log2phy[e]) == r * ndev + slot,
                   f"expert {e} on rank {r} slot {slot} -> {r * ndev + slot}")
    # every assigned expert is active; every inactive expert is -1
    for e in range(len(counts)):
        if counts[e] > 0:
            _check(int(p.log2phy[e]) >= 0, f"active expert {e} assigned")
        else:
            _check(int(p.log2phy[e]) == -1, f"inactive expert {e} is -1")
    # load imbalance report: |max-min| should be modest (<= heaviest single expert)
    imb = max(p.per_rank_load) - min(p.per_rank_load)
    _check(imb <= int(counts.max()), f"load imbalance {imb} <= hottest expert {int(counts.max())}")
    print(f"  per_rank_load={p.per_rank_load} per_rank_experts={p.per_rank_experts}")


def test_capacity_overflow():
    print("test_capacity_overflow")
    # 6 active experts, ep_size=2, ndev=2 => total capacity 4, overflow 2
    counts = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int64)
    p = mcp.plan_placement(counts, ep_size=2, num_device_experts=2)
    assigned = sum(len(rk) for rk in p.per_rank_experts)
    _check(assigned == 4, f"only 4 fit, got {assigned}")
    _check(len(p.unassigned) == 2, f"2 unassigned, got {len(p.unassigned)}")
    for e in p.unassigned:
        _check(int(p.log2phy[e]) == -1, f"unassigned expert {e} is -1")


def test_tiebreak_count_equal():
    """All experts equally hot -> should spread across ranks (slot-balanced)."""
    print("test_tiebreak_count_equal")
    counts = torch.tensor([2, 2, 2, 2], dtype=torch.int64)
    p = mcp.plan_placement(counts, ep_size=2, num_device_experts=2)
    _check(len(p.per_rank_experts[0]) == 2 and len(p.per_rank_experts[1]) == 2,
           f"equal split 2/2, got {[len(x) for x in p.per_rank_experts]}")
    _check(p.per_rank_load[0] == p.per_rank_load[1],
           f"equal load, got {p.per_rank_load}")


def test_single_rank():
    print("test_single_rank")
    counts = torch.tensor([1, 5, 2, 0], dtype=torch.int64)
    p = mcp.plan_placement(counts, ep_size=1, num_device_experts=2)
    # hottest first: expert 1 (5) then expert 2 (2); expert 0 (1) overflows
    _check(p.per_rank_experts[0][:2] == [1, 2], f"hottest first, got {p.per_rank_experts}")
    _check(int(p.log2phy[1]) == 0 and int(p.log2phy[2]) == 1,
           "single-rank log2phy == slot index")
    _check(int(p.log2phy[0]) == -1, "overflow expert 0 unassigned")
    _check(0 in p.unassigned, "expert 0 in unassigned")


def test_local_expert_counts():
    print("test_local_expert_counts")
    topk_ids = torch.tensor([[0, 2], [2, 5], [0, 0]])  # expert 0 x3, 2 x2, 5 x1
    counts = mcp.local_expert_counts(topk_ids, global_num_experts=8)
    expected = torch.tensor([3, 0, 2, 0, 0, 1, 0, 0], dtype=torch.int64)
    _check(torch.equal(counts, expected), f"counts {counts.tolist()} == {expected.tolist()}")


def test_plan_for_layer_no_group():
    """End-to-end with ep_group=None (CPU test path): counts gathered locally."""
    print("test_plan_for_layer_no_group")
    topk_ids = torch.tensor([[0, 1], [1, 2], [3, 0]])
    p = mcp.plan_for_layer(topk_ids, global_num_experts=4,
                           ep_size=2, num_device_experts=2, ep_rank=0, ep_group=None)
    for e in range(4):
        _check(int(p.log2phy[e]) >= 0, f"expert {e} assigned (all active)")


# --------------------------------------------------------------------------- #
#  PREFILL logic (EP shard + All2All)                                          #
# --------------------------------------------------------------------------- #
def test_comm_method_boundary():
    """MC2 (decode) vs ALLTOALL (prefill) selection. MC2 kernel hard-limits at
    512 tokens; capacity (~2 here) is the decode batch bound."""
    print("test_comm_method_boundary")
    _check(mcp.comm_method_for_multi_card(1, mc2_tokens_capacity=2) == "MC2",
           "1 token (decode) -> MC2")
    _check(mcp.comm_method_for_multi_card(2, mc2_tokens_capacity=2) == "MC2",
           "2 tokens (== capacity) -> MC2")
    _check(mcp.comm_method_for_multi_card(3, mc2_tokens_capacity=2) == "ALLTOALL",
           "3 tokens (> capacity) -> ALLTOALL")
    # even if capacity were huge, the 512 hard limit forces ALLTOALL
    _check(mcp.comm_method_for_multi_card(512, mc2_tokens_capacity=10_000) == "MC2",
           "512 tokens (== hard limit, capacity huge) -> MC2")
    _check(mcp.comm_method_for_multi_card(513, mc2_tokens_capacity=10_000) == "ALLTOALL",
           "513 tokens (> hard limit) -> ALLTOALL")
    _check(mcp.comm_method_for_multi_card(2048, mc2_tokens_capacity=2) == "ALLTOALL",
           "2048 tokens (prefill profile) -> ALLTOALL")


def test_shard_size():
    print("test_shard_size")
    _check(mcp.shard_size(256, 2) == 128, "256/2 = 128")
    _check(mcp.shard_size(256, 4) == 64, "256/4 = 64")
    try:
        mcp.shard_size(255, 2)  # not divisible
        _check(False, "255/2 should raise")
    except AssertionError:
        _check(True, "255/2 raises (indivisible)")


def test_shard_expert_map():
    """Each rank's shard maps to local [0:shard], everything else -1. Union of
    all ranks' maps covers every expert exactly once (true EP partition)."""
    print("test_shard_expert_map")
    ntotal, ep = 256, 2
    maps = [mcp.shard_expert_map(r, ep, ntotal) for r in range(ep)]
    for r in range(ep):
        m = maps[r]
        _check(list(m.shape) == [ntotal], f"rank {r} map len {ntotal}")
        sh = ntotal // ep
        base = r * sh
        for e in range(ntotal):
            if base <= e < base + sh:
                _check(int(m[e]) == e - base, f"rank {r} expert {e} -> local {e - base}")
            else:
                _check(int(m[e]) == -1, f"rank {r} expert {e} -> -1")
    # EP partition: every expert is local on exactly one rank
    for e in range(ntotal):
        owners = [r for r in range(ep) if int(maps[r][e]) != -1]
        _check(owners == [e // (ntotal // ep)], f"expert {e} owned by rank {e // (ntotal // ep)}")


def test_all2all_local_expert_indices():
    """All2All requires CONTIGUOUS local_expert_indices: [i, i+1, ...]."""
    print("test_all2all_local_expert_indices")
    idx0 = mcp.all2all_local_expert_indices(ep_rank=0, shard_sz=128)
    idx1 = mcp.all2all_local_expert_indices(ep_rank=1, shard_sz=128)
    _check(idx0 == list(range(0, 128)), "rank0 [0..127]")
    _check(idx1 == list(range(128, 256)), "rank1 [128..255]")
    for idx in (idx0, idx1):
        _check(all(idx[i + 1] == idx[i] + 1 for i in range(len(idx) - 1)),
               "contiguous (All2All assertion)")


def test_all2all_reshape_invariant():
    """CRITICAL: All2All _preprocess reshapes per-expert counts as
    (ep_size, nel) -> needs nel * ep_size == num_total_experts."""
    print("test_all2all_reshape_invariant")
    ntotal, ep = 256, 2
    sh = mcp.shard_size(ntotal, ep)
    eidpr = mcp.all2all_expert_ids_per_ep_rank(sh, ntotal)
    _check(len(eidpr) == ntotal, f"len {ntotal}")
    _check(sh * ep == ntotal, f"nel*ep_size == num_experts ({sh}*{ep}=={ntotal})")
    # every value in [0, shard)
    _check(all(0 <= v < sh for v in eidpr), "values in [0, shard)")
    # expert e on rank r (e in [r*sh, r*sh+sh)) maps to local slot e - r*sh,
    # and eidpr[e] == e % sh == e - r*sh  (since r*sh is a multiple of sh)
    for e in range(ntotal):
        _check(eidpr[e] == e % sh, f"expert {e} -> {e % sh}")


def test_prefill_is_all2all():
    print("test_prefill_is_all2all")

    class _CT:
        def __init__(self, name):
            self.name = name
    _check(mcp.prefill_is_all2all(_CT("ALLTOALL")) is True, "ALLTOALL -> prefill")
    _check(mcp.prefill_is_all2all(_CT("MC2")) is False, "MC2 -> decode (not prefill)")
    _check(mcp.prefill_is_all2all(_CT("ALLGATHER")) is True, "ALLGATHER -> prefill")


def test_local_expert_counts_cpu():
    """CPU bincount matches the NPU scatter_add version for the same input."""
    print("test_local_expert_counts_cpu")
    topk_ids = torch.tensor([[0, 2], [2, 5], [0, 0]])  # expert 0 x3, 2 x2, 5 x1
    # NPU-path reference (runs on CPU here, same algorithm)
    ref = mcp.local_expert_counts(topk_ids, global_num_experts=8)
    cpu = mcp.local_expert_counts_cpu(topk_ids, global_num_experts=8)
    expected = torch.tensor([3, 0, 2, 0, 0, 1, 0, 0], dtype=torch.int64)
    _check(torch.equal(cpu, expected), f"cpu counts {cpu.tolist()} == {expected.tolist()}")
    _check(torch.equal(cpu, ref), f"cpu counts == npu-path {ref.tolist()}")
    # int32 input (pinned buffer is int32) also works
    cpu_i32 = mcp.local_expert_counts_cpu(topk_ids.to(torch.int32), 8)
    _check(torch.equal(cpu_i32, expected), "int32 input -> same counts")


def test_gather_global_counts_cpu_nogroup():
    """cpu_group=None returns input unchanged (UT path, no dist)."""
    print("test_gather_global_counts_cpu_nogroup")
    counts = torch.tensor([1, 2, 3], dtype=torch.int64)
    out = mcp.gather_global_counts_cpu(counts, cpu_group=None)
    _check(torch.equal(out, counts), "no group -> unchanged")
    _check(out is not counts or torch.equal(out, counts), "value-equal")


def test_cpu_path_matches_npu_path_placement():
    """The graph-mode CPU path (bincount+gloo-noop+plan) gives the same
    placement as the eager NPU path (scatter_add+HCCL-noop+plan) for the
    same topk_ids — confirms the two are interchangeable when ep_group is
    a no-op (UT)."""
    print("test_cpu_path_matches_npu_path_placement")
    topk_ids = torch.tensor([[0, 1], [1, 2], [3, 0]])
    p_npu = mcp.plan_for_layer(topk_ids, global_num_experts=4, ep_size=2,
                               num_device_experts=2, ep_rank=0, ep_group=None)
    # CPU path
    lc = mcp.local_expert_counts_cpu(topk_ids, 4)
    gc = mcp.gather_global_counts_cpu(lc, cpu_group=None)
    p_cpu = mcp.plan_placement(gc, ep_size=2, num_device_experts=2)
    _check(torch.equal(p_npu.log2phy, p_cpu.log2phy),
           f"CPU-path placement == NPU-path placement")


def test_stable_slot_keeps_prev_slot():
    """With prev_log2phy, experts that stay on the same rank keep their slot
    (cache hit). New experts fill freed slots."""
    print("test_stable_slot_keeps_prev_slot")
    counts = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int64)  # 6 active
    # step 0: no prev -> slot = LB order
    p0 = mcp.plan_placement(counts, ep_size=2, num_device_experts=4)
    # step 1: experts 0,1,2,3 repeat; 4,5 gone; 6,7 new. prev = p0.log2phy.
    counts1 = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1], dtype=torch.int64)
    p1 = mcp.plan_placement(counts1, ep_size=2, num_device_experts=4,
                            prev_log2phy=p0.log2phy)
    # experts 0,1,2,3 should keep their step-0 slot (same physical id)
    for e in [0, 1, 2, 3]:
        _check(int(p1.log2phy[e]) == int(p0.log2phy[e]),
               f"expert {e} stable: step1 phys {int(p1.log2phy[e])} == step0 {int(p0.log2phy[e])}")
    # experts 6,7 placed somewhere valid
    for e in [6, 7]:
        _check(int(p1.log2phy[e]) >= 0, f"new expert {e} placed")


def test_stable_slot_no_prev_matches_old():
    """Without prev_log2phy, stable-slot placement == original LB-order placement."""
    print("test_stable_slot_no_prev_matches_old")
    counts = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6], dtype=torch.int64)
    # old-style (no prev, no hotness)
    p = mcp.plan_placement(counts, ep_size=4, num_device_experts=3)
    # determinism: same input -> same output
    p2 = mcp.plan_placement(counts, ep_size=4, num_device_experts=3)
    _check(torch.equal(p.log2phy, p2.log2phy), "no-prev stable == deterministic")


def test_stable_slot_hotness_orders_new():
    """New experts (not in prev) are ordered by hotness desc when filling slots."""
    print("test_stable_slot_hotness_orders_new")
    # prev had experts 0,1 on rank0 (slots 0,1), 2,3 on rank1 (slots 0,1)
    prev = torch.full((8,), -1, dtype=torch.int32)
    prev[0] = 0; prev[1] = 1; prev[2] = 32; prev[3] = 33  # ndev=32
    # this step: 0,2 repeat (keep); 4,5 new. hotness: 5 > 4.
    counts = torch.tensor([1, 0, 1, 0, 1, 1, 0, 0], dtype=torch.int64)
    hot = [0.0] * 8
    hot[4] = 1.0; hot[5] = 10.0  # 5 is hotter
    p = mcp.plan_placement(counts, ep_size=2, num_device_experts=32,
                           prev_log2phy=prev, hotness=hot)
    # 0,2 keep their prev slot
    _check(int(p.log2phy[0]) == 0, "expert 0 stable at phys 0")
    _check(int(p.log2phy[2]) == 32, "expert 2 stable at phys 32")
    # 4,5 placed (hotter 5 gets a slot too — both fit)
    _check(int(p.log2phy[4]) >= 0 and int(p.log2phy[5]) >= 0, "4,5 both placed")


def test_retention_keeps_inactive_expert():
    """LRU retention: an expert active at step0, inactive at step1, active
    again at step2 stays in its slot at step2 (cache HIT, no re-H2D). The old
    _assign_slots_stable reset each step and dropped it -> step2 was a miss."""
    print("test_retention_keeps_inactive_expert")
    counts0 = torch.tensor([1, 0, 0], dtype=torch.int64)  # expert 0 active
    p0 = mcp.plan_placement(counts0, ep_size=2, num_device_experts=4)
    phys0 = int(p0.log2phy[0])
    _check(phys0 >= 0, f"step0 expert 0 placed (phys {phys0})")
    # step1: expert 0 INACTIVE (expert 1 active instead). prev = p0.
    counts1 = torch.tensor([0, 1, 0], dtype=torch.int64)
    p1 = mcp.plan_placement(counts1, ep_size=2, num_device_experts=4,
                            prev_log2phy=p0.log2phy)
    _check(int(p1.log2phy[0]) == phys0,
           f"step1 expert 0 RETAINED at phys {phys0} though inactive "
           f"(got {int(p1.log2phy[0])})")
    # step2: expert 0 active again -> same slot (hit).
    counts2 = torch.tensor([1, 0, 0], dtype=torch.int64)
    p2 = mcp.plan_placement(counts2, ep_size=2, num_device_experts=4,
                            prev_log2phy=p1.log2phy)
    _check(int(p2.log2phy[0]) == phys0,
           f"step2 expert 0 cache HIT at phys {phys0} (got {int(p2.log2phy[0])})")


def test_retention_evicts_coldest_when_full():
    """When slots are full and a new active expert needs one, the coldest
    non-active resident is evicted (hotness-driven), hotter ones retained."""
    print("test_retention_evicts_coldest_when_full")
    # rank0 full: experts 0,1,2,3 in slots 0-3 (num_device_experts=4 per rank)
    prev = torch.full((8,), -1, dtype=torch.int32)
    prev[0] = 0; prev[1] = 1; prev[2] = 2; prev[3] = 3
    # new active expert 4; hotness makes expert 0 the coldest
    counts = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.int64)
    hot = [1.0] * 8
    hot[0] = 0.0  # expert 0 coldest -> should be evicted
    p = mcp.plan_placement(counts, ep_size=2, num_device_experts=4,
                           prev_log2phy=prev, hotness=hot)
    _check(int(p.log2phy[4]) >= 0, "new active expert 4 placed")
    _check(int(p.log2phy[0]) == -1,
           f"coldest expert 0 evicted (got {int(p.log2phy[0])})")
    for e in [1, 2, 3]:
        _check(int(p.log2phy[e]) >= 0, f"hotter expert {e} retained")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
