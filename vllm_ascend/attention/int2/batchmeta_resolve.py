# SPDX-License-Identifier: Apache-2.0
"""Batch-metadata generation mixin: the request->row/slot resolution
contract (explicit finish/preempt frees; absence is not finish; fail-closed
at capacity; zero-on-assignment)."""

from __future__ import annotations

from typing import Any

from vllm_ascend.attention.int2.helpers import (  # noqa: F401
    _R144_TT_DYN,
)


class Int2BatchMetaResolveMixin:
    # Host-state annotations only: the live attributes are initialized and
    # owned by the backend assembly (slot-cache init + the HP-buffer owner).
    # Bare annotations create no runtime attributes; they pin types for
    # mypy --check-untyped-defs on the mixin bodies.
    _tt_slotmap: dict[str, int]
    _tt_free: set[int]
    _tt_next: int
    _tt_mseq: int
    _tt_pref: dict[str, Any]
    _tt_rcnt: dict[str, Any]
    _tt_rcnt_seen: dict[str, Any]
    _tt_ring_org: dict[str, Any]
    _tt_known: set[str]
    _tt_cur_rids: list[str] | None
    _tt_cur_slots: list[int] | None
    _tt_hp_bufs: Any

    def _tt_resolve_slots(self, attn_metadata, lid, device, dtype):
        """Build per-call b->req_id and b->slot maps from the STABLE request
        identity plumbed through AscendMetadata.request_ids (InputBatch.req_ids)
        + EXPLICIT SchedulerOutput lifecycle events (finished_req_ids /
        preempted_req_ids / new_req_ids).

        CONTRACT (directive P0.1):
        1. A req_id absent from the current batch but with NO finished/removed
           event is TEMPORARILY absent -> state MUST be preserved (NOT freed).
           Absence is NOT a finish signal.
        2. Explicit finished (finished_req_ids): free slot + clear owner/state.
        3. Preempted (preempted_req_ids): vLLM V1 preemption = recompute (KV
           freed, re-prefills) -> free slot + zero state (same as finished; the
           request re-initializes on re-prefill). If V1 ever adds KV-preserving
           swap, this branch must keep slot/state instead.
        4. Map full with NO explicit finished slot -> fail-closed (NO LRU).
        5. On slot assignment (new/reuse) -> ZERO HP/prefix/recent/rcnt
           immediately, NOT deferred to a later prefill.

        The req_id does NOT collide on prefix-cache sharing (two prefix-sharing
        requests have the same first physical block but DIFFERENT req_ids) and
        does NOT recycle on block-pool reuse. Replaces the rejected
        block_tables[b][0] scheme (physical address) + absence-inference +
        LRU eviction."""
        rids = getattr(attn_metadata, "request_ids", None)
        if rids is None:
            # R144_TT_REQID=0 (b-keyed path) or plumbing absent: fall back to
            # row index. _tt_cur_* stay None so _tt_rid/_tt_slot return b.
            return
        _tt_dbg = getattr(self, "_tt_dbg", False)
        finished = getattr(attn_metadata, "finished_req_ids", None) or set()
        preempted = getattr(attn_metadata, "preempted_req_ids", None) or set()
        gone = finished | preempted
        # 1) Explicit finish/preempt -> free slot + ZERO HP immediately.
        if gone:
            try:
                hk, hv = self._tt_hp_bufs(lid, device, dtype)
                _hp = True
            except Exception:
                _hp = False
            for r in gone:
                _slot = self._tt_slotmap.pop(r, None)
                if _slot is not None:
                    self._tt_free.add(_slot)
                    if _hp:
                        hk[_slot].zero_()
                        hv[_slot].zero_()
                self._tt_pref.pop(r, None)
                self._tt_rcnt.pop(r, None)
                self._tt_rcnt_seen.pop(r, None)
                self._tt_ring_org.pop(r, None)  # R198 VRING
                self._tt_known.discard(r)
                if _tt_dbg:
                    _R144_TT_DYN["fin"] += 1
                    try:
                        with open("/tmp/r144_dyn_inst.log", "a") as _lf:
                            _lf.write(f"FIN req={r[:12]} slot={_slot}\n")
                    except Exception:
                        pass
        rids = list(rids)
        cur = set(rids)
        self._tt_known = cur  # live set (diagnostic; slotmap is the source of truth)
        # 2) Assign slots for new/re-appearing req_ids (not in slotmap).
        #    Temporarily-absent req_ids (not in rids, not in gone) stay in
        #    slotmap untouched -> state preserved (contract item 1).
        try:
            hk, hv = self._tt_hp_bufs(lid, device, dtype)
            _hp = True
        except Exception:
            _hp = False
        for r in rids:
            if r not in self._tt_slotmap:
                if self._tt_free:
                    # deterministic reuse (lowest slot first): the slot maps
                    # are rank-local (no collective ever carries them), so
                    # cross-rank agreement is not required for correctness —
                    # but set.pop() ordering also varies per process via
                    # PYTHONHASHSEED, and deterministic assignment removes
                    # the question entirely
                    _slot = min(self._tt_free)
                    self._tt_free.discard(_slot)
                    _ev = "REUSE"
                elif self._tt_next < self._tt_mseq:
                    _slot = self._tt_next
                    self._tt_next += 1
                    _ev = "NEW"
                else:
                    # map full + no explicit finished slot -> fail-closed (NO LRU)
                    raise RuntimeError(
                        f"[3T] slot capacity exceeded: {self._tt_next} >= "
                        f"{self._tt_mseq} (max_num_seqs); no explicit finished "
                        f"slot free; cannot assign {r!r}"
                    )
                self._tt_slotmap[r] = _slot
                # contract item 5: immediate zero on assignment
                if _hp:
                    hk[_slot].zero_()
                    hv[_slot].zero_()
                if _tt_dbg:
                    _R144_TT_DYN["new" if _ev == "NEW" else "reuse"] += 1
                    try:
                        with open("/tmp/r144_dyn_inst.log", "a") as _lf:
                            _lf.write(f"{_ev} req={r[:12]} slot={_slot}\n")
                    except Exception:
                        pass
        self._tt_cur_slots = [self._tt_slotmap[r] for r in rids]
        self._tt_cur_rids = rids
