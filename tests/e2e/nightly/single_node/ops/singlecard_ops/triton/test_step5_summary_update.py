# SPDX-License-Identifier: Apache-2.0
"""Single-operator accuracy tests for the Step5 CSA summary-update Triton ops.

Covers update_decode, update_verify, build_prefill_groups and update_prefill
from vllm_ascend/ops/triton/step5/summary_update.py against pure-torch
references that mirror the kernels' numerics:

* ring rows are bf16 stores of bf16 inputs -> compared bit-exact;
* FP8 e4m3 summaries (fp32 -> bf16 -> e4m3 double rounding) use the FP8
  summary tolerance: at least 99% of touched bytes bit-equal and every
  differing byte within one e4m3 quantization step;
* build_prefill_groups emits integers -> compared exactly.

Each scenario pre-fills the open region's earlier ring rows (the state an
earlier step would have left), then clones the reference state, runs the
kernel on one copy and the reference on the other.
"""

import gc

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.step5.summary_update import (
    build_prefill_groups,
    update_decode,
    update_prefill,
    update_verify,
)

PROXY = 256
ROW_ELEMS = 512  # [index_k(256) | index_z(256)] bf16


def quantize_summary(numerator, denominator):
    """fp32 -> bf16 -> e4m3 double rounding, as _store_summary does."""
    summary = numerator / torch.maximum(denominator, torch.full_like(denominator, 1e-20))
    return summary.to(torch.bfloat16).to(torch.float8_e4m3fn).view(torch.uint8)


def fp8_ulp(values):
    """One e4m3 quantization step around each reference value."""
    absf = values.abs()
    normal = absf >= 2.0**-6
    step = torch.exp2(torch.floor(torch.log2(absf.clamp(min=2.0**-9))) - 3)
    return torch.where(normal, step, torch.full_like(step, 2.0**-9))


class SummaryState:
    """Reference ring + summary cache with the kernels' addressing rules."""

    def __init__(self, num_ring_rows, blocks, regions_per_block, *, pad_regions=0, device="npu"):
        self.capacity = 16  # ring rows per request block (16 KiB / 1 KiB row)
        self.ring = (torch.randn(num_ring_rows, ROW_ELEMS, device=device) * 3).to(torch.bfloat16)
        # Optionally strided blocks (stride(0) > regions*256) to exercise the
        # MEAN_BLOCK addressing path.
        self._mean_full = torch.zeros(blocks, regions_per_block + pad_regions, PROXY, dtype=torch.uint8, device=device)
        self.mean = self._mean_full[:, :regions_per_block]
        self.regions_per_block = regions_per_block

    def clone(self):
        """Reference copy taken AFTER the scenario's history pre-fill."""
        other = SummaryState.__new__(SummaryState)
        other.__dict__.update(
            {
                "capacity": self.capacity,
                "ring": self.ring.clone(),
                "_mean_full": self._mean_full.clone(),
                "mean": None,
                "regions_per_block": self.regions_per_block,
            }
        )
        other.mean = other._mean_full[:, : self.mean.shape[1]]
        return other

    def summary_address(self, slot):
        """Flat element address of the summary of the region holding slot."""
        stride0 = self._mean_full.stride(0)
        tokens_per_block = self.mean.shape[1] * 8
        return slot // tokens_per_block * stride0 + slot % tokens_per_block // 8 * PROXY

    def write_summary(self, slot, numerator, denominator):
        address = self.summary_address(slot)
        self._mean_full.view(-1)[address : address + PROXY] = quantize_summary(numerator, denominator)

    def store_ring_row(self, slot, key, z):
        self.ring[slot, :PROXY] = key
        self.ring[slot, PROXY:] = z

    def ring_row(self, slot):
        return self.ring[slot, :PROXY].float(), self.ring[slot, PROXY:].float()


def check_summaries(state, ref_state, touched_slots):
    """FP8 tolerance over the touched summaries; untouched must be equal."""
    actual = state._mean_full.view(-1)
    reference = ref_state._mean_full.view(-1)
    if not touched_slots:
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg="summaries changed without any close")
        return
    addresses = torch.tensor([ref_state.summary_address(s) for s in touched_slots], device=actual.device)
    lanes = torch.arange(PROXY, device=actual.device)
    flat_idx = (addresses[:, None] + lanes[None, :]).view(-1)
    a = actual[flat_idx].view(torch.float8_e4m3fn).float()
    r = reference[flat_idx].view(torch.float8_e4m3fn).float()
    bit_equal = (actual[flat_idx] == reference[flat_idx]).float().mean().item()
    assert bit_equal >= 0.99, f"only {bit_equal:.4%} summary bytes are bit-equal"
    assert ((a - r).abs() <= fp8_ulp(r) + 1e-12).all(), "summary differs by > 1 e4m3 step"
    # Untouched bytes identical.
    mask = torch.ones_like(actual, dtype=torch.bool)
    mask[flat_idx] = False
    torch.testing.assert_close(actual[mask], reference[mask], rtol=0, atol=0, msg="untouched summary bytes changed")


def random_rows(rows, device, groups=1):
    key = (torch.randn(rows, groups, PROXY, device=device) * 2).to(torch.bfloat16)
    z = (torch.randn(rows, groups, PROXY, device=device) * 2).to(torch.bfloat16)
    return key, z


def prefill_open_region_history(state, ring_block, start_position, first_position):
    """Store the open region's earlier rows, as an earlier step would have."""
    for pos in range(first_position, start_position):
        state.store_ring_row(
            ring_block * state.capacity + pos % state.capacity,
            (torch.randn(PROXY, device=state.ring.device) * 2).to(torch.bfloat16),
            (torch.randn(PROXY, device=state.ring.device) * 2).to(torch.bfloat16),
        )


def ref_update_decode(key, z, slots, positions, ring_base, state):
    """Flat fold of the closing row's 7 ring rows + the row itself."""
    touched = []
    for row in range(slots.numel()):
        slot = int(slots[row])
        base = int(ring_base[row])
        pos = int(positions[row])
        if slot < 0 or base < 0:
            continue
        if pos % 8 == 7:
            offs = torch.arange(7, device=key.device)
            ring_slots = base + (pos - 7 + offs) % state.capacity
            rkey = torch.stack([state.ring[s, :PROXY] for s in ring_slots]).float()
            rz = torch.stack([state.ring[s, PROXY:] for s in ring_slots]).float()
            cur_key = key[row].float()
            cur_z = z[row].float()
            maximum = torch.maximum(rz.max(), cur_z.max())
            rweight = torch.exp(rz - maximum)
            cweight = torch.exp(cur_z - maximum)
            numerator = (rweight * rkey).sum(0) + cweight * cur_key
            denominator = rweight.sum(0) + cweight
            state.write_summary(slot, numerator, denominator)
            touched.append(slot)
        else:
            state.store_ring_row(base + pos % state.capacity, key[row], z[row])
    return touched


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("blocks_pad", [0, 3], ids=["contiguous", "strided-blocks"])
def test_update_decode(blocks_pad):
    torch.manual_seed(0)
    device = "npu"
    num_reqs = 6
    state = SummaryState(
        num_ring_rows=16 * (num_reqs + 2),
        blocks=num_reqs + 2,
        regions_per_block=16,
        pad_regions=blocks_pad,
        device=device,
    )

    rows = []
    for req in range(num_reqs):
        seq_len = 8 * (req + 2) + req  # positions 15..63: several closes
        ring_block = req + 1  # distinct ring block per request
        summary_block = req + 1
        pos = seq_len - 1
        # Carry history: the open region's earlier rows ([pos - pos%8, pos)),
        # stored by earlier decode steps (empty for the first position).
        prefill_open_region_history(state, ring_block, pos, pos - pos % 8)
        rows.append(
            dict(
                slot=summary_block * 16 * 8 + pos,
                pos=pos,
                base=ring_block * state.capacity,
            )
        )
    # Padding rows: slot/base = -1 must be skipped entirely.
    rows.append(dict(slot=-1, pos=7, base=2 * state.capacity))
    rows.append(dict(slot=16 * 8, pos=7, base=-1))

    ref_state = state.clone()
    slots = torch.tensor([r["slot"] for r in rows], dtype=torch.int64, device=device)
    positions = torch.tensor([r["pos"] for r in rows], dtype=torch.int64, device=device)
    ring_base = torch.tensor([r["base"] for r in rows], dtype=torch.int64, device=device)
    key, z = random_rows(len(rows), device)

    update_decode(key, z, slots, positions, ring_base, state.mean, state.ring, state.capacity)

    touched = ref_update_decode(key, z, slots, positions, ring_base, ref_state)
    torch.testing.assert_close(state.ring, ref_state.ring, rtol=0, atol=0, msg="ring stores are not bit-exact")
    check_summaries(state, ref_state, touched)

    gc.collect()
    torch.npu.empty_cache()


# ---------------------------------------------------------------- update_verify


def ref_update_verify(key, z, slots, query_start_loc, seq_lens, ring_base, state):
    """Sequential online-softmax replay, in the kernel's row order."""
    touched = []
    for request in range(seq_lens.shape[0]):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        if start >= end:
            continue
        base = int(ring_base[start])
        seq = int(seq_lens[request])
        for row in range(start, end):
            slot = int(slots[row])
            if slot < 0 or base < 0:
                continue
            pos = seq - (end - row)
            state.store_ring_row(base + pos % state.capacity, key[row, 0], z[row, 0])
            if pos % 8 == 7:
                numerator = torch.zeros(PROXY, device=key.device)
                denominator = torch.zeros(PROXY, device=key.device)
                maximum = torch.tensor(-float("inf"), device=key.device)
                for p in range(pos - 7, pos + 1):
                    ring_key, ring_z = state.ring_row(base + p % state.capacity)
                    updated_max = torch.maximum(maximum, ring_z.max())
                    scale = torch.where(
                        denominator > 0,
                        torch.exp(maximum - updated_max),
                        torch.zeros((), device=key.device),
                    )
                    weight = torch.exp(ring_z - updated_max)
                    numerator = numerator * scale + weight * ring_key
                    denominator = denominator * scale + weight
                    maximum = updated_max
                state.write_summary(slot, numerator, denominator)
                touched.append(slot)
    return touched


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("draft_tokens", [1, 2, 3])
def test_update_verify(draft_tokens):
    torch.manual_seed(0)
    device = "npu"
    query_len = draft_tokens + 1
    num_reqs = 5
    state = SummaryState(
        num_ring_rows=16 * (num_reqs + 2),
        blocks=num_reqs + 2,
        regions_per_block=16,
        device=device,
    )

    rows = []
    spans = [0]
    for req in range(num_reqs):
        seq = 8 * (req + 3) + 5 + query_len  # each span closes 0-1 regions
        ring_block = req + 1
        first_pos = seq - query_len
        # History rows before the span's first position.
        prefill_open_region_history(state, ring_block, first_pos, first_pos - first_pos % 8)
        for j in range(query_len):
            pos = first_pos + j
            slot = (req + 1) * 16 * 8 + pos if j != query_len // 2 else -1
            rows.append(dict(slot=slot, pos=pos, base=ring_block * state.capacity))
        spans.append(len(rows))

    ref_state = state.clone()
    slots = torch.tensor([r["slot"] for r in rows], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor(spans, dtype=torch.int64, device=device)
    seq_lens = torch.tensor(
        [8 * (req + 3) + 5 + query_len for req in range(num_reqs)],
        dtype=torch.int64,
        device=device,
    )
    ring_base = torch.tensor([r["base"] for r in rows], dtype=torch.int64, device=device)
    key, z = random_rows(len(rows), device)

    update_verify(key, z, slots, query_start_loc, seq_lens, ring_base, state.mean, state.ring, state.capacity)

    touched = ref_update_verify(key, z, slots, query_start_loc, seq_lens, ring_base, ref_state)
    torch.testing.assert_close(state.ring, ref_state.ring, rtol=0, atol=0, msg="ring stores are not bit-exact")
    check_summaries(state, ref_state, touched)

    gc.collect()
    torch.npu.empty_cache()


def ref_build_prefill_groups(slots, positions, owners, query_start_loc):
    rows = slots.shape[0]
    metadata = torch.full((rows, 10), -1, dtype=torch.int64)
    offset = torch.arange(8)
    for row in range(rows):
        owner = int(owners[row])
        start = int(query_start_loc[owner])
        end = int(query_start_loc[owner + 1])
        pos = int(positions[row])
        head = (row == start) or (pos % 8 == 0)
        step = offset - pos % 8
        cand = row + step
        in_group = head & (step >= 0) & (cand < end)
        slot_vals = torch.where(in_group, slots[cand.clamp(0, rows - 1)], torch.tensor(-1))
        valid = in_group & (slot_vals >= 0)
        first = int(torch.where(valid, offset, torch.tensor(8)).min())
        region = int(torch.where(valid, slot_vals // 8, torch.tensor(-1)).max())
        metadata[row, 0] = region if first < 8 else -1
        metadata[row, 1] = first
        metadata[row, 2:] = torch.where(valid, cand, torch.tensor(-1))
    return metadata


def ref_update_prefill(key, z, metadata, positions, ring_base, ring_slots, state):
    """Close folds first (flat tiles), then the ring stores - launch order."""
    touched = []
    device = key.device
    offset = torch.arange(8, device=device)
    for row in range(metadata.shape[0]):
        region = int(metadata[row, 0])
        rows8 = metadata[row, 2:]
        if region < 0 or int(rows8[7]) < 0:
            continue
        first = int(metadata[row, 1])
        head = int(rows8[first])
        pos = int(positions[head])
        base = int(ring_base[head])

        carry = offset < first
        ring_slots_t = base + (pos - first + offset) % state.capacity
        rkey = torch.stack([state.ring[s, :PROXY] for s in ring_slots_t]).float()
        rz = torch.stack([state.ring[s, PROXY:] for s in ring_slots_t]).float()
        rkey = torch.where(carry[:, None], rkey, torch.zeros_like(rkey))
        rz = torch.where(carry[:, None], rz, torch.full_like(rz, -float("inf")))

        valid = rows8 >= 0
        safe = torch.where(valid, rows8, torch.zeros_like(rows8))
        ckey = key[safe, 0].float()
        cz = z[safe, 0].float()
        ckey = torch.where(valid[:, None], ckey, torch.zeros_like(ckey))
        cz = torch.where(valid[:, None], cz, torch.full_like(cz, -float("inf")))

        maximum = torch.maximum(
            torch.max(torch.where(carry[:, None], rz, torch.full_like(rz, -float("inf")))),
            torch.max(torch.where(valid[:, None], cz, torch.full_like(cz, -float("inf")))),
        )
        rweight = torch.exp(rz - maximum)
        cweight = torch.exp(cz - maximum)
        numerator = (rweight * rkey).sum(0) + (cweight * ckey).sum(0)
        denominator = rweight.sum(0) + cweight.sum(0)
        state.write_summary(region * 8, numerator, denominator)
        touched.append(region * 8)

    for row in range(ring_slots.shape[0]):
        slot = int(ring_slots[row])
        if slot >= 0:
            state.store_ring_row(slot, key[row, 0], z[row, 0])
    return touched


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
@pytest.mark.parametrize("blocks_pad", [0, 3], ids=["contiguous", "strided-blocks"])
def test_build_prefill_groups_and_update_prefill(blocks_pad):
    torch.manual_seed(0)
    device = "npu"
    num_reqs = 5
    state = SummaryState(
        num_ring_rows=16 * (num_reqs + 2),
        blocks=num_reqs + 2,
        regions_per_block=16,
        pad_regions=blocks_pad,
        device=device,
    )

    slots, positions, owners = [], [], []
    ring_base, ring_slots = [], []
    spans = [0]
    for req in range(num_reqs):
        # Unaligned start so regions straddle steps; some chunks close, some
        # stay open; chunk lengths cross the 8-row group size.
        start_pos = 3 + 11 * req + 8 * (req % 3)
        chunk_len = 5 + 3 * req + (req % 2)
        ring_block = req + 1
        # History: rows of the open region stored by an earlier step.
        prefill_open_region_history(state, ring_block, start_pos, start_pos - start_pos % 8)
        for j in range(chunk_len):
            pos = start_pos + j
            valid = not (req == 2 and j == 4)  # one padding row with slot -1
            summary_slot = (req + 1) * 16 * 8 + pos if valid else -1
            base = ring_block * state.capacity if valid else -1
            slots.append(summary_slot)
            positions.append(pos)
            owners.append(req)
            ring_base.append(base)
            # Ring keep rule of _ring_metadata_kernel (pos >= seq_len -
            # capacity): rows older than the capacity window get slot -1, so
            # each ring row is written at most once per launch (a chunk
            # longer than the capacity would otherwise wrap onto itself).
            seq_end = start_pos + chunk_len
            keep = valid and pos >= seq_end - state.capacity
            ring_slots.append(base + pos % state.capacity if keep else -1)
        spans.append(len(slots))

    ref_state = state.clone()
    slots_t = torch.tensor(slots, dtype=torch.int64, device=device)
    positions_t = torch.tensor(positions, dtype=torch.int64, device=device)
    owners_t = torch.tensor(owners, dtype=torch.int64, device=device)
    query_start_loc = torch.tensor(spans, dtype=torch.int64, device=device)
    ring_base_t = torch.tensor(ring_base, dtype=torch.int64, device=device)
    ring_slots_t = torch.tensor(ring_slots, dtype=torch.int64, device=device)
    key, z = random_rows(len(slots), device)

    metadata = build_prefill_groups(slots_t, positions_t, owners_t, query_start_loc)
    metadata_ref = ref_build_prefill_groups(slots_t.cpu(), positions_t.cpu(), owners_t.cpu(), query_start_loc.cpu())
    torch.testing.assert_close(metadata.cpu(), metadata_ref, rtol=0, atol=0, msg="grouping metadata mismatch")

    update_prefill(
        key,
        z,
        metadata,
        positions_t,
        ring_base_t,
        ring_slots_t,
        state.mean,
        state.ring,
        state.capacity,
    )

    touched = ref_update_prefill(key, z, metadata, positions_t, ring_base_t, ring_slots_t, ref_state)
    torch.testing.assert_close(state.ring, ref_state.ring, rtol=0, atol=0, msg="ring stores are not bit-exact")
    check_summaries(state, ref_state, touched)

    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires Ascend A5",
)
def test_summary_update_empty_batches():
    device = "npu"
    state = SummaryState(num_ring_rows=64, blocks=4, regions_per_block=16, device=device)
    empty_i64 = torch.zeros(0, dtype=torch.int64, device=device)
    key, z = random_rows(0, device)
    # update_decode / update_verify wrappers must no-op on empty batches.
    update_decode(key, z, empty_i64, empty_i64, empty_i64, state.mean, state.ring, state.capacity)
    update_verify(
        key,
        z,
        empty_i64,
        torch.zeros(1, dtype=torch.int64, device=device),
        empty_i64,
        empty_i64,
        state.mean,
        state.ring,
        state.capacity,
    )
    metadata = build_prefill_groups(empty_i64, empty_i64, empty_i64, torch.zeros(1, dtype=torch.int64, device=device))
    assert metadata.shape == (0, 10)
    update_prefill(
        key,
        z,
        metadata,
        empty_i64,
        empty_i64,
        empty_i64,
        state.mean,
        state.ring,
        state.capacity,
    )
    gc.collect()
    torch.npu.empty_cache()
