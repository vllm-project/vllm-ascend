# SPDX-License-Identifier: Apache-2.0
"""Commit compatible GDN layers in one launch, without copying state pools."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.mamba.gdn_recover import _check_cache, _check_conv


@triton.jit(do_not_specialize=["batch_size"])
def _batch_commit(
    StateBase,
    ConvBase,
    LayerOffsets,
    LayerGroups,
    IDs,
    CU,
    Prefix,
    batch_size,
    SS: tl.constexpr,
    RS: tl.constexpr,
    CS: tl.constexpr,
    HS: tl.constexpr,
    W: tl.constexpr,
    HV: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HISTORY: tl.constexpr,
    CHANNELS: tl.constexpr,
    CONV_BLOCK: tl.constexpr,
):
    layer, row, piece = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group = tl.load(LayerGroups + layer)
    slot = tl.load(IDs + group * batch_size + row).to(tl.int64)
    available = tl.load(CU + group * (batch_size + 1) + row + 1) - tl.load(CU + group * (batch_size + 1) + row)
    length = tl.minimum(tl.maximum(tl.load(Prefix + group * batch_size + row), 0), available)
    conv_offset = tl.load(LayerOffsets + layer * 4).to(tl.int64)
    state_offset = tl.load(LayerOffsets + layer * 4 + 1).to(tl.int64)
    record_offset = tl.load(LayerOffsets + layer * 4 + 2).to(tl.int64)
    history_offset = tl.load(LayerOffsets + layer * 4 + 3).to(tl.int64)
    if piece < HV * tl.cdiv(VD, BV):
        head = piece // tl.cdiv(VD, BV)
        tile = piece % tl.cdiv(VD, BV)
        kk = tl.arange(0, BK)
        vv = tile * BV + tl.arange(0, BV)
        mask = (vv[:, None] < VD) & (kk[None, :] < KD)
        if slot > 0 and length > 0:
            offset = state_offset + slot * SS + head * VD * KD + vv[:, None] * KD + kk[None, :]
            state = tl.load(StateBase + offset, mask=mask, other=0)
            for step in range(W):
                if step < length:
                    record = record_offset + slot * RS + (head * W + step) * (VD + KD + 1)
                    key = tl.load(StateBase + record + VD + kk, mask=kk < KD, other=0)
                    correction = tl.load(StateBase + record + vv, mask=vv < VD, other=0)
                    decay = tl.load(StateBase + record + VD + KD)
                    state = state * decay
                    state = state + correction[:, None] * key[None, :]
            tl.store(StateBase + offset, state, mask=mask)
    if piece < tl.cdiv(CHANNELS, CONV_BLOCK) and (slot > 0 and available > 0):
        cc = piece * CONV_BLOCK + tl.arange(0, CONV_BLOCK)
        hh = tl.arange(0, triton.next_power_of_2(HISTORY))
        mask = (hh[:, None] < HISTORY) & (cc[None, :] < CHANNELS)
        if length > 0:
            offset = conv_offset + slot * CS + (length - 1 + hh[:, None]) * CHANNELS + cc[None, :]
        else:
            offset = history_offset + slot * HS + hh[:, None] * CHANNELS + cc[None, :]
        history = tl.load(ConvBase + offset, mask=mask, other=0)
        destination = conv_offset + slot * CS + hh[:, None] * CHANNELS + cc[None, :]
        tl.store(ConvBase + destination, history, mask=mask)


class GDNRecoverBatchLayout:
    """Cold physical descriptors; cache fields may alias across KV groups.

    The cache manager owns disjoint selected slots for groups sharing physical
    views. No new SSM pool is allocated. Cache reinitialization drops this layout.
    """

    def __init__(self, caches):
        self.caches = tuple(caches)
        self.state_base, self.conv_base = caches[0][1], caches[0][0]
        self.strides = tuple(t.stride(0) for t in caches[0])
        self.geometry = _check_cache(caches[0][1], caches[0][2])
        self.history, self.channels = _check_conv(caches[0][0], caches[0][3])
        offsets = []
        for cache in caches:
            offsets.append(
                [
                    cache[0].storage_offset() - self.conv_base.storage_offset(),
                    cache[1].storage_offset() - self.state_base.storage_offset(),
                    cache[2].storage_offset() - self.state_base.storage_offset(),
                    cache[3].storage_offset() - self.conv_base.storage_offset(),
                ]
            )
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=self.state_base.device)
        self._group_signature = None
        self._groups = None

    @classmethod
    def try_create(cls, caches):
        if not caches or any(len(cache) != 4 for cache in caches):
            return None
        first = caches[0]
        storage = first[1].untyped_storage().data_ptr()
        geometry = _check_cache(first[1], first[2])
        conv_geometry = _check_conv(first[0], first[3])
        strides = tuple(t.stride(0) for t in first)
        for cache in caches:
            if any(t.untyped_storage().data_ptr() != storage for t in cache):
                return None
            if tuple(t.stride(0) for t in cache) != strides or cache[0].dtype != first[0].dtype:
                return None
            if _check_cache(cache[1], cache[2]) != geometry or _check_conv(cache[0], cache[3]) != conv_geometry:
                return None
        return cls(caches)

    @torch.no_grad()
    def commit(self, entries):
        groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        group_numbers: list[int] = []
        by_key: dict[tuple[int, int, int], int] = {}
        for _, ids, offsets, prefixes in entries:
            key = (ids.data_ptr(), offsets.data_ptr(), prefixes.data_ptr())
            if key not in by_key:
                by_key[key] = len(groups)
                groups.append((ids, offsets, prefixes))
            group_numbers.append(by_key[key])
        signature = tuple(group_numbers)
        if signature != self._group_signature:
            self._groups = torch.tensor(group_numbers, dtype=torch.int32, device=self.state_base.device)
            self._group_signature = signature
        batch = groups[0][0].numel()
        if any(
            ids.numel() != batch or offsets.numel() != batch + 1 or p.numel() != batch for ids, offsets, p in groups
        ):
            # Groups with different request counts cannot share this grid.
            from vllm_ascend.ops.triton.mamba.gdn_recover import gdn_commit_conv_history, gdn_recover_commit

            for cache, ids, offsets, prefixes in entries:
                gdn_recover_commit(cache[1], cache[2], ids, offsets, prefixes)
                gdn_commit_conv_history(cache[0], cache[3], ids, offsets, prefixes)
            return
        if batch == 0:
            return
        ids = torch.stack([g[0] for g in groups])
        offsets = torch.stack([g[1] for g in groups])
        prefixes = torch.stack([g[2] for g in groups])
        heads, vd, kd, window = self.geometry
        value_tile = 32 if kd > 128 else 64
        pieces = max(heads * triton.cdiv(vd, value_tile), triton.cdiv(self.channels, 128))
        cs, ss, rs, hs = self.strides
        _batch_commit[(len(entries), batch, pieces)](
            self.state_base,
            self.conv_base,
            self.offsets,
            self._groups,
            ids,
            offsets,
            prefixes,
            batch,
            SS=ss,
            RS=rs,
            CS=cs,
            HS=hs,
            W=window,
            HV=heads,
            KD=kd,
            VD=vd,
            BK=triton.next_power_of_2(kd),
            BV=value_tile,
            HISTORY=self.history,
            CHANNELS=self.channels,
            CONV_BLOCK=128,
            enable_fp_fusion=False,
        )
