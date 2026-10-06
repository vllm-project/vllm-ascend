# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in this directory for the full text of the License.

"""Interleaved/BF16 compatibility variant of ``indexer_prologue_qw`` NPU kernel (CANNBotDSL, Ascend950 / dav-3510).

One MIX_AIC_1_2 kernel, two cube paths and one vector epilogue::

    C_w  BF16  x @ ww^T  --fixpipe(deq_scale=softmax_scale)-->  GM w
    C_q  MXFP8 qr @ wqb^T  =fixpipe(dual_dst)=>  cv_ub  --VF-->  GM q, descale_q

Tiling for the frozen case ``dim=5120, q_lora=1280, N=32, D=128, Dr=64``: the
``T`` axis is split over AICs in tiles of 128 rows, ``qr`` plus its MX scale
stay resident in L1 for the whole 32-head loop, ``wqb`` streams in 80 KB K
windows, and the W path's ``x`` traffic is chopped into 32 windows retired one
per head so it never stalls the cube.  The cube roofline at T=4096 is about 52.7 us vs HBM 38.7 us.

Everything the vector unit does is raw-VF register code.  ``vcast`` has no fp4
lowering in this wheel (verified), so E2M1 nibbles are produced by integer bit
manipulation and a small gather table, then packed two-per-byte.
"""

from __future__ import annotations

# Vendor tuning knobs are frozen to the tested defaults in this integration.
# The A5 Flash indexer selects this implementation after weight validation.
from collections import deque

import torch
from cannbotdsl import (
    Buffer,
    Channel,
    ChannelKind,
    Dim,
    Float8E4M3FN,
    Float8E8M0,
    MemLoc,
    Tensor,
    TensorSpec,
    const_expr,
    dtypes,
    get_block_idx,
    get_block_num,
    get_subblock_dim,
    get_subblock_id,
    jit,
    kernel,
    make_copy_engine,
    matmul,
    mem_copy,
    range_constexpr,
    vf,
)
from cannbotdsl.arena import _current_channel_arena
from cannbotdsl.ops.reg import (
    PackMode,
    create_mask,
    full_mask,
    vabs,
    vadd,
    vadds,
    varange,
    vbitwise_and,
    vbitwise_or,
    vcast,
    vdups,
    vgather,
    vgather_reg,
    vgts,
    vload,
    vlts,
    vmaxs,
    vmem_bar,
    vmul,
    vne,
    vpair_reduce_sum,
    vreduce_max,
    vreinterpret,
    vselect,
    vshl,
    vshr,
    vstore,
    vstore_first,
    vstore_pack,
    vsub,
)
from cannbotdsl.ops.sync import (
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_block_wait,
    global_sync_all,
    vec_sync_all,
    vec_sync_block_wait,
)
from cannbotdsl.tensor import (
    local_slice,
    make_bounded_tiler,
    make_partition_tiler,
    partition_view,
    tile_view,
)
from cannbotdsl.types.scalar import PIPE

try:
    from device_properties import get_device_properties
except ImportError:  # packaged net/ops wheel has no samples/ on sys.path

    def get_device_properties():
        import torch

        return torch.npu.get_device_properties(0)


__all__ = [
    "IndexerPrologueQw",
    "build_e2m1_lut",
    "plan",
    "recent_calls",
    "resolve_cube_cores",
]

I32 = dtypes.int32
# ``get_block_idx`` and ``get_block_num`` are int64, and a dynamic range wants
# start, stop and step in one dtype, so the runtime work count is int64 too.
I64 = dtypes.int64
U32 = dtypes.uint32
F32 = dtypes.float32
U8 = dtypes.uint8

VL = 64  # fp32 lanes in one 256-byte raw vector register
SUBBLOCKS = 2  # AIVs per block on MIX_AIC_1_2; get_subblock_dim() at runtime
MX_GROUP = 32  # elements per E8M0 scale
MX_K_ALIGN = 64  # K elements per public paired-scale group
MX_PAIR = 2
E8M0_BIAS = 127
FP4_E2M1_MAX = 6.0
FP32_EXP_SHIFT = 23
FP32_MANTISSA_MASK = 0x7FFFFF
FP32_HALF_MANTISSA = 0x400000  # significand exactly 1.5
FP32_MAG_MASK = 0x7FFFFFFF
# |y| <= 6 always, so (bits(|y|) >> LUT_SHIFT) <= 1036.
LUT_SHIFT = 20
LUT_LEN = 1088


# Channel depths, overridable while tuning so a sweep is one env var per run.
# Read at trace time, so every one of these is a compile-time constant.
def _depth(name: str, default: int) -> int:
    return int(default)


D_CV = _depth("CV", 2)
D_QUB = _depth("QUB", 2)
# How many K windows one head's ``wqb`` reduction is filled in.  A core with
# several heads has head h+1 to overlap with, so a coarse split is enough; a
# single-head core has no next head and instead wants its own windows small
# enough to keep MTE2 ahead of the mmads.  5 windows (256 elements each, the
# value the old window-size knob had converged on) measured best there.
D_WQB = _depth("WQB", 2)
D_WQB_SINGLE_HEAD = _depth("WQB_SINGLE_HEAD", 5)
# One buf id per channel slot, and the CUBE side has 32.  Everything except
# wqb_l1 and its scales is fixed-depth, hence the constant.
CUBE_BUF_IDS = 32
FIXED_CUBE_SLOTS = 16
D_L0C_Q = _depth("L0C_Q", 2)
D_L0AB_Q = _depth("L0AB_Q", 2)
D_L0B_Q = _depth("L0B_Q", D_L0AB_Q)
D_X = _depth("X", 2)
# The W path's L0 staging is deliberately single-buffered.  Every L0A byte it
# gives up buys a larger ``base_k_w`` (32 -> 80 for the frozen case), and a
# fatter, rarer mmad is worth far more here than double buffering: W is 3% of
# the cube work but its mmad count was setting the rate at which the Q path
# could get its own L0 loads through the in-order MTE1 pipe.  Measured at
# T=4096: 80.3 us -> 70.2 us, chip MFU 65.7% -> 75.2%.
D_L0AB_W = _depth("L0AB_W", 1)

# W-path schedule inside the Q head loop.  MTE2 is one in-order pipe, so where
# the W fills sit relative to the Q fills decides whether Q's mmads wait behind
# them.  "wfirst" is the original: W fills and W mmads both ahead of Q.
W_SCHED = "wfirst"

# Diagnostic-only ablations; some produce wrong numbers by construction and
# exist purely to attribute wall time to a pipe.  "" is the real kernel.
PROBE = ""

# Where the W reduction's all-core rendezvous goes, and how many phases it
# runs.  What it has to wait for is "every partial is in GM", which is true a
# microsecond into the kernel -- but a barrier blocks until the slowest core
# *arrives*, so its cost is whatever work the cores still have between them at
# that point, not the dependency.  T=72, min of 20 launches:
#
#     placement   barrier                 T=72
#     ----------  ----------------------  --------
#     none        none (racy)              10.67    reduction alone: +0.16
#     tail        lean, 2 phases           10.86    <- default
#     tail        global_sync_all          12.41
#     early       lean, 2 phases           12.58
#     early       global_sync_all          12.44
#     (the atomic-add reduction this replaced)      12.95
#
# "early" (right after this core's partial store) looks like the natural place
# -- the arrivals are tight there because every core has the same tiny slice
# of W behind it -- but it is the worst one: the wait lands *before* the Q head
# loop, so nothing has been issued that could cover it and every core's Q path
# starts late.  At the tail each core has ~10 us of Q work behind it and the
# rendezvous is down to the skew.
#
# The four-phase barrier is expensive at the tail for a specific reason: its
# phase 1 funnels each block's AIVs into their AIC, and at the tail the AIV is
# behind the AIC (it is still retiring the last head's epilogue), so phase 1
# puts the AIV's tail latency on the AIC's critical path.  Nothing here needs
# it -- see store_to_vec_barrier.
W_REDUCE_AT = "tail"
# "lean" for the two-phase barrier below, "full" for global_sync_all's four.
W_SYNC = "lean"

# L2 retention hints.  At one tile/core, caching Q-path weights in L2 is a
# wash.  At two-plus tiles/core the same hint *hurts*: T=8192 went 144 us
# (ctl=1) -> 139 us (ctl=0).  Plan.l2_q therefore defaults to 0 when the
# core owns more than one T-tile; IPQW_L2_Q still overrides.  ``x`` is
# streamed once and does not want L2.
L2_Q = None
L2_W = 1
# Issue the next tile's ``qr`` load after the last Q mmad of this tile so it
# overlaps the last head's vector epilogue.  Default on when a core has more
# than one T-tile.  IPQW_PREFETCH=0/1 overrides.
PREFETCH = None
UNIT_FLAG = 0
# Split the head axis over leftover AICs when T-tiles alone cannot fill the
# chip.  Once there are at least as many T-tiles as cube cores the original
# T-only schedule is kept, so the 75% MFU path is unchanged.
# IPQW_HEAD_SPLIT=0 restores the old 1-core-per-T-tile launch for A/B.
HEAD_SPLIT = 1
# What one extra launched block costs, in units of one head GEMM's per-core
# weight traffic.  Fitted from the pinned block-count sweep in
# ``_plan_head_split``; see there for the measurements and why the head split
# has an optimum rather than wanting every core.
BLOCK_COST_HEADS = 0.146
# Split the W path's K reduction across the cores that share a T-tile, each
# writing its partial to rows it alone owns, and sum them on the vector side.
# Letting one leader per tile own the whole reduction leaves it reading the
# entire x tile alone: at T=72 that is 1388 KB on the leader against 268 KB on
# each of the other cores, and the wall clock is the leader.
W_K_SPLIT = 1
# Launch the whole chip even when there are fewer work items than cores, so that
# the grid stops being a specialisation dimension.  A block with no work items
# falls straight out of the work loop; see the ``used_cores`` comment in
# ``_plan_head_split`` for what that is worth measured.
GRID_FULL = 1
# Where the two templates meet, counted in T-tiles rather than in cores.
#
# This being a core count is what used to put the core count in the binary. The
# boundary was ``n_tiles >= cube_cores``, and since the head split and the band
# geometry are chosen against the same number, a 28-, 32- and 36-core part each
# needed their own build of both templates -- and within a part the split moved
# with T, which is where the nine binaries came from.
#
# The tile count a T-only schedule needs before it fills a chip on its own is
# roughly the core count, so this is that value for the parts in play, frozen. A
# part with more cores switches to split-T slightly early and one with fewer runs
# split-K slightly long; that costs a little occupancy at the boundary and no
# correctness, because neither template assumes anything about the grid -- the
# work loop strides by ``get_block_num()``, which is a runtime value.
SPLIT_T_TILES = 25
# The split-K head split, pinned.  ``_plan_head_split`` used to score this per
# (T, cores), and it is genuinely T-dependent -- 16 wins at T=1 and 2 wins at
# T=2048 -- but every distinct value is another binary, because the block count
# is a loop trip count, a ``k_block`` width and a band UB width.  16 is the
# decode optimum, and decode is the phase where this operator's latency is on
# the critical path; see the measured table in ``_plan_head_split`` for what the
# large end of the split-K range gives up for it.
HEAD_BLOCKS_K = 16
# The band height of the workspace W reduction, pinned for the same reason: it is
# a UB height and a register unroll.  The narrowest legal band spreads best over
# the AIVs and 2 is what every decode shape already chose, so pinning it here
# costs the large end of split-K some DMA round trips and decode nothing.
ROWS_RED_K = 2
# Upper bound on the launch grid, used only to bound the ``ws`` extent.
#
# The grid cannot be a plain host integer without the core count becoming part of
# the binary: ``compute_binary_key`` hashes the verified IR, and the IR carries
# the block dim, so a 28-, 32- and 36-core export of identical source produce
# three different keys.  It also cannot exceed the physical core count, because
# the W reduction rendezvous waits on every block and a second scheduling wave
# deadlocks it.  So the grid is carried the way ``mixed_quant_sparse_flash_mla``
# carries it: the host sizes a tensor to it and the kernel reads the extent back,
# which keeps it symbolic.  ``ws`` gets ``used_cores`` marker rows past its
# partials for that, and the Dim that bounds them has to admit any core count on
# any part or the bound itself would put the core count back in the contract.
MAX_CORES = 128
# ``IPQW_T_DYN=0`` goes back to a binary per T, so the price of T being a runtime
# value can be measured rather than argued about.  See ``_plan_for``.
T_DYN = 1
# Rows of T one core owns per work item, i.e. the M of every GEMM here.  It
# cannot go above 128: the L1 residents (``qr``, the ``x`` and ``ww`` buffers)
# reach 439 KB of the 512 KB there, and 192 would need 565 KB.  Below 128 it
# only re-reads the 5.24 MB ``wqb`` more times per row of T, which loses even
# where the round quantisation is at its worst -- 28 cores, where T=4096 is 32
# tiles over 28 cores, measures 158.3 us at 128 against 172.7 at 64 and 241.1
# at 32.  ``IPQW_BASE_M`` is that A/B.
BASE_M = 128
# T is the only free axis, and it is ``batch * seq``, so nothing about it is
# aligned: decode contributes 1, 4, 6, 8, 12, 24 (batch times 1 or MTP's 6) and
# prefill contributes batch times 1024..8192.  Any T in this range is legal; the
# row fractal is satisfied by the tile height, not by T (see ``base_m``).
T_MIN = 1
T_MAX = 256 * 1024
# FRACTAL_NZ grid of the two weight inputs: 16 rows per fractal, and a C0 that
# is 32 bytes wide, i.e. 32 one-byte or 16 two-byte elements.
NZ_M_FRAC = 16
NZ_C0_1B = 32
NZ_C0_2B = 16

L1_BYTES = 512 * 1024
L0AB_BYTES = 64 * 1024
L0C_BYTES = 256 * 1024
UB_BYTES = 248 * 1024


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def resolve_cube_cores(cube_cores: int | None = None) -> int:
    """Cube-core count for the tile plan: a device property, not a constant.

    Every core-count decision in :class:`Plan` (head split, used cores, the
    L2/prefetch thresholds) is keyed on this.  It is resolved on the host and
    then carried as a compile-time constant so the compile-only path never
    enters the NPU runtime.  ``IPQW_CUBE_CORES`` overrides for A/B runs.
    """
    if cube_cores is None:
        forced = None
        cube_cores = int(forced) if forced else get_device_properties().cube_core_num
    cube_cores = int(cube_cores)
    if cube_cores <= 0:
        raise ValueError(f"cube_cores must be positive, got {cube_cores}")
    return cube_cores


def scale_l1_len(k: int) -> int:
    """Contiguous E8M0 length in L1 for a K extent (paired public layout)."""
    return ceil_div(k, MX_K_ALIGN) * MX_PAIR


# ---------------------------------------------------------------------------
# E2M1 rounding table
# ---------------------------------------------------------------------------


def build_e2m1_lut() -> torch.Tensor:
    """Gather table mapping ``bits(|y|) >> 20`` to a 3-bit E2M1 magnitude code.

    ``y`` is the already-descaled value, so ``|y| <= 6``.  Index bits are the
    biased fp32 exponent times eight plus the top three mantissa bits; every
    E2M1 midpoint (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0) is exactly on such a
    bucket boundary, so a truncating index reproduces round-to-nearest with
    ties away from zero -- the rule the CPU golden uses.
    """
    codes = torch.zeros(LUT_LEN, dtype=torch.int32)
    midpoints = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    for exponent in range(120, 133):
        for mantissa in range(8):
            index = exponent * 8 + mantissa
            if index >= LUT_LEN:
                continue
            value = 2.0 ** (exponent - E8M0_BIAS) * (1.0 + mantissa / 8.0)
            codes[index] = sum(1 for m in midpoints if value >= m)
    codes[133 * 8 :] = 7
    return codes


# ---------------------------------------------------------------------------
# Host tiling
# ---------------------------------------------------------------------------


class Plan:
    """Static tile plan; every field is a compile-time constant.

    ``t_dyn`` builds a plan whose kernel reads its tile count from a runtime
    argument instead of this object, so one binary serves every T the same
    template covers.  It is only legal where no *other* field depends on T --
    see :meth:`_check_t_dynamic` -- which is exactly the split-T regime.
    """

    def __init__(
        self,
        t: int,
        dim: int,
        q_lora: int,
        n_heads: int,
        d: int,
        dr: int,
        cube_cores: int | None = None,
        t_dyn: bool = False,
    ):
        if t < T_MIN or t > T_MAX:
            raise ValueError(f"T must be in [{T_MIN}, {T_MAX}], got {t}")
        self.t = t
        self.t_dyn = bool(t_dyn)
        self.dim = dim
        self.q_lora = q_lora
        self.n_heads = n_heads
        self.d = d
        self.dr = dr
        self.cube_cores = resolve_cube_cores(cube_cores)

        # Sizing the tile to T is what used to make T compile-time below one
        # tile per core: ``base_m`` is the height of every L1/L0/UB buffer, so a
        # T-dependent value here fragments the small-T range into one binary per
        # T.  A dynamic plan pins it and lets the short tile be short -- which
        # costs nothing measurable, because the vector epilogue is already cut
        # to the live rows (``vec.epilogue(cv_ub, rows)``) and the wasted mmad
        # rows land in a cube that is only ~22% busy down here.
        self.base_m = BASE_M if (self.t_dyn or t >= BASE_M) else max(16, ceil_div(t, 16) * 16)
        self.rows_vec = self.base_m // 2
        self.n_tiles = ceil_div(t, self.base_m)
        self._plan_head_split()
        # See the L2_Q / PREFETCH comments at module level.  Both want to know
        # whether a core gets more than one T-tile, which is a runtime question
        # now, so it is answered by the template instead: heads are split exactly
        # when the T-tiles cannot fill the chip on their own, so a split-K binary
        # is the one whose cores get a single tile.  This agrees with the old
        # per-T choice everywhere except tiles 25..31, which used to be its own
        # set of binaries and which the split-T setting measured faster on.
        more_than_one_tile = self.n_head_blocks == 1
        self.l2_q = L2_Q if L2_Q is not None else (0 if more_than_one_tile else 1)
        self.prefetch = PREFETCH if PREFETCH is not None else (1 if more_than_one_tile else 0)

        # W path always produces the full (T_tile, N) tile: splitting it over
        # N turns the BF16 GEMM into a GEMV and makes every AIC read the whole
        # ``x`` tile.  Split the K reduction instead: the ``n_head_blocks``
        # cores that share a T-tile each take ``dim / n_head_blocks`` columns
        # and write their partial to their own workspace rows, so ``x`` is
        # still read once but spread over every core.  Without heads to split
        # (T >= 4096) the T-tiles already balance and one core owns the whole
        # reduction, which lands straight in ``w`` with no workspace at all.
        self.base_n_w = n_heads
        self.w_k_split = bool(W_K_SPLIT) and self.n_head_blocks > 1 and dim % self.n_head_blocks == 0
        self.w_ws = self.w_k_split
        self._plan_reduce()
        self.k_block = dim // self.n_head_blocks if self.w_k_split else dim
        self.w_leader_only = self.n_head_blocks > 1 and not self.w_k_split
        self.k_l1_w = self._pick_k_l1_w()
        self.base_k_w = self._pick_base_k_w()
        self.step_k_w = self.k_l1_w // self.base_k_w
        self.k_l1_tiles_w = ceil_div(self.k_block, self.k_l1_w)
        self.w_interleaved = self.k_l1_tiles_w % self.heads_per_block == 0
        # Interleaving W under Q only pays when this core has enough Q heads
        # to hide the ``x`` traffic.  A 1-head leader cannot hide 1.25 MB, so
        # it runs W as one contiguous reduction instead; a K slice is small
        # enough that it does not need hiding either.
        self.w_separate = self.w_k_split or (self.w_leader_only and self.heads_per_block <= 8)
        self.w_per_head = (
            0 if self.w_separate else (self.k_l1_tiles_w // self.heads_per_block if self.w_interleaved else 0)
        )

        # Q path: one head per output tile keeps the full D on chip for RoPE.
        self.base_n_q = d
        self.base_k_q = 0 or min(128, ceil_div(q_lora, MX_K_ALIGN) * MX_K_ALIGN)
        self.k_l1_tiles_q = self._pick_k_windows_q()
        self.k_l1_q = q_lora // self.k_l1_tiles_q
        self.step_k_q = self.k_l1_q // self.base_k_q
        self.n_buf_wqb = self.k_l1_tiles_q
        self.k_l0_tiles_q = ceil_div(q_lora, self.base_k_q)
        self.scale_k_l0_len = self.base_k_q // MX_GROUP
        self.scale_k_l1_len_q = scale_l1_len(self.k_l1_q)
        self.scale_k_len_q = scale_l1_len(q_lora)
        self.qr_k_l1 = ceil_div(q_lora, MX_K_ALIGN) * MX_K_ALIGN

        # Vector epilogue geometry.
        self.n_reg = d // VL
        self.groups = d // MX_GROUP
        self.rope_regs = dr // VL
        self.enc_regs = ceil_div(self.rows_vec * self.groups, VL)
        self.amax_len = self.enc_regs * VL

        self._check()
        if self.t_dyn:
            self._check_t_dynamic()

    def _plan_head_split(self) -> None:
        """Split the head axis so the T-tiles alone do not have to fill the chip.

        Decode T is 1..192, which is one or two row tiles -- a T-only schedule
        would launch one or two cores and leave the other 30 idle, reporting the
        2-5% chip MFU the official small-T cases used to show.  The missing work
        is the other 31 heads, which are independent GEMMs, so the free axis for
        occupancy is N, not T.  Once the heads occupy the chip the shape is
        weight-load bound (``wqb`` is 5.24 MB whatever T is) and the target
        flips from MFU to MBU.

        Above ``cube_cores`` T-tiles the T axis fills the chip on its own and the
        head split is off: a core then owns a whole T-tile, keeps its ``qr`` in
        L1 for all 32 heads, and owns the W reduction outright.  That is the
        split-T template.  Measured at 28 cores and T=4096 the head split makes
        no difference at all there (157.6 .. 159.7 us over every block count from
        1 to 32, including the 4-cores-do-two-tiles imbalance), because the shape
        is MTE2-throughput bound and no split changes the bytes; so the cheapest
        schedule wins by default.

        Below that, more blocks is *not* simply better, and this is the one place
        where chasing occupancy actively costs time.  Splitting heads ``b`` ways
        launches ``b`` blocks, and every launched block pays a share of the
        the grid barrier and the band sum the workspace W reduction needs, while
        it only divides the per-core weight traffic.  Measured at 32 cores with
        the block count pinned (on the atomic-add reduction this replaced; the
        shape of the curve is what the score is fitted to, and the workspace
        reduction moved every column down by about the same 2 us):

            T     4 blk   8 blk   16 blk   32 blk
            1     16.20   11.10    9.58    12.02   us
            16    16.77   11.70   11.22    13.53
            72    20.57   14.80   13.83    15.88
            192   24.50   18.43   18.80    20.39

        so 16 launched blocks wins everywhere and 32 is 13-25% worse than that.
        Fitting those four columns to ``A + B*blocks + C*bytes_per_core`` gives
        B = 0.27 us per launched block against C = 1.8 us per head of weight
        traffic.  ``BLOCK_COST_HEADS`` is that B/C ratio: one launched block
        priced in head GEMMs.

        The other term is what a round costs beyond its head GEMMs: a core
        stages the T-tile's ``qr`` into L1 once per work item, so a round is
        ``base_m/d`` head GEMMs of weight traffic on top of ``heads_per_block`` of
        them (``qr`` is ``base_m`` rows of the same K extent that one head's
        ``wqb`` has ``d`` rows of).  Without that term the score prefers splits
        that shave a fraction off the round count and pay for it many times over
        in re-staged ``qr`` -- at 36 cores and T=2048 it would pick 32 blocks
        over 2, which measures 60.9 us against 54.3 us.

        With both terms the score reproduces the measured best block count at
        every (cores, T) pair that was swept: 28, 32 and 36 cores across
        T = 1, 16, 72, 192, 1024, 2048 and 4096.

        Lowering the per-block cost is the way to make finer splits pay.  The
        workspace reduction already took out the zero-fill and cut the barrier
        from four handshakes to two; ``BLOCK_COST_HEADS`` is still the fitted
        constant from before that, so it is now, if anything, pessimistic about
        fine splits.  Re-fit it if the reduction changes again.

        ``IPQW_HEAD_BLOCKS`` pins the count for A/B runs.
        """
        forced = 0 or HEAD_BLOCKS_K
        if not HEAD_SPLIT or forced == 1 or self.n_tiles >= SPLIT_T_TILES:
            self.n_head_blocks = 1
            self.heads_per_block = self.n_heads
            self.used_cores = self._grid(self.n_tiles)
            return

        if self.n_heads % forced:
            raise ValueError(f"head block count {forced} does not divide n_heads={self.n_heads}")
        self.n_head_blocks = forced
        self.heads_per_block = self.n_heads // self.n_head_blocks
        self.used_cores = self._grid(self.n_tiles * self.n_head_blocks)

    def _grid(self, n_work: int) -> int:
        """How many blocks to launch for ``n_work`` work items.

        The grid is a launch parameter, so unlike the work count itself it cannot
        be a runtime value -- which makes it a specialisation dimension, and a
        bad one: taking it as ``min(cores, n_work)`` gives a separate binary for
        every work count below the core count, seven of them in T=3073..3968
        alone, all doing the same thing.  Launching the whole chip instead costs
        only the prologue of the blocks that find no work item, since the work
        loop is strided by the grid and they fall straight out of it.
        """
        if GRID_FULL:
            return self.cube_cores
        return min(self.cube_cores, n_work)

    # The fields the kernel has to know at compile time, and what each one is.
    # Nothing else about the plan reaches the kernel as a constant: the counts
    # that move with T -- ``n_tiles``, ``n_work``, ``n_bands``, ``n_bands_ws``,
    # ``ws_rows`` -- are all derived from the live row count at runtime.  So this
    # tuple *is* the specialisation key: two T that agree on it share a binary.
    SHAPE_KEY = (
        "base_m",  # height of every L1/L0/UB tile
        "rows_vec",  # vector-side tile height
        "n_head_blocks",  # head loop trip count, k_block, band UB width
        "heads_per_block",
        "rows_red",  # band UB height
        "red_regs",  # register unroll in the band sum
        # ``used_cores`` is deliberately absent: the grid is a symbolic
        # expression over the padded row count now, not a constant, so it is
        # neither a specialisation dimension nor a route for the core count.
        "w_ws",  # whether the workspace path is compiled in at all
        "w_separate",
        "w_leader_only",
        "w_interleaved",
        "prefetch",
        "l2_q",
        "k_l1_w",
        "base_k_w",
        "k_l1_tiles_w",
        "qr_k_l1",
        "n_buf_wqb",
        "reduce_at",
    )

    def shape_key(self) -> tuple:
        """What this plan pins at compile time. Neither T nor the core count is.

        ``cube_cores`` used to be in here, and it had to be while the head split,
        the band height and the launch grid were all chosen against it.  All three
        are now pinned or runtime, so the same binary serves a 28-, 32- or 36-core
        part; the core count is a host-side quantity only.
        """
        return (self.dim, self.q_lora, self.n_heads, self.d, self.dr) + tuple(getattr(self, f) for f in self.SHAPE_KEY)

    def _check_t_dynamic(self) -> None:
        """Reject a dynamic-T plan that would still bake a T-derived count in.

        Two things have to hold, and neither is about which template this is.

        ``base_m`` must be the pinned tile height, because it is a buffer shape
        rather than a count -- a plan that sized it to T could not serve any
        other T.  ``__init__`` pins it under ``t_dyn``, so this is a guard on
        that, not a restriction on the caller.

        ``reduce_at`` must be "tail".  The "early" rendezvous sits inside the
        work loop and is only sound while every launched core runs that loop
        exactly once; whether that holds depends on the tile count, which is
        now a runtime value, so it cannot be decided here.
        """
        problems = []
        if self.base_m != BASE_M:
            problems.append(f"base_m={self.base_m} is sized to T (want {BASE_M})")
        if self.w_ws and self.reduce_at != "tail":
            problems.append(
                f"reduce_at={self.reduce_at!r} needs one work item per core, which is a runtime property under t_dyn"
            )
        if problems:
            raise ValueError(f"T cannot be dynamic for this plan (T={self.t}): " + "; ".join(problems))

    def _plan_reduce(self) -> None:
        """Band geometry for the AIV-side W reduction.

        A band is ``rows_red`` rows of ``w``.  One AIV pulls all
        ``n_head_blocks`` partials of its band into UB, one DMA each, and sums
        them register-wise, so a band has to be a whole number of vector
        registers wide.  Bands are handed out round-robin over every AIV on
        the chip, so the narrowest legal band spreads best; it is only grown
        when that would leave more than two bands per AIV.
        """
        if not self.w_ws:
            # No partials to hold, so ``ws`` is nothing but the grid marker.
            self.ws_rows = self.used_cores
            self.rows_red = 0
            self.n_bands = 0
            self.n_bands_ws = 0
            self.red_regs = 0
            self.reduce_at = "tail"
            return
        total_rows = self.n_tiles * self.base_m
        # Partials first, then the grid marker, so the partial indexing the
        # kernel does is untouched and the marker rows are never read.
        self.ws_partial_rows = self.n_head_blocks * total_rows
        self.ws_rows = self.ws_partial_rows + self.used_cores
        legal = [
            rows for rows in range(1, self.base_m + 1) if self.base_m % rows == 0 and (rows * self.n_heads) % VL == 0
        ]
        if not legal:
            raise NotImplementedError(
                f"no band height divides base_m={self.base_m} into whole {VL}-lane registers at N={self.n_heads}"
            )
        # Pinned rather than fitted to the AIV count: sizing it to
        # ``used_cores * SUBBLOCKS`` is a second way for the core count to reach
        # the binary, since this is a UB height and a register unroll.  See
        # ``ROWS_RED_K``.  Still checked against the legal set, because a band
        # that does not divide ``base_m`` into whole registers cannot be summed.
        if ROWS_RED_K not in legal:
            raise ValueError(
                f"ROWS_RED_K={ROWS_RED_K} is not a legal band height for "
                f"base_m={self.base_m} at N={self.n_heads}; legal: {legal}"
            )
        self.rows_red = ROWS_RED_K
        self.n_bands_ws = total_rows // self.rows_red
        # ``ws`` is indexed by tile, so it carries the padded tile height; only
        # the live rows of ``w`` are produced, so the band loop stops at T and
        # the last band may be short.
        self.n_bands = ceil_div(self.t, self.rows_red)
        self.red_regs = self.rows_red * self.n_heads // VL
        # The "early" rendezvous sits inside the work loop, which is only
        # sound while every launched core runs that loop exactly once -- a
        # core with two work items would arrive twice and one with none never.
        # Head splitting always lands on one item per core, so this holds
        # wherever w_ws is on, but it is cheap to keep honest.
        self.reduce_at = (
            W_REDUCE_AT if (not self.t_dyn and self.n_tiles * self.n_head_blocks == self.used_cores) else "tail"
        )

    def _pick_k_l1_w(self) -> int:
        """Largest <=256 K window dividing this core's slice of the reduction.

        The slice is ``k_block``: the whole ``dim`` for a leader, or
        ``dim / n_head_blocks`` under ``w_k_split``.  A leader additionally
        needs its window count to divide ``heads_per_block`` so the fills can
        interleave under the Q heads; dim=5120 with 32 heads gives 160, one
        window per head.  A K slice is retired in one go, so it only needs the
        fattest window that fits.
        """
        forced = 0
        if forced:
            return forced
        best = 0
        for k in range(32, 257, 32):
            if self.k_block % k:
                continue
            if self.w_k_split or (self.k_block // k) % self.heads_per_block == 0:
                best = k
        return best if best else 128

    def _pick_base_k_w(self) -> int:
        """L0 K step for the W path: as large as L0A allows next to the Q path."""
        q_l0a = D_L0AB_Q * self.base_m * (0 or min(128, ceil_div(self.q_lora, MX_K_ALIGN) * MX_K_ALIGN))
        budget = L0AB_BYTES - q_l0a
        best = 16
        for k in range(16, self.k_l1_w + 1, 16):
            if self.k_l1_w % k:
                continue
            if D_L0AB_W * self.base_m * k * 2 <= budget:
                best = k
        return best

    def row_split(self, rows, cols: int):
        """Row-balanced cube-tile -> vector-subblock tiler for a ``cols``-wide view."""
        return make_partition_tiler((rows, cols), (self.base_m, cols))

    def _pick_k_windows_q(self) -> int:
        """How many K windows the ``wqb`` reduction is filled in per head.

        See the D_WQB comment at module level for why a single-head core wants
        more, smaller windows.  A count is legal only if it cuts ``q_lora``
        into whole L0 K steps that are also whole MX groups; the requested
        count is otherwise walked down to the nearest legal one.
        Each window gets its own L1 buffer, so the count is also capped by the
        CUBE buf-id file.  ``IPQW_K_L1_Q`` still names the window and is
        converted here.
        """
        forced_window = 0
        want = (
            self.q_lora // forced_window
            if forced_window
            else (D_WQB_SINGLE_HEAD if self.heads_per_block == 1 else D_WQB)
        )
        max_tiles = (CUBE_BUF_IDS - FIXED_CUBE_SLOTS) // 2
        for tiles in range(min(want, max_tiles), 0, -1):
            window = self.q_lora // tiles
            if self.q_lora % tiles or window % self.base_k_q or window % MX_K_ALIGN:
                continue
            return tiles
        raise NotImplementedError(f"no legal wqb K window count for q_lora={self.q_lora} with base_k_q={self.base_k_q}")

    def _check(self) -> None:
        if self.d % VL or self.dr % VL:
            raise NotImplementedError(f"fused kernel needs D and Dr as multiples of {VL}, got D={self.d}, Dr={self.dr}")
        if self.rope_regs != 1:
            # Dr == 64 keeps rotate-half inside one register, so pass 1 can read
            # the pre-RoPE partner lanes from the register it is about to
            # overwrite.  Dr > 64 would need every rope register live at once
            # before any write-back; not implemented, and Dr=64 is the frozen
            # case (and what MLA uses).
            raise NotImplementedError(f"rotate-half is implemented for Dr == {VL} only, got Dr={self.dr}")
        if self.dr > self.d:
            raise ValueError(f"Dr={self.dr} exceeds D={self.d}")
        if self.q_lora % MX_K_ALIGN:
            raise NotImplementedError(f"q_lora must be a multiple of {MX_K_ALIGN}, got {self.q_lora}")
        if self.dim % self.k_l1_w:
            raise NotImplementedError(f"dim must be a multiple of {self.k_l1_w}, got {self.dim}")
        # ``wqb`` and ``ww`` arrive in FRACTAL_NZ and are copied to L1 by an
        # identity engine, which cannot pad a partial fractal the way nd2nz
        # does.  Every window this kernel cuts out of them therefore has to
        # land on the fractal grid: 16 rows, and C0 columns (32 for the
        # one-byte wqb, 16 for BF16 ww).
        for name, extent, align in (
            ("D (wqb NZ row tile)", self.d, NZ_M_FRAC),
            ("k_l1_q (wqb NZ C0)", self.k_l1_q, NZ_C0_1B),
            ("N (ww NZ row tile)", self.n_heads, NZ_M_FRAC),
            ("k_l1_w (ww NZ C0)", self.k_l1_w, NZ_C0_2B),
        ):
            if extent % align:
                raise NotImplementedError(f"FRACTAL_NZ inputs need {name} as a multiple of {align}, got {extent}")
        self.assert_budgets()

    def assert_budgets(self) -> None:
        l1 = (
            self.base_m * self.qr_k_l1  # qr resident
            + self.base_m * self.scale_k_len_q
            + self.n_buf_wqb * self.base_n_q * self.k_l1_q  # wqb, fully resident
            + self.n_buf_wqb * self.scale_k_l1_len_q * self.base_n_q
            + D_X * self.base_m * self.k_l1_w * 2  # x
            + D_X * self.base_n_w * self.k_l1_w * 2  # ww
        )
        l0a = D_L0AB_Q * self.base_m * self.base_k_q + D_L0AB_W * self.base_m * self.base_k_w * 2
        l0b = D_L0B_Q * self.base_n_q * self.base_k_q + D_L0AB_W * self.base_n_w * self.base_k_w * 2
        l0c = 4 * (D_L0C_Q * self.base_m * self.base_n_q + 2 * self.base_m * self.base_n_w)
        ub = (
            D_CV * self.rows_vec * self.d * 4  # cv_ub
            + D_QUB * self.rows_vec * (self.d // 2)  # q_ub
            + D_QUB * self.rows_vec * self.groups  # descale
            + self.amax_len * 8  # amax fp32 + off int32
            + LUT_LEN * 4
            + 2 * self.rows_vec * max(self.dr, 1) * 4  # cos + sin
            # one band's partials plus the summed band
            + (self.n_head_blocks + 1) * self.rows_red * self.n_heads * 4
        )
        for name, used, cap in (
            ("L1", l1, L1_BYTES),
            ("L0A", l0a, L0AB_BYTES),
            ("L0B", l0b, L0AB_BYTES),
            ("L0C", l0c, L0C_BYTES),
            ("UB", ub, UB_BYTES),
        ):
            if used > cap:
                raise ValueError(f"{name} budget exceeded: {used} > {cap} bytes; retune Plan")
        self.bytes_used = {"L1": l1, "L0A": l0a, "L0B": l0b, "L0C": l0c, "UB": ub}


def plan(
    t: int,
    dim: int,
    q_lora: int,
    n_heads: int,
    d: int,
    dr: int,
    cube_cores: int | None = None,
) -> Plan:
    """The plan that will actually serve this T.

    Goes through the same cache the host entry does, so a caller sizing its
    buffers off ``base_m`` gets the tile height the kernel will really use.
    Building a fresh ``Plan`` here would size the tile to T and hand back a
    smaller height than the binary serving that T was compiled with.
    """
    return _plan_for(t, dim, q_lora, n_heads, d, dr, cube_cores).p


# ---------------------------------------------------------------------------
# Cross-core rendezvous for the W reduction
# ---------------------------------------------------------------------------


@jit
def store_to_vec_barrier():
    """All-core rendezvous for "every AIC's W partial is in GM".

    ``global_sync_all`` emits four handshakes, two of which serve
    dependencies this barrier does not have.  Its phase 1 funnels each
    block's AIVs into their own AIC, which matters when an AIV produced
    something the grid has to see; here the producers are the AICs and the
    AIVs are the consumers.  Its phase 3 is an all-AIV meeting point its own
    docstring calls redundant on a mix kernel.  What is left is the grid
    barrier over the AICs, then the handoff to each block's own AIVs.

    Flag ids come from the same arena allocator ``global_sync_all`` draws
    from, so they cannot collide with a Channel's or another barrier's.

    Worth 0.75 us of the 1.77 us the full barrier costs at T=72.  The arrive
    goes out on FIXPIPE and is preceded by ``cube_sync_all`` because an FFTS
    arrive drains only the pipe it is issued on: the partial is a fixpipe
    store, and a peer must not see the flag before the store lands.
    """
    grid_flag, funnel_flag = _current_channel_arena().alloc_flag_ids(2)
    cube_sync_all()
    cube_sync_block_arrive(PIPE.FIXPIPE, grid_flag, mode=0)
    cube_sync_block_wait(PIPE.S, grid_flag, mode=0)
    cube_sync_block_arrive(PIPE.MTE3, funnel_flag, mode=2)
    vec_sync_block_wait(PIPE.S, funnel_flag)


@jit
def _prefetch_qr(cube_q, qr_gm, descale_qr_gm, p, work, tile, n_work):
    """Stage the next work item's ``qr`` so it rides under this tile's epilogue.

    It sits after the head loop rather than inside its last iteration, where it
    used to: the staging channels are depth 1 and the read transaction now spans
    the whole head loop, so a fill issued before the release would block on its
    own reader.  On the cube's own timeline that moves it past one
    ``drain_to_vec``, which is a fixpipe op the MTE2 fill does not wait for.
    """
    if const_expr(p.prefetch):
        next_work = work + get_block_num()
        if next_work < n_work:
            next_tile = next_work // p.n_head_blocks
            if next_tile != tile:
                cube_q.load_qr(qr_gm, descale_qr_gm, next_tile)


@jit
def w_barrier():
    if const_expr(W_SYNC == "lean"):
        store_to_vec_barrier()
    else:
        global_sync_all()


# ---------------------------------------------------------------------------
# Cube modules
# ---------------------------------------------------------------------------


class CubeQ:
    """MXFP8 ``qr @ wqb^T`` for one (T tile, head); owns its L1/L0 staging."""

    def __init__(self, p: Plan):
        self.p = p
        self.qr_l1 = Channel(MemLoc.L1, (p.base_m, p.qr_k_l1), Float8E4M3FN, depth=1, data_format="nz")
        self.scale_qr_l1 = Channel(MemLoc.L1, (p.base_m, p.scale_k_len_q), Float8E8M0, depth=1, data_format="zn")
        # ``wqb`` is handed over already in FRACTAL_NZ.  The GM-to-L1 identity
        # copy is a raw byte move and insists both sides share a dtype, and the
        # one-byte elements FRACTAL_NZ can describe are only the integers -- so
        # the staging channel is declared uint8 for the fill and read back
        # through an fp8 alias.
        #
        # One depth-1 channel per K window rather than one channel of that
        # depth, because the fp8 view has to be taken on the slot (see
        # gemm_head) and a slot only has a statically known byte origin when
        # its channel is depth 1.
        self.wqb_l1 = [
            Channel(
                MemLoc.L1,
                (p.base_n_q, p.k_l1_q),
                U8,
                depth=1,
                data_format="nz",
            )
            for _ in range(p.n_buf_wqb)  # == k_l1_tiles_q
        ]
        self.wqb_l1_fp8 = [ch.reinterpret(Float8E4M3FN) for ch in self.wqb_l1]
        # No alias on the scales, so one rotating channel is fine here.
        self.scale_wqb_l1 = Channel(
            MemLoc.L1,
            (p.scale_k_l1_len_q, p.base_n_q),
            Float8E8M0,
            depth=p.n_buf_wqb,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (p.base_m, p.base_k_q), Float8E4M3FN, depth=D_L0AB_Q)
        self.l0b = Channel(MemLoc.L0B, (p.base_n_q, p.base_k_q), Float8E4M3FN, depth=D_L0B_Q)
        self.l0c = Channel(MemLoc.L0C, (p.base_m, p.base_n_q), F32, depth=D_L0C_Q)

        self.eng_nd2nz = make_copy_engine(format_transform="nd2nz", dtype=Float8E4M3FN, pad_value=0.0)
        # ``wqb`` arrives in FRACTAL_NZ, so its L1 fill is a plain burst copy.
        # The ND route had to gather base_n_q separate q_lora-strided rows per
        # window; the NZ route reads whole (16, 32) fractals back to back,
        # which is what makes this the cheapest 5.24 MB on the chip.
        self.eng_nz = make_copy_engine(format_transform="identity", dtype=U8)
        self.eng_scale_a = make_copy_engine(format_transform="mx_scale_and", dtype=Float8E8M0, pad_value=0.0)
        self.eng_scale_b = make_copy_engine(format_transform="mx_scale_bdn", dtype=Float8E8M0, pad_value=0.0)
        # split-M L0C -> UB handoff requires dual_dst_ctl=1 (API note #16).
        if UNIT_FLAG:
            # Lets the fixpipe drain start on the last accumulating mmad instead
            # of after a full L0C sync, which shortens the cube -> vector chain.
            self.eng_c2v = make_copy_engine(dtype=F32, dual_dst_ctl=1, unit_flag_mode=3)
        else:
            self.eng_c2v = make_copy_engine(dtype=F32, dual_dst_ctl=1)

    @jit
    def load_qr(self, qr_gm, descale_qr_gm, tile):
        """Stage one T-tile's ``qr`` and its scales into L1.

        The GM side is the tile view's own extent, so a short last tile reads
        only its live rows and the rest of the L1 tile keeps whatever the last
        tile through this buffer left there.  The mmads reduce those rows and
        the epilogue's store is cut to the live ones (see ``VectorQ.store``), so
        nothing reads what they produced.

        The transaction is manual only so that the read side can name the row
        height -- see ``rows_view``.
        """
        p = self.p
        qr_slot = self.qr_l1.acquire()
        mem_copy(
            qr_slot,
            tile_view(qr_gm, (p.base_m, p.qr_k_l1), (tile, 0)),
            engine=self.eng_nd2nz,
            l2_cache_ctl=p.l2_q,
        )
        self.qr_l1.commit(qr_slot)
        scale_slot = self.scale_qr_l1.acquire()
        mem_copy(
            scale_slot,
            tile_view(
                descale_qr_gm,
                (p.base_m, p.scale_k_len_q // MX_PAIR, MX_PAIR),
                (tile, 0, 0),
            ),
            engine=self.eng_scale_a,
            l2_cache_ctl=p.l2_q,
        )
        self.scale_qr_l1.commit(scale_slot)

    @jit
    def wait_qr(self):
        """The staged ``qr`` tile and its scales, as one transaction pair."""
        return self.qr_l1.wait(), self.scale_qr_l1.wait()

    @jit
    def release_qr(self, qr_slot, scale_slot):
        self.qr_l1.release(qr_slot)
        self.scale_qr_l1.release(scale_slot)

    @jit
    def rows_view(self, qr_slot, scale_slot, rows_l1):
        """The staged tile cut to the row height its fill actually wrote.

        A GM-to-L1 nd2nz copy of a short tile programs its NZ row stride as
        ``ceil(rows/16)*16``, so a read issued at the declared ``base_m`` walks
        at ``base_m/16`` fractals and lands on the wrong rows for every K window
        but the first -- wrong values, not a short read.  T=8229 read back 1955
        wrong ``descale_q`` bytes that way and T=4296 read back 3645; it needs a
        T past the prefetch threshold that is not a multiple of 128, and every
        prefill T in the original shape list is a multiple of 1024, which is why
        it went unnoticed.

        The layout pass infers exactly this height on its own, but only while
        one unconditional fill reaches the reads.  The prefetch adds a second
        fill site behind a runtime ``if``, the two runtime extents join to the
        declared height, and an explicit view is rejected on an implicit channel
        ("cannot project typed Channel reaching shape through explicit tile_view
        read") -- hence the manual transaction, which hands back a plain slot
        this can slice.  Measured: with the fill unconditional the inference is
        right at every T; with the prefetch on it is wrong at every T that is
        not a multiple of 128.

        The scales are cut to the same height for their row step only.  Their
        fill is ``mx_scale_and``, which does not pack to the tail, so their row
        stride is the declared one either way.
        """
        p = self.p
        qr = local_slice(
            qr_slot,
            make_bounded_tiler((rows_l1, p.qr_k_l1), (p.base_m, p.qr_k_l1), (NZ_M_FRAC, 1)),
            stride=(p.qr_k_l1, 1),
        )
        scale = local_slice(
            scale_slot,
            make_bounded_tiler(
                (rows_l1, p.scale_k_len_q),
                (p.base_m, p.scale_k_len_q),
                (NZ_M_FRAC, 1),
            ),
            stride=(p.scale_k_len_q, 1),
        )
        return qr, scale

    @jit
    def _fill_wqb(self, wqb_gm, descale_wqb_gm, head, kw, buf):
        """Stage K window ``kw`` of this head's ``wqb`` and its scales into L1."""
        p = self.p
        groups = p.scale_k_l1_len_q // MX_PAIR
        # Filled through the uint8 face, read back through the fp8 alias in
        # gemm_head.  Both halves are written out by hand so the fill and the
        # read name the same depth-1 buffer.
        wqb_slot = self.wqb_l1[buf].acquire()
        mem_copy(
            wqb_slot,
            tile_view(wqb_gm, (p.base_n_q, p.k_l1_q), (head, kw)),
            engine=self.eng_nz,
            l2_cache_ctl=p.l2_q,
        )
        self.wqb_l1[buf].commit(wqb_slot)
        scale_slot = self.scale_wqb_l1.acquire()
        mem_copy(
            scale_slot,
            tile_view(descale_wqb_gm, (p.base_n_q, groups, MX_PAIR), (head, kw, 0)),
            engine=self.eng_scale_b,
            l2_cache_ctl=p.l2_q,
        )
        self.scale_wqb_l1.commit(scale_slot)

    @jit
    def gemm_head(self, wqb_gm, descale_wqb_gm, head, qr_rows, scale_rows):
        """Accumulate the whole q_lora reduction for one head into L0C.

        Window kw+1 is staged before window kw is consumed, so ``wqb_l1`` keeps
        MTE2 busy underneath the mmads instead of showing each window's burst
        up as a cube stall.
        """
        p = self.p
        self._fill_wqb(wqb_gm, descale_wqb_gm, head, 0, 0)
        # Unrolled: keeps ``init`` a compile-time bool, so the hottest loop in
        # the kernel has no scf.if around its mmad.
        for kw in range_constexpr(p.k_l1_tiles_q):
            if const_expr(kw + 1 < p.k_l1_tiles_q):
                self._fill_wqb(wqb_gm, descale_wqb_gm, head, kw + 1, (kw + 1) % p.n_buf_wqb)
            buf = kw % p.n_buf_wqb
            wqb_ready = self.wqb_l1_fp8[buf].wait()
            scale_ready = self.scale_wqb_l1.wait()
            for kk in range_constexpr(p.step_k_q):
                k_l0 = kw * p.step_k_q + kk
                mem_copy(
                    self.l0a,
                    tile_view(qr_rows, (p.base_m, p.base_k_q), (0, k_l0)),
                    mx_scale=tile_view(scale_rows, (p.base_m, p.scale_k_l0_len), (0, k_l0)),
                )
                mem_copy(
                    self.l0b,
                    tile_view(wqb_ready, (p.base_n_q, p.base_k_q), (0, kk)),
                    mx_scale=tile_view(scale_ready, (p.scale_k_l0_len, p.base_n_q), (kk, 0)),
                )
                if const_expr(UNIT_FLAG):
                    last_k = k_l0 == p.k_l0_tiles_q - 1
                    matmul(
                        self.l0c,
                        self.l0a,
                        self.l0b,
                        init=(k_l0 == 0),
                        unit_flag=3 if last_k else 2,
                    )
                else:
                    matmul(self.l0c, self.l0a, self.l0b, init=(k_l0 == 0))
            self.wqb_l1_fp8[buf].release(wqb_ready)
            self.scale_wqb_l1.release(scale_ready)

    @jit
    def drain_to_vec(self, cv_ub, split_m):
        mem_copy(cv_ub, self.l0c, engine=self.eng_c2v, partition=split_m)


class CubeW:
    """BF16 ``x @ ww^T`` scaled in fixpipe; writes GM directly, no vector hop."""

    def __init__(self, p: Plan):
        self.p = p
        bf16 = dtypes.bfloat16
        self.x_l1 = Channel(MemLoc.L1, (p.base_m, p.k_l1_w), bf16, depth=D_X, data_format="nz")
        self.ww_l1 = Channel(MemLoc.L1, (p.base_n_w, p.k_l1_w), bf16, depth=D_X, data_format="nz")
        self.l0a = Channel(MemLoc.L0A, (p.base_m, p.base_k_w), bf16, depth=D_L0AB_W)
        self.l0b = Channel(MemLoc.L0B, (p.base_n_w, p.base_k_w), bf16, depth=D_L0AB_W)
        self.l0c = Channel(MemLoc.L0C, (p.base_m, p.base_n_w), F32, depth=2)
        self.eng_nd2nz = make_copy_engine(format_transform="nd2nz", dtype=bf16, pad_value=0.0)
        # ``ww`` arrives in FRACTAL_NZ; ``x`` is an activation and stays ND.
        self.eng_nz = make_copy_engine(format_transform="identity", dtype=bf16)
        self.eng_fp = make_copy_engine(dtype=F32)

    @jit
    def x_rows(self, slot, rows_l1):
        """This tile's ``x`` in L1, at the row height its fill actually wrote.

        The GM-to-L1 nd2nz copy of a short last tile writes the live rows and
        rounds the fractal grid up: the NZ row stride it programs is
        ``ceil(rows/16)*16``, and the rows the rounding adds are the engine's
        pad value.  A read has to be issued at that same height or its own
        fractal stride is the declared ``base_m/16``, which walks past every K
        window but the first -- which is how a short tail used to corrupt the
        upper half of the head axis and why ``x`` used to be handed over grown
        to whole tiles in GM.

        The height cannot be the channel's declared shape, because only the last
        tile is short, and it cannot be inferred either: a manual acquire/wait
        slot is a fresh root, so the fill's row count never reaches the read.
        (The ``qr`` staging is an implicit channel, where the layout pass does
        propagate it; see ``CubeQ.load_qr``.)  So it is passed in, and this is
        the same ``local_slice`` a dynamic-shape Channel would apply itself.
        """
        p = self.p
        return local_slice(
            slot,
            make_bounded_tiler((rows_l1, p.k_l1_w), (p.base_m, p.k_l1_w), (NZ_M_FRAC, 1)),
            stride=(p.k_l1_w, 1),
        )

    @jit
    def c_rows(self, slot, rows_l1):
        """This tile's L0C, at the row height the mmads wrote it at.

        An mmad lays C out as ``ceil(N/16)`` blocks of (M, 16), so its M is the
        stride between one head fractal and the next.  ``x_rows`` makes that M
        the rounded-up row count of the tile, which leaves the fixpipe store
        reading at the declared ``base_m`` stride and picking up the wrong half
        of the head axis -- the exact corruption the GM padding was hiding.
        """
        p = self.p
        return local_slice(
            slot,
            make_bounded_tiler((rows_l1, p.base_n_w), (p.base_m, p.base_n_w), (NZ_M_FRAC, 1)),
            stride=(p.base_n_w, 1),
        )

    @jit
    def gemm_head_slice(self, x_gm, ww_gm, tile, local_head, h_block, l0c_slot, rows_l1):
        """Retire this local head's share of the K reduction into ``l0c_slot``.

        Called from inside the Q head loop so the ``x`` traffic trickles in on
        MTE2 underneath the Q path's cube work.  ``local_head`` is 0 ..
        heads_per_block-1 on this core; ``h_block`` selects which slice of
        ``ww`` / ``w`` this core owns.  ``local_head`` is a runtime index,
        hence the manual L0C transaction: with an implicit channel the
        ``scf.if`` around the ``init`` matmul reads as a second writer.
        """
        p = self.p
        self._fill_window(x_gm, ww_gm, tile, h_block, local_head * p.w_per_head)
        for j in range_constexpr(p.w_per_head):
            if const_expr(j + 1 < p.w_per_head):
                self._fill_window(x_gm, ww_gm, tile, h_block, local_head * p.w_per_head + j + 1)
            x_ready = self.x_l1.wait()
            ww_ready = self.ww_l1.wait()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a, tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                first = kk == 0 and j == 0
                if const_expr(PROBE == "no_runtime_init"):
                    matmul(l0c_slot, self.l0a, self.l0b, init=False)
                elif const_expr(PROBE == "no_w_mmad"):
                    pass
                else:
                    matmul(
                        l0c_slot,
                        self.l0a,
                        self.l0b,
                        init=(local_head == 0) if const_expr(first) else False,
                    )
            self.x_l1.release(x_ready)
            self.ww_l1.release(ww_ready)

    @jit
    def _fill_window(self, x_gm, ww_gm, tile, k_base, kw):
        """Stage K window ``k_base + kw`` of ``x`` and ``ww`` into L1.

        ``k_base`` is this core's offset into the reduction, in windows: zero
        for a leader that owns the whole ``dim``, ``h_block * k_l1_tiles_w``
        under ``w_k_split``.
        """
        p = self.p
        x_slot = self.x_l1.acquire()
        mem_copy(
            x_slot,
            tile_view(x_gm, (p.base_m, p.k_l1_w), (tile, k_base + kw)),
            engine=self.eng_nd2nz,
            l2_cache_ctl=L2_W,
        )
        self.x_l1.commit(x_slot)
        ww_slot = self.ww_l1.acquire()
        mem_copy(
            ww_slot,
            tile_view(ww_gm, (p.base_n_w, p.k_l1_w), (0, k_base + kw)),
            engine=self.eng_nz,
            l2_cache_ctl=L2_W,
        )
        self.ww_l1.commit(ww_slot)

    @jit
    def fill(self, x_gm, ww_gm, tile, local_head, h_block):
        """Issue this local head's W-path L1 fills; does not wait for them."""
        p = self.p
        for j in range_constexpr(p.w_per_head):
            self._fill_window(x_gm, ww_gm, tile, h_block, local_head * p.w_per_head + j)

    @jit
    def drain(self, local_head, l0c_slot, rows_l1):
        """Retire the oldest staged window pair into ``l0c_slot``."""
        p = self.p
        for j in range_constexpr(p.w_per_head):
            x_ready = self.x_l1.wait()
            ww_ready = self.ww_l1.wait()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a, tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                first = kk == 0 and j == 0
                matmul(
                    l0c_slot,
                    self.l0a,
                    self.l0b,
                    init=(local_head == 0) if const_expr(first) else False,
                )
            self.x_l1.release(x_ready)
            self.ww_l1.release(ww_ready)

    @jit
    def gemm_full(self, x_gm, ww_gm, tile, k_base, l0c_slot, rows_l1):
        """This core's whole K slice as one reduction; used when Q cannot hide it."""
        p = self.p
        self._fill_window(x_gm, ww_gm, tile, k_base, 0)
        for kw in range_constexpr(p.k_l1_tiles_w):
            if const_expr(kw + 1 < p.k_l1_tiles_w):
                self._fill_window(x_gm, ww_gm, tile, k_base, kw + 1)
            x_ready = self.x_l1.wait()
            ww_ready = self.ww_l1.wait()
            x_rows = self.x_rows(x_ready, rows_l1)
            for kk in range_constexpr(p.step_k_w):
                mem_copy(self.l0a, tile_view(x_rows, (p.base_m, p.base_k_w), (0, kk)))
                mem_copy(self.l0b, tile_view(ww_ready, (p.base_n_w, p.base_k_w), (0, kk)))
                first = kw == 0 and kk == 0
                matmul(l0c_slot, self.l0a, self.l0b, init=const_expr(first))
            self.x_l1.release(x_ready)
            self.ww_l1.release(ww_ready)

    @jit
    def store(self, w_gm, tile, softmax_scale, l0c_slot, rows_l1):
        """Whole-reduction store, straight into ``w``. Not used under w_ws."""
        # softmax_scale folded into the fixpipe DEQSCALE register: the W path
        # never touches the vector unit.
        p = self.p
        self.l0c.commit(l0c_slot)
        ready = self.l0c.wait()
        mem_copy(
            tile_view(w_gm, (p.base_m, p.base_n_w), (tile, 0)),
            self.c_rows(ready, rows_l1),
            engine=self.eng_fp,
            deq_scale_val=softmax_scale,
            l2_cache_ctl=L2_W,
        )
        self.l0c.release(ready)

    @jit
    def store_partial(self, ws_gm, tile, h_block, n_tiles, softmax_scale, l0c_slot, rows_l1):
        """Write this core's K-slice partial to the rows it alone owns.

        A plain store, not an atomic add: nothing else writes these rows, so
        they need no pre-zeroing and no barrier ahead of the store.  Scaling
        each partial by ``softmax_scale`` in the fixpipe is equivalent to
        scaling the sum because the scale is a constant.

        ``ws`` is allocated on the padded tile grid, so the destination is
        always full height while the source is only ``rows_l1`` tall on a short
        last tile; the rows past that are left holding whatever the previous
        launch put there.  The reduction below drops them because its own output
        view is cut to the live rows of ``w``.
        """
        p = self.p
        self.l0c.commit(l0c_slot)
        ready = self.l0c.wait()
        mem_copy(
            tile_view(ws_gm, (p.base_m, p.base_n_w), (h_block * n_tiles + tile, 0)),
            self.c_rows(ready, rows_l1),
            engine=self.eng_fp,
            deq_scale_val=softmax_scale,
            l2_cache_ctl=L2_W,
        )
        self.l0c.release(ready)


# ---------------------------------------------------------------------------
# Vector module
# ---------------------------------------------------------------------------


class VectorQ:
    """Inplace RoPE plus MXFP4 quantisation of one head tile, raw VF only."""

    def __init__(self, p: Plan):
        self.p = p
        rows = p.rows_vec
        # D is a multiple of 64, so q_ub's rows are already 32-byte aligned and
        # its declared stride is its real stride.  dsc_ub's natural shape
        # (rows, groups) is not: UB rounds the innermost axis up to 32 bytes, so
        # a 4-byte row would really be 32 bytes apart and the linear offsets
        # pass 2 writes at would land in the padding.  Declare it in the shape
        # pass 2 actually writes (one 64-byte register per row) and let ``store``
        # re-view it as (rows, groups) with an explicit packed stride.
        self.q_ub = Channel(MemLoc.UB, (rows, p.d // 2), U8, depth=D_QUB)
        self.dsc_ub = Channel(MemLoc.UB, (p.enc_regs, VL), U8, depth=D_QUB)
        # Buffer vs Channel is a synchronisation decision, not a sizing one.
        # amax/off are written and read only by this vector core, so a plain
        # Buffer plus vmem_bar orders them.  The LUT and the cos/sin tables
        # arrive by MTE2 DMA and are then read by VF: that crosses pipes, and a
        # Buffer carries no dependency, so the VF read can beat the DMA.  As
        # Channels (depth=1, one fill per T-tile) the handoff is tracked.
        self.amax_ub = Buffer(MemLoc.UB, (p.amax_len,), F32)
        self.off_ub = Buffer(MemLoc.UB, (p.amax_len,), I32)
        self.lut_ub = Channel(MemLoc.UB, (LUT_LEN,), I32, depth=1)
        self.cos_ub = Channel(MemLoc.UB, (rows, max(p.dr, 1)), F32, depth=1)
        self.sin_ub = Channel(MemLoc.UB, (rows, max(p.dr, 1)), F32, depth=1)
        # W reduction staging.  ``band_ub`` holds every partial of one band so
        # the sum is whole-register adds; partial ``i`` starts exactly
        # ``red_regs`` registers in, because a band's innermost extent
        # (n_heads fp32 = 128 B) is already 32-byte aligned and so the
        # declared stride is the real one.  Plain Buffers: the DMA-to-VF and
        # VF-to-DMA handoffs are both ordered by vec_sync_all, which is
        # coarse but runs at most twice per band.
        if p.w_ws:
            self.band_ub = Buffer(MemLoc.UB, (p.n_head_blocks * p.rows_red, p.n_heads), F32)
            self.wsum_ub = Buffer(MemLoc.UB, (p.rows_red, p.n_heads), F32)

    @jit
    def reduce_all(self, ws_gm, w_gm, n_bands, n_bands_ws):
        """Sum the bands this AIV owns, round-robin over every AIV on chip.

        ``get_subblock_dim()`` is 2 on the vector core and 1 on the cube, so
        the cube side walks a different, and empty, band range.  Both counts are
        runtime under ``t_dyn``: a band is a fixed ``rows_red`` rows, so how many
        there are is the only thing T changes here.
        """
        aiv = get_block_idx() * get_subblock_dim() + get_subblock_id()
        for band in range(aiv, n_bands, get_block_num() * get_subblock_dim()):
            self.reduce_w(ws_gm, w_gm, band, n_bands_ws)

    @jit
    def reduce_w(self, ws_gm, w_gm, band, n_bands_ws):
        """Sum every partial of one band out of UB and write it to ``w``."""
        p = self.p
        for part in range_constexpr(p.n_head_blocks):
            mem_copy(
                local_slice(
                    self.band_ub,
                    (p.rows_red, p.n_heads),
                    offset=part * p.red_regs * VL * 4,
                ),
                tile_view(
                    ws_gm,
                    (p.rows_red, p.n_heads),
                    (part * n_bands_ws + band, 0),
                ),
            )
        vec_sync_all()
        with vf(mode="raw"):
            m32 = full_mask()
            for reg in range_constexpr(p.red_regs):
                acc = vload(self.band_ub, reg * VL)
                for part in range_constexpr(1, p.n_head_blocks):
                    acc = vadd(
                        acc,
                        vload(self.band_ub, (part * p.red_regs + reg) * VL),
                        mask=m32,
                    )
                vstore(self.wsum_ub, reg * VL, acc, m32)
        vec_sync_all()
        out = tile_view(w_gm, (p.rows_red, p.n_heads), (band, 0))
        mem_copy(out, local_slice(self.wsum_ub, (out.shape[0], p.n_heads)))

    @jit
    def load_tile_constants(self, lut_gm, cos_half, sin_half):
        """Per-T-tile prologue: DMA the LUT and the cos/sin tables into UB."""
        mem_copy(self.lut_ub, tile_view(lut_gm, (LUT_LEN,), (0,)))
        if const_expr(self.p.dr > 0):
            mem_copy(self.cos_ub, cos_half)
            mem_copy(self.sin_ub, sin_half)

    @jit
    def epilogue(self, cv_ub, rows):
        """RoPE then MXFP4 quantise ``cv_ub`` into ``q_ub`` / ``dsc_ub``."""
        p = self.p
        with vf(mode="raw"):
            m32 = full_mask()
            m_lo = create_mask("h", 32)
            lane = varange(0, I32)

            zero_i = vdups(0, I32, mask=m32)
            one_i = vdups(1, I32, mask=m32)
            k8 = vdups(8, I32, mask=m32)
            k_bias = vdups(E8M0_BIAS, I32, mask=m32)
            k_lane_wrap = vdups(VL - 1, I32, mask=m32)
            mant_mask = vdups(FP32_MANTISSA_MASK, I32, mask=m32)
            mag_mask = vdups(FP32_MAG_MASK, I32, mask=m32)

            # lane i -> (i + VL/2) % VL, the intra-register rotate-half permute
            rot_idx = vreinterpret(vbitwise_and(vadds(lane, VL // 2, mask=m32), k_lane_wrap, mask=m32), U32)
            # Existing V4.1 caches use adjacent real/imaginary pairs.
            # Keep rot_idx above for the two 32-element scale reductions;
            # only the RoPE permutation changes to adjacent partners.
            odd_rope = vne(vbitwise_and(lane, one_i, mask=m32), zero_i, mask=m32)
            rope_idx = vreinterpret(
                vselect(
                    vsub(lane, one_i, mask=m32),
                    vadd(lane, one_i, mask=m32),
                    cond_mask=odd_rope,
                ),
                U32,
            )
            rope_sgn = vselect(
                vdups(1.0, F32, mask=m32),
                vdups(-1.0, F32, mask=m32),
                cond_mask=odd_rope,
            )
            # 1.0 on even lanes, 16.0 on odd lanes: pair-sum then packs two
            # nibbles.  f32 because asc_pair_reduce_sum only exists for f16/f32
            # -- there is no integer pair-reduce.  Nibbles are 0..15 and the
            # products 0..240, all exactly representable, so this is not a
            # precision compromise.
            odd = vne(vbitwise_and(lane, one_i, mask=m32), zero_i, mask=m32)
            pack_w = vselect(vdups(16.0, F32, mask=m32), vdups(1.0, F32, mask=m32), cond_mask=odd)
            # lanes 0..31 -> group g, lanes 32..63 -> group g+1
            gsel = vshr(lane, 5, mask=m32)

            # ---- pass 1: inplace RoPE, then |max| per 32-element MX group ----
            for r in range(rows):
                base = r * p.d
                for i in range_constexpr(p.n_reg):
                    x = vload(cv_ub, base + i * VL)
                    # Match the BF16 materialization in the unfused projection/RoPE.
                    bits = vreinterpret(x, I32)
                    tie = vbitwise_and(vshr(bits, 16, mask=m32), one_i, mask=m32)
                    bits = vadd(vadds(bits, 32767, mask=m32), tie, mask=m32)
                    x = vreinterpret(vbitwise_and(bits, vdups(-65536, I32, mask=m32), mask=m32), F32)
                    pe = i - (p.n_reg - p.rope_regs)
                    if const_expr(p.rope_regs > 0 and pe >= 0):
                        cosv = vload(self.cos_ub, r * p.dr + pe * VL)
                        sinv = vload(self.sin_ub, r * p.dr + pe * VL)
                        # Adjacent-pair RoPE without any cache permutation.
                        rot = vmul(
                            vreinterpret(vgather_reg(vreinterpret(x, I32), rope_idx), F32),
                            rope_sgn,
                            mask=m32,
                        )
                        # Match separate FP32 multiply/add, then BF16 output.
                        x = vadd(vmul(rot, sinv, mask=m32), vmul(x, cosv, mask=m32), mask=m32)
                        # Match the BF16 materialization in the unfused projection/RoPE.
                        bits = vreinterpret(x, I32)
                        tie = vbitwise_and(vshr(bits, 16, mask=m32), one_i, mask=m32)
                        bits = vadd(vadds(bits, 32767, mask=m32), tie, mask=m32)
                        x = vreinterpret(vbitwise_and(bits, vdups(-65536, I32, mask=m32), mask=m32), F32)
                    vstore(cv_ub, base + i * VL, x, m32)
                    a = vabs(x, mask=m32)
                    # Both MX groups reduce under the same "lowest half" pset:
                    # the upper group is rotated down first (rot_idx is already
                    # live for the RoPE) so neither reduction depends on where a
                    # masked reduce parks its result for a high lane range.
                    a_hi = vreinterpret(vgather_reg(vreinterpret(a, I32), rot_idx), F32)
                    slot = r * p.groups + 2 * i
                    vstore_first(self.amax_ub, slot, vreduce_max(a, mask=m_lo))
                    vstore_first(self.amax_ub, slot + 1, vreduce_max(a_hi, mask=m_lo))

            # Pass 2 reads back the amax values pass 1 just wrote, and pass 3
            # reads back the RoPE result stored into cv_ub.  Inside one raw VF
            # region a UB store is not ordered against a later UB load, so
            # without this the last few rows of each subblock read stale data
            # (which is exactly how this showed up on hardware: only the rows at
            # the end of an AIV's range, and only for the first head).
            vmem_bar("vst_vld")

            # ---- pass 2: E8M0 scales for 64 groups at a time ----
            #
            # The golden takes ceil(log2(amax/6)) as
            # ``exp(fl(amax/6)) + (mant(fl(amax/6)) != 0)``.  Reproducing that
            # through a vector divide would need bit-exact IEEE division, which
            # the VPU reciprocal does not give, so use the algebraic identity
            # instead: with amax = 2^E * m, m in [1,2), amax/6 = 2^(E-3)*(m*4/3)
            # and m*4/3 in [4/3, 8/3), hence
            #     ceil(log2(amax/6)) = E - 2 + (m > 1.5).
            # Exact for every normal fp32 amax (checked against the golden over
            # the mantissa==0x400000 boundary and random bit patterns).
            for e in range(p.enc_regs):
                ab = vreinterpret(vload(self.amax_ub, e * VL), I32)
                exp_bits = vshr(ab, FP32_EXP_SHIFT, mask=m32)
                mant = vbitwise_and(ab, mant_mask, mask=m32)
                bump = vselect(one_i, zero_i, cond_mask=vgts(mant, FP32_HALF_MANTISSA, mask=m32))
                er = vadd(vadds(exp_bits, -2, mask=m32), bump, mask=m32)
                # amax < 1.5*2^-125 makes the golden's amax/6.0 land on an fp32
                # subnormal, where the identity above no longer holds; the golden
                # always yields 1 there, and er >= 1 is legal for every other
                # amax, so a clamp is enough.
                er = vmaxs(er, 1, mask=m32)
                # amax == 0 -> golden uses scale 1.0, i.e. the E8M0 bias.
                er = vselect(er, k_bias, cond_mask=vne(ab, zero_i, mask=m32))
                vstore(
                    self.off_ub,
                    e * VL,
                    vshl(vsub(er, k_bias, mask=m32), 3, mask=m32),
                    m32,
                )
                vstore_pack(self.dsc_ub, e * VL, er, m32, pack_mode=PackMode.B32_TO_B8)

            # Pass 3 gathers the per-group offsets pass 2 just wrote.
            vmem_bar("vst_vld")

            # ---- pass 3: descale, round to E2M1, pack two nibbles per byte ----
            for r in range(rows):
                base = r * p.d
                gbase = vadds(gsel, r * p.groups, mask=m32)
                for i in range_constexpr(p.n_reg):
                    off = vgather(
                        self.off_ub,
                        vreinterpret(vadds(gbase, 2 * i, mask=m32), U32),
                        mask=m32,
                    )
                    xv = vload(cv_ub, base + i * VL)
                    mag = vbitwise_and(vreinterpret(xv, I32), mag_mask, mask=m32)
                    idx = vsub(vshr(mag, LUT_SHIFT, mask=m32), off, mask=m32)
                    # Only a low clamp: the high side is bounded analytically.
                    # er is derived from this group's own amax in pass 1 and
                    # ``2^(er-127) >= amax/6``, so |x|/scale <= 6 and idx <= 8*129
                    # + 4 = 1036 < LUT_LEN.  Clamping er up to 1 only enlarges
                    # the scale, which lowers idx.  Do not reuse this table with
                    # an externally supplied scale without adding vmins.
                    idx = vmaxs(idx, 0, mask=m32)
                    code = vgather(self.lut_ub, vreinterpret(idx, U32), mask=m32)
                    # Float compare, not the raw sign bit: the golden takes the
                    # sign as ``q < 0``, and -0.0 < 0 is false, so -0.0 must
                    # encode as nibble 0x0 rather than 0x8.  This matches the
                    # golden wherever the fp32 quotient is nonzero or x is +-0.
                    # It still differs when the quotient itself underflows to
                    # -0.0 while x != 0; making that exact needs the group's
                    # flush threshold max((er-150)<<23, 0) gathered alongside
                    # off, plus a compare and a select -- 3 more instructions on
                    # the path that is already the critical one, to cover a
                    # within-group dynamic range of 2^152 that an fp32
                    # accumulation cannot produce (see test_vf_math.py).
                    neg = vlts(xv, 0.0, mask=m32)
                    sign = vselect(k8, zero_i, cond_mask=neg)
                    nib = vbitwise_or(code, sign, mask=m32)
                    # even lane * 1 + odd lane * 16 -> one byte per lane pair,
                    # low nibble = even element, which is the golden's layout.
                    # Only VL/2 lanes carry a result, so the store is masked to
                    # the lowest half or it would clobber the next row.
                    nibf = vcast(nib, F32, mask=m32)
                    packed = vcast(
                        vpair_reduce_sum(vmul(nibf, pack_w, mask=m32), mask=m32),
                        I32,
                        mask=m32,
                    )
                    vstore_pack(
                        self.q_ub,
                        base // 2 + i * (VL // 2),
                        packed,
                        m_lo,
                        pack_mode=PackMode.B32_TO_B8,
                    )

    @jit
    def store(self, q_half, dsc_half, rows):
        """Write back exactly ``rows`` rows, so a short T tail cannot overrun GM."""
        p = self.p
        mem_copy(
            q_half,
            local_slice(self.q_ub, (rows, p.d // 2), stride=(p.d // 2, 1)),
        )
        mem_copy(
            dsc_half,
            local_slice(self.dsc_ub, (rows, p.groups), stride=(p.groups, 1)),
        )


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------


@kernel
class indexer_prologue_qw_interleaved_kernel:
    def __init__(self, t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=False):
        self.p = Plan(t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn)

    def __call__(
        self,
        x: Tensor,
        qr: Tensor,
        wqb: Tensor,
        ww: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        lut: Tensor,
        q: Tensor,
        descale_q: Tensor,
        w: Tensor,
        ws: Tensor,
        # Scalars last, widest first, so the 4-byte one is the final argument.
        # Every tensor argument is a struct of a pointer and int64 shape/stride
        # arrays, so it needs 8-byte alignment; a 4-byte scalar ahead of one
        # forces the toolchain to pad, and the two CANN 9.2.0 builds here do not
        # agree on that padding.  With ``softmax_scale`` at slot 9 the 20260917
        # build read every pointer after it from the wrong offset: T=2048 came
        # back with a different ``descale_q`` on a repeat call with identical
        # inputs, and the field saw the same misalignment as an MTE access to an
        # invalid GM address.  Nothing follows the float now, so there is no
        # padding to disagree about.
        t_live: I64,
        softmax_scale: F32,
    ):
        p = self.p
        cube_q = CubeQ(p)
        cube_w = CubeW(p)
        vec = VectorQ(p)
        # C_q -> V_q, the only cross-core edge. depth 2 lets the cube run head
        # h+1 while the vector still owns head h.
        cv_ub = Channel(
            MemLoc.UB,
            (p.rows_vec, p.d),
            F32,
            depth=D_CV,
            kind=ChannelKind.CrossCore,
        )
        subblock = get_subblock_id()

        # Every T-derived count, recomputed here from the live row count exactly
        # as ``Plan`` computes it on the host.  Under ``t_dyn`` these are runtime
        # values and the binary is tied to none of them; a static plan keeps them
        # constants so the trip counts stay visible to the compiler.  Nothing
        # else in this kernel reads T -- that is what ``Plan.SHAPE_KEY`` is the
        # inventory of, and ``_check_t_dynamic`` the guard on.
        if const_expr(p.t_dyn):
            n_tiles = (t_live + (p.base_m - 1)) // p.base_m
            n_work = n_tiles * p.n_head_blocks
            # Stride between one K slice's band of ``ws`` and the next.  ``ws``
            # is laid out on the padded tile grid, so this counts padded rows.
            if const_expr(p.w_ws):
                n_bands = (t_live + (p.rows_red - 1)) // p.rows_red
                n_bands_ws = n_tiles * p.base_m // p.rows_red
            else:
                n_bands = 0
                n_bands_ws = 0
        else:
            n_tiles = p.n_tiles
            n_work = p.n_tiles * p.n_head_blocks
            n_bands = p.n_bands
            n_bands_ws = p.n_bands_ws
        first_work = get_block_idx()
        for work in range(first_work, n_work, get_block_num()):
            tile = work // p.n_head_blocks
            h_block = work % p.n_head_blocks
            # One tiler per distinct row width: the split is balanced over axis 0
            # only, so every tiler built from the same tile row count agrees on
            # where subblock 0 ends and subblock 1 begins.
            rows_full = tile_view(w, (p.base_m, p.n_heads), (tile, 0)).shape[0]
            # Row height of this tile's L1 staging: the live rows rounded up to
            # the fractal grid, which is what a GM-to-L1 copy of a short tile
            # writes.  See ``CubeW.x_rows``.
            rows_l1 = ceil_div(rows_full, NZ_M_FRAC) * NZ_M_FRAC
            split_cv = p.row_split(rows_full, p.d)
            split_q = p.row_split(rows_full, p.d // 2)
            split_dsc = p.row_split(rows_full, p.groups)
            rows = partition_view(tile_view(descale_q, (p.base_m, p.groups), (tile, 0)), split_dsc, subblock).shape[0]

            if const_expr(p.dr > 0):
                split_rope = p.row_split(rows_full, p.dr)
                cos_half = partition_view(tile_view(rope_cos, (p.base_m, p.dr), (tile, 0)), split_rope, subblock)
                sin_half = partition_view(tile_view(rope_sin, (p.base_m, p.dr), (tile, 0)), split_rope, subblock)
            else:
                cos_half = None
                sin_half = None
            # ``qr`` is on the first-mmad critical path; LUT/sin/cos only need
            # to be in UB before the first epilogue, so they go second and can
            # ride MTE2 under the first head's cube work.  Later tiles skip
            # ``load_qr`` when the previous tile already prefetched it.
            if (not const_expr(p.prefetch)) or work == first_work:
                cube_q.load_qr(qr, descale_qr, tile)
            vec.load_tile_constants(lut, cos_half, sin_half)
            # ``w_leader_only`` is compile-time (depends on T).  The leader
            # path is the only one allowed to acquire the W L0C: a runtime
            # if around commit/release is read as a second writer.
            if const_expr(not p.w_leader_only) or h_block == 0:
                # The separate-W path retires the L0C before the head loop, so
                # under IPQW_PROBE=no_w its acquire/release interval would hold
                # no data op at all and no sync could be emitted for it.
                if const_expr(not (PROBE == "no_w" and p.w_separate)):
                    w_slot = cube_w.l0c.acquire()
                else:
                    w_slot = None
                if const_expr(p.w_separate) and const_expr(PROBE != "no_w"):
                    w_k_base = h_block * p.k_l1_tiles_w if const_expr(p.w_k_split) else 0
                    cube_w.gemm_full(x, ww, tile, w_k_base, w_slot, rows_l1)
                    if const_expr(p.w_ws):
                        cube_w.store_partial(
                            ws,
                            tile,
                            h_block,
                            n_tiles,
                            softmax_scale,
                            w_slot,
                            rows_l1,
                        )
                    else:
                        cube_w.store(w, tile, softmax_scale, w_slot, rows_l1)
                    w_slot = None
                    if const_expr(p.w_ws and p.reduce_at == "early" and PROBE != "no_reduce"):
                        # Every core is a microsecond into the kernel with the
                        # same W slice behind it, so the arrivals are tight;
                        # and the AIVs are still idle here, which is what the
                        # reduction needs.  See W_REDUCE_AT.
                        if const_expr(PROBE != "no_sync"):
                            w_barrier()
                        vec.reduce_all(ws, w, n_bands, n_bands_ws)
                if const_expr(W_SCHED == "wpipe") and const_expr(PROBE != "no_w") and const_expr(not p.w_separate):
                    cube_w.fill(x, ww, tile, 0, 0)
                qr_ready, dqr_ready = cube_q.wait_qr()
                qr_rows, dqr_rows = cube_q.rows_view(qr_ready, dqr_ready, rows_l1)
                for local_head in range(p.heads_per_block):
                    head = h_block * p.heads_per_block + local_head
                    if const_expr(PROBE != "no_w") and const_expr(W_SCHED == "wfirst") and const_expr(not p.w_separate):
                        cube_w.gemm_head_slice(x, ww, tile, local_head, 0, w_slot, rows_l1)
                    cube_q.gemm_head(wqb, descale_wqb, head, qr_rows, dqr_rows)
                    if const_expr(PROBE != "no_w") and const_expr(W_SCHED == "wafter") and const_expr(not p.w_separate):
                        cube_w.gemm_head_slice(x, ww, tile, local_head, 0, w_slot, rows_l1)
                    elif (
                        const_expr(PROBE != "no_w") and const_expr(W_SCHED == "wpipe") and const_expr(not p.w_separate)
                    ):
                        if local_head + 1 < p.heads_per_block:
                            cube_w.fill(x, ww, tile, local_head + 1, 0)
                        cube_w.drain(local_head, w_slot, rows_l1)
                    cube_q.drain_to_vec(cv_ub, split_cv)
                    vec.epilogue(cv_ub, rows)
                    vec.store(
                        partition_view(
                            tile_view(q, (p.base_m, p.d // 2), (tile, head)),
                            split_q,
                            subblock,
                        ),
                        partition_view(
                            tile_view(descale_q, (p.base_m, p.groups), (tile, head)),
                            split_dsc,
                            subblock,
                        ),
                        rows,
                    )
                cube_q.release_qr(qr_ready, dqr_ready)
                _prefetch_qr(cube_q, qr, descale_qr, p, work, tile, n_work)
                if const_expr(PROBE != "no_w") and const_expr(not p.w_separate):
                    cube_w.store(w, tile, softmax_scale, w_slot, rows_l1)
                elif const_expr(PROBE == "no_w") and const_expr(not p.w_separate):
                    cube_w.l0c.commit(w_slot)
                    cube_w.l0c.release(cube_w.l0c.wait())
            else:
                qr_ready, dqr_ready = cube_q.wait_qr()
                qr_rows, dqr_rows = cube_q.rows_view(qr_ready, dqr_ready, rows_l1)
                for local_head in range(p.heads_per_block):
                    head = h_block * p.heads_per_block + local_head
                    cube_q.gemm_head(wqb, descale_wqb, head, qr_rows, dqr_rows)
                    cube_q.drain_to_vec(cv_ub, split_cv)
                    vec.epilogue(cv_ub, rows)
                    vec.store(
                        partition_view(
                            tile_view(q, (p.base_m, p.d // 2), (tile, head)),
                            split_q,
                            subblock,
                        ),
                        partition_view(
                            tile_view(descale_q, (p.base_m, p.groups), (tile, head)),
                            split_dsc,
                            subblock,
                        ),
                        rows,
                    )
                cube_q.release_qr(qr_ready, dqr_ready)
                _prefetch_qr(cube_q, qr, descale_qr, p, work, tile, n_work)

        if const_expr(p.w_ws and p.reduce_at == "tail" and PROBE != "no_reduce"):
            # Every partial has to be in GM before anyone sums it.  Waiting
            # for that here costs far more than the dependency does: the
            # partials all landed in the first microsecond, but a barrier
            # waits for the slowest arrival, so this one also absorbs the Q
            # path's load imbalance.  IPQW_PROBE=no_sync drops it to price
            # it, which is racy by construction.
            if const_expr(PROBE != "no_sync"):
                w_barrier()
            vec.reduce_all(ws, w, n_bands, n_bands_ws)


class IndexerPrologueQw:
    """Host wrapper: validates, plans tiles, launches the fused kernel."""

    def __init__(self, t, dim, q_lora, n_heads, d, dr, cube_cores=None, t_dyn=False, plan=None):
        # ``plan`` lets ``_plan_for`` hand over the plan it already built to read
        # the shape key off, instead of building an identical second one.
        self.p = plan or Plan(t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn)

    @jit
    def run(
        self,
        x: Tensor,
        qr: Tensor,
        wqb: Tensor,
        ww: Tensor,
        descale_qr: Tensor,
        descale_wqb: Tensor,
        rope_sin: Tensor,
        rope_cos: Tensor,
        lut: Tensor,
        q: Tensor,
        descale_q: Tensor,
        w: Tensor,
        ws: Tensor,
        t_live: I64,
        softmax_scale: F32,
    ):
        p = self.p
        # The grid, read back out of the marker rows the host put on the end of
        # ``ws``.  Symbolic, so it is not in the binary key -- a host integer here
        # is what used to make the core count a specialisation, since the block
        # dim rides along in the verified IR that key is taken over.
        #
        # It has to be the core count and not the work count: one block per work
        # item deadlocks as soon as the work outruns the chip, because the surplus
        # blocks are scheduled in a second wave while the W reduction rendezvous
        # waits on blocks that have not launched.  T=1024 is 8 tiles x 16 head
        # blocks = 128 blocks on a 64-AIV part, and it hangs to an aicore timeout.
        if const_expr(p.w_ws):
            tiles = (x.shape[0] + (p.base_m - 1)) // p.base_m
            grid = ws.shape[0] - tiles * p.n_head_blocks * p.base_m
        else:
            grid = ws.shape[0]
        indexer_prologue_qw_interleaved_kernel(p.t, p.dim, p.q_lora, p.n_heads, p.d, p.dr, p.cube_cores, p.t_dyn)[grid](
            x,
            qr,
            wqb,
            ww,
            descale_qr,
            descale_wqb,
            rope_sin,
            rope_cos,
            lut,
            q,
            descale_q,
            w,
            ws,
            t_live,
            softmax_scale,
        )


# ---------------------------------------------------------------------------
# NPU-only public host (FRACTAL_NZ weights; no CPU golden fallback)
# ---------------------------------------------------------------------------


try:
    from cannbotdsl.package.native import register
except ImportError:  # 0.3.dev441 has the kernel APIs but not the native packager

    def register(name):
        def decorate(function):
            return function

        return decorate


MX_GROUP_SIZE = 32
# torch_npu.Format.FRACTAL_NZ; spelled out so importing this module stays free
# of torch_npu until the NPU path actually runs.
FRACTAL_NZ = 29
# T values the packaged export compiles a binary for.  The host entry point
# below plans any T; this list only bounds the AOT build.
#
# The default is the deployment set: T is batch * seq, so decode contributes
# batch x {1 accepted token, 1 + 5 MTP draft tokens} and prefill contributes
# batch x chunk length with no MTP.  Kept as a literal rather than derived from
# the verification shape module so that ``net/ops`` stays importable on its own;
# ``net/verification/test/indexer_prologue_qw_shapes.py`` generates it.
DEPLOYMENT_T = (
    1,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    72,
    96,
    192,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)
EXPORT_T = tuple(int(v) for v in (",".join(str(t) for t in DEPLOYMENT_T)).split(","))


def to_nz(weight: torch.Tensor) -> torch.Tensor:
    """Convert one ND device weight to the FRACTAL_NZ layout this op expects.

    ``wqb`` and ``ww`` are static weights, so the intended deployment converts
    them once at load time and the operator never pays for a layout change.
    This helper exists for tests and for callers that still hold ND weights;
    do not call it per step.
    """
    import torch_npu

    if weight.device.type != "npu":
        raise ValueError("to_nz requires an NPU tensor")
    return torch_npu.npu_format_cast(weight, FRACTAL_NZ)


def split_t_min(cube_cores: int | None = None) -> int:
    """Smallest T whose T-tiles fill the chip without splitting heads.

    This is where the two templates meet -- split-K below, split-T above -- and
    nothing more.  It is not a floor on T being dynamic: T is a runtime value on
    both sides of it.
    """
    return BASE_M * resolve_cube_cores(cube_cores)


# One entry per distinct compile-time shape (``Plan.shape_key``), not per T.  T
# is a runtime value in every entry, so a T nobody listed costs nothing; what
# picks an entry is the head split and the launch grid, which are buffer shapes
# and a grid size rather than counts.  See ``Plan.SHAPE_KEY``.
_PLAN_CACHE: dict[tuple, IndexerPrologueQw] = {}


def plan_buckets(dim, q_lora, n_heads, d, dr, cube_cores=None, t_max=None):
    """Every distinct binary the T range needs, and the T range each one serves.

    With ``base_m`` pinned, every compile-time field of the plan is a function of
    the tile count alone, so walking the tile count enumerates the binaries
    exactly -- no need to walk T, of which there are 262144.  Tile count ``k``
    serves ``T`` in ``((k-1)*base_m, k*base_m]``.

    Returns a list of ``(plan, t_lo, t_hi)`` ordered by ``t_lo``.  The ranges can
    overlap, because the head-split score is not monotonic in the tile count: a
    binary's range is the hull of the tile counts that chose it, and the host
    picks by score rather than by range.
    """
    cube_cores = resolve_cube_cores(cube_cores)
    t_max = t_max or T_MAX
    seen: dict[tuple, list] = {}
    for tiles in range(1, ceil_div(t_max, BASE_M) + 1):
        plan = Plan(tiles * BASE_M, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=True)
        entry = seen.setdefault(plan.shape_key(), [plan, tiles, tiles])
        entry[1] = min(entry[1], tiles)
        entry[2] = max(entry[2], tiles)
    out = [(plan, (lo - 1) * BASE_M + 1, min(hi * BASE_M, t_max)) for plan, lo, hi in seen.values()]
    out.sort(key=lambda item: item[1])
    return out


def _plan_for(t, dim, q_lora, n_heads, d, dr, cube_cores=None):
    """The dynamic-T plan serving this T, reusing one per distinct shape key.

    Neither phase can enumerate its T at build time.  Prefill T is
    ``batch * chunk length`` and the chunk length is the scheduler's; decode T is
    ``batch * accepted tokens`` and the accepted count moves with how many MTP
    drafts survive, so it is 1..6 per request rather than a fixed 1 or 6.  So T
    is a dynamic axis everywhere and this never specialises on it.

    What it does specialise on is ``Plan.shape_key()``: the head split, the band
    geometry and the launch grid, which are chosen by tile count against core
    count and cannot be runtime values because they are buffer shapes and a grid
    size.  Those still move with T, but they take only a handful of distinct
    values over the whole range -- so the plan is built at the smallest T that
    lands on each one and then serves every other T that lands there too.
    """
    cube_cores = resolve_cube_cores(cube_cores)
    if not T_DYN:
        # A/B only: one binary per T, with ``base_m`` sized to T and the grid cut
        # to the work count.  This is what the operator did before T was a
        # runtime value, and it is how the cost of making it one gets priced.
        key = (t, dim, q_lora, n_heads, d, dr, cube_cores)
        op = _PLAN_CACHE.get(key)
        if op is None:
            op = IndexerPrologueQw(t, dim, q_lora, n_heads, d, dr, cube_cores)
            _PLAN_CACHE[key] = op
        return op
    probe = Plan(t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=True)
    key = probe.shape_key()
    op = _PLAN_CACHE.get(key)
    if op is None:
        op = IndexerPrologueQw(t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=True, plan=probe)
        _PLAN_CACHE[key] = op
    return op


def _arg_specs(dim, q_lora, n_heads, d, dr, rows_live, ws_rows):
    """The spec ``op.run`` is compiled against, for one bucket.

    Shared by the packaged export and the host entry deliberately.  A native
    package is looked up by the digest of this spec, so a host that compiled
    against concrete row counts could never find the binary the packager built
    against ``Dim``s: the lookup misses on the contract digest and the operator
    silently re-traces on every call instead of launching the shipped ``.so``.

    One row count for every T-axis tensor.  ``x`` is the only one the caller may
    hand over taller than T, and the host truncates it before the launch, so the
    contract does not have to admit two different row extents any more.

    Every tensor first, then the scalars widest last, which is an alignment
    requirement rather than a style: see the kernel signature for the padding
    the two CANN 9.2.0 builds disagreed about when the float sat mid-frame.
    """
    groups = ceil_div(d, MX_GROUP_SIZE)
    return (
        TensorSpec((rows_live, dim), dtypes.bfloat16),
        TensorSpec((rows_live, q_lora), U8),
        TensorSpec((n_heads * d, q_lora), U8, storage_format="nz"),
        TensorSpec((n_heads, dim), dtypes.bfloat16, storage_format="nz"),
        TensorSpec((rows_live, ceil_div(q_lora, 64), 2), U8),
        TensorSpec((n_heads * d, ceil_div(q_lora, 64), 2), U8),
        TensorSpec((rows_live, dr), F32),
        TensorSpec((rows_live, dr), F32),
        TensorSpec((LUT_LEN,), I32),
        TensorSpec((rows_live, n_heads * (d // 2)), U8),
        TensorSpec((rows_live, n_heads * groups), U8),
        TensorSpec((rows_live, n_heads), F32),
        TensorSpec((ws_rows, n_heads), F32),
        I64,
        F32,
    )


def _bucket_dims(index, plan, t_lo, t_hi):
    """The two T-derived ``Dim``s of one bucket: live rows and ws rows.

    ``index`` is the bucket's position in ``plan_buckets``, and it goes into the
    Dim names.  It is part of the contract digest, so the host has to number the
    buckets exactly the way the export does or the lookup misses.
    """
    tiles_lo = ceil_div(t_lo, BASE_M)
    tiles_hi = ceil_div(t_hi, BASE_M)
    # ``ws`` is one band of rows per K slice per padded tile, so it scales with
    # the tile count too and cannot be a fixed extent either, plus the grid
    # marker rows on the end (see MAX_CORES).  The marker is why the bounds are
    # slack and why there is no ``multiple_of``: they have to admit every core
    # count from 1 to MAX_CORES, since a bound is part of the contract and a
    # bound spelled in terms of the core count would specialise on it again.
    ws_rows = (
        Dim(
            f"WS{index}",
            min=plan.n_head_blocks * tiles_lo * BASE_M + 1,
            max=plan.n_head_blocks * tiles_hi * BASE_M + MAX_CORES,
        )
        if plan.w_ws
        else Dim(f"WS{index}", min=1, max=MAX_CORES)
    )
    return (
        Dim(f"T{index}", min=t_lo, max=t_hi),
        ws_rows,
    )


# ``plan_buckets`` builds a Plan per tile count over the whole T range, so it is
# far too expensive to redo per call; the host needs it to find which bucket owns
# this T.
_BUCKET_CACHE: dict[tuple, list] = {}
# One open ProviderCallable per bucket.  Compiled against the bucket's Dims, so
# it hits the packaged binary when one is installed and is traced once when not.
_COMPILED_CACHE: dict[tuple, object] = {}


def _buckets_cached(dim, q_lora, n_heads, d, dr, cube_cores):
    key = (dim, q_lora, n_heads, d, dr, cube_cores)
    buckets = _BUCKET_CACHE.get(key)
    if buckets is None:
        buckets = plan_buckets(dim, q_lora, n_heads, d, dr, cube_cores)
        _BUCKET_CACHE[key] = buckets
    return buckets


def _run_for(op, dim, q_lora, n_heads, d, dr, cube_cores):
    """``op.run`` compiled against its bucket's spec, not against this call.

    Calling the ``@jit`` method directly would work, but it builds a contract
    from the concrete row counts of this one call.  The packaged binaries were
    built from ``Dim``s, so that contract can never match one: every call would
    miss the native lookup and re-trace, which costs a few hundred ms of host
    time against a kernel that runs in tens of microseconds.  Compiling through
    the bucket's own spec is what makes the shipped ``.so`` reachable.
    """
    if not T_DYN:
        # A/B path: there are no buckets, the plan is specialised on T.
        return op.run
    if not hasattr(op.run, "compile"):
        # The export harness substitutes an already-compiled launcher for
        # ``op.run`` so it can watch which artifact each case lands on.  There is
        # nothing left to compile against in that case, and re-compiling here
        # would defeat the point of the substitution.
        return op.run
    want = op.p.shape_key()
    key = (want, dim, q_lora, n_heads, d, dr, cube_cores)
    fn = _COMPILED_CACHE.get(key)
    if fn is None:
        for index, (plan, t_lo, t_hi) in enumerate(_buckets_cached(dim, q_lora, n_heads, d, dr, cube_cores)):
            if plan.shape_key() == want:
                break
        else:
            # No bucket owns this shape: let the @jit path trace it rather than
            # refuse to run.  Reachable only if the plan policy and the bucket
            # enumeration disagree, which is a bug, not a caller error.
            return op.run
        fn = op.run.compile(*_arg_specs(dim, q_lora, n_heads, d, dr, *_bucket_dims(index, plan, t_lo, t_hi)))
        _COMPILED_CACHE[key] = fn
    return fn


_CALL_LOG: deque = deque(maxlen=32)
_LOGGED_ONCE: set = set()


def _trace_call(op, t, n_tiles, partial_rows, ws_rows) -> None:
    """Check the workspace covers what the kernel will write, and optionally say so.

    The check is the point; the print is for reading it back.  One ``Plan`` serves
    every T in its bucket, and it was built at whichever T reached the bucket
    first, so anything sized off the plan rather than off this call's ``t`` is
    sized for the wrong T.  Sizing ``ws`` that way is a real failure that has been
    seen in the field: the partials of a large T written into a workspace
    allocated for a small one land outside it, which surfaces as an aicore error
    with a non-zero MTE status, or as "The address to Write GM is invalid".  It
    needs a small T and then a large one in the same process, so no single-shape
    test reaches it -- hence a guard on every call instead.

    ``need`` is recomputed from ``t`` rather than reusing ``n_tiles`` so that the
    two are not the same expression: the highest row the reduction touches is
    ``n_head_blocks * n_bands_ws * rows_red``, and ``n_bands_ws`` comes from
    ``t_live`` in the kernel, so this is that bound spelled from T alone.

    Printing is controlled by ``IPQW_LOG``: ``1`` for every call, ``once`` for one
    line per distinct shape.  ``once`` is the one to use in a running network,
    where per-call output is one line per token.  Either way the line goes into
    the ring buffer that ``recent_calls`` reads back; see there for why.
    """
    p = op.p
    record = (t, n_tiles, partial_rows, ws_rows, p.n_head_blocks, p.used_cores, p.t)
    _CALL_LOG.append(record)

    if p.w_ws:
        need = p.n_head_blocks * ceil_div(t, p.base_m) * p.base_m
        if partial_rows < need:
            raise RuntimeError(
                f"workspace too small for T={t}: {partial_rows} partial rows "
                f"allocated, {need} needed ({p.n_head_blocks} K slices x "
                f"{ceil_div(t, p.base_m)} tiles x {p.base_m} rows). The plan this "
                f"call went through was built at T={p.t}; size ws from this "
                f"call's tile count, not the plan's.\n" + recent_calls()
            )

    mode = ""
    if mode == "once":
        key = (p.shape_key(), t)
        if key in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(key)
    elif mode != "1":
        return
    print(f"[ipqw] {_describe(record)}", flush=True)


def _describe(record) -> str:
    t, n_tiles, partial_rows, ws_rows, hblk, grid, plan_t = record
    template = "split-T" if hblk == 1 else "split-K"
    return (
        f"T={t} {template} tiles={n_tiles} hblk={hblk} grid={grid} "
        f"ws={ws_rows}({partial_rows}+{grid} marker) plan_built_at_T={plan_t}"
    )


def recent_calls() -> str:
    """The last ``IPQW_CALL_LOG`` calls, oldest first, as text.

    For localising a device-side fault, which is what the per-call print cannot
    do.  A bad GM write is reported asynchronously: by the time the runtime
    raises, the call that made it has returned and the traceback points at
    whichever synchronisation happened to notice.  In a network doing thousands
    of calls a second that is no help, and the one failure mode known here is
    order-dependent -- it needs a small T and then a large one -- so what is
    wanted is the sequence of shapes leading up to the fault, not the line the
    exception surfaced on.

    Kept unconditionally, because a fault that needs a flag set to be diagnosable
    is a fault that has to be reproduced first.  The cost is appending a tuple to
    a bounded deque.  Call it from an exception handler around the model step.
    """
    if not _CALL_LOG:
        return "[ipqw] no calls recorded"
    lines = [f"[ipqw] last {len(_CALL_LOG)} call(s), oldest first:"]
    lines += [f"[ipqw]   {_describe(record)}" for record in _CALL_LOG]
    return "\n".join(lines)


_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _lut_cache(device: torch.device) -> torch.Tensor:
    """E2M1 rounding table, uploaded once per device."""
    cached = _LUT_CACHE.get(device)
    if cached is None:
        cached = build_e2m1_lut().to(device)
        _LUT_CACHE[device] = cached
    return cached


def indexer_prologue_qw(
    x: torch.Tensor,
    qr: torch.Tensor,
    wqb: torch.Tensor,
    ww: torch.Tensor,
    descale_qr: torch.Tensor,
    descale_wqb: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    softmax_scale: float,
    q: torch.Tensor | None = None,
    descale_q: torch.Tensor | None = None,
    w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused indexer prologue: MXFP8 Q GEMM + RoPE + MXFP4 quant, and BF16 W GEMM.

    NPU only. The two weights ``wqb`` and ``ww`` must already be in FRACTAL_NZ
    (see :func:`to_nz`); ``x``, ``qr``, both descales and the RoPE tables are ND.

    ``T`` comes from ``qr``.  ``x`` may have more rows than that -- a caller
    whose hidden-state buffer is allocated at its own tile height can hand it
    over as it stands -- and the extra rows are dropped by a view, not read.

    No GM is allocated for any input, at any T.  A short last row tile is
    handled where the row fractal is, in L1; see ``CubeQ.rows_view``.

    Returns ``(q, descale_q, w)``. ``q`` is packed uint8 e2m1 with shape
    ``(T, N, D/2)``; ``descale_q`` is e8m0; ``w`` is fp32.
    """
    tensors = (x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos)
    if not all(t.device.type == "npu" for t in tensors):
        raise ValueError("indexer_prologue_qw is NPU-only; all tensors must be on npu")
    if x.dim() != 2 or qr.dim() != 2 or wqb.dim() != 2 or ww.dim() != 2:
        raise ValueError("x, qr, wqb, ww must be rank-2")
    if x.shape[0] < qr.shape[0]:
        raise ValueError(f"x has fewer rows than T: x={x.shape[0]} qr={qr.shape[0]}")
    if rope_sin.shape != rope_cos.shape:
        raise ValueError("rope_sin and rope_cos shapes must match")
    if softmax_scale is None:
        raise ValueError("softmax_scale is required")
    # The kernel declares an element type per input and the DMA burst length
    # follows it, so a caller handing over fp16 or bf16 where fp32 is declared
    # makes MTE2 read twice the bytes the tensor owns.  Check the widths here:
    # dtype is metadata, so this costs nothing and reads no element.
    for name, tensor, want in (
        ("x", x, torch.bfloat16),
        ("ww", ww, torch.bfloat16),
        ("rope_sin", rope_sin, torch.float32),
        ("rope_cos", rope_cos, torch.float32),
    ):
        if tensor.dtype != want:
            raise ValueError(f"{name} must be {want}, got {tensor.dtype}")
    # Both weights are staged into L1 by a copy engine that cannot change
    # layout, so an ND weight does not run slowly -- it aborts the trace with a
    # raw ``cannir.dma_gm2l1 ... identical layout map kinds`` error that says
    # nothing about what the caller should have done.  Name it here instead.
    # These are static weights: the cast belongs at load time, not per step.
    import torch_npu

    for name, tensor in (("wqb", wqb), ("ww", ww)):
        if torch_npu.get_npu_format(tensor) != FRACTAL_NZ:
            raise ValueError(
                f"{name} must be in FRACTAL_NZ; convert it once at load time with to_nz({name}), not per step"
            )
    for name, tensor in (
        ("qr", qr),
        ("wqb", wqb),
        ("descale_qr", descale_qr),
        ("descale_wqb", descale_wqb),
    ):
        if tensor.element_size() != 1:
            raise ValueError(f"{name} must be a one-byte fp8/e8m0 dtype, got {tensor.dtype}")

    t = qr.shape[0]
    dim = x.shape[1]
    q_lora = qr.shape[1]
    n_heads = ww.shape[0]
    d = wqb.shape[0] // n_heads
    dr = rope_sin.shape[-1]
    groups = ceil_div(d, MX_GROUP_SIZE)
    dev = x.device

    cube_cores = resolve_cube_cores()
    op = _plan_for(t, dim, q_lora, n_heads, d, dr, cube_cores)
    lut = _lut_cache(dev)

    # Every T-axis input goes to the kernel at exactly ``t`` rows.  The short
    # last tile is a row-fractal problem, and it is solved where the fractal is:
    # the L1 staging pads it (see ``rows_l1`` in the kernel).  This operator
    # allocates no GM for an input.
    #
    # ``x`` is the one input a caller may hand over taller than T -- a decode
    # caller's hidden-state buffer is allocated at its own tile height -- and a
    # view is free, so the extra rows are dropped here rather than described in
    # the ABI.
    n_tiles = ceil_div(t, op.p.base_m)
    if x.shape[0] != t:
        x = x[:t]

    qr_in = qr.view(torch.uint8)
    dqr_in = descale_qr.reshape(t, -1, 2).view(torch.uint8)

    q_flat = torch.empty((t, n_heads * (d // 2)), dtype=torch.uint8, device=dev)
    dsc_flat = torch.empty((t, n_heads * groups), dtype=torch.uint8, device=dev)
    w_out = torch.empty((t, n_heads), dtype=torch.float32, device=dev)
    # One padded (T_tile, N) fp32 tile per K slice.  ``empty`` is enough: every
    # row is written by the core that owns it before the reduction reads it, so
    # there is nothing to initialise and no Zeros op in a captured graph.  One
    # workspace is needed.
    # Sized from this call's tile count, not the plan's: the plan was built at
    # whichever T first landed on this shape key.  Same formula as _plan_reduce.
    # The ``used_cores`` rows on the end are the grid marker the kernel reads the
    # launch width back out of; see MAX_CORES for why the grid travels this way.
    partial_rows = op.p.n_head_blocks * n_tiles * op.p.base_m if op.p.w_ws else 0
    ws_rows = partial_rows + op.p.used_cores
    ws = torch.empty((ws_rows, n_heads), dtype=torch.float32, device=dev)
    _trace_call(op, t, n_tiles, partial_rows, ws_rows)

    _run_for(op, dim, q_lora, n_heads, d, dr, cube_cores)(
        x.view(torch.bfloat16),
        qr_in,
        wqb.view(torch.uint8),
        ww.view(torch.bfloat16),
        dqr_in,
        descale_wqb.reshape(n_heads * d, -1, 2).view(torch.uint8),
        rope_sin,
        rope_cos,
        lut,
        q_flat,
        dsc_flat,
        w_out,
        ws,
        t,
        float(softmax_scale),
    )

    q_out = q_flat.view(t, n_heads, d // 2)
    dsc_out = dsc_flat.view(t, n_heads, ceil_div(d, 64), 2)
    if q is not None:
        q.copy_(q_out)
        q_out = q
    if descale_q is not None:
        descale_q.copy_(dsc_out)
        dsc_out = descale_q
    if w is not None:
        w.copy_(w_out)
        w_out = w
    return q_out, dsc_out, w_out


@register("indexer_prologue_qw_interleaved")
def export_indexer_prologue_qw():
    """One binary per distinct compile-time shape, each serving a range of T.

    T is a ``Dim`` in every binary emitted here -- neither phase can enumerate
    its T at build time.  Prefill T is ``batch * chunk length`` and the chunk
    length is the scheduler's; decode T is ``batch * accepted tokens`` and the
    accepted count depends on how many MTP drafts survive, so a build-time list
    of T would be a list of the shapes somebody happened to think of.

    What is still compile-time is the head split and the band geometry it
    implies, because those are buffer heights, a register unroll and the launch
    grid rather than counts -- see ``Plan.SHAPE_KEY``.  They are chosen by tile
    count against core count, so they change a handful of times over the whole
    range; ``plan_buckets`` enumerates those changes and this emits one binary
    per bucket with T as a Dim over the tile counts that chose it.

    The core count is compile-time for the same reason, and it is a property of
    the part being built for, not of the build host -- so it comes from
    ``resolve_cube_cores`` / ``IPQW_CUBE_CORES``.  A 28-core and a 32-core part
    need separately exported binaries.
    """
    dim, q_lora, n_heads, d, dr = 5120, 1280, 32, 128, 64
    cube_cores = resolve_cube_cores()

    buckets = plan_buckets(dim, q_lora, n_heads, d, dr, cube_cores)
    # ``IPQW_EXPORT_T`` no longer names the binaries -- T is a Dim, so there is no
    # binary per T to name.  It narrows to the buckets that serve the listed T,
    # which is what a caller asking for a subset actually wants.
    wanted = {_plan_for(t, dim, q_lora, n_heads, d, dr, cube_cores).p.shape_key() for t in EXPORT_T}
    for index, (plan, t_lo, t_hi) in enumerate(buckets):
        if plan.shape_key() not in wanted:
            continue
        op = IndexerPrologueQw(plan.t, dim, q_lora, n_heads, d, dr, cube_cores, t_dyn=True, plan=plan)
        fn = op.run.compile(*_arg_specs(dim, q_lora, n_heads, d, dr, *_bucket_dims(index, plan, t_lo, t_hi)))
        fn.close()
