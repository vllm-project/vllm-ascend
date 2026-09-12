# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# This file is a part of the vllm-ascend project.
#
# Ascend Triton implementation of the Qrita fused top-k / top-p sampler.
# Ported from vllm/v1/sample/ops/topk_topp_triton.py.
#
# Algorithm: "Qrita: High-performance Top-k and Top-p Algorithm for GPUs
# using Pivot-based Truncation and Selection" by Park et al.
# (https://arxiv.org/abs/2602.01518). A pivot-based truncation and
# selection scheme replaces the sort-based reference: candidate outliers
# are gathered once, then ternary/binary searches over a value pivot find
# the top-k and top-p cut points without any full-vocabulary sort.

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_TRITON_TABLE_CACHE: dict[tuple[torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_TRITON_BUFFER_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

# Ascend A2/A3 vector cores have a 192 KB Unified Buffer. A 8192-wide fp32
# tile (32 KB) fits but the pivot searches keep two live tile variants
# (pivot_0 / pivot_1 masks) plus dense temporaries, so the scan tile is
# halved to 4096 and the truncation tile to 2048. This follows the
# BLOCK_SIZE=4096 precedent of the other sampler kernels under
# vllm_ascend/ops/triton.
_NPU_BLOCK_SIZE = 4096
_NPU_BLOCK_SIZE_TRUNC = 2048

# fmt: off
_NORMAL_CDF_TO_SIGMA_TABLE = [
  3.656,  3.650,  3.650,  3.650,  3.626,  3.626,  3.626,  3.514,  3.514,  3.503,
  3.503,  3.434,  3.434,  3.428,  3.428,  3.387,  3.380,  3.380,  3.376,  3.373,
  3.373,  3.356,  3.354,  3.354,  3.291,  3.249,  3.234,  3.214,  3.198,  3.198,
  3.185,  3.177,  3.177,  3.165,  3.164,  3.161,  3.138,  3.120,  3.115,  3.113,
  3.093,  3.066,  3.054,  3.043,  3.037,  3.023,  2.993,  2.991,  2.976,  2.970,
  2.952,  2.946,  2.932,  2.908,  2.902,  2.895,  2.886,  2.874,  2.861,  2.844,
  2.836,  2.810,  2.801,  2.790,  2.784,  2.779,  2.767,  2.757,  2.745,  2.733,
  2.723,  2.716,  2.693,  2.678,  2.671,  2.656,  2.649,  2.629,  2.611,  2.595,
  2.592,  2.585,  2.574,  2.550,  2.543,  2.534,  2.521,  2.518,  2.497,  2.485,
  2.468,  2.450,  2.441,  2.430,  2.412,  2.402,  2.389,  2.383,  2.377,  2.364,
  2.349,  2.338,  2.332,  2.319,  2.310,  2.301,  2.282,  2.274,  2.266,  2.250,
  2.242,  2.236,  2.226,  2.215,  2.207,  2.196,  2.179,  2.171,  2.162,  2.147,
  2.135,  2.121,  2.109,  2.095,  2.085,  2.073,  2.063,  2.045,  2.030,  2.016,
  2.003,  1.992,  1.983,  1.972,  1.960,  1.949,  1.940,  1.928,  1.912,  1.897,
  1.881,  1.869,  1.854,  1.838,  1.824,  1.807,  1.792,  1.779,  1.764,  1.751,
  1.739,  1.726,  1.711,  1.697,  1.685,  1.668,  1.652,  1.636,  1.622,  1.603,
  1.585,  1.568,  1.551,  1.534,  1.513,  1.499,  1.480,  1.464,  1.441,  1.422,
  1.394,  1.373,  1.347,  1.320,  1.296,  1.270,  1.246,  1.219,  1.190,  1.163,
  1.135,  1.104,  1.073,  1.041,  1.006,  0.969,  0.931,  0.894,  0.851,  0.806,
  0.757,  0.702,  0.643,  0.574,  0.498,  0.405,  0.288,  0.134, -0.110, -3.813
]

_PERCENTILE_TO_STD_TABLE = [
  2.576,  2.319,  2.178,  2.064,  1.968,  1.892,  1.819,  1.757,  1.708,  1.659,
  1.616,  1.568,  1.526,  1.492,  1.456,  1.420,  1.382,  1.342,  1.309,  1.280,
  1.249,  1.221,  1.193,  1.169,  1.145,  1.121,  1.095,  1.073,  1.050,  1.030,
  1.008,  0.987,  0.966,  0.945,  0.926,  0.910,  0.891,  0.871,  0.854,  0.837,
  0.819,  0.803,  0.784,  0.767,  0.753,  0.734,  0.719,  0.702,  0.690,  0.675,
  0.658,  0.640,  0.625,  0.609,  0.595,  0.578,  0.564,  0.550,  0.537,  0.521,
  0.509,  0.495,  0.481,  0.466,  0.453,  0.439,  0.424,  0.410,  0.397,  0.383,
  0.370,  0.356,  0.343,  0.330,  0.316,  0.302,  0.289,  0.274,  0.261,  0.247,
  0.235,  0.223,  0.209,  0.196,  0.184,  0.172,  0.159,  0.149,  0.137,  0.124,
  0.112,  0.100,  0.086,  0.074,  0.062,  0.050,  0.035,  0.023,  0.009, -0.003,
 -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122,
 -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248,
 -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382,
 -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521,
 -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673,
 -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838,
 -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027,
 -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272,
 -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658,
 -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813
]
# fmt: on


@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """Update running (min, count) of values above a pivot across tiles.

    Tracks the smallest value strictly above a pivot and how many times
    it occurs.  Called once per tile per pivot; the running state is
    carried across tiles via `min_larger` / `num_min_larger`.

    Merge rule:
      - tile min < running min  -> replace both
      - tile min == running min -> accumulate count
      - tile min > running min  -> keep running values
    """
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_cnt = tl.sum(tl.where(tile_eq, 1, 0).to(tl.int32))
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tl.where(is_same, tile_cnt, 0))
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _topk_topp_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.int32)
        num_keep = tl.zeros((), dtype=tl.int32)
        num_kept = tl.zeros((), dtype=tl.int32)

        max_logit = -float("inf")
        min_logit = float("inf")

        # The dense BUFFER layout keeps every candidate in its original
        # position, so all pivot searches scan the full vocab width in
        # BLOCK_SIZE_TRUNC tiles.
        search_iters: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC

        if TOPK_ENABLED:
            k = tl.load(K + row_id)
            if k < VOCAB_SIZE:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(LOGITS_ROW + offs, mask=mask_n, other=-float("inf"))
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(tl.where(finite_mask, 1, 0).to(tl.int32))
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0)
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0))

                # Calculate outlier pivot t for Gaussian sigma-truncation.
                # triton-ascend cannot lower a scalar-indexed tl.load from a
                # pointer, so the tables are looked up with a masked vector
                # load reduced by tl.sum instead.
                table_offs = tl.arange(0, 256)
                percentile = (k * 200) // VOCAB_SIZE
                sigma = tl.sum(
                    tl.where(
                        table_offs == percentile,
                        tl.load(
                            PERCENTILE_TO_STD_TABLE + table_offs,
                            mask=table_offs < 200,
                            other=0.0,
                        ),
                        0.0,
                    )
                )
                sigma = sigma + tl.abs(sigma) * -0.15
                outlier_pivot = avg_logit + std_logit * sigma
                num_outliers = tl.zeros((), dtype=tl.int32)

                # First pass: compute max and min logits and gather outliers
                num_finite_total = tl.zeros((), dtype=tl.int32)
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))

                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk_mask = logits_blk > -float("inf")
                    finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                    num_finite_total += tl.sum(tl.where(finite_blk_mask & mask_n, 1, 0).to(tl.int32))

                    outlier_mask = (logits_blk > outlier_pivot) & mask_n
                    num_outliers += tl.sum(tl.where(outlier_mask, 1, 0).to(tl.int32))

                    # triton-ascend lowers the upstream runtime-indexed
                    # scatter (BUFFER_ROW + write_pos, write_pos derived
                    # from tl.cumsum) to DiscreteMemAccess / SyncBlockLock,
                    # which serializes the whole kernel. Keep the candidate
                    # set in a positionally dense buffer (-inf elsewhere)
                    # and use contiguous GM stores instead. The searches
                    # below must then scan the full vocab width, but the
                    # extra -inf lanes are skipped by the value masks.
                    buffer_blk = tl.where(outlier_mask, logits_blk, -float("inf"))
                    tl.store(BUFFER_ROW + offs_n, buffer_blk, mask=mask_n)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                # Second pass: Ternary search for the top-k pivot. Both
                # probe pivots share one scan over the data.
                num_iters = 0
                k_pivot = float("inf")
                k_pivots_num = tl.zeros((), dtype=tl.int32)
                min_larger = float("inf")
                num_min_larger = tl.zeros((), dtype=tl.int32)
                if num_outliers > k:
                    max_range = max_logit
                    min_range = outlier_pivot
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.int32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.int32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.int32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.int32)

                        # Single fused pass: compute k_pivots_num,
                        # min_larger, and num_min_larger together to avoid
                        # a second data scan. See _update_min_larger_stats
                        # for the tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                            mask_n_2 = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n,
                                mask=mask_n_2,
                                other=-float("inf"),
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(tl.where(above_0, 1, 0).to(tl.int32))
                            k_pivots_num_1 += tl.sum(tl.where(above_1, 1, 0).to(tl.int32))

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination
                        if k_pivots_num_0 >= k and k_pivots_num_0 - num_min_larger_0 < k:
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if k_pivots_num_1 >= k and k_pivots_num_1 - num_min_larger_1 < k:
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                else:
                    # If top-k outlier gathering failed, search whole space
                    max_range = max_logit
                    min_range = min_logit
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.int32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.int32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.int32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.int32)

                        # Single fused pass over the raw logits (same
                        # approach as the buffer path above).
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n,
                                mask=mask_n,
                                other=-float("inf"),
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(tl.where(above_0, 1, 0).to(tl.int32))
                            k_pivots_num_1 += tl.sum(tl.where(above_1, 1, 0).to(tl.int32))

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination
                        if k_pivots_num_0 >= k and k_pivots_num_0 - num_min_larger_0 < k:
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if k_pivots_num_1 >= k and k_pivots_num_1 - num_min_larger_1 < k:
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1

                duplicate_logit = min_larger
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - (k_pivots_num - k)
                num_kept = tl.zeros((), dtype=tl.int32)

                # Top-k only path.  If there are fewer finite values
                # than k (e.g. grammar mask), keep everything.
                final_pivot = k_pivot if num_finite_total > k else -float("inf")

                if TOPP_ENABLED and num_finite_total > k:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
                        min_logit = k_pivot
                        sum_exp_logits = 0.0

                        # Third pass: Calculate exp logits and sum. The
                        # top-k survivors stay in their original positions
                        # in BUFFER; rejected entries become exp(-inf)==0.
                        if num_outliers > k:
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n_2

                                # Duplicate logit handling for Top-k
                                if num_duplicate_logit > 0:
                                    duplicate_mask = (tl.abs(probs_blk - duplicate_logit) < 1e-9) & mask_n_2
                                    duplicate_count = tl.cumsum(tl.where(duplicate_mask, 1, 0).to(tl.int32)) + num_kept
                                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                                    outlier_mask = (outlier_mask & ~duplicate_mask) | duplicate_keep_mask
                                    num_kept += tl.sum(tl.where(duplicate_keep_mask, 1, 0).to(tl.int32))

                                probs_blk = tl.where(outlier_mask, probs_blk, -float("inf"))
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                            # Fourth pass: Normalize BUFFER into probabilities.
                            # exp(-inf) == 0 for the rejected entries, so the
                            # dense layout only adds zero-mass lanes.
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        else:
                            # If top-k outlier gathering failed,
                            # softmax the raw logits into BUFFER.
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n

                                # Duplicate logit handling for Top-k
                                duplicate_mask = (tl.abs(probs_blk - duplicate_logit) < 1e-9) & mask_n
                                duplicate_count = tl.cumsum(tl.where(duplicate_mask, 1, 0).to(tl.int32)) + num_kept
                                duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                                outlier_mask = (outlier_mask & ~duplicate_mask) | duplicate_keep_mask
                                num_kept += tl.sum(tl.where(duplicate_keep_mask, 1, 0).to(tl.int32))

                                probs_blk = tl.where(outlier_mask, probs_blk, -float("inf"))
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                                # Store the full tile densely (rejected
                                # entries become 0) to keep BUFFER_ROW
                                # stores contiguous.
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                            # Fourth pass: Normalize BUFFER into probabilities
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)

                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

                        p_pivot = 1.0
                        num_iters = 0
                        min_larger_prob = 1.0
                        num_min_larger = tl.zeros((), dtype=tl.int32)
                        p_pivots_sum = 0.0

                        # Fifth pass: Binary search for the top-p pivot
                        found_pivot = 0
                        while found_pivot == 0:
                            p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.int32)

                            # Single fused pass: compute p_pivots_sum,
                            # min_larger, and num_min_larger together.
                            # See _update_min_larger_stats for the
                            # tile-level merge logic.
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                                mask_n_2 = offs_n < VOCAB_SIZE
                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=0.0,
                                )

                                above_0 = probs_blk > p_pivot_0
                                p_pivots_sum_0 += tl.sum(tl.where(above_0, probs_blk, 0.0))

                                min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                    probs_blk,
                                    above_0,
                                    min_larger_0,
                                    num_min_larger_0,
                                    1.0,
                                )

                            # Check if the pivot satisfies termination
                            if p_pivots_sum_0 >= p and (p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p):
                                p_pivot = p_pivot_0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                            # Update range
                            if p_pivots_sum_0 > p:
                                min_range = p_pivot_0
                            elif p_pivots_sum_0 < p:
                                max_range = p_pivot_0

                            num_iters += 1
                            if (max_range - min_range) < 1e-9 or num_iters >= 18:
                                # Keep the pivot and its boundary statistics
                                # from the SAME evaluation. Mixing a midpoint
                                # pivot with stats from a different probe
                                # makes final_pivot and duplicate_logit
                                # inconsistent (observed as a top-p boundary
                                # accuracy failure on A3).
                                p_pivot = p_pivot_0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                        duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_logit
                        num_duplicate_logit = num_min_larger
                        remove_count = tl.maximum(((p_pivots_sum - p) / min_larger_prob).to(tl.int32), 0)
                        num_keep = tl.maximum(num_duplicate_logit - remove_count, 1)
                        num_kept = tl.zeros((), dtype=tl.int32)

                        # Top-k + Top-p path
                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            p = tl.load(P + row_id)
            if p < 1.0:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(LOGITS_ROW + offs, mask=mask_n, other=-float("inf"))
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(tl.where(finite_mask, 1, 0).to(tl.int32))
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0)
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0))
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # First pass: compute max and min logits and sum_exp_logits
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk = tl.where(logits_blk > -float("inf"), logits_blk, float("inf"))
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                # Vectorized table lookup (scalar-index loads unsupported
                # on triton-ascend); matches upstream idx = clamp(p*200).
                table_offs = tl.arange(0, 256)
                table_offs_f32 = table_offs.to(tl.float32)
                scaled_p = tl.maximum(0.0, tl.minimum(p * 200.0, 199.0))
                sigma = tl.sum(
                    tl.where(
                        (table_offs < 200) & (scaled_p >= table_offs_f32) & (scaled_p < table_offs_f32 + 1.0),
                        tl.load(
                            NORMAL_CDF_TO_SIGMA_TABLE + table_offs,
                            mask=table_offs < 200,
                            other=0.0,
                        ),
                        0.0,
                    )
                )
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.int32)

                # Second pass: Calculate softmax and gather outliers.
                # Candidates keep their positions in BUFFER; non-candidates
                # are stored as 0.0 probability (contiguous stores, and the
                # zero mass is ignored by every pivot test above 0).
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(tl.where(outlier_mask, probs_blk, 0.0))
                    num_outliers += tl.sum(tl.where(outlier_mask, 1, 0).to(tl.int32))

                    buffer_probs = tl.where(outlier_mask, probs_blk, 0.0)
                    tl.store(BUFFER_ROW + offs_n, buffer_probs, mask=mask_n)

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.int32)
                p_pivots_sum = 0.0

                # Third pass: Search for the top-p pivot. When the outlier
                # candidates already carry more mass than p, the top-p head
                # lies strictly inside them and the search can start from
                # outlier_prob. Otherwise re-populate BUFFER with the full
                # softmax distribution, because the top-p head may extend
                # past the candidate set into sub-threshold tokens.
                if sum_outlier_probs > p:
                    min_range = outlier_prob
                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.int32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                            mask_n_2 = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(tl.where(above_0, probs_blk, 0.0))

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination
                        if p_pivots_sum_0 >= p and (p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            # Keep pivot and boundary statistics from the
                            # same evaluation (see the note above).
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1
                else:
                    # Re-populate the buffer with the full softmax probs.
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.int32)

                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                            mask_n_2 = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(tl.where(above_0, probs_blk, 0.0))

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination
                        if p_pivots_sum_0 >= p and (p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            # Keep pivot and boundary statistics from the
                            # same evaluation (see the note above).
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
                num_duplicate_logit = num_min_larger
                remove_count = tl.maximum(((p_pivots_sum - p) / min_larger_prob).to(tl.int32), 0)
                num_keep = tl.maximum(num_duplicate_logit - remove_count, 1)
                num_kept = tl.zeros((), dtype=tl.int32)

                # Top-p only path
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # Sixth pass: Apply mask and store final output.
        # If the pivot >= max logit (or is NaN), no token would
        # survive the strict `>` keep_mask.  Skip masking.
        # Using `not <` instead of `>=` so that NaN is also caught.
        if not (final_pivot < max_logit):
            final_pivot = -float("inf")
        elif final_pivot != -float("inf"):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                keep_mask = (logits_blk > final_pivot) & mask_n

                # Duplicate logit handling
                if num_duplicate_logit > 0:
                    duplicate_mask = (tl.abs(logits_blk - duplicate_logit) < 1e-9) & mask_n
                    duplicate_count = tl.cumsum(tl.where(duplicate_mask, 1, 0).to(tl.int32)) + num_kept
                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                    num_kept += tl.sum(tl.where(duplicate_keep_mask, 1, 0).to(tl.int32))
                    keep_mask = (keep_mask & ~duplicate_mask) | duplicate_keep_mask

                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


def _get_vectorcore_count() -> int:
    try:
        return int(get_vectorcore_num())
    except AssertionError:
        init_device_properties_triton()
        return int(get_vectorcore_num())


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Apply combined top-k and top-p masking using Triton-Ascend.

    Top-k is applied first (by logit value), then top-p is applied
    to the remaining k values (by probability).

    Args:
        logits: [batch_size, vocab_size] float32 tensor. The returned tensor
            may alias this input or be a new contiguous tensor for unsupported
            layouts.
        k: [batch_size] int32 tensor of top-k values per row, or None to
            disable top-k.
        p: [batch_size] float32 tensor of top-p values per row (0 to 1),
            or None to disable top-p.
        mask_value: Value for masked positions (default: -inf)

    Returns:
        The masked logits tensor. It may or may not be modified in-place.
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None

    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits

    # The kernel supports arbitrary row strides, but it still assumes
    # the vocab dimension is laid out contiguously within each row.
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits  # Dummy pointer (won't be read)

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits  # Dummy pointer (won't be read)

    num_vector_cores = _get_vectorcore_count()
    NUM_PROGRAMS = min(num_vector_cores, batch_size)

    # Cache per-Triton Program buffer on each device.
    buf_key = (logits.device, logits.dtype, vocab_size)
    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.shape[0] < NUM_PROGRAMS:
        size = min(next_power_of_2(NUM_PROGRAMS), num_vector_cores)
        buffer = logits.new_empty((size, vocab_size))
        _TRITON_BUFFER_CACHE[buf_key] = buffer
    if buffer.shape[0] > NUM_PROGRAMS:
        buffer = buffer[:NUM_PROGRAMS]

    # Cache lookup table entries on each device.
    tables = _TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        normal_cdf_to_sigma_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
        percentile_to_std_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
        _TRITON_TABLE_CACHE[logits.device] = (
            normal_cdf_to_sigma_table,
            percentile_to_std_table,
        )
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables

    # Smaller tiles compile and run faster on CPU; the A2/A3 Unified
    # Buffer (192 KB) benefits from larger tiles. See _NPU_BLOCK_SIZE.
    if logits.device.type == "cpu":
        block_size, block_size_trunc = 256, 128
    else:
        block_size, block_size_trunc = _NPU_BLOCK_SIZE, _NPU_BLOCK_SIZE_TRUNC

    _topk_topp_kernel[(NUM_PROGRAMS,)](
        logits,
        logits.stride(0),
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
        BATCH_SIZE=batch_size,
        MASK_VALUE=mask_value,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_TRUNC=block_size_trunc,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
        multibuffer=False,
    )

    return logits


def reset_buffer_cache():
    _TRITON_BUFFER_CACHE.clear()
    _TRITON_TABLE_CACHE.clear()
    torch.accelerator.empty_cache()
