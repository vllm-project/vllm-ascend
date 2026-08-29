/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GroupedMatmulSituQuantV2 op_host.
// Fused GMM1 (fake-A8W8) + SiTU + per-token INT8 quant, single operator.
//
// DEVICE-SIDE group_list parsing (this revision):
//   The fused single-launch path performs NO host-side group_list parse. The
//   group_list device tensor is passed straight to the kernel, which parses it
//   on-device every forward (mE / startRow / mPadOff / sumMPad / MValid are
//   derived in-kernel from GM). This makes the operator correct for DYNAMIC
//   production MoE group_list (new tensor per forward OR in-place value update)
//   while keeping the host call path zero-copy / zero-sync on cache hit.
//
//   The persistent FusedMetaCache key is ONLY the static weight configuration:
//   (full weight/scale pointer set, K, N, C, group_list_type). group_list
//   identity/values are NOT part of the key — any group_list change is handled
//   by the on-device parse each forward.
//
//   The P34 expanded-w8 cache covers ALL E experts ([E,K,N] int8). On cache miss
//   the kernel's AIV stage fills the whole cache (unpack all experts), then the
//   AIC GEMM reads the active experts' w8 — so any group_list's active set is
//   always covered. skipUnpack=1 (cache hit) reuses the persistent expanded w8.
//
//   All mechanisms preserved: column-blocked w8, single-launch 3-stage pipeline,
//   NZ (FRACTAL_NZ) input, stacked/TensorList, capacity-padded x, gl_type 0/1.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>

#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "grouped_matmul_situ_quant_v2.h"

namespace vllm_ascend {

// Device kernel launcher exported by libvllm_ascend_kernels (ccec-compiled TU
// owns the kernel entry and the tripple-chevron launch; see op_kernel/).
void gmsq2_fused_256_impl(uint32_t blockDim, void *stream, void *x, void *wPtrTbl, void *scPtrTbl,
                          void *w8, void *acc, void *scaleF32, void *xScale, void *y, void *yScale,
                          void *groupList, int32_t E, int32_t kNum, int32_t nNum, int32_t nBlock,
                          int32_t NP, int32_t N, int32_t N2, int32_t K, int32_t C, int32_t glType,
                          float beta, float invBeta, int32_t hasLinear, float linBeta,
                          float invLinBeta, int32_t skipUnpack);

constexpr int32_t GMSQ2_BM = 128;
constexpr int32_t GMSQ2_BN = 128;
constexpr int32_t GMSQ2_BK = 64;
constexpr int32_t GMSQ2_EPI_M = 32;

// ACL tensor format ids (see acl_base.h / torch_npu Format enum).
constexpr int64_t ACL_FMT_ND = 2;
constexpr int64_t ACL_FMT_FRACTAL_NZ = 29;

// ---------------------------------------------------------------------------
// Persistent device-side metadata cache for the FUSED single-launch path.
// Key = static weight configuration ONLY (weights are static during inference):
//   fullWPtrs/fullSPtrs — data_ptr of every expert's weight / weight_scale
//   K, N, C            — tensor shapes
//   glType             — group_list interpretation mode (0 cumsum / 1 count)
// group_list identity is deliberately NOT in the key (it changes every forward
// in production MoE; the kernel parses it on-device each forward).
// ---------------------------------------------------------------------------
struct FusedMetaCache {
    // key
    std::vector<uintptr_t> fullWPtrs;
    std::vector<uintptr_t> fullSPtrs;
    int64_t K = 0;
    int64_t N = 0;
    int64_t C = 0;
    int64_t glType = 0;
    // NZ -> ND one-time converted tensors (all E, kept alive).
    bool srcIsNz = false;
    std::vector<at::Tensor> ndWeights;
    std::vector<at::Tensor> ndScales;
    // Strong refs to ORIGINAL weight/weight_scale tensors (all E) so the
    // allocator cannot reuse their data_ptr while this cache is alive.
    std::vector<at::Tensor> weightRefs;
    std::vector<at::Tensor> scaleRefs;
    // device tensors (static per weight config)
    at::Tensor tbl;     // int32 [4*E] = wPtrV(2E) + scPtrV(2E)
    at::Tensor wsFlat;  // workspace (accWs + scaleF32)
    int64_t scOff = 0;
    int64_t accOffWs = 0;
    int64_t scOffWs = 0;
    int64_t accBytes = 0;
    int64_t scBytes = 0;
    int32_t kNum = 0;
    int32_t nNum256 = 0;
    int32_t gemmNBlock = 1024;  // P54: MTE2 load-balance N-block, computed ONCE per
                                // cache build (removed from steady-state launch path)
    int64_t wBytes = 0;   // E*K*N
    int64_t accRows = 0;  // C + 128*E (max padded rows)
    bool valid = false;
};
static FusedMetaCache g_fusedCache;

// P34: expanded-w8 persistent cache, ALL experts [E,K,N] int8. Keyed on the
// full weight pointer set + shapes (static). A hit lets the kernel skip the
// whole unpack phase; a miss triggers a one-time in-kernel cache fill.
static std::vector<uintptr_t> g_cachedWPtrs;
static int64_t g_cachedK = 0;
static int64_t g_cachedN = 0;
static at::Tensor g_cachedW8;

// Return an ND tensor for `t` (original if already ND, else FRACTAL_NZ -> ND).
static at::Tensor EnsureNd(const at::Tensor &t, std::vector<at::Tensor> &converted)
{
    if (at_npu::native::get_npu_format(t) == ACL_FMT_ND) {
        return t;
    }
    at::Tensor nd = at_npu::native::npu_format_cast(t, ACL_FMT_ND);
    converted.push_back(nd);
    return nd;
}

static std::vector<int32_t> BuildPtrVec(const std::vector<uintptr_t> &ptrs)
{
    std::vector<int32_t> v;
    v.reserve(ptrs.size() * 2);
    for (uintptr_t p : ptrs) {
        v.push_back(static_cast<int32_t>(static_cast<uint32_t>(p & 0xFFFFFFFFu)));
        v.push_back(static_cast<int32_t>(p >> 32));
    }
    return v;
}

std::tuple<at::Tensor, at::Tensor> grouped_matmul_situ_quant_v2(
    const at::Tensor &x, at::TensorList weight, at::TensorList weight_scale,
    const at::Tensor &x_scale, const at::Tensor &group_list, at::TensorList weight_assist_matrix,
    double beta, std::optional<double> linear_beta, int64_t group_list_type)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(x.scalar_type() == at::kChar, "x must be int8");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "x_scale must be float32");
    TORCH_CHECK(!weight.empty(), "weight list must be non-empty");
    TORCH_CHECK(weight.size() == weight_scale.size(), "weight/weight_scale size mismatch");
    TORCH_CHECK(weight_assist_matrix.empty() || weight_assist_matrix.size() == weight.size(),
                "weight_assist_matrix size mismatch");
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1, "group_list_type must be 0 or 1");

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t aivCoreNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAiv());
    int32_t aicCoreNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAic());
    TORCH_CHECK(aivCoreNum > 0 && aicCoreNum > 0, "failed to query core numbers");

    int64_t C = x.sizes()[0];      // x rows (capacity, >= M_valid)
    int64_t K = x.sizes()[1];
    int64_t experts = static_cast<int64_t>(weight.size());
    int64_t NP = weight[0].sizes()[1];  // packed N/8 per row
    int64_t N = NP * 8;
    int64_t N2 = N / 2;
    TORCH_CHECK(K % GMSQ2_BK == 0, "K must be a multiple of 64");
    TORCH_CHECK(N % GMSQ2_BN == 0, "N must be a multiple of 128");
    TORCH_CHECK(N2 % GMSQ2_BN == 0, "N/2 must be a multiple of 128");
    TORCH_CHECK(weight[0].scalar_type() == at::kInt, "weight must be int32 packed");
    TORCH_CHECK(weight_scale[0].scalar_type() == at::kLong, "weight_scale must be int64 carrier");

    auto devOptI8 = at::device(at::kPrivateUse1).dtype(at::kChar);
    auto devOptI32 = at::device(at::kPrivateUse1).dtype(at::kInt);
    auto devOptF32 = at::device(at::kPrivateUse1).dtype(at::kFloat);

    // outputs: y [C, N2] int8, y_scale [C] fp32 (only first M_valid rows meaningful)
    at::Tensor y = at::empty({C, N2}, devOptI8);
    at::Tensor y_scale = at::empty({C}, devOptF32);

    // EP empty rank (M_valid == 0): x may be [0, K] with an all-zero group_list
    // (short requests under TP8+EP leave some ranks without routed tokens). All
    // parameter validation is done above and the outputs are already empty with
    // the right shapes — return without building caches, reading group_list or
    // launching the kernel. A C > 0 input whose device-side group_list is all
    // zero stays on the normal launch path; the kernel tolerates
    // sum(group_list) == 0 (the P54 waveBatch clamp keeps batches at 0, so no
    // tile is scheduled and no cross-core flag pair is exercised).
    if (C == 0) {
        return {y, y_scale};
    }

    float betaF = static_cast<float>(beta);
    float invBeta = 1.0f / betaF;
    int32_t hasLinear = linear_beta.has_value() ? 1 : 0;
    float lbF = hasLinear ? static_cast<float>(*linear_beta) : 1.0f;
    float invLb = hasLinear ? 1.0f / lbF : 1.0f;

    // ---- FUSED single-launch path (device-side group_list parse) ----
    int32_t nNum256 = static_cast<int32_t>(N / 256);
    bool fusedEligible = (N % 256 == 0) && (nNum256 <= 128) && (experts > 0);
    if (fusedEligible) {
        // Cheap host-side key (no H2D/D2H, no sync): full pointer sets + shapes +
        // gl_type. group_list is deliberately NOT in the key.
        std::vector<uintptr_t> fullWPtrs(static_cast<size_t>(experts));
        std::vector<uintptr_t> fullSPtrs(static_cast<size_t>(experts));
        bool ptrsOk = true;
        for (int64_t e = 0; e < experts; e++) {
            if (weight[e].data_ptr() == nullptr || weight_scale[e].data_ptr() == nullptr) {
                ptrsOk = false;
                break;
            }
            fullWPtrs[static_cast<size_t>(e)] = reinterpret_cast<uintptr_t>(weight[e].data_ptr());
            fullSPtrs[static_cast<size_t>(e)] = reinterpret_cast<uintptr_t>(weight_scale[e].data_ptr());
            int64_t wfmt = at_npu::native::get_npu_format(weight[e]);
            if (wfmt != ACL_FMT_ND && wfmt != ACL_FMT_FRACTAL_NZ) {
                ptrsOk = false;
            }
        }

        FusedMetaCache &fc = g_fusedCache;
        bool hit = ptrsOk && fc.valid && (fc.K == K) && (fc.N == N) && (fc.C == C) &&
                   (fc.glType == group_list_type) && (fc.fullWPtrs == fullWPtrs) &&
                   (fc.fullSPtrs == fullSPtrs) && fc.tbl.defined() && fc.wsFlat.defined();

        if (!hit) {
            // ---- CACHE MISS: build static pointer tables + workspace (no D2H) ----
            fc.ndWeights.clear();
            fc.ndScales.clear();
            fc.weightRefs.clear();
            fc.scaleRefs.clear();
            // pointer tables for ALL experts (expert index = table index)
            std::vector<uintptr_t> wPtrsAll;
            std::vector<uintptr_t> scPtrsAll;
            wPtrsAll.reserve(static_cast<size_t>(experts));
            scPtrsAll.reserve(static_cast<size_t>(experts));
            for (int64_t e = 0; e < experts; e++) {
                at::Tensor wNd = EnsureNd(weight[static_cast<size_t>(e)], fc.ndWeights);
                at::Tensor sNd = EnsureNd(weight_scale[static_cast<size_t>(e)], fc.ndScales);
                fc.weightRefs.push_back(weight[static_cast<size_t>(e)]);
                fc.scaleRefs.push_back(weight_scale[static_cast<size_t>(e)]);
                TORCH_CHECK(wNd.is_contiguous(), "weight[e] must be contiguous (ND)");
                TORCH_CHECK(sNd.is_contiguous(), "weight_scale[e] must be contiguous (ND)");
                wPtrsAll.push_back(reinterpret_cast<uintptr_t>(wNd.data_ptr()));
                scPtrsAll.push_back(reinterpret_cast<uintptr_t>(sNd.data_ptr()));
            }
            std::vector<int32_t> wPtrV = BuildPtrVec(wPtrsAll);
            std::vector<int32_t> scPtrV = BuildPtrVec(scPtrsAll);
            int64_t wPtrLen = static_cast<int64_t>(wPtrV.size());
            int64_t scOff = wPtrLen;
            int64_t tblLen = wPtrLen + static_cast<int64_t>(scPtrV.size());
            std::vector<int32_t> tblV;
            tblV.reserve(static_cast<size_t>(tblLen));
            tblV.insert(tblV.end(), wPtrV.begin(), wPtrV.end());
            tblV.insert(tblV.end(), scPtrV.begin(), scPtrV.end());
            at::Tensor tblCpu = at::from_blob(
                tblV.data(), {tblLen}, [](void *) {}, at::TensorOptions().dtype(at::kInt));
            at::Tensor tbl = at::empty({tblLen}, devOptI32);
            tbl.copy_(tblCpu, /*non_blocking=*/false);  // H2D — one-time per weight config

            // workspace with MAX bounds (device-side group_list parse determines the
            // actual active rows; we allocate the worst case C + 64*E padded rows).
            auto align64 = [](int64_t v) { return (v + 63) / 64 * 64; };
            int64_t wBytes = experts * K * N;
            int64_t accRows = C + 128 * experts;  // max sumMPad (BM=128)
            // P49: accumulator is now half (epilogue dequant int32->half), halving
            // the GM intermediate traffic vs. the int32 baseline.
            int64_t accBytes = accRows * N * static_cast<int64_t>(sizeof(at::Half));
            int64_t scBytes = experts * 2 * N2 * static_cast<int64_t>(sizeof(float));
            int64_t accOffWs = align64(wBytes);
            int64_t scOffWs = align64(accOffWs + accBytes);
            int64_t wsTotal = scOffWs + scBytes;
            at::Tensor wsFlat = at::empty({wsTotal}, devOptI8);

            fc.fullWPtrs = std::move(fullWPtrs);
            fc.fullSPtrs = std::move(fullSPtrs);
            fc.K = K; fc.N = N; fc.C = C; fc.glType = group_list_type;
            fc.tbl = tbl;
            fc.scOff = scOff;
            fc.accOffWs = accOffWs; fc.scOffWs = scOffWs;
            fc.accBytes = accBytes; fc.scBytes = scBytes;
            fc.kNum = static_cast<int32_t>(K / GMSQ2_BK);
            fc.nNum256 = nNum256;
            fc.wBytes = wBytes;
            fc.accRows = accRows;
            fc.wsFlat = wsFlat;  // plain copy (refcount) — avoids move-reorder hazards
            fc.valid = true;

            // P54: MTE2 load-balance N-block scheduling computed ONCE per weight
            // config (cache build), removing the per-call scalar scheduling loop
            // from the steady-state launch path. Same first-principles rule as the
            // original per-call loop: critB = waves*blk minimized, largest blk on
            // ties to minimize A re-fetch / per-tile scalar overhead.
            {
                int32_t cores = aicCoreNum;
                int64_t bestCritB = INT64_MAX;
                int32_t gblock = 1024;
                for (int32_t blk = 1024; blk >= 256; blk -= 256) {
                    if (N % blk != 0) {
                        continue;  // clean tiling only (N is a multiple of 256)
                    }
                    int64_t tiles = experts * (N / blk);
                    int32_t waves = static_cast<int32_t>((tiles + cores - 1) / cores);
                    int64_t critB = static_cast<int64_t>(waves) * blk;
                    if (critB < bestCritB) {
                        bestCritB = critB;
                        gblock = blk;
                    }
                }
                fc.gemmNBlock = gblock;
            }
        }

        // ---- STEADY STATE (cache hit): pure kernel launch below ----
        at::Tensor wPtrTf = fc.tbl.narrow(0, 0, static_cast<int64_t>(experts * 2));
        at::Tensor scPtrTf = fc.tbl.narrow(0, fc.scOff, static_cast<int64_t>(experts * 2));
        at::Tensor accWs = fc.wsFlat.narrow(0, fc.accOffWs, fc.accBytes).view(at::kHalf);
        at::Tensor scaleF32 = fc.wsFlat.narrow(0, fc.scOffWs, fc.scBytes).view(at::kFloat);

        // P34 expanded-w8 cache (ALL experts, static key).
        bool w8Hit = (g_cachedK == fc.K) && (g_cachedN == fc.N) &&
                     (g_cachedWPtrs == fc.fullWPtrs) && g_cachedW8.defined() &&
                     (g_cachedW8.numel() == fc.wBytes);
        if (!w8Hit) {
            g_cachedWPtrs = fc.fullWPtrs;
            g_cachedK = fc.K;
            g_cachedN = fc.N;
            g_cachedW8 = at::empty({fc.wBytes}, devOptI8);
        }
        at::Tensor wInt8 = g_cachedW8;
        int32_t skipUnpack = w8Hit ? 1 : 0;

        int32_t E32f = static_cast<int32_t>(experts);
        int32_t NP32f = static_cast<int32_t>(NP);
        int32_t N32f = static_cast<int32_t>(N);
        int32_t N232f = static_cast<int32_t>(N2);
        int32_t K32f = static_cast<int32_t>(K);
        int32_t C32f = static_cast<int32_t>(C);
        int32_t glType32 = static_cast<int32_t>(group_list_type);

        // P54: N-block scheduling was computed once per cache build (see the
        // cache-miss path above); the steady-state launch path just reads it.
        int32_t gemmNBlock = fc.gemmNBlock;

        int32_t kNumV = fc.kNum;
        int32_t nNum256V = fc.nNum256;
        uint32_t fusedBlockDim = static_cast<uint32_t>(aicCoreNum);
        aclrtStream gmsqStream = c10_npu::getCurrentNPUStream().stream();
        at_npu::native::OpCommand opCmd;
        opCmd.Name("grouped_matmul_situ_quant_v2");
        opCmd.SetCustomHandler(
            [gmsqStream, fusedBlockDim, x, wPtrTf, scPtrTf, wInt8, accWs, scaleF32, x_scale, y,
             y_scale, group_list, E32f, kNumV, nNum256V, gemmNBlock, NP32f, N32f, N232f, K32f,
             C32f, glType32, betaF, invBeta, hasLinear, lbF, invLb, skipUnpack]() -> int {
                gmsq2_fused_256_impl(fusedBlockDim, gmsqStream, x.data_ptr(), wPtrTf.data_ptr(),
                                     scPtrTf.data_ptr(), wInt8.data_ptr(), accWs.data_ptr(),
                                     scaleF32.data_ptr(), x_scale.data_ptr(), y.data_ptr(),
                                     y_scale.data_ptr(), group_list.data_ptr(), E32f, kNumV,
                                     nNum256V, gemmNBlock, NP32f, N32f, N232f, K32f, C32f,
                                     glType32, betaF, invBeta, hasLinear, lbF, invLb,
                                     skipUnpack);
                return 0;
            });
        opCmd.Run();
        return {y, y_scale};
    }

    // The fused single-launch path is the only path: TORCH_CHECK(N2 %% GMSQ2_BN == 0)
    // above (GMSQ2_BN=128) forces N %% 256 == 0, and fusedEligible gates nNum256 <= 128.
    // No 4-kernel fallback exists in this MMImplType delivery.
    TORCH_CHECK(fusedEligible,
                "unsupported shape for fused kernel: N must be a multiple of 256 and N/256 <= 128");
    return {y, y_scale};
}

}  // namespace vllm_ascend
