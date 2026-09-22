/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GroupedMatmulSituQuant op_host.
// Exact-MSD GMM1 (A8W4) + SiTU + per-token INT8 quant, single operator.
// ND and native packed INT8-NZ weights are read directly from caller storage.
// Runtime shape support follows alignment, metadata/UB capacity and arithmetic bounds.
//
// DEVICE-SIDE group_list parsing (this revision):
//   The fused single-launch path performs NO host-side group_list parse. The
//   group_list device tensor is normalized to contiguous int64 on-device, then
//   parsed by the kernel every forward (mE / startRow / mPadOff / sumMPad / MValid are
//   derived in-kernel from GM). This makes the operator correct for DYNAMIC
//   production MoE group_list (new tensor per forward OR in-place value update)
//   without host-side value copies or synchronization.
//
//   Persistent metadata entries are keyed by device, full weight/scale pointer
//   and format sets, K, N and group_list_type. Neither capacity C nor group_list
//   identity/values are part of this key. Every layer must be warmed up eagerly:
//   metadata misses during graph capture are rejected before conversion/H2D.
//
//   Fixed MSD scratch is separate for each device/stream/K/N combination. The framework may
//   allocate scratch during capture, including on its internal side stream.
//   Graphs sharing a capture stream's scratch must replay serially; concurrent
//   replay is not supported by this reuse scheme. Weights/scales remain static
//   after warmup. Every accepted weight is read from the original storage.
//
//   This host has no W8 fallback or weight materialization path.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/core/Storage.h>

#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "grouped_matmul_situ_quant.h"

namespace vllm_ascend {

// Exact MSD device kernel launcher.
void gmsq_msd_exact_impl(uint32_t blockDim, void *stream, void *x, void *wPtrTbl,
    void *scPtrTbl, void *packedA, void *rawAcc, void *xScale, void *y,
    void *yScale, void *groupList, int32_t E, int32_t K, int32_t N,
    int32_t C, int32_t glType, float beta, float invBeta, int32_t hasLinear,
    float linBeta, float invLinBeta, int32_t nzInput);

// x_scale uses fixed-size UB chunks. Bound the AIC expert-of-block table.
constexpr int64_t GMSQ_MAX_M_BLOCKS = 256;

constexpr int32_t GMSQ_BM = 128;
constexpr int32_t GMSQ_BN = 128;
constexpr int32_t GMSQ_BK = 64;
constexpr int64_t GMSQ_MAX_EXPERTS = 128;
constexpr int64_t GMSQ_EXACT_MAX_K = (1LL << 24) / 1024;

// ACL tensor format ids (see acl_base.h / torch_npu Format enum).
constexpr int64_t ACL_FMT_ND = 2;
constexpr int64_t ACL_FMT_NCHW = 0; // ordinary contiguous base-format storage
constexpr int64_t ACL_FMT_FRACTAL_NZ = 29;

// ---------------------------------------------------------------------------
// Persistent device-side metadata cache for the FUSED single-launch path.
// Key = static weight configuration ONLY (weights are static during inference):
//   fullWPtrs/fullSPtrs — data_ptr of every expert's weight / weight_scale
//   device             — owning NPU
//   weight/scale format sets and K, N — static tensor layout
//   glType             — group_list interpretation mode (0 cumsum / 1 count)
// group_list identity is deliberately NOT in the key (it changes every forward
// in production MoE; the kernel parses it on-device each forward).
// ---------------------------------------------------------------------------
struct FusedMetaCache {
    // key
    int64_t device = -1;
    std::vector<uintptr_t> fullWPtrs;
    std::vector<uintptr_t> fullSPtrs;
    std::vector<int64_t> weightFormats;
    std::vector<int64_t> scaleFormats;
    int64_t K = 0;
    int64_t N = 0;
    int64_t glType = 0;
    // Only scale tensors may have a one-time ND conversion. Weight pointers
    // always refer to the caller's original ND or native packed INT8-NZ storage.
    bool srcIsNz = false;
    std::vector<at::Tensor> ndScales;
    // Strong refs to ORIGINAL weight/weight_scale tensors (all E) so the
    // allocator cannot reuse their data_ptr while this cache is alive.
    std::vector<at::Tensor> weightRefs;
    std::vector<at::Tensor> scaleRefs;
    // device tensors (static per weight config)
    at::Tensor tbl;     // int32 [4*E] = wPtrV(2E) + scPtrV(2E)
    int64_t scOff = 0;
};

// Same shape shares one pair across experts, layers, and ND/NZ layouts on a
// stream. Different K/N pairs never replace existing graph-visible addresses.
struct FusedScratchCache {
    int64_t device = -1;
    aclrtStream stream = nullptr;
    int64_t K = 0;
    int64_t N = 0;
    at::Tensor msdPackedA;
    at::Tensor msdRawAcc;
};

static std::mutex g_fusedCacheMutex;
static std::vector<std::unique_ptr<FusedMetaCache>> g_fusedMetaEntries;
static std::vector<std::unique_ptr<FusedScratchCache>> g_fusedScratchEntries;

// Mirrors GmsqFusedAivKernel256::Init(rawInt32=true). All rows are already
// 32-byte aligned; metadata and the scalar reduction outputs are rounded up.
// Sigmoid explicitly reuses aBuf_, so it requires no hidden UB stack space.
static int64_t MsdUbBytes(int64_t experts, int64_t N2)
{
    const auto align32 = [](int64_t bytes) { return (bytes + 31) / 32 * 32; };
    const int64_t packing = 2 * 4096 + 16384 + 2 * 8192;
    const int64_t rawAndScale = 4 * N2 + std::max<int64_t>(4 * N2, 8192);
    const int64_t rows = 4 * 4 * N2 + 2 * N2 + N2;
    const int64_t scalarAndReduction = 32 + 32 + 4096 + align32(((N2 + 1023) / 1024) * 4);
    return packing + rawAndScale + rows + scalarAndReduction + align32((3 * experts + 4) * 4);
}

// Scale-only conversion helper; weight data must never be passed here.
static at::Tensor EnsureNdScale(const at::Tensor &t, std::vector<at::Tensor> &converted)
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

std::tuple<at::Tensor, at::Tensor> grouped_matmul_situ_quant(
    const at::Tensor &x, at::TensorList weight, at::TensorList weight_scale,
    const at::Tensor &x_scale, const at::Tensor &group_list, at::TensorList weight_assist_matrix,
    double beta, std::optional<double> linear_beta, int64_t group_list_type)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(x.scalar_type() == at::kChar, "x must be int8");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "x_scale must be float32");
    TORCH_CHECK(x_scale.is_contiguous() && x_scale.numel() >= x.size(0),
                "x_scale must be contiguous with at least x.size(0) elements");
    TORCH_CHECK(!weight.empty(), "weight list must be non-empty");
    TORCH_CHECK(weight.size() == weight_scale.size(), "weight/weight_scale size mismatch");
    TORCH_CHECK(weight_assist_matrix.empty() || weight_assist_matrix.size() == weight.size(),
                "weight_assist_matrix size mismatch");
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1, "group_list_type must be 0 or 1");
    TORCH_CHECK(group_list.dim() == 1, "group_list must be 1D");
    TORCH_CHECK(group_list.numel() >= static_cast<int64_t>(weight.size()),
                "group_list numel must be at least the number of experts");
    TORCH_CHECK(group_list.scalar_type() == at::kLong || group_list.scalar_type() == at::kInt ||
                    group_list.scalar_type() == at::kFloat,
                "group_list dtype must be int64, int32 or float32");

    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on an NPU device");
    const int64_t device = x.get_device();
    TORCH_CHECK(c10_npu::current_device() == device,
                "grouped_matmul_situ_quant: current NPU device must match x.device");
    TORCH_CHECK(x_scale.device() == x.device() && group_list.device() == x.device(),
                "x_scale and group_list must be on x.device");
    for (size_t e = 0; e < weight.size(); ++e) {
        TORCH_CHECK(weight[e].device() == x.device() && weight_scale[e].device() == x.device(),
                    "weight and weight_scale must be on x.device (expert ", e, ")");
    }
    for (const auto &assist : weight_assist_matrix) {
        TORCH_CHECK(assist.device() == x.device(), "weight_assist_matrix must be on x.device");
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t aivCoreNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAiv());
    int32_t aicCoreNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAic());
    TORCH_CHECK(aivCoreNum > 0 && aicCoreNum > 0, "failed to query core numbers");

    int64_t C = x.sizes()[0];      // x rows (capacity, >= M_valid)
    int64_t K = x.sizes()[1];
    int64_t experts = static_cast<int64_t>(weight.size());
    TORCH_CHECK(experts <= GMSQ_MAX_EXPERTS, "grouped_matmul_situ_quant supports at most 128 experts");
    TORCH_CHECK(weight[0].dim() == 2, "weight[e] must be [K, N/8]");
    int64_t NP = weight[0].sizes()[1];  // packed N/8 per row
    TORCH_CHECK(NP > 0 && NP <= std::numeric_limits<int32_t>::max() / 8,
                "packed N/8 must be positive and fit the kernel int32 shape ABI");
    int64_t N = NP * 8;
    int64_t N2 = N / 2;
    TORCH_CHECK(K > 0 && K % GMSQ_BK == 0, "unsupported K: K must be a positive multiple of 64");
    TORCH_CHECK(N % GMSQ_BN == 0, "N must be a multiple of 128");
    TORCH_CHECK(N2 % GMSQ_BN == 0, "N/2 must be a multiple of 128");
    TORCH_CHECK(weight[0].scalar_type() == at::kInt, "weight must be int32 packed");
    TORCH_CHECK(weight_scale[0].scalar_type() == at::kLong, "weight_scale must be int64 carrier");
    for (int64_t e = 0; e < experts; ++e) {
        TORCH_CHECK(weight[e].dim() == 2 && weight[e].size(0) == K && weight[e].size(1) == NP &&
                        weight[e].scalar_type() == at::kInt,
                    "weight[e] must be int32 packed [K,N/8] with a common shape; expert ", e);
        TORCH_CHECK(weight_scale[e].scalar_type() == at::kLong && weight_scale[e].numel() >= N,
                    "weight_scale[e] must contain at least N int64 carriers; expert ", e);
    }

    // This is a computation limit, independent of the ND/NZ storage contract.
    // (low-aux)+16*high has integer intermediates of magnitude <=1024*K;
    // all are exactly representable in FP32 through K=16384.
    TORCH_CHECK(K <= GMSQ_EXACT_MAX_K,
                "unsupported K for exact FP32 MSD reconstruction: 1024*K must be <=2^24; got K=", K);
    uint64_t ubBytes = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const int64_t requiredUb = MsdUbBytes(experts, N2);
    TORCH_CHECK(ubBytes > 0 && static_cast<uint64_t>(requiredUb) <= ubBytes,
                "unsupported N/E for the row epilogue: requires ", requiredUb,
                " UB bytes, device provides ", ubBytes, "; E=", experts, ", N=", N);

    const int64_t activeUpperBound = std::min(experts, C);
    const int64_t paddedBlockUpperBound = activeUpperBound + (C - activeUpperBound) / GMSQ_BM;
    TORCH_CHECK(paddedBlockUpperBound <= GMSQ_MAX_M_BLOCKS,
                "grouped_matmul_situ_quant: capacity/expert combination exceeds the ",
                GMSQ_MAX_M_BLOCKS, " padded M-block metadata limit");

    auto devOptI8 = at::TensorOptions().device(x.device()).dtype(at::kChar);
    auto devOptI32 = at::TensorOptions().device(x.device()).dtype(at::kInt);
    auto devOptF32 = at::TensorOptions().device(x.device()).dtype(at::kFloat);

    // outputs: y [C, N2] int8, y_scale [C] fp32 (only first M_valid rows meaningful)
    at::Tensor y = at::empty({C, N2}, devOptI8);
    at::Tensor y_scale = at::empty({C}, devOptF32);

    // EP empty rank (M_valid == 0): x may be [0, K] with an all-zero group_list
    // (short requests under TP8+EP leave some ranks without routed tokens). All
    // shape/capacity validation is done above and the outputs are already empty with
    // the right shapes — return without building caches, reading group_list or
    // launching the kernel. A C > 0 input whose device-side group_list is all
    // zero stays on the normal launch path; the kernel tolerates
    // sum(group_list) == 0 (the P54 waveBatch clamp keeps batches at 0, so no
    // tile is scheduled and no cross-core flag pair is exercised).
    if (C == 0) {
        return {y, y_scale};
    }

    // AllToAll histograms may arrive as float32. Normalize on the input device
    // and current stream on EVERY call: counts change between graph replays.
    // Contiguous int64 input is reused without a copy. Float inputs must carry
    // finite, nonnegative integer counts (or valid cumulative counts for type 0),
    // with total valid rows <= C; value validation is the caller's contract.
    // Do not read values on the CPU or cache this converted tensor.
    at::Tensor kernelGroupList = group_list;
    if (kernelGroupList.scalar_type() != at::kLong) {
        kernelGroupList = kernelGroupList.to(at::kLong);
    }
    kernelGroupList = kernelGroupList.contiguous();

    float betaF = static_cast<float>(beta);
    float invBeta = 1.0f / betaF;
    int32_t hasLinear = linear_beta.has_value() ? 1 : 0;
    float lbF = hasLinear ? static_cast<float>(*linear_beta) : 1.0f;
    float invLb = hasLinear ? 1.0f / lbF : 1.0f;

    // ---- FUSED single-launch path (device-side group_list parse) ----
    {
        const aclrtStream gmsqStream = c10_npu::getCurrentNPUStream().stream();
        aclmdlRICaptureStatus captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        aclmdlRI captureModel = nullptr;
        const aclError captureError = aclmdlRICaptureGetInfo(gmsqStream, &captureStatus, &captureModel);
        TORCH_CHECK(captureError == ACL_SUCCESS,
                    "grouped_matmul_situ_quant: capture status query failed: ", captureError);
        TORCH_CHECK(captureStatus != ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
                    "grouped_matmul_situ_quant: cannot enqueue on an invalidated graph capture");
        const bool capturing = captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
        // Cheap host-side key (no H2D/D2H, no sync): full pointer sets + shapes +
        // gl_type. group_list is deliberately NOT in the key.
        std::vector<uintptr_t> fullWPtrs(static_cast<size_t>(experts));
        std::vector<uintptr_t> fullSPtrs(static_cast<size_t>(experts));
        std::vector<int64_t> weightFormats(static_cast<size_t>(experts));
        std::vector<int64_t> scaleFormats(static_cast<size_t>(experts));
        int64_t nzCount = 0;
        for (int64_t e = 0; e < experts; e++) {
            TORCH_CHECK(weight[e].data_ptr() != nullptr && weight_scale[e].data_ptr() != nullptr,
                        "weight and weight_scale must have non-null storage");
            fullWPtrs[static_cast<size_t>(e)] = reinterpret_cast<uintptr_t>(weight[e].data_ptr());
            fullSPtrs[static_cast<size_t>(e)] = reinterpret_cast<uintptr_t>(weight_scale[e].data_ptr());
            int64_t wfmt = at_npu::native::get_npu_format(weight[e]);
            TORCH_CHECK(wfmt == ACL_FMT_ND || wfmt == ACL_FMT_NCHW || wfmt == ACL_FMT_FRACTAL_NZ,
                        "grouped_matmul_situ_quant: weight[e] must use contiguous ND (format 0/2) "
                        "or native packed INT8-NZ viewed as INT32 (format 29); weight conversion "
                        "is not performed, got format ", wfmt, " for expert ", e);
            if (wfmt != ACL_FMT_FRACTAL_NZ) {
                // Validate on cache hits too: a new view can reuse the same
                // pointer and shape while changing its logical strides.
                TORCH_CHECK(weight[e].is_contiguous(), "weight[e] must be contiguous (ND)");
            }
            if (wfmt == ACL_FMT_FRACTAL_NZ) {
                // The logical [K,N/8] INT32 carrier must retain row-major
                // strides from native INT8-NZ -> view(INT32). Expert unbind
                // slices may have nonzero storage_offset; data_ptr includes it.
                // Reject transposed/strided views without materializing them.
                TORCH_CHECK(weight[e].is_contiguous(),
                            "grouped_matmul_situ_quant: native packed INT8-NZ viewed as INT32 "
                            "must have a contiguous [K,N/8] carrier with strides [N/8,1]; "
                            "strided views are unsupported and no weight copy is performed; expert ", e);
                // dtype views share storage, so the storage descriptor retains
                // the packed INT8 dtype and NZ geometry through view(INT32).
                // The checks above require every weight to be on x's NPU;
                // torch_npu owns these storages as NPUStorageImpl. Read the
                // installed header's public descriptor directly: NPUBridge's
                // equivalent accessor is not exported by this torch_npu build.
                // This is host metadata only, with no materialization/cast.
                const auto *storage = static_cast<const torch_npu::NPUStorageImpl *>(
                    weight[e].storage().unsafeGetStorageImpl());
                const auto &desc = storage->npu_desc_;
                TORCH_CHECK(desc.data_type_ == caffe2::TypeMeta::Make<int8_t>(),
                            "grouped_matmul_situ_quant: NZ weight must originate from native "
                            "packed INT8 storage before view(INT32), expert ", e);
                const auto &shape = desc.storage_sizes_;
                const size_t rank = shape.size();
                TORCH_CHECK(rank >= 4 && shape[rank - 4] == N / 64 &&
                                shape[rank - 3] == K / 16 && shape[rank - 2] == 16 &&
                                shape[rank - 1] == 32,
                            "grouped_matmul_situ_quant: native packed INT8-NZ storage must have "
                            "trailing geometry [N/64,K/16,16,32], expert ", e);
            }
            weightFormats[static_cast<size_t>(e)] = wfmt;
            scaleFormats[static_cast<size_t>(e)] = at_npu::native::get_npu_format(weight_scale[e]);
            nzCount += (wfmt == ACL_FMT_FRACTAL_NZ) ? 1 : 0;
        }
        TORCH_CHECK(nzCount == 0 || nzCount == experts,
                    "grouped_matmul_situ_quant: mixed ND/FRACTAL_NZ weight experts are not "
                    "supported; all experts must use the same format");

        const bool srcIsNz = (nzCount == experts);
        // Entries never move or expire; the mutex protects construction and
        // scratch growth. Published metadata is immutable and shared across
        // eager and capture streams on the same device.
        std::unique_lock<std::mutex> cacheLock(g_fusedCacheMutex);
        FusedMetaCache *meta = nullptr;
        for (const auto &entry : g_fusedMetaEntries) {
            if (entry->device == device && entry->K == K && entry->N == N &&
                entry->glType == group_list_type && entry->srcIsNz == srcIsNz &&
                entry->fullWPtrs == fullWPtrs && entry->fullSPtrs == fullSPtrs &&
                entry->weightFormats == weightFormats && entry->scaleFormats == scaleFormats) {
                meta = entry.get();
                break;
            }
        }

        if (meta == nullptr) {
            TORCH_CHECK(!capturing,
                        "grouped_matmul_situ_quant: metadata cache miss during graph capture; "
                        "warm up every weight/scale configuration eagerly on this device first "
                        "(pointer-table H2D and format conversion are forbidden during capture)");
            // Construct privately so a failed build cannot publish partial data.
            auto newEntry = std::make_unique<FusedMetaCache>();
            FusedMetaCache &fc = *newEntry;
            fc.device = device;
            fc.srcIsNz = srcIsNz;
            // pointer tables for ALL experts (expert index = table index)
            std::vector<uintptr_t> wPtrsAll;
            std::vector<uintptr_t> scPtrsAll;
            wPtrsAll.reserve(static_cast<size_t>(experts));
            scPtrsAll.reserve(static_cast<size_t>(experts));
            for (int64_t e = 0; e < experts; e++) {
                const at::Tensor &originalWeight = weight[static_cast<size_t>(e)];
                if (!srcIsNz) {
                    TORCH_CHECK(originalWeight.is_contiguous(), "weight[e] must be contiguous (ND)");
                }
                at::Tensor sNd = EnsureNdScale(weight_scale[static_cast<size_t>(e)], fc.ndScales);
                fc.weightRefs.push_back(weight[static_cast<size_t>(e)]);
                fc.scaleRefs.push_back(weight_scale[static_cast<size_t>(e)]);
                TORCH_CHECK(sNd.is_contiguous(), "weight_scale[e] must be contiguous (ND)");
                wPtrsAll.push_back(reinterpret_cast<uintptr_t>(originalWeight.data_ptr()));
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

            fc.fullWPtrs = std::move(fullWPtrs);
            fc.fullSPtrs = std::move(fullSPtrs);
            fc.weightFormats = std::move(weightFormats);
            fc.scaleFormats = std::move(scaleFormats);
            fc.K = K; fc.N = N; fc.glType = group_list_type;
            fc.tbl = tbl;
            fc.scOff = scOff;
            meta = newEntry.get();
            g_fusedMetaEntries.push_back(std::move(newEntry));
        }

        const FusedMetaCache &fc = *meta;
        FusedScratchCache *scratch = nullptr;
        for (const auto &entry : g_fusedScratchEntries) {
            if (entry->device == device && entry->stream == gmsqStream && entry->K == K && entry->N == N) {
                scratch = entry.get();
                break;
            }
        }
        if (scratch == nullptr) {
            auto entry = std::make_unique<FusedScratchCache>();
            entry->device = device;
            entry->stream = gmsqStream;
            entry->K = K;
            entry->N = N;
            scratch = entry.get();
            g_fusedScratchEntries.push_back(std::move(entry));
        }


        // Fixed scratch for this device/stream/K/N. E, C, routing, layer and
        // weight layout do not change its size or its recorded graph address.
        {
            constexpr int64_t msdSlots = 8;
            constexpr int64_t msdRawRows = 2 * GMSQ_BM + 16;
            const int64_t packedBytes = msdSlots * msdRawRows * K / 2;
            const int64_t rawBytes = msdSlots * msdRawRows * N * sizeof(int32_t);
            if (!scratch->msdPackedA.defined()) {
                // Publish only after both allocations succeed, so an OOM
                // cannot leave a half-initialized reusable scratch pair.
                at::Tensor newPackedA = at::empty({packedBytes}, devOptI8);
                at::Tensor newRawAcc = at::empty({rawBytes}, devOptI8);
                scratch->msdPackedA = std::move(newPackedA);
                scratch->msdRawAcc = std::move(newRawAcc);
            }
            at::Tensor packedA = scratch->msdPackedA;
            at::Tensor rawAcc = scratch->msdRawAcc;
            at::Tensor wPtrTf = fc.tbl.narrow(0, 0, experts * 2);
            at::Tensor scPtrTf = fc.tbl.narrow(0, fc.scOff, experts * 2);
            cacheLock.unlock();
            void *xPtr = x.data_ptr();
            void *wPtr = wPtrTf.data_ptr();
            void *sPtr = scPtrTf.data_ptr();
            void *aPtr = packedA.data_ptr();
            void *cPtr = rawAcc.data_ptr();
            void *xsPtr = x_scale.data_ptr();
            void *yPtr = y.data_ptr();
            void *ysPtr = y_scale.data_ptr();
            void *glPtr = kernelGroupList.data_ptr();
            std::vector<c10::Storage> keepAlive;
            for (const at::Tensor &tensor : {x, wPtrTf, scPtrTf, packedA, rawAcc,
                                           x_scale, y, y_scale, kernelGroupList}) {
                keepAlive.emplace_back(tensor.storage());
            }
            std::function<int()> handler = [gmsqStream, aicCoreNum, xPtr, wPtr,
                sPtr, aPtr, cPtr, xsPtr, yPtr, ysPtr, glPtr, experts, K, N, C,
                group_list_type, betaF, invBeta, hasLinear, lbF, invLb, srcIsNz,
                keepAlive = std::move(keepAlive)]() -> int {
                (void)keepAlive;
                gmsq_msd_exact_impl(aicCoreNum, gmsqStream, xPtr, wPtr, sPtr,
                    aPtr, cPtr, xsPtr, yPtr, ysPtr, glPtr, experts, K, N, C,
                    group_list_type, betaF, invBeta, hasLinear, lbF, invLb, srcIsNz ? 1 : 0);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("grouped_matmul_situ_quant", handler,
                                               /*sync=*/false);
            return {y, y_scale};
        }

    }
}

}  // namespace vllm_ascend
