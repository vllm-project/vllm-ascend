/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_mla_with_kvcache_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>
#include "flash_mla_with_kvcache_metadata.h"
#include "flash_mla_with_kvcache_metadata_aicpu.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "../../flash_attn/op_host/fa_adjust_sinner_souter.h"

constexpr uint32_t FA_KERNEL_STATUS_OK = 0;
constexpr uint32_t FA_KERNEL_STATUS_PARAM_INVALID = 1;

using namespace optiling;

namespace aicpu {
uint32_t FlashMlaWithKvcacheMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    load_balance::SectionStreamKResult splitRes{};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");

    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    return FA_KERNEL_STATUS_OK;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cacheSeqlens_ = ctx.Input(static_cast<uint32_t>(ParamId::cacheSeqlens));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim_qk", headDimQk_) && GetAttrValue(ctx, "head_dim_v", headDimV_) &&
        GetAttrValue(ctx, "soc_version", socVersion_) && GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
        GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    KERNEL_CHECK_FALSE(requiredAttrs, false, "Missing Required attrs missing!");

    // attributes optional
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);

    KERNEL_CHECK_FALSE(ParamsCheck(), false, "Params check failed");
    return ParamsInit();
}

bool FlashMlaWithKvcacheMetadataCpuKernel::ParamsInit()
{
    InitDeviceInfo();
    InitBaseInfo();
    InitLoadBalanceParams();
    return true;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_FALSE(CheckActualQuerySeq(), false, "Check query sequence failed");
    KERNEL_CHECK_FALSE(CheckActualKvSeq(), false, "Check kv sequence failed");
    // batch size 由必传的 cacheSeqlens 长度推导（每 batch 一项），batch_size attr 已移除
    KERNEL_CHECK_FALSE(!actualSeqlenKv_.empty(), false, "cacheSeqlens must be provided to derive batch_size");
    return true;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::CheckActualQuerySeq()
{
    isActualSeqlenQAccum_ = false;
    actualSeqlenQ_.clear();
    std::vector<int64_t> cuSeqlensQ{};
    std::vector<int64_t> sequsedQ{};

    cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_);
    sequsedQ = GetTensorDataAsInt64(sequsedQ_);

    for (size_t i = 0; i < sequsedQ.size(); ++i) {
        if (sequsedQ[i] < 0) {
            KERNEL_LOG_ERROR("The elements of sequsedQ must be non-negative, but %zuth element is %ld", i, sequsedQ[i]);
            return false;
        }
    }

    if (!cuSeqlensQ.empty()) {
        if (cuSeqlensQ[0] != 0) {
            KERNEL_LOG_ERROR("The first element of cuSeqlensQ must be 0, but got %ld", cuSeqlensQ[0]);
            return false;
        }
    }

    for (size_t i = 1; i < cuSeqlensQ.size(); ++i) {
        if (cuSeqlensQ[i] < cuSeqlensQ[i - 1]) {
            KERNEL_LOG_ERROR(
                "The %zuth element of cuSeqlensQ must not be less than the %zuth element, but got %ld and %ld", i,
                i - 1, cuSeqlensQ[i], cuSeqlensQ[i - 1]);
            return false;
        }
    }

    if (!sequsedQ.empty()) {
        isActualSeqlenQAccum_ = false;
        actualSeqlenQ_ = sequsedQ;
    } else if (!cuSeqlensQ.empty()) {
        isActualSeqlenQAccum_ = true;
        actualSeqlenQ_.assign(cuSeqlensQ.begin() + 1, cuSeqlensQ.end());
    }

    return true;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::CheckActualKvSeq()
{
    // cacheSeqlens carries per-batch kv lengths (non-cumulative, no cumulative
    // branch): the scheduler must treat them as non-cumulative
    // (isCumulativeKvSeq = false), preserving the previously effective per-batch
    // kv length semantics.
    isActualSeqlenKvAccum_ = false;
    actualSeqlenKv_.clear();
    std::vector<int64_t> cacheSeqlens{};

    cacheSeqlens = GetTensorDataAsInt64(cacheSeqlens_);

    for (size_t i = 0; i < cacheSeqlens.size(); ++i) {
        if (cacheSeqlens[i] < 0) {
            KERNEL_LOG_ERROR("The elements of cacheSeqlens must be non-negative, but %zuth element is %ld", i,
                             cacheSeqlens[i]);
            return false;
        }
    }

    actualSeqlenKv_ = cacheSeqlens;
    return true;
}

void FlashMlaWithKvcacheMetadataCpuKernel::InitDeviceInfo()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
}

void FlashMlaWithKvcacheMetadataCpuKernel::InitLoadBalanceParams()
{
    uint32_t qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BNSD;
    if (baseInfo.layoutQuery == load_balance::Layout::BSH || baseInfo.layoutQuery == load_balance::Layout::BSND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BSH;
    } else if (baseInfo.layoutQuery == load_balance::Layout::TND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_TND;
    }
    uint32_t gSize = static_cast<uint32_t>(numHeadsQ_ / numHeadsKv_);
    optiling::flash_attn::fa_tiling_util::AdjustSinnerAndSouter(baseInfo.headDimQk, gSize, maxSeqlenQ_, maxSeqlenKv_,
                                                                baseInfo.sparseMode, baseInfo.preToken,
                                                                baseInfo.nextToken, qlayout, mBaseSize_, s2BaseSize_);
    mBaseSize_ *= (aivCoreNum_ / aicCoreNum_);
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 96U * 1024U * 1024U; // 96: 96MB, 1024: Mb2Kb, 1024:Kb2Mb
    param.fdTolerance = 10;             // 10: tolerance block
    param.fdLeastBlock = 3;             // 3: least block
    // SectionStreamK 的 FD 调度 = 长 KV 的 s2 核间切分 + workspace 归约（decode 提速）。flash_mla
    // kernel 当前只在上板验证了 FA 单核 s2 迭代路径（fa-portion），FD 归约路径未经历跨 section
    // 真机校验（isFd 路由修正后 b8/kvS=32768 上板挂起）。暂关闭 FD 调度 → 所有 (bn2,m) 块在单核
    // 内顺序覆盖全部 s2（cachedS2LoopTimes），无需核间归约，LSE/out 全程在 FA 路径完成；
    // 正确性优先，性能留待后续按需恢复（与 flash_attn 运行时开关一致后可回退本行）。
    param.fdOn = false;
}

void FlashMlaWithKvcacheMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = actualSeqlenKv_.size();
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    // head_dim_qk = q/k_cache 最后维 576（nope 512 + rope 64，meta.txt 约束）；head_dim_v = 512
    // 为 value/输出宽。调度（AdjustSinnerAndSouter）用 head_dim_qk；head_dim_v 由主算子
    // head_dim_v attr 独立约束（rebase 后新版 SectionStreamK 的 s2vCost 切分模型独立读取
    // GetHeadDimQk()/GetHeadDimV()，故同时写入，避免退化为默认 64）。
    load_balance::SparseMode maskMode = load_balance::SparseMode::BUTT;
    if (maskMode_ != 0) {
        maskMode = static_cast<load_balance::SparseMode>(maskMode_);
    }
    baseInfo.attenMaskFlag = (maskMode != load_balance::SparseMode::BUTT);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode);
    // window attrs removed: window is unlimited, preToken/nextToken are
    // unconditionally UINT32_MAX (keeps the effective values of the previous
    // always-(-1) window attrs path).
    baseInfo.preToken = std::numeric_limits<uint32_t>::max();
    baseInfo.nextToken = std::numeric_limits<uint32_t>::max();
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeKvSeq = isActualSeqlenKvAccum_;
    baseInfo.actualKvSeqSize = actualSeqlenKv_;
    baseInfo.isCumulativeQuerySeq = isActualSeqlenQAccum_;
    baseInfo.actualQuerySeqSize = actualSeqlenQ_;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult &splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult &splitRes)
{
    detail::FaMetadata faMetadata(metadata_->GetData(), splitRes.sectionNum);
    faMetadata.Clear(); // set to all 0

    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    // D18 fallback (unconditional, D29/R6): HEAD_M_BASE_SIZE / HEAD_S2_BASE_SIZE are FORCED to the MLA
    // kernel template base sizes 64/128 (D17) and do NOT trust the value derived by
    // fa_tiling_util::AdjustSinnerAndSouter + the aiv/aic ratio scaling above. The invariant source is
    // the kernel-side static_assert mBaseSize==64 && s2BaseSize==128 (fia_block_cube_noquant_mla.h:106,
    // ported as flash_mla_block_cube_noquant_mla.h) for D=576; this forced head-field write keeps the
    // AICPU producer in agreement. The per-core split math (BalanceSchedule -> SectionStreamK::Compute)
    // is intentionally UNCHANGED - only the two head fields are overridden. Trigger rule: if the Todo 13
    // byte-golden / on-board smoke shows SectionStreamK mis-splitting D=576, port FIA split_core_v2
    // per D18 and re-run.
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, HEAD_M_BASE_SIZE_MLA);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, HEAD_S2_BASE_SIZE_MLA);
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }

    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)}; // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto &faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto &prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bn2Start = (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_START_INDEX, bn2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_END_INDEX, faRes.bNEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.mEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }

        auto &fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN2_IDX_INDEX, fdRes.bNIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.mIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
    return true;
}

namespace {
static const char *kernelType = "FlashMlaWithKvcacheMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashMlaWithKvcacheMetadataCpuKernel);
} // namespace

} // namespace aicpu
