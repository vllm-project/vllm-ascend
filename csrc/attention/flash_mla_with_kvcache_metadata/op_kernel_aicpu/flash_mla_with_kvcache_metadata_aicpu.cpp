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

#include "flash_mla_with_kvcache_metadata_aicpu.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>
#include "log.h"
#include "status.h"
#include "../../a5_mla_common/op_kernel/aicpu_common.h"
#include "flash_mla_with_kvcache_metadata.h"

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

    success = CheckMetadataCapacity(splitRes.sectionNum);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Metadata buffer capacity insufficient!");

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
    GetAttrValueOpt(ctx, "is_c8", isC8_);

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
    KERNEL_CHECK_FALSE(CheckAttrs(), false, "Check attrs failed");
    KERNEL_CHECK_FALSE(CheckActualKvSeq(), false, "Check kv sequence failed");
    // batch size 由必传的 cacheSeqlens 长度推导（每 batch 一项），batch_size attr 已移除
    KERNEL_CHECK_FALSE(!actualSeqlenKv_.empty(), false, "cacheSeqlens must be provided to derive batch_size");
    KERNEL_CHECK_FALSE(CheckActualQuerySeq(static_cast<int64_t>(actualSeqlenKv_.size())), false,
                       "Check query sequence failed");
    return true;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::CheckAttrs()
{
    KERNEL_CHECK_FALSE(numHeadsQ_ > 0, false, "numHeadsQ must be greater than 0, but got %d", numHeadsQ_);
    KERNEL_CHECK_FALSE(numHeadsKv_ == MLA_KV_HEADS, false, "MLA kv head num must be %ld, but got %d", MLA_KV_HEADS,
                       numHeadsKv_);
    KERNEL_CHECK_FALSE(headDimQk_ == MLA_HEAD_DIM_QK, false,
                       "MLA headDimQk (nope 512 + rope 64) must be %ld, but got %d", MLA_HEAD_DIM_QK, headDimQk_);
    KERNEL_CHECK_FALSE(headDimV_ == MLA_HEAD_DIM_V, false, "MLA headDimV (value/output width) must be %ld, but got %d",
                       MLA_HEAD_DIM_V, headDimV_);
    KERNEL_CHECK_FALSE((maskMode_ == 0 || maskMode_ == 3), false,
                       "maskMode only supports 0(NO_MASK), 3(RIGHT_DOWN), but got %d", maskMode_);
    KERNEL_CHECK_FALSE((layoutQ_ == "TND" || layoutQ_ == "BNSD" || layoutQ_ == "BSND"), false,
                       "layoutQ only supports TND, BNSD, BSND, but got %s", layoutQ_.c_str());
    // 核数由 host 平台信息（硬件）透传，仅做基本合法性检查；buffer 是否够用由
    // CheckMetadataCapacity 在实际 sectionNum 出来后把关
    KERNEL_CHECK_FALSE(aicCoreNum_ > 0, false, "aicCoreNum must be positive, but got %d", aicCoreNum_);
    KERNEL_CHECK_FALSE(aivCoreNum_ > 0, false, "aivCoreNum must be positive, but got %d", aivCoreNum_);
    return true;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::CheckMetadataCapacity(uint32_t sectionNum)
{
    // 布局：[16-word HEAD][sectionNum * aicCoreNum * 16 FA][sectionNum * aivCoreNum * 16 FD]，
    // 核数为 host 平台信息透传的硬件值
    int64_t needWords = (1 + static_cast<int64_t>(sectionNum) * (aicCoreNum_ + aivCoreNum_)) *
                        static_cast<int64_t>(HEAD_METADATA_STRIDE);
    KERNEL_CHECK_FALSE(metadata_->NumElements() >= needWords, false,
                       "metadata buffer too small: %ld words, but sectionNum %u needs %ld words",
                       metadata_->NumElements(), sectionNum, needWords);
    return true;
}

bool FlashMlaWithKvcacheMetadataCpuKernel::CheckActualQuerySeq(int64_t batchSize)
{
    isActualSeqlenQAccum_ = false;
    actualSeqlenQ_.clear();
    std::vector<int64_t> cuSeqlensQ{};
    std::vector<int64_t> sequsedQ{};

    cuSeqlensQ = GetTensorDataAsInt64(cuSeqlensQ_);
    sequsedQ = GetTensorDataAsInt64(sequsedQ_);

    // batch 一致性：cuSeqlensQ 为 (batch+1,)，sequsedQ 为 (batch,)
    if (!cuSeqlensQ.empty() && static_cast<int64_t>(cuSeqlensQ.size()) != batchSize + 1) {
        KERNEL_LOG_ERROR("cuSeqlensQ shape must be (batchSize+1,)=(%ld,), but got %zu", batchSize + 1,
                         cuSeqlensQ.size());
        return false;
    }
    if (!sequsedQ.empty() && static_cast<int64_t>(sequsedQ.size()) != batchSize) {
        KERNEL_LOG_ERROR("sequsedQ shape must be (batchSize,)=(%ld,), but got %zu", batchSize, sequsedQ.size());
        return false;
    }

    // layout 存在性规则：TND 必传 cuSeqlensQ，非 TND 禁传；非 TND 时 maxSeqlenQ 与 sequsedQ 至少提供一个
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ.empty()) {
            KERNEL_LOG_ERROR("When layoutQ is TND, cuSeqlensQ should be provided, but got empty");
            return false;
        }
    } else if (!cuSeqlensQ.empty()) {
        KERNEL_LOG_ERROR("When layoutQ is not TND, cuSeqlensQ should not be provided, but got non-empty");
        return false;
    }

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
    param.mBaseSize = isC8_ ? HEAD_M_BASE_SIZE_MLA_C8 : HEAD_M_BASE_SIZE_MLA;
    param.s2BaseSize = isC8_ ? HEAD_S2_BASE_SIZE_MLA_C8 : HEAD_S2_BASE_SIZE_MLA;
    // Long-context replicated-Q decode: keep the 96 M tiles in one stream-K
    // schedule instead of restarting the CV pipeline for four L2 sections.
    // This is tuned for 128K + 1K generation with DCP8 and a shared prefix.
    // It remains correct for unshared pages; other shapes retain L2 sectioning.
    constexpr uint32_t LONG_DECODE_BATCH = 16;
    constexpr uint32_t LONG_DECODE_HEADS = 96;
    constexpr uint32_t LONG_DECODE_QUERY = 4;
    constexpr int64_t LONG_DECODE_KV_MIN = 16 * 1024;
    constexpr int64_t LONG_DECODE_KV_MAX = LONG_DECODE_KV_MIN + 2 * HEAD_S2_BASE_SIZE_MLA_C8;
    const bool longC8Decode = isC8_ && maskMode_ == 0 && numHeadsQ_ == LONG_DECODE_HEADS &&
        maxSeqlenQ_ == LONG_DECODE_QUERY && actualSeqlenKv_.size() == LONG_DECODE_BATCH &&
        std::all_of(actualSeqlenKv_.begin(), actualSeqlenKv_.end(), [](int64_t length) {
            return length >= LONG_DECODE_KV_MIN && length <= LONG_DECODE_KV_MAX;
        });
    // M 方向总行数 = g * S1 = (N1/N2) * maxSeqlenQ。若总行数 <= 2*mBaseSize（M 最多 2 块），
    // 多 section 切分无收益，仅增加 metadata/re-init/SyncAll 开销，故关闭。
    {
        uint32_t g = numHeadsQ_ / numHeadsKv_;
        if (g * maxSeqlenQ_ <= param.mBaseSize * 2 || longC8Decode) {
            param.l2Byte = 0U; // 0: disable section splitting
        } else {
            param.l2Byte = 96U * 1024U * 1024U; // 96MB
        }
    }
    param.fdTolerance = 10; // 10: tolerance block
    param.fdLeastBlock = 3; // 3: least block
    // 允许 SectionStreamK 沿 KV 轴跨核切分；是否采用 FD 由代价模型与 fdTolerance/fdLeastBlock 决定。
    // 分片结果写入 workspace，由 AIV 在 section 的 FA 阶段结束后执行 FD 规约。
    param.fdOn = true;
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
    baseInfo.headDimQk = static_cast<uint32_t>(headDimQk_);
    baseInfo.headDimV = static_cast<uint32_t>(headDimV_);
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
    baseInfo.queryType = isC8_ ? load_balance::DataType::INT8 : load_balance::DataType::FP16;
    baseInfo.kvType = isC8_ ? load_balance::DataType::INT8 : load_balance::DataType::FP16;
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
    detail::FaMetadata faMetadata(metadata_->GetData(), splitRes.sectionNum, static_cast<uint32_t>(aicCoreNum_),
                                  static_cast<uint32_t>(aivCoreNum_));
    faMetadata.Clear(); // set to all 0

    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    // Publish the selected tile sizes for the matching BF16 or C8 kernel.
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, param.mBaseSize);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, param.s2BaseSize);
    faMetadata.SetHeadMetadata(HEAD_AIC_NUM_INDEX, static_cast<FA_METADATA_T>(aicCoreNum_));
    faMetadata.SetHeadMetadata(HEAD_AIV_NUM_INDEX, static_cast<FA_METADATA_T>(aivCoreNum_));
    faMetadata.SetHeadMetadata(HEAD_OUTPUT_LAYOUT_INDEX,
                               static_cast<FA_METADATA_T>(load_balance::OutputLayout::BN2_S1G));
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
