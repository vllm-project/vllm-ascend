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
 * \file flash_attn_metadata_aicpu.cpp
 * \brief
 */

#include "flash_attn_metadata_aicpu.h"
#include <cmath>
#include <cstdio>
#include <numeric>
#include <algorithm>
#include "log.h"
#include "status.h"
#include "flash_attn_grad_metadata_split.h"
#include "../../a5_mla_common/op_kernel/aicpu_common.h"
#include "../../flash_attn/op_host/fa_adjust_sinner_souter.h"

constexpr uint32_t FA_KERNEL_STATUS_OK = 0;
constexpr uint32_t FA_KERNEL_STATUS_PARAM_INVALID = 1;

using namespace optiling;

namespace aicpu {
uint32_t FlashAttnMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    load_balance::SectionStreamKResult splitRes{};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");

    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    // ===== FAG metadata start =====
    // FAG (Flash Attn Grad) metadata: only executed when is_grad_enabled=true.
    if (fagIsGradEnabled_) {
        InitFagParams();
        DoFagSparse();
        GenFagMetadata(splitRes.sectionNum);
    }
    // ===== FAG metadata end =====

    return FA_KERNEL_STATUS_OK;
}

bool FlashAttnMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    KERNEL_CHECK_FALSE(requiredAttrs, false, "Missing Required attrs missing!");

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    GetAttrValueOpt(ctx, "head_dim_v", headDimV_);
    // head_dim_v 未指定(-1)时, v 的 head_dim 等于 head_dim
    if (headDimV_ == -1) {
        headDimV_ = headDim_;
    }
    GetAttrValueOpt(ctx, "is_grad_enabled", fagIsGradEnabled_);

    KERNEL_CHECK_FALSE(ParamsCheck(), false, "Params check failed");
    return ParamsInit();
}

bool FlashAttnMetadataCpuKernel::ParamsInit()
{
    InitDeviceInfo();
    InitBaseInfo();
    InitLoadBalanceParams();
    return true;
}

bool FlashAttnMetadataCpuKernel::ParamsCheck()
{
    KERNEL_CHECK_FALSE(CheckActualQuerySeq(), false, "Check query sequence failed");
    KERNEL_CHECK_FALSE(CheckActualKvSeq(), false, "Check kv sequence failed");
    return true;
}

bool FlashAttnMetadataCpuKernel::CheckActualQuerySeq()
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

bool FlashAttnMetadataCpuKernel::CheckActualKvSeq()
{
    isActualSeqlenKvAccum_ = false;
    actualSeqlenKv_.clear();
    std::vector<int64_t> cuSeqlensKv{};
    std::vector<int64_t> sequsedKv{};

    cuSeqlensKv = GetTensorDataAsInt64(cuSeqlensKv_);
    sequsedKv = GetTensorDataAsInt64(sequsedKv_);

    for (size_t i = 0; i < sequsedKv.size(); ++i) {
        if (sequsedKv[i] < 0) {
            KERNEL_LOG_ERROR("The elements of sequsedKv must be non-negative, but %zuth element is %ld", i,
                             sequsedKv[i]);
            return false;
        }
    }

    if (!cuSeqlensKv.empty()) {
        if (cuSeqlensKv[0] != 0) {
            KERNEL_LOG_ERROR("The first element of cuSeqlensKv must be 0, but got %ld", cuSeqlensKv[0]);
            return false;
        }
    }

    for (size_t i = 1; i < cuSeqlensKv.size(); ++i) {
        if (cuSeqlensKv[i] < cuSeqlensKv[i - 1]) {
            KERNEL_LOG_ERROR(
                "The %zuth element of cuSeqlensKv must not be less than the %zuth element, but got %ld and %ld", i,
                i - 1, cuSeqlensKv[i], cuSeqlensKv[i - 1]);
            return false;
        }
    }

    if (!sequsedKv.empty()) {
        isActualSeqlenKvAccum_ = false;
        actualSeqlenKv_ = sequsedKv;
    } else if (!cuSeqlensKv.empty()) {
        isActualSeqlenKvAccum_ = true;
        actualSeqlenKv_.assign(cuSeqlensKv.begin() + 1, cuSeqlensKv.end());
    }

    return true;
}

void FlashAttnMetadataCpuKernel::InitDeviceInfo()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
}

void FlashAttnMetadataCpuKernel::InitLoadBalanceParams()
{
    uint32_t qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BNSD;
    if (baseInfo.layoutQuery == load_balance::Layout::BSH || baseInfo.layoutQuery == load_balance::Layout::BSND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_BSH;
    } else if (baseInfo.layoutQuery == load_balance::Layout::TND) {
        qlayout = optiling::flash_attn::fa_tiling_util::LAYOUT_TND;
    }
    uint32_t gSize = static_cast<uint32_t>(numHeadsQ_ / numHeadsKv_);
    optiling::flash_attn::fa_tiling_util::AdjustSinnerAndSouter(headDimV_, gSize, maxSeqlenQ_, maxSeqlenKv_, maskMode_,
                                                                baseInfo.preToken, baseInfo.nextToken, qlayout,
                                                                mBaseSize_, s2BaseSize_);
    mBaseSize_ *= (aivCoreNum_ / aicCoreNum_);
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 96U * 1024U * 1024U; // 96: 96MB, 1024: Mb2Kb, 1024:Kb2Mb
    param.fdTolerance = 10;             // 10: tolerance block
    param.fdLeastBlock = 3;             // 3: least block
    param.fdOn = true;
    param.outputLayout = load_balance::OutputLayout::BN2_S1G;
}

void FlashAttnMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = (actualSeqlenQ_.empty()) ? batchSize_ : actualSeqlenQ_.size();
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.headDimQk = headDim_;
    baseInfo.headDimV = headDimV_;
    load_balance::SparseMode maskMode = load_balance::SparseMode::BUTT;
    if (maskMode_ != 0) {
        maskMode = static_cast<load_balance::SparseMode>(maskMode_);
    }
    baseInfo.attenMaskFlag = (maskMode != load_balance::SparseMode::BUTT);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode);
    baseInfo.preToken = (winLeft_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winLeft_);
    baseInfo.nextToken = (winRight_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winRight_);
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = load_balance::ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    baseInfo.isCumulativeKvSeq = isActualSeqlenKvAccum_;
    baseInfo.actualKvSeqSize = actualSeqlenKv_;
    baseInfo.isCumulativeQuerySeq = isActualSeqlenQAccum_;
    baseInfo.actualQuerySeqSize = actualSeqlenQ_;
}

bool FlashAttnMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult &splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool FlashAttnMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult &splitRes)
{
    detail::FaMetadata faMetadata(aicCoreNum_, aivCoreNum_, splitRes.sectionNum, metadata_->GetData());
    faMetadata.Clear(); // set to all 0

    SetMetadataHead(splitRes, faMetadata);
    SetMetadataFa(splitRes, faMetadata);
    SetMetadataFd(splitRes, faMetadata);
    return true;
}

void FlashAttnMetadataCpuKernel::SetMetadataHead(const load_balance::SectionStreamKResult &splitRes,
                                                 optiling::detail::FaMetadata &faMetadata)
{
    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }
    faMetadata.SetHeadMetadata(HEAD_AIC_NUM_INDEX, aicCoreNum_);
    faMetadata.SetHeadMetadata(HEAD_AIV_NUM_INDEX, aivCoreNum_);
    faMetadata.SetHeadMetadata(HEAD_OUTPUT_LAYOUT_INDEX, static_cast<FA_METADATA_T>(param.outputLayout));
}

void FlashAttnMetadataCpuKernel::SetMetadataFa(const load_balance::SectionStreamKResult &splitRes,
                                               optiling::detail::FaMetadata &faMetadata)
{
    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)}; // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto &faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto &prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bnStart = (aicIdx == 0) ? prevFaRes.bNEnd[prevLastCore] : faRes.bNEnd[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.mEnd[prevLastCore] : faRes.mEnd[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN_START_INDEX, bnStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN_END_INDEX, faRes.bNEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.mEnd[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }
    }
}

void FlashAttnMetadataCpuKernel::SetMetadataFd(const load_balance::SectionStreamKResult &splitRes,
                                               optiling::detail::FaMetadata &faMetadata)
{
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto &fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN_IDX_INDEX, fdRes.bNIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.mIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
}

namespace {
static const char *kernelType = "FlashAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, FlashAttnMetadataCpuKernel);
} // namespace

} // namespace aicpu
