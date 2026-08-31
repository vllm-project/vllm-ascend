/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cmath>
#include "../../fused_infer_attention_score_v2_sink/op_kernel/fused_infer_attention_score_v2_sink_metadata.h"
#include "../../common/aicpu/cpu_context_util.h"
#include "fused_infer_attention_score_v2_sink_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
namespace {
    const char* const FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA = "FusedInferAttentionScoreV2SinkMetadata";

// Normalize TND_NTD/NTD_TND to the TND parsing semantics.
// TND 族 layout（含双布局形态 TND_NTD/NTD_TND）: query 的 actual_seq_qlen 为逐 request 累积和语义,
// bSize 取 qlen 数组长度, s1Size 取逐 req 差分最大值, isAccumQSeq=true。
static inline bool IsTndFamilyLayout(const std::string &layout)
{
    return layout == "TND" || layout == "NTD" || layout == "TND_NTD" || layout == "NTD_TND";
}

// M 轴 token-major（S1 外层 G 内层）的 query layout: TND / BSH_BSND / TND_NTD。
// （NTD_TND 的 query 为 NTD 头主序, 不属于该集合。）
static inline bool IsS1GLayout(const std::string &layout)
{
    return layout == "TND" || layout == "BSH_BSND" || layout == "TND_NTD";
}
}  // namespace

uint32_t FusedInferAttentionScoreV2SinkMetadataKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    CheckIsMla();
    cvRatio_ = aivCoreNum_ / aicCoreNum_;
    if (s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        s2Size = 1024;
    }
    BaseInfo baseInfo{};
    SplitParam splitParam{};
    if (isMlaSink) {
        CalcInnerSizeMla(static_cast<uint32_t>(s2Size));
        CreateSplitInputMla(baseInfo, splitParam);
    } else {
        CalcInnerSizeGqa(static_cast<uint32_t>(s2Size));
        CreateSplitInputGqa(baseInfo, splitParam);
    }
    SplitResult res{aicCoreNum_, cvRatio_};
    SplitCore(aicCoreNum_, baseInfo, splitParam, res);
    if (res.numOfFdHead > aicCoreNum_ || res.usedCoreNum > aicCoreNum_ || res.maxS2SplitNum > aicCoreNum_ + 1) {
        KERNEL_LOG_ERROR("used_core_num: %u, num_of_fd_head: %u, max_s2_split_num: %u, aic_num: %u",
                         res.usedCoreNum,
                         res.numOfFdHead,
                         res.maxS2SplitNum,
                         aicCoreNum_);
    }
    success = GenMetaData(res);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GenMetaData(const SplitResult &splitRes)
{
    optiling::detail::FiaSinkMetaData *metaDataPtr = (optiling::detail::FiaSinkMetaData *)metaData_->GetData();
    // aic Metadata Generate
    for (uint32_t i = 0; i < splitRes.usedCoreNum; ++i) {
        if (i >= splitRes.usedCoreNum) {
            metaDataPtr->aicMetadata[i][AIC_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        metaDataPtr->aicMetadata[i][AIC_CORE_ENABLE_INDEX] = 1; // AIC enable
        metaDataPtr->aicMetadata[i][BN2_END_PTR_INDEX] = splitRes.bN2End[i];
        metaDataPtr->aicMetadata[i][GS1_END_PTR_INDEX] = splitRes.gS1End[i];
        metaDataPtr->aicMetadata[i][S2_END_PTR] = splitRes.s2End[i];
        metaDataPtr->aicMetadata[i][BN2_IDX_OF_FD_HEAD_INDEX] = splitRes.fdRes.bN2IdxOfFdHead[i];
        metaDataPtr->aicMetadata[i][GS1_IDX_OF_FD_HEAD_INDEX] = splitRes.fdRes.gS1IdxOfFdHead[i];
        metaDataPtr->aicMetadata[i][S2_SPLIT_NUM_OF_FD_HEAD_INDEX] = splitRes.fdRes.s2SplitNumOfFdHead[i];
        metaDataPtr->aicMetadata[i][S2_SPLIT_START_IDX_OF_CORE_INDEX] = splitRes.fdRes.s2SplitStartIdxOfCore[i];
        metaDataPtr->aicMetadata[i][GS1_SPLIT_NUM_OF_FD_HEAD_INDEX] = splitRes.fdRes.gS1SplitNumOfFdHead[i];
        metaDataPtr->aicMetadata[i][GS1_LAST_PART_SIZE_OF_FD_HEAD_INDEX] = splitRes.fdRes.gS1LastPartSizeOfFdHead[i];
    }

    // aiv Metadata Generate,当前仅支持C:V=1:2
    for (uint32_t i = 0; i < splitRes.usedCoreNum * 2; ++i) {
        if (i >= (splitRes.usedCoreNum * 2)) {
            metaDataPtr->aivMetadata[i][AIV_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metaDataPtr->aivMetadata[i][AIV_CORE_ENABLE_INDEX] = 1; // AIV enable
        metaDataPtr->aivMetadata[i][GS1_IDX_END_OF_FD_HEAD_INDEX] = splitRes.fdRes.gS1IdxEndOfFdHead[i];
        metaDataPtr->aivMetadata[i][GS1_IDX_END_OF_FD_HEAD_SPLIT_INDEX] = splitRes.fdRes.gS1IdxEndOfFdHeadSplit[i];
    }

    for (size_t i = 0; i < BASE_METADATA_SIZE; ++i) {
        metaDataPtr->baseMetadata[M_BASE_SIZE_INDEX] = mBaseSize_;
        metaDataPtr->baseMetadata[S_INNER_SIZE_INDEX] = sInnerSize_;
        metaDataPtr->baseMetadata[M_FD_BASE_SIZE_INDEX] = mFdBaseSize_;
        metaDataPtr->baseMetadata[NUM_OF_FD_INDEX] = splitRes.numOfFdHead;
        metaDataPtr->baseMetadata[USED_CORE_NUM_INDEX] = splitRes.usedCoreNum;
        metaDataPtr->baseMetadata[USED_VEC_NUM_OF_FD_INDEX] = splitRes.usedVecNumOfFd;
        metaDataPtr->baseMetadata[S1_SIZE_INDEX] = s1Size;
        metaDataPtr->baseMetadata[S2_SIZE_INDEX] = s2Size;
        metaDataPtr->baseMetadata[ACUTAL_LEN_Q_DIM_INDEX] = actualLenQDims;
        metaDataPtr->baseMetadata[ACUTAL_LEN_KV_DIM_INDEX] = actualLenKvDims;
    }
    return true;
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcInnerSizeMla(uint32_t seqSize)
{
    /*
     * CANN reduceSum函数在规约长度S  S>=512、S<512两种场景内部实现的累加顺序不同，
     * 为了保障batchInvariant场景计算结果的一致性，将算子在S2方向的基本块大小固定为256，
     * 从而确保不同shape下reduceSum函数的累加序的一致性
    */
    if (batchInvariant_) {
        sInnerSize_ = 256U;
    } else {
        sInnerSize_ = 512U;
        uint32_t s1G = s1Size * gSize;
        /*
         * BAND场景，带上sinkToken后，分块位置发生变化，从而导致计算band边界覆盖的基本块增加，导致无效计算量增大
         * 将基本块大小减小至256，减少无效计算量，缓解极端场景下分核劣势
         * s1G >= 128为实测最优值
        */
        if (sinkNum_ > 0 && sinkNum_ <= 256 && sparseMode_ == static_cast<uint32_t>(SparseMode::BAND) && s1G >= 128) {
            // decode阶段sparseMode4场景sinkToken耗时优化，s1G >= 128经验值
            sInnerSize_ = 256;
        }

        /*
         * 在decode阶段 sinkTensor场景，使得小batch时尽可能使用满核数
         * 在SWA阶段，窗口大小为512时，单batch在s2base=512时，分3个核，
         * 在s2base=256时分4个核，batch=6时，刚好用满24个核。
        */
        if ((kSinkNum_ > 0 && (sparseMode_ == static_cast<uint32_t>(SparseMode::BAND)) &&
            bSize <= 6 && (preToken + nextToken < 512))) {
            sInnerSize_ = 256;
        }

        // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
        if (splitKVFlag_ && !IsS1GLayout(inputLayout_)) {
            if (seqSize == 256U) {
                sInnerSize_ = 128U;
            } else if (seqSize > 256U && seqSize <= sInnerSize_) {
                sInnerSize_ = (sInnerSize_ + 1U) / 2U;
            }
        }

        // PA特性泛化场景，blockSize可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
        if (blockSize_ != 0) {
            uint32_t blockSize = static_cast<uint32_t>(blockSize_);
            if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
                sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
            }
        }

        sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
        sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
        // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
        // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
        if (sInnerSize_ > seqSize) {
            sInnerSize_ = seqSize;
        }
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CalcMBaseSizeMla();
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcMBaseSizeMla()
{
    if (batchInvariant_) {
        uint32_t mBaseSizeBatchInvariant = 256;
        mBaseSize_ = mBaseSizeBatchInvariant;
        return;
    }

    uint32_t bN2 = bSize * n2Size;
    uint32_t s1G = s1Size * gSize;
    if (bN2 == 0U || aicCoreNum_ == 0U) {
        return;
    }
    // 若BN2能整除核数或者KVS不等长，不考虑调整mBaseSize
    if (bN2 % aicCoreNum_ == 0U || !DealSameSeqEachBatch()) {
        return;
    }

    // 求B*N2与核数的最小公倍数
    uint32_t originalA = bN2;
    uint32_t originalB = aicCoreNum_;
    // 欧几里得算法计算最大公约数
    while (originalB != 0U) {
        originalA %= originalB;
        std::swap(originalA, originalB);
    }

    // 计算最小公倍数
    uint32_t lcm = (bN2 / originalA) * aicCoreNum_;
    uint32_t splitOfS1G = lcm / bN2; // 求S1G被切多少份，可以不开FD
    // 如果不能等分S1G，则不切S2无法做到负载不均衡，不修改mBaseSize，直接按照最大mBaseSize进行切分
    if (splitOfS1G == 0U || s1G % splitOfS1G != 0U) {
        return;
    }
    uint32_t mBaseSizeTmp = s1G / splitOfS1G; // 计算可以等分S1G的mBaseSize
    // 128：最小的mBaseSize，计算的理论mBaseSize需要在原定mBaseSize和最小mBaseSize区间内
    if (mBaseSizeTmp > mBaseSize_ || mBaseSizeTmp < 128U) {
        return;
    }

    mBaseSize_ = mBaseSizeTmp; // 满足以上条件则更新mBaseSize
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CreateSplitInputMla(BaseInfo &baseInfo, SplitParam &splitParam) const
{
    // 构造分核输入参数
    baseInfo.bSize = bSize;
    baseInfo.n2Size = n2Size;
    baseInfo.gSize = gSize;
    baseInfo.s2Size = static_cast<uint32_t>(s2Size);
    baseInfo.s1Size = s1Size;
    baseInfo.actualLenQDims = actualLenQDims;
    baseInfo.actualLenKvDims = actualLenKvDims;
    baseInfo.preToken = preToken;
    baseInfo.nextToken = nextToken;
    baseInfo.isS1G = IsS1GLayout(inputLayout_);
    baseInfo.sparseMode = sparseMode_;
    baseInfo.attenMaskFlag = attenMaskFlag;
    baseInfo.sinkNumber = sinkNum_;
    baseInfo.keySinkNumber = kSinkNum_;
    baseInfo.supportFD = true;  // sink做为独立tensor传入时,decode默认走FD
    baseInfo.batchInvariant = batchInvariant_;
    if (actualSeqLengths_ != nullptr) {
        baseInfo.isAccumSeqS1 = isAccumQSeq;
        baseInfo.actualSeqS1Size.reserve(bSize);
        const int64_t *s1Ptr = static_cast<const int64_t *>(actualSeqLengths_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (actualSeqLengthsKv_ != nullptr) {
        baseInfo.isAccumSeqS2 = isAccumKVSeq;
        baseInfo.actualSeqS2Size.reserve(bSize);
        const int64_t *s2Ptr = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    splitParam.mBaseSize = mBaseSize_;
    splitParam.s2BaseSize = sInnerSize_;
    splitParam.gS1BaseSizeOfFd = mFdBaseSize_;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::DealSameSeqEachBatch()
{
    batchContinuousFlag = blockSize_ == 0 ? true : false;
    if (batchContinuousFlag) {
        if (actualSeqLenFlag) {
            return isSameActualseq;
        } else {
            return isSameSeqAllKVTensor;
        }
    } else {
        return isSameActualseq;
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcInnerSizeGqa(uint32_t s2Size)
{
    /*
     * CANN reduceSum函数在规约长度S  S>=512、S<512两种场景内部实现的累加顺序不同，
     * 为了保障batchInvariant场景计算结果的一致性，将算子在S2方向的基本块大小固定为256，
     * 从而确保不同shape下reduceSum函数的累加序的一致性
    */
    if (batchInvariant_) {
        sInnerSize_ = 256;
    } else {
        if (IsTndFamilyLayout(inputLayout_)) {
            sInnerSize_ = S_INNER_SIZE_512;
            /*
             * BAND场景，带上sinkToken后，分块位置发生变化，从而导致计算band边界覆盖的基本块增加，导致无效计算量增大
             * 将基本块大小减小至256，减少无效计算量，缓解极端场景下分核劣势
            */
            if ((sinkNum_ > 0 && sinkNum_ <= 256 && s1Size > S1_SIZE_16 &&
                sparseMode_ == static_cast<uint32_t>(SparseMode::BAND))) {
                sInnerSize_ = 256;
            }

            /*
             * prefill阶段，SWA场景(窗口大小为512)，针对sinkTensor场景做了小块跳过性能优化操作，需要将s2基本块修改为256
             * 由于prefill阶段数据量较大，在长序列场景为计算bound，跳过操作的搬运收益拿不到
             * 且将s2基本块修改为256后，scale操作增多，性能反而劣化，实验测试边界值为qTSize=32k
             *
             * kernel侧使用的sInnerSize_是通过aicpu传递的，该处的sInnerSize_不起作用
             * 该处和aicpu的差异在于，该处使用qTSize，aicpu使用actseqlen[-1]
            */
            const int64_t *s1Ptr = static_cast<const int64_t*>(actualSeqLengths_->GetData());
            if ((kSinkNum_ > 0 && sparseMode_ == static_cast<uint32_t>(SparseMode::BAND) &&
                s1Ptr[bSize - 1] < 32 * 1024 && (preToken + nextToken < 512))) {
                sInnerSize_ = 256;
            }
        } else {
            if (s1Size <= S1_SIZE_16) {
                /**
                * V1阶段分配用于存放mm1结果的UB大小为32K, 当计算的数据类型为float时，其可以存放8192个元素.
                * 另外, 需要保证单次计算不会切分S2, 那么S2的内切大小最大为8192, 所以将默认值设置为8192
                */
                sInnerSize_ = MAX_SPLIT_SIZE; // 8192

                /** 当前版本限制workspace大小不超过32MB，否则会影响网络中前后算子性能，
                *  GQA场景下 nNumOfQInOneGroup和sInnerSize_切分大小直接影响workspace大小,
                *  具体计算参考CalcWorkSpace函数，这里根据nNumOfQInOneGroup将sInnerSize_
                *  分为8192，4096，2048三档，nNumOfQInOneGroup增大时减小sInnerSize_，
                *   保证最终workspace大小不超过32MB。
                */
                uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
                uint32_t idx = std::min(gSize / 5U, 2U);
                sInnerSize_ = sInnerSize[idx];
            } else {
                sInnerSize_ = S_INNER_SIZE_512;
            }
        }
        if (attenMaskFlag && (sparseMode_ == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
                sparseMode_ == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL) ||
                sparseMode_ == static_cast<uint32_t>(SparseMode::BAND))) {
            sInnerSize_ = std::min(sInnerSize_, S_INNER_SIZE_1024); // attention mask压缩场景，基本块最大支持1024*1024
        }

        // PA特性泛化场景，blockSize可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
        if (blockSize_ != 0) {
            uint32_t blockSize = static_cast<uint32_t>(blockSize_);
            if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
                sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
            }
        }
        sInnerLoopTimes_ = (s2Size + sInnerSize_ - static_cast<uint32_t>(1)) / sInnerSize_;
        sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - static_cast<uint32_t>(1)) * sInnerSize_;
        // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
        // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
        if (sInnerSize_ > s2Size) {
            sInnerSize_ = s2Size;
        }
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    CalcMBaseSizeGqa();
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcMBaseSizeGqa()
{
    if (IsTndFamilyLayout(inputLayout_)) {
        mBaseSize_ = M_BASE_SIZE_512;
        /*
         * BAND场景，带上sinkToken后，分块位置发生变化，从而导致计算band边界覆盖的基本块增加，导致无效计算量增大
         * 将基本块大小减小至256，减少无效计算量，缓解极端场景下分核劣势
        */
        if (sinkNum_ > 0 && sinkNum_ <= 256 && s1Size > S1_SIZE_16 &&
            sparseMode_ == static_cast<uint32_t>(SparseMode::BAND) &&
            nextToken == 0 && preToken % 512 >= sinkNum_) {
            // prefill阶段sparseMode4场景sinkToken耗时优化，当swa窗口比sinkNumber大时，会导致s2方向切块变多，计算量增加
            mBaseSize_ = M_BASE_SIZE_256;
        }
    } else {
        if (s1Size <= S1_SIZE_16) {
            if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_512) {
                mBaseSize_ = M_BASE_SIZE_512;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_1024) {
                mBaseSize_ = M_BASE_SIZE_256;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_2048) {
                mBaseSize_ = M_BASE_SIZE_128;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_4096) {
                mBaseSize_ = M_BASE_SIZE_64;
            } else { // sInnerSizeAlign_最大值为8192
                mBaseSize_ = M_BASE_SIZE_32;
            }
        } else {
            mBaseSize_ = M_BASE_SIZE_512;
        }
    }

    if (batchInvariant_) {
        mBaseSize_ = M_BASE_SIZE_256;
    }
    softmaxWithBrcbFlag_ = (mBaseSize_ <= M_BASE_SIZE_128);
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CreateSplitInputGqa(BaseInfo &baseInfo, SplitParam &splitParam) const
{
    // 构造分核输入参数
    baseInfo.bSize = bSize;
    baseInfo.n2Size = n2Size;
    baseInfo.gSize = gSize;
    baseInfo.s2Size = s2Size;
    baseInfo.s1Size = s1Size;
    baseInfo.actualLenQDims = actualLenQDims;
    baseInfo.actualLenKvDims = actualLenKvDims;
    baseInfo.preToken = preToken;
    baseInfo.nextToken = nextToken;
    baseInfo.isS1G = IsS1GLayout(inputLayout_);
    baseInfo.sparseMode = sparseMode_;
    baseInfo.attenMaskFlag = attenMaskFlag;
    baseInfo.sinkNumber = sinkNum_;
    baseInfo.keySinkNumber = kSinkNum_;
    baseInfo.supportFD = (baseInfo.keySinkNumber == 0);  // sink做为独立tensor传入时,prefill不走FD
    baseInfo.batchInvariant = batchInvariant_;
    if (actualSeqLengths_ != nullptr) {
        baseInfo.isAccumSeqS1 = isAccumQSeq;
        baseInfo.actualSeqS1Size.reserve(bSize);
        const int64_t *s1Ptr = static_cast<const int64_t *>(actualSeqLengths_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (actualSeqLengthsKv_ != nullptr) {
        baseInfo.isAccumSeqS2 = isAccumKVSeq;
        const int64_t *s2Ptr = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
        for (uint32_t i = 0; i < bSize; ++i) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }

    splitParam.mBaseSize = mBaseSize_;
    splitParam.s2BaseSize = sInnerSize_;
    splitParam.gS1BaseSizeOfFd = mFdBaseSize_;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    actualSeqLengths_ = ctx.Input(static_cast<uint32_t>(InputParamId::ACTUAL_SEQ_LENGTHS_OPTIONAL));
    actualSeqLengthsKv_ = ctx.Input(static_cast<uint32_t>(InputParamId::ACTUAL_SEQ_LENGTHS_KV_OPTIONAL));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(OutputParamId::META_DATA));

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
                         GetAttrValue(ctx, "head_dim_qk", headDimQK_) && GetAttrValue(ctx, "head_dim_v", headDimV_);
    GetAttrValueOpt(ctx, "soc_version", socVersion_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
    GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "rope_head_dim", ropeHeadDim_);
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "sparse_mode", sparseMode_);
    GetAttrValueOpt(ctx, "pre_tokens", preTokens_);
    GetAttrValueOpt(ctx, "next_tokens", nextTokens_);
    GetAttrValueOpt(ctx, "input_layout", inputLayout_);
    GetAttrValueOpt(ctx, "input_layout_kv", inputLayoutKv_);
    GetAttrValueOpt(ctx, "sink_num", sinkNum_);
    GetAttrValueOpt(ctx, "k_sink_num", kSinkNum_);
    GetAttrValueOpt(ctx, "batch_invariant", batchInvariant_);
    GetAttrValueOpt(ctx, "block_size", blockSize_);

    return (ParamsParse() && ParamsCheck());
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::ParamsParse()
{
    return (CheckSingleParam() && ParseAxisInfo());
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckSingleParam()
{
    // 1. 基础输出校验
    KERNEL_CHECK_NULLPTR(metaData_, false, "metadata is null");
    auto metaShape = metaData_->GetTensorShape();
    KERNEL_CHECK_NULLPTR(metaShape, false, "shape of metadata is null");
    KERNEL_CHECK_NULLPTR(metaData_->GetData(), false, "data of metadata is null");
    // 2. 核心数校验
    if (aicCoreNum_ == 0 || aivCoreNum_ == 0 || (aivCoreNum_ % aicCoreNum_ != 0)) {
        KERNEL_LOG_ERROR("Core num invalid: aic:%u, aiv:%u", aicCoreNum_, aivCoreNum_);
        return false;
    }

    if (batchSize_ < 0) {
        KERNEL_LOG_ERROR("batchSize is %d, it should be greater than 0.", batchSize_);
        return false;
    }

    constexpr int32_t BLOCK_SIZE_ALIGN_SIZE = 16;
    constexpr int32_t BLOCK_SIZE_MAX_SIZE = 1024;
    if (blockSize_ % BLOCK_SIZE_ALIGN_SIZE != 0) {
        KERNEL_LOG_ERROR("blockSize should aligned to 16, but got %d.", blockSize_);
        return false;
    }
    if (blockSize_ > BLOCK_SIZE_MAX_SIZE) {
        KERNEL_LOG_ERROR("blockSize should less equal than 1024, but got %d.", blockSize_);
        return false;
    }

    if ((sparseMode_ != static_cast<uint32_t>(SparseMode::DEFAULT_MASK)) &&
        (sparseMode_ != static_cast<uint32_t>(SparseMode::ALL_MASK)) &&
        (sparseMode_ != static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL)) &&
        (sparseMode_ != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) &&
        (sparseMode_ != static_cast<uint32_t>(SparseMode::BAND))) {
        KERNEL_LOG_ERROR("sparseMode only supports 0/1/2/3/4, but got %u.", sparseMode_);
        return false;
    }

    if (headDimQK_ <= 0) {
        KERNEL_LOG_ERROR("qk head num is %d, it should be greater than 0.", headDimQK_);
        return false;
    }

    if (headDimV_ <= 0) {
        KERNEL_LOG_ERROR("value head num is %d, it should be greater than 0.", headDimV_);
        return false;
    }

    return CheckSingleParaInputLayout();
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckSingleParaInputLayout()
{
    const std::vector<std::string> inputLayoutList = {"BSH",
                                                      "BSND",
                                                      "BNSD",
                                                      "TND",
                                                      "NTD",
                                                      "BSH_NBSD",
                                                      "BSND_NBSD",
                                                      "BNSD_NBSD",
                                                      "TND_NTD",
                                                      "NTD_TND",
                                                      "BSH_BNSD",
                                                      "BSND_BNSD",
                                                      "BNSD_BSND"};
    if (std::find(inputLayoutList.begin(), inputLayoutList.end(), inputLayout_) == inputLayoutList.end()) {
        KERNEL_LOG_ERROR("input layout only supports BSH, BSND, BNSD, TND, NTD, BSH_NBSD, BSND_NBSD, BNSD_NBSD, "
                         "TND_NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s.",
                         inputLayout_);
        return false;
    }

    const std::vector<std::string> inputLayoutKvList = {
        "BSH", "BSND", "BNSD", "TND", "NTD", "NZ", "BnBsH", "BnNBsD", "PA"};
    if (std::find(inputLayoutKvList.begin(), inputLayoutKvList.end(), inputLayoutKv_) == inputLayoutKvList.end()) {
        KERNEL_LOG_ERROR("input layout only supports BSH, BSND, BNSD, TND, NTD, NZ, BnBsH, BnNBsD, PA, but got %s",
                         inputLayoutKv_);
        return false;
    }

    if (IsTndFamilyLayout(inputLayout_)) {
        KERNEL_CHECK_NULLPTR(
            actualSeqLengths_, false, "When inputLayout is TND or NTD, actualSeqQLengths should not be null.");
        KERNEL_CHECK_NULLPTR(
            actualSeqLengthsKv_, false, "When inputLayout is TND or NTD, actualSeqKVLengths should not be null.");
    }

    if (blockSize_ == 0) {
        if (inputLayoutKv_ != "BSH" && inputLayoutKv_ != "BSND" && inputLayoutKv_ != "BNSD" &&
            inputLayoutKv_ != "TND" && inputLayoutKv_ != "NTD") {
            KERNEL_LOG_ERROR("In BATCH_CONTINUOUS situation, key/value's layout only support BSH, BSND, BNSD, TND and "
                             "NTD in batch continuous scene, but got %s",
                             inputLayoutKv_);
            return false;
        }
        if (inputLayoutKv_ != inputLayout_) {
            KERNEL_LOG_ERROR("In BATCH_CONTINUOUS situation, key/value's layout and query's layout should be same in "
                             "batch continuous scene.");
            return false;
        }
    } else {
        if ((inputLayoutKv_ == "BnBsH" || inputLayoutKv_ == "BnNBsD" || inputLayoutKv_ == "NZ" ||
             inputLayoutKv_ == "PA") &&
            (inputLayout_ != "BSH" && inputLayout_ != "BSND" && inputLayout_ != "BNSD" && inputLayout_ != "TND" &&
             inputLayout_ != "NTD" && inputLayout_ != "TND_NTD" && inputLayout_ != "NTD_TND")) {
            KERNEL_LOG_ERROR("In PAGE_ATTENTION situation, query layout must be BSH, BSND, BNSD TND and TND in page "
                             "attention scene, but got %s",
                             inputLayout_);
            return false;
        }
    }

    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::ParseAxisInfo()
{
    return (CheckSingleParaNumHeads() && GetBatchSize() && GetS1Size() && GetS2Size() && GetPreNextToken() &&
            GetActualSeqInfo());
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckSingleParaNumHeads()
{
    if (numHeadsQ_ <= 0) {
        KERNEL_LOG_ERROR("the query's heads num(%u) should be greater than 0", numHeadsQ_);
        return false;
    }

    if (numHeadsKv_ <= 0) {
        KERNEL_LOG_ERROR("the query's heads num(%u) should be greater than 0", numHeadsQ_);
        return false;
    }

    n1Size = static_cast<uint32_t>(numHeadsQ_);
    n2Size = (numHeadsKv_ == 0) ? n1Size : static_cast<uint32_t>(numHeadsKv_);
    if (n1Size % n2Size != 0U) {
        KERNEL_LOG_ERROR(
            "the query's heads num(%u) should be a multiple of the key/value's heads num(%u)", n1Size, n2Size);
        return false;
    }
    gSize = n1Size / n2Size;
    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (IsTndFamilyLayout(inputLayout_)) {
        return GetActualSeqLenSize(bSize);
    } else { // BSH/BSND/BNSD
        bSize = batchSize_;
        return true;
    }
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND/NTD时, 以max_seqlen_q为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (IsTndFamilyLayout(inputLayout_)) {
        const int64_t *actualSeqQ = static_cast<const int64_t *>(actualSeqLengths_->GetData());
        if (actualSeqQ == nullptr) {
            return false;
        }

        uint32_t b = 0;
        if (!GetActualSeqLenSize(b)) {
            return false;
        }
        int64_t qActualSeqMax = 0;
        for (uint32_t i = 0; i < b; i++) {
            int64_t tmpS1 = (i == 0U) ? actualSeqQ[0] : (actualSeqQ[i] - actualSeqQ[i - 1U]);
            if (tmpS1 > qActualSeqMax) {
                qActualSeqMax = tmpS1;
            }
        }
        s1Size = static_cast<uint32_t>(qActualSeqMax);
    }

    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GetActualSeqLenSize(uint32_t &size)
{
    int64_t shapeSize = actualSeqLengths_->GetTensorShape()->GetDimSize(0);
    if (shapeSize <= 0) {
        KERNEL_LOG_ERROR("actualSeqLengths's shape size is %ld, it should be greater than 0.", shapeSize);
        return false;
    }
    size = static_cast<uint32_t>(shapeSize);
    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GetActualSeqKVLenSize(uint32_t &size)
{
    int64_t shapeSize = actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0);
    if (shapeSize <= 0) {
        KERNEL_LOG_ERROR("actualSeqKVLengths's shape size is %ld, it should be greater than 0.", shapeSize);
        return false;
    }
    size = static_cast<uint32_t>(shapeSize);
    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GetS2Size()
{
    // 获取S2基准值 连续或者tensorlist，都通过前缀和获取，其他取最大
    const int64_t *actualLenData = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
    if (actualLenData == nullptr) {
        return false;
    }
    uint32_t loop = std::min(static_cast<uint32_t>(actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0)), bSize);
    if (IsTndFamilyLayout(inputLayoutKv_)) {
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = 0;
            tmpS = (i == 0U) ? actualLenData[0] : (actualLenData[i] - actualLenData[i - 1U]);
            s2Size = std::max(s2Size, tmpS);
        }
        if (kSinkNum_ > 0) {
            s2Size += kSinkNum_;
        }
        return true;
    } else {
        for (uint32_t i = 0; i < loop; i++) {
            s2Size = std::max(s2Size, actualLenData[i]);
        }
        if (kSinkNum_ > 0) {
            s2Size += kSinkNum_;
        }
        return true;
    }

    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GetPreNextToken()
{
    static int64_t SPARSE_MODE_INT_MAX = 2147483647;
    if (sparseMode_ == static_cast<uint32_t>(SparseMode::DEFAULT_MASK) && preTokens_ == SPARSE_MODE_INT_MAX &&
        nextTokens_ == SPARSE_MODE_INT_MAX) {
        attenMaskFlag = false;
    }
    if (sparseMode_ == static_cast<uint32_t>(SparseMode::ALL_MASK)) {
        preToken = SPARSE_MODE_INT_MAX;
        nextToken = SPARSE_MODE_INT_MAX;
    } else if (sparseMode_ == static_cast<uint32_t>(SparseMode::LEFT_UP_CAUSAL) ||
               sparseMode_ == static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        nextToken = 0;
        preToken = SPARSE_MODE_INT_MAX;
    } else {
        preToken = preTokens_ == SPARSE_MODE_INT_MAX ? 0 : preTokens_;
        nextToken = nextTokens_ == SPARSE_MODE_INT_MAX ? 0 : nextTokens_;
    }

    if (preToken > SPARSE_MODE_INT_MAX) {
        preToken = SPARSE_MODE_INT_MAX;
    } else if (preToken < -(SPARSE_MODE_INT_MAX)) {
        preToken = -(SPARSE_MODE_INT_MAX);
    }
    if (nextToken > SPARSE_MODE_INT_MAX) {
        nextToken = SPARSE_MODE_INT_MAX;
    } else if (nextToken < -(SPARSE_MODE_INT_MAX)) {
        nextToken = -(SPARSE_MODE_INT_MAX);
    }

    if (preToken < 0 && nextToken < 0) {
        KERNEL_LOG_ERROR("preTokens(%ld) and nextTokens(%ld) cannot neither be negative number.", preToken, nextToken);
        return false;
    }
    if (nextToken * (-1) > preToken) {
        KERNEL_LOG_ERROR("nextToken line(%ld) should be higher than preToken line(%ld).", preToken, nextToken);
        return false;
    }
    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::GetActualSeqInfo()
{
    maxActualseq = s2Size;
    if (actualSeqLengthsKv_ != nullptr && actualSeqLengthsKv_->GetData() != nullptr) {
        const int64_t *actualLenData = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
        actualLenKvDims = actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0);
        if (IsTndFamilyLayout(inputLayout_)) {
            isAccumQSeq = true;
        }
        uint32_t loop = ((actualLenKvDims == 1) && (kvListSeqLens.size() == 1)) ? 1 : bSize;
        loop = std::min(loop, actualLenKvDims);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t actLen = actualLenData[i];
            needInit = (actLen == 0);
            maxActualseq = maxActualseq < actLen ? actLen : maxActualseq;
            if (actualLenData[i] != actualLenData[0]) {
                isSameActualseq = false;
                break;
            }
        }
    }

    if (actualSeqLengths_ != nullptr && actualSeqLengths_->GetData() != nullptr) {
        const int64_t *actualLenQData = static_cast<const int64_t *>(actualSeqLengths_->GetData());
        actualLenQDims = actualSeqLengths_->GetTensorShape()->GetDimSize(0);
        if (IsTndFamilyLayout(inputLayoutKv_)) {
            isAccumKVSeq = true;
        }
        uint32_t loop = std::min(bSize, actualLenQDims);
        for (uint32_t i = 0; i < loop; i++) {
            if (actualLenQData[i] != static_cast<int64_t>(s1Size)) {
                needInit = true;
                break;
            }
        }
    }
    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::ParamsCheck()
{
    return (CheckFeature() && CheckMultiParaConsistency());
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CheckIsMla()
{
    // 仅支持ROPE分开传输
    if (ropeHeadDim_ == 0U) {
        isMlaSink = false;
        return;
    }

    // 仅支持Q&K&V的HeadDim为512, ROPE的HeadDim为64
    constexpr uint32_t QK_HEAD_DIM_512 = 512U;
    constexpr uint32_t ROPE_HEAD_DIM_64 = 64U;
    if ((headDimQK_ != headDimV_) || (headDimQK_ != QK_HEAD_DIM_512) || (ropeHeadDim_ != ROPE_HEAD_DIM_64)) {
        isMlaSink = false;
        return;
    }

    // 仅支持KV_N为1
    if (n2Size != 1U) {
        isMlaSink = false;
        return;
    }

    // 支持的input_layout范围
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD"};
    if (std::find(layoutSupportList.begin(), layoutSupportList.end(), inputLayout_) == layoutSupportList.end()) {
        isMlaSink = false;
        return;
    }

    // 支持的sparse_mode值
    if ((sparseMode_ != static_cast<uint32_t>(SparseMode::DEFAULT_MASK)) &&
        (sparseMode_ != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) &&
        (sparseMode_ != static_cast<uint32_t>(SparseMode::BAND))) {
        isMlaSink = false;
        return;
    }

    // 支持的G的取值范围
    std::vector<uint32_t> gSizeSupportList = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    if (std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize) == gSizeSupportList.end()) {
        isMlaSink = false;
        return;
    }

    isMlaSink = true;
    return;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckFeature()
{
    return (CheckFeatureActualSeqLensQData() && CheckFeatureActualSeqLensKvData() && CheckFeatureAxisInfo());
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckFeatureAxisInfo() const
{
    constexpr uint32_t MAX_ACTUAL_SEQ_LEN_BYTE = 64U * 1024U;

    if (actualSeqLengthsQSize_ > MAX_ACTUAL_SEQ_LEN_BYTE) {
        KERNEL_LOG_ERROR("actual sequence length q should be smaller or equal to 64K, but got %u",
                         actualSeqLengthsQSize_);
        return false;
    }

    if (actualSeqLengthsKvSize_ > MAX_ACTUAL_SEQ_LEN_BYTE) {
        KERNEL_LOG_ERROR("actual sequence length kv should be smaller or equal to 64K, but got %u",
                         actualSeqLengthsKvSize_);
        return false;
    }

    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckFeatureActualSeqLensQData()
{
    if (actualSeqLengths_ == nullptr) {
        qSize.push_back(s1Size);
        return true;
    }

    const int64_t *actualSeq = static_cast<const int64_t *>(actualSeqLengths_->GetData());
    if (actualSeq == nullptr) {
        qSize.push_back(s1Size);
        return true;
    }

    if (!GetActualSeqLenSize(actualSeqLengthsQSize_)) {
        return false;
    }

    uint32_t loop = std::min(actualSeqLengthsQSize_, bSize);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS1 = 0;
        if (IsTndFamilyLayout(inputLayout_)) {
            if (actualSeq[i] < 0) {
                KERNEL_LOG_ERROR("When q's Layout is TND or NTD, actualSeqQLengths[%u] should not be a negative "
                                 "number, but got %ld.",
                                 i,
                                 actualSeq[i]);
                return false;
            }

            if (i > 0U && actualSeq[i] < actualSeq[i - 1U]) {
                KERNEL_LOG_ERROR("When q's Layout is TND or NTD, actualSeqQLengths[%u](%ld) should not be less than "
                                 "actualSeqQLengths[%u](%ld).",
                                 i,
                                 actualSeq[i],
                                 (i - 1U),
                                 actualSeq[i - 1U]);
                return false;
            }

            tmpS1 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS1 = actualSeq[i];
        }
        if (tmpS1 > static_cast<int64_t>(s1Size) || tmpS1 < 0) {
            KERNEL_LOG_ERROR(
                "actualSeqQLengths[%u] computed is %ld, it should be in range [0, Q_S(%u)].", i, tmpS1, s1Size);
            return false;
        }
        qSize.push_back(tmpS1);
    }

    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckFeatureActualSeqLensKvData()
{
    if (actualSeqLengthsKv_ == nullptr) {
        kvSize.push_back(s2Size);
        return true;
    }
    actualSeqLenFlag = true;

    const int64_t *actualSeq = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
    if (actualSeq == nullptr) {
        kvSize.push_back(s2Size);
        return true;
    }

    if (!GetActualSeqKVLenSize(actualSeqLengthsKvSize_)) {
        return false;
    }

    uint32_t loop = std::min(actualSeqLengthsKvSize_, bSize);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS2 = 0;
        if (IsTndFamilyLayout(inputLayoutKv_)) {
            if (actualSeq[i] < 0) {
                KERNEL_LOG_ERROR("When kv's Layout is TND or NTD, actualSeqKVLengths[%u] should not be a negative "
                                 "number, but got %ld.",
                                 i,
                                 actualSeq[i]);
                return false;
            }

            if (i > 0U && actualSeq[i] < actualSeq[i - 1U]) {
                KERNEL_LOG_ERROR("When kv's Layout is TND or NTD, actualSeqKVLengths[%u](%ld) should not be less than "
                                 "actualSeqKVLengths[%u](%ld).",
                                 i,
                                 actualSeq[i],
                                 (i - 1U),
                                 actualSeq[i - 1U]);
                return false;
            }

            tmpS2 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS2 = actualSeq[i];
        }

        if (tmpS2 < 0 || tmpS2 > s2Size) {
            KERNEL_LOG_ERROR(
                "actualSeqKVLengths[%u] computed is %ld, it should be in range [0, KV_S(%u)].", i, tmpS2, s2Size);
            return false;
        }
        kvSize.push_back(tmpS2);
    }

    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckMultiParaConsistency()
{
    return (CheckActualSeqLensQ() && CheckActualSeqLensKv() && CheckParamSinkNumber() && CheckSinkNumber());
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckParamSinkNumber()
{
    if (kSinkNum_ < 0) {
        KERNEL_LOG_ERROR("invalid Parameter kSinkNum");
        return false;
    }

    if (kSinkNum_ == 0) {
        return true;
    }

    if (sinkNum_ > 0) {
        KERNEL_LOG_ERROR("Conflict paramSink and learnable attribute cannot coexist");
        return false;
    }

    if (sparseMode_ != static_cast<uint32_t>(SparseMode::BAND)) {
        KERNEL_LOG_ERROR("paramSink only support sparseMode=4(SWA), but got %u", sparseMode_);
        return false;
    }

    if (inputLayout_ != "TND" && inputLayout_ != "TND_NTD") {
        KERNEL_LOG_ERROR("query layout must be TND, but query layout is %s", inputLayout_.c_str());
        return false;
    }

    if ((actualSeqLengths_ == nullptr) || (actualSeqLengths_->GetData() == nullptr) ||
        (actualSeqLengthsKv_ == nullptr) || (actualSeqLengthsKv_->GetData() == nullptr)) {
        KERNEL_LOG_ERROR("actualSeqLengths_ or actualSeqLengthsKv_ cannot be empty.");
        return false;
    }

    const int64_t *actualSeqLensDataQ = static_cast<const int64_t *>(actualSeqLengths_->GetData());
    const int64_t *actualSeqLensData = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
    int64_t tmpS2Size = 0;
    for (uint32_t i = 0; i < actualSeqLengthsQSize_; i++) {
        int64_t tmpS1Size = (i == 0U) ? actualSeqLensDataQ[0] : (actualSeqLensDataQ[i] - actualSeqLensDataQ[i - 1U]);
        if (blockSize_ != 0) {
            tmpS2Size = actualSeqLensData[i];
        } else {
            tmpS2Size = (i == 0U) ? actualSeqLensData[0] : (actualSeqLensData[i] - actualSeqLensData[i - 1U]);
        }
        if (tmpS2Size == 0) {
            // 当s2=0时，为T轴padding场景，此时不需要校验行无效
            continue;
        }
        // 不支持行无效
        if (preToken < 0 || (nextToken < (tmpS1Size - tmpS2Size))) {
            KERNEL_LOG_ERROR("preToken < 0 or nextToken < (s1 - s2) is not supported when sink tensor exists");
            return false;
        }
    }

    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckSinkNumber()
{
    int64_t maxSinkNumber = 512;

    if (sinkNum_ == 0) {
        return true;
    }

    // sinkNum不能为奇数
    if (sinkNum_ < 0 || sinkNum_ > maxSinkNumber || sinkNum_ % 2 == 1) {
        KERNEL_LOG_ERROR("invalid Parameter sinkNumber");
        return false;
    }

    if (sinkNum_ > 0 && inputLayout_ != "TND" && inputLayout_ != "TND_NTD") {
        KERNEL_LOG_ERROR("when sinkNumber > 0, qLayout should TND");
        return false;
    }

    uint64_t minActualKvSeqLen = 1000000000000;
    const int64_t *actualKvLenData = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
    if (actualKvLenData != nullptr) {
        uint32_t loop = actualSeqLengthsKv_->GetTensorShape()->GetDimSize(0);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = 0;
            if (blockSize_ != 0) {
                tmpS = actualKvLenData[i];
            } else {
                tmpS = (i == 0U) ? actualKvLenData[0] : (actualKvLenData[i] - actualKvLenData[i - 1U]);
            }
            minActualKvSeqLen = minActualKvSeqLen < tmpS ? minActualKvSeqLen : tmpS;
        }
    }

    if (minActualKvSeqLen < sinkNum_) {
        KERNEL_LOG_ERROR("when sinkNumber > 0, and qLayout is TND, sinknumber(%ld) should <= minActualKvSeqLen(%ld)",
                         sinkNum_,
                         minActualKvSeqLen);
        return false;
    }

    return CheckMlaSinkNumber();
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckMlaSinkNumber()
{
    // D=128时，走GQA模板
    if (ropeHeadDim_ == 0U || headDimV_ != 512U) {
        return true;
    }
    if (sparseMode_ != static_cast<uint32_t>(SparseMode::BAND)) {
        KERNEL_LOG_ERROR("sparseMode only support 4 when sinkNumber(%ld) exist and D is 512", sinkNum_);
        return false;
    }
    // MLA sink 支持 TND、TND_NTD layout，继承GQA的校验

    const int64_t *actualSeqLensDataQ = static_cast<const int64_t *>(actualSeqLengths_->GetData());
    const int64_t *actualSeqLensData = static_cast<const int64_t *>(actualSeqLengthsKv_->GetData());
    int64_t tmpS2Size = 0;
    for (uint32_t i = 0; i < actualSeqLengthsQSize_; i++) {
        int64_t tmpS1Size = (i == 0U) ? actualSeqLensDataQ[0] : (actualSeqLensDataQ[i] - actualSeqLensDataQ[i - 1U]);
        if (blockSize_ != 0) {
            tmpS2Size = actualSeqLensData[i];
        } else {
            tmpS2Size = (i == 0U) ? actualSeqLensData[0] : (actualSeqLensData[i] - actualSeqLensData[i - 1U]);
        }
        if (tmpS2Size == sinkNum_) {
            // 当s2=sinkNumber时，为T轴padding场景，此时不需要校验行无效
            continue;
        }
        // 不支持行无效
        if (preToken < 0 || (nextToken < (tmpS1Size - tmpS2Size + sinkNum_))) {
            KERNEL_LOG_ERROR("preToken < 0 or nextToken < (s1 - s2 + sinkNumber) is not supported "
                             "when sinkNumber(%ld) exist and D is 512",
                             sinkNum_);
            return false;
        }
    }
    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckActualSeqLensQ() const
{
    if ((actualSeqLengths_ == nullptr) || (actualSeqLengths_->GetData() == nullptr)) {
        return true;
    }

    if (inputLayout_ == "TND" || inputLayout_ == "TND_NTD" || inputLayout_ == "NTD_TND") {
        if (actualSeqLengthsQSize_ != bSize && actualSeqLengthsQSize_ != 1U) {
            KERNEL_LOG_ERROR("actual seq q len shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                             actualSeqLengthsQSize_,
                             bSize);
            return false;
        }
    } else {
        if (actualSeqLengthsQSize_ < bSize && actualSeqLengthsQSize_ != 1U) {
            KERNEL_LOG_ERROR(
                "actual seq q len shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                actualSeqLengthsQSize_,
                bSize);
            return false;
        }
    }
    return true;
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::CheckActualSeqLensKv() const
{
    if ((actualSeqLengthsKv_ == nullptr) || (actualSeqLengthsKv_->GetData() == nullptr)) {
        return true;
    }

    if (inputLayout_ == "TND" || inputLayout_ == "TND_NTD" || inputLayout_ == "NTD_TND") {
        if (actualSeqLengths_ != nullptr && actualSeqLengths_->GetData() != nullptr &&
            actualSeqLengthsKvSize_ != actualSeqLengthsQSize_) {
            KERNEL_LOG_ERROR(
                "actual seq kv len shape size is %u, it should be equal to actual seq kv len shape size(%u).",
                actualSeqLengthsKvSize_,
                actualSeqLengthsQSize_);
            return false;
        }
        if (actualSeqLengthsKvSize_ != bSize && actualSeqLengthsKvSize_ != 1U) {
            KERNEL_LOG_ERROR("actual seq kv len shape size is %u, it should be equal to batch size(%u) or equal to 1.",
                             actualSeqLengthsKvSize_,
                             bSize);
            return false;
        }
    } else {
        if (actualSeqLengthsKvSize_ < bSize && actualSeqLengthsKvSize_ != 1U) {
            KERNEL_LOG_ERROR(
                "actual seq kv len shape size is %u, it should be bigger or equal to batch size(%u) or equal to 1.",
                actualSeqLengthsKvSize_,
                bSize);
            return false;
        }
    }

    return true;
}

//============================= 以下为split_core.cpp中的接口 ===========================

uint32_t FusedInferAttentionScoreV2SinkMetadataKernel::GetS1SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    if (baseInfo.actualSeqS1Size.empty()) {
        return baseInfo.s1Size;
    }

    if (baseInfo.actualLenQDims == 1U) {
        return static_cast<uint32_t>(baseInfo.actualSeqS1Size[0]);
    }

    if (!baseInfo.isAccumSeqS1) {
        return static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx]);
    }

    return (bIdx == 0) ? static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx])
                       : static_cast<uint32_t>(baseInfo.actualSeqS1Size[bIdx] - baseInfo.actualSeqS1Size[bIdx - 1U]);
}

uint32_t FusedInferAttentionScoreV2SinkMetadataKernel::GetS2SeqSize(uint32_t bIdx, const BaseInfo &baseInfo)
{
    uint32_t prefix = static_cast<uint32_t>(baseInfo.actualSeqPrefixSize);
    if (baseInfo.actualSeqS2Size.empty()) {
        return prefix + baseInfo.s2Size;
    }

    if (baseInfo.actualLenKvDims == 1U) {
        return prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[0]);
    }

    if (!baseInfo.isAccumSeqS2) {
        return prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx]);
    }

    return (bIdx == 0)
               ? prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx])
               : prefix + static_cast<uint32_t>(baseInfo.actualSeqS2Size[bIdx] - baseInfo.actualSeqS2Size[bIdx - 1U]);
}

int64_t FusedInferAttentionScoreV2SinkMetadataKernel::CalcPreTokenLeftUp(uint32_t s1Size,
                                                                         uint32_t s2Size,
                                                                         const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + baseInfo.preToken;
    }
    return baseInfo.preToken;
}

int64_t FusedInferAttentionScoreV2SinkMetadataKernel::CalcNextTokenLeftUp(uint32_t s1Size,
                                                                          uint32_t s2Size,
                                                                          const BaseInfo &baseInfo)
{
    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return baseInfo.nextToken;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + baseInfo.nextToken;
        default:
            return baseInfo.nextToken;
    }
}

int64_t FusedInferAttentionScoreV2SinkMetadataKernel::CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2); // 6：M轴系数，10：S2轴系数
}

BlockCost<int64_t> FusedInferAttentionScoreV2SinkMetadataKernel::CalcCostTable(uint32_t s1NormalSize,
                                                                               uint32_t s2NormalSize,
                                                                               uint32_t s1GTailSize,
                                                                               uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost{};
    typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)][static_cast<size_t>(BlockType::NORMAL_BLOCK)] =
        CalcCost(s1NormalSize, s2NormalSize);
    typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)][static_cast<size_t>(BlockType::NORMAL_BLOCK)] =
        (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)][static_cast<size_t>(BlockType::TAIL_BLOCK)] =
        (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)][static_cast<size_t>(BlockType::TAIL_BLOCK)] =
        (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> FusedInferAttentionScoreV2SinkMetadataKernel::CalcS2Range(uint32_t s1GIdx,
                                                                          const BaseInfo &baseInfo,
                                                                          const SplitParam &splitParam,
                                                                          const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    // no mask
    if (!baseInfo.attenMaskFlag) {
        s2Start = 0U;
        s2End = (batchCache.s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        return std::make_pair(s2Start, s2End);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(splitParam.mBaseSize);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(splitParam.mBaseSize),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(baseInfo.gSize)) -
                           1;

    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (baseInfo.isS1G) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(baseInfo.gSize);
        s1LastToken = s1GLastToken / static_cast<int64_t>(baseInfo.gSize);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 2. trans index of token to index of block
    // no valid token
    if (s2FirstToken >= batchCache.s2Size || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }

    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2Start = static_cast<uint32_t>(s2FirstToken) / splitParam.s2BaseSize;
    s2End = static_cast<uint32_t>(s2LastToken) / splitParam.s2BaseSize +
            1U; // end of block index, +1 for Right-open interval
    return std::make_pair(s2Start, s2End);
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcSplitInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    uint32_t s2Size;
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx, baseInfo);
        if (baseInfo.keySinkNumber && baseInfo.supportFD) {
            s2Size = baseInfo.keySinkNumber + GetS2SeqSize(bIdx, baseInfo);
        } else {
            s2Size = GetS2SeqSize(bIdx, baseInfo);
        }

        splitInfo.s1GBaseNum[bIdx] = (s1Size * baseInfo.gSize + (splitParam.mBaseSize - 1U)) / splitParam.mBaseSize;
        splitInfo.s1GTailSize[bIdx] = (s1Size * baseInfo.gSize) % splitParam.mBaseSize;
        splitInfo.s2BaseNum[bIdx] = (s2Size + splitParam.s2BaseSize - 1U) / splitParam.s2BaseSize;
        splitInfo.s2TailSize[bIdx] = s2Size % splitParam.s2BaseSize;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcBatchCache(uint32_t bIdx,
                                                                  const SplitContext &splitContext,
                                                                  BatchCache &batchCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx, baseInfo);
    if (baseInfo.keySinkNumber && baseInfo.supportFD) {
        batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo) + baseInfo.keySinkNumber;
    } else {
        batchCache.s2Size = GetS2SeqSize(bIdx, baseInfo);
    }

    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size, baseInfo);
    batchCache.typeCost = CalcCostTable(
        splitParam.mBaseSize, splitParam.s2BaseSize, splitInfo.s1GTailSize[bIdx], splitInfo.s2TailSize[bIdx]);
}

// FusedInferAttentionScoreV2Sink自定义算子新增函数：计算s1GIdx行sink块的Cost和块数
void FusedInferAttentionScoreV2SinkMetadataKernel::CalcS1GCacheWithSinkNumber(uint32_t s1GIdx,
                                                                              const SplitContext &splitContext,
                                                                              const BatchCache &batchCache,
                                                                              S1GCache &s1GCache,
                                                                              uint32_t sinkNumber)
{
    s1GCache.s2Start = 0;
    if (s1GCache.s2End == 0) {
        return;
    }
    if (s1GCache.firstTokenWithSinkNumber == 0) {
        return;
    }
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    uint32_t sinkBlockLastIndex =
        (static_cast<uint32_t>(sinkNumber) + splitParam.s2BaseSize - 1) / splitParam.s2BaseSize - 1;
    if (s1GCache.firstTokenWithSinkNumber > sinkBlockLastIndex) {
        s1GCache.s1GBlock += sinkBlockLastIndex + 1;
        if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
            s1GCache.s1GCost += batchCache.typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)]
                                                   [static_cast<size_t>(BlockType::NORMAL_BLOCK)] *
                                (sinkBlockLastIndex + 1);
        } else {
            s1GCache.s1GCost += batchCache.typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)]
                                                   [static_cast<size_t>(BlockType::NORMAL_BLOCK)] *
                                (sinkBlockLastIndex + 1);
        }
    } else {
        s1GCache.s1GBlock += s1GCache.firstTokenWithSinkNumber;
        if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
            s1GCache.s1GCost += batchCache.typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)]
                                                   [static_cast<size_t>(BlockType::NORMAL_BLOCK)] *
                                s1GCache.firstTokenWithSinkNumber;
        } else {
            s1GCache.s1GCost += batchCache.typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)]
                                                   [static_cast<size_t>(BlockType::NORMAL_BLOCK)] *
                                s1GCache.firstTokenWithSinkNumber;
        }
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcS1GCache(uint32_t s1GIdx,
                                                                const SplitContext &splitContext,
                                                                const BatchCache &batchCache,
                                                                S1GCache &s1GCache)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, baseInfo, splitParam, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;
    s1GCache.firstTokenWithSinkNumber =
        s2Range
            .first; // FusedInferAttentionScoreV2Sink自定义算子新增参数：表达sinkNumber>0时，s1GIdx行preToken所在的第一个块的下标
    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num =
        (splitInfo.s2TailSize[batchCache.bIdx] != 0U && s1GCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U
                                                                                                                : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
    } else if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost =
            batchCache.typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)]
                               [static_cast<size_t>(BlockType::NORMAL_BLOCK)] *
                curNormalS2Num +
            batchCache
                    .typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)][static_cast<size_t>(BlockType::TAIL_BLOCK)] *
                curTailS2Num;
        s1GCache.s1GLastBlockCost =
            curTailS2Num > 0U
                ? batchCache
                      .typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)][static_cast<size_t>(BlockType::TAIL_BLOCK)]
                : batchCache.typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)]
                                     [static_cast<size_t>(BlockType::NORMAL_BLOCK)];
        s1GCache.s1GNormalBlockCost =
            batchCache
                .typeCost[static_cast<size_t>(BlockType::TAIL_BLOCK)][static_cast<size_t>(BlockType::NORMAL_BLOCK)];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)]
                                              [static_cast<size_t>(BlockType::NORMAL_BLOCK)] *
                               curNormalS2Num +
                           batchCache.typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)]
                                              [static_cast<size_t>(BlockType::TAIL_BLOCK)] *
                               curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U
                                        ? batchCache.typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)]
                                                             [static_cast<size_t>(BlockType::TAIL_BLOCK)]
                                        : batchCache.typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)]
                                                             [static_cast<size_t>(BlockType::NORMAL_BLOCK)];
        s1GCache.s1GNormalBlockCost =
            batchCache
                .typeCost[static_cast<size_t>(BlockType::NORMAL_BLOCK)][static_cast<size_t>(BlockType::NORMAL_BLOCK)];
    }
    uint32_t sinkNumber = (baseInfo.keySinkNumber && baseInfo.supportFD) ? baseInfo.keySinkNumber : baseInfo.sinkNumber;
    if (sinkNumber > 0 &&
        baseInfo.sparseMode == 4U) { // FusedInferAttentionScoreV2Sink自定义算子新增部分：计算s1GIdx行sink块的Cost和块数
        CalcS1GCacheWithSinkNumber(s1GIdx, splitContext, batchCache, s1GCache, sinkNumber);
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.numOfFdHead = tmpRes.numOfFdHead;
    splitRes.maxS2SplitNum = tmpRes.maxS2SplitNum;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.gS1End[i] = tmpRes.gS1End[i];
        splitRes.s2End[i] = tmpRes.s2End[i];

        splitRes.fdRes.bN2IdxOfFdHead[i] = tmpRes.fdRes.bN2IdxOfFdHead[i];
        splitRes.fdRes.gS1IdxOfFdHead[i] = tmpRes.fdRes.gS1IdxOfFdHead[i];
        splitRes.fdRes.s2SplitNumOfFdHead[i] = tmpRes.fdRes.s2SplitNumOfFdHead[i];
        splitRes.fdRes.s2SplitStartIdxOfCore[i] = tmpRes.fdRes.s2SplitStartIdxOfCore[i];
        splitRes.fdRes.gS1SplitNumOfFdHead[i] = tmpRes.fdRes.gS1SplitNumOfFdHead[i];
        splitRes.fdRes.gS1LastPartSizeOfFdHead[i] = tmpRes.fdRes.gS1LastPartSizeOfFdHead[i];
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::ClearTmpResult(SplitResult &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.numOfFdHead = 0U;
    tmpResult.maxS2SplitNum = 0U;
    tmpResult.usedVecNumOfFd = 0U;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.gS1End[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.fdRes.bN2IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitStartIdxOfCore[i] = 0U;
        tmpResult.fdRes.gS1SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1LastPartSizeOfFdHead[i] = 0U;
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcBatchCost(uint32_t bIdx,
                                                                 const SplitContext &splitContext,
                                                                 CostInfo &costInfo)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx, baseInfo) == 0U || GetS2SeqSize(bIdx, baseInfo) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcCostInfo(SplitContext &splitContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;

    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * baseInfo.n2Size;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * baseInfo.n2Size;
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::UpdateCursor(const SplitContext &splitContext,
                                                                AssignContext &assignContext)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool upDataS1G = false;
    bool upDateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        upDataS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx ==
        baseInfo.bSize * baseInfo.n2Size) { // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
        assignContext.curS1GIdx = 0U;
        upDateBatch = true;
        upDataS1G = true;
    }

    // Update Cache
    if (upDateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (upDataS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::AssignByBatch(const SplitContext &splitContext,
                                                                 AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    const BaseInfo &baseInfo = splitContext.baseInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == baseInfo.bSize * baseInfo.n2Size) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / baseInfo.n2Size != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / baseInfo.n2Size;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::AssignByRow(const SplitContext &splitContext,
                                                               AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost
                                    ? assignContext.bN2Cost - assignContext.s1GCache.s1GCost
                                    : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock
                                     ? assignContext.bN2Block - assignContext.s1GCache.s1GBlock
                                     : 0U;
        // 计算新一行的信息
        assignContext.curS1GIdx++;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        while (assignContext.s1GCache.s1GBlock == 0) {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        }
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

// FusedInferAttentionScoreV2Sink自定义算子新增函数：按block块分配时，分配完之后需要更新curS2Idx,需要跳过sink和preToken之间被掩掉的块
void FusedInferAttentionScoreV2SinkMetadataKernel::UpdateCurS2IdxWithSinkNumber(const SplitContext &splitContext,
                                                                                AssignContext &assignContext,
                                                                                uint32_t sinkNumber)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    if (assignContext.s1GCache.s2End == 0) {
        return;
    }
    if (assignContext.s1GCache.firstTokenWithSinkNumber == 0) {
        assignContext.curS2Idx++;
        return;
    }
    uint32_t sinkBlockLastIndex =
        (static_cast<uint32_t>(sinkNumber) + splitParam.s2BaseSize - 1) / splitParam.s2BaseSize - 1;
    if (assignContext.curS2Idx < sinkBlockLastIndex ||
        assignContext.curS2Idx >= assignContext.s1GCache.firstTokenWithSinkNumber) {
        assignContext.curS2Idx++;
        return;
    }
    if (assignContext.s1GCache.firstTokenWithSinkNumber <= sinkBlockLastIndex) {
        assignContext.curS2Idx++;
    } else {
        assignContext.curS2Idx = assignContext.s1GCache.firstTokenWithSinkNumber;
    }
    return;
}

void FusedInferAttentionScoreV2SinkMetadataKernel::AssignByBlock(const SplitContext &splitContext,
                                                                 AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }
    uint32_t sinkNumber = (splitContext.baseInfo.keySinkNumber && splitContext.baseInfo.supportFD)
                              ? splitContext.baseInfo.keySinkNumber
                              : splitContext.baseInfo.sinkNumber;
    while (IsWithinTolerance(assignContext.coreCache.costLimit,
                             curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost +
                                 curCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;

        if (sinkNumber > 0 &&
            splitContext.baseInfo.sparseMode == 4U) { // FusedInferAttentionScoreV2Sink自定义算子新增部分：更新curS2Idx
            UpdateCurS2IdxWithSinkNumber(splitContext, assignContext, sinkNumber);
        } else {
            assignContext.curS2Idx++;
        }
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void FusedInferAttentionScoreV2SinkMetadataKernel::ForceAssign(const SplitContext &splitContext,
                                                               AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    uint32_t sinkNumber = (splitContext.baseInfo.keySinkNumber && splitContext.baseInfo.supportFD)
                              ? splitContext.baseInfo.keySinkNumber
                              : splitContext.baseInfo.sinkNumber;
    if (sinkNumber > 0 &&
        splitContext.baseInfo.sparseMode == 4U) { // FusedInferAttentionScoreV2Sink自定义算子新增部分：更新curS2Idx
        UpdateCurS2IdxWithSinkNumber(splitContext, assignContext, sinkNumber);
    } else {
        assignContext.curS2Idx++;
    }
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool FusedInferAttentionScoreV2SinkMetadataKernel::IsNeedRecordFDInfo(const AssignContext &assignContext,
                                                                      const SplitResult &splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void FusedInferAttentionScoreV2SinkMetadataKernel::RecordFDInfo(const SplitContext &splitContext,
                                                                const AssignContext &assignContext,
                                                                SplitResult &result)
{
    const BaseInfo &baseInfo = splitContext.baseInfo;
    const SplitParam &splitParam = splitContext.splitParam;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / baseInfo.n2Size;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx, baseInfo);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U)
                                ? (s1Size * baseInfo.gSize - splitS1GIdx * splitParam.mBaseSize)
                                : splitParam.mBaseSize;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + splitParam.gS1BaseSizeOfFd - 1U) / splitParam.gS1BaseSizeOfFd;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (splitParam.gS1BaseSizeOfFd * (curFdS1gSplitPart - 1U));
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.bN2IdxOfFdHead[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.gS1IdxOfFdHead[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.s2SplitNumOfFdHead[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.gS1SplitNumOfFdHead[result.numOfFdHead] = curFdS1gSplitPart;
    result.fdRes.gS1LastPartSizeOfFdHead[result.numOfFdHead] = curFdS1gLastPartSize;
    result.numOfFdHead++;
}

void FusedInferAttentionScoreV2SinkMetadataKernel::CalcSplitPlan(uint32_t coreNum,
                                                                 int64_t costLimit,
                                                                 const SplitContext &splitContext,
                                                                 SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    const BaseInfo &baseInfo = splitContext.baseInfo;

    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.fdRes.s2SplitStartIdxOfCore[assignContext.curCoreIdx] = assignContext.curKvSplitPart - 1U;

        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);

        // 1、按整batch分配
        AssignByBatch(splitContext, assignContext);
        // 2、按行分配
        AssignByRow(splitContext, assignContext);
        if (baseInfo.supportFD && !baseInfo.batchInvariant) {
            // 3、按块分配
            AssignByBlock(splitContext, assignContext);
            // 4、强制分配
            if (assignContext.coreCache.block == 0) {
                ForceAssign(splitContext, assignContext);
            }
        }
        result.bN2End[i] = assignContext.curBN2Idx;
        result.gS1End[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;
        if (baseInfo.supportFD && !baseInfo.batchInvariant) {
            // 对之前的归约信息进行记录并清理
            if (IsNeedRecordFDInfo(assignContext, result)) {
                RecordFDInfo(splitContext, assignContext, result);
                assignContext.curKvSplitPart = 1U;
            }

            // 更新S2切分信息
            if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
                assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
                assignContext.curKvSplitPart++;
            }
        }
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void FusedInferAttentionScoreV2SinkMetadataKernel::SplitFD(SplitResult &result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector =
        static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum); // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = result.fdRes.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < result.fdRes.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector); // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * FD_TOLERANCE_RATIO) {
                result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector); // 当前未分配的总负载
                // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    result.usedVecNumOfFd = curCoreIndex + 1;
}

void FusedInferAttentionScoreV2SinkMetadataKernel::SplitCore(uint32_t coreNum,
                                                             const BaseInfo &baseInfo,
                                                             const SplitParam &param,
                                                             SplitResult &result)
{
    SplitContext splitContext(baseInfo, param);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        result.usedCoreNum = 1U;
        result.bN2End[0] = baseInfo.bSize * baseInfo.n2Size;
        result.gS1End[0] = 0U;
        result.s2End[0] = 0U;
        return;
    }

    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    uint32_t maxCore = std::min(coreNum, splitContext.costInfo.totalBlockNum);
    uint32_t minCore =
        static_cast<uint32_t>(std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    result.maxCost = INT64_MAX;
    result.usedCoreNum = 1U;

    SplitResult tmpResult{coreNum, result.vecCubeRatio};
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, result.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < result.maxCost) {
            CopyTmpResult(tmpResult, result);
        }
        ClearTmpResult(tmpResult);
    }

    // 3、存在FD任务，对FD进行负载均衡分配
    if (result.numOfFdHead > 0U) {
        SplitFD(result);
    }
    result.usedCoreNum = std::max(result.usedCoreNum, 1U); // 至少使用1个core
}

REGISTER_CPU_KERNEL(FUSED_INFER_ATTENTION_SCORE_V2_SINK_METADATA, FusedInferAttentionScoreV2SinkMetadataKernel);
}  // namespace aicpu
