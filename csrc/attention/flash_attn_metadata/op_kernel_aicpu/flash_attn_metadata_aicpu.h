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
 * \file flash_attn_metadata_aicpu.h
 * \brief
 */

#ifndef FLASH_ATTN_METADATA_AICPU_H
#define FLASH_ATTN_METADATA_AICPU_H

#include <string>
#include <cstring>
#include <vector>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "flash_attn_metadata.h"
#include "../../a5_mla_common/op_kernel/load_balance/section_stream_k/section_stream_k.h"

namespace aicpu {

using optiling::FAG_CORE_LIST_NUM;
using optiling::FAG_ARRAY_LENGTH;
using optiling::FAG_SPLIT_AXIS_BN2GS1S2;
using optiling::FAG_SPARSE_NO_MASK;
using optiling::FAG_INPUT_FORMAT_BS2N2GD;

class FlashAttnMetadataCpuKernel : public CpuKernel {
public:
    FlashAttnMetadataCpuKernel() = default;
    ~FlashAttnMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool BalanceSchedule(load_balance::SectionStreamKResult &splitRes);
    bool GenMetadata(load_balance::SectionStreamKResult &splitRes);

    bool ParamsCheck();
    bool CheckActualQuerySeq();
    bool CheckActualKvSeq();

    bool ParamsInit();
    void InitDeviceInfo();
    void InitBaseInfo();
    void InitLoadBalanceParams();
    void LoadActualQuerySeq();
    void LoadActualKvSeq();

    void SetMetadataHead(const load_balance::SectionStreamKResult &splitRes, optiling::detail::FaMetadata &faMetadata);
    void SetMetadataFa(const load_balance::SectionStreamKResult &splitRes, optiling::detail::FaMetadata &faMetadata);
    void SetMetadataFd(const load_balance::SectionStreamKResult &splitRes, optiling::detail::FaMetadata &faMetadata);

private:
    CpuKernelContext *context_ = nullptr;
    // input tensor
    Tensor *cuSeqlensQ_ = nullptr;
    Tensor *cuSeqlensKv_ = nullptr;
    Tensor *sequsedQ_ = nullptr;
    Tensor *sequsedKv_ = nullptr;
    // output tensor
    Tensor *metadata_ = nullptr;

    // input attr
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t headDimV_ = -1; // -1: 未指定, 归一化后等于 headDim_
    int32_t maskMode_ = 1;
    int32_t winLeft_ = -1;
    int32_t winRight_ = -1;
    std::string layoutQ_ = "BSND";
    std::string layoutKv_ = "BSND";
    std::string layoutOut_ = "BSND";
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36U; // 36: default aic num
    int32_t aivCoreNum_ = 72U; // 72: default aiv num

    // BaseInfo
    bool isActualSeqlenQAccum_ = false;
    bool isActualSeqlenKvAccum_ = false;
    std::vector<int64_t> actualSeqlenQ_{};
    std::vector<int64_t> actualSeqlenKv_{};

    // SplitParams
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 64;   // 64: default value
    uint32_t s2BaseSize_ = 128; // 128: default value
    load_balance::DeviceInfo deviceInfo;
    load_balance::BaseInfo baseInfo;
    load_balance::SectionStreamKParam param;

    // FAG split params (Flash Attn Grad metadata)
    int64_t fagS1Inner_ = 64;
    int64_t fagS2Inner_ = 128;
    int64_t fagS1CvInner_ = 128;
    int64_t fagCvS2Inner_ = 128;
    int64_t fagS1Outer_ = 0;
    int64_t fagS2Outer_ = 0;
    int64_t fagG_ = 0;
    int64_t fagBlockStarts_[FAG_CORE_LIST_NUM] = {0};
    int64_t fagBlockEnds_[FAG_CORE_LIST_NUM] = {0};
    int64_t fagTndStartBIdx_[FAG_CORE_LIST_NUM] = {0};
    uint32_t fagBlockOuter_ = 0;
    int64_t fagBlockFactor_ = 0;
    uint32_t fagSplitAxis_ = FAG_SPLIT_AXIS_BN2GS1S2;
    bool fagIsSparse_ = false;
    uint32_t fagSparseMode_ = FAG_SPARSE_NO_MASK;
    uint32_t fagLayoutType_ = FAG_INPUT_FORMAT_BS2N2GD;
    int64_t fagS1Token_ = -1;
    int64_t fagS2Token_ = -1;
    bool fagIsBn2_ = false;
    bool fagIsBn2MultiBlk_ = false;
    bool fagIsAllSame_ = false;
    bool fagIsInvalidCol_ = false;
    bool fagIsInvalidRow_ = false;
    bool fagIsGradEnabled_ = false;
    bool fagUseTndSplit_ = false;
    bool fagIsSeqExistZero_ = false;

    // FAG method declarations
    void InitFagParams();
    void InitFagBaseDims();
    void InitFagLayoutAndSparse();
    void InitFagActualSeqlen();
    void DoFagSparse();
    void GenFagMetadata(uint32_t sectionNum);

    // FAG split implementations (defined in flash_attn_grad_metadata_split.h)
    void SetFagSplitAxis();
    void SupportTransBSND();
    void DoFagDenseSplit();
    void DoFagTndSplit();
    void DoFagSparseBlockInfo();
    void GetFagParseS1S2OuterInfo(std::vector<std::vector<int64_t>> &parseInfo);
    bool FagCheckSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx);
    bool FagIsValid(int64_t blockIdx);
    bool FagIsValidUnpad(int64_t blockIdx);
    bool FagCheckUnpadSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx);
    void FagFillBlockInfoLoadBalance(std::vector<std::vector<int64_t>> &totalBlockInfo,
                                     std::vector<std::vector<float>> &acturalBlockInfo);

    // BN2S2 split implementations
    void TryBn2s2Sparse();
    void DoFagBn2s2DenseSplit();
    void DoFagBn2s2SparseTndSplit();
    float Bn2s2BinarySearchMaxBlockNumPerCore(int64_t b, int64_t n2, int64_t g, int64_t aicNum,
                                              std::vector<std::vector<int64_t>> &totalBlockInfo,
                                              std::vector<std::vector<float>> &acturalBlockInfo);
    bool Bn2s2IsPossible(int64_t b, int64_t n2, int64_t g, int64_t aicNum, float possibleMax,
                         std::vector<std::vector<float>> &acturalBlockInfo);
    bool Bn2s2CaclePerCoreBlockInfo(int64_t b, int64_t n2, int64_t g, int64_t aicNum,
                                    std::vector<std::vector<int64_t>> &totalBlockInfo,
                                    std::vector<std::vector<float>> &acturalBlockInfo, float maxBlockNumPerCore);

    // BN2 split implementations
    bool TryBn2MultiBlkSparse();
    void DoFagBn2DenseSplit();
    bool DoFagBn2SparseBlockInfo();

private:
    enum ParamId {
        // input
        cuSeqlensQ = 0,
        cuSeqlensKv = 1,
        sequsedQ = 2,
        sequsedKv = 3,
        // output
        metaData = 0,
    };
};
} // namespace aicpu

#endif
