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
 * \file sparse_flash_mla_csa_kernel_arch35.h
 * \brief
 */

#ifndef SPARSE_FLASH_MLA_CSA_KERNEL_ARCH35_H
#define SPARSE_FLASH_MLA_CSA_KERNEL_ARCH35_H
#include "sparse_flash_mla_common_arch35.h"
#include "sparse_flash_mla_kvcache.h"
#include "sparse_flash_mla_csa_block_cube_arch35.h"
#include "sparse_flash_mla_csa_block_vector_arch35.h"
#include "kernel_operator.h"
#include "../sparse_flash_mla_kernel_metadata.h"

#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif
#if __has_include("common/buffers_policy_3buff_sfa.h")
#include "common/buffers_policy_3buff_sfa.h"
#endif

#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using AttentionCommon::FdRunInfo;

namespace SMLAKernel {
template <typename CubeBlockType, typename VecBlockType>
class SparseFlashMlaCsaKernel {
public:
    ARGS_TRAITS;
    __aicore__ inline SparseFlashMlaCsaKernel(){};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV,
                                __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
                                __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *workspace, const SparseFlashMlaTilingData *__restrict tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline int64_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                        GlobalTensor<int32_t> &actualSeqGm, GlobalTensor<int32_t> &cuSeqlensGm,
                                        int64_t defaultSize);
    __aicore__ inline void ParseTilingData(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                           __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
                                           __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
                                           __gm__ uint8_t *cmpResidualKV);
    __aicore__ inline void InitGlobalBuffer(
        __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
        __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
        __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
        __gm__ uint8_t *cmpResidualKV, __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength,
        __gm__ uint8_t *sinks, __gm__ uint8_t *workspace, const SparseFlashMlaTilingData *__restrict tiling);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void FreeEvent();
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *workspace);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo);
    __aicore__ inline void ParseFdRunInfo(FdRunInfo &fdRunInfo);
    __aicore__ inline int64_t ConvertS2MetadataBlockToToken(const RunParamStr &runParam, const ConstInfo &constInfo,
                                                            uint32_t s2BlockIdx);
    __aicore__ inline bool ApplyS2MetadataRange(RunParamStr &runParam, ConstInfo &constInfo, int64_t s2StartPoint,
                                                int64_t s2EndPoint, bool isFirstS2RangeTask, bool isLastS2RangeTask);
    const SparseFlashMlaTilingData *__restrict tilingData;
    /* 编译期常量的基本块信息 */
    static constexpr uint32_t PRELOAD_NUM = 3;
    static constexpr uint32_t crossCoreMte2SyncFlagId = 15; // IS_SPLIT_G 核间 MTE2 同步 flag ID
    static constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;

    /* 核间通道 */
    BufferManager<BufferType::GM> v0ResGmBufferManager;

    StaticBuffer<T> bmm1Buffers[2];
    StaticBuffer<T> bmm2Buffers;
    uint32_t bmm1GetFlag = 0;
    uint32_t vUbBase = 0;

    // mm2左矩阵P
    StaticBuffer<Q_T> l1PBuffers[2];
    uint32_t l1PGetFlag = 0;
    uint32_t l1CubeBase = 0;
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqOriKvlenGm;
    GlobalTensor<int32_t> actualSeqCmpKvlenGm;
    GlobalTensor<int32_t> cmpResidualKvGm;
    GlobalTensor<int32_t> actualSeqQlenGm;
    GlobalTensor<int32_t> oriTopkLengthGm;
    GlobalTensor<int32_t> cmpTopkLengthGm;

    bool hasCuSeqlensQ = false;
    bool hasCuSeqlensOriKv = false;
    bool hasCuSeqlensCmpKv = false;
    bool hasActualSeqQlen = false;
    bool hasActualSeqOriKvlen = false;
    bool hasActualSeqCmpKvlen = false;
    /* workspace 空间 */
    BuffersPolicy3buffSFA<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> v0ResGmBuffers;
    BufferManager<BufferType::GM> fdStagingBufferManager;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> fdStagingBuffer;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> intraCoreCombineBuffer;
    BuffersPolicySingleBuffer<BufferType::GM, SyncType::NO_SYNC> crossCoreCombineBuffer;
    /* 核Index信息 */
    int32_t aicIdx;

    /* Init阶段metadata解析结果 */
    uint32_t bN2StartIdx;
    uint32_t gS1StartIdx;
    uint32_t bN2EndIdx;
    uint32_t nextGs1Idx;
    uint32_t hasLoad;

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const SparseFlashMlaTilingData *__restrict tiling)
{
    fa_base_matmul::ResetIdCounter();
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        constInfo.aivIdx = 0;
        this->tilingData = tiling;
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }

    if (metadata == nullptr) {
        return;
    }
    this->metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);

    constInfo.s1BaseSize = 64;
    constInfo.s2BaseSize = 128;

    this->ParseTilingData(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV,
                          cmpResidualKV);
    vecBlock.InitVecBlock(cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV);
    vecBlock.CleanOutput(attentionOut, softmaxLse, constInfo);

    // 从meta data解析分核信息
    bN2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_START_INDEX, false));
    gS1StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_START_INDEX, false));
    bN2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_END_INDEX, false));
    nextGs1Idx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_END_INDEX, false));
    hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    this->InitGlobalBuffer(query, oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable,
                           cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, seqUsedOriKV, seqUsedCmpKV,
                           cmpResidualKV, oriTopkLength, cmpTopkLength, sinks, workspace, tiling); // gm设置

    if ASCEND_IS_AIV {
        if constexpr ((TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                       TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                       TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                      IS_VEC_S2PHYADDR) {
            this->vecBlock.GetKVPhyAddr(hasLoad, bN2StartIdx, bN2EndIdx, gS1StartIdx, nextGs1Idx, hasActualSeqQlen,
                                        hasCuSeqlensQ, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm,
                                        cuSeqlensOriKvGm, oriTopkLengthGm, hasActualSeqCmpKvlen, hasCuSeqlensCmpKv,
                                        actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm, cmpResidualKvGm,
                                        actualSeqQlenGm, cuSeqlensQGm, workspace, constInfo);
        }
    }

    InitMMResBuf(workspace);
    if constexpr (IS_BATCH_CONSISTENCY) {
        vecBlock.InitS2SplitStaging(intraCoreCombineBuffer.Get(), crossCoreCombineBuffer.Get());
    } else {
        vecBlock.InitS2SplitStaging(fdStagingBuffer.Get());
    }
    this->ComputeConstexpr();
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::GetSeqLen(
    int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens, GlobalTensor<int32_t> &actualSeqGm,
    GlobalTensor<int32_t> &cuSeqlensGm, int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    } else if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    } else {
        return defaultSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ParseTilingData(
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
    __gm__ uint8_t *cmpResidualKV)
{
    auto &sparseFlashMLABaseParams = this->tilingData->baseParams;
    auto &sparseFlashMLACmpParams = this->tilingData->cmpParams;
    constInfo.bSize = sparseFlashMLABaseParams.batchSize;
    constInfo.n2Size = 1;
    constInfo.gSize = sparseFlashMLABaseParams.nNumOfQInOneGroup;
    constInfo.s1Size = sparseFlashMLABaseParams.qSeqSize;
    constInfo.s2Size = sparseFlashMLABaseParams.kvSeqSize;
    constInfo.cmpS2Size = sparseFlashMLACmpParams.cmpKvSeqSize;
    constInfo.oriSparseBlockCount = sparseFlashMLABaseParams.oriSparseBlockCount;
    constInfo.cmpSparseBlockCount = sparseFlashMLACmpParams.cmpSparseBlockCount;
    constInfo.alignedOriSparseBlockCount =
        (constInfo.oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) / SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    constInfo.alignedCmpSparseBlockCount =
        (constInfo.cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) / SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    constInfo.cmpRatio = sparseFlashMLACmpParams.cmpRatio;
    constInfo.oriMaskMode = sparseFlashMLABaseParams.oriMaskMode;
    constInfo.cmpMaskMode = sparseFlashMLACmpParams.cmpMaskMode;
    constInfo.oriWinLeft = sparseFlashMLABaseParams.oriWinLeft;
    constInfo.oriWinRight = sparseFlashMLABaseParams.oriWinRight;
    constInfo.layoutType = sparseFlashMLABaseParams.outputLayout;
    constInfo.returnSoftmaxLse = sparseFlashMLABaseParams.returnSoftmaxLse;
    constInfo.tileSize = 0;
    constInfo.dSizeRope = 64;
    constInfo.oriKeyStride0 = sparseFlashMLABaseParams.oriKeyStride0;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        constInfo.cmpKeyStride0 = sparseFlashMLACmpParams.cmpKeyStride0;
    }
    if ASCEND_IS_AIV {
        constInfo.softmaxScale = sparseFlashMLABaseParams.softmaxScale;
    }
    constInfo.dSize = 512;
    constInfo.dSizeV = constInfo.dSize;
    constInfo.dSizeVInput = constInfo.dSize;
    constInfo.dSizeNope = constInfo.dSize - constInfo.dSizeRope;
    constInfo.sparseBlockSize = 1;
    constInfo.actualSeqLenSize = constInfo.bSize + 1;
    constInfo.actualLenDimsOriKV = sparseFlashMLABaseParams.actualLenDimsOriKV;
    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
        constInfo.actualLenDimsCmpKV = sparseFlashMLABaseParams.actualLenDimsCmpKV;
        constInfo.cmpResidualKVSize = sparseFlashMLABaseParams.cmpResidualKVSize;
    }
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        this->constInfo.isActualLenDimsOriKVNull = 0U;
    } else {
        this->constInfo.isActualLenDimsOriKVNull = (seqUsedOriKV == nullptr);
    }

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        constInfo.oriBlockSize = sparseFlashMLABaseParams.oriBlockSize;
        constInfo.cmpBlockSize = sparseFlashMLABaseParams.cmpBlockSize;
        constInfo.oriMaxBlockNumPerBatch = sparseFlashMLABaseParams.oriMaxBlockNumPerBatch;
        constInfo.cmpMaxBlockNumPerBatch = sparseFlashMLACmpParams.cmpMaxBlockNumPerBatch;
    }

    if (cuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensOriKv != nullptr) {
        cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
        hasCuSeqlensOriKv = true;
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (cuSeqlensCmpKv != nullptr) {
            cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            hasCuSeqlensCmpKv = true;
        }
    }

    if (sequsedQ != nullptr) {
        actualSeqQlenGm.SetGlobalBuffer((__gm__ int32_t *)sequsedQ);
        hasActualSeqQlen = true;
    }
    if (seqUsedOriKV != nullptr) {
        actualSeqOriKvlenGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedOriKV);
        hasActualSeqOriKvlen = true;
    }

    if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                  TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
        if (seqUsedCmpKV != nullptr) {
            actualSeqCmpKvlenGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV);
            hasActualSeqCmpKvlen = true;
        }
        if (cmpResidualKV != nullptr) {
            cmpResidualKvGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKV);
        }
    }

    constInfo.needInit = 0;
    if (TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
        TEMPLATE_MODE != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE && constInfo.oriMaskMode != 0) {
        for (uint32_t bIdx = 0; bIdx < constInfo.bSize; bIdx++) {
            int64_t s2Size;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                s2Size = actualSeqOriKvlenGm.GetValue(bIdx);
            } else {
                s2Size = GetSeqLen(bIdx, hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm, cuSeqlensOriKvGm,
                                   constInfo.s2Size);
            }
            int64_t s1Size =
                GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
            int64_t expectQs;
            if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
                expectQs = GetSeqLen(bIdx, false, hasCuSeqlensQ, actualSeqQlenGm, cuSeqlensQGm, constInfo.s1Size);
            } else {
                expectQs = constInfo.s1Size;
            }
            if (s1Size > s2Size || s1Size < expectQs) {
                constInfo.needInit = 1;
                break;
            }
        }
    } else {
        constInfo.needInit = 1;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
    __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
    const SparseFlashMlaTilingData *__restrict tiling)
{
    vecBlock.InitGlobalBuffer(oriKV, cmpKV, oriSparseIndices, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ,
                              sinks, seqUsedOriKV, seqUsedCmpKV, cmpResidualKV);
    cubeBlock.InitGlobalBuffer(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ,
                               cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, constInfo);

    if (oriTopkLength != nullptr) {
        constInfo.hasOriTopkLength = true;
        oriTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)oriTopkLength);
    } else {
        constInfo.hasOriTopkLength = false;
    }
    if (cmpTopkLength != nullptr) {
        constInfo.hasCmpTopkLength = true;
        cmpTopkLengthGm.SetGlobalBuffer((__gm__ int32_t *)cmpTopkLength);
    } else {
        constInfo.hasCmpTopkLength = false;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitMMResBuf(__gm__ uint8_t *workspace)
{
    // L1: [l1P x2][cube L1], l1P 必须放在最前面保证与 vec 申请地址相同
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize;
    uint32_t l1PAddr = 0;
    l1PBuffers[0] = {LocalTensor<Q_T>(TPosition::A1, l1PAddr, mm2LeftSize), 0};
    l1PAddr += (mm2LeftSize * sizeof(Q_T));
    l1PBuffers[1] = {LocalTensor<Q_T>(TPosition::A1, l1PAddr, mm2LeftSize), 1};
    l1PAddr += (mm2LeftSize * sizeof(Q_T));
    l1CubeBase = l1PAddr;

    // UB: [bmm2][bmm1 x2][vec UB]
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize;
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512;
    uint32_t ubAddr = 0;
    bmm2Buffers = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm2ResultSize), 0};
    ubAddr += (mm2ResultSize * sizeof(T));
    bmm1Buffers[0] = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm1ResultSize), 0};
    ubAddr += (mm1ResultSize * sizeof(T));
    bmm1Buffers[1] = {LocalTensor<T>(TPosition::VECIN, ubAddr, mm1ResultSize), 1};
    ubAddr += (mm1ResultSize * sizeof(T));
    vUbBase = ubAddr;

    if ASCEND_IS_AIV {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1Buffers[0].idx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM1(bmm1Buffers[1].idx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSSCORE_BMM2);
    }

    if constexpr (IS_SPLIT_G || TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        uint32_t v0ResSize = constInfo.s2BaseSize * 512U * sizeof(Q_T);
        int64_t v0ResTotalOffset;
        if constexpr (IS_SPLIT_G) {
            v0ResTotalOffset = v0ResSize * 3 * (aicIdx >> 1U);
        } else {
            v0ResTotalOffset = v0ResSize * 3 * aicIdx;
        }
        v0ResGmBufferManager.Init(workspace + v0ResTotalOffset);
        v0ResGmBuffers.Init(v0ResGmBufferManager, v0ResSize);
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(0));
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(1));
        v0ResGmBuffers.Get().SetCrossCoreID(INVALID_CROSS_CORE_EVENT_ID, CROSSCORE_V0RES(2));
    }
    int64_t fdStagingOffset = 0LL;
    if constexpr (IS_SPLIT_G || TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        constexpr int64_t TRIPLE_BUFFER_NUM = 3LL;
        int64_t v0ResSize = static_cast<int64_t>(constInfo.s2BaseSize) * constInfo.dSize * sizeof(Q_T);
        uint32_t v0LogicalSlotCount = IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum();
        fdStagingOffset = v0ResSize * TRIPLE_BUFFER_NUM * v0LogicalSlotCount;
        fdStagingOffset += TRIPLE_BUFFER_NUM * constInfo.s2BaseSize * sizeof(int32_t) * GetBlockNum();
        if constexpr (IS_VEC_S2PHYADDR) {
            int64_t totalBS1 = (LAYOUT_T == SMLA_LAYOUT::TND) ?
                                   static_cast<int64_t>(constInfo.s1Size) :
                                   static_cast<int64_t>(constInfo.bSize) * constInfo.s1Size;
            if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                          TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
                fdStagingOffset += totalBS1 * constInfo.alignedOriSparseBlockCount * sizeof(int64_t);
            }
            if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                          TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
                fdStagingOffset += totalBS1 * constInfo.alignedCmpSparseBlockCount * sizeof(int64_t);
            }
        }
    }
    fdStagingBufferManager.Init(workspace + fdStagingOffset);
    constexpr uint32_t FD_MAX_SUM_REGION_NUM = 2U;
    uint32_t gSize = static_cast<uint32_t>(constInfo.gSize);
    uint32_t combineElemSize =
        gSize * constInfo.dSize +
        FD_MAX_SUM_REGION_NUM * gSize * static_cast<uint32_t>(AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW);
    if constexpr (IS_BATCH_CONSISTENCY) {
        uint32_t intraCoreSlotNum = IS_SPLIT_G ? GetBlockNum() : (GetBlockNum() << 1U);
        uint32_t intraCoreCombineSize = intraCoreSlotNum * combineElemSize * sizeof(float);
        uint32_t crossCoreCombineSize =
            GetBlockNum() * BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM * combineElemSize * sizeof(float);
        intraCoreCombineBuffer.Init(fdStagingBufferManager, intraCoreCombineSize);
        crossCoreCombineBuffer.Init(fdStagingBufferManager, crossCoreCombineSize);
    } else {
        uint32_t fdSlotCount = static_cast<uint32_t>(AttentionCommon::FD_MAX_S2_SPLIT_NUM) *
                               (IS_SPLIT_G ? (GetBlockNum() >> 1U) : GetBlockNum());
        fdStagingBuffer.Init(fdStagingBufferManager, fdSlotCount * combineElemSize * sizeof(float));
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(constInfo, vUbBase);
    cubeBlock.InitLocalBuffer(l1CubeBase);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积

    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = constInfo.s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;

    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if constexpr (IS_VEC_S2PHYADDR) {
        SyncAll<false>();
    } else if (this->constInfo.needInit) {
        SyncAll<false>();
    }
    FdRunInfo fdRunInfo;
    if ASCEND_IS_AIV {
        ParseFdRunInfo(fdRunInfo);
    }
    ProcessMainLoop();
    if ASCEND_IS_AIV {
        SyncAll();
        if (fdRunInfo.coreEnable) {
            this->vecBlock.ProcessFlashDecode(fdRunInfo, this->constInfo);
        }
    }
    FreeEvent();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false));
    int64_t maxS2LoopCnt = 0;
    if constexpr (IS_SPLIT_G) {
        maxS2LoopCnt = static_cast<int64_t>(metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_MAX_NUM, false)));
    }
    if (hasLoad == 0) {
        if ASCEND_IS_AIC {
            if constexpr (IS_SPLIT_G) {
                for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                    CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                    CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                }
            }
        }
        return;
    }

    // 从meta data解析分核信息
    uint32_t bN2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_START_INDEX, false));
    uint32_t gS1StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_START_INDEX, false));
    uint32_t s2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_START_INDEX, false));
    uint32_t bN2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_END_INDEX, false));
    uint32_t nextGs1Idx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_END_INDEX, false));
    uint32_t s2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_END_INDEX, false));
    uint32_t firstFdDataWorkspaceIdx =
        metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, false));

    uint32_t s2LoopLimit = 0;

    if (nextGs1Idx != 0 || s2EndIdx != 0) {
        bN2EndIdx++;
    }

    int64_t taskId = 0;
    bool notLast = true;
    bool isFirstLoop = true;
    RunInfo runInfo[4];
    RunParamStr runParam;
    runParam.firstFdDataWorkspaceIdx = firstFdDataWorkspaceIdx;
    int64_t multiCoreInnerIdx = 1;
    int64_t s2SplitIdxCounter = 0;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        runParam.boIdx = bnIdx;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(
            runParam, this->constInfo, this->cuSeqlensQGm, this->cuSeqlensOriKvGm, this->cuSeqlensCmpKvGm,
            this->actualSeqQlenGm, this->actualSeqOriKvlenGm, this->actualSeqCmpKvlenGm, this->cmpResidualKvGm,
            this->hasActualSeqQlen, this->hasActualSeqOriKvlen, this->hasActualSeqCmpKvlen, this->hasCuSeqlensCmpKv);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, lastBN, nextGs1Idx, gS1StartIdx, s2EndIdx);

        int64_t gS1LoopEnd = lastBN ? (runParam.gs1LoopEndIdx + PRELOAD_NUM) : runParam.gs1LoopEndIdx;
        for (int64_t gS1Index = runParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool notLastThreeLoop = true;
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.gs1LoopEndIdx;
                switch (extraGS1) {
                    case 0:
                        notLastThreeLoop = false;
                        break;
                    case 1:
                        notLastTwoLoop = false;
                        notLastThreeLoop = false;
                        break;
                    case 2:
                        notLast = false;
                        notLastTwoLoop = false;
                        notLastThreeLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastThreeLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc =
                    ComputeParamS1<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, gS1Index, this->cuSeqlensQGm);
                bool s2NoNeedCalc = ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(
                    bnIdx, gS1Index, this->cuSeqlensQGm, oriTopkLengthGm, cmpTopkLengthGm, runParam, this->constInfo);
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t oriLoad = runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx;
                    int64_t cmpLoad = runParam.s2CmpLineEndIdx - runParam.s2CmpLineStartIdx;
                    int64_t totalLoad = oriLoad + cmpLoad;
                    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
                    int64_t rawReductionBlockSize = totalLoad / 32LL;
                    int64_t reductionBlockSize = (rawReductionBlockSize + s2BaseSize - 1LL) / s2BaseSize * s2BaseSize;
                    runParam.baseBlockNumPerReductionBlock =
                        reductionBlockSize > 0 ? reductionBlockSize / s2BaseSize : 1LL;
                }
                if (!s2NoNeedCalc) {
                    bool isFirstS2RangeTask = (bnIdx == bN2StartIdx && gS1Index == runParam.gs1LoopStartIdx);
                    bool isLastS2RangeTask = (lastBN && gS1Index == runParam.gs1LoopEndIdx - 1);
                    int64_t s2StartPoint = ConvertS2MetadataBlockToToken(runParam, this->constInfo, s2StartIdx);
                    int64_t s2EndPoint = (isLastS2RangeTask && s2EndIdx == 0) ?
                                             0 :
                                             ConvertS2MetadataBlockToToken(runParam, this->constInfo, s2EndIdx);
                    s2NoNeedCalc = ApplyS2MetadataRange(runParam, this->constInfo, s2StartPoint, s2EndPoint,
                                                        isFirstS2RangeTask, isLastS2RangeTask);
                } else {
                    runParam.isCrossCoreSplit = false;
                }
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                if constexpr (!IS_BATCH_CONSISTENCY) {
                    if (runParam.isCrossCoreSplit) {
                        runParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
                if constexpr (IS_SPLIT_G) {
                    maxS2LoopCnt -= (s2LoopLimit + 1);
                }
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if constexpr (IS_BATCH_CONSISTENCY) {
                    int64_t safeBaseBlockNum =
                        runParam.baseBlockNumPerReductionBlock > 0 ? runParam.baseBlockNumPerReductionBlock : 1LL;
                    if (runParam.isCrossCoreSplit && s2LoopCount % safeBaseBlockNum == 0) {
                        runParam.s2SplitIdx = s2SplitIdxCounter++;
                    }
                }
                if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                              TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                              TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
                    if (notLastThreeLoop) {
                        RunInfo &runInfo1 = runInfo[taskId % 4];
                        this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    }
                    if ASCEND_IS_AIV {
                        if (notLastThreeLoop) {
                            RunInfo &runInfo1 = runInfo[taskId % 4];
                            this->vecBlock.ProcessVec0(this->v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                       this->constInfo);
                        }
                        if (taskId > 1 && notLast) {
                            uint32_t bmm1Slot = bmm1GetFlag;
                            bmm1GetFlag ^= 1;
                            uint32_t l1PSlot = l1PGetFlag;
                            l1PGetFlag ^= 1;
                            auto &runInfo2 = runInfo[(taskId + 2) % 4];
                            this->vecBlock.ProcessVec1(this->l1PBuffers[l1PSlot], this->bmm1Buffers[bmm1Slot], runInfo2,
                                                       this->constInfo);
                        }
                        if (taskId > 2) {
                            RunInfo &runInfo3 = runInfo[(taskId + 1) % 4];
                            this->vecBlock.ProcessVec2(this->bmm2Buffers, runInfo3, this->constInfo);
                        }
                    } else {
                        if (taskId > 0 && notLastTwoLoop) {
                            RunInfo &runInfo1 = runInfo[(taskId + 3) % 4];
                            this->cubeBlock.IterateLoadQK(this->v0ResGmBuffers.Get(runInfo1.taskIdMod3), runInfo1,
                                                          this->constInfo, isFirstLoop);
                            isFirstLoop = false;
                        } else {
                            if constexpr (IS_SPLIT_G) {
                                if (taskId > 0 && maxS2LoopCnt > 0) {
                                    maxS2LoopCnt--;
                                    CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                                    CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                                }
                            }
                        }
                        if (taskId > 1 && notLast) {
                            uint32_t bmm1Slot = bmm1GetFlag;
                            bmm1GetFlag ^= 1;
                            auto &runInfo2 = runInfo[(taskId + 2) % 4];
                            RunInfo &runInfoNext = runInfo[(taskId + 3) % 4];
                            this->cubeBlock.IterateBmm1(this->bmm1Buffers[bmm1Slot], notLastTwoLoop, runInfoNext,
                                                        runInfo2, this->constInfo);
                        }
                        if (taskId > 2) {
                            uint32_t l1PSlot = l1PGetFlag;
                            l1PGetFlag ^= 1;
                            RunInfo &runInfo3 = runInfo[(taskId + 1) % 4];
                            this->cubeBlock.IterateBmm2(this->bmm2Buffers, this->l1PBuffers[l1PSlot], runInfo3,
                                                        this->constInfo);
                        }
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
    if ASCEND_IS_AIC {
        if constexpr (IS_SPLIT_G) {
            for (int64_t loopCnt = 0; loopCnt < maxS2LoopCnt; loopCnt++) {
                CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
                CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
            }
        }
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam)
{
    // GS1合轴, 不切G, 只切S1
    runParam.s1oIdx = gS1Index * runParam.qSNumInOneBlock;
    if constexpr (IS_SPLIT_G) {
        int64_t halfG = (constInfo.gSize + 1) / 2; // ceil(gSize/2), 第一个AIC多处理一行
        runParam.goIdx = (aicIdx % 2 == 0) ? 0 : halfG;
        runParam.gSplitSize = (aicIdx % 2 == 0) ? halfG : (constInfo.gSize - halfG);
    } else {
        runParam.goIdx = 0;
        runParam.gSplitSize = constInfo.gSize;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit,
    int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2OriLineStartIdx;
        runInfo.s2EndIdx = runParam.s2OriLineEndIdx;
    } else {
        runInfo.s2StartIdx = runParam.s2CmpLineStartIdx;
        runInfo.s2EndIdx = runParam.s2CmpLineEndIdx;
    }
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }

    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    runInfo.firstFdDataWorkspaceIdx = runParam.firstFdDataWorkspaceIdx;
    runInfo.isCrossCoreSplit = runParam.isCrossCoreSplit;
    runInfo.s2SplitIdx = runParam.s2SplitIdx;
    runInfo.isFirstS2SplitCore = runParam.isFirstS2SplitCore;
    int64_t safeBaseBlockNum =
        runParam.baseBlockNumPerReductionBlock > 0 ? runParam.baseBlockNumPerReductionBlock : 1LL;
    int64_t baseBlockIdInReduceBlock = s2LoopCount % safeBaseBlockNum;
    runInfo.reduceBlockId = s2LoopCount / safeBaseBlockNum;
    runInfo.isFirstBase = baseBlockIdInReduceBlock == 0;
    runInfo.isLastBase = baseBlockIdInReduceBlock == safeBaseBlockNum - 1LL || s2LoopCount == s2LoopLimit;
    runInfo.needReduce = runInfo.reduceBlockId > 0;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ComputeBmm1Tail(RunInfo &runInfo,
                                                                                             RunParamStr &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    runInfo.firstHalfMRealSize = runParam.firstHalfMRealSize;

    runInfo.vec2MBaseSize = runInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = constInfo.s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    int64_t curS2LoopCnt = (runInfo.s2LoopCount >= runParam.oriKvLoopEndIdx) ?
                               (runInfo.s2LoopCount - runParam.oriKvLoopEndIdx) :
                               runInfo.s2LoopCount;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ParseFdRunInfo(FdRunInfo &fdRunInfo)
{
    uint32_t aivIdx = static_cast<uint32_t>(this->constInfo.aivIdx);
    fdRunInfo.coreEnable = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_CORE_ENABLE_INDEX, true)) != 0;
    if (!fdRunInfo.coreEnable) {
        return;
    }
    fdRunInfo.bn2Idx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_BN2_IDX_INDEX, true));
    fdRunInfo.mIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_IDX_INDEX, true));
    fdRunInfo.workspaceIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_IDX_INDEX, true));
    fdRunInfo.workspaceNum = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_WORKSPACE_NUM_INDEX, true));
    fdRunInfo.mStartIdx = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_START_INDEX, true));
    fdRunInfo.mNum = metadataGm.GetValue(GetAttrAbsIndex(aivIdx, FD_M_NUM_INDEX, true));
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline int64_t SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ConvertS2MetadataBlockToToken(
    const RunParamStr &runParam, const ConstInfo &constInfo, uint32_t s2BlockIdx)
{
    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriLen = runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx;
    int64_t cmpLen = runParam.s2CmpLineEndIdx - runParam.s2CmpLineStartIdx;
    int64_t reductionBlockSize = runParam.baseBlockNumPerReductionBlock * s2BaseSize;
    int64_t oriReductionBlockNum = (oriLen + reductionBlockSize - 1) / reductionBlockSize;
    int64_t reductionBlockIdx = static_cast<int64_t>(s2BlockIdx);
    if (reductionBlockIdx < oriReductionBlockNum) {
        int64_t oriToken = reductionBlockIdx * reductionBlockSize;
        return oriToken < oriLen ? oriToken : oriLen;
    }
    int64_t cmpToken = (reductionBlockIdx - oriReductionBlockNum) * reductionBlockSize;
    return oriLen + (cmpToken < cmpLen ? cmpToken : cmpLen);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::FreeEvent()
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[0].idx));
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[0].idx) + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[1].idx));
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(bmm1Buffers[1].idx) + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2 + AIV0_AIV1_OFFSET);
        this->cubeBlock.FreeEvent();
    } else {
        this->vecBlock.FreeEvent(constInfo);
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline bool SparseFlashMlaCsaKernel<CubeBlockType, VecBlockType>::ApplyS2MetadataRange(
    RunParamStr &runParam, ConstInfo &constInfo, int64_t s2StartPoint, int64_t s2EndPoint, bool isFirstS2RangeTask,
    bool isLastS2RangeTask)
{
    int64_t oriStart = runParam.s2OriLineStartIdx;
    int64_t oriEnd = runParam.s2OriLineEndIdx;
    int64_t oriLen = oriEnd - oriStart;
    int64_t cmpStart = runParam.s2CmpLineStartIdx;
    int64_t cmpEnd = runParam.s2CmpLineEndIdx;
    int64_t cmpLen = cmpEnd - cmpStart;
    int64_t totalLen = oriLen + cmpLen;

    int64_t effectiveS2EndPoint = (isLastS2RangeTask && s2EndPoint == 0) ? totalLen : s2EndPoint;
    int64_t rangeStart = isFirstS2RangeTask ? s2StartPoint : 0;
    rangeStart = rangeStart < 0 ? 0 : rangeStart;
    rangeStart = rangeStart < totalLen ? rangeStart : totalLen;
    int64_t rangeEnd = isLastS2RangeTask ? effectiveS2EndPoint : totalLen;
    rangeEnd = rangeEnd < 0 ? 0 : rangeEnd;
    rangeEnd = rangeEnd < totalLen ? rangeEnd : totalLen;
    if (rangeEnd <= rangeStart) {
        runParam.oriKvLoopEndIdx = 0;
        runParam.cmpKvLoopEndIdx = 0;
        runParam.s2LoopEndIdx = 0;
        runParam.isCrossCoreSplit = false;
        return true;
    }

    bool hasPrevCore = rangeStart > 0;
    bool hasNextCore = rangeEnd < totalLen;
    runParam.isCrossCoreSplit = hasPrevCore || hasNextCore;
    runParam.isFirstS2SplitCore = !hasPrevCore;

    int64_t oriRangeStart = rangeStart < oriLen ? rangeStart : oriLen;
    int64_t oriRangeEnd = rangeEnd < oriLen ? rangeEnd : oriLen;
    runParam.s2OriLineStartIdx = oriStart + oriRangeStart;
    runParam.s2OriLineEndIdx = oriStart + oriRangeEnd;

    int64_t cmpRangeStart = rangeStart > oriLen ? rangeStart - oriLen : 0;
    cmpRangeStart = cmpRangeStart < cmpLen ? cmpRangeStart : cmpLen;
    int64_t cmpRangeEnd = rangeEnd > oriLen ? rangeEnd - oriLen : 0;
    cmpRangeEnd = cmpRangeEnd < cmpLen ? cmpRangeEnd : cmpLen;
    runParam.s2CmpLineStartIdx = cmpStart + cmpRangeStart;
    runParam.s2CmpLineEndIdx = cmpStart + cmpRangeEnd;

    int64_t s2BaseSize = static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t oriRangeLen = runParam.s2OriLineEndIdx - runParam.s2OriLineStartIdx;
    int64_t cmpRangeLen = runParam.s2CmpLineEndIdx - runParam.s2CmpLineStartIdx;
    runParam.oriKvLoopEndIdx = (oriRangeLen + s2BaseSize - 1) / s2BaseSize;
    runParam.cmpKvLoopEndIdx = (cmpRangeLen + s2BaseSize - 1) / s2BaseSize;
    runParam.s2LoopEndIdx = runParam.oriKvLoopEndIdx + runParam.cmpKvLoopEndIdx;
    return runParam.s2LoopEndIdx == 0;
}
} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_KERNEL_ARCH35_H
