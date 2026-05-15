/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_scfa_kernel.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "kv_quant_sparse_attn_sharedkv_kvcache.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_cube.h"
#include "kv_quant_sparse_attn_sharedkv_scfa_block_vector.h"
#include "kernel_operator.h"
#include "../kv_quant_sparse_attn_sharedkv_metadata.h"

// 线上编包
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"

#include "kernel_operator_list_tensor_intf.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace optiling::detail;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename CubeBlockType, typename VecBlockType>
class KvQuantSparseAttnSharedkvScfa {
public:
    ARGS_TRAITS;
    __aicore__ inline KvQuantSparseAttnSharedkvScfa() {};

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                       __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                       __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
                                       __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                       const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ProcessMainLoop();
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
        __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
        const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam);
    __aicore__ inline void InitUniqueConstInfo();
    __aicore__ inline void ComputeAxisIdxByBnAndGs1(int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam);
    __aicore__ inline void InitUniqueRunInfo(const RunParamStr &runParam, RunInfo &runInfo);
    TPipe *pipe;

    const KvQuantSparseAttnSharedkvTilingData *__restrict tilingData;
    /* 编译期常量的基本块信息 */
    static constexpr TPosition bmm2OutPos = CubeBlockType::bmm2OutPos;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};
    static constexpr uint32_t PRELOAD_NUM = 2;
    /* 核间通道 */
    BufferManager<BufferType::GM> gmBufferManager;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1RightBuffers;
    CVSharedParams sharedParams;
    /* GM信息 */
    using keyGmType = typename std::conditional<TEMPLATE_MODE == SASTemplateMode::SWA_TEMPLATE_MODE, int8_t, GlobalTensor<KV_T>>::type;
    GlobalTensor<KV_T> oriKeyGm; // kv不连续场景需要使用来获取shape
    keyGmType cmpKeyGm; // kv不连续场景需要使用来获取shape
    GlobalTensor<uint32_t> metadataGm;
    __gm__ int32_t *cuSeqlensQAddr;
    __gm__ int32_t *actualSeqKvlenAddr;
    __gm__ int32_t *actualSeqQlenAddr;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 初始化后不变的信息 */
    ConstInfo constInfo;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
        constInfo.aivIdx = 0;
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

    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, cuSeqlensQ, sequsedKv);
    if ASCEND_IS_AIV {
        constInfo.bSize = this->sharedParams.bSize;
        constInfo.gSize = this->sharedParams.gSize;
        constInfo.s1Size = this->sharedParams.s1Size;
        constInfo.dSizeV = this->sharedParams.dSize;
        constInfo.needInit = this->sharedParams.needInit;
    }
    vecBlock.CleanOutput(attentionOut, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf();
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
        #pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ, sequsedQ, sequsedKv, sinks,
        workspace, tiling, tPipe); // gm设置
    this->InitLocalBuffer();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv, __gm__ uint8_t *sinks, __gm__ uint8_t *workspace,
    const KvQuantSparseAttnSharedkvTilingData *__restrict tiling, TPipe *tPipe)
{
    // 初始化vector用到的global buffer
    oriKeyGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE) {
        cmpKeyGm.SetGlobalBuffer((__gm__ KV_T *)(cmpKV));
    }
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQAddr = (__gm__ int32_t *)cuSeqlensQ;
    }
    if (sequsedKv != nullptr) {
        actualSeqKvlenAddr = (__gm__ int32_t *)sequsedKv;
    }

    if (sequsedQ != nullptr) {
        actualSeqQlenAddr = (__gm__ int32_t *)sequsedQ;
    } else {
        actualSeqQlenAddr = nullptr;
    }

    uint64_t singleCoreOffset = 0;
    vecBlock.InitGlobalBuffer(oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, sequsedQ, sinks);
    cubeBlock.InitCubeInput(cuSeqlensQ, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitMMResBuf()
{
    uint32_t mm1ResultSize = constInfo.s1BaseSize / CV_RATIO * constInfo.s2BaseSize * sizeof(T);
    uint32_t mm2ResultSize = constInfo.s1BaseSize / CV_RATIO * 512 * sizeof(T);
    uint32_t mm2LeftSize = constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(Q_T);
    uint32_t mm1RightSize = constInfo.s2BaseSize * 512 * sizeof(Q_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    l1RightBuffers.Init(l1BufferManager, mm1RightSize);
    if ASCEND_IS_AIC {
        l1PBuffers.Get().SetCrossCore();
        l1PBuffers.Get().SetCrossCore();
        l1RightBuffers.Get().SetCrossCore();
        l1RightBuffers.Get().SetCrossCore();
        l1RightBuffers.Get().SetCrossCore();
    }
    ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
    bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    if ASCEND_IS_AIV {
        bmm2Buffers.Get().SetCrossCore();
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }
}
 
template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    // 计算轴的乘积
    if ASCEND_IS_AIC {
        constInfo.bSize = this->sharedParams.bSize;
        constInfo.gSize = this->sharedParams.gSize;
        constInfo.s1Size = this->sharedParams.s1Size;
        constInfo.dSizeV = this->sharedParams.dSize;
        constInfo.needInit = this->sharedParams.needInit;
    }
    constInfo.n2Size = sharedParams.n2Size;
    constInfo.s2Size = sharedParams.s2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeVInput = sharedParams.dSizeVInput;
    constInfo.dSizeRope = sharedParams.dSizeRope;
    constInfo.dSizeNope = constInfo.dSize - constInfo.dSizeRope;
    constInfo.tileSize = sharedParams.tileSize;
    constInfo.sparseBlockCount = sharedParams.sparseBlockCount;
    constInfo.sparseBlockSize = 1;
    constInfo.cmpRatio = sharedParams.cmpRatio;
    constInfo.oriWinLeft = sharedParams.oriWinLeft;
    constInfo.oriWinRight = sharedParams.oriWinRight;
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
    constInfo.layoutType = sharedParams.layoutType;

    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;

        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (LAYOUT_T == SAS_LAYOUT::BSND) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = constInfo.s1BaseSize * constInfo.n2GDv;
        constInfo.mm1Ka = constInfo.n2Size * constInfo.dSize;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    }
    if ASCEND_IS_AIV {
        constInfo.softmaxScale = sharedParams.softmaxScale;
        constInfo.oriBlockSize = sharedParams.oriBlockSize;
        constInfo.cmpBlockSize = sharedParams.cmpBlockSize;
        constInfo.oriMaxBlockNumPerBatch = sharedParams.oriMaxBlockNumPerBatch;
        constInfo.cmpMaxBlockNumPerBatch = sharedParams.cmpMaxBlockNumPerBatch;
    }

    InitUniqueConstInfo();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitUniqueConstInfo()
{
    this->constInfo.actualSeqLenSize = this->sharedParams.bSize + 1;
    this->constInfo.actualSeqLenKVSize = this->sharedParams.bSize;
    this->constInfo.isActualLenDimsKVNull = static_cast<bool>(this->sharedParams.isActualSeqLengthsKVNull);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::Process()
{
    // SyncAll Cube和Vector都需要调用
    if (this->sharedParams.needInit) {
        SyncAll<false>();
    }
    ProcessMainLoop();
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ProcessMainLoop()
{
    uint32_t hasLoad = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_CORE_ENABLE_INDEX, false, 0));
    if (hasLoad == 0) {
        return;
    }

    // 从meta data解析分核信息
    uint32_t bN2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_START_INDEX, false, 0));
    uint32_t gS1StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_START_INDEX, false, 0));
    uint32_t s2StartIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_START_INDEX, false, 0));
    uint32_t bN2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_BN2_END_INDEX, false, 0));
    uint32_t nextGs1Idx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_M_END_INDEX, false, 0));
    uint32_t s2EndIdx = metadataGm.GetValue(GetAttrAbsIndex(aicIdx, FA_S2_END_INDEX, false, 0));
    uint32_t s2LoopLimit = 0;

    if (nextGs1Idx != 0) {
        bN2EndIdx++;
    }

    int64_t taskId = 0;
    bool notLast = true;
    RunInfo runInfo[3];
    RunParamStr runParam;
    int64_t multiCoreInnerIdx = 1;
    for (int64_t bnIdx = bN2StartIdx; bnIdx < bN2EndIdx; bnIdx++) {
        bool lastBN = (bnIdx == bN2EndIdx - 1);
        runParam.boIdx = bnIdx;
        runParam.n2oIdx = 0;
        ComputeParamBatch<TEMPLATE_INTF_ARGS>(runParam, this->constInfo,
            this->cuSeqlensQAddr, this->actualSeqQlenAddr, this->actualSeqKvlenAddr);
        ComputeS1LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo, lastBN, nextGs1Idx, gS1StartIdx);

        int64_t gS1LoopEnd = lastBN ? (runParam.gs1LoopEndIdx + PRELOAD_NUM) : runParam.gs1LoopEndIdx;
        for (int64_t gS1Index = runParam.gs1LoopStartIdx; gS1Index < gS1LoopEnd; gS1Index++) {
            bool notLastTwoLoop = true;
            if (lastBN) {
                int32_t extraGS1 = gS1Index - runParam.gs1LoopEndIdx;
                switch (extraGS1) {
                    case 0:
                        notLastTwoLoop = false;
                        break;
                    case 1:
                        notLast = false;
                        notLastTwoLoop = false;
                        break;
                    default:
                        break;
                }
            }
            if (notLastTwoLoop) {
                this->ComputeAxisIdxByBnAndGs1(bnIdx, gS1Index, runParam);
                bool s1NoNeedCalc = ComputeParamS1<TEMPLATE_INTF_ARGS>(
                    runParam, this->constInfo, gS1Index, this->cuSeqlensQAddr);
                bool s2NoNeedCalc =
                    ComputeS2LoopInfo<TEMPLATE_INTF_ARGS>(runParam, this->constInfo);
                // s1和s2有任意一个不需要算, 则continue, 如果是当前核最后一次循环，则补充计算taskIdx+2的部分
                if (s1NoNeedCalc || s2NoNeedCalc) {
                    continue;
                }
                s2LoopLimit = runParam.s2LoopEndIdx - 1;
            } else {
                s2LoopLimit = 0;
            }
            for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; ++s2LoopCount) {
                if (notLastTwoLoop) {
                    RunInfo &runInfo1 = runInfo[taskId % 3];
                    this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);
                    if ASCEND_IS_AIC {
                        this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), this->l1RightBuffers.Get(), runInfo1,
                            this->constInfo);
                    } else {
                        this->vecBlock.ProcessVec0(this->l1RightBuffers.Get(), runInfo1, this->constInfo);
                    }
                }
                if (taskId > 0 && notLast) {
                    auto &runInfo2 = runInfo[(taskId + 2) % 3];
                    if ASCEND_IS_AIV {
                        this->vecBlock.ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo2,
                            this->constInfo);
                    } else {
                        RunInfo &runInfo2 = runInfo[(taskId + 2) % 3];
                        this->cubeBlock.IterateBmm2(this->bmm2Buffers.Get(), this->l1PBuffers, this->l1RightBuffers.GetReused(), runInfo2,
                            this->constInfo);
                    }
                }
                if (taskId > 1) {
                    if ASCEND_IS_AIV {
                        RunInfo &runInfo3 = runInfo[(taskId + 1) % 3];
                        this->vecBlock.ProcessVec2(this->bmm2Buffers.Get(), runInfo3, this->constInfo);
                    }
                }
                ++taskId;
            }
            ++multiCoreInnerIdx;
        }
        gS1StartIdx = 0;
    }
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ComputeAxisIdxByBnAndGs1(
    int64_t bnIndex, int64_t gS1Index, RunParamStr &runParam)
{
    // GS1合轴, 不切G, 只切S1
    runParam.s1oIdx = gS1Index * runParam.qSNumInOneBlock;
    runParam.goIdx = 0;
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    if (s2LoopCount < runParam.oriKvLoopEndIdx) {
        runInfo.s2StartIdx = runParam.s2LineStartIdx;
        runInfo.s2EndIdx = runParam.s2LineEndIdx;
    } else {
        runInfo.s2StartIdx = 0;
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
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.sOuterOffset = runParam.sOuterOffset;
    this->ComputeBmm1Tail(runInfo, runParam);
    InitUniqueRunInfo(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void
KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::InitUniqueRunInfo(
    const RunParamStr &runParam, RunInfo &runInfo)
{
    InitTaskParamByRun<TEMPLATE_INTF_ARGS>(runParam, runInfo);
}

template <typename CubeBlockType, typename VecBlockType>
__aicore__ inline void KvQuantSparseAttnSharedkvScfa<CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo &runInfo, RunParamStr &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;
    runInfo.mRealSize = runParam.mRealSize;
    runInfo.halfMRealSize = runParam.halfMRealSize;
    runInfo.firstHalfMRealSize = runParam.firstHalfMRealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;  // D>128 这里需要适配
    runInfo.vec2MBaseSize = runInfo.halfMRealSize;

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = constInfo.s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    int64_t curS2LoopCnt = (runInfo.s2LoopCount >= runParam.oriKvLoopEndIdx) ? (runInfo.s2LoopCount - runParam.oriKvLoopEndIdx) : runInfo.s2LoopCount;
    if (runInfo.s2StartIdx + (curS2LoopCnt + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
        runInfo.s2RealSize = runInfo.s2EndIdx - curS2LoopCnt * runInfo.s2RealSize - runInfo.s2StartIdx;
        runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
    }
}
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_KERNEL_H