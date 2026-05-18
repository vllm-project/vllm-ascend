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
 * \file grrouped_matmul_swiglu_quant_spilit_fusion.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_SPLIT_FUSION_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_SPLIT_FUSION_H

#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "grouped_matmul_swiglu_quant_v2_utils.h"

namespace GroupedMatmulDequantSwigluQuant {
using namespace AscendC;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr int64_t SWI_FACTOR = 2;
constexpr float DYNAMIC_QUANT_FACTOR = 1.0 / static_cast<float>(127.0);
constexpr uint64_t MAX_CALC_NUM = 64;
constexpr uint64_t REDUCEMAX_CALC_NUM = 64;
constexpr uint64_t SPILI_NUM = 2;
constexpr uint64_t VC_SYNC_MAX_TIMES = 14;
constexpr uint64_t RESRERVE_MEM_SIZE = 192;

class GroupedMatmulDequantSwigluQuantFusion {
public:
    using aType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
    using bType = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using matmulType = MMImplType<aType, bType, cType, biasType, matmulCFGUnitFlag>;
    matmulType::MT mm;

    __aicore__ inline GroupedMatmulDequantSwigluQuantFusion(
        TPipe* pipe, const GMMSwigluQuantV2TilingFusionData* __restrict tiling,
        const TCubeTiling* __restrict matmulTilingData)
        : pipe_(pipe), tilingData_(tiling), matmulTilingData_(matmulTilingData) {
        }

    __aicore__ inline int CeilDiv(int a, int b) {
        return (a + b  - 1) / b;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weight_scale, GM_ADDR activation_scale,
        GM_ADDR weightAssistanceMatrix, GM_ADDR group_list,
        GM_ADDR y, GM_ADDR scale, GM_ADDR workspace) {
        xGm_.SetGlobalBuffer((__gm__ int8_t*)x);
        groupListGm_.SetGlobalBuffer((__gm__ int64_t*)group_list);
        weightGm_.SetGlobalBuffer(GetTensorAddr<int8_t>(0, weight));
        weightScaleGm_.SetGlobalBuffer(GetTensorAddr<float>(0, weight_scale));
        workspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace);
        activateScaleGm_.SetGlobalBuffer((__gm__ float*)activation_scale);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
        yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
        weightScaleTensorPtr_ = weight_scale;
        weightTensorPtr_ = weight;

        nBasicsBlocks = CeilDiv(tilingData_->N, matmulTilingData_->baseN);
        totalBasicBlocks = 0;
        for (int groupId = 0; groupId < tilingData_->groupNum; groupId++) {
            int tokens = groupListGm_.GetValue(groupId);
            if (tilingData_->groupListType == 0 && groupId > 0) {
                tokens = groupListGm_.GetValue(groupId) - groupListGm_.GetValue(groupId - 1);
            }
            int mBasicBlocks = CeilDiv(tokens, matmulTilingData_->baseM);
            totalBasicBlocks += mBasicBlocks * nBasicsBlocks;
        }

        totalSyncTimes = CeilDiv(totalBasicBlocks, tilingData_->cubeBlockDim);
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(xActQueue_, 1, (tilingData_->ubFactorDimx * (tilingData_->N / SPILI_NUM) * SWI_FACTOR + tilingData_->ubFactorDimx * BLOCK_ELEM) * sizeof(int32_t));
            pipe_->InitBuffer(inScaleQueue_, 1, ((tilingData_->N / SPILI_NUM) * SWI_FACTOR + (tilingData_->N / SPILI_NUM)) * sizeof(float));
            pipe_->InitBuffer(outQueue_, 1, tilingData_->ubFactorDimx * (tilingData_->N / SPILI_NUM) * sizeof(int8_t) + tilingData_->ubFactorDimx * sizeof(float) + RESRERVE_MEM_SIZE);
            pipe_->InitBuffer(tmpBuf1_, tilingData_->ubFactorDimx * (tilingData_->N / SPILI_NUM) * SWI_FACTOR * sizeof(float));
        }
    }

    __aicore__ inline void FindCurrentGroup(uint32_t basicBlockIdxInGlobal, uint32_t& currentGroupId,
        uint32_t& globalMOffset, uint32_t& processedBasicBlock) {
        for (int groupId = currentGroupId; groupId < tilingData_->groupNum; groupId++) {
            int tokens = groupListGm_.GetValue(groupId);
            if (tilingData_->groupListType == 0 && groupId > 0) {
                tokens = groupListGm_.GetValue(groupId) - groupListGm_.GetValue(groupId - 1);
            }
            int mBasicBlocks = CeilDiv(tokens, matmulTilingData_->baseM);
            if (processedBasicBlock + mBasicBlocks * nBasicsBlocks > basicBlockIdxInGlobal) {
                currentGroupId = groupId;
                break;
            } else {
                globalMOffset += tokens;
                processedBasicBlock += mBasicBlocks * nBasicsBlocks;
            }
        }
    }

    __aicore__ inline void CalculateBlockSizes(int tokens, int currentBasicBlockMId, int currentBasicBlockNId,
        int& realMSize, int& realNSize) {
        realMSize = matmulTilingData_->baseM;
        if (currentBasicBlockMId * matmulTilingData_->baseM + realMSize > tokens) {
            realMSize = tokens - currentBasicBlockMId * matmulTilingData_->baseM;
        }
        realNSize = matmulTilingData_->baseN;
        if (currentBasicBlockNId * matmulTilingData_->baseN + realNSize > tilingData_->N) {
            realNSize = tilingData_->N - currentBasicBlockNId * matmulTilingData_->baseN;
        }
    }

    __aicore__ inline void SetupMatmulShape(int tokens, int realMSize, int realNSize) {
        mm.SetOrgShape(tokens, tilingData_->N, tilingData_->K);
        mm.SetSingleShape(realMSize, realNSize, tilingData_->K);
    }

    __aicore__ inline void SetupMatmulWeight(int currentGroupId, int currentBasicBlockNId) {
        if (tilingData_->isSingleTensor == 0) {
            weightGm_.SetGlobalBuffer(GetTensorAddr<int8_t>(currentGroupId, weightTensorPtr_));
            mm.SetTensorB(weightGm_[0x8 * currentBasicBlockNId * tilingData_->K * 0x20]);
        } else {
            int64_t tensorBOffset = currentGroupId * tilingData_->K * tilingData_->N + 0x8 * currentBasicBlockNId * tilingData_->K * 0x20;
            mm.SetTensorB(weightGm_[tensorBOffset]);
        }
    }

    __aicore__ inline void ProcessCubeBlock(uint32_t basicBlockIdxInGlobal, uint32_t& currentGroupId,
        uint32_t& globalMOffset, uint32_t& processedBasicBlock) {
        FindCurrentGroup(basicBlockIdxInGlobal, currentGroupId, globalMOffset, processedBasicBlock);
        int tokens = groupListGm_.GetValue(currentGroupId);
        if (tilingData_->groupListType == 0 && currentGroupId > 0) {
            tokens = groupListGm_.GetValue(currentGroupId) - groupListGm_.GetValue(currentGroupId - 1);
        }
        int basicBlockIdxInCurrentGroup = basicBlockIdxInGlobal - processedBasicBlock;
        int mBasicBlocks = CeilDiv(tokens, matmulTilingData_->baseM);
        int currentBasicBlockMId = basicBlockIdxInCurrentGroup / nBasicsBlocks;
        int currentBasicBlockNId = basicBlockIdxInCurrentGroup % nBasicsBlocks;
        int realMSize = 0;
        int realNSize = 0;
        CalculateBlockSizes(tokens, currentBasicBlockMId, currentBasicBlockNId, realMSize, realNSize);
        SetupMatmulShape(tokens, realMSize, realNSize);
        int64_t tensorAOffset = currentBasicBlockMId * matmulTilingData_->baseM * tilingData_->K + globalMOffset * tilingData_->K;
        mm.SetTensorA(xGm_[tensorAOffset]);
        SetupMatmulWeight(currentGroupId, currentBasicBlockNId);
        int64_t workspaceOffset = globalMOffset * tilingData_->N + currentBasicBlockMId * matmulTilingData_->baseM * tilingData_->N
            + currentBasicBlockNId * matmulTilingData_->baseN;
        mm.template IterateAll<false>(workspaceGm_[workspaceOffset]);
    }

    __aicore__ inline void FinalizeCubeSync(uint32_t& syncId) {
        while (syncId < totalSyncTimes) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
            syncId += 1;
        }
    }

    __aicore__ inline void CubeProcess() {
        if ASCEND_IS_AIC {
            uint32_t currentBlockId = GetBlockIdx();
            uint32_t rsvBlockNum = 0;
            uint32_t calcBlockNum = 0;
            uint32_t cvTimes = 0;
            uint32_t syncId = 0;
            uint32_t globalMOffset = 0;
            uint32_t processedBasicBlock = 0;
            uint32_t currentGroupId = 0;
            uint32_t realSyncId = 0;
            while (currentBlockId < totalBasicBlocks) {
                cvTimes = CeilDiv(nBasicsBlocks - rsvBlockNum, tilingData_->cubeBlockDim);
                calcBlockNum += cvTimes * tilingData_->cubeBlockDim;
                rsvBlockNum = calcBlockNum % nBasicsBlocks;

                for (uint32_t cvId = 0; cvId < cvTimes; cvId++) {
                    uint32_t basicBlockIdxInGlobal = currentBlockId;
                    if (basicBlockIdxInGlobal >= totalBasicBlocks) {
                        break;
                    }
                    ProcessCubeBlock(basicBlockIdxInGlobal, currentGroupId, globalMOffset, processedBasicBlock);
                    currentBlockId += tilingData_->cubeBlockDim;
                    syncId += 1;
                }
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);

                realSyncId += 1;
                if (realSyncId > 0 && realSyncId % VC_SYNC_MAX_TIMES == 0) {
                    AscendC::CrossCoreWaitFlag(0x9);
                }
            }
            FinalizeCubeSync(syncId);
        }
    }

    __aicore__ inline void CalculateEndGroupInfo(int endBasicBlockId, int endGroupId, int& endGroupMOffset,
        int& basicBlockCountBeforeEndGroup) {
        endGroupMOffset = 0;
        basicBlockCountBeforeEndGroup = 0;
        for (int gId = 0; gId < endGroupId; gId++) {
            int tokens = groupListGm_.GetValue(gId);
            if (tilingData_->groupListType == 0 && gId > 0) {
                tokens = groupListGm_.GetValue(gId) - groupListGm_.GetValue(gId - 1);
            }
            int mBasicBlocks = CeilDiv(tokens, matmulTilingData_->baseM);
            basicBlockCountBeforeEndGroup += mBasicBlocks * nBasicsBlocks;
            endGroupMOffset += tokens;
        }
        int basicBlockIdxInCurrentGroup = endBasicBlockId - basicBlockCountBeforeEndGroup;
        int currentBasicBlockMId = basicBlockIdxInCurrentGroup / nBasicsBlocks;
        endGroupMOffset += currentBasicBlockMId * matmulTilingData_->baseM;
    }

    __aicore__ inline void ProcessGroupRange(int startGroupId, int endGroupId, int endGroupMOffset,
        uint32_t& globalMOffset, bool &isSyncAll) {
        int currentGroupMOffset = 0;
        for (int gId = 0; gId < startGroupId; gId++) {
            currentGroupMOffset += groupListGm_.GetValue(gId);
        }
        for (int groupId = startGroupId; groupId <= endGroupId; groupId++) {
            if (tilingData_->groupListType == 0 && groupId > 0) {
                currentGroupMOffset = groupListGm_.GetValue(groupId);
            } else {
                currentGroupMOffset += groupListGm_.GetValue(groupId);
            }
            int calcCount = 0;
            if (currentGroupMOffset <= endGroupMOffset) {
                calcCount = currentGroupMOffset - globalMOffset;
            } else {
                calcCount = endGroupMOffset - globalMOffset;
            }
            ProcessDSQ(groupId, globalMOffset, calcCount, isSyncAll);
            globalMOffset += calcCount;
        }
    }

    __aicore__ inline void ProcessVectorBlock(uint32_t syncId, bool& isSyncAll, uint32_t& globalMOffset) {
        int startBasicBlockId = syncId * tilingData_->cubeBlockDim;
        int endBasicBlockId = startBasicBlockId + tilingData_->cubeBlockDim;
        if (totalBasicBlocks < endBasicBlockId) {
            endBasicBlockId = totalBasicBlocks;
        }
        int startGroupId = GetGroupId(startBasicBlockId);
        int endGroupId = GetGroupId(endBasicBlockId);
        int endGroupMOffset = 0;
        int basicBlockCountBeforeEndGroup = 0;
        CalculateEndGroupInfo(endBasicBlockId, endGroupId, endGroupMOffset, basicBlockCountBeforeEndGroup);
        ProcessGroupRange(startGroupId, endGroupId, endGroupMOffset, globalMOffset, isSyncAll);
    }

    __aicore__ inline void VectorProcess() {
        if ASCEND_IS_AIV {
            weightCacheGroupId_ = -1;
            uint32_t currentBlockId = GetBlockIdx() / 2;
            uint32_t rsvBlockNum = 0;
            uint32_t calcBlockNum = 0;
            uint32_t cvTimes = 0;
            uint32_t syncId = 0;
            uint32_t globalMOffset = 0;
            uint32_t processedBasicBlock = 0;
            uint32_t currentGroupId = 0;
            uint32_t realSyncId = 0;
            bool isSyncAll = false;
            while (syncId < totalSyncTimes) {
                cvTimes = CeilDiv(nBasicsBlocks - rsvBlockNum, tilingData_->cubeBlockDim);
                calcBlockNum += cvTimes * tilingData_->cubeBlockDim;
                rsvBlockNum = calcBlockNum % nBasicsBlocks;
                isSyncAll = true;

                for (uint32_t cvId = 0; cvId < cvTimes; cvId++) {
                    ProcessVectorBlock(syncId, isSyncAll, globalMOffset);
                    currentBlockId += tilingData_->cubeBlockDim;
                    syncId += 1;
                }

                realSyncId += 1;
                if (realSyncId > 0 && (realSyncId % VC_SYNC_MAX_TIMES == 0)) {
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(0x9);
                }
            }
        }
    }

    __aicore__ inline void Process() {
        CubeProcess();
        VectorProcess();
    }

    __aicore__ inline int GetGroupId(int basicBlockId) {
        int processedBasicBlock = 0;
        int currentGroupId = 0;
        int globalMOffset = 0;
        for (int groupId = 0; groupId < tilingData_->groupNum; groupId++) {
            int tokens = groupListGm_.GetValue(groupId);
            if (tilingData_->groupListType == 0 && groupId > 0) {
                tokens = groupListGm_.GetValue(groupId) - groupListGm_.GetValue(groupId - 1);
            }
            int mBasicBlocks = CeilDiv(tokens, matmulTilingData_->baseM);
            if (processedBasicBlock + mBasicBlocks * nBasicsBlocks >= basicBlockId) {
                return groupId;
            } else {
                processedBasicBlock += mBasicBlocks * nBasicsBlocks;
            }
        }
        return tilingData_->groupNum - 1;
    }

    __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& tempRes, int32_t calcCount) {
        uint32_t vectorCycles = calcCount / MAX_CALC_NUM;
        uint32_t remainElements = calcCount % MAX_CALC_NUM;

        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 0;
        repeatParams.src0RepStride = 0x8;
        repeatParams.src1RepStride = 0;

        if (vectorCycles > 0 && remainElements > 0) {
            Max(tempRes, tempRes, tempRes[vectorCycles * MAX_CALC_NUM], remainElements, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }

        if (vectorCycles > 1) {
            Max(tempRes,  tempRes[MAX_CALC_NUM], tempRes, MAX_CALC_NUM, vectorCycles - 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessDSQ(int groupId, int globalOffset, int calcCount, bool &isSyncAll) {
        int32_t blockDimxFactor = (calcCount + tilingData_->vectorBlockDim - 1) / tilingData_->vectorBlockDim;
        int32_t realCoreDim = calcCount == 0 ? 0 : (calcCount + blockDimxFactor - 1) / blockDimxFactor;

        if (GetBlockIdx() >= realCoreDim) {
            if (isSyncAll) {
                AscendC::CrossCoreWaitFlag(0x8);
                SyncAll<true>();
                isSyncAll = false;
            }
            return;
        }

        DataCopyPadParams padParams{false, 0, 0, 0};
        LocalTensor<float> inScaleLocal = inScaleQueue_.AllocTensor<float>();

        if (weightCacheGroupId_ != groupId) {
            DataCopyParams dataCopyWeightScaleParams;
            dataCopyWeightScaleParams.blockCount = 1;
            dataCopyWeightScaleParams.blockLen = tilingData_->N * sizeof(float);
            dataCopyWeightScaleParams.srcStride = 0;
            dataCopyWeightScaleParams.dstStride = 0;
            if (tilingData_->isSingleTensor == 0) {
                weightScaleGm_.SetGlobalBuffer(GetTensorAddr<float>(groupId, weightScaleTensorPtr_));
                DataCopyPad(inScaleLocal, weightScaleGm_, dataCopyWeightScaleParams, padParams);
            } else {
                DataCopyPad(inScaleLocal, weightScaleGm_[groupId * tilingData_->N], dataCopyWeightScaleParams, padParams);
            }
            DataCopyParams dataCopyQuantScaleParams;
            dataCopyQuantScaleParams.blockCount = 1;
            dataCopyQuantScaleParams.blockLen = (tilingData_->N / SPILI_NUM) * sizeof(float);
            dataCopyQuantScaleParams.srcStride = 0;
            dataCopyQuantScaleParams.dstStride = 0;
            weightCacheGroupId_ = groupId;
        }

        inScaleQueue_.EnQue(inScaleLocal);
        inScaleLocal = inScaleQueue_.DeQue<float>();

        int32_t blockDimxTailFactor = calcCount - blockDimxFactor * (realCoreDim - 1);
        int32_t DimxCore = GetBlockIdx() == (realCoreDim - 1) ? blockDimxTailFactor : blockDimxFactor;

        int32_t ubDimxLoop = (DimxCore + tilingData_->ubFactorDimx - 1) / tilingData_->ubFactorDimx;
        int32_t ubDimxTailFactor = DimxCore - tilingData_->ubFactorDimx * (ubDimxLoop - 1);

        int64_t coreDimxOffset = blockDimxFactor * GetBlockIdx();
        int32_t actOffset = tilingData_->actRight * tilingData_->ubFactorDimy;
        int32_t gateOffset = tilingData_->ubFactorDimy - actOffset;

        LocalTensor<float> weightScaleLocal = inScaleLocal;
        LocalTensor<float> quantScaleLocal = inScaleLocal[tilingData_->N];

        for (uint32_t loopIdx = 0; loopIdx < ubDimxLoop; loopIdx++) {
            int64_t xDimxOffset = (coreDimxOffset + loopIdx * tilingData_->ubFactorDimx) + globalOffset;
            int32_t proDimsx = loopIdx == (ubDimxLoop - 1) ? ubDimxTailFactor : tilingData_->ubFactorDimx;
            LocalTensor<float> tmpUbF32 = tmpBuf1_.AllocTensor<float>();
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tilingData_->ubFactorDimy * SWI_FACTOR);
            Copy<float, false>(tmpUbF32, weightScaleLocal, MASK_PLACEHOLDER, proDimsx,
                {1, 1, static_cast<uint16_t>((tilingData_->ubFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 0});
            SetMaskNorm();
            ResetMask();

            LocalTensor<int32_t> xActLocal = xActQueue_.AllocTensor<int32_t>();
            DataCopyParams dataCopyActScaleParams;
            dataCopyActScaleParams.blockCount = proDimsx;
            dataCopyActScaleParams.blockLen = sizeof(float);
            dataCopyActScaleParams.srcStride = 0;
            dataCopyActScaleParams.dstStride = 0;
            LocalTensor<float> xActLocalF32 = xActLocal.template ReinterpretCast<float>();
            DataCopyPad(xActLocalF32[tilingData_->ubFactorDimx * tilingData_->N], activateScaleGm_[xDimxOffset],
                dataCopyActScaleParams, padParams);

            if (isSyncAll) {
                AscendC::CrossCoreWaitFlag(0x8);
                SyncAll<true>();
                isSyncAll = false;
            }

            DataCopyParams dataCopyXParams;
            dataCopyXParams.blockCount = proDimsx;
            dataCopyXParams.blockLen = tilingData_->N * sizeof(int32_t);
            dataCopyXParams.srcStride = 0;
            dataCopyXParams.dstStride = 0;
            DataCopyPad(xActLocal, workspaceGm_[xDimxOffset * tilingData_->N], dataCopyXParams, padParams);
            xActQueue_.EnQue(xActLocal);
            xActLocal = xActQueue_.DeQue<int32_t>();

            LocalTensor<int32_t> xLocal = xActLocal;
            xActLocalF32 = xActLocal.template ReinterpretCast<float>();
            LocalTensor<float> xLocalF32 = xActLocalF32;
            LocalTensor<float> activationScaleLocal = xActLocalF32[tilingData_->ubFactorDimx * tilingData_->N];

            Cast(xLocalF32, xLocal, RoundMode::CAST_NONE, SWI_FACTOR * proDimsx * tilingData_->ubFactorDimy);
            PipeBarrier<PIPE_V>();

            Mul(xLocalF32, tmpUbF32, xLocalF32, tilingData_->ubFactorDimy * SWI_FACTOR * proDimsx);
            PipeBarrier<PIPE_V>();

            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tilingData_->ubFactorDimy * SWI_FACTOR);
            Copy<float, false>(tmpUbF32, activationScaleLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
                {1, 0, static_cast<uint16_t>((tilingData_->ubFactorDimy * SWI_FACTOR) / BLOCK_ELEM), 1});
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();

            Mul(xLocalF32, tmpUbF32, xLocalF32, tilingData_->ubFactorDimy * SWI_FACTOR * proDimsx);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> tmpUbF32Act = tmpUbF32;
            LocalTensor<float> tmpUbF32Gate = tmpUbF32[tilingData_->ubFactorDimy * proDimsx];
            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tilingData_->ubFactorDimy);
            Copy<float, false>(tmpUbF32Act, xLocalF32[actOffset], AscendC::MASK_PLACEHOLDER, proDimsx,
                {1, 1, static_cast<uint16_t>(tilingData_->ubFactorDimy / BLOCK_ELEM),
                static_cast<uint16_t>(tilingData_->ubFactorDimy / BLOCK_ELEM * SWI_FACTOR)});
            Copy<float, false>(tmpUbF32Gate, xLocalF32[gateOffset], AscendC::MASK_PLACEHOLDER, proDimsx,
                {1, 1, static_cast<uint16_t>(tilingData_->ubFactorDimy / BLOCK_ELEM),
                static_cast<uint16_t>(tilingData_->ubFactorDimy / BLOCK_ELEM * SWI_FACTOR)});
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();
            limited=tilingData_->swigluLimit;

            Mins(tmpUbF32Gate, tmpUbF32Gate, limited,tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
            Maxs(tmpUbF32Gate, tmpUbF32Gate, (-1.0f * limited), tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
            Mins(tmpUbF32Act, tmpUbF32Act, limited, tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();

            Muls(xLocalF32, tmpUbF32Act, static_cast<float>(-1.0), tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
            Exp(xLocalF32, xLocalF32, tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
            Adds(xLocalF32, xLocalF32, static_cast<float>(1.0), tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();
            Div(tmpUbF32Act, tmpUbF32Act, xLocalF32, tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();

            xActQueue_.FreeTensor(xActLocal);
            Mul(tmpUbF32Act, tmpUbF32Gate, tmpUbF32Act, tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();

            Abs(tmpUbF32Gate, tmpUbF32Act, tilingData_->ubFactorDimy * proDimsx);

            LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

            uint64_t scaleOutOffset = tilingData_->ubFactorDimx * (tilingData_->N / SPILI_NUM) * sizeof(int8_t) / sizeof(float);
            uint64_t alignScaleOutOffset = Ceil(scaleOutOffset, uint32_t(8)) * 8; // 8: num int32_t in 32B ub block
            LocalTensor<float> scaleOut = outLocal[alignScaleOutOffset];
            LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();
            PipeBarrier<PIPE_V>();

            for (uint32_t i = 0; i < proDimsx; i++) {
                ComputeReduceMax(tmpUbF32Gate[i * tilingData_->ubFactorDimy], tilingData_->ubFactorDimy);
            }

            uint64_t realReduceMaxCalcNum = REDUCEMAX_CALC_NUM;
            if (tilingData_->ubFactorDimy < REDUCEMAX_CALC_NUM) {
                realReduceMaxCalcNum = tilingData_->ubFactorDimy;
            }

            WholeReduceMax(tmpUbF32Gate, tmpUbF32Gate, realReduceMaxCalcNum,  proDimsx, 1, 1,
                tilingData_->ubFactorDimy / BLOCK_ELEM, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();

            Muls(scaleOut, tmpUbF32Gate, DYNAMIC_QUANT_FACTOR, proDimsx);
            PipeBarrier<PIPE_V>();

            int64_t blockCount = (proDimsx + BLOCK_ELEM - 1) / BLOCK_ELEM;
            Brcb(outLocal, scaleOut, blockCount, {1, 8});
            PipeBarrier<PIPE_V>();

            SetMaskCount();
            SetVectorMask<float, MaskMode::COUNTER>(tilingData_->ubFactorDimy);
            Copy<float, false>(tmpUbF32Gate, outLocal, AscendC::MASK_PLACEHOLDER, proDimsx,
                {1, 0, static_cast<uint16_t>(tilingData_->ubFactorDimy / BLOCK_ELEM), 1});
            SetMaskNorm();
            ResetMask();
            PipeBarrier<PIPE_V>();

            Div(tmpUbF32Act, tmpUbF32Act, tmpUbF32Gate, tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();

            LocalTensor<int32_t> tmpUbF32ActI32 = tmpUbF32Act.ReinterpretCast<int32_t>();
            Cast(tmpUbF32ActI32, tmpUbF32Act, RoundMode::CAST_RINT, tilingData_->ubFactorDimy * proDimsx);
            SetDeqScale((half)1.000000e+00f);

            LocalTensor<half> tmpUbF32Gate16 = tmpUbF32Gate.template ReinterpretCast<half>();
            Cast(tmpUbF32Gate16, tmpUbF32ActI32, RoundMode::CAST_ROUND, tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();

            Cast(yOut, tmpUbF32Gate16, RoundMode::CAST_TRUNC, tilingData_->ubFactorDimy * proDimsx);
            PipeBarrier<PIPE_V>();

            tmpBuf1_.FreeTensor(tmpUbF32);
            outQueue_.EnQue<float>(outLocal);
            outLocal = outQueue_.DeQue<float>();
            scaleOut = outLocal[alignScaleOutOffset];
            yOut = outLocal.template ReinterpretCast<int8_t>();

            DataCopyParams dataCopyOutScaleParams;
            dataCopyOutScaleParams.blockCount = 1;
            dataCopyOutScaleParams.blockLen = proDimsx * sizeof(float);
            dataCopyOutScaleParams.srcStride = 0;
            dataCopyOutScaleParams.dstStride = 0;
            DataCopyPad(scaleGm_[xDimxOffset], scaleOut, dataCopyOutScaleParams);

            DataCopyParams dataCopyOutyParams;
            dataCopyOutyParams.blockCount = 1;
            dataCopyOutyParams.blockLen = proDimsx * (tilingData_->N / SPILI_NUM) * sizeof(int8_t);
            dataCopyOutyParams.srcStride = 0;
            dataCopyOutyParams.dstStride = 0;
            DataCopyPad(yGm_[xDimxOffset * (tilingData_->N / SPILI_NUM)], yOut, dataCopyOutyParams);
            outQueue_.FreeTensor(outLocal);
        }
        inScaleQueue_.FreeTensor(inScaleLocal);
    }

private:
    TPipe *pipe_ = nullptr;
    const GMMSwigluQuantV2TilingFusionData* __restrict tilingData_;
    const TCubeTiling* __restrict matmulTilingData_;
    static constexpr float FLOAT_INF = 3e+99;
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<int8_t> weightGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<int32_t> workspaceGm_;
    GlobalTensor<float> weightScaleGm_;
    GlobalTensor<float> activateScaleGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int64_t> groupListGm_;
    int nBasicsBlocks = 0;
    int totalBasicBlocks = 0;
    int totalSyncTimes = 0;
    int32_t weightCacheGroupId_ = -1;
    float limited = FLOAT_INF;

    TQue<TPosition::VECIN, 1> inQue_;
    TQue<TPosition::VECIN, 1> xQue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TQue<TPosition::VECOUT, 1> scaleOutQue_;
    TQue<TPosition::VECOUT, 1> yOutQue_;
    TQue<QuePosition::VECIN, 1> xActQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECIN, 1> inScaleQueue_;
    TBuf<TPosition::VECCALC> tmpBuf1_;

    GM_ADDR weightTensorPtr_;
    GM_ADDR weightScaleTensorPtr_;
};
}

#endif