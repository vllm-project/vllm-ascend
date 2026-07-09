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
 * \file scatter_nd_update_common.h
 * \brief scatter_nd_update
 */
#ifndef ASCENDC_SCATTER_ND_UPDATE_COMMON_H_
#define ASCENDC_SCATTER_ND_UPDATE_COMMON_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../inc/load_store_utils.h"
#include "scatter_nd_update_struct.h"
#include "indices_sort_utils.h"
#include "op_kernel/math_util.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
namespace ScatterNdUpdate {
using namespace AscendC;

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint64_t SORT_PAD_NUM = 2;
constexpr uint16_t MAX_RANK_COUNT = 7;
constexpr uint16_t MIN_SAME_IDX_ACCM_COUNT = 256;
constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
constexpr uint16_t MAX_SHAPE_RANK = 8;
constexpr uint16_t INDICE_RANK_TWO = 2;
constexpr float SORT_HIST_THRESHOLD = 0.01f;
constexpr uint32_t HASH_SCORE_BUF_SIZE = 128;
constexpr uint32_t MASK_DEFAULT = 0;
constexpr uint64_t LEAST_DEAL_SIZE = 256;

constexpr MicroAPI::CastTrait castTraitB322B64 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_DETERMINISTIC = 256;
#else
constexpr uint32_t THREAD_NUM_DETERMINISTIC = 1024;
#endif

static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

template <typename OFFSET_T>
__simd_vf__ inline void ComputeUniqueIdNumVf(__ubuf__ OFFSET_T* indicesAddr, __ubuf__ int32_t* uniqueIdCountsAddr,
                                             int64_t vfLen, uint16_t loopCnt, uint32_t counter)
{
    AscendC::MicroAPI::RegTensor<int32_t> orderReg;
    AscendC::MicroAPI::RegTensor<OFFSET_T> sortedIdxReg;
    AscendC::MicroAPI::RegTensor<OFFSET_T> sortedIdxShiftOneReg;
    AscendC::MicroAPI::RegTensor<int32_t> selReg;
    AscendC::MicroAPI::MaskReg cmpMask;
    AscendC::MicroAPI::MaskReg maskReg;
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::UnalignReg uOut;
    AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * vfLen);
        maskReg = AscendC::MicroAPI::UpdateMask<OFFSET_T>(counter);
        auto startAddr = indicesAddr + i * vfLen;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<OFFSET_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<OFFSET_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        if constexpr (std::is_same<int64_t, OFFSET_T>::value) {
            AscendC::MicroAPI::MaskReg maskHalf;
            AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                 maskHalf);
        } else {
            AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                                 cmpMask);
        }
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIdCountsAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template <typename T, typename U, typename OFFSET_T = U>
class ScatterNdUpdateBase {
public:
    int64_t indicesFactor_ = 0;
    int64_t afterAxisFactor_ = 0;
    int64_t afterAxis_ = 0;
    int64_t indexRankSize_ = 0;
    int64_t eachCoreAfterAxisCount_ = 0;
    int64_t shiftOffset_ = UB_AGLIN_VALUE / sizeof(OFFSET_T);
    uint32_t uniqueIdNum_ = 0;
    float maxScore_ = 0;
    int64_t varInAxis_ = 0;
    int64_t simdWithSort_ = 0;

    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;

    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> outOfstBuf_;
    TBuf<QuePosition::VECCALC> strideBuf_;
    TBuf<QuePosition::VECCALC> maxScoreBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> dataQueue_;

    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesOriginIdexQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> uniqueIdCountQue_;

    using IndexRegType = typename std::conditional<
        IsSameType<U, int64_t>::value,
        typename AscendC::MicroAPI::RegTensor<uint64_t, AscendC::MicroAPI::RegTraitNumTwo>,
        typename AscendC::MicroAPI::RegTensor<uint32_t>>::type;
    using InnerRegType = typename std::conditional<
        IsSameType<OFFSET_T, int64_t>::value,
        typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
        typename AscendC::MicroAPI::RegTensor<int32_t>>::type;

    using selRegType = typename std::conditional<IsSameType<T, bool>::value, int8_t, T>::type;

    __aicore__ inline void InitBaseBuffer(TPipe& pipe, uint32_t indicesNumber, GM_ADDR indices, GM_ADDR updates,
                                          GM_ADDR y)
    {
        indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
        yGm_.SetGlobalBuffer((__gm__ T*)(y));

        pipe.InitBuffer(strideBuf_, MAX_SHAPE_RANK * sizeof(U));
        pipe.InitBuffer(dataQueue_, DOUBLE_BUFFER, indicesFactor_ * afterAxisFactor_ * sizeof(T));
        pipe.InitBuffer(outOfstBuf_, indicesFactor_ * sizeof(OFFSET_T));
        pipe.InitBuffer(indicesBuf_, indicesFactor_ * indexRankSize_ * sizeof(U));
        pipe.InitBuffer(maxScoreBuf_, HASH_SCORE_BUF_SIZE * sizeof(float));

        pipe.InitBuffer(
            sortIndicesQue_,
            Ops::Base::CeilAlign(indicesFactor_ * sizeof(OFFSET_T) + SORT_PAD_NUM * UB_AGLIN_VALUE, UB_AGLIN_VALUE));
        pipe.InitBuffer(updatesOriginIdexQue_, DOUBLE_BUFFER, indicesFactor_ * sizeof(uint32_t));
        pipe.InitBuffer(uniqueIdCountQue_, DOUBLE_BUFFER,
                        Ops::Base::CeilAlign((indicesFactor_ + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
    }

    template <typename PARAM_T>
    __aicore__ inline void CopyIn(const LocalTensor<PARAM_T>& dstTensor, const GlobalTensor<PARAM_T>& srcTensor,
                                  int64_t dataLen)
    {
        DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<PARAM_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                   static_cast<PARAM_T>(0)};
        DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
    }

    template <typename PARAM_T>
    __aicore__ inline void CopyOut(const GlobalTensor<PARAM_T>& dstTensor, const LocalTensor<PARAM_T>& srcTensor,
                                   int64_t dataLen)
    {
        DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(PARAM_T)),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPad(dstTensor, srcTensor, copyParams);
    }

    __aicore__ inline void ComputeOutOfset(const LocalTensor<U> indicesLocal, const LocalTensor<OFFSET_T> outOfstLocal,
                                           int32_t indicesLen, int32_t rankSize)
    {
        LocalTensor<U> strideLocal = strideBuf_.Get<U>();

        __local_mem__ U* indicesLocalPtr = ((__local_mem__ U*)indicesLocal.GetPhyAddr());
        __local_mem__ OFFSET_T* outOfstLocalPtr = (__local_mem__ OFFSET_T*)outOfstLocal.GetPhyAddr();

        uint32_t dataLen = indicesLen;
        uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
        uint32_t indicesLenTimes = Ops::Base::CeilDiv(dataLen, vfLen);
        uint16_t loopCnt = static_cast<uint16_t>(indicesLenTimes);
        uint16_t rankSizeLoops = static_cast<uint16_t>(rankSize);

        __VEC_SCOPE__
        {
            InnerRegType inReg;
            InnerRegType outReg;
            InnerRegType orderReg;
            IndexRegType indexReg;
            AscendC::MicroAPI::MaskReg pregLoop;

            for (uint16_t i = 0; i < loopCnt; i++) {
                if constexpr (IsSameType<OFFSET_T, int64_t>::value) {
                    pregLoop = AscendC::MicroAPI::UpdateMask<OFFSET_T, AscendC::MicroAPI::RegTraitNumTwo>(dataLen);
                } else {
                    pregLoop = AscendC::MicroAPI::UpdateMask<OFFSET_T>(dataLen);
                }
                AscendC::MicroAPI::Duplicate(outReg, 0, pregLoop);
                AscendC::MicroAPI::Arange(orderReg, i * vfLen);
                AscendC::MicroAPI::Muls(orderReg, orderReg, rankSize, pregLoop);
                for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
                    OFFSET_T strideValue = static_cast<OFFSET_T>(strideLocal(dim));
                    indexReg = (IndexRegType&)orderReg;
                    if constexpr (IsSameType<U, int32_t>::value && IsSameType<OFFSET_T, int64_t>::value) {
                        AscendC::MicroAPI::RegTensor<int32_t> castReg;
                        AscendC::MicroAPI::DataCopyGather(castReg, indicesLocalPtr, indexReg, pregLoop);
                        MicroAPI::Cast<int64_t, int32_t, castTraitB322B64>(inReg, castReg, pregLoop);
                    } else {
                        AscendC::MicroAPI::DataCopyGather(inReg, indicesLocalPtr, indexReg, pregLoop);
                    }
                    AscendC::MicroAPI::Muls(inReg, inReg, strideValue, pregLoop);
                    AscendC::MicroAPI::Add(outReg, inReg, outReg, pregLoop);
                    AscendC::MicroAPI::Adds(orderReg, orderReg, (OFFSET_T)(1), pregLoop);
                }
                auto outOfstAddr = outOfstLocalPtr + i * vfLen;
                AscendC::MicroAPI::DataCopy(outOfstAddr, outReg, pregLoop);
            }
        }
    }

    __aicore__ inline uint32_t ComputeUniqueIdNum(LocalTensor<OFFSET_T> indicesLocal,
                                                  LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
    {
        __local_mem__ OFFSET_T* indicesAddr = (__local_mem__ OFFSET_T*)indicesLocal[shiftOffset_].GetPhyAddr();
        __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

        int64_t vfLen = Ops::Base::GetVRegSize() / sizeof(OFFSET_T);
        uint16_t loopCnt = Ops::Base::CeilDiv(dataLen + 1, vfLen);
        uint32_t counter = dataLen + 1;
        ComputeUniqueIdNumVf<OFFSET_T>((__ubuf__ OFFSET_T*)indicesAddr, (__ubuf__ int32_t*)uniqueIdCountsAddr, vfLen,
                                       loopCnt, counter);
        uint32_t elementCount = AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>() / sizeof(int32_t);
        if (elementCount > 0) {
            return static_cast<uint32_t>(elementCount - 1);
        } else {
            return 0;
        }
    }

    __aicore__ inline void SortAndComputeUniqueIdx(LocalTensor<OFFSET_T> outOfstLocal, int64_t rowLen)
    {
        LocalTensor<OFFSET_T> sortIndicesLocal = sortIndicesQue_.Get<OFFSET_T>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
        LocalTensor<OFFSET_T> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        AscendC::Sort<OFFSET_T, true, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, outOfstLocal,
                                                  static_cast<uint32_t>(rowLen));
        Duplicate(sortIndicesLocal, (OFFSET_T)-1, shiftOffset_);
        shiftSortLocal(rowLen) = -1;

        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        uniqueIdNum_ = ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);

        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
    }

    __aicore__ inline void CopyIndiceInSplitAfter(int64_t rowIdx, int64_t rowLen)
    {
        LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
        LocalTensor<OFFSET_T> outOfstLocal = outOfstBuf_.Get<OFFSET_T>();
        LocalTensor<float> dstLocal = maxScoreBuf_.Get<float>();

        int64_t rankSize = indexRankSize_;
        int64_t indicesOfset = rowIdx * indicesFactor_;
        CopyIn<U>(indicesLocal, indicesGm_[indicesOfset * rankSize], rowLen * rankSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, rankSize);
        if (simdWithSort_ == 0) {
            return;
        }
        if constexpr (IsSameType<OFFSET_T, int32_t>::value) {
            IndexStatisticInt32(outOfstLocal, dstLocal, maxScore_, rowLen, afterAxis_);
        } else {
            IndexStatisticInt64(outOfstLocal, dstLocal, maxScore_, rowLen, afterAxis_);
        }
    }

    __aicore__ inline void CopyUpdatesInSplitAfter(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
    {
        LocalTensor<T> updatesLocal = this->dataQueue_.template AllocTensor<T>();
        int64_t indicesOfset = rowIdx * indicesFactor_;
        DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                        static_cast<uint32_t>((afterAxis_ - colLen) * sizeof(T)),
                                        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
        int64_t rowOfset = indicesOfset * afterAxis_;
        int64_t updatesOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
        DataCopyPad(updatesLocal, updatesGm_[updatesOfset], copyParams, updatePadParams);
        this->dataQueue_.template EnQue(updatesLocal);
    }

    __aicore__ inline void CopyOutSplitAfter(int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        LocalTensor<T> dataLocal = dataQueue_.DeQue<T>();
        LocalTensor<OFFSET_T> outOfstLocal = outOfstBuf_.Get<OFFSET_T>();
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t i = 0; i < rowLen; i++) {
            int64_t rowOfset = outOfstLocal(i);
            int64_t outOfset = rowOfset + GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
            int64_t indicesValue = outOfstLocal(i);
            if (indicesValue >= 0 && indicesValue < varInAxis_) {
                CopyOut<T>(yGm_[outOfset], dataLocal[i * colLenAlignSize], colLen);
            }
        }
        dataQueue_.FreeTensor(dataLocal);
    }

    __aicore__ inline void CopyOutSplitAfterWithSort(int64_t rowLen, int64_t colLen, int64_t colIdx)
    {
        LocalTensor<T> dataLocal = dataQueue_.DeQue<T>();
        LocalTensor<OFFSET_T> sortIndicesLocal = sortIndicesQue_.Get<OFFSET_T>();
        LocalTensor<OFFSET_T> shiftSortLocal = sortIndicesLocal[shiftOffset_];
        LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.DeQue<uint32_t>();
        LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.DeQue<int32_t>();
        int64_t colLenAlignSize = Ops::Base::CeilAlign(colLen * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (int64_t i = 0; i < rowLen; i++) {
            int32_t uniqueIdx = uniqueIdCountLocal(i);
            int64_t inOfset = updatesOriginIdexLocal(uniqueIdx) * colLenAlignSize;
            int64_t outOfset = shiftSortLocal(uniqueIdx);
            outOfset += GetBlockIdx() * eachCoreAfterAxisCount_ + colIdx * afterAxisFactor_;
            int64_t indicesValue = shiftSortLocal(uniqueIdx);
            if (indicesValue >= 0 && indicesValue < varInAxis_) {
                CopyOut<T>(yGm_[outOfset], dataLocal[inOfset], colLen);
            }
        }

        uniqueIdCountQue_.EnQue(uniqueIdCountLocal);
        updatesOriginIdexQue_.EnQue(updatesOriginIdexLocal);
        dataQueue_.FreeTensor(dataLocal);
    }
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCalcMask(
    __gm__ INDICES_T* idxGmAddr, __gm__ TYPE_T* maskGmAddr, __gm__ PARAMS_T* outputGmAddr,
    const __ubuf__ TYPE_T* strideListAddr, const __ubuf__ TYPE_T* outputShapeAddr, TYPE_T indiceBlockOffSet,
    uint32_t rankSize, uint32_t currBlockHandleIdx, int64_t varInAxis, __gm__ TYPE_T* varIdxGmAddr)
{
    for (uint32_t index = threadIdx.x; index < currBlockHandleIdx; index += blockDim.x) {
        TYPE_T globalIndiceRowOffset = indiceBlockOffSet + index;
        INDICES_T idx = 0;
        bool outOfBound = false;
        for (TYPE_T dim = 0; dim < rankSize; ++dim) {
            INDICES_T indiceVal = idxGmAddr[globalIndiceRowOffset * rankSize + dim];
            outOfBound |= static_cast<TYPE_T>(indiceVal) > outputShapeAddr[dim];
            idx += indiceVal * strideListAddr[dim];
        }
        if (!outOfBound) {
            if (idx >= 0 && idx < varInAxis) {
                asc_atomic_max(maskGmAddr + idx, globalIndiceRowOffset);
                varIdxGmAddr[globalIndiceRowOffset] = idx;
            }
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERMINISTIC) inline void ScatterNdUpdateSimtCalcMaskUnSort(
    uint32_t indicesCount, int64_t varFullDimSize, uint64_t indicesStartGmOffset, __gm__ TYPE_T* workspaceMaskAddr,
    __gm__ TYPE_T* varIdxGmAddr, __local_mem__ OFFSET_T* indicesLocalAddr, uint32_t sliceSize)
{
    for (uint32_t i = threadIdx.x; i < indicesCount; i += blockDim.x) {
        OFFSET_T indicesValue = indicesLocalAddr[i];
        if (!(indicesValue >= 0 && indicesValue < varFullDimSize)) {
            continue;
        }
        asc_atomic_max(workspaceMaskAddr + indicesValue / sliceSize, static_cast<TYPE_T>(indicesStartGmOffset + i));
        varIdxGmAddr[indicesStartGmOffset + i] = static_cast<TYPE_T>(indicesValue);
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERMINISTIC) inline void ScatterNdUpdateSimtCalcMaskSort(
    uint32_t uniqueIdNum, int64_t varFirstDimSize, uint64_t indicesStartGmOffset, __gm__ TYPE_T* workspaceMaskAddr,
    __local_mem__ OFFSET_T* indicesSortedPtr, __local_mem__ uint32_t* updatesOriginIdxAddr,
    __local_mem__ int32_t* uniqueIdCountAddr, uint32_t sliceSize)
{
    for (uint32_t i = threadIdx.x; i < uniqueIdNum; i += blockDim.x) {
        int32_t repeatTimes = uniqueIdCountAddr[i + 1] - uniqueIdCountAddr[i];
        int32_t lastIndicesIdx = uniqueIdCountAddr[i] + repeatTimes - 1;
        OFFSET_T indicesValue = indicesSortedPtr[lastIndicesIdx];
        indicesValue /= sliceSize;
        if (!(indicesValue >= 0 && indicesValue < varFirstDimSize)) {
            continue;
        }

        uint32_t indicesLocalOffset = updatesOriginIdxAddr[lastIndicesIdx];
        asc_atomic_max(workspaceMaskAddr + indicesValue,
                       static_cast<TYPE_T>(indicesStartGmOffset + indicesLocalOffset));
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERMINISTIC) inline void ScatterNdUpdateSimtWriteVarIdx(
    uint32_t indicesCount, int64_t varFullDimSize, uint64_t indicesStartGmOffset, __gm__ TYPE_T* varIdxGmAddr,
    __local_mem__ OFFSET_T* indicesLocalAddr)
{
    for (uint32_t i = threadIdx.x; i < indicesCount; i += blockDim.x) {
        OFFSET_T indicesValue = indicesLocalAddr[i];
        if (indicesValue >= 0 && indicesValue < varFullDimSize) {
            varIdxGmAddr[indicesStartGmOffset + i] = static_cast<TYPE_T>(indicesValue);
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
class ScatterNdUpdateDeterministicCommon : public ScatterNdUpdateBase<PARAMS_T, INDICES_T, OFFSET_T> {
public:
    __aicore__ inline ScatterNdUpdateDeterministicCommon(const ScatterNdUpdateRegBaseTilingData& tilingData,
                                                         TPipe& pipe)
        : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CalcMask();
    __aicore__ inline void InitUpdateBuffer();
    __aicore__ inline void InitMaskGm(uint64_t totalSize, GM_ADDR workspace);
    __aicore__ inline void CopyInIndices(uint64_t indicesGmOffset, uint32_t indicesCount);
    __aicore__ inline uint32_t DeterministicSortAndComputeUniqueIdx(int64_t rowLen,
                                                                    LocalTensor<OFFSET_T> indicesSrcLocal,
                                                                    LocalTensor<OFFSET_T> sortIndicesLocal,
                                                                    LocalTensor<int32_t> uniqueIdCountLocal,
                                                                    LocalTensor<uint32_t> updatesOriginIdexLocal);

protected:
    TPipe& pipe_;
    const ScatterNdUpdateRegBaseTilingData& tiling_;
    GlobalTensor<INDICES_T> idxGm;
    GlobalTensor<PARAMS_T> updateGm;
    GlobalTensor<PARAMS_T> outputGm;
    GlobalTensor<TYPE_T> maskGm;
    GlobalTensor<TYPE_T> varIdxGm;
    GlobalTensor<TYPE_T> maskBlockGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inQueX;
    TQue<QuePosition::VECIN, 1> indicesQue_;
    TBuf<QuePosition::VECCALC> deterUpdatesOriginIdxBuf_;
    TBuf<QuePosition::VECCALC> deterUniqueIdCountBuf_;

    uint32_t blockIdx;
    TYPE_T currBlockHandleIdx = 0;
    TYPE_T indiceBlockOffSet = 0;
    int64_t indicesUbFactor = 0;
    uint32_t rankSize_ = 0;
    uint64_t indicesBlockLoop_{0};
    uint64_t indicesTailLoopSize_{0};
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::InitMaskGm(
    uint64_t totalSize, GM_ADDR workspace)
{
    uint64_t blockNum = GetBlockNum();
    uint64_t perCoreInitNum = Ops::Base::CeilDiv(totalSize, blockNum);
    uint64_t alignFactor = LEAST_DEAL_SIZE / sizeof(PARAMS_T);
    perCoreInitNum = Ops::Base::CeilDiv(perCoreInitNum, alignFactor) * alignFactor;

    uint64_t initUsedCore = Ops::Base::CeilDiv(totalSize, perCoreInitNum);
    if (GetBlockIdx() >= initUsedCore) {
        return;
    }
    uint64_t tailCoreInitNum = totalSize - (initUsedCore - 1) * perCoreInitNum;

    uint64_t maskBlockOffset = GetBlockIdx() * perCoreInitNum;
    maskBlockGm.SetGlobalBuffer((__gm__ TYPE_T*)workspace + maskBlockOffset);

    uint64_t maskBlockLen = perCoreInitNum;
    if (GetBlockIdx() == initUsedCore - 1) {
        maskBlockLen = tailCoreInitNum;
    }
    InitGlobalMemory(maskBlockGm, maskBlockLen, static_cast<TYPE_T>(MASK_DEFAULT));
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::InitBase(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx = GetBlockIdx();
    indicesUbFactor = tiling_.indicesUbFactor;
    rankSize_ = tiling_.rankSize;

    idxGm.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    updateGm.SetGlobalBuffer((__gm__ PARAMS_T*)updates);
    outputGm.SetGlobalBuffer((__gm__ PARAMS_T*)y);
    maskGm.SetGlobalBuffer((__gm__ TYPE_T*)workspace);
    varIdxGm.SetGlobalBuffer((__gm__ TYPE_T*)workspace + (tiling_.varStorageInAxis + 1));

    InitMaskGm(tiling_.varInAxis, workspace);
    if (blockIdx >= tiling_.calcMaskUsedCoreNum) {
        return;
    }

    this->indiceBlockOffSet = static_cast<TYPE_T>(blockIdx * tiling_.normCoreHandleIdx);

    if (blockIdx == tiling_.calcMaskUsedCoreNum - 1) {
        this->currBlockHandleIdx = tiling_.tailCoreHandleIdx;
    } else {
        this->currBlockHandleIdx = tiling_.normCoreHandleIdx;
    }

    pipe_.InitBuffer(indicesQue_, 1,
                     Ops::Base::CeilAlign(tiling_.indicesUbFactor * rankSize_ * sizeof(INDICES_T), UB_AGLIN_VALUE));

    pipe_.InitBuffer(this->strideBuf_, MAX_SHAPE_RANK * sizeof(INDICES_T));
    pipe_.InitBuffer(this->outOfstBuf_,
                     Ops::Base::CeilAlign(tiling_.indicesUbFactor * sizeof(OFFSET_T), UB_AGLIN_VALUE));

    pipe_.InitBuffer(this->maxScoreBuf_, HASH_SCORE_BUF_SIZE * sizeof(float));

    indicesBlockLoop_ = tiling_.normBlockLoop;
    indicesTailLoopSize_ = tiling_.normBlockTail;
    if (blockIdx == tiling_.calcMaskUsedCoreNum - 1) {
        indicesBlockLoop_ = tiling_.tailBlockLoop;
        indicesTailLoopSize_ = tiling_.tailBlockTail;
    }

    pipe_.InitBuffer(this->sortIndicesQue_,
                     Ops::Base::CeilAlign(tiling_.indicesUbFactor * sizeof(OFFSET_T), UB_AGLIN_VALUE) +
                         SORT_PAD_NUM * UB_AGLIN_VALUE);
    pipe_.InitBuffer(deterUpdatesOriginIdxBuf_,
                     Ops::Base::CeilAlign(tiling_.indicesUbFactor * sizeof(uint32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(deterUniqueIdCountBuf_,
                     Ops::Base::CeilAlign(tiling_.indicesUbFactor * sizeof(int32_t), UB_AGLIN_VALUE) +
                         SORT_PAD_NUM * UB_AGLIN_VALUE);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyInIndices(
    uint64_t indicesGmOffset, uint32_t indicesCount)
{
    LocalTensor<INDICES_T> indicesLocal = indicesQue_.AllocTensor<INDICES_T>();

    DataCopyExtParams indicesCopyParams{1, static_cast<uint32_t>(indicesCount * sizeof(INDICES_T)), 0, 0, 0};
    DataCopyPadExtParams<INDICES_T> indicesPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, idxGm[indicesGmOffset], indicesCopyParams, indicesPadParams);
    indicesQue_.EnQue<INDICES_T>(indicesLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline uint32_t
ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::DeterministicSortAndComputeUniqueIdx(
    int64_t rowLen, LocalTensor<OFFSET_T> indicesSrcLocal, LocalTensor<OFFSET_T> sortIndicesLocal,
    LocalTensor<int32_t> uniqueIdCountLocal, LocalTensor<uint32_t> updatesOriginIdexLocal)
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    LocalTensor<OFFSET_T> shiftSortLocal = sortIndicesLocal[this->shiftOffset_];
    AscendC::Sort<OFFSET_T, true, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesSrcLocal,
                                              static_cast<uint32_t>(rowLen));
    Duplicate(sortIndicesLocal, (OFFSET_T)-1, this->shiftOffset_);
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    shiftSortLocal(rowLen) = -1;

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    return this->ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CalcMask()
{
    if (blockIdx >= tiling_.calcMaskUsedCoreNum) {
        return;
    }

    LocalTensor<INDICES_T> strideLocal = this->strideBuf_.template Get<INDICES_T>();
    for (uint32_t i = 0; i < MAX_SHAPE_RANK; i++) {
        strideLocal(i) = static_cast<INDICES_T>(tiling_.strideList[i]);
    }

    int64_t varFirstDimSize = tiling_.varStorageInAxis;
    int64_t varFullDimSize = tiling_.outputStorageShapeSize;
    __gm__ TYPE_T* workspaceMaskAddr = (__gm__ TYPE_T*)(maskGm.GetPhyAddr());
    __gm__ TYPE_T* varIdxGmAddr = (__gm__ TYPE_T*)(varIdxGm.GetPhyAddr());

    uint32_t indicesCount = tiling_.indicesUbFactor;
    for (uint64_t idx = 0; idx < indicesBlockLoop_; idx++) {
        if (idx == indicesBlockLoop_ - 1) {
            indicesCount = indicesTailLoopSize_;
        }
        uint64_t indicesStartGmOffset = blockIdx * tiling_.normCoreHandleIdx + idx * tiling_.indicesUbFactor;

        LocalTensor<OFFSET_T> flatOfstLocal;

        CopyInIndices(indicesStartGmOffset * rankSize_, indicesCount * rankSize_);
        LocalTensor<INDICES_T> indicesLocal = indicesQue_.DeQue<INDICES_T>();

        flatOfstLocal = this->outOfstBuf_.template Get<OFFSET_T>();
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        this->ComputeOutOfset(indicesLocal, flatOfstLocal, indicesCount, rankSize_);
        indicesQue_.FreeTensor(indicesLocal);

        LocalTensor<float> hashLocal = this->maxScoreBuf_.template Get<float>();
        float maxScore = 0.0f;
        if constexpr (IsSameType<OFFSET_T, int32_t>::value) {
            IndexStatisticInt32(flatOfstLocal, hashLocal, maxScore, indicesCount, tiling_.afterAxis);
        } else {
            IndexStatisticInt64(flatOfstLocal, hashLocal, maxScore, indicesCount, tiling_.afterAxis);
        }

        __local_mem__ OFFSET_T* flatOfstAddr = (__local_mem__ OFFSET_T*)(flatOfstLocal.GetPhyAddr());

        if (maxScore > SORT_HIST_THRESHOLD) {
            asc_vf_call<ScatterNdUpdateSimtWriteVarIdx<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>>(
                dim3(THREAD_NUM_DETERMINISTIC), indicesCount, varFullDimSize, indicesStartGmOffset, varIdxGmAddr,
                flatOfstAddr);

            LocalTensor<OFFSET_T> indicesSortedLocal = this->sortIndicesQue_.template Get<OFFSET_T>();
            LocalTensor<uint32_t> updatesOriginIdxLocal = deterUpdatesOriginIdxBuf_.template Get<uint32_t>();
            LocalTensor<int32_t> uniqueIdCountLocal = deterUniqueIdCountBuf_.template Get<int32_t>();
            __local_mem__ OFFSET_T* indicesSortedPtr = (__local_mem__ OFFSET_T*)(indicesSortedLocal.GetPhyAddr()) +
                                                       this->shiftOffset_;
            __local_mem__ uint32_t* updatesOriginIdxAddr = (__local_mem__ uint32_t*)(updatesOriginIdxLocal
                                                                                         .GetPhyAddr());
            __local_mem__ int32_t* uniqueIdCountAddr = (__local_mem__ int32_t*)(uniqueIdCountLocal.GetPhyAddr());
            uint32_t uniqueIdNum = this->DeterministicSortAndComputeUniqueIdx(
                indicesCount, flatOfstLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);

            asc_vf_call<ScatterNdUpdateSimtCalcMaskSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>>(
                dim3(THREAD_NUM_DETERMINISTIC), uniqueIdNum, varFirstDimSize, indicesStartGmOffset, workspaceMaskAddr,
                indicesSortedPtr, updatesOriginIdxAddr, uniqueIdCountAddr, this->tiling_.sliceSize);
        } else {
            asc_vf_call<ScatterNdUpdateSimtCalcMaskUnSort<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>>(
                dim3(THREAD_NUM_DETERMINISTIC), indicesCount, varFullDimSize, indicesStartGmOffset, workspaceMaskAddr,
                varIdxGmAddr, flatOfstAddr, this->tiling_.sliceSize);
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::InitUpdateBuffer()
{
    pipe_.Reset();
    pipe_.InitBuffer(inQueX, DOUBLE_BUFFER,
                     Ops::Base::CeilAlign(tiling_.afterAxisFactor * sizeof(PARAMS_T), UB_AGLIN_VALUE));
}

} // namespace ScatterNdUpdate

#endif