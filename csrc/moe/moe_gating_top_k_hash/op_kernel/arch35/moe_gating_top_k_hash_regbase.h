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
 * \file moe_gating_top_k_hash_regbase.h
 * \brief
 */
#ifndef MOE_GATING_TOP_K_REGBASE_H
#define MOE_GATING_TOP_K_REGBASE_H

#include <cmath>
#include "common.h"
#include "kernel_operator.h"
#include "kernel_utils.h"

namespace MoeGatingTopKHashRegbaseNS
{
    using namespace AscendC;
    using MicroAPI::RegTensor;

    constexpr int32_t CONSTANT_TWO = 2;
    constexpr int32_t CONSTANT_THREE = 3;
    constexpr int32_t CONSTANT_FOUR = 4;
    constexpr int32_t CONSTANT_EIGHT = 8;
    constexpr uint32_t VL_FLOAT_SIZE = VECTOR_REG_WIDTH / sizeof(float);
    constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, true};

    template <typename T, typename  U1, typename U2>
    class MoeGatingTopKHashRegbase
    {
    public:
        __aicore__ inline MoeGatingTopKHashRegbase(){};
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out, GM_ADDR workspace,
                                    const MoeGatingTopKHashRegbaseTilingData *tilingData, TPipe *tPipe);
        __aicore__ inline void Process();

    private:
        __aicore__ inline void CopyInBias();
        __aicore__ inline void CopyInX(int64_t progress);
        __aicore__ inline void ComputeX();
        __aicore__ inline void ComputeSoftmax();
        __aicore__ inline void ComputeSigmoid();
        __aicore__ inline void ComputeSoftplus();
        __aicore__ inline void CopyOutXNorm(int64_t progress);
        __aicore__ inline void SortInGroup();
        __aicore__ inline void SelectTopKGroupIndex();
        __aicore__ inline void FinalSortByKGroup();
        __aicore__ inline void FinalSortAfterKGroup();
        __aicore__ inline void SelectTopKExpertScore();
        __aicore__ inline void TopKCompute();
        __aicore__ inline void HashCompute(int64_t row);
        __aicore__ inline void CopyOut(int64_t progress);

        __aicore__ inline void TopKGatherWithSmallKAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                          LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                          float eps, float routedScalingFactor);
        __aicore__ inline void TopKGatherWithLargeKAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                          LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                          float eps, float routedScalingFactor);
        __aicore__ inline void TopKGatherWithSmallKNotAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                             LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                             float eps, float routedScalingFactor, int32_t expertIdxPad,
                                                             int32_t perGroupExpertCountAlign);
        __aicore__ inline void TopKGatherWithLargeKNotAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                             LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                             float eps, float routedScalingFactor, int32_t expertIdxPad,
                                                             int32_t perGroupExpertCountAlign);
        __aicore__ inline void HashGatherWithSmallKAlignE(LocalTensor<float> xSigmoidTensor,
                                                          LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                          float eps, float routedScalingFactor);
        __aicore__ inline void HashGatherWithSmallKNotAlignE(LocalTensor<float> xSigmoidTensor,
                                                             LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                             float eps, float routedScalingFactor, int32_t expertIdxPad,
                                                             int32_t perGroupExpertCountAlign);

    private:
        TPipe *pipe_;
        TQue<QuePosition::VECIN, 1> xInQueue_;
        TQue<QuePosition::VECOUT, 1> yOutQueue_;
        TQue<QuePosition::VECOUT, 1> expertIdxOutQueue_;
        TQue<QuePosition::VECOUT, 1> outOutQueue_;

        TBuf<TPosition::VECCALC> biasBuf_;
        TBuf<QuePosition::VECCALC> xBiasBuf_;
        TBuf<QuePosition::VECCALC> xSigmoidBuf_;
        TBuf<QuePosition::VECCALC> groupBuf_;
        TBuf<QuePosition::VECCALC> sortedInGroupBuf_;
        TBuf<QuePosition::VECCALC> sortedGroupBuf_;
        TBuf<TPosition::VECCALC> indexBuffer_;
        TBuf<TPosition::VECCALC> finalSortBuffer_;

        GlobalTensor<T> xGm_;
        GlobalTensor<T> biasGm_;
        GlobalTensor<T> yGm_;
        GlobalTensor<int32_t> expertIdxGm_;
        GlobalTensor<float> outGm_;
        GlobalTensor<U1> inputIdsGm_;
        GlobalTensor<U2> tid2eidGm_;

        LocalTensor<uint32_t> indexTensor;
        LocalTensor<float> sortedInGroupTensor;
        LocalTensor<float> sortedGroupTensor;
        LocalTensor<float> mrgSortTensor;

        int64_t blockIdx_;
        int64_t curCoreRowCount_;
        int64_t expertCount_;
        int64_t k_;
        int64_t kGroup_;
        int64_t groupCount_;
        float routedScalingFactor_;
        float eps_;
        bool hasBias_ = false;
        bool hashFlag_ = false;

        int64_t perGroupExpertCount_;
        int64_t perGroupExpertCountAlign_;
        int64_t expertCountAlign_;
        int64_t groupSelectMode_;
        const MoeGatingTopKHashRegbaseTilingData *tilingData_;
    };

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::CopyInBias()
    {
        if (!hasBias_)
        {
            return;
        }
        LocalTensor<T> biasTensor = biasBuf_.Get<T>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = groupCount_;
        dataCopyParams.blockLen = perGroupExpertCount_ * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(T) / BLOCK_BYTES;
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};

        DataCopyPad(biasTensor, biasGm_, dataCopyParams, dataCopyPadParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::CopyInX(int64_t row)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = groupCount_;
        dataCopyParams.blockLen = perGroupExpertCount_ * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(T) / BLOCK_BYTES;
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
        if (tilingData_->normType == 0)
        {
            LocalTensor<float> xInLocalTensor = xInQueue_.AllocTensor<float>();
            if constexpr (IsSameType<T, float>::value)
            {
                DataCopyPad(xInLocalTensor, xGm_[row * expertCount_], dataCopyParams, dataCopyPadParams);
            }
            else
            {
                DataCopyPad(xInLocalTensor[expertCountAlign_].ReinterpretCast<T>(), xGm_[row * expertCount_], dataCopyParams,
                            dataCopyPadParams);
            }
            xInQueue_.EnQue(xInLocalTensor);
        }
        else
        {
            LocalTensor<T> xInLocalTensor = xInQueue_.AllocTensor<T>();
            DataCopyPad(xInLocalTensor, xGm_[row * expertCount_], dataCopyParams, dataCopyPadParams);
            xInQueue_.EnQue(xInLocalTensor);
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::ComputeSoftmax()
    {
        LocalTensor<float> xSoftmaxTensor = xSigmoidBuf_.Get<float>();
        LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
        indexTensor = indexBuffer_.Get<uint32_t>();

        uint32_t size = perGroupExpertCountAlign_ * groupCount_;
        uint32_t perGroupExpertCount0 = perGroupExpertCount_;
        uint32_t perGroupExpertCountAlign0 = perGroupExpertCountAlign_;
        uint16_t groupCount0 = groupCount_;
        LocalTensor<float> xInLocalTensor = xInQueue_.DeQue<float>();
        if constexpr (!IsSameType<T, float>::value)
        {
            Cast(xInLocalTensor, xInLocalTensor[expertCountAlign_].template ReinterpretCast<T>(), RoundMode::CAST_NONE,
                 expertCountAlign_);
        }

        int64_t duplicateNum = perGroupExpertCount_ % ONE_REPEAT_SORT_NUM;
        int duplicateIndex = perGroupExpertCount_ - duplicateNum;
        if (perGroupExpertCount_ != perGroupExpertCountAlign_ && duplicateNum > 0)
        {
            uint64_t mask0 = UINT64_MAX;
            mask0 = mask0 << duplicateNum;
            mask0 = mask0 & (UINT64_MAX >> ONE_REPEAT_SORT_NUM);
            uint64_t mask[2] = {mask0, 0};
            Duplicate(xInLocalTensor.ReinterpretCast<int32_t>()[duplicateIndex], MIN_FP32, mask, groupCount_, 1,
                      (perGroupExpertCountAlign_ * sizeof(float)) / BLOCK_BYTES);
        }

        SoftMaxShapeInfo softmaxShapeInfo;
        softmaxShapeInfo.srcM = 1;
        softmaxShapeInfo.srcK = groupCount_ * perGroupExpertCountAlign_;
        softmaxShapeInfo.oriSrcM = 1;
        softmaxShapeInfo.oriSrcK = groupCount_ * perGroupExpertCountAlign_;
        SoftMax<float, true, false>(xSoftmaxTensor, xInLocalTensor, tilingData_->softmaxTilingData, softmaxShapeInfo);

        __local_mem__ float *softmaxOutAddr = (__local_mem__ float *)xSoftmaxTensor.GetPhyAddr();
        __local_mem__ int32_t *indexOutAddr = (__local_mem__ int32_t *)indexTensor.GetPhyAddr();
        __local_mem__ float *addBiasOutAddr = (__local_mem__ float *)xBiasTensor.GetPhyAddr();
        if (hasBias_)
        {
            LocalTensor<T> biasTensor = biasBuf_.Get<T>();
            __VEC_SCOPE__
            {
                RegTensor<float> vregBiasFp32;
                RegTensor<int32_t> vregIndex;
                RegTensor<float> vregSoftmaxResult;
                RegTensor<float> vregBiasResult;
                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));
                __local_mem__ T *biasAddr = (__local_mem__ T *)biasTensor.GetPhyAddr();

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<float>(size);
                    LoadOneTensorForDtypeT<float>(softmaxOutAddr, vregSoftmaxResult, preg0, i * VL_FLOAT_SIZE);
                    LoadOneTensorForDtypeT<T>(biasAddr, vregBiasFp32, preg0, i * VL_FLOAT_SIZE);
                    // add bias
                    MicroAPI::Add(vregBiasResult, vregSoftmaxResult, vregBiasFp32, preg0);
                    // 使用Arange生成排序索引, 起始值为i乘每个循环的veclen
                    MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * VL_FLOAT_SIZE));
                    MicroAPI::DataCopy(addBiasOutAddr + i * VL_FLOAT_SIZE, vregBiasResult, preg0);
                    MicroAPI::DataCopy(indexOutAddr + i * VL_FLOAT_SIZE, vregIndex, preg0);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();

                // pad min fp32
                RegTensor<float> vregPad;
                MicroAPI::UnalignReg u0;
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    auto padUbAddr = addBiasOutAddr + i * perGroupExpertCountAlign0 + perGroupExpertCount0;
                    MicroAPI::DataCopyUnAlign(padUbAddr, vregPad, u0, perGroupExpertCountAlign0 - perGroupExpertCount0);
                    MicroAPI::DataCopyUnAlignPost(padUbAddr, u0, 0);
                }
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<int32_t> vregIndex;
                RegTensor<float> vregSoftmaxResult;
                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<float>(size);
                    LoadOneTensorForDtypeT<float>(softmaxOutAddr, vregSoftmaxResult, preg0, i * VL_FLOAT_SIZE);
                    // 使用Arange生成排序索引, 起始值为i乘每个循环的veclen
                    MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * VL_FLOAT_SIZE));
                    MicroAPI::DataCopy(addBiasOutAddr + i * VL_FLOAT_SIZE, vregSoftmaxResult, preg0);
                    MicroAPI::DataCopy(indexOutAddr + i * VL_FLOAT_SIZE, vregIndex, preg0);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();

                // pad min fp32
                RegTensor<float> vregPad;
                MicroAPI::UnalignReg u0;
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    auto padUbAddr = addBiasOutAddr + i * perGroupExpertCountAlign0 + perGroupExpertCount0;
                    MicroAPI::DataCopyUnAlign(padUbAddr, vregPad, u0, perGroupExpertCountAlign0 - perGroupExpertCount0);
                    MicroAPI::DataCopyUnAlignPost(padUbAddr, u0, 0);
                }
            }
        }
        xInQueue_.FreeTensor(xInLocalTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::ComputeSigmoid()
    {
        LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
        LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
        indexTensor = indexBuffer_.Get<uint32_t>();

        uint32_t size = perGroupExpertCountAlign_ * groupCount_;
        uint32_t perGroupExpertCount0 = perGroupExpertCount_;
        uint32_t perGroupExpertCountAlign0 = perGroupExpertCountAlign_;
        uint16_t groupCount0 = groupCount_;
        LocalTensor<T> xInLocalTensor = xInQueue_.DeQue<T>();
        if (hasBias_)
        {
            LocalTensor<T> biasTensor = biasBuf_.Get<T>();
            __VEC_SCOPE__
            {
                RegTensor<float> vregBiasFp32;
                RegTensor<int32_t> vregIndex;
                RegTensor<float> vregSigmoidResult;
                RegTensor<float> vregBiasResult;
                RegTensor<float> vregOne;
                RegTensor<float> vregInFp32;
                RegTensor<float> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));

                __local_mem__ T *inputAddr = (__local_mem__ T *)xInLocalTensor.GetPhyAddr();
                __local_mem__ T *biasAddr = (__local_mem__ T *)biasTensor.GetPhyAddr();
                __local_mem__ float *sigmoidOutAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
                __local_mem__ int32_t *indexOutAddr = (__local_mem__ int32_t *)indexTensor.GetPhyAddr();
                __local_mem__ float *addBiasOutAddr = (__local_mem__ float *)xBiasTensor.GetPhyAddr();

                // sigmoid
                MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);
                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<float>(size);
                    LoadTwoTensorForDtypeT<T>(inputAddr, biasAddr, vregInFp32, vregBiasFp32, preg0, preg0,
                                              i * VL_FLOAT_SIZE, i * VL_FLOAT_SIZE);
                    MicroAPI::Muls(vreg1, vregInFp32, static_cast<float>(-1), preg0);
                    MicroAPI::Exp(vreg2, vreg1, preg0);
                    MicroAPI::Adds(vreg3, vreg2, static_cast<float>(1), preg0);
                    MicroAPI::Div<float, &mode>(vregSigmoidResult, vregOne, vreg3, preg0);
                    // add bias
                    MicroAPI::Add(vregBiasResult, vregSigmoidResult, vregBiasFp32, preg0);
                    // 使用Arange生成排序索引, 起始值为i乘每个循环的veclen
                    MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * VL_FLOAT_SIZE));
                    MicroAPI::DataCopy(sigmoidOutAddr + i * VL_FLOAT_SIZE, vregSigmoidResult, preg0);
                    MicroAPI::DataCopy(addBiasOutAddr + i * VL_FLOAT_SIZE, vregBiasResult, preg0);
                    MicroAPI::DataCopy(indexOutAddr + i * VL_FLOAT_SIZE, vregIndex, preg0);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();

                // pad min fp32
                RegTensor<float> vregPad;
                MicroAPI::UnalignReg u0;
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    auto padUbAddr = addBiasOutAddr + i * perGroupExpertCountAlign0 + perGroupExpertCount0;
                    MicroAPI::DataCopyUnAlign(padUbAddr, vregPad, u0, perGroupExpertCountAlign0 - perGroupExpertCount0);
                    MicroAPI::DataCopyUnAlignPost(padUbAddr, u0, 0);
                }
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<float> vregBiasFp32;
                RegTensor<int32_t> vregIndex;
                RegTensor<float> vregSigmoidResult;
                RegTensor<float> vregOne;
                RegTensor<float> vregInFp32;
                RegTensor<float> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));

                __local_mem__ T *inputAddr = (__local_mem__ T *)xInLocalTensor.GetPhyAddr();
                __local_mem__ float *sigmoidOutAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
                __local_mem__ int32_t *indexOutAddr = (__local_mem__ int32_t *)indexTensor.GetPhyAddr();
                __local_mem__ float *addBiasOutAddr = (__local_mem__ float *)xBiasTensor.GetPhyAddr();

                // sigmoid
                MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);
                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<float>(size);
                    LoadOneTensorForDtypeT<T>(inputAddr, vregInFp32, preg0, i * VL_FLOAT_SIZE);
                    MicroAPI::Muls(vreg1, vregInFp32, static_cast<float>(-1), preg0);
                    MicroAPI::Exp(vreg2, vreg1, preg0);
                    MicroAPI::Adds(vreg3, vreg2, static_cast<float>(1), preg0);
                    MicroAPI::Div<float, &mode>(vregSigmoidResult, vregOne, vreg3, preg0);
                    // 使用Arange生成排序索引, 起始值为i乘每个循环的veclen
                    MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * VL_FLOAT_SIZE));
                    MicroAPI::DataCopy(sigmoidOutAddr + i * VL_FLOAT_SIZE, vregSigmoidResult, preg0);
                    MicroAPI::DataCopy(addBiasOutAddr + i * VL_FLOAT_SIZE, vregSigmoidResult, preg0);
                    MicroAPI::DataCopy(indexOutAddr + i * VL_FLOAT_SIZE, vregIndex, preg0);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();

                // pad inf
                RegTensor<float> vregPad;
                MicroAPI::UnalignReg u0;
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    auto padUbAddr = addBiasOutAddr + i * perGroupExpertCountAlign0 + perGroupExpertCount0;
                    MicroAPI::DataCopyUnAlign(padUbAddr, vregPad, u0, perGroupExpertCountAlign0 - perGroupExpertCount0);
                    MicroAPI::DataCopyUnAlignPost(padUbAddr, u0, 0);
                }
            }
        }
        xInQueue_.FreeTensor(xInLocalTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::ComputeSoftplus()
    {
        LocalTensor<float> xSoftplusTensor = xSigmoidBuf_.Get<float>();
        LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
        indexTensor = indexBuffer_.Get<uint32_t>();

        uint32_t size = perGroupExpertCountAlign_ * groupCount_;
        uint32_t perGroupExpertCount0 = perGroupExpertCount_;
        uint32_t perGroupExpertCountAlign0 = perGroupExpertCountAlign_;
        uint16_t groupCount0 = groupCount_;
        LocalTensor<T> xInLocalTensor = xInQueue_.DeQue<T>();
        if (hasBias_)
        {
            LocalTensor<T> biasTensor = biasBuf_.Get<T>();
            __VEC_SCOPE__
            {
                RegTensor<float> vregBiasFp32;
                RegTensor<int32_t> vregIndex;
                RegTensor<float> vregSoftplusResult;
                RegTensor<float> vregBiasResult;
                RegTensor<float> vregOne;
                RegTensor<float> vregInFp32;
                RegTensor<float> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));

                __local_mem__ T *inputAddr = (__local_mem__ T *)xInLocalTensor.GetPhyAddr();
                __local_mem__ T *biasAddr = (__local_mem__ T *)biasTensor.GetPhyAddr();
                __local_mem__ float *softplusOutAddr = (__local_mem__ float *)xSoftplusTensor.GetPhyAddr();
                __local_mem__ int32_t *indexOutAddr = (__local_mem__ int32_t *)indexTensor.GetPhyAddr();
                __local_mem__ float *addBiasOutAddr = (__local_mem__ float *)xBiasTensor.GetPhyAddr();

                // softplus
                MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);
                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<float>(size);
                    LoadTwoTensorForDtypeT<T>(inputAddr, biasAddr, vregInFp32, vregBiasFp32, preg0, preg0,
                                              i * VL_FLOAT_SIZE, i * VL_FLOAT_SIZE);
                    MicroAPI::Exp(vreg1, vregInFp32, preg0);
                    MicroAPI::Adds(vreg2, vreg1, static_cast<float>(1), preg0);
                    MicroAPI::Ln(vreg3, vreg2, preg0);
                    MicroAPI::Sqrt(vregSoftplusResult, vreg3, preg0);
                    // add bias
                    MicroAPI::Add(vregBiasResult, vregSoftplusResult, vregBiasFp32, preg0);
                    // 使用Arange生成排序索引, 起始值为i乘每个循环的veclen
                    MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * VL_FLOAT_SIZE));
                    MicroAPI::DataCopy(softplusOutAddr + i * VL_FLOAT_SIZE, vregSoftplusResult, preg0);
                    MicroAPI::DataCopy(addBiasOutAddr + i * VL_FLOAT_SIZE, vregBiasResult, preg0);
                    MicroAPI::DataCopy(indexOutAddr + i * VL_FLOAT_SIZE, vregIndex, preg0);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();

                // pad min fp32
                RegTensor<float> vregPad;
                MicroAPI::UnalignReg u0;
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    auto padUbAddr = addBiasOutAddr + i * perGroupExpertCountAlign0 + perGroupExpertCount0;
                    MicroAPI::DataCopyUnAlign(padUbAddr, vregPad, u0, perGroupExpertCountAlign0 - perGroupExpertCount0);
                    MicroAPI::DataCopyUnAlignPost(padUbAddr, u0, 0);
                }
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<float> vregBiasFp32;
                RegTensor<int32_t> vregIndex;
                RegTensor<float> vregSoftplusResult;
                RegTensor<float> vregOne;
                RegTensor<float> vregInFp32;
                RegTensor<float> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));

                __local_mem__ T *inputAddr = (__local_mem__ T *)xInLocalTensor.GetPhyAddr();
                __local_mem__ float *softplusOutAddr = (__local_mem__ float *)xSoftplusTensor.GetPhyAddr();
                __local_mem__ int32_t *indexOutAddr = (__local_mem__ int32_t *)indexTensor.GetPhyAddr();
                __local_mem__ float *addBiasOutAddr = (__local_mem__ float *)xBiasTensor.GetPhyAddr();

                // sigmoid
                MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::ZEROING, float>(vregOne, static_cast<float>(1), preg0);
                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<float>(size);
                    LoadOneTensorForDtypeT<T>(inputAddr, vregInFp32, preg0, i * VL_FLOAT_SIZE);
                    MicroAPI::Exp(vreg1, vregInFp32, preg0);
                    MicroAPI::Adds(vreg2, vreg1, static_cast<float>(1), preg0);
                    MicroAPI::Ln(vreg3, vreg2, preg0);
                    MicroAPI::Sqrt(vregSoftplusResult, vreg3, preg0);
                    // 使用Arange生成排序索引, 起始值为i乘每个循环的veclen
                    MicroAPI::Arange(vregIndex, static_cast<int32_t>(i * VL_FLOAT_SIZE));
                    MicroAPI::DataCopy(softplusOutAddr + i * VL_FLOAT_SIZE, vregSoftplusResult, preg0);
                    MicroAPI::DataCopy(addBiasOutAddr + i * VL_FLOAT_SIZE, vregSoftplusResult, preg0);
                    MicroAPI::DataCopy(indexOutAddr + i * VL_FLOAT_SIZE, vregIndex, preg0);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();

                // pad inf
                RegTensor<float> vregPad;
                MicroAPI::UnalignReg u0;
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    auto padUbAddr = addBiasOutAddr + i * perGroupExpertCountAlign0 + perGroupExpertCount0;
                    MicroAPI::DataCopyUnAlign(padUbAddr, vregPad, u0, perGroupExpertCountAlign0 - perGroupExpertCount0);
                    MicroAPI::DataCopyUnAlignPost(padUbAddr, u0, 0);
                }
            }
        }
        xInQueue_.FreeTensor(xInLocalTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::ComputeX()
    {
        if (tilingData_->normType == 0)
        {
            ComputeSoftmax();
            return;
        }
        if (tilingData_->normType == 1)
        {
            ComputeSigmoid();
            return;
        }
        ComputeSoftplus();
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::CopyOutXNorm(int64_t progress)
    {
        if (tilingData_->outFlag == 0)
        {
            return;
        }
        LocalTensor<float> outOutTensor = outOutQueue_.AllocTensor<float>();
        LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
        DataCopy(outOutTensor, xSigmoidTensor, expertCountAlign_);
        outOutQueue_.EnQue<float>(outOutTensor);
        outOutTensor = outOutQueue_.DeQue<float>();
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(groupCount_), static_cast<uint32_t>(perGroupExpertCount_ * sizeof(float)),
            static_cast<uint32_t>((perGroupExpertCountAlign_ - perGroupExpertCount_) * sizeof(float) / BLOCK_BYTES), 0, 0};
        DataCopyPad(outGm_[progress * expertCount_], outOutTensor, dataCopyParams);
        outOutQueue_.FreeTensor(outOutTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::SortInGroup()
    {
        LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
        LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>(); // 组内排序的结果, 后续归并需要
        LocalTensor<float> tmpLocal = finalSortBuffer_.Get<float>();

        if (perGroupExpertCountAlign_ == ONE_REPEAT_SORT_NUM)
        {
            Sort32(sortedInGroupTensor, xBiasTensor, indexTensor, groupCount_);
        }
        else
        {
            for (uint16_t i = 0; i < groupCount_; i++)
            {
                Sort<float, true>(sortedInGroupTensor[i * perGroupExpertCountAlign_ * CONSTANT_TWO],
                                  xBiasTensor[i * perGroupExpertCountAlign_], indexTensor[i * perGroupExpertCountAlign_],
                                  tmpLocal, perGroupExpertCountAlign_ / ONE_REPEAT_SORT_NUM);
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::TopKCompute()
    {
        LocalTensor<float> xBiasTensor = xBiasBuf_.Get<float>();
        LocalTensor<float> sortedInGroupTensor = sortedInGroupBuf_.Get<float>(); // 组内排序的结果, 后续归并需要
        LocalTensor<float> tmpLocal = finalSortBuffer_.Get<float>();

        Sort<float, true>(sortedInGroupTensor, xBiasTensor, indexTensor, tmpLocal,
                          perGroupExpertCountAlign_ * groupCount_ / ONE_REPEAT_SORT_NUM);

        LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
        LocalTensor<T> yTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();

        LocalTensor<int32_t> sortedInGroupTensorCast = sortedInGroupTensor.template ReinterpretCast<int32_t>();

        int32_t expertIdxPad = perGroupExpertCountAlign_ - perGroupExpertCount_;
        if (k_ <= VL_FLOAT_SIZE)
        {
            if (expertIdxPad != 0)
            {
                TopKGatherWithSmallKNotAlignE(xSigmoidTensor, sortedInGroupTensorCast, expertIdxTensor, yTensor, k_, eps_,
                                              routedScalingFactor_, expertIdxPad, perGroupExpertCountAlign_);
            }
            else
            {
                TopKGatherWithSmallKAlignE(xSigmoidTensor, sortedInGroupTensorCast, expertIdxTensor, yTensor, k_, eps_,
                                           routedScalingFactor_);
            }
        }
        else
        {
            if (expertIdxPad != 0)
            {
                TopKGatherWithLargeKNotAlignE(xSigmoidTensor, sortedInGroupTensorCast, expertIdxTensor, yTensor, k_, eps_,
                                              routedScalingFactor_, expertIdxPad, perGroupExpertCountAlign_);
            }
            else
            {
                TopKGatherWithLargeKAlignE(xSigmoidTensor, sortedInGroupTensorCast, expertIdxTensor, yTensor, k_, eps_,
                                           routedScalingFactor_);
            }
        }

        yOutQueue_.EnQue(yTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void
    MoeGatingTopKHashRegbase<T, U1, U2>::HashGatherWithSmallKAlignE(LocalTensor<float> xSigmoidTensor,
                                                        LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                        float eps, float routedScalingFactor)
    {
        __local_mem__ float *inputAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
        __local_mem__ uint32_t *expertIdxAddr = (__local_mem__ uint32_t *)expertIdxTensor.GetPhyAddr();
        __local_mem__ T *outputAddr = (__local_mem__ T *)yTensor.GetPhyAddr();

        if (tilingData_->normType == 0)
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                MicroAPI::DataCopy(vreg1, expertIdxAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::Muls(vreg2, vreg2, routedScalingFactor, preg0);
                StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg0, 0);
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                RegTensor<float> vreg4;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                MicroAPI::DataCopy(vreg1, expertIdxAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::ReduceSum(vreg3, vreg2, preg0);
                MicroAPI::Adds(vreg3, vreg3, eps, preg0);
                MicroAPI::Duplicate(vreg4, vreg3, preg0);
                MicroAPI::Div(vreg4, vreg2, vreg4, preg0);
                MicroAPI::Muls(vreg4, vreg4, routedScalingFactor, preg0);
                StoreOneTensorForDtypeT<T>(outputAddr, vreg4, preg0, 0);
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void
    MoeGatingTopKHashRegbase<T, U1, U2>::HashGatherWithSmallKNotAlignE(LocalTensor<float> xSigmoidTensor,
                                                           LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                           float eps, float routedScalingFactor, int32_t expertIdxPad,
                                                           int32_t perGroupExpertCountAlign)
    {
        __local_mem__ float *inputAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
        __local_mem__ uint32_t *expertIdxAddr = (__local_mem__ uint32_t *)expertIdxTensor.GetPhyAddr();
        __local_mem__ T *outputAddr = (__local_mem__ T *)yTensor.GetPhyAddr();

        if (tilingData_->normType == 0)
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<uint32_t> vregAlign;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(k);

                MicroAPI::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

                MicroAPI::DataCopy(vreg1, expertIdxAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::Muls(vreg2, vreg2, routedScalingFactor, preg0);

                // compute expertIdx: id = id - floor_div(id, perGroupExpertCountAlign) * pad
                MicroAPI::Div(vregAlign, vreg1, vregAlign, preg0);
                MicroAPI::Muls(vregAlign, vregAlign, expertIdxPad, preg0);
                MicroAPI::Sub(vreg1, vreg1, vregAlign, preg0);

                StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg0, 0);
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                RegTensor<float> vreg4;
                RegTensor<uint32_t> vregAlign;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(k);

                MicroAPI::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

                MicroAPI::DataCopy(vreg1, expertIdxAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::ReduceSum(vreg3, vreg2, preg0);
                MicroAPI::Adds(vreg3, vreg3, eps, preg0);
                MicroAPI::Duplicate(vreg4, vreg3, preg0);
                MicroAPI::Div(vreg4, vreg2, vreg4, preg0);
                MicroAPI::Muls(vreg4, vreg4, routedScalingFactor, preg0);

                // compute expertIdx: id = id - floor_div(id, perGroupExpertCountAlign) * pad
                MicroAPI::Div(vregAlign, vreg1, vregAlign, preg0);
                MicroAPI::Muls(vregAlign, vregAlign, expertIdxPad, preg0);
                MicroAPI::Sub(vreg1, vreg1, vregAlign, preg0);

                StoreOneTensorForDtypeT<T>(outputAddr, vreg4, preg0, 0);
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::HashCompute(int64_t row)
    {
        LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();
        LocalTensor<U2> hashExpertId = sortedInGroupBuf_.Get<U2>();
        LocalTensor<int32_t> hashExpertIdInt32 = hashExpertId.template ReinterpretCast<int32_t>();

        U1 key = inputIdsGm_.GetValue(row);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(U2)), 0, 0, 0};
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<U2>(0)};
        DataCopyPad(hashExpertId, tid2eidGm_[key * k_], dataCopyParams, dataCopyPadParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        if constexpr (IsSameType<U2, int32_t>::value) {
            DataCopy(expertIdxTensor, hashExpertId, Align(k_, sizeof(int32_t)));
        } else {
            Cast(hashExpertIdInt32, hashExpertId, RoundMode::CAST_NONE, Align(k_, sizeof(U2)));
            PipeBarrier<PIPE_V>();
            DataCopy(expertIdxTensor, hashExpertIdInt32, Align(k_, sizeof(int32_t)));
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
        LocalTensor<T> yTensor = yOutQueue_.AllocTensor<T>();
        LocalTensor<int32_t> sortedInGroupTensorCast = sortedInGroupTensor.template ReinterpretCast<int32_t>();
        int32_t expertIdxPad = perGroupExpertCountAlign_ - perGroupExpertCount_;
        // 当前只支持k小于64场景
        if (k_ <= VL_FLOAT_SIZE)
        {
            if (expertIdxPad != 0)
            {
                HashGatherWithSmallKNotAlignE(xSigmoidTensor, expertIdxTensor, yTensor, k_, eps_,
                                              routedScalingFactor_, expertIdxPad, perGroupExpertCountAlign_);
            }
            else
            {
                HashGatherWithSmallKAlignE(xSigmoidTensor, expertIdxTensor, yTensor, k_, eps_,
                                           routedScalingFactor_);
            }
        }
        yOutQueue_.EnQue(yTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::SelectTopKGroupIndex()
    {
        sortedInGroupTensor = sortedInGroupBuf_.Get<float>();
        LocalTensor<float> top2InGroupTensor = groupBuf_.Get<float>();
        LocalTensor<float> tmpLocal = xBiasBuf_.Get<float>();
        // 排序，将kgroup选出来
        sortedGroupTensor = sortedGroupBuf_.Get<float>();

        uint16_t groupCount0 = groupCount_;
        uint32_t perGroupExpertCountAlign0 = perGroupExpertCountAlign_;
        int32_t groupCountNumAlign = (groupCount_ + 31) / 32 * 32;
        uint32_t padNegInfNum = groupCountNumAlign - groupCount_;
        if (groupSelectMode_ == 1)
        {
            __VEC_SCOPE__
            {
                RegTensor<float> vreg0;
                RegTensor<float> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vregPad;

                MicroAPI::UnalignReg u0;
                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();

                __local_mem__ float *inputAddr = (__local_mem__ float *)sortedInGroupTensor.GetPhyAddr();
                __local_mem__ float *outputAddr = (__local_mem__ float *)top2InGroupTensor.GetPhyAddr();

                // pair reduce sum
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                        vreg0, vreg1, inputAddr + i * perGroupExpertCountAlign0 * 2);
                    MicroAPI::PairReduceSum(vreg2, vreg0, preg0);
                    MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputAddr, vreg2, u0, 1);
                }
                MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputAddr, vregPad, u0,
                                                                                          padNegInfNum);
                MicroAPI::DataCopyUnAlignPost(outputAddr, u0, 0);
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<float> vreg0;
                RegTensor<float> vreg1;
                RegTensor<float> vregPad;

                MicroAPI::UnalignReg u0;

                __local_mem__ float *inputAddr = (__local_mem__ float *)sortedInGroupTensor.GetPhyAddr();
                __local_mem__ float *outputAddr = (__local_mem__ float *)top2InGroupTensor.GetPhyAddr();

                // max
                MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
                for (uint16_t i = 0; i < groupCount0; i++)
                {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_DINTLV_B32>(
                        vreg0, vreg1, inputAddr + i * perGroupExpertCountAlign0 * 2);
                    MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputAddr, vreg0, u0, 1);
                }
                MicroAPI::DataCopyUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(outputAddr, vregPad, u0,
                                                                                          padNegInfNum);
                MicroAPI::DataCopyUnAlignPost(outputAddr, u0, 0);
            }
        }

        Sort<float, true>(sortedGroupTensor, top2InGroupTensor, indexTensor, tmpLocal,
                          groupCountNumAlign / ONE_REPEAT_SORT_NUM);

        uint32_t size = groupCountNumAlign;
        int32_t kGroup0 = kGroup_;
        int32_t kGroupNumAlign = (kGroup0 + 31) / 32 * 32;
        uint32_t padkGroupNum = kGroupNumAlign - kGroup0;
        __VEC_SCOPE__
        {
            RegTensor<int32_t> vreg0;
            RegTensor<int32_t> vreg1;
            RegTensor<float> vregPad;
            MicroAPI::UnalignReg u0;

            __local_mem__ int32_t *inputAddr = (__local_mem__ int32_t *)sortedGroupTensor.GetPhyAddr();
            __local_mem__ int32_t *outputAddr = (__local_mem__ int32_t *)top2InGroupTensor.GetPhyAddr();

            MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<int32_t>();
            uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(size, VL_FLOAT_SIZE));

            for (uint16_t i = 0; i < vfLoopNum; i++)
            {
                preg0 = MicroAPI::UpdateMask<int32_t>(size);
                MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                 inputAddr + i * 2 * VL_FLOAT_SIZE);
                MicroAPI::DataCopy(outputAddr + i * VL_FLOAT_SIZE, vreg1, preg0);
            }
            MicroAPI::Duplicate(vregPad, *((float *)&MIN_FP32));
            outputAddr = outputAddr + kGroup0;
            MicroAPI::DataCopyUnAlign(outputAddr, (RegTensor<int32_t> &)vregPad, u0, padkGroupNum);
            MicroAPI::DataCopyUnAlignPost(outputAddr, u0, 0);
        }

        Sort<float, true>(sortedGroupTensor, top2InGroupTensor, indexTensor, tmpLocal,
                          kGroupNumAlign / ONE_REPEAT_SORT_NUM);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::FinalSortByKGroup()
    {
        mrgSortTensor = finalSortBuffer_.Get<float>();
        LocalTensor<uint32_t> tmpLocal = sortedGroupTensor.template ReinterpretCast<uint32_t>();
        uint32_t offset[MRG_SORT_ELEMENT_LEN] = {0, 0, 0, 0};

        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        uint16_t lenArr[CONSTANT_FOUR] = {
            static_cast<uint16_t>(perGroupExpertCount_), static_cast<uint16_t>(perGroupExpertCount_),
            static_cast<uint16_t>(perGroupExpertCount_), static_cast<uint16_t>(perGroupExpertCount_)};
        MrgSort4Info params{lenArr, false, 0b1111, 1};
        MrgSortSrcList<float> srcList;

        for (int32_t i = kGroup_ - 1; i >= 0; i -= CONSTANT_FOUR)
        {
#if defined(__CCE_KT_TEST__)
            int32_t mrgLen = std::min(i + 1, CONSTANT_FOUR);
#else
            int32_t mrgLen = min(i + 1, CONSTANT_FOUR);
#endif
            if (mrgLen > 1)
            {
                if (mrgLen == CONSTANT_FOUR)
                {
                    offset[0] = tmpLocal.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
                    offset[1] = tmpLocal.GetValue((i - 1) * 2) * perGroupExpertCountAlign_ * 2;
                    offset[CONSTANT_TWO] = tmpLocal.GetValue((i - 2) * 2) * perGroupExpertCountAlign_ * 2;
                    offset[CONSTANT_THREE] = tmpLocal.GetValue((i - 3) * 2) * perGroupExpertCountAlign_ * 2;
                }
                else if (mrgLen == CONSTANT_THREE)
                {
                    offset[0] = tmpLocal.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
                    offset[1] = tmpLocal.GetValue((i - 1) * 2) * perGroupExpertCountAlign_ * 2;
                    offset[CONSTANT_TWO] = tmpLocal.GetValue((i - 2) * 2) * perGroupExpertCountAlign_ * 2;
                    offset[CONSTANT_THREE] = 0;
                    params.elementLengths[CONSTANT_THREE] = 0;
                    params.validBit = 0b111;
                }
                else
                {
                    offset[0] = tmpLocal.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
                    offset[1] = tmpLocal.GetValue((i - 1) * 2) * perGroupExpertCountAlign_ * 2;
                    offset[CONSTANT_TWO] = 0;
                    offset[CONSTANT_THREE] = 0;
                    params.elementLengths[CONSTANT_TWO] = 0;
                    params.elementLengths[CONSTANT_THREE] = 0;
                    params.validBit = 0b11;
                }
                srcList.src1 = sortedInGroupTensor[offset[0]];
                srcList.src2 = sortedInGroupTensor[offset[1]];
                srcList.src3 = sortedInGroupTensor[offset[CONSTANT_TWO]];
                srcList.src4 = sortedInGroupTensor[offset[CONSTANT_THREE]];
                MrgSort(mrgSortTensor[(kGroup_ - 1 - i) * perGroupExpertCountAlign_ * 2], srcList, params);
            }
            else
            {
                offset[0] = tmpLocal.GetValue(i * 2) * perGroupExpertCountAlign_ * 2;
                DataCopy(mrgSortTensor[(kGroup_ - 1 - i) * perGroupExpertCountAlign_ * 2], sortedInGroupTensor[offset[0]],
                         perGroupExpertCountAlign_ * 2);
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::FinalSortAfterKGroup()
    {
        LocalTensor<float> srcTensor;
        LocalTensor<float> dstTensor;
        int32_t sortedBaseRow = CONSTANT_FOUR;
        for (int32_t i = 0; i < tilingData_->vmsCount; i++)
        {
            if (i % CONSTANT_TWO == 0)
            {
                srcTensor = mrgSortTensor;
                dstTensor = sortedInGroupTensor;
            }
            else
            {
                srcTensor = sortedInGroupTensor;
                dstTensor = mrgSortTensor;
            }
            int32_t nextBaseRow = sortedBaseRow * CONSTANT_FOUR;
            int32_t quotient = kGroup_ / nextBaseRow;
            int32_t remainder = kGroup_ - quotient * nextBaseRow;

            if (quotient > 0)
            {
                MrgSort4Info params;
                MrgSortSrcList<float> srcList;
                params.ifExhaustedSuspension = false;
                params.elementLengths[0] = perGroupExpertCount_ * sortedBaseRow;
                params.elementLengths[1] = perGroupExpertCount_ * sortedBaseRow;
                params.elementLengths[CONSTANT_TWO] = perGroupExpertCount_ * sortedBaseRow;
                params.elementLengths[CONSTANT_THREE] = perGroupExpertCount_ * sortedBaseRow;
                params.validBit = 0b1111;
                params.repeatTimes = 1;
                for (int j = 0; j < quotient; j++)
                {
                    srcList.src1 = srcTensor[perGroupExpertCountAlign_ * sortedBaseRow * 8 * j];
                    srcList.src2 = srcTensor[perGroupExpertCountAlign_ * sortedBaseRow * (8 * j + 2)];
                    srcList.src3 = srcTensor[perGroupExpertCountAlign_ * sortedBaseRow * (8 * j + 4)];
                    srcList.src4 = srcTensor[perGroupExpertCountAlign_ * sortedBaseRow * (8 * j + 6)];
                    MrgSort(dstTensor[perGroupExpertCountAlign_ * sortedBaseRow * 8 * j], srcList, params);
                }
            }
            if (remainder > 0)
            {
                int32_t baseOffset = quotient * nextBaseRow * perGroupExpertCountAlign_ * 2;
                int32_t mrgLen = CeilDiv(remainder, sortedBaseRow);
                int32_t tailRow = remainder - (mrgLen - 1) * sortedBaseRow;
                if (mrgLen > 1)
                {
                    MrgSort4Info params;
                    MrgSortSrcList<float> srcList;
                    params.repeatTimes = 1;
                    params.ifExhaustedSuspension = false;
                    params.elementLengths[0] = perGroupExpertCount_ * sortedBaseRow;
                    params.elementLengths[1] = perGroupExpertCount_ * sortedBaseRow;
                    params.elementLengths[CONSTANT_TWO] = perGroupExpertCount_ * sortedBaseRow;
                    params.elementLengths[CONSTANT_THREE] = perGroupExpertCount_ * sortedBaseRow;
                    srcList.src1 = srcTensor[baseOffset];
                    srcList.src2 = srcTensor[baseOffset + perGroupExpertCountAlign_ * sortedBaseRow * 2];
                    if (mrgLen == CONSTANT_FOUR)
                    {
                        srcList.src3 = srcTensor[baseOffset + perGroupExpertCountAlign_ * sortedBaseRow * 2 * 2];
                        srcList.src4 = srcTensor[baseOffset + perGroupExpertCountAlign_ * sortedBaseRow * 2 * 3];
                        params.elementLengths[CONSTANT_THREE] = perGroupExpertCount_ * tailRow;
                        params.validBit = 0b1111;
                    }
                    else if (mrgLen == CONSTANT_THREE)
                    {
                        srcList.src3 = srcTensor[baseOffset + perGroupExpertCountAlign_ * sortedBaseRow * 2 * 2];
                        params.elementLengths[CONSTANT_TWO] = perGroupExpertCount_ * tailRow;
                        params.elementLengths[CONSTANT_THREE] = 0;
                        params.validBit = 0b111;
                    }
                    else
                    {
                        params.elementLengths[1] = perGroupExpertCount_ * tailRow;
                        params.elementLengths[CONSTANT_TWO] = 0;
                        params.elementLengths[CONSTANT_THREE] = 0;
                        params.validBit = 0b11;
                    }
                    MrgSort(dstTensor[baseOffset], srcList, params);
                }
                else
                {
                    DataCopy(dstTensor[baseOffset], srcTensor[baseOffset], tailRow * perGroupExpertCountAlign_ * 2);
                }
            }

            sortedBaseRow = nextBaseRow;
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void
    MoeGatingTopKHashRegbase<T, U1, U2>::TopKGatherWithSmallKAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                        LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                        float eps, float routedScalingFactor)
    {
        __local_mem__ float *inputAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
        __local_mem__ uint32_t *mrgSortAddr = (__local_mem__ uint32_t *)mrgSortTensor.GetPhyAddr();
        __local_mem__ uint32_t *expertIdxAddr = (__local_mem__ uint32_t *)expertIdxTensor.GetPhyAddr();
        __local_mem__ T *outputAddr = (__local_mem__ T *)yTensor.GetPhyAddr();

        if (tilingData_->normType == 0)
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1, mrgSortAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::Muls(vreg2, vreg2, routedScalingFactor, preg0);
                StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg0, 0);
                MicroAPI::DataCopy(expertIdxAddr, vreg1, preg0);
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                RegTensor<float> vreg4;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1, mrgSortAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::ReduceSum(vreg3, vreg2, preg0);
                MicroAPI::Adds(vreg3, vreg3, eps, preg0);
                MicroAPI::Duplicate(vreg4, vreg3, preg0);
                MicroAPI::Div(vreg4, vreg2, vreg4, preg0);
                MicroAPI::Muls(vreg4, vreg4, routedScalingFactor, preg0);
                StoreOneTensorForDtypeT<T>(outputAddr, vreg4, preg0, 0);
                MicroAPI::DataCopy(expertIdxAddr, vreg1, preg0);
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void
    MoeGatingTopKHashRegbase<T, U1, U2>::TopKGatherWithLargeKAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                        LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                        float eps, float routedScalingFactor)
    {
        uint32_t k1 = k_;
        __local_mem__ float *inputAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
        __local_mem__ uint32_t *mrgSortAddr = (__local_mem__ uint32_t *)mrgSortTensor.GetPhyAddr();
        __local_mem__ uint32_t *expertIdxAddr = (__local_mem__ uint32_t *)expertIdxTensor.GetPhyAddr();
        __local_mem__ T *outputAddr = (__local_mem__ T *)yTensor.GetPhyAddr();

        if (tilingData_->normType == 0)
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vregSum;

                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<float>();
                MicroAPI::Duplicate(vregSum, static_cast<float>(0), preg0);
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::Duplicate(vreg2, static_cast<float>(0), preg1);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                    MicroAPI::Add(vregSum, vregSum, vreg2, preg1);
                }

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg1 = MicroAPI::UpdateMask<uint32_t>(k1);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg1);
                    MicroAPI::Muls(vreg2, vreg2, routedScalingFactor, preg1);
                    StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg1, i * VL_FLOAT_SIZE);
                    MicroAPI::DataCopy(expertIdxAddr + i * VL_FLOAT_SIZE, vreg1, preg1);
                }
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                RegTensor<float> vreg4;
                RegTensor<float> vreg5;
                RegTensor<float> vregSum;

                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<float>();

                MicroAPI::Duplicate(vregSum, static_cast<float>(0), preg0);
                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::Duplicate(vreg2, static_cast<float>(0), preg1);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                    MicroAPI::Add(vregSum, vregSum, vreg2, preg1);
                }
                MicroAPI::ReduceSum(vregSum, vregSum, preg1);
                MicroAPI::Adds(vregSum, vregSum, eps, preg1);
                MicroAPI::Duplicate(vreg4, vregSum, preg1);
                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg1 = MicroAPI::UpdateMask<uint32_t>(k1);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg1);
                    MicroAPI::Div(vreg5, vreg2, vreg4, preg1);
                    MicroAPI::Muls(vreg5, vreg5, routedScalingFactor, preg1);
                    StoreOneTensorForDtypeT<T>(outputAddr, vreg5, preg1, i * VL_FLOAT_SIZE);
                    MicroAPI::DataCopy(expertIdxAddr + i * VL_FLOAT_SIZE, vreg1, preg1);
                }
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void
    MoeGatingTopKHashRegbase<T, U1, U2>::TopKGatherWithSmallKNotAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                           LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                           float eps, float routedScalingFactor, int32_t expertIdxPad,
                                                           int32_t perGroupExpertCountAlign)
    {
        __local_mem__ float *inputAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
        __local_mem__ uint32_t *mrgSortAddr = (__local_mem__ uint32_t *)mrgSortTensor.GetPhyAddr();
        __local_mem__ uint32_t *expertIdxAddr = (__local_mem__ uint32_t *)expertIdxTensor.GetPhyAddr();
        __local_mem__ T *outputAddr = (__local_mem__ T *)yTensor.GetPhyAddr();

        if (tilingData_->normType == 0)
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<uint32_t> vregAlign;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(k);

                MicroAPI::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

                MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1, mrgSortAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::Muls(vreg2, vreg2, routedScalingFactor, preg0);

                // compute expertIdx: id = id - floor_div(id, perGroupExpertCountAlign) * pad
                MicroAPI::Div(vregAlign, vreg1, vregAlign, preg0);
                MicroAPI::Muls(vregAlign, vregAlign, expertIdxPad, preg0);
                MicroAPI::Sub(vreg1, vreg1, vregAlign, preg0);

                StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg0, 0);
                MicroAPI::DataCopy(expertIdxAddr, vreg1, preg0);
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                RegTensor<float> vreg4;
                RegTensor<uint32_t> vregAlign;

                MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<float>(k);

                MicroAPI::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

                MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1, mrgSortAddr);
                MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                MicroAPI::ReduceSum(vreg3, vreg2, preg0);
                MicroAPI::Adds(vreg3, vreg3, eps, preg0);
                MicroAPI::Duplicate(vreg4, vreg3, preg0);
                MicroAPI::Div(vreg4, vreg2, vreg4, preg0);
                MicroAPI::Muls(vreg4, vreg4, routedScalingFactor, preg0);

                // compute expertIdx: id = id - floor_div(id, perGroupExpertCountAlign) * pad
                MicroAPI::Div(vregAlign, vreg1, vregAlign, preg0);
                MicroAPI::Muls(vregAlign, vregAlign, expertIdxPad, preg0);
                MicroAPI::Sub(vreg1, vreg1, vregAlign, preg0);

                StoreOneTensorForDtypeT<T>(outputAddr, vreg4, preg0, 0);
                MicroAPI::DataCopy(expertIdxAddr, vreg1, preg0);
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void
    MoeGatingTopKHashRegbase<T, U1, U2>::TopKGatherWithLargeKNotAlignE(LocalTensor<float> xSigmoidTensor, LocalTensor<int32_t> mrgSortTensor,
                                                           LocalTensor<int32_t> expertIdxTensor, LocalTensor<T> yTensor, uint32_t k,
                                                           float eps, float routedScalingFactor, int32_t expertIdxPad,
                                                           int32_t perGroupExpertCountAlign)
    {
        uint32_t k1 = k_;
        __local_mem__ float *inputAddr = (__local_mem__ float *)xSigmoidTensor.GetPhyAddr();
        __local_mem__ uint32_t *mrgSortAddr = (__local_mem__ uint32_t *)mrgSortTensor.GetPhyAddr();
        __local_mem__ uint32_t *expertIdxAddr = (__local_mem__ uint32_t *)expertIdxTensor.GetPhyAddr();
        __local_mem__ T *outputAddr = (__local_mem__ T *)yTensor.GetPhyAddr();

        if (tilingData_->normType == 0)
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<uint32_t> vreg3;
                RegTensor<float> vregSum;
                RegTensor<uint32_t> vregAlign;

                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<float>();

                MicroAPI::Duplicate(vregSum, static_cast<float>(0), preg0);
                MicroAPI::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::Duplicate(vreg2, static_cast<float>(0), preg1);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                    MicroAPI::Add(vregSum, vregSum, vreg2, preg1);
                }

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg1 = MicroAPI::UpdateMask<uint32_t>(k1);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg1);
                    MicroAPI::Muls(vreg2, vreg2, routedScalingFactor, preg1);

                    // compute expertIdx: id = id - floor_div(id, perGroupExpertCountAlign) * pad
                    MicroAPI::Div(vreg3, vreg1, vregAlign, preg1);
                    MicroAPI::Muls(vreg3, vreg3, expertIdxPad, preg1);
                    MicroAPI::Sub(vreg1, vreg1, vreg3, preg1);
                    StoreOneTensorForDtypeT<T>(outputAddr, vreg2, preg1, i * VL_FLOAT_SIZE);
                    MicroAPI::DataCopy(expertIdxAddr + i * VL_FLOAT_SIZE, vreg1, preg1);
                }
            }
        }
        else
        {
            __VEC_SCOPE__
            {
                RegTensor<uint32_t> vreg0;
                RegTensor<uint32_t> vreg1;
                RegTensor<float> vreg2;
                RegTensor<float> vreg3;
                RegTensor<float> vreg4;
                RegTensor<uint32_t> vreg5;
                RegTensor<float> vreg6;
                RegTensor<float> vregSum;
                RegTensor<uint32_t> vregAlign;

                MicroAPI::MaskReg preg0 = MicroAPI::CreateMask<float>();
                MicroAPI::MaskReg preg1 = MicroAPI::CreateMask<float>();

                MicroAPI::Duplicate(vregSum, static_cast<float>(0), preg0);
                MicroAPI::Duplicate(vregAlign, perGroupExpertCountAlign, preg0);

                uint16_t vfLoopNum = static_cast<uint16_t>(CeilDiv(k, VL_FLOAT_SIZE));

                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg0 = MicroAPI::UpdateMask<uint32_t>(k);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::Duplicate(vreg2, static_cast<float>(0), preg1);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg0);
                    MicroAPI::Add(vregSum, vregSum, vreg2, preg1);
                }
                MicroAPI::ReduceSum(vregSum, vregSum, preg1);
                MicroAPI::Adds(vregSum, vregSum, eps, preg1);
                MicroAPI::Duplicate(vreg4, vregSum, preg1);
                for (uint16_t i = 0; i < vfLoopNum; i++)
                {
                    preg1 = MicroAPI::UpdateMask<uint32_t>(k1);
                    MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                                      mrgSortAddr + i * 2 * VL_FLOAT_SIZE);
                    MicroAPI::DataCopyGather(vreg2, inputAddr, vreg1, preg1);
                    MicroAPI::Div(vreg6, vreg2, vreg4, preg1);
                    MicroAPI::Muls(vreg6, vreg6, routedScalingFactor, preg1);

                    // compute expertIdx: id = id - floor_div(id, perGroupExpertCountAlign) * pad
                    MicroAPI::Div(vreg5, vreg1, vregAlign, preg1);
                    MicroAPI::Muls(vreg5, vreg5, expertIdxPad, preg1);
                    MicroAPI::Sub(vreg1, vreg1, vreg5, preg1);
                    StoreOneTensorForDtypeT<T>(outputAddr, vreg6, preg1, i * VL_FLOAT_SIZE);
                    MicroAPI::DataCopy(expertIdxAddr + i * VL_FLOAT_SIZE, vreg1, preg1);
                }
            }
        }
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::SelectTopKExpertScore()
    {
        LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> mrgSortTensor = finalSortBuffer_.Get<int32_t>();

        if (tilingData_->vmsCount % CONSTANT_TWO == 1)
        {
            mrgSortTensor = sortedInGroupTensor.ReinterpretCast<int32_t>();
        }

        LocalTensor<float> xSigmoidTensor = xSigmoidBuf_.Get<float>();
        LocalTensor<T> yTensor = yOutQueue_.AllocTensor<T>();

        int32_t expertIdxPad = perGroupExpertCountAlign_ - perGroupExpertCount_;
        if (k_ <= VL_FLOAT_SIZE)
        {
            if (expertIdxPad != 0)
            {
                TopKGatherWithSmallKNotAlignE(xSigmoidTensor, mrgSortTensor, expertIdxTensor, yTensor, k_, eps_, routedScalingFactor_,
                                              expertIdxPad, perGroupExpertCountAlign_);
            }
            else
            {
                TopKGatherWithSmallKAlignE(xSigmoidTensor, mrgSortTensor, expertIdxTensor, yTensor, k_, eps_, routedScalingFactor_);
            }
        }
        else
        {
            if (expertIdxPad != 0)
            {
                TopKGatherWithLargeKNotAlignE(xSigmoidTensor, mrgSortTensor, expertIdxTensor, yTensor, k_, eps_, routedScalingFactor_,
                                              expertIdxPad, perGroupExpertCountAlign_);
            }
            else
            {
                TopKGatherWithLargeKAlignE(xSigmoidTensor, mrgSortTensor, expertIdxTensor, yTensor, k_, eps_, routedScalingFactor_);
            }
        }

        yOutQueue_.EnQue(yTensor);
        expertIdxOutQueue_.EnQue<int32_t>(expertIdxTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::CopyOut(int64_t row)
    {
        LocalTensor<T> yOutTensor = yOutQueue_.DeQue<T>();
        LocalTensor<int32_t> expertIdxTensor = expertIdxOutQueue_.DeQue<int32_t>();
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(k_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[row * k_], yOutTensor, dataCopyParams);

        dataCopyParams.blockLen = k_ * sizeof(int32_t);
        DataCopyPad(expertIdxGm_[row * k_], expertIdxTensor, dataCopyParams);

        expertIdxOutQueue_.FreeTensor(expertIdxTensor);
        yOutQueue_.FreeTensor(yOutTensor);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::Init(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid, GM_ADDR y, GM_ADDR expertIdx, GM_ADDR out,
                                                         GM_ADDR workspace,
                                                         const MoeGatingTopKHashRegbaseTilingData *tilingData, TPipe *tPipe)
    {
        tilingData_ = tilingData;
        pipe_ = tPipe;
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ == GetBlockNum() - 1)
        {
            curCoreRowCount_ = tilingData_->lastCoreRowCount;
        }
        else
        {
            curCoreRowCount_ = tilingData_->perCoreRowCount;
        }
        expertCount_ = tilingData_->expertCount;
        k_ = tilingData_->k;
        kGroup_ = tilingData_->kGroup;
        groupCount_ = tilingData_->groupCount;
        perGroupExpertCount_ = tilingData_->perGroupExpertCount;
        perGroupExpertCountAlign_ = tilingData_->perGroupExpertCountAlign;
        expertCountAlign_ = Align(perGroupExpertCountAlign_ * groupCount_, sizeof(float));
        routedScalingFactor_ = tilingData_->routedScalingFactor;
        groupSelectMode_ = tilingData_->groupSelectMode;
        eps_ = tilingData_->eps;
        hashFlag_ = tilingData_->hashFlag == 1;

        // init input gm buf
        xGm_.SetGlobalBuffer((__gm__ T *)x + tilingData_->perCoreRowCount * expertCount_ * blockIdx_, expertCount_);
        if (bias != nullptr)
        {
            hasBias_ = true;
            biasGm_.SetGlobalBuffer((__gm__ T *)bias, expertCount_);
        }
        yGm_.SetGlobalBuffer((__gm__ T *)y + tilingData_->perCoreRowCount * k_ * blockIdx_, k_);
        expertIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expertIdx + tilingData_->perCoreRowCount * k_ * blockIdx_, k_);
        outGm_.SetGlobalBuffer((__gm__ float *)out + tilingData_->perCoreRowCount * expertCount_ * blockIdx_, expertCount_);

        inputIdsGm_.SetGlobalBuffer((__gm__ U1 *)inputIds);
        tid2eidGm_.SetGlobalBuffer((__gm__ U2 *)tid2eid);

        // init queue
        int32_t expertGroupAlign = groupCount_ * perGroupExpertCountAlign_;
        int32_t groupAlign = static_cast<int32_t>(CeilAlign(groupCount_, ONE_REPEAT_SORT_NUM));
        pipe_->InitBuffer(xInQueue_, CONSTANT_TWO, expertGroupAlign * sizeof(float) * (sizeof(float) / sizeof(T)));
        pipe_->InitBuffer(yOutQueue_, CONSTANT_TWO, AlignBytes(k_, sizeof(T)));
        pipe_->InitBuffer(expertIdxOutQueue_, CONSTANT_TWO, AlignBytes(k_, sizeof(int32_t)));
        pipe_->InitBuffer(outOutQueue_, CONSTANT_TWO, expertGroupAlign * sizeof(float));

        pipe_->InitBuffer(biasBuf_, expertGroupAlign * sizeof(T));
        pipe_->InitBuffer(xSigmoidBuf_, expertGroupAlign * sizeof(float));
        pipe_->InitBuffer(xBiasBuf_, expertGroupAlign * sizeof(float));
        pipe_->InitBuffer(indexBuffer_, expertGroupAlign * sizeof(int32_t));
        pipe_->InitBuffer(sortedInGroupBuf_, expertGroupAlign * sizeof(U2) * 2);
        pipe_->InitBuffer(finalSortBuffer_, expertGroupAlign * sizeof(float) * 2);
        pipe_->InitBuffer(groupBuf_, groupAlign * sizeof(float));
        pipe_->InitBuffer(sortedGroupBuf_, groupAlign * sizeof(float) * 2);
    }

    template <typename T, typename  U1, typename U2>
    __aicore__ inline void MoeGatingTopKHashRegbase<T, U1, U2>::Process()
    {
        CopyInBias();
        if (kGroup_ == groupCount_ || groupCount_ == expertCount_)
        {
            CopyInX(0);
            for (int64_t row = 1; row < curCoreRowCount_; row++)
            {
                ComputeX();
                CopyOutXNorm(row - 1);
                CopyInX(row);
                if (hashFlag_)
                {
                    HashCompute(row - 1 + tilingData_->perCoreRowCount * blockIdx_);
                }
                else
                {
                    TopKCompute();
                }
                CopyOut(row - 1);
            }
            ComputeX();
            CopyOutXNorm(curCoreRowCount_ - 1);
            if (hashFlag_)
            {
                HashCompute(curCoreRowCount_ - 1 + tilingData_->perCoreRowCount * blockIdx_);
            }
            else
            {
                TopKCompute();
            }
            CopyOut(curCoreRowCount_ - 1);
            return;
        }

        CopyInX(0);
        for (int64_t row = 1; row < curCoreRowCount_; row++)
        {
            ComputeX();
            CopyOutXNorm(row - 1);
            SortInGroup();
            SelectTopKGroupIndex();
            CopyInX(row);
            FinalSortByKGroup();
            FinalSortAfterKGroup();
            SelectTopKExpertScore();
            CopyOut(row - 1);
        }
        ComputeX();
        CopyOutXNorm(curCoreRowCount_ - 1);
        SortInGroup();
        SelectTopKGroupIndex();
        FinalSortByKGroup();
        FinalSortAfterKGroup();
        SelectTopKExpertScore();
        CopyOut(curCoreRowCount_ - 1);
    }
} // namespace MoeGatingTopKHash
#endif // MOE_GATING_TOP_K_REGBASE_H