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
 * \file kernel_mla_prolog_split_n.h
 * \brief
 */

#ifndef KERNEL_MLA_PROLOG_SPLIT_N_H
#define KERNEL_MLA_PROLOG_SPLIT_N_H

#include "mla_prolog_comm.h"
#include "mla_prolog_vector_comm.h"
#include "service_matmul.h"
#include "service_rms_norm.h"
#include "service_dequant.h"
#include "service_dynamic_quant_qn_mul_qr.h"
#include "service_gather_sin_cos.h"
#include "service_rotary_position_embedding.h"
#include "service_scatter_cache.h"
#include "../mla_prolog_tiling_data.h"
#include "../mla_prolog_template_tiling_key.h"

namespace MlaProlog {
template <typename MLAPT>
class MlaPrologVecS1CubS2 {
public:
    static constexpr bool isPertile = MLAPT::isPertile;

    using mmInputType = typename MLAPT::mmInputType;
    using mmQnInputType = typename MLAPT::mmQnInputType;
    using mmQcQrInputType = typename MLAPT::mmQcQrInputType;
    using mmQnOutputType = typename MLAPT::mmQnOutputType;
    using mmQcQrOutputType = typename MLAPT::mmQcQrOutputType;
    using mmCkvKrOutputType = typename MLAPT::mmCkvKrOutputType;
    using mmCqOutputType = typename MLAPT::mmCqOutputType;
    using kvCacheType = typename MLAPT::kvCacheType;
    using krCacheType = typename MLAPT::krCacheType;
    using queryOutputType = typename std::conditional<isPertile, bfloat16_t, typename MLAPT::kvCacheType>::type;
    using ropeSinCosType = typename MLAPT::ropeSinCosType;
    using ropeComputType = typename MLAPT::ropeComputType;
    using ropeOutputType = typename MLAPT::ropeOutputType;
    using rmsNormGammaType = typename MLAPT::rmsNormGammaType;
    using rmsNormComputType = typename MLAPT::rmsNormComputType;
    using rmsNormCqOutputType = typename MLAPT::rmsNormCqOutputType;
    using rmsNormCkvOutputType = typename MLAPT::rmsNormCkvOutputType;
    using dequantScaleQNopeType = typename MLAPT::dequantScaleQNopeType;
    using dequantScaleQNormType = typename MLAPT::dequantScaleQNormType;
    using dequantScaleType = typename MLAPT::dequantScaleType;

    MMParams mmQnParam_;
    MMParams mmQcQrParam_;
    MMParams mmCkvKrParam_;
    MMParams mmCqParam_;

    __aicore__ inline MlaPrologVecS1CubS2(TPipe *pipe, const optiling::MlaPrologTilingData *__restrict tilingData,
                                          const optiling::MlaPrologBaseParams *__restrict baseParams)
        : pipe_(pipe), tilingData_(tilingData), baseParams_(baseParams)
    {
    }

    __aicore__ inline void Process();
    __aicore__ inline void Init(__gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq, __gm__ uint8_t *weightUqQr,
                                __gm__ uint8_t *weightUk, __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq,
                                __gm__ uint8_t *rmsnormGammaCkv, __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos,
                                __gm__ uint8_t *cacheIndex, __gm__ uint8_t *kvCache, __gm__ uint8_t *krCache,
                                __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
                                __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
                                __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *actualSeqLen,
                                __gm__ uint8_t *kNopeClipAlpha, __gm__ uint8_t *queryOut, __gm__ uint8_t *queryRopeOut,
                                __gm__ uint8_t *dequantScaleQNopeOut, __gm__ uint8_t *queryNormOut,
                                __gm__ uint8_t *dequantScaleQNormOut, __gm__ uint8_t *workspace);

private:
    __aicore__ inline void CopyGlobalParams();
    __aicore__ inline void ScaleInit(__gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                     __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
                                     __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
                                     __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *kNopeClipAlpha);
    __aicore__ inline void WorkspaceInit(__gm__ uint8_t *workspace);
    __aicore__ inline void MmParamInit();
    __aicore__ inline void MmQnParamInit();
    __aicore__ inline void MmQcQrParamInit();
    __aicore__ inline void MmCkvKrParamInit();
    __aicore__ inline void MmCqParamInit();
    __aicore__ inline void CubeBufferInit();
    __aicore__ inline void VectorBufferInit();
    __aicore__ inline void UpdateStepBatchParams(int64_t curStepBatchSize);
    __aicore__ inline void ComputeAivOffset(AivOffset &aivOffset, int64_t batchOffset);
    __aicore__ inline void ComputeAicOffset(AicOffset &aicOffset, int64_t numHeadOffset);
    __aicore__ inline void ComputeBlkScatterOffsets(GlobalTensor<int64_t> indexGm, int64_t tokenIndex, int64_t rows,
                                                    CkvkrParams &rmsNormAndScatterCkvParams,
                                                    CkvkrParams &ropeAndScatterKrParams);
    template <bool needQnDynamicQuant>
    __aicore__ inline void AivProcess(AivOffset &aivOffset, int64_t batchOffset, int64_t curStepBatchSize,
                                      int64_t numHeadOffset, int64_t mmQnLoops);
    template <bool needQnDynamicQuant>
    __aicore__ inline void AicProcess(AicOffset &aicOffset, int64_t batchOffset, int64_t mmQnLoops);
    template <typename T, typename O, typename S, bool needCheckEmptyTensor = false, bool needCheckAFullLoad = false,
              bool isContinuousCopy = true>
    __aicore__ inline void
    MatmulSplitN(const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm,
                 const MMParams &mmPara, const UsedBlockParams &mmBlockParams,
                 const GlobalTensor<S> &tensorAScaleGm = {}, const GlobalTensor<S> &tensorBScaleGm = {});
    __aicore__ inline void MatmulQcQr(AicOffset &aicOffset);
    __aicore__ inline void MatmulAndSyncQcQr(AicOffset &aicOffset);
    __aicore__ inline void MatmulQnWeightPreload(int64_t weightUkOffset, int64_t mmQnLoops);
    __aicore__ inline void PreloadQnAndSync(AicOffset &aicOffset, int64_t mmQnLoops);
    template <bool needQnDynamicQuant>
    __aicore__ inline void MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset, int64_t weightUkOffset,
                                                            int64_t qnResOffset, int64_t mmQnLoops);
    __aicore__ inline void CopyInSinCos(int64_t tokenIndex, int64_t curVecToken, int64_t batchOffset,
                                        int64_t curStepBatchSize);
    __aicore__ inline void RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset, int64_t rmsNormCqResOffset,
                                     int64_t curVecToken, int64_t curBlockTokenOffset);
    __aicore__ inline void RopeAndScatterKr(LocalTensor<dequantScaleType> &dequantScaleXLocal,
                                            LocalTensor<uint8_t> &shareTmpUb,
                                            LocalTensor<typename MLAPT::ropeComputType> &cosLocalCkvKr,
                                            LocalTensor<typename MLAPT::ropeComputType> &sinLocalCkvKr,
                                            CkvkrParams ropeAndScatterKrParams);
    __aicore__ inline void RmsNormAndScatterCkv(LocalTensor<dequantScaleType> &dequantScaleXLocal,
                                                LocalTensor<uint8_t> &shareTmpUb,
                                                LocalTensor<typename MLAPT::ropeComputType> &cosLocalCkvKr,
                                                LocalTensor<typename MLAPT::ropeComputType> &sinLocalCkvKr,
                                                CkvkrParams rmsNormAndScatterCkvParams);
    __aicore__ inline void RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset, int64_t ropeKrOffset,
                                                   int64_t curVecToken);
    __aicore__ inline void RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset, int64_t curVecToken,
                                  int64_t curBlockTokenOffset);
    __aicore__ inline void DequantQc(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, int64_t curVecToken,
                                     int64_t curBlockTokenOffset);
    __aicore__ inline void CastQc(int64_t mmQnPreCastOffset, int64_t mmQnPreCastResOffset, int64_t curVecToken);
    // 低时延算力分组场景
    __aicore__ inline void DequantAndRopeSplitNSyncMMQcQr(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset,
                                                          int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset,
                                                    int64_t qcQrScaleOffset);
    __aicore__ inline void RopeQrSplitNGroupCase(int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DequantQcQrSplitN(const DequantQcQrSplitNParams &dequantQcQrSplitN);
    __aicore__ inline void CastQcQrSplitN(const CastQcQrSplitNParams &castQcQrSplitN);
    __aicore__ inline void RopeQrSplitN(const RopeQrSplitNParams &ropeQrSplitNParams);
    __aicore__ inline void DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset, int64_t curStepBatchSize,
                                                          int64_t numHeadOffset, int64_t mmQnLoops);

    const optiling::MlaPrologBaseParams *__restrict baseParams_;
    const optiling::MlaPrologTilingData *__restrict tilingData_;
    TPipe *pipe_;
    uint32_t cubeBlockIdx_ = 0U; // AIV上使用AIC的blockIdx
    uint32_t blockIdx_ = 0U;
    int64_t vectorCoreNum_;
    int64_t curVectorBlockNum_;
    int64_t vectorRow_ = 1;
    uint32_t curVecTokenMax_;
    uint32_t curStepVecBackToken_;
    uint32_t curStepVecFrontListNum_;
    uint32_t curStepVecFrontToken_;
    uint64_t dequantScaleCqSize_ = 1;
    bool enableSmoothScalesCq_;
    static constexpr uint32_t cvMode = MLAPT::cvRatio; // 编译态，默认cv1:2
    uint32_t cvRatio_ = 2U;                            // 默认cv 1:2

    struct DequantTool {
        LocalTensor<dequantScaleType> dequantScaleCqLocal_;
        GlobalTensor<dequantScaleType> dequantScaleCqGm_;
        TBuf<TPosition::VECCALC> dequantScaleCqBuffer_; // 用于临时存储每一行的Scale，以及汇总最终每一行的Scale参数
        __aicore__ inline DequantTool()
        {
        }
    };

    // 算子分组开关
    DequantTool dequantTool_;

    // GM
    GlobalTensor<mmInputType> tokenXGm_;
    GlobalTensor<mmInputType> weightDqGm_;
    GlobalTensor<mmInputType> weightDkvKrGm_;
    GlobalTensor<mmQcQrInputType> weightUqQrGm_;
    GlobalTensor<mmQnInputType> weightUkGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCqGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCkvGm_;
    GlobalTensor<ropeSinCosType> ropeSinGm_;
    GlobalTensor<ropeSinCosType> ropeCosGm_;
    GlobalTensor<int64_t> cacheIndexGm_;
    GlobalTensor<ropeOutputType> qrOutGm_;
    GlobalTensor<kvCacheType> kvCacheGm_;
    GlobalTensor<krCacheType> krCacheGm_;

    GlobalTensor<dequantScaleType> dequantScaleXGm_;
    GlobalTensor<dequantScaleType> dequantScaleWDqGm_;
    GlobalTensor<dequantScaleType> dequantScaleWDkvkrGm_;
    GlobalTensor<dequantScaleType> deqScaleQcQrW_; // per-channel反量化参数
    GlobalTensor<float> smoothScaleCqGm_;
    GlobalTensor<float> quantScaleCkvGm_;
    GlobalTensor<float> quantScaleCkrGm_;

    GlobalTensor<int32_t> actualSeqLenGm_;
    GlobalTensor<float> kNopeClipAlphaGm_;

    GlobalTensor<mmCqOutputType> mmCqResGm_;
    GlobalTensor<mmCkvKrOutputType> mmCkvKrResGm_;
    GlobalTensor<mmQcQrOutputType> mmQcQrResGm_;
    GlobalTensor<mmQnInputType> mmQcQrResDequantGm_;
    GlobalTensor<mmQnOutputType> mmQnResGm_;
    GlobalTensor<rmsNormCqOutputType> rmsNormCqResGm_;
    GlobalTensor<queryOutputType> queryOutGm_;
    GlobalTensor<dequantScaleQNopeType> dequantScaleQNopeGm_;
    GlobalTensor<dequantScaleQNormType> dequantScaleQNormGm_;

    // UB
    TBuf<TPosition::VECCALC> shareBuffer_;
    TBuf<TPosition::VECCALC> sincosBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDqBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDkvKrBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCqBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCkvBuffer_;
    TBuf<TPosition::VECCALC> smoothScaleCqBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkvBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkrBuffer_;
    TBuf<TPosition::VECCALC> stepActualSeqBuffer_;

    LocalTensor<ropeComputType> sinLocal_;
    LocalTensor<ropeComputType> cosLocal_;
    LocalTensor<dequantScaleType> dequantScaleWDqLocal_;
    LocalTensor<dequantScaleType> dequantScaleWDkvKrLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCkvLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCqLocal_;
    LocalTensor<float> smoothScaleCqLocal_;
    LocalTensor<float> quantScaleCkrLocal_;
    LocalTensor<float> quantScaleCkvLocal_;
    LocalTensor<int64_t> stepActualSeqLocal_;

    struct ActSeqState {
        bool inited = false;         // whether state is initialized for the run
        int64_t curBatch = 0;        // current batch index for the running token index
        int64_t curPrefix = 0;       // prefix sum of seq lengths up to curBatch (exclusive high bound)
        int64_t prevPrefix = 0;      // prefix sum of seq lengths up to (curBatch - 1)
        int64_t curIndexOffset = 0;  // prefix sum of blocks up to curBatch
        int64_t prevIndexOffset = 0; // prefix sum of blocks up to (curBatch - 1)
    };
    ActSeqState actSeqState_;
    int64_t blockSpanPerBatch_ = 0; // blockNum * blockSize

    TBuf<TPosition::B1> bBufL1_;
    TBuf<TPosition::A1> aBufL1_;
    LocalTensor<mmInputType> bL1Tensor_;
    LocalTensor<mmInputType> aL1Tensor_;
    MMBufParams bufParam_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::CO1> cBufL0_;
};

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::Init(
    __gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq, __gm__ uint8_t *weightUqQr, __gm__ uint8_t *weightUk,
    __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq, __gm__ uint8_t *rmsnormGammaCkv,
    __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos, __gm__ uint8_t *cacheIndex, __gm__ uint8_t *kvCache,
    __gm__ uint8_t *krCache, __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr, __gm__ uint8_t *quantScaleCkv,
    __gm__ uint8_t *quantScaleCkr, __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *actualSeqLen,
    __gm__ uint8_t *kNopeClipAlpha, __gm__ uint8_t *queryOut, __gm__ uint8_t *queryRopeOut,
    __gm__ uint8_t *dequantScaleQNopeOut, __gm__ uint8_t *queryNormOut, __gm__ uint8_t *dequantScaleQNormOut,
    __gm__ uint8_t *workspace)
{
    blockIdx_ = GetBlockIdx(); // cube:0-23  vec:0-47
    if ASCEND_IS_AIC {
        cubeBlockIdx_ = blockIdx_;
    } else {
        cvRatio_ = GetSubBlockNum(); // CV1:2场景返回2，其他场景返回1
        cubeBlockIdx_ = blockIdx_ / cvRatio_;
    }
    vectorCoreNum_ = static_cast<int64_t>(baseParams_->vectorBlockNum); // aivNum 48
    curVectorBlockNum_ = static_cast<int64_t>(baseParams_->stepBatchSize);
    if (cvMode == 1 && cvRatio_ == 2) { // 编译态cv1:1，运行态cv1:2
        if (vectorCoreNum_ < curVectorBlockNum_) {
            vectorCoreNum_ = vectorCoreNum_ * 2; // 修正为运行态vector数目
        }
    }
    enableSmoothScalesCq_ = smoothScaleCq == nullptr ? false : true;
    curVecTokenMax_ = (curVectorBlockNum_ + vectorCoreNum_ - 1) / vectorCoreNum_;
    // GM
    tokenXGm_.SetGlobalBuffer((__gm__ mmInputType *)tokenX);
    weightDqGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDq);         // NZ
    weightDkvKrGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDkvKr);   // NZ
    weightUqQrGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)weightUqQr); // NZ
    weightUkGm_.SetGlobalBuffer((__gm__ mmQnInputType *)weightUk);
    rmsnormGammaCkvGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCkv);
    rmsnormGammaCqGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCq);
    ropeCosGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeCos);
    ropeSinGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeSin);
    if constexpr (MLAPT::cacheMode != CACHE_MODE::ND) {
        cacheIndexGm_.SetGlobalBuffer((__gm__ int64_t *)cacheIndex);
    }
    krCacheGm_.SetGlobalBuffer((__gm__ krCacheType *)krCache);
    kvCacheGm_.SetGlobalBuffer((__gm__ kvCacheType *)kvCache);
    qrOutGm_.SetGlobalBuffer((__gm__ ropeOutputType *)queryRopeOut);
    if constexpr ((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) &&
                  !isPertile) {
        dequantScaleQNopeGm_.SetGlobalBuffer((__gm__ dequantScaleQNopeType *)dequantScaleQNopeOut);
        queryOutGm_.SetGlobalBuffer((__gm__ queryOutputType *)queryOut);
    } else {
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)queryOut);
    }
    if (baseParams_->queryNormFlag == 1U) {
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)queryNormOut);
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            dequantScaleQNormGm_.SetGlobalBuffer((__gm__ dequantScaleQNormType *)dequantScaleQNormOut);
        }
    }
    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        actualSeqLenGm_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLen);
    }

    MmParamInit();
    ScaleInit(dequantScaleX, dequantScaleWDq, deqScaleQcQrW, dequantScaleWDkvkr, quantScaleCkv, quantScaleCkr,
              smoothScaleCq, kNopeClipAlpha);
    WorkspaceInit(workspace);
    if ASCEND_IS_AIC {
        CubeBufferInit();
    } else {
        VectorBufferInit();
    }
}

template <typename MLAPT>
__aicore__ inline void
MlaPrologVecS1CubS2<MLAPT>::ScaleInit(__gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                      __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
                                      __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
                                      __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *kNopeClipAlpha)
{
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        dequantScaleXGm_.SetGlobalBuffer((__gm__ dequantScaleType *)dequantScaleX);
        dequantScaleWDqGm_.SetGlobalBuffer((__gm__ dequantScaleType *)dequantScaleWDq);
        dequantScaleWDkvkrGm_.SetGlobalBuffer((__gm__ dequantScaleType *)dequantScaleWDkvkr);
    }
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        smoothScaleCqGm_.SetGlobalBuffer((__gm__ float *)smoothScaleCq);
        deqScaleQcQrW_.SetGlobalBuffer((__gm__ dequantScaleType *)deqScaleQcQrW);
        quantScaleCkvGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkv);
        quantScaleCkrGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkr);
    }

    if constexpr (isPertile) {
        kNopeClipAlphaGm_.SetGlobalBuffer((__gm__ float *)kNopeClipAlpha);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmCqParamInit()
{
    if (cubeBlockIdx_ == baseParams_->mm1BlockNum - 1) {
        mmCqParam_.n = baseParams_->headSizeCq - baseParams_->mm1SingleCoreN * cubeBlockIdx_;
    } else {
        mmCqParam_.n = baseParams_->mm1SingleCoreN; // 1536 / 24 = 64
    }
    mmCqParam_.m = baseParams_->stepBatchSize; // 32
    mmCqParam_.k = baseParams_->headSizeX; // 7168
    mmCqParam_.needSetOrgShape = 1;
    mmCqParam_.orgKb = mmCqParam_.k;
    mmCqParam_.orgKa = mmCqParam_.k;
    mmCqParam_.orgN = mmCqParam_.n;
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.orgKc = baseParams_->headSizeCq; // 1536
    mmCqParam_.baseK =
        (sizeof(mmInputType) != ONE_BYTE_TYPE_SIZE) ? 128 : 256; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCqParam_.baseN = 128;
    mmCqParam_.stepK = 4;
    if ((mmCqParam_.k / mmCqParam_.baseK) % mmCqParam_.stepK != 0) {
        mmCqParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    mmCqParam_.kL1StepSize = mmCqParam_.baseK * mmCqParam_.stepK;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmCkvKrParamInit()
{
    if (cubeBlockIdx_ == baseParams_->mm2BlockNum - 1) {
        mmCkvKrParam_.n =
            baseParams_->headSizeCkv + baseParams_->headSizeKr - baseParams_->mm2SingleCoreN * cubeBlockIdx_;
    } else {
        mmCkvKrParam_.n = baseParams_->mm2SingleCoreN;
    }
    mmCkvKrParam_.m = baseParams_->stepBatchSize; // 32
    mmCkvKrParam_.k = baseParams_->headSizeX; // 7168
    mmCkvKrParam_.needSetOrgShape = 1;
    mmCkvKrParam_.orgKb = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKa = mmCkvKrParam_.k;
    mmCkvKrParam_.orgN = mmCkvKrParam_.n;
    mmCkvKrParam_.orgM = mmCkvKrParam_.m;
    mmCkvKrParam_.orgKc = (baseParams_->dimHeadRope + baseParams_->headSizeCkv); // 576
    mmCkvKrParam_.baseK =
        (sizeof(mmInputType) != ONE_BYTE_TYPE_SIZE) ? 128 : 256; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCkvKrParam_.baseN = 128;
    mmCkvKrParam_.stepK = 4;
    if ((mmCkvKrParam_.k / mmCkvKrParam_.baseK) % mmCkvKrParam_.stepK != 0) {
        mmCkvKrParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    mmCkvKrParam_.kL1StepSize = mmCkvKrParam_.baseK * mmCkvKrParam_.stepK;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmQcQrParamInit()
{
    if constexpr (MLAPT::enableGroupComputeOpt) {
        // 算力分组仅考虑G8 Qc Qr(8+4核), n固定128, 这里不处理尾核
        mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
    } else {
        if (cubeBlockIdx_ == baseParams_->mm3BlockNum - 1) {
            mmQcQrParam_.n =
                baseParams_->headSizeQr + baseParams_->headSizeQc - baseParams_->mm3SingleCoreN * cubeBlockIdx_;
        } else {
            mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
        }
    }
    mmQcQrParam_.m = baseParams_->stepBatchSize; // 32

    mmQcQrParam_.k = baseParams_->headSizeCq; // 1536
    mmQcQrParam_.needSetOrgShape = 1;
    mmQcQrParam_.orgKb = mmQcQrParam_.k;
    mmQcQrParam_.orgKa = mmQcQrParam_.k;
    mmQcQrParam_.orgN = mmQcQrParam_.n;
    mmQcQrParam_.orgM = mmQcQrParam_.m;
    mmQcQrParam_.orgKc = (baseParams_->headSizeQr + baseParams_->headSizeQc); // (128 * 32 + 64 * 32)
    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) != ONE_BYTE_TYPE_SIZE) ? 64 : 128;
    if constexpr (MLAPT::enableGroupComputeOpt) {
        mmQcQrParam_.baseN = 128;
    } else {
        if (mmQcQrParam_.m <= 64) {
            mmQcQrParam_.baseN = 256;
        } else {
            mmQcQrParam_.baseN = 128;
        }
    }
    mmQcQrParam_.stepK = 4;
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmQnParamInit()
{
    mmQnParam_.m = baseParams_->stepBatchSize; // 32
    mmQnParam_.n = baseParams_->headSizeCkv;   // 512
    mmQnParam_.k = baseParams_->dimHeadSizeQc; // 128, 这里numHeadSize被分核，matmul设置里不体现
    mmQnParam_.needSetOrgShape = 1;
    mmQnParam_.orgN = mmQnParam_.n;
    mmQnParam_.orgM = mmQnParam_.m;
    if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value || std::is_same<mmQcQrOutputType, float>::value) {
        mmQnParam_.orgKa = baseParams_->headSizeQc;
    } else {
        mmQnParam_.orgKa = baseParams_->headSizeQc + baseParams_->headSizeQr;
    }
    mmQnParam_.orgKc = baseParams_->headSizeCkv * baseParams_->numHeadSize;
    mmQnParam_.orgKb = baseParams_->dimHeadSizeQc;
    mmQnParam_.baseN = 128;
    mmQnParam_.baseK = 128;
    mmQnParam_.stepK = 1;
    if ((mmQnParam_.k > mmQnParam_.baseK) && (mmQnParam_.k % mmQnParam_.baseK != 0)) {
        mmQnParam_.baseK = 64;
        mmQnParam_.stepK = 3; // support D = 192
    }
    mmQnParam_.kL1StepSize = mmQnParam_.baseK * mmQnParam_.stepK;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MmParamInit()
{
    MmCqParamInit();
    MmCkvKrParamInit();
    MmQcQrParamInit();
    MmQnParamInit();
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::VectorBufferInit()
{
    // 暂存已分配UB大小，其余为shareBuffer
    uint64_t usedBytes = 0;
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        uint64_t dequantScaleWDqSize = baseParams_->headSizeCq * sizeof(float);
        pipe_->InitBuffer(dequantScaleWDqBuffer_, dequantScaleWDqSize); // [1, 1536]
        usedBytes += dequantScaleWDqSize;
        dequantScaleWDqLocal_ = dequantScaleWDqBuffer_.Get<float>();

        uint64_t dequantScaleWDkvKrSize = (baseParams_->dimHeadRope + baseParams_->headSizeCkv) * sizeof(float);
        pipe_->InitBuffer(dequantScaleWDkvKrBuffer_, dequantScaleWDkvKrSize); // [1, 512 + 64]
        usedBytes += dequantScaleWDkvKrSize;
        dequantScaleWDkvKrLocal_ = dequantScaleWDkvKrBuffer_.Get<float>();
    }
    uint64_t rmsnormGammaCqSize = baseParams_->headSizeCq * sizeof(rmsNormGammaType);
    pipe_->InitBuffer(rmsnormGammaCqBuffer_, rmsnormGammaCqSize); // [1, 1536] bf16
    usedBytes += rmsnormGammaCqSize;
    rmsnormGammaCqLocal_ = rmsnormGammaCqBuffer_.Get<rmsNormGammaType>();

    uint64_t rmsnormGammaCkvSize = baseParams_->headSizeCkv * sizeof(rmsNormGammaType);
    pipe_->InitBuffer(rmsnormGammaCkvBuffer_, rmsnormGammaCkvSize); // [1, 512] bf16
    usedBytes += rmsnormGammaCkvSize;
    rmsnormGammaCkvLocal_ = rmsnormGammaCkvBuffer_.Get<rmsNormGammaType>();

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        if (enableSmoothScalesCq_) {
            uint64_t smoothScaleCqSize = baseParams_->headSizeCq * sizeof(float);
            pipe_->InitBuffer(smoothScaleCqBuffer_, smoothScaleCqSize); // [1, 1536]
            usedBytes += smoothScaleCqSize;
            smoothScaleCqLocal_ = smoothScaleCqBuffer_.Get<float>();
        }
    }

    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        uint64_t quantScaleCkrSize = baseParams_->dimHeadRope * sizeof(float);
        pipe_->InitBuffer(quantScaleCkrBuffer_, baseParams_->dimHeadRope * sizeof(float)); // [1, 64]
        usedBytes += quantScaleCkrSize;
        quantScaleCkrLocal_ = quantScaleCkrBuffer_.Get<float>();
    }

    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
        uint64_t quantScaleCkvSize = 0;
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value ||
                      std::is_same<mmCkvKrOutputType, float>::value) {
            quantScaleCkvSize = ALIGN_BLOCK_SIZE;
            pipe_->InitBuffer(quantScaleCkvBuffer_, quantScaleCkvSize);
            usedBytes += quantScaleCkvSize;
        } else { // per_channel量化场景走else分支
            quantScaleCkvSize = baseParams_->headSizeCkv * sizeof(float);
            pipe_->InitBuffer(quantScaleCkvBuffer_, quantScaleCkvSize); // [1, 512]
            usedBytes += quantScaleCkvSize;
        }
        quantScaleCkvLocal_ = quantScaleCkvBuffer_.Get<float>();
    }

    // 预留brcb的空间
    uint64_t deQuantScaleCqSize = (baseParams_->stepBatchSize + 7) * ALIGN_BLOCK_SIZE;
    pipe_->InitBuffer(dequantTool_.dequantScaleCqBuffer_, deQuantScaleCqSize);
    usedBytes += deQuantScaleCqSize;
    dequantTool_.dequantScaleCqLocal_ = dequantTool_.dequantScaleCqBuffer_.template Get<dequantScaleType>();

    uint64_t sincosSize = 0;
    if constexpr (MLAPT::enableDequantOpt) {
        // 在ropeQr进行切N处理后，会复用shareBuffer的内存，不需要额外申请
        // 开启开关后会按照head切分rope qr，此时需要加载一半batchsize数量的sin和cos值
        // 需要2倍的空间分别存储sin和cos
        sincosSize = sizeof(ropeComputType) * 2 * baseParams_->dimHeadRope *
                     ((baseParams_->stepBatchSize + cvRatio_ - 1) / cvRatio_);
        pipe_->InitBuffer(sincosBuffer_, sincosSize);
    } else {
        // 需要2倍的空间分别存储sin和cos
        sincosSize = sizeof(ropeComputType) * 2 * baseParams_->dimHeadRope * curVecTokenMax_;
        pipe_->InitBuffer(sincosBuffer_, sincosSize); // [2, 64] float
    }
    usedBytes += sincosSize;

    if constexpr (!MLAPT::enableDequantOpt) {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * curVecTokenMax_];
    } else {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + cvRatio_ - 1) / cvRatio_)];
    }

    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        uint64_t stepActualSeqSize = sizeof(int64_t) * baseParams_->stepBatchSize;
        pipe_->InitBuffer(stepActualSeqBuffer_, stepActualSeqSize); // [1, stepBatchSize]
        usedBytes += stepActualSeqSize;
        stepActualSeqLocal_ = stepActualSeqBuffer_.Get<int64_t>();
    }

    // 由于shareBuffer属于各个vector操作临时申请内存的区域内存使用不固定，建议shareBuffer始终放在最后
    // 防止写入shareBuffer越界导致前面固定申请的UB内存被踩。
    pipe_->InitBuffer(shareBuffer_, MAX_UB_SIZE - usedBytes);
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_CKVKR);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CubeBufferInit()
{
    // cube相关Buffer初始化
    pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
    pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);

    aL1Tensor_ = aBufL1_.Get<mmInputType>();
    bL1Tensor_ = bBufL1_.Get<mmInputType>();
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
    AscendC::SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
    bufParam_.aL1BufAddr = aBufL1_.GetBufferAddr(aL1Tensor_.GetBufferHandle());
    bufParam_.bL1BufAddr = bBufL1_.GetBufferAddr(bL1Tensor_.GetBufferHandle());

    pipe_->InitBuffer(cBufL0_, L0C_PP_SIZE * 2); // 128K
    pipe_->InitBuffer(bBufL0_, L0B_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(aBufL0_, L0A_PP_SIZE * 2); // 64K

    AscendC::SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    AscendC::SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    AscendC::SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    AscendC::SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

    AscendC::SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    AscendC::SetFlag<HardEvent::FIX_M>(L0C_EVENT1);

    AscendC::SetFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);

    bufParam_.cL0BufAddr = cBufL0_.GetBufferAddr(cBufL0_.Get<float>().GetBufferHandle());
    bufParam_.bL0BufAddr = bBufL0_.GetBufferAddr(bBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.aL0BufAddr = aBufL0_.GetBufferAddr(aBufL0_.Get<mmInputType>().GetBufferHandle());
}

/*
 * workspace管理
 * 1. 常驻：dequantTool_.dequantScaleCqGm_ stepBs * 32 Byte
 * 2. 中间结果：
 *      tokenXGm_──────>mmCkvKrResGm_
 *          |           [stepBS, HCkv + Dr]
 *          |           (bf16 | int32)
 *          └─────────>mmCqResGm_──────>rmsNormCqResGm_──────>mmQcQrResGm_──────>mmQcQrResDequantGm_──────>mmQnResGm_
 *                     [stepBS, HCq]    [stepBS, HCq]     [stepBS, N1, D + Dr]   [stepBS, N1, D]           [stepBS, N1,
 * HCkv] (bf16 | int32)   (bf16 | int8)     (bf16 | int32)         (bf16)                    (bf16)
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::WorkspaceInit(__gm__ uint8_t *workspace)
{
    int64_t workspaceOffset = 0;
    dequantScaleCqSize_ = Align(dequantScaleCqSize_, BYTE_BLOCK);
    if constexpr (MLAPT::enableGroupComputeOpt || MLAPT::enableDequantOpt) {
        dequantTool_.dequantScaleCqGm_.SetGlobalBuffer((__gm__ dequantScaleType *)(workspace + workspaceOffset));
        workspaceOffset += baseParams_->stepBatchSize * dequantScaleCqSize_;
    }

    mmCkvKrResGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType *)(workspace + workspaceOffset));
    workspaceOffset += static_cast<int64_t>(sizeof(mmCkvKrOutputType)) *
                       static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                       static_cast<int64_t>(baseParams_->stepBatchSize);

    mmCqResGm_.SetGlobalBuffer((__gm__ mmCqOutputType *)(workspace + workspaceOffset));

    if (baseParams_->queryNormFlag == 0U) {
        // 全量化场景下`mmCqResGm_`与`rmsNormCqResGm_`的dtype不同，无法共用workspace，此处需要偏移`mmCqResGm_`占用的大小；
        if constexpr (!std::is_same<rmsNormCqOutputType, mmCqOutputType>::value) {
            workspaceOffset += static_cast<int64_t>(sizeof(mmCqOutputType)) *
                               static_cast<int64_t>(baseParams_->headSizeCq) *
                               static_cast<int64_t>(baseParams_->stepBatchSize);
        }
        // 不返回queryNorm时，rmsNormCq结果放在workspace
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ rmsNormCqOutputType *)(workspace + workspaceOffset));
        workspaceOffset += static_cast<int64_t>(baseParams_->headSizeCq) *
                           static_cast<int64_t>(baseParams_->stepBatchSize) * sizeof(rmsNormCqOutputType);
    } else {
        // 返回queryNorm时，rmsNormCq结果直接输出gm，此处需要偏移`mmCqResGm_`占用的大小
        workspaceOffset += static_cast<int64_t>(baseParams_->headSizeCq) *
                           static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(sizeof(mmCqOutputType));
    }

    mmQcQrResGm_.SetGlobalBuffer((__gm__ mmQcQrOutputType *)(workspace + workspaceOffset));
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeQc + baseParams_->headSizeQr) *
                           static_cast<int64_t>(sizeof(mmQcQrOutputType));
    }

    mmQcQrResDequantGm_.SetGlobalBuffer((__gm__ mmQnInputType *)(workspace + workspaceOffset));
    if constexpr ((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) &&
                  !isPertile) {
        workspaceOffset += sizeof(mmQnInputType) * static_cast<int64_t>(baseParams_->dimHeadSizeQc) *
                           static_cast<int64_t>(baseParams_->numHeadSize) *
                           static_cast<int64_t>(baseParams_->stepBatchSize);
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)(workspace + workspaceOffset));
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::UpdateStepBatchParams(int64_t curStepBatchSize)
{
    mmQnParam_.m = curStepBatchSize;
    mmQcQrParam_.m = curStepBatchSize;
    mmCkvKrParam_.m = curStepBatchSize;
    mmCqParam_.m = curStepBatchSize;
    curVectorBlockNum_ = curStepBatchSize;
}

/*
 * MlaProlog算子计算&CV流水同步流程
 *                    ┌───────────────── token_x ─────────────────┐
 *                    |                                           ▼
 *                    |                                      MatmulCkvKr
 *                    ▼                                           | wait mm CkvKr(0x1)
 *                MatmulCq                                        ▼
 *                    | wait mm Cq(0x1)                  ┌─────────────────┐
 *                    ▼                                  ▼                 ▼
 *               RmsNorm(Cq)                         RmsNorm(Ckv)       Rope(Kr)
 *                    | wait rmsNorm cq(0x1)             |                 |
 *                    ▼                                  ▼                 ▼
 *        ┌───────MatmulQcQr───────┐                 Scatter(Ckv)      Scatter(Kr)
 *        | wait mm Qc(0x1)        |                     |                 |
 *        ▼                        |                     ▼                 ▼
 *    DequantQc                    |                 kv_cache_out      kr_cache_out
 *        | wait dequant qc(0x1)   | wait mm Qr(0x2)
 *        ▼                        ▼
 *     MatmulQn                 Rope(Qr)
 *        | wait mm Qn(0x1)        |
 *   DynamicQuantQn──┐             |
 *        |          ▼             |
 *        ▼   dequant_scale_out    ▼
 *    query_out               query_rope_out
 * 注：仅为表明基本计算与CV同步流程，仅包含了影响CV同步的量化分支，其余量化分支应参考设计文档。
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::Process()
{
    constexpr bool needQnDynamicQuant =
        (std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) && !isPertile;
    int64_t numHeadOffset = cubeBlockIdx_ * baseParams_->mm4SingleCoreBatch;
    int64_t mmQnLoops;
    if (cubeBlockIdx_ == baseParams_->mm4BlockNum - 1) {
        mmQnLoops = static_cast<int64_t>(baseParams_->numHeadSize) - numHeadOffset;
    } else {
        mmQnLoops = static_cast<int64_t>(baseParams_->mm4SingleCoreBatch);
    }

    // AIV的offset参数
    AivOffset aivOffset;
    ComputeAivOffset(aivOffset, 0);

    // AIC的offset参数
    AicOffset aicOffset;
    ComputeAicOffset(aicOffset, numHeadOffset);

    int64_t bsSize = static_cast<int64_t>(baseParams_->tokenSize);
    // 需要考虑BS合轴的尾块情况
    for (int64_t batchOffset = 0; batchOffset < bsSize;
         batchOffset += static_cast<int64_t>(baseParams_->stepBatchSize)) {
        if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
            if (!actSeqState_.inited) {
                actSeqState_.curBatch = 0;
                actSeqState_.curPrefix = (baseParams_->batchSize > 0) ? actualSeqLenGm_(0) : 0;
                actSeqState_.prevPrefix = 0;
                actSeqState_.inited = true;
                actSeqState_.curIndexOffset =
                    CeilDivT(actSeqState_.curPrefix, static_cast<int64_t>(baseParams_->blockSize));
                actSeqState_.prevIndexOffset = 0;
            }
        }
        int64_t curStepBatchSize = bsSize - batchOffset;
        if (curStepBatchSize >= static_cast<int64_t>(baseParams_->stepBatchSize)) {
            curStepBatchSize = static_cast<int64_t>(baseParams_->stepBatchSize);
        } else {
            UpdateStepBatchParams(curStepBatchSize); // 320 - 256
            if (batchOffset != 0) {
                ComputeAivOffset(aivOffset, batchOffset);
            }
        }
        if ASCEND_IS_AIV {
            AivProcess<needQnDynamicQuant>(aivOffset, batchOffset, curStepBatchSize, numHeadOffset, mmQnLoops);
        }
        if ASCEND_IS_AIC {
            AicProcess<needQnDynamicQuant>(aicOffset, batchOffset, mmQnLoops);
        }
    }
    if ASCEND_IS_AIC {
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);

        AscendC::WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
        AscendC::WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
        AscendC::WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
        AscendC::WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

        AscendC::WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        AscendC::WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);

        AscendC::WaitFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
        CrossCoreWaitFlag(FINISH_VEC_CKVKR);
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AicProcess(AicOffset &aicOffset, int64_t batchOffset,
                                                              int64_t mmQnLoops)
{
    int64_t dequantScaleXOffset = static_cast<int64_t>(baseParams_->headSizeX) * batchOffset / 32;
    int64_t tokenXOffset = static_cast<int64_t>(baseParams_->headSizeX) * batchOffset;
    // MatmulCq ──> RmsNorm(Cq)
    // [32, 7168] * [7168, 1536] = [32, 1536]
    MatmulSplitN<mmInputType, mmCqOutputType, dequantScaleType>(
        mmCqResGm_[aicOffset.cqResOffset], tokenXGm_[tokenXOffset], weightDqGm_[aicOffset.weightDqOffset], mmCqParam_,
        UsedBlockParams{0, baseParams_->mm1BlockNum});
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ);
    // MatmulCkvKr ──> RmsNorm(Ckv)
    //            └──> Rope(Kr)
    // [32, 7168] * [7168, 512+64] = [32, 576]
    CrossCoreWaitFlag(FINISH_VEC_CKVKR);
    MatmulSplitN<mmInputType, mmCkvKrOutputType, dequantScaleType, true>(
        mmCkvKrResGm_[aicOffset.ckvKrResOffset], tokenXGm_[tokenXOffset], weightDkvKrGm_[aicOffset.weightDkvKrOffset],
        mmCkvKrParam_, UsedBlockParams{0, baseParams_->mm2BlockNum});
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CKVKR);
    CrossCoreWaitFlag(FINISH_VEC_RMSNORM_CQ);

    MatmulAndSyncQcQr(aicOffset);

    PreloadQnAndSync(aicOffset, mmQnLoops);
    MatmulQnSyncDynamicQuantAndMulQr<needQnDynamicQuant>(aicOffset.qcOffset, aicOffset.weightUkOffset,
                                                         aicOffset.qnResOffset, mmQnLoops);
    if constexpr (!needQnDynamicQuant) {
        // MatmulQn的结果直接输出到 queryOut, qnOffset需要按Batch轴偏移
        aicOffset.qnResOffset += static_cast<int64_t>(baseParams_->numHeadSize) *
                                 static_cast<int64_t>(baseParams_->headSizeCkv) *
                                 static_cast<int64_t>(baseParams_->stepBatchSize);
    }
    if (unlikely(baseParams_->queryNormFlag == 1U)) {
        aicOffset.rmsNormCqResOffset +=
            static_cast<int64_t>(baseParams_->headSizeCq) * static_cast<int64_t>(baseParams_->stepBatchSize);
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::AivProcess(AivOffset &aivOffset, int64_t batchOffset,
                                                              int64_t curStepBatchSize, int64_t numHeadOffset,
                                                              int64_t mmQnLoops)
{
    int64_t tokenIndex = aivOffset.curBlockTokenOffset + batchOffset;
    if (batchOffset == 0) {
        // 只需要搬运一次
        CopyGlobalParams();
    }
    int64_t rmsNormCqResOffset;
    if (baseParams_->queryNormFlag == 1U) {
        rmsNormCqResOffset = tokenIndex * baseParams_->headSizeCq;
    } else {
        rmsNormCqResOffset = aivOffset.rmsNormCqOffset;
    }
    CopyInSinCos(tokenIndex, aivOffset.curVecToken, batchOffset, curStepBatchSize);
    CrossCoreWaitFlag(FINISH_MM_CQ);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
    RmsNormCq(tokenIndex, aivOffset.rmsNormCqOffset, rmsNormCqResOffset, aivOffset.curVecToken,
              aivOffset.curBlockTokenOffset);
    // 由于RmsNormCq和MatmulQcQr的分核策略不一样，需要等所有vector上的RmsNormCq执行完成后才能启动MatmulQcQr
    // 需要所有vector核上的RmsNormCq执行完成后，才发起MatmulQcQr的执行
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);

    // 聚合全部scale结果
    if constexpr ((MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) &&
                  (std::is_same<rmsNormCqOutputType, int8_t>::value)) {
        DataCopy(dequantTool_.dequantScaleCqLocal_, dequantTool_.dequantScaleCqGm_,
                 ALIGN_BLOCK_SIZE / sizeof(float) * baseParams_->stepBatchSize);
    }
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
    CrossCoreWaitFlag(FINISH_MM_CKVKR);
    WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
    RmsNormRopeScatterCkvKr(tokenIndex, aivOffset.rmsNormCkvOffset, aivOffset.ropeKrOffset, aivOffset.curVecToken);
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_CKVKR);

    // 根据不同分支条件处理
    if constexpr (!MLAPT::enableGroupComputeOpt) {
        if constexpr (MLAPT::enableDequantOpt) {
            DequantAndRopeSplitNSyncMMQcQr(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset,
                                           aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset);
        } else if constexpr (std::is_same<mmQcQrInputType, bfloat16_t>::value) {
            CrossCoreWaitFlag(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset, aivOffset.curVecToken,
                   aivOffset.curBlockTokenOffset);
        } else {
            CrossCoreWaitFlag(FINISH_MM_QCQR);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            DequantQc(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset, aivOffset.curVecToken,
                      aivOffset.curBlockTokenOffset);
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
            RopeQr(aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset, aivOffset.curVecToken,
                   aivOffset.curBlockTokenOffset);
        }

        if constexpr (!(MLAPT::enableDequantOpt) && needQnDynamicQuant) {
            // 非切N场景，需要等待全部rope的结果做完并搬到GM
            WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(FINISH_VEC_ALL);
        }
        if constexpr (needQnDynamicQuant) {
            DynamicQuantQnAndMulQrSyncMMQn(batchOffset, curStepBatchSize, numHeadOffset, mmQnLoops);
        }
        aivOffset.ropeQrResOffset +=
            static_cast<int64_t>(baseParams_->headSizeQr) * static_cast<int64_t>(baseParams_->stepBatchSize);
    } else {
        CrossCoreWaitFlag(FINISH_MM_QC);
        DequantQcSplitNGroupCase(aivOffset.mmQnPreDequantOffset, aivOffset.mmQnPreDequantResOffset,
                                 aivOffset.qcScaleOffsetSplitN);
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC);
        CrossCoreWaitFlag(FINISH_MM_QR);
        RopeQrSplitNGroupCase(aivOffset.ropeQrSplitNOffset, aivOffset.ropeQrResSplitNOffset);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ComputeAicOffset(AicOffset &aicOffset, int64_t numHeadOffset)
{
    aicOffset.cqResOffset = baseParams_->mm1SingleCoreN * blockIdx_; // 64 * idx
    aicOffset.dequantScaleWDqOffset =
        static_cast<int64_t>(baseParams_->headSizeX) / 32 * aicOffset.cqResOffset;                   // 7168 * 64 * idx
    aicOffset.weightDqOffset = static_cast<int64_t>(baseParams_->headSizeX) * aicOffset.cqResOffset; // 7168 * 64 * idx

    aicOffset.ckvKrResOffset = baseParams_->mm2SingleCoreN * blockIdx_; //  (512 + 64) / 9 * idx  = 64 * idx
    aicOffset.dequantScaleWDkvKrOffset = static_cast<int64_t>(baseParams_->headSizeX) / 32 *
                                         aicOffset.ckvKrResOffset; // 7168 * (512 + 64) / 9 * idx = 7168 * 64 * idx
    aicOffset.weightDkvKrOffset = static_cast<int64_t>(baseParams_->headSizeX) *
                                  aicOffset.ckvKrResOffset; // 7168 * (512 + 64) / 9 * idx = 7168 * 64 * idx

    if constexpr (MLAPT::enableGroupComputeOpt) {
        aicOffset.weightUqOffset =
            blockIdx_ * (baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) *
            static_cast<int64_t>(baseParams_->headSizeCq); // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
        aicOffset.weightQrOffset = static_cast<int64_t>(baseParams_->headSizeCq) *
                                       (baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) * 2 *
                                       (blockIdx_ - QC_CORE_NUM) +
                                   baseParams_->dimHeadSizeQc * static_cast<int64_t>(baseParams_->headSizeCq);
        aicOffset.qCResOffset = baseParams_->dimHeadSizeQc * blockIdx_;
        aicOffset.qRResOffset =
            QC_CORE_NUM * baseParams_->dimHeadSizeQc * baseParams_->stepBatchSize +
            2 * baseParams_->dimHeadRope * (blockIdx_ - QC_CORE_NUM); // m=baseParams_->stepBatchSize
    } else {
        aicOffset.qcQrResOffset = baseParams_->mm3SingleCoreN * blockIdx_; // 192 * idx
        aicOffset.weightUqQrOffset =
            aicOffset.qcQrResOffset *
            static_cast<int64_t>(baseParams_->headSizeCq); // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
        aicOffset.dequantScaleWuqqrOffset = aicOffset.weightUqQrOffset / 32;
    }

    if (cubeBlockIdx_ < baseParams_->mm4BlockNum) {
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value || std::is_same<mmQcQrOutputType, float>::value) {
            aicOffset.qcOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) * numHeadOffset; // 128 * idx
        } else {
            aicOffset.qcOffset = static_cast<int64_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) *
                                 numHeadOffset; // (128 + 64) * idx
        }
        aicOffset.weightUkOffset = numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv) *
                                   static_cast<int64_t>(baseParams_->dimHeadSizeQc);            // (128 * 512) * idx
        aicOffset.qnResOffset = static_cast<int64_t>(baseParams_->headSizeCkv) * numHeadOffset; // BS, N, Dq
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ComputeAivOffset(AivOffset &aivOffset, int64_t batchOffset)
{
    curStepVecFrontListNum_ = curVectorBlockNum_ % static_cast<uint32_t>(vectorCoreNum_);
    curStepVecBackToken_ = curVectorBlockNum_ / static_cast<uint32_t>(vectorCoreNum_);
    curStepVecFrontToken_ = curStepVecFrontListNum_ != 0 ? curStepVecBackToken_ + 1 : curStepVecBackToken_;

    aivOffset.curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    aivOffset.curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ?
                                        blockIdx_ * aivOffset.curVecToken :
                                        blockIdx_ * aivOffset.curVecToken + curStepVecFrontListNum_;
    aivOffset.rmsNormCqOffset = baseParams_->headSizeCq * aivOffset.curBlockTokenOffset; //  1536 * idx
    aivOffset.rmsNormCkvOffset =
        (baseParams_->dimHeadRope + baseParams_->headSizeCkv) * aivOffset.curBlockTokenOffset; // (512 + 64) * idx
    aivOffset.ropeKrOffset = aivOffset.rmsNormCkvOffset + baseParams_->headSizeCkv;            // 512 + (512 + 64) * idx
    if constexpr (!MLAPT::enableDequantOpt) {
        aivOffset.mmQnPreDequantOffset =
            (baseParams_->headSizeQr + baseParams_->headSizeQc) * aivOffset.curBlockTokenOffset;
        aivOffset.mmQnPreDequantResOffset = baseParams_->headSizeQc * aivOffset.curBlockTokenOffset;
        aivOffset.ropeQrOffset = static_cast<int64_t>(baseParams_->dimHeadSizeQc) +
                                 aivOffset.curBlockTokenOffset *
                                     static_cast<int64_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) *
                                     static_cast<int64_t>(baseParams_->numHeadSize); // 128 + 32 * (128 + 64) * idx
        aivOffset.ropeQrResOffset =
            static_cast<int64_t>(baseParams_->headSizeQr) *
            (aivOffset.curBlockTokenOffset + batchOffset); // 32 * 64 * idx;    // 按BS合轴切分step
    }

    // 以下的Offset均是与batchOffset无关的，仅初始化一次即可。
    if (batchOffset == 0) {
        if constexpr (MLAPT::enableDequantOpt) {
            aivOffset.mmQnPreDequantOffset = baseParams_->mm3SingleCoreN * cubeBlockIdx_; // qcQrResOffset
            aivOffset.mmQnPreDequantResOffset = baseParams_->mm3SingleCoreN /
                                                (baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) *
                                                baseParams_->dimHeadSizeQc * cubeBlockIdx_;
            aivOffset.ropeQrOffset = aivOffset.mmQnPreDequantOffset + baseParams_->dimHeadSizeQc;
            aivOffset.ropeQrResOffset =
                (baseParams_->mm3SingleCoreN / (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope)) *
                baseParams_->dimHeadRope * cubeBlockIdx_;
        }

        if constexpr (MLAPT::enableGroupComputeOpt) {
            aivOffset.qcScaleOffsetSplitN =
                (baseParams_->headSizeQr + baseParams_->headSizeQc) / QC_CORE_NUM * (blockIdx_ / cvRatio_);
            aivOffset.mmQnPreDequantResOffset = (baseParams_->headSizeQc) / QC_CORE_NUM * (blockIdx_ / cvRatio_);
            aivOffset.mmQnPreDequantOffset = aivOffset.mmQnPreDequantResOffset;
            aivOffset.ropeQrResSplitNOffset = (blockIdx_ - QC_CORE_NUM * cvRatio_) * baseParams_->dimHeadRope;
            aivOffset.ropeQrSplitNOffset = baseParams_->headSizeQc + aivOffset.ropeQrResSplitNOffset;
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CopyInSinCos(int64_t tokenIndex, int64_t curVecToken,
                                                                int64_t batchOffset, int64_t curStepBatchSize)
{
    if constexpr (!MLAPT::enableRope) {
        return;
    }
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    if constexpr (!MLAPT::enableDequantOpt) {
        if (blockIdx_ >= curVectorBlockNum_) {
            return;
        }
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, tokenIndex,
                                                     curVecToken, shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    } else {
        // 如果是切N场景，mm3的每个C核都会做rope
        if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
            return;
        }
        // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
        // >> 1 是将curStepBatchSize分到每个vec核上；
        uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
        int64_t offset = (curStepBatchSize / cvRatio_) * subBlockIdx_ + batchOffset;
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, offset,
                                                     (curStepBatchSize + cvRatio_ - 1) / cvRatio_, shareTmpUb,
                                                     vectorRow_, baseParams_->dimHeadRope);
    }
}

// Mlaprolog 支持int8进int32出, 参考MatmulQcQr
template <typename MLAPT>
template <typename T, typename O, typename S, bool needCheckEmptyTensor, bool needCheckAFullLoad, bool isContinuousCopy>
__aicore__ inline void
MlaPrologVecS1CubS2<MLAPT>::MatmulSplitN(const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm,
                                         const GlobalTensor<T> &tensorBGm, const MMParams &mmPara,
                                         const UsedBlockParams &mmBlockParams, const GlobalTensor<S> &tensorAScaleGm,
                                         const GlobalTensor<S> &tensorBScaleGm)
{
    if constexpr (needCheckEmptyTensor) {
        if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
            return;
        }
    }
    if (blockIdx_ >= mmBlockParams.blockEndIdx || blockIdx_ < mmBlockParams.blockStartIdx) {
        return;
    }
    // 用于enableGroupComputeOpt场景
    if constexpr (needCheckAFullLoad) {
        constexpr uint32_t mSize =
            (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
        bool isAFullLoad = mmQcQrParam_.m <= mSize;
        if (isAFullLoad) {
            MatmulGroupComputeAFullLoad<T, O, S, isContinuousCopy>(tensorResGm, tensorAGm, tensorBGm, mmPara,
                                                                   bufParam_);
            return;
        }
    }
    uint32_t nL1SplitSize = mmPara.baseN;
    uint32_t nInput = mmPara.n;
    uint32_t subNL1SplitSize = nL1SplitSize;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        MatmulSplitK<T, O, S>(tensorResGm, tensorAGm, tensorBGm, mmPara, bufParam_, nL1 * nL1SplitSize,
                              subNL1SplitSize);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulAndSyncQcQr(AicOffset &aicOffset)
{
    if constexpr (!MLAPT::enableGroupComputeOpt) {
        MatmulQcQr(aicOffset);
    } else {
        // MatmulQc
        // 复用mmCqResGm_ workspace
        MatmulSplitN<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, false, true>(
            mmQcQrResGm_[aicOffset.qCResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
            weightUqQrGm_[aicOffset.weightUqOffset], mmQcQrParam_, UsedBlockParams{0, QC_CORE_NUM});
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QC);
        // MatmulQr
        MatmulSplitN<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, false, true, false>(
            mmQcQrResGm_[aicOffset.qRResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
            weightUqQrGm_[aicOffset.weightQrOffset], mmQcQrParam_,
            UsedBlockParams{QC_CORE_NUM, QC_CORE_NUM + QR_CORE_NUM});
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQcQr(AicOffset &aicOffset)
{
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    // RmsNorm(Cq) ──> MatmulQcQr ──> MatmulQn
    //                           └──> Rope(Qr)
    // [32, 1536] * [1536, 32*(128+64)] = [32, 32*192]
    constexpr uint32_t mSize =
        (sizeof(mmQcQrInputType) != sizeof(int8_t)) ? BF16_AFULLLOAD_MAX_MSIZE : INT8_AFULLLOAD_MAX_MSIZE;
    bool isAFullLoad = mmQcQrParam_.m <= mSize;

    uint32_t nL1SplitSize = mmQcQrParam_.baseN;
    uint32_t nInput = mmQcQrParam_.n;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    if (isAFullLoad) {
        LoadL1A(rmsNormCqResGm_[aicOffset.rmsNormCqResOffset], mmQcQrParam_.m, mmQcQrParam_.k, mmQcQrParam_.k,
                bufParam_);
        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
    }

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        if (!isAFullLoad) {
            MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType>(
                mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nL1 * nL1SplitSize,
                subNL1SplitSize);
        } else {
            MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, true>(
                mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nL1 * nL1SplitSize,
                subNL1SplitSize);
        }
        if constexpr (MLAPT::enableDequantOpt) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(FINISH_MM_QCQR_SPLIT_N);
        }
    }
    if (isAFullLoad) {
        AscendC::SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
        bufParam_.aL1BufIter++;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::PreloadQnAndSync(AicOffset &aicOffset, int64_t mmQnLoops)
{
    MatmulQnWeightPreload(aicOffset.weightUkOffset, mmQnLoops);
    if constexpr (MLAPT::enableDequantOpt) {
        // enableDequantOpt分支的CV同步由更细粒度的子函数控制
        return;
    } else if constexpr (MLAPT::enableGroupComputeOpt) {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QR);
    } else {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QCQR);
    }

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC);
    } else {
        WaitAllCore<SYNC_MODE_ALL_CUBE, PIPE_FIX>(FINISH_MM_ALL);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQnWeightPreload(int64_t weightUkOffset, int64_t subLoopTimes)
{
    if constexpr (MLAPT::enableGroupComputeOpt) {
        if (blockIdx_ >= QC_CORE_NUM) {
            return;
        }
    } else {
        if (blockIdx_ >= baseParams_->mm4BlockNum) {
            return;
        }
    }
    if (mmQnParam_.baseK < mmQnParam_.k) {
        return;
    }
    int64_t weightOffset = weightUkOffset;
    for (int32_t i = 0; i < subLoopTimes; ++i) {
        if (i < 1) { // preload double buffer
            LoadL1B<mmQnInputType, DataFormat::ND, false>(weightUkGm_[weightOffset], mmQnParam_.n, mmQnParam_.n,
                                                          mmQnParam_.k, mmQnParam_.k, bufParam_);
            AscendC::WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam_.bL1BufIter & 1u));
            weightOffset +=
                static_cast<int64_t>(baseParams_->headSizeCkv) * static_cast<int64_t>(baseParams_->dimHeadSizeQc);
        }
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void
MlaPrologVecS1CubS2<MLAPT>::MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset, int64_t weightUkOffset,
                                                             int64_t qnResOffset, int64_t subLoopTimes)
{
    uint32_t maxBlockIdx = !MLAPT::enableGroupComputeOpt ? baseParams_->mm4BlockNum : QC_CORE_NUM;
    if (blockIdx_ >= maxBlockIdx) {
        return;
    }
    // MatmulQcQr ──> MatmulQn ──> query_out
    // [32, 128] * [128, 512] = [32, 512]
    // [32, 2, 128] * [2, 128, 512] = [32, 2, 512]
    bool needSparseSync = subLoopTimes > MAX_SYNC_FLAG_COUNT;
    for (int64_t i = 0; i < subLoopTimes; i++) {
        if constexpr (MLAPT::enableDequantOpt) {
            if (!needSparseSync || (i % 2 == 0) || (i == subLoopTimes - 1)) {
                CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC_SPLIT_N);
            }
        }
        if (unlikely(mmQnParam_.k > mmQnParam_.baseK)) {
            uint32_t nL1SplitSize = mmQnParam_.baseN;
            uint32_t nInput = baseParams_->headSizeCkv;
            uint32_t subNL1SplitSize = nL1SplitSize;
            uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
            for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
                if (nL1 == nL1loops - 1) {
                    subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
                }
                MatmulSplitK<mmQnInputType, mmQnOutputType, dequantScaleType, false, false, DataFormat::ND>(
                    mmQnResGm_[qnResOffset], mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset], mmQnParam_,
                    bufParam_, nL1 * nL1SplitSize, subNL1SplitSize);
            }
        } else {
            if (i >= 1) {
                MatmulFullLoad<mmQnInputType, mmQnOutputType, dequantScaleType, false, true>(
                    mmQnResGm_[qnResOffset], mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset], mmQnParam_,
                    bufParam_);
            } else {
                MatmulFullLoad<mmQnInputType, mmQnOutputType, dequantScaleType, true, true>(
                    mmQnResGm_[qnResOffset], mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset], mmQnParam_,
                    bufParam_);
            }
        }
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
            qcOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc);
        } else {
            qcOffset += static_cast<int64_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc);
        }
        weightUkOffset +=
            static_cast<int64_t>(baseParams_->headSizeCkv) * static_cast<int64_t>(baseParams_->dimHeadSizeQc);
        qnResOffset += static_cast<int64_t>(baseParams_->headSizeCkv);

        if constexpr (needQnDynamicQuant) {
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QN_SPLIT_N);
        }
    }
}


template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CopyGlobalParams()
{
    // dequantScaleWDq
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        DataCopyExtParams dequantCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDqLocal_, dequantScaleWDqGm_, dequantCopyParams, dequantPadParams);
    }

    // rmsnormGammaCq
    DataCopy(rmsnormGammaCqLocal_, rmsnormGammaCqGm_, baseParams_->headSizeCq);

    // rmsnormGammaCkv
    DataCopy(rmsnormGammaCkvLocal_, rmsnormGammaCkvGm_, baseParams_->headSizeCkv);

    // smoothScaleCq
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        if (enableSmoothScalesCq_) {
            DataCopyExtParams smoothCopyParams{1, static_cast<uint32_t>(baseParams_->headSizeCq * sizeof(float)), 0, 0,
                                               0};
            DataCopyPadExtParams<float> smoothPadParams{false, 0, 0, 0};
            DataCopyPad(smoothScaleCqLocal_, smoothScaleCqGm_, smoothCopyParams, smoothPadParams);
        }
    }

    // dequantScaleWDkvKr
    if constexpr (std::is_same<mmInputType, int8_t>::value) {
        DataCopyExtParams dequantCopyParams{
            1, static_cast<uint32_t>((baseParams_->headSizeCkv + baseParams_->dimHeadRope) * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dequantPadParams{false, 0, 0, 0};
        DataCopyPad(dequantScaleWDkvKrLocal_, dequantScaleWDkvkrGm_, dequantCopyParams, dequantPadParams);
    }

    // quantScaleCkv
    if constexpr ((std::is_same<rmsNormCkvOutputType, int8_t>::value) && !isPertile) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value) {
            DataCopyExtParams quantCopyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
            DataCopyPad(quantScaleCkvLocal_, quantScaleCkvGm_, quantCopyParams, quantPadParams);
        } else { // per_channel量化场景走else分支
            DataCopy(quantScaleCkvLocal_, quantScaleCkvGm_, baseParams_->headSizeCkv);
        }
    }

    // quantScaleCkr
    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
        DataCopyExtParams quantCopyParams{1, static_cast<uint32_t>(baseParams_->dimHeadRope * sizeof(float)), 0, 0, 0};
        DataCopyPad(quantScaleCkrLocal_, quantScaleCkrGm_, quantCopyParams, quantPadParams);
    }
}


/**
 * @brief RmsNormCq流程，融合了dynamicquant
          内部所需空间约为 curVecToken(128) * 8*4 + 8*4 + (4*vectorRow_*baseParams_->headSizeCq + 8)*4 = 28.0625K
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset,
                                                             int64_t rmsNormCqResOffset, int64_t curVecToken,
                                                             int64_t curBlockTokenOffset)
{
    if (curVectorBlockNum_ <= blockIdx_) {
        return;
    }
    uint64_t dequantScaleXSize = 1;
    uint64_t dequantScaleCqElementNum = dequantScaleCqSize_ / sizeof(dequantScaleType);
    int64_t stepTokenIndex = tokenIndex;
    uint64_t dequantScaleXElementNum = Align(dequantScaleXSize, BYTE_BLOCK) / sizeof(dequantScaleType);
    LocalTensor<rmsNormCqOutputType> outputLocal = shareBuffer_.Get<rmsNormCqOutputType>();
    LocalTensor<dequantScaleType> dequantScaleQcQr =
        outputLocal[baseParams_->headSizeCq].template ReinterpretCast<dequantScaleType>();
    LocalTensor<dequantScaleType> dequantScaleXLocal =
        dequantScaleQcQr[curVecToken * dequantScaleCqElementNum].template ReinterpretCast<dequantScaleType>();
    LocalTensor<uint8_t> shareTmpUb = dequantScaleXLocal[dequantScaleXElementNum].template ReinterpretCast<uint8_t>();
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx += 1) {
        // MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish

        // dequantScaleXGm_  [BS , 1] 每个每个token对应一个系数，此处扩展为一个DataBlock
        if constexpr (std::is_same<mmCqOutputType, int32_t>::value) {
            DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        RmsNormParam rmsNormParams = {
            baseParams_->reciprocalCq,      // reciprocal
            baseParams_->epsilonCq,         // epsilon
            (uint32_t)vectorRow_,           // row
            baseParams_->headSizeCq,        // col
            baseParams_->qcQrScale,         // scale
            baseParams_->isQcQrScaleEnable, // isScaleEnable
        };
        uint64_t scaleOffset = curVecTokenIdx * dequantScaleCqElementNum;
        if constexpr (std::is_same<rmsNormCqOutputType, bfloat16_t>::value) {
            RmsNormNormal<mmCqOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType, dequantScaleType>(
                outputLocal, mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqLocal_, dequantScaleWDqLocal_,
                dequantScaleXLocal, shareTmpUb, rmsNormParams);
        } else {
            RmsNormDynamicQuant<mmCqOutputType, rmsNormGammaType, float, rmsNormComputType, rmsNormCqOutputType,
                                dequantScaleType>(outputLocal, dequantScaleQcQr[scaleOffset],
                                                  mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqLocal_,
                                                  smoothScaleCqLocal_, dequantScaleWDqLocal_, dequantScaleXLocal,
                                                  shareTmpUb, rmsNormParams, enableSmoothScalesCq_);
        }
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // RmsNorm(Cq)的结果拷进mmCqResGm_中，用于MatmulQcQr的A矩阵
        DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, baseParams_->headSizeCq);

        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        tokenIndex++;
        rmsNormCqResOffset += static_cast<int64_t>(baseParams_->headSizeCq);
        rmsNormCqOffset += static_cast<int64_t>(baseParams_->headSizeCq);
    }

    Brcb(dequantTool_.dequantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM], dequantScaleQcQr, curVecToken,
         {1, 1});
    if constexpr (MLAPT::enableDequantOpt || MLAPT::enableGroupComputeOpt) {
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(dequantTool_.dequantScaleCqGm_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                 dequantTool_.dequantScaleCqLocal_[curBlockTokenOffset * FP32_BLOCK_ELEMENT_NUM],
                 FP32_BLOCK_ELEMENT_NUM * curVecToken);
    }
    if (unlikely(baseParams_->queryNormFlag == 1U)) {
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            DataCopyPad(dequantScaleQNormGm_[stepTokenIndex], dequantScaleQcQr,
                        {static_cast<uint16_t>(curVecToken), sizeof(dequantScaleQNormType), 0, 0});
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ComputeBlkScatterOffsets(GlobalTensor<int64_t> indexGm,
                                                                            int64_t tokenIndex, int64_t rows,
                                                                            CkvkrParams &rmsNormAndScatterCkvParams,
                                                                            CkvkrParams &ropeAndScatterKrParams)
{
    bool spill = false;
    int64_t rowsThisStep = 0;
    int64_t nextPageId = 0;
    int64_t batchTokenIndex = 0;
    int64_t batchSeqSize = 0;
    int64_t batchIndexOffset = 0;

    // Advance batch cursor until the token index falls in [prevPrefix, curPrefix)
    // curPrefix is an exclusive upper bound for the current batch
    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        while (actSeqState_.curBatch + 1 < static_cast<int64_t>(baseParams_->batchSize) &&
               tokenIndex >= actSeqState_.curPrefix) {
            actSeqState_.prevPrefix = actSeqState_.curPrefix;
            actSeqState_.prevIndexOffset = actSeqState_.curIndexOffset;
            actSeqState_.curBatch += 1;
            actSeqState_.curPrefix = actualSeqLenGm_(actSeqState_.curBatch);
            actSeqState_.curIndexOffset += CeilDivT(actSeqState_.curPrefix - actSeqState_.prevPrefix,
                                                    static_cast<int64_t>(baseParams_->blockSize));
        }
        batchTokenIndex = tokenIndex - actSeqState_.prevPrefix;
        batchSeqSize = actSeqState_.curPrefix - actSeqState_.prevPrefix;
        batchIndexOffset = actSeqState_.prevIndexOffset;
    } else {
        const int64_t batchIndex = tokenIndex / baseParams_->seq1Size;
        batchSeqSize = baseParams_->seq1Size;
        batchTokenIndex = tokenIndex % baseParams_->seq1Size;
        batchIndexOffset =
            batchIndex * CeilDivT(batchSeqSize, static_cast<int64_t>(baseParams_->blockSize)); // cacheIndex的offset
    }

    int64_t tokenOffsetInPage = batchTokenIndex % baseParams_->blockSize;

    int64_t leftRowsInPage = baseParams_->blockSize - tokenOffsetInPage;
    int64_t indexOffset = batchIndexOffset + batchTokenIndex / baseParams_->blockSize;
    int64_t paBlkId = indexGm(indexOffset); // 取cacheIdx

    int64_t pageTokenOffset = paBlkId * baseParams_->blockSize;
    if (leftRowsInPage < rows) {
        rowsThisStep = rows - leftRowsInPage;
        spill = true;
        nextPageId = indexGm(indexOffset + 1);
    } else {
        rowsThisStep = rows;
        spill = false;
        nextPageId = -1;
    }


    MaterializeOffsetsWithHeadSize<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
        pageTokenOffset, tokenOffsetInPage, rowsThisStep, spill, nextPageId, baseParams_->dimHeadRope,
        ropeAndScatterKrParams);

    // --- Materialize for RMSNorm/CKV ---
    MaterializeOffsetsWithHeadSize<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
        pageTokenOffset, tokenOffsetInPage, rowsThisStep, spill, nextPageId, baseParams_->headSizeCkv,
        rmsNormAndScatterCkvParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormAndScatterCkv(
    LocalTensor<dequantScaleType> &dequantScaleXLocal, LocalTensor<uint8_t> &shareTmpUb,
    LocalTensor<ropeComputType> &cosLocalCkvKr, LocalTensor<ropeComputType> &sinLocalCkvKr,
    CkvkrParams rmsNormAndScatterCkvParams)
{
    // MatmulCkvKr ──> RmsNorm(Ckv) ──> Scatter(Ckv)
    uint32_t outputLocalSize = vectorRow_ * baseParams_->headSizeCkv * sizeof(kvCacheType);
    LocalTensor<kvCacheType> outputLocal = shareTmpUb.ReinterpretCast<kvCacheType>();
    if constexpr (isPertile) { // pertile量化场景，按照concat的最长长度申请内存
        uint32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
        outputLocalSize = vectorRow_ * (baseParams_->headSizeCkv * sizeof(kvCacheType) + tileNum * sizeof(float));
        outputLocalSize = Align(outputLocalSize, static_cast<uint32_t>(BYTE_BLOCK));
    }

    RmsNormParam rmsNormParams = {
        baseParams_->reciprocalCkv,   // reciprocal
        baseParams_->epsilonCkv,      // epsilon
        (uint32_t)vectorRow_,         // row
        baseParams_->headSizeCkv,     // col
        baseParams_->kcScale,         // scale
        baseParams_->isKcScaleEnable, // isScaleEnable
    };

    LocalTensor<uint8_t> rmsNormShareTmpUb = shareTmpUb[outputLocalSize].template ReinterpretCast<uint8_t>();

    if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
        // row = vectorRow_ = 1     col = baseParams_->headSizeCkv
        LocalTensor<float> inputLocal = rmsNormShareTmpUb.ReinterpretCast<float>();
        LocalTensor<uint8_t> sharedBuf =
            inputLocal[vectorRow_ * baseParams_->headSizeCkv].template ReinterpretCast<uint8_t>();
        RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, float, dequantScaleType>(
            inputLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_,
            dequantScaleWDkvKrLocal_, dequantScaleXLocal, sharedBuf, rmsNormParams);

        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        float kNopeClipAlpha = kNopeClipAlphaGm_.GetValue(0);

        if constexpr (isPertile) {
            PerTileQuantParams perTileQuantParams = {
                static_cast<uint32_t>(baseParams_->tileSize),                            // baseParams_->tileSize
                static_cast<uint32_t>(baseParams_->headSizeCkv / baseParams_->tileSize), // tileNum
                kNopeClipAlpha,                                                          // alpha
                static_cast<uint32_t>(vectorRow_),                                       // row
                baseParams_->headSizeCkv                                                 // col
            };
            QuantPerTile(outputLocal, inputLocal, sharedBuf, perTileQuantParams);
        } else if constexpr (std::is_same<mmCkvKrOutputType, bfloat16_t>::value) {
            Rectangle rectangleParams{
                (uint32_t)vectorRow_,               // row
                (uint32_t)baseParams_->headSizeCkv, // col
                (uint32_t)baseParams_->headSizeCkv  // columnStride
            };
            QuantPerChannel(outputLocal, inputLocal, quantScaleCkvLocal_, sharedBuf, rectangleParams);
        } else {
            Rectangle rectangleParams{
                (uint32_t)vectorRow_,               // row
                (uint32_t)baseParams_->headSizeCkv, // col
                (uint32_t)baseParams_->headSizeCkv  // columnStride
            };
            QuantPerTensor(outputLocal, inputLocal, quantScaleCkvLocal_, sharedBuf, rectangleParams);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else {
        RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType, dequantScaleType>(
            outputLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_,
            dequantScaleWDkvKrLocal_, dequantScaleXLocal, rmsNormShareTmpUb, rmsNormParams);
    }

    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // Scatter(Ckv)
    // RmsNorm(Ckv) ──> Scatter(Ckv) ──> kv_cache_out
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) || (MLAPT::cacheMode == CACHE_MODE::PA_BSND) ||
                  (MLAPT::cacheMode == CACHE_MODE::ND)) {
        int64_t paTokenIndex = 0;
        if constexpr (MLAPT::cacheMode != CACHE_MODE::ND) {
            paTokenIndex = cacheIndexGm_(rmsNormAndScatterCkvParams.tokenIndex);
        } else {
            paTokenIndex = rmsNormAndScatterCkvParams.tokenIndex;
        }
        ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
            kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->headSizeCkv,
                               baseParams_->dtileSize});
        // 刷新量化scale
        if (baseParams_->quantScaleRepoMode == 1U && isPertile) {
            // BSND:
            int64_t startColOffset = baseParams_->headSizeCkv;
            if (baseParams_->ckvkrRepoMode == 1U) {
                startColOffset += baseParams_->headSizeKr * sizeof(krCacheType);
            }
            int64_t startOffset = 0;
            if constexpr ((MLAPT::cacheMode != CACHE_MODE::PA_NZ)) {
                startOffset = startColOffset;
            }

            LocalTensor<kvCacheType> quantScaleCkvInt8Tensor =
                outputLocal[vectorRow_ * baseParams_->headSizeCkv].template ReinterpretCast<kvCacheType>();
            uint32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
            ScatterCacheUnAligned<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                kvCacheGm_[startOffset], quantScaleCkvInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_,
                                   static_cast<int64_t>(sizeof(float) * tileNum), baseParams_->dtileSize});
        }
    } else {
        ScatterCacheMultiRows<kvCacheType, MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ>(
            kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, rmsNormAndScatterCkvParams.cacheOffset, vectorRow_,
                               baseParams_->headSizeCkv, baseParams_->headSizeCkv, baseParams_->seq1Size,
                               rmsNormAndScatterCkvParams.tokenIndex},
            rmsNormAndScatterCkvParams.rowsInCurBatch, rmsNormAndScatterCkvParams.cacheOffset,
            rmsNormAndScatterCkvParams.nextBatchOffset);
    }

    AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeAndScatterKr(LocalTensor<dequantScaleType> &dequantScaleXLocal,
                                                                    LocalTensor<uint8_t> &shareTmpUb,
                                                                    LocalTensor<ropeComputType> &cosLocalCkvKr,
                                                                    LocalTensor<ropeComputType> &sinLocalCkvKr,
                                                                    CkvkrParams ropeAndScatterKrParams)
{
    // MatmulCkvKr ──> Rope(Ckv) ──> Scatter(Kr)
    int64_t stride = static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->headSizeKr); // 512 + 64
    LocalTensor<krCacheType> outputKrLocal = shareTmpUb.ReinterpretCast<krCacheType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputKrLocal[baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();

    LocalTensor<ropeComputType> sinLocal;
    LocalTensor<ropeComputType> cosLocal;
    if constexpr (MLAPT::enableDequantOpt) {
        cosLocal = cosLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    } else {
        cosLocal = cosLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    }
    Rectangle ropeParams{
        (uint32_t)vectorRow_,               // row
        (uint32_t)baseParams_->dimHeadRope, // col
        (uint32_t)stride                    // stride
    };
    if constexpr ((std::is_same<mmCkvKrOutputType, int32_t>::value) && std::is_same<krCacheType, bfloat16_t>::value) {
        LocalTensor<uint8_t> sharedBuf =
            ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope * sizeof(ropeSinCosType)];
        // input为int32_t需在rope中做反量化, intput为float根据模板参数判断是否做反量化
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType, true, MLAPT::enableRope>(
            outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, sharedBuf, ropeParams,
            dequantScaleWDkvKrLocal_[baseParams_->headSizeCkv], dequantScaleXLocal);
    } else if constexpr (std::is_same<krCacheType, int8_t>::value) {
        LocalTensor<uint8_t> sharedBuf =
            ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope * sizeof(ropeSinCosType)];
        LocalTensor<ropeSinCosType> inputLocal = ropeShareTmpUb.ReinterpretCast<ropeSinCosType>();
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, ropeSinCosType,
                              !std::is_same<mmCkvKrOutputType, bfloat16_t>::value, MLAPT::enableRope>(
            inputLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, sharedBuf, ropeParams);
        RopePostQuantPerChannel(outputKrLocal, inputLocal, quantScaleCkrLocal_, sharedBuf,
                                baseParams_->dimHeadRope * vectorRow_);
    } else {
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType, false, MLAPT::enableRope>(
            outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, ropeShareTmpUb,
            ropeParams);
    }
    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // scatter(Kr)
    // Rope(Kr) ──> Scatter(Kr) ──> kr_cache_out
    int64_t paTokenIndex;
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) || (MLAPT::cacheMode == CACHE_MODE::PA_BSND) ||
                  (MLAPT::cacheMode == CACHE_MODE::ND)) {
        if constexpr (MLAPT::cacheMode != CACHE_MODE::ND) {
            paTokenIndex = cacheIndexGm_(ropeAndScatterKrParams.tokenIndex);
        } else {
            paTokenIndex = ropeAndScatterKrParams.tokenIndex;
        }
        if (baseParams_->ckvkrRepoMode == 1U && isPertile) {
            int64_t startOffset;
            if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ)) {
                constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(kvCacheType);
                startOffset = baseParams_->blockSize * col0 * CeilDiv(baseParams_->headSizeCkv, col0); // 列方向的偏移
            } else {
                startOffset = baseParams_->headSizeCkv;
            }
            LocalTensor<kvCacheType> outputKrInt8Tensor = outputKrLocal.template ReinterpretCast<kvCacheType>();
            ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                kvCacheGm_[startOffset], outputKrInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_,
                                   static_cast<int64_t>(sizeof(krCacheType) * baseParams_->dimHeadRope),
                                   baseParams_->dtileSize});
        } else {
            ScatterCache<krCacheType, MLAPT::cacheMode == CACHE_MODE::PA_NZ>(
                krCacheGm_, outputKrLocal,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->dimHeadRope,
                                   baseParams_->dimHeadRope});
        }
    } else {
        ScatterCacheMultiRows<krCacheType, MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ>(
            krCacheGm_, outputKrLocal,
            ScatterCacheParams{baseParams_->blockSize, ropeAndScatterKrParams.cacheOffset, vectorRow_,
                               baseParams_->dimHeadRope, baseParams_->dimHeadRope, baseParams_->seq1Size,
                               ropeAndScatterKrParams.tokenIndex},
            ropeAndScatterKrParams.rowsInCurBatch, ropeAndScatterKrParams.cacheOffset,
            ropeAndScatterKrParams.nextBatchOffset);
    }

    AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset,
                                                                           int64_t ropeKrOffset, int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    LocalTensor<dequantScaleType> dequantScaleXLocal = shareBuffer_.Get<dequantScaleType>();

    LocalTensor<ropeComputType> cosLocalCkvKr =
        dequantScaleXLocal[FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<ropeComputType>();
    LocalTensor<ropeComputType> sinLocalCkvKr = cosLocalCkvKr[curVecToken * baseParams_->dimHeadRope];
    LocalTensor<uint8_t> shareTmpUb =
        sinLocalCkvKr[curVecToken * baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt && MLAPT::enableRope) {
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocalCkvKr, sinLocalCkvKr, ropeCosGm_, ropeSinGm_, tokenIndex,
                                                     curVecToken, shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
    if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return;
    }

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        if constexpr (std::is_same<mmCqOutputType, int32_t>::value) {
            DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        CkvkrParams ropeAndScatterKrParams{tokenIndex, ropeKrOffset, curVecTokenIdx, 0, 0, 0};

        CkvkrParams rmsNormAndScatterCkvParams{tokenIndex, rmsNormCkvOffset, curVecTokenIdx, 0, 0, 0};

        if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_BLK_BSND) || (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)) {
            // --- Compute headSize-independent variables once ---
            ComputeBlkScatterOffsets(cacheIndexGm_, tokenIndex, vectorRow_, rmsNormAndScatterCkvParams,
                                     ropeAndScatterKrParams);
        }

        RmsNormAndScatterCkv(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, rmsNormAndScatterCkvParams);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        RopeAndScatterKr(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, ropeAndScatterKrParams);

        tokenIndex += 1;
        ropeKrOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
        rmsNormCkvOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset,
                                                          int64_t curVecToken, int64_t curBlockTokenOffset)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    uint64_t stride = static_cast<uint64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);

    
    LocalTensor<ropeOutputType> outputLocal;
    LocalTensor<float> channelDeqScaleLocal = shareBuffer_.Get<float>();

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        uint64_t col = baseParams_->dimHeadRope;
        uint64_t row = baseParams_->numHeadSize;
        DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(sizeof(float) * col),
                                     static_cast<uint32_t>(sizeof(float) * (stride - col)), 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[baseParams_->dimHeadSizeQc], copyParams,
                    padParams); // 复用内存
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        outputLocal = channelDeqScaleLocal[col * row].template ReinterpretCast<ropeOutputType>();
    } else {
        outputLocal = shareBuffer_.Get<ropeOutputType>();
    }
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();

    // row, col, stride
    Rectangle ropeParams{(uint32_t)baseParams_->numHeadSize, (uint32_t)baseParams_->dimHeadRope, (uint32_t)stride};

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulQcQr ──> Rope(Qr) ──> query_rope_out
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType, true, MLAPT::enableRope>(
                outputLocal, mmQcQrResGm_[ropeQrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb, ropeParams, channelDeqScaleLocal,
                dequantTool_.dequantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM]);
        } else {
            RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
                outputLocal, mmQcQrResGm_[ropeQrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], ropeShareTmpUb, ropeParams);
        }

        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, baseParams_->headSizeQr);

        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        ropeQrOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                        static_cast<int64_t>(baseParams_->numHeadSize);
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

// 用于enableGroupComputeOpt场景 BS = 1 ， G = 8
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeQrSplitNGroupCase(int64_t ropeQrOffset, int64_t ropeQrResOffset)
{
    if (blockIdx_ < QC_CORE_NUM * cvRatio_ || blockIdx_ >= (QC_CORE_NUM + QR_CORE_NUM) * cvRatio_) {
        return;
    }

    uint32_t col = baseParams_->dimHeadRope;
    uint32_t row = baseParams_->numHeadSize / QC_CORE_NUM;
    int64_t stride = col;
    int64_t strideScale = static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
    int64_t deqScaleOffset = baseParams_->dimHeadSizeQc + row *
                                                              (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                                              (blockIdx_ - QC_CORE_NUM * cvRatio_);
    LocalTensor<float> channelDeqScaleLocal = shareBuffer_.Get<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(sizeof(float) * col),
                                 static_cast<uint32_t>(sizeof(float) * (strideScale - col)), 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(channelDeqScaleLocal, deqScaleQcQrW_[deqScaleOffset], copyParams, padParams); // 复用内存
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    LocalTensor<ropeOutputType> outputLocal =
        channelDeqScaleLocal[row * col].template ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocal[baseParams_->headSizeQr].template ReinterpretCast<uint8_t>();

    Rectangle ropeParams{
        (uint32_t)col,   // row
        (uint32_t)col,   // col
        (uint32_t)stride // stride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVectorBlockNum_; curVecTokenIdx++) {
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        // sharetmp共用，待V算完再执行MTE2搬运规避风险
        if constexpr (MLAPT::enableRope) {
            GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_,
                                                         curVecTokenIdx * baseParams_->dimHeadRope, 1, ropeShareTmpUb,
                                                         vectorRow_, baseParams_->dimHeadRope);
        }

        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

        RotaryPosEmbPerTensor<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
            outputLocal, mmQcQrResGm_[ropeQrOffset], cosLocal_, sinLocal_, ropeShareTmpUb, ropeParams,
            channelDeqScaleLocal, dequantTool_.dequantScaleCqLocal_);

        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, col * row);

        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        ropeQrOffset += static_cast<int64_t>(baseParams_->dimHeadRope) * static_cast<int64_t>(baseParams_->numHeadSize);
        ropeQrResOffset += static_cast<int64_t>(baseParams_->headSizeQr);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeQrSplitN(const RopeQrSplitNParams &ropeQrSplitNParams)
{
    uint32_t colQr = baseParams_->dimHeadRope;
    uint32_t ropeCnt = ropeQrSplitNParams.ropeCnt;
    uint32_t deQuantScaleCqOffset = ropeQrSplitNParams.deQuantScaleCqOffset;
    DataCopyParams outputRopeParams{
        static_cast<uint16_t>(ropeCnt), static_cast<uint16_t>(sizeof(ropeOutputType) * colQr / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>(sizeof(ropeOutputType) * ropeQrSplitNParams.ropeDstStride / ALIGN_BLOCK_SIZE)};

    Rectangle ropeParams{
        ropeCnt,                                // row
        colQr,                                  // col
        (uint32_t)ropeQrSplitNParams.ropeStride // stride
    };

    GlobalTensor<ropeOutputType> outputGmRope = qrOutGm_[ropeQrSplitNParams.ropeQrResOffset];
    GlobalTensor<mmQcQrOutputType> inputGmRope = mmQcQrResGm_[ropeQrSplitNParams.ropeQrOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<ropeOutputType> outputLocalRope = shareTmpUb.ReinterpretCast<ropeOutputType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputLocalRope[colQr * ropeCnt].template ReinterpretCast<uint8_t>();

    AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        GlobalTensor<float> deqScaleRope = deqScaleQcQrW_[ropeQrSplitNParams.ropeQrOffset];
        RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType, true, MLAPT::enableRope>(
            outputLocalRope, inputGmRope[ropeQrSplitNParams.inputOffsetRope],
            cosLocal_[ropeQrSplitNParams.sinCosOffset], sinLocal_[ropeQrSplitNParams.sinCosOffset], ropeShareTmpUb,
            ropeParams, ropeQrSplitNParams.ropeStride, deqScaleRope[ropeQrSplitNParams.deqScaleOffset],
            dequantTool_.dequantScaleCqLocal_[deQuantScaleCqOffset]);
    } else {
        RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
            outputLocalRope, inputGmRope[ropeQrSplitNParams.inputOffsetRope],
            cosLocal_[ropeQrSplitNParams.sinCosOffset], sinLocal_[ropeQrSplitNParams.sinCosOffset], ropeShareTmpUb,
            ropeParams, ropeQrSplitNParams.ropeStride);
    }

    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

    DataCopy(outputGmRope[ropeQrSplitNParams.outputOffsetRope], outputLocalRope, outputRopeParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CastQcQrSplitN(const CastQcQrSplitNParams &castQcQrSplitN)
{
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t row = mmQcQrParam_.m;
    uint32_t colQcSingle = colQc / cvRatio_;
    uint32_t count = colQcSingle * row;

    DataCopyParams inputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(sizeof(mmQcQrOutputType) * colQcSingle / ALIGN_BLOCK_SIZE),
        static_cast<uint16_t>(sizeof(mmQcQrOutputType) * castQcQrSplitN.srcStride / ALIGN_BLOCK_SIZE), 0};

    DataCopyParams outputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(sizeof(mmQnInputType) * colQcSingle / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>(sizeof(mmQnInputType) * castQcQrSplitN.dstStride / ALIGN_BLOCK_SIZE)};


    GlobalTensor<mmQnInputType> outputGm = mmQcQrResDequantGm_[castQcQrSplitN.mmQnPreCastResOffset];
    GlobalTensor<mmQcQrOutputType> inputGm = mmQcQrResGm_[castQcQrSplitN.mmQnPreCastOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    LocalTensor<mmQnInputType> outputLocal = inputLocal.template ReinterpretCast<mmQnInputType>();

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::DataCopy(inputLocal, inputGm[castQcQrSplitN.inputOffset], inputCopyParams);
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    // cast
    AscendC::Cast(outputLocal, inputLocal, RoundMode::CAST_RINT, count);
    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    // copy out
    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::DataCopy(outputGm[castQcQrSplitN.outputOffset], outputLocal, outputCopyParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantQcQrSplitN(const DequantQcQrSplitNParams &dequantQcQrSplitN)
{
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t row = mmQcQrParam_.m;
    uint32_t colQcSingle = colQc / cvRatio_;
    uint32_t count = colQcSingle * row;

    DataCopyParams inputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(sizeof(mmQcQrOutputType) * colQcSingle / ALIGN_BLOCK_SIZE),
        static_cast<uint16_t>(sizeof(mmQcQrOutputType) * dequantQcQrSplitN.srcStride / ALIGN_BLOCK_SIZE), 0};

    DataCopyParams outputCopyParams{
        static_cast<uint16_t>(row), static_cast<uint16_t>(sizeof(mmQnInputType) * colQcSingle / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>(sizeof(mmQnInputType) * dequantQcQrSplitN.dstStride / ALIGN_BLOCK_SIZE)};


    GlobalTensor<mmQcQrOutputType> inputGm = mmQcQrResGm_[dequantQcQrSplitN.mmQnPreDequantOffset];
    GlobalTensor<float> scale1Gm = deqScaleQcQrW_[dequantQcQrSplitN.mmQnPreDequantOffset];
    GlobalTensor<mmQnInputType> outputGm = mmQcQrResDequantGm_[dequantQcQrSplitN.mmQnPreDequantResOffset];

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<dequantScaleType> scale2Local = dequantTool_.dequantScaleCqLocal_;
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    // 可以和inputLocal共享内存地址，减少UB使用
    LocalTensor<float> computeLocal = shareTmpUb.ReinterpretCast<float>();
    LocalTensor<float> scaleLocal = inputLocal[count + FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<float>();
    // outputLocal比scaleLocal占用UB少，且不会同时使用，故可以复用UB内存
    LocalTensor<mmQnInputType> outputLocal = scaleLocal.template ReinterpretCast<mmQnInputType>();

    Rectangle dequantParams{
        row,         //  row
        colQcSingle, // col
        colQcSingle  // columnStride
    };
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    AscendC::DataCopy(inputLocal, inputGm[dequantQcQrSplitN.inputOffset], inputCopyParams);
    AscendC::DataCopy(scaleLocal, scale1Gm[dequantQcQrSplitN.inputOffset], colQc / cvRatio_);
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    // cast
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Dequant(computeLocal, inputLocal, scaleLocal, scale2Local, dequantParams);
    AscendC::PipeBarrier<PIPE_V>();
    // cast
    AscendC::Cast(outputLocal, computeLocal, RoundMode::CAST_RINT, count);
    AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    // copy out
    AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    AscendC::DataCopy(outputGm[dequantQcQrSplitN.outputOffset], outputLocal, outputCopyParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantAndRopeSplitNSyncMMQcQr(int64_t mmQnPreDequantOffset,
                                                                                  int64_t mmQnPreDequantResOffset,
                                                                                  int64_t ropeQrOffset,
                                                                                  int64_t ropeQrResOffset)
{
    if (baseParams_->mm3BlockNum <= cubeBlockIdx_) {
        return;
    }
    // mmQcQr一个C核算stepBatchSize * singleN，每两个V核做对应C核输出的Qc部分的dequant
    // singleN一定整除(dimHeadSizeQc + dimHeadSizeQc)，保证dequant能找到dimHeadSizeQc部分的起始位置
    uint32_t colQr = baseParams_->dimHeadRope;
    uint32_t colQc = baseParams_->dimHeadSizeQc;
    uint32_t colCube = mmQcQrParam_.baseN;
    uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
    uint32_t oriCol = mmQcQrParam_.n;
    uint32_t colOffsetCube = 0;

    // DequantSplitN参数
    uint32_t srcStride = baseParams_->headSizeQr + baseParams_->headSizeQc - baseParams_->dimHeadSizeQc / cvRatio_;
    uint32_t dstStride = baseParams_->headSizeQc - baseParams_->dimHeadSizeQc / cvRatio_;
    uint32_t colQcSingle = colQc / cvRatio_;
    uint32_t colOffsetVec = 0;
    uint32_t inputOffset = subBlockIdx_ * colQcSingle;
    uint32_t outputOffset = inputOffset;

    // RopeQrSplitN参数
    uint32_t ropeDstStride = baseParams_->headSizeQr - colQr;
    // CV1:2和1:1，M轴方向切成2块处理
    uint32_t ropeCnt = (mmQcQrParam_.m + subBlockIdx_) / cvRatio_;
    uint32_t ropeCntDown = mmQcQrParam_.m / 2;
    int64_t ropeStride = static_cast<int64_t>(colQr + colQc) * static_cast<int64_t>(baseParams_->numHeadSize);
    uint32_t outputOffsetRope = ropeCntDown * subBlockIdx_ * baseParams_->headSizeQr;
    uint32_t inputOffsetRope = ropeCntDown * subBlockIdx_ * ropeStride;
    uint32_t colOffsetRope = 0;
    uint32_t deqScaleOffset = 0;
    uint32_t deQuantScaleCqOffset = ropeCntDown * subBlockIdx_ * FP32_BLOCK_ELEMENT_NUM;
    // cube一次处理row*colCube，对应的两个vec一次处理row*colQc，两vec之间切colQc
    // 等cube生产足够数据了以后，vec开始消费
    uint32_t dequantLoopCount = 0;
    uint32_t totalDequantLoops = CeilDiv(oriCol, (colQr + colQc));
    bool needSparseSync = totalDequantLoops > MAX_SYNC_FLAG_COUNT;
    while (colOffsetCube < oriCol) { // 循环CeilDiv(oriCol, colCube)次
        colOffsetCube += colCube;
        if (colOffsetCube > oriCol) { // 当oriCol不被colCube整除时，mm最后一个base块需要刷新col end
            colOffsetCube = oriCol;
        }
        CrossCoreWaitFlag(FINISH_MM_QCQR_SPLIT_N);
        // DequantSplitN
        while (colOffsetVec + colQc <= colOffsetCube) { // 循环singleNumHeadSize次
            if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
                DequantQcQrSplitN(DequantQcQrSplitNParams{mmQnPreDequantOffset, mmQnPreDequantResOffset, inputOffset,
                                                          outputOffset, srcStride, dstStride});
            }
            if (!needSparseSync || dequantLoopCount % 2 == 0 || dequantLoopCount == totalDequantLoops - 1) {
                CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N);
            }
            dequantLoopCount++;
            outputOffset += colQc;
            inputOffset += (colQr + colQc);
            colOffsetVec += (colQr + colQc);
        }
        // RopeQrSplitN
        while ((colOffsetRope + colQc + colQr) <= colOffsetCube) {
            // cv1:1时，为了防止UB内存溢出，每次处理一半的数量
            if (cvRatio_ == 1) {
                ropeCnt = ropeCntDown;
            }
            RopeQrSplitN(RopeQrSplitNParams{ropeQrOffset, ropeQrResOffset, inputOffsetRope, deqScaleOffset,
                                            outputOffsetRope, ropeStride, ropeDstStride, deQuantScaleCqOffset, 0,
                                            ropeCnt});
            // cv1:1时，还需处理第二次，第二次在第一次的基础上计算偏移等
            if (cvRatio_ == 1) {
                RopeQrSplitN(RopeQrSplitNParams{
                    ropeQrOffset, ropeQrResOffset, static_cast<uint32_t>(inputOffsetRope + ropeStride * ropeCntDown),
                    deqScaleOffset, static_cast<uint32_t>(outputOffsetRope + baseParams_->headSizeQr * ropeCntDown),
                    ropeStride, ropeDstStride, ropeCntDown * FP32_BLOCK_ELEMENT_NUM,
                    ropeCntDown * baseParams_->dimHeadRope, (mmQcQrParam_.m + 1) / 2});
            }

            outputOffsetRope += colQr;
            deqScaleOffset += (colQr + colQc);
            inputOffsetRope += (colQr + colQc);
            colOffsetRope += (colQr + colQc);
        }
    }
}

// 用于算力切分dequant切N场景
template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantQcSplitNGroupCase(int64_t mmQnPreDequantOffset,
                                                                            int64_t mmQnPreDequantResOffset,
                                                                            int64_t qcQrScaleOffset)
{
    if (blockIdx_ >= cvRatio_ * QC_CORE_NUM) {
        return;
    }
    uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
    uint32_t curCol = (baseParams_->headSizeQc) / (cvRatio_ * QC_CORE_NUM);
    uint32_t oriCol = (baseParams_->headSizeQc) / QC_CORE_NUM;
    uint32_t dstStride = baseParams_->headSizeQc - curCol;
    uint32_t srcStride = baseParams_->headSizeQc - curCol;
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    Rectangle rectangleParams{
        (uint32_t)mmQcQrParam_.m,         // row
        (uint32_t)curCol,                 // col
        (uint32_t)baseParams_->headSizeQc // columnStride
    };
    DequantSplitNQc(mmQcQrResDequantGm_[mmQnPreDequantResOffset], mmQcQrResGm_[mmQnPreDequantOffset],
                    deqScaleQcQrW_[qcQrScaleOffset], dequantTool_.dequantScaleCqLocal_, shareTmpUb, rectangleParams,
                    oriCol, dstStride, subBlockIdx_);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::DequantQc(int64_t mmQnPreDequantOffset,
                                                             int64_t mmQnPreDequantResOffset, int64_t curVecToken,
                                                             int64_t curBlockTokenOffset)
{
    if (curVectorBlockNum_ <= blockIdx_) {
        return;
    }

    Rectangle rectangleParams{
        static_cast<uint32_t>(baseParams_->stepNumHeadDequant),                      // row
        static_cast<uint32_t>(baseParams_->dimHeadSizeQc),                           // col
        static_cast<uint32_t>(baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc) // columnStride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        DequantPerTokenQc(
            mmQcQrResDequantGm_[mmQnPreDequantResOffset], mmQcQrResGm_[mmQnPreDequantOffset], deqScaleQcQrW_,
            dequantTool_.dequantScaleCqLocal_[(curBlockTokenOffset + curVecTokenIdx) * FP32_BLOCK_ELEMENT_NUM],
            shareBuffer_.Get<uint8_t>(), rectangleParams, baseParams_->numHeadSize);
        mmQnPreDequantOffset += baseParams_->headSizeQr + baseParams_->headSizeQc;
        mmQnPreDequantResOffset += baseParams_->headSizeQc;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::CastQc(int64_t mmQnPreCastOffset, int64_t mmQnPreCastResOffset,
                                                          int64_t curVecToken)
{
    if (curVectorBlockNum_ <= blockIdx_) {
        return;
    }

    Rectangle rectangleCastParams{
        static_cast<uint32_t>(baseParams_->stepNumHeadDequant),                      // row
        static_cast<uint32_t>(baseParams_->dimHeadSizeQc),                           // col
        static_cast<uint32_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) // columnStride
    };

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
        CastPerTokenQc(mmQcQrResDequantGm_[mmQnPreCastResOffset], mmQcQrResGm_[mmQnPreCastOffset], shareTmpUb,
                       rectangleCastParams, baseParams_->numHeadSize);

        mmQnPreCastOffset += baseParams_->headSizeQr + baseParams_->headSizeQc;
        mmQnPreCastResOffset += baseParams_->headSizeQc;
    }
}

template <typename MLAPT>
__aicore__ inline void
MlaPrologVecS1CubS2<MLAPT>::DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset, int64_t curStepBatchSize,
                                                           int64_t numHeadOffset, int64_t mmQnLoops)
{
    // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
    // >> 1 是将curStepBatchSize分到每个vec核上；
    int64_t curStepBatchSizeVec = ((blockIdx_ % cvRatio_) + curStepBatchSize) / cvRatio_;
    if (blockIdx_ >= baseParams_->mm4BlockNum * cvRatio_) {
        return;
    }
    // 等待前面的Qr部分完成
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY = EVENT_ID3;
    constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3;
    constexpr uint32_t DYNAMIC_QUANT_INPUT_READY = EVENT_ID0;

    int64_t blockBatchOffset = (blockIdx_ % cvRatio_) * (curStepBatchSize / cvRatio_);
    int64_t totalSizeCkv =
        static_cast<int64_t>(baseParams_->headSizeCkv) * static_cast<int64_t>(baseParams_->numHeadSize);
    // 由于两个vec核分curStepBatchSize，各处理curStepBatchSize/2，blockIdx_ & 1 表示是否为第二个vec核
    int64_t dynamicQuantQueryOffset =
        blockBatchOffset * totalSizeCkv + numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t dynamicQuantQueryResOffset = batchOffset * totalSizeCkv + blockBatchOffset * totalSizeCkv +
                                         numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t scaleQueryNopeOffset = batchOffset * static_cast<int64_t>(baseParams_->numHeadSize) +
                                   blockBatchOffset * static_cast<int64_t>(baseParams_->numHeadSize) + numHeadOffset;
    int64_t queryOutStride = totalSizeCkv;
    int64_t qrOutputStride =
        static_cast<int64_t>(baseParams_->dimHeadRope) * static_cast<int64_t>(baseParams_->numHeadSize);
    int64_t qrPostProcessResOffset = blockBatchOffset * static_cast<int64_t>(baseParams_->headSizeQr) +
                                     numHeadOffset * static_cast<int64_t>(baseParams_->dimHeadRope) +
                                     batchOffset * static_cast<int64_t>(baseParams_->headSizeQr);


    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();

    float quantScaleCkv = quantScaleCkvGm_.GetValue(0);

    // Dynamic Quant
    AscendC::SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
    AscendC::SetFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);

    // Rope Post Process
    AscendC::SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    // per-head循环
    for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
        CrossCoreWaitFlag(FINISH_MM_QN_SPLIT_N);
        DynamicQuantQnWithMulQr<ropeOutputType, dequantScaleQNopeType, queryOutputType>(
            dequantScaleQNopeGm_[scaleQueryNopeOffset], queryOutGm_[dynamicQuantQueryResOffset],
            qrOutGm_[qrPostProcessResOffset], mmQnResGm_[dynamicQuantQueryOffset], shareTmpUb, curStepBatchSizeVec,
            baseParams_->headSizeCkv, baseParams_->numHeadSize, queryOutStride,
            // Rope Post Process
            qrOutGm_[qrPostProcessResOffset], quantScaleCkv, baseParams_->dimHeadRope, qrOutputStride, cvRatio_);

        scaleQueryNopeOffset += 1;
        dynamicQuantQueryOffset += static_cast<int64_t>(baseParams_->headSizeCkv);
        qrPostProcessResOffset += static_cast<int64_t>(baseParams_->dimHeadRope);
        dynamicQuantQueryResOffset += static_cast<int64_t>(baseParams_->headSizeCkv);
    }
    // Rope Post Process
    AscendC::WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    // Dynamic Quant
    AscendC::WaitFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
    AscendC::WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
}

} // namespace MlaProlog

#endif // MLA_PROLOG_VEC_S1_CUB_S2_H
