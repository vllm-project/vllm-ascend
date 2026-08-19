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
 * \file kernel_mla_prolog_split_m.h
 * \brief
 */

#ifndef KERNEL_MLA_PROLOG_SPLIT_M_H
#define KERNEL_MLA_PROLOG_SPLIT_M_H

#include "mla_prolog_comm.h"
#include "mla_prolog_vector_comm.h"
#include "service_matmul.h"
#include "service_rms_norm.h"
#include "service_gather_sin_cos.h"
#include "service_rotary_position_embedding.h"
#include "service_scatter_cache.h"
#include "service_dequant.h"
#include "service_dynamic_quant_qn_mul_qr.h"
#include "../mla_prolog_tiling_data.h"
#include "../mla_prolog_template_tiling_key.h"

namespace MlaProlog {
template <typename MLAPT>
class MlaPrologV3SplitM {
public:
    static constexpr bool isPertile = MLAPT::isPertile;

    using mmInputType = typename MLAPT::mmInputType;
    using mmQcQrInputType = typename MLAPT::mmQcQrInputType;
    using mmQnInputType = typename MLAPT::mmQnInputType;
    using mmCqOutputType = typename MLAPT::mmCqOutputType;
    using mmCkvKrOutputType = typename MLAPT::mmCkvKrOutputType;
    using mmQcQrOutputType = typename MLAPT::mmQcQrOutputType;
    using mmQnOutputType = typename MLAPT::mmQnOutputType;
    using rmsNormGammaType = typename MLAPT::rmsNormGammaType;
    using rmsNormComputType = typename MLAPT::rmsNormComputType;
    using rmsNormCqOutputType = typename MLAPT::rmsNormCqOutputType;
    using rmsNormCkvOutputType = typename MLAPT::rmsNormCkvOutputType;
    using ropeSinCosType = typename MLAPT::ropeSinCosType;
    using ropeComputType = typename MLAPT::ropeComputType;
    using ropeOutputType = typename MLAPT::ropeOutputType;
    using queryOutputType = typename std::conditional<isPertile, bfloat16_t, typename MLAPT::kvCacheType>::type;
    using kvCacheType = typename MLAPT::kvCacheType;
    using krCacheType = typename MLAPT::krCacheType;
    using dequantScaleQNopeType = typename MLAPT::dequantScaleQNopeType;
    using dequantScaleQNormType = typename MLAPT::dequantScaleQNormType;
    using dequantScaleType = typename MLAPT::dequantScaleType;

    MMParams mmCqParam_;
    MMParams mmCkvKrParam_;
    MMParams mmQcQrParam_;
    MMParams mmQnParam_;

    __aicore__ inline MlaPrologV3SplitM(TPipe *pipe, const optiling::MlaPrologTilingData *__restrict tilingData,
                                        const optiling::MlaPrologBaseParams *__restrict baseParams)
        : pipe_(pipe), tilingData_(tilingData), baseParams_(baseParams)
    {
    }

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
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyGlobalParams();
    __aicore__ inline void OutputInit(__gm__ uint8_t *actualSeqLen, __gm__ uint8_t *queryOut,
                                      __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *dequantScaleQNopeOut,
                                      __gm__ uint8_t *queryNormOut, __gm__ uint8_t *dequantScaleQNormOut);
    __aicore__ inline void ScaleInit(__gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                     __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
                                     __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
                                     __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *kNopeClipAlpha);
    __aicore__ inline void WorkspaceInit(__gm__ uint8_t *workspace);
    __aicore__ inline void MmParamInit();
    __aicore__ inline void MmCqParamInit();
    __aicore__ inline void MmCkvKrParamInit();
    __aicore__ inline void MmQcQrParamInit();
    __aicore__ inline void MmQnParamInit();
    __aicore__ inline void CubeBufferInit();
    __aicore__ inline void VectorBufferInit();
    __aicore__ inline void UpdateStepBatchParams(int64_t curMSize);
    __aicore__ inline void ComputeBlkScatterOffsets(GlobalTensor<int64_t> indexGm, int64_t tokenIndex, int64_t rows,
                                                    CkvkrParams &rmsNormAndScatterCkvParams,
                                                    CkvkrParams &ropeAndScatterKrParams);
    template <bool needQnDynamicQuant>
    __aicore__ inline void AicProcess(AicOffset &aicOffset, int64_t batchOffset, int64_t mmQnLoops);
    template <bool needQnDynamicQuant>
    __aicore__ inline void AivProcess(AivOffset &aivOffset, int64_t batchOffset, int64_t curMSize,
                                      int64_t numHeadOffset, int64_t mmQnLoops);
    template <typename T, typename O, typename S, bool needCheckEmptyTensor = false, bool needCheckAFullLoad = false,
              bool isContinuousCopy = true>
    __aicore__ inline void
    MatmulSplitM(const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm, const GlobalTensor<T> &tensorBGm,
                 const MMParams &mmPara, const UsedBlockParams &mmBlockParams,
                 const GlobalTensor<S> &tensorAScaleGm = {}, const GlobalTensor<S> &tensorBScaleGm = {});
    __aicore__ inline void MatmulQcQr(AicOffset &aicOffset);
    template <bool needQnDynamicQuant>
    __aicore__ inline void MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset, int64_t weightUkOffset,
                                                            int64_t qnResOffset, int64_t mmQnLoops);
    __aicore__ inline void CopyInSinCos(int64_t tokenIndex, int64_t curVecToken, int64_t batchOffset, int64_t curMSize);
    __aicore__ inline void RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset, int64_t rmsNormCqResOffset,
                                     int64_t curVecToken, int64_t curBlockTokenOffset);
    __aicore__ inline void RopeAndScatterKr(LocalTensor<float> &dequantScaleXLocal, LocalTensor<uint8_t> &shareTmpUb,
                                            LocalTensor<typename MLAPT::ropeComputType> &cosLocalCkvKr,
                                            LocalTensor<typename MLAPT::ropeComputType> &sinLocalCkvKr,
                                            CkvkrParams ropeAndScatterKrParams);
    __aicore__ inline void ScatterKr(LocalTensor<krCacheType> &outputKrLocal, CkvkrParams ropeAndScatterKrParams);
    __aicore__ inline void RmsNormAndScatterCkv(LocalTensor<float> &dequantScaleXLocal,
                                                LocalTensor<uint8_t> &shareTmpUb,
                                                LocalTensor<typename MLAPT::ropeComputType> &cosLocalCkvKr,
                                                LocalTensor<typename MLAPT::ropeComputType> &sinLocalCkvKr,
                                                CkvkrParams rmsNormAndScatterCkvParams);
    __aicore__ inline void RmsNormAndQuantizeCkv(LocalTensor<kvCacheType> &outputLocal,
                                                 LocalTensor<uint8_t> &rmsNormShareTmpUb,
                                                 LocalTensor<float> &dequantScaleXLocal, RmsNormParam rmsNormParams,
                                                 CkvkrParams rmsNormAndScatterCkvParams);
    __aicore__ inline void ScatterCkv(LocalTensor<kvCacheType> &outputLocal, CkvkrParams rmsNormAndScatterCkvParams);
    __aicore__ inline void RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset, int64_t ropeKrOffset,
                                                   int64_t curVecToken);
    // 低时延算力分组场景
    __aicore__ inline void QcQrSplit(int64_t curVecToken, int64_t curBlockTokenOffset, int64_t curMSize,
                                     int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset,
                                     int64_t ropeQrOffset, int64_t ropeQrResOffset);
    __aicore__ inline void DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset, int64_t curMSize, int64_t numHeadOffset,
                                                          int64_t mmQnLoops);

private:
    TPipe *pipe_;
    const optiling::MlaPrologTilingData *__restrict tilingData_;
    const optiling::MlaPrologBaseParams *__restrict baseParams_;
    uint32_t blockIdx_ = 0U;
    uint32_t cubeBlockIdx_ = 0U; // AIV上使用AIC的blockIdx
    int64_t vectorRow_ = 1;
    int64_t curVectorBlockNum_;
    int64_t vectorCoreNum_;
    uint64_t dequantScaleCqSize_ = 1;
    uint32_t curStepVecFrontToken_;
    uint32_t curStepVecFrontListNum_;
    uint32_t curStepVecBackToken_;
    uint32_t curVecTokenMax_;
    bool enableSmoothScalesCq_;
    static constexpr uint32_t cvRatio_ = MLAPT::cvRatio; // 默认cv 1:2
    static constexpr bool isFp8E8m0 = std::is_same<dequantScaleType, FP8E8M0>::value;

    uint32_t mSubSize_ = 0;
    uint32_t mOffsetStart_ = 0;

    struct DequantTool {
        GlobalTensor<dequantScaleType> deQuantScaleCqGm_;
        TBuf<TPosition::VECCALC> deQuantScaleCqBuffer_; // 用于临时存储每一行的Scale，以及汇总最终每一行的Scale参数
        LocalTensor<dequantScaleType> deQuantScaleCqLocal_;
        __aicore__ inline DequantTool()
        {
        }
    };

    // 算子分组开关
    DequantTool dequantTool_;

    // GM
    GlobalTensor<mmInputType> tokenXGm_;
    GlobalTensor<mmInputType> weightDqGm_;
    GlobalTensor<mmQcQrInputType> weightUqQrGm_;
    GlobalTensor<mmQnInputType> weightUkGm_;
    GlobalTensor<mmInputType> weightDkvKrGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCqGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCkvGm_;
    GlobalTensor<ropeSinCosType> ropeSinGm_;
    GlobalTensor<ropeSinCosType> ropeCosGm_;
    GlobalTensor<int64_t> cacheIndexGm_;
    GlobalTensor<kvCacheType> kvCacheGm_;
    GlobalTensor<krCacheType> krCacheGm_;
    GlobalTensor<ropeOutputType> qrOutGm_;

    GlobalTensor<dequantScaleType> dequantScaleXGm_;
    GlobalTensor<dequantScaleType> dequantScaleWDqGm_;
    GlobalTensor<dequantScaleType> dequantScaleWDkvkrGm_;
    GlobalTensor<float> smoothScaleCqGm_;
    GlobalTensor<dequantScaleType> deqScaleQcQrW_; // per-channel反量化参数
    GlobalTensor<float> quantScaleCkvGm_;
    GlobalTensor<float> quantScaleCkrGm_;

    GlobalTensor<int32_t> actualSeqLenGm_;
    GlobalTensor<float> kNopeClipAlphaGm_;

    GlobalTensor<rmsNormCqOutputType> rmsNormCqResGm_;
    GlobalTensor<mmCqOutputType> mmCqResGm_;
    GlobalTensor<mmCkvKrOutputType> mmCkvKrResGm_;
    GlobalTensor<mmQcQrOutputType> mmQcQrResGm_;
    GlobalTensor<mmQnInputType> mmQcQrResDequantGm_;
    GlobalTensor<mmQnOutputType> mmQnResGm_;
    GlobalTensor<dequantScaleQNopeType> dequantScaleQNopeGm_;
    GlobalTensor<queryOutputType> queryOutGm_;
    GlobalTensor<dequantScaleQNormType> dequantScaleQNormGm_;

    // UB
    TBuf<TPosition::VECCALC> sincosBuffer_;
    TBuf<TPosition::VECCALC> shareBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDqBuffer_;
    TBuf<TPosition::VECCALC> dequantScaleWDkvKrBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCqBuffer_;
    TBuf<TPosition::VECCALC> rmsnormGammaCkvBuffer_;
    TBuf<TPosition::VECCALC> smoothScaleCqBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkvBuffer_;
    TBuf<TPosition::VECCALC> quantScaleCkrBuffer_;
    TBuf<TPosition::VECCALC> stepActualSeqBuffer_;

    LocalTensor<ropeComputType> cosLocal_;
    LocalTensor<ropeComputType> sinLocal_;
    LocalTensor<float> dequantScaleWDqLocal_;
    LocalTensor<float> dequantScaleWDkvKrLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCqLocal_;
    LocalTensor<rmsNormGammaType> rmsnormGammaCkvLocal_;
    LocalTensor<float> smoothScaleCqLocal_;
    LocalTensor<float> quantScaleCkvLocal_;
    LocalTensor<float> quantScaleCkrLocal_;
    LocalTensor<int64_t> stepActualSeqLocal_;

    struct ActSeqState {
        bool inited = false;         // whether state is initialized for the run
        int64_t curBatch = 0;        // current batch index for the running token index
        int64_t prevPrefix = 0;      // prefix sum of seq lengths up to (curBatch - 1)
        int64_t curPrefix = 0;       // prefix sum of seq lengths up to curBatch (exclusive high bound)
        int64_t prevIndexOffset = 0; // prefix sum of blocks up to (curBatch - 1)
        int64_t curIndexOffset = 0;  // prefix sum of blocks up to curBatch
    };
    ActSeqState actSeqState_;
    int64_t blockSpanPerBatch_ = 0; // blockNum * blockSize

    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::B1> bBufL1_;
    LocalTensor<mmInputType> aL1Tensor_;
    LocalTensor<mmInputType> bL1Tensor_;
    MMBufParams bufParam_;
    TBuf<TPosition::A2> aBufL0_;
    TBuf<TPosition::B2> bBufL0_;
    TBuf<TPosition::CO1> cBufL0_;
    TBuf<TPosition::VECCALC> zeroBuf_;
    LocalTensor<float> zeroSrc_;
};


template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::Init(
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
    if ASCEND_IS_AIV {
        cubeBlockIdx_ = blockIdx_ / cvRatio_;
    } else {
        cubeBlockIdx_ = blockIdx_;
    }
    curVectorBlockNum_ = static_cast<int64_t>(baseParams_->stepBatchSize);
    vectorCoreNum_ = static_cast<int64_t>(baseParams_->vectorBlockNum);
    curVecTokenMax_ = (curVectorBlockNum_ + vectorCoreNum_ - 1) / vectorCoreNum_;
    enableSmoothScalesCq_ = smoothScaleCq == nullptr ? false : true;
    // GM
    tokenXGm_.SetGlobalBuffer((__gm__ mmInputType *)tokenX);
    weightDqGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDq);         // NZ
    weightUqQrGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)weightUqQr); // NZ
    weightUkGm_.SetGlobalBuffer((__gm__ mmQnInputType *)weightUk);
    weightDkvKrGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDkvKr); // NZ
    rmsnormGammaCqGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCq);
    rmsnormGammaCkvGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCkv);
    ropeSinGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeCos);
    if constexpr (MLAPT::cacheMode != CACHE_MODE::ND) {
        cacheIndexGm_.SetGlobalBuffer((__gm__ int64_t *)cacheIndex);
    }
    kvCacheGm_.SetGlobalBuffer((__gm__ kvCacheType *)kvCache);
    krCacheGm_.SetGlobalBuffer((__gm__ krCacheType *)krCache);

    OutputInit(actualSeqLen, queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut);
    ScaleInit(dequantScaleX, dequantScaleWDq, deqScaleQcQrW, dequantScaleWDkvkr, quantScaleCkv, quantScaleCkr,
              smoothScaleCq, kNopeClipAlpha);
    MmParamInit();
    WorkspaceInit(workspace);
    if ASCEND_IS_AIV {
        VectorBufferInit();
    } else {
        CubeBufferInit();
        constexpr int64_t zeroChunk = 512;
        pipe_->InitBuffer(zeroBuf_, zeroChunk * sizeof(float));
        zeroSrc_ = zeroBuf_.Get<float>();
        Duplicate(zeroSrc_, static_cast<float>(0), zeroChunk);
    }
}

template <typename MLAPT>
__aicore__ inline void
MlaPrologV3SplitM<MLAPT>::OutputInit(__gm__ uint8_t *actualSeqLen, __gm__ uint8_t *queryOut,
                                     __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *dequantScaleQNopeOut,
                                     __gm__ uint8_t *queryNormOut, __gm__ uint8_t *dequantScaleQNormOut)
{
    qrOutGm_.SetGlobalBuffer((__gm__ ropeOutputType *)queryRopeOut);
    if constexpr (((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) ||
                   (std::is_same<mmInputType, FP8E4M3>::value && std::is_same<kvCacheType, FP8E4M3>::value) ||
                   (std::is_same<mmInputType, HIF8>::value && std::is_same<kvCacheType, HIF8>::value)) &&
                  !isPertile) {
        dequantScaleQNopeGm_.SetGlobalBuffer((__gm__ dequantScaleQNopeType *)dequantScaleQNopeOut);
        queryOutGm_.SetGlobalBuffer((__gm__ queryOutputType *)queryOut);
    } else {
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)queryOut);
    }
    if (baseParams_->queryNormFlag == 1U) {
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)queryNormOut);
        if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, false>()) {
            dequantScaleQNormGm_.SetGlobalBuffer((__gm__ dequantScaleQNormType *)dequantScaleQNormOut);
        }
    }
    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::EN_Q_LEN) {
        actualSeqLenGm_.SetGlobalBuffer((__gm__ int32_t *)actualSeqLen);
    }
}

template <typename MLAPT>
__aicore__ inline void
MlaPrologV3SplitM<MLAPT>::ScaleInit(__gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                    __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
                                    __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,
                                    __gm__ uint8_t *smoothScaleCq, __gm__ uint8_t *kNopeClipAlpha)
{
    if constexpr (IsFullQuantMode<mmInputType, dequantScaleType, false>()) {
        dequantScaleXGm_.SetGlobalBuffer((__gm__ dequantScaleType *)dequantScaleX);
        dequantScaleWDqGm_.SetGlobalBuffer((__gm__ dequantScaleType *)dequantScaleWDq);
        dequantScaleWDkvkrGm_.SetGlobalBuffer((__gm__ dequantScaleType *)dequantScaleWDkvkr);
    }
    if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
        deqScaleQcQrW_.SetGlobalBuffer((__gm__ dequantScaleType *)deqScaleQcQrW);
        quantScaleCkvGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkv);
    }

    if constexpr (isPertile) {
        kNopeClipAlphaGm_.SetGlobalBuffer((__gm__ float *)kNopeClipAlpha);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmParamInit()
{
    MmCqParamInit();
    MmCkvKrParamInit();
    MmQcQrParamInit();
    MmQnParamInit();
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmCqParamInit()
{
    mmCqParam_.m = baseParams_->stepBatchSize;
    mmCqParam_.n = baseParams_->mm1SingleCoreN; // 1536 / 24 = 64
    mmCqParam_.k = baseParams_->headSizeX;      // 7168
    mmCqParam_.needSetOrgShape = 1;
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.orgN = mmCqParam_.n;
    mmCqParam_.orgKa = mmCqParam_.k;
    mmCqParam_.orgKb = mmCqParam_.k;
    mmCqParam_.orgKc = baseParams_->headSizeCq; // 1536
    mmCqParam_.baseK =
        (sizeof(mmInputType) == ONE_BYTE_TYPE_SIZE) ? 256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCqParam_.baseN = (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) ? 64 : 128;
    mmCqParam_.stepK = 4;
    if ((mmCqParam_.k / mmCqParam_.baseK) % mmCqParam_.stepK != 0) {
        mmCqParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    mmCqParam_.kL1StepSize = mmCqParam_.baseK * mmCqParam_.stepK;
    mmCqParam_.kScale = mmCqParam_.k / FP8_E4M3_BLOCK_SIZE;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmCkvKrParamInit()
{
    mmCkvKrParam_.m = baseParams_->stepBatchSize;
    mmCkvKrParam_.n = baseParams_->mm2SingleCoreN;
    mmCkvKrParam_.k = baseParams_->headSizeX; // 7168
    mmCkvKrParam_.needSetOrgShape = 1;
    mmCkvKrParam_.orgM = mmCkvKrParam_.m;
    mmCkvKrParam_.orgN = mmCkvKrParam_.n;
    mmCkvKrParam_.orgKa = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKb = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKc = (baseParams_->headSizeCkv + baseParams_->dimHeadRope); // 576
    mmCkvKrParam_.baseK =
        (sizeof(mmInputType) == sizeof(int8_t)) ? 256 : 128; // 128KB / (128 max baseN * 4 stepK * sizeof(type))
    mmCkvKrParam_.baseN = (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) ? 64 : 128;
    mmCkvKrParam_.stepK = 4;
    if ((mmCkvKrParam_.k / mmCkvKrParam_.baseK) % mmCkvKrParam_.stepK != 0) {
        mmCkvKrParam_.stepK = 3; // support k = 7680, mmInputType int8, no tail
    }
    mmCkvKrParam_.kL1StepSize = mmCkvKrParam_.baseK * mmCkvKrParam_.stepK;
    mmCkvKrParam_.kScale = mmCkvKrParam_.k / FP8_E4M3_BLOCK_SIZE;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmQcQrParamInit()
{
    mmQcQrParam_.m = baseParams_->stepBatchSize;
    mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
    mmQcQrParam_.k = baseParams_->headSizeCq; // 1536
    mmQcQrParam_.needSetOrgShape = 1;
    mmQcQrParam_.orgM = mmQcQrParam_.m;
    mmQcQrParam_.orgN = mmQcQrParam_.n;
    mmQcQrParam_.orgKa = mmQcQrParam_.k;
    mmQcQrParam_.orgKb = mmQcQrParam_.k;
    mmQcQrParam_.orgKc = (baseParams_->headSizeQc + baseParams_->headSizeQr); // (128 * 32 + 64 * 32)
    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? 128 : 64;
    if constexpr (MLAPT::enableGroupComputeOpt) {
        mmQcQrParam_.baseN = 128;
    } else {
        mmQcQrParam_.baseN = 128;
    }
    mmQcQrParam_.stepK = 4;
    mmQcQrParam_.kL1StepSize = mmQcQrParam_.baseK * mmQcQrParam_.stepK;
    mmQcQrParam_.kScale = mmQcQrParam_.k / FP8_E4M3_BLOCK_SIZE;
}


template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MmQnParamInit()
{
    mmQnParam_.m = baseParams_->stepBatchSize;
    mmQnParam_.n = baseParams_->headSizeCkv;   // 512
    mmQnParam_.k = baseParams_->dimHeadSizeQc; // 128, 这里numHeadSize被分核，matmul设置里不体现
    mmQnParam_.needSetOrgShape = 1;
    mmQnParam_.orgM = mmQnParam_.m;
    mmQnParam_.orgN = mmQnParam_.n;
    if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value || std::is_same<mmQcQrOutputType, float>::value) {
        mmQnParam_.orgKa = baseParams_->headSizeQc;
    } else {
        mmQnParam_.orgKa = baseParams_->headSizeQc + baseParams_->headSizeQr;
    }
    mmQnParam_.orgKb = baseParams_->dimHeadSizeQc;
    mmQnParam_.orgKc = baseParams_->headSizeCkv * baseParams_->numHeadSize;
    mmQnParam_.baseN = 128;
    mmQnParam_.baseK = 128;
    mmQnParam_.stepK = 1;
    if ((mmQnParam_.k > mmQnParam_.baseK) && (mmQnParam_.k % mmQnParam_.baseK != 0)) {
        mmQnParam_.baseK = 64;
        mmQnParam_.stepK = 3; // support D = 192
    }
    mmQnParam_.kL1StepSize = mmQnParam_.baseK * mmQnParam_.stepK;
    mmQnParam_.kScale = mmQnParam_.k / FP8_E4M3_BLOCK_SIZE;
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::VectorBufferInit()
{
    pipe_->InitBuffer(rmsnormGammaCqBuffer_, baseParams_->headSizeCq * sizeof(rmsNormGammaType)); // [1, 1536] bf16
    rmsnormGammaCqLocal_ = rmsnormGammaCqBuffer_.Get<rmsNormGammaType>();

    pipe_->InitBuffer(rmsnormGammaCkvBuffer_, baseParams_->headSizeCkv * sizeof(rmsNormGammaType)); // [1, 512] bf16
    rmsnormGammaCkvLocal_ = rmsnormGammaCkvBuffer_.Get<rmsNormGammaType>();

    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        pipe_->InitBuffer(quantScaleCkrBuffer_, baseParams_->dimHeadRope * sizeof(float)); // [1, 64]
        quantScaleCkrLocal_ = quantScaleCkrBuffer_.Get<float>();
    }

    if constexpr (IsFullQuantMode<rmsNormCkvOutputType, dequantScaleType, false>()) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value ||
                      (std::is_same<mmCkvKrOutputType, float>::value && !isFp8E8m0)) {
            pipe_->InitBuffer(quantScaleCkvBuffer_, ALIGN_BLOCK_SIZE);
        } else {
            pipe_->InitBuffer(quantScaleCkvBuffer_, baseParams_->headSizeCkv * sizeof(float)); // [1, 512]
        }
        quantScaleCkvLocal_ = quantScaleCkvBuffer_.Get<float>();
    }

    // 预留brcb的空间
    pipe_->InitBuffer(dequantTool_.deQuantScaleCqBuffer_, (baseParams_->stepBatchSize + 7) * ALIGN_BLOCK_SIZE);
    dequantTool_.deQuantScaleCqLocal_ = dequantTool_.deQuantScaleCqBuffer_.template Get<dequantScaleType>();

    if constexpr (MLAPT::enableDequantOpt) {
        // 在ropeQr进行切M处理后，会复用shareBuffer的内存，不需要额外申请
        // 开启开关后会按照head切分rope qr，此时需要加载一半batchsize数量的sin和cos值
        // 需要2倍的空间分别存储sin和cos
        pipe_->InitBuffer(sincosBuffer_, 2 * baseParams_->dimHeadRope * sizeof(ropeComputType) *
                                             ((baseParams_->stepBatchSize + 1) >> 1));
    } else {
        // 需要2倍的空间分别存储sin和cos
        pipe_->InitBuffer(sincosBuffer_,
                          2 * baseParams_->dimHeadRope * sizeof(ropeComputType) * curVecTokenMax_); // [2, 64] float
    }

    uint64_t usedAddr;
    if constexpr (MLAPT::enableDequantOpt) {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + 1) >> 1)];
        usedAddr = reinterpret_cast<uint64_t>(
            sinLocal_[baseParams_->dimHeadRope * ((baseParams_->stepBatchSize + 1) >> 1)].GetPhyAddr());
    } else {
        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * curVecTokenMax_];
        usedAddr = reinterpret_cast<uint64_t>(sinLocal_[baseParams_->dimHeadRope * curVecTokenMax_].GetPhyAddr());
    }

    constexpr int64_t zeroChunk = 512;
    pipe_->InitBuffer(zeroBuf_, zeroChunk * sizeof(float));
    zeroSrc_ = zeroBuf_.Get<float>();
    Duplicate(zeroSrc_, static_cast<float>(0), zeroChunk);

    // 由于shareBuffer属于各个vector操作临时申请内存的区域内存使用不固定，建议shareBuffer始终放在最后，防止写入shareBuffer越界导致前面固定申请的UB内存被踩。
    pipe_->InitBuffer(shareBuffer_, MAX_UB_SIZE - usedAddr - zeroChunk * sizeof(float));
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::CubeBufferInit()
{
    // cube相关Buffer初始化
    pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
    pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);

    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
    aL1Tensor_ = aBufL1_.Get<mmInputType>();
    bL1Tensor_ = bBufL1_.Get<mmInputType>();
    bufParam_.aL1BufAddr = aBufL1_.GetBufferAddr(aL1Tensor_.GetBufferHandle());
    bufParam_.bL1BufAddr = bBufL1_.GetBufferAddr(bL1Tensor_.GetBufferHandle());

    pipe_->InitBuffer(aBufL0_, L0A_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(bBufL0_, L0B_PP_SIZE * 2); // 64K
    pipe_->InitBuffer(cBufL0_, L0C_PP_SIZE * 2); // 128K

    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);

    SetFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);

    bufParam_.aL0BufAddr = aBufL0_.GetBufferAddr(aBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.bL0BufAddr = bBufL0_.GetBufferAddr(bBufL0_.Get<mmInputType>().GetBufferHandle());
    bufParam_.cL0BufAddr = cBufL0_.GetBufferAddr(cBufL0_.Get<float>().GetBufferHandle());
}

/*
 * workspace管理
 * 1. 常驻：dequantTool_.deQuantScaleCqGm_ stepBs * 32 Byte
 * 2. 中间结果：
 *      tokenXGm_──────>mmCkvKrResGm_
 *          |           [stepBS, HCkv + Dr]
 *          |           (bf16 | int32)
 *          └─────────>mmCqResGm_──────>rmsNormCqResGm_──────>mmQcQrResGm_──────>mmQcQrResDequantGm_──────>mmQnResGm_
 *                     [stepBS, HCq]    [stepBS, HCq]     [stepBS, N1, D + Dr]   [stepBS, N1, D]           [stepBS, N1,
 * HCkv] (bf16 | int32)   (bf16 | int8)     (bf16 | int32)         (bf16)                    (bf16)
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::WorkspaceInit(__gm__ uint8_t *workspace)
{
    int64_t workspaceOffset = 0;
    if constexpr (std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0) {
        dequantScaleCqSize_ = baseParams_->headSizeCq / FP8_E4M3_BLOCK_SIZE;
    }
    dequantScaleCqSize_ = Align(dequantScaleCqSize_, BYTE_BLOCK);

    if constexpr (MLAPT::enableGroupComputeOpt || MLAPT::enableDequantOpt) {
        dequantTool_.deQuantScaleCqGm_.SetGlobalBuffer((__gm__ dequantScaleType *)(workspace + workspaceOffset));
        workspaceOffset += baseParams_->stepBatchSize * dequantScaleCqSize_ * baseParams_->mm1BlockNum;
    }
    // query_norm_flag有影响
    mmCqResGm_.SetGlobalBuffer((__gm__ mmCqOutputType *)(workspace + workspaceOffset)); // aicOffset.cqResOffset
    if (baseParams_->queryNormFlag == 0U) {
        // 全量化场景下`mmCqResGm_`与`rmsNormCqResGm_`的dtype不同，无法共用workspace，此处需要偏移`mmCqResGm_`占用的大小；
        if constexpr (!std::is_same<rmsNormCqOutputType, mmCqOutputType>::value) {
            workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                               static_cast<int64_t>(baseParams_->headSizeCq) * baseParams_->mm1BlockNum *
                               static_cast<int64_t>(sizeof(mmCqOutputType));
        }
        // 不返回queryNorm时，rmsNormCq结果放在workspace
        rmsNormCqResGm_.SetGlobalBuffer((__gm__ rmsNormCqOutputType *)(workspace + workspaceOffset));
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) * baseParams_->mm1BlockNum *
                           static_cast<int64_t>(baseParams_->headSizeCq) * sizeof(rmsNormCqOutputType);
    } else {
        // 返回queryNorm时，rmsNormCq结果直接输出gm，此处需要偏移`mmCqResGm_`占用的大小
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeCq) * baseParams_->mm1BlockNum *
                           static_cast<int64_t>(sizeof(mmCqOutputType));
    }

    mmCkvKrResGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType *)(workspace + workspaceOffset));
    workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                       static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                       baseParams_->mm2BlockNum * sizeof(mmCkvKrOutputType);

    mmQcQrResGm_.SetGlobalBuffer((__gm__ mmQcQrOutputType *)(workspace + workspaceOffset)); // aicOffset.qcQrResOffset
    
    if constexpr (IsFullQuantMode<mmQcQrInputType, dequantScaleType, false>()) {
        workspaceOffset += static_cast<int64_t>(baseParams_->stepBatchSize) *
                           static_cast<int64_t>(baseParams_->headSizeQc + baseParams_->headSizeQr) *
                           baseParams_->mm3BlockNum * sizeof(mmQcQrOutputType);
    }

    mmQcQrResDequantGm_.SetGlobalBuffer(
        (__gm__ mmQnInputType *)(workspace + workspaceOffset)); // qcOffset mmQnPreDequantResOffset
    workspaceOffset +=
        static_cast<int64_t>(baseParams_->stepBatchSize) * static_cast<int64_t>(baseParams_->numHeadSize) *
        static_cast<int64_t>(baseParams_->dimHeadSizeQc) * baseParams_->mm4BlockNum * sizeof(mmQnInputType);
    if constexpr (((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) ||
                   (std::is_same<mmInputType, FP8E4M3>::value && std::is_same<kvCacheType, FP8E4M3>::value) ||
                   (std::is_same<mmInputType, HIF8>::value && std::is_same<kvCacheType, HIF8>::value)) &&
                  !isPertile) {
        workspaceOffset +=
            static_cast<int64_t>(baseParams_->stepBatchSize) * static_cast<int64_t>(baseParams_->numHeadSize) *
            static_cast<int64_t>(baseParams_->dimHeadSizeQc) * baseParams_->mm4BlockNum * sizeof(mmQnInputType);
        mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)(workspace + workspaceOffset)); // aicOffset.qnResOffset
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::UpdateStepBatchParams(int64_t curMSize)
{
    mmCqParam_.m = curMSize;
    mmCkvKrParam_.m = curMSize;
    mmQcQrParam_.m = curMSize;
    mmQnParam_.m = curMSize;
    curVectorBlockNum_ = curMSize;
}

/*
 * MlaPrologV3算子计算&CV流水同步流程
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
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::Process()
{
    constexpr bool needQnDynamicQuant =
        ((std::is_same<mmInputType, int8_t>::value && std::is_same<kvCacheType, int8_t>::value) ||
         (std::is_same<mmInputType, FP8E4M3>::value && std::is_same<kvCacheType, FP8E4M3>::value) ||
         (std::is_same<mmInputType, HIF8>::value && std::is_same<kvCacheType, HIF8>::value)) &&
        !isPertile;
    int64_t numHeadOffset = 0;
    int64_t mmQnLoops = baseParams_->mm4SingleCoreBatch;

    // moffsetStart_ 当前核处理的token起始偏移
    // mSubSize_ 当前核处理的token数量
    // curMsize 当前step内处理的token数量
    // maxMOffset 当前核处理的token结束偏移
    // mSubCoreNum 前多少个核需要多处理1个token
    if (cubeBlockIdx_ < baseParams_->mSubCoreNum) {
        mSubSize_ = baseParams_->mSubSize;
        mOffsetStart_ = mSubSize_ * cubeBlockIdx_;
    } else {
        mSubSize_ = baseParams_->mSubSize - 1;
        mOffsetStart_ =
            baseParams_->mSubSize * baseParams_->mSubCoreNum + mSubSize_ * (cubeBlockIdx_ - baseParams_->mSubCoreNum);
    }

    // AIC的offset参数
    AicOffset aicOffset;

    // AIV的offset参数
    AivOffset aivOffset;

    // 需要考虑BS合轴的尾块情况
    int64_t maxMOffset = mOffsetStart_ + mSubSize_;
    for (int64_t mOffset = mOffsetStart_; mOffset < maxMOffset; mOffset += baseParams_->stepBatchSize) {
        int64_t curMSize =
            (maxMOffset - mOffset) < baseParams_->stepBatchSize ? (maxMOffset - mOffset) : baseParams_->stepBatchSize;

        UpdateStepBatchParams(curMSize);
        // AIV核用MTE3 DMA清零对应的AIC workspace (UB→GM仅AIV的MTE3支持)
        if ASCEND_IS_AIV {
            if (cubeBlockIdx_ < baseParams_->mm1BlockNum && (blockIdx_ % cvRatio_ == 0)) {
                int64_t stepBS = static_cast<int64_t>(baseParams_->stepBatchSize);
                int64_t cbIdx = static_cast<int64_t>(cubeBlockIdx_);
                LocalTensor<float> zeroSrc = zeroBuf_.Get<float>();
                constexpr int64_t zeroChunk = 512;
                Duplicate(zeroSrc, static_cast<float>(0), zeroChunk);

                int64_t cqN = stepBS * static_cast<int64_t>(baseParams_->headSizeCq);
                int64_t ckvN = stepBS * static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);

                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                for (int64_t off = 0; off < cqN; off += zeroChunk) {
                    int64_t len = (cqN - off < zeroChunk) ? (cqN - off) : zeroChunk;
                    DataCopy(mmCqResGm_[cbIdx * cqN + off], zeroSrc, static_cast<uint16_t>(len));
                }
                for (int64_t off = 0; off < ckvN; off += zeroChunk) {
                    int64_t len = (ckvN - off < zeroChunk) ? (ckvN - off) : zeroChunk;
                    DataCopy(mmCkvKrResGm_[cbIdx * ckvN + off], zeroSrc, static_cast<uint16_t>(len));
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(SYNC_MODE_CUBE_VEC);
        }

        if ASCEND_IS_AIC {
            AicProcess<needQnDynamicQuant>(aicOffset, mOffset, mmQnLoops);
        }
        if ASCEND_IS_AIV {
            AivProcess<needQnDynamicQuant>(aivOffset, mOffset, curMSize, numHeadOffset, mmQnLoops);
        }
    }

    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::AicProcess(AicOffset &aicOffset, int64_t mOffset, int64_t mmQnLoops)
{
    int64_t tokenXOffset = mOffset * static_cast<int64_t>(baseParams_->headSizeX);
    int64_t dequantScaleXOffset = mOffset * static_cast<int64_t>(baseParams_->headSizeX) / 32; // mxfp8需要除32
    aicOffset.weightDqOffset = 0;
    aicOffset.dequantScaleWDqOffset = 0;
    aicOffset.cqResOffset = baseParams_->stepBatchSize * baseParams_->headSizeCq * cubeBlockIdx_;

    CrossCoreWaitFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(SYNC_MODE_CUBE_VEC);

    // MatmulCq ──> RmsNorm(Cq)  [K-outer: kL1→nb→stepK]
    if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
        if (cubeBlockIdx_ < baseParams_->mm1BlockNum) {
            MatmulSplitMKOuter<mmInputType, mmCqOutputType, dequantScaleType>(
                mmCqResGm_[aicOffset.cqResOffset], tokenXGm_[tokenXOffset], weightDqGm_[aicOffset.weightDqOffset],
                mmCqParam_, bufParam_, dequantScaleXGm_[dequantScaleXOffset],
                dequantScaleWDqGm_[aicOffset.dequantScaleWDqOffset]);
        }
    }

    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CQ);

    aicOffset.weightDkvKrOffset = 0;
    aicOffset.dequantScaleWDkvKrOffset = 0;
    aicOffset.ckvKrResOffset =
        baseParams_->stepBatchSize * (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * cubeBlockIdx_;

    // MatmulCkvKr ──> RmsNorm(Ckv) / Rope(Kr)  [K-outer: kL1→nb→stepK]
    if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
        if (cubeBlockIdx_ < baseParams_->mm2BlockNum) {
            MatmulSplitMKOuter<mmInputType, mmCkvKrOutputType, dequantScaleType>(
                mmCkvKrResGm_[aicOffset.ckvKrResOffset], tokenXGm_[tokenXOffset],
                weightDkvKrGm_[aicOffset.weightDkvKrOffset], mmCkvKrParam_, bufParam_,
                dequantScaleXGm_[dequantScaleXOffset], dequantScaleWDkvkrGm_[aicOffset.dequantScaleWDkvKrOffset]);
        }
    }

    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_CKVKR);
    CrossCoreWaitFlag(FINISH_VEC_RMSNORM_CQ);

    // 设置 MatmulQcQr 需要的 offset
    if constexpr (std::is_same<mmInputType, FP8E4M3>::value && isFp8E8m0) {
        if (baseParams_->queryNormFlag == 1U) {
            aicOffset.rmsNormCqResOffset = mOffset * baseParams_->headSizeCq;
        } else {
            aicOffset.rmsNormCqResOffset = baseParams_->stepBatchSize * baseParams_->headSizeCq * cubeBlockIdx_;
        }
        aicOffset.dequantScaleCqOffset = baseParams_->stepBatchSize * cubeBlockIdx_ * dequantScaleCqSize_;
        aicOffset.weightUqQrOffset = 0;
        aicOffset.dequantScaleWuqqrOffset = 0;
        aicOffset.qcQrResOffset =
            baseParams_->stepBatchSize * (baseParams_->headSizeQc + baseParams_->headSizeQr) * cubeBlockIdx_;
        MatmulQcQr(aicOffset);
    }

    if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
        WaitFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
    }

    if constexpr (!needQnDynamicQuant) {
        aicOffset.qnResOffset = mOffset * baseParams_->headSizeCkv * baseParams_->numHeadSize;
    } else {
        aicOffset.qnResOffset =
            baseParams_->stepBatchSize * baseParams_->numHeadSize * baseParams_->headSizeCkv * cubeBlockIdx_;
    }

    aicOffset.qcOffset =
        baseParams_->stepBatchSize * baseParams_->numHeadSize * baseParams_->dimHeadSizeQc * cubeBlockIdx_;
    aicOffset.weightUkOffset = 0;
    MatmulQnSyncDynamicQuantAndMulQr<needQnDynamicQuant>(aicOffset.qcOffset, aicOffset.weightUkOffset,
                                                         aicOffset.qnResOffset, mmQnLoops);
    if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
        SetFlag<HardEvent::MTE1_MTE2>(SCALE_EVENT);
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::AivProcess(AivOffset &aivOffset, int64_t mOffset, int64_t curMSize,
                                                            int64_t numHeadOffset, int64_t mmQnLoops)
{
    if (mOffset == mOffsetStart_) {
        // 只需要搬运一次
        CopyGlobalParams();
    }

    uint64_t isBackVec = blockIdx_ % cvRatio_;
    curStepVecBackToken_ = curMSize / cvRatio_;
    curStepVecFrontListNum_ = curMSize % cvRatio_;

    aivOffset.curVecToken =
        curStepVecBackToken_ + isBackVec * curStepVecFrontListNum_;   // 当前核中每个vec处理多少个token
    aivOffset.curBlockTokenOffset = isBackVec * curStepVecBackToken_; // 当前核中每个vec在这个m中的偏移

    int64_t tokenIndex = mOffset + aivOffset.curBlockTokenOffset;
    int64_t curTokenStart = mOffset + aivOffset.curBlockTokenOffset;
    int64_t curTokenEnd = mOffset + aivOffset.curBlockTokenOffset + aivOffset.curVecToken;

    CopyInSinCos(curTokenStart, aivOffset.curVecToken, mOffset, curMSize);

    CrossCoreWaitFlag(FINISH_MM_CQ);

    aivOffset.rmsNormCqOffset = baseParams_->stepBatchSize * baseParams_->headSizeCq * cubeBlockIdx_ +
                                baseParams_->headSizeCq * aivOffset.curBlockTokenOffset;
    int64_t rmsNormCqResOffset;
    if (baseParams_->queryNormFlag == 1U) {
        rmsNormCqResOffset = tokenIndex * baseParams_->headSizeCq;
    } else {
        rmsNormCqResOffset = aivOffset.rmsNormCqOffset;
    }
    RmsNormCq(curTokenStart, aivOffset.rmsNormCqOffset, rmsNormCqResOffset, aivOffset.curVecToken,
              aivOffset.curBlockTokenOffset);

    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_RMSNORM_CQ);
    CrossCoreWaitFlag(FINISH_MM_CKVKR);

    aivOffset.rmsNormCkvOffset =
        baseParams_->stepBatchSize * (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * cubeBlockIdx_ +
        (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * aivOffset.curBlockTokenOffset;

    aivOffset.ropeKrOffset = baseParams_->headSizeCkv + aivOffset.rmsNormCkvOffset; // 512 + (512 + 64) * idx
    RmsNormRopeScatterCkvKr(tokenIndex, aivOffset.rmsNormCkvOffset, aivOffset.ropeKrOffset, aivOffset.curVecToken);

    aivOffset.mmQnPreDequantOffset =
        ((baseParams_->stepBatchSize) * (baseParams_->headSizeQc + baseParams_->headSizeQr) * cubeBlockIdx_) +
        (baseParams_->headSizeQc + baseParams_->headSizeQr) * aivOffset.curBlockTokenOffset;

    aivOffset.mmQnPreDequantResOffset =
        baseParams_->stepBatchSize * baseParams_->numHeadSize * baseParams_->dimHeadSizeQc * cubeBlockIdx_ +
        baseParams_->dimHeadSizeQc * baseParams_->numHeadSize * aivOffset.curBlockTokenOffset;
    aivOffset.ropeQrOffset =
        (baseParams_->stepBatchSize * (baseParams_->headSizeQc + baseParams_->headSizeQr) * cubeBlockIdx_) +
        baseParams_->dimHeadSizeQc +
        (baseParams_->headSizeQc + baseParams_->headSizeQr) * aivOffset.curBlockTokenOffset;
    aivOffset.ropeQrResOffset = (baseParams_->headSizeQr) * (aivOffset.curBlockTokenOffset + mOffset);

    QcQrSplit(aivOffset.curVecToken, aivOffset.curBlockTokenOffset, curMSize, aivOffset.mmQnPreDequantOffset,
              aivOffset.mmQnPreDequantResOffset, aivOffset.ropeQrOffset, aivOffset.ropeQrResOffset);

    if constexpr (needQnDynamicQuant) {
        DynamicQuantQnAndMulQrSyncMMQn(mOffset, curMSize, numHeadOffset, mmQnLoops);
    }
}

template <typename MLAPT>
template <typename T, typename O, typename S, bool needCheckEmptyTensor, bool needCheckAFullLoad, bool isContinuousCopy>
__aicore__ inline void
MlaPrologV3SplitM<MLAPT>::MatmulSplitM(const GlobalTensor<O> &tensorResGm, const GlobalTensor<T> &tensorAGm,
                                       const GlobalTensor<T> &tensorBGm, const MMParams &mmPara,
                                       const UsedBlockParams &mmBlockParams, const GlobalTensor<S> &tensorAScaleGm,
                                       const GlobalTensor<S> &tensorBScaleGm)
{
    if constexpr (needCheckEmptyTensor) {
        if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
            return;
        }
    }
    if (blockIdx_ < mmBlockParams.blockStartIdx || blockIdx_ >= mmBlockParams.blockEndIdx) {
        return;
    }
    // 用于enableGroupComputeOpt场景
    if constexpr (needCheckAFullLoad) {
        constexpr uint32_t mSize =
            (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
        bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;
        if (isAFullLoad) {
            MatmulGroupComputeAFullLoad<T, O, S, isContinuousCopy>(tensorResGm, tensorAGm, tensorBGm, mmPara,
                                                                   bufParam_);
            return;
        }
    }

    uint32_t nInput = mmPara.n;
    uint32_t nL1SplitSize = mmPara.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        MatmulSplitK<T, O, S>(tensorResGm, tensorAGm, tensorBGm, mmPara, bufParam_, nL1 * nL1SplitSize, subNL1SplitSize,
                              tensorAScaleGm, tensorBScaleGm);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::MatmulQcQr(AicOffset &aicOffset)
{
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    // RmsNorm(Cq) ──> MatmulQcQr ──> MatmulQn
    //                           └──> Rope(Qr)
    // [32, 1536] * [1536, 32*(128+64)] = [32, 32*192]
    constexpr uint32_t mSize =
        (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? INT8_AFULLLOAD_MAX_MSIZE : BF16_AFULLLOAD_MAX_MSIZE;
    bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;

    uint32_t nInput = mmQcQrParam_.n;
    uint32_t nL1SplitSize = mmQcQrParam_.baseN;
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t subNL1SplitSize = nL1SplitSize;
    if (isAFullLoad) {
        if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
            uint32_t offsetL1B = L1_B_SIZE / 2 / sizeof(rmsNormCqOutputType);
            LoadL1AAndScale<rmsNormCqOutputType, dequantScaleType, false, true>(
                rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                dequantTool_.deQuantScaleCqGm_[aicOffset.dequantScaleCqOffset], mmQcQrParam_.m, mmQcQrParam_.k,
                mmQcQrParam_.k, mmQcQrParam_.kScale, offsetL1B, bufParam_);
        }
        WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
    }

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
            if (isAFullLoad) {
                MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, true, true>(
                    mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                    weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nL1 * nL1SplitSize,
                    subNL1SplitSize, dequantTool_.deQuantScaleCqGm_[aicOffset.dequantScaleCqOffset],
                    deqScaleQcQrW_[aicOffset.dequantScaleWuqqrOffset]);
            } else {
                MatmulSplitK<rmsNormCqOutputType, mmQcQrOutputType, dequantScaleType, false, true>(
                    mmQcQrResGm_[aicOffset.qcQrResOffset], rmsNormCqResGm_[aicOffset.rmsNormCqResOffset],
                    weightUqQrGm_[aicOffset.weightUqQrOffset], mmQcQrParam_, bufParam_, nL1 * nL1SplitSize,
                    subNL1SplitSize, dequantTool_.deQuantScaleCqGm_[aicOffset.dequantScaleCqOffset],
                    deqScaleQcQrW_[aicOffset.dequantScaleWuqqrOffset]);
            }
        }
        if constexpr (MLAPT::enableDequantOpt) {
            uint32_t spacing = CeilDivT(nL1loops, static_cast<uint32_t>(MAX_SYNC_FLAG_COUNT - 1));
            if (nL1 % spacing == 0 || (nL1 + 1) == static_cast<int64_t>(nL1loops)) {
                CrossCoreSetFlag<0x2, PIPE_FIX>(FINISH_MM_QCQR_SPLIT_N);
            }
            if ((nL1 + 1) % spacing == 0 || (nL1 + 1) == static_cast<int64_t>(nL1loops)) {
                CrossCoreSetFlag<0x2, PIPE_FIX>(FINISH_MM_QCQR_SPLIT_BATCH);
            }
        }
    }
    if (isAFullLoad) {
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam_.aL1BufIter & 1u));
        bufParam_.aL1BufIter++;
    }
}

template <typename MLAPT>
template <bool needQnDynamicQuant>
__aicore__ inline void
MlaPrologV3SplitM<MLAPT>::MatmulQnSyncDynamicQuantAndMulQr(int64_t qcOffset, int64_t weightUkOffset,
                                                           int64_t qnResOffset, int64_t subLoopTimes)
{
    if (blockIdx_ >= baseParams_->mm4BlockNum) {
        return;
    }

    // MatmulQcQr ──> MatmulQn ──> query_out
    // [32, 128] * [128, 512] = [32, 512]
    // [32, 2, 128] * [2, 128, 512] = [32, 2, 512]

    for (int64_t i = 0; i < subLoopTimes; i++) {
        uint32_t coarse = CeilDivT(static_cast<uint32_t>(subLoopTimes), static_cast<uint32_t>(MAX_SYNC_FLAG_COUNT - 1));
        if (i % coarse == 0) {
            CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC_SPLIT_N);
        } else if ((subLoopTimes - CeilDivT(static_cast<uint32_t>(subLoopTimes), coarse)) <= MAX_SYNC_FLAG_COUNT) {
            CrossCoreWaitFlag(FINISH_VEC_DEQUANT_QC_SPLIT_N_GAP);
        }

        // Qn 用 MatmulSplitK 避免 L0A 溢出：每 L0 步 128×64=8192<16384，bf16 无 scale
        for (int64_t nL1 = 0; nL1 < mmQnParam_.n; nL1 += mmQnParam_.baseN) {
            uint32_t subN = static_cast<uint32_t>((nL1 + mmQnParam_.baseN > mmQnParam_.n) ? (mmQnParam_.n - nL1) :
                                                                                            mmQnParam_.baseN);
            MatmulSplitK<mmQnInputType, mmQnOutputType, dequantScaleType, false, false, true, DataFormat::ND>(
                mmQnResGm_[qnResOffset], mmQcQrResDequantGm_[qcOffset], weightUkGm_[weightUkOffset], mmQnParam_,
                bufParam_, static_cast<uint32_t>(nL1), subN);
        }

        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value || std::is_same<mmQcQrInputType, FP8E4M3>::value ||
                      std::is_same<mmQcQrOutputType, float>::value) {
            qcOffset += static_cast<int64_t>(baseParams_->dimHeadSizeQc);
        }
        weightUkOffset +=
            static_cast<int64_t>(baseParams_->dimHeadSizeQc) * static_cast<int64_t>(baseParams_->headSizeCkv);
        qnResOffset += static_cast<int64_t>(baseParams_->headSizeCkv);
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (needQnDynamicQuant) {
        CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_FIX>(FINISH_MM_QN_SPLIT_N);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::CopyInSinCos(int64_t tokenIndex, int64_t curVecToken,
                                                              int64_t batchOffset, int64_t curMSize)
{
    if constexpr (!MLAPT::enableRope) {
        return;
    }
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt) {
        // 如果是切N场景，mm3的每个C核都会做rope
        if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
            return;
        }
        // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
        // >> 1 是将curStepBatchSize分到每个vec核上；
        uint32_t subBlockIdx_ = blockIdx_ % cvRatio_;
        int64_t offset = (curMSize / cvRatio_) * subBlockIdx_ + batchOffset;
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, offset,
                                                     (curMSize + cvRatio_ - 1) / cvRatio_, shareTmpUb, vectorRow_,
                                                     baseParams_->dimHeadRope);
    } else {
        if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
            return;
        }
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocal_, sinLocal_, ropeCosGm_, ropeSinGm_, tokenIndex,
                                                     curVecToken, shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::CopyGlobalParams()
{
    // rmsnormGammaCq
    DataCopy(rmsnormGammaCqLocal_, rmsnormGammaCqGm_, baseParams_->headSizeCq);

    // rmsnormGammaCkv
    DataCopy(rmsnormGammaCkvLocal_, rmsnormGammaCkvGm_, baseParams_->headSizeCkv);

    // quantScaleCkv
    
    if constexpr (IsFullQuantMode<rmsNormCkvOutputType, dequantScaleType, false>() && !isPertile) {
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value ||
                      (std::is_same<mmCkvKrOutputType, float>::value && !isFp8E8m0)) {
            DataCopyExtParams quantCopyParams{1, sizeof(float), 0, 0, 0};
            DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
            DataCopyPad(quantScaleCkvLocal_, quantScaleCkvGm_, quantCopyParams, quantPadParams);
        } else {
            DataCopy(quantScaleCkvLocal_, quantScaleCkvGm_, baseParams_->headSizeCkv);
        }
    }

    // quantScaleCkr
    if constexpr (std::is_same<krCacheType, int8_t>::value) {
        DataCopyExtParams quantCopyParams{1, static_cast<uint32_t>(baseParams_->dimHeadRope * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> quantPadParams{false, 0, 0, 0};
        DataCopyPad(quantScaleCkrLocal_, quantScaleCkrGm_, quantCopyParams, quantPadParams);
    }
}


/**
 * @brief RmsNormCq流程，融合了dynamicquant
          内部所需空间约为 curVecToken(128) * 8*4 + 8*4 + (4*vectorRow_*baseParams_->headSizeCq + 8)*4 = 28.0625K
 */
template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset,
                                                           int64_t rmsNormCqResOffset, int64_t curVecToken,
                                                           int64_t curBlockTokenOffset)
{
    if (blockIdx_ < 0 || blockIdx_ >= cvRatio_ * baseParams_->mm1BlockNum) {
        return;
    }

    uint64_t dequantScaleCqElementNum = dequantScaleCqSize_ / sizeof(dequantScaleType);
    LocalTensor<rmsNormCqOutputType> outputLocal = shareBuffer_.Get<rmsNormCqOutputType>();
    LocalTensor<dequantScaleType> dequantScaleQcQr =
        outputLocal[baseParams_->headSizeCq].template ReinterpretCast<dequantScaleType>();
    LocalTensor<float> dequantScaleXLocal =
        dequantScaleQcQr[curVecToken * dequantScaleCqElementNum].template ReinterpretCast<float>();

    uint64_t dequantScaleXSize = 1;
    if constexpr (std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0) {
        dequantScaleXSize = baseParams_->headSizeX / FP8_E4M3_BLOCK_SIZE;
    }
    uint64_t dequantScaleXElementNum = Align(dequantScaleXSize, BYTE_BLOCK) / sizeof(float);
    LocalTensor<uint8_t> shareTmpUb = dequantScaleXLocal[dequantScaleXElementNum].template ReinterpretCast<uint8_t>();
    int64_t stepTokenIndex = tokenIndex;
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish
        uint64_t scaleOffset;
        if constexpr (std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0) {
            scaleOffset = curVecTokenIdx * dequantScaleCqElementNum;
        } else {
            scaleOffset = curVecTokenIdx * FP32_BLOCK_ELEMENT_NUM;
        }
        RmsNormParam rmsNormParams = {baseParams_->reciprocalCq,         // reciprocal
                                      baseParams_->epsilonCq,            // epsilon
                                      static_cast<uint32_t>(vectorRow_), // row
                                      baseParams_->headSizeCq,           // col
                                      baseParams_->qcQrScale,
                                      baseParams_->isQcQrScaleEnable};

                                      
        if constexpr (IsFullQuantMode<rmsNormCqOutputType, dequantScaleType, false>()) {
            RmsNormDynamicQuant<mmCqOutputType, rmsNormGammaType, float, rmsNormComputType, rmsNormCqOutputType,
                                dequantScaleType>(outputLocal, dequantScaleQcQr[scaleOffset],
                                                  mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqLocal_,
                                                  smoothScaleCqLocal_, dequantScaleWDqLocal_, dequantScaleXLocal,
                                                  shareTmpUb, rmsNormParams, enableSmoothScalesCq_);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        // RmsNorm(Cq)的结果拷进mmCqResGm_中，用于MatmulQcQr的A矩阵
        DataCopy(rmsNormCqResGm_[rmsNormCqResOffset], outputLocal, baseParams_->headSizeCq);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        rmsNormCqOffset += baseParams_->headSizeCq;
        rmsNormCqResOffset += baseParams_->headSizeCq;
        tokenIndex++;
    }

    if constexpr (std::is_same<rmsNormCqOutputType, FP8E4M3>::value && isFp8E8m0) {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(dequantTool_.deQuantScaleCqGm_[(curBlockTokenOffset + baseParams_->stepBatchSize * cubeBlockIdx_) *
                                                dequantScaleCqElementNum],
                 dequantScaleQcQr, curVecToken * dequantScaleCqElementNum);
    }
    if (unlikely(baseParams_->queryNormFlag == 1U)) {
        if constexpr (std::is_same<mmQcQrInputType, FP8E4M3>::value && isFp8E8m0) {
            DataCopyPad(
                dequantScaleQNormGm_[stepTokenIndex *
                                     static_cast<uint16_t>((baseParams_->headSizeCq / FP8_E4M3_BLOCK_SIZE))],
                dequantScaleQcQr,
                {static_cast<uint16_t>(curVecToken),
                 static_cast<uint16_t>(sizeof(dequantScaleQNormType) * (baseParams_->headSizeCq / FP8_E4M3_BLOCK_SIZE)),
                 0, 0});
        }
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::ComputeBlkScatterOffsets(GlobalTensor<int64_t> indexGm,
                                                                          int64_t tokenIndex, int64_t rows,
                                                                          CkvkrParams &rmsNormAndScatterCkvParams,
                                                                          CkvkrParams &ropeAndScatterKrParams)
{
    int64_t batchTokenIndex = 0;
    int64_t batchIndexOffset = 0;
    int64_t batchSeqSize = 0;
    int64_t rowsThisStep = 0;
    int64_t nextPageId = 0;
    bool spill = false;

    if constexpr (MLAPT::actualSeqMode == ACTUAL_SEQ_MODE::DISABLED) {
        const int64_t batchIndex = tokenIndex / baseParams_->seq1Size;
        batchTokenIndex = tokenIndex % baseParams_->seq1Size;
        batchSeqSize = baseParams_->seq1Size;
        batchIndexOffset = batchIndex * CeilDivT(batchSeqSize, static_cast<int64_t>(baseParams_->blockSize));
    } else {
        while (actSeqState_.curBatch + 1 < static_cast<int64_t>(baseParams_->batchSize) &&
               tokenIndex >= actSeqState_.curPrefix) {
            actSeqState_.curBatch += 1;
            actSeqState_.prevPrefix = actSeqState_.curPrefix;
            actSeqState_.curPrefix = actualSeqLenGm_(actSeqState_.curBatch);
            actSeqState_.prevIndexOffset = actSeqState_.curIndexOffset;
            actSeqState_.curIndexOffset += CeilDivT(actSeqState_.curPrefix - actSeqState_.prevPrefix,
                                                    static_cast<int64_t>(baseParams_->blockSize));
        }
        batchTokenIndex = tokenIndex - actSeqState_.prevPrefix;
        batchSeqSize = actSeqState_.curPrefix - actSeqState_.prevPrefix;
        batchIndexOffset = actSeqState_.prevIndexOffset;
    }

    int64_t indexOffset = batchIndexOffset + batchTokenIndex / baseParams_->blockSize;
    int64_t paBlkId = indexGm(indexOffset);

    int64_t tokenOffsetInPage = batchTokenIndex % baseParams_->blockSize;

    int64_t leftRowsInPage = baseParams_->blockSize - tokenOffsetInPage;
    if (leftRowsInPage >= rows) {
        rowsThisStep = rows;
        spill = false;
        nextPageId = -1;
    } else {
        rowsThisStep = rows - leftRowsInPage;
        spill = true;
        nextPageId = indexGm(indexOffset + 1);
    }

    MaterializeOffsetsWithHeadSize<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
        paBlkId, tokenOffsetInPage, rowsThisStep, spill, nextPageId, baseParams_->headSizeCkv,
        baseParams_->kvCacheStride0,
        rmsNormAndScatterCkvParams);

    MaterializeOffsetsWithHeadSize<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
        paBlkId, tokenOffsetInPage, rowsThisStep, spill, nextPageId, baseParams_->dimHeadRope,
        baseParams_->krCacheStride0,
        ropeAndScatterKrParams);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RmsNormAndScatterCkv(LocalTensor<float> &dequantScaleXLocal,
                                                                      LocalTensor<uint8_t> &shareTmpUb,
                                                                      LocalTensor<ropeComputType> &cosLocalCkvKr,
                                                                      LocalTensor<ropeComputType> &sinLocalCkvKr,
                                                                      CkvkrParams rmsNormAndScatterCkvParams)
{
    // MatmulCkvKr ──> RmsNorm(Ckv) ──> Scatter(Ckv)
    LocalTensor<kvCacheType> outputLocal = shareTmpUb.ReinterpretCast<kvCacheType>();
    int32_t outputLocalSize = vectorRow_ * baseParams_->headSizeCkv * sizeof(kvCacheType);
    if constexpr (isPertile) { // pertile量化场景，按照concat的最长长度申请内存
        int32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
        outputLocalSize = vectorRow_ * (baseParams_->headSizeCkv * sizeof(kvCacheType) + tileNum * sizeof(float));
        outputLocalSize = Align(outputLocalSize, static_cast<int32_t>(BYTE_BLOCK));
    }
    LocalTensor<uint8_t> rmsNormShareTmpUb = shareTmpUb[outputLocalSize].template ReinterpretCast<uint8_t>();

    RmsNormParam rmsNormParams = {
        baseParams_->reciprocalCkv,        // reciprocal
        baseParams_->epsilonCkv,           // epsilon
        static_cast<uint32_t>(vectorRow_), // row
        baseParams_->headSizeCkv,          // col
        baseParams_->kcScale,              // scale
        baseParams_->isKcScaleEnable,      // isScaleEnable
    };

    if constexpr (IsFullQuantMode<rmsNormCkvOutputType, dequantScaleType, false>()) {
        RmsNormAndQuantizeCkv(outputLocal, rmsNormShareTmpUb, dequantScaleXLocal, rmsNormParams,
                              rmsNormAndScatterCkvParams);
    } else {
        RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType, dequantScaleType>(
            outputLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_,
            dequantScaleWDkvKrLocal_, dequantScaleXLocal, rmsNormShareTmpUb, rmsNormParams);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // Scatter(Ckv)
    // RmsNorm(Ckv) ──> Scatter(Ckv) ──> kv_cache_out
    ScatterCkv(outputLocal, rmsNormAndScatterCkvParams);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RmsNormAndQuantizeCkv(LocalTensor<kvCacheType> &outputLocal,
                                                                       LocalTensor<uint8_t> &rmsNormShareTmpUb,
                                                                       LocalTensor<float> &dequantScaleXLocal,
                                                                       RmsNormParam rmsNormParams,
                                                                       CkvkrParams rmsNormAndScatterCkvParams)
{
    // row = vectorRow_ = 1     col = baseParams_->headSizeCkv
    LocalTensor<float> inputLocal = rmsNormShareTmpUb.ReinterpretCast<float>();
    LocalTensor<uint8_t> sharedBuf =
        inputLocal[vectorRow_ * baseParams_->headSizeCkv].template ReinterpretCast<uint8_t>();
    RmsNormNormal<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, float, dequantScaleType>(
        inputLocal, mmCkvKrResGm_[rmsNormAndScatterCkvParams.offset], rmsnormGammaCkvLocal_, dequantScaleWDkvKrLocal_,
        dequantScaleXLocal, sharedBuf, rmsNormParams);

    DataSyncBarrier<MemDsbT::UB>();

    if constexpr (isPertile) {
        float kNopeClipAlpha = isFp8E8m0 ? 1.0f : kNopeClipAlphaGm_.GetValue(0);
        PerTileQuantParams perTileQuantParams = {
            static_cast<uint32_t>(baseParams_->tileSize),                            // baseParams_->tileSize
            static_cast<uint32_t>(baseParams_->headSizeCkv / baseParams_->tileSize), // tileNum
            kNopeClipAlpha,                                                          // alpha
            static_cast<uint32_t>(vectorRow_),                                       // row
            baseParams_->headSizeCkv                                                 // col
        };
        if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
            QuantPerTile(outputLocal, inputLocal, sharedBuf, perTileQuantParams);
        } else {
            QuantPerTile8Bit(outputLocal, inputLocal, perTileQuantParams);
        }
    } else {
        Rectangle rectangleParams{
            static_cast<uint32_t>(vectorRow_),               // row
            static_cast<uint32_t>(baseParams_->headSizeCkv), // col
            static_cast<uint32_t>(baseParams_->headSizeCkv)  // columnStride
        };
        if constexpr (std::is_same<mmCkvKrOutputType, int32_t>::value ||
                         std::is_same<mmCkvKrOutputType, float>::value) {
            QuantPerTensor(outputLocal, inputLocal, quantScaleCkvLocal_, sharedBuf, rectangleParams);
        } else {
            QuantPerChannel(outputLocal, inputLocal, quantScaleCkvLocal_, sharedBuf, rectangleParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::ScatterCkv(LocalTensor<kvCacheType> &outputLocal,
                                                            CkvkrParams rmsNormAndScatterCkvParams)
{
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) || (MLAPT::cacheMode == CACHE_MODE::PA_BSND) ||
                  (MLAPT::cacheMode == CACHE_MODE::ND)) {
        int64_t paTokenIndex;
        if constexpr (MLAPT::cacheMode == CACHE_MODE::ND) {
            paTokenIndex = rmsNormAndScatterCkvParams.tokenIndex;
        } else {
            paTokenIndex = cacheIndexGm_(rmsNormAndScatterCkvParams.tokenIndex);
        }
        ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
            kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->headSizeCkv,
                               baseParams_->dtileSize, static_cast<int64_t>(baseParams_->kvCacheStride0)});
        // 刷新量化scale
        if (isPertile && baseParams_->quantScaleRepoMode == 1U) {
            // BSND:
            int32_t tileNum = baseParams_->headSizeCkv / baseParams_->tileSize;
            LocalTensor<kvCacheType> quantScaleCkvInt8Tensor =
                outputLocal[vectorRow_ * baseParams_->headSizeCkv].template ReinterpretCast<kvCacheType>();
            int64_t startOffset = 0;
            int64_t startColOffset = baseParams_->headSizeCkv;
            if (baseParams_->ckvkrRepoMode == 1U) {
                startColOffset += baseParams_->headSizeKr * sizeof(krCacheType);
            }
            if constexpr ((MLAPT::cacheMode != CACHE_MODE::PA_NZ)) {
                startOffset = startColOffset;
            }

            ScatterCacheUnAligned<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                kvCacheGm_[startOffset], quantScaleCkvInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_,
                                   static_cast<int64_t>(tileNum * sizeof(float)), baseParams_->dtileSize,
                                   static_cast<int64_t>(baseParams_->kvCacheStride0)});
        }
    } else {
        // 使用  已计算好的偏移参数
        ScatterCacheMultiRows<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
            kvCacheGm_, outputLocal,
            ScatterCacheParams{baseParams_->blockSize, rmsNormAndScatterCkvParams.cacheOffset, vectorRow_,
                               baseParams_->headSizeCkv, baseParams_->headSizeCkv,
                               static_cast<int64_t>(baseParams_->kvCacheStride0), baseParams_->seq1Size,
                               rmsNormAndScatterCkvParams.tokenIndex},
            rmsNormAndScatterCkvParams.rowsInCurBatch, rmsNormAndScatterCkvParams.cacheOffset,
            rmsNormAndScatterCkvParams.nextBatchOffset);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RopeAndScatterKr(LocalTensor<float> &dequantScaleXLocal,
                                                                  LocalTensor<uint8_t> &shareTmpUb,
                                                                  LocalTensor<ropeComputType> &cosLocalCkvKr,
                                                                  LocalTensor<ropeComputType> &sinLocalCkvKr,
                                                                  CkvkrParams ropeAndScatterKrParams)
{
    // MatmulCkvKr ──> Rope(Ckv) ──> Scatter(Kr)
    LocalTensor<krCacheType> outputKrLocal = shareTmpUb.ReinterpretCast<krCacheType>();
    LocalTensor<uint8_t> ropeShareTmpUb = outputKrLocal[baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();
    int64_t stride = baseParams_->headSizeCkv + baseParams_->headSizeKr; // 512 + 64

    LocalTensor<ropeComputType> cosLocal;
    LocalTensor<ropeComputType> sinLocal;
    if constexpr (MLAPT::enableDequantOpt) {
        cosLocal = cosLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocalCkvKr[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    } else {
        cosLocal = cosLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
        sinLocal = sinLocal_[baseParams_->dimHeadRope * ropeAndScatterKrParams.curVecTokenIdx];
    }
    Rectangle ropeParams{
        static_cast<uint32_t>(vectorRow_),               // row
        static_cast<uint32_t>(baseParams_->dimHeadRope), // col
        static_cast<uint32_t>(stride)                    // stride
    };

    if constexpr ((std::is_same<mmCkvKrOutputType, int32_t>::value ||
                   (std::is_same<mmCkvKrOutputType, float>::value && !isFp8E8m0)) &&
                  std::is_same<krCacheType, bfloat16_t>::value) {
        // Use ropeShareTmpUb directly (same as RopeQr full-quant). cos/sin live in cosLocal/sinLocal,
        // not in this temp buffer; the old dimHeadRope*sizeof(ropeSinCosType) skip caused
        // ENABLE_ROPE=0 write-through to read/write the wrong UB region for Kr dequant.
        LocalTensor<uint8_t> sharedBuf = ropeShareTmpUb;
        // input为int32_t需在rope中做反量化, intput为float根据模板参数判断是否做反量化
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType, true, MLAPT::enableRope>(
            outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, sharedBuf, ropeParams,
            dequantScaleWDkvKrLocal_[baseParams_->headSizeCkv], dequantScaleXLocal);
    } else if constexpr (std::is_same<krCacheType, int8_t>::value) {
        LocalTensor<ropeSinCosType> inputLocal = ropeShareTmpUb.ReinterpretCast<ropeSinCosType>();
        LocalTensor<uint8_t> sharedBuf =
            ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope * sizeof(ropeSinCosType)];
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, ropeSinCosType,
                              !std::is_same<mmCkvKrOutputType, bfloat16_t>::value, MLAPT::enableRope>(
            inputLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, sharedBuf, ropeParams);
        RopePostQuantPerChannel(outputKrLocal, inputLocal, quantScaleCkrLocal_, sharedBuf,
                                vectorRow_ * baseParams_->dimHeadRope);
    } else {
        RotaryPosEmbPerTensor<mmCkvKrOutputType, ropeComputType, krCacheType, false, MLAPT::enableRope>(
            outputKrLocal, mmCkvKrResGm_[ropeAndScatterKrParams.offset], cosLocal, sinLocal, ropeShareTmpUb,
            ropeParams);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // scatter(Kr)
    // Rope(Kr) ──> Scatter(Kr) ──> kr_cache_out
    ScatterKr(outputKrLocal, ropeAndScatterKrParams);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::ScatterKr(LocalTensor<krCacheType> &outputKrLocal,
                                                           CkvkrParams ropeAndScatterKrParams)
{
    int64_t paTokenIndex;
    if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ) || (MLAPT::cacheMode == CACHE_MODE::PA_BSND) ||
                  (MLAPT::cacheMode == CACHE_MODE::ND)) {
        paTokenIndex = cacheIndexGm_(ropeAndScatterKrParams.tokenIndex);
        if constexpr (MLAPT::cacheMode == CACHE_MODE::ND) {
            paTokenIndex = ropeAndScatterKrParams.tokenIndex;
        }
        if (isPertile && baseParams_->ckvkrRepoMode == 1) {
            int64_t startOffset;
            if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_NZ)) {
                constexpr uint8_t col0 = ALIGN_BLOCK_SIZE / sizeof(kvCacheType);
                startOffset = CeilDiv(baseParams_->headSizeCkv, col0) * col0 * baseParams_->blockSize; // 列方向的偏移
            } else {
                startOffset = baseParams_->headSizeCkv;
            }
            LocalTensor<kvCacheType> outputKrInt8Tensor = outputKrLocal.template ReinterpretCast<kvCacheType>();
            ScatterCache<kvCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                kvCacheGm_[startOffset], outputKrInt8Tensor,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_,
                                   static_cast<int64_t>(baseParams_->dimHeadRope * sizeof(krCacheType)),
                                   baseParams_->dtileSize, static_cast<int64_t>(baseParams_->kvCacheStride0)});
        } else {
            ScatterCache<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_NZ)>(
                krCacheGm_, outputKrLocal,
                ScatterCacheParams{baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->dimHeadRope,
                                   baseParams_->dimHeadRope, static_cast<int64_t>(baseParams_->krCacheStride0)});
        }
    } else if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_BLK_BSND) || (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)) {
        // 使用  已计算好的偏移参数
        ScatterCacheMultiRows<krCacheType, (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)>(
            krCacheGm_, outputKrLocal,
            ScatterCacheParams{baseParams_->blockSize, ropeAndScatterKrParams.cacheOffset, vectorRow_,
                               baseParams_->dimHeadRope, baseParams_->dimHeadRope,
                               static_cast<int64_t>(baseParams_->krCacheStride0), baseParams_->seq1Size,
                               ropeAndScatterKrParams.tokenIndex},
            ropeAndScatterKrParams.rowsInCurBatch, ropeAndScatterKrParams.cacheOffset,
            ropeAndScatterKrParams.nextBatchOffset);
    } else {
        int64_t batchTokenIndex = 0;
        int64_t batchIndex = 0;
        int64_t batchSeqSize = 0;
        int64_t batchIndexOffset = 0;
        int64_t rowsInCurBatch = 0;
        int64_t cacheOffset = 0;
        int64_t nextBatchOffset = 0;
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset,
                                                                         int64_t ropeKrOffset, int64_t curVecToken)
{
    if (blockIdx_ < 0 || blockIdx_ >= cvRatio_ * baseParams_->mm2BlockNum) {
        return;
    }
    LocalTensor<float> dequantScaleXLocal = shareBuffer_.Get<float>();
    LocalTensor<ropeComputType> cosLocalCkvKr =
        dequantScaleXLocal[FP32_BLOCK_ELEMENT_NUM].template ReinterpretCast<ropeComputType>();
    LocalTensor<ropeComputType> sinLocalCkvKr = cosLocalCkvKr[baseParams_->dimHeadRope * curVecToken];
    LocalTensor<uint8_t> shareTmpUb =
        sinLocalCkvKr[baseParams_->dimHeadRope * curVecToken].template ReinterpretCast<uint8_t>();
    if constexpr (MLAPT::enableDequantOpt && MLAPT::enableRope) {
        GatherSinCos<ropeSinCosType, ropeComputType>(cosLocalCkvKr, sinLocalCkvKr, ropeCosGm_, ropeSinGm_, tokenIndex,
                                                     curVecToken, shareTmpUb, vectorRow_, baseParams_->dimHeadRope);
    }
    if constexpr (MLAPT::emptyMode == EMPTY_TENSOR_MODE::EMPTY_CACHE) {
        return;
    }

    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        if constexpr (std::is_same<mmCqOutputType, int32_t>::value ||
                      (std::is_same<mmCqOutputType, float>::value && !isFp8E8m0)) {
            DataCopyPad(dequantScaleXLocal, dequantScaleXGm_[tokenIndex], {1, sizeof(float), 0, 0}, {false, 0, 0, 0});
        }

        CkvkrParams rmsNormAndScatterCkvParams{tokenIndex, rmsNormCkvOffset, curVecTokenIdx, 0, 0, 0};

        CkvkrParams ropeAndScatterKrParams{tokenIndex, ropeKrOffset, curVecTokenIdx, 0, 0, 0};

        if constexpr ((MLAPT::cacheMode == CACHE_MODE::PA_BLK_BSND) || (MLAPT::cacheMode == CACHE_MODE::PA_BLK_NZ)) {
            ComputeBlkScatterOffsets(cacheIndexGm_, tokenIndex, vectorRow_, rmsNormAndScatterCkvParams,
                                     ropeAndScatterKrParams);
        }

        RmsNormAndScatterCkv(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, rmsNormAndScatterCkvParams);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        RopeAndScatterKr(dequantScaleXLocal, shareTmpUb, cosLocalCkvKr, sinLocalCkvKr, ropeAndScatterKrParams);

        tokenIndex += 1;
        rmsNormCkvOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
        ropeKrOffset += static_cast<int64_t>(baseParams_->headSizeCkv + baseParams_->dimHeadRope);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologV3SplitM<MLAPT>::QcQrSplit(int64_t curVecToken, int64_t curBlockTokenOffset,
                                                           int64_t curMSize, int64_t mmQcQrOffset,
                                                           int64_t mmQnPreDequantResOffset, int64_t ropeQrOffset,
                                                           int64_t ropeQrResOffset)
{
    if (cubeBlockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }

    DataCopyParams inputCopyParams{
        static_cast<uint16_t>(curVecToken),
        static_cast<uint16_t>(baseParams_->dimHeadSizeQc * sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        static_cast<uint16_t>((baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->dimHeadSizeQc) *
                              sizeof(mmQcQrOutputType) / ALIGN_BLOCK_SIZE),
        0};

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    LocalTensor<mmQcQrOutputType> inputLocal = shareTmpUb.ReinterpretCast<mmQcQrOutputType>();
    LocalTensor<mmQnInputType> outputLocal = inputLocal.template ReinterpretCast<mmQnInputType>();

    Rectangle dequantParams{
        static_cast<uint16_t>(curVecToken), //  row
        baseParams_->dimHeadSizeQc,         // col
        baseParams_->dimHeadSizeQc          // columnStride
    };
    DataCopyParams outputCopyParams{
        static_cast<uint16_t>(curVecToken),
        static_cast<uint16_t>(baseParams_->dimHeadSizeQc * sizeof(mmQnInputType) / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>((baseParams_->headSizeQc - baseParams_->dimHeadSizeQc) * sizeof(mmQnInputType) /
                              ALIGN_BLOCK_SIZE)};

    DataCopyParams outputRopeParams{
        static_cast<uint16_t>(curVecToken),
        static_cast<uint16_t>(baseParams_->dimHeadRope * sizeof(ropeOutputType) / ALIGN_BLOCK_SIZE), 0,
        static_cast<uint16_t>((baseParams_->headSizeQr - baseParams_->dimHeadRope) * sizeof(ropeOutputType) /
                              ALIGN_BLOCK_SIZE)};

    Rectangle ropeParams{
        static_cast<uint16_t>(curVecToken), //  row
        baseParams_->dimHeadRope,           // col
        static_cast<uint32_t>(baseParams_->numHeadSize *
                              (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope)) // stride
    };
    int64_t ropeStride = static_cast<int64_t>(baseParams_->numHeadSize) *
                         static_cast<int64_t>(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
    uint32_t deqScaleOffset = 0;

    int64_t deqScaleQcQrWOffset = 0;
    uint32_t colOffsetCube = 0;
    uint32_t colOffsetVec = 0;
    uint32_t colOffsetVecEnd = 0;
    uint32_t colOffsetRope = 0;
    // cube一次处理row*colCube，对应的两个vec一次处理row*colQc，两vec之间切row
    // 等cube生产足够数据了以后，vec开始消费
    uint32_t qcCount = 0;
    uint32_t splitCount = 0;
    uint32_t totalLoops = CeilDivT(mmQcQrParam_.n, mmQcQrParam_.baseN);
    uint32_t totalQcLoops = CeilDivT(mmQcQrParam_.n, (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope));

    for (uint32_t colOffsetCube = 0; colOffsetCube < mmQcQrParam_.n; colOffsetCube += mmQcQrParam_.baseN) {
        colOffsetVecEnd = colOffsetCube + mmQcQrParam_.baseN;
        if (colOffsetVecEnd > mmQcQrParam_.n) { // 当oriCol不被colCube整除时，mm最后一个base块需要刷新col end
            colOffsetVecEnd = mmQcQrParam_.n;
        }
        uint32_t qcQrSpacing = CeilDivT(totalLoops, static_cast<uint32_t>(MAX_SYNC_FLAG_COUNT - 1));
        if (splitCount % qcQrSpacing == 0 || splitCount == totalLoops - 1) {
            CrossCoreWaitFlag(FINISH_MM_QCQR_SPLIT_N);
        }
        if (splitCount % qcQrSpacing == 0) {
            CrossCoreWaitFlag(FINISH_MM_QCQR_SPLIT_BATCH);
        }
        while (colOffsetVec + baseParams_->dimHeadSizeQc <= colOffsetVecEnd) { // 循环singleNumHeadSize次
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

            DataCopy(inputLocal, mmQcQrResGm_[mmQcQrOffset], inputCopyParams);

            SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

            Cast(outputLocal, inputLocal, RoundMode::CAST_RINT, curVecToken * baseParams_->dimHeadSizeQc);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

            DataCopy(mmQcQrResDequantGm_[mmQnPreDequantResOffset], outputLocal, outputCopyParams);
            {
                uint32_t qcSpacing = CeilDivT(totalQcLoops, static_cast<uint32_t>(MAX_SYNC_FLAG_COUNT - 1));
                if ((qcCount + 1) % qcSpacing == 0 || qcCount + 1 == totalQcLoops) {
                    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N);
                } else if ((totalQcLoops - CeilDivT(totalQcLoops, qcSpacing)) <=
                           static_cast<uint32_t>(MAX_SYNC_FLAG_COUNT)) {
                    CrossCoreSetFlag<SYNC_MODE_CUBE_VEC, PIPE_MTE3>(FINISH_VEC_DEQUANT_QC_SPLIT_N_GAP);
                }
            }
            colOffsetVec += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            mmQcQrOffset += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            mmQnPreDequantResOffset += baseParams_->dimHeadSizeQc;
            qcCount++;
        }

        while (colOffsetRope + baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope <=
               colOffsetVecEnd) { // 循环singleNumHeadSize次

            GlobalTensor<ropeOutputType> outputGmRope = qrOutGm_[ropeQrResOffset];

            LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
            LocalTensor<ropeOutputType> outputLocalRope = shareTmpUb.ReinterpretCast<ropeOutputType>();
            LocalTensor<uint8_t> ropeShareTmpUb =
                outputLocalRope[curVecToken * baseParams_->dimHeadRope].template ReinterpretCast<uint8_t>();

            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

            RotaryPosEmbPerHead<mmQcQrOutputType, ropeComputType, ropeOutputType, false, MLAPT::enableRope>(
                outputLocalRope, mmQcQrResGm_[ropeQrOffset], cosLocal_, sinLocal_, ropeShareTmpUb, ropeParams,
                ropeStride);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);

            DataCopy(qrOutGm_[ropeQrResOffset], outputLocalRope, outputRopeParams);

            colOffsetRope += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            ropeQrOffset += (baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
            ropeQrResOffset += baseParams_->dimHeadRope;
        }
        splitCount++;
    }
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void
MlaPrologV3SplitM<MLAPT>::DynamicQuantQnAndMulQrSyncMMQn(int64_t batchOffset, int64_t curStepBatchSize,
                                                         int64_t numHeadOffset, int64_t mmQnLoops)
{
    // 如果curStepBatchSize是偶数，则两个核平分；如果curStepBatchSize是奇数，则奇数核比偶数核多分一个
    // >> 1 是将curStepBatchSize分到每个vec核上；
    int64_t curStepBatchSizeVec = (curStepBatchSize + (blockIdx_ % cvRatio_)) >> 1;
    if (blockIdx_ >= baseParams_->mm4BlockNum * cvRatio_) {
        return;
    }

    // 等待前面的Qr部分完成
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

    constexpr uint32_t DYNAMIC_QUANT_INPUT_READY = EVENT_ID0;
    constexpr uint32_t MUL_QR_INPUT_COPY_READY = EVENT_ID3;
    constexpr uint32_t DYNAMIC_QUANT_OUTPUT_READY = EVENT_ID3;

    int64_t blockBatchOffset = (blockIdx_ % cvRatio_) * (curStepBatchSize / cvRatio_);
    int64_t totalSizeCkv =
        static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(baseParams_->headSizeCkv);
    // 由于两个vec核分curStepBatchSize，各处理curStepBatchSize/2，blockIdx_ & 1 表示是否为第二个vec核
    int64_t dynamicQuantQueryOffset =
        baseParams_->stepBatchSize * baseParams_->numHeadSize * baseParams_->headSizeCkv * cubeBlockIdx_ +
        blockBatchOffset * totalSizeCkv + numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t dynamicQuantQueryResOffset = batchOffset * totalSizeCkv + blockBatchOffset * totalSizeCkv +
                                         numHeadOffset * static_cast<int64_t>(baseParams_->headSizeCkv);
    int64_t scaleQueryNopeOffset = batchOffset * static_cast<int64_t>(baseParams_->numHeadSize) +
                                   blockBatchOffset * static_cast<int64_t>(baseParams_->numHeadSize) + numHeadOffset;
    int64_t queryOutStride = totalSizeCkv;
    int64_t qrOutputStride =
        static_cast<int64_t>(baseParams_->numHeadSize) * static_cast<int64_t>(baseParams_->dimHeadRope);
    int64_t qrPostProcessResOffset = batchOffset * static_cast<int64_t>(baseParams_->headSizeQr) +
                                     numHeadOffset * static_cast<int64_t>(baseParams_->dimHeadRope) +
                                     blockBatchOffset * static_cast<int64_t>(baseParams_->headSizeQr);

    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();

    float quantScaleCkv = quantScaleCkvGm_.GetValue(0);

    // Dynamic Quant
    SetFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
    SetFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);

    // Rope Post Process
    SetFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    CrossCoreWaitFlag(FINISH_MM_QN_SPLIT_N);
    // per-head循环
    for (int64_t loopIdx = 0; loopIdx < mmQnLoops; loopIdx++) {
        DynamicQuantQnWithMulQr<ropeOutputType, dequantScaleQNopeType, queryOutputType>(
            dequantScaleQNopeGm_[scaleQueryNopeOffset], queryOutGm_[dynamicQuantQueryResOffset],
            qrOutGm_[qrPostProcessResOffset], mmQnResGm_[dynamicQuantQueryOffset], shareTmpUb, curStepBatchSizeVec,
            baseParams_->headSizeCkv, baseParams_->numHeadSize, queryOutStride,
            // Rope Post Process
            qrOutGm_[qrPostProcessResOffset], quantScaleCkv, baseParams_->dimHeadRope, qrOutputStride, cvRatio_);
        dynamicQuantQueryOffset += baseParams_->headSizeCkv;
        scaleQueryNopeOffset += 1;
        dynamicQuantQueryResOffset += baseParams_->headSizeCkv;
        qrPostProcessResOffset += baseParams_->dimHeadRope;
    }
    // Rope Post Process
    WaitFlag<HardEvent::V_MTE2>(MUL_QR_INPUT_COPY_READY);
    // Dynamic Quant
    WaitFlag<HardEvent::V_MTE2>(DYNAMIC_QUANT_INPUT_READY);
    WaitFlag<HardEvent::MTE3_V>(DYNAMIC_QUANT_OUTPUT_READY);
}

} // namespace MlaProlog

#endif // MLA_PROLOG_V3_SPLIT_M