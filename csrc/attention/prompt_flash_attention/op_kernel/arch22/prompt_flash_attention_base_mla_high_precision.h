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
 * \file prompt_flash_attention_base_mla_high_precision.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_BASE_MLA_HIGH_PRECISION_H
#define PROMPT_FLASH_ATTENTION_BASE_MLA_HIGH_PRECISION_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

constexpr int32_t ZERO_PRECISION = 0;
constexpr int32_t ONE_PRECISION = 1;
constexpr int32_t TWO_PRECISION = 2;
constexpr int32_t FOUR_PRECISION = 4;
constexpr int32_t FIVE_PRECISION = 5;
constexpr int32_t EIGHT_PRECISION = 8;
constexpr int32_t THIRTY_TWO_PRECISION = 32;
constexpr int32_t SIXTY_FOUR_PRECISION = 64;

constexpr int32_t MASK_TYPE_ALIBI_COMPRESS_PRECISION = 2;
constexpr int32_t ALIBI_LEFT_ALIGN_DISABLE_PRECISION = 0;
constexpr int32_t ALIBI_COMPRESS_OFFSET_DISABLE_PRECISION = 0;
constexpr int32_t MASK_TYPE_NONE_PRECISION = 0; 
constexpr int32_t SBLOCK_STACK_DOUBLE_PRECISION = 2;

constexpr uint32_t PINGPONG_FLAG_M_MTE1_OFFSET_TWO_PRECISION = 2;
constexpr uint32_t PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_PRECISION = 4;
constexpr uint32_t PINGPONG_FLAG_M_MTE1_OFFSET_FIVE_PRECISION = 5;

constexpr int32_t MAX_ALLOWED_LENGTH_PRECISION = 16;  
constexpr int32_t MIN_ALLOWED_LENGTH_PRECISION = 1; 

template <typename TILING_TYPE, typename IN_DATA_TYPE, const bool IS_BF16 = true, typename...Args>
struct PFAHighPrecisionMLAType {
    using tilingType = TILING_TYPE;
    using inDataType = IN_DATA_TYPE;
    static constexpr bool isBF16 = IS_BF16;
};

#ifdef __DAV_C220_CUBE__

template<typename PFAT>
class FlashAttentionEncoderHighPrecisionMLA {
public:
    __aicore__ __attribute__((always_inline)) inline FlashAttentionEncoderHighPrecisionMLA(
        __gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
        __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
        __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
        const typename PFAT::tilingType* __restrict tiling,
        __gm__ uint8_t* gmTiling, TPipe* tPipe)
    {
        AscendC::SetLoadDataPaddingValue<uint64_t>(0);
        AscendC::SetAtomicNone();
        AscendC::SetFixpipeNz2ndFlag(1, 0, 0);
        AscendC::SetMaskNorm();

        qGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DATA_TYPE *>(query));
        kGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DATA_TYPE *>(key));
        vGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DATA_TYPE *>(value));
        uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
        sGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        pGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DATA_TYPE *>(workspace + workSize));
        oTmpGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + 2 * workSize));
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);
        this->tiling = tiling;
        this->batchSize = tiling->promptAttentionBaseApiBaseParams.batchSize;
        this->maxSeqlen = tiling->promptAttentionBaseApiBaseParams.maxSeqLen;
        this->qHeads = tiling->promptAttentionBaseApiBaseParams.headNumSize;
        this->embd = tiling->promptAttentionBaseApiBaseParams.headSize;
        this->embdv = tiling->promptAttentionBaseApiBaseParams.embeddingSizeV;
        this->kvHeads = tiling->promptAttentionBaseApiBaseParams.kvHeadNumSize;
        this->isTriuMask = tiling->promptAttentionBaseApiBaseParams.isTriuMask;
        this->totalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNum;
        this->maxKvSeqlen = tiling->promptAttentionBaseApiBaseParams.maxKvSeqLen;
        this->tilingDataShape = tiling->promptAttentionBaseApiBaseParams.quantType;
        this->inputLayoutType = tiling->promptAttentionBaseApiBaseParams.inputLayoutType;
        this->groupNum = qHeads / kvHeads;
        this->strideQo = qHeads * embd;
        this->strideO = qHeads * embdv;
        this->strideK = kvHeads * embd;
        this->strideV = kvHeads * embdv;
        if (this->tilingDataShape == 1)
        {
            this->strideQo = embd;
            this->strideO = embdv;
            this->strideK = embd;
            this->strideV = embdv;
        }
        this->batchStrideK = batchSize * maxKvSeqlen * strideK * 2;
        this->batchStrideV = batchSize * maxKvSeqlen * strideV * 2;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID5));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID6));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));
    }

    __aicore__ __attribute__((always_inline)) inline ~FlashAttentionEncoderHighPrecisionMLA()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID5));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID6));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ __attribute__((always_inline)) inline void LoadDataToCa(
        AscendC::LocalTensor<typename PFAT::inDataType> dstTensor, AscendC::LocalTensor<typename PFAT::inDataType> srcTensor,
        uint32_t roundK, uint32_t qkRoundM, uint32_t qkM)
    {
        if (qkM == 1) {
            AscendC::LoadData(dstTensor,
                            srcTensor,
                            AscendC::LoadData2dParams(0,           // baseIdx
                                                    (roundK + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                                                    1,  // srcStride
                                                    0,           // sid
                                                    0,  // dstStride
                                                    false, // transpose
                                                    0));         // addrCalMode
        } else {
            for (uint32_t l0aLoadIdx = 0; l0aLoadIdx < qkRoundM / BLOCK_SIZE; ++l0aLoadIdx) {
                AscendC::LoadData(dstTensor[l0aLoadIdx * roundK * BLOCK_SIZE],
                                srcTensor[l0aLoadIdx * CUBE_MATRIX_SIZE],
                                AscendC::LoadData2dParams(0,           // baseIdx
                                                            roundK / BLOCK_SIZE,   // repeat
                                                            qkRoundM / BLOCK_SIZE,  // srcStride
                                                            0,           // sid
                                                            0,  // dstStride
                                                            false, // transpose
                                                            0));         // addrCalMode
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        uint64_t curBatch = 0;
        uint32_t preTotalQBlkNum = 0;
        uint32_t qSeqlen = maxSeqlen;
        uint32_t kvSeqlen = maxKvSeqlen;
        if (this->inputLayoutType == 0) {
            uint32_t qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
            uint32_t kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
        }
        uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
        uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
        uint64_t addrQScalar = 0;
        uint64_t addrKScalar = 0;
        uint64_t addrVScalar = 0;
        uint32_t curTotalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNumFirst;

        uint32_t processNum = totalQBlkNum * qHeads;
        uint32_t nextProcess = 0;
        uint32_t isMLA = 1;
        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
        static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(IN_DATA_TYPE);
        for (uint32_t process = GetBlockIdx(); process < processNum; process = nextProcess) {
            while (process >= curTotalQBlkNum * qHeads) {
                curBatch++;
                preTotalQBlkNum = curTotalQBlkNum;
                addrQScalar += static_cast<uint64_t>(qSeqlen * qHeads * embd);
                addrKScalar += static_cast<uint64_t>(kvSeqlen * kvHeads * embd);
                addrVScalar += static_cast<uint64_t>(kvSeqlen * kvHeads * embdv);
                if (this->inputLayoutType == 0) {
                    qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
                    kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
                }
                MNibd mnIbd = GetPPmnIbd(qSeqlen, kvSeqlen, embd, isMLA);
                ppMScalar = PP_MM[mnIbd.mIbd];
                ppNScalar = PP_NN[mnIbd.nIbd];
                curTotalQBlkNum += (qSeqlen != ZERO_PRECISION) ? ((qSeqlen + ppMScalar - ONE_PRECISION) / ppMScalar) : ZERO_PRECISION;
            }
            nextProcess = process + GetBlockNum();
            if (isTriuMask) {
                uint32_t currIter = process / GetBlockNum();
                nextProcess = currIter % 2 == 1 ? (currIter + 1) * GetBlockNum() + GetBlockIdx() : (currIter + 2) * GetBlockNum() - 1 - GetBlockIdx(); // 2 is the later num
            }
            // get tiling args
            if (qSeqlen == 0 || kvSeqlen == 0) {
                continue;
            }

            uint32_t processIdx = process - preTotalQBlkNum * qHeads;
            uint32_t mIdx = processIdx / qHeads;
            uint64_t headIdx = processIdx % qHeads;

            uint32_t mLoop = (qSeqlen + ppMScalar - 1) / ppMScalar;
            uint32_t nLoop = (kvSeqlen + ppNScalar - 1) / ppNScalar;

            uint32_t qkM = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * ppMScalar) : ppMScalar;
            uint32_t qkRoundM = (qkM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            /**************** pre_load *****************/
            uint32_t qkN = nLoop == 1 ? kvSeqlen : ppNScalar;
            uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint32_t pingpongFlag = 0;
            uint32_t offset = pingpongFlag * L0AB_HALF_BUF_SIZE;

            uint64_t qOffset = addrQScalar + headIdx * embd + mIdx * ppMScalar * strideQo;
            uint64_t kOffset = addrKScalar + (headIdx / groupNum) * embd;
            if (tilingDataShape == 1)
            {
               qOffset = addrQScalar + headIdx * embd * maxSeqlen+ mIdx * ppMScalar * strideQo;
               kOffset = addrKScalar + (headIdx / groupNum) * embd * maxKvSeqlen;
            }

            uint32_t loopQK = (this->embd + BLOCK_QK - 1) / BLOCK_QK;
            uint32_t loopV = (this->embdv + BLOCK_QK - 1) / BLOCK_QK;
            uint32_t qkK = BLOCK_QK;
            uint32_t qkRoundK = BLOCK_QK;

            uint32_t svN = nLoop == 1 ? kvSeqlen : ppNScalar;
            uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            uint64_t vOffset = addrVScalar + (headIdx / groupNum) * embdv;
            if (tilingDataShape == 1)
            {
                vOffset = addrVScalar + (headIdx / groupNum) * embdv * maxKvSeqlen;
            }

            uint32_t nEnd = nLoop;
            if (isTriuMask) {
                nEnd = mIdx + 1;
            }
            uint32_t sBlockStack = nEnd > FOUR_PRECISION ? TWO_PRECISION : ONE_PRECISION; // Currently not splitting K
            uint32_t launchDelay = sBlockStack * TWO_PRECISION;
            uint32_t vectMod = 2 * launchDelay;
            for (uint32_t nIdx = 0; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
                if (nIdx < nEnd) {
                    for (uint32_t splitIdx = 0; splitIdx < sBlockStack && nIdx + splitIdx < nEnd; splitIdx++) {
                        pingpongFlag = (nIdx + splitIdx) % TWO_PRECISION;
                        offset = pingpongFlag * L0AB_HALF_BUF_SIZE;
                        if (nIdx + splitIdx == (nLoop - 1)) {
                            qkN = (kvSeqlen - (nIdx + splitIdx) * ppNScalar);
                            qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                        }
                        bool lastSplit = splitIdx == sBlockStack - 1 || nIdx + splitIdx == nEnd - 1;
                        for(uint32_t kIdx = 0; kIdx < loopQK; kIdx++){
                            uint32_t initc = (kIdx == 0) ? 1 : 0;
                            qkK = (kIdx == (loopQK - 1))? embd - kIdx * BLOCK_QK : BLOCK_QK;
                            qkRoundK = (qkK + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                            uint64_t qQffsetk = qOffset + BLOCK_QK * kIdx;
                            uint64_t kQffsetk = kOffset + BLOCK_QK * kIdx;
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FIVE_PRECISION));
                            if (qkM == 1) {
                                AscendC::DataCopy(l1qBufAddrTensor[offset],
                                                qGmTensor[qQffsetk],
                                                AscendC::DataCopyParams(1,                                              // nBurst
                                                                        CeilDiv<BLOCK_SIZE_COPY>(1 * qkRoundK), // lenBurst
                                                                        0,                                              // srcGap
                                                                        0));
                            } else {
                                if (strideQo < STRIDE_LIMIT) {
                                    AscendC::DataCopy(l1qBufAddrTensor[offset],
                                                    qGmTensor[qQffsetk],
                                                    AscendC::Nd2NzParams(1,           // ndNum
                                                                        qkM, // nValue
                                                                        qkK, // dValue
                                                                        0,           // srcNdMatrixStride, unused
                                                                        strideQo,        // srcDValue
                                                                        qkRoundM,   // dstNzC0Stride
                                                                        1,           // dstNzNStride
                                                                        0));         // dstNzMatrixStride, unused
                                } else {
                                    for (uint32_t i = 0; i < qkM; ++i) {
                                        AscendC::DataCopy(l1qBufAddrTensor[offset][i * BLOCK_SIZE_COPY],
                                                        qGmTensor[qQffsetk][i * strideQo],
                                                        AscendC::Nd2NzParams(1,           // ndNum
                                                                            1,           // nValue
                                                                            qkK, // dValue
                                                                            0,           // srcNdMatrixStride, unused
                                                                            0,           // srcDValue
                                                                            qkRoundM,   // dstNzC0Stride
                                                                            0,           // dstNzNStride
                                                                            0));         // dstNzMatrixStride, unused
                                    }
                                }
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag));
                            LoadDataToCa(l0aBufTensor[offset], l1qBufAddrTensor[offset], qkRoundK, qkRoundM, qkM);
                            // *** Prepare K to L1
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FIVE_PRECISION));
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag));

                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag));
                            if (strideK < STRIDE_LIMIT) {
                                AscendC::DataCopy(l1kBufAddrTensor[offset],
                                                kGmTensor[kQffsetk],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    qkN, // nValue
                                                                    qkK, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    strideK,        // srcDValue
                                                                    qkRoundN,   // dstNzC0Stride
                                                                    1,           // dstNzNStride
                                                                    0));         // dstNzMatrixStride, unused
                            } else {
                                for (uint32_t i = 0; i < qkN; ++i) {
                                    AscendC::DataCopy(l1kBufAddrTensor[offset][i * BLOCK_SIZE_COPY],
                                                    kGmTensor[kQffsetk][i * strideK],
                                                    AscendC::Nd2NzParams(1,           // ndNum
                                                                        1,           // nValue
                                                                        qkK, // dValue
                                                                        0,           // srcNdMatrixStride, unused
                                                                        0,           // srcDValue
                                                                        qkRoundN,   // dstNzC0Stride
                                                                        0,           // dstNzNStride
                                                                        0)           // dstNzMatrixStride, unused
                                    );
                                }
                            }

                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_PRECISION));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_PRECISION));
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_PRECISION));
                            AscendC::LoadData(l0bBufTensor[offset],
                                            l1kBufAddrTensor[offset],
                                            AscendC::LoadData2dParams(0,
                                                                    qkRoundK * qkRoundN / CUBE_MATRIX_SIZE,
                                                                    1,
                                                                    0,
                                                                    0,
                                                                    false,
                                                                    0));
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag));
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_PRECISION));

                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_PRECISION));
                            if (splitIdx == 0 && initc) {
                                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
                                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));
                            }
                            AscendC::Mmad(l0cBufTensor[splitIdx * qkRoundM * ppNScalar],
                                        l0aBufTensor[offset],
                                        l0bBufTensor[offset],
                                        AscendC::MmadParams(qkM, qkN, qkK, 0, false, initc));
                            AscendC::PipeBarrier<PIPE_M>();
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag));
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_PRECISION));
                        }
                        kOffset += ppNScalar * strideK;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>((EVENT_ID0));
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((EVENT_ID0));
                    uint32_t svNTriu = nEnd * ppNScalar;
                    if (nIdx + sBlockStack > nEnd - 1) {
                        svN = svNTriu > kvSeqlen ? kvSeqlen - nIdx * ppNScalar : svNTriu - nIdx * ppNScalar;
                    } else {
                        svN = ppNScalar * sBlockStack;
                    }
                    uint32_t svRoundN = (svN + BLOCK_SIZE - ONE_PRECISION) / BLOCK_SIZE * BLOCK_SIZE;
                    // copy S to gm
                    auto intriParams = AscendC::FixpipeParamsV220(svRoundN, // nSize
                                                                qkM, // mSize
                                                                qkRoundM,   // srcStride
                                                                svRoundN,   // dstStride
                                                                false);      // enRelu
                    intriParams.quantPre = QuantMode_t::NoQuant;
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod], l0cBufTensor, intriParams);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(QK_READY); // 2 is the Sync mode
                }
                if (nIdx >= launchDelay) {
                    uint32_t l0cPingpongFlag = nIdx % TWO_PRECISION;
                    uint32_t l0cOffset = l0cPingpongFlag * L0AB_HALF_BUF_SIZE;
                    uint32_t svNTriu = nEnd * ppNScalar;
                    if (nIdx + sBlockStack > nEnd + launchDelay - 1) {
                        svN = svNTriu > kvSeqlen ? kvSeqlen - (nIdx - launchDelay) * ppNScalar : svNTriu - (nIdx - launchDelay) * ppNScalar;
                    } else {
                        svN = ppNScalar * sBlockStack;
                    }
                    uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    for(uint32_t kIdx = 0; kIdx < loopV; kIdx++){
                        qkK = (kIdx == (loopV - 1))? embdv - kIdx * BLOCK_QK : BLOCK_QK;
                        qkRoundK = (qkK + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                        uint64_t vOffsetk = vOffset + BLOCK_QK * kIdx;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
                        if (strideV < STRIDE_LIMIT) {
                            AscendC::DataCopy(l1vBufAddrTensor,
                                            vGmTensor[vOffsetk],
                                            AscendC::Nd2NzParams(1,           // ndNum
                                                                svN, // nValue
                                                                qkK, // dValue
                                                                0,           // srcNdMatrixStride, unused
                                                                strideV,        // srcDValue
                                                                svRoundN,   // dstNzC0Stride
                                                                1,           // dstNzNStride
                                                                0));         // dstNzMatrixStride, unused
                        } else {
                            for (uint32_t i = 0; i < svN; ++i) {
                                AscendC::DataCopy(l1vBufAddrTensor[i * BLOCK_SIZE_COPY],
                                                vGmTensor[vOffsetk][i * strideV],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    1,           // nValue
                                                                    qkK, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    0,           // srcDValue
                                                                    svRoundN,   // dstNzC0Stride
                                                                    0,           // dstNzNStride
                                                                    0));         // dstNzMatrixStride, unused
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID2));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID2));
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
                        for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < svRoundN / BLOCK_SIZE; ++l0bLoadIdx) {
                            AscendC::LoadData(l0bBufTensor[l0bLoadIdx * qkRoundK * BLOCK_SIZE],
                                            l1vBufAddrTensor[l0bLoadIdx * CUBE_MATRIX_SIZE],
                                            AscendC::LoadData2dParams(0,
                                                                    qkRoundK / BLOCK_SIZE,
                                                                    svRoundN / BLOCK_SIZE,
                                                                    0,
                                                                    0,
                                                                    true,
                                                                    0));
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID6));
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
                        if(kIdx == 0){
                            AscendC::CrossCoreWaitFlag(SOFTMAX_READY);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
                            if (qkM == 1) {
                                AscendC::DataCopy(l1pBufAddrTensor,
                                                pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod],
                                                AscendC::DataCopyParams(1,                                    // nBurst
                                                                        CeilDiv<BLOCK_SIZE_COPY>(1 * svRoundN),  // lenBurst
                                                                        0,                                    // srcGap
                                                                        0)                                    // dstGap
                                );
                            } else {
                                if (svRoundN < STRIDE_LIMIT) {
                                    AscendC::DataCopy(l1pBufAddrTensor,
                                                    pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod],
                                                    AscendC::Nd2NzParams(1,           // ndNum
                                                                        qkM, // nValue
                                                                        svN, // dValue
                                                                        0,           // srcNdMatrixStride, unused
                                                                        svRoundN,        // srcDValue
                                                                        qkRoundM,   // dstNzC0Stride
                                                                        1,           // dstNzNStride
                                                                        0));         // dstNzMatrixStride, unused
                                } else {
                                    for (uint32_t i = 0; i < qkM; ++i) {
                                        AscendC::DataCopy(l1pBufAddrTensor[i * BLOCK_SIZE_COPY],
                                                        pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod][i * strideQo],
                                                        AscendC::Nd2NzParams(1,           // ndNum
                                                                            1,           // nValue
                                                                            qkM, // dValue
                                                                            0,           // srcNdMatrixStride, unused
                                                                            0,           // srcDValue
                                                                            qkRoundM,   // dstNzC0Stride
                                                                            0,           // dstNzNStride
                                                                            0));         // dstNzMatrixStride, unused
                                    }
                                }
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID3));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID3));
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
                            if (qkM == 1) {
                                AscendC::LoadData(l0aBufTensor,
                                                l1pBufAddrTensor,
                                                AscendC::LoadData2dParams(0,           // baseIdx
                                                                        (svRoundN + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                                                                        1,  // srcStride
                                                                        0,           // sid
                                                                        0,  // dstStride
                                                                        false, // transpose
                                                                        0));         // addrCalMode
                            } else {
                                for (uint32_t l0aLoadIdx = 0; l0aLoadIdx < qkRoundM / BLOCK_SIZE; ++l0aLoadIdx) {
                                    AscendC::LoadData(l0aBufTensor[l0aLoadIdx * svRoundN * BLOCK_SIZE],
                                                    l1pBufAddrTensor[l0aLoadIdx * CUBE_MATRIX_SIZE],
                                                    AscendC::LoadData2dParams(0,           // baseIdx
                                                                            svRoundN / BLOCK_SIZE,   // repeat
                                                                            qkRoundM / BLOCK_SIZE,  // srcStride
                                                                            0,           // sid
                                                                            0,  // dstStride
                                                                            false, // transpose
                                                                            0));         // addrCalMode
                                }
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID5));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID6));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID5));
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((l0cPingpongFlag));
                        AscendC::Mmad(l0cBufTensor[l0cOffset],
                                    l0aBufTensor,
                                    l0bBufTensor,
                                    AscendC::MmadParams(qkM, qkK, svN, 0, false, 1));
                        AscendC::PipeBarrier<PIPE_M>();
                        if(kIdx == loopV - 1){
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>((l0cPingpongFlag));
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((l0cPingpongFlag));
                        // copy O to gm
                        auto intriParams = AscendC::FixpipeParamsV220(qkRoundK, // nSize
                                                                    qkM, // mSize
                                                                    qkRoundM,   // srcStride
                                                                    qkRoundK,   // dstStride
                                                                    false);      // enRelu
                        intriParams.quantPre = QuantMode_t::NoQuant;
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE * LOCAL_SIZE + kIdx * TMP_SIZET + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod * loopV], l0cBufTensor[l0cOffset], intriParams);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((l0cPingpongFlag));
                    }
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(UPDATE_READY); // 2 is the sync mode
                    vOffset += svN * strideV;
                }
            }
        }
    }

private:
    using IN_DATA_TYPE = typename PFAT::inDataType;
    static constexpr bool IS_BF16 = PFAT::isBF16;

    const typename PFAT::tilingType* __restrict tiling;
    const uint32_t l1qBufAddrOffset = 0;
    const uint32_t l1kBufAddrOffset = 2 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1pBufAddrOffset = 4 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1vBufAddrOffset = 6 * L0AB_UINT8_BLOCK_SIZE;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::LocalTensor<IN_DATA_TYPE> l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DATA_TYPE>(l1qBufAddrOffset);
    AscendC::LocalTensor<IN_DATA_TYPE> l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DATA_TYPE>(l1kBufAddrOffset);
    AscendC::LocalTensor<IN_DATA_TYPE> l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DATA_TYPE>(l1pBufAddrOffset);
    AscendC::LocalTensor<IN_DATA_TYPE> l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, IN_DATA_TYPE>(l1vBufAddrOffset);

    AscendC::GlobalTensor<IN_DATA_TYPE> qGmTensor;
    AscendC::GlobalTensor<IN_DATA_TYPE> kGmTensor;
    AscendC::GlobalTensor<IN_DATA_TYPE> vGmTensor;
    AscendC::GlobalTensor<float> sGmTensor;
    AscendC::GlobalTensor<IN_DATA_TYPE> pGmTensor;
    AscendC::GlobalTensor<float> oTmpGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    AscendC::LocalTensor<IN_DATA_TYPE> l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, IN_DATA_TYPE>(0);
    AscendC::LocalTensor<IN_DATA_TYPE> l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, IN_DATA_TYPE>(0);
    AscendC::LocalTensor<float> l0cBufTensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    uint32_t batchSize{0};
    uint32_t maxSeqlen{0};
    uint32_t maxKvSeqlen{0};
    uint32_t qHeads{0};
    uint32_t embd{0};
    uint32_t embdv{0};
    uint32_t kvHeads{0};
    uint32_t isTriuMask{0};
    uint32_t totalQBlkNum{0};
    uint32_t groupNum{0};
    uint64_t strideQo{0};
    uint64_t strideO{0};
    uint64_t strideK{0};
    uint64_t strideV{0};
    uint64_t batchStrideV{0};
    uint64_t batchStrideK{0};
    uint32_t tilingDataShape{0};
    uint32_t inputLayoutType{0};
};
#elif __DAV_C220_VEC__
template<typename PFAT>
class FlashAttentionEncoderHighPrecisionMLAVec {
public:
    __aicore__ __attribute__((always_inline)) inline FlashAttentionEncoderHighPrecisionMLAVec(
        __gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
        __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
        __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
        const typename PFAT::tilingType* __restrict tiling,
        __gm__ uint8_t* gmTiling, TPipe* tPipe, __gm__ uint8_t* alibiCoeff) : alibiCoeffGm(alibiCoeff)
    {
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        this->subBlockIdx = AscendC::GetSubBlockIdx();
        this->tiling = tiling;

        uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
        maskGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DATA_TYPE *>(pseShift));
        oGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DATA_TYPE *>(attentionOut));
        sGmTensor.SetGlobalBuffer((__gm__ float *)(workspace));
        pGmTensor.SetGlobalBuffer((__gm__ IN_DATA_TYPE *)(workspace + workSize));
        oTmpGmTensor.SetGlobalBuffer((__gm__ float *)(workspace + 2 * workSize));
        upoGmTensor.SetGlobalBuffer((__gm__ float *)(workspace + 3 * workSize));
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);
        this->batchSize = tiling->promptAttentionBaseApiBaseParams.batchSize;
        this->maxSeqlen = tiling->promptAttentionBaseApiBaseParams.maxSeqLen;
        this->maxKvSeqlen = tiling->promptAttentionBaseApiBaseParams.maxKvSeqLen;
        this->qHeads = tiling->promptAttentionBaseApiBaseParams.headNumSize;
        this->embd = tiling->promptAttentionBaseApiBaseParams.headSize;
        this->embdv = tiling->promptAttentionBaseApiBaseParams.embeddingSizeV;
        this->tor = tiling->promptAttentionBaseApiBaseParams.tor;
        this->headStride = tiling->promptAttentionBaseApiBaseParams.headStride;
        this->maskStride = tiling->promptAttentionBaseApiBaseParams.maskStride;
        this->isTriuMask = tiling->promptAttentionBaseApiBaseParams.isTriuMask;
        this->totalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNum;
        this->isClamp = tiling->promptAttentionBaseApiBaseParams.isClamp;
        this->clampMin = 0;
        this->clampMax = 0;
        this->longSeq = tiling->promptAttentionBaseApiBaseParams.isLongSeq;
        this->isSqrt = 0;
        this->maskType = tiling->promptAttentionBaseApiBaseParams.maskType;
        this->alibiCompressOffset = 0;
        this->alibiLeftAlign = 0;
        this->tilingDataShapeType = tiling->promptAttentionBaseApiBaseParams.quantType;
        this->inputLayoutType = tiling->promptAttentionBaseApiBaseParams.inputLayoutType;
        this->strideQo = qHeads * embdv;

        if (tilingDataShapeType == 1)
        {
            this->strideQo = embdv;
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID0));
    }

    __aicore__ __attribute__((always_inline)) inline ~FlashAttentionEncoderHighPrecisionMLAVec()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID0));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ __attribute__((always_inline)) inline void __set_mask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE || len == 0) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }

    __aicore__ __attribute__((always_inline)) inline void __set_vcg_mask(int32_t len)
    {
        if (len > MAX_ALLOWED_LENGTH_PRECISION || len < MIN_ALLOWED_LENGTH_PRECISION) 
        {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }

        uint64_t subMask = ((uint64_t) 1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask +
                             (subMask << 56) + (subMask << 40) + (subMask << 24) + (subMask << 8);
                             
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        uint64_t curBatch = 0;
        uint32_t preTotalQBlkNum = 0;
        uint32_t qSeqlen = maxSeqlen;
        uint32_t kvSeqlen = maxKvSeqlen;
        if (this->inputLayoutType == 0) {
            qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
            kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
        }
        uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
        uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
        uint64_t addrOScalar = 0;
        uint32_t curTotalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNumFirst;
        uint32_t processNum = totalQBlkNum * qHeads;
        float alibiCoeff = 1;
        uint32_t loopV = (embdv + BLOCK_QK - 1) / BLOCK_QK;
        uint32_t qkK = BLOCK_QK;
        uint32_t qkRoundK = BLOCK_QK;

        uint32_t nextProcess = 0;
        uint32_t isMLA = 1;
        for (uint32_t process = GetBlockIdx(); process < processNum; process = nextProcess) {
            while (process >= curTotalQBlkNum * qHeads) {
                curBatch++;
                preTotalQBlkNum = curTotalQBlkNum;
                addrOScalar += static_cast<uint64_t>(qSeqlen * qHeads * embdv);
                if (this->inputLayoutType == 0) {
                    qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
                    kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
                }
                MNibd mnIbd = GetPPmnIbd(qSeqlen, kvSeqlen, embd, isMLA);
                ppMScalar = PP_MM[mnIbd.mIbd];
                ppNScalar = PP_NN[mnIbd.nIbd];
                curTotalQBlkNum += (qSeqlen != 0) ? ((qSeqlen + ppMScalar - 1) / ppMScalar) : 0;
            }
            nextProcess = process + GetBlockNum();
            if (isTriuMask) {
                uint32_t currIter = process / GetBlockNum();
                nextProcess = currIter % 2 == 1 ? (currIter + 1) * GetBlockNum() + GetBlockIdx() : (currIter + 2) * GetBlockNum() - 1 - GetBlockIdx(); // 2 is the later num
            }

            // get tiling args
            if (qSeqlen == 0 || kvSeqlen == 0) {
                continue;
            }

            uint32_t processIdx = process - preTotalQBlkNum * qHeads;
            uint32_t mIdx = processIdx / qHeads;
            uint64_t headIdx = processIdx % qHeads;
            if (alibiCoeffGm !=nullptr) {
                alibiCoeff = (float)(*((__gm__ float *)alibiCoeffGm + headIdx));
            }

            uint32_t mLoop = (qSeqlen + ppMScalar - 1) / ppMScalar;
            uint32_t nLoop = (kvSeqlen + ppNScalar - 1) / ppNScalar;

            uint32_t qkM = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * ppMScalar) : ppMScalar;
            uint32_t subM = (subBlockIdx == 1) ? (qkM - qkM / 2) : qkM / 2;
            uint32_t subMD128 = (subM + VECTOR_SIZE - 1) / VECTOR_SIZE;            // up aligned to 128
            uint32_t subMD64 = (subM + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE; // up aligned to 64
            uint32_t roundSubM = (subM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            // ******** pre_load *******
            uint32_t qkN = nLoop == 1 ? kvSeqlen : ppNScalar;
            uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint64_t maskBatchOffset = curBatch * maskStride * maxSeqlen;
            uint64_t maskHeadOffset = headIdx * headStride * maxSeqlen;
            uint64_t maskOffset = maskBatchOffset + maskHeadOffset;
            uint32_t deltaUint = 0;
            float baseY = -128;
            float delta = 0;

            if (longSeq == 0) {
                maskOffset += mIdx * ppMScalar * maxSeqlen;
            } else {
                AscendC::DataCopy(mask16UbufTensor,
                                maskGmTensor[(uint64_t)subBlockIdx * qkM / 2 * VECTOR_SIZE], // 2 is the CV_RATIO
                                AscendC::DataCopyParams(subM,
                                                        VECTOR_SIZE / BLOCK_SIZE,
                                                        0,
                                                        0)
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                AscendC::Cast<float, IN_DATA_TYPE, false>(
                    maskUbufTensor,
                    mask16UbufTensor,
                    AscendC::RoundMode::CAST_NONE,
                    (uint64_t)0,
                    subM * VECTOR_SIZE / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                );
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls<float, false>(
                    maskUbufTensor,
                    maskUbufTensor,
                    (float)-3e38,
                    (uint64_t)0,
                    subM * VECTOR_SIZE / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                );
                AscendC::PipeBarrier<PIPE_V>();
            }

            uint64_t oOffset = addrOScalar + headIdx * embdv + mIdx * ppMScalar * strideQo;
            if(tilingDataShapeType == 1){
                oOffset = addrOScalar + headIdx * embdv * maxSeqlen + mIdx * ppMScalar * strideQo;
            }
            uint32_t nEnd = nLoop;
            if (isTriuMask) {
                nEnd = mIdx + 1;
            }
            uint32_t qkNTriu = nEnd * ppNScalar;
            uint32_t sBlockStack = nEnd > 4 ? 2 : 1;
            uint32_t launchDelay = sBlockStack * 2;
            uint32_t vectMod = 2 * launchDelay;
            uint32_t mSlice = subM > 32 ? 32 : 0; // sBlockStack=2时，UB可以放下
            uint32_t mEnd = subM > 32 ? 2 : 1;
            for (uint32_t nIdx = 0; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
                if (nIdx < nEnd) {
                    if (nIdx + sBlockStack > nEnd - 1) {
                        qkN = qkNTriu > kvSeqlen ? kvSeqlen - nIdx * ppNScalar : qkNTriu - nIdx * ppNScalar;
                    } else {
                        qkN = ppNScalar * sBlockStack;
                    }
                    qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    if (qkN <= VECTOR_SIZE) {
                        if (subM > ZERO_PRECISION && maskType != MASK_TYPE_NONE_PRECISION) {
                            if (alibiCoeffGm != nullptr) {
                                if (alibiLeftAlign == ALIBI_LEFT_ALIGN_DISABLE_PRECISION) {
                                    if (nIdx == nEnd - ONE_PRECISION) {
                                        maskOffset = ZERO_PRECISION;
                                        deltaUint = ZERO_PRECISION;
                                        delta = ZERO_PRECISION;
                                    } else {
                                        maskOffset = BASE_MASK_SIZE * maxSeqlen;
                                        deltaUint = mIdx * ppMScalar - nIdx * ppNScalar;
                                        delta = baseY + deltaUint;
                                    }
                                } else {
                                    if (nIdx == nEnd - ONE_PRECISION) {
                                        maskOffset = ZERO_PRECISION;
                                    } else {
                                        maskOffset = BASE_MASK_SIZE * maxSeqlen;
                                    }
                                    delta = -baseY * nIdx;
                                }
                            } else if (maskType == MASK_TYPE_ALIBI_COMPRESS_PRECISION && alibiCompressOffset > ALIBI_COMPRESS_OFFSET_DISABLE_PRECISION) {
                                if (nIdx == nEnd - ONE_PRECISION) {
                                    maskOffset = headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                } else {
                                    deltaUint = mIdx * ppMScalar - nIdx * ppNScalar;
                                    maskOffset = BASE_MASK_SIZE * deltaUint + headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                }
                            }
                            if (longSeq == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                AscendC::DataCopyPad(mask16UbufTensor,
                                                    maskGmTensor[maskOffset + subBlockIdx * qkM / TWO_PRECISION * maxSeqlen],
                                                    AscendC::DataCopyExtParams(subM, qkN * TWO_PRECISION, (maxSeqlen - qkN) * TWO_PRECISION, 0, 0),
                                                    AscendC::DataCopyPadExtParams<IN_DATA_TYPE>(false, 0, 0, 0));
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                maskOffset += qkN;
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::Cast<float, IN_DATA_TYPE, false>(
                                    maskUbufTensor,
                                    mask16UbufTensor,
                                    AscendC::RoundMode::CAST_NONE,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                );
                                if (alibiCoeffGm != nullptr) {
                                    AscendC::PipeBarrier<PIPE_V>();
                                    if (isSqrt == 1 && mIdx != nIdx) {
                                        AscendC::Mul<float, false>(
                                            maskUbufTensor,
                                            maskUbufTensor,
                                            maskUbufTensor,
                                            (uint64_t)0,
                                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    }
                                    AscendC::Adds<float, false>(
                                        maskUbufTensor,
                                        maskUbufTensor,
                                        (float)delta,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    if (isSqrt == 1 && mIdx != nIdx) {
                                        AscendC::Sqrt<float, false>(
                                            maskUbufTensor,
                                            maskUbufTensor,
                                            (uint64_t)0,
                                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    }
                                    AscendC::Muls<float, false>(
                                        maskUbufTensor,
                                        maskUbufTensor,
                                        (float)alibiCoeff,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                if (headStride == ZERO_PRECISION && maskType != MASK_TYPE_ALIBI_COMPRESS_PRECISION) {
                                    AscendC::Muls<float, false>(
                                        maskUbufTensor,
                                        maskUbufTensor,
                                        (float)-3e38,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                }
                            }
                        }
                        AscendC::CrossCoreWaitFlag(QK_READY);
                        if (subM > 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
                            // input QK
                            AscendC::DataCopy(lsUbufTensor,
                                            sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                (uint64_t)subBlockIdx * qkM / 2 * qkRoundN], // 2 is the CV_RATIO
                                            AscendC::DataCopyParams(subM,
                                                                    qkRoundN / FLOAT_BLOCK_SIZE,
                                                                    0,
                                                                    0)
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                            // *** ls = tor * ls
                            AscendC::Muls<float, false>(
                                lsUbufTensor,
                                lsUbufTensor,
                                tor,
                                (uint64_t)0,
                                (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();

                            if (isClamp == 1) {
                                // get min(clampMin, ls_ubuf)
                                AscendC::Maxs<float, false>(
                                    lsUbufTensor,
                                    lsUbufTensor,
                                    clampMin,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                // get max(clampMin，ls_ubuf)
                                AscendC::Mins<float, false>(
                                    lsUbufTensor,
                                    lsUbufTensor,
                                    clampMax,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                            }

                            // *** ls = ls + mask
                            if (maskType != MASK_TYPE_NONE_PRECISION) {
                                if (longSeq == ZERO_PRECISION) {
                                    AscendC::Add<float, false>(
                                        lsUbufTensor,
                                        lsUbufTensor,
                                        maskUbufTensor,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - ONE_PRECISION) / FLOAT_VECTOR_SIZE,
                                        AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                    );
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                } else if (ppNScalar == FLOAT_VECTOR_SIZE && sBlockStack == SBLOCK_STACK_DOUBLE_PRECISION && nIdx == nEnd - TWO_PRECISION) {
                                    __set_mask(qkN - FLOAT_VECTOR_SIZE);
                                    AscendC::Add<float, false>(
                                        lsUbufTensor[FLOAT_VECTOR_SIZE],
                                        lsUbufTensor[FLOAT_VECTOR_SIZE],
                                        maskUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, STRIDE_16)
                                    );
                                } else if (nIdx == nEnd - 1) {
                                    if (qkN < FLOAT_VECTOR_SIZE){
                                        __set_mask(qkN);
                                    } else {
                                        __set_mask(FLOAT_VECTOR_SIZE);
                                    }
                                    AscendC::Add<float, false>(
                                        lsUbufTensor,
                                        lsUbufTensor,
                                        maskUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, STRIDE_16)
                                    );
                                    if (qkN > FLOAT_VECTOR_SIZE) {
                                        __set_mask(qkN - FLOAT_VECTOR_SIZE);
                                        AscendC::Add<float, false>(
                                            lsUbufTensor[FLOAT_VECTOR_SIZE],
                                            lsUbufTensor[FLOAT_VECTOR_SIZE],
                                            maskUbufTensor[FLOAT_VECTOR_SIZE],
                                            (uint64_t)0,
                                            subM,
                                            AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, STRIDE_16)
                                        );
                                    }
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            // *** lm = rowmax(ls)
                            if (qkN <= FLOAT_VECTOR_SIZE) {
                                __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                AscendC::BlockReduceMax<float, false>(
                                    tvUbufTensor,
                                    lsUbufTensor,
                                    subM,
                                    0,
                                    1,
                                    1,
                                    qkRoundN / FLOAT_BLOCK_SIZE
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                __set_vcg_mask((qkN + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
                                AscendC::BlockReduceMax<float, false>(
                                    lmUbufTensor,
                                    tvUbufTensor,
                                    (subM * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    0,
                                    1,
                                    1,
                                    STRIDE_8
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            } else {
                                AscendC::BlockReduceMax<float, false>(
                                    tvUbufTensor,
                                    lsUbufTensor,
                                    subM,
                                    0,
                                    1,
                                    1,
                                    qkRoundN / EIGHT_PRECISION
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::BlockReduceMax<float, false>(
                                    lmUbufTensor,
                                    tvUbufTensor,
                                    roundSubM * EIGHT_PRECISION / SIXTY_FOUR_PRECISION,
                                    0,
                                    1,
                                    1,
                                    STRIDE_8
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                for (uint32_t rowMaxIdx = 1; rowMaxIdx < qkN / FLOAT_VECTOR_SIZE; ++rowMaxIdx) {
                                    AscendC::BlockReduceMax<float, false>(
                                        tvUbufTensor,
                                        lsUbufTensor[rowMaxIdx * FLOAT_VECTOR_SIZE],
                                        subM,
                                        0,
                                        1,
                                        1,
                                        qkRoundN / EIGHT_PRECISION
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::BlockReduceMax<float, false>(
                                        tvUbufTensor,
                                        tvUbufTensor,
                                        roundSubM * 8 / 64, // roundSubM * 8 / 64 is the repeate time
                                        0,
                                        1,
                                        1,
                                        STRIDE_8
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    __set_mask(subM);
                                    AscendC::Max<float, false>(
                                            lmUbufTensor,
                                            lmUbufTensor,
                                            tvUbufTensor,
                                            (uint64_t)0,
                                            1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                    AscendC::PipeBarrier<PIPE_V>();
                                }
                                if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                    __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                    AscendC::BlockReduceMax<float, false>(
                                        tvUbufTensor,
                                        lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        subM,
                                        0,
                                        1,
                                        1,
                                        qkRoundN / EIGHT_PRECISION
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    __set_vcg_mask((qkN % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
                                    AscendC::BlockReduceMax<float, false>(
                                        tvUbufTensor,
                                        tvUbufTensor,
                                        roundSubM * 8 / 64, // roundSubM * 8 / 64 is the repeate time
                                        0,
                                        1,
                                        1,
                                        STRIDE_8
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    __set_mask(subM);
                                    AscendC::Max<float, false>(
                                        lmUbufTensor,
                                        lmUbufTensor,
                                        tvUbufTensor,
                                        (uint64_t)0,
                                        1,
                                        AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                    );
                                }
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            if (nIdx == 0) {
                                // *** hm = lm
                                AscendC::DataCopy(hmUbufTensor,
                                                lmUbufTensor,
                                                AscendC::DataCopyParams(1,
                                                roundSubM / FLOAT_BLOCK_SIZE,
                                                0,
                                                0)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                            } else {
                                // *** hm = MAX(lm, gm)
                                AscendC::Max<float, false>(
                                    hmUbufTensor,
                                    lmUbufTensor,
                                    gmUbufTensor,
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** dm = gm - hm
                                AscendC::Sub<float, false>(
                                    dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE], // 4 is the BlockStack num
                                    gmUbufTensor,
                                    hmUbufTensor,
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** dm = exp(dm)
                                AscendC::Exp<float, false>(
                                    dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE], // 4 is the BlockStack num
                                    dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE], // 4 is the BlockStack num
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                            }
                            // *** gm = hm
                            AscendC::DataCopy(gmUbufTensor,
                                            hmUbufTensor,
                                            AscendC::DataCopyParams(1,
                                            roundSubM / FLOAT_BLOCK_SIZE,
                                            0,
                                            0)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** hm_block = expand_to_block(hm), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint32_t>(),
                                hmUbufTensor.ReinterpretCast<uint32_t>(),
                                roundSubM / FLOAT_BLOCK_SIZE,
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** ls = ls - hm_block
                            for (uint32_t vsubIdx = 0; vsubIdx < qkN / FLOAT_VECTOR_SIZE; ++vsubIdx) {
                                AscendC::Sub<float, false>(
                                    lsUbufTensor[vsubIdx * FLOAT_VECTOR_SIZE],
                                    lsUbufTensor[vsubIdx * FLOAT_VECTOR_SIZE],
                                    tvUbufTensor,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                );
                            }
                            if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                AscendC::Sub<float, false>(
                                    lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                    lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                    tvUbufTensor,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** ls = exp(ls)
                            AscendC::Exp<float, false>(
                                ls32UbufTensor,
                                lsUbufTensor,
                                (uint64_t)0,
                                (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** lp = castfp32to16(ls)
                            if (IS_BF16) {
                                AscendC::Cast<IN_DATA_TYPE, float, false>(
                                    lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                    ls32UbufTensor,
                                    AscendC::RoundMode::CAST_RINT,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                );
                            } else {
                                if constexpr(std::is_same<float, float>::value && std::is_same<IN_DATA_TYPE, __bf16>::value){
                                    AscendC::Cast<IN_DATA_TYPE, float, false>(
                                        lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                        ls32UbufTensor,
                                        AscendC::RoundMode::CAST_RINT,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                    );
                                }else {
                                    AscendC::Cast<IN_DATA_TYPE, float, false>(
                                        lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                        ls32UbufTensor,
                                        AscendC::RoundMode::CAST_NONE,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                    );
                                }
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            AscendC::DataCopy(pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                        (uint64_t)subBlockIdx * qkM / 2 * qkRoundN], // 2 is the CV_RATIO
                                            lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                            AscendC::DataCopyParams(1,
                                                                    subM * qkRoundN / BLOCK_SIZE,
                                                                    0,
                                                                    0)
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
                            // *** ll = rowsum(ls32)
                            if (qkN <= FLOAT_VECTOR_SIZE) {
                                __set_mask(qkN);
                                AscendC::RepeatReduceSum<float, false>(
                                    llUbufTensor,
                                    ls32UbufTensor,
                                    subM,
                                    0,
                                    0,
                                    1,                             // dstRepeatStride
                                    1,                             // srcBlockStride
                                    qkRoundN / FLOAT_BLOCK_SIZE  // srcRepeatStride
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            } else {
                                for (uint32_t rowSumIdx = 1; rowSumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                                    AscendC::Add<float, false>(
                                        ls32UbufTensor,
                                        ls32UbufTensor,
                                        ls32UbufTensor[rowSumIdx * FLOAT_VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE)
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                }
                                if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                    __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                    AscendC::Add<float, false>(
                                        ls32UbufTensor,
                                        ls32UbufTensor,
                                        ls32UbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE)
                                    );
                                }
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::RepeatReduceSum<float, false>(
                                    llUbufTensor,
                                    ls32UbufTensor,
                                    subM,
                                    0,
                                    0,
                                    1,                             // dstRepeatStride
                                    1,                             // srcBlockStride
                                    qkRoundN / FLOAT_BLOCK_SIZE  // srcRepeatStride
                                );
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            if (nIdx == 0) {
                                // *** gl = ll
                                AscendC::DataCopy(glUbufTensor,
                                                llUbufTensor,
                                                AscendC::DataCopyParams(1,
                                                roundSubM / FLOAT_BLOCK_SIZE,
                                                0,
                                                0)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                            } else {
                                // *** gl = dm * gl
                                AscendC::Mul<float, false>(
                                    glUbufTensor,
                                    dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE], // 4 is the BlockStack num
                                    glUbufTensor,
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** gl = ll + gl
                                AscendC::Add<float, false>(
                                    glUbufTensor,
                                    glUbufTensor,
                                    llUbufTensor,
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                        }
                    }
                    else {
                        bool last_n_loop = nIdx + sBlockStack > nEnd - 1;
                        for (uint32_t splitIdx = 0; splitIdx < mEnd; splitIdx++) {
                            bool lastMLoop = splitIdx == mEnd - 1;
                            uint32_t mSplit =  lastMLoop ? subM - splitIdx * mSlice : mSlice;
                            uint32_t roundMSplit = (mSplit + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
                            if (subM > 0 && maskType != 0 && longSeq == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                uint64_t maskOffsetTail = 0;
                                if (alibiCoeffGm != nullptr) {
                                    maskOffset = BASE_MASK_SIZE * SOFTMAX_MAX_LENGTH;
                                    if (alibiLeftAlign == ALIBI_LEFT_ALIGN_DISABLE_PRECISION) {
                                        delta = baseY * (nIdx + 1 - mIdx);
                                    } else {
                                        delta = -baseY * nIdx;
                                    }
                                    AscendC::DataCopy(mask16UbufTensor,
                                                    maskGmTensor[maskOffset + (subBlockIdx * qkM / TWO_PRECISION + splitIdx * mSlice) * SOFTMAX_MAX_LENGTH],
                                                    AscendC::DataCopyParams(mSplit,
                                                                            qkRoundN / BLOCK_SIZE,
                                                                            (SOFTMAX_MAX_LENGTH - qkRoundN) / BLOCK_SIZE,
                                                                            0)
                                    );
                                } else if (maskType == MASK_TYPE_ALIBI_COMPRESS_PRECISION && alibiCompressOffset > ALIBI_COMPRESS_OFFSET_DISABLE_PRECISION) {
                                    deltaUint = mIdx * ppMScalar - nIdx * ppNScalar;
                                    maskOffset = BASE_MASK_SIZE * deltaUint + headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                    maskOffsetTail = maskOffset - BASE_MASK_SIZE * ppNScalar;
                                    if (nIdx == nEnd - TWO_PRECISION) {
                                        maskOffsetTail = headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                    }
                                    AscendC::DataCopy(mask16UbufTensor,
                                                    maskGmTensor[maskOffset + (subBlockIdx * qkM / 2 + splitIdx * mSlice) * VECTOR_SIZE], // 2 is the CV_RATIO
                                                    AscendC::DataCopyParams(mSplit,
                                                                            8, // 8 is the block count
                                                                            0,
                                                                            (qkRoundN - VECTOR_SIZE)/ BLOCK_SIZE)
                                    );
                                    AscendC::DataCopy(mask16UbufTensor[VECTOR_SIZE],
                                                    maskGmTensor[maskOffsetTail + (subBlockIdx * qkM / 2 + splitIdx * mSlice) * VECTOR_SIZE],  // 2 is the CV_RATIO
                                                    AscendC::DataCopyParams(mSplit,
                                                                            (qkRoundN - VECTOR_SIZE)/ BLOCK_SIZE,
                                                                            (SOFTMAX_MAX_LENGTH - qkRoundN)/ BLOCK_SIZE,
                                                                            STRIDE_8)
                                    );
                                } else {
                                    AscendC::DataCopyPad(mask16UbufTensor, maskGmTensor[maskOffset + (subBlockIdx * qkM / TWO_PRECISION + splitIdx * mSlice) * maxSeqlen], AscendC::DataCopyExtParams(mSplit, qkN * TWO_PRECISION, (maxSeqlen - qkN) * TWO_PRECISION, 0, 0),
                                                        AscendC::DataCopyPadExtParams<IN_DATA_TYPE>(false, 0, 0, 0));
                                }
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                            }
                            if (splitIdx == 0) {
                                AscendC::CrossCoreWaitFlag(QK_READY);
                            }
                            if (subM > 0) {
                                if (mSplit > 0) {
                                    if (maskType != 0 && longSeq == 0) {
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                        AscendC::Cast<float, IN_DATA_TYPE, false>(
                                            maskUbufTensor,
                                            mask16UbufTensor,
                                            AscendC::RoundMode::CAST_NONE,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                        );
                                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                        if (alibiCoeffGm != nullptr) {
                                            AscendC::PipeBarrier<PIPE_V>();
                                            if (nIdx != nEnd - TWO_PRECISION) {
                                                if (isSqrt == 1) {
                                                    AscendC::Mul<float, false>(
                                                        maskUbufTensor,
                                                        maskUbufTensor,
                                                        maskUbufTensor,
                                                        (uint64_t)0,
                                                        (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                        AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                                    );
                                                    AscendC::PipeBarrier<PIPE_V>();
                                                }
                                                AscendC::Adds<float, false>(
                                                    maskUbufTensor,
                                                    maskUbufTensor,
                                                    (float)delta,
                                                    (uint64_t)0,
                                                    (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                                );
                                                AscendC::PipeBarrier<PIPE_V>();
                                                if (alibiLeftAlign == 1) {
                                                    AscendC::Adds<float, false>(
                                                        maskUbufTensor[128], // 128 is the Tensor length
                                                        maskUbufTensor,
                                                        (float)-baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / EIGHT_PRECISION, qkRoundN / EIGHT_PRECISION)
                                                    );
                                                    AscendC::Adds<float, false>(
                                                        maskUbufTensor[128 + FLOAT_VECTOR_SIZE], // 128 is the Tensor length
                                                        maskUbufTensor[FLOAT_VECTOR_SIZE],
                                                        (float)-baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / EIGHT_PRECISION, qkRoundN / EIGHT_PRECISION)
                                                    );
                                                } else {
                                                    AscendC::Adds<float, false>(
                                                        maskUbufTensor[128], // 128 is the Tensor length
                                                        maskUbufTensor,
                                                        baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / EIGHT_PRECISION, qkRoundN / EIGHT_PRECISION)
                                                    );
                                                    AscendC::Adds<float, false>(
                                                        maskUbufTensor[128 + FLOAT_VECTOR_SIZE],// 128 + FLOAT_VECTOR_SIZE is the Tensor length
                                                        maskUbufTensor[FLOAT_VECTOR_SIZE],
                                                        baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / EIGHT_PRECISION, qkRoundN / EIGHT_PRECISION)
                                                    );
                                                }
                                                AscendC::PipeBarrier<PIPE_V>();
                                                if (isSqrt == 1) {
                                                    AscendC::Sqrt<float, false>(
                                                        maskUbufTensor,
                                                        maskUbufTensor,
                                                        (uint64_t)0,
                                                        (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                                    );
                                                    AscendC::PipeBarrier<PIPE_V>();
                                                }
                                            } else if (alibiLeftAlign == 1) {
                                                AscendC::Adds<float, false>(
                                                    maskUbufTensor,
                                                    maskUbufTensor,
                                                    (float)delta,
                                                    (uint64_t)0,
                                                    (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                                );
                                                AscendC::PipeBarrier<PIPE_V>();
                                            }
                                            AscendC::Muls<float, false>(
                                                maskUbufTensor,
                                                maskUbufTensor,
                                                (float)alibiCoeff,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                            );
                                        }
                                        AscendC::PipeBarrier<PIPE_V>();
                                        if (headStride == 0 && maskType != 2) { // 2is the mask type
                                            AscendC::Muls<float, false>(
                                                maskUbufTensor,
                                                maskUbufTensor,
                                                (float)-3e38,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                            );
                                            AscendC::PipeBarrier<PIPE_V>();
                                        }
                                    }
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
                                    AscendC::DataCopy(lsUbufTensor,
                                                    sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                        (uint64_t)(subBlockIdx * qkM / 2 + splitIdx * mSlice) * qkRoundN], // 2 is the CV_RATIO
                                                    AscendC::DataCopyParams(mSplit,
                                                                            qkRoundN / FLOAT_BLOCK_SIZE,
                                                                            0,
                                                                            0)
                                    );
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                    // *** ls = tor * ls
                                    AscendC::Muls<float, false>(
                                        lsUbufTensor,
                                        lsUbufTensor,
                                        tor,
                                        (uint64_t)0,
                                        (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    if (isClamp == 1) {
                                        // get min(clampMin，ls_ubuf)
                                        AscendC::Maxs<float, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            clampMin,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        // get max(clampMin，ls_ubuf)
                                        AscendC::Mins<float, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            clampMax,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    }
                                    // *** ls = ls + mask
                                    if (maskType != 0) {
                                        if (longSeq == 0) {
                                            AscendC::Add<float, false>(
                                                lsUbufTensor,
                                                lsUbufTensor,
                                                maskUbufTensor,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                            );
                                        } else if (nIdx == nEnd - TWO_PRECISION) {
                                            if (qkN - ppNScalar < FLOAT_VECTOR_SIZE){
                                                __set_mask(qkN - ppNScalar);
                                            } else {
                                                __set_mask(FLOAT_VECTOR_SIZE);
                                            }
                                            AscendC::Add<float, false>(
                                                lsUbufTensor[ppNScalar],
                                                lsUbufTensor[ppNScalar],
                                                maskUbufTensor[splitIdx * mSlice * VECTOR_SIZE],
                                                (uint64_t)0,
                                                mSplit,
                                                AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, STRIDE_16)
                                            );
                                            if (qkN - ppNScalar > FLOAT_VECTOR_SIZE) {
                                                __set_mask(qkN - ppNScalar - FLOAT_VECTOR_SIZE);
                                                AscendC::Add<float, false>(
                                                    lsUbufTensor[ppNScalar + FLOAT_VECTOR_SIZE],
                                                    lsUbufTensor[ppNScalar + FLOAT_VECTOR_SIZE],
                                                    maskUbufTensor[FLOAT_VECTOR_SIZE + splitIdx * mSlice * VECTOR_SIZE],
                                                    (uint64_t)0,
                                                    mSplit,
                                                    AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, STRIDE_16)
                                                );
                                            }
                                        }
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                    }
                                    if (qkN == SOFTMAX_MAX_LENGTH) {
                                        AscendC::BlockReduceMax<float, false>(
                                            tvUbufTensor,
                                            lsUbufTensor,
                                            mSplit * qkN / FLOAT_VECTOR_SIZE,
                                            0,
                                            1,
                                            1,
                                            STRIDE_8
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        __set_mask(THIRTY_TWO_PRECISION);
                                        AscendC::BlockReduceMax<float, false>(
                                             tvUbufTensor,
                                            tvUbufTensor,
                                            mSplit,
                                            0,
                                            1,
                                            1,
                                            STRIDE_4
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        __set_vcg_mask(FOUR_PRECISION);
                                        AscendC::BlockReduceMax<float, false>(
                                            lmUbufTensor,
                                            tvUbufTensor,
                                            (mSplit * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            0,
                                            1,
                                            1,
                                            STRIDE_8
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        __set_mask(mSplit);
                                    } else {
                                        AscendC::BlockReduceMax<float, false>(
                                            tvUbufTensor,
                                            lsUbufTensor,
                                            mSplit,
                                            0,
                                            1,
                                            1,
                                            qkRoundN / FLOAT_BLOCK_SIZE
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::BlockReduceMax<float, false>(
                                            lmUbufTensor,
                                            tvUbufTensor,
                                            (mSplit * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            0,
                                            1,
                                            1,
                                            STRIDE_8
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        for (uint64_t rowmaxIdx = 1; rowmaxIdx < (uint64_t)qkN / FLOAT_VECTOR_SIZE; ++rowmaxIdx) {
                                            AscendC::BlockReduceMax<float, false>(
                                                tvUbufTensor,
                                                lsUbufTensor[rowmaxIdx * FLOAT_VECTOR_SIZE],
                                                mSplit,
                                                0,
                                                1,
                                                1,
                                                qkRoundN / FLOAT_BLOCK_SIZE
                                            );
                                            AscendC::PipeBarrier<PIPE_V>();
                                            AscendC::BlockReduceMax<float, false>(
                                                tvUbufTensor,
                                                tvUbufTensor,
                                                (mSplit * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                0,
                                                1,
                                                1,
                                                STRIDE_8
                                            );
                                            AscendC::PipeBarrier<PIPE_V>();
                                            __set_mask(mSplit);
                                            AscendC::Max<float, false>(
                                                lmUbufTensor,
                                                lmUbufTensor,
                                                tvUbufTensor,
                                                (uint64_t)0,
                                                1,
                                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                            );
                                            AscendC::PipeBarrier<PIPE_V>();
                                        }
                                        if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                            __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                            AscendC::BlockReduceMax<float, false>(
                                                tvUbufTensor,
                                                lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                mSplit,
                                                0,
                                                1,
                                                1,
                                                qkRoundN / FLOAT_BLOCK_SIZE
                                            );
                                            AscendC::PipeBarrier<PIPE_V>();
                                            __set_vcg_mask((qkN % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
                                            AscendC::BlockReduceMax<float, false>(
                                                tvUbufTensor,
                                                tvUbufTensor,
                                                (mSplit * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                0,
                                                1,
                                                1,
                                                STRIDE_8
                                            );
                                            AscendC::PipeBarrier<PIPE_V>();
                                            __set_mask(mSplit);
                                            AscendC::Max<float, false>(
                                                lmUbufTensor,
                                                lmUbufTensor,
                                                tvUbufTensor,
                                                (uint64_t)0,
                                                1,
                                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                            );
                                        }
                                    }
                                    AscendC::PipeBarrier<PIPE_V>();
                                    if (nIdx == 0) {
                                        // *** hm = lm
                                        AscendC::DataCopy(hmUbufTensor[splitIdx * mSlice],
                                                        lmUbufTensor,
                                                        AscendC::DataCopyParams(1,
                                                        roundMSplit / FLOAT_BLOCK_SIZE,
                                                        0,
                                                        0)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    } else {
                                        // *** hm = MAX(lm, gm)
                                        AscendC::Max<float, false>(
                                            hmUbufTensor[splitIdx * mSlice],
                                            lmUbufTensor,
                                            gmUbufTensor[splitIdx * mSlice],
                                            (uint64_t)0,
                                            1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        // *** dm = gm - hm
                                        AscendC::Sub<float, false>(
                                            dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
                                            gmUbufTensor[splitIdx * mSlice],
                                            hmUbufTensor[splitIdx * mSlice],
                                            (uint64_t)0,
                                            1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        // *** dm = exp(dm)
                                        AscendC::Exp<float, false>(
                                            dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
                                            dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
                                            (uint64_t)0,
                                            1,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                    }
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                    AscendC::PipeBarrier<PIPE_V>();
                                    // *** gm = hm
                                    AscendC::DataCopy(gmUbufTensor[splitIdx * mSlice],
                                                    hmUbufTensor[splitIdx * mSlice],
                                                    AscendC::DataCopyParams(1,
                                                    roundMSplit / FLOAT_BLOCK_SIZE,
                                                    0,
                                                    0)
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    // *** hm_block = expand_to_block(hm), 存放于 tv
                                    AscendC::Brcb(
                                        tvUbufTensor.ReinterpretCast<uint32_t>(),
                                        hmUbufTensor.ReinterpretCast<uint32_t>()[splitIdx * mSlice],
                                        roundMSplit / FLOAT_BLOCK_SIZE,
                                        AscendC::BrcbRepeatParams(1, STRIDE_8)
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    // *** ls = ls - hm_block
                                    for (uint32_t vsubIdx = 0; vsubIdx < qkN / FLOAT_VECTOR_SIZE; ++vsubIdx) {
                                        AscendC::Sub<float, false>(
                                            lsUbufTensor[vsubIdx * FLOAT_VECTOR_SIZE],
                                            lsUbufTensor[vsubIdx * FLOAT_VECTOR_SIZE],
                                            tvUbufTensor,
                                            (uint64_t)0,
                                            mSplit,
                                            AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                        );
                                    }
                                    if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                        __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                        AscendC::Sub<float, false>(
                                            lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            tvUbufTensor,
                                            (uint64_t)0,
                                            mSplit,
                                            AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                        );
                                        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                    }
                                    AscendC::PipeBarrier<PIPE_V>();
                                    // *** ls = exp(ls)
                                    AscendC::Exp<float, false>(
                                        ls32UbufTensor,
                                        lsUbufTensor,
                                        (uint64_t)0,
                                        (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    // *** lp = castfp32to16(ls)
                                    if (IS_BF16) {
                                        AscendC::Cast<IN_DATA_TYPE, float, false>(
                                            lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                            ls32UbufTensor,
                                            AscendC::RoundMode::CAST_RINT,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                        );
                                    } else {
                                        if constexpr(std::is_same<float, float>::value && std::is_same<IN_DATA_TYPE, __bf16>::value) {
                                            AscendC::Cast<IN_DATA_TYPE, float, false>(
                                                lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                                ls32UbufTensor,
                                                AscendC::RoundMode::CAST_RINT,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                            );
                                        }else {
                                            AscendC::Cast<IN_DATA_TYPE, float, false>(
                                                lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                                ls32UbufTensor,
                                                AscendC::RoundMode::CAST_NONE,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                            );
                                        }
                                    }
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                                    AscendC::DataCopy(pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                    ((uint64_t)subBlockIdx * qkM / 2 + splitIdx * mSlice) * qkRoundN], // 2 is the CV_RAITO
                                                    lpUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                                    AscendC::DataCopyParams(mSplit,
                                                                            qkRoundN / BLOCK_SIZE,
                                                                            0,
                                                                            0)
                                    );
                                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
                                    // *** ll = rowsum(ls32)
                                    for (uint32_t rowsumIdx = 1; rowsumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowsumIdx) {
                                        AscendC::Add<float, false>(
                                            ls32UbufTensor,
                                            ls32UbufTensor,
                                            ls32UbufTensor[rowsumIdx * FLOAT_VECTOR_SIZE],
                                            (uint64_t)0,
                                            mSplit,
                                            AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    }
                                    if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                        __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                        AscendC::Add<float, false>(
                                            ls32UbufTensor,
                                            ls32UbufTensor,
                                            ls32UbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                            (uint64_t)0,
                                            mSplit,
                                            AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE)
                                        );
                                        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                    }
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::RepeatReduceSum<float, false>(
                                        llUbufTensor,
                                        ls32UbufTensor,
                                        mSplit,
                                        0,
                                        0,
                                        1,                             // dstRepeatStride
                                        1,                             // srcBlockStride
                                        qkRoundN / FLOAT_BLOCK_SIZE  // srcRepeatStride
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    if (nIdx == 0) {
                                        // *** gl = ll
                                        AscendC::DataCopy(glUbufTensor[splitIdx * mSlice],
                                                        llUbufTensor,
                                                        AscendC::DataCopyParams(1,
                                                        roundMSplit / FLOAT_BLOCK_SIZE,
                                                        0,
                                                        0)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    } else {
                                        __set_mask(mSplit);
                                        // *** gl = dm * gl
                                        AscendC::Mul<float, false>(
                                            glUbufTensor[splitIdx * mSlice],
                                            dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
                                            glUbufTensor[splitIdx * mSlice],
                                            (uint64_t)0,
                                            1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        // *** gl = ll + gl
                                        AscendC::Add<float, false>(
                                            glUbufTensor[splitIdx * mSlice],
                                            glUbufTensor[splitIdx * mSlice],
                                            llUbufTensor,
                                            (uint64_t)0,
                                            1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                    }
                                }
                            }
                        }
                        maskOffset += qkN;
                    }
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SOFTMAX_READY); // 2 is the Sync mode
                }
                if (nIdx >= launchDelay) {
                    AscendC::CrossCoreWaitFlag(UPDATE_READY);
                    for(uint32_t kIdx = 0; kIdx < loopV; kIdx++){
                        uint64_t oOffsetk = oOffset + kIdx * BLOCK_QK;
                        qkK = (kIdx == (loopV - 1))? embdv - kIdx * BLOCK_QK : BLOCK_QK;
                        qkRoundK = (qkK + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                        if (subM > 0) {
                            if(nIdx == launchDelay){
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                                AscendC::DataCopy(goUbufTensor,
                                                oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE * LOCAL_SIZE + kIdx * TMP_SIZET  + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod * loopV +
                                                    (uint64_t)subBlockIdx * qkM / 2 * qkRoundK], // 2 is the CV_RATIO
                                                AscendC::DataCopyParams(1,
                                                                        subM * qkRoundK / FLOAT_BLOCK_SIZE,
                                                                        0,
                                                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3)); // (kIdx+loopV) * (TMP_SIZE / 2)
                            }
                            else{
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                                AscendC::DataCopy(goUbufTensor,
                                                upoGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + kIdx * TMP_SIZET +
                                                    (uint64_t)subBlockIdx * qkM / 2 * qkRoundK], // 2 is the CV_RATIO
                                                AscendC::DataCopyParams(1,
                                                                        subM * qkRoundK / FLOAT_BLOCK_SIZE,
                                                                        0,
                                                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                                AscendC::DataCopy(loUbufTensor,
                                                oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE * LOCAL_SIZE + kIdx * TMP_SIZET  + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod * loopV +
                                                    (uint64_t)subBlockIdx * qkM / 2 * qkRoundK], // 2 is the CV_RATIO
                                                AscendC::DataCopyParams(1,
                                                                        subM * qkRoundK / FLOAT_BLOCK_SIZE,
                                                                        0,
                                                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** dm_block = expand_to_block(dm),存放于 tv
                                AscendC::Brcb(
                                    tvUbufTensor.ReinterpretCast<uint32_t>(),
                                    dmUbufTensor[((nIdx - launchDelay) / sBlockStack % 4) * UB_FLOAT_LINE_SIZE].ReinterpretCast<uint32_t>(), // 4 is the BlockStack num
                                    roundSubM / FLOAT_BLOCK_SIZE,
                                    AscendC::BrcbRepeatParams(1, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));
                                // *** go = go * dm_block
                                for (uint32_t vmulIdx = 0; vmulIdx < qkK / FLOAT_VECTOR_SIZE; ++vmulIdx) {
                                    AscendC::Mul<float, false>(
                                        goUbufTensor[vmulIdx * FLOAT_VECTOR_SIZE],
                                        goUbufTensor[vmulIdx * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundK / FLOAT_BLOCK_SIZE, qkRoundK / FLOAT_BLOCK_SIZE, 1)
                                    );
                                }
                                if (qkK % FLOAT_VECTOR_SIZE > 0) {
                                    __set_mask(qkK % FLOAT_VECTOR_SIZE);
                                    AscendC::Mul<float, false>(
                                        goUbufTensor[qkK / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        goUbufTensor[qkK / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundK / FLOAT_BLOCK_SIZE, qkRoundK / FLOAT_BLOCK_SIZE, 1)
                                    );
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));
                                // *** go = lo + go
                                AscendC::Add<float, false>(
                                    goUbufTensor,
                                    goUbufTensor,
                                    loUbufTensor,
                                    (uint64_t)0,
                                    (subM * qkRoundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                            }
                            // *** gl = castfp32to16(gl)
                            if (nIdx + sBlockStack > nEnd + launchDelay - 1){
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** gl_block = expand_to_block(gl), 存放于 tv
                                AscendC::Brcb(
                                    tvUbufTensor.ReinterpretCast<uint32_t>(),
                                    glUbufTensor.ReinterpretCast<uint32_t>(),
                                    roundSubM / FLOAT_BLOCK_SIZE,
                                    AscendC::BrcbRepeatParams(1, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** go = go / gl_block
                                for (uint32_t vdivIdx = 0; vdivIdx < qkK / FLOAT_VECTOR_SIZE; ++vdivIdx) {
                                    AscendC::Div<float, false>(
                                        goUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                        goUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundK / FLOAT_BLOCK_SIZE, qkRoundK / FLOAT_BLOCK_SIZE, 1)
                                    );
                                }
                                if (qkK % FLOAT_VECTOR_SIZE > 0) {
                                    __set_mask(qkK % FLOAT_VECTOR_SIZE);
                                    AscendC::Div<float, false>(
                                        goUbufTensor[qkK / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        goUbufTensor[qkK / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundK / FLOAT_BLOCK_SIZE, qkRoundK / FLOAT_BLOCK_SIZE, 1)
                                    );
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                if (IS_BF16) {
                                    AscendC::Cast<IN_DATA_TYPE, float, false>(
                                        goUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                        goUbufTensor,
                                        AscendC::RoundMode::CAST_RINT,
                                        (uint64_t)0,
                                        (subM * qkRoundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                    );
                                } else {
                                    if constexpr(std::is_same<float, float>::value && std::is_same<IN_DATA_TYPE, __bf16>::value){
                                        AscendC::Cast<IN_DATA_TYPE, float, false>(
                                            goUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                            goUbufTensor,
                                            AscendC::RoundMode::CAST_RINT,
                                            (uint64_t)0,
                                            (subM * qkRoundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                        );
                                    }else {
                                        AscendC::Cast<IN_DATA_TYPE, float, false>(
                                            goUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                            goUbufTensor,
                                            AscendC::RoundMode::CAST_NONE,
                                            (uint64_t)0,
                                            (subM * qkRoundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                        );
                                    }
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                // ********************* move O to GM ************************
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID1));
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID1));
                                AscendC::DataCopyPad(oGmTensor[oOffsetk + (uint64_t)subBlockIdx * qkM / TWO_PRECISION * strideQo],
                                                    goUbufTensor.ReinterpretCast<IN_DATA_TYPE>(),
                                                    AscendC::DataCopyExtParams(subM,
                                                                            qkK * TWO_PRECISION,
                                                                            0,
                                                                            (strideQo - qkK) * TWO_PRECISION,
                                                                            0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                            }
                            else{
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID2));
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID2));
                                AscendC::DataCopy(upoGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE +  kIdx * TMP_SIZET +
                                                (uint64_t)subBlockIdx * qkM / 2 * qkRoundK], // 2 is the CV_RATIO
                                                goUbufTensor,
                                                AscendC::DataCopyParams(1,
                                                                        subM * qkRoundK / FLOAT_BLOCK_SIZE,
                                                                        0,
                                                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                            }
                        }
                    }
                }
            }
        }
    }

private:
    using IN_DATA_TYPE = typename PFAT::inDataType;
    static constexpr bool IS_BF16 = PFAT::isBF16;

    const typename PFAT::tilingType* __restrict tiling;
    const uint32_t lsUbufOffset = 0;
    const uint32_t lpUbufOffset = 0;
    const uint32_t ls32UbufOffset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t maskUbufOffset = 4 * UB_UINT8_BLOCK_SIZE;
    const uint32_t loUbufOffset = 6 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lmUbufOffset = 8 * UB_UINT8_BLOCK_SIZE;
    const uint32_t hmUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE;
    const uint32_t gmUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE;
    const uint32_t dmUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 4 * UB_UINT8_LINE_SIZE;
    const uint32_t llUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 8 * UB_UINT8_LINE_SIZE;
    const uint32_t glUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_LINE_SIZE;
    const uint32_t tvUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_LINE_SIZE;
    const uint32_t goUbufOffset = 9 * UB_UINT8_BLOCK_SIZE;
    const uint32_t mask16UbufOffset = 11 * UB_UINT8_BLOCK_SIZE;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::LocalTensor<float> lsUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lsUbufOffset);
    AscendC::LocalTensor<float> lpUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lpUbufOffset);
    AscendC::LocalTensor<float> ls32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32UbufOffset);
    AscendC::LocalTensor<float> maskUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(maskUbufOffset);
    AscendC::LocalTensor<float> loUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbufOffset);
    AscendC::LocalTensor<float> lmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lmUbufOffset);
    AscendC::LocalTensor<float> hmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(hmUbufOffset);
    AscendC::LocalTensor<float> gmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(gmUbufOffset);
    AscendC::LocalTensor<float> dmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(dmUbufOffset);
    AscendC::LocalTensor<float> llUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(llUbufOffset);
    AscendC::LocalTensor<float> glUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(glUbufOffset);
    AscendC::LocalTensor<float> tvUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(tvUbufOffset);
    AscendC::LocalTensor<float> goUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbufOffset);
    AscendC::LocalTensor<IN_DATA_TYPE> mask16UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, IN_DATA_TYPE>(mask16UbufOffset);

    AscendC::GlobalTensor<IN_DATA_TYPE> maskGmTensor;
    AscendC::GlobalTensor<IN_DATA_TYPE> oGmTensor;
    AscendC::GlobalTensor<float> sGmTensor;
    AscendC::GlobalTensor<IN_DATA_TYPE> pGmTensor;
    AscendC::GlobalTensor<float> oTmpGmTensor;
    AscendC::GlobalTensor<float> upoGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    __gm__ uint8_t *__restrict__ alibiCoeffGm{nullptr};

    uint32_t batchSize{0};
    uint32_t maxSeqlen{0};
    uint32_t maxKvSeqlen{0};
    uint32_t qHeads{0};
    uint32_t embd{0};
    uint32_t embdv{0};
    float tor{0};
    uint32_t headStride{0};
    uint32_t maskStride{0};
    uint32_t isTriuMask{0};
    uint32_t totalQBlkNum{0};
    uint32_t isClamp{0};
    float clampMin;
    float clampMax;
    uint64_t strideQo{0};
    uint32_t goFlagScalar{1};
    int32_t subBlockIdx{-1};
    uint32_t longSeq{0};
    uint32_t isSqrt{0};
    uint32_t maskType{0};
    uint32_t alibiCompressOffset{0};
    uint32_t alibiLeftAlign{0};
    uint32_t tilingDataShapeType{0};
    uint32_t inputLayoutType{0};
};
#endif

template <typename PFAT>
class PromptFlashAttentionBaseMLAHighPrecision {
public:
    __aicore__ inline PromptFlashAttentionBaseMLAHighPrecision() {};
    __aicore__ inline void Init(__gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
                                const typename PFAT::tilingType* __restrict tiling,
                                __gm__ uint8_t* gmTiling, TPipe* tPipe, __gm__ uint8_t* alibiCoeff) {
        #ifdef __DAV_C220_CUBE__
            FlashAttentionEncoderHighPrecisionMLA<PFAT> fa_cube(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                                queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                                attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe);
            fa_cube.Run();
        #elif __DAV_C220_VEC__
            FlashAttentionEncoderHighPrecisionMLAVec<PFAT> fa_vec(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                                queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                                attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe, alibiCoeff);
            fa_vec.Run();
        #endif
    }
};

#endif  // PROMPT_FLASH_ATTENTION_BASE_MLA_HIGH_PRECISION_H
