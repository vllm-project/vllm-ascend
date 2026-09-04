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
 * \file prompt_flash_attention_base_api_high_precision_no_mask.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_BASE_API_HIGH_PRECISION_NO_MASK_H
#define PROMPT_FLASH_ATTENTION_BASE_API_HIGH_PRECISION_NO_MASK_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common_func.h"
#include "hardware.h"
#include "mem.h"

constexpr int32_t ONE_NO_MASK = 1;
constexpr int32_t TWO_NO_MASK = 2;
constexpr uint32_t PINGPONG_FLAG_M_MTE1_OFFSET_ONE_NO_MASK = 1;
constexpr uint32_t PINGPONG_FLAG_M_MTE1_OFFSET_TWO_NO_MASK = 2;
constexpr uint32_t PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_NO_MASK = 4;
constexpr uint32_t PINGPONG_FLAG_M_MTE1_OFFSET_SIX_NO_MASK = 6;
constexpr uint32_t L0B_LOAD_TOTAL_SIZE_128_NO_MASK = 128;

constexpr size_t LEN_BURST_TOTAL_BYTES_32_NO_MASK = 32;

template <typename TILING_TYPE, typename S_DT, typename QKV_DT, typename O_DT, typename P_DT, typename M_DT, typename E_DT, MaskType M = MaskType::MASK_TYPE_NONE, ScaleType S = ScaleType::SCALE_TOR, typename...Args>
struct PFAHighPrecisionBaseType {
    using tilingType = TILING_TYPE;
    using qkvType = QKV_DT;
    using outputType = O_DT;
    using sType = S_DT;
    using pType = P_DT;
    using mType = M_DT;
    using expType = E_DT;
    static constexpr MaskType maskType = M;
    static constexpr ScaleType scaleType = S;
};

#ifdef __DAV_C220_CUBE__
constexpr int32_t KV_DB_SIZE = 65536;  // 128 * 128 * 2B

template<typename PFAT>
class FlashAttentionEncoderHighPrecisionCubeOpt {
public:
    __aicore__ __attribute__((always_inline)) inline FlashAttentionEncoderHighPrecisionCubeOpt(
        __gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
        __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
        __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
        const typename PFAT::tilingType* __restrict tiling,
        __gm__ uint8_t* gmTiling, TPipe* tPipe, __gm__ uint8_t* deqScale1, __gm__ uint8_t* scale1,
         __gm__ uint8_t* deqScale2, __gm__ uint8_t* scale2, __gm__ uint8_t* offset2)
    {
        AscendC::SetLoadDataPaddingValue<uint64_t>(0);
        AscendC::SetAtomicNone();
        AscendC::SetFixpipeNz2ndFlag(1, 0, 0);
        AscendC::SetMaskNorm();
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
        this->dataShapeType = tiling->promptAttentionBaseApiBaseParams.dataShapeType;
        this->groupNum = qHeads / kvHeads;
        this->strideQo = qHeads * embd;
        this->strideKv = kvHeads * embd;
        if (dataShapeType == 1) {
            this->strideQo = embd;
            this->strideKv = embd;
        }
        this->batchStrideKv = batchSize * maxKvSeqlen * kvHeads * embd * sizeof(QKV_DT);
        this->__k = embd;
        this->roundK = (__k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        qGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(query));
        kGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(key));
        vGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QKV_DT *>(value));
        uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
        sGmTensor.SetGlobalBuffer((__gm__ float *)(workspace));
        pGmTensor.SetGlobalBuffer((__gm__ QKV_DT *)(workspace + workSize));
        oTmpGmTensor.SetGlobalBuffer((__gm__ float *)(workspace + 2 * workSize));
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    }

    __aicore__ __attribute__((always_inline)) inline ~FlashAttentionEncoderHighPrecisionCubeOpt()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ __attribute__((always_inline)) inline void LoadDataToCa(
        AscendC::LocalTensor<typename PFAT::qkvType> dstTensor, AscendC::LocalTensor<typename PFAT::qkvType> srcTensor,
        uint32_t roundK, uint32_t qkRoundM, uint32_t qkM)
    {
        uint32_t roundRow = RoundUp<uint32_t>(roundK, 32 / sizeof(QKV_DT));
        if (qkM == 1) {
            AscendC::LoadData(dstTensor,
                            srcTensor,
                            AscendC::LoadData2dParams(0,           // baseIdx
                                                        NumMatrixsRoundUp<QKV_DT>(roundRow),   // repeat
                                                        1,  // srcStride
                                                        0,           // sid
                                                        0,  // dstStride
                                                        false, // transpose
                                                        0));         // addrCalMode
        } else {
            using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
            // 16 * 32
            static constexpr uint32_t ROW_BLOCK_SIZE = 16;
            static constexpr uint32_t COL_BLOCK_SIZE = 32 / sizeof(QKV_DT);
            static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(QKV_DT);
            for (uint32_t i = 0; i < qkRoundM / ROW_BLOCK_SIZE; i++) {
                AscendC::LoadData(dstTensor[i * ROW_BLOCK_SIZE * roundRow],
                                srcTensor[i * FRACTAL_SIZE],
                                AscendC::LoadData2dParams(0,
                                                            static_cast<uint16_t>(roundRow / COL_BLOCK_SIZE),
                                                            qkRoundM / ROW_BLOCK_SIZE,
                                                            0,
                                                            0,
                                                            false,
                                                            0));
            }
        }
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        uint64_t curBatch = 0;
        uint32_t qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
        uint32_t kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
        uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
        uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
        uint64_t addrQScalar = 0;
        uint64_t addrKScalar = 0;
        uint64_t addrVScalar = 0;
        uint32_t preTotalQBlkNum = 0;
        uint32_t curTotalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNumFirst;
        uint32_t processNum = totalQBlkNum * qHeads;
        uint32_t nextProcess = 0;
        uint32_t isMLA = 0;
        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
        static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(QKV_DT);
        for (uint32_t process = GetBlockIdx(); process < processNum; process = nextProcess) {
            while (process >= curTotalQBlkNum * qHeads) {
                curBatch++;
                preTotalQBlkNum = curTotalQBlkNum;
                addrQScalar += static_cast<uint64_t>(qSeqlen * qHeads * embd);
                addrKScalar += static_cast<uint64_t>(kvSeqlen * kvHeads * embd);
                addrVScalar += static_cast<uint64_t>(kvSeqlen * kvHeads * embdv);
                qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
                kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
                MNibd mnIbd = GetPPmnIbd(qSeqlen, kvSeqlen, embd, isMLA);
                ppMScalar = PP_MM[mnIbd.mIbd];
                ppNScalar = PP_NN[mnIbd.nIbd];
                curTotalQBlkNum += (qSeqlen != 0) ? ((qSeqlen + ppMScalar - 1) / ppMScalar) : 0;
            }
            nextProcess = process + GetBlockNum();
            if (isTriuMask) {
                uint32_t currIter = process / GetBlockNum();
                nextProcess = currIter % TWO_NO_MASK == ONE_NO_MASK ? (currIter + ONE_NO_MASK) * GetBlockNum() + GetBlockIdx() : (currIter + TWO_NO_MASK) * GetBlockNum() - ONE_NO_MASK - GetBlockIdx();
            }
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
            if (dataShapeType == 1) {
                qOffset = addrQScalar + headIdx * embd * maxSeqlen + mIdx * ppMScalar * strideQo;
                kOffset = addrKScalar + (headIdx / groupNum) * embd * maxKvSeqlen;
            }

            // Only need load Q once
            if (qkM == 1) {
                AscendC::DataCopy(l1qBufAddrTensor,
                                qGmTensor[qOffset],
                                AscendC::DataCopyParams(1,                                              // nBurst
                                                        CeilDiv<BLOCK_SIZE_COPY>(1 * RoundUp<uint32_t>(roundK, LEN_BURST_TOTAL_BYTES_32_NO_MASK / sizeof(QKV_DT))), // lenBurst
                                                        0,                                              // srcGap
                                                        0));
            } else {
                if (strideQo < STRIDE_LIMIT) {
                    AscendC::DataCopy(l1qBufAddrTensor,
                                    qGmTensor[qOffset],
                                    AscendC::Nd2NzParams(1,           // ndNum
                                                        qkM, // nValue
                                                        __k, // dValue
                                                        0,           // srcNdMatrixStride, unused
                                                        strideQo,        // srcDValue
                                                        qkRoundM,   // dstNzC0Stride
                                                        1,           // dstNzNStride
                                                        0));         // dstNzMatrixStride, unused
                } else {
                    for (uint32_t i = 0; i < qkM; ++i) {
                        AscendC::DataCopy(l1qBufAddrTensor,
                                        qGmTensor[qOffset],
                                        AscendC::Nd2NzParams(1,           // ndNum
                                                            1,           // nValue
                                                            __k, // dValue
                                                            0,           // srcNdMatrixStride, unused
                                                            0,           // srcDValue
                                                            qkRoundM,   // dstNzC0Stride
                                                            0,           // dstNzNStride
                                                            0));         // dstNzMatrixStride, unused
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlag);
            uint32_t svN = nLoop == 1 ? kvSeqlen : ppNScalar;
            uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            uint64_t vOffset = addrVScalar + (headIdx / groupNum) * embd;
            if (dataShapeType == 1) {
                vOffset = addrVScalar + (headIdx / groupNum) * embd * maxKvSeqlen;
            }
            uint32_t nEnd = nLoop;
            if (isTriuMask) {
                nEnd = mIdx + 1;
            }
            uint32_t sBlockStack = nEnd > 8 ? 4 : (nEnd > 4 ? 2 : 1);
            uint32_t launchDelay = sBlockStack * 2;
            uint32_t vectMod = 2 * launchDelay;
            uint32_t kvPingpongFlag = 0;
            uint64_t kvPingpongOffset = kvPingpongFlag * KV_DB_SIZE;
            for (uint32_t nIdx = 0; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
                if (nIdx < nEnd) {
                    uint32_t svNTriu = nEnd * ppNScalar;
                    if (nIdx + sBlockStack > nEnd - 1) {
                        svN = svNTriu > kvSeqlen ? kvSeqlen - nIdx * ppNScalar : svNTriu - nIdx * ppNScalar;
                    } else {
                        svN = ppNScalar * sBlockStack;
                    }
                    uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    for (uint32_t splitIdx = 0; splitIdx < sBlockStack && nIdx + splitIdx < nEnd; splitIdx++) {
                        pingpongFlag = (nIdx + splitIdx) % TWO_NO_MASK;
                        offset = pingpongFlag * L0AB_HALF_BUF_SIZE;
                        if (nIdx + splitIdx == (nLoop - 1)) {
                            qkN = (kvSeqlen - (nIdx + splitIdx) * ppNScalar);
                            qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                        }
                        bool lastSplit = splitIdx == sBlockStack - 1 || nIdx + splitIdx == nEnd - 1;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + TWO_NO_MASK * kvPingpongFlag);
                        if (strideKv < STRIDE_LIMIT) {
                            AscendC::DataCopy(l1kBufAddrTensor[kvPingpongOffset + offset],
                                            kGmTensor[kOffset],
                                            AscendC::Nd2NzParams(1,           // ndNum
                                                                qkN, // nValue
                                                                __k, // dValue
                                                                0,           // srcNdMatrixStride, unused
                                                                strideKv,        // srcDValue
                                                                qkRoundN,   // dstNzC0Stride
                                                                1,           // dstNzNStride
                                                                0));         // dstNzMatrixStride, unused
                        } else {
                            for (uint32_t i = 0; i < qkN; i++) {
                                AscendC::DataCopy(l1kBufAddrTensor[i * BLOCK_SIZE_COPY],
                                                kGmTensor[kOffset][i * strideKv],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    1,           // nValue
                                                                    __k, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    0,           // srcDValue
                                                                    qkRoundN,   // dstNzC0Stride
                                                                    0,           // dstNzNStride
                                                                    0));         // dstNzMatrixStride, unused
                            }
                        }
                        kOffset += ppNScalar * strideKv;
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
                        LoadDataToCa(l0aBufTensor[offset], l1qBufAddrTensor, roundK, qkRoundM, qkM);
                        // *** Prepare K to L1
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_NO_MASK);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(pingpongFlag);
                        AscendC::LoadData(
                            l0bBufTensor[offset],
                            l1kBufAddrTensor[kvPingpongOffset + offset],
                            AscendC::LoadData2dParams(0,
                                                        NumMatrixsRoundUp<QKV_DT>(RoundUp<uint32_t>(roundK, LEN_BURST_TOTAL_BYTES_32_NO_MASK / sizeof(QKV_DT)) * qkRoundN),
                                                        1,
                                                        0,
                                                        0,
                                                        false,
                                                        0)
                        );
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(pingpongFlag + TWO_NO_MASK * kvPingpongFlag);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_NO_MASK);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_NO_MASK);
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(pingpongFlag);
                        if constexpr (int8Flag) {
                            AscendC::Mmad(l0cBufTensor.ReinterpretCast<int32_t>()[pingpongFlag * L0AB_HALF_BUF_SIZE],
                                        l0aBufTensor[offset],
                                        l0bBufTensor[offset],
                                        AscendC::MmadParams(qkM, qkN, __k, 0, false, 1));
                        } else {
                            AscendC::Mmad(l0cBufTensor[pingpongFlag * L0AB_HALF_BUF_SIZE],
                                        l0aBufTensor[offset],
                                        l0bBufTensor[offset],
                                        AscendC::MmadParams(qkM, qkN, __k, 0, false, 1));
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(pingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_NO_MASK);
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(pingpongFlag);
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(pingpongFlag);
                        // copy S to gm
                        auto intriParams = AscendC::FixpipeParamsV220(qkRoundN, // nSize
                                                                        qkM, // mSize
                                                                        qkRoundM,   // srcStride
                                                                        svRoundN,   // dstStride
                                                                        false);      // enRelu
                        intriParams.quantPre = QuantMode_t::NoQuant;
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod + splitIdx * ppNScalar], l0cBufTensor[pingpongFlag * L0AB_HALF_BUF_SIZE], intriParams);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(pingpongFlag);
                    }
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(QK_READY); // 2 is the Sync mode
                    kvPingpongFlag = 1 - kvPingpongFlag;
                    kvPingpongOffset = kvPingpongFlag * KV_DB_SIZE;
                }
                if (nIdx >= launchDelay) {
                    uint32_t l0cPingpongFlag = (nIdx + 1) % TWO_NO_MASK;
                    uint32_t l0cOffset = l0cPingpongFlag * L0AB_HALF_BUF_SIZE;
                    uint32_t svNTriu = nEnd * ppNScalar;
                    if (nIdx + sBlockStack > nEnd + launchDelay - 1) {
                        svN = svNTriu > kvSeqlen ? kvSeqlen - (nIdx - launchDelay) * ppNScalar : svNTriu - (nIdx - launchDelay) * ppNScalar;
                    } else {
                        svN = ppNScalar * sBlockStack;
                    }
                    uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    uint32_t nSlice = ppNScalar * ((sBlockStack + 1) / 2);
                    uint32_t l1SplitLoop = (svN + nSlice - 1) / nSlice;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kvPingpongFlag * TWO_NO_MASK);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(kvPingpongFlag * TWO_NO_MASK + PINGPONG_FLAG_M_MTE1_OFFSET_ONE_NO_MASK);
                    if (strideKv < STRIDE_LIMIT) {
                        AscendC::DataCopy(l1vBufAddrTensor[kvPingpongOffset],
                                        vGmTensor[vOffset],
                                        AscendC::Nd2NzParams(1,           // ndNum
                                                                svN, // nValue
                                                                __k, // dValue
                                                                0,           // srcNdMatrixStride, unused
                                                                strideKv,        // srcDValue
                                                                svRoundN,   // dstNzC0Stride
                                                                1,           // dstNzNStride
                                                                0));         // dstNzMatrixStride, unused
                    } else {
                        for (uint32_t i = 0; i < svN; i++) {
                            AscendC::DataCopy(l1vBufAddrTensor[i * BLOCK_SIZE_COPY],
                                            vGmTensor[vOffset][i * strideKv],
                                            AscendC::Nd2NzParams(1,           // ndNum
                                                                1,           // nValue
                                                                __k, // dValue
                                                                0,           // srcNdMatrixStride, unused
                                                                0,           // srcDValue
                                                                svRoundN,   // dstNzC0Stride
                                                                0,           // dstNzNStride
                                                                0));         // dstNzMatrixStride, unused
                        }
                    }
                    vOffset += svN * strideKv;
                    AscendC::WaitEvent(SOFTMAX_READY);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingpongFlag);
                    for (uint32_t l1KSplitIdx = 0; l1KSplitIdx < l1SplitLoop; l1KSplitIdx++) {
                        uint32_t l1PingpongFlag = l1KSplitIdx % 2;
                        uint32_t l1Offset = l1PingpongFlag * 128 * 256;
                        bool l1LastSplit = l1KSplitIdx == l1SplitLoop - 1;
                        uint32_t d = l1LastSplit ? svN - l1KSplitIdx * nSlice : nSlice;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_NO_MASK);
                        if (qkM == 1) {
                        AscendC::DataCopy(l1pBufAddrTensor[l1Offset],
                                        pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE +
                                                    (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod) *
                                                    2 / sizeof(QKV_DT) + l1KSplitIdx * nSlice],  // 2 is the Byte Convert Factor
                                        AscendC::DataCopyParams(1,                                              // nBurst
                                                                CeilDiv<BLOCK_SIZE_COPY>(1 * RoundUp<uint64_t>(svRoundN, BlockSize<QKV_DT>())), // lenBurst
                                                                0,                                              // srcGap
                                                                0));
                        } else {
                            if (svRoundN * TWO_NO_MASK / sizeof(QKV_DT) < STRIDE_LIMIT) {
                                AscendC::DataCopy(l1pBufAddrTensor[l1Offset],
                                                pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE +
                                                            (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod) *
                                                            2 / sizeof(QKV_DT) + l1KSplitIdx * nSlice],  // 2 is the Byte Convert Factor
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                        qkM, // nValue
                                                                        __k, // dValue
                                                                        0,           // srcNdMatrixStride, unused
                                                                        svRoundN * TWO_NO_MASK / sizeof(QKV_DT),        // srcDValue
                                                                        qkRoundM,   // dstNzC0Stride
                                                                        1,           // dstNzNStride
                                                                        0));         // dstNzMatrixStride, unused
                            } else {
                                for (uint32_t i = 0; i < qkM; ++i) {
                                    AscendC::DataCopy(l1pBufAddrTensor[l1Offset],
                                                    pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE +
                                                                (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod) *
                                                                2 / sizeof(QKV_DT) + l1KSplitIdx * nSlice],  // 2 is the Byte Convert Factor
                                                    AscendC::Nd2NzParams(1,           // ndNum
                                                                            1,           // nValue
                                                                            __k, // dValue
                                                                            0,           // srcNdMatrixStride, unused
                                                                            0,           // srcDValue
                                                                            qkRoundM,   // dstNzC0Stride
                                                                            0,           // dstNzNStride
                                                                            0));         // dstNzMatrixStride, unused
                                }
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_NO_MASK);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_NO_MASK);
                        uint32_t dSplitLoop = (d + 127) / 128;
                        for (uint32_t l0KSplitIdx = 0; l0KSplitIdx < dSplitLoop; l0KSplitIdx++) {
                            uint32_t l0PingpongFlag = l0KSplitIdx % 2;
                            uint32_t l0Offset = l0PingpongFlag * 128 * 128;
                            bool l0LastSplit = l0KSplitIdx == dSplitLoop - 1;
                            int32_t l0POffset = qkM == 1 ? l0KSplitIdx * 128 : l0KSplitIdx * 128 * qkRoundM;
                            bool initC = l0KSplitIdx== 0 && l1KSplitIdx == 0;
                            uint32_t reduceD = l0LastSplit ? d - l0KSplitIdx * 128 : 128;
                            uint32_t roundReduceD = (reduceD + 15) / 16 * 16;
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingpongFlag);
                            LoadDataToCa(l0aBufTensor[l0Offset], l1pBufAddrTensor[l1Offset + l0POffset],
                                        RoundUp<uint64_t>(roundReduceD, BlockSize<QKV_DT>()), qkRoundM, qkM);
                            if (l0LastSplit){
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_FOUR_NO_MASK);
                            }
                            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_NO_MASK);
                            for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < L0B_LOAD_TOTAL_SIZE_128_NO_MASK / BLOCK_SIZE; ++l0bLoadIdx) {
                                AscendC::LoadData(
                                    l0bBufTensor[l0Offset + l0bLoadIdx * roundK * BLOCK_SIZE],
                                    l1vBufAddrTensor[kvPingpongOffset + l0bLoadIdx * CUBE_MATRIX_SIZE +
                                                        l1KSplitIdx * nSlice * BLOCK_SIZE +
                                                        l0KSplitIdx * 128 * BLOCK_SIZE],  // 128 is the Block num
                                    AscendC::LoadData2dParams(0,
                                                                roundK / BLOCK_SIZE,
                                                                svRoundN / BLOCK_SIZE,
                                                                0,
                                                                0,
                                                                true,
                                                                0)
                                );
                            }
                            if (l0LastSplit && l1LastSplit){
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kvPingpongFlag * TWO_NO_MASK);
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kvPingpongFlag * TWO_NO_MASK + PINGPONG_FLAG_M_MTE1_OFFSET_ONE_NO_MASK);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_SIX_NO_MASK);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_SIX_NO_MASK );
                            if constexpr (int8Flag) {
                                AscendC::Mmad(l0cBufTensor.template ReinterpretCast<int32_t>()[l0cOffset],
                                            l0aBufTensor[l0Offset],
                                            l0bBufTensor[l0Offset],
                                            AscendC::MmadParams(qkM, __k, svN, 0, false, 1));
                            } else {
                                AscendC::Mmad(l0cBufTensor[l0cOffset],
                                            l0aBufTensor[l0Offset],
                                            l0bBufTensor[l0Offset],
                                            AscendC::MmadParams(qkM, __k, reduceD, 0, false, initC));
                            }
                            AscendC::PipeBarrier<PIPE_M>();
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingpongFlag);
                            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0PingpongFlag + PINGPONG_FLAG_M_MTE1_OFFSET_TWO_NO_MASK);
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingpongFlag);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingpongFlag);
                    // copy O to gm
                    auto intriParams = AscendC::FixpipeParamsV220(roundK, // nSize
                                                                    qkM, // mSize
                                                                    qkRoundM,   // srcStride
                                                                    roundK,   // dstStride
                                                                    false);      // enRelu
                    intriParams.quantPre = QuantMode_t::NoQuant;
                    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod], l0cBufTensor[l0cOffset], intriParams);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingpongFlag);
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(UPDATE_READY); // 2 is the Sync mode
                    kvPingpongFlag = 1 - kvPingpongFlag;
                    kvPingpongOffset = kvPingpongFlag * KV_DB_SIZE;
                }
            }
        }
    }

private:
    using QKV_DT = typename PFAT::qkvType;
    using O_DT = typename PFAT::outputType;
    using IN_DATA_TYPE = typename PFAT::mType;
    static constexpr bool int8Flag = IsSameType<QKV_DT, int8_t>::value;
    const typename PFAT::tilingType* __restrict tiling;
    const uint32_t l1qBufAddrOffset = 0;
    const uint32_t l1kBufAddrOffset = 4 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1vBufAddrOffset = 4 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1pBufAddrOffset = 12 * L0AB_UINT8_BLOCK_SIZE;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<QKV_DT> l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1qBufAddrOffset);
    AscendC::LocalTensor<QKV_DT> l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1kBufAddrOffset);
    AscendC::LocalTensor<QKV_DT> l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1pBufAddrOffset);
    AscendC::LocalTensor<QKV_DT> l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, QKV_DT>(l1vBufAddrOffset);
    AscendC::GlobalTensor<QKV_DT> qGmTensor;
    AscendC::GlobalTensor<QKV_DT> kGmTensor;
    AscendC::GlobalTensor<QKV_DT> vGmTensor;
    AscendC::GlobalTensor<float> sGmTensor;
    AscendC::GlobalTensor<QKV_DT> pGmTensor;
    AscendC::GlobalTensor<float> oTmpGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    AscendC::LocalTensor<QKV_DT> l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, QKV_DT>(0);
    AscendC::LocalTensor<QKV_DT> l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, QKV_DT>(0);
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
    uint64_t strideKv{0};
    uint64_t batchStrideKv{0};
    uint32_t __k{0};
    uint32_t roundK{0};
    uint32_t dataShapeType{0};
};
#elif __DAV_C220_VEC__
template<typename PFAT>
class FlashAttentionEncoderHighPrecisionVecOpt {
public:
    __aicore__ __attribute__((always_inline)) inline FlashAttentionEncoderHighPrecisionVecOpt(
        __gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
        __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
        __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
        const typename PFAT::tilingType* __restrict tiling,
        __gm__ uint8_t* gmTiling, TPipe* tPipe, __gm__ uint8_t* deqScale1, __gm__ uint8_t* scale1,
        __gm__ uint8_t* deqScale2, __gm__ uint8_t* scale2, __gm__ uint8_t* offset2,
         __gm__ uint8_t* alibiCoeff) : deqQkGm(deqScale1), offQkGm(scale1), quantPGm(deqScale2),
          deqPvGm(scale2), offPvGm(offset2), alibiCoeffGm(alibiCoeff)
    {
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
        maskGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ MASK_DTYPE *>(pseShift));
        oGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ O_DTYPE *>(attentionOut));
        sGmTensor.SetGlobalBuffer((__gm__ float *)(workspace));
        pGmTensor.SetGlobalBuffer((__gm__ P_DTYPE *)(workspace + workSize));
        oTmpGmTensor.SetGlobalBuffer((__gm__ float *)(workspace + 2 * workSize));
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);
        logNGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ S_DTYPE *>(softmaxLse));
        logNFloatGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmaxLse));
        this->tiling = tiling;
        this->subBlockIdx = AscendC::GetSubBlockIdx();
        this->batchSize = tiling->promptAttentionBaseApiBaseParams.batchSize;
        this->maxSeqlen = tiling->promptAttentionBaseApiBaseParams.maxSeqLen;
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
        this->quantType = tiling->promptAttentionBaseApiBaseParams.quantType;
        this->dataShapeType = tiling->promptAttentionBaseApiBaseParams.dataShapeType;
        this->strideQo = qHeads * embd;
        if (this->dataShapeType == 1) {
            this->strideQo = embd;
        }
        this->__k = embd;
        this->roundK = (__k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ __attribute__((always_inline)) inline ~FlashAttentionEncoderHighPrecisionVecOpt()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ __attribute__((always_inline)) inline void Run()
    {
        uint64_t curBatch = 0;
        uint32_t preTotalQBlkNum = 0;
        uint32_t qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
        uint32_t kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
        uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
        uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
        uint64_t addrOScalar = 0;
        uint32_t curTotalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNumFirst;
        uint32_t processNum = totalQBlkNum * qHeads;
        uint32_t nextProcess = 0;
        uint32_t isMLA = 0;
        for (uint32_t process = GetBlockIdx(); process < processNum; process = nextProcess) {
            while (process >= curTotalQBlkNum * qHeads) {
                curBatch++;
                preTotalQBlkNum = curTotalQBlkNum;
                addrOScalar += static_cast<uint64_t>(qSeqlen * qHeads * embdv);
                qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
                kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
                MNibd mnIbd = GetPPmnIbd(qSeqlen, kvSeqlen, embd, isMLA);
                ppMScalar = PP_MM[mnIbd.mIbd];
                ppNScalar = PP_NN[mnIbd.nIbd];
                curTotalQBlkNum += (qSeqlen != 0) ? ((qSeqlen + ppMScalar - 1) / ppMScalar) : 0;
            }
            nextProcess = process + GetBlockNum();
            if (qSeqlen == 0 || kvSeqlen == 0) {
                continue;
            }
            uint32_t processIdx = process - preTotalQBlkNum * qHeads;
            uint32_t mIdx = processIdx / qHeads;
            uint64_t headIdx = processIdx % qHeads;
            uint32_t mLoop = (qSeqlen + ppMScalar - 1) / ppMScalar;
            uint32_t nLoop = (kvSeqlen + ppNScalar - 1) / ppNScalar;
            uint32_t qkM = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * ppMScalar) : ppMScalar;
            uint32_t subM = (subBlockIdx == 1) ? (qkM - qkM / 2) : qkM / 2;
            uint32_t subMD128 = (subM + VECTOR_SIZE - 1) / VECTOR_SIZE;            // up aligned to 128
            uint32_t subMD64 = (subM + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE; // up aligned to 64
            uint32_t roundSubM = (subM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            /******** pre_load *******/
            uint32_t qkN = nLoop == 1 ? kvSeqlen : ppNScalar;
            uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            uint64_t oOffset = addrOScalar + headIdx * embd + mIdx * ppMScalar * strideQo;
            if (dataShapeType == 1) {
                oOffset = addrOScalar + headIdx * embd * maxSeqlen + mIdx * ppMScalar * strideQo;
            }
            uint32_t nEnd = nLoop;
            uint32_t qkNTriu = nEnd * ppNScalar;
            uint32_t sBlockStack = nEnd > 8 ? 4 : (nEnd > 4 ? 2 : 1);
            uint32_t launchDelay = sBlockStack * 2;
            uint32_t vectMod = 2 * launchDelay;
            uint32_t mSlice = FLOAT_VECTOR_SIZE / sBlockStack;
            uint32_t mEnd = (subM + mSlice - 1) / mSlice;
            for (uint32_t nIdx = 0; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
                if (nIdx < nEnd) {
                    if (nIdx + sBlockStack > nEnd - 1) {
                        qkN = qkNTriu > kvSeqlen ? kvSeqlen - nIdx * ppNScalar : qkNTriu - nIdx * ppNScalar;
                    } else {
                        qkN = ppNScalar * sBlockStack;
                    }
                    qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    if (subM == 0) {
                        AscendC::WaitEvent(QK_READY);
                    }
                    uint32_t pingpongFlag = 0;
                    for (uint32_t mInd = 0; mInd < mEnd; mInd++) {
                        uint32_t rowOffset = mInd * mSlice;
                        uint32_t currM = mInd == mEnd - 1 ? subM - rowOffset : mSlice;
                        uint32_t sUbOffset = pingpongFlag * S_DB_SIZE;
                        uint64_t spGmOffset = (uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                    (uint64_t)subBlockIdx * qkM / 2 * qkRoundN + rowOffset * qkRoundN;
                        if (mInd == 0) {
                            AscendC::WaitEvent(QK_READY);
                        }
                        if (currM == 0) {
                            continue;
                        }
                        OnlineSoftmaxStage1<S_DTYPE, EXP_DTYPE, P_DTYPE, MASK_DTYPE, MASK_TYPE> (
                            lsUbufTensor[sUbOffset],
                            mask16UbufTensor,
                            maskUbufTensor,
                            lmUbufTensor[rowOffset],
                            hmUbufTensor[rowOffset],
                            gmUbufTensor[rowOffset],
                            dmUbufTensor[((nIdx / sBlockStack) % 6) * UB_FLOAT_LINE_SIZE + rowOffset], // 6 is the Block Stack Num
                            lsUbufTensor[sUbOffset],
                            llUbufTensor[rowOffset],
                            glUbufTensor[rowOffset],
                            lpUbufTensor[sUbOffset * 2],  // 2 is the offset factor
                            tvUbufTensor,
                            sGmTensor[spGmOffset],
                            pGmTensor[spGmOffset],
                            nIdx == 0, this->tor,
                            currM, qkN, qkRoundN, pingpongFlag
                        );
                        pingpongFlag = 1 - pingpongFlag;
                    }
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SOFTMAX_READY); // 2 is the Sync Mode
                }
                if (nIdx >= launchDelay) {
                    AscendC::WaitEvent(UPDATE_READY); // 4
                    if (subM == 0) {
                        continue;
                    }
                    //  *** 更新 L 和 O
                    if (nIdx != launchDelay) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                        AscendC::DataCopy(loUbufTensor,
                                            oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod +
                                                            (uint64_t)subBlockIdx * qkM / 2 * roundK], // 2 is the CV_RATIO
                                            AscendC::DataCopyParams(
                                                1,
                                                subM * roundK / FLOAT_BLOCK_SIZE,
                                                0,
                                                0)
                        );
			            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                        // *** dm_block = expand_to_block(dm), 存放于 tv
                        AscendC::Brcb(
                            tvUbufTensor.ReinterpretCast<uint32_t>(),
                            dmUbufTensor[((nIdx - launchDelay) / sBlockStack % 6) * UB_FLOAT_LINE_SIZE].ReinterpretCast<uint32_t>(),  // 6 is the Block Stack Num
                            roundSubM / FLOAT_BLOCK_SIZE,
                            AscendC::BrcbRepeatParams(1, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** go = go * dm_block
                        for (uint32_t vmulIdx = 0; vmulIdx < __k / FLOAT_VECTOR_SIZE; ++vmulIdx) {
                            AscendC::Mul<float, false>(
                                goUbufTensor[vmulIdx * FLOAT_VECTOR_SIZE],
                                goUbufTensor[vmulIdx * FLOAT_VECTOR_SIZE],
                                tvUbufTensor,
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                            );
                        }
                        if (__k % FLOAT_VECTOR_SIZE > 0) {
                            SetVecMask(__k % FLOAT_VECTOR_SIZE);
                            AscendC::Mul<float, false>(
                                goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                tvUbufTensor,
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** go = lo + go
			            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                        AscendC::Add<float, false>(
                            goUbufTensor,
                            goUbufTensor,
                            loUbufTensor,
                            (uint64_t)0,
                            (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
			            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    } else {
			            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
                        AscendC::DataCopy(goUbufTensor,
                                            oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod +
                                                            (uint64_t)subBlockIdx * qkM / 2 * roundK], // 2 is the CV_RATIO
                                            AscendC::DataCopyParams(
                                                1,
                                                subM * roundK / FLOAT_BLOCK_SIZE,
                                                0,
                                                0)
                        );
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                    }
                    if (nIdx + sBlockStack > nEnd + launchDelay - 1)  {
                        // *** gl_block = expand_to_block(gl), 存放于 tv
                        AscendC::Brcb(
                            tvUbufTensor.ReinterpretCast<uint32_t>(),
                            glUbufTensor.ReinterpretCast<uint32_t>(),
                            roundSubM / FLOAT_BLOCK_SIZE,
                            AscendC::BrcbRepeatParams(1, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** go = go / gl_block
                        for (uint32_t vdivIdx = 0; vdivIdx < __k / FLOAT_VECTOR_SIZE; ++vdivIdx) {
                            AscendC::Div<float, false>(
                                goUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                goUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                tvUbufTensor,
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                            );
                        }
                        if (__k % FLOAT_VECTOR_SIZE > 0) {
                            SetVecMask(__k % FLOAT_VECTOR_SIZE);
                            AscendC::Div<float, false>(
                                goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                tvUbufTensor,
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        if constexpr (std::is_same<float, float>::value && std::is_same<O_DTYPE, __bf16>::value) {
                            AscendC::Cast<O_DTYPE, float, false>(
                                goUbufTensor.ReinterpretCast<O_DTYPE>(),
                                goUbufTensor,
                                AscendC::RoundMode::CAST_RINT,
                                (uint64_t)0,
                                (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)); // 4 is the RepStrid
                        } else {
                            AscendC::Cast<O_DTYPE, float, false>(
                                goUbufTensor.ReinterpretCast<O_DTYPE>(),
                                goUbufTensor,
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)); // 4 is the RepStrid
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        // ********************* move O to GM ************************
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::DataCopyPad(
                            oGmTensor[oOffset + (uint64_t)subBlockIdx * qkM / TWO_NO_MASK * strideQo],
                            goUbufTensor.ReinterpretCast<O_DTYPE>(),
                            AscendC::DataCopyExtParams(
                                subM,
                                __k * TWO_NO_MASK,
                                0,
                                (strideQo - __k) * TWO_NO_MASK,
                                0)
                        );
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
                    }
                }
            }
        }
    }

private:
    using O_DTYPE = typename PFAT::outputType;
    using S_DTYPE = typename PFAT::sType;
    using P_DTYPE = typename PFAT::pType;
    using EXP_DTYPE = typename PFAT::expType;
    using MASK_DTYPE = typename PFAT::mType;
    static constexpr MaskType MASK_TYPE = PFAT::maskType;
    static constexpr ScaleType SCALE_TYPE = PFAT::scaleType;
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
    const uint32_t glUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_LINE_SIZE;
    const uint32_t tvUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_LINE_SIZE;
    const uint32_t pScaleUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 21 * UB_UINT8_LINE_SIZE;
    const uint32_t logUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 30 * UB_UINT8_LINE_SIZE;
    const uint32_t logUbufFloatOffset = 8 * UB_UINT8_BLOCK_SIZE + 30 * UB_UINT8_LINE_SIZE;
    const uint32_t goUbufOffset = 9 * UB_UINT8_BLOCK_SIZE;
    const uint32_t mask16UbufOffset = 11 * UB_UINT8_BLOCK_SIZE;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<S_DTYPE> lsUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DTYPE>(lsUbufOffset);
    AscendC::LocalTensor<P_DTYPE> lpUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, P_DTYPE>(lpUbufOffset);
    AscendC::LocalTensor<EXP_DTYPE> ls32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, EXP_DTYPE>(ls32UbufOffset);
    AscendC::LocalTensor<S_DTYPE> maskUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DTYPE>(maskUbufOffset);
    AscendC::LocalTensor<float> loUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbufOffset);
    AscendC::LocalTensor<float> lmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lmUbufOffset);
    AscendC::LocalTensor<float> hmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(hmUbufOffset);
    AscendC::LocalTensor<float> gmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(gmUbufOffset);
    AscendC::LocalTensor<float> dmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(dmUbufOffset);
    AscendC::LocalTensor<float> llUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(llUbufOffset);
    AscendC::LocalTensor<float> glUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(glUbufOffset);
    AscendC::LocalTensor<float> tvUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(tvUbufOffset);
    AscendC::LocalTensor<float> pScaleUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(pScaleUbufOffset);
    AscendC::LocalTensor<float> goUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbufOffset);
    AscendC::LocalTensor<MASK_DTYPE> mask16UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, MASK_DTYPE>(mask16UbufOffset);
    AscendC::LocalTensor<half> logUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(logUbufOffset);
    AscendC::LocalTensor<float> logUbufFloatTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(logUbufFloatOffset);
    AscendC::GlobalTensor<MASK_DTYPE> maskGmTensor;
    AscendC::GlobalTensor<O_DTYPE> oGmTensor;
    AscendC::GlobalTensor<S_DTYPE> sGmTensor;
    AscendC::GlobalTensor<P_DTYPE> pGmTensor;
    AscendC::GlobalTensor<float> oTmpGmTensor;
    AscendC::GlobalTensor<S_DTYPE> logNGmTensor;
    AscendC::GlobalTensor<float> logNFloatGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    __gm__ uint8_t *__restrict__ alibiCoeffGm{nullptr};
    __gm__ uint8_t *__restrict__ deqQkGm{nullptr};
    __gm__ uint8_t *__restrict__ offQkGm{nullptr};
    __gm__ uint8_t *__restrict__ quantPGm{nullptr};
    __gm__ uint8_t *__restrict__ deqPvGm{nullptr};
    __gm__ uint8_t *__restrict__ offPvGm{nullptr};
    uint32_t batchSize{0};
    uint32_t maxSeqlen{0};
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
    uint32_t __k{0};
    uint32_t roundK{0};
    int32_t subBlockIdx{-1};
    uint32_t tilingKey{0};
    uint32_t tilingHeadSize{0};
    uint32_t tilingParaSize{0};
    uint32_t longSeq{0};
    uint32_t isSqrt{0};
    uint32_t maskType{0};
    uint32_t alibiCompressOffset{0};
    uint32_t alibiLeftAlign{0};
    uint32_t dataShapeType{0};
    uint32_t quantType{0};
};
#endif

template <typename PFAT>
class PromptFlashAttentionBaseApiHighPrecisionNoMask {
public:
    __aicore__ inline PromptFlashAttentionBaseApiHighPrecisionNoMask() {};
    __aicore__ inline void Init(__gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
                                const typename PFAT::tilingType* __restrict tiling,
                                __gm__ uint8_t* gmTiling, TPipe* tPipe, __gm__ uint8_t* deqScale1, __gm__ uint8_t* scale1,
                                __gm__ uint8_t* deqScale2, __gm__ uint8_t* scale2, __gm__ uint8_t* offset2, __gm__ uint8_t* alibiCoeff) {
        #ifdef __DAV_C220_CUBE__
            FlashAttentionEncoderHighPrecisionCubeOpt<PFAT> faCube(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                                    queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                                    attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe, deqScale1, scale1,
                                    deqScale2, scale2, offset2);
            faCube.Run();
        #elif __DAV_C220_VEC__
            FlashAttentionEncoderHighPrecisionVecOpt<PFAT> faVec(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                                    queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                                    attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe, deqScale1, scale1,
                                    deqScale2, scale2, offset2, alibiCoeff);
            faVec.Run();
        #endif
    }
};

#endif  // PROMPT_FLASH_ATTENTION_BASE_API_HIGH_PRECISION_NO_MASK_H
