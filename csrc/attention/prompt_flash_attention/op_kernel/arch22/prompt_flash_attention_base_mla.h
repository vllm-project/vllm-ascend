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
 * \file prompt_flash_attention_base_mla.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_BASE_MLA_H
#define PROMPT_FLASH_ATTENTION_BASE_MLA_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

constexpr int32_t TMP_SIZET = 16384;               // 128 * 256 * 2
constexpr int32_t BLOCK_QK = 128;
constexpr int32_t LOCAL_SIZE = 6;
constexpr int32_t MAX_ALLOWED_LENGTH_BASE_MLA = 16;  
constexpr int32_t MIN_ALLOWED_LENGTH_BASE_MLA = 1; 

template <typename TILING_TYPE, typename...Args>
struct PFAMLAType {
    using tilingType = TILING_TYPE;
};

#ifdef __DAV_C220_CUBE__
template<typename PFAT>
__aicore__ inline void unpad_flashattention_mla(
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

    const uint32_t l1qBufAddrOffset = 0;
    const uint32_t l1kBufAddrOffset = 2 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1pBufAddrOffset = 4 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1vBufAddrOffset = 6 * L0AB_UINT8_BLOCK_SIZE;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::LocalTensor<half> l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1qBufAddrOffset);
    AscendC::LocalTensor<half> l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1kBufAddrOffset);
    AscendC::LocalTensor<half> l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1pBufAddrOffset);
    AscendC::LocalTensor<half> l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1vBufAddrOffset);
    AscendC::LocalTensor<half> l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, half>(0);
    AscendC::LocalTensor<half> l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, half>(0);
    AscendC::LocalTensor<float> l0cBufTensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    uint32_t batchSize = tiling->promptAttentionBaseApiBaseParams.batchSize;
    uint32_t maxSeqlen = tiling->promptAttentionBaseApiBaseParams.maxSeqLen;
    uint32_t qHeads = tiling->promptAttentionBaseApiBaseParams.headNumSize;
    uint32_t embd = tiling->promptAttentionBaseApiBaseParams.headSize;
    uint32_t embdv = tiling->promptAttentionBaseApiBaseParams.embeddingSizeV;
    uint32_t kvHeads = tiling->promptAttentionBaseApiBaseParams.kvHeadNumSize;
    uint32_t isTriuMask = tiling->promptAttentionBaseApiBaseParams.isTriuMask;
    uint32_t totalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNum;
    uint32_t maxKvSeqlen = tiling->promptAttentionBaseApiBaseParams.maxKvSeqLen;
    uint32_t windowSize = 0;
    uint32_t dataShapeType = tiling->promptAttentionBaseApiBaseParams.quantType;
    uint32_t inputLayoutType = tiling->promptAttentionBaseApiBaseParams.inputLayoutType;
    uint32_t groupNum = qHeads / kvHeads;
    uint64_t strideQo = qHeads * embd;
    uint64_t strideK = kvHeads * embd;
    uint64_t strideV = kvHeads * embdv;
    if (dataShapeType == 1)
    {
        strideQo = embd;
        strideK = embd;
        strideV = embdv;
    }

    uint64_t batchStrideK = batchSize * maxKvSeqlen * kvHeads * embd * 2;
    uint64_t batchStrideV = batchSize * maxKvSeqlen * kvHeads * embdv * 2;
    AscendC::GlobalTensor<half> qGmTensor;
    AscendC::GlobalTensor<half> kGmTensor;
    AscendC::GlobalTensor<half> vGmTensor;
    AscendC::GlobalTensor<half> sGmTensor;
    AscendC::GlobalTensor<half> pGmTensor;
    AscendC::GlobalTensor<float> oTmpGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    qGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query));
    kGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key));
    vGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(value));
    uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
    sGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace));
    pGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace + workSize));
    oTmpGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + 2 * workSize));
    actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
    actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);
    
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

    uint64_t curBatch = 0;
    uint32_t qSeqlen = maxSeqlen;
    uint32_t kvSeqlen = maxKvSeqlen;
    if (inputLayoutType == 0) {
        uint32_t qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
        uint32_t kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
    }
    uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
    uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
    uint64_t addrQScalar = 0;
    uint64_t addrKScalar = 0;
    uint64_t addrVScalar = 0;
    uint32_t preTotalQBlkNum = 0;
    uint32_t curTotalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNumFirst;
    uint32_t processNum = totalQBlkNum * qHeads;
    uint32_t nextProcess = 0;
    uint32_t isMLA = 1;
    using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
    static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(half);
    for (uint32_t process = GetBlockIdx(); process < processNum; process = nextProcess) {
        while (process >= curTotalQBlkNum * qHeads) {
            curBatch++;
            preTotalQBlkNum = curTotalQBlkNum;
            addrQScalar += static_cast<uint64_t>(qSeqlen * qHeads * embd);
            addrKScalar += static_cast<uint64_t>(kvSeqlen * kvHeads * embd);
            addrVScalar += static_cast<uint64_t>(kvSeqlen * kvHeads * embdv);
            if (inputLayoutType == 0) {
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
            nextProcess = currIter % 2 == 1 ? (currIter + 1) * GetBlockNum() + GetBlockIdx() : (currIter + 2) * GetBlockNum() - 1 - GetBlockIdx();
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

        if (dataShapeType == 1)
        {
            qOffset = addrQScalar + headIdx * embd * maxSeqlen + mIdx * strideQo * ppMScalar;
            kOffset = addrKScalar + (headIdx / groupNum) * embd * maxKvSeqlen;
        }

        uint32_t svN = nLoop == 1 ? kvSeqlen : ppNScalar;
        uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        uint64_t vOffset = addrVScalar + (headIdx / groupNum) * embdv;
        if(dataShapeType == 1){
            vOffset = addrVScalar + (headIdx / groupNum) * embdv * maxKvSeqlen;
        }

        uint32_t nEnd = nLoop;
        if (isTriuMask) {
            nEnd = mIdx + 1;
        }
        uint32_t sBlockStack = nEnd > NO_STACK_S_BLOCK_LIMIT ? 2 : 1; // Currently not splitting K
        uint32_t launchDelay = sBlockStack * 2;
        uint32_t vectMod = 2 * launchDelay;
        uint32_t loopQK = (embd + BLOCK_QK - 1) / BLOCK_QK;
        uint32_t loopV = (embdv + BLOCK_QK - 1) / BLOCK_QK;
        uint32_t qkK = BLOCK_QK;
        uint32_t qkRoundK = BLOCK_QK;
        for (uint32_t nIdx = 0; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
            if (nIdx < nEnd) {
                for (uint32_t splitIdx = 0; splitIdx < sBlockStack && nIdx + splitIdx < nEnd; splitIdx++) {
                    pingpongFlag = (nIdx + splitIdx) % 2;
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
                        uint64_t qOffsetk = qOffset + BLOCK_QK * kIdx;
                        uint64_t kOffsetk = kOffset + BLOCK_QK * kIdx;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag + 5));
                        if (qkM == 1) {
                            AscendC::DataCopy(l1qBufAddrTensor[offset],
                                            qGmTensor[qOffsetk],
                                            AscendC::DataCopyParams(1,                                              // nBurst
                                                                    CeilDiv<BLOCK_SIZE_COPY>(1 * qkRoundK), // lenBurst
                                                                    0,                                              // srcGap
                                                                    0));
                        } else {
                            if (strideQo < STRIDE_LIMIT) {
                                AscendC::DataCopy(l1qBufAddrTensor[offset],
                                                qGmTensor[qOffsetk],
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
                                                    qGmTensor[qOffsetk][i * strideQo],
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
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag + 4));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag + 4));
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag));
                        if (qkM == 1) {
                            AscendC::LoadData(l0aBufTensor[offset],
                                            l1qBufAddrTensor[offset],
                                            AscendC::LoadData2dParams(0,           // baseIdx
                                                                    (qkRoundK + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE,   // repeat
                                                                    1,  // srcStride
                                                                    0,           // sid
                                                                    0,  // dstStride
                                                                    false, // transpose
                                                                    0));         // addrCalMode
                        } else {
                            for (uint32_t l0aLoadIdx = 0; l0aLoadIdx < qkRoundM / BLOCK_SIZE; ++l0aLoadIdx) {
                                AscendC::LoadData(l0aBufTensor[offset + l0aLoadIdx * qkRoundK * BLOCK_SIZE],
                                                l1qBufAddrTensor[offset + l0aLoadIdx * CUBE_MATRIX_SIZE],
                                                AscendC::LoadData2dParams(0,           // baseIdx
                                                                            qkRoundK / BLOCK_SIZE,   // repeat
                                                                            qkRoundM / BLOCK_SIZE,  // srcStride
                                                                            0,           // sid
                                                                            0,  // dstStride
                                                                            false, // transpose
                                                                            0));         // addrCalMode
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag + 5));
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag));

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag));
                        if (strideK < STRIDE_LIMIT) {
                            AscendC::DataCopy(l1kBufAddrTensor[offset],
                                            kGmTensor[kOffsetk],
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
                                                kGmTensor[kOffsetk][i * strideK],
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
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag + 2));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));
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
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag + 2));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag + 2));
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
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag + 2));
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
                svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                auto intriParams = AscendC::FixpipeParamsV220(svRoundN, // nSize
                                                            qkM, // mSize
                                                            qkRoundM,   // srcStride
                                                            svRoundN,   // dstStride
                                                            false);      // enRelu
                intriParams.quantPre = QuantMode_t::F322F16;
                AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod], l0cBufTensor, intriParams);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(QK_READY);
            }
            if (nIdx >= launchDelay) {
                uint32_t l0cPingpongFlag = nIdx % 2;
                uint32_t l0cOffset = l0cPingpongFlag * L0AB_HALF_BUF_SIZE;
                uint32_t svNTriu = nEnd * ppNScalar;
                if (nIdx + sBlockStack > nEnd + launchDelay - 1) {
                    svN = svNTriu > kvSeqlen ? kvSeqlen - (nIdx - launchDelay) * ppNScalar : svNTriu - (nIdx - launchDelay) * ppNScalar;
                } else {
                    svN = ppNScalar * sBlockStack;
                }
                svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
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
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID5));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID6));
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
                AscendC::CrossCoreSetFlag<2, PIPE_FIX>(UPDATE_READY);
                vOffset += svN * strideV;
            }
        }
    }

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
#endif

#ifdef __DAV_C220_VEC__

__aicore__ __attribute__((always_inline)) inline void __set_vcg_mask_mla(int32_t len)
{
    if (len > MAX_ALLOWED_LENGTH_BASE_MLA || len < MIN_ALLOWED_LENGTH_BASE_MLA) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t) 1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
    AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
}

template <typename PFAT>
__aicore__ inline void unpad_flashattention_mla(
    __gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
    __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
    __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
    __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
    __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
    __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
    const typename PFAT::tilingType* __restrict tiling,
    __gm__ uint8_t* gmTiling, TPipe* tPipe)
{
    int32_t subblockIdx = AscendC::GetSubBlockIdx();
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

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
    const uint32_t glUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 16 * UB_UINT8_LINE_SIZE;
    const uint32_t tvUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 17 * UB_UINT8_LINE_SIZE;
    const uint32_t goUbufOffset = 9 * UB_UINT8_BLOCK_SIZE;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::LocalTensor<half> lsUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lsUbufOffset);
    AscendC::LocalTensor<half> lpUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lpUbufOffset);
    AscendC::LocalTensor<float> ls32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32UbufOffset);
    AscendC::LocalTensor<half> maskUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(maskUbufOffset);
    AscendC::LocalTensor<half> maskValueUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(11 * UB_UINT8_BLOCK_SIZE);
    AscendC::LocalTensor<uint8_t> maskU8UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(11 * UB_UINT8_BLOCK_SIZE);
    AscendC::LocalTensor<float> loUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbufOffset);
    AscendC::LocalTensor<half> lmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lmUbufOffset);
    AscendC::LocalTensor<half> hmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hmUbufOffset);
    AscendC::LocalTensor<half> gmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gmUbufOffset);
    AscendC::LocalTensor<half> dmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dmUbufOffset);
    AscendC::LocalTensor<float> llUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(llUbufOffset);
    AscendC::LocalTensor<float> glUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(glUbufOffset);
    AscendC::LocalTensor<float> tvUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(tvUbufOffset);
    AscendC::LocalTensor<half> tv16UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tvUbufOffset);
    AscendC::LocalTensor<uint8_t> tvU8UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(tvUbufOffset);
    AscendC::LocalTensor<float> goUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbufOffset);

    AscendC::GlobalTensor<half> maskGmTensor;
    AscendC::GlobalTensor<uint8_t> maskU8GmTensor;
    AscendC::GlobalTensor<half> oGmTensor;
    AscendC::GlobalTensor<half> sGmTensor;
    AscendC::GlobalTensor<half> pGmTensor;
    AscendC::GlobalTensor<float> oTmpGmTensor;
    AscendC::GlobalTensor<float> upoGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    maskGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(pseShift));
    maskU8GmTensor.SetGlobalBuffer(pseShift);
    oGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOut));
    actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
    actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);
    uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
    sGmTensor.SetGlobalBuffer((__gm__ half *)(workspace));
    pGmTensor.SetGlobalBuffer((__gm__ half *)(workspace + workSize));
    oTmpGmTensor.SetGlobalBuffer((__gm__ float *)(workspace + 2 * workSize));
    upoGmTensor.SetGlobalBuffer((__gm__ float *)(workspace + 3 * workSize));

    uint32_t batchSize = tiling->promptAttentionBaseApiBaseParams.batchSize;
    uint32_t maxSeqlen = tiling->promptAttentionBaseApiBaseParams.maxSeqLen;
    uint32_t qHeads = tiling->promptAttentionBaseApiBaseParams.headNumSize;
    uint32_t embd = tiling->promptAttentionBaseApiBaseParams.headSize;
    uint32_t embdv = tiling->promptAttentionBaseApiBaseParams.embeddingSizeV;
    uint32_t kvHeads = tiling->promptAttentionBaseApiBaseParams.kvHeadNumSize;
    uint32_t isTriuMask = tiling->promptAttentionBaseApiBaseParams.isTriuMask;
    uint32_t totalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNum;
    uint32_t maxKvSeqlen = tiling->promptAttentionBaseApiBaseParams.maxKvSeqLen;
    uint32_t windowSize = 0;
    uint32_t dataShapeType = tiling->promptAttentionBaseApiBaseParams.quantType;
    half tor = static_cast<half>(tiling->promptAttentionBaseApiBaseParams.tor);
    uint32_t headStride = tiling->promptAttentionBaseApiBaseParams.headStride;
    uint32_t maskStride = tiling->promptAttentionBaseApiBaseParams.maskStride;
    uint32_t isClamp = tiling->promptAttentionBaseApiBaseParams.isClamp;
    half clampMin = 0;
    half clampMax = 0;
    uint32_t maskType = tiling->promptAttentionBaseApiBaseParams.maskType;
    uint32_t longSeq = tiling->promptAttentionBaseApiBaseParams.isLongSeq;
    uint32_t inputLayoutType = tiling->promptAttentionBaseApiBaseParams.inputLayoutType;
    uint64_t strideQo = qHeads * embdv;
    if(dataShapeType == 1){
        strideQo = embdv;
    }
    uint32_t loopV = (embdv + BLOCK_QK - 1) / BLOCK_QK;
    uint32_t qkK = BLOCK_QK;
    uint32_t qkRoundK = BLOCK_QK;

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID0));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID3));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID0));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID1));

    uint64_t curBatch = 0;
    uint32_t preTotalQBlkNum = 0;
    uint32_t qSeqlen = maxSeqlen;
    uint32_t kvSeqlen = maxKvSeqlen;
    if (inputLayoutType == 0) {
        uint32_t qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
        uint32_t kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
    }
    uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
    uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
    uint64_t addrOScalar = 0;
    uint32_t curTotalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNumFirst;
    uint32_t processNum = totalQBlkNum * qHeads;
    uint32_t nextProcess = 0;
    uint32_t isMLA = 1;
    for (uint32_t process = GetBlockIdx(); process < processNum; process = nextProcess) {
        while (process >= curTotalQBlkNum * qHeads) {
            curBatch++;
            preTotalQBlkNum = curTotalQBlkNum;
            addrOScalar += static_cast<uint64_t>(qSeqlen * qHeads * embdv);
            if (inputLayoutType == 0) {
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
            nextProcess = currIter % 2 == 1 ? (currIter + 1) * GetBlockNum() + GetBlockIdx() : (currIter + 2) * GetBlockNum() - 1 - GetBlockIdx();
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
        uint32_t subM = (subblockIdx == 1) ? (qkM - qkM / 2) : qkM / 2;
        uint32_t subMD128 = (subM + VECTOR_SIZE - 1) / VECTOR_SIZE;             // up aligned to 128
        uint32_t subMD64 = (subM + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;  // up aligned to 64
        uint32_t roundSubM = (subM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        //******** pre_load *******
        uint32_t qkN = nLoop == 1 ? kvSeqlen : ppNScalar;
        uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        uint32_t pingpongFlag = 0;
        uint32_t offset = pingpongFlag * UB_HALF_BUF_SIZE;
        uint64_t maskBatchOffset = curBatch * maskStride * maxSeqlen;
        uint64_t mask_head_offset = headIdx * ((uint64_t)headStride) * maxSeqlen;
        uint64_t maskOffset = maskBatchOffset + mask_head_offset;
        if (longSeq == 0) {
            maskOffset += mIdx * ppMScalar * maxSeqlen;
        } else {
            AscendC::DataCopy(maskUbufTensor,
                            maskGmTensor[(uint64_t)subblockIdx * qkM / 2 * VECTOR_SIZE],
                            AscendC::DataCopyParams(subM,
                                                    VECTOR_SIZE / BLOCK_SIZE,
                                                    0,
                                                    0)
            );
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
        }

        uint64_t oOffset = addrOScalar + headIdx * embdv + mIdx * ppMScalar * strideQo;
        if(dataShapeType == 1){
            oOffset = addrOScalar + headIdx * embdv * maxSeqlen + mIdx * ppMScalar * strideQo;
        }

        uint32_t nEnd = nLoop;
        if (isTriuMask) {
            nEnd = mIdx + 1;
        }
        uint32_t qkNTriu = nEnd * ppNScalar;
        uint32_t sBlockStack = nEnd > NO_STACK_S_BLOCK_LIMIT ? 2 : 1;
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
                if (subM > 0 && maskType != 0 && longSeq == 0) {
                    if (qkN <= ppNScalar) {
                        pingpongFlag = nIdx % 2;
                        offset = pingpongFlag * UB_HALF_BUF_SIZE;
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((pingpongFlag + 2));
                        AscendC::DataCopyPad(maskUbufTensor[offset], maskGmTensor[maskOffset + (uint64_t)subblockIdx * qkM / 2 * maxSeqlen], AscendC::DataCopyExtParams(subM, qkN * 2, (maxSeqlen - qkN) * 2, 0, 0),
                                            AscendC::DataCopyPadExtParams<half>(false, 0, 0, 0));
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((pingpongFlag + 2));
                    } else {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                        AscendC::DataCopyPad(maskUbufTensor, maskGmTensor[maskOffset + (uint64_t)subblockIdx * qkM / 2 * maxSeqlen], AscendC::DataCopyExtParams(subM, qkN * 2, (maxSeqlen - qkN) * 2, 0, 0),
                                            AscendC::DataCopyPadExtParams<half>(false, 0, 0, 0));
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));
                    }
                    maskOffset += qkN;
                }
                AscendC::CrossCoreWaitFlag(QK_READY);
                uint32_t qkNReduceSum = qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE;
                if (qkN <= VECTOR_SIZE) {
                    pingpongFlag = nIdx % 2;
                    offset = pingpongFlag * UB_HALF_BUF_SIZE;
                    if (subM > 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((pingpongFlag));
                        if (sBlockStack == 2) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((1 - pingpongFlag));
                        }
                        // input QK
                        AscendC::DataCopy(lsUbufTensor[offset],
                                        sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                    (uint64_t)subblockIdx * qkM / 2 * qkRoundN],
                                        AscendC::DataCopyParams(subM,
                                                                qkRoundN / BLOCK_SIZE,
                                                                0,
                                                                0)
                        );
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                        // *** ls = tor * ls
                        AscendC::Muls<half, false>(
                            lsUbufTensor[offset],
                            lsUbufTensor[offset],
                            tor,
                            (uint64_t)0,
                            (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        if (isClamp == 1){
                            // get min(clampMin，ls_ubuf)
                            AscendC::Maxs<half, false>(
                                lsUbufTensor[offset],
                                lsUbufTensor[offset],
                                clampMin,
                                (uint64_t)0,
                                (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();

                            // get max(clampMin，ls_ubuf)
                            AscendC::Mins<half, false>(
                                lsUbufTensor[offset],
                                lsUbufTensor[offset],
                                clampMax,
                                (uint64_t)0,
                                (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        // *** ls = ls + mask
                        if (maskType != 0) {
                            if (longSeq == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((pingpongFlag + 2));
                                AscendC::Add<half, false>(
                                    lsUbufTensor[offset],
                                    lsUbufTensor[offset],
                                    maskUbufTensor[offset],
                                    (uint64_t)0,
                                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((pingpongFlag + 2));
                             } else if (ppNScalar == FLOAT_VECTOR_SIZE && sBlockStack == 2 && nIdx == nEnd - 2) {
                                __set_mask(qkN - FLOAT_VECTOR_SIZE);
                                AscendC::Add<half, false>(
                                    lsUbufTensor[offset + FLOAT_VECTOR_SIZE],
                                    lsUbufTensor[offset + FLOAT_VECTOR_SIZE],
                                    maskUbufTensor,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, STRIDE_8)
                                );
                            } else if (nIdx == nEnd - 1) {
                                __set_mask(qkN);
                                AscendC::Add<half, false>(
                                    lsUbufTensor[offset],
                                    lsUbufTensor[offset],
                                    maskUbufTensor,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, STRIDE_8)
                                );
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        // *** lm = rowmax(ls)
                        if (qkN <= VECTOR_SIZE) {
                            __set_mask(qkN);
                            AscendC::BlockReduceMax<half, false>(
                                tvUbufTensor.ReinterpretCast<half>(),
                                lsUbufTensor[offset],
                                subM,
                                0,
                                2,
                                1,
                                qkRoundN / BLOCK_SIZE
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            __set_vcg_mask_mla(qkRoundN / BLOCK_SIZE);
                            AscendC::BlockReduceMax<half, false>(
                                lmUbufTensor,
                                tvUbufTensor.ReinterpretCast<half>(),
                                (subM * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                0,
                                1,
                                1,
                                STRIDE_8
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        } else {
                            AscendC::BlockReduceMax<half, false>(
                                tvUbufTensor.ReinterpretCast<half>(),
                                lsUbufTensor[offset],
                                subM,
                                0,
                                2,
                                1,
                                qkRoundN / BLOCK_SIZE
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            __set_mask(qkN - VECTOR_SIZE);
                            AscendC::BlockReduceMax<half, false>(
                                tvUbufTensor.ReinterpretCast<half>()[ROWMAX_TEMP_BUF_OFFSET],
                                lsUbufTensor[offset + VECTOR_SIZE],
                                subM,
                                0,
                                2,
                                1,
                                qkRoundN / BLOCK_SIZE
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            __set_vcg_mask_mla((qkRoundN - VECTOR_SIZE) / BLOCK_SIZE);
                            AscendC::Max<half, false>(
                                tvUbufTensor.ReinterpretCast<half>(),
                                tvUbufTensor.ReinterpretCast<half>(),
                                tvUbufTensor.ReinterpretCast<half>()[ROWMAX_TEMP_BUF_OFFSET],
                                (uint64_t)0,
                                (subM * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            AscendC::PipeBarrier<PIPE_V>();
                            __set_vcg_mask_mla(VECTOR_SIZE / BLOCK_SIZE);
                            AscendC::BlockReduceMax<half, false>(
                                lmUbufTensor,
                                tvUbufTensor.ReinterpretCast<half>(),
                                (subM * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                0,
                                1,
                                1,
                                STRIDE_8
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        if (nIdx == 0) {
                            // *** hm = lm
                            AscendC::DataCopy(hmUbufTensor,
                                            lmUbufTensor,
                                            AscendC::DataCopyParams(1,
                                            roundSubM / BLOCK_SIZE,
                                            0,
                                            0)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        } else {
                            // *** hm = MAX(lm, gm)
                            AscendC::Max<half, false>(
                                hmUbufTensor,
                                lmUbufTensor,
                                gmUbufTensor,
                                (uint64_t)0,
                                subMD128,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** dm = gm - hm
                            AscendC::Sub<half, false>(
                                dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_HALF_LINE_SIZE],
                                gmUbufTensor,
                                hmUbufTensor,
                                (uint64_t)0,
                                subMD128,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        // *** gm = hm
                        AscendC::DataCopy(gmUbufTensor,
                                        hmUbufTensor,
                                        AscendC::DataCopyParams(1,
                                        roundSubM / BLOCK_SIZE,
                                        0,
                                        0)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** hm_block = expand_to_block(hm), 存放于 tv
                        AscendC::Brcb(
                            tvUbufTensor.ReinterpretCast<uint16_t>(),
                            hmUbufTensor.ReinterpretCast<uint16_t>(),
                            roundSubM / FLOAT_BLOCK_SIZE,
                            AscendC::BrcbRepeatParams(1, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** ls = ls - hm_block
                        for (uint32_t vsubIdx = 0; vsubIdx < qkN / VECTOR_SIZE; ++vsubIdx) {
                            AscendC::Sub<half, false>(
                                lsUbufTensor[offset + vsubIdx * VECTOR_SIZE],
                                lsUbufTensor[offset + vsubIdx * VECTOR_SIZE],
                                tvUbufTensor.ReinterpretCast<half>(),
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, 1)
                            );
                        }
                        if (qkN % VECTOR_SIZE > 0) {
                            __set_mask(qkN % VECTOR_SIZE);
                            AscendC::Sub<half, false>(
                                lsUbufTensor[offset + qkN / VECTOR_SIZE * VECTOR_SIZE],
                                lsUbufTensor[offset + qkN / VECTOR_SIZE * VECTOR_SIZE],
                                tvUbufTensor.ReinterpretCast<half>(),
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, 1)
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** ls = castfp16to32(ls)
                        AscendC::Cast<float, half, false>(
                            ls32UbufTensor,
                            lsUbufTensor[offset],
                            AscendC::RoundMode::CAST_NONE,
                            (uint64_t)0,
                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** ls = exp(ls)
                        AscendC::Exp<float, false>(
                            ls32UbufTensor,
                            ls32UbufTensor,
                            (uint64_t)0,
                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        // *** lp = castfp32to16(ls)
                        AscendC::Cast<half, float, false>(
                            lpUbufTensor[offset],
                            ls32UbufTensor,
                            AscendC::RoundMode::CAST_NONE,
                            (uint64_t)0,
                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                            AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));

                        // *** ll = rowsum(ls32)
                        if (qkN <= FLOAT_VECTOR_SIZE) {
                            __set_mask(qkN);
                            AscendC::RepeatReduceSum<float, false>(
                                llUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],
                                ls32UbufTensor,
                                subM,
                                0,
                                0,
                                1,
                                1,
                                qkRoundN / FLOAT_BLOCK_SIZE
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        } else {
                            for (uint32_t rowsumIdx = 1; rowsumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowsumIdx) {
                                AscendC::Add<float, false>(
                                    ls32UbufTensor,
                                    ls32UbufTensor,
                                    ls32UbufTensor[rowsumIdx * FLOAT_VECTOR_SIZE],
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
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::RepeatReduceSum<float, false>(
                                llUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],
                                ls32UbufTensor,
                                subM,
                                0,
                                0,
                                1,
                                1,
                                qkRoundN / FLOAT_BLOCK_SIZE
                            );
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        if (nIdx == 0) {
                            AscendC::DataCopy(glUbufTensor,
                                            llUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],
                                            AscendC::DataCopyParams(1,
                                            roundSubM / FLOAT_BLOCK_SIZE,
                                            0,
                                            0)
                            );
                        } else {
                            AscendC::Cast<float, half, false>(
                                tvUbufTensor,
                                dmUbufTensor[(nIdx / sBlockStack) % 4 * UB_HALF_LINE_SIZE],
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                subMD64,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** dm = exp(dm)
                            AscendC::Exp<float, false>(
                                tvUbufTensor,
                                tvUbufTensor,
                                (uint64_t)0,
                                subMD64,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** dm_block = expand_to_block(dm), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE],
                                tvUbufTensor.ReinterpretCast<uint32_t>(),
                                roundSubM / FLOAT_BLOCK_SIZE,
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** gl = dm * gl
                            AscendC::Mul<float, false>(
                                glUbufTensor,
                                tvUbufTensor,
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
                                llUbufTensor[(nIdx / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],
                                (uint64_t)0,
                                subMD64,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                        AscendC::DataCopy(pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                    (uint64_t)subblockIdx * qkM / 2 * qkRoundN],
                                        lpUbufTensor[offset],
                                        AscendC::DataCopyParams(1,
                                                                subM * qkRoundN / BLOCK_SIZE,
                                                                0,
                                                                0)
                        );
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((pingpongFlag));
                        if (sBlockStack == 2) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((1 - pingpongFlag));
                        }
                    }
                } else {
                    bool lastNLoop = nIdx + sBlockStack > nEnd - 1;
                    if (subM > 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
                        // input QK
                        AscendC::DataCopy(lsUbufTensor,
                                        sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                            (uint64_t)subblockIdx * qkM / 2 * qkRoundN],
                                        AscendC::DataCopyParams(mSlice,
                                                                qkRoundN / BLOCK_SIZE,
                                                                0,
                                                                0)
                        );
                        if (subM > mSlice) {
                            if (mEnd > 1) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
                            }
                            AscendC::DataCopy(lsUbufTensor[mSlice * qkRoundN],
                                            sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                        (uint64_t)subblockIdx * qkM / 2 * qkRoundN + mSlice * qkRoundN],
                                            AscendC::DataCopyParams(subM - mSlice,
                                                                    qkRoundN / BLOCK_SIZE,
                                                                    0,
                                                                    0)
                            );
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                        // *** ls = tor * ls
                        AscendC::Muls<half, false>(
                            lsUbufTensor,
                            lsUbufTensor,
                            tor,
                            (uint64_t)0,
                            (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                        );
                        AscendC::PipeBarrier<PIPE_V>();
                        if (maskType != 0) {
                            if (longSeq == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));
                                AscendC::Add<half, false>(
                                    lsUbufTensor,
                                    lsUbufTensor,
                                    maskUbufTensor,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                            } else if (nIdx == nEnd - 2) {
                                __set_mask(qkN - ppNScalar);
                                AscendC::Add<half, false>(
                                    lsUbufTensor[ppNScalar],
                                    lsUbufTensor[ppNScalar],
                                    maskUbufTensor,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, STRIDE_8)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        if (isClamp == 1){
                            // get min(clampMin，ls_ubuf)
                            AscendC::Maxs<half, false>(
                                lsUbufTensor,
                                lsUbufTensor,
                                clampMin,
                                (uint64_t)0,
                                (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // get max(clampMin，ls_ubuf)
                            AscendC::Mins<half, false>(
                                lsUbufTensor,
                                lsUbufTensor,
                                clampMax,
                                (uint64_t)0,
                                (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        if (qkN != SOFTMAX_MAX_LENGTH) {
                            AscendC::DataCopy(ls32UbufTensor.ReinterpretCast<half>(),
                                            lsUbufTensor,
                                            AscendC::DataCopyParams(1,
                                            VECTOR_SIZE / BLOCK_SIZE,
                                            (qkRoundN - VECTOR_SIZE) / BLOCK_SIZE,
                                            0)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            __set_mask(qkN - VECTOR_SIZE);
                            AscendC::Max<half, false>(
                                ls32UbufTensor.ReinterpretCast<half>(),
                                ls32UbufTensor.ReinterpretCast<half>(),
                                lsUbufTensor[VECTOR_SIZE],
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, qkRoundN / BLOCK_SIZE)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            AscendC::BlockReduceMax<half, false>(
                                tvUbufTensor.ReinterpretCast<half>(),
                                ls32UbufTensor.ReinterpretCast<half>(),
                                subM,
                                0,
                                2,
                                1,
                                STRIDE_8
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            __set_vcg_mask_mla(VECTOR_SIZE / BLOCK_SIZE);
                            AscendC::BlockReduceMax<half, false>(
                                lmUbufTensor,
                                tvUbufTensor.ReinterpretCast<half>(),
                                (subM * BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                0,
                                1,
                                1,
                                STRIDE_8
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        } else {
                            AscendC::BlockReduceMax<half, false>(
                                ls32UbufTensor.ReinterpretCast<half>(),
                                lsUbufTensor,
                                subM * qkRoundN / VECTOR_SIZE,
                                0,
                                1,
                                1,
                                STRIDE_8
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::BlockReduceMax<half, false>(
                                lmUbufTensor,
                                ls32UbufTensor.ReinterpretCast<half>(),
                                (subM *  BLOCK_SIZE + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                0,
                                1,
                                1,
                                STRIDE_8
                            );
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        if (nIdx == 0) {
                            // *** hm = lm
                            AscendC::DataCopy(gmUbufTensor,
                                            lmUbufTensor,
                                            AscendC::DataCopyParams(1,
                                            roundSubM / BLOCK_SIZE,
                                            0,
                                            0)
                            );
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint16_t>(),
                                lmUbufTensor.ReinterpretCast<uint16_t>(),
                                roundSubM / FLOAT_BLOCK_SIZE,
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        } else {
                            // *** hm = MAX(lm, gm)
                            AscendC::Max<half, false>(
                                hmUbufTensor,
                                lmUbufTensor,
                                gmUbufTensor,
                                (uint64_t)0,
                                1,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** hm_block = expand_to_block(hm), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint16_t>(),
                                hmUbufTensor.ReinterpretCast<uint16_t>(),
                                roundSubM / FLOAT_BLOCK_SIZE,
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                            );
                            // *** dm = gm - hm
                            AscendC::Sub<half, false>(
                                dmUbufTensor[(nIdx / sBlockStack) % launchDelay * UB_HALF_LINE_SIZE],
                                gmUbufTensor,
                                hmUbufTensor,
                                (uint64_t)0,
                                1,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** gm = hm
                            AscendC::DataCopy(gmUbufTensor,
                                            hmUbufTensor,
                                            AscendC::DataCopyParams(1,
                                            roundSubM / BLOCK_SIZE,
                                            0,
                                            0)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        // *** ls = ls - hm_block
                        for (uint32_t vsubIdx = 0; vsubIdx < qkN / VECTOR_SIZE; ++vsubIdx) {
                            AscendC::Sub<half, false>(
                                lsUbufTensor[vsubIdx * VECTOR_SIZE],
                                lsUbufTensor[vsubIdx * VECTOR_SIZE],
                                tvUbufTensor.ReinterpretCast<half>(),
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, 1)
                            );
                        }
                        if (qkN % VECTOR_SIZE > 0) {
                            __set_mask(qkN % VECTOR_SIZE);
                            AscendC::Sub<half, false>(
                                lsUbufTensor[qkN / VECTOR_SIZE * VECTOR_SIZE],
                                lsUbufTensor[qkN / VECTOR_SIZE * VECTOR_SIZE],
                                tvUbufTensor.ReinterpretCast<half>(),
                                (uint64_t)0,
                                subM,
                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, 1)
                            );
                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        for (uint32_t splitIdx = 0; splitIdx < mEnd; splitIdx++) {
                            bool lastMLoop = splitIdx == mEnd - 1;
                            uint32_t mSplit =  lastMLoop ? subM - splitIdx * mSlice : mSlice;
                            // *** ls = castfp16to32(ls)
                            AscendC::Cast<float, half, false>(
                                ls32UbufTensor,
                                lsUbufTensor[splitIdx * mSlice * qkRoundN],
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** ls = exp(ls)
                            AscendC::Exp<float, false>(
                                ls32UbufTensor,
                                ls32UbufTensor,
                                (uint64_t)0,
                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** lp = castfp32to16(ls)
                            AscendC::Cast<half, float, false>(
                                lpUbufTensor[splitIdx * mSlice * qkRoundN],
                                ls32UbufTensor,
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));

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
                                    ls32UbufTensor[qkNReduceSum],
                                    (uint64_t)0,
                                    mSplit,
                                    AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::RepeatReduceSum<float, false>(
                                llUbufTensor[(nIdx / sBlockStack) % launchDelay * UB_FLOAT_LINE_SIZE + splitIdx * mSlice],
                                ls32UbufTensor,
                                mSplit,
                                0,
                                0,
                                1,
                                1,
                                qkRoundN / FLOAT_BLOCK_SIZE
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            AscendC::DataCopy(pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                ((uint64_t)subblockIdx * qkM / 2 + splitIdx * mSlice) * qkRoundN],
                                            lpUbufTensor[splitIdx * mSlice * qkRoundN],
                                            AscendC::DataCopyParams(mSplit,
                                                                    qkRoundN / BLOCK_SIZE,
                                                                    0,
                                                                    0)
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((splitIdx));
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        if (nIdx == 0) {
                            AscendC::DataCopy(glUbufTensor,
                                            llUbufTensor[(nIdx / sBlockStack) % launchDelay * UB_FLOAT_LINE_SIZE],
                                            AscendC::DataCopyParams(1,
                                            roundSubM / FLOAT_BLOCK_SIZE,
                                            0,
                                            0)
                            );
                        } else {
                            AscendC::Cast<float, half, false>(
                                tvUbufTensor,
                                dmUbufTensor[(nIdx / sBlockStack) % launchDelay * UB_HALF_LINE_SIZE],
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                subMD64,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** dm = exp(dm)
                            AscendC::Exp<float, false>(
                                tvUbufTensor,
                                tvUbufTensor,
                                (uint64_t)0,
                                subMD64,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** dm_block = expand_to_block(dm), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE],
                                tvUbufTensor.ReinterpretCast<uint32_t>(),
                                roundSubM / FLOAT_BLOCK_SIZE,
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** gl = dm * gl
                            AscendC::Mul<float, false>(
                                glUbufTensor,
                                tvUbufTensor,
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
                                llUbufTensor[(nIdx / sBlockStack) % launchDelay * UB_FLOAT_LINE_SIZE],
                                (uint64_t)0,
                                subMD64,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                    }
                }
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SOFTMAX_READY);
            }
            if (nIdx >= launchDelay) {
                // *** 更新 L 和 O
                AscendC::CrossCoreWaitFlag(UPDATE_READY);
                for(uint32_t kIdx = 0; kIdx < loopV; kIdx++){
                    uint64_t oOffsetk = oOffset + kIdx * BLOCK_QK;
                    if (subM > 0) {
                        qkK = (kIdx == (loopV - 1))? embdv - kIdx * BLOCK_QK : BLOCK_QK;
                        qkRoundK = (qkK + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

                        if(nIdx == launchDelay){
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                            AscendC::DataCopy(goUbufTensor,
                                            oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE * LOCAL_SIZE + kIdx * TMP_SIZET  + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod * loopV+
                                                            (uint64_t)subblockIdx * qkM / 2 * qkRoundK],
                                            AscendC::DataCopyParams(1,
                                                                    subM * qkRoundK / FLOAT_BLOCK_SIZE,
                                                                    0,
                                                                    0)
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));
                        }
                        else{
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                            AscendC::DataCopy(goUbufTensor,
                                            upoGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + kIdx * TMP_SIZET +
                                                        (uint64_t)subblockIdx * qkM / 2 * qkRoundK],
                                            AscendC::DataCopyParams(1,
                                                                    subM * qkRoundK / FLOAT_BLOCK_SIZE,
                                                                    0,
                                                                    0)
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));

                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));

                            AscendC::DataCopy(loUbufTensor,
                                            oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE * LOCAL_SIZE + kIdx * TMP_SIZET  + (nIdx - launchDelay) % vectMod * TMP_SIZE / vectMod * loopV +
                                                            (uint64_t)subblockIdx * qkM / 2 * qkRoundK],
                                            AscendC::DataCopyParams(1,
                                                                    subM * qkRoundK / FLOAT_BLOCK_SIZE,
                                                                    0,
                                                                    0)
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID4));

                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::Cast<float, half, false>(
                                tvUbufTensor,
                                dmUbufTensor[((nIdx - launchDelay) / sBlockStack % 4)  * UB_HALF_LINE_SIZE],
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                subMD64,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** dm = exp(dm)
                            AscendC::Exp<float, false>(
                                tvUbufTensor,
                                tvUbufTensor,
                                (uint64_t)0,
                                subMD64,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** dm_block = expand_to_block(dm), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE],
                                tvUbufTensor.ReinterpretCast<uint32_t>(),
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
                                    tvUbufTensor[VECTOR_SIZE],
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
                                    tvUbufTensor[VECTOR_SIZE],
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundK / FLOAT_BLOCK_SIZE, qkRoundK / FLOAT_BLOCK_SIZE, 1)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID4));
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
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
                        }
                        // *** gl = castfp32to16(gl)
                        if (nIdx + sBlockStack > nEnd + launchDelay - 1){
                            AscendC::PipeBarrier<PIPE_V>();
                            if(kIdx == 0){
                                AscendC::Cast<half, float, false>(
                                    glUbufTensor.ReinterpretCast<half>(),
                                    glUbufTensor,
                                    AscendC::RoundMode::CAST_NONE,
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                );
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** go = castfp32to16(go)
                            AscendC::Cast<half, float, false>(
                                goUbufTensor.ReinterpretCast<half>(),
                                goUbufTensor,
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                (subM * qkRoundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** gl_block = expand_to_block(gl), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint16_t>(),
                                glUbufTensor.ReinterpretCast<uint16_t>(),
                                roundSubM / FLOAT_BLOCK_SIZE,
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** go = go / gl_block
                            for (uint32_t vdiv_idx = 0; vdiv_idx < qkK / VECTOR_SIZE; ++vdiv_idx) {
                                AscendC::Div<half, false>(
                                    goUbufTensor.ReinterpretCast<half>()[vdiv_idx * VECTOR_SIZE],
                                    goUbufTensor.ReinterpretCast<half>()[vdiv_idx * VECTOR_SIZE],
                                    tvUbufTensor.ReinterpretCast<half>(),
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundK / BLOCK_SIZE, qkRoundK / BLOCK_SIZE, 1)
                                );
                            }
                            if (qkK % VECTOR_SIZE > 0) {
                                __set_mask(qkK % VECTOR_SIZE);
                                AscendC::Div<half, false>(
                                    goUbufTensor.ReinterpretCast<half>()[qkK / VECTOR_SIZE * VECTOR_SIZE],
                                    goUbufTensor.ReinterpretCast<half>()[qkK / VECTOR_SIZE * VECTOR_SIZE],
                                    tvUbufTensor.ReinterpretCast<half>(),
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundK / BLOCK_SIZE, qkRoundK / BLOCK_SIZE, 1)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            // ********************* move O to GM ************************
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID1));
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID1));
                            AscendC::DataCopyPad(oGmTensor[oOffsetk + (uint64_t)subblockIdx * qkM / 2 * strideQo],
                                                goUbufTensor.ReinterpretCast<half>(),
                                                AscendC::DataCopyExtParams(subM,
                                                                        qkK * 2,
                                                                        0,
                                                                        (strideQo - qkK) * 2,
                                                                        0)
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                        }
                        else{
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID2));
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID2));
                                AscendC::DataCopy(upoGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE +  kIdx * TMP_SIZET +
                                                (uint64_t)subblockIdx * qkM / 2 * qkRoundK],
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

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID0));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID3));
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID0));
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID1));
    AscendC::PipeBarrier<PIPE_ALL>();
}
#endif

template <typename PFAT>
class PromptFlashAttentionBaseMLA {
public:
    __aicore__ inline PromptFlashAttentionBaseMLA() {};
    __aicore__ inline void Init(__gm__ uint8_t*  query, __gm__ uint8_t*  key, __gm__ uint8_t*  value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t*  attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t*  attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t*  workspace,
                                const typename PFAT::tilingType* __restrict tiling,
                                __gm__ uint8_t* gmTiling, TPipe* tPipe, __gm__ uint8_t* alibi_coeff);
};

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionBaseMLA<PFAT>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                            __gm__ uint8_t* value, __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                            __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                            __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                            __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                            __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                            const typename PFAT::tilingType* __restrict tiling,
                            __gm__ uint8_t* gmTiling, TPipe* tPipe, __gm__ uint8_t* alibi_coeff) {
    #ifdef __DAV_C220_CUBE__
        unpad_flashattention_mla<PFAT>(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                                queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                                attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe);

    #elif __DAV_C220_VEC__
        unpad_flashattention_mla<PFAT>(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                                queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                                attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe);
    #endif
}

#endif  // PROMPT_FLASH_ATTENTION_BASE_MLA_H
