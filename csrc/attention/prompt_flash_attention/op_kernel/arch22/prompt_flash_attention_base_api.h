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
 * \file prompt_flash_attention_base_api.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_BASE_API_H
#define PROMPT_FLASH_ATTENTION_BASE_API_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common_func.h"
#include "hardware.h"
#include "mem.h"

constexpr int32_t QK_READY = 1;
constexpr int32_t UPDATE_READY = 3;
constexpr int32_t BIT_SHIFT = 8;
constexpr int32_t SOFTMAX_MAX_LENGTH = 256;
constexpr int32_t SOFTMAX_READY = 2;
constexpr int32_t NO_STACK_S_BLOCK_LIMIT = 4;
constexpr int32_t BLOCK_SIZE = 16;
constexpr uint32_t STRIDE_LIMIT = 65536;

constexpr int32_t MLA_THRESHOLD = 256;
constexpr int32_t LONG_SEQ_LEN = 128;
const int32_t PP_BLOCK_BUFFER_SIZE = 128 * 128;
const int32_t PP_MM_NUM = 8;
const int32_t PP_INDEX = 16;
const int32_t PP_NN_NUM = 16;

constexpr int32_t PP_MM[] = {16, 32, 48, 64, 80, 96, 112, 128};
constexpr int32_t PP_NN[] = {16,  32,  48,  64,  80,  96,  112, 128,
                            144, 160, 176, 192, 208, 224, 240, 256};

constexpr int32_t quantType_3 = 3;
constexpr int32_t INT8_MAX_127 = 127;

const uint32_t PARITY_CHECK_BASE_KER_BASE_API = 2;

constexpr int32_t ZERO_BASE_API = 0;
constexpr int32_t ONE_BASE_API = 1;
constexpr int32_t TWO_BASE_API = 2;
constexpr int32_t FOUR_BASE_API = 4;
constexpr int32_t EIGHT_BASE_API = 8;
constexpr int32_t SIXTEEN_BASE_API = 16;
constexpr int32_t THIRTY_TWO_BASE_API = 32;
constexpr int32_t SIXTY_FOUR_BASE_API = 64;

constexpr int32_t HEAD_STRIDE_INVALID_BASE_API = 0;
constexpr int32_t MASK_TYPE_NO_MASK_BASE_API = 2;
constexpr int32_t ELEMENT_SIZE_BYTES_BASE_API = 4; 
constexpr int32_t NO_SUB_BLOCK_BASE_API = 0;
constexpr int32_t DEFAULT_MASK_TYPE_BASE_API = 0;

constexpr int32_t MAX_WINDOW_BLOCKS_BASE_API = 3;
constexpr int32_t MASK_OFFSET_NONE_BASE_API = 0; 
constexpr int32_t MASK_OFFSET_1X_BASE_API = 1; 
constexpr int32_t MASK_OFFSET_2X_BASE_API = 2; 
constexpr int32_t MASK_OFFSET_3X_BASE_API = 3; 

constexpr int32_t MAX_ALLOWED_LENGTH_BASE_API = 16;  
constexpr int32_t MIN_ALLOWED_LENGTH_BASE_API = 1; 

constexpr int32_t STRIDE_4 = 8;  // 4 is the stride
constexpr int32_t STRIDE_8 = 8;  // 8 is the stride
constexpr int32_t STRIDE_16 = 8;  // 16 is the stride

struct MNibd {
    int32_t mIbd;
    int32_t nIbd;
};

enum ScaleType {
    SCALE_TOR = 0,
    SCALE_LOGN = 1,
    SCALE_LOGN_FP32 = 2
};

enum class MaskType 
{
    MASK_TYPE_NONE = 0,
    MASK_TYPE_TRIU = 1
};

__aicore__ __attribute__((always_inline)) inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound)
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

__aicore__ __attribute__((always_inline)) inline int32_t ConvertValueToIndexNN(int32_t val, int32_t idxBound)
{
    return (val > PP_NN[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

__aicore__ __attribute__((always_inline)) inline int32_t GetMin(int32_t valA, int32_t valB)
{
    return (valA <= valB) ? valA : valB;
}

__aicore__ __attribute__((always_inline)) inline int32_t GetMax(int32_t valA, int32_t valB)
{
    return (valA > valB) ? valA : valB;
}

__aicore__ __attribute__((always_inline)) inline MNibd GetPPmnIbd(uint32_t qSeqLen, uint32_t kvSeqLen, uint32_t headSize, uint32_t isMLA)
{
    int32_t qSeqlenAligned = (qSeqLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int32_t kvSeqlenAligned = (kvSeqLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int32_t embeddingSizeAligned = (headSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int32_t tilingK = embeddingSizeAligned < LONG_SEQ_LEN ? LONG_SEQ_LEN : embeddingSizeAligned;
    int32_t nUbd = isMLA ? GetMin((PP_BLOCK_BUFFER_SIZE / tilingK / BLOCK_SIZE) * BLOCK_SIZE, kvSeqlenAligned)
                        : GetMin(LONG_SEQ_LEN, kvSeqlenAligned);
    int32_t nIbd = ConvertValueToIndexNN(nUbd, PP_NN_NUM - 1);
    int32_t mUbd = isMLA ? GetMin((PP_BLOCK_BUFFER_SIZE / GetMax(embeddingSizeAligned, PP_NN[nIbd]) / BLOCK_SIZE) * BLOCK_SIZE, 
                            qSeqlenAligned)
                        : GetMin(LONG_SEQ_LEN, qSeqlenAligned);
    int32_t mIbd = ConvertValueToIndexMM(mUbd, PP_MM_NUM - 1);
    MNibd mnIbd;
    mnIbd.mIbd = mIbd;
    mnIbd.nIbd = nIbd;
    return mnIbd;
}

template <typename TILING_TYPE, typename Q_T, typename KV_T, typename O_T, typename S_T, typename P_T, typename TMP_T, typename U_T = O_T, OptimizationMode M = OptimizationMode::HighPerformance, const int MASK_TYPE = 0, const bool SWA_FLAG = false, const bool SWA_COMPRESS = false, typename...Args>
struct PFATypeNew {
    using tilingType = TILING_TYPE;
    using qDType = Q_T;
    using kvDType = KV_T;
    using outputDType = O_T;
    using sDType = S_T;
    using pDType = P_T;
    using oTmpDType = TMP_T;
    using maskDType = U_T;
    static constexpr int maskType = MASK_TYPE;
    static constexpr bool swaFlag = SWA_FLAG;
    static constexpr bool swaCompress = SWA_COMPRESS;
    static constexpr OptimizationMode calcMode = M;
};

#ifdef __DAV_C220_CUBE__
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;  // 128 * 128
constexpr int32_t CUBE_MATRIX_SIZE = 256;         // 16 * 16
constexpr int32_t L0AB_UINT8_BLOCK_SIZE = 32768;  // 128 * 128 * 2B
constexpr int32_t TMP_SIZE = 32768 * 4;               // 128 * 256 * 2

template<typename PFAT>
class UnpadAttentionDecoderAic {
public:
    __aicore__ inline UnpadAttentionDecoderAic() {};

    __aicore__ __attribute__((always_inline)) inline void Run(
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

        // GlobalTensor init set
        qGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T *>(query));
        kGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T *>(key));
        vGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T *>(value));
        oGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ O_T *>(attentionOut));

        // 待修改workspace
        uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
        sGmTensor.SetGlobalBuffer((__gm__ S_T *)(workspace));
        pGmTensor.SetGlobalBuffer((__gm__ P_T *)(workspace + workSize));
        oTmpGmTensor.SetGlobalBuffer((__gm__ TMP_T *)(workspace + 2 * workSize));
        deqQkGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(deqScale1));

        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);

        // LocalTensor init set
        l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1qBufAddrOffset);
        l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1kBufAddrOffset);
        l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, KV_T>(l1vBufAddrOffset);
        l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, P_T>(l1pBufAddrOffset);

        l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, Q_T>(0);
        l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, Q_T>(0);
        l0cBufTensor = buf.GetBuffer<BufferType::ASCEND_L0C, TMP_T>(0);

        // tiling
        uint32_t batchSize = tiling->promptAttentionBaseApiBaseParams.batchSize;
        uint32_t maxSeqlen = tiling->promptAttentionBaseApiBaseParams.maxSeqLen;
        uint32_t qHeads = tiling->promptAttentionBaseApiBaseParams.headNumSize;
        uint32_t embd = tiling->promptAttentionBaseApiBaseParams.headSize;
        uint32_t embdv = tiling->promptAttentionBaseApiBaseParams.embeddingSizeV;
        uint32_t kvHeads = tiling->promptAttentionBaseApiBaseParams.kvHeadNumSize;
        uint32_t isTriuMask = tiling->promptAttentionBaseApiBaseParams.isTriuMask;
        uint32_t totalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNum;
        uint32_t maxKvSeqlen = tiling->promptAttentionBaseApiBaseParams.maxKvSeqLen;
        uint32_t quantType = tiling->promptAttentionBaseApiBaseParams.quantType;
        uint32_t dataShapeType = tiling->promptAttentionBaseApiBaseParams.dataShapeType;
        uint32_t windowSize = 0;
        uint32_t qSeqlen = actualSeqLengthsGm.GetValue(0);
        uint32_t kvSeqlen = actualSeqLengthsKVGm.GetValue(0);
        uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
        uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
        uint64_t addrQScalar = 0;
        uint64_t addrKScalar = 0;
        uint64_t addrVScalar = 0;

        uint32_t groupNum = qHeads / kvHeads;
        uint64_t strideQo = qHeads * embd;
        uint64_t strideKv = kvHeads * embd;
        if (dataShapeType == 1) {
            strideQo = embd;
            strideKv = embd;
        }

        uint64_t batchStrideKv = batchSize * maxKvSeqlen * kvHeads * embd * sizeof(Q_T);

        uint32_t __k = embd;
        uint32_t roundK = (__k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));

        uint64_t curBatch = 0;
        uint32_t preTotalQBlkNum = 0;

        uint32_t curTotalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNumFirst;
        uint32_t processNum = totalQBlkNum * qHeads;
        uint32_t nextProcess = 0;
        uint32_t isMLA = 0;
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

            // **************** pre_load *****************
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
                using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(Q_T);
                AscendC::DataCopy(
                    l1qBufAddrTensor,
                    qGmTensor[qOffset],
                    AscendC::DataCopyParams(1, CeilDiv<BLOCK_SIZE_COPY>(1 * RoundUp<uint32_t>(roundK, 32 / sizeof(Q_T))), 0, 0)
                );
            } else {
                using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(Q_T);
                if(strideQo < STRIDE_LIMIT){
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
                }else {
                    for (uint32_t i = 0; i < qkM; ++i) {
                        AscendC::DataCopy(l1qBufAddrTensor[i * BLOCK_SIZE_COPY],
                                    qGmTensor[qOffset + i * strideQo],
                                        AscendC::Nd2NzParams(1,           // ndNum
                                                            1,           // nValue
                                                            __k, // dValue
                                                            0,           // srcNdMatrixStride, unused
                                                            0,           // srcDValue
                                                            qkRoundM,  // dstNzC0Stride
                                                            0,           // dstNzNStride
                                                            0));         // dstNzMatrixStride, unused
                    }
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));

            uint32_t svN = nLoop == 1 ? kvSeqlen : ppNScalar;
            uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            uint64_t vOffset = addrVScalar + (headIdx / groupNum) * embd;
            if (dataShapeType == 1) {
                vOffset = addrVScalar + (headIdx / groupNum) * embd * maxKvSeqlen;
            }

            uint32_t nEnd = nLoop;
            uint32_t windowStart = (windowSize + ppNScalar - 1) / ppNScalar;
            uint32_t nStart = 0;
            if (isTriuMask || windowSize > 0) {
                nEnd = mIdx + 1;
            }
            if constexpr (swaFlag) {
                if (windowSize > 0 && windowSize < kvSeqlen) {
                    nStart = (mIdx < windowStart) ? 0 : mIdx - windowStart;
                    kOffset += nStart * strideKv * ppNScalar;
                    vOffset += nStart * strideKv * ppNScalar;
                }
            }
            uint32_t sBlockStack = nEnd > NO_STACK_S_BLOCK_LIMIT ? 2 : 1; // Currently not splitting K
            uint32_t launchDelay = sBlockStack * 2;
            uint32_t vectMod = 2 * launchDelay;

            for (uint32_t nIdx = nStart; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
                if (nIdx < nEnd) {
                    for (uint32_t splitIdx = 0; splitIdx < sBlockStack && nIdx + splitIdx < nEnd; splitIdx++) {
                        pingpongFlag = (nIdx + splitIdx - nStart) % 2;
                        offset = pingpongFlag * L0AB_HALF_BUF_SIZE;
                        if (nIdx + splitIdx == (nLoop - 1)) {
                            qkN = (kvSeqlen - (nIdx + splitIdx) * ppNScalar);
                            qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                        }
                        bool lastSplit = splitIdx == sBlockStack - 1 || nIdx + splitIdx == nEnd - 1;
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag));
                        uint32_t roundRow1 = RoundUp<uint32_t>(roundK, 32 / sizeof(Q_T));
                        if (qkM == 1) {
                            AscendC::LoadData(l0aBufTensor[offset],
                                            l1qBufAddrTensor,
                                            AscendC::LoadData2dParams(0,           // baseIdx
                                                                    NumMatrixsRoundUp<Q_T>(roundRow1),  // repeat
                                                                    1,  // srcStride
                                                                    0,           // sid
                                                                    0,  // dstStride
                                                                    false, // transpose
                                                                    0));         // addrCalMode
                        } else {
                            using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                            static constexpr uint32_t ROW_BLOCK_SIZE = 16;
                            static constexpr uint32_t COL_BLOCK_SIZE = 32 / sizeof(Q_T);
                            static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(Q_T);
                            static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;
                            for(uint32_t i = 0; i < qkRoundM / ROW_BLOCK_SIZE; i++){
                                AscendC::LoadData(l0aBufTensor[offset][i *  ROW_BLOCK_SIZE * roundRow1],
                                            l1qBufAddrTensor[i * FRACTAL_SIZE],
                                            AscendC::LoadData2dParams(0,           // baseIdx
                                                                    static_cast<uint16_t>(roundRow1 / COL_BLOCK_SIZE),  // repeat
                                                                    qkRoundM / ROW_BLOCK_SIZE,  // srcStride
                                                                    0,           // sid
                                                                    0,  // dstStride
                                                                    false, // transpose
                                                                    0));         // addrCalMode
                            }
                        }
                        // *** Prepare K to L1
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag));
                        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                        static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(Q_T);
                        if (strideKv < STRIDE_LIMIT) {
                                AscendC::DataCopy(l1kBufAddrTensor[offset],
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
                            for (uint32_t i = 0; i < qkN; ++i) {
                                AscendC::DataCopy(l1kBufAddrTensor[offset + i * BLOCK_SIZE_COPY],
                                                kGmTensor[kOffset + i * strideKv],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    1,           // nValue
                                                                    __k, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    0,           // srcDValue
                                                                    qkRoundN,   // dstNzC0Stride
                                                                    0,           // dstNzNStride
                                                                    0)           // dstNzMatrixStride, unused
                                );
                            }
                        }
                        kOffset += ppNScalar * strideKv;
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag + 2));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((pingpongFlag));
                        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                        static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(Q_T);
                        AscendC::LoadData(
                            l0bBufTensor[offset],
                            l1kBufAddrTensor[offset],
                            AscendC::LoadData2dParams(0, NumMatrixsRoundUp<Q_T>(RoundUp<uint32_t>(roundK, 32 / sizeof(Q_T)) * qkRoundN), 1, 0, 0, false, 0)
                        );
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((pingpongFlag));
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag + 2));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag));
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((pingpongFlag + 2));
                        if (splitIdx == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
                            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));
                        }
                        AscendC::Mmad(
                                    l0cBufTensor[splitIdx * qkRoundM * ppNScalar],
                                    l0aBufTensor[offset],
                                    l0bBufTensor[offset],
                                    AscendC::MmadParams(
                                        qkM,  // m
                                        qkN,  // n
                                        __k,   // k
                                        0,
                                        false,
                                        1
                                    )
                        );
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag));
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((pingpongFlag + 2));
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
                    if constexpr (IsSameType<KV_T, int8_t>::value) {
                        float tmp = deqQkGmTensor.GetValue(headIdx);
                        uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
                        AscendC::SetFixpipeNz2ndFlag(1, 0, 0);
                        AscendC::DataCopyCO12DstParams intriParams(svRoundN, qkM, svRoundN, qkRoundM, QuantMode_t::DEQF16, 0, false, true);
                        AscendC::SetFixpipePreQuantFlag(deqScalar);
                        AscendC::PipeBarrier<PIPE_FIX>();
                        AscendC::DataCopy(sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod], l0cBufTensor, intriParams);
                    }
                    else {
                        auto intriParams = AscendC::FixpipeParamsV220(svRoundN, // nSize
                                                      qkM, // mSize
                                                      qkRoundM,   // srcStride
                                                      svRoundN,   // dstStride
                                                      false);      // enRelu
                        if(IsSameType<S_T, half>::value){
                            intriParams.quantPre = QuantMode_t::F322F16;
                        }else{
                            intriParams.quantPre = QuantMode_t::NoQuant;
                        }

                        AscendC::Fixpipe<S_T, TMP_T, AscendC::CFG_ROW_MAJOR>(sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod],
                            l0cBufTensor, intriParams);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(QK_READY);
                }
                if (nIdx >= launchDelay + nStart) {
                    uint32_t l0cPingpongFlag = (nIdx - nStart) % 2;
                    uint32_t l0cOffset = l0cPingpongFlag * L0AB_HALF_BUF_SIZE;
                    uint32_t svNTriu = nEnd * ppNScalar;
                    if (nIdx + sBlockStack > nEnd + launchDelay - 1) {
                        svN = svNTriu > kvSeqlen ? kvSeqlen - (nIdx - launchDelay) * ppNScalar : svNTriu - (nIdx - launchDelay) * ppNScalar;
                    } else {
                        svN = ppNScalar * sBlockStack;
                    }
                    svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
                    if constexpr (IsSameType<KV_T, int8_t>::value) {
                        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                        static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(KV_T);
                        if (strideKv < STRIDE_LIMIT) {
                                AscendC::DataCopy(l1vBufAddrTensor,
                                                vGmTensor[vOffset],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    svN, // nValue
                                                                    __k, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    strideKv,        // srcDValue
                                                                    RoundUp<uint64_t>(svRoundN, 32),   // dstNzC0Stride
                                                                    1,           // dstNzNStride
                                                                    0));         // dstNzMatrixStride, unused
                        } else {
                            for (uint32_t i = 0; i < svN; ++i) {
                                AscendC::DataCopy(l1vBufAddrTensor[i * BLOCK_SIZE_COPY],
                                                vGmTensor[vOffset + i * strideKv],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    1,           // nValue
                                                                    __k, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    0,           // srcDValue
                                                                    RoundUp<uint64_t>(svRoundN, 32),   // dstNzC0Stride
                                                                    0,           // dstNzNStride
                                                                    0)           // dstNzMatrixStride, unused
                                );
                            }
                        }
                    }
                    else {
                        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                        static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(KV_T);
                        if (strideKv < STRIDE_LIMIT) {
                                AscendC::DataCopy(l1vBufAddrTensor,
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
                            for (uint32_t i = 0; i < svN; ++i) {
                                AscendC::DataCopy(l1vBufAddrTensor[i * BLOCK_SIZE_COPY],
                                                vGmTensor[vOffset + i * strideKv],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    1,           // nValue
                                                                    __k, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    0,           // srcDValue
                                                                    svRoundN,   // dstNzC0Stride
                                                                    0,           // dstNzNStride
                                                                    0)           // dstNzMatrixStride, unused
                                );
                            }
                        }
                    }
                    vOffset += svN * strideKv;
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID2));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID2));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
                    if constexpr (IsSameType<KV_T, int8_t>::value) {
                        for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < (svRoundN + 31) / 32 * 32 / BlockSize<int8_t>(); ++l0bLoadIdx) {
                            AscendC::LoadDataWithTranspose(l0bBufTensor[l0bLoadIdx * roundK * BlockSize<int8_t>()],
                                                           l1vBufAddrTensor[l0bLoadIdx * BlockSize<int8_t>() * BlockSize<int8_t>()],
                                                           AscendC::LoadData2dTransposeParams(0,                      // startIndexIn
                                                                                              (roundK + 31) / 32 * 32 / BlockSize<int8_t>(), // repeatTimesIn
                                                                                              (svRoundN + 31) / 32 * 32 / BlockSize<int8_t>(), // srcStrideIn
                                                                                              1,                      // dstGapIn
                                                                                              0,                      // dstfracGapIn
                                                                                              0)                      // addrModeIn
                            );
                        }
                    }
                    else {
                        for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < svRoundN / BLOCK_SIZE; ++l0bLoadIdx) {
                            using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                            static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(KV_T);
                            AscendC::LoadData(
                                l0bBufTensor[l0bLoadIdx * roundK * BLOCK_SIZE],
                                l1vBufAddrTensor[l0bLoadIdx * CUBE_MATRIX_SIZE],
                                AscendC::LoadData2dParams(0, roundK / BLOCK_SIZE, svRoundN / BLOCK_SIZE, 0, 0, true, 0)
                            );
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID6));
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
                    AscendC::WaitEvent(SOFTMAX_READY);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
                    if (qkM == 1) {
                        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                        static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(KV_T);
                        AscendC::DataCopy(l1pBufAddrTensor,
                                                pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod) * 2 / sizeof(KV_T)],
                                                AscendC::DataCopyParams(1,                                              // nBurst
                                                                        CeilDiv<BLOCK_SIZE_COPY>(1 * RoundUp<uint64_t>(svRoundN, BlockSize<KV_T>())), // lenBurst
                                                                        0,                                              // srcGap
                                                                        0));
                    } else {
                        using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                        static constexpr uint32_t BLOCK_SIZE_COPY = HardwareParams::l1l0BlockSize / sizeof(KV_T);
                        if (svRoundN * 2 / sizeof(KV_T) < STRIDE_LIMIT) {
                                AscendC::DataCopy(l1pBufAddrTensor,
                                                pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod) * 2 / sizeof(KV_T)],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    qkM, // nValue
                                                                    svN, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    svRoundN * 2 / sizeof(KV_T),        // srcDValue
                                                                    qkRoundM,   // dstNzC0Stride
                                                                    1,           // dstNzNStride
                                                                    0));         // dstNzMatrixStride, unused
                        } else {
                            for (uint32_t i = 0; i < svN; ++i) {
                                AscendC::DataCopy(l1pBufAddrTensor[i * BLOCK_SIZE_COPY],
                                                pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod) * 2 / sizeof(KV_T) + i * svRoundN * 2 / sizeof(KV_T)],
                                                AscendC::Nd2NzParams(1,           // ndNum
                                                                    1,           // nValue
                                                                    svN, // dValue
                                                                    0,           // srcNdMatrixStride, unused
                                                                    0,           // srcDValue
                                                                    qkRoundM,   // dstNzC0Stride
                                                                    0,           // dstNzNStride
                                                                    0)           // dstNzMatrixStride, unused
                                );
                            }
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID3));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>((EVENT_ID3));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
                    uint32_t roundRow = RoundUp<uint32_t>(RoundUp<uint64_t>(svRoundN, BlockSize<KV_T>()), 32 / sizeof(KV_T));
                    if (qkM == 1) {
                        AscendC::LoadData(l0aBufTensor,
                                            l1pBufAddrTensor,
                                            AscendC::LoadData2dParams(0,           // baseIdx
                                                                    NumMatrixsRoundUp<KV_T>(roundRow),  // repeat
                                                                    1,  // srcStride
                                                                    0,           // sid
                                                                    0,  // dstStride
                                                                    false, // transpose
                                                                    0));         // addrCalMode
                    } else {
                            using HardwareParams = HardwareInfo<ArchType::ASCEND_V220>;
                            static constexpr uint32_t ROW_BLOCK_SIZE = 16;
                            static constexpr uint32_t COL_BLOCK_SIZE = 32 / sizeof(KV_T);
                            static constexpr uint32_t FRACTAL_SIZE = HardwareParams::fractalSize / sizeof(KV_T);
                            static constexpr uint32_t BLOCK_NUM_PER_FRACTAL = HardwareParams::fractalSize / HardwareParams::l1l0BlockSize;
                            for(int32_t i = 0; i < qkRoundM / ROW_BLOCK_SIZE; i++){
                                AscendC::LoadData(l0aBufTensor[i *  ROW_BLOCK_SIZE * roundRow],
                                            l1pBufAddrTensor[i * FRACTAL_SIZE],
                                            AscendC::LoadData2dParams(0,           // baseIdx
                                                                    static_cast<uint16_t>(roundRow / COL_BLOCK_SIZE),  // repeat
                                                                    qkRoundM / ROW_BLOCK_SIZE,  // srcStride
                                                                    0,           // sid
                                                                    0,  // dstStride
                                                                    false, // transpose
                                                                    0));         // addrCalMode
                            }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID5));
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID5));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>((EVENT_ID6));
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((l0cPingpongFlag));
                     if constexpr (IsSameType<KV_T, int8_t>::value) {
                        AscendC::Mmad(
                            l0cBufTensor.template ReinterpretCast<int32_t>()[l0cOffset],
                            l0aBufTensor,
                            l0bBufTensor,
                            AscendC::MmadParams(
                                qkM,  // m
                                __k,  // n
                                svN, // k
                                0,
                                false,
                                1
                            )
                        );
                    } else {
                        AscendC::Mmad(
                            l0cBufTensor[l0cOffset],
                            l0aBufTensor,
                            l0bBufTensor,
                            AscendC::MmadParams(
                                qkM,  // m
                                __k,  // n
                                svN, // k
                                0,
                                false,
                                1
                            )
                        );
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>((l0cPingpongFlag));
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>((l0cPingpongFlag));
                    // copy O to gm
                    auto intriParams = AscendC::FixpipeParamsV220(roundK, // nSize
                                                      qkM, // mSize
                                                      qkRoundM,   // srcStride
                                                      roundK,   // dstStride
                                                      false);      // enRelu

                    intriParams.quantPre = QuantMode_t::NoQuant;
                    AscendC::Fixpipe<TMP_T, TMP_T, AscendC::CFG_ROW_MAJOR>(oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod], l0cBufTensor[l0cOffset], intriParams);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>((l0cPingpongFlag));
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(UPDATE_READY);
                }
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>((EVENT_ID3));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>((EVENT_ID3));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>((EVENT_ID1));

        AscendC::PipeBarrier<PIPE_ALL>();
    }
protected:
    using Q_T = typename PFAT::qDType;
    using KV_T = typename PFAT::kvDType;
    using O_T = typename PFAT::outputDType;
    using U_T = typename PFAT::maskDType;
    using S_T = typename PFAT::sDType;
    using P_T = typename PFAT::pDType;
    using TMP_T = typename PFAT::oTmpDType;
    static constexpr bool swaFlag = PFAT::swaFlag;
    static constexpr bool swaCompress = PFAT::swaCompress;

    GlobalTensor<Q_T> qGmTensor;
    GlobalTensor<Q_T> kGmTensor;
    GlobalTensor<KV_T> vGmTensor;
    GlobalTensor<O_T> oGmTensor;
    GlobalTensor<S_T> sGmTensor;
    GlobalTensor<P_T> pGmTensor;
    GlobalTensor<TMP_T> oTmpGmTensor;
    GlobalTensor<float> deqQkGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    const uint32_t l1qBufAddrOffset = 0;
    const uint32_t l1kBufAddrOffset = 2 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1pBufAddrOffset = 4 * L0AB_UINT8_BLOCK_SIZE;
    const uint32_t l1vBufAddrOffset = 6 * L0AB_UINT8_BLOCK_SIZE;

    LocalTensor<Q_T> l1qBufAddrTensor;
    LocalTensor<Q_T> l1kBufAddrTensor;
    LocalTensor<KV_T> l1vBufAddrTensor;
    LocalTensor<P_T> l1pBufAddrTensor;
    LocalTensor<Q_T> l0aBufTensor;
    LocalTensor<Q_T> l0bBufTensor;
    LocalTensor<TMP_T> l0cBufTensor;
};

#elif __DAV_C220_VEC__
constexpr int32_t VECTOR_SIZE = 128;
constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int32_t FLOAT_BLOCK_SIZE = 8;
constexpr int32_t LONG_SEQ_MASK_LEN = 128;
constexpr int32_t COMPRESS_MASK_SIZE = 8192;
constexpr int32_t UB_UINT8_BLOCK_SIZE = 16384;  // 64 * 128 * 2B
constexpr int32_t UB_HALF_BUF_SIZE = 8192;      // 64 * 128
constexpr int32_t UB_UINT8_LINE_SIZE = 512;     // 128 * 4B
constexpr int32_t UB_FLOAT_LINE_SIZE = 64;     // 128
constexpr int32_t UB_HALF_LINE_SIZE = 128;      // UB_FLOAT_LINE_SIZE * 2
constexpr int32_t TMP_SIZE = 32768 * 4;             // 128 * 256
constexpr int32_t TOTAL_UB_SIZE = 192 * 1024;
constexpr int32_t ROWMAX_TEMP_BUF_OFFSET = 1024;

template<typename PFAT>
class UnpadAttentionDecoderAiv {
public:
    __aicore__ inline UnpadAttentionDecoderAiv() {};

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
        if (len > SIXTEEN_BASE_API || len < ONE_BASE_API) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t) 1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    template<typename T>
    __aicore__ __attribute__((always_inline)) inline uint32_t VectorSize()
    {
        return 256 / sizeof(T);
    }

    template<typename T>
    __aicore__ __attribute__((always_inline)) inline uint64_t NumVectorsRoundUp(uint64_t num)
    {
        return (num + VectorSize<T>() - 1) / VectorSize<T>();
    }

    template <typename T>
    __aicore__ inline void RowMaxRepeatM(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src,
        const AscendC::LocalTensor<T> &tempTensor,
        const uint32_t& subM,
        const uint32_t& qkN,
        const uint32_t& qkRoundN
    )
    {
        uint32_t T_BLOCK_SIZE = 32 / sizeof(T);
        uint32_t T_VECTOR_SIZE = 256 / sizeof(T);
        if (qkN <= T_VECTOR_SIZE) {
            __set_mask(qkN);
            AscendC::WholeReduceMax<T, false>(
                dst,
                src,
                (int32_t)0,
                subM,
                1,
                1,
                qkRoundN / T_BLOCK_SIZE
            );
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        else {
            AscendC::DataCopy(tempTensor, src, AscendC::DataCopyParams(subM, 8, (qkRoundN - T_VECTOR_SIZE) / T_BLOCK_SIZE, 0));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t rowmaxIdx = 1; rowmaxIdx < qkN / T_VECTOR_SIZE; ++rowmaxIdx) {
                AscendC::Max<T, false>(
                    tempTensor,
                    tempTensor,
                    src[rowmaxIdx * T_VECTOR_SIZE],
                    (uint64_t)0,
                    subM,
                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, qkRoundN / T_BLOCK_SIZE)
                );
                AscendC::PipeBarrier<PIPE_V>();
            }
            if (qkN % T_VECTOR_SIZE > 0) {
                __set_mask(qkN % T_VECTOR_SIZE);
                AscendC::Max<T, false>(
                    tempTensor,
                    tempTensor,
                    src[qkN / T_VECTOR_SIZE * T_VECTOR_SIZE],
                    (uint64_t)0,
                    subM,
                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, qkRoundN / T_BLOCK_SIZE)
                );
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::WholeReduceMax<T, false>(
                dst,
                tempTensor,
                (int32_t)0,
                subM,
                1,
                1,
                8
            );
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void MulRepeatM(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src0,
        const AscendC::LocalTensor<T> &src1,
        const uint32_t& subM,
        const uint32_t& subN,
        const uint32_t& subRoundN
    )
    {
        uint32_t T_BLOCK_SIZE = 32 / sizeof(T);
        uint32_t T_VECTOR_SIZE = 256 / sizeof(T);

        for (uint32_t vmulsIdx = 0; vmulsIdx < subN / T_VECTOR_SIZE; ++vmulsIdx) {
            AscendC::Mul<T, false>(
                dst[vmulsIdx * T_VECTOR_SIZE],
                src0[vmulsIdx * T_VECTOR_SIZE],
                src1,
                (uint64_t)0,
                subM,
                AscendC::BinaryRepeatParams(1, 1, 0, subRoundN / T_BLOCK_SIZE, subRoundN / T_BLOCK_SIZE, 1)
            );
        }
        if (subN % FLOAT_VECTOR_SIZE > 0) {
            __set_mask(subN % FLOAT_VECTOR_SIZE);
            AscendC::Mul<T, false>(
                dst[subN / T_VECTOR_SIZE * T_VECTOR_SIZE],
                src0[subN / T_VECTOR_SIZE * T_VECTOR_SIZE],
                src1,
                (uint64_t)0,
                subM,
                AscendC::BinaryRepeatParams(1, 1, 0, subRoundN / T_BLOCK_SIZE, subRoundN / T_BLOCK_SIZE, 1)
            );
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void DivRepeatM(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src0,
        const AscendC::LocalTensor<T> &src1,
        const uint32_t& subM,
        const uint32_t& subN,
        const uint32_t& subRoundN
    )
    {
        uint32_t temp = sizeof(T);
        uint32_t T_BLOCK_SIZE = 32 / sizeof(T);
        uint32_t T_VECTOR_SIZE = 256 / sizeof(T);

        for (uint32_t vdivIdx = 0; vdivIdx < subN / T_VECTOR_SIZE; ++vdivIdx) {
            AscendC::Div<T, false>(
                dst[vdivIdx * T_VECTOR_SIZE],
                src0[vdivIdx * T_VECTOR_SIZE],
                src1,
                (uint64_t)0,
                subM,
                AscendC::BinaryRepeatParams(1, 1, 0, subRoundN / T_BLOCK_SIZE, subRoundN / T_BLOCK_SIZE, 1)
            );
        }
        if (subN % T_VECTOR_SIZE > 0) {
            __set_mask(subN % T_VECTOR_SIZE);
            AscendC::Div<T, false>(
                dst[subN / T_VECTOR_SIZE * T_VECTOR_SIZE],
                src0[subN / T_VECTOR_SIZE * T_VECTOR_SIZE],
                src1,
                (uint64_t)0,
                subM,
                AscendC::BinaryRepeatParams(1, 1, 0, subRoundN / T_BLOCK_SIZE, subRoundN / T_BLOCK_SIZE, 1)
            );
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void SymmetricQuant(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::LocalTensor<T> &src,
        const AscendC::LocalTensor<T> &scaleUbufTensor,
        const AscendC::LocalTensor<T> &tempTensor,
        const AscendC::GlobalTensor<T> &quantPGmTensor,
        const AscendC::LocalTensor<half> &lmUbufTensor,
        const AscendC::LocalTensor<half> &hmUbufTensor,
        const uint32_t& headIdx,
        const uint32_t& quantType,
        const uint32_t& mSplit,
        const uint32_t& roundMSplit,
        const uint32_t& qkN,
        const uint32_t& qkRoundN,
        const uint32_t& nIdx
        )
    {
        if (quantType == quantType_3) {
            if (nIdx == 0){
                AscendC::Duplicate<T>(scaleUbufTensor, (T)((T)1.0 / (T)INT8_MAX_127), roundMSplit);
            }
            else {
                AscendC::Sub(tempTensor.template ReinterpretCast<half>(), lmUbufTensor, hmUbufTensor, mSplit);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(tempTensor, tempTensor.template ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, mSplit);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(tempTensor, tempTensor, mSplit);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(scaleUbufTensor, tempTensor, (T)((T) 1 / (T)INT8_MAX_127), mSplit);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(
                tempTensor,
                scaleUbufTensor,
                roundMSplit / 8, // roundMSplit / 8 is repeate time
                AscendC::BrcbRepeatParams(1, STRIDE_8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            DivRepeatM(dst, src, tempTensor, mSplit, qkN, qkRoundN);
        }
        else {
            float value = quantPGmTensor.GetValue(headIdx);
            AscendC::SetFlag<AscendC::HardEvent::S_V>((EVENT_ID0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((EVENT_ID0));
            AscendC::Muls(dst, src, (float)((float)1.0 / (float)value), roundMSplit * qkRoundN);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    template <typename T>
    __aicore__ inline void SymmetricDeQuant(const AscendC::LocalTensor<float> &loUbufTensor,
                                            const AscendC::LocalTensor<T> &scaleUbufTensor,
                                            const AscendC::LocalTensor<T> &tvUbufTensor,
                                            const AscendC::GlobalTensor<T> &deqPvGmTensor,
                                            const AscendC::GlobalTensor<T> &quantPGmTensor,
                                            uint32_t headIdx, uint32_t subM, uint32_t roundSubM,
                                            uint32_t qkN, uint32_t qkRoundN, uint32_t quantType)
    {
        AscendC::Cast<float, int32_t, false>(
            loUbufTensor,
            loUbufTensor.template ReinterpretCast<int32_t>(),
            AscendC::RoundMode::CAST_NONE,
            (uint64_t)0,
            NumVectorsRoundUp<int32_t>(subM * qkRoundN),
            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
        );
        AscendC::PipeBarrier<PIPE_V>();
        float deqPv = deqPvGmTensor.GetValue(headIdx);
        if (quantType == quantType_3) {
            AscendC::SetFlag<AscendC::HardEvent::S_V>((EVENT_ID0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((EVENT_ID0));
            AscendC::Muls(scaleUbufTensor, scaleUbufTensor, deqPv, subM);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(
                tvUbufTensor,
                scaleUbufTensor,
                roundSubM / 8,
                AscendC::BrcbRepeatParams(1, STRIDE_8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            MulRepeatM<float>(loUbufTensor, loUbufTensor, tvUbufTensor, subM, qkN, qkRoundN);
            AscendC::PipeBarrier<PIPE_V>();
        }
        else {
            float quant_p = quantPGmTensor.GetValue(headIdx);
            float value = deqPv * quant_p;
            AscendC::SetFlag<AscendC::HardEvent::S_V>((EVENT_ID0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>((EVENT_ID0));
            AscendC::Muls<float, false>(
                loUbufTensor,
                loUbufTensor,
                (float)value,
                (uint64_t)0,
                (subM * qkRoundN + VectorSize<float>() - 1) / VectorSize<float>(),
                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
            );
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ __attribute__((always_inline)) inline void Run(
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
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        maskGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(pseShift));
        maskU8GmTensor.SetGlobalBuffer(pseShift);
        oGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOut));

        uint64_t workSize = tiling->promptAttentionBaseApiBaseParams.workSize;
        sGmTensor.SetGlobalBuffer((__gm__ half *)(workspace));
        pGmTensor.SetGlobalBuffer((__gm__ P_T *)(workspace + workSize));
        oTmpGmTensor.SetGlobalBuffer((__gm__ TMP_T *)(workspace + 2 * workSize));

        deqPvGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(deqScale2));
        quantPGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale1));
        logNGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(softmaxLse));
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);

        // LocalTensor
        lsUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, S_T>(lsUbufOffset);
        lpUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, P_T>(lpUbufOffset);
        lpInt8UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(lpUbufOffset);
        lp32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lpUbufOffset);
        ls32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32UbufOffset);
        ls32QuantUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32QuantUbufOffset);
        maskUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(maskUbufOffset);
        maskValueUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(11 * UB_UINT8_BLOCK_SIZE);
        maskU8UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(11 * UB_UINT8_BLOCK_SIZE);
        loUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbufOffset);
        lmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lmUbufOffset);
        hmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hmUbufOffset);
        gmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gmUbufOffset);
        dmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dmUbufOffset);
        llUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(llUbufOffset);
        glUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(glUbufOffset);
        tvUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(tvUbufOffset);
        tv32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(tvUbufOffset);
        tv16UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tvUbufOffset);
        tvU8UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(tvUbufOffset);
        goUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbufOffset);
        scaleUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(scaleUbufOffset);
        logUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(logUbufOffset);

        int32_t subBlockIdx = AscendC::GetSubBlockIdx();
        // tiling value
        uint32_t batchSize = tiling->promptAttentionBaseApiBaseParams.batchSize;
        uint32_t maxSeqlen = tiling->promptAttentionBaseApiBaseParams.maxSeqLen;
        uint32_t qHeads = tiling->promptAttentionBaseApiBaseParams.headNumSize;
        uint32_t embd = tiling->promptAttentionBaseApiBaseParams.headSize;
        uint32_t embdv = tiling->promptAttentionBaseApiBaseParams.embeddingSizeV;
        half tor = static_cast<half>(tiling->promptAttentionBaseApiBaseParams.tor);
        uint32_t headStride = tiling->promptAttentionBaseApiBaseParams.headStride;
        uint32_t maskStride = tiling->promptAttentionBaseApiBaseParams.maskStride;
        uint32_t isTriuMask = tiling->promptAttentionBaseApiBaseParams.isTriuMask;
        uint32_t totalQBlkNum = tiling->promptAttentionBaseApiBaseParams.totalQBlkNum;
        uint32_t isClamp = tiling->promptAttentionBaseApiBaseParams.isClamp;
        half clampMin = 0;
        half clampMax = 0;
        uint32_t tilingHeadSize = (uint32_t)tiling->promptAttentionBaseApiBaseParams.tilingHeadSize;
        uint32_t tilingParaSize = (uint32_t)tiling->promptAttentionBaseApiBaseParams.tilingParaSize;
        uint32_t longSeq = (uint32_t)tiling->promptAttentionBaseApiBaseParams.isLongSeq;
        uint32_t maskType = (uint32_t)tiling->promptAttentionBaseApiBaseParams.maskType;
        uint32_t quantType = (uint32_t)tiling->promptAttentionBaseApiBaseParams.quantType;
        uint32_t dataShapeType = (uint32_t)tiling->promptAttentionBaseApiBaseParams.dataShapeType;
        uint32_t windowSize = 0;

        uint64_t strideQo = qHeads * embd;
        if (dataShapeType == 1) {
            strideQo = embd;
        }

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
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));

        uint32_t __k = embd;
        uint32_t roundK = (__k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        uint32_t curBatch = 0;
        uint32_t qSeqlen = actualSeqLengthsGm.GetValue(curBatch);
        uint32_t kvSeqlen = actualSeqLengthsKVGm.GetValue(curBatch);
        uint32_t ppMScalar = tiling->promptAttentionBaseApiBaseParams.ppMScalar;
        uint32_t ppNScalar = tiling->promptAttentionBaseApiBaseParams.ppNScalar;
        uint64_t addrOScalar = 0;

        uint32_t preTotalQBlkNum = 0;
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
            if (isTriuMask) {
                uint32_t currIter = process / GetBlockNum();
                nextProcess = currIter % 2 == 1 ? (currIter + 1) * GetBlockNum() + GetBlockIdx() : (currIter + 2) * GetBlockNum() - 1 - GetBlockIdx();
            }

            // get tiling args
            if (qSeqlen == 0 || kvSeqlen == 0) {
                continue;
            }
            ScaleType scaleType = (ScaleType)(tiling->promptAttentionBaseApiBaseParams.scaleType);

            uint32_t processIdx = process - preTotalQBlkNum * qHeads;
            uint32_t mIdx = processIdx / qHeads;
            uint64_t headIdx = processIdx % qHeads;

            uint32_t mLoop = (qSeqlen + ppMScalar - 1) / ppMScalar;
            uint32_t nLoop = (kvSeqlen + ppNScalar - 1) / ppNScalar;

            uint32_t qkM = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * ppMScalar) : ppMScalar;
            uint32_t subM = (subBlockIdx == 1) ? (qkM - qkM / 2) : qkM / 2;
            uint32_t subMD128 = (subM + VECTOR_SIZE - 1) / VECTOR_SIZE;             // up aligned to 128
            uint32_t subMD64 = (subM + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;  // up aligned to 64
            uint32_t roundSubM = (subM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            /******** pre_load *******/
            uint32_t qkN = nLoop == 1 ? kvSeqlen : ppNScalar;
            uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint32_t pingpongFlag = 0;
            uint32_t offset = pingpongFlag * UB_HALF_BUF_SIZE;
            uint64_t maskBatchOffset = curBatch * maskStride * maxSeqlen;
            uint64_t maskHeadOffset = headIdx * ((uint64_t)headStride) * maxSeqlen;
            uint64_t maskOffset = maskBatchOffset + maskHeadOffset;
            if (longSeq == 0) {
                maskOffset += mIdx * ppMScalar * maxSeqlen;
            } else {
                AscendC::DataCopy(
                    maskUbufTensor,
                    maskGmTensor[(uint64_t)subBlockIdx * qkM / 2 * VECTOR_SIZE],
                    AscendC::DataCopyParams(
                        subM,                                // nBurst
                        VECTOR_SIZE / BLOCK_SIZE,  // lenBurst
                        0,                                // srcGap
                        0
                    )
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
            }

            uint64_t oOffset = addrOScalar + headIdx * embd + mIdx * ppMScalar * strideQo;
            if (dataShapeType == 1) {
                oOffset = addrOScalar + headIdx * embd * maxSeqlen + mIdx * ppMScalar * strideQo;
            }

            uint32_t nEnd = nLoop;
            if (isTriuMask || windowSize > 0) {
                nEnd = mIdx + 1;
            }
            uint32_t windowStart = (windowSize + ppNScalar - 1) / ppNScalar;
            uint32_t nStart = 0;
            if constexpr (swaFlag) {
                if (windowSize > 0 && windowSize < kvSeqlen) {
                    nStart = (mIdx < windowStart) ? 0 : mIdx - windowStart;
                    if constexpr (!swaCompress) {
                        maskOffset += nStart * ppNScalar;
                    }
                }
            }
            uint32_t qkNTriu = nEnd * ppNScalar;
            uint32_t sBlockStack = nEnd > NO_STACK_S_BLOCK_LIMIT ? 2 : 1;
            uint32_t pvStage = 3;
            uint32_t launchDelay = sBlockStack * 2;
            uint32_t vectMod = 2 * launchDelay;
            uint32_t mSlice = subM > 32 ? 32 : 0; // sBlockStack=2时，UB可以放下
            uint32_t mEnd = subM > 32 ? 2 : 1;

            for (uint32_t nIdx = nStart; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
                if (nIdx < nEnd) {
                    uint32_t pScaleOffset =
                        nIdx / sBlockStack % pvStage * RoundUp<uint32_t>(ppMScalar, FLOAT_VECTOR_SIZE);
                    if (nIdx + sBlockStack > nEnd - 1) {
                        qkN = qkNTriu > kvSeqlen ? kvSeqlen - nIdx * ppNScalar : qkNTriu - nIdx * ppNScalar;
                    } else {
                        qkN = ppNScalar * sBlockStack;
                    }
                    uint32_t deltaIdx = mIdx - nIdx;
                    bool skipMask = windowStart > 3 && deltaIdx > 1 && deltaIdx < windowStart - 1;
                    if constexpr (swaCompress) {
                        maskOffset = 0;
                        if (windowStart <= 3) {    // window < 128*3
                            if (mIdx < nIdx) {
                                maskOffset = ppNScalar;  // swa with midx<nidx, offset = pp_n
                            } else {
                                maskOffset = deltaIdx * maxSeqlen * ppMScalar;
                            }
                        } else {
                            if (deltaIdx == 0) {
                                maskOffset = 0;    // m = n
                            } else if (deltaIdx == windowStart) {
                                maskOffset = 3 * maxSeqlen * ppMScalar;
                            } else if (deltaIdx == 1) {
                                maskOffset = maxSeqlen * ppMScalar;
                            } else if (deltaIdx == windowStart - 1) {
                                maskOffset = 2 * maxSeqlen * ppMScalar;
                            }   // delta idx in [2, window-1) do not move and add mask
                        }
                    }
                    qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

                    if (subM > 0 && maskType != 0 && longSeq == 0) {
                        if (qkN <= ppNScalar) {
                            pingpongFlag = (nIdx - nStart) % 2;
                            offset = pingpongFlag * UB_HALF_BUF_SIZE;
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((pingpongFlag + 2));
                            if constexpr(swaCompress) {
                                if (!skipMask) {
                                    AscendC::DataCopyPad(
                                        maskUbufTensor[offset],
                                        maskGmTensor[maskOffset + (uint64_t)subBlockIdx * qkM / 2 * maxSeqlen],
                                        AscendC::DataCopyExtParams(
                                            subM,                        // nBurst
                                            qkN * 2,                // lenBurst
                                            (maxSeqlen - qkN) * 2, // srcGap
                                            0,                             // dstGap
                                            0),
                                        AscendC::DataCopyPadExtParams<half>(
                                            false,
                                            0,
                                            0,
                                            0)
                                    );
                                }

                            } else {
                                AscendC::DataCopyPad(
                                    maskUbufTensor[offset],
                                    maskGmTensor[maskOffset + (uint64_t)subBlockIdx * qkM / 2 * maxSeqlen],
                                    AscendC::DataCopyExtParams(
                                        subM,                        // nBurst
                                        qkN * 2,                // lenBurst
                                        (maxSeqlen - qkN) * 2, // srcGap
                                        0,                             // dstGap
                                        0),
                                    AscendC::DataCopyPadExtParams<half>(
                                        false,
                                        0,
                                        0,
                                        0)
                                );
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((pingpongFlag + 2));
                        } else {
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                            if constexpr (swaCompress) {
                                if (!skipMask) {
                                    AscendC::DataCopyPad(
                                        maskUbufTensor,
                                        maskGmTensor[maskOffset + (uint64_t)subBlockIdx * qkM / 2 * maxSeqlen],
                                        AscendC::DataCopyExtParams(
                                            subM,                        // nBurst
                                            qkN * 2,                // lenBurst
                                            (maxSeqlen - qkN) * 2, // srcGap
                                            0,                             // dstGap
                                            0),
                                        AscendC::DataCopyPadExtParams<half>(
                                            false,
                                            0,
                                            0,
                                            0)
                                    );
                                }
                            } else {
                                AscendC::DataCopyPad(
                                    maskUbufTensor,
                                    maskGmTensor[maskOffset + (uint64_t)subBlockIdx * qkM / 2 * maxSeqlen],
                                    AscendC::DataCopyExtParams(
                                        subM,                        // nBurst
                                        qkN * 2,                // lenBurst
                                        (maxSeqlen - qkN) * 2, // srcGap
                                        0,                             // dstGap
                                        0),
                                    AscendC::DataCopyPadExtParams<half>(
                                        false,
                                        0,
                                        0,
                                        0)
                                );
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));
                        }
                        if constexpr (!swaCompress) {
                            maskOffset += qkN;
                        }
                    }
                    AscendC::WaitEvent(QK_READY);
                    uint32_t qkNReduceSum = qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE;
                    if (qkN <= VECTOR_SIZE) {
                        pingpongFlag = (nIdx - nStart) % 2;
                        offset = pingpongFlag * UB_HALF_BUF_SIZE;
                        if (subM > 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((pingpongFlag));
                            if (sBlockStack == 2) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((1 - pingpongFlag));
                            }
                            // input QK
                            AscendC::DataCopy(
                                lsUbufTensor[offset],
                                sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                    (uint64_t)subBlockIdx * qkM / 2 * qkRoundN],
                                AscendC::DataCopyParams(
                                    subM,                                // nBurst
                                    qkRoundN / BLOCK_SIZE,  // lenBurst
                                    0,                                // srcGap
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                            if(scaleType == ScaleType::SCALE_LOGN){
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                AscendC::DataCopyPad(
                                    logUbufTensor,
                                    logNGmTensor[mIdx * ppMScalar + (uint64_t)subBlockIdx * qkM / 2 ],
                                    AscendC::DataCopyExtParams(
                                        1,                   // nBurst
                                        subM * 2,                // lenBurst
                                        0, // srcGap byte
                                        (roundSubM - subM) * 2,    // dstGap block
                                        0),
                                    AscendC::DataCopyPadExtParams<half>(
                                        false,
                                        0,
                                        0,
                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::Brcb(
                                    tvUbufTensor.ReinterpretCast<uint16_t>()[VECTOR_SIZE],
                                    logUbufTensor.ReinterpretCast<uint16_t>(),
                                    (subM + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE,
                                    AscendC::BrcbRepeatParams(1, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                for (uint32_t vdivIdx = 0; vdivIdx < qkN / VECTOR_SIZE; ++vdivIdx) {
                                    AscendC::Mul<S_T, false>(
                                        lsUbufTensor[offset + vdivIdx * VECTOR_SIZE],
                                        lsUbufTensor[offset + vdivIdx * VECTOR_SIZE],
                                        tvUbufTensor.ReinterpretCast<half>()[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE,qkRoundN / BLOCK_SIZE, 1)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                if (qkN % VECTOR_SIZE > 0) {
                                    __set_mask(qkN % VECTOR_SIZE);
                                    AscendC::Mul<S_T, false>(
                                        lsUbufTensor[offset + qkN / VECTOR_SIZE * VECTOR_SIZE],
                                        lsUbufTensor[offset + qkN / VECTOR_SIZE * VECTOR_SIZE],
                                        tvUbufTensor.ReinterpretCast<half>()[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE,qkRoundN / BLOCK_SIZE, 1)
                                    );
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Muls<S_T, false>(
                                    lsUbufTensor[offset],
                                    lsUbufTensor[offset],
                                    tor,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                            } else {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                // *** ls = tor * ls
                                AscendC::Muls<S_T, false>(
                                    lsUbufTensor[offset],
                                    lsUbufTensor[offset],
                                    tor,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            if (isClamp == 1){
                                // get min(clampMin，ls_ubuf)
                                AscendC::Maxs<S_T, false>(
                                    lsUbufTensor[offset],
                                    lsUbufTensor[offset],
                                    clampMin,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                // get max(clampMin，ls_ubuf)
                                AscendC::Mins<S_T, false>(
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
                                    if constexpr (swaCompress) {
                                        if (!skipMask) {
                                             AscendC::Add<half, false>(
                                                lsUbufTensor[offset],
                                                lsUbufTensor[offset],
                                                maskUbufTensor[offset],
                                                (uint64_t)0,
                                                (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                            );
                                        }
                                    } else {
                                        AscendC::Add<S_T, false>(
                                            lsUbufTensor[offset],
                                            lsUbufTensor[offset],
                                            maskUbufTensor[offset],
                                            (uint64_t)0,
                                            (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                    }
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((pingpongFlag + 2));
                                } else if (ppNScalar == FLOAT_VECTOR_SIZE && sBlockStack == 2 && nIdx == nEnd - 2) {
                                    __set_mask(qkN - FLOAT_VECTOR_SIZE);
                                    AscendC::Add<S_T, false>(
                                        lsUbufTensor[offset + FLOAT_VECTOR_SIZE],
                                        lsUbufTensor[offset + FLOAT_VECTOR_SIZE],
                                        maskUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, STRIDE_8)
                                    );
                                } else if (nIdx == nEnd - 1) {
                                    __set_mask(qkN);
                                    AscendC::Add<S_T, false>(
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
                                AscendC::BlockReduceMax<S_T, false>(
                                    tvUbufTensor.ReinterpretCast<half>(),
                                    lsUbufTensor[offset],
                                    subM,
                                    0,
                                    2,
                                    1,
                                    qkRoundN / BLOCK_SIZE
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                __set_vcg_mask(qkRoundN / BLOCK_SIZE);
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
                                AscendC::BlockReduceMax<S_T, false>(
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
                                AscendC::BlockReduceMax<S_T, false>(
                                    tvUbufTensor.ReinterpretCast<half>()[ROWMAX_TEMP_BUF_OFFSET],
                                    lsUbufTensor[offset + VECTOR_SIZE],
                                    subM,
                                    0,
                                    2,
                                    1,
                                    qkRoundN / BLOCK_SIZE
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                __set_vcg_mask((qkRoundN - VECTOR_SIZE) / BLOCK_SIZE);
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
                                __set_vcg_mask(VECTOR_SIZE / BLOCK_SIZE);
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
                            if (nIdx == nStart) {
                                // *** hm = lm
                                AscendC::DataCopy(hmUbufTensor,
                                                lmUbufTensor,
                                                AscendC::DataCopyParams(1,
                                                                        roundSubM / BLOCK_SIZE,
                                                                        0,
                                                                        0 )
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
                                    dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_HALF_LINE_SIZE],
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
                                                                        0 )
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
                                AscendC::Sub<S_T, false>(
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
                                AscendC::Sub<S_T, false>(
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
                            AscendC::Cast<float, S_T, false>(
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
                            if constexpr (IsSameType<KV_T, int8_t>::value) {
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
                                SymmetricQuant(ls32QuantUbufTensor, ls32UbufTensor, scaleUbufTensor[pScaleOffset], tv32UbufTensor,
                                                quantPGmTensor, lmUbufTensor, hmUbufTensor, headIdx, quantType, subM, roundSubM, qkN, qkRoundN, nIdx);
                                AscendC::Cast<half, float, false>(lpUbufTensor.template ReinterpretCast<half>()[offset], ls32QuantUbufTensor.template ReinterpretCast<float>(),
                                                                AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                                                                (subM * qkRoundN + VectorSize<float>() - 1) / VectorSize<float>(), {1, 1, 4, 8});
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
                                for (uint32_t rowIdx = 0; rowIdx < qkN / VectorSize<half>(); ++rowIdx) {
                                    AscendC::Cast<int8_t, half, false>(lpUbufTensor.template ReinterpretCast<int8_t>()[offset * 2 + rowIdx * VectorSize<half>()],
                                                                    lpUbufTensor.template ReinterpretCast<half>()[offset + rowIdx * VectorSize<half>()], AscendC::RoundMode::CAST_RINT,
                                                                    (uint64_t)0, subM, {1, 1, (uint8_t)(qkRoundN / BlockSize<half>()), (uint8_t)(qkRoundN / BlockSize<half>())});
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                if (qkN % VectorSize<half>() > 0) {
                                    __set_mask(qkN % VectorSize<half>());
                                    AscendC::Cast<int8_t, half, false>(lpUbufTensor.template ReinterpretCast<int8_t>()[offset * 2 + qkN / VectorSize<half>() * VectorSize<half>()],
                                                                    lpUbufTensor.template ReinterpretCast<half>()[offset + qkN / VectorSize<half>() * VectorSize<half>()], AscendC::RoundMode::CAST_RINT,
                                                                    (uint64_t)0, subM, {1, 1, (uint8_t)(qkRoundN / BlockSize<half>()), (uint8_t)(qkRoundN / BlockSize<half>())});
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                            else {
                                // *** lp = castfp32to16(ls)
                                AscendC::Cast<KV_T, float, false>(
                                    lpUbufTensor[offset],
                                    ls32UbufTensor,
                                    AscendC::RoundMode::CAST_NONE,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                );
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));

                            // *** ll = rowsum(ls32)
                            if (qkN <= FLOAT_VECTOR_SIZE) {
                                __set_mask(qkN);
                                AscendC::RepeatReduceSum<float, false>(
                                    llUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],
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
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                 AscendC::RepeatReduceSum<float, false>(
                                    llUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],
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
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            if constexpr (IsSameType<KV_T, int8_t>::value) {
                                AscendC::DataCopy(
                                    pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                (uint64_t)subBlockIdx * qkM / 2 * qkRoundN) * 2 / sizeof(KV_T)],
                                    lpUbufTensor.template ReinterpretCast<KV_T>()[offset * 2],
                                    AscendC::DataCopyParams(
                                        1,                                            // nBurst
                                        subM * qkRoundN * 2 / BlockSize<int8_t>(), // lenBurst
                                        0, // srcGap
                                        0  // dstGap
                                    )
                                );
                            }
                            else {
                                AscendC::DataCopy(
                                    pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                        (uint64_t)subBlockIdx * qkM / 2 * qkRoundN],
                                    lpUbufTensor[offset],
                                    AscendC::DataCopyParams(
                                        1,                                            // nBurst
                                        subM * qkRoundN / BLOCK_SIZE, // lenBurst
                                        0, // srcGap
                                        0  // dstGap
                                    )
                                );
                            }
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
                            AscendC::DataCopy(
                                lsUbufTensor,
                                sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                    (uint64_t)subBlockIdx * qkM / 2 * qkRoundN],
                                AscendC::DataCopyParams(
                                    mSlice,                                // nBurst
                                    qkRoundN / BLOCK_SIZE,  // lenBurst
                                    0,                                // srcGap
                                    0
                                )
                            );
                            if (subM > mSlice) {
                                if (mEnd > 1) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
                                }
                                AscendC::DataCopy(
                                    lsUbufTensor[mSlice * qkRoundN],
                                    sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                        (uint64_t)subBlockIdx * qkM / 2 * qkRoundN + mSlice * qkRoundN],
                                    AscendC::DataCopyParams(
                                        subM - mSlice,                                // nBurst
                                        qkRoundN / BLOCK_SIZE,  // lenBurst
                                        0,                                // srcGap
                                        0
                                    )
                                );
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                            if(scaleType == ScaleType::SCALE_LOGN){
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                AscendC::DataCopyPad(
                                    logUbufTensor,
                                    logNGmTensor[mIdx * ppMScalar + (uint64_t)subBlockIdx * qkM / 2 ],
                                    AscendC::DataCopyExtParams(
                                        1,                   // nBurst
                                        subM * 2,                // lenBurst
                                        0, // srcGap byte
                                        (roundSubM - subM) * 2,    // dstGap block
                                        0),
                                    AscendC::DataCopyPadExtParams<half>(
                                        false,
                                        0,
                                        0,
                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::Brcb(
                                    tvUbufTensor.ReinterpretCast<uint16_t>()[VECTOR_SIZE],
                                    logUbufTensor.ReinterpretCast<uint16_t>(),
                                    (subM + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE,
                                    AscendC::BrcbRepeatParams(1, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                for (uint32_t vdivIdx = 0; vdivIdx < qkN / VECTOR_SIZE; ++vdivIdx) {
                                    AscendC::Mul<half, false>(
                                        lsUbufTensor[vdivIdx * VECTOR_SIZE],
                                        lsUbufTensor[vdivIdx * VECTOR_SIZE],
                                        tvUbufTensor.ReinterpretCast<half>()[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE,qkRoundN / BLOCK_SIZE, 1)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                if (qkN % VECTOR_SIZE > 0) {
                                    __set_mask(qkN % VECTOR_SIZE);
                                    AscendC::Mul<half, false>(
                                        lsUbufTensor[qkN / VECTOR_SIZE * VECTOR_SIZE],
                                        lsUbufTensor[qkN / VECTOR_SIZE * VECTOR_SIZE],
                                        tvUbufTensor.ReinterpretCast<half>()[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BLOCK_SIZE,qkRoundN / BLOCK_SIZE, 1)
                                    );
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Muls<half, false>(
                                    lsUbufTensor,
                                    lsUbufTensor,
                                    tor,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                            } else {
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
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            if (maskType != 0) {
                                if (longSeq == 0) {
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));
                                    if constexpr(swaCompress) {
                                        if (!skipMask) {
                                            AscendC::Add<half, false>(
                                                lsUbufTensor,
                                                lsUbufTensor,
                                                maskUbufTensor,
                                                (uint64_t)0,
                                                (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                            );
                                        }
                                    } else {
                                        AscendC::Add<half, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            maskUbufTensor,
                                            (uint64_t)0,
                                            (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                    }
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                                } else if (nIdx == nEnd - 2) {
                                    __set_mask(qkN - ppNScalar);
                                    AscendC::Add<half, false>(
                                        lsUbufTensor[ppNScalar],
                                        lsUbufTensor[ppNScalar],
                                        maskUbufTensor,
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 1, qkRoundN / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, 8)
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
                                                AscendC::DataCopyParams(subM,                                    // nBurst
                                                                        VECTOR_SIZE / BLOCK_SIZE,                 // lenBurst
                                                                        (qkRoundN - VECTOR_SIZE) / BLOCK_SIZE,  // srcGap
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
                                    AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, qkRoundN / BLOCK_SIZE )
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
                                __set_vcg_mask(VECTOR_SIZE / BLOCK_SIZE);
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
                            if (nIdx == nStart) {
                                // *** hm = lm
                                AscendC::DataCopy( gmUbufTensor,
                                                lmUbufTensor,
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundSubM / BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
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
                                    dmUbufTensor[((nIdx - nStart) / sBlockStack) % launchDelay * UB_HALF_LINE_SIZE],
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
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundSubM / BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
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
                                uint32_t roundMSplit = (mSplit + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
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
                                if constexpr (IsSameType<KV_T, int8_t>::value) {
                                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
                                    SymmetricQuant(ls32QuantUbufTensor, ls32UbufTensor, scaleUbufTensor[pScaleOffset + splitIdx * mSlice], tv32UbufTensor,
                                                    quantPGmTensor, lmUbufTensor[splitIdx * mSlice], hmUbufTensor[splitIdx * mSlice], headIdx, quantType, mSplit, roundMSplit, qkN, qkRoundN, nIdx);

                                    AscendC::Cast<half, float, false>(lpUbufTensor.template ReinterpretCast<half>()[splitIdx * mSlice * qkRoundN], ls32QuantUbufTensor,
                                                                AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                                                                (mSplit * qkRoundN + VectorSize<float>() - 1) / VectorSize<float>(), {1, 1, 4, 8});
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
                                    for (uint32_t rowIdx = 0; rowIdx < qkN / VectorSize<half>(); ++rowIdx) {
                                        AscendC::Cast<int8_t, half, false>(lpUbufTensor.template ReinterpretCast<int8_t>()[(splitIdx * mSlice * qkRoundN) * 2 + rowIdx * VectorSize<half>()],
                                                                        lpUbufTensor.template ReinterpretCast<half>()[splitIdx * mSlice * qkRoundN + rowIdx * VectorSize<half>()], AscendC::RoundMode::CAST_RINT,
                                                                        (uint64_t)0, mSplit, {1, 1, (uint8_t)(qkRoundN / BlockSize<half>()), (uint8_t)(qkRoundN / BlockSize<half>())});
                                    }
                                    if (qkN % VectorSize<half>() > 0) {
                                        __set_mask(qkN % VectorSize<half>());
                                        AscendC::Cast<int8_t, half, false>(lpUbufTensor.template ReinterpretCast<int8_t>()[(splitIdx * mSlice * qkRoundN) * 2 + qkN / VectorSize<half>() * VectorSize<half>()],
                                                                        lpUbufTensor.template ReinterpretCast<half>()[splitIdx * mSlice * qkRoundN + qkN / VectorSize<half>() * VectorSize<half>()], AscendC::RoundMode::CAST_RINT,
                                                                        (uint64_t)0, mSplit, {1, 1, (uint8_t)(qkRoundN / BlockSize<half>()), (uint8_t)(qkRoundN / BlockSize<half>())});
                                        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                    }
                                }
                                else {
                                    // *** lp = castfp32to16(ls)
                                    AscendC::Cast<KV_T, float, false>(
                                        lpUbufTensor[splitIdx * mSlice * qkRoundN],
                                        ls32UbufTensor,
                                        AscendC::RoundMode::CAST_NONE,
                                        (uint64_t)0,
                                        (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                                // *** ll = rowsum(ls32)
                                for (uint32_t rowSumIdx = 1; rowSumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                                    AscendC::Add<float,false>(
                                        ls32UbufTensor,
                                        ls32UbufTensor,
                                        ls32UbufTensor[rowSumIdx * FLOAT_VECTOR_SIZE],
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
                                    llUbufTensor[((nIdx - nStart) / sBlockStack) % launchDelay * UB_FLOAT_LINE_SIZE + splitIdx * mSlice],
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
                                if constexpr (IsSameType<KV_T, int8_t>::value) {
                                    AscendC::DataCopy(
                                        pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE + nIdx % vectMod * TMP_SIZE / vectMod +
                                                    ((uint64_t)subBlockIdx * qkM / 2 + splitIdx * mSlice) * qkRoundN) * 2 / sizeof(KV_T)],
                                        lpUbufTensor.template ReinterpretCast<KV_T>()[(splitIdx * mSlice * qkRoundN) * 2],
                                        AscendC::DataCopyParams(
                                            mSplit,                              // nBurst
                                            qkRoundN * 2 / BlockSize<int8_t>(), // lenBurst
                                            0,                                  // srcGap
                                            0                                   // dstGap
                                        )
                                    );
                                }
                                else {
                                    AscendC::DataCopy(
                                        pGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                            ((uint64_t)subBlockIdx * qkM / 2 + splitIdx * mSlice) * qkRoundN],
                                        lpUbufTensor[splitIdx * mSlice * qkRoundN],
                                        AscendC::DataCopyParams(
                                            mSplit,                              // nBurst
                                            qkRoundN / BLOCK_SIZE,  // lenBurst
                                            0,                                  // srcGap
                                            0                                   // dstGap
                                        )
                                    );
                                }
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((splitIdx));
                            }
                        }
                    }
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SOFTMAX_READY);
                }
                if (nIdx >= launchDelay + nStart) {
                    uint32_t pScaleOffset = (nIdx - launchDelay) / sBlockStack % pvStage * RoundUp<uint32_t>(ppMScalar, FLOAT_VECTOR_SIZE);
                    AscendC::CrossCoreWaitFlag(UPDATE_READY);
                    if (subM > 0) {
                        // *** 更新 L 和 O
                        if (nIdx != launchDelay + nStart) {
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
                            AscendC::DataCopy(
                                loUbufTensor.template ReinterpretCast<TMP_T>(),
                                oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod +
                                    (uint64_t)subBlockIdx * qkM / 2 * roundK],
                                AscendC::DataCopyParams(
                                    1,                                   // nBurst
                                    subM * roundK / FLOAT_BLOCK_SIZE,  // lenBurst
                                    0,                                // srcGap
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID4));
                            // *** dm32 = castfp16to32(dm), 存放于 tv
                            AscendC::Cast<float, half, false>(
                                tvUbufTensor,
                                dmUbufTensor[((nIdx - launchDelay - nStart) / sBlockStack % 4)  * UB_HALF_LINE_SIZE],
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
                                llUbufTensor[((nIdx - launchDelay - nStart) / sBlockStack % 4) * UB_FLOAT_LINE_SIZE],
                                (uint64_t)0,
                                subMD64,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** go = go * dm_block
                            for (uint32_t vmulIdx = 0; vmulIdx < __k / FLOAT_VECTOR_SIZE; ++vmulIdx) {
                                AscendC::Mul<float, false>(
                                    goUbufTensor[vmulIdx * FLOAT_VECTOR_SIZE],
                                    goUbufTensor[vmulIdx * FLOAT_VECTOR_SIZE],
                                    tvUbufTensor[VECTOR_SIZE],
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                                );
                            }
                            if (__k % FLOAT_VECTOR_SIZE > 0) {
                                __set_mask(__k % FLOAT_VECTOR_SIZE);
                                AscendC::Mul<float, false>(
                                    goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                    goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                    tvUbufTensor[VECTOR_SIZE],
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID4));

                            if constexpr (IsSameType<KV_T, int8_t>::value) {
                                SymmetricDeQuant(loUbufTensor, scaleUbufTensor[pScaleOffset], tvUbufTensor, deqPvGmTensor,
                                                quantPGmTensor, headIdx, subM, roundSubM, __k, roundK, quantType);
                            }
                            // *** go = lo + go
                            AscendC::Add<float, false>(
                                goUbufTensor,
                                goUbufTensor,
                                loUbufTensor,
                                (uint64_t)0,
                                (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID4));
                        } else {
                            // *** gl = ll
                            AscendC::DataCopy(glUbufTensor,
                                            llUbufTensor[((nIdx - launchDelay - nStart) / sBlockStack % 4) * UB_FLOAT_LINE_SIZE],
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundSubM / FLOAT_BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
                                                                        0)
                                );
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                            AscendC::DataCopy(
                                goUbufTensor.template ReinterpretCast<TMP_T>(),
                                oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod +
                                    (uint64_t)subBlockIdx * qkM / 2 * roundK],
                                AscendC::DataCopyParams(
                                    1,                                   // nBurst
                                    subM * roundK / FLOAT_BLOCK_SIZE,  // lenBurst
                                    0,                                // srcGap
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID5));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID5));
                            AscendC::PipeBarrier<PIPE_V>();
                            if constexpr (IsSameType<KV_T, int8_t>::value) {
                                SymmetricDeQuant(goUbufTensor, scaleUbufTensor[pScaleOffset], tvUbufTensor, deqPvGmTensor,
                                                quantPGmTensor, headIdx, subM, roundSubM, __k, roundK, quantType);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        if (nIdx + sBlockStack > nEnd + launchDelay - 1) {
                            // *** gl = castfp32to16(gl)
                            AscendC::Cast<half, float, false>(
                                glUbufTensor.ReinterpretCast<half>(),
                                glUbufTensor,
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                subMD64,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** go = castfp32to16(go)
                            AscendC::Cast<half, float, false>(
                                goUbufTensor.ReinterpretCast<half>(),
                                goUbufTensor,
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
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
                            for (uint32_t vdivIdx = 0; vdivIdx < __k / VECTOR_SIZE; ++vdivIdx) {
                                AscendC::Div<half, false>(
                                    goUbufTensor.ReinterpretCast<half>()[vdivIdx * VECTOR_SIZE],
                                    goUbufTensor.ReinterpretCast<half>()[vdivIdx * VECTOR_SIZE],
                                    tvUbufTensor.ReinterpretCast<half>(),
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, roundK / BLOCK_SIZE, roundK / BLOCK_SIZE, 1)
                                );
                            }
                            if (__k % VECTOR_SIZE > 0) {
                                __set_mask(__k % VECTOR_SIZE);
                                AscendC::Div<half, false>(
                                    goUbufTensor.ReinterpretCast<half>()[__k / VECTOR_SIZE * VECTOR_SIZE],
                                    goUbufTensor.ReinterpretCast<half>()[__k / VECTOR_SIZE * VECTOR_SIZE],
                                    tvUbufTensor.ReinterpretCast<half>(),
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, roundK / BLOCK_SIZE, roundK / BLOCK_SIZE, 1)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            // ********************* move O to GM ************************
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID1));
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID1));
                            AscendC::DataCopyPad(
                                oGmTensor[oOffset + (uint64_t)subBlockIdx * qkM / 2 * strideQo],
                                goUbufTensor.ReinterpretCast<half>(),
                                AscendC::DataCopyExtParams(
                                    subM,                 // nBurst
                                    __k * 2,               // lenBurst
                                    0,
                                    (strideQo - __k) * 2,
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
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
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));

        AscendC::PipeBarrier<PIPE_ALL>();
    }
protected:
    using Q_T = typename PFAT::qDType;
    using KV_T = typename PFAT::kvDType;
    using O_T = typename PFAT::outputDType;
    using U_T = typename PFAT::maskDType;
    using S_T = typename PFAT::sDType;
    using P_T = typename PFAT::pDType;
    using TMP_T = typename PFAT::oTmpDType;
    static constexpr bool swaFlag = PFAT::swaFlag;
    static constexpr bool swaCompress = PFAT::swaCompress;

    GlobalTensor<half> oGmTensor;
    GlobalTensor<uint8_t> maskU8GmTensor;
    GlobalTensor<half> maskGmTensor;
    GlobalTensor<half> sGmTensor;
    GlobalTensor<P_T> pGmTensor;
    GlobalTensor<TMP_T> oTmpGmTensor;

    GlobalTensor<float> deqPvGmTensor;
    GlobalTensor<float> quantPGmTensor;
    GlobalTensor<half> logNGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

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
    const uint32_t scaleUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 18 * UB_UINT8_LINE_SIZE;
    const uint32_t logUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 30 * UB_UINT8_LINE_SIZE;
    const uint32_t tvUbufOffset = 11 * UB_UINT8_BLOCK_SIZE;
    const uint32_t goUbufOffset = 9 * UB_UINT8_BLOCK_SIZE;
    const uint32_t ls32QuantUbufOffset = 6 * UB_UINT8_BLOCK_SIZE;

    LocalTensor<S_T> lsUbufTensor;
    LocalTensor<P_T> lpUbufTensor;
    LocalTensor<int8_t> lpInt8UbufTensor;
    LocalTensor<float> lp32UbufTensor;
    LocalTensor<float> ls32UbufTensor;
    LocalTensor<float> ls32QuantUbufTensor;
    LocalTensor<half> maskUbufTensor;
    LocalTensor<half> maskValueUbufTensor;
    LocalTensor<uint8_t> maskU8UbufTensor;
    LocalTensor<float> loUbufTensor;
    LocalTensor<half> lmUbufTensor;
    LocalTensor<half> hmUbufTensor;
    LocalTensor<half> gmUbufTensor;
    LocalTensor<half> dmUbufTensor;
    LocalTensor<float> llUbufTensor;
    LocalTensor<float> glUbufTensor;
    LocalTensor<float> tvUbufTensor;
    LocalTensor<float> tv32UbufTensor;
    LocalTensor<half> tv16UbufTensor;
    LocalTensor<uint8_t> tvU8UbufTensor;
    LocalTensor<float> goUbufTensor;
    LocalTensor<float> scaleUbufTensor;
    LocalTensor<half> logUbufTensor;
};

#include "prompt_flash_attention_base_common.h"

constexpr int32_t BASE_MASK_SIZE = 128;

__aicore__ __attribute__((always_inline)) inline void __set_mask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % FLOAT_VECTOR_SIZE;
    
    for (int64_t i = 0; i < temp; i++) 
    {
        mask |= one << i;
    }
    if (len == VECTOR_SIZE || len == 0) 
    {
       AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= FLOAT_VECTOR_SIZE) 
    {
       AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else 
    {
       AscendC::SetVectorMask<int8_t>(0x0, mask);
    }
}

__aicore__ __attribute__((always_inline)) inline void __set_vcg_mask(int32_t len)
{
    if (len > MAX_ALLOWED_LENGTH_BASE_API || len < MIN_ALLOWED_LENGTH_BASE_API) {
       AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t) 1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask +
                            (subMask << 56) + (subMask << 40) + (subMask << 24) + (subMask << 8);
   AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
}

template <typename PFAT>
class FlashAttentionEncoderHighPrecisionVec {
public:
    __aicore__ __attribute__((always_inline)) inline FlashAttentionEncoderHighPrecisionVec(
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
        maskGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ U_T *>(pseShift));
        oGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ O_T *>(attentionOut));
        sGmTensor.SetGlobalBuffer((__gm__ float *)(workspace));
        pGmTensor.SetGlobalBuffer((__gm__ P_T *)(workspace + workSize));
        oTmpGmTensor.SetGlobalBuffer((__gm__ TMP_T *)(workspace + 2 * workSize));
        logNGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ U_T *>(softmaxLse));
        logNFloatGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmaxLse));
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tiling->promptAttentionBaseApiBaseParams.batchSize);
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tiling->promptAttentionBaseApiBaseParams.batchSize);
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
        this->windowSize = 0;

        this->strideQo = qHeads * embd;
        if (this->dataShapeType == 1) {
            this->strideQo = embd;
        }

        this->__k = embd;
        this->roundK = (__k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        this->scaleType = (ScaleType)(tiling->promptAttentionBaseApiBaseParams.scaleType);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
    }
    __aicore__ __attribute__((always_inline)) inline ~FlashAttentionEncoderHighPrecisionVec()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>((EVENT_ID0));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template<typename Dtype>
    __aicore__ __attribute__((always_inline)) inline uint32_t VectorSize()
    {
        return 256 / sizeof(Dtype);
    }

    template<typename Dtype>
    __aicore__ __attribute__((always_inline)) inline uint64_t NumVectorsRoundUp(uint64_t num)
    {
        return (num + VectorSize<Dtype>() - 1) / VectorSize<Dtype>();
    }

    __aicore__ __attribute__((always_inline)) inline void DeqPerHeadS322F32(AscendC::LocalTensor<float> &s,
                                                                            __gm__ uint8_t *deqQkGm,
                                                                            __gm__ uint8_t *offQkGm,
                                                                            uint32_t headIdx, uint32_t len)
    {
        // dequant QK
        // int32_t转成float类型
        AscendC::Cast<float, int32_t, false>(
                                s, s.ReinterpretCast<int32_t>(),
                                AscendC::RoundMode::CAST_NONE,
                                (uint64_t)0,
                                NumVectorsRoundUp<int32_t>(len),
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
        AscendC::PipeBarrier<PIPE_V>();

        // scale
        float s_quant_scale = *((__gm__ float *)deqQkGm + headIdx);
        AscendC::Muls<float, false>(
                                    s, s, s_quant_scale,
                                    (uint64_t)0,
                                    NumVectorsRoundUp<float>(len),
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void DivRepeatM(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src0,
                                      const AscendC::LocalTensor<T> &src1, const uint32_t subM, const uint32_t qkN,
                                      const uint32_t qkRoundN)
    {
        uint32_t T_BLOCK_SIZE = BlockSize<T>();
        uint32_t T_VECTOR_SIZE = VectorSize<T>();
        for (uint32_t rowIdx = 0; rowIdx < qkN / T_VECTOR_SIZE; ++rowIdx) {
            AscendC::Div<T, false>(
                                    dst[rowIdx * T_VECTOR_SIZE], src0[rowIdx * T_VECTOR_SIZE], src1,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / T_BLOCK_SIZE, qkRoundN / T_BLOCK_SIZE, 1)
            );
        }
        if (qkN % T_VECTOR_SIZE > 0) {
            __set_mask(qkN % T_VECTOR_SIZE);
            AscendC::Div<T, false>(
                                    dst[qkN / T_VECTOR_SIZE * T_VECTOR_SIZE],
                                    src0[qkN / T_VECTOR_SIZE * T_VECTOR_SIZE], src1,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / T_BLOCK_SIZE, qkRoundN / T_BLOCK_SIZE, 1)
            );
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SymmetricDeQuant(const AscendC::LocalTensor<float> &loUbufTensor,
                                            const AscendC::LocalTensor<float> &pScaleUbufTensor, uint32_t subM,
                                            uint32_t roundSubM, uint32_t qkN, uint32_t qkRoundN, uint32_t headIdx)
    {
        if (quantType == quantType_3) {
            AscendC::Brcb(
                                tvUbufTensor, pScaleUbufTensor,
                                roundSubM / BlockSize<float>(),
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                        );
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t rowIdx = 0; rowIdx < qkN / VectorSize<float>(); ++rowIdx) {
                AscendC::Mul<float, false>(
                    loUbufTensor[rowIdx * VectorSize<float>()],
                    loUbufTensor[rowIdx * VectorSize<float>()],
                    tvUbufTensor,
                    (uint64_t)0,
                    subM,
                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BlockSize<float>(), qkRoundN / BlockSize<float>(), 1)
                );
            }
            if (qkN % VectorSize<float>() > 0) {
                __set_mask(qkN % VectorSize<float>());
                AscendC::Mul<float, false>(
                    loUbufTensor[qkN / VectorSize<float>() * VectorSize<float>()],
                    loUbufTensor[qkN / VectorSize<float>() * VectorSize<float>()],
                    tvUbufTensor,
                    (uint64_t)0,
                    subM,
                    AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / BlockSize<float>(), qkRoundN / BlockSize<float>(), 1)
                );

                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            float pScale = *((__gm__ float *)quantPGm + headIdx);
            AscendC::Muls<float, false>(
                                    loUbufTensor, loUbufTensor, pScale,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VectorSize<float>() - 1) / VectorSize<float>(),
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void SymmetricQuant(const AscendC::LocalTensor<float> &lpUbufTensor,
                                          const AscendC::LocalTensor<float> &ls32UbufTensor,
                                          const AscendC::LocalTensor<float> &lmUbufTensor,
                                          const AscendC::LocalTensor<float> &hmUbufTensor,
                                          const AscendC::LocalTensor<float> &pScaleUbufTensor, const uint32_t subM,
                                          const uint32_t roundSubM, const uint32_t qkN, const uint32_t qkRoundN,
                                          uint32_t headIdx)
    {
        // online quant
        if (quantType == quantType_3) {
            AscendC::Sub<float, false>(
                                    lmUbufTensor, lmUbufTensor, hmUbufTensor,
                                    (uint64_t)0,
                                    1,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp<float, false>(
                                lmUbufTensor,
                                lmUbufTensor,
                                (uint64_t)0,
                                1,
                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                            );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls<float, false>(
                                    pScaleUbufTensor, lmUbufTensor, ((float)1 / (float)INT8_MAX_127),
                                    (uint64_t)0,
                                    1,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(
                                tvUbufTensor, pScaleUbufTensor,
                                roundSubM / BlockSize<float>(),
                                AscendC::BrcbRepeatParams(1, STRIDE_8)
                        );
            AscendC::PipeBarrier<PIPE_V>();
            DivRepeatM<float>(lpUbufTensor, ls32UbufTensor, tvUbufTensor, subM, qkN, qkRoundN);
        } else {  // offline quant
            float pScale = (float)1.0 / *((__gm__ float *)quantPGm + headIdx);
            AscendC::Muls<float, false>(
                                    lpUbufTensor, ls32UbufTensor, pScale,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VectorSize<float>() - 1) / VectorSize<float>(),
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Cast<half, float, false>(lpUbufTensor.ReinterpretCast<half>(), lpUbufTensor,
                                          AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                                          (subM * qkRoundN + VectorSize<float>() - 1) / VectorSize<float>(), {1, 1, STRIDE_4, STRIDE_8});
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t rowIdx = 0; rowIdx < qkN / VectorSize<half>(); ++rowIdx) {
            AscendC::Cast<int8_t, half, false>(lpUbufTensor.ReinterpretCast<int8_t>()[rowIdx * VectorSize<half>()],
                                               lpUbufTensor.ReinterpretCast<half>()[rowIdx * VectorSize<half>()], AscendC::RoundMode::CAST_RINT,
                                               (uint64_t)0, subM, {1, 1, (uint8_t)(qkRoundN / BlockSize<half>()), (uint8_t)(qkRoundN / BlockSize<half>())});
        }
        if (qkN % VectorSize<half>() > 0) {
            __set_mask(qkN % VectorSize<half>());
            AscendC::Cast<int8_t, half, false>(lpUbufTensor.ReinterpretCast<int8_t>()[qkN / VectorSize<half>() * VectorSize<half>()],
                                               lpUbufTensor.ReinterpretCast<half>()[qkN / VectorSize<half>() * VectorSize<half>()], AscendC::RoundMode::CAST_RINT,
                                               (uint64_t)0, subM, {1, 1, (uint8_t)(qkRoundN / BlockSize<half>()), (uint8_t)(qkRoundN / BlockSize<half>())});
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
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
        float alibiCoeff = 1;

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
            if (isTriuMask) {
                uint32_t currIter = process / GetBlockNum();
                nextProcess = currIter % PARITY_CHECK_BASE_KER_BASE_API == 1 ? (currIter + 1) * GetBlockNum() + GetBlockIdx() : (currIter + PARITY_CHECK_BASE_KER_BASE_API) * GetBlockNum() - 1 - GetBlockIdx();
            }

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

            /******** pre_load *******/
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
                AscendC::DataCopy(
                    mask16UbufTensor,
                    maskGmTensor[(uint64_t)subBlockIdx * qkM / 2 * VECTOR_SIZE],  // 2 is the CV_RATIO
                    AscendC::DataCopyParams(
                        subM,                                // nBurst
                        VECTOR_SIZE / BLOCK_SIZE,  // lenBurst
                        0,                                // srcGap
                        0
                    )
                );
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                AscendC::Cast<float, U_T, false>(
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
                    maskUbufTensor, (float)-3e38,
                    (uint64_t)0,
                    subM * VECTOR_SIZE / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                );
                AscendC::PipeBarrier<PIPE_V>();
            }

            uint64_t oOffset = addrOScalar + headIdx * embd + mIdx * ppMScalar * strideQo;
            if (dataShapeType == 1) {
                oOffset = addrOScalar + headIdx * embd * maxSeqlen + mIdx * ppMScalar * strideQo;
            }

            uint32_t nEnd = nLoop;
            if (isTriuMask || windowSize > 0) {
                nEnd = mIdx + 1;
            }
            uint32_t windowStart = (windowSize + ppNScalar - 1) / ppNScalar;
            uint32_t nStart = 0;
            if constexpr (swaFlag) {
                if (windowSize > 0 && windowSize < kvSeqlen) {
                    nStart = (mIdx < windowStart) ? 0 : mIdx - windowStart;
                    if constexpr (!swaCompress) {
                        maskOffset += nStart * ppNScalar;
                    }
                }
            }
            uint32_t qkNTriu = nEnd * ppNScalar;
            uint32_t sBlockStack = nEnd > 4 ? 2 : 1;
            // PV is in stage 3, which means the 1st PV block corresponding to the 3th QK in our pipeline strategy

            uint32_t pvStage = 3;
            uint32_t launchDelay = sBlockStack * 2;
            uint32_t vectMod = 2 * launchDelay;
            uint32_t mSlice = subM > 32 ? 32 : 0; // sBlockStack=2时，UB可以放下
            uint32_t mEnd = subM > 32 ? 2 : 1;
            for (uint32_t nIdx = nStart; nIdx < nEnd + launchDelay; nIdx += sBlockStack) {
                if (nIdx < nEnd) {
                    uint32_t pScaleOffset =
                        nIdx / sBlockStack % pvStage * RoundUp<uint32_t>(ppMScalar, FLOAT_VECTOR_SIZE);
                    if (nIdx + sBlockStack > nEnd - 1) {
                        qkN = qkNTriu > kvSeqlen ? kvSeqlen - nIdx * ppNScalar : qkNTriu - nIdx * ppNScalar;
                    } else {
                        qkN = ppNScalar * sBlockStack;
                    }
		            uint32_t deltaIdx = mIdx - nIdx;
                    bool skipMask = windowStart > MAX_WINDOW_BLOCKS_BASE_API && deltaIdx > ONE_BASE_API && deltaIdx < windowStart - ONE_BASE_API;
                    if constexpr (swaCompress) {
                        maskOffset = 0;
                        if (windowStart <= MAX_WINDOW_BLOCKS_BASE_API) {    // window < 128*3 最多跨4个基块
                            if (mIdx < nIdx) {
                                maskOffset = ppNScalar; // 偏移128个数, midx=0, nidx=1
                            } else {
                                maskOffset = deltaIdx * maxSeqlen * ppMScalar;
                            }
                        } else {
                            if (deltaIdx == MASK_OFFSET_NONE_BASE_API) {
                                maskOffset = MASK_OFFSET_NONE_BASE_API;    // m = n
                            } else if (deltaIdx == windowStart) {
                                maskOffset = MASK_OFFSET_3X_BASE_API * maxSeqlen * ppMScalar;
                            } else if (deltaIdx == MASK_OFFSET_1X_BASE_API) {
                                maskOffset = maxSeqlen * ppMScalar;
                            } else if (deltaIdx == windowStart - MASK_OFFSET_1X_BASE_API) {
                                maskOffset = MASK_OFFSET_2X_BASE_API * maxSeqlen * ppMScalar;
                            }   // delta idx in [2, window-1) do not move and add mask
                        }
                    }
                    qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    if (qkN <= VECTOR_SIZE) {
                        if (subM > NO_SUB_BLOCK_BASE_API && maskType != DEFAULT_MASK_TYPE_BASE_API) {
                            if (alibiCoeffGm != nullptr) {
                                if (alibiLeftAlign == ZERO_BASE_API) {
                                    if (nIdx == nEnd - 1) {
                                        maskOffset = 0;
                                        deltaUint = 0;
                                        delta = 0;
                                    } else {
                                        maskOffset = BASE_MASK_SIZE * maxSeqlen;
                                        deltaUint = mIdx * ppMScalar - nIdx * ppNScalar;
                                        delta = baseY + deltaUint;
                                    }
                                } else {
                                    if (nIdx == nEnd - 1) {
                                        maskOffset = 0;
                                    } else {
                                        maskOffset = BASE_MASK_SIZE * maxSeqlen;
                                    }
                                    delta = -baseY * nIdx;
                                }
                            } else if (maskType == MASK_TYPE_NO_MASK_BASE_API && alibiCompressOffset > ZERO_BASE_API) {
                                if (nIdx == nEnd - 1) {
                                    maskOffset = headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                } else {
                                    deltaUint = mIdx * ppMScalar - nIdx * ppNScalar;
                                    maskOffset = BASE_MASK_SIZE * deltaUint + headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                }
                            }
                            if (longSeq == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                if constexpr (!swaCompress) {
                                    AscendC::DataCopyPad(
                                        mask16UbufTensor,
                                        maskGmTensor[maskOffset + subBlockIdx * qkM / 2 * maxSeqlen], // 2 is the CV_RATIO
                                        AscendC::DataCopyExtParams(
                                                subM,                   // nBurst
                                                qkN * TWO_BASE_API,                // lenBurst
                                                (maxSeqlen - qkN) * TWO_BASE_API, // srcGap
                                                0,                        // dstGap
                                                0),
                                        AscendC::DataCopyPadExtParams<U_T>(
                                            false,
                                            0,
                                            0,
                                            0)
                                    );
                                } else {
                                    if (!(skipMask)) {
                                        AscendC::DataCopyPad(
                                            mask16UbufTensor,
                                            maskGmTensor[maskOffset + subBlockIdx * qkM / 2 * maxSeqlen], // 2 is the CV_RATIO
                                            AscendC::DataCopyExtParams(
                                                    subM,                   // nBurst
                                                    qkN * TWO_BASE_API,                // lenBurst
                                                    (maxSeqlen - qkN) * TWO_BASE_API, // srcGap
                                                    0,                        // dstGap
                                                    0),
                                            AscendC::DataCopyPadExtParams<U_T>(
                                                false,
                                                0,
                                                0,
                                                0)
                                        );
                                    }
                                }
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                if constexpr (!swaCompress) {
                                    maskOffset += qkN;
                                }
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                if constexpr (!swaCompress) {
                                    AscendC::Cast<float, U_T, false>(
                                        maskUbufTensor,
                                        mask16UbufTensor,
                                        AscendC::RoundMode::CAST_NONE,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                    );
                                } else {
                                    if (!(skipMask)) {
                                        AscendC::Cast<float, U_T, false>(
                                            maskUbufTensor,
                                            mask16UbufTensor,
                                            AscendC::RoundMode::CAST_NONE,
                                            (uint64_t)0,
                                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                        );
                                    }
                                }
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
                                        maskUbufTensor, (float)alibiCoeff,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                if (headStride == HEAD_STRIDE_INVALID_BASE_API && maskType != MASK_TYPE_NO_MASK_BASE_API) {
                                    if constexpr (!swaCompress) {
                                        AscendC::Muls<float, false>(
                                            maskUbufTensor,
                                            maskUbufTensor, (float)-3e38,
                                            (uint64_t)0,
                                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                    } else {
                                        if (!(skipMask)) {
                                            AscendC::Muls<float, false>(
                                                maskUbufTensor,
                                                maskUbufTensor, (float)-3e38,
                                                (uint64_t)0,
                                                (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                            );
                                        }
                                    }
                                    AscendC::PipeBarrier<PIPE_V>();
                                }
                            }
                        }
                        AscendC::CrossCoreWaitFlag(QK_READY);
                        if (subM > 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
                            // input QK
                            AscendC::DataCopy(
                                lsUbufTensor,
                                sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                    (uint64_t)subBlockIdx * qkM / 2 * qkRoundN], // 2 is the CV_RATIO
                                AscendC::DataCopyParams(
                                    subM,                                // nBurst
                                    qkRoundN / FLOAT_BLOCK_SIZE,  // lenBurst
                                    0,                                // srcGap
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                            if(scaleType == ScaleType::SCALE_LOGN_FP32){
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                AscendC::DataCopyPad(
                                    logUbufFloatTensor,
                                    logNFloatGmTensor[mIdx * ppMScalar + (uint64_t)subBlockIdx * qkM / TWO_BASE_API],
                                    AscendC::DataCopyExtParams(
                                             1,                   // nBurst
                                            subM * ELEMENT_SIZE_BYTES_BASE_API,                // lenBurst
                                            0, // srcGap byte
                                            (roundSubM - subM) * ELEMENT_SIZE_BYTES_BASE_API,    // dstGap block
                                            0),
                                    AscendC::DataCopyPadExtParams<float>(
                                        false,
                                        0,
                                        0,
                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::Brcb(
                                    tvUbufTensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE],
                                    logUbufFloatTensor.ReinterpretCast<uint32_t>(),
                                    (subM + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE,
                                    AscendC::BrcbRepeatParams(1, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                if constexpr (int8Flag) {
                                    DeqPerHeadS322F32(lsUbufTensor, deqQkGm, offQkGm, headIdx, subM * qkRoundN);
                                }
                                for (uint32_t vdivIdx = 0; vdivIdx < qkN / FLOAT_VECTOR_SIZE; ++vdivIdx) {
                                    AscendC::Mul<float, false>(
                                        lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                        lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0,  qkRoundN / FLOAT_BLOCK_SIZE,  qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                    __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                    AscendC::Mul<float, false>(
                                        lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0,  qkRoundN / FLOAT_BLOCK_SIZE,  qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                    );
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                 // *** ls = tor * ls
                                AscendC::Muls<float, false>(
                                    lsUbufTensor, lsUbufTensor, tor,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                            } else if(scaleType == ScaleType::SCALE_LOGN) {
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                AscendC::DataCopyPad(
                                    logUbufTensor,
                                    logNGmTensor[mIdx * ppMScalar + (uint64_t)subBlockIdx * qkM / 2 ], // 2 is the CV_RATIO
                                    AscendC::DataCopyExtParams(
                                            1,                   // nBurst
                                            subM * TWO_BASE_API,                // lenBurst
                                            0, // srcGap byte
                                            (roundSubM - subM) * TWO_BASE_API,    // dstGap block
                                            0),
                                    AscendC::DataCopyPadExtParams<U_T>(
                                        false,
                                        0,
                                        0,
                                        0)
                                );
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                AscendC::Cast<float, U_T, false>(
                                    tvUbufTensor,
                                    logUbufTensor,
                                    AscendC::RoundMode::CAST_NONE,
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::Brcb(
                                    tvUbufTensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE],
                                    tvUbufTensor.ReinterpretCast<uint32_t>(),
                                    (subM + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE,
                                    AscendC::BrcbRepeatParams(1, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                if constexpr (int8Flag) {
                                    DeqPerHeadS322F32(lsUbufTensor, deqQkGm, offQkGm, headIdx, subM * qkRoundN);
                                }
                                for (uint32_t vdivIdx = 0; vdivIdx < qkN / FLOAT_VECTOR_SIZE; ++vdivIdx) {
                                    AscendC::Mul<float, false>(
                                        lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                        lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0,  qkRoundN / FLOAT_BLOCK_SIZE,  qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                               if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                    __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                    AscendC::Mul<float, false>(
                                        lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                        tvUbufTensor[VECTOR_SIZE],
                                        (uint64_t)0,
                                        subM,
                                        AscendC::BinaryRepeatParams(1, 1, 0,  qkRoundN / FLOAT_BLOCK_SIZE,  qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                    );
                                    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** ls = tor * ls
                                AscendC::Muls<float, false>(
                                    lsUbufTensor, lsUbufTensor, tor,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                            } else {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                if constexpr (int8Flag) {
                                    DeqPerHeadS322F32(lsUbufTensor, deqQkGm, offQkGm, headIdx, subM * qkRoundN);
                                }
                                // *** ls = tor * ls
                                AscendC::Muls<float, false>(
                                    lsUbufTensor, lsUbufTensor, tor,
                                    (uint64_t)0,
                                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                            }
                            AscendC::PipeBarrier<PIPE_V>();

                            if (isClamp == 1) {
                                // get min(clampMin，ls_ubuf)
                                AscendC::Maxs<float, false>(
                                    lsUbufTensor,
                                    lsUbufTensor,
                                    clampMin,
                                    (uint64_t)0,
                                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();

                                // get max(clampMin，ls_ubuf)
                                AscendC::Mins<float, false>(
                                    lsUbufTensor,
                                    lsUbufTensor,
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
                                    if constexpr (!swaCompress) {
                                        AscendC::Add<float, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            maskUbufTensor,
                                            (uint64_t)0,
                                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                    } else {
                                        if (!(skipMask)) {
                                            AscendC::Add<float, false>(
                                                lsUbufTensor,
                                                lsUbufTensor,
                                                maskUbufTensor,
                                                (uint64_t)0,
                                                (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                            );
                                        }
                                    }
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                } else if (ppNScalar == FLOAT_VECTOR_SIZE && sBlockStack == TWO_BASE_API && nIdx == nEnd - TWO_BASE_API) {
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
                                    qkRoundN / EIGHT_BASE_API
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                AscendC::BlockReduceMax<float, false>(
                                    lmUbufTensor,
                                    tvUbufTensor,
                                    roundSubM * EIGHT_BASE_API / SIXTY_FOUR_BASE_API,
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
                                        qkRoundN / 8 // (qkRoundN / 8) is the repeate stride
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    AscendC::BlockReduceMax<float, false>(
                                        tvUbufTensor,
                                        tvUbufTensor,
                                        roundSubM * 8 / 64, // (8 / 64) is the repeate time 
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
                                        qkRoundN / 8 // (qkRoundN / 8) is the repeate stride
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    __set_vcg_mask((qkN % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
                                    AscendC::BlockReduceMax<float, false>(
                                        tvUbufTensor,
                                        tvUbufTensor,
                                        roundSubM * 8 / 64, // (8 / 64) is the repeate time 
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
                            if (nIdx == nStart) {
                                // *** hm = lm
                                AscendC::DataCopy( hmUbufTensor,
                                                lmUbufTensor,
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundSubM / FLOAT_BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
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
                                    dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],  // 4 is the BlockStack Num
                                    gmUbufTensor,
                                    hmUbufTensor,
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                                // *** dm = exp(dm)
                                AscendC::Exp<float, false>(
                                    dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],  // 4 is the BlockStack Num
                                    dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],  // 4 is the BlockStack Num
                                    (uint64_t)0,
                                    subMD64,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                );
                            }
                            // *** gm = hm
                            AscendC::DataCopy(gmUbufTensor,
                                                hmUbufTensor,
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundSubM / FLOAT_BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
                                                                        0)
                                );
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** hm_block = expand_to_block(hm), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint32_t>(),
                                hmUbufTensor.ReinterpretCast<uint32_t>(),
                                roundSubM / FLOAT_BLOCK_SIZE,
                                AscendC::BrcbRepeatParams(1, STRIDE_8)  // 8 is the src data repeatStride
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
                            if constexpr (int8Flag) {
                                SymmetricQuant(lpUbufTensor, ls32UbufTensor, lmUbufTensor, hmUbufTensor,
                                               pScaleUbufTensor[pScaleOffset], subM, roundSubM, qkN,
                                               qkRoundN, headIdx);
                            } else {
                                // *** lp = castfp32to16(ls)
                                if (ISBF16) {
                                    AscendC::Cast<U_T, float, false>(
                                        lpUbufTensor.ReinterpretCast<U_T>(),
                                        ls32UbufTensor,
                                        AscendC::RoundMode::CAST_RINT,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                    );
                                } else {
                                    AscendC::Cast<P_T, float, false>(
                                        lpUbufTensor.ReinterpretCast<P_T>(), ls32UbufTensor,
                                        AscendC::RoundMode::CAST_NONE,
                                        (uint64_t)0,
                                        (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                    );
                                }
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            AscendC::DataCopy(
                                pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                            (uint64_t)subBlockIdx * qkM / 2 * qkRoundN) * 2 / sizeof(P_T)], // 2 is the CV_RATIO
                                lpUbufTensor.ReinterpretCast<P_T>(),
                                AscendC::DataCopyParams(
                                    1,                                            // nBurst
                                    subM * qkRoundN * TWO_BASE_API / BlockSize<int8_t>(), // lenBurst
                                    0,                                  // srcGap
                                    0                                   // dstGap
                                )
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
                                    1,
                                    1,
                                    qkRoundN / FLOAT_BLOCK_SIZE
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
                                    1,
                                    1,
                                    qkRoundN / FLOAT_BLOCK_SIZE
                                );
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            if (nIdx == nStart) {
                                // *** gl = ll
                                AscendC::DataCopy(glUbufTensor,
                                                llUbufTensor,
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundSubM / FLOAT_BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
                                                                        0)
                                );
                                AscendC::PipeBarrier<PIPE_V>();
                            } else {
                                // *** gl = dm * gl
                                AscendC::Mul<float, false>(
                                    glUbufTensor,
                                    dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE],  // 4 is the BlockStack Num
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
                        bool lastNLoop = nIdx + sBlockStack > nEnd - 1;
                        for (uint32_t splitIdx = 0; splitIdx < mEnd; splitIdx++) {
                            bool lastMLoop = splitIdx == mEnd - 1;
                            uint32_t mSplit =  lastMLoop ? subM - splitIdx * mSlice : mSlice;
                            uint32_t roundMSplit = (mSplit + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
                            if (subM > 0 && maskType != 0 && longSeq == 0) {
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                uint64_t maskOffsetTail = 0;
                                if (alibiCoeffGm != nullptr) {
                                    maskOffset = BASE_MASK_SIZE * SOFTMAX_MAX_LENGTH;
                                    if (alibiLeftAlign == ZERO_BASE_API) {
                                        delta = baseY * (nIdx + 1 - mIdx);
                                    } else {
                                        delta = -baseY * nIdx;
                                    }
                                    AscendC::DataCopy(
                                        mask16UbufTensor,
                                        maskGmTensor[maskOffset + (subBlockIdx * qkM / 2 + splitIdx * mSlice) * SOFTMAX_MAX_LENGTH], // 2 is the CV_RATIO
                                        AscendC::DataCopyParams(
                                            mSplit,                                // nBurst
                                            qkRoundN / BLOCK_SIZE,  // lenBurst
                                            (SOFTMAX_MAX_LENGTH - qkRoundN) / BLOCK_SIZE, // srcGap
                                            0
                                        )
                                    );
                                } else if (maskType == 2 && alibiCompressOffset > 0) { // 2 is the mask mode
                                    deltaUint = mIdx * ppMScalar - nIdx * ppNScalar;
                                    maskOffset = BASE_MASK_SIZE * deltaUint + headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                    maskOffsetTail = maskOffset - BASE_MASK_SIZE * ppNScalar;
                                    if (nIdx == nEnd - 2) {  // 2 is the early start num
                                        maskOffsetTail = headIdx * alibiCompressOffset * BASE_MASK_SIZE;
                                    }
                                    AscendC::DataCopy(
                                        mask16UbufTensor,
                                        maskGmTensor[maskOffset + (subBlockIdx * qkM / 2 + splitIdx * mSlice) * VECTOR_SIZE], // 2 is the CV_RATIO
                                        AscendC::DataCopyParams(
                                            mSplit,                                // nBurst
                                            8,  // 8 is the lenBurst
                                            0, // srcGap
                                            (qkRoundN - VECTOR_SIZE)/ BLOCK_SIZE
                                        )
                                    );
                                    AscendC::DataCopy(
                                        mask16UbufTensor[VECTOR_SIZE],
                                        maskGmTensor[maskOffset + (subBlockIdx * qkM / 2 + splitIdx * mSlice) * VECTOR_SIZE], // 2 is the CV_RATIO
                                        AscendC::DataCopyParams(
                                            mSplit,                                // nBurst
                                            (qkRoundN - VECTOR_SIZE)/ BLOCK_SIZE,  // lenBurst
                                            (SOFTMAX_MAX_LENGTH - qkRoundN)/ BLOCK_SIZE, // srcGap
                                            8 // 8 is the dst Stride
                                        )
                                    );
                                } else {
                                    if constexpr (!swaCompress) {
                                        AscendC::DataCopyPad(
                                            mask16UbufTensor,
                                            maskGmTensor[maskOffset + (subBlockIdx * qkM / 2 + splitIdx * mSlice) * maxSeqlen], // 2 is the CV_RATIO
                                            AscendC::DataCopyExtParams(
                                                    mSplit,                   // nBurst
                                                    qkN * 2,                // 2 is the lenBurst factor
                                                    (maxSeqlen - qkN) * 2, // 2 is the srcGap factor
                                                    0,                        // dstGap
                                                    0),
                                            AscendC::DataCopyPadExtParams<U_T>(
                                                false,
                                                0,
                                                0,
                                                0)
                                        );
                                    } else {
                                        if (!skipMask) {
                                            AscendC::DataCopyPad(
                                                mask16UbufTensor,
                                                maskGmTensor[maskOffset + (subBlockIdx * qkM / 2 + splitIdx * mSlice) * maxSeqlen], // 2 is the CV_RATIO
                                                AscendC::DataCopyExtParams(
                                                        mSplit,                   // nBurst
                                                        qkN * 2,                // 2 is the lenBurst
                                                        (maxSeqlen - qkN) * 2, // 2 is the srcGap factor
                                                        0,                        // dstGap
                                                        0),
                                                AscendC::DataCopyPadExtParams<U_T>(
                                                    false,
                                                    0,
                                                    0,
                                                    0)
                                            );
                                        }
                                    }
                                }
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                            }
                            if (splitIdx == 0) {
                                AscendC::WaitEvent(QK_READY);
                            }
                            if (subM > 0) {
                                if (mSplit > 0) {
                                    if (maskType != 0 && longSeq == 0) {
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                        if constexpr (!swaCompress) {
                                            AscendC::Cast<float, U_T, false>(
                                                maskUbufTensor,
                                                mask16UbufTensor,
                                                AscendC::RoundMode::CAST_NONE,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                            );
                                        } else {
                                            if (!skipMask) {
                                                AscendC::Cast<float, U_T, false>(
                                                maskUbufTensor,
                                                mask16UbufTensor,
                                                AscendC::RoundMode::CAST_NONE,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                            );
                                            }
                                        }

                                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID1));
                                        if (alibiCoeffGm != nullptr) {
                                            AscendC::PipeBarrier<PIPE_V>();
                                            if (nIdx != nEnd - TWO_BASE_API) {
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
                                                        maskUbufTensor[VECTOR_SIZE],
                                                        maskUbufTensor,
                                                        (float)-baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / 8, qkRoundN / 8) // (qkRoundN / 8) is the repeate stride 
                                                    );
                                                    AscendC::Adds<float, false>(
                                                        maskUbufTensor[VECTOR_SIZE + FLOAT_VECTOR_SIZE],
                                                        maskUbufTensor[FLOAT_VECTOR_SIZE],
                                                        (float)-baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / 8, qkRoundN / 8) // (qkRoundN / 8) is the repeate stride 
                                                    );
                                                } else {
                                                    AscendC::Adds<float, false>(
                                                        maskUbufTensor[VECTOR_SIZE],
                                                        maskUbufTensor,
                                                        (float)baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / 8, qkRoundN / 8) // (qkRoundN / 8) is the repeate stride 
                                                    );
                                                    AscendC::Adds<float, false>(
                                                        maskUbufTensor[VECTOR_SIZE + FLOAT_VECTOR_SIZE],
                                                        maskUbufTensor[FLOAT_VECTOR_SIZE],
                                                        (float)baseY,
                                                        (uint64_t)0,
                                                        mSplit,
                                                        AscendC::UnaryRepeatParams(1, 1, qkRoundN / 8, qkRoundN / 8) // (qkRoundN / 8) is the repeate stride 
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
                                        if (headStride == 0 && maskType != 2) {  // 2 is the mask mode
                                            if constexpr (!swaCompress) {
                                                AscendC::Muls<float, false>(
                                                    maskUbufTensor,
                                                    maskUbufTensor,
                                                    (float)-3e38,
                                                    (uint64_t)0,
                                                    (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                                );
                                            } else {
                                                if (!skipMask) {
                                                    AscendC::Muls<float, false>(
                                                        maskUbufTensor,
                                                        maskUbufTensor,
                                                        (float)-3e38,
                                                        (uint64_t)0,
                                                        (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                        AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                                    );
                                                }
                                            }
                                            AscendC::PipeBarrier<PIPE_V>();
                                        }
                                    }
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));
                                    // input QK
                                    AscendC::DataCopy(
                                        lsUbufTensor,
                                        sGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                            (uint64_t)(subBlockIdx * qkM / 2 + splitIdx * mSlice) * qkRoundN], // 2 is the CV_RATIO
                                        AscendC::DataCopyParams(
                                            mSplit,                                // nBurst
                                            qkRoundN / FLOAT_BLOCK_SIZE,  // lenBurst
                                            0,                                // srcGap
                                            0
                                        )
                                    );
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                    if constexpr (int8Flag) {
                                        DeqPerHeadS322F32(lsUbufTensor, deqQkGm, offQkGm, headIdx,
                                                          mSplit * qkRoundN);
                                    }
                                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                    if(scaleType == ScaleType::SCALE_LOGN_FP32){
                                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                        AscendC::DataCopyPad(
                                            logUbufFloatTensor,
                                            logNFloatGmTensor[mIdx * ppMScalar +  (uint64_t)(subBlockIdx * qkM / 2 + splitIdx * mSlice)], // 2 is the CV_RATIO
                                            AscendC::DataCopyExtParams(
                                                    1,                          // nBurst
                                                    mSplit * ELEMENT_SIZE_BYTES_BASE_API,                // lenBurst
                                                    0,                          // srcGap byte
                                                    (roundMSplit - mSplit) * ELEMENT_SIZE_BYTES_BASE_API,   // dstGap block
                                                    0),
                                            AscendC::DataCopyPadExtParams<float>(
                                                false,
                                                0,
                                                0,
                                                0)
                                        );
                                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                        AscendC::Brcb(
                                            tvUbufTensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE],
                                            logUbufFloatTensor.ReinterpretCast<uint32_t>(),
                                            roundMSplit / FLOAT_BLOCK_SIZE,
                                            AscendC::BrcbRepeatParams(1, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));

                                        for (uint32_t vdivIdx = 0; vdivIdx < qkN / FLOAT_VECTOR_SIZE; ++vdivIdx) {
                                            AscendC::Mul<float, false>(
                                                lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                                lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                                tvUbufTensor[VECTOR_SIZE],
                                                (uint64_t)0,
                                                mSplit,
                                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                            );
                                        }
                                        AscendC::PipeBarrier<PIPE_V>();
                                        if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                            __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                            AscendC::Mul<float, false>(
                                                lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                tvUbufTensor[VECTOR_SIZE],
                                                (uint64_t)0,
                                                mSplit,
                                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                            );
                                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                        }
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::Muls<float, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            tor,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                    } else if(scaleType == ScaleType::SCALE_LOGN) {
                                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));
                                        AscendC::DataCopyPad(
                                            logUbufTensor,
                                            logNGmTensor[mIdx * ppMScalar +  (uint64_t)(subBlockIdx * qkM / 2 + splitIdx * mSlice)], // 2 is the CV_RATIO
                                            AscendC::DataCopyExtParams(
                                                    1,                          // nBurst
                                                    mSplit * TWO_BASE_API,                // lenBurst
                                                    0,                          // srcGap byte
                                                    (roundMSplit - mSplit) * TWO_BASE_API,   // dstGap block
                                                    0),
                                            AscendC::DataCopyPadExtParams<U_T>(
                                                false,
                                                0,
                                                0,
                                                0)
                                        );
                                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID1));
                                        AscendC::Cast<float, U_T, false>(
                                            tvUbufTensor,
                                            logUbufTensor,
                                            AscendC::RoundMode::CAST_NONE,
                                            (uint64_t)0,
                                            (mSplit + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_4)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::Brcb(
                                            tvUbufTensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE],
                                            tvUbufTensor.ReinterpretCast<uint32_t>(),
                                            roundMSplit / FLOAT_BLOCK_SIZE,
                                            AscendC::BrcbRepeatParams(1, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();

                                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID0));
                                        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID7));

                                        for (uint32_t vdivIdx = 0; vdivIdx < qkN / FLOAT_VECTOR_SIZE; ++vdivIdx) {
                                            AscendC::Mul<float, false>(
                                                 lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                                lsUbufTensor[vdivIdx * FLOAT_VECTOR_SIZE],
                                                tvUbufTensor[VECTOR_SIZE],
                                                (uint64_t)0,
                                                mSplit,
                                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                            );
                                        }
                                        AscendC::PipeBarrier<PIPE_V>();
                                        if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                            __set_mask(qkN % FLOAT_VECTOR_SIZE);
                                            AscendC::Mul<float, false>(
                                                lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                lsUbufTensor[qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                                tvUbufTensor[VECTOR_SIZE],
                                                (uint64_t)0,
                                                mSplit,
                                                AscendC::BinaryRepeatParams(1, 1, 0, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 1)
                                            );
                                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                        }
                                        AscendC::PipeBarrier<PIPE_V>();
                                        AscendC::Muls<float, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            tor,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                    } else {
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
                                    }
                                    AscendC::PipeBarrier<PIPE_V>();
                                    if (isClamp == 1) {
                                        // get min(clampMin，ls_ubuf)
                                        AscendC::Maxs<float, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            clampMin,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        // get max(clampMin，ls_ubuf)
                                        AscendC::Mins<float, false>(
                                            lsUbufTensor,
                                            lsUbufTensor,
                                            clampMax,
                                            (uint64_t)0,
                                            (mSplit * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    }
                                    // *** ls = ls + mask
                                    if (maskType != 0) {
                                        if (longSeq == 0) {
                                            if constexpr (!swaCompress) {
                                                AscendC::Add<float, false>(
                                                    lsUbufTensor,
                                                    lsUbufTensor,
                                                    maskUbufTensor,
                                                    (uint64_t)0,
                                                    (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                                );
                                                            } else {
                                                if (!skipMask) {
                                                    AscendC::Add<float, false>(
                                                        lsUbufTensor,
                                                        lsUbufTensor,
                                                        maskUbufTensor,
                                                        (uint64_t)0,
                                                        (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                        AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                                    );
                                                }
                                            }
                                        } else if (nIdx == nEnd - TWO_BASE_API) {
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
                                        __set_mask(THIRTY_TWO_BASE_API);
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
                                        __set_vcg_mask(FOUR_BASE_API);
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
                                                hmUbufTensor,
                                                lmUbufTensor,
                                                tvUbufTensor,
                                                (uint64_t)0,
                                                1,
                                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                            );
                                            AscendC::PipeBarrier<PIPE_V>();
                                            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
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
                                    if (nIdx == nStart) {
                                        // *** hm = lm
                                        AscendC::DataCopy(hmUbufTensor[splitIdx * mSlice],
                                                lmUbufTensor,
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundMSplit / FLOAT_BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
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
                                            dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
                                            gmUbufTensor[splitIdx * mSlice],
                                            hmUbufTensor[splitIdx * mSlice],
                                            (uint64_t)0,
                                            1,
                                            AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                        // *** dm = exp(dm)
                                        AscendC::Exp<float, false>(
                                            dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
                                            dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
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
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundMSplit / FLOAT_BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
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
                                    if constexpr (int8Flag) {
                                        SymmetricQuant(lpUbufTensor, ls32UbufTensor, lmUbufTensor,
                                                       hmUbufTensor[splitIdx * mSlice],
                                                       pScaleUbufTensor[pScaleOffset + splitIdx * mSlice],
                                                       mSplit, roundMSplit, qkN, qkRoundN, headIdx);
                                    } else {
                                        // *** lp = castfp32to16(ls)
                                        if (ISBF16) {
                                            AscendC::Cast<P_T, float, false>(
                                                lpUbufTensor.ReinterpretCast<P_T>(),
                                                ls32UbufTensor,
                                                AscendC::RoundMode::CAST_RINT,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                            );
                                        } else {
                                            AscendC::Cast<P_T, float, false>(
                                                lpUbufTensor.ReinterpretCast<P_T>(), ls32UbufTensor,
                                                AscendC::RoundMode::CAST_NONE,
                                                (uint64_t)0,
                                                (mSplit * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                            );
                                        }
                                        AscendC::PipeBarrier<PIPE_V>();
                                    }
                                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                                    AscendC::DataCopy(
                                        pGmTensor[((uint64_t)GetBlockIdx() * TMP_SIZE +
                                                    (nIdx - nStart) % vectMod * TMP_SIZE / vectMod +
                                                    ((uint64_t)subBlockIdx * qkM / TWO_BASE_API + splitIdx * mSlice) *
                                                        qkRoundN) * TWO_BASE_API / sizeof(P_T)],
                                        lpUbufTensor.ReinterpretCast<P_T>(),
                                        AscendC::DataCopyParams(
                                            mSplit,                              // nBurst
                                            qkRoundN * TWO_BASE_API / BlockSize<int8_t>(), // lenBurst
                                            0,                                  // srcGap
                                            0                                   // dstGap
                                        )
                                    );
                                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID0));

                                    // *** ll = rowsum(ls32)
                                    for (uint32_t rowSumIdx = 1; rowSumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowSumIdx) {
                                        AscendC::Add<float, false>(
                                            ls32UbufTensor,
                                            ls32UbufTensor,
                                            ls32UbufTensor[rowSumIdx * FLOAT_VECTOR_SIZE],
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
                                        1,
                                        1,
                                        qkRoundN / FLOAT_BLOCK_SIZE
                                    );
                                    AscendC::PipeBarrier<PIPE_V>();
                                    if (nIdx == nStart) {
                                        // *** gl = ll
                                        AscendC::DataCopy(glUbufTensor[splitIdx * mSlice],
                                                llUbufTensor,
                                                AscendC::DataCopyParams(1,                         // nBurst
                                                                        roundMSplit / FLOAT_BLOCK_SIZE,  // lenBurst
                                                                        0,                         // srcGap
                                                                        0)
                                        );
                                        AscendC::PipeBarrier<PIPE_V>();
                                    } else {
                                        __set_mask(mSplit);
                                        // *** gl = dm * gl
                                        AscendC::Mul<float, false>(
                                            glUbufTensor[splitIdx * mSlice],
                                            dmUbufTensor[((nIdx - nStart) / sBlockStack) % 4 * UB_FLOAT_LINE_SIZE + splitIdx * mSlice], // 4 is the BlockStack num
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
                        if constexpr(!swaCompress) {
                            maskOffset += qkN;
                        }
                    }
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SOFTMAX_READY); // 2 is the Sync Mode
                }
                if (nIdx >= launchDelay + nStart) {
                    uint32_t pScaleOffset = (nIdx - launchDelay) / sBlockStack % pvStage * RoundUp<uint32_t>(ppMScalar, FLOAT_VECTOR_SIZE);
                    AscendC::CrossCoreWaitFlag(UPDATE_READY);
                    if (subM > 0) {
                        // *** 更新 L 和 O
                        if (nIdx != launchDelay + nStart) {
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                            AscendC::DataCopy(
                                loUbufTensor,
                                oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod +
                                                (uint64_t)subBlockIdx * qkM / 2 * roundK],  // 2 is the CV_RATIO
                                AscendC::DataCopyParams(
                                     1,                                  // nBurst
                                    subM * roundK / FLOAT_BLOCK_SIZE, // lenBurst
                                    0,                                // srcGap
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));

                            // *** dm_block = expand_to_block(dm), 存放于 tv
                            AscendC::Brcb(
                                tvUbufTensor.ReinterpretCast<uint32_t>(),
                                 dmUbufTensor[((nIdx - launchDelay - nStart) / sBlockStack % 4) * UB_FLOAT_LINE_SIZE].ReinterpretCast<uint32_t>(), // 4 is the BlockStack Num
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
                                    AscendC::BinaryRepeatParams(1, 1, 0,  roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                                );
                            }
                            if (__k % FLOAT_VECTOR_SIZE > 0) {
                                __set_mask(__k % FLOAT_VECTOR_SIZE);
                                AscendC::Mul<float, false>(
                                    goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                    goUbufTensor[__k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                                    tvUbufTensor,
                                    (uint64_t)0,
                                    subM,
                                    AscendC::BinaryRepeatParams(1, 1, 0,  roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1)
                                );
                                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            // *** go = lo + go

                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID2));
                            if constexpr (int8Flag) {
                                DeqPerHeadS322F32(loUbufTensor, deqPvGm, offPvGm, headIdx, subM * roundK);
                                SymmetricDeQuant(loUbufTensor, pScaleUbufTensor[pScaleOffset], subM,
                                                 roundSubM, __k, roundK, headIdx);
                            }
                            AscendC::Add<float, false>(
                                goUbufTensor,
                                goUbufTensor,
                                loUbufTensor,
                                (uint64_t)0,
                                (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                            );
                            AscendC::PipeBarrier<PIPE_V>();
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>((EVENT_ID2));
                        } else {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                            AscendC::DataCopy(
                                goUbufTensor,
                                oTmpGmTensor[(uint64_t)GetBlockIdx() * TMP_SIZE + (nIdx - launchDelay - nStart) % vectMod * TMP_SIZE / vectMod +
                                    (uint64_t)subBlockIdx * qkM / 2 * roundK], // 2 is the CV_RATIO
                                AscendC::DataCopyParams(
                                    1,                                  // nBurst
                                    subM * roundK / FLOAT_BLOCK_SIZE, // lenBurst
                                    0,                                // srcGap
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((EVENT_ID3));
                            if constexpr (int8Flag) {
                                DeqPerHeadS322F32(goUbufTensor, deqPvGm, offPvGm, headIdx, subM * roundK);
                                SymmetricDeQuant(goUbufTensor, pScaleUbufTensor[pScaleOffset], subM,
                                                 roundSubM, __k, roundK, headIdx);
                            }
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
                                __set_mask(__k % FLOAT_VECTOR_SIZE);
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
                            if (ISBF16) {
                                AscendC::Cast<O_T, float, false>(
                                    goUbufTensor.ReinterpretCast<O_T>(),
                                    goUbufTensor,
                                    AscendC::RoundMode::CAST_RINT,
                                    (uint64_t)0,
                                    (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                    AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                );
                            } else {
                                AscendC::Cast<O_T, float, false>(
                                                goUbufTensor.ReinterpretCast<O_T>(),
                                                 goUbufTensor,
                                                AscendC::RoundMode::CAST_NONE,
                                                (uint64_t)0,
                                                (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                                                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8)
                                            );
                            }
                            AscendC::PipeBarrier<PIPE_V>();
                            // ********************* move O to GM ************************
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((EVENT_ID0));
                            AscendC::DataCopyPad(
                                oGmTensor[oOffset + (uint64_t)subBlockIdx * qkM / TWO_BASE_API * strideQo],
                                goUbufTensor.ReinterpretCast<O_T>(),
                                AscendC::DataCopyExtParams(
                                    subM,                 // nBurst
                                    __k * TWO_BASE_API,               // lenBurst
                                    0,
                                    (strideQo - __k) * TWO_BASE_API,
                                    0
                                )
                            );
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((EVENT_ID2));
                        }
                    }
                }
            }
        }
    }

private:
    using Q_T = typename PFAT::qDType;
    using KV_T = typename PFAT::kvDType;
    using O_T = typename PFAT::outputDType;
    using U_T = typename PFAT::maskDType;
    using S_T = typename PFAT::sDType;
    using P_T = typename PFAT::pDType;
    using TMP_T = typename PFAT::oTmpDType;
    static constexpr bool swaFlag = PFAT::swaFlag;
    static constexpr bool swaCompress = PFAT::swaCompress;
    static constexpr bool ISBF16 = IsSameType<Q_T, bfloat16_t>::value;
    static constexpr bool int8Flag = IsSameType<KV_T, int8_t>::value;

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
    const uint32_t pScaleUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 21 * UB_UINT8_LINE_SIZE;
    const uint32_t logUbufOffset = 8 * UB_UINT8_BLOCK_SIZE + 30 * UB_UINT8_LINE_SIZE;
    const uint32_t logUbufFLoatOffset = 8 * UB_UINT8_BLOCK_SIZE + 30 * UB_UINT8_LINE_SIZE;
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
    AscendC::LocalTensor<float> pScaleUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(pScaleUbufOffset);
    AscendC::LocalTensor<float> goUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbufOffset);
    AscendC::LocalTensor<U_T> mask16UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, U_T>(mask16UbufOffset);

    AscendC::LocalTensor<U_T> logUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, U_T>(logUbufOffset);
    AscendC::LocalTensor<float> logUbufFloatTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(logUbufFLoatOffset);

    AscendC::GlobalTensor<U_T> maskGmTensor;
    AscendC::GlobalTensor<O_T> oGmTensor;
    AscendC::GlobalTensor<float> sGmTensor;
    AscendC::GlobalTensor<P_T> pGmTensor;
    AscendC::GlobalTensor<TMP_T> oTmpGmTensor;
    AscendC::GlobalTensor<U_T> logNGmTensor;
    AscendC::GlobalTensor<float> logNFloatGmTensor;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;

    __gm__ uint8_t *__restrict__ alibiCoeffGm{nullptr};
    __gm__ uint8_t *__restrict__ deqQkGm{nullptr};
    __gm__ uint8_t *__restrict__ deqPvGm{nullptr};
    __gm__ uint8_t *__restrict__ quantPGm{nullptr};
    __gm__ uint8_t *__restrict__ offQkGm{nullptr};
    __gm__ uint8_t *__restrict__ offPvGm{nullptr};

    ScaleType scaleType = ScaleType::SCALE_TOR;
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
    uint32_t goFlagScalar{1};

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
	uint32_t windowSize{0};
};
#endif

template <typename PFAT>
class PromptFlashAttentionBaseApiHighPerformance {
public:
    __aicore__ inline PromptFlashAttentionBaseApiHighPerformance() {};
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
            UnpadAttentionDecoderAic<PFAT> fa_aic_fp16;
            fa_aic_fp16.Run(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                            queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                            attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe, deqScale1, scale1,
                            deqScale2, scale2, offset2);
        #elif __DAV_C220_VEC__
            UnpadAttentionDecoderAiv<PFAT> fa_aiv_fp16;
            fa_aiv_fp16.Run(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                            queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                            attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe, deqScale1, scale1,
                            deqScale2, scale2, offset2);
        #endif
    }
};

template <typename PFAT>
class PromptFlashAttentionBaseApiHighPrecisionV {
public:
    __aicore__ inline PromptFlashAttentionBaseApiHighPrecisionV() {};
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
            UnpadAttentionDecoderAic<PFAT> fa_aic_fp16;
            fa_aic_fp16.Run(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable,
                            queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
                            attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe, deqScale1, scale1,
                            deqScale2, scale2, offset2);
        #elif __DAV_C220_VEC__
            FlashAttentionEncoderHighPrecisionVec<PFAT> fa_vec(query, key, value, pseShift, attenMask, actualSeqLengths,
                        actualSeqLengthsKV, blocktable, queryPaddingSize, kvPaddingSize, keySharedPrefix, valueSharedPrefix,
                        actualSharedPrefixLen, attentionOut, softmaxLse, workspace, tiling, gmTiling, tPipe, deqScale1,
                        scale1, deqScale2, scale2, offset2, alibiCoeff);
            fa_vec.Run();
        #endif
    }
};

#endif // PROMPT_FLASH_ATTENTION_BASE_API_H

