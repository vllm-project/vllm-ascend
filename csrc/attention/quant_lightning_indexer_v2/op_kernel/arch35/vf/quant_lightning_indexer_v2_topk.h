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
 * \file quant_lightning_indexer_v2_topk.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_TOPK_H
#define QUANT_LIGHTNING_INDEXER_V2_TOPK_H

#include "kernel_operator.h"
#include "vf_topk.h"
#include "vf_topk_16_gather_quant_v2.h"

namespace topk {
template<typename T>
class LITopk {
public:
    __aicore__ inline void operator()(LocalTensor<uint32_t>& outputIdxLocal,
                                      LocalTensor<T>& inputLocal,
                                      uint32_t s2SeqLen)
    {
    }
};

template<>
class LITopk<uint32_t> {
public:
    static __aicore__ inline uint32_t GetSharedTmpBufferSize(uint32_t topK)
    {
        return 2 * topK * sizeof(uint32_t) + 5 * 256 * sizeof(uint32_t) + 64 * sizeof(uint32_t) +
               (topK + 64) * sizeof(uint32_t); // for output value tensor
    }

    static __aicore__ inline uint32_t GetIndexBufferSize(uint32_t topK)
    {
        return (topK + 64) * sizeof(uint32_t);
    }

    __aicore__ inline void Init(uint32_t topK)
    {
        this->topK = topK;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer)
    {
        tmpIdxLocal = sharedTmpBuffer[0];
        tmpValueLocal = tmpIdxLocal[topK];
        histogramsLocal = tmpValueLocal[topK];
        idx0Local = histogramsLocal[256];
        idx1Local = idx0Local[256];
        idx2Local = idx1Local[256];
        idx3Local = idx2Local[256];
        nkValueLocal = idx3Local[256];
        outputValueLocal = nkValueLocal[64];
    }

    __aicore__ inline void operator()(LocalTensor<uint32_t>& outputIdxLocal,
                                      LocalTensor<uint32_t>& inputLocal,
                                      uint32_t s2SeqLen)
    {
        topkb32::LiTopKVF(outputIdxLocal, // filter阶段使用输出value Buf topK * 4B
                          outputValueLocal, // filter阶段使用输出 Idx Buf topK * 4B
                          inputLocal, // 输入 s2SeqLen * 4B
                          tmpIdxLocal, // filter阶段使用暂存index Buf topK * 4B
                          tmpValueLocal, // filter阶段使用暂存value Buf topK * 4B
                          histogramsLocal, // 直方图的临时Buf 256 * 4B
                          idx0Local, // 输入数据第1个8位Buf 256 * 4B
                          idx1Local, // 输入数据第2个8位Buf 256 * 4B
                          idx2Local, // 输入数据第3个8位Buf 256 * 4B
                          idx3Local, // 输入数据第4个8位Buf 256 * 4B
                          nkValueLocal, // next_k 暂存Buf 64 * 4B
                          topK,       // topk数量
                          s2SeqLen); // 输入元素总数
    }
private:
    LocalTensor<uint32_t> tmpIdxLocal;     // filter阶段使用暂存index Buf topK * 4B
    LocalTensor<uint32_t> tmpValueLocal;   // filter阶段使用暂存value Buf topK * 4B
    LocalTensor<uint32_t> histogramsLocal; // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idx0Local;       // 输入数据第1个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx1Local;       // 输入数据第2个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx2Local;       // 输入数据第3个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx3Local;       // 输入数据第4个8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal; // next_k 暂存Buf 64 * 4B
    LocalTensor<uint32_t> outputValueLocal; // 输出value tensor
    uint32_t topK;
};

template<>
class LITopk<uint16_t> {
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        // 2 * QLIV2Common::Align(topK, (uint32_t)256): 两块hisIndexLocal;
        // 3 * 256: histogramsLocal idxHighLocal idxLowLocal; 64: nkValueLocal
        uint64_t bufferSize1 = (2 * QLIV2Common::Align(topK, (uint32_t)256) + 3 * 256 + 64)  * sizeof(uint32_t);
        // QLIV2Common::Align(topK, (uint32_t)256) + trunkLen：tmpIndexLocal
        uint64_t bufferSize2 = (QLIV2Common::Align(topK, (uint32_t)256) + trunkLen)  * sizeof(uint16_t);
        uint64_t reuseBufferSize = QLIV2Common::Align(topK, (uint32_t)256) * sizeof(uint32_t);
        return bufferSize1 + bufferSize2 - reuseBufferSize;
    }

    __aicore__ inline void Init(uint32_t topK, uint32_t trunkLen)
    {
        this->topK = topK;
        this->trunkLen =  trunkLen;
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer, LocalTensor<uint32_t>& indicesOutLocal)
    {
        LocalTensor<uint32_t> hisIndexLocal1 = indicesOutLocal;
        LocalTensor<uint32_t> hisIndexLocal2 = sharedTmpBuffer[0];
        hisIndexLocal[0] = hisIndexLocal1;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = hisIndexLocal2[QLIV2Common::Align(topK, (uint32_t)256)];
        idxHighLocal = histogramsLocal[256];
        idxLowLocal = idxHighLocal[256];
        nkValueLocal = idxLowLocal[256];
        LocalTensor<uint32_t> tmpIndexLocalTmp = nkValueLocal[64];
        tmpIndexLocal = tmpIndexLocalTmp.template ReinterpretCast<uint16_t>();
    }

    __aicore__ inline void TopK(LocalTensor<uint16_t>& mrgValueLocal, LocalTensor<uint32_t>& indicesOutLocal,
                                LocalTensor<uint16_t>& hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx,
                                uint32_t s2LoopNum, bool isNeedLD, bool returnValueFlag,
                                uint32_t outputIdxOffset)
    {
        // true: 开启返回hisValueLocal
        if (s2LoopNum == 1) {
            if (isNeedLD || returnValueFlag) {
                topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                              idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            } else {
                topkb16gather::LiTopKVF<false>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                               idxHighLocal, idxLowLocal, nkValueLocal, topK, s2SeqLen);
            }
            PipeBarrier<PIPE_V>();
            Cast(indicesOutLocal, tmpIndexLocal, RoundMode::CAST_NONE, topK);
            if (outputIdxOffset != 0) {
                topkb16gather::IndicesAddOffset(indicesOutLocal, outputIdxOffset, topK);
            }
            return;
        }
        if (loopIdx == 0 && !isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                          idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            Cast(hisIndexLocal[(loopIdx + 1) % 2], tmpIndexLocal, RoundMode::CAST_NONE, topK);
        } else if (loopIdx != 0 && !isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                          idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            uint32_t curProcess = topK < trunkLen ? loopIdx * trunkLen - QLIV2Common::Align(topK, (uint32_t)256) :
                                  (loopIdx - 1) * trunkLen;
            topkb16gather::LiTopKGatherVF(hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, mrgValueLocal,
                                          tmpIndexLocal, hisIndexLocal[loopIdx % 2], topK,
                                          curProcess, s2SeqLen);
            if (loopIdx == s2LoopNum - 1) {
                PipeBarrier<PIPE_V>();
                if ((loopIdx + 1) % 2 == 1) { // 2:pingpong
                    AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2],
                                      QLIV2Common::Align(topK, (uint32_t)256));
                }
            }
        }

        if (loopIdx == 0 && isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            Cast(hisIndexLocal[(loopIdx + 1) % 2], tmpIndexLocal, RoundMode::CAST_NONE, topK);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2],
                              QLIV2Common::Align(topK, (uint32_t)256));
        } else if (loopIdx != 0 && isNeedLD) {
            topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                          idxLowLocal, nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            uint32_t curProcess = topK < trunkLen ? loopIdx * trunkLen - QLIV2Common::Align(topK, (uint32_t)256) :
                                  (loopIdx - 1) * trunkLen;
            topkb16gather::LiTopKGatherVF(hisIndexLocal[(loopIdx + 1) % 2], hisValueLocal, mrgValueLocal,
                                          tmpIndexLocal, hisIndexLocal[loopIdx % 2],
                                          topK, curProcess, s2SeqLen);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal, hisIndexLocal[(loopIdx + 1) % 2],
                              QLIV2Common::Align(topK, (uint32_t)256));
        }
        if (outputIdxOffset != 0) {
            topkb16gather::IndicesAddOffset(indicesOutLocal, outputIdxOffset, topK);
        }
    }
    __aicore__ inline void LdTopK(LocalTensor<uint16_t>& mrgValueLocal, LocalTensor<uint32_t> indexLocal,
                                  LocalTensor<uint32_t>& indicesOutLocal,
                                  LocalTensor<uint16_t>& hisValueLocal, uint32_t s2SeqLen, uint32_t loopIdx,
                                  uint32_t s2LoopNum)
    {
        topkb16gather::LiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal, idxHighLocal,
                                      idxLowLocal, nkValueLocal, topK, s2SeqLen);
        PipeBarrier<PIPE_V>();
        topkb16gather::LiTopKLDGatherVF(indicesOutLocal, tmpIndexLocal, indexLocal, topK);
    }

private:
    LocalTensor<uint32_t> hisIndexLocal[2];      // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;       // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idxHighLocal;          // 输入数据高8位Buf 256 * 4B
    LocalTensor<uint32_t> idxLowLocal;           // 输入数据低8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;          // next_k 暂存Buf 64 * 4B
    LocalTensor<uint16_t> tmpIndexLocal;         // 每trunkLen + topK的临时index
    uint32_t topK = 512;
    uint32_t trunkLen = 16384;
};
}
#endif  // QUANT_LIGHTNING_INDEXER_V2_TOPK_H
