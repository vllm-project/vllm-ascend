/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"
#include "kernel_tensor_impl.h"
#include "kernel_type.h"
#include "kernel_operator_intf.h"
#include "inner_interface/inner_kernel_operator_intf.h"
#include <stdio.h>
#include "types.h"
#include "utils.h"

static constexpr int BUFFER_NUM = 2;
static constexpr int ROTARY_SIZE = 2;

using vllm_ascend::AccType;
using vllm_ascend::smem2smem;
template <typename scalar_t, bool isNeox> class RotaryEmbedding {
    // NOTE(ganyi): we use 32K as load stride for pipe, need to find another way to
    // retrive this size from runtime for more Soc support
    static int constexpr loadSize = 1024 * 4;
    using dst_t = scalar_t;
    using acc_t = typename AccType<scalar_t>::type;
    // only half tensor have cast instruct to int8, hardcode acc_dst_t as half
    using LocT = AscendC::LocalTensor<scalar_t>;
    using LocAcc = AscendC::LocalTensor<acc_t>;
    using LocDst = AscendC::LocalTensor<dst_t>;

public:
    __aicore__ inline RotaryEmbedding()
    {
    }

    __aicore__ inline void init(__gm__ int64_t *positions, __gm__ void *queryDst, __gm__ void *keyDst,
                                __gm__ scalar_t *query, __gm__ scalar_t *key, __gm__ scalar_t *cosSinCache,
                                const int rotDim, const int64_t dstQueryStride,
                                const int64_t dstKeyStride, const int64_t queryStride, const int64_t keyStride,
                                const int numHeads, const int numKvHeads, const int headSize, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        rotDim_ = rotDim;
        queryStride_ = queryStride;
        keyStride_ = keyStride;
        dstQueryStride_ = dstQueryStride;
        dstKeyStride_ = dstKeyStride;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        embedDim_ = rotDim / ROTARY_SIZE;

        pipe_->InitBuffer(inQue_, 1 /* buffer_num */, loadSize /* buffer_size */);
        pipe_->InitBuffer(inQueSinCos_, 1, rotDim_ * sizeof(scalar_t));
        pipe_->InitBuffer(outQue_, 1, loadSize);
        // 2 temperary calculation buffer
        calcTmpBufferOffset_ = 0;
        // 1 upcast buffer for bf16 (headSize)
        upcastInputBufferOffset_ = calcTmpBufferOffset_ + sizeof(acc_t) * embedDim_ * BUFFER_NUM;
        // 1 upcast temp buffer for bf16 (2 * embed_dim)
        upcastTempBufferOffset_ = upcastInputBufferOffset_ + sizeof(acc_t) * headSize_;
        // 2 sin cos upcast buffer for bf16
        cosSinUpcastBufferOffset_ = upcastTempBufferOffset_ + sizeof(acc_t) * BUFFER_NUM * embedDim_;
        // 2. bf16 path: needs 2 cos sin upcast buffer size
        // 3. fp16 path: needs 2 temperary calculation buffer size
        tempBufferSize_ = cosSinUpcastBufferOffset_ + BUFFER_NUM * embedDim_ * sizeof(acc_t);
        // need to consider upcast the bf16 to fp32, so we might need 4 buffer just in case
        // 2 temperary buffer, 2 input buffer, 1 cos buffer, 1 sin buffer, 2 scale buffer (headSize), 2 zp
        // buffer(headSize int8), 1 dst_temp buffer(headSize, int32)
        pipe_->InitBuffer(calcBuf_, tempBufferSize_);
        if constexpr (!std::is_same_v<scalar_t, acc_t>) {
            pipe_->InitBuffer(copyBuf_, loadSize);
        }
    }
    __aicore__ inline void update(__gm__ int64_t *positions, __gm__ void *queryDst, __gm__ void *keyDst,
                                  __gm__ scalar_t *query, __gm__ scalar_t *key, __gm__ scalar_t *cosSinCache,
                                  const int rotDim, const int64_t dstQueryStride, const int64_t dstKeyStride,
                                  const int64_t queryStride, const int64_t keyStride, const int numHeads,
                                  const int numKvHeads, const int headSize, const int64_t idx)
    {
        int64_t pos = positions[idx];
        cosSin_.SetGlobalBuffer(cosSinCache + pos * rotDim_, rotDim_);
        query_.SetGlobalBuffer(query + queryStride * idx, headSize * numHeads_);
        key_.SetGlobalBuffer(key + keyStride * idx, headSize * numKvHeads_);
        queryDst_.SetGlobalBuffer(reinterpret_cast<__gm__ dst_t *>(queryDst) + dstQueryStride * idx,
                                  headSize * numHeads_);
        keyDst_.SetGlobalBuffer(reinterpret_cast<__gm__ dst_t *>(keyDst) + dstKeyStride * idx, headSize * numKvHeads_);
    }

    // compute per head for neox on bf16
    template <typename acc_t_, typename std::enable_if<!std::is_same_v<acc_t_, scalar_t>, void>::type * = nullptr>
    __aicore__ inline void
    neox_compute(LocT src, LocDst dst, AscendC::LocalTensor<acc_t_> sin, AscendC::LocalTensor<acc_t_> cos,
                 AscendC::LocalTensor<acc_t_> upcastInputBuffer, AscendC::LocalTensor<acc_t_> calcTmpBuffer)
    {
        // slice dst
        LocDst dstX = dst;
        LocDst dstY = dst[embedDim_];

        // slice src
        LocT srcX = src;
        LocT srcY = src[embedDim_];

        // slice temp buffer
        LocAcc calcTmpBufferX = calcTmpBuffer;
        LocAcc calcTmpBufferY = calcTmpBuffer[embedDim_];

        // slice upcast input buffer
        LocAcc upcastBufferX = upcastInputBuffer;
        LocAcc upcastBufferY = upcastBufferX[embedDim_];

        // dst x calc
        Cast(upcastInputBuffer, src, AscendC::RoundMode::CAST_NONE, headSize_);
        Mul(calcTmpBufferX, upcastBufferX, cos, embedDim_);
        Mul(calcTmpBufferY, upcastBufferY, sin, embedDim_);
        Sub(calcTmpBufferX, calcTmpBufferX, calcTmpBufferY, embedDim_);
        Cast(dstX, calcTmpBufferX, AscendC::RoundMode::CAST_TRUNC, embedDim_);

        // dst y calc
        Mul(calcTmpBufferX, upcastBufferX, sin, embedDim_);
        Mul(calcTmpBufferY, upcastBufferY, cos, embedDim_);
        Add(calcTmpBufferX, calcTmpBufferX, calcTmpBufferY, embedDim_);
        Cast(dstY, calcTmpBufferX, AscendC::RoundMode::CAST_TRUNC, embedDim_);
    }

    // compute per head output for neox
    template <typename acc_t_, typename std::enable_if<std::is_same_v<acc_t_, scalar_t>, void>::type * = nullptr>
    __aicore__ inline void
    neox_compute(LocT src, LocDst dst, AscendC::LocalTensor<acc_t_> sin, AscendC::LocalTensor<acc_t_> cos,
                 AscendC::LocalTensor<acc_t_> upcastInputBuffer, AscendC::LocalTensor<acc_t_> calcTmpBuffer)
    {
        // slice dst buffer
        LocDst dstX = dst;
        LocDst dstY = dst[embedDim_];
        // slice src buffer
        LocT srcX = src;
        LocT srcY = src[embedDim_];
        // slice temp buffer
        LocAcc calcTmpBufferX = calcTmpBuffer;
        LocAcc calcTmpBufferY = calcTmpBuffer[embedDim_];

        // dst x calc
        Mul(calcTmpBufferX, srcX, cos, embedDim_);
        Mul(calcTmpBufferY, srcY, sin, embedDim_);
        Sub(dstX, calcTmpBufferX, calcTmpBufferY, embedDim_);

        // dst y calc
        Mul(calcTmpBufferX, srcX, sin, embedDim_);
        Mul(calcTmpBufferY, srcY, cos, embedDim_);
        Add(dstY, calcTmpBufferX, calcTmpBufferY, embedDim_);
    }

    __aicore__ inline void compute_tensor(AscendC::GlobalTensor<scalar_t> srcG, AscendC::GlobalTensor<dst_t> dstG,
                                          LocAcc localCos, LocAcc localSin, LocAcc upcastInputBuffer,
                                          LocAcc calcTmpBuffer, int loopCnt, int tailHeads, int loadStride,
                                          int headNumPerLoad)
    {
        for (int loopNum = 0; loopNum < loopCnt; ++loopNum) {
            LocT src = inQue_.AllocTensor<scalar_t>();
            LocDst dst = outQue_.AllocTensor<dst_t>();
            int repeat_cnt = (loadStride + 127) / 128;
            AscendC::DataCopy(src, srcG[loopNum * loadStride], loadStride);
            inQue_.EnQue(src);

            LocT srcDeque = inQue_.DeQue<scalar_t>();
            if constexpr (!std::is_same_v<scalar_t, acc_t>) {
                int elem_num = loadStride / sizeof(scalar_t);
                AscendC::LocalTensor<acc_t> upBuffer = copyBuf_.GetWithOffset<acc_t>(elem_num, 0);
                Cast(upBuffer, srcDeque, AscendC::RoundMode::CAST_TRUNC, elem_num);
                Cast(dst, upBuffer, AscendC::RoundMode::CAST_TRUNC, elem_num);
            } else {
                smem2smem(dst, srcDeque, loadStride);
            }
            for (int i = 0; i < headNumPerLoad; ++i) {
                neox_compute(srcDeque[i * headSize_], dst[i * headSize_], localSin, localCos, upcastInputBuffer,
                             calcTmpBuffer);
            }
            outQue_.EnQue(dst);
            LocDst dstDeque = outQue_.DeQue<dst_t>();
            AscendC::DataCopy(dstG[loopNum * loadStride], dstDeque, loadStride);
            outQue_.FreeTensor(dstDeque);
            inQue_.FreeTensor(srcDeque);
        }
        // process tail
        {
            LocT src = inQue_.AllocTensor<scalar_t>();
            LocDst dst = outQue_.AllocTensor<dst_t>();
            int repeat_cnt = (tailHeads * headSize_ * sizeof(scalar_t) + 255) / 256;
            AscendC::DataCopy(src, srcG[loopCnt * loadStride], tailHeads * headSize_);
            inQue_.EnQue(src);
            LocT srcDeque = inQue_.DeQue<scalar_t>();

            if constexpr (!std::is_same_v<scalar_t, acc_t>) {
                int elem_num = tailHeads * headSize_ / sizeof(scalar_t);
                AscendC::LocalTensor<acc_t> upBuffer = copyBuf_.GetWithOffset<acc_t>(elem_num, 0);
                Cast(upBuffer, srcDeque, AscendC::RoundMode::CAST_TRUNC, elem_num);
                Cast(dst, upBuffer, AscendC::RoundMode::CAST_TRUNC, elem_num);
            } else {
                smem2smem(dst, srcDeque, tailHeads * headSize_);
            }

            for (int i = 0; i < tailHeads; ++i) {
                neox_compute(srcDeque[i * headSize_], dst[i * headSize_], localSin, localCos, upcastInputBuffer,
                             calcTmpBuffer);
            }
            outQue_.EnQue(dst);
            LocDst dstDeque = outQue_.DeQue<dst_t>();
            AscendC::DataCopy(dstG[loopCnt * loadStride], dstDeque, tailHeads * headSize_);
            outQue_.FreeTensor(dstDeque);
            inQue_.FreeTensor(srcDeque);
        }
    }

    __aicore__ inline void compute()
    {
        LocT cosSinLocal = inQueSinCos_.AllocTensor<scalar_t>();

        AscendC::DataCopy(cosSinLocal, cosSin_, embedDim_ * BUFFER_NUM);

        inQueSinCos_.EnQue(cosSinLocal);
        LocT localSinCosDeque = inQueSinCos_.DeQue<scalar_t>();
        LocT localCos = localSinCosDeque;
        LocT localSin = localSinCosDeque[embedDim_];

        LocAcc calcTmpBuffer;
        LocAcc upcastInputBuffer;
        LocAcc upcastTempBuffer;
        LocAcc cosSinUpcastBuffer;
        LocAcc scaleBuffer;
        LocAcc offsetBuffer;
        calcTmpBuffer = calcBuf_.GetWithOffset<acc_t>(embedDim_ * BUFFER_NUM, calcTmpBufferOffset_);
        upcastInputBuffer = calcBuf_.GetWithOffset<acc_t>(headSize_, upcastInputBufferOffset_);
        upcastTempBuffer = calcBuf_.GetWithOffset<acc_t>(embedDim_ * BUFFER_NUM, upcastTempBufferOffset_);
        cosSinUpcastBuffer = calcBuf_.GetWithOffset<acc_t>(embedDim_ * BUFFER_NUM, cosSinUpcastBufferOffset_);

        LocAcc cosAccBuffer;
        LocAcc sinAccBuffer;

        if constexpr (!std::is_same_v<scalar_t, acc_t>) {
            Cast(cosSinUpcastBuffer, localSinCosDeque, AscendC::RoundMode::CAST_NONE, BUFFER_NUM * embedDim_);
            cosAccBuffer = cosSinUpcastBuffer;
            sinAccBuffer = cosSinUpcastBuffer[embedDim_];
        } else {
            cosAccBuffer = localCos;
            sinAccBuffer = localSin;
        }

        constexpr const int loadSizeByElem = loadSize / sizeof(scalar_t);
        int64_t headNumPerLoad = loadSizeByElem / headSize_;
        int64_t loopCnt = numHeads_ / headNumPerLoad;
        int64_t tailHeads = numHeads_ - loopCnt * headNumPerLoad;
        int64_t loadStride = headNumPerLoad * headSize_;
        int64_t loopCntKv = numKvHeads_ / headNumPerLoad;
        int64_t tailHeadsKv = numKvHeads_ - loopCntKv * headNumPerLoad;
        compute_tensor(query_, queryDst_, cosAccBuffer, sinAccBuffer, upcastInputBuffer,
                       calcTmpBuffer, loopCnt, tailHeads, loadStride, headNumPerLoad);

        compute_tensor(key_, keyDst_, cosAccBuffer, sinAccBuffer, upcastInputBuffer, calcTmpBuffer,
                       loopCntKv, tailHeadsKv, loadStride, headNumPerLoad);

        inQueSinCos_.FreeTensor(localSinCosDeque);
    }

private:
    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue_, inQueSinCos_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> copyBuf_;
    AscendC::GlobalTensor<dst_t> queryDst_;
    AscendC::GlobalTensor<dst_t> keyDst_;
    AscendC::GlobalTensor<scalar_t> query_;
    AscendC::GlobalTensor<scalar_t> key_;
    AscendC::GlobalTensor<scalar_t> cosSin_;
    int rotDim_;
    int embedDim_;
    int64_t queryStride_;
    int64_t keyStride_;
    int64_t dstQueryStride_;
    int64_t dstKeyStride_;
    int numHeads_;
    int numKvHeads_;
    int headSize_;
    int calcTmpBufferOffset_;
    int upcastInputBufferOffset_;
    int upcastTempBufferOffset_;
    int cosSinUpcastBufferOffset_;
    int tempBufferSize_;
};

#define ROPE_CUSTOM_KERNEL_INSTANTIATION(TYPE, NEOX)                                                                 \
    extern "C" __global__ __aicore__ void rope_custom_##NEOX##_##TYPE(                                               \
        __gm__ int64_t* positions, __gm__ void* queryDst, __gm__ void* keyDst, __gm__ TYPE* query, __gm__ TYPE* key, \
        __gm__ TYPE* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride,              \
        const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads,          \
        const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)                           \
    {                                                                                                                \
        AscendC::TPipe pipe;                                                                                         \
        RotaryEmbedding<TYPE, NEOX> op{};                                                                            \
        op.init(positions, queryDst, keyDst, query, key, cosSinCache, rotDim, dstQueryStride, dstKeyStride,          \
                queryStride, keyStride, numHeads, numKvHeads, headSize, &pipe);                                      \
        for (int64_t i = AscendC::GetBlockIdx(); i < numTokens; i += coreNum) {                                      \
            op.update(positions, queryDst, keyDst, query, key, cosSinCache, rotDim, dstQueryStride, dstKeyStride,    \
                      queryStride, keyStride, numHeads, numKvHeads, headSize, i);                                    \
            op.compute();                                                                                            \
        }                                                                                                            \
    }

#define ITERMEDIATE_EXPAND(TYPE, NEOX) ROPE_CUSTOM_KERNEL_INSTANTIATION(TYPE, NEOX)

#define ROPE_CUSTOM_KERNEL(TYPE)    \
    ITERMEDIATE_EXPAND(TYPE, true); \
    ITERMEDIATE_EXPAND(TYPE, false);

ROPE_CUSTOM_KERNEL(half)
ROPE_CUSTOM_KERNEL(bfloat16_t)

namespace vllm_ascend {

#define ROPE_KERNEL_CALL(TYPE)                                                                                   \
    if (isNeox)                                                                                                  \
        rope_custom_true_##TYPE<<<blockDim, nullptr, stream>>>(                                                  \
            positions, queryDst, keyDst, reinterpret_cast<TYPE *>(query), reinterpret_cast<TYPE *>(key),         \
            reinterpret_cast<TYPE *>(cosSinCache), rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, \
            numHeads, numKvHeads, headSize, numTokens, loopCnt, blockDim);                                       \
    else                                                                                                         \
        rope_custom_false_##TYPE<<<blockDim, nullptr, stream>>>(                                                 \
            positions, queryDst, keyDst, reinterpret_cast<TYPE *>(query), reinterpret_cast<TYPE *>(key),         \
            reinterpret_cast<TYPE *>(cosSinCache), rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, \
            numHeads, numKvHeads, headSize, numTokens, loopCnt, blockDim);

static const int64_t maxParallelSize = 65535;

extern void rotary_embedding_kernel(AscendType type, bool isNeox, void *stream, int64_t *positions, void *queryDst,
                                    void *keyDst, void *query, void *key, void *cosSinCache, const int rotDim,
                                    const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride,
                                    const int64_t dstKeyStride, const int numHeads, const int numKvHeads,
                                    const int headSize, const int64_t numTokens, const uint32_t loopCnt,
                                    uint32_t aivNum)
{

    int blockDim = maxParallelSize > numTokens ? numTokens : maxParallelSize;

    if (type == AscendType::FP16) {
        ROPE_KERNEL_CALL(half);
    } else if (type == AscendType::BF16) {
        ROPE_KERNEL_CALL(bfloat16_t);
    } else {
        return;
    }
}

} // namespace vllm_ascend