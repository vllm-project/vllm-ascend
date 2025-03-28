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


using vllm_ascend::AccType;
using vllm_ascend::local_mem_copy;
template <typename scalar_t, bool isNeox> class RotaryEmbedding {
    // NOTE(ganyi): we use 32K as load stride for pipe, need to find another way to
    // retrive this size from runtime for more Soc support
    static int constexpr loadSize = 1024 * 4;
    using dst_t = scalar_t;
    using acc_t = typename AccType<scalar_t>::type;
    // only half tensor have cast instruct to int8, hardcode acc_dst_t as half
    using local_scalar_t = AscendC::LocalTensor<scalar_t>;
    using local_acc_t = AscendC::LocalTensor<acc_t>;
    using local_dst_t = AscendC::LocalTensor<dst_t>;

public:
    __aicore__ inline RotaryEmbedding()
    {
    }

    // Allocate buffers for input and output queue and the temp buffer used during kernel compute process,
    // this init process happens only in the kernel compute on a single vector core.
    __aicore__ inline void init(__gm__ int64_t *positions, __gm__ void *queryDst, __gm__ void *keyDst,
                                __gm__ scalar_t *query, __gm__ scalar_t *key, __gm__ scalar_t *cosSinCache,
                                const int rotDim, const int64_t dstQueryStride,
                                const int64_t dstKeyStride, const int64_t queryStride, const int64_t keyStride,
                                const int numHeads, const int numKvHeads, const int headSize, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        rotDim_ = rotDim;
        // query stride and key stride is used to handle the strided tensor which is not contiguous on num_tokens dim
        queryStride_ = queryStride;
        keyStride_ = keyStride;
        dstQueryStride_ = dstQueryStride;
        dstKeyStride_ = dstKeyStride;
        numHeads_ = numHeads;
        numKvHeads_ = numKvHeads;
        headSize_ = headSize;
        embedDim_ = rotDim / 2;

        pipe_->InitBuffer(inQue_, 1 /* buffer_num */, loadSize /* buffer_size */);
        pipe_->InitBuffer(inQueSinCos_, 1 /* buffer_num */, rotDim_ * sizeof(scalar_t) /* buffer_size */);
        pipe_->InitBuffer(outQue_, 1 /* buffer_num */, loadSize /* buffer_size */);
        // 2 temperary calculation buffer
        calcTmpBufferOffset_ = 0;
        // 1 upcast buffer for bf16 (headSize)
        upcastInputBufferOffset_ = calcTmpBufferOffset_ + sizeof(acc_t) * embedDim_ * 2;
        // 1 upcast temp buffer for bf16 (2 * embed_dim)
        upcastTempBufferOffset_ = upcastInputBufferOffset_ + sizeof(acc_t) * headSize_;
        // 2 sin cos upcast buffer for bf16
        cosSinUpcastBufferOffset_ = upcastTempBufferOffset_ + sizeof(acc_t) * 2 * embedDim_;
        // 2. bf16 path: needs 2 cos sin upcast buffer size
        // 3. fp16 path: needs 2 temperary calculation buffer size
        tempBufferSize_ = cosSinUpcastBufferOffset_ + 2 * embedDim_ * sizeof(acc_t);
        // need to consider upcast the bf16 to fp32, so we might need 4 buffer just in case
        // 2 temperary buffer, 2 input buffer, 1 cos buffer, 1 sin buffer, 2 scale buffer (headSize), 2 zp
        // buffer(headSize int8), 1 dst_temp buffer(headSize, int32)
        pipe_->InitBuffer(calcBuf_, tempBufferSize_ /* buffer_size */);
        if constexpr (!std::is_same_v<scalar_t, acc_t>) {
            pipe_->InitBuffer(copyBuf_, loadSize);
        }
    }
    __aicore__ inline void update_mem_offset(__gm__ int64_t *positions, __gm__ void *queryDst, __gm__ void *keyDst,
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
    neox_compute(local_scalar_t src, local_dst_t dst, AscendC::LocalTensor<acc_t_> sin, AscendC::LocalTensor<acc_t_> cos,
                 AscendC::LocalTensor<acc_t_> upcastInputBuffer, AscendC::LocalTensor<acc_t_> calcTmpBuffer)
    {
        // slice dst
        local_dst_t dstX = dst;
        local_dst_t dstY = dst[embedDim_];

        // slice src
        local_scalar_t srcX = src;
        local_scalar_t srcY = src[embedDim_];

        // slice temp buffer
        local_acc_t calcTmpBufferX = calcTmpBuffer;
        local_acc_t calcTmpBufferY = calcTmpBuffer[embedDim_];

        // slice upcast input buffer
        local_acc_t upcastBufferX = upcastInputBuffer;
        local_acc_t upcastBufferY = upcastBufferX[embedDim_];

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
    neox_compute(local_scalar_t src, local_dst_t dst, AscendC::LocalTensor<acc_t_> sin, AscendC::LocalTensor<acc_t_> cos,
                 AscendC::LocalTensor<acc_t_> upcastInputBuffer, AscendC::LocalTensor<acc_t_> calcTmpBuffer)
    {
        // slice dst buffer
        local_dst_t dstX = dst;
        local_dst_t dstY = dst[embedDim_];
        // slice src buffer
        local_scalar_t srcX = src;
        local_scalar_t srcY = src[embedDim_];
        // slice temp buffer
        local_acc_t calcTmpBufferX = calcTmpBuffer;
        local_acc_t calcTmpBufferY = calcTmpBuffer[embedDim_];

        // dst x calc
        Mul(calcTmpBufferX, srcX, cos, embedDim_);
        Mul(calcTmpBufferY, srcY, sin, embedDim_);
        Sub(dstX, calcTmpBufferX, calcTmpBufferY, embedDim_);

        // dst y calc
        Mul(calcTmpBufferX, srcX, sin, embedDim_);
        Mul(calcTmpBufferY, srcY, cos, embedDim_);
        Add(dstY, calcTmpBufferX, calcTmpBufferY, embedDim_);
    }

    __aicore__ inline void compute_qk(AscendC::GlobalTensor<scalar_t> srcG, AscendC::GlobalTensor<dst_t> dstG,
                                          local_acc_t localCos, local_acc_t localSin, local_acc_t upcastInputBuffer,
                                          local_acc_t calcTmpBuffer, int loopCnt, int tailHeads, int loadStride,
                                          int headNumPerLoad)
    {
        for (int loopNum = 0; loopNum < loopCnt; ++loopNum) {
            local_scalar_t src = inQue_.AllocTensor<scalar_t>();
            local_dst_t dst = outQue_.AllocTensor<dst_t>();
            AscendC::DataCopy(src, srcG[loopNum * loadStride], loadStride);
            inQue_.EnQue(src);

            local_scalar_t srcDeque = inQue_.DeQue<scalar_t>();
            if constexpr (!std::is_same_v<scalar_t, acc_t>) {
                int elem_num = loadStride / sizeof(scalar_t);
                AscendC::LocalTensor<acc_t> upBuffer = copyBuf_.GetWithOffset<acc_t>(elem_num, 0);
                Cast(upBuffer, srcDeque, AscendC::RoundMode::CAST_TRUNC, elem_num);
                Cast(dst, upBuffer, AscendC::RoundMode::CAST_TRUNC, elem_num);
            } else {
                local_mem_copy(dst, srcDeque, loadStride);
            }
            for (int i = 0; i < headNumPerLoad; ++i) {
                neox_compute(srcDeque[i * headSize_], dst[i * headSize_], localSin, localCos, upcastInputBuffer,
                             calcTmpBuffer);
            }
            outQue_.EnQue(dst);
            local_dst_t dstDeque = outQue_.DeQue<dst_t>();
            AscendC::DataCopy(dstG[loopNum * loadStride], dstDeque, loadStride);
            outQue_.FreeTensor(dstDeque);
            inQue_.FreeTensor(srcDeque);
        }
        // process tail
        {
            local_scalar_t src = inQue_.AllocTensor<scalar_t>();
            local_dst_t dst = outQue_.AllocTensor<dst_t>();

            AscendC::DataCopy(src, srcG[loopCnt * loadStride], tailHeads * headSize_);
            inQue_.EnQue(src);
            local_scalar_t srcDeque = inQue_.DeQue<scalar_t>();

            if constexpr (!std::is_same_v<scalar_t, acc_t>) {
                int elem_num = tailHeads * headSize_ / sizeof(scalar_t);
                AscendC::LocalTensor<acc_t> upBuffer = copyBuf_.GetWithOffset<acc_t>(elem_num, 0);
                Cast(upBuffer, srcDeque, AscendC::RoundMode::CAST_TRUNC, elem_num);
                Cast(dst, upBuffer, AscendC::RoundMode::CAST_TRUNC, elem_num);
            } else {
                local_mem_copy(dst, srcDeque, tailHeads * headSize_);
            }

            for (int i = 0; i < tailHeads; ++i) {
                neox_compute(srcDeque[i * headSize_], dst[i * headSize_], localSin, localCos, upcastInputBuffer,
                             calcTmpBuffer);
            }
            outQue_.EnQue(dst);
            local_dst_t dstDeque = outQue_.DeQue<dst_t>();
            AscendC::DataCopy(dstG[loopCnt * loadStride], dstDeque, tailHeads * headSize_);
            outQue_.FreeTensor(dstDeque);
            inQue_.FreeTensor(srcDeque);
        }
    }

    __aicore__ inline void compute_function()
    {
        local_scalar_t cosSinLocal = inQueSinCos_.AllocTensor<scalar_t>();

        AscendC::DataCopy(cosSinLocal, cosSin_, embedDim_ * 2);

        inQueSinCos_.EnQue(cosSinLocal);
        local_scalar_t localSinCosDeque = inQueSinCos_.DeQue<scalar_t>();
        local_scalar_t localCos = localSinCosDeque;
        local_scalar_t localSin = localSinCosDeque[embedDim_];

        local_acc_t calcTmpBuffer;
        local_acc_t upcastInputBuffer;
        local_acc_t upcastTempBuffer;
        local_acc_t cosSinUpcastBuffer;
        local_acc_t scaleBuffer;
        local_acc_t offsetBuffer;
        calcTmpBuffer = calcBuf_.GetWithOffset<acc_t>(embedDim_ * 2, calcTmpBufferOffset_);
        upcastInputBuffer = calcBuf_.GetWithOffset<acc_t>(headSize_, upcastInputBufferOffset_);
        upcastTempBuffer = calcBuf_.GetWithOffset<acc_t>(embedDim_ * 2, upcastTempBufferOffset_);
        cosSinUpcastBuffer = calcBuf_.GetWithOffset<acc_t>(embedDim_ * 2, cosSinUpcastBufferOffset_);

        local_acc_t cosAccBuffer;
        local_acc_t sinAccBuffer;

        if constexpr (!std::is_same_v<scalar_t, acc_t>) {
            Cast(cosSinUpcastBuffer, localSinCosDeque, AscendC::RoundMode::CAST_NONE, 2 * embedDim_);
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
        compute_qk(query_, queryDst_, cosAccBuffer, sinAccBuffer, upcastInputBuffer,
                       calcTmpBuffer, loopCnt, tailHeads, loadStride, headNumPerLoad);

        compute_qk(key_, keyDst_, cosAccBuffer, sinAccBuffer, upcastInputBuffer, calcTmpBuffer,
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

// Note: Need to use macro to instaniate all the target functions here, for the current build system dose not support template call in cpp
// We use C style symbol here for kernel compilation, cpp style kernel entry may lead to compilation failure
#define ROPE_CUSTOM_KERNEL_TYPE_DECLARE(TYPE, NEOX)                                                                            \
    extern "C" __global__ __aicore__ void rope_custom_##NEOX##_##TYPE(                                                          \
        __gm__ int64_t* positions, __gm__ void* queryDst, __gm__ void* keyDst, __gm__ TYPE* query, __gm__ TYPE* key,            \
        __gm__ TYPE* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride,                         \
        const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads,                     \
        const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)                                      \
    {                                                                                                                           \
        AscendC::TPipe pipe;                                                                                                    \
        RotaryEmbedding<TYPE, NEOX> op{};                                                                                       \
        op.init(positions, queryDst, keyDst, query, key, cosSinCache, rotDim, dstQueryStride, dstKeyStride,                     \
                queryStride, keyStride, numHeads, numKvHeads, headSize, &pipe);                                                 \
        for (int64_t i = AscendC::GetBlockIdx(); i < numTokens; i += coreNum) {                                                 \
            op.update_mem_offset(positions, queryDst, keyDst, query, key, cosSinCache, rotDim, dstQueryStride, dstKeyStride,    \
                      queryStride, keyStride, numHeads, numKvHeads, headSize, i);                                               \
            op.compute_function();                                                                                              \
        }                                                                                                                       \
    }

#define ROPE_CUSTOM_KERNEL_DECLARE(TYPE)    \
    ROPE_CUSTOM_KERNEL_TYPE_DECLARE(TYPE, true); \
    ROPE_CUSTOM_KERNEL_TYPE_DECLARE(TYPE, false);

// Declare all the kernel entry here
ROPE_CUSTOM_KERNEL_DECLARE(half)
ROPE_CUSTOM_KERNEL_DECLARE(bfloat16_t)

namespace vllm_ascend {

#define ROTARY_EMBEDDING_KERNEL_CALL(TYPE)                                                                       \
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

extern void rotary_embedding_impl(AscendType type, bool isNeox, void *stream, int64_t *positions, void *queryDst,
                                    void *keyDst, void *query, void *key, void *cosSinCache, const int rotDim,
                                    const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride,
                                    const int64_t dstKeyStride, const int numHeads, const int numKvHeads,
                                    const int headSize, const int64_t numTokens, const uint32_t loopCnt,
                                    uint32_t aivNum)
{

    int blockDim = maxParallelSize > numTokens ? numTokens : maxParallelSize;
    if (type == AscendType::FP16) {
        ROTARY_EMBEDDING_KERNEL_CALL(half);
    } else if (type == AscendType::BF16) {
        ROTARY_EMBEDDING_KERNEL_CALL(bfloat16_t);
    } else {
        return;
    }
}

} // namespace vllm_ascend