/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#include "kernel_type.h"
#include "types.h"

namespace {

template <typename scalar_t, typename pos_t>
class InterleaveRopeByCache {
public:
    __aicore__ inline InterleaveRopeByCache() {}

    __aicore__ inline void Init(
        __gm__ scalar_t* qk,
        __gm__ pos_t* positions,
        __gm__ scalar_t* cos_sin_cache,
        __gm__ scalar_t* output,
        int64_t num_tokens,
        int64_t num_heads,
        int64_t head_dim,
        int64_t qk_row_stride,
        int64_t qk_head_stride,
        int64_t cos_sin_row_stride,
        bool is_neox_style)
    {
        qk_gm_.SetGlobalBuffer(qk);
        positions_gm_.SetGlobalBuffer(positions);
        cos_sin_cache_gm_.SetGlobalBuffer(cos_sin_cache);
        output_gm_.SetGlobalBuffer(output);

        num_tokens_ = num_tokens;
        num_heads_ = num_heads;
        head_dim_ = head_dim;
        half_dim_ = head_dim / 2;
        qk_row_stride_ = qk_row_stride;
        qk_head_stride_ = qk_head_stride;
        cos_sin_row_stride_ = cos_sin_row_stride;
        (void)is_neox_style;

        pipe_.InitBuffer(qk_queue_, 1, head_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(cache_queue_, 1, head_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(out_queue_, 1, head_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(even_buf_, half_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(odd_buf_, half_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(even_fp32_buf_, half_dim_ * sizeof(float));
        pipe_.InitBuffer(odd_fp32_buf_, half_dim_ * sizeof(float));
        pipe_.InitBuffer(cos_fp32_buf_, half_dim_ * sizeof(float));
        pipe_.InitBuffer(sin_fp32_buf_, half_dim_ * sizeof(float));
        pipe_.InitBuffer(tmp_fp32_buf_, half_dim_ * sizeof(float));
        pipe_.InitBuffer(out_first_fp32_buf_, half_dim_ * sizeof(float));
        pipe_.InitBuffer(out_second_fp32_buf_, half_dim_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t total_heads = num_tokens_ * num_heads_;
        int64_t block_num = AscendC::GetBlockNum();
        for (int64_t linear_idx = AscendC::GetBlockIdx(); linear_idx < total_heads; linear_idx += block_num) {
            int64_t token_idx = linear_idx / num_heads_;
            int64_t head_idx = linear_idx - token_idx * num_heads_;
            CopyIn(token_idx, head_idx);
            Compute();
            CopyOut(linear_idx);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t token_idx, int64_t head_idx)
    {
        AscendC::LocalTensor<scalar_t> qk_local = qk_queue_.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> cache_local = cache_queue_.AllocTensor<scalar_t>();

        int64_t qk_offset = token_idx * qk_row_stride_ + head_idx * qk_head_stride_;
        pos_t position = positions_gm_.GetValue(token_idx);
        int64_t cache_offset = static_cast<int64_t>(position) * cos_sin_row_stride_;

        AscendC::DataCopy(qk_local, qk_gm_[qk_offset], head_dim_);
        AscendC::DataCopy(cache_local, cos_sin_cache_gm_[cache_offset], head_dim_);
        qk_queue_.EnQue(qk_local);
        cache_queue_.EnQue(cache_local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<scalar_t> qk_local = qk_queue_.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> cache_local = cache_queue_.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> out_local = out_queue_.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> even_local = even_buf_.Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> odd_local = odd_buf_.Get<scalar_t>();
        AscendC::LocalTensor<float> even_fp32 = even_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> odd_fp32 = odd_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> cos_fp32 = cos_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> sin_fp32 = sin_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> tmp_fp32 = tmp_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> out_first_fp32 = out_first_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> out_second_fp32 = out_second_fp32_buf_.Get<float>();

        for (int64_t dim = 0; dim < half_dim_; ++dim) {
            even_local.SetValue(dim, qk_local.GetValue(2 * dim));
            odd_local.SetValue(dim, qk_local.GetValue(2 * dim + 1));
        }

        AscendC::Cast(even_fp32, even_local, AscendC::RoundMode::CAST_NONE, half_dim_);
        AscendC::Cast(odd_fp32, odd_local, AscendC::RoundMode::CAST_NONE, half_dim_);
        AscendC::Cast(cos_fp32, cache_local, AscendC::RoundMode::CAST_NONE, half_dim_);
        AscendC::Cast(sin_fp32, cache_local[half_dim_], AscendC::RoundMode::CAST_NONE, half_dim_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(out_first_fp32, even_fp32, cos_fp32, half_dim_);
        AscendC::Mul(tmp_fp32, odd_fp32, sin_fp32, half_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(out_first_fp32, out_first_fp32, tmp_fp32, half_dim_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(out_second_fp32, odd_fp32, cos_fp32, half_dim_);
        AscendC::Mul(tmp_fp32, even_fp32, sin_fp32, half_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(out_second_fp32, out_second_fp32, tmp_fp32, half_dim_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(even_local, out_first_fp32, AscendC::RoundMode::CAST_RINT, half_dim_);
        AscendC::Cast(odd_local, out_second_fp32, AscendC::RoundMode::CAST_RINT, half_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t dim = 0; dim < half_dim_; ++dim) {
            out_local.SetValue(dim, even_local.GetValue(dim));
            out_local.SetValue(half_dim_ + dim, odd_local.GetValue(dim));
        }

        out_queue_.EnQue(out_local);
        qk_queue_.FreeTensor(qk_local);
        cache_queue_.FreeTensor(cache_local);
    }

    __aicore__ inline void CopyOut(int64_t linear_idx)
    {
        AscendC::LocalTensor<scalar_t> out_local = out_queue_.DeQue<scalar_t>();
        AscendC::DataCopy(output_gm_[linear_idx * head_dim_], out_local, head_dim_);
        out_queue_.FreeTensor(out_local);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> qk_queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> cache_queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_queue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> even_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> odd_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> even_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> odd_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cos_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sin_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> out_first_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> out_second_fp32_buf_;
    AscendC::GlobalTensor<scalar_t> qk_gm_;
    AscendC::GlobalTensor<pos_t> positions_gm_;
    AscendC::GlobalTensor<scalar_t> cos_sin_cache_gm_;
    AscendC::GlobalTensor<scalar_t> output_gm_;
    int64_t num_tokens_;
    int64_t num_heads_;
    int64_t head_dim_;
    int64_t half_dim_;
    int64_t qk_row_stride_;
    int64_t qk_head_stride_;
    int64_t cos_sin_row_stride_;
};

#define INTERLEAVE_ROPE_BY_CACHE_TYPE_DECLARE(TYPE, POS_TYPE, SUFFIX)                                                \
    extern "C" __global__ __aicore__ void interleave_rope_by_cache_##TYPE##_##SUFFIX(                              \
        __gm__ TYPE* qk, __gm__ POS_TYPE* positions, __gm__ TYPE* cos_sin_cache, __gm__ TYPE* output,                \
        int64_t num_tokens, int64_t num_heads, int64_t head_dim, int64_t qk_row_stride, int64_t qk_head_stride,      \
        int64_t cos_sin_row_stride, bool is_neox_style)                                                              \
    {                                                                                                                \
        InterleaveRopeByCache<TYPE, POS_TYPE> op;                                                                    \
        op.Init(qk, positions, cos_sin_cache, output, num_tokens, num_heads, head_dim, qk_row_stride,                \
                qk_head_stride, cos_sin_row_stride, is_neox_style);                                                  \
        op.Process();                                                                                                \
    }

INTERLEAVE_ROPE_BY_CACHE_TYPE_DECLARE(half, int32_t, int32)
INTERLEAVE_ROPE_BY_CACHE_TYPE_DECLARE(half, int64_t, int64)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
INTERLEAVE_ROPE_BY_CACHE_TYPE_DECLARE(bfloat16_t, int32_t, int32)
INTERLEAVE_ROPE_BY_CACHE_TYPE_DECLARE(bfloat16_t, int64_t, int64)
#endif

}  // namespace

namespace vllm_ascend {

void interleave_rope_by_cache_impl(
    AscendType type,
    bool positions_is_int32,
    void* stream,
    void* qk,
    void* positions,
    void* cos_sin_cache,
    void* output,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t qk_row_stride,
    int64_t qk_head_stride,
    int64_t cos_sin_row_stride,
    bool is_neox_style,
    uint32_t block_dim)
{
    if (type == AscendType::FP16) {
        if (positions_is_int32) {
            interleave_rope_by_cache_half_int32<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(qk), static_cast<int32_t*>(positions), static_cast<half*>(cos_sin_cache),
                static_cast<half*>(output), num_tokens, num_heads, head_dim, qk_row_stride, qk_head_stride,
                cos_sin_row_stride, is_neox_style);
        } else {
            interleave_rope_by_cache_half_int64<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(qk), static_cast<int64_t*>(positions), static_cast<half*>(cos_sin_cache),
                static_cast<half*>(output), num_tokens, num_heads, head_dim, qk_row_stride, qk_head_stride,
                cos_sin_row_stride, is_neox_style);
        }
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        if (positions_is_int32) {
            interleave_rope_by_cache_bfloat16_t_int32<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(qk), static_cast<int32_t*>(positions), static_cast<bfloat16_t*>(cos_sin_cache),
                static_cast<bfloat16_t*>(output), num_tokens, num_heads, head_dim, qk_row_stride, qk_head_stride,
                cos_sin_row_stride, is_neox_style);
        } else {
            interleave_rope_by_cache_bfloat16_t_int64<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(qk), static_cast<int64_t*>(positions), static_cast<bfloat16_t*>(cos_sin_cache),
                static_cast<bfloat16_t*>(output), num_tokens, num_heads, head_dim, qk_row_stride, qk_head_stride,
                cos_sin_row_stride, is_neox_style);
        }
#endif
    }
}

}  // namespace vllm_ascend
