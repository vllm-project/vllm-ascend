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

constexpr int64_t REDUCE_TMP_BYTES = 32 * 1024;

template <typename scalar_t, typename pos_t, typename slot_t>
class KvRmsNormRopeCacheByCache {
public:
    __aicore__ inline KvRmsNormRopeCacheByCache() {}

    __aicore__ inline void Init(
        __gm__ scalar_t* kv,
        __gm__ scalar_t* weight,
        __gm__ pos_t* positions,
        __gm__ scalar_t* cos_sin_cache,
        __gm__ slot_t* slots,
        __gm__ scalar_t* kv_cache_rope,
        __gm__ scalar_t* kv_cache_nope,
        __gm__ scalar_t* out_rope,
        __gm__ scalar_t* out_nope,
        int64_t num_tokens,
        int64_t kv_row_stride,
        int64_t cos_sin_row_stride,
        int64_t cache_block_size,
        int64_t nope_dim,
        int64_t rope_dim,
        float epsilon,
        bool is_neox_style,
        bool is_output_kv,
        bool cache_mode_is_nz)
    {
        kv_gm_.SetGlobalBuffer(kv);
        weight_gm_.SetGlobalBuffer(weight);
        positions_gm_.SetGlobalBuffer(positions);
        cos_sin_cache_gm_.SetGlobalBuffer(cos_sin_cache);
        slots_gm_.SetGlobalBuffer(slots);
        kv_cache_rope_gm_.SetGlobalBuffer(kv_cache_rope);
        kv_cache_nope_gm_.SetGlobalBuffer(kv_cache_nope);
        out_rope_gm_.SetGlobalBuffer(out_rope);
        out_nope_gm_.SetGlobalBuffer(out_nope);

        num_tokens_ = num_tokens;
        kv_row_stride_ = kv_row_stride;
        cos_sin_row_stride_ = cos_sin_row_stride;
        cache_block_size_ = cache_block_size;
        nope_dim_ = nope_dim;
        rope_dim_ = rope_dim;
        total_dim_ = nope_dim + rope_dim;
        half_rope_dim_ = rope_dim / 2;
        epsilon_ = epsilon;
        is_neox_style_ = is_neox_style;
        is_output_kv_ = is_output_kv;
        cache_mode_is_nz_ = cache_mode_is_nz;

        pipe_.InitBuffer(kv_queue_, 1, total_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(weight_queue_, 1, nope_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(cache_queue_, 1, rope_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(nope_out_queue_, 1, nope_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(rope_out_queue_, 1, rope_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(nope_fp32_buf_, nope_dim_ * sizeof(float));
        pipe_.InitBuffer(weight_fp32_buf_, nope_dim_ * sizeof(float));
        pipe_.InitBuffer(tmp_fp32_buf_, nope_dim_ * sizeof(float));
        pipe_.InitBuffer(reduce_tmp_buf_, REDUCE_TMP_BYTES);
        pipe_.InitBuffer(sum_buf_, 32);
        pipe_.InitBuffer(even_buf_, half_rope_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(odd_buf_, half_rope_dim_ * sizeof(scalar_t));
        pipe_.InitBuffer(even_fp32_buf_, half_rope_dim_ * sizeof(float));
        pipe_.InitBuffer(odd_fp32_buf_, half_rope_dim_ * sizeof(float));
        pipe_.InitBuffer(cos_fp32_buf_, half_rope_dim_ * sizeof(float));
        pipe_.InitBuffer(sin_fp32_buf_, half_rope_dim_ * sizeof(float));
        pipe_.InitBuffer(rope_tmp_fp32_buf_, half_rope_dim_ * sizeof(float));
        pipe_.InitBuffer(rope_first_fp32_buf_, half_rope_dim_ * sizeof(float));
        pipe_.InitBuffer(rope_second_fp32_buf_, half_rope_dim_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t block_num = AscendC::GetBlockNum();
        for (int64_t token_idx = AscendC::GetBlockIdx(); token_idx < num_tokens_; token_idx += block_num) {
            CopyIn(token_idx);
            Compute();
            CopyOut(token_idx);
        }
    }

protected:
    __aicore__ inline void CopyIn(int64_t token_idx)
    {
        AscendC::LocalTensor<scalar_t> kv_local = kv_queue_.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> weight_local = weight_queue_.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> cache_local = cache_queue_.AllocTensor<scalar_t>();

        pos_t position = positions_gm_.GetValue(token_idx);
        int64_t cache_offset = static_cast<int64_t>(position) * cos_sin_row_stride_;

        AscendC::DataCopy(kv_local, kv_gm_[token_idx * kv_row_stride_], total_dim_);
        AscendC::DataCopy(weight_local, weight_gm_, nope_dim_);
        AscendC::DataCopy(cache_local, cos_sin_cache_gm_[cache_offset], rope_dim_);
        kv_queue_.EnQue(kv_local);
        weight_queue_.EnQue(weight_local);
        cache_queue_.EnQue(cache_local);
    }

    __aicore__ inline void ComputeNope(
        const AscendC::LocalTensor<scalar_t>& kv_local,
        const AscendC::LocalTensor<scalar_t>& weight_local,
        const AscendC::LocalTensor<scalar_t>& nope_out)
    {
        AscendC::LocalTensor<float> nope_fp32 = nope_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> weight_fp32 = weight_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> tmp_fp32 = tmp_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> reduce_tmp = reduce_tmp_buf_.Get<float>();
        AscendC::LocalTensor<float> sum_local = sum_buf_.Get<float>();

        AscendC::Cast(nope_fp32, kv_local, AscendC::RoundMode::CAST_NONE, nope_dim_);
        AscendC::Cast(weight_fp32, weight_local, AscendC::RoundMode::CAST_NONE, nope_dim_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(tmp_fp32, nope_fp32, nope_fp32, nope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float>(sum_local, tmp_fp32, reduce_tmp, nope_dim_);
        AscendC::PipeBarrier<PIPE_V>();

        float sum = sum_local.GetValue(0);
        float rstd = 1.0f / sqrt(sum / static_cast<float>(nope_dim_) + epsilon_);
        AscendC::Mul(nope_fp32, nope_fp32, weight_fp32, nope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(nope_fp32, nope_fp32, rstd, nope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(nope_out, nope_fp32, AscendC::RoundMode::CAST_RINT, nope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadRopePairs(const AscendC::LocalTensor<scalar_t>& rope_in)
    {
        AscendC::LocalTensor<scalar_t> even_local = even_buf_.Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> odd_local = odd_buf_.Get<scalar_t>();
        if (is_neox_style_) {
            for (int64_t dim = 0; dim < half_rope_dim_; ++dim) {
                even_local.SetValue(dim, rope_in.GetValue(dim));
                odd_local.SetValue(dim, rope_in.GetValue(half_rope_dim_ + dim));
            }
        } else {
            for (int64_t dim = 0; dim < half_rope_dim_; ++dim) {
                even_local.SetValue(dim, rope_in.GetValue(2 * dim));
                odd_local.SetValue(dim, rope_in.GetValue(2 * dim + 1));
            }
        }
    }

    __aicore__ inline void LoadRopeCacheFirst(const AscendC::LocalTensor<scalar_t>& cache_local)
    {
        AscendC::LocalTensor<float> cos_fp32 = cos_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> sin_fp32 = sin_fp32_buf_.Get<float>();

        AscendC::Cast(cos_fp32, cache_local, AscendC::RoundMode::CAST_NONE, half_rope_dim_);
        AscendC::Cast(sin_fp32, cache_local[half_rope_dim_], AscendC::RoundMode::CAST_NONE, half_rope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeRopeFromInput(
        const AscendC::LocalTensor<scalar_t>& rope_in,
        const AscendC::LocalTensor<scalar_t>& cache_local,
        const AscendC::LocalTensor<scalar_t>& rope_out)
    {
        AscendC::LocalTensor<scalar_t> even_local = even_buf_.Get<scalar_t>();
        AscendC::LocalTensor<scalar_t> odd_local = odd_buf_.Get<scalar_t>();
        AscendC::LocalTensor<float> even_fp32 = even_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> odd_fp32 = odd_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> cos_fp32 = cos_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> sin_fp32 = sin_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> tmp_fp32 = rope_tmp_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> rope_first_fp32 = rope_first_fp32_buf_.Get<float>();
        AscendC::LocalTensor<float> rope_second_fp32 = rope_second_fp32_buf_.Get<float>();

        LoadRopePairs(rope_in);
        AscendC::Cast(even_fp32, even_local, AscendC::RoundMode::CAST_NONE, half_rope_dim_);
        AscendC::Cast(odd_fp32, odd_local, AscendC::RoundMode::CAST_NONE, half_rope_dim_);
        LoadRopeCacheFirst(cache_local);

        AscendC::Mul(rope_first_fp32, even_fp32, cos_fp32, half_rope_dim_);
        AscendC::Mul(tmp_fp32, odd_fp32, sin_fp32, half_rope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(rope_first_fp32, rope_first_fp32, tmp_fp32, half_rope_dim_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(rope_second_fp32, odd_fp32, cos_fp32, half_rope_dim_);
        AscendC::Mul(tmp_fp32, even_fp32, sin_fp32, half_rope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(rope_second_fp32, rope_second_fp32, tmp_fp32, half_rope_dim_);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Cast(even_local, rope_first_fp32, AscendC::RoundMode::CAST_RINT, half_rope_dim_);
        AscendC::Cast(odd_local, rope_second_fp32, AscendC::RoundMode::CAST_RINT, half_rope_dim_);
        AscendC::PipeBarrier<PIPE_V>();
        for (int64_t dim = 0; dim < half_rope_dim_; ++dim) {
            rope_out.SetValue(dim, even_local.GetValue(dim));
            rope_out.SetValue(half_rope_dim_ + dim, odd_local.GetValue(dim));
        }
    }

    __aicore__ inline void ComputeRope(
        const AscendC::LocalTensor<scalar_t>& kv_local,
        const AscendC::LocalTensor<scalar_t>& cache_local,
        const AscendC::LocalTensor<scalar_t>& rope_out)
    {
        ComputeRopeFromInput(kv_local[nope_dim_], cache_local, rope_out);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<scalar_t> kv_local = kv_queue_.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> weight_local = weight_queue_.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> cache_local = cache_queue_.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> nope_out = nope_out_queue_.AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> rope_out = rope_out_queue_.AllocTensor<scalar_t>();

        ComputeNope(kv_local, weight_local, nope_out);
        ComputeRope(kv_local, cache_local, rope_out);

        nope_out_queue_.EnQue(nope_out);
        rope_out_queue_.EnQue(rope_out);
        kv_queue_.FreeTensor(kv_local);
        weight_queue_.FreeTensor(weight_local);
        cache_queue_.FreeTensor(cache_local);
    }

    __aicore__ inline int64_t CacheOffset(int64_t slot, int64_t dim, int64_t dim_size) const
    {
        int64_t block_idx = slot / cache_block_size_;
        int64_t block_offset = slot - block_idx * cache_block_size_;
        if (cache_mode_is_nz_) {
            return block_idx * cache_block_size_ * dim_size
                + (dim / 16) * cache_block_size_ * 16
                + block_offset * 16
                + (dim % 16);
        }
        return (block_idx * cache_block_size_ + block_offset) * dim_size + dim;
    }

    __aicore__ inline void StoreCache(
        const AscendC::LocalTensor<scalar_t>& rope_out,
        const AscendC::LocalTensor<scalar_t>& nope_out,
        int64_t slot)
    {
        if (slot < 0) {
            return;
        }

        if (cache_mode_is_nz_) {
            for (int64_t dim = 0; dim < nope_dim_; dim += 16) {
                AscendC::DataCopy(kv_cache_nope_gm_[CacheOffset(slot, dim, nope_dim_)], nope_out[dim], 16);
            }
            for (int64_t dim = 0; dim < rope_dim_; dim += 16) {
                AscendC::DataCopy(kv_cache_rope_gm_[CacheOffset(slot, dim, rope_dim_)], rope_out[dim], 16);
            }
        } else {
            AscendC::DataCopy(kv_cache_nope_gm_[CacheOffset(slot, 0, nope_dim_)], nope_out, nope_dim_);
            AscendC::DataCopy(kv_cache_rope_gm_[CacheOffset(slot, 0, rope_dim_)], rope_out, rope_dim_);
        }
    }

    __aicore__ inline void CopyOut(int64_t token_idx)
    {
        AscendC::LocalTensor<scalar_t> nope_out = nope_out_queue_.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> rope_out = rope_out_queue_.DeQue<scalar_t>();
        slot_t slot = slots_gm_.GetValue(token_idx);

        StoreCache(rope_out, nope_out, static_cast<int64_t>(slot));
        if (is_output_kv_) {
            AscendC::DataCopy(out_rope_gm_[token_idx * rope_dim_], rope_out, rope_dim_);
            AscendC::DataCopy(out_nope_gm_[token_idx * nope_dim_], nope_out, nope_dim_);
        }

        nope_out_queue_.FreeTensor(nope_out);
        rope_out_queue_.FreeTensor(rope_out);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> kv_queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> weight_queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> cache_queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> nope_out_queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> rope_out_queue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> nope_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> weight_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> reduce_tmp_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sum_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> even_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> odd_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> even_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> odd_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cos_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sin_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rope_tmp_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rope_first_fp32_buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> rope_second_fp32_buf_;
    AscendC::GlobalTensor<scalar_t> kv_gm_;
    AscendC::GlobalTensor<scalar_t> weight_gm_;
    AscendC::GlobalTensor<pos_t> positions_gm_;
    AscendC::GlobalTensor<scalar_t> cos_sin_cache_gm_;
    AscendC::GlobalTensor<slot_t> slots_gm_;
    AscendC::GlobalTensor<scalar_t> kv_cache_rope_gm_;
    AscendC::GlobalTensor<scalar_t> kv_cache_nope_gm_;
    AscendC::GlobalTensor<scalar_t> out_rope_gm_;
    AscendC::GlobalTensor<scalar_t> out_nope_gm_;
    int64_t num_tokens_;
    int64_t kv_row_stride_;
    int64_t cos_sin_row_stride_;
    int64_t cache_block_size_;
    int64_t nope_dim_;
    int64_t rope_dim_;
    int64_t total_dim_;
    int64_t half_rope_dim_;
    float epsilon_;
    bool is_neox_style_;
    bool is_output_kv_;
    bool cache_mode_is_nz_;
};

template <typename scalar_t, typename pos_t, typename slot_t>
class KvRmsNormRopeCacheAndInterleaveByCache
    : public KvRmsNormRopeCacheByCache<scalar_t, pos_t, slot_t> {
public:
    __aicore__ inline KvRmsNormRopeCacheAndInterleaveByCache() {}

    __aicore__ inline void Init(
        __gm__ scalar_t* kv,
        __gm__ scalar_t* weight,
        __gm__ scalar_t* q,
        __gm__ pos_t* positions,
        __gm__ scalar_t* cos_sin_cache,
        __gm__ slot_t* slots,
        __gm__ scalar_t* kv_cache_rope,
        __gm__ scalar_t* kv_cache_nope,
        __gm__ scalar_t* out_rope,
        __gm__ scalar_t* out_nope,
        __gm__ scalar_t* q_out,
        int64_t num_tokens,
        int64_t kv_row_stride,
        int64_t q_row_stride,
        int64_t q_head_stride,
        int64_t q_num_heads,
        int64_t cos_sin_row_stride,
        int64_t cache_block_size,
        int64_t nope_dim,
        int64_t rope_dim,
        float epsilon,
        bool is_neox_style,
        bool is_output_kv,
        bool cache_mode_is_nz)
    {
        KvRmsNormRopeCacheByCache<scalar_t, pos_t, slot_t>::Init(
            kv, weight, positions, cos_sin_cache, slots, kv_cache_rope, kv_cache_nope, out_rope, out_nope,
            num_tokens, kv_row_stride, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
            is_neox_style, is_output_kv, cache_mode_is_nz);
        q_gm_.SetGlobalBuffer(q);
        q_out_gm_.SetGlobalBuffer(q_out);
        q_row_stride_ = q_row_stride;
        q_head_stride_ = q_head_stride;
        q_num_heads_ = q_num_heads;
    }

    __aicore__ inline void Process()
    {
        int64_t total_kv_tasks = this->num_tokens_;
        int64_t total_q_tasks = this->num_tokens_ * q_num_heads_;
        int64_t total_tasks = total_kv_tasks + total_q_tasks;
        int64_t block_num = AscendC::GetBlockNum();
        for (int64_t task_idx = AscendC::GetBlockIdx(); task_idx < total_tasks; task_idx += block_num) {
            if (task_idx < total_kv_tasks) {
                this->CopyIn(task_idx);
                this->Compute();
                this->CopyOut(task_idx);
            } else {
                int64_t q_linear_idx = task_idx - total_kv_tasks;
                int64_t token_idx = q_linear_idx / q_num_heads_;
                int64_t head_idx = q_linear_idx - token_idx * q_num_heads_;
                CopyInQ(token_idx, head_idx);
                ComputeQ();
                CopyOutQ(q_linear_idx);
            }
        }
    }

private:
    __aicore__ inline void CopyInQ(int64_t token_idx, int64_t head_idx)
    {
        AscendC::LocalTensor<scalar_t> q_local = this->kv_queue_.template AllocTensor<scalar_t>();
        AscendC::LocalTensor<scalar_t> cache_local = this->cache_queue_.template AllocTensor<scalar_t>();

        int64_t q_offset = token_idx * q_row_stride_ + head_idx * q_head_stride_;
        pos_t position = this->positions_gm_.GetValue(token_idx);
        int64_t cache_offset = static_cast<int64_t>(position) * this->cos_sin_row_stride_;

        AscendC::DataCopy(q_local, q_gm_[q_offset], this->rope_dim_);
        AscendC::DataCopy(cache_local, this->cos_sin_cache_gm_[cache_offset], this->rope_dim_);
        this->kv_queue_.EnQue(q_local);
        this->cache_queue_.EnQue(cache_local);
    }

    __aicore__ inline void ComputeQ()
    {
        AscendC::LocalTensor<scalar_t> q_local = this->kv_queue_.template DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> cache_local = this->cache_queue_.template DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> rope_out = this->rope_out_queue_.template AllocTensor<scalar_t>();

        this->ComputeRopeFromInput(q_local, cache_local, rope_out);

        this->rope_out_queue_.EnQue(rope_out);
        this->kv_queue_.FreeTensor(q_local);
        this->cache_queue_.FreeTensor(cache_local);
    }

    __aicore__ inline void CopyOutQ(int64_t q_linear_idx)
    {
        AscendC::LocalTensor<scalar_t> rope_out = this->rope_out_queue_.template DeQue<scalar_t>();
        AscendC::DataCopy(q_out_gm_[q_linear_idx * this->rope_dim_], rope_out, this->rope_dim_);
        this->rope_out_queue_.FreeTensor(rope_out);
    }

    AscendC::GlobalTensor<scalar_t> q_gm_;
    AscendC::GlobalTensor<scalar_t> q_out_gm_;
    int64_t q_row_stride_;
    int64_t q_head_stride_;
    int64_t q_num_heads_;
};

#define KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(TYPE, POS_TYPE, SLOT_TYPE, POS_SUFFIX, SLOT_SUFFIX)                  \
    extern "C" __global__ __aicore__ void kv_rmsnorm_rope_cache_by_cache_##TYPE##_##POS_SUFFIX##_##SLOT_SUFFIX(   \
        __gm__ TYPE* kv, __gm__ TYPE* weight, __gm__ POS_TYPE* positions, __gm__ TYPE* cos_sin_cache,              \
        __gm__ SLOT_TYPE* slots, __gm__ TYPE* kv_cache_rope, __gm__ TYPE* kv_cache_nope,                           \
        __gm__ TYPE* out_rope, __gm__ TYPE* out_nope, int64_t num_tokens, int64_t kv_row_stride,                   \
        int64_t cos_sin_row_stride, int64_t cache_block_size, int64_t nope_dim, int64_t rope_dim, float epsilon,   \
        bool is_neox_style, bool is_output_kv, bool cache_mode_is_nz)                                               \
    {                                                                                                              \
        KvRmsNormRopeCacheByCache<TYPE, POS_TYPE, SLOT_TYPE> op;                                                   \
        op.Init(kv, weight, positions, cos_sin_cache, slots, kv_cache_rope, kv_cache_nope, out_rope, out_nope,     \
                num_tokens, kv_row_stride, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,      \
                is_neox_style, is_output_kv, cache_mode_is_nz);                                                    \
        op.Process();                                                                                              \
    }

KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(half, int32_t, int32_t, int32, int32)
KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(half, int32_t, int64_t, int32, int64)
KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(half, int64_t, int32_t, int64, int32)
KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(half, int64_t, int64_t, int64, int64)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(bfloat16_t, int32_t, int32_t, int32, int32)
KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(bfloat16_t, int32_t, int64_t, int32, int64)
KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(bfloat16_t, int64_t, int32_t, int64, int32)
KV_RMSNORM_ROPE_CACHE_BY_CACHE_DECLARE(bfloat16_t, int64_t, int64_t, int64, int64)
#endif

#define KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(TYPE, POS_TYPE, SLOT_TYPE, POS_SUFFIX, SLOT_SUFFIX)    \
    extern "C" __global__ __aicore__ void                                                                             \
    kv_rmsnorm_rope_cache_and_interleave_by_cache_##TYPE##_##POS_SUFFIX##_##SLOT_SUFFIX(                            \
        __gm__ TYPE* kv, __gm__ TYPE* weight, __gm__ TYPE* q, __gm__ POS_TYPE* positions,                           \
        __gm__ TYPE* cos_sin_cache, __gm__ SLOT_TYPE* slots, __gm__ TYPE* kv_cache_rope,                            \
        __gm__ TYPE* kv_cache_nope, __gm__ TYPE* out_rope, __gm__ TYPE* out_nope, __gm__ TYPE* q_out,               \
        int64_t num_tokens, int64_t kv_row_stride, int64_t q_row_stride, int64_t q_head_stride,                    \
        int64_t q_num_heads, int64_t cos_sin_row_stride, int64_t cache_block_size, int64_t nope_dim,                \
        int64_t rope_dim, float epsilon, bool is_neox_style, bool is_output_kv, bool cache_mode_is_nz)              \
    {                                                                                                               \
        KvRmsNormRopeCacheAndInterleaveByCache<TYPE, POS_TYPE, SLOT_TYPE> op;                                       \
        op.Init(kv, weight, q, positions, cos_sin_cache, slots, kv_cache_rope, kv_cache_nope, out_rope, out_nope,   \
                q_out, num_tokens, kv_row_stride, q_row_stride, q_head_stride, q_num_heads, cos_sin_row_stride,     \
                cache_block_size, nope_dim, rope_dim, epsilon, is_neox_style, is_output_kv, cache_mode_is_nz);      \
        op.Process();                                                                                               \
    }

KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(half, int32_t, int32_t, int32, int32)
KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(half, int32_t, int64_t, int32, int64)
KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(half, int64_t, int32_t, int64, int32)
KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(half, int64_t, int64_t, int64, int64)
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(bfloat16_t, int32_t, int32_t, int32, int32)
KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(bfloat16_t, int32_t, int64_t, int32, int64)
KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(bfloat16_t, int64_t, int32_t, int64, int32)
KV_RMSNORM_ROPE_CACHE_AND_INTERLEAVE_BY_CACHE_DECLARE(bfloat16_t, int64_t, int64_t, int64, int64)
#endif

}  // namespace

namespace vllm_ascend {

void kv_rmsnorm_rope_cache_by_cache_impl(
    AscendType type,
    bool positions_is_int32,
    bool slots_is_int32,
    void* stream,
    void* kv,
    void* weight,
    void* positions,
    void* cos_sin_cache,
    void* slots,
    void* kv_cache_rope,
    void* kv_cache_nope,
    void* out_rope,
    void* out_nope,
    int64_t num_tokens,
    int64_t kv_row_stride,
    int64_t cos_sin_row_stride,
    int64_t cache_block_size,
    int64_t nope_dim,
    int64_t rope_dim,
    float epsilon,
    bool is_neox_style,
    bool is_output_kv,
    bool cache_mode_is_nz,
    uint32_t block_dim)
{
    if (type == AscendType::FP16) {
        if (positions_is_int32 && slots_is_int32) {
            kv_rmsnorm_rope_cache_by_cache_half_int32_int32<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<int32_t*>(positions),
                static_cast<half*>(cos_sin_cache), static_cast<int32_t*>(slots), static_cast<half*>(kv_cache_rope),
                static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope), static_cast<half*>(out_nope),
                num_tokens, kv_row_stride, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        } else if (positions_is_int32) {
            kv_rmsnorm_rope_cache_by_cache_half_int32_int64<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<int32_t*>(positions),
                static_cast<half*>(cos_sin_cache), static_cast<int64_t*>(slots), static_cast<half*>(kv_cache_rope),
                static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope), static_cast<half*>(out_nope),
                num_tokens, kv_row_stride, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        } else if (slots_is_int32) {
            kv_rmsnorm_rope_cache_by_cache_half_int64_int32<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<int64_t*>(positions),
                static_cast<half*>(cos_sin_cache), static_cast<int32_t*>(slots), static_cast<half*>(kv_cache_rope),
                static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope), static_cast<half*>(out_nope),
                num_tokens, kv_row_stride, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        } else {
            kv_rmsnorm_rope_cache_by_cache_half_int64_int64<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<int64_t*>(positions),
                static_cast<half*>(cos_sin_cache), static_cast<int64_t*>(slots), static_cast<half*>(kv_cache_rope),
                static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope), static_cast<half*>(out_nope),
                num_tokens, kv_row_stride, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        }
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        if (positions_is_int32 && slots_is_int32) {
            kv_rmsnorm_rope_cache_by_cache_bfloat16_t_int32_int32<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<int32_t*>(positions),
                static_cast<bfloat16_t*>(cos_sin_cache), static_cast<int32_t*>(slots),
                static_cast<bfloat16_t*>(kv_cache_rope), static_cast<bfloat16_t*>(kv_cache_nope),
                static_cast<bfloat16_t*>(out_rope), static_cast<bfloat16_t*>(out_nope), num_tokens, kv_row_stride,
                cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon, is_neox_style, is_output_kv,
                cache_mode_is_nz);
        } else if (positions_is_int32) {
            kv_rmsnorm_rope_cache_by_cache_bfloat16_t_int32_int64<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<int32_t*>(positions),
                static_cast<bfloat16_t*>(cos_sin_cache), static_cast<int64_t*>(slots),
                static_cast<bfloat16_t*>(kv_cache_rope), static_cast<bfloat16_t*>(kv_cache_nope),
                static_cast<bfloat16_t*>(out_rope), static_cast<bfloat16_t*>(out_nope), num_tokens, kv_row_stride,
                cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon, is_neox_style, is_output_kv,
                cache_mode_is_nz);
        } else if (slots_is_int32) {
            kv_rmsnorm_rope_cache_by_cache_bfloat16_t_int64_int32<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<int64_t*>(positions),
                static_cast<bfloat16_t*>(cos_sin_cache), static_cast<int32_t*>(slots),
                static_cast<bfloat16_t*>(kv_cache_rope), static_cast<bfloat16_t*>(kv_cache_nope),
                static_cast<bfloat16_t*>(out_rope), static_cast<bfloat16_t*>(out_nope), num_tokens, kv_row_stride,
                cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon, is_neox_style, is_output_kv,
                cache_mode_is_nz);
        } else {
            kv_rmsnorm_rope_cache_by_cache_bfloat16_t_int64_int64<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<int64_t*>(positions),
                static_cast<bfloat16_t*>(cos_sin_cache), static_cast<int64_t*>(slots),
                static_cast<bfloat16_t*>(kv_cache_rope), static_cast<bfloat16_t*>(kv_cache_nope),
                static_cast<bfloat16_t*>(out_rope), static_cast<bfloat16_t*>(out_nope), num_tokens, kv_row_stride,
                cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon, is_neox_style, is_output_kv,
                cache_mode_is_nz);
        }
#endif
    }
}

void kv_rmsnorm_rope_cache_and_interleave_by_cache_impl(
    AscendType type,
    bool positions_is_int32,
    bool slots_is_int32,
    void* stream,
    void* kv,
    void* weight,
    void* q,
    void* positions,
    void* cos_sin_cache,
    void* slots,
    void* kv_cache_rope,
    void* kv_cache_nope,
    void* out_rope,
    void* out_nope,
    void* q_out,
    int64_t num_tokens,
    int64_t kv_row_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    int64_t q_num_heads,
    int64_t cos_sin_row_stride,
    int64_t cache_block_size,
    int64_t nope_dim,
    int64_t rope_dim,
    float epsilon,
    bool is_neox_style,
    bool is_output_kv,
    bool cache_mode_is_nz,
    uint32_t block_dim)
{
    if (type == AscendType::FP16) {
        if (positions_is_int32 && slots_is_int32) {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_half_int32_int32<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<half*>(q),
                static_cast<int32_t*>(positions), static_cast<half*>(cos_sin_cache), static_cast<int32_t*>(slots),
                static_cast<half*>(kv_cache_rope), static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope),
                static_cast<half*>(out_nope), static_cast<half*>(q_out), num_tokens, kv_row_stride, q_row_stride,
                q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        } else if (positions_is_int32) {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_half_int32_int64<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<half*>(q),
                static_cast<int32_t*>(positions), static_cast<half*>(cos_sin_cache), static_cast<int64_t*>(slots),
                static_cast<half*>(kv_cache_rope), static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope),
                static_cast<half*>(out_nope), static_cast<half*>(q_out), num_tokens, kv_row_stride, q_row_stride,
                q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        } else if (slots_is_int32) {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_half_int64_int32<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<half*>(q),
                static_cast<int64_t*>(positions), static_cast<half*>(cos_sin_cache), static_cast<int32_t*>(slots),
                static_cast<half*>(kv_cache_rope), static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope),
                static_cast<half*>(out_nope), static_cast<half*>(q_out), num_tokens, kv_row_stride, q_row_stride,
                q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        } else {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_half_int64_int64<<<block_dim, nullptr, stream>>>(
                static_cast<half*>(kv), static_cast<half*>(weight), static_cast<half*>(q),
                static_cast<int64_t*>(positions), static_cast<half*>(cos_sin_cache), static_cast<int64_t*>(slots),
                static_cast<half*>(kv_cache_rope), static_cast<half*>(kv_cache_nope), static_cast<half*>(out_rope),
                static_cast<half*>(out_nope), static_cast<half*>(q_out), num_tokens, kv_row_stride, q_row_stride,
                q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim, rope_dim, epsilon,
                is_neox_style, is_output_kv, cache_mode_is_nz);
        }
    } else if (type == AscendType::BF16) {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
        if (positions_is_int32 && slots_is_int32) {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_bfloat16_t_int32_int32<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<bfloat16_t*>(q),
                static_cast<int32_t*>(positions), static_cast<bfloat16_t*>(cos_sin_cache),
                static_cast<int32_t*>(slots), static_cast<bfloat16_t*>(kv_cache_rope),
                static_cast<bfloat16_t*>(kv_cache_nope), static_cast<bfloat16_t*>(out_rope),
                static_cast<bfloat16_t*>(out_nope), static_cast<bfloat16_t*>(q_out), num_tokens, kv_row_stride,
                q_row_stride, q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim,
                rope_dim, epsilon, is_neox_style, is_output_kv, cache_mode_is_nz);
        } else if (positions_is_int32) {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_bfloat16_t_int32_int64<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<bfloat16_t*>(q),
                static_cast<int32_t*>(positions), static_cast<bfloat16_t*>(cos_sin_cache),
                static_cast<int64_t*>(slots), static_cast<bfloat16_t*>(kv_cache_rope),
                static_cast<bfloat16_t*>(kv_cache_nope), static_cast<bfloat16_t*>(out_rope),
                static_cast<bfloat16_t*>(out_nope), static_cast<bfloat16_t*>(q_out), num_tokens, kv_row_stride,
                q_row_stride, q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim,
                rope_dim, epsilon, is_neox_style, is_output_kv, cache_mode_is_nz);
        } else if (slots_is_int32) {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_bfloat16_t_int64_int32<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<bfloat16_t*>(q),
                static_cast<int64_t*>(positions), static_cast<bfloat16_t*>(cos_sin_cache),
                static_cast<int32_t*>(slots), static_cast<bfloat16_t*>(kv_cache_rope),
                static_cast<bfloat16_t*>(kv_cache_nope), static_cast<bfloat16_t*>(out_rope),
                static_cast<bfloat16_t*>(out_nope), static_cast<bfloat16_t*>(q_out), num_tokens, kv_row_stride,
                q_row_stride, q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim,
                rope_dim, epsilon, is_neox_style, is_output_kv, cache_mode_is_nz);
        } else {
            kv_rmsnorm_rope_cache_and_interleave_by_cache_bfloat16_t_int64_int64<<<block_dim, nullptr, stream>>>(
                static_cast<bfloat16_t*>(kv), static_cast<bfloat16_t*>(weight), static_cast<bfloat16_t*>(q),
                static_cast<int64_t*>(positions), static_cast<bfloat16_t*>(cos_sin_cache),
                static_cast<int64_t*>(slots), static_cast<bfloat16_t*>(kv_cache_rope),
                static_cast<bfloat16_t*>(kv_cache_nope), static_cast<bfloat16_t*>(out_rope),
                static_cast<bfloat16_t*>(out_nope), static_cast<bfloat16_t*>(q_out), num_tokens, kv_row_stride,
                q_row_stride, q_head_stride, q_num_heads, cos_sin_row_stride, cache_block_size, nope_dim,
                rope_dim, epsilon, is_neox_style, is_output_kv, cache_mode_is_nz);
        }
#endif
    }
}

}  // namespace vllm_ascend
