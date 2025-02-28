#pragma once
#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"
#include "kernel_tensor_impl.h"
#include "kernel_type.h"
#include "kernel_operator_intf.h"
#include "inner_interface/inner_kernel_operator_intf.h"
#include <stdio.h>
#include "utils.h"




template <typename scalar_t, bool IS_NEOX>
class KernelRope {
    static constexpr int BLK_SIZE = 32;
    static constexpr int STRIDE_SIZE = 32 * 8;
    static constexpr int  ROPE_TILING = 128;
    static constexpr int LOAD_SIZE = 1024 * 64;
    typedef typename acc_type<scalar_t>::type acc_t;
    using LocT = AscendC::LocalTensor<scalar_t>;
    using LocAcc = AscendC::LocalTensor<acc_t>;
    static constexpr bool  same_type = std::is_same<scalar_t, acc_t>::type;
    public:
        __aicore__ inline KernelRope() {}

        __aicore__ inline void Init(
            __gm__ int64_t* positions,
            __gm__ dst_t* query_dst,
            __gm__ dst_t* key_dst,
            __gm__ scalar_t* query,
            __gm__ scalar_t* key,
            __gm__ scalar_t* cos_sin_cache,
            const int rot_dim,
            const int64_t dst_query_stride,
            const int64_t dst_key_stride,
            const int64_t query_stride,
            const int64_t key_stride,
            const int num_heads,
            const int num_kv_heads,
            const int head_size,
            AscendC::TPipe* pipe
        ) {
            pipe_ = pipe;
            rot_dim_ = rot_dim;
            query_stride_ = query_stride;
            key_stride_ = key_stride;
            num_heads_ = num_heads;
            num_kv_heads_ = num_kv_heads;
            head_size_ = head_size;
            embed_dim_ = rot_dim / 2;

            pipe_->InitBuffer(in_que_, 1, LOAD_SIZE);
            pipe_->InitBuffer(in_que_sin_cos_, 1, rot_dim_ * sizeof(scalar_t));
            pipe_->InitBuffer(out_que_, 1, LOAD_SIZE);
            // 2 temperary calculation buffer
            int calc_tmp_buffer_offset_ = 0;
            // 2 upcast buffer for bf16
            int upcast_buffer_offset_ = calc_tmp_buffer_offset_ + !same_type ? sizeof(acc_t) * embed_dim_ * 2 : 0;
            // 2 sin cos upcast buffer for bf16
            int cos_sin_upcast_buffer_offset_ = upcast_buffer_offset_ + !same_type ? sizeof(acc_t) * embed_dim_ * 2 : 0;
            // 1. quant path: needs 1 addtional dst dump buffer size
            // 2. bf16 path: needs 2 cos sin upcast buffer size
            // 3. fp16 path: needs 2 temperary calculation buffer size
            int temp_buffer_size_ = dst_quant_temp_buffer_offset_ + 2 * embed_dim_ * sizeof(acc_t);
            // need to consider upcast the bf16 to fp32, so we might need 4 buffer just in case
            // 2 temperary buffer, 2 input buffer, 1 cos buffer, 1 sin buffer, 2 scale buffer (head_size), 2 zp buffer(head_size int8), 1 dst_temp buffer(head_size, int32)
            pipe_->InitBuffer(calc_buf_, temp_buffer_size_);
            if constexpr (IS_NEOX) {
                // note: We allocate two mask buffer as its max valid mask size, namely 256bit * 4 for NEOX scenario.
                pipe_->InitBuffer(interleave_mask_, embed_dim_ * sizeof(uint32_t) * 6);
            }
        }
    __aicore__ inline void Update(
            __gm__ int64_t* positions,
            __gm__ dst_t* query_dst,
            __gm__ dst_t* key_dst,
            __gm__ scalar_t* query,
            __gm__ scalar_t* key,
            __gm__ scalar_t* cos_sin_cache,
            const int rot_dim,
            const int64_t dst_query_stride,
            const int64_t dst_key_stride,
            const int64_t query_stride,
            const int64_t key_stride,
            const int num_heads,
            const int num_kv_heads,
            const int head_size,
            const int64_t idx) {
            int64_t pos = positions[idx];
            cos_sin_cache = cos_sin_cache + pos * rot_dim_;
            cos_sin_.SetGlobalBuffer(cos_sin_cache, rot_dim_);
            // sin_.SetGlobalBuffer(cos_sin_cache + embed_dim_, embed_dim_);
            query_.SetGlobalBuffer(query + query_stride * idx, head_size * num_heads_);
            key_.SetGlobalBuffer(key + key_stride * idx, head_size * num_kv_heads_);
            query_dst_.SetGlobalBuffer(query_dst + dst_query_stride * idx, head_size * num_heads_);
            key_dst_.SetGlobalBuffer(key_dst + dst_key_stride * idx, head_size * num_kv_heads_);
    }



    template <typename op_type=scalar_t, class = typename std::enable_if<!std::is_same<acc_t, scalar_t>::value, void>::type>
      __aicore__ inline void neox_compute(
      LocT src,
      LocT dst,
      LocAcc sin,
      LocAcc cos,
      LocAcc upcast_buffer,
      LocAcc calc_tmp_buffer
    ) {
      // slice dst
      LocT dst_x = dst;
      LocT dst_y = dst[embed_dim_];

      // slice src
      LocT src_x = src;
      LocT src_y = src[embed_dim_];

      // slice temp buffer
      LocAcc calc_tmp_buffer_x = calc_tmp_buffer;
      LocAcc calc_tmp_buffer_y = calc_tmp_buffer[embed_dim_];

      // slice upcast buffer
      LocAcc upcast_buffer_x = upcast_buffer;
      LocAcc upcast_buffer_y = upcast_buffer[embed_dim_];

      // dst x calc
      Cast(upcast_buffer, src_x, AscendC::RoundMode::CAST_NONE, 2 * embed_dim_);
      Mul(calc_tmp_buffer_x, upcast_buffer_x, cos, embed_dim_);
      Mul(calc_tmp_buffer_y, upcast_buffer_y, sin, embed_dim_);
      Sub(calc_tmp_buffer_x, calc_tmp_buffer_x, calc_tmp_buffer_y, embed_dim_);
      Cast(dst_x, calc_tmp_buffer_x, AscendC::RoundMode::CAST_TRUNC, embed_dim_);

      // dst y calc
      Mul(calc_tmp_buffer_x, upcast_buffer_x, sin, embed_dim_);
      Mul(calc_tmp_buffer_y, upcast_buffer_y, cos, embed_dim_);
      Add(calc_tmp_buffer_x, calc_tmp_buffer_x, calc_tmp_buffer_y, embed_dim_);
      Cast(dst_y, calc_tmp_buffer_x, AscendC::RoundMode::CAST_TRUNC, embed_dim_);

    }



    template<typename op_type=scalar_t, class = typename std::enable_if<std::is_same<acc_t, scalar_t>::value, void>::type>
    __aicore__ inline void neox_compute(
      LocT src,
      LocT dst,
      LocAcc sin,
      LocAcc cos,
      LocAcc upcast_buffer,
      LocAcc calc_tmp_buffer
    ) {
      // slice dst buffer
      LocT dst_x = dst;
      LocT dst_y = dst[embed_dim_];
      // slice src buffer
      LocT src_x = src;
      LocT src_y = src[embed_dim_];
      // slice temp buffer
      LocAcc calc_tmp_buffer_x = calc_tmp_buffer;
      LocAcc calc_tmp_buffer_y = calc_tmp_buffer[embed_dim_];

      // dst x calc
      Mul(calc_tmp_buffer_x, src_x, cos, embed_dim_);
      Mul(calc_tmp_buffer_y, src_y, sin, embed_dim_);
      Sub(dst_x, calc_tmp_buffer_x, calc_tmp_buffer_y, embed_dim_);

      // dst y calc
      Mul(calc_tmp_buffer_x, src_x, sin, embed_dim_);
      Mul(calc_tmp_buffer_y, src_y, cos, embed_dim_);
      Add(dst_y, calc_tmp_buffer_x, calc_tmp_buffer_y, embed_dim_);
    }


  __aicore__ inline void compute_tensor(
    AscendC::GlobalTensor<scalar_t> src_g,
    AscendC::GlobalTensor<dst_t> dst_g,
    LocAcc local_cos,
    LocAcc local_sin,
    LocAcc upcast_buffer,
    LocAcc calc_tmp_buffer,
    LocAcc dst_temp_buffer,
    int loop_cnt,
    int tail_heads,
    int load_stride,
    int head_num_per_load) {

    for (int loop_num = 0; loop_num < loop_cnt; ++loop_num) {
        LocT src = in_que_.AllocTensor<scalar_t>();
        LocT dst = out_que_.AllocTensor<scalar_t>();
        int repeat_cnt = (load_stride + 127) / 128;
        AscendC::DataCopy(src, src_g[loop_num * load_stride], load_stride);
        in_que_.EnQue(src);

        LocT src_deque = in_que_.DeQue<scalar_t>();
        AscendC::Copy(dst, src_deque, 256 / sizeof(scalar_t), repeat_cnt, {1, 1, 8, 8});
        for (int i = 0; i < head_num_per_load; ++i) {
          if constexpr (std::is_same<acc_t, scalar_t>::type)
            neox_compute(src_deque, dst, local_sin, local_cos, calc_tmp_buffer, dst_temp_buffer);
          else
            neox_compute(src_deque, dst, local_sin, local_cos, upcast_buffer, calc_tmp_buffer, dst_temp_buffer);
        }
        out_que_.EnQue(dst);
        LocT dst_deque = out_que_.DeQue<scalar_t>();
        AscendC::DataCopy(dst_g[loop_num * load_stride], dst_deque, load_stride);
        out_que_.FreeTensor(dst_deque);
        in_que_.FreeTensor(src_deque);
    }
    // process tail
    {
        LocT src = in_que_.AllocTensor<scalar_t>();
        LocT dst = out_que_.AllocTensor<scalar_t>();
        int repeat_cnt = (tail_heads * head_size_ * sizeof(scalar_t) + 255) / 256;
        AscendC::DataCopy(src, src_g[loop_cnt * load_stride], tail_heads * head_size_);
        in_que_.EnQue(src);
        LocT src_deque = in_que_.DeQue<scalar_t>();

        AscendC::Copy(dst, src_deque, 256 / sizeof(scalar_t), repeat_cnt, {1, 1, 8, 8});

        for (int i = 0; i < tail_heads; ++i) {
          if constexpr (std::is_same<acc_t, scalar_t>::type)
            neox_compute(src_deque, dst, local_sin, local_cos, calc_tmp_buffer, dst_temp_buffer);
          else
            neox_compute(src_deque, dst, local_sin, local_cos, upcast_buffer, calc_tmp_buffer, dst_temp_buffer);
        }
        out_que_.EnQue(dst);
        LocT dst_deque = out_que_.DeQue<scalar_t>();
        AscendC::DataCopy(dst_g[loop_cnt * load_stride], dst_deque, tail_heads * head_size_);
        out_que_.FreeTensor(dst_deque);
        in_que_.FreeTensor(src_deque);
    }
  }

    __aicore__ inline void Compute() {
        LocT cos_sin_local = in_que_sin_cos_.AllocTensor<scalar_t>();

        AscendC::DataCopy(cos_sin_local, cos_sin_, embed_dim_ * 2);

        in_que_sin_cos_.EnQue(cos_sin_local);
        LocT local_sin_cos_deque = in_que_sin_cos_.DeQue<scalar_t>();
        LocT local_cos = local_sin_cos_deque;
        LocT local_sin = local_sin_cos_deque[embed_dim_];
        // We define the interleave mask 1, 2, 3 for shuffle and deshuffle operation in non-neox scenario, mask1 and mask2 are build for
        // shuffle the x and y into two consecutive buffer, and mask3 is build for de-shuffle the scattered data back to the data that 
        // can rewrite back to the global memory.
        AscendC::LocalTensor<int32_t> interleave_mask1, interleave_mask2, interleave_mask3, tmp_mask;
        AscendC::LocalTensor<uint32_t> umask1, umask2, umask3;
        uint8_t loop_cnt_mask = (embed_dim_ + STRIDE_SIZE - 1) / STRIDE_SIZE;
        int32_t start_val = 0;

        if constexpr (!IS_NEOX) {
            interleave_mask1 = interleave_mask_.Get<int32_t>(embed_dim_ * 6);
            interleave_mask2 = interleave_mask1[embed_dim_];
            interleave_mask3 = interleave_mask1[rot_dim_];
            tmp_mask = interleave_mask1[2 * rot_dim_];
            int32_t mul_div_val = 2;
            int32_t add_val = 1;
            CreateVecIndex(interleave_mask1, start_val, embed_dim_);
            CreateVecIndex(interleave_mask3, start_val, rot_dim_);
            Muls(interleave_mask1, interleave_mask1, mul_div_val, embed_dim_);
            Adds(interleave_mask2, interleave_mask1, add_val, embed_dim_);
            // Div instruction dose not have scalar version, use tmp mask for this place.
            Duplicate(tmp_mask, mul_div_val, rot_dim_);
            AscendC::LocalTensor<float> tmp_interleave_mask3, tmp_tmp_mask;
            Cast(tmp_interleave_mask3, interleave_mask3, AscendC::RoundMode::CAST_ROUND, rot_dim_);
            Cast(tmp_tmp_mask, tmp_mask, AscendC::RoundMode::CAST_ROUND, rot_dim_);
            Div(tmp_interleave_mask3, tmp_interleave_mask3, tmp_tmp_mask, rot_dim_);
            Cast(interleave_mask3, tmp_interleave_mask3, AscendC::RoundMode::CAST_FLOOR, rot_dim_);

            uint64_t mask[2] = {uint64_t(0x5555555555555555ULL), uint64_t(0x5555555555555555ULL)};
            Adds(interleave_mask3, interleave_mask3, embed_dim_, mask, loop_cnt_mask, {});
            umask1 = interleave_mask1.ReinterpretCast<uint32_t>();
            umask2 = interleave_mask2.ReinterpretCast<uint32_t>();
            umask3 = interleave_mask3.ReinterpretCast<uint32_t>();
        }
        LocAcc calc_tmp_buffer, upcast_buffer, cos_sin_upcast_buffer, scale_buffer;
        calc_tmp_buffer = calc_buf_.GetWithOffset<acc_t>(embed_dim_ * 2, calc_tmp_buffer_offset_);
        upcast_buffer = calc_buf_.GetWithOffset<acc_t>(embed_dim_ * 2, upcast_buffer_offset_);
        cos_sin_upcast_buffer = calc_buf_.GetWithOffset<acc_t>(embed_dim_ * 2, cos_sin_upcast_buffer_offset_);

        LocAcc cos_acc_buffer, sin_acc_buffer;

        if constexpr (!std::is_same<scalar_t, acc_t>::value) {
          Cast(cos_sin_upcast_buffer, local_sin_cos_deque, AscendC::RoundMode::CAST_NONE, 2 * embed_dim_);
          cos_acc_buffer = cos_sin_upcast_buffer;
          sin_acc_buffer = cos_sin_upcast_buffer[embed_dim_];
        } else {
          cos_acc_buffer = local_cos;
          sin_acc_buffer = local_sin;
        }

        constexpr const int load_size = LOAD_SIZE / sizeof(scalar_t);
        int64_t head_num_per_load = load_size / head_size_;
        int64_t loop_cnt = num_heads_ / head_num_per_load;
        int64_t tail_heads = num_heads_ - loop_cnt * head_num_per_load;
        int64_t load_stride = head_num_per_load * head_size_;
        int64_t loop_cnt_kv = num_kv_heads_ / head_num_per_load;
        int64_t tail_heads_kv = num_kv_heads_ - loop_cnt_kv * head_num_per_load;
        // AscendC::printf("loop cnt: %d, tail heads: %d, loop cnt kv: %d, tail heads kv: %d, head_size: %d",loop_cnt, tail_heads, loop_cnt_kv, tail_heads_kv, head_size_);
        compute_tensor(query_, query_dst_, cos_acc_buffer, sin_acc_buffer, upcast_buffer, calc_tmp_buffer, dst_quant_tmp_buffer, loop_cnt, tail_heads, load_stride, head_num_per_load);
        compute_tensor(key_, key_dst_, cos_acc_buffer, sin_acc_buffer, upcast_buffer, calc_tmp_buffer, dst_quant_tmp_buffer, loop_cnt_kv, tail_heads_kv, load_stride, head_num_per_load);


        // #endif
    }


    private:
        AscendC::TPipe* pipe_;
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> in_que_, in_que_sin_cos_;
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_que_;
        AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_;
        AscendC::TBuf<AscendC::TPosition::VECCALC> interleave_mask_;
        AscendC::GlobalTensor<scalar_t> query_dst_;
        AscendC::GlobalTensor<scalar_t> key_dst_;
        AscendC::GlobalTensor<scalar_t> query_;
        AscendC::GlobalTensor<scalar_t> key_;
        AscendC::GlobalTensor<scalar_t> cos_sin_;
        int rot_dim_;
        int embed_dim_;
        int64_t query_stride_;
        int64_t key_stride_;
        int64_t dst_query_stride_;
        int64_t dst_key_stride_;
        int num_heads_;
        int num_kv_heads_;
        int head_size_;
        int calc_tmp_buffer_offset_;
        int upcast_buffer_offset_;
        int cos_sin_upcast_buffer_offset_;
        int dst_quant_temp_buffer_offset_;
        int temp_buffer_size_;

};

#define ROPE_CUSTOM_KERNEL_INSTANTIATION(TYPE, NEOX)    \
extern "C" __global__ __aicore__ void rope_custom_quant_##NEOX##_##TYPE(      \
    __gm__ int64_t* positions,                          \
    __gm__ int8_t* query_dst,                             \
    __gm__ int8_t* key_dst,                               \
    __gm__ TYPE* query,                                 \
    __gm__ TYPE* key,                                   \
    __gm__ TYPE* cos_sin_cache,                         \
    const int rot_dim,                                  \
    const int64_t query_stride,                         \
    const int64_t key_stride,                           \
    const int64_t dst_query_stride,                     \
    const int64_t dst_key_stride,                       \
    const int num_heads,                                \
    const int num_kv_heads,                             \
    const int head_size,                                \
    const int64_t num_tokens,                               \
    const int loop_num,                                 \
    const int core_num) {                               \
        AscendC::TPipe pipe;                            \
        KernelRopeQuant<TYPE, NEOX, true> op{};                    \
        op.Init(                                        \
            positions,                                  \
            query_dst,                                  \
            key_dst,                                    \
            query,                                      \
            key,                                        \
            cos_sin_cache,                              \
            rot_dim,                                    \
            dst_query_stride,                           \
            dst_key_stride,                             \
            query_stride,                               \
            key_stride,                                 \
            num_heads,                                  \
            num_kv_heads,                               \
            head_size,                                  \
            &pipe                                       \
        );                                              \
        for (int64_t i = AscendC::GetBlockIdx(); i < num_tokens; i += core_num) {   \
            op.Update(positions, query_dst, key_dst,    \
                query, key, cos_sin_cache, rot_dim,     \
                dst_query_stride, dst_key_stride,       \
                query_stride, key_stride, num_heads,    \
                num_kv_heads, head_size, i);            \
            op.Compute();                               \
        }                                               \
    }


#define ITERMEDIATE_EXPAND(TYPE, NEOX)  ROPE_CUSTOM_KERNEL_INSTANTIATION(TYPE, NEOX)

#define ROPE_CUSTOM_KERNEL(TYPE)      \
    ITERMEDIATE_EXPAND(TYPE, true);   \
    ITERMEDIATE_EXPAND(TYPE, false);

// ROPE_CUSTOM_KERNEL(float)
ROPE_CUSTOM_KERNEL(half)
ROPE_CUSTOM_KERNEL(bfloat16_t)



#define ROPE_KERNEL_CALL(TYPE)                                  \
    if (is_neox)                                                \
    rope_custom_true_##TYPE<<<BlockDim, nullptr, stream>>>(       \
        positions,                                              \
        reinterpret_cast<TYPE*>(query_dst),                     \
        reinterpret_cast<TYPE*>(key_dst),                       \
        reinterpret_cast<TYPE*>(query),                                                  \
        reinterpret_cast<TYPE*>(key),                                                    \
        reinterpret_cast<TYPE*>(cos_sin_cache),                                          \
        rot_dim,                                                \
        query_stride,                                           \
        key_stride,                                             \
        dst_query_stride,                                       \
        dst_key_stride,                                         \
        num_heads,                                              \
        num_kv_heads,                                           \
        head_size,                                              \
        num_tokens,                                             \
        loop_cnt,                                               \
        BlockDim);                                                \
    else                                                        \
    rope_custom_false_##TYPE<<<BlockDim, nullptr, stream>>>(    \
        positions,                                              \
        reinterpret_cast<TYPE*>(query_dst),                     \
        reinterpret_cast<TYPE*>(key_dst),                       \
        reinterpret_cast<TYPE*>(query),                                                  \
        reinterpret_cast<TYPE*>(key),                                                    \
        reinterpret_cast<TYPE*>(cos_sin_cache),                                          \
        rot_dim,                                                \
        query_stride,                                           \
        key_stride,                                             \
        dst_query_stride,                                       \
        dst_key_stride,                                         \
        num_heads,                                              \
        num_kv_heads,                                           \
        head_size,                                              \
        num_tokens,                                             \
        loop_cnt,                                               \
        BlockDim);

#define INSTANTIATE_ROPE(TYPE, NEOX)        \
    template void rope_custom_do<TYPE, NEOX>(    \
    uint32_t blockDim,                      \
    void* stream,                           \
    int64_t* positions,                     \
    TYPE* query,                            \
    TYPE* key,                              \
    TYPE* cos_sin_cache,                    \
    const int rot_dim,                      \
    const int64_t query_stride,             \
    const int64_t key_stride,               \
    const int num_heads,                    \
    const int num_kv_heads,                 \
    const int head_size);



static const int64_t max_parallel_size = 65535;
// template <typename scalar_t, bool IS_NEOX>

extern void rotary_embedding(
    turbo_types type,
    bool is_neox,
    void* stream,
    int64_t* positions,
    void* query_dst,
    void* key_dst,
    void* query,
    void* key,
    void* cos_sin_cache,
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int64_t dst_query_stride,
    const int64_t dst_key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int64_t num_tokens,
    const int64_t loop_cnt,
    int aivNum) {

        int BlockDim = max_parallel_size > num_tokens ? num_tokens : max_parallel_size;

        if (type == turbo_types::FP16) {
            ROPE_KERNEL_CALL(half);
        } else if (type == turbo_types::BF16) {
            ROPE_KERNEL_CALL(bfloat16_t);
        } else {
            return;
        }
    }
