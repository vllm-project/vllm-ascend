/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_MROPE_H
#define SPLIT_MROPE_H

#include "kernel_operator.h"

namespace SplitMrope {

using namespace AscendC;

template <typename T>
class SplitMropeBF16 {
public:
    __aicore__ inline SplitMropeBF16(){};
    __aicore__ inline void Init(GM_ADDR positions,
        GM_ADDR in_qkv,
        GM_ADDR in_cos_sin_cache,
        GM_ADDR out_query,
        GM_ADDR out_key,
        GM_ADDR out_value,
        SplitMropeTilingData* tiling_data,
        TPipe* pipe
    );
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(SplitMropeTilingData* tiling_data);
    __aicore__ inline void CopyIn(uint64_t index, uint64_t loop_num_token);
    __aicore__ inline void Compute(uint64_t index, uint64_t loop_num_token);
    __aicore__ inline void CopyOut(uint64_t index, uint64_t loop_num_token);
    __aicore__ inline void ConvertRotTensor(
        LocalTensor<T>& in_cal_local,
        LocalTensor<T>& in_queue_local,
        LocalTensor<float>& temp_local,
        LocalTensor<float>& in_cal_fp32_local,
        LocalTensor<float>& in_cal_reverse_fp32_local,
        LocalTensor<float>& neg_one,
        uint64_t num_heads,
        uint64_t loop_num_token
    );
    __aicore__ inline void ComputeMrope(
        LocalTensor<T>& in_queue_local,
        LocalTensor<T>& in_cal_local,
        LocalTensor<float>& in_cal_fp32_local,
        LocalTensor<float>& in_cal_reverse_fp32_local,
        LocalTensor<float>& cos_sin_fp32_local,
        LocalTensor<uint32_t>& offset_local,
        LocalTensor<float>& temp_local,
        uint64_t num_heads,
        uint64_t loop_num_token
    );
    __aicore__ inline void CopyCosSin(
        LocalTensor<float>& cos_sin_fp32_local,
        LocalTensor<T>& in_cos_sin_cache_queue_local,
        LocalTensor<float>& in_cos_sin_cache_fp32_local,
        uint64_t index,
        uint64_t loop_num_token,
        uint64_t cos_sin_offset,
        int64_t num_heads
    );


private:
    // tiling
    uint64_t num_q_heads;
    uint64_t num_kv_heads;
    uint64_t head_size;
    uint16_t rotary_dim;
    uint16_t mrope_section_0;
    uint16_t mrope_section_1;
    uint16_t mrope_section_2;
    uint64_t is_neox_style;
    uint64_t num_tokens;
    uint64_t num_tokens_each_tail_core;
    uint64_t num_tokens_each_front_core;
    uint64_t front_core;
    uint64_t tail_core;
    uint64_t loop_time_each_front_core;
    uint64_t loop_time_each_tail_core;
    uint64_t num_tokens_front_core_each_loop;
    uint64_t num_tokens_tail_core_each_loop;
    uint64_t num_tokens_front_core_last_loop;
    uint64_t num_tokens_tail_core_last_loop;

    // tmp
    uint16_t head_block_len;
    uint16_t rotary_block_len;
    uint16_t cal_block_len;
    uint64_t num_heads_max;
    uint64_t block_idx;
    uint64_t q_size;
    uint64_t kv_size;
    uint64_t num_tokens_each_loop_current_core;
    uint64_t block_offset;
    uint64_t loop_time_current_core;
    uint64_t num_tokens_last_loop_current_core;
    uint16_t half_rotary_dim;

    // global
    GlobalTensor<uint64_t> in_positions_global;
    GlobalTensor<T> in_qkv_global;
    GlobalTensor<T> in_cos_sin_cache_global;

    GlobalTensor<T> out_query_global;
    GlobalTensor<T> out_key_global;
    GlobalTensor<T> out_value_global;

    // pipe
    TQue<QuePosition::VECIN, 1> in_q_queue;
    TQue<QuePosition::VECIN, 1> in_k_queue;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> in_v_queue;

    TQue<QuePosition::VECIN, 1> in_cos_sin_cache_queue;
    TQue<QuePosition::VECIN, 1> in_positions_queue;
    TQue<QuePosition::VECIN, 1> copy_mrope_queue_0;
    TQue<QuePosition::VECIN, 1> copy_mrope_queue_1;
    TQue<QuePosition::VECIN, 1> copy_mrope_queue_2;

    TQue<QuePosition::VECOUT, 1> out_q_queue;
    TQue<QuePosition::VECOUT, 1> out_k_queue;

    // buf
    TBuf<TPosition::VECCALC> neg_one_buf;

    TBuf<TPosition::VECCALC> in_q_cal_buf;
    TBuf<TPosition::VECCALC> in_q_cal_fp32_buf;
    TBuf<TPosition::VECCALC> in_q_cal_reverse_fp32_buf;
    TBuf<TPosition::VECCALC> q_temp_buf;
    TBuf<TPosition::VECCALC> q_offset_buf;

    TBuf<TPosition::VECCALC> in_k_cal_buf;
    TBuf<TPosition::VECCALC> in_k_cal_fp32_buf;
    TBuf<TPosition::VECCALC> in_k_cal_reverse_fp32_buf;
    TBuf<TPosition::VECCALC> k_temp_buf;
    TBuf<TPosition::VECCALC> k_offset_buf;

    TBuf<TPosition::VECCALC> in_cos_sin_cache_fp32_buf;
    TBuf<TPosition::VECCALC> cos_sin_fp32_buf;
};

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::Init(
    GM_ADDR positions,
    GM_ADDR in_qkv,
    GM_ADDR in_cos_sin_cache,
    GM_ADDR out_query,
    GM_ADDR out_key,
    GM_ADDR out_value,
    SplitMropeTilingData* tiling_data,
    TPipe* pipe
) {
    InitTiling(tiling_data);

    this->block_idx = AscendC::GetBlockIdx();

    this->num_tokens_each_loop_current_core = (block_idx < front_core) ? num_tokens_front_core_each_loop :
                                              num_tokens_tail_core_each_loop;
    this->loop_time_current_core = (block_idx < front_core) ? loop_time_each_front_core : loop_time_each_tail_core;
    this->num_tokens_last_loop_current_core = (block_idx < front_core) ? num_tokens_front_core_last_loop :
                                              num_tokens_tail_core_last_loop;

    this->head_block_len = static_cast<uint16_t>(head_size / 16);
    this->rotary_block_len = static_cast<uint16_t>(rotary_dim / 16);

    // cal_block_len: FP32格式占用的块数
    this->cal_block_len = rotary_block_len;
    this->num_heads_max = (num_q_heads > num_kv_heads) ? num_q_heads : num_kv_heads;

    if (this->block_idx < this->front_core) {
        this->block_offset = num_tokens_each_front_core * block_idx;
    } else {
        this->block_offset = num_tokens_each_front_core * front_core + (block_idx - front_core) * num_tokens_each_tail_core;
    }

    // temp
    this->q_size = num_q_heads * head_size;
    this->kv_size = num_kv_heads * head_size;
    this->half_rotary_dim = static_cast<uint16_t>(rotary_dim / 2);

    in_qkv_global.SetGlobalBuffer((__gm__ T* )(in_qkv));
    in_positions_global.SetGlobalBuffer((__gm__ uint64_t *)(positions));
    in_cos_sin_cache_global.SetGlobalBuffer((__gm__ T *)(in_cos_sin_cache));

    out_query_global.SetGlobalBuffer((__gm__ T *)(out_query));
    out_key_global.SetGlobalBuffer((__gm__ T *)(out_key));
    out_value_global.SetGlobalBuffer((__gm__ T *)(out_value));

    // pipe-input
    pipe->InitBuffer(in_q_queue, 1, num_tokens_each_loop_current_core * num_q_heads * rotary_dim * sizeof(T));
    pipe->InitBuffer(in_k_queue, 1, num_tokens_each_loop_current_core * num_kv_heads * rotary_dim * sizeof(T));
    pipe->InitBuffer(in_v_queue, 1, num_tokens_each_loop_current_core * num_kv_heads * rotary_dim * sizeof(T));

    pipe->InitBuffer(in_cos_sin_cache_queue, 1, rotary_dim * sizeof(T));
    pipe->InitBuffer(in_positions_queue, 1, 12 * sizeof(uint64_t));
    pipe->InitBuffer(copy_mrope_queue_0, 1, half_rotary_dim * sizeof(T));
    pipe->InitBuffer(copy_mrope_queue_1, 1, half_rotary_dim * sizeof(T));
    pipe->InitBuffer(copy_mrope_queue_2, 1, half_rotary_dim * sizeof(T));

    // pipe-output
    pipe->InitBuffer(out_q_queue, 1, num_tokens_each_loop_current_core * num_q_heads * head_size * sizeof(T));
    pipe->InitBuffer(out_k_queue, 1, num_tokens_each_loop_current_core * num_kv_heads * head_size * sizeof(T));

    // pipe-buffer
    pipe->InitBuffer(in_q_cal_buf, num_tokens_each_loop_current_core * num_q_heads * rotary_dim * sizeof(T));
    pipe->InitBuffer(in_q_cal_fp32_buf, num_tokens_each_loop_current_core * num_q_heads * rotary_dim * sizeof(float));
    pipe->InitBuffer(in_q_cal_reverse_fp32_buf, num_tokens_each_loop_current_core * num_q_heads * rotary_dim * sizeof(float));

    pipe->InitBuffer(in_k_cal_buf, num_tokens_each_loop_current_core * num_kv_heads * rotary_dim * sizeof(T));
    pipe->InitBuffer(in_k_cal_fp32_buf, num_tokens_each_loop_current_core * num_kv_heads * rotary_dim * sizeof(float));
    pipe->InitBuffer(in_k_cal_reverse_fp32_buf, num_tokens_each_loop_current_core * num_kv_heads * rotary_dim * sizeof(float));

    pipe->InitBuffer(neg_one_buf, num_tokens_each_loop_current_core * num_heads_max * rotary_dim * sizeof(float));
    pipe->InitBuffer(in_cos_sin_cache_fp32_buf, rotary_dim * sizeof(float));
    pipe->InitBuffer(cos_sin_fp32_buf, num_tokens_each_loop_current_core * num_heads_max * rotary_dim * sizeof(float));

    if (is_neox_style == 0) {
        pipe->InitBuffer(q_offset_buf, rotary_dim * sizeof(uint32_t));
        pipe->InitBuffer(k_offset_buf, rotary_dim * sizeof(uint32_t));
        pipe->InitBuffer(q_temp_buf, num_tokens_each_loop_current_core * num_q_heads * rotary_dim * sizeof(float));
        pipe->InitBuffer(k_temp_buf, num_tokens_each_loop_current_core * num_kv_heads * rotary_dim * sizeof(float));
    } else {
        pipe->InitBuffer(q_offset_buf, 0 * sizeof(uint32_t));
        pipe->InitBuffer(k_offset_buf, 0 * sizeof(uint32_t));
        pipe->InitBuffer(q_temp_buf, 0 * sizeof(float));
        pipe->InitBuffer(k_temp_buf, 0 * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::Process() {
    for (uint64_t n = 0; n < loop_time_current_core - 1; n++) {
        CopyIn(n, num_tokens_each_loop_current_core);
        Compute(n, num_tokens_each_loop_current_core);
        CopyOut(n, num_tokens_each_loop_current_core);
    }

    if (num_tokens_last_loop_current_core == 0) {
        CopyIn(loop_time_current_core - 1, num_tokens_each_loop_current_core);
        Compute(loop_time_current_core - 1, num_tokens_each_loop_current_core);
        CopyOut(loop_time_current_core - 1, num_tokens_each_loop_current_core);
    } else {
        CopyIn(loop_time_current_core - 1, num_tokens_last_loop_current_core);
        Compute(loop_time_current_core - 1, num_tokens_last_loop_current_core);
        CopyOut(loop_time_current_core - 1, num_tokens_last_loop_current_core);
    }
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::InitTiling(SplitMropeTilingData* tiling_data) {
    this->num_q_heads = tiling_data->num_q_heads;
    this->num_kv_heads = tiling_data->num_kv_heads;
    this->head_size = tiling_data->head_size;
    this->rotary_dim = tiling_data->rotary_dim;
    this->mrope_section_0 = tiling_data->mrope_section_0;
    this->mrope_section_1 = tiling_data->mrope_section_1;
    this->mrope_section_2 = tiling_data->mrope_section_2;
    this->num_tokens = tiling_data->num_tokens;
    this->num_tokens_each_front_core = tiling_data->num_tokens_each_front_core;
    this->num_tokens_each_tail_core = tiling_data->num_tokens_each_tail_core;
    this->front_core = tiling_data->front_core;
    this->tail_core = tiling_data->tail_core;
    this->num_tokens_front_core_each_loop = tiling_data->num_tokens_front_core_each_loop;
    this->num_tokens_tail_core_each_loop = tiling_data->num_tokens_tail_core_each_loop;
    this->loop_time_each_front_core = tiling_data->loop_time_each_front_core;
    this->loop_time_each_tail_core = tiling_data->loop_time_each_tail_core;
    this->num_tokens_front_core_last_loop = tiling_data->num_tokens_front_core_last_loop;
    this->num_tokens_tail_core_last_loop = tiling_data->num_tokens_tail_core_last_loop;
    this->is_neox_style = tiling_data->is_neox_style;
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::CopyIn(uint64_t index, uint64_t loop_num_token) {
    LocalTensor<T> in_q_queue_local = in_q_queue.AllocTensor<T>();
    LocalTensor<T> in_k_queue_local = in_k_queue.AllocTensor<T>();
    LocalTensor<T> in_v_queue_local = in_v_queue.AllocTensor<T>();

    DataCopy(
        in_q_queue_local,
        in_qkv_global[(block_offset + index * num_tokens_each_loop_current_core) * (q_size + 2 * kv_size)],
        {
            static_cast<uint16_t>(loop_num_token),
            static_cast<uint16_t>(q_size * sizeof(T) / 32),
            static_cast<uint16_t>(2 * kv_size * sizeof(T) / 32),
            0
        }
    );

    DataCopy(
        in_k_queue_local,
        this->in_qkv_global[(block_offset + index * num_tokens_each_loop_current_core) * (q_size + 2 * kv_size) + q_size],
        {
            static_cast<uint16_t>(loop_num_token),
            static_cast<uint16_t>(kv_size * sizeof(T) / 32),
            static_cast<uint16_t>((q_size + kv_size) * sizeof(T) / 32),
            0
        }
    );

    DataCopy(
        in_v_queue_local,
        in_qkv_global[(block_offset + index * num_tokens_each_loop_current_core) * (q_size + 2 * kv_size) + q_size + kv_size],
        {
            static_cast<uint16_t>(loop_num_token),
            static_cast<uint16_t>(kv_size * sizeof(T) / 32),
            static_cast<uint16_t>((q_size + kv_size) * sizeof(T) / 32),
            0
        }
    );

    in_q_queue.EnQue(in_q_queue_local);
    in_k_queue.EnQue(in_k_queue_local);
    in_v_queue.EnQue(in_v_queue_local);
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::Compute(uint64_t index, uint64_t loop_num_token) {
    LocalTensor<T> in_q_queue_local = in_q_queue.DeQue<T>();
    LocalTensor<T> in_k_queue_local = in_k_queue.DeQue<T>();

    LocalTensor<T> out_q_queue_local = out_q_queue.AllocTensor<T>();
    LocalTensor<T> out_k_queue_local = out_k_queue.AllocTensor<T>();

    LocalTensor<T> in_cos_sin_cache_queue_local = in_cos_sin_cache_queue.AllocTensor<T>();

    LocalTensor<T> in_q_cal_local = in_q_cal_buf.Get<T>();
    LocalTensor<float> in_q_cal_fp32_local = in_q_cal_fp32_buf.Get<float>();
    LocalTensor<float> in_q_cal_reverse_fp32_local = in_q_cal_reverse_fp32_buf.Get<float>();

    LocalTensor<T> in_k_cal_local = in_k_cal_buf.Get<T>();
    LocalTensor<float> in_k_cal_fp32_local = in_k_cal_fp32_buf.Get<float>();
    LocalTensor<float> in_k_cal_reverse_fp32_local = in_k_cal_reverse_fp32_buf.Get<float>();

    LocalTensor<float> neg_one = neg_one_buf.Get<float>();

    uint32_t src_shape[2] = {
        1,
        static_cast<uint32_t>(rotary_dim)
    };
    uint32_t dst_shape_4_neg_one[2] = {
        static_cast<uint32_t>(loop_num_token * num_heads_max),
        static_cast<uint32_t>(rotary_dim)
    };

    float positive = 1.0;
    float negative = -1.0;
    Duplicate<float>(neg_one, negative, half_rotary_dim);
    Duplicate<float>(neg_one[half_rotary_dim], positive, half_rotary_dim);
    Broadcast<float, 2, 0, false>(neg_one[rotary_dim], neg_one, dst_shape_4_neg_one, src_shape);

    LocalTensor<uint32_t> q_offset_local= q_offset_buf.Get<uint32_t>();
    LocalTensor<uint32_t> k_offset_local= k_offset_buf.Get<uint32_t>();
    LocalTensor<float> q_temp_local = q_temp_buf.Get<float>();
    LocalTensor<float> k_temp_local = k_temp_buf.Get<float>();
    LocalTensor<float> cos_sin_fp32_local = cos_sin_fp32_buf.Get<float>();
    LocalTensor<float> in_cos_sin_cache_fp32_local = in_cos_sin_cache_fp32_buf.Get<float>();

    ConvertRotTensor(
        in_q_cal_local,
        in_q_queue_local,
        q_temp_local,
        in_q_cal_fp32_local,
        in_q_cal_reverse_fp32_local,
        neg_one,
        num_q_heads,
        loop_num_token
    );
    ConvertRotTensor(
        in_k_cal_local,
        in_k_queue_local,
        k_temp_local,
        in_k_cal_fp32_local,
        in_k_cal_reverse_fp32_local,
        neg_one,
        num_kv_heads,
        loop_num_token
    );

    // copy sin : GM->UB
    CopyCosSin(
        cos_sin_fp32_local,
        in_cos_sin_cache_queue_local,
        in_cos_sin_cache_fp32_local,
        index,
        loop_num_token,
        static_cast<uint64_t>(half_rotary_dim),
        num_heads_max
    );

    // [−x2*sin, x1*sin]
    // q
    Mul(
        in_q_cal_reverse_fp32_local,
        cos_sin_fp32_local,
        in_q_cal_reverse_fp32_local,
        static_cast<uint16_t>(loop_num_token * num_q_heads * rotary_dim)
    );
    // k
    Mul(
        in_k_cal_reverse_fp32_local,
        cos_sin_fp32_local,
        in_k_cal_reverse_fp32_local,
        static_cast<uint16_t>(loop_num_token * num_kv_heads * rotary_dim)
    );

    // copy cos : GM->UB
    CopyCosSin(
        cos_sin_fp32_local,
        in_cos_sin_cache_queue_local,
        in_cos_sin_cache_fp32_local,
        index,
        loop_num_token,
        static_cast<uint64_t>(0),
        num_heads_max
    );
    // q
    ComputeMrope(
        in_q_queue_local,
        in_q_cal_local,
        in_q_cal_fp32_local,
        in_q_cal_reverse_fp32_local,
        cos_sin_fp32_local,
        q_offset_local,
        q_temp_local,
        num_q_heads,
        loop_num_token
    );
    DataCopy(
        out_q_queue_local,
        in_q_queue_local,
        static_cast<uint16_t>(loop_num_token * q_size)
    );
    // k
    ComputeMrope(
        in_k_queue_local,
        in_k_cal_local,
        in_k_cal_fp32_local,
        in_k_cal_reverse_fp32_local,
        cos_sin_fp32_local,
        k_offset_local,
        k_temp_local,
        num_kv_heads,
        loop_num_token
    );
    DataCopy(
        out_k_queue_local,
        in_k_queue_local,
        static_cast<uint16_t>(loop_num_token * kv_size)
    );

    out_q_queue.EnQue<T>(out_q_queue_local);
    out_k_queue.EnQue<T>(out_k_queue_local);

    in_cos_sin_cache_queue.FreeTensor(in_cos_sin_cache_queue_local);
    in_q_queue.FreeTensor(in_q_queue_local);
    in_k_queue.FreeTensor(in_k_queue_local);
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::CopyOut(uint64_t index, uint64_t loop_num_token) {
    LocalTensor<T> out_q_queue_local = out_q_queue.DeQue<T>();
    LocalTensor<T> out_k_queue_local = out_k_queue.DeQue<T>();
    LocalTensor<T> in_v_queue_local = in_v_queue.DeQue<T>();

    DataCopy(
        out_query_global[(block_offset + index * num_tokens_each_loop_current_core) * q_size],
        out_q_queue_local,
        static_cast<uint16_t>(loop_num_token * q_size)
    );

    DataCopy(
        out_key_global[(block_offset + index * num_tokens_each_loop_current_core) * kv_size],
        out_k_queue_local,
        static_cast<uint16_t>(loop_num_token * kv_size)
    );

    DataCopy(
        out_value_global[(block_offset + index * num_tokens_each_loop_current_core) * kv_size],
        in_v_queue_local,
        static_cast<uint16_t>(loop_num_token * kv_size)
    );

    out_q_queue.FreeTensor(out_q_queue_local);
    out_k_queue.FreeTensor(out_k_queue_local);
    in_v_queue.FreeTensor(in_v_queue_local);
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::ConvertRotTensor(
    LocalTensor<T>& in_cal_local,
    LocalTensor<T>& in_queue_local,
    LocalTensor<float>& temp_local,
    LocalTensor<float>& in_cal_fp32_local,
    LocalTensor<float>& in_cal_reverse_fp32_local,
    LocalTensor<float>& neg_one,
    uint64_t num_heads,
    uint64_t loop_num_token
) {
    DataCopy(
        in_cal_local,
        in_queue_local,
        {
            static_cast<uint16_t>(loop_num_token * num_heads),
            rotary_block_len,
            static_cast<uint16_t>(head_block_len - rotary_block_len),
            0
        });

    if (is_neox_style == 0) {
        Cast(
            temp_local,
            in_cal_local,
            AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(loop_num_token * num_heads * rotary_dim)
        );

        uint64_t rsv = 0;
        for (uint16_t i = 0; i < static_cast<uint16_t>(loop_num_token * num_heads); i++) {
            GatherMask(
                in_cal_fp32_local[i * rotary_dim],
                temp_local[i * rotary_dim],
                static_cast<uint8_t>(1),
                true,
                rotary_dim,
                {1, 1, 0, 0},
                rsv
            );
            GatherMask(
                in_cal_fp32_local[static_cast<uint16_t>(i * rotary_dim + half_rotary_dim)],
                temp_local[i * rotary_dim],
                static_cast<uint8_t>(2),
                true,
                rotary_dim,
                {1, 1, 0, 0},
                rsv
            );
        }
    } else {
        Cast(
            in_cal_fp32_local,
            in_cal_local,
            AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(loop_num_token * num_heads * rotary_dim)
        );

    }

    // [x1,x2] -> [x2,x1]
    DataCopy(
        in_cal_reverse_fp32_local,
        in_cal_fp32_local[static_cast<uint16_t>(half_rotary_dim)],
        {
            static_cast<uint16_t>(loop_num_token * num_heads),
            cal_block_len,
            cal_block_len,
            cal_block_len
        });
    DataCopy(
        in_cal_reverse_fp32_local[static_cast<uint16_t>(half_rotary_dim)],
        in_cal_fp32_local,
        {
            static_cast<uint16_t>(loop_num_token * num_heads),
            cal_block_len,
            cal_block_len,
            cal_block_len
        });

    // [x2,x1] -> [-x2, x1]
    Mul(
        in_cal_reverse_fp32_local,
        neg_one,
        in_cal_reverse_fp32_local,
        static_cast<uint16_t>(loop_num_token * num_heads * rotary_dim)
    );
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::ComputeMrope(
    LocalTensor<T>& in_queue_local,
    LocalTensor<T>& in_cal_local,
    LocalTensor<float>& in_cal_fp32_local,
    LocalTensor<float>& in_cal_reverse_fp32_local,
    LocalTensor<float>& cos_sin_fp32_local,
    LocalTensor<uint32_t>& offset_local,
    LocalTensor<float>& temp_local,
    uint64_t num_heads,
    uint64_t loop_num_token
) {
    // [x1 * cos, x2 * cos]
    Mul(
        in_cal_fp32_local,
        cos_sin_fp32_local,
        in_cal_fp32_local,
        static_cast<uint16_t>(loop_num_token * num_heads * rotary_dim)
    );

    // [(x1 * cos - x2 * sin), (x2 * cos + x1 * sin)]
    Add(
        in_cal_fp32_local,
        in_cal_reverse_fp32_local,
        in_cal_fp32_local,
        static_cast<uint16_t>(loop_num_token * num_heads * rotary_dim)
    );

    // fp32 -> bf16
    if (is_neox_style == 0) {
        for (uint32_t i = 0; i < half_rotary_dim; i++) {
            offset_local.SetValue(i * 2, i * 4);
            offset_local.SetValue(i * 2 + 1, (half_rotary_dim + i) * 4);
        }

        for (uint32_t i = 0; i < loop_num_token * num_heads; i++) {
            Gather(
                temp_local[i * rotary_dim],
                in_cal_fp32_local[i * rotary_dim],
                offset_local,
                (uint32_t)0,
                rotary_dim
            );
        }

        Cast(
            in_cal_local,
            temp_local,
            AscendC::RoundMode::CAST_RINT,
            static_cast<uint16_t>(loop_num_token * num_heads * rotary_dim)
        );
    } else {
        Cast(
            in_cal_local,
            in_cal_fp32_local,
            AscendC::RoundMode::CAST_RINT,
            static_cast<uint16_t>(loop_num_token * num_heads * rotary_dim)
        );
    }

    // query = [queryRot, queryPass]
    if (head_size != rotary_dim) {
        DataCopy(
            in_queue_local,
            in_cal_local,
            {
                static_cast<uint16_t>(loop_num_token * num_heads),
                rotary_block_len,
                0,
                static_cast<uint16_t>(head_block_len - rotary_block_len)
            });
    } else {
        DataCopy(
            in_queue_local,
            in_cal_local,
            {
                static_cast<uint16_t>(loop_num_token),
                static_cast<uint16_t>(num_heads * head_block_len),
                0,
                0
            });
    }
}

template <typename T>
__aicore__ inline void SplitMropeBF16<T>::CopyCosSin(
    LocalTensor<float>& cos_sin_fp32_local,
    LocalTensor<T>& in_cos_sin_cache_queue_local,
    LocalTensor<float>& in_cos_sin_cache_fp32_local,
    uint64_t index,
    uint64_t loop_num_token,
    uint64_t cos_sin_offset,
    int64_t num_heads)
{
    uint64_t localStartAddr = 0;
    uint32_t src_shape[2] = {
        1,
        static_cast<uint32_t>(rotary_dim)
    };
    uint32_t dst_shape[2] = {
        static_cast<uint32_t>(num_heads_max),
        static_cast<uint32_t>(rotary_dim)
    };

    for (uint64_t i = 0; i < loop_num_token; ++i) {
        uint64_t offset_pos = block_offset + num_tokens_each_loop_current_core * index + i;

        LocalTensor<uint64_t> in_positions_queue_local = in_positions_queue.AllocTensor<uint64_t>();

        if (mrope_section_0 > 0) {
            // mrope model
            LocalTensor<T> copy_mrope_0_local = copy_mrope_queue_0.AllocTensor<T>();
            LocalTensor<T> copy_mrope_1_local = copy_mrope_queue_1.AllocTensor<T>();
            LocalTensor<T> copy_mrope_2_local = copy_mrope_queue_2.AllocTensor<T>();

            DataCopyPad(
                in_positions_queue_local,
                in_positions_global[offset_pos],
                {3, static_cast<uint16_t>(sizeof(uint64_t)), static_cast<uint32_t>((num_tokens - 1) * sizeof(uint64_t)), 0, 0},
                {false, 0, 0, 0}
            );
            in_positions_queue.EnQue(in_positions_queue_local);
            in_positions_queue_local = in_positions_queue.DeQue<uint64_t>();

            uint64_t pos0 = in_positions_queue_local.GetValue(0);
            uint64_t pos1 = in_positions_queue_local.GetValue(4);
            uint64_t pos2 = in_positions_queue_local.GetValue(8);

            DataCopyPad(
                copy_mrope_0_local,
                in_cos_sin_cache_global[pos0 * rotary_dim + cos_sin_offset],
                {1, static_cast<uint16_t>(mrope_section_0 * sizeof(T)), 0, 0},
                {false, 0, 0, 0}
            );
            copy_mrope_queue_0.EnQue(copy_mrope_0_local);

            DataCopyPad(
                copy_mrope_1_local,
                in_cos_sin_cache_global[pos1 * rotary_dim + mrope_section_0 + cos_sin_offset],
                {1, static_cast<uint16_t>(mrope_section_1 * sizeof(T)), 0, 0},
                {false, 0, 0, 0}
            );
            copy_mrope_queue_1.EnQue(copy_mrope_1_local);

            DataCopyPad(
                copy_mrope_2_local,
                in_cos_sin_cache_global[pos2 * rotary_dim + mrope_section_0 + mrope_section_1 + cos_sin_offset],
                {1, static_cast<uint16_t>(mrope_section_2 * sizeof(T)), 0, 0},
                {true, 8, 0, 0}
            );
            copy_mrope_queue_2.EnQue(copy_mrope_2_local);

            copy_mrope_0_local = copy_mrope_queue_0.DeQue<T>();
            copy_mrope_1_local = copy_mrope_queue_1.DeQue<T>();
            copy_mrope_2_local = copy_mrope_queue_2.DeQue<T>();

            Copy(in_cos_sin_cache_queue_local[32], copy_mrope_2_local, 32, 1, {1, 1, 0, 0});
            Copy(in_cos_sin_cache_queue_local[16], copy_mrope_1_local, 24, 1, {1, 1, 0, 0});
            Copy(in_cos_sin_cache_queue_local, copy_mrope_0_local, 16, 1, {1, 1, 0, 0});

            copy_mrope_queue_0.FreeTensor(copy_mrope_0_local);
            copy_mrope_queue_1.FreeTensor(copy_mrope_1_local);
            copy_mrope_queue_2.FreeTensor(copy_mrope_2_local);
        } else {
            // rope mode
            DataCopyPad(
                in_positions_queue_local,
                in_positions_global[offset_pos],
                {1, static_cast<uint16_t>(sizeof(uint64_t)), 0, 0},
                {false, 0, 0, 0}
            );
            in_positions_queue.EnQue(in_positions_queue_local);
            in_positions_queue_local = in_positions_queue.DeQue<uint64_t>();

            uint64_t pos = in_positions_queue_local.GetValue(0);

            DataCopy(
                in_cos_sin_cache_queue_local,
                in_cos_sin_cache_global[pos * rotary_dim + cos_sin_offset],
                {1, rotary_block_len, 0, 0}
            );

            in_cos_sin_cache_queue.EnQue(in_cos_sin_cache_queue_local);
            in_cos_sin_cache_queue_local = in_cos_sin_cache_queue.DeQue<T>();
        }

        Cast(
            in_cos_sin_cache_fp32_local,
            in_cos_sin_cache_queue_local,
            AscendC::RoundMode::CAST_NONE,
            static_cast<uint16_t>(rotary_dim)
        );

        DataCopy(
            in_cos_sin_cache_fp32_local[half_rotary_dim],
            in_cos_sin_cache_fp32_local,
            {1, static_cast<uint16_t>(cal_block_len), 0, 0}
        );

        Broadcast<float, 2, 0, false>(
            cos_sin_fp32_local[localStartAddr],
            in_cos_sin_cache_fp32_local,
            dst_shape,
            src_shape
        );

        localStartAddr += num_heads * rotary_dim;

        in_positions_queue.FreeTensor(in_positions_queue_local);
    }
}

} // namespace SplitMrope

#endif // SPLIT_MROPE_H