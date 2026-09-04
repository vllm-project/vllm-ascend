/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNPAD_FLASH_ATTENTION_COMMON_H
#define UNPAD_FLASH_ATTENTION_COMMON_H
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common.h"
#include "iterator.h"
#include "gm_to_l1_iterator.h"
#include "gm_to_ub_iterator.h"
#include "l0c_to_gm_iterator.h"
#include "l0c_to_l1_iterator.h"
#include "l0c_to_ub_iterator.h"
#include "l1_to_bt_iterator.h"
#include "l1_to_fb_iterator.h"
#include "l1_to_l0_iterator.h"
#include "l1_to_ub_iterator.h"
#include "common_func.h"
#include "simd.h"
#include "mma.h"
using namespace AscendC;

constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int32_t VECTOR_SIZE = 128;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t BLOCK_LIMIT = 128;
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;
constexpr int32_t UB_FLOAT_BUF_SIZE = 8192;
constexpr int32_t FLOAT_BLOCK_SIZE = 8;
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr int32_t BASE_MASK_SIZE = 128;
constexpr int32_t STRIDE_UPPER_BOUND = 65535;
constexpr int64_t L1_UINT8_BLOCK_SIZE = 131072;   // 128K
constexpr int64_t UB_UINT8_BLOCK_SIZE = 32768;    // 128 * 128 * 2 = 32K
constexpr int32_t DEC_UB_UINT8_BLOCK_SIZE = 8192; // 16 * 128 * 2
constexpr int64_t UB_UINT8_LINE_SIZE = 1024;
constexpr int64_t UB_FLOAT_LINE_SIZE = 256;
constexpr int64_t UB_HALF_LINE_SIZE = 512;
constexpr int32_t INNER_PRECISE_PTR = 1;

__aicore__ __attribute__((always_inline)) inline int32_t GetMin(int32_t valA, int32_t valB)
{
    return (valA <= valB) ? valA : valB;
}

__aicore__ __attribute__((always_inline)) inline int32_t GetMax(int32_t valA, int32_t valB)
{
    return (valA > valB) ? valA : valB;
}

enum AttentonMaskType {
    MASK_TYPE_NONE = 0,
    MASK_TYPE_NORM = 1,
    MASK_TYPE_ALIBI = 2,
    MASK_TYPE_LOOK_AHEAD = 3,
    MASK_TYPE_SWA_NORM = 4,
    MASK_TYPE_SWA_COMPRESS = 5
};

enum PrecType { 
    BMM1_FP16_EXP_FP32 = 0,
    BMM1_FP32_EXP_FP32 = 1,
    BMM2_ONLINE_SOFTMAX_FP16 = 4
};

__aicore__ inline void SyncStart()
{
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID3);
    SET_FLAG(V, M, EVENT_ID0);
    SET_FLAG(V, M, EVENT_ID1);
    SET_FLAG(V, M, EVENT_ID2);
    SET_FLAG(V, M, EVENT_ID3);
    SET_FLAG(V, MTE1, EVENT_ID0);
    SET_FLAG(V, MTE1, EVENT_ID1);
    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);
    SET_FLAG(MTE3, V, EVENT_ID2);
    SET_FLAG(MTE3, V, EVENT_ID3);
    SET_FLAG(MTE1, MTE3, EVENT_ID0);
    SET_FLAG(MTE1, MTE3, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID5);
    SET_FLAG(MTE1, MTE2, EVENT_ID6);
    SET_FLAG(MTE1, MTE2, EVENT_ID7);
    SET_FLAG(V, MTE2, EVENT_ID6);
}

__aicore__ inline void SyncEnd()
{
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
    WAIT_FLAG(V, MTE1, EVENT_ID0);
    WAIT_FLAG(V, MTE1, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE3, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE3, EVENT_ID1);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
    WAIT_FLAG(MTE3, V, EVENT_ID2);
    WAIT_FLAG(MTE3, V, EVENT_ID3);
    WAIT_FLAG(V, M, EVENT_ID0);
    WAIT_FLAG(V, M, EVENT_ID1);
    WAIT_FLAG(V, M, EVENT_ID2);
    WAIT_FLAG(V, M, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(M, MTE1, EVENT_ID3);
    WAIT_FLAG(V, MTE2, EVENT_ID6);
    PIPE_BARRIER(ALL);
}

template <typename T, PrecType prec_type2>
__aicore__ inline void UpdateExp(AscendC::LocalTensor<T> src, uint32_t repeat);

template <>
__aicore__ inline void UpdateExp<float, PrecType::BMM1_FP16_EXP_FP32>(AscendC::LocalTensor<float> src, uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V200, float>(src, src, repeat, 1, 1, uint16_t(8), uint16_t(8));
}

// tensor type, tiling type, exp type, softmax type
template <typename T = half, typename SType = half, PrecType prec_type1 = PrecType::BMM1_FP16_EXP_FP32,
          PrecType prec_type2 = PrecType::BMM1_FP16_EXP_FP32>
class UnpadFlashAttentionCommon {
public:
    __aicore__ inline UnpadFlashAttentionCommon(
        __gm__ uint8_t *__restrict__ gmSrcq, __gm__ uint8_t *__restrict__ gmSrck, __gm__ uint8_t *__restrict__ gmSrcv,
        __gm__ uint8_t *__restrict__ gmSrcm, __gm__ uint8_t *__restrict__ gmSrcLayerid,
        __gm__ uint8_t *__restrict__ gmSrcAlibiCoeff, __gm__ uint8_t *__restrict__ gmSrcLogn,
        __gm__ uint8_t *__restrict__ gmDsto)
        : gmSrcq(gmSrcq), gmSrck(gmSrck), gmSrcv(gmSrcv), gmSrcm(gmSrcm), gmSrcLayerid(gmSrcLayerid),
          gmSrcAlibiCoeff(gmSrcAlibiCoeff), gmSrcLogn(gmSrcLogn), gmDsto(gmDsto)
    {
    }

    __aicore__ inline void Init(int32_t mReal, int32_t nReal, int32_t kReal, int64_t srcqOffsetReal,
                                int64_t srckOffsetReal, int64_t srcvOffsetReal, int64_t srcmOffsetReal0,
                                int64_t srcmOffsetReal1, int64_t dstoOffsetReal, int32_t initGReal, int32_t wrapOReal,
                                int32_t ntokensQReal, int32_t maskStrideReal, int64_t lognOffsetReal,
                                int32_t cubeUpdateSwitch = 0)
    {
        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srcmOffset0 = srcmOffsetReal0;
        srcmOffset1 = srcmOffsetReal1;
        dstoOffset = dstoOffsetReal;
        maskStride = maskStrideReal;

        initG = initGReal;
        wrapO = wrapOReal;
        ntokensQ = ntokensQReal;
        cubeUpdateO = cubeUpdateSwitch;

        lognOffset = lognOffsetReal;
    }

    __aicore__ inline void SetEncoderParams(SType torIn, int32_t kvCopyStrideIn, int32_t isSqrtIn, int32_t precTypeIn)
    {
        tor = torIn;
        kvCopyStride = kvCopyStrideIn;
        isSqrt = isSqrtIn;
        precType = precTypeIn;
        if (precType == PrecType::BMM1_FP32_EXP_FP32) {
            lpUbufSize = UB_FLOAT_BUF_SIZE;
        }
    }

public:
    __aicore__ inline void FlashAttentionNzPrefillCompute(
        const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn, const int32_t __m0, const int32_t __n0,
        const int32_t __n1, const int32_t pp_n_scalar, const int32_t q_tight, const int32_t add_mask_n0,
        const int32_t add_mask_n1, const int32_t long_seq, const SType alibi_coeff, const SType delta0,
        const SType delta1, const uint32_t scale_type, const uint32_t alibi_left_align);

    __aicore__ inline void
    Run(const PromptFlashAttentionBaseApiTilingData *__restrict tilingData,
        __gm__ uint8_t *__restrict__ alibi_coeff_gm,AscendC::GlobalTensor<int64_t> actualSeqLengthsGm,AscendC::GlobalTensor<int64_t> actualSeqLengthsKVGm,
        uint32_t mask_type, uint32_t window_len, uint32_t long_seq, uint64_t stride_qo, uint64_t stride_kv,
        int64_t head_mask_stride, int64_t batch_mask_stride, uint32_t start_batch, uint32_t end_batch,
        int32_t start_blk, int32_t end_blk, uint32_t is_triu, uint32_t alibi_compress_offset, int32_t group_num,
        uint32_t mask_stride, uint32_t q_tokens, int32_t embd, uint32_t q_tight, uint32_t scaleType, SType tor,
        int32_t kv_copy_stride, uint32_t is_sqrt, int64_t heads, uint32_t max_seqlen, uint32_t batch_size,
        int32_t kv_real_heads, const uint32_t alibi_left_align, uint32_t inputLayout);
    __aicore__ inline void InitBatchParam(const PromptFlashAttentionBaseApiTilingData *__restrict tilingData,
    int32_t heads, uint32_t max_seqlen, uint32_t max_kv_seqlen,int32_t embd, uint32_t kvHead, uint32_t embeddingSizeV, uint32_t inputLayout, uint32_t q_tight);
    __aicore__ inline void RowSum(const int32_t __n0, const int32_t fm, int32_t Pingflag);
    __aicore__ inline void SoftmaxUpdate(int32_t fm, int32_t fk, int32_t oSize, int32_t Pingflag, int32_t initGgO,
                                         int32_t mD64);
    __aicore__ inline void UpdateOutput(int32_t fm, int32_t fk, int32_t oSize, int32_t mD64, int32_t __m0);
    __aicore__ inline void SoftMax(const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn,
                                   const int32_t __m0, const int32_t __n0, const int32_t __n1,
                                   const int32_t add_mask_n0, const int32_t add_mask_n1, const SType alibi_coeff,
                                   const SType delta0, const SType delta1, const uint32_t scale_type,
                                   const uint32_t alibi_left_align, uint32_t initGgDm);

public:
    __aicore__ void __set_mask(int32_t len)
    {
        if (len >= 128) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - 64 > 0 ? len - 64 : 0;
        int32_t lowMask = len - 64 >= 0 ? 64 : len;
        if (len < 64) {
            AscendC::SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            AscendC::SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    __aicore__ void __set_vcg_mask(int32_t len)
    {
        if (len > 16) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ void ExpandToBlockHalf(AscendC::LocalTensor<half> dst_tensor, AscendC::LocalTensor<half> src_tensor,
                                      int32_t len)
    {
        for (int32_t vaddsIdx = 0; vaddsIdx < 2; ++vaddsIdx) { // (len,) -> len / 16 个 (16, 16)
            adds_v<ArchType::ASCEND_V200, half>(dst_tensor[vaddsIdx * 8 * BLOCK_SIZE], src_tensor, (half)(0.0),
                                                len / BLOCK_SIZE, // repeat
                                                1, 0, 16, 1);
        }
        PIPE_BARRIER(V);
        for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE); ++vtransIdx) { // (16, len) -> (len, 16)
            tranpose_v<ArchType::ASCEND_V200, half>(dst_tensor[vtransIdx * CUBE_MATRIX_SIZE],
                                                    dst_tensor[vtransIdx * CUBE_MATRIX_SIZE]);
        }
        PIPE_BARRIER(V);
    }
    __aicore__ void ExpandToBlockFloat(AscendC::LocalTensor<float> dst_tensor, AscendC::LocalTensor<float> src_tensor,
                                       int32_t len)
    {
        for (int32_t rowIdx = 0; rowIdx < len; rowIdx++) {
            float scale = (float)*((__ubuf__ float *)src_tensor.GetPhyAddr() + rowIdx);
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            PIPE_BARRIER(V);
            Duplicate<float>(dst_tensor[rowIdx * 16], scale, MASK_PLACEHOLDER, 1, 1, 8);
        }
        PIPE_BARRIER(V);
    }

public:
    int32_t vmPingpongFlag = 1;

    __gm__ uint8_t *__restrict__ gmSrcq;
    __gm__ uint8_t *__restrict__ gmSrck;
    __gm__ uint8_t *__restrict__ gmSrcv;
    __gm__ uint8_t *__restrict__ gmSrcm;
    __gm__ uint8_t *__restrict__ gmSrcLayerid;
    __gm__ uint8_t *__restrict__ gmSrcAlibiCoeff;
    __gm__ uint8_t *__restrict__ gmSrcLogn;
    __gm__ uint8_t *__restrict__ gmDsto;

    const uint32_t lq_buf_offset = 0;
    const uint32_t lk_buf_offset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lv_buf_offset = 2 * (L1_UINT8_BLOCK_SIZE + UB_UINT8_BLOCK_SIZE);
    const uint32_t lp_buf_offset = 2 * L1_UINT8_BLOCK_SIZE;
    const uint32_t lmask_buf_offset = 4 * L1_UINT8_BLOCK_SIZE;
    const uint32_t lalibi_coeff_buf_offset = 4 * (L1_UINT8_BLOCK_SIZE + UB_UINT8_BLOCK_SIZE);
    const uint32_t ldiag_buf_offset = 5 * L1_UINT8_BLOCK_SIZE; // 128 * 128 * 2(fp16) * 2(PingPong) = 64k
    const uint32_t lo_buf_offset = 6 * L1_UINT8_BLOCK_SIZE;    // 128(qSeqStep) * 128(embedDim) * 2(fp16) = 32k

    const uint32_t ls_ubuf_offset = 0;
    const uint32_t lp_ubuf_offset = 0;
    const uint32_t ls32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lo_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE;
    const uint32_t hm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE;
    const uint32_t gm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE;
    const uint32_t dm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 5 * UB_UINT8_LINE_SIZE;
    const uint32_t gl_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 7 * UB_UINT8_LINE_SIZE;
    const uint32_t tiling_para_ub_offset = 4 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_LINE_SIZE;
    const uint32_t logn_ub_offset = 4 * UB_UINT8_BLOCK_SIZE + 31 * UB_UINT8_LINE_SIZE;
    const uint32_t tv_ubuf_offset = 5 * UB_UINT8_BLOCK_SIZE;
    const uint32_t go_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE;
    const uint32_t mask_ubuf_offset = DEC_UB_UINT8_BLOCK_SIZE * 8;

    __cbuf__ uint8_t *l1qBufAddr;
    __cbuf__ uint8_t *l1kBufAddr;
    __cbuf__ uint8_t *l1pBufAddr;
    __cbuf__ uint8_t *l1vBufAddr;
    __cbuf__ uint8_t *l1maskBufAddr;

    AsdopsBuffer<ArchType::ASCEND_V200> buf;

    AscendC::LocalTensor<half> l1qBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lq_buf_offset);
    AscendC::LocalTensor<half> l1kBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lk_buf_offset);
    AscendC::LocalTensor<half> l1vBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lv_buf_offset);
    AscendC::LocalTensor<half> l1pBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lp_buf_offset);
    AscendC::LocalTensor<half> l1maskBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lmask_buf_offset);
    AscendC::LocalTensor<half> l1diagBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(ldiag_buf_offset);
    AscendC::LocalTensor<half> l1oBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lo_buf_offset);

    AscendC::LocalTensor<half> lsUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(ls_ubuf_offset);
    // diagUbuf_tensor复用lsUbuf_tensor空间，128*128*2*2=64k
    AscendC::LocalTensor<half> diagUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(ls_ubuf_offset);
    AscendC::LocalTensor<half> lpUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lp_ubuf_offset);
    AscendC::LocalTensor<T> ls32Ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(ls32_ubuf_offset);
    AscendC::LocalTensor<float> loUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lo_ubuf_offset);
    AscendC::LocalTensor<half> lmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lm_ubuf_offset);
    AscendC::LocalTensor<half> hmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hm_ubuf_offset);
    AscendC::LocalTensor<half> gmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gm_ubuf_offset);
    AscendC::LocalTensor<half> dmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dm_ubuf_offset);
    AscendC::LocalTensor<T> llUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(ll_ubuf_offset);
    AscendC::LocalTensor<T> glUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(gl_ubuf_offset);
    AscendC::LocalTensor<half> tvUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tv_ubuf_offset);
    AscendC::LocalTensor<T> goUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(go_ubuf_offset);
    // toUbuf_tensor复用goUbuf_tensor空间，128*128*2=32k
    AscendC::LocalTensor<half> toUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(go_ubuf_offset);
    AscendC::LocalTensor<half> maskUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(mask_ubuf_offset);
    AscendC::LocalTensor<half> logn_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(logn_ub_offset);

    AscendC::LocalTensor<half> l0aBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, half>(0);
    AscendC::LocalTensor<half> l0bBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, half>(0);
    AscendC::LocalTensor<float> l0cBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    AscendC::GlobalTensor<half> gmSrcq_tensor;
    AscendC::GlobalTensor<half> gmSrck_tensor;
    AscendC::GlobalTensor<half> gmSrcv_tensor;
    AscendC::GlobalTensor<half> gmSrcm_tensor;
    AscendC::GlobalTensor<half> gmDsto_tensor;

    SType tor = 0;
    int32_t precType = 0;
    int32_t ntokensQ = 0;
    int32_t kvCopyStride = 0;
    int64_t srcqOffset = 0;
    int64_t srckOffset = 0;
    int64_t srcvOffset = 0;
    int64_t dstoOffset = 0;
    int64_t srcmOffset0 = 0;
    int64_t srcmOffset1 = 0;
    int64_t lognOffset = 0;
    int32_t maskStride = 0;
    int32_t initG = 0;
    int32_t wrapO = 0;
    int32_t isSqrt = 0;
    int32_t cubeUpdateO = 0;
    int32_t lpUbufSize = L0AB_HALF_BUF_SIZE;
    int32_t q_seqlen_aligned = 0;
    int32_t kv_seqlen_aligned = 0;
    uint32_t q_seqlen_real = 0;
    uint32_t kv_seqlen_real = 0;
    int32_t pp_m_scalar = 0;
    int32_t pp_n_scalar = 0;
    uint64_t addr_q_scalar = 0; // batch offset
    uint64_t addr_k_scalar = 0;
    uint64_t addr_v_scalar = 0;
    uint64_t addr_o_scalar = 0;
    uint32_t cur_total_qblk = 0;
    uint32_t cur_proc_num = 0;
    uint64_t totalQBlkNum = 0;
    uint64_t addrQSeqOffset = 0;
    uint64_t addrKSeqOffset = 0;
    uint64_t addrVSeqOffset = 0;
    uint64_t addrOSeqOffset = 0;
};

template <typename T, typename SType, PrecType prec_type1, PrecType prec_type2>
__aicore__ inline void UnpadFlashAttentionCommon<T, SType, prec_type1, prec_type2>::RowSum(
   const int32_t __n0, const int32_t fm, int32_t Pingflag)
{
    if (__n0 / BLOCK_SIZE > 1) {
        add_v<ArchType::ASCEND_V200, T>(tvUbuf_tensor.ReinterpretCast<T>(),
                                            ls32Ubuf_tensor,
                                            ls32Ubuf_tensor[fm * BLOCK_SIZE],
                                            fm * BLOCK_SIZE * sizeof(T) / 256, // repeat
                                            1,                                   // dstBlockStride
                                            1,                                   // src0BlockStride
                                            1,                                   // src1BlockStride
                                            8,                                   // dstRepeatStride
                                            8,                                   // src0RepeatStride
                                            8                                    // src1RepeatStride
        );
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, T>(tvUbuf_tensor.ReinterpretCast<T>(),
                                            ls32Ubuf_tensor,
                                            0,                   // sid
                                            1,                   // nBurst
                                            fm * BLOCK_SIZE * sizeof(T) / 32, // lenBurst
                                            0,                   // srcGap
                                            0                    // dstGap
        );
        PIPE_BARRIER(V);
    }
    for (int32_t rowsumIdx = 2; rowsumIdx < (__n0 / BLOCK_SIZE); ++rowsumIdx) {
        add_v<ArchType::ASCEND_V200, T>(tvUbuf_tensor.ReinterpretCast<T>(),
                                        tvUbuf_tensor.ReinterpretCast<T>(),
                                        ls32Ubuf_tensor[rowsumIdx * fm * BLOCK_SIZE],
                                        fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, // repeat
                                        1,                                   // dstBlockStride
                                        1,                                   // src0BlockStride
                                        1,                                   // src1BlockStride
                                        8,                                   // dstRepeatStride
                                        8,                                   // src0RepeatStride
                                        8                                    // src1RepeatStride
        );
        PIPE_BARRIER(V);
    }
    AscendC::SetMaskNorm();
    if (__n0 % BLOCK_SIZE > 0) {
        __set_mask(__n0 % BLOCK_SIZE);
        if (__n0 / BLOCK_SIZE > 0) {
            add_v<ArchType::ASCEND_V200, T>(tvUbuf_tensor.ReinterpretCast<T>(),
                                            tvUbuf_tensor.ReinterpretCast<T>(),
                                            ls32Ubuf_tensor[__n0 / BLOCK_SIZE * fm * BLOCK_SIZE],
                                            fm, // repeat
                                            1,  // dstBlockStride
                                            1,  // src0BlockStride
                                            1,  // src1BlockStride
                                            sizeof(T) / 2,  // dstRepeatStride
                                            sizeof(T) / 2,  // src0RepeatStride
                                            sizeof(T) / 2   // src1RepeatStride
            );
            PIPE_BARRIER(V);
            AscendC::SetVectorMask<int8_t>(0x0, 0xffff);
        }
    } else {
        AscendC::SetVectorMask<int8_t>(0x0, 0xffff);
    }

    cadd_v<ArchType::ASCEND_V200, T>(llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE],
                                    tvUbuf_tensor.ReinterpretCast<T>(), fm,
                                    1,  // dstRepeatStride
                                    1,  // srcBlockStride
                                    sizeof(T) / 2); // srcRepeatStride, fp32 2 block
    PIPE_BARRIER(V);
    // Must Sync, otherwise error
    SET_FLAG(V, MTE1, EVENT_ID0);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
}

template <typename T, typename SType, PrecType prec_type1, PrecType prec_type2>
__aicore__ inline void UnpadFlashAttentionCommon<T, SType, prec_type1, prec_type2>::SoftmaxUpdate(
    int32_t fm, int32_t fk, int32_t oSize, int32_t Pingflag, int32_t initGgO, int32_t mD64)
{
    if (cubeUpdateO == 0 && initGgO == 0) { // 需要更新O
        conv_v<ArchType::ASCEND_V200, half, float>(tvUbuf_tensor.ReinterpretCast<float>(),
                                                dmUbuf_tensor[Pingflag * UB_HALF_LINE_SIZE], mD64, 1, 1, uint16_t(8),
                                                uint16_t(4));
        PIPE_BARRIER(V);
        UpdateExp<T, prec_type2>(tvUbuf_tensor.ReinterpretCast<float>(), mD64);
        PIPE_BARRIER(V);
        mul_v<ArchType::ASCEND_V200, T>(glUbuf_tensor, tvUbuf_tensor.ReinterpretCast<T>(), glUbuf_tensor,
                                            mD64, // repeat
                                            1,    // dstBlockStride
                                            1,    // src0BlockStride
                                            1,    // src1BlockStride
                                            8,    // dstRepeatStride
                                            8,    // src0RepeatStride
                                            8     // src1RepeatStride
        );
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, T>(glUbuf_tensor, glUbuf_tensor, llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE],
                                            mD64, // repeat
                                            1,    // dstBlockStride
                                            1,    // src0BlockStride
                                            1,    // src1BlockStride
                                            8,    // dstRepeatStride
                                            8,    // src0RepeatStride
                                            8     // src1RepeatStride
        );

        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbuf_tensor, // broadcast(m_j-1 - m_j)
                      dmUbuf_tensor[Pingflag * UB_HALF_LINE_SIZE], fm);

        conv_v<ArchType::ASCEND_V200, half, float>(tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE / 2],
                                               tvUbuf_tensor,
                                               fm * BLOCK_SIZE / 64, // repeat
                                               1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        UpdateExp<T, prec_type2>(tvUbuf_tensor.ReinterpretCast<T>()[fm * BLOCK_SIZE / 4 * sizeof(SType)],
                                                fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE);
        PIPE_BARRIER(V);

        if (vmPingpongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingpongFlag = 0;
        }

        for (int32_t vmulIdx = 0; vmulIdx < (fk / BLOCK_SIZE); ++vmulIdx) { // e^broadcast(m_j-1 - m_j) * Oj_1
            mul_v<ArchType::ASCEND_V200, T>(goUbuf_tensor[vmulIdx * fm * BLOCK_SIZE],
                                            goUbuf_tensor[vmulIdx * fm * BLOCK_SIZE],
                                            tvUbuf_tensor.ReinterpretCast<T>()[fm * BLOCK_SIZE / 4 * sizeof(SType)],
                                            fm * BLOCK_SIZE * sizeof(T) / 256, // repeat
                                            1,                                   // dstBlockStride
                                            1,                                   // src0BlockStride
                                            1,                                   // src1BlockStride
                                            8,                                   // dstRepeatStride
                                            8,                                   // src0RepeatStride
                                            8                                    // src1RepeatStride
            );
        }
        PIPE_BARRIER(V);

        // 2 for double buffer
        for (int32_t vaddIdx = 0; vaddIdx < 2; ++vaddIdx) { // update Oj
            add_v<ArchType::ASCEND_V200, T>(goUbuf_tensor[vaddIdx * oSize / 2], goUbuf_tensor[vaddIdx * oSize / 2],
                                            loUbuf_tensor.ReinterpretCast<T>()[vaddIdx * oSize / 2],
                                            (oSize * sizeof(T)) / 8 / FLOAT_VECTOR_SIZE, // repeat
                                            1,                             // dstBlockStride
                                            1,                             // src0BlockStride
                                            1,                             // src1BlockStride
                                            8,                             // dstRepeatStride
                                            8,                             // src0RepeatStride
                                            8                              // src1RepeatStride
            );
        }
        PIPE_BARRIER(V);
    } else if (cubeUpdateO == 0) {
        ub_to_ub<ArchType::ASCEND_V200, T>(glUbuf_tensor, llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE],
                                               0,      // sid
                                               1,      // nBurst
                                               fm * sizeof(T) / 32, // lenBurst
                                               0,      // srcGap
                                               0       // dstGap
        );

        PIPE_BARRIER(V);
        if (vmPingpongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingpongFlag = 0;
        }
        ub_to_ub<ArchType::ASCEND_V200, T>(goUbuf_tensor,
                                           loUbuf_tensor.ReinterpretCast<T>(),
                                            0,         // sid
                                            1,         // nBurst
                                            oSize * sizeof(T) / 32, // lenBurst
                                            0,         // srcGap
                                            0          // dstGap
        );
        PIPE_BARRIER(V);
    }
    SET_FLAG(V, MTE1, EVENT_ID0);
}

template <typename T, typename SType, PrecType prec_type1, PrecType prec_type2>
__aicore__ inline void UnpadFlashAttentionCommon<T, SType, prec_type1, prec_type2>::UpdateOutput(
    int32_t fm, int32_t fk, int32_t oSize, int32_t mD64, int32_t __m0)
{
    if (wrapO == 1) {
        conv_v<ArchType::ASCEND_V200, T, half>(glUbuf_tensor.template ReinterpretCast<half>(),
                                                   glUbuf_tensor,
                                                   mD64,        // repeat
                                                   1,           // dstBlockStride
                                                   1,           // srcBlockStride
                                                   uint16_t(4), // dstRepeatStride
                                                   uint16_t(8)  // srcRepeatStride
        );
        PIPE_BARRIER(V);
        for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
            conv_v<ArchType::ASCEND_V200, T, half>(goUbuf_tensor.template ReinterpretCast<half>()[vconvIdx * oSize / 2],
                                                    goUbuf_tensor[vconvIdx * oSize / 2],
                                                    oSize / 2 / FLOAT_VECTOR_SIZE, // repeat
                                                    1,                             // dstBlockStride
                                                    1,                             // srcBlockStride
                                                    uint16_t(4),                   // dstRepeatStride
                                                    uint16_t(8)                    // srcRepeatStride
            );
            PIPE_BARRIER(V);
        }
        ExpandToBlockHalf(tvUbuf_tensor, glUbuf_tensor.template ReinterpretCast<half>(), fm);
        for (int32_t vdivIdx = 0; vdivIdx < (fk / BLOCK_SIZE); ++vdivIdx) { // Oi / li
            div_v<ArchType::ASCEND_V200, half>(goUbuf_tensor.template ReinterpretCast<half>()[vdivIdx * fm * BLOCK_SIZE],
                                               goUbuf_tensor.template ReinterpretCast<half>()[vdivIdx * fm * BLOCK_SIZE],
                                               tvUbuf_tensor,
                                               __m0 * BLOCK_SIZE / VECTOR_SIZE, // repeat
                                               1,                               // dstBlockStride
                                               1,                               // src0BlockStride
                                               1,                               // src1BlockStride
                                               8,                               // dstRepeatStride
                                               8,                               // src0RepeatStride
                                               8                                // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        int32_t blockV = VECTOR_SIZE / BLOCK_SIZE;
        if (__m0 % blockV != 0) {
            __set_mask(__m0 * BLOCK_SIZE % 128);
            div_v<ArchType::ASCEND_V200, half>(goUbuf_tensor.template ReinterpretCast<half>()[__m0 * BLOCK_SIZE / 128 * 128],
                                               goUbuf_tensor.template ReinterpretCast<half>()[__m0 * BLOCK_SIZE / 128 * 128],
                                               tvUbuf_tensor[__m0 / blockV * blockV * 16],
                                               fk / BLOCK_SIZE, // repeat
                                               1,               // dstBlockStride
                                               1,               // src0BlockStride
                                               1,               // src1BlockStride
                                               fm,              // dstRepeatStride
                                               fm,              // src0RepeatStride
                                               0                // src1RepeatStride
            );
            AscendC::SetVectorMask<int8_t>(-1, -1);
        }
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, EVENT_ID2);
        WAIT_FLAG(V, MTE3, EVENT_ID2);

        // move O to gm
        if (ntokensQ <= STRIDE_UPPER_BOUND + fm) {

            ub_to_gm<ArchType::ASCEND_V200, half>(gmDsto_tensor[(int64_t)dstoOffset],
                                                  goUbuf_tensor.template ReinterpretCast<half>(), 0,
                                                  fk / BLOCK_SIZE,  // nburst 32/16=2
                                                  __m0,             // burstlen 32
                                                  fm - __m0,        // strStride 32-32=0
                                                  ntokensQ - __m0); // dstStride 32-32=0
        } else {
            for (uint64_t gmBurstIdx = 0; gmBurstIdx < (fk / BLOCK_SIZE); ++gmBurstIdx) {
                ub_to_gm<ArchType::ASCEND_V200, half>(
                    gmDsto_tensor[(int64_t)dstoOffset + gmBurstIdx * ntokensQ * BLOCK_SIZE],
                    goUbuf_tensor.template ReinterpretCast<half>()[gmBurstIdx * fm * BLOCK_SIZE], 0, 1, __m0, 0, 0);
            }
        }

        if (vmPingpongFlag == 0) {
            SET_FLAG(MTE3, V, EVENT_ID2);
            vmPingpongFlag = 1;
        }
    }
}

template <>
__aicore__ inline void UnpadFlashAttentionCommon<float, half, PrecType::BMM1_FP16_EXP_FP32 , PrecType::BMM1_FP16_EXP_FP32>::SoftMax(
                                                const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn,
                                                const int32_t __m0, const int32_t __n0, const int32_t __n1,
                                                const int32_t add_mask_n0, const int32_t add_mask_n1,
                                                const half alibi_coeff, const half delta0, const half delta1,
                                                const uint32_t scaleType, const uint32_t alibi_left_align, uint32_t initGgDm)
{
    int32_t Pingflag = 0;
    int32_t Pongflag = 1;
    int32_t pSize = fm * fn;
    int32_t pSize_b = fm * bn;
    int32_t mD128 = (fm + VECTOR_SIZE - 1) / VECTOR_SIZE;
    // 3. ################ Softmax Ping Starts #######################
    WAIT_FLAG(M, V, Pingflag);
    WAIT_FLAG(MTE3, V, Pingflag);

    l0c_to_ub<ArchType::ASCEND_V200, float, half>(lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                                  l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], 1,
                                                  pSize / CUBE_MATRIX_SIZE, 0, 0);
    SET_FLAG(V, M, Pingflag);
    PIPE_BARRIER(V);

    if (scaleType == 1) {
        WAIT_FLAG(V, MTE2, EVENT_ID6);
        AscendC::GlobalTensor<half> logn_gm_tensor;
        logn_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcLogn));
        gm_to_ub<ArchType::ASCEND_V200, half>(logn_ub_tensor, logn_gm_tensor[lognOffset], 0, 1, fm / BLOCK_SIZE, 0, 0);

        SET_FLAG(MTE2, V, Pingflag + 2);
        WAIT_FLAG(MTE2, V, Pingflag + 2);

        // expand logn to block
        ExpandToBlockHalf(tvUbuf_tensor, logn_ub_tensor, fm); // (fm,) -> (fm, 16)
        PIPE_BARRIER(V);
        // loop for column, repeat for row
        for (uint32_t fn_block_idx = 0; fn_block_idx < (__n0 / VECTOR_SIZE); ++fn_block_idx) {
            mul_v<ArchType::ASCEND_V200, half>(
                lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + fn_block_idx * fm * VECTOR_SIZE],
                lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + fn_block_idx * fm * VECTOR_SIZE],
                tvUbuf_tensor.ReinterpretCast<half>(),
                __m0,                    // repeat
                fm,                        // dstBlockStride
                fm,                        // src0BlockStride
                0,                        // src1BlockStride
                1,  // dstRepeatStride
                1,  // src0RepeatStride
                1                         // src1RepeatStride
            );
        }
        if (__n0 % VECTOR_SIZE > 0) {
            __set_mask(__n0 % VECTOR_SIZE);
            mul_v<ArchType::ASCEND_V200, half>(
                lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + __n0 / VECTOR_SIZE * fm * VECTOR_SIZE],
                lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + __n0 / VECTOR_SIZE * fm * VECTOR_SIZE],
                tvUbuf_tensor.ReinterpretCast<half>(),
                __m0,                    // repeat
                fm,                        // dstBlockStride
                fm,                        // src0BlockStride
                0,                        // src1BlockStride
                1,  // dstRepeatStride
                1,  // src0RepeatStride
                1                         // src1RepeatStride
            );
            __set_mask(VECTOR_SIZE);
        }
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE2, EVENT_ID6);
    }


    // 3.1. mask(attention score * tor)
    muls_v<ArchType::ASCEND_V200, half>(lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                        lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], tor,
                                        pSize / 128,             // repeat
                                        1,                       // dstBlockStride
                                        1,                       // srcBlockStride
                                        uint16_t(8), uint16_t(8) // dstRepeatStride
    );
    PIPE_BARRIER(V);
    // Must Sync, otherwise error
    WAIT_FLAG(V, MTE1, EVENT_ID0);
    // Mask Ping Load UB
    if ((gmSrcm != nullptr) && (add_mask_n0 == 1)) {
        WAIT_FLAG(MTE2, MTE1, Pingflag + 2);
        l1_to_ub<ArchType::ASCEND_V200, half>(loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                                              l1maskBufAddr_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                              1,                    // nBurst, 次数
                                              fm * fn / BLOCK_SIZE, // lenBurst
                                              0,                    // srcStride，尾-头,32byte
                                              0);
        SET_FLAG(MTE1, MTE2, Pingflag + 2);
        SET_FLAG(MTE1, V, Pingflag);
        WAIT_FLAG(MTE1, V, Pingflag);
        if (gmSrcAlibiCoeff != nullptr) {
            if (srcmOffset0) {
                if (isSqrt == 1) {
                    mul_v<ArchType::ASCEND_V200, half>(
                        loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                        loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                        loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                        fm * fn / VECTOR_SIZE, // repeat
                        1,                     // dstBlockStride
                        1,                     // src0BlockStride
                        1,                     // src1BlockStride
                        8,                     // dstRepeatStride
                        8,                     // src0RepeatStride
                        8                      // src1RepeatStride
                    );
                    PIPE_BARRIER(V);
                }
                adds_v<ArchType::ASCEND_V200, half>(
                    loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                    loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE], (half)delta0,
                    fm * fn / VECTOR_SIZE, // repeat
                    1,                     // dstBlockStride
                    1,                     // src0BlockStride
                    8,                     // dstRepeatStride
                    8                      // src0RepeatStride
                );
                PIPE_BARRIER(V);
                if (isSqrt == 1) {
                    sqrt_v<ArchType::ASCEND_V200, half>(
                        loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                        loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                        fm * fn / VECTOR_SIZE, // repeat
                        1,                     // dstBlockStride
                        1,                     // src0BlockStride
                        8,                     // dstRepeatStride
                        8                      // src0RepeatStride
                    );
                    PIPE_BARRIER(V);
                }
            }
            muls_v<ArchType::ASCEND_V200, half>(loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                                                loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                                                (half)alibi_coeff,
                                                fm * fn / VECTOR_SIZE, // repeat
                                                1,                     // dstBlockStride
                                                1,                     // src0BlockStride
                                                8,                     // dstRepeatStride
                                                8                      // src0RepeatStride
            );
            PIPE_BARRIER(V);
        }
        add_v<ArchType::ASCEND_V200, half>(lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                           lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                           loUbuf_tensor.ReinterpretCast<half>()[Pingflag * L0AB_HALF_BUF_SIZE],
                                           fm * fn / VECTOR_SIZE, // repeat
                                           1,                     // dstBlockStride
                                           1,                     // src0BlockStride
                                           1,                     // src1BlockStride
                                           8,                     // dstRepeatStride
                                           8,                     // src0RepeatStride
                                           8                      // src1RepeatStride
        );

        PIPE_BARRIER(V);
    }
    // 3. softmax part
    if (__n0 / BLOCK_SIZE > 1) { // 前两个(fm, 16)求最大值
        max_v<ArchType::ASCEND_V200, half>(tvUbuf_tensor, lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                           lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + fm * BLOCK_SIZE],
                                           fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                                           1,                             // dstBlockStride
                                           1,                             // src0BlockStride
                                           1,                             // src1BlockStride
                                           8,                             // dstRepeatStride
                                           8,                             // src0RepeatStride
                                           8                              // src1RepeatStride
        );
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, half>(tvUbuf_tensor, lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                              0,  // sid
                                              1,  // nBurst
                                              fm, // lenBurst
                                              0,  // srcGap
                                              0   // dstGap
        );

        PIPE_BARRIER(V);
    }
    for (int32_t rowmaxIdx = 2; rowmaxIdx < (__n0 / BLOCK_SIZE); ++rowmaxIdx) { // 循环比较(fm, 16)
        max_v<ArchType::ASCEND_V200, half>(tvUbuf_tensor, tvUbuf_tensor,
                                           lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + rowmaxIdx * fm * BLOCK_SIZE],
                                           fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                                           1,                             // dstBlockStride
                                           1,                             // src0BlockStride
                                           1,                             // src1BlockStride
                                           8,                             // dstRepeatStride
                                           8,                             // src0RepeatStride
                                           8                              // src1RepeatStride
        );
        PIPE_BARRIER(V);
    }
    if (__n0 % BLOCK_SIZE > 0) {
        __set_mask(__n0 % BLOCK_SIZE);
        if (__n0 / BLOCK_SIZE > 0) {
            max_v<ArchType::ASCEND_V200, half>(
                tvUbuf_tensor, tvUbuf_tensor,
                lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + __n0 / BLOCK_SIZE * fm * BLOCK_SIZE],
                fm, // repeat
                1,  // dstBlockStride
                1,  // src0BlockStride
                1,  // src1BlockStride
                1,  // dstRepeatStride
                1,  // src0RepeatStride
                1   // src1RepeatStride
            );

            PIPE_BARRIER(V);
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
    if (__n0 < BLOCK_SIZE) {
        __set_vcg_mask(__n0);
    }
    cgmax_v<ArchType::ASCEND_V200, half>(lmUbuf_tensor, tvUbuf_tensor, fm * BLOCK_SIZE / VECTOR_SIZE, 1, 1, 8);
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);

    if (initGgDm == 0) { // 需要update m_j
        max_v<ArchType::ASCEND_V200, half>(hmUbuf_tensor, lmUbuf_tensor, gmUbuf_tensor,
                                           mD128, // repeat
                                           1,     // dstBlockStride
                                           1,     // src0BlockStride
                                           1,     // src1BlockStride
                                           8,     // dstRepeatStride
                                           8,     // src0RepeatStride
                                           8      // src1RepeatStride
        );
        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(dmUbuf_tensor[Pingflag * UB_HALF_LINE_SIZE], gmUbuf_tensor, hmUbuf_tensor,
                                           mD128, // repeat
                                           1,     // dstBlockStride
                                           1,     // src0BlockStride
                                           1,     // src1BlockStride
                                           8,     // dstRepeatStride
                                           8,     // src0RepeatStride
                                           8      // src1RepeatStride
        );

        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, half>(hmUbuf_tensor, lmUbuf_tensor,
                                              0,               // sid
                                              1,               // nBurst
                                              fm / BLOCK_SIZE, // lenBurst
                                              0,               // srcGap
                                              0                // dstGap
        );
        PIPE_BARRIER(V);
    }

    ub_to_ub<ArchType::ASCEND_V200, half>(gmUbuf_tensor, hmUbuf_tensor,
                                          0,               // sid
                                          1,               // nBurst
                                          fm / BLOCK_SIZE, // lenBurst
                                          0,               // srcGap
                                          0                // dstGap
    );

    initGgDm = 0;
    PIPE_BARRIER(V);
    ExpandToBlockHalf(tvUbuf_tensor, hmUbuf_tensor, fm);                // (fm,) -> (fm, 16)
    for (int32_t vsubIdx = 0; vsubIdx < (fn / BLOCK_SIZE); ++vsubIdx) { // (fm, fn)
        sub_v<ArchType::ASCEND_V200, half>(lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE],
                                           lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE],
                                           tvUbuf_tensor,
                                           fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                                           1,                             // dstBlockStride
                                           1,                             // src0BlockStride
                                           1,                             // src1BlockStride
                                           8,                             // dstRepeatStride
                                           8,                             // src0RepeatStride
                                           8                              // src1RepeatStride
        );
    }
    PIPE_BARRIER(V);
    // 2 for Repeatimes above 255
    for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
        conv_v<ArchType::ASCEND_V200, half, float>(ls32Ubuf_tensor[vconvIdx * pSize / 2],
                                                   lsUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2],
                                                   pSize / 2 / FLOAT_VECTOR_SIZE, // repeat
                                                   1,                             // dstBlockStride
                                                   1,                             // srcBlockStride
                                                   uint16_t(8),                   // dstRepeatStride
                                                   uint16_t(4)                    // srcRepeatStride
        );
    }
    PIPE_BARRIER(V);
    // 2 for Repeatimes above 255
    for (int32_t vexpIdx = 0; vexpIdx < 2; ++vexpIdx) {
        exp_v<ArchType::ASCEND_V200, float>(ls32Ubuf_tensor[vexpIdx * pSize / 2], ls32Ubuf_tensor[vexpIdx * pSize / 2],
                                            pSize / 2 / FLOAT_VECTOR_SIZE, // repeat
                                            1,                             // dstBlockStride
                                            1,                             // srcBlockStride
                                            8,                             // dstRepeatStride
                                            8                              // srcRepeatStride
        );
    }
    PIPE_BARRIER(V);
    // 2 for Repeatimes above 255
    for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
        conv_v<ArchType::ASCEND_V200, float, half>(lpUbuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2],
                                                   ls32Ubuf_tensor[vconvIdx * pSize / 2],
                                                   pSize / 2 / FLOAT_VECTOR_SIZE, // repeat
                                                   1,                             // dstBlockStride
                                                   1,                             // srcBlockStride
                                                   4,                             // dstRepeatStride
                                                   8                              // srcRepeatStride
        );
    }
    PIPE_BARRIER(V);
    SET_FLAG(V, MTE3, Pingflag);
    if (__n0 / BLOCK_SIZE > 1) {
        add_v<ArchType::ASCEND_V200, float>(tvUbuf_tensor.ReinterpretCast<float>(), ls32Ubuf_tensor,
                                            ls32Ubuf_tensor[fm * BLOCK_SIZE],
                                            fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, // repeat
                                            1,                                   // dstBlockStride
                                            1,                                   // src0BlockStride
                                            1,                                   // src1BlockStride
                                            8,                                   // dstRepeatStride
                                            8,                                   // src0RepeatStride
                                            8                                    // src1RepeatStride
        );
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, float>(tvUbuf_tensor.ReinterpretCast<float>(), ls32Ubuf_tensor,
                                               0,                   // sid
                                               1,                   // nBurst
                                               fm * BLOCK_SIZE / 8, // lenBurst
                                               0,                   // srcGap
                                               0                    // dstGap
        );
        PIPE_BARRIER(V);
    }
    for (int32_t rowsumIdx = 2; rowsumIdx < (__n0 / BLOCK_SIZE); ++rowsumIdx) {
        add_v<ArchType::ASCEND_V200, float>(tvUbuf_tensor.ReinterpretCast<float>(),
                                            tvUbuf_tensor.ReinterpretCast<float>(),
                                            ls32Ubuf_tensor[rowsumIdx * fm * BLOCK_SIZE],
                                            fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, // repeat
                                            1,                                   // dstBlockStride
                                            1,                                   // src0BlockStride
                                            1,                                   // src1BlockStride
                                            8,                                   // dstRepeatStride
                                            8,                                   // src0RepeatStride
                                            8                                    // src1RepeatStride
        );
        PIPE_BARRIER(V);
    }
    AscendC::SetMaskNorm();
    if (__n0 % BLOCK_SIZE > 0) {
        __set_mask(__n0 % BLOCK_SIZE);
        if (__n0 / BLOCK_SIZE > 0) {
            add_v<ArchType::ASCEND_V200, float>(tvUbuf_tensor.ReinterpretCast<float>(),
                                                tvUbuf_tensor.ReinterpretCast<float>(),
                                                ls32Ubuf_tensor[__n0 / BLOCK_SIZE * fm * BLOCK_SIZE],
                                                fm, // repeat
                                                1,  // dstBlockStride
                                                1,  // src0BlockStride
                                                1,  // src1BlockStride
                                                2,  // dstRepeatStride
                                                2,  // src0RepeatStride
                                                2   // src1RepeatStride
            );
            PIPE_BARRIER(V);
            AscendC::SetVectorMask<int8_t>(0x0, 0xffff);
        }
    } else {
        AscendC::SetVectorMask<int8_t>(0x0, 0xffff);
    }

    cadd_v<ArchType::ASCEND_V200, float>(llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE],
                                         tvUbuf_tensor.ReinterpretCast<float>(), fm,
                                         1,  // dstRepeatStride
                                         1,  // srcBlockStride
                                         2); // srcRepeatStride, fp32 2 block

    PIPE_BARRIER(V);
    // Must Sync, otherwise error
    SET_FLAG(V, MTE1, EVENT_ID0);
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    // 3. ################ Softmax Ping Ends #######################
    // 4. ################ Softmax Pong Starts #######################
    if (__n1 != -1) {
        WAIT_FLAG(M, V, Pongflag);
        WAIT_FLAG(MTE3, V, Pongflag);
        l0c_to_ub<ArchType::ASCEND_V200, float, half>(lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                                      l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], 1,
                                                      pSize_b / CUBE_MATRIX_SIZE, 0, 0);
        PIPE_BARRIER(V);
        SET_FLAG(V, M, Pongflag);

        if (scaleType == 1) {
            // expand logn to block
            ExpandToBlockHalf(tvUbuf_tensor, logn_ub_tensor, fm); // (fm,) -> (fm, 16)
            PIPE_BARRIER(V);
            // loop for column, repeat for row
            for (uint32_t fn_block_idx = 0; fn_block_idx < (__n1 / VECTOR_SIZE); ++fn_block_idx) {
                mul_v<ArchType::ASCEND_V200, half>(
                    lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + fn_block_idx * fm * VECTOR_SIZE],
                    lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + fn_block_idx * fm * VECTOR_SIZE],
                    tvUbuf_tensor.ReinterpretCast<half>(),
                    __m0,                    // repeat
                    fm,                        // dstBlockStride
                    fm,                        // src0BlockStride
                    0,                        // src1BlockStride
                    1,  // dstRepeatStride
                    1,  // src0RepeatStride
                    1                         // src1RepeatStride
                );
            }
            if (__n1 % VECTOR_SIZE > 0) {
                __set_mask(__n1 % VECTOR_SIZE);
                mul_v<ArchType::ASCEND_V200, half>(
                    lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + __n1 / VECTOR_SIZE * fm * VECTOR_SIZE],
                    lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + __n1 / VECTOR_SIZE * fm * VECTOR_SIZE],
                    tvUbuf_tensor.ReinterpretCast<half>(),
                    __m0,                    // repeat
                    fm,                        // dstBlockStride
                    fm,                        // src0BlockStride
                    0,                        // src1BlockStride
                    1,  // dstRepeatStride
                    1,  // src0RepeatStride
                    1                         // src1RepeatStride
                );
                __set_mask(VECTOR_SIZE);
            }
            PIPE_BARRIER(V);
        }

        // 3.1. mask(attention score * tor)
        muls_v<ArchType::ASCEND_V200, half>(lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                            lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], tor,
                                            pSize_b / 128, // repeat
                                            1,             // dstBlockStride
                                            1,             // srcBlockStride
                                            uint16_t(8),   // dstRepeatStride
                                            uint16_t(8)    // srcRepeatStride
        );
        PIPE_BARRIER(V);
        WAIT_FLAG(V, MTE1, EVENT_ID0);
        if ((gmSrcm != nullptr) && (add_mask_n1 == 1)) {
            // Mask Pong Load UB
            WAIT_FLAG(MTE2, MTE1, Pongflag + 2);
            l1_to_ub<ArchType::ASCEND_V200, half>(loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                                                  l1maskBufAddr_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                                  1,                    // nBurst, 次数
                                                  fm * bn / BLOCK_SIZE, // lenBurst
                                                  0,                    // srcStride，尾-头,32byte
                                                  0);
            SET_FLAG(MTE1, MTE2, Pongflag + 2);
            SET_FLAG(MTE1, V, Pongflag);
            WAIT_FLAG(MTE1, V, Pongflag);
            if (gmSrcAlibiCoeff != nullptr) {
                if (srcmOffset1) {
                    if (isSqrt == 1) {
                        mul_v<ArchType::ASCEND_V200, half>(

                            loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                            loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                            loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                            fm * fn / VECTOR_SIZE, // repeat
                            1,                     // dstBlockStride
                            1,                     // src0BlockStride
                            1,                     // src1BlockStride
                            8,                     // dstRepeatStride
                            8,                     // src0RepeatStride
                            8                      // src1RepeatStride
                        );
                        PIPE_BARRIER(V);
                    }
                    adds_v<ArchType::ASCEND_V200, half>(
                        loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                        loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE], (half)delta1,
                        fm * fn / VECTOR_SIZE, // repeat
                        1,                     // dstBlockStride
                        1,                     // src0BlockStride
                        8,                     // dstRepeatStride
                        8                      // src0RepeatStride
                    );
                    PIPE_BARRIER(V);
                    if (isSqrt == 1) {
                        sqrt_v<ArchType::ASCEND_V200, half>(
                            loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                            loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                            fm * fn / VECTOR_SIZE, // repeat
                            1,                     // dstBlockStride
                            1,                     // src0BlockStride
                            8,                     // dstRepeatStride
                            8                      // src0RepeatStride
                        );
                        PIPE_BARRIER(V);
                    }
                }
                muls_v<ArchType::ASCEND_V200, half>(
                    loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                    loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE], (half)alibi_coeff,
                    fm * fn / VECTOR_SIZE, // repeat
                    1,                     // dstBlockStride
                    1,                     // src0BlockStride
                    8,                     // dstRepeatStride
                    8                      // src0RepeatStride
                );
                PIPE_BARRIER(V);
            }
            add_v<ArchType::ASCEND_V200, half>(lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                               lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                               loUbuf_tensor.ReinterpretCast<half>()[Pongflag * L0AB_HALF_BUF_SIZE],
                                               fm * bn / VECTOR_SIZE, // repeat
                                               1,                     // dstBlockStride
                                               1,                     // src0BlockStride
                                               1,                     // src1BlockStride
                                               8,                     // dstRepeatStride
                                               8,                     // src0RepeatStride
                                               8                      // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        // 3. softmax part
        if (__n1 / BLOCK_SIZE > 1) { // 前两个(fm, 16)求最大值
            max_v<ArchType::ASCEND_V200, half>(tvUbuf_tensor, lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                               lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + fm * BLOCK_SIZE],
                                               fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                                               1,                             // dstBlockStride
                                               1,                             // src0BlockStride
                                               1,                             // src1BlockStride
                                               8,                             // dstRepeatStride
                                               8,                             // src0RepeatStride
                                               8                              // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            ub_to_ub<ArchType::ASCEND_V200, half>(tvUbuf_tensor, lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                                  0,  // sid
                                                  1,  // nBurst
                                                  fm, // lenBurst
                                                  0,  // srcGap
                                                  0   // dstGap
            );

            PIPE_BARRIER(V);
        }
        for (int32_t rowmaxIdx = 2; rowmaxIdx < (__n1 / BLOCK_SIZE); ++rowmaxIdx) { // 循环比较(fm, 16)
            max_v<ArchType::ASCEND_V200, half>(
                tvUbuf_tensor, tvUbuf_tensor,
                lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + rowmaxIdx * fm * BLOCK_SIZE],
                fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                1,                             // dstBlockStride
                1,                             // src0BlockStride
                1,                             // src1BlockStride
                8,                             // dstRepeatStride
                8,                             // src0RepeatStride
                8                              // src1RepeatStride
            );

            PIPE_BARRIER(V);
        }
        if (__n1 % BLOCK_SIZE > 0) {
            __set_mask(__n1 % BLOCK_SIZE);
            if (__n1 / BLOCK_SIZE > 0) {
                max_v<ArchType::ASCEND_V200, half>(
                    tvUbuf_tensor, tvUbuf_tensor,
                    lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + __n1 / BLOCK_SIZE * fm * BLOCK_SIZE],
                    fm, // repeat
                    1,  // dstBlockStride
                    1,  // src0BlockStride
                    1,  // src1BlockStride
                    1,  // dstRepeatStride
                    1,  // src0RepeatStride
                    1   // src1RepeatStride
                );
                max_v<ArchType::ASCEND_V200, half>(
                    tvUbuf_tensor, tvUbuf_tensor,
                    lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + __n1 / BLOCK_SIZE * fm * BLOCK_SIZE], fm, 1, 1, 1, 1,
                    1, 1);
                PIPE_BARRIER(V);
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (__n1 < BLOCK_SIZE) {
            __set_vcg_mask(__n1);
        }
        cgmax_v<ArchType::ASCEND_V200, half>(lmUbuf_tensor, tvUbuf_tensor, fm * BLOCK_SIZE / VECTOR_SIZE, 1, 1, 8);
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        PIPE_BARRIER(V);

        if (initGgDm == 0) { // 需要update m_j
            max_v<ArchType::ASCEND_V200, half>(hmUbuf_tensor, lmUbuf_tensor, gmUbuf_tensor,
                                               mD128, // repeat
                                               1,     // dstBlockStride
                                               1,     // src0BlockStride
                                               1,     // src1BlockStride
                                               8,     // dstRepeatStride
                                               8,     // src0RepeatStride
                                               8      // src1RepeatStride
            );

            PIPE_BARRIER(V);
            sub_v<ArchType::ASCEND_V200, half>(dmUbuf_tensor[Pongflag * UB_HALF_LINE_SIZE], gmUbuf_tensor,
                                               hmUbuf_tensor,
                                               mD128, // repeat
                                               1,     // dstBlockStride
                                               1,     // src0BlockStride
                                               1,     // src1BlockStride
                                               8,     // dstRepeatStride
                                               8,     // src0RepeatStride
                                               8      // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            ub_to_ub<ArchType::ASCEND_V200, half>(hmUbuf_tensor, lmUbuf_tensor,
                                                  0,               // sid
                                                  1,               // nBurst
                                                  fm / BLOCK_SIZE, // lenBurst
                                                  0,               // srcGap
                                                  0                // dstGap
            );
            PIPE_BARRIER(V);
        }
        ub_to_ub<ArchType::ASCEND_V200, half>(gmUbuf_tensor, hmUbuf_tensor,
                                              0,               // sid
                                              1,               // nBurst
                                              fm / BLOCK_SIZE, // lenBurst
                                              0,               // srcGap
                                              0                // dstGap
        );
        ub_to_ub<ArchType::ASCEND_V200, half>(gmUbuf_tensor, hmUbuf_tensor, 0, 1, fm / BLOCK_SIZE, 0, 0);
        initGgDm = 0;
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbuf_tensor, hmUbuf_tensor, fm);                // (fm,) -> (fm, 16)
        for (int32_t vsubIdx = 0; vsubIdx < (bn / BLOCK_SIZE); ++vsubIdx) { // (fm, bn)
            sub_v<ArchType::ASCEND_V200, half>(lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE],
                                               lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE],
                                               tvUbuf_tensor,
                                               fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                                               1,                             // dstBlockStride
                                               1,                             // src0BlockStride
                                               1,                             // src1BlockStride
                                               8,                             // dstRepeatStride
                                               8,                             // src0RepeatStride
                                               8                              // src1RepeatStride
            );
        }
        PIPE_BARRIER(V);
        // 2 for Repeatimes above 255
        for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
            conv_v<ArchType::ASCEND_V200, half, float>(
                ls32Ubuf_tensor[vconvIdx * pSize_b / 2],
                lsUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize_b / 2],
                pSize_b / 2 / FLOAT_VECTOR_SIZE, // repeat
                1,                               // dstBlockStride
                1,                               // srcBlockStride
                uint16_t(8),                     // dstRepeatStride
                uint16_t(4)                      // srcRepeatStride
            );
        }
        PIPE_BARRIER(V);
        for (int32_t vexpIdx = 0; vexpIdx < 2; ++vexpIdx) {
            exp_v<ArchType::ASCEND_V200, float>(ls32Ubuf_tensor[vexpIdx * pSize_b / 2],
                                                ls32Ubuf_tensor[vexpIdx * pSize_b / 2],
                                                pSize_b / 2 / FLOAT_VECTOR_SIZE, // repeat
                                                1,                               // dstBlockStride
                                                1,                               // srcBlockStride
                                                8,                               // dstRepeatStride
                                                8                                // srcRepeatStride
            );
        }
        PIPE_BARRIER(V);

        // 2 for double buffer
        for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
            conv_v<ArchType::ASCEND_V200, float, half>(
                lpUbuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize_b / 2],
                ls32Ubuf_tensor[vconvIdx * pSize_b / 2],
                pSize_b / 2 / FLOAT_VECTOR_SIZE, // repeat
                1,                               // dstBlockStride
                1,                               // srcBlockStride
                4,                               // dstRepeatStride
                8                                // srcRepeatStride
            );
        }
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, Pongflag);
        if (__n1 / BLOCK_SIZE > 1) {
            add_v<ArchType::ASCEND_V200, float>(tvUbuf_tensor.ReinterpretCast<float>(), ls32Ubuf_tensor,
                                                ls32Ubuf_tensor[fm * BLOCK_SIZE],
                                                fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, // repeat
                                                1,                                   // dstBlockStride
                                                1,                                   // src0BlockStride
                                                1,                                   // src1BlockStride
                                                8,                                   // dstRepeatStride
                                                8,                                   // src0RepeatStride
                                                8                                    // src1RepeatStride
            );
            PIPE_BARRIER(V);
        } else {
            ub_to_ub<ArchType::ASCEND_V200, float>(tvUbuf_tensor.ReinterpretCast<float>(), ls32Ubuf_tensor,
                                                   0,                   // sid
                                                   1,                   // nBurst
                                                   fm * BLOCK_SIZE / 8, // lenBurst
                                                   0,                   // srcGap
                                                   0                    // dstGap
            );
            PIPE_BARRIER(V);
        }
        for (int32_t rowsumIdx = 2; rowsumIdx < (__n1 / BLOCK_SIZE); ++rowsumIdx) {
            add_v<ArchType::ASCEND_V200, float>(tvUbuf_tensor.ReinterpretCast<float>(),
                                                tvUbuf_tensor.ReinterpretCast<float>(),
                                                ls32Ubuf_tensor[rowsumIdx * fm * BLOCK_SIZE],
                                                fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, // repeat
                                                1,                                   // dstBlockStride
                                                1,                                   // src0BlockStride
                                                1,                                   // src1BlockStride
                                                8,                                   // dstRepeatStride
                                                8,                                   // src0RepeatStride
                                                8                                    // src1RepeatStride
            );
            PIPE_BARRIER(V);
        }
        AscendC::SetMaskNorm();
        if (__n1 % BLOCK_SIZE > 0) {
            __set_mask(__n1 % BLOCK_SIZE);
            if (__n1 / BLOCK_SIZE > 0) {
                add_v<ArchType::ASCEND_V200, float>(
                    tvUbuf_tensor.ReinterpretCast<float>(), tvUbuf_tensor.ReinterpretCast<float>(),
                    ls32Ubuf_tensor[__n1 / BLOCK_SIZE * fm * BLOCK_SIZE], fm, 1, 1, 1, 2, 2, 2);
                PIPE_BARRIER(V);
                AscendC::SetVectorMask<int8_t>(0x0, 0xffff);
            }
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, 0xffff);
        }
        cadd_v<ArchType::ASCEND_V200, float>(llUbuf_tensor[Pongflag * UB_FLOAT_LINE_SIZE],
                                             tvUbuf_tensor.ReinterpretCast<float>(), fm,
                                             1,  // dstRepeatStride
                                             1,  // srcBlockStride
                                             2); // srcRepeatStride, fp32 2 block
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE1, EVENT_ID0);
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    }
    // 4. ################ Softmax Pong Ends #######################
}

template <typename T, typename SType, PrecType prec_type1, PrecType prec_type2>
__aicore__ inline void UnpadFlashAttentionCommon<T, SType, prec_type1, prec_type2>::FlashAttentionNzPrefillCompute(
    const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn, const int32_t __m0, const int32_t __n0,
    const int32_t __n1, const int32_t pp_n_scalar, const int32_t q_tight, const int32_t add_mask_n0,
    const int32_t add_mask_n1, const int32_t long_seq, const SType alibi_coeff, const SType delta0, const SType delta1,
    const uint32_t scale_type, const uint32_t alibi_left_align)
{
    int32_t Pingflag = 0; // manual PingPong attempt
    int32_t Pongflag = 1;

    const uint32_t l1q_buf_addr_offset = 0;
    const uint32_t l1kpv_buf_addr_offset = 4 * L1_UINT8_BLOCK_SIZE;
    const uint32_t l1diag_buf_addr_offset = UB_UINT8_BLOCK_SIZE; // 32k

    AscendC::LocalTensor<half> l1qBuf_tensor =
        l1qBufAddr_tensor.ReinterpretCast<uint8_t>()[l1q_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1kPingBuf_tensor =
        l1kBufAddr_tensor.ReinterpretCast<uint8_t>()[Pingflag * l1kpv_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1kPongBuf_tensor =
        l1kBufAddr_tensor.ReinterpretCast<uint8_t>()[Pongflag * l1kpv_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1vPingBuf_tensor =
        l1vBufAddr_tensor.ReinterpretCast<uint8_t>()[Pingflag * l1kpv_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1vPongBuf_tensor =
        l1vBufAddr_tensor.ReinterpretCast<uint8_t>()[Pongflag * l1kpv_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1pPingBuf_tensor =
        l1pBufAddr_tensor.ReinterpretCast<uint8_t>()[Pingflag * l1kpv_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1pPongBuf_tensor =
        l1pBufAddr_tensor.ReinterpretCast<uint8_t>()[Pongflag * l1kpv_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1dmDiagPingBuf_tensor =
        l1diagBufAddr_tensor.ReinterpretCast<uint8_t>()[Pingflag * l1diag_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1dmDiagPongBuf_tensor =
        l1diagBufAddr_tensor.ReinterpretCast<uint8_t>()[Pongflag * l1diag_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1oTempBuf_tensor = l1oBufAddr_tensor.ReinterpretCast<half>();

    gmSrcq_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcq));
    gmSrck_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrck));
    gmSrcv_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcv));
    gmSrcm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcm));
    gmDsto_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmDsto));

    // 4 for ping-pong memory offset in L1
    __cbuf__ uint8_t *l1qBuf = l1qBufAddr;
    // 4 for ping-pong memory offset in L1
    __cbuf__ uint8_t *l1kPingBuf = l1kBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE;
    __cbuf__ uint8_t *l1kPongBuf = l1kBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE;
    // 4 for ping-pong memory offset in L1
    __cbuf__ uint8_t *l1vPingBuf = l1vBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE;
    __cbuf__ uint8_t *l1vPongBuf = l1vBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE;
    // 4 for ping-pong memory offset in L1
    __cbuf__ uint8_t *l1pPingBuf = l1pBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE;
    __cbuf__ uint8_t *l1pPongBuf = l1pBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE;

    int32_t oSize = fm * fk;
    int32_t mD64 = (fm + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
    int32_t mD128 = (fm + VECTOR_SIZE - 1) / VECTOR_SIZE;
    int32_t initGgDm = (initG == 1) ? 1 : 0; // n_idx =0->1
    int32_t initGgO = (initG == 1) ? 1 : 0;

    int32_t pSize = fm * fn;
    int32_t pSize_b = fm * bn;

    // 1. ################ Bmm1 Ping Start #######################
    // 1.1 ################ QK Ping LOAD ################
    if (initGgO != 0) {
        WAIT_FLAG(MTE1, MTE2, Pingflag);
        WAIT_FLAG(MTE1, MTE2, Pongflag);
        if (__m0 == 1) {
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1qBuf_tensor, gmSrcq_tensor[(int64_t)srcqOffset], 1, ntokensQ, 1, fk, fk, fk);
        } else if (ntokensQ <= STRIDE_UPPER_BOUND + fm) { // (fm, fk)
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1qBuf_tensor, gmSrcq_tensor[(int64_t)srcqOffset], fm, ntokensQ, fm, fk, fk, fk);
        } else {
            for (int32_t l1qBurstIdx = 0; l1qBurstIdx < (fk / BLOCK_SIZE); ++l1qBurstIdx) {
                gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                    l1qBuf_tensor[l1qBurstIdx * fm * BLOCK_SIZE],
                    gmSrcq_tensor[(int64_t)srcqOffset + l1qBurstIdx * ntokensQ * BLOCK_SIZE], fm, fm, fm, BLOCK_SIZE,
                    BLOCK_SIZE, BLOCK_SIZE);
            }
        }
        SET_FLAG(MTE2, MTE1, Pingflag);
        if (__n1 != -1) {
            SET_FLAG(MTE2, MTE1, Pongflag);
        }
    }

    //  Mask Preload L1
    if (gmSrcm != nullptr) {
        if (add_mask_n0 == 1) {
            WAIT_FLAG(MTE1, MTE2, Pingflag + 2);
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1maskBufAddr_tensor[Pingflag * L0AB_HALF_BUF_SIZE], gmSrcm_tensor[srcmOffset0], fm, maskStride, fm, fn,
                fn, fn);
            SET_FLAG(MTE2, MTE1, Pingflag + 2);
        }
        if (__n1 != -1) {
            if (add_mask_n1 == 1) {
                WAIT_FLAG(MTE1, MTE2, Pongflag + 2);
                gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                    l1maskBufAddr_tensor[Pongflag * L0AB_HALF_BUF_SIZE], gmSrcm_tensor[srcmOffset1], fm, maskStride, fm,
                    bn, bn, bn);
                SET_FLAG(MTE2, MTE1, Pongflag + 2);
            }
        }
    }

    WAIT_FLAG(M, MTE1, Pingflag);
    if (initGgO == 1) {
        WAIT_FLAG(MTE2, MTE1, Pingflag);
    }
    if (__m0 == 1) {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], l1qBuf_tensor, 0,
            1, // repeat
            0,
            1, // srcStride
            0,
            0 // dstStride
        );

    } else if (fk == BLOCK_SIZE) {
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], l1qBuf_tensor, 0,
            fm / BLOCK_SIZE, // repeat
            0,
            1, // srcStride
            0,
            0 // dstStride
        );

    } else {
        for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, fk) Nz-> zZ
            l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * fk * BLOCK_SIZE], // 512
                l1qBuf_tensor[l0aLoadIdx * CUBE_MATRIX_SIZE], // 256
                0,
                fk / BLOCK_SIZE, // repeat 2
                0,
                fm / BLOCK_SIZE, // srcStride 2
                0,
                0 // dstStride
            );
        }
    }

    SET_FLAG(MTE1, M, Pingflag);
    WAIT_FLAG(MTE1, MTE2, Pingflag + 4);

    if (kvCopyStride <= STRIDE_UPPER_BOUND + fn) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1kPingBuf_tensor, gmSrck_tensor[(int64_t)srckOffset], fn, kvCopyStride, fn, fk, fk, fk);
    } else {
        for (int32_t l1kBurstIdx = 0; l1kBurstIdx < (fk / BLOCK_SIZE); ++l1kBurstIdx) {
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1kPingBuf_tensor[l1kBurstIdx * fn * BLOCK_SIZE],
                gmSrck_tensor[(int64_t)srckOffset + l1kBurstIdx * kvCopyStride * BLOCK_SIZE], fn, fn, fn, BLOCK_SIZE,
                BLOCK_SIZE, BLOCK_SIZE);
        }
    }
    SET_FLAG(MTE2, MTE1, Pingflag);
    WAIT_FLAG(MTE2, MTE1, Pingflag);

    WAIT_FLAG(M, MTE1, Pingflag + 2);
    l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], l1kPingBuf_tensor, 0,
        fk * fn / CUBE_MATRIX_SIZE, // repeat
        0,
        1, // srcStride
        0,
        0 // dstStride
    );
    SET_FLAG(MTE1, MTE2, Pingflag + 4);
    SET_FLAG(MTE1, M, Pingflag + 2);

    WAIT_FLAG(MTE1, MTE2, Pingflag + 6);
    if (kvCopyStride <= STRIDE_UPPER_BOUND + fn) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1vPingBuf_tensor, gmSrcv_tensor[(int64_t)srcvOffset], fn, kvCopyStride, fn, fk, fk, fk);
    } else {
        for (int32_t l1vBurstIdx = 0; l1vBurstIdx < (fk / BLOCK_SIZE); ++l1vBurstIdx) {
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1vPingBuf_tensor[l1vBurstIdx * fn * BLOCK_SIZE],
                gmSrcv_tensor[(int64_t)srcvOffset + l1vBurstIdx * kvCopyStride * BLOCK_SIZE], fn, fn, fn, BLOCK_SIZE,
                BLOCK_SIZE, BLOCK_SIZE);
        }
    }
    SET_FLAG(MTE2, MTE1, Pingflag + 4);

    // bmm1 ping
    WAIT_FLAG(MTE1, M, Pingflag + 2);
    WAIT_FLAG(MTE1, M, Pingflag);
    WAIT_FLAG(V, M, Pingflag);
    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
        l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], __m0, __n0, fk, 1);

    SET_FLAG(M, MTE1, Pingflag);
    SET_FLAG(M, MTE1, Pingflag + 2);
    SET_FLAG(M, V, Pingflag);

    // 1. ################ Bmm1 Ping Ends #######################
    // 2. ################ Bmm1 Pong Starts #######################
    // 2.1 ################ QK Pong LOAD ################
    if (__n1 != -1) {
        WAIT_FLAG(M, MTE1, Pongflag);
        if (initGgO == 1) {
            WAIT_FLAG(MTE2, MTE1, Pongflag);
        }
        if (__m0 == 1) {
            l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], l1qBuf_tensor, 0,
                1, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else if (fk == BLOCK_SIZE) {
            l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], l1qBuf_tensor, 0,
                fm / BLOCK_SIZE, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else {
            for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, fk) Nz-> zZ
                l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * fk * BLOCK_SIZE],
                    l1qBuf_tensor[l0aLoadIdx * CUBE_MATRIX_SIZE], 0,
                    fk / BLOCK_SIZE, // repeat
                    0,
                    fm / BLOCK_SIZE, // srcStride
                    0,
                    0 // dstStride
                );
            }
        }
        SET_FLAG(MTE1, M, Pongflag);
        WAIT_FLAG(MTE1, MTE2, Pongflag + 4);
        if (kvCopyStride <= STRIDE_UPPER_BOUND + bn) {
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1kPongBuf_tensor, gmSrck_tensor[(int64_t)srckOffset + Pongflag * pp_n_scalar * BLOCK_SIZE], bn,
                kvCopyStride, bn, fk, fk, fk);
        } else {
            for (int32_t l1kBurstIdx = 0; l1kBurstIdx < (fk / BLOCK_SIZE); ++l1kBurstIdx) {
                gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                    l1kPongBuf_tensor[l1kBurstIdx * bn * BLOCK_SIZE],
                    gmSrck_tensor[(int64_t)srckOffset + Pongflag * pp_n_scalar * BLOCK_SIZE +
                                  l1kBurstIdx * kvCopyStride * BLOCK_SIZE],
                    bn, bn, bn, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
            }
        }
        SET_FLAG(MTE2, MTE1, Pongflag);
        WAIT_FLAG(MTE2, MTE1, Pongflag);
        WAIT_FLAG(M, MTE1, Pongflag + 2);
        l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], l1kPongBuf_tensor, 0,
            fk * bn / CUBE_MATRIX_SIZE, // repeat
            0,
            1, // srcStride
            0,
            0 // dstStride
        );
        SET_FLAG(MTE1, MTE2, Pongflag + 4);
        SET_FLAG(MTE1, M, Pongflag + 2);

        WAIT_FLAG(MTE1, MTE2, Pongflag + 6);
        if (kvCopyStride <= STRIDE_UPPER_BOUND + bn) {
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1vPongBuf_tensor, gmSrcv_tensor[(int64_t)srcvOffset + Pongflag * pp_n_scalar * BLOCK_SIZE], bn,
                kvCopyStride, bn, fk, fk, fk);
        } else {
            for (int32_t l1vBurstIdx = 0; l1vBurstIdx < (fk / BLOCK_SIZE); ++l1vBurstIdx) {
                gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                    l1vPongBuf_tensor[l1vBurstIdx * bn * BLOCK_SIZE],
                    gmSrcv_tensor[(int64_t)srcvOffset + Pongflag * pp_n_scalar * BLOCK_SIZE +
                                  l1vBurstIdx * kvCopyStride * BLOCK_SIZE],
                    bn, bn, bn, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
            }
        }
        SET_FLAG(MTE2, MTE1, Pongflag + 4);
        WAIT_FLAG(MTE1, M, Pongflag + 2);
        WAIT_FLAG(MTE1, M, Pongflag);
        WAIT_FLAG(V, M, Pongflag);
        mmad<ArchType::ASCEND_V200, half, half, float, false>(
            l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
            l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], __m0, __n1, fk, 1);
        SET_FLAG(M, MTE1, Pongflag);
        SET_FLAG(M, V, Pongflag);
        SET_FLAG(M, MTE1, Pongflag + 2);
    }
    // 2. ################ Bmm1 Pong Ends #######################
    SoftMax(fm, fn, fk, bn, __m0, __n0, __n1, add_mask_n0, add_mask_n1,
            alibi_coeff, delta0, delta1, scale_type, alibi_left_align, initGgDm
            );
    if (cubeUpdateO == 1) {
        initGgO = 0;
    }

    // 5. ################ Bmm2 Ping Starts #######################
    if (cubeUpdateO == 0) {
        WAIT_FLAG(MTE2, MTE1, Pingflag + 4);
        WAIT_FLAG(M, MTE1, Pingflag + 2);
        // 16 is blocksize in format zN
        if (fk == 16) {
            l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], l1vPingBuf_tensor, 0,
                fn / BLOCK_SIZE, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else {
            for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (fn / BLOCK_SIZE); ++l0bLoadIdx) { // Nz -> nZ
                l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + l0bLoadIdx * fk * BLOCK_SIZE],
                    l1vPingBuf_tensor[l0bLoadIdx * CUBE_MATRIX_SIZE], 0,
                    fk / BLOCK_SIZE, // repeat
                    0,
                    fn / BLOCK_SIZE, // srcStride
                    0,
                    0 // dstStride
                );
            }
        }
        SET_FLAG(MTE1, M, Pingflag + 2);
        SET_FLAG(MTE1, MTE2, Pingflag + 6);
        // BMM2 P LOAD Ping
        WAIT_FLAG(V, MTE3, Pingflag);
        WAIT_FLAG(MTE1, MTE3, Pingflag);
        if (__m0 == 1) {
            ub_to_l1<ArchType::ASCEND_V200, half>(l1pPingBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                                lpUbuf_tensor[Pingflag * lpUbufSize], fn / BLOCK_SIZE, 1, fm - 1,
                                                0);
        } else {
            ub_to_l1<ArchType::ASCEND_V200, half>(l1pPingBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
                                                lpUbuf_tensor[Pingflag * lpUbufSize], 1, pSize / BLOCK_SIZE, 0,
                                                0);
        }
        SET_FLAG(MTE3, V, Pingflag);
        SET_FLAG(MTE3, MTE1, Pingflag);
        WAIT_FLAG(MTE3, MTE1, Pingflag);
        WAIT_FLAG(M, MTE1, Pingflag);
        // 16 is blocksize in format zN
        if (__m0 == 1) {
            l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], l1pPingBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], 0,
                1, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else if (fn == BLOCK_SIZE) {
            l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], l1pPingBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], 0,
                fm / BLOCK_SIZE, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else {
            for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, fn)
                l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * fn * BLOCK_SIZE],
                    l1pPingBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * CUBE_MATRIX_SIZE], 0,
                    fn / BLOCK_SIZE, // repeat
                    0,
                    fm / BLOCK_SIZE, // srcStride
                    0,
                    0 // dstStride
                );
            }
        }
        SET_FLAG(MTE1, M, Pingflag);
        SET_FLAG(MTE1, MTE3, Pingflag);
        WAIT_FLAG(MTE1, M, Pingflag);
        // 4. bmm2 partr
        WAIT_FLAG(MTE1, M, Pingflag + 2);
        WAIT_FLAG(V, M, Pingflag + 2);
        mmad<ArchType::ASCEND_V200, __fp16, __fp16, float, false>(
            l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE],
            l0aBuf_tensor.ReinterpretCast<__fp16>()[Pingflag * L0AB_HALF_BUF_SIZE],
            l0bBuf_tensor.ReinterpretCast<__fp16>()[Pingflag * L0AB_HALF_BUF_SIZE], __m0, fk, __n0, 1);
        SET_FLAG(M, V, Pingflag);
        SET_FLAG(M, MTE1, Pingflag);
        SET_FLAG(M, MTE1, Pingflag + 2);
        if (wrapO == 1) {
            SET_FLAG(MTE1, MTE2, Pingflag);
            if (__n1 == -1) {
                SET_FLAG(MTE1, MTE2, Pongflag);
            }
        }
    }
    // 5. ################ Bmm2 Ping Ends #######################
    // 6. ################ Bmm2 Pong Starts #######################
    if (__n1 != -1) {
        WAIT_FLAG(MTE2, MTE1, Pongflag + 4);
        WAIT_FLAG(M, MTE1, Pongflag + 2);
        // 16 is blocksize in format zN
        if (fk == 16) {
            l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], l1vPongBuf_tensor, 0,
                bn / BLOCK_SIZE, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else {
            for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (bn / BLOCK_SIZE); ++l0bLoadIdx) { // Nz -> nZ
                l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + l0bLoadIdx * fk * BLOCK_SIZE],
                    l1vPongBuf_tensor[l0bLoadIdx * CUBE_MATRIX_SIZE], 0,
                    fk / BLOCK_SIZE, // repeat
                    0,
                    bn / BLOCK_SIZE, // srcStride
                    0,
                    0 // dstStride
                );
            }
        }
        SET_FLAG(MTE1, MTE2, Pongflag + 6);
        SET_FLAG(MTE1, M, Pongflag + 2);
        // BMM2 P LOAD Pong
        WAIT_FLAG(MTE1, MTE3, Pongflag);
        WAIT_FLAG(V, MTE3, Pongflag);
        if (__m0 == 1) {
            ub_to_l1<ArchType::ASCEND_V200, half>(l1pPongBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                                  lpUbuf_tensor[Pongflag * lpUbufSize], bn / BLOCK_SIZE, 1,
                                                  fm - 1, 0);
        } else {
            ub_to_l1<ArchType::ASCEND_V200, half>(l1pPongBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                                                  lpUbuf_tensor[Pongflag * lpUbufSize], 1, pSize_b / BLOCK_SIZE,
                                                  0, 0);
        }
        SET_FLAG(MTE3, V, Pongflag);
        SET_FLAG(MTE3, MTE1, Pongflag);
        WAIT_FLAG(MTE3, MTE1, Pongflag);
        WAIT_FLAG(M, MTE1, Pongflag);

        // 16 is blocksize in format zN
        if (__m0 == 1) {
            l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], l1pPongBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], 0,
                1, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else if (bn == BLOCK_SIZE) {
            l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], l1pPongBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], 0,
                fm / BLOCK_SIZE, // repeat
                0,
                1, // srcStride
                0,
                0 // dstStride
            );
        } else {
            for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, bn)
                l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * bn * BLOCK_SIZE],
                    l1pPongBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * CUBE_MATRIX_SIZE], 0,
                    bn / BLOCK_SIZE, // repeat
                    0,
                    fm / BLOCK_SIZE, // srcStride
                    0,
                    0 // dstStride
                );
            }
        }
        SET_FLAG(MTE1, M, Pongflag);
        SET_FLAG(MTE1, MTE3, Pongflag);
        WAIT_FLAG(MTE1, M, Pongflag);
        // 4. bmm2 part
        WAIT_FLAG(MTE1, M, Pongflag + 2);
        WAIT_FLAG(V, M, Pongflag + 2);
        mmad<ArchType::ASCEND_V200, __fp16, __fp16, float, false>(
            l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
            l0aBuf_tensor.ReinterpretCast<__fp16>()[Pongflag * L0AB_HALF_BUF_SIZE],
            l0bBuf_tensor.ReinterpretCast<__fp16>()[Pongflag * L0AB_HALF_BUF_SIZE], __m0, fk, __n1, 1);

        SET_FLAG(M, MTE1, Pongflag);
        SET_FLAG(M, MTE1, Pongflag + 2);

        // online softmax
        if (cubeUpdateO == 1 && initGgO == 0) {
            WAIT_FLAG(MTE3, MTE1, Pongflag + 2);
            WAIT_FLAG(M, MTE1, Pongflag);
            if (__m0 == 1) {
                l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                    l1dmDiagPongBuf_tensor,
                    0,
                    1, // repeat
                    0,
                    1, // srcStride
                    0,
                    0 // dstStride
                );

            } else {
                for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, fk) Nz-> zZ
                    l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                        l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * fm * BLOCK_SIZE],
                        l1dmDiagPongBuf_tensor[l0aLoadIdx * CUBE_MATRIX_SIZE],
                        0,
                        fm / BLOCK_SIZE, // repeat
                        0,
                        fm / BLOCK_SIZE, // srcStride
                        0,
                        0 // dstStride
                    );
                }
            }

            WAIT_FLAG(MTE3, MTE1, EVENT_ID0);
            WAIT_FLAG(M, MTE1, Pongflag + 2);
            if (fk == BLOCK_SIZE) {
                l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                    l1oTempBuf_tensor,
                    0,
                    fm / BLOCK_SIZE, // repeat
                    0,
                    1, // srcStride
                    0,
                    0 // dstStride
                );
            } else {
                for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (fm / BLOCK_SIZE); ++l0bLoadIdx) { // Nz -> nZ
                    l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                        l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE + l0bLoadIdx * fk * BLOCK_SIZE],
                        l1oTempBuf_tensor[l0bLoadIdx * CUBE_MATRIX_SIZE],
                        0,
                        fk / BLOCK_SIZE, // repeat
                        0,
                        fm / BLOCK_SIZE, // srcStride
                        0,
                        0 // dstStride
                    );
                }
            }

            SET_FLAG(MTE1, M, Pongflag);
            WAIT_FLAG(MTE1, M, Pongflag);

            mmad<ArchType::ASCEND_V200, __fp16, __fp16, float, false>(
                l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE],
                l0aBuf_tensor.ReinterpretCast<__fp16>()[Pongflag * L0AB_HALF_BUF_SIZE],
                l0bBuf_tensor.ReinterpretCast<__fp16>()[Pongflag * L0AB_HALF_BUF_SIZE],
                __m0,
                fk,
                __m0,
                0
            );

            SET_FLAG(M, MTE1, Pongflag);
            SET_FLAG(M, MTE1, Pongflag + 2);
        }
        if (cubeUpdateO == 1) {
            initGgO = 0;
        }

        SET_FLAG(M, V, Pongflag);
        if (wrapO == 1) {
            SET_FLAG(MTE1, MTE2, Pongflag);
        }
    }
    // 6. ################ Bmm2 Pong Ends #######################
    // 7. ################ Update Ping Starts #######################
    WAIT_FLAG(V, MTE1, EVENT_ID0);
    if (cubeUpdateO == 0) {
        WAIT_FLAG(M, V, Pingflag);
        l0c_to_ub<ArchType::ASCEND_V200, float, T>(loUbuf_tensor.ReinterpretCast<T>(),
                                                   l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], 1,
                                                   oSize / CUBE_MATRIX_SIZE, 0, 0);
    } else if (__n1 == -1) {
        WAIT_FLAG(M, V, Pingflag);
        l0c_to_ub<ArchType::ASCEND_V200, float, float>(goUbuf_tensor,
                                                       l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE], 1,
                                                       oSize / CUBE_MATRIX_SIZE, 0, 0);
    }

    PIPE_BARRIER(V);
    SoftmaxUpdate(fm, fk, oSize, Pingflag, initGgO, mD64);
    initGgO = 0;
    // 7. ################ Update Ping Ends #######################
    // 8. ################ Update Pong Starts #######################
    if (__n1 != -1) {
        WAIT_FLAG(V, MTE1, EVENT_ID0);
        WAIT_FLAG(M, V, Pongflag);

        if (cubeUpdateO == 0) {
            l0c_to_ub<ArchType::ASCEND_V200, float, T>(loUbuf_tensor.ReinterpretCast<T>(), l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], 1,
                                                       oSize / CUBE_MATRIX_SIZE, 0, 0);

        } else if (wrapO == 0) {
            l0c_to_ub<ArchType::ASCEND_V200, float, half>(toUbuf_tensor, l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], 1,
                                                          oSize / CUBE_MATRIX_SIZE, 0, 0);

            SET_FLAG(V, MTE3, Pongflag);
            WAIT_FLAG(V, MTE3, Pongflag);

            ub_to_l1<ArchType::ASCEND_V200, half>(l1oTempBuf_tensor,
                                                  toUbuf_tensor,
                                                  1,
                                                  oSize / BLOCK_SIZE,
                                                  0,
                                                  0);

        } else {
            l0c_to_ub<ArchType::ASCEND_V200, float, float>(goUbuf_tensor, l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE], 1,
                                                           oSize / CUBE_MATRIX_SIZE, 0, 0);
        }
        PIPE_BARRIER(V);
        // 5. update for outer loop
        SoftmaxUpdate(fm, fk, oSize, Pongflag, initGgO, mD64);
        SET_FLAG(V, M, Pongflag + 2);
        PIPE_BARRIER(V);
        initGgO = 0;
    }
    SET_FLAG(V, M, Pingflag + 2);
    // 8. ################ Update Pong Ends #######################
    // 9. ################ Line Output Starts #####################
    UpdateOutput(fm, fk, oSize, mD64, __m0);
    // 9. ################ Line Output Ends #####################
}

template <typename T, typename SType, PrecType prec_type1, PrecType prec_type2>
__aicore__ inline void UnpadFlashAttentionCommon<T, SType, prec_type1, prec_type2>::InitBatchParam(const PromptFlashAttentionBaseApiTilingData *__restrict tilingData,
    int32_t heads, uint32_t max_seqlen, uint32_t max_kv_seqlen,int32_t embd, uint32_t kvHead, uint32_t embeddingSizeV, uint32_t inputLayout, uint32_t q_tight){
    const int32_t PP_BLOCK_BUFFER_SIZE = 128 * 128;
    const int32_t PP_MM_NUM = 8;
    const int32_t PP_NN_NUM = 16;
    const int32_t PP_INDEX = 16;
    constexpr int32_t PP_MM[] = {16, 32, 48, 64, 80, 96, 112, 128};
    constexpr int32_t PP_NN[] = {16,  32,  48,  64,  80,  96,  112, 128,
                                144, 160, 176, 192, 208, 224, 240, 256};

    q_seqlen_aligned = (q_seqlen_real + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    kv_seqlen_aligned = (kv_seqlen_real + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int32_t embeddingSizeAligned = (heads + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int32_t tilingK = embeddingSizeAligned < BLOCK_LIMIT ? BLOCK_LIMIT : embeddingSizeAligned;
    int32_t nUbd = GetMin((PP_BLOCK_BUFFER_SIZE / tilingK / BLOCK_SIZE) * BLOCK_SIZE, kv_seqlen_aligned); // hidden_size = 40 ➡️ n_blk=256 n_blk = 64
    int32_t nIbd = (nUbd > PP_NN[PP_NN_NUM - 1]) ? (PP_NN_NUM - 1) : (nUbd / PP_INDEX - 1);

    int32_t embeddingSize = heads;
    int32_t mUbd = GetMin((PP_BLOCK_BUFFER_SIZE / GetMax(embeddingSize, PP_NN[nIbd]) / BLOCK_SIZE) * BLOCK_SIZE,  q_seqlen_aligned);
    mUbd = mUbd > PP_MM[3] && INNER_PRECISE_PTR ? PP_MM[3] : mUbd;

    // PP_MM 是 16 对齐的数组
    int32_t mIbd = (mUbd > PP_MM[PP_MM_NUM - 1]) ? (PP_MM_NUM - 1) : (mUbd / PP_INDEX - 1);

    // Batch 的 Q 的切块数目
    int32_t curQBlockNum = ((q_seqlen_real + PP_MM[mIbd] - 1) / PP_MM[mIbd]); // 对齐处理
    // 整个任务有多少个 Q 切块
    totalQBlkNum += curQBlockNum;

    pp_m_scalar = PP_MM[mIbd];
    pp_n_scalar = PP_MM[nIbd];

    addr_q_scalar = addrQSeqOffset;
    addr_k_scalar = addrKSeqOffset;
    addr_v_scalar = addrVSeqOffset;
    addr_o_scalar = addrOSeqOffset;

    cur_total_qblk = heads * totalQBlkNum;
    cur_proc_num = heads * curQBlockNum;

    auto kvFactor = kv_seqlen_real;

    if (inputLayout == 3) {
        addrQSeqOffset += static_cast<uint64_t>(max_seqlen * heads * embd);
        addrKSeqOffset += static_cast<uint64_t>(max_kv_seqlen * kvHead * embeddingSizeV);
        addrVSeqOffset += static_cast<uint64_t>(max_kv_seqlen * kvHead * embeddingSizeV);
        addrOSeqOffset += static_cast<uint64_t>(max_seqlen * heads * embeddingSizeV);
    }
    else if (inputLayout == 1 || inputLayout == 2) {
        addrQSeqOffset += q_tight != 0 ? static_cast<uint64_t>(q_seqlen_real * BLOCK_SIZE)
                                                         : static_cast<uint64_t>(q_seqlen_aligned * BLOCK_SIZE);
        addrKSeqOffset += static_cast<uint64_t>(max_seqlen * kvHead * embeddingSizeV);
        addrVSeqOffset += static_cast<uint64_t>(max_seqlen * kvHead * embeddingSizeV);
        addrOSeqOffset += q_tight != 0 ? static_cast<uint64_t>(q_seqlen_real * BLOCK_SIZE)
                                                         : static_cast<uint64_t>(q_seqlen_aligned * BLOCK_SIZE);
    }
}

template <>
__aicore__ inline void UnpadFlashAttentionCommon<float, half, PrecType::BMM1_FP16_EXP_FP32, PrecType::BMM1_FP16_EXP_FP32>
                    ::Run(const PromptFlashAttentionBaseApiTilingData *__restrict tilingData,
                           __gm__ uint8_t *__restrict__ alibi_coeff_gm,AscendC::GlobalTensor<int64_t> actualSeqLengthsGm,AscendC::GlobalTensor<int64_t> actualSeqLengthsKVGm,
                           uint32_t mask_type, uint32_t window_len, uint32_t long_seq,
                           uint64_t stride_qo, uint64_t stride_kv, int64_t head_mask_stride,
                           int64_t batch_mask_stride,
                           uint32_t start_batch, uint32_t end_batch,
                           int32_t start_blk, int32_t end_blk,
                           uint32_t is_triu, uint32_t alibi_compress_offset, int32_t group_num,
                           uint32_t mask_stride, uint32_t q_tokens, int32_t embd,
                           uint32_t q_tight, uint32_t scaleType,
                           half tor, int32_t kv_copy_stride, uint32_t is_sqrt,
                           int64_t heads, uint32_t max_seqlen, uint32_t batch_size, int32_t kv_real_heads, const uint32_t alibi_left_align, uint32_t inputLayout)
{
    if (gmSrcLayerid != nullptr) {
        stride_kv = max_seqlen * embd;
        kv_copy_stride = max_seqlen;
        uint64_t stride_batch_kv = batch_size * max_seqlen * kv_real_heads * embd * 2;
    }
    SetEncoderParams(tor, kv_copy_stride, is_sqrt, 0);
    SET_FLAG(S, MTE2, EVENT_ID0);
    WAIT_FLAG(S, MTE2, EVENT_ID0);

    SET_FLAG(MTE2, S, EVENT_ID0);
    WAIT_FLAG(MTE2, S, EVENT_ID0);
    SyncStart();

    int32_t cur_batch = 0;
    int32_t pre_batch = -1;
    int64_t cur_bms = batch_mask_stride * start_batch;

    uint32_t max_kv_seqlen = tilingData->promptAttentionBaseApiBaseParams.maxSeqLen;
    uint32_t kvHead = tilingData->promptAttentionBaseApiBaseParams.kvHeadNumSize;
    uint32_t embeddingSizeV = tilingData->promptAttentionBaseApiBaseParams.embeddingSizeV;
    for (uint32_t curr_q_blk = start_blk; curr_q_blk < end_blk; curr_q_blk++) {
        q_seqlen_real = actualSeqLengthsGm.GetValue(cur_batch);
        kv_seqlen_real = actualSeqLengthsKVGm.GetValue(cur_batch);

        if (cur_batch > pre_batch){
            InitBatchParam(tilingData, heads, max_seqlen,
            max_kv_seqlen, embd, kvHead, embeddingSizeV, inputLayout, q_tight);
            pre_batch = cur_batch;
        }
        uint64_t cur_q_blk_id = curr_q_blk - (cur_total_qblk - cur_proc_num);

        // SWA calc mode condition
        uint32_t swa_mode = ((mask_type == AttentonMaskType::MASK_TYPE_SWA_NORM
                            || mask_type == AttentonMaskType::MASK_TYPE_SWA_COMPRESS)
                            && kv_seqlen_real > window_len) ? 1 : 0;
        is_triu = ((mask_type == AttentonMaskType::MASK_TYPE_SWA_NORM
                    || mask_type == AttentonMaskType::MASK_TYPE_SWA_COMPRESS)
                    && kv_seqlen_real <= window_len) ? 1 : is_triu;    // Regular Triu Mask

        int32_t m_loop = (q_seqlen_aligned + pp_m_scalar - 1) / pp_m_scalar;
        int32_t n_loop = 0;
        if (swa_mode) {
            n_loop = kv_seqlen_aligned > window_len + pp_n_scalar
                    ? ((window_len + pp_n_scalar - 1) / pp_n_scalar + 1)
                    : ((kv_seqlen_aligned + pp_n_scalar - 1) / pp_n_scalar);
        } else {
            n_loop = (kv_seqlen_aligned + pp_n_scalar - 1) / pp_n_scalar;
        }
        int32_t start = cur_q_blk_id * n_loop;
        int32_t end = start + n_loop;

        for (int32_t loop_idx = start; loop_idx < end; loop_idx += 2) {
            int32_t head_idx0 = loop_idx / (m_loop * n_loop);
            int32_t m_idx0 = loop_idx % (m_loop * n_loop) / n_loop;
            int32_t n_idx0 = loop_idx % (m_loop * n_loop) % n_loop;
            int32_t window_offset = 0; // window_offset will only be overwritten in swa mode.
            if (swa_mode) {
                window_offset = (m_idx0 + 1 > n_loop) ? (m_idx0 - n_loop + 1) : 0;
                is_triu = (window_offset == 0) ? 1 : 0;
            }
            if (is_triu == 1 && n_idx0 > m_idx0) {
                continue;
            }
            int32_t add_mask_n0 = ((long_seq == 0) || ((long_seq == 1) && (n_idx0 == m_idx0)) ||
                                   alibi_coeff_gm != nullptr ||
                                   (mask_type == AttentonMaskType::MASK_TYPE_ALIBI && alibi_compress_offset > 0))
                                      ? 1
                                      : 0;
            int32_t add_mask_n1 = ((long_seq == 0) || ((long_seq == 1) && (n_idx0 + 1 == m_idx0)) ||
                                   alibi_coeff_gm != nullptr ||
                                   (mask_type == AttentonMaskType::MASK_TYPE_ALIBI && alibi_compress_offset > 0))
                                      ? 1
                                      : 0;
                                      
            int64_t q_offset = addr_q_scalar + head_idx0 * stride_qo + m_idx0 * pp_m_scalar * BLOCK_SIZE;
            int64_t k_offset = addr_k_scalar + head_idx0 / group_num * stride_kv + (n_idx0 + window_offset) * pp_n_scalar * BLOCK_SIZE;
            int64_t v_offset = addr_v_scalar + head_idx0 / group_num * stride_kv + (n_idx0 + window_offset) * pp_n_scalar * BLOCK_SIZE;
            int64_t o_offset = addr_o_scalar + head_idx0 * stride_qo + m_idx0 * pp_m_scalar * BLOCK_SIZE;
            int64_t logn_offset = m_idx0 * pp_m_scalar;

            int64_t mask_offset0 = cur_bms + head_mask_stride * head_idx0;
            int64_t mask_offset1 = cur_bms + head_mask_stride * head_idx0;
            int32_t delta_uint = 0;
            int32_t delta0 = 0;
            int32_t delta1 = 0;
            half alibi_coeff = 1;
            if (alibi_coeff_gm != nullptr) {
                AsdopsBuffer<ArchType::ASCEND_V200> buf;
                AscendC::LocalTensor<half> alibi_coeff_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>( 5 * UB_UINT8_BLOCK_SIZE + 28 * UB_UINT8_LINE_SIZE);
                AscendC::GlobalTensor<half> alibi_coeff_gm_tensor;
                alibi_coeff_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(alibi_coeff_gm));
                gm_to_ub<ArchType::ASCEND_V200, half>(alibi_coeff_ub_tensor,
                                                    alibi_coeff_gm_tensor, 0, 1, (heads + 15) / 16, 0, 0);
                SET_FLAG(MTE2, S, EVENT_ID0);
                WAIT_FLAG(MTE2, S, EVENT_ID0);
                alibi_coeff = *(__ubuf__ half *)(alibi_coeff_ub_tensor[head_idx0].GetPhyAddr());
                if (m_idx0 == n_idx0) {
                    mask_offset0 = 0;
                } else {
                    mask_offset0 = BASE_MASK_SIZE * BLOCK_SIZE;
                    delta_uint = m_idx0 * pp_m_scalar - n_idx0 * pp_n_scalar;
                    delta0 = delta_uint - BASE_MASK_SIZE;
                }

                if (m_idx0 == n_idx0 + 1) {
                    mask_offset1 = 0;
                } else {
                    mask_offset1 = BASE_MASK_SIZE * BLOCK_SIZE;
                    delta_uint = m_idx0 * pp_m_scalar - (n_idx0 + 1) * pp_n_scalar;
                    delta1 = delta_uint - BASE_MASK_SIZE;
                }
            } else if (mask_type == AttentonMaskType::MASK_TYPE_ALIBI && alibi_compress_offset > 0) {
                if (m_idx0 != n_idx0) {
                    mask_offset0 += (m_idx0 * pp_m_scalar - n_idx0 * pp_n_scalar) * BLOCK_SIZE;
                }

                if (m_idx0 != n_idx0 + 1) {
                    mask_offset1 += (m_idx0 * pp_m_scalar - (n_idx0 + 1) * pp_n_scalar) * BLOCK_SIZE;
                }
            } else if (mask_type == AttentonMaskType::MASK_TYPE_SWA_COMPRESS) { //SWA Compress mode - mask offset
                int32_t window_n_scalar = (window_len > 2 * pp_n_scalar) ? window_len / pp_n_scalar : 2;
                if (n_idx0 == m_idx0 - window_offset) {
                    mask_offset1 += mask_stride * pp_n_scalar;
                } else if ((n_idx0 < m_idx0 - window_offset) && ((m_idx0 - n_idx0) < (window_offset + window_n_scalar))) {
                    mask_offset0 += pp_m_scalar * BLOCK_SIZE;
                    add_mask_n0 = (window_len >= 2 * pp_n_scalar) ? 0 : 1;
                    if (n_idx0 + 1 < m_idx0 - window_offset){
                        mask_offset1 += pp_m_scalar * BLOCK_SIZE;
                        add_mask_n1 = (window_len >= 2 * pp_n_scalar) ? 0 : 1;
                    }
                } else if (n_idx0 == (m_idx0 - window_offset - window_n_scalar)) {
                    mask_offset0 += 2 * pp_m_scalar * BLOCK_SIZE;
                    mask_offset1 += pp_m_scalar * BLOCK_SIZE;
                    add_mask_n1 = (window_len >= 2 * pp_n_scalar) ? 0 : 1;
                } else if (n_idx0 == (m_idx0 - window_offset - window_n_scalar - 1)) {
                    mask_offset0 += 3 * pp_m_scalar * BLOCK_SIZE;
                    mask_offset1 += 2 * pp_m_scalar * BLOCK_SIZE;
                } else {
                    mask_offset0 += mask_stride * pp_n_scalar;
                    mask_offset1 += mask_stride * pp_n_scalar;
                }
            } else if (long_seq == 0) {
                mask_offset0 += (m_idx0 * pp_m_scalar * BLOCK_SIZE + (n_idx0 + window_offset) * mask_stride * pp_n_scalar);
                mask_offset1 += (m_idx0 * pp_m_scalar * BLOCK_SIZE + (n_idx0 + window_offset + 1) * mask_stride * pp_n_scalar);
            }
            int32_t wrap_o = (n_idx0 == (n_loop - 1) || (n_idx0 + 1) == (n_loop - 1)) ? 1 : 0;
            if (is_triu == 1) {
                wrap_o = (n_idx0 == m_idx0 || (n_idx0 + 1) == m_idx0) ? 1 : 0;
            }
            if (swa_mode) {
                if (window_offset == 0) {
                    wrap_o = (n_idx0 == m_idx0 || (n_idx0 + 1) == m_idx0) ? 1 : 0;
                } else {
                    wrap_o = (n_idx0 == (n_loop - 1) || (n_idx0 + 1) == (n_loop - 1)) ? 1 : 0;
                }
            }
            int32_t init_g = (n_idx0 == 0) ? 1 : 0;
            int32_t __m0 = (m_idx0 == (m_loop - 1)) ? (q_seqlen_real - m_idx0 * pp_m_scalar) : pp_m_scalar;
            int32_t __n0 = 0;
            int32_t __n1 = 0;
            if (swa_mode) {
                __n0 = ((n_idx0 + window_offset + 1) * pp_n_scalar <= kv_seqlen_real)
                        ? pp_n_scalar
                        : (kv_seqlen_real - (n_idx0 + window_offset) * pp_n_scalar);
                __n1 = (((n_idx0 + 1) + window_offset + 1) * pp_n_scalar <= kv_seqlen_real)
                        ? pp_n_scalar
                        : (kv_seqlen_real - ((n_idx0 + 1) + window_offset) * pp_n_scalar);
            } else {
                __n0 = (n_idx0 == (n_loop - 1)) ? (kv_seqlen_real - n_idx0 * pp_n_scalar) : pp_n_scalar;
                __n1 = ((n_idx0 + 1) == (n_loop - 1)) ? (kv_seqlen_real - (n_idx0 + 1) * pp_n_scalar) : pp_n_scalar;
            }
            int32_t __k0 = embd;
            int32_t round_m0 = (__m0 + 15) / 16 * 16;
            int32_t round_n0 = (__n0 + 15) / 16 * 16;
            int32_t round_k0 = (__k0 + 15) / 16 * 16;
            int32_t round_n1 = (__n1 + 15) / 16 * 16;

            if ((n_idx0 + 1) == (n_loop) || (n_idx0 == m_idx0 && is_triu == 1)) {
                __n1 = -1;
            }

            Init(round_m0, round_n0, round_k0, q_offset, k_offset, v_offset, mask_offset0,
                                       mask_offset1, o_offset, init_g, wrap_o, q_tokens, mask_stride, logn_offset);
            FlashAttentionNzPrefillCompute(round_m0, round_n0, round_k0, round_n1, __m0, __n0,
                                                                 __n1, pp_n_scalar, q_tight, add_mask_n0, add_mask_n1,
                                                                 long_seq, alibi_coeff, delta0, delta1, scaleType, alibi_left_align);
        }
        if (cur_q_blk_id == cur_proc_num - 1) {
            cur_batch++;
            cur_bms += batch_mask_stride;
        }
    }
    SyncEnd();
}
#endif