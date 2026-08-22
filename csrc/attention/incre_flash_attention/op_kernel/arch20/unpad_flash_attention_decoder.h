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
 * \file unpad_flash_attention_decoder.h
 * \brief
 */

#ifndef UNPAD_FLASH_ATTENTION_DECODER_H
#define UNPAD_FLASH_ATTENTION_DECODER_H

#include "common_func.h"
#include "simd.h"
#include "iterator.h"
#include "common.h"
#include "mma.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

extern constexpr int32_t LOCAL_STORAGE_BUFFER_SIZE = 4096;
extern constexpr int32_t FLOAT_VECTOR_SIZE_T = 64;
extern constexpr int32_t VECTOR_SIZE_T = 128;
extern constexpr int32_t BLOCK_SIZE_T = 16;
extern constexpr int32_t L0AB_HALF_BUF_SIZE_T = 16384;
extern constexpr int32_t UB_FLOAT_BUF_SIZE = 8192;
extern constexpr int32_t CUBE_MATRIX_SIZE_T = 256;
extern constexpr int64_t L1_UINT8_BLOCK_SIZE_T = 131072;
extern constexpr int64_t UB_UINT8_BLOCK_SIZE_T = 32768;
extern constexpr int32_t DEC_UB_UINT8_BLOCK_SIZE_T = 8192;
extern constexpr int64_t UB_UINT8_LINE_SIZE_T = 1024;
extern constexpr int64_t UB_FLOAT_LINE_SIZE_T = 256;
extern constexpr int64_t UB_HALF_LINE_SIZE_T = 512;

extern namespace Enums{
    enum AttentonMaskType {
        MASK_TYPE_NONE = 0,
        MASK_TYPE_NORM = 1,
        MASK_TYPE_SWA_NORM = 2,
        MASK_TYPE_SWA_COMPRESS = 3
    };
    
    enum PrecType { 
        BMM1_FP16_EXP_FP32 = 0, 
        BMM1_FP32_EXP_FP32 = 1, 
        BMM2_ONLINE_SOFTMAX_FP16 = 4
    };
}


template <typename T, typename SType, Enums::PrecType prec_type1>
__aicore__ inline void SoftmaxExp(AscendC::LocalTensor<half> dst, AscendC::LocalTensor<SType> src,
                                  AscendC::LocalTensor<T> tmp, uint32_t pSize);

template <>
__aicore__ inline void
SoftmaxExp<float, float, Enums::PrecType::BMM1_FP32_EXP_FP32>(AscendC::LocalTensor<half> dst, AscendC::LocalTensor<float> src,
                                                       AscendC::LocalTensor<float> tmp, uint32_t pSize)
{
    ub_to_ub<ArchType::ASCEND_V200, float>(tmp, src, 0, 1, pSize / 8, 0, 0);
    PIPE_BARRIER(V);
    // 2 for Repeatimes above 255
    for (int32_t vexpIdx = 0; vexpIdx < 2; ++vexpIdx) {
        exp_v<ArchType::ASCEND_V200, float>(tmp[vexpIdx * pSize / 2], tmp[vexpIdx * pSize / 2],
                                            pSize / 2 / FLOAT_VECTOR_SIZE_T, // repeat
                                            1,                             // dstBlockStride
                                            1,                             // srcBlockStride
                                            8,                             // dstRepeatStride
                                            8                              // srcRepeatStride
        );
    }
    PIPE_BARRIER(V);
    // 2 for Repeatimes above 255
    for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
        conv_v<ArchType::ASCEND_V200, float, half>(dst[vconvIdx * pSize / 2], tmp[vconvIdx * pSize / 2],
                                                   pSize / 2 / FLOAT_VECTOR_SIZE_T, // repeat
                                                   1,                             // dstBlockStride
                                                   1,                             // srcBlockStride
                                                   uint16_t(4),                   // dstRepeatStride
                                                   uint16_t(8)                    // srcRepeatStride
        );
    }
}

template <typename T, Enums::PrecType prec_type2>
__aicore__ inline void UpdateExp(AscendC::LocalTensor<T> src, uint32_t repeat);

template <>
__aicore__ inline void UpdateExp<float, Enums::PrecType::BMM1_FP16_EXP_FP32>(AscendC::LocalTensor<float> src, uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V200, float>(src, src, repeat, 1, 1, uint16_t(8), uint16_t(8));
}

template <>
__aicore__ inline void UpdateExp<float, Enums::PrecType::BMM1_FP32_EXP_FP32>(AscendC::LocalTensor<float> src, uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V200, float>(src, src, repeat, 1, 1, uint16_t(8), uint16_t(8));
}

template <>
__aicore__ inline void UpdateExp<float, Enums::PrecType::BMM2_ONLINE_SOFTMAX_FP16>(AscendC::LocalTensor<float> src,
                                                                            uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V200, float>(src, src, repeat, 1, 1, uint16_t(8), uint16_t(8));
}

template <typename T = half, typename SType = half, Enums::PrecType prec_type1 = Enums::PrecType::BMM1_FP16_EXP_FP32,
          Enums::PrecType prec_type2 = Enums::PrecType::BMM1_FP16_EXP_FP32>
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
        m = mReal;
        n = nReal;
        k = kReal;
        d = kReal;

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

    __aicore__ inline void SetParams(half torIn, int32_t kvCopyStrideIn)
    {
        tor = torIn;
        kvCopyStride = kvCopyStrideIn;
    }
    __aicore__ inline void SetEncoderParams(SType torIn, int32_t kvCopyStrideIn, int32_t isSqrtIn, int32_t Enums::PrecTypeIn)
    {
        tor = torIn;
        kvCopyStride = kvCopyStrideIn;
        isSqrt = isSqrtIn;
        Enums::PrecType = PrecTypeIn;
        if (Enums::PrecType == Enums::PrecType::BMM1_FP32_EXP_FP32) {
            lpUbufSize = UB_FLOAT_BUF_SIZE;
        }
    }

    __aicore__ inline void InitKVgmBatchwise(uint64_t kBatchPtr, uint64_t vBatchPtr)
    {
        gmSrck = reinterpret_cast<__gm__ uint8_t *>(kBatchPtr);
        gmSrcv = reinterpret_cast<__gm__ uint8_t *>(vBatchPtr);
    }

    __aicore__ inline void InitDec(int32_t mReal, int32_t nReal, int32_t kReal, int64_t srcqOffsetReal,
                                   int64_t srckOffsetReal, int64_t srcvOffsetReal, int64_t srcmOffsetReal,
                                   int64_t dstoOffsetReal, int32_t initGReal, int32_t wrapOReal, int32_t ntokensQReal,
                                   int32_t maskStrideReal)
    {
        m = mReal;
        n = nReal;
        k = kReal;
        d = kReal;

        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srcmOffset0 = srcmOffsetReal;
        dstoOffset = dstoOffsetReal;
        maskStride = maskStrideReal;

        initG = initGReal;
        wrapO = wrapOReal;
        ntokensQ = ntokensQReal;

        const uint32_t lsUbuf_offset = 0;
        const uint32_t lpUbuf_offset = DEC_UB_UINT8_BLOCK_SIZE_T * 2;
        const uint32_t ls32Ubuf_offset = DEC_UB_UINT8_BLOCK_SIZE_T * 4;
        const uint32_t maskUbuf_offset = DEC_UB_UINT8_BLOCK_SIZE_T * 8;
        const uint32_t loUbuf_offset = DEC_UB_UINT8_BLOCK_SIZE_T * 10;

        lsUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lsUbuf_offset);
        lpUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lpUbuf_offset);
        ls32Ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32Ubuf_offset);
        maskUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(maskUbuf_offset);
        loUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbuf_offset);
    }

public:
    __aicore__ inline void FlashAttentionNzPrefillCompute(
        const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn, const int32_t m0, const int32_t n0,
        const int32_t n1, const int32_t pp_n_scalar, const int32_t q_tight, const int32_t add_mask_n0,
        const int32_t add_mask_n1, const int32_t long_seq, const SType alibi_coeff, const SType delta0,
        const SType delta1, const uint32_t scale_type, const uint32_t alibi_left_align);
    __aicore__ inline void FlashAttentionNzDecoderCompute(const int32_t fm, const int32_t fn, const int32_t fk,
                                                          const int32_t bn, const int32_t m0, const int32_t n0,
                                                          const int32_t n1, const int32_t pp_n_scalar, half local_tor,
                                                          const uint32_t scale_type);

public:
    __aicore__ inline void
    Run(AscendC::LocalTensor<int32_t> tiling_para_ub_tensor, AscendC::GlobalTensor<int32_t> tiling_para_gm_tensor,
        AscendC::LocalTensor<int32_t> layerId_ub_tensor, __gm__ uint8_t *__restrict__ alibi_coeff_gm,
        uint32_t mask_type, uint32_t window_len, uint32_t long_seq, uint64_t stride_qo, uint64_t stride_kv,
        int64_t head_mask_stride, int64_t batch_mask_stride, uint32_t start_batch, uint32_t end_batch,
        int32_t start_blk, int32_t end_blk, uint32_t is_triu, uint32_t alibi_compress_offset, int32_t group_num,
        uint32_t mask_stride, uint32_t q_tokens, int32_t embd, uint32_t q_tight, uint32_t scaleType, SType tor,
        int32_t kv_copy_stride, uint32_t is_sqrt, int64_t heads, uint32_t max_seqlen, uint32_t batch_size,
        int32_t kv_real_heads, const uint32_t alibi_left_align);
    __aicore__ inline void RowSum(const int32_t n0, const int32_t fm, int32_t Pingflag);
    __aicore__ inline void SoftmaxUpdate(int32_t fm, int32_t fk, int32_t oSize, int32_t Pingflag, int32_t initGgO,
                                         int32_t mD64);
    __aicore__ inline void UpdateOutput(int32_t fm, int32_t fk, int32_t oSize, int32_t mD64, int32_t m0);
    __aicore__ inline void UpdateFP32(const int32_t fm, const int32_t fk, const int32_t n1, uint32_t initGgO);
    __aicore__ inline void UpdateFP16(const int32_t fm, const int32_t fk, const int32_t n1, uint32_t initGgO);
    __aicore__ inline void SoftMax(const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn,
                                   const int32_t m0, const int32_t n0, const int32_t n1,
                                   const int32_t add_mask_n0, const int32_t add_mask_n1, const SType alibi_coeff,
                                   const SType delta0, const SType delta1, const uint32_t scale_type,
                                   const uint32_t alibi_left_align, uint32_t initGgDm);

public:
    __aicore__ void __set_mask(int32_t len)
    {
        if (len >= VECTOR_SIZE_T) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - FLOAT_VECTOR_SIZE_T > 0 ? len - FLOAT_VECTOR_SIZE_T : 0;
        int32_t lowMask = len - FLOAT_VECTOR_SIZE_T >= 0 ? FLOAT_VECTOR_SIZE_T : len;
        if (len < FLOAT_VECTOR_SIZE_T) {
            SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    __aicore__ void __set_vcg_mask(int32_t len)
    {
        if (len > BLOCK_SIZE_T) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << BLOCK_SIZE_T) + subMask;
        SetVectorMask<int8_t>(maskValue, maskValue);
    }
    __aicore__ void __set_vcg_mask_fp32(int32_t len)
    {   
        uint32_t float_block_size = 8;
        if (len > float_block_size) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << BLOCK_SIZE_T) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        SetVectorMask<int8_t>(0x0, maskValue);
    }
    __aicore__ void ExpandToBlockHalf(AscendC::LocalTensor<half> dst_tensor, AscendC::LocalTensor<half> src_tensor,
                                      int32_t len)
    {
        constexpr int32_t BLOCK_EXPAND_ROUNDS = 2;
        constexpr uint32_t FP16_COMPUTE_GRANULARITY = 8;
        for (int32_t vaddsIdx = 0; vaddsIdx < BLOCK_EXPAND_ROUNDS; ++vaddsIdx) { // (len,) -> len / BLOCK_SIZE_T 个 (BLOCK_SIZE_T, BLOCK_SIZE_T)
            adds_v<ArchType::ASCEND_V200, half>(dst_tensor[vaddsIdx * FP16_COMPUTE_GRANULARITY * BLOCK_SIZE_T], src_tensor, (half)(0.0),
                                                len / BLOCK_SIZE_T, // repeat
                                                1, 0, BLOCK_SIZE_T, 1);
        }
        PIPE_BARRIER(V);
        for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE_T); ++vtransIdx) { // (BLOCK_SIZE_T, len) -> (len, BLOCK_SIZE_T)
            tranpose_v<ArchType::ASCEND_V200, half>(dst_tensor[vtransIdx * CUBE_MATRIX_SIZE_T],
                                                    dst_tensor[vtransIdx * CUBE_MATRIX_SIZE_T]);
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
            Duplicate<float>(dst_tensor[owIdx * BLOCK_SIZE_T], scale, MASK_PLACEHOLDER, 1, 1, 8);
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
    const uint32_t lk_buf_offset = 2 * UB_UINT8_BLOCK_SIZE_T;
    const uint32_t lv_buf_offset = 2 * (L1_UINT8_BLOCK_SIZE_T + UB_UINT8_BLOCK_SIZE_T);
    const uint32_t lp_buf_offset = 2 * L1_UINT8_BLOCK_SIZE_T;
    const uint32_t lmask_buf_offset = 4 * L1_UINT8_BLOCK_SIZE_T;
    const uint32_t lalibi_coeff_buf_offset = 4 * (L1_UINT8_BLOCK_SIZE_T + UB_UINT8_BLOCK_SIZE_T);
    const uint32_t ldiag_buf_offset = 5 * L1_UINT8_BLOCK_SIZE_T; // VECTOR_SIZE_T * VECTOR_SIZE_T * 2(fp16) * 2(PingPong) = 64k
    const uint32_t lo_buf_offset = 6 * L1_UINT8_BLOCK_SIZE_T;    // VECTOR_SIZE_T(qSeqStep) * VECTOR_SIZE_T(embedDim) * 2(fp16) = 32k

    const uint32_t ls_ubuf_offset = 0;
    const uint32_t lp_ubuf_offset = 0;
    const uint32_t ls32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_T;
    const uint32_t lo_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE_T;
    const uint32_t lm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_T;
    const uint32_t hm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_T + 1 * UB_UINT8_LINE_SIZE_T;
    const uint32_t gm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_T + 2 * UB_UINT8_LINE_SIZE_T;
    const uint32_t dm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_T + 3 * UB_UINT8_LINE_SIZE_T;
    const uint32_t ll_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_T + 5 * UB_UINT8_LINE_SIZE_T;
    const uint32_t gl_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE_T + 7 * UB_UINT8_LINE_SIZE_T;
    const uint32_t tiling_para_ub_offset = 4 * UB_UINT8_BLOCK_SIZE_T + 9 * UB_UINT8_LINE_SIZE_T;
    const uint32_t logn_ub_offset = 4 * UB_UINT8_BLOCK_SIZE_T + 31 * UB_UINT8_LINE_SIZE_T;
    const uint32_t tv_ubuf_offset = 5 * UB_UINT8_BLOCK_SIZE_T;
    const uint32_t go_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE_T;
    const uint32_t mask_ubuf_offset = DEC_UB_UINT8_BLOCK_SIZE_T * 8;

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
    // diagUbuf_tensor复用lsUbuf_tensor空间，128*VECTOR_SIZE_T*2*2=64k
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
    // toUbuf_tensor复用goUbuf_tensor空间，128*VECTOR_SIZE_T*2=32k
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
    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
    int32_t d = 0;
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
    int32_t lpUbufSize = L0AB_HALF_BUF_SIZE_T;
};


template <>
__aicore__ inline void UnpadFlashAttentionCommon<float, half, Enums::PrecType::BMM1_FP16_EXP_FP32, Enums::PrecType::BMM1_FP16_EXP_FP32>::FlashAttentionNzDecoderCompute(const int32_t fm, const int32_t fn,
                                                                                 const int32_t fk, const int32_t bn,
                                                                                 const int32_t m0, const int32_t n0,
                                                                                 const int32_t n1,
                                                                                 const int32_t pp_n_scalar,
                                                                                 half local_tor,
                                                                                 const uint32_t scale_type)
{
    int32_t Pingflag = 0; // manual PingPong attempt
    int32_t Pongflag = 1;

    if (scale_type == 0) {
        local_tor = tor;
    }

    const uint32_t l1q_buf_addr_offset = 0;
    const uint32_t l1kpv_buf_addr_offset = 4 * L1_UINT8_BLOCK_SIZE_T;
    const uint32_t l1mask_buf_addr_offset = 2 * L0AB_HALF_BUF_SIZE_T;

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
    AscendC::LocalTensor<half> l1maskPingBuf_tensor =
        l1maskBufAddr_tensor.ReinterpretCast<uint8_t>()[Pingflag * l1mask_buf_addr_offset].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1maskPongBuf_tensor =
        l1maskBufAddr_tensor.ReinterpretCast<uint8_t>()[Pongflag * l1mask_buf_addr_offset].ReinterpretCast<half>();

    gmSrcq_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcq));
    gmSrck_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrck));
    gmSrcv_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcv));
    gmSrcm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcm));
    gmDsto_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmDsto));

    __cbuf__ uint8_t *l1qBuf = l1qBufAddr;
    __cbuf__ uint8_t *l1kPingBuf = l1kBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE_T;
    __cbuf__ uint8_t *l1kPongBuf = l1kBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE_T;
    __cbuf__ uint8_t *l1vPingBuf = l1vBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE_T;
    __cbuf__ uint8_t *l1vPongBuf = l1vBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE_T;
    __cbuf__ uint8_t *l1pPingBuf = l1pBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE_T;
    __cbuf__ uint8_t *l1pPongBuf = l1pBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE_T;
    __cbuf__ uint8_t *l1maskPingBuf = l1maskBufAddr + Pingflag * 2 * L0AB_HALF_BUF_SIZE_T;
    __cbuf__ uint8_t *l1maskPongBuf = l1maskBufAddr + Pongflag * 2 * L0AB_HALF_BUF_SIZE_T;

    int32_t oSize = fm * fk;
    int32_t mD64 = (fm + FLOAT_VECTOR_SIZE_T - 1) / FLOAT_VECTOR_SIZE_T;
    int32_t mD128 = (fm + VECTOR_SIZE_T - 1) / VECTOR_SIZE_T;
    int32_t initGgDm = (initG == 1) ? 1 : 0;
    int32_t initGgO = (initG == 1) ? 1 : 0;

    int32_t pSize = fm * fn;
    int32_t pSize_b = fm * bn;

    // 1. ################ Bmm1 Ping Start #######################
    // 1.1 ################ QK Ping LOAD ################
    if (initGgO != 0) {
        WAIT_FLAG(MTE1, MTE2, Pingflag);
        WAIT_FLAG(MTE1, MTE2, Pongflag);
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1qBuf_tensor,
            gmSrcq_tensor[(int64_t)srcqOffset],
            1, ntokensQ, 1, fk, fk, fk);
        SET_FLAG(MTE2, MTE1, Pingflag);
        if (n1 != -1) {
            SET_FLAG(MTE2, MTE1, Pongflag);
        }
    }
    if (gmSrcm != nullptr) {
        WAIT_FLAG(MTE1, MTE2, Pingflag + 2);
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1maskPingBuf_tensor,
            gmSrcm_tensor[srcmOffset0],
            1, maskStride, 1, fn, fn, fn);
        SET_FLAG(MTE2, MTE1, Pingflag + 2);
        WAIT_FLAG(MTE2, MTE1, Pingflag + 2);
        WAIT_FLAG(V, MTE1, Pingflag);
        l1_to_ub<ArchType::ASCEND_V200, half>(
            maskUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            l1maskPingBuf_tensor,
            1, fn / BLOCK_SIZE_T, 0, 0);
        SET_FLAG(MTE1, V, Pingflag);
        SET_FLAG(MTE1, MTE2, Pingflag + 2);
        if (n1 != -1) {
            WAIT_FLAG(MTE1, MTE2, Pongflag + 2);
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1maskPongBuf_tensor,
                gmSrcm_tensor[srcmOffset0 + maskStride * pp_n_scalar],
                1, maskStride, 1, bn, bn, bn);
            SET_FLAG(MTE2, MTE1, Pongflag + 2);
            WAIT_FLAG(MTE2, MTE1, Pongflag + 2);
            WAIT_FLAG(V, MTE1, Pongflag);
            l1_to_ub<ArchType::ASCEND_V200, half>(
                maskUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                l1maskPongBuf_tensor,
                1, bn / BLOCK_SIZE_T, 0, 0);
            SET_FLAG(MTE1, V, Pongflag);
            SET_FLAG(MTE1, MTE2, Pongflag + 2);
        }
    }
    WAIT_FLAG(M, MTE1, Pingflag);
    if (initGgO == 1) {
        WAIT_FLAG(MTE2, MTE1, Pingflag);
    }
    l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        l1qBuf_tensor,
        0, 1, 0, 1, 0, 0);

    SET_FLAG(MTE1, M, Pingflag);
    WAIT_FLAG(MTE1, MTE2, Pingflag + 4);
    gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
        l1kPingBuf_tensor,
        gmSrck_tensor[(int64_t)srckOffset],
        fn, kvCopyStride, fn, fk, fk, fk);
    SET_FLAG(MTE2, MTE1, Pingflag);
    WAIT_FLAG(MTE2, MTE1, Pingflag);
    WAIT_FLAG(M, MTE1, Pingflag + 2);
    l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        l1kPingBuf_tensor,
        0, fk * fn / CUBE_MATRIX_SIZE_T, 0, 1, 0, 0);
    SET_FLAG(MTE1, MTE2, Pingflag + 4);
    SET_FLAG(MTE1, M, Pingflag + 2);
    // 2. ################ Bmm1 Pong Starts #######################
    // 2.1 ################ QK Pong PRELOAD ################
    if (n1 != -1) {
        WAIT_FLAG(M, MTE1, Pongflag);
        if (initGgO == 1) {
            WAIT_FLAG(MTE2, MTE1, Pongflag);
        }
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            l1qBuf_tensor,
            0, 1, 0, 1, 0, 0);
        SET_FLAG(MTE1, M, Pongflag);
        WAIT_FLAG(MTE1, MTE2, Pongflag + 4);
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1kPongBuf_tensor,
            gmSrck_tensor[(int64_t)srckOffset + Pongflag * pp_n_scalar * BLOCK_SIZE_T],
            bn, kvCopyStride, bn, fk, fk, fk);
        SET_FLAG(MTE2, MTE1, Pongflag);
        WAIT_FLAG(MTE2, MTE1, Pongflag);
        WAIT_FLAG(M, MTE1, Pongflag + 2);
        l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            l1kPongBuf_tensor,
            0, fk * bn / CUBE_MATRIX_SIZE_T, 0, 1, 0, 0);
        SET_FLAG(MTE1, M, Pongflag + 2);
        SET_FLAG(MTE1, MTE2, Pongflag + 4);
    }
    // 1.2 ################ Bmm1 Ping + V PRELOAD ################
    WAIT_FLAG(MTE1, MTE2, Pingflag + 6);
    gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
        l1vPingBuf_tensor,
        gmSrcv_tensor[(int64_t)srcvOffset],
        fn, kvCopyStride, fn, fk, fk, fk);
    SET_FLAG(MTE2, MTE1, Pingflag + 4);
    WAIT_FLAG(MTE1, M, Pingflag + 2);
    WAIT_FLAG(MTE1, M, Pingflag);
    WAIT_FLAG(V, M, Pingflag);
    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        m0, n0, fk, 1);
    SET_FLAG(M, V, Pingflag);
    SET_FLAG(M, MTE1, Pingflag);
    SET_FLAG(M, MTE1, Pingflag + 2);
    WAIT_FLAG(MTE2, MTE1, Pingflag + 4);
    WAIT_FLAG(M, MTE1, Pingflag + 2);
    if (fk == BLOCK_SIZE_T) {
        l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
            l1vPingBuf_tensor,
            0, fn / BLOCK_SIZE_T, 0, 1, 0, 0);
    } else {
        for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (fn / BLOCK_SIZE_T); ++l0bLoadIdx) { // Nz -> nZ
            l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T + l0bLoadIdx * fk * BLOCK_SIZE_T],
                l1vPingBuf_tensor[l0bLoadIdx * CUBE_MATRIX_SIZE_T],
                0, fk / BLOCK_SIZE_T, 0, fn / BLOCK_SIZE_T, 0, 0);
        }
    }
    SET_FLAG(MTE1, MTE2, Pingflag + 6);
    SET_FLAG(MTE1, M, Pingflag + 2);
    // 1. ################ Bmm1 Ping Ends #######################
    // 2.2 ################ Bmm1 Pong + V PRELOAD ################
    if (n1 != -1) {
        WAIT_FLAG(MTE1, MTE2, Pongflag + 6);
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1vPongBuf_tensor,
            gmSrcv_tensor[(int64_t)srcvOffset + Pongflag * pp_n_scalar * BLOCK_SIZE_T],
            bn, kvCopyStride, bn, fk, fk, fk);
        SET_FLAG(MTE2, MTE1, Pongflag + 4);
        WAIT_FLAG(MTE1, M, Pongflag + 2);
        WAIT_FLAG(MTE1, M, Pongflag);
        WAIT_FLAG(V, M, Pongflag);
        mmad<ArchType::ASCEND_V200, half, half, float, false>(
            l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            m0, n1, fk, 1);
        SET_FLAG(M, V, Pongflag);
        SET_FLAG(M, MTE1, Pongflag);
        SET_FLAG(M, MTE1, Pongflag + 2);
        WAIT_FLAG(MTE2, MTE1, Pongflag + 4);
        WAIT_FLAG(M, MTE1, Pongflag + 2);
        if (fk == BLOCK_SIZE_T) {
            l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
                l1vPongBuf_tensor,
                0, bn / BLOCK_SIZE_T, 0, 1, 0, 0);
        } else {
            for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (bn / BLOCK_SIZE_T); ++l0bLoadIdx) { // Nz -> nZ
                l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0bBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T + l0bLoadIdx * fk * BLOCK_SIZE_T],
                    l1vPongBuf_tensor[l0bLoadIdx * CUBE_MATRIX_SIZE_T],
                    0, fk / BLOCK_SIZE_T, 0, bn / BLOCK_SIZE_T, 0, 0);
            }
        }
        SET_FLAG(MTE1, MTE2, Pongflag + 6);
        SET_FLAG(MTE1, M, Pongflag + 2);
    }
    // 2. ################ Bmm1 Pong Ends #######################
    // 3. ################ Softmax Ping Starts #######################
    WAIT_FLAG(M, V, Pingflag);
    l0c_to_ub<ArchType::ASCEND_V200, float, half>(
        lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        1, pSize / CUBE_MATRIX_SIZE_T, 0, 0);
    PIPE_BARRIER(V);
    SET_FLAG(V, M, Pingflag);
    muls_v<ArchType::ASCEND_V200, half>(
        lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        local_tor,
        (fn + 127) / VECTOR_SIZE_T, 1, fm, 8, fm * 8);

    PIPE_BARRIER(V);
    if (gmSrcm != nullptr) {
        WAIT_FLAG(MTE1, V, Pingflag);
        add_v<ArchType::ASCEND_V200, half>(
            lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            maskUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            n0 / VECTOR_SIZE_T, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        if (n0 % VECTOR_SIZE_T != 0) {
            __set_mask(n0 % VECTOR_SIZE_T);
            add_v<ArchType::ASCEND_V200, half>(
                lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE + (n0 / VECTOR_SIZE_T) * VECTOR_SIZE_T],
                lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE + (n0 / VECTOR_SIZE_T) * VECTOR_SIZE_T],
                maskUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE + (n0 / VECTOR_SIZE_T) * VECTOR_SIZE_T],
                1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        }
        SET_FLAG(V, MTE1, Pingflag);
    }
    if (n0 <= VECTOR_SIZE_T) {
        if (n0 != VECTOR_SIZE_T) {
            __set_mask(n0 % VECTOR_SIZE_T);
        }
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbuf_tensor,
            lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            1, 1, 1, 8);
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, half>(
            tvUbuf_tensor,
            lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            0, 1, 8, 8, 8);

        PIPE_BARRIER(V);
        if (n0 % VECTOR_SIZE_T != 0) {
            __set_mask(n0 % VECTOR_SIZE_T);
        }
        max_v<ArchType::ASCEND_V200, half>(
            tvUbuf_tensor,
            tvUbuf_tensor,
            lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE + VECTOR_SIZE_T],
            1, 1, 1, 1, 8, 8, 8 );
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbuf_tensor,
            tvUbuf_tensor,
            1, 1, 1, 8);
        PIPE_BARRIER(V);
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);

    if (initGgDm == 0) {
        max_v<ArchType::ASCEND_V200, half>(
            hmUbuf_tensor,
            lmUbuf_tensor,
            gmUbuf_tensor,
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(
            dmUbuf_tensor[Pingflag * UB_HALF_LINE_SIZE_T],
            gmUbuf_tensor,
            hmUbuf_tensor,
            1, 1, 1, 1, 8, 8, 8);

        PIPE_BARRIER(V);
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbuf_tensor,
            hmUbuf_tensor,
            0, 1, 1, 0, 0);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbuf_tensor, hmUbuf_tensor, fm);
    } else {
        initGgDm = 0;
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbuf_tensor,
            lmUbuf_tensor,
            0, 1, 1, 0, 0);
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbuf_tensor,
            lmUbuf_tensor,
            0, 1, 1, 0, 0);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbuf_tensor, gmUbuf_tensor, fm);
    }
    sub_v<ArchType::ASCEND_V200, half>(
        lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        tvUbuf_tensor.ReinterpretCast<half>(),
        (fn + 127) / VECTOR_SIZE_T, 1, 1, 0, 8, 8, 0);
    PIPE_BARRIER(V);
    conv_v<ArchType::ASCEND_V200, half, float>(
        ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        lsUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        (fn + 63) / FLOAT_VECTOR_SIZE_T, 1, 1, 8, 4);

    PIPE_BARRIER(V);
    exp_v<ArchType::ASCEND_V200, float>(
        ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        (fn + FLOAT_VECTOR_SIZE_T - 1) / FLOAT_VECTOR_SIZE_T, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    WAIT_FLAG(MTE3, V, Pingflag);
    conv_v<ArchType::ASCEND_V200, float, half>(
        lpUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        (fn + FLOAT_VECTOR_SIZE_T - 1) / FLOAT_VECTOR_SIZE_T, 1, 1, 4, 8);

    PIPE_BARRIER(V);
    SET_FLAG(V, MTE3, Pingflag);
    SetMaskNorm();
    if (n0 < FLOAT_VECTOR_SIZE_T) {
        if (n0 != FLOAT_VECTOR_SIZE_T) {
            SetVectorMask<int8_t>(0x0, ((long)1 << n0) - 1);
        }
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE_T],
            ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            1, 1, 1, 2);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    } else {
        for (int64_t vcalcIdx = 1; vcalcIdx < n0 / FLOAT_VECTOR_SIZE_T; vcalcIdx++) {
            add_v<ArchType::ASCEND_V200, float>(
                ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE + vcalcIdx * FLOAT_VECTOR_SIZE_T],
                1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
        }
        if (n0 % FLOAT_VECTOR_SIZE_T != 0) {
            __set_mask(n0 % FLOAT_VECTOR_SIZE_T);
            add_v<ArchType::ASCEND_V200, float>(
                ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE + (n0 / FLOAT_VECTOR_SIZE_T) * FLOAT_VECTOR_SIZE_T],
                1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        }
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE_T],
            ls32Ubuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            1, 1, 1, 2);
    }
    PIPE_BARRIER(V);
    SetMaskNorm();
    SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    WAIT_FLAG(MTE1, MTE3, Pingflag);
    WAIT_FLAG(V, MTE3, Pingflag);
    ub_to_l1<ArchType::ASCEND_V200, half>(
        l1pPingBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        lpUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        1, fn / BLOCK_SIZE_T, 0, 0);
    SET_FLAG(MTE3, V, Pingflag);
    SET_FLAG(MTE3, MTE1, Pingflag);
    // 3. ################ Softmax Ping Ends #######################
    // 4. ################ Softmax Pong Starts #######################
    if (n1 != -1) {
        WAIT_FLAG(M, V, Pongflag);
        l0c_to_ub<ArchType::ASCEND_V200, float, half>(
            lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            1, pSize / CUBE_MATRIX_SIZE_T, 0, 0);
        PIPE_BARRIER(V);
        SET_FLAG(V, M, Pongflag);

        muls_v<ArchType::ASCEND_V200, half>(
            lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            local_tor,
            (bn + 127) / VECTOR_SIZE_T, 1, fm, 8, fm * 8);
        PIPE_BARRIER(V);
        if (gmSrcm != nullptr) {
            WAIT_FLAG(MTE1, V, Pongflag);
            add_v<ArchType::ASCEND_V200, half>(
                lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                maskUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                n1 / VECTOR_SIZE_T, 1, 1, 1, 8, 8, 8);

            PIPE_BARRIER(V);
            if (n1 % VECTOR_SIZE_T != 0) {
                __set_mask(n1 % VECTOR_SIZE_T);
                add_v<ArchType::ASCEND_V200, half>(lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE + (n1 / VECTOR_SIZE_T) * VECTOR_SIZE_T],
                                                   lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE + (n1 / VECTOR_SIZE_T) * VECTOR_SIZE_T],
                                                   maskUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE + (n1 / VECTOR_SIZE_T) * VECTOR_SIZE_T],
                                                   1, 1, 1, 1, 8, 8, 8);

                PIPE_BARRIER(V);
                SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
            }
            SET_FLAG(V, MTE1, Pongflag);
        }
        // 3. softmax part
        if (n1 <= VECTOR_SIZE_T) {
            if (n1 != VECTOR_SIZE_T) {
                __set_mask(n1 % VECTOR_SIZE_T);
            }
            cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
                lmUbuf_tensor,
                lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                1, 1, 1, 8);
            PIPE_BARRIER(V);
        } else {
            ub_to_ub<ArchType::ASCEND_V200, half>(
                tvUbuf_tensor,
                lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                0, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            if (n1 % VECTOR_SIZE_T != 0) {
                __set_mask(n1 % VECTOR_SIZE_T);
            }
            max_v<ArchType::ASCEND_V200, half>(
                tvUbuf_tensor,
                tvUbuf_tensor,
                lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE + VECTOR_SIZE_T],
                1, 1, 1, 1, 8, 8, 8);

            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
            cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
                lmUbuf_tensor,
                tvUbuf_tensor,
                1, 1, 1, 8);
            PIPE_BARRIER(V);
        }
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        PIPE_BARRIER(V);
        max_v<ArchType::ASCEND_V200, half>(
            hmUbuf_tensor,
            lmUbuf_tensor,
            gmUbuf_tensor,
            1, 1, 1, 1, 8, 8, 8);

        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(
            dmUbuf_tensor[Pongflag * UB_HALF_LINE_SIZE_T],
            gmUbuf_tensor,
            hmUbuf_tensor,
            1, 1, 1, 1, 8, 8, 8);

        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbuf_tensor, hmUbuf_tensor, fm); 
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbuf_tensor,
            hmUbuf_tensor,
            0, 1, 1, 0, 0);

        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(
            lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            tvUbuf_tensor,
            (bn + 127) / VECTOR_SIZE_T, 1, 1, 0, 8, 8, 0);

        PIPE_BARRIER(V);
        conv_v<ArchType::ASCEND_V200, half, float>(
            ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            lsUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            (bn + FLOAT_VECTOR_SIZE_T - 1) / FLOAT_VECTOR_SIZE_T, 1, 1, 8, 4);
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            (bn + FLOAT_VECTOR_SIZE_T - 1) / FLOAT_VECTOR_SIZE_T, 1, 1, 8, 8);

        PIPE_BARRIER(V);
        WAIT_FLAG(MTE3, V, Pongflag);
        conv_v<ArchType::ASCEND_V200, float, half>(
            lpUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            (bn + FLOAT_VECTOR_SIZE_T - 1) / FLOAT_VECTOR_SIZE_T, 1, 1, 4, 8);
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, Pongflag);
        SetMaskNorm();
        if (n1 < FLOAT_VECTOR_SIZE_T) {
            if (n1 != FLOAT_VECTOR_SIZE_T) {
                SetVectorMask<int8_t>(0x0, ((long)1 << n1) - 1);
            }
            cadd_v<ArchType::ASCEND_V200, float>(
                llUbuf_tensor[Pongflag * UB_FLOAT_LINE_SIZE_T],
                ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                1, 1, 1, 2);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        } else {
            for (int64_t vcalcIdx = 1; vcalcIdx < n1 / FLOAT_VECTOR_SIZE_T; vcalcIdx++) {
                add_v<ArchType::ASCEND_V200, float>(
                    ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE + vcalcIdx * FLOAT_VECTOR_SIZE_T],
                    1, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);
            }
            if (n1 % FLOAT_VECTOR_SIZE_T != 0) {
                SetVectorMask<int8_t>(0x0, ((long)1 << (n1 % FLOAT_VECTOR_SIZE_T)) - 1);
                add_v<ArchType::ASCEND_V200, float>(
                    ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE + n1 / FLOAT_VECTOR_SIZE_T * FLOAT_VECTOR_SIZE_T],
                    1, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);
                SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
            }
            cadd_v<ArchType::ASCEND_V200, float>(
                llUbuf_tensor[Pongflag * UB_FLOAT_LINE_SIZE_T],
                ls32Ubuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
                1, 1, 1, 2);
        }
        PIPE_BARRIER(V);
        SetMaskNorm();
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        WAIT_FLAG(MTE1, MTE3, Pongflag);
        WAIT_FLAG(V, MTE3, Pongflag);
        ub_to_l1<ArchType::ASCEND_V200, half>(
            l1pPongBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            lpUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            1, bn / BLOCK_SIZE_T, 0, 0);
        SET_FLAG(MTE3, V, Pongflag);
        SET_FLAG(MTE3, MTE1, Pongflag);
    }
    // 4. ################ Softmax Pong Ends #######################
    // 5. ################ Bmm2 Ping Starts #######################
    WAIT_FLAG(MTE3, MTE1, Pingflag);
    WAIT_FLAG(M, MTE1, Pingflag);
    l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0aBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        l1pPingBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        0, 1, 0, 1, 0, 0);
    SET_FLAG(MTE1, MTE3, Pingflag);
    SET_FLAG(MTE1, M, Pingflag);
    WAIT_FLAG(MTE1, M, Pingflag);
    WAIT_FLAG(MTE1, M, Pingflag + 2);
    WAIT_FLAG(V, M, Pingflag + 2);
    mmad<ArchType::ASCEND_V200, __fp16, __fp16, float, false>(
        l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        l0aBuf_tensor.ReinterpretCast<__fp16>()[Pingflag * L0AB_HALF_BUF_SIZE_T],
        l0bBuf_tensor.ReinterpretCast<__fp16>()[Pingflag * L0AB_HALF_BUF_SIZE_T],
        m0, fk, n0, 1);
    SET_FLAG(M, MTE1, Pingflag);
    SET_FLAG(M, MTE1, Pingflag + 2);
    SET_FLAG(M, V, Pingflag);
    if (wrapO == 1) {
        SET_FLAG(MTE1, MTE2, Pingflag);
        if (n1 == -1) {
            SET_FLAG(MTE1, MTE2, Pongflag);
        }
    }
    // 5. ################ Bmm2 Ping Ends #######################
    // 6. ################ Bmm2 Pong Starts #######################
    if (n1 != -1) {
        WAIT_FLAG(MTE3, MTE1, Pongflag);
        WAIT_FLAG(M, MTE1, Pongflag);
        // BLOCK_SIZE_T is blocksize in format zN
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            l1pPongBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            0, 1, 0, 1, 0, 0
        );
        SET_FLAG(MTE1, MTE3, Pongflag);
        SET_FLAG(MTE1, M, Pongflag);
        WAIT_FLAG(MTE1, M, Pongflag);
        WAIT_FLAG(MTE1, M, Pongflag + 2);
        WAIT_FLAG(V, M, Pongflag + 2);

        mmad<ArchType::ASCEND_V200, __fp16, __fp16, float, false>(
            l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            l0aBuf_tensor.ReinterpretCast<__fp16>()[Pongflag * L0AB_HALF_BUF_SIZE_T],
            l0bBuf_tensor.ReinterpretCast<__fp16>()[Pongflag * L0AB_HALF_BUF_SIZE_T],
            m0, fk, n1, 1);

        SET_FLAG(M, MTE1, Pongflag);
        SET_FLAG(M, MTE1, Pongflag + 2);
        SET_FLAG(M, V, Pongflag);
        if (wrapO == 1) {
            SET_FLAG(MTE1, MTE2, Pongflag);
        }
    }
    // 6. ################ Bmm2 Pong Ends #######################
    // 7. ################ Update Ping Starts #######################
    WAIT_FLAG(M, V, Pingflag);
    l0c_to_ub<ArchType::ASCEND_V200, float, float>(
        loUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
        l0cBuf_tensor[Pingflag * L0AB_HALF_BUF_SIZE_T],
        1, oSize / CUBE_MATRIX_SIZE_T, 0, 0);
    PIPE_BARRIER(V);
    // 8. ################ Update Pong Starts #######################
    if (n1 != -1) {
        WAIT_FLAG(M, V, Pongflag);
        l0c_to_ub<ArchType::ASCEND_V200, float, float>(
            loUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE],
            l0cBuf_tensor[Pongflag * L0AB_HALF_BUF_SIZE_T],
            1, oSize / CUBE_MATRIX_SIZE_T, 0, 0);
        PIPE_BARRIER(V);
    }
    if (initGgO == 0) {
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbuf_tensor.ReinterpretCast<float>(),
            dmUbuf_tensor[Pingflag * UB_HALF_LINE_SIZE_T],
            mD64, 1, 1, uint16_t(8), uint16_t(4));

        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbuf_tensor.ReinterpretCast<float>(),
            tvUbuf_tensor.ReinterpretCast<float>(),
            mD64, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        mul_v<ArchType::ASCEND_V200, float>(
            glUbuf_tensor,
            tvUbuf_tensor.ReinterpretCast<float>(),
            glUbuf_tensor,
            mD64, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            glUbuf_tensor,
            glUbuf_tensor,
            llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE_T],
            mD64, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbuf_tensor, dmUbuf_tensor[Pingflag * UB_HALF_LINE_SIZE_T], fm);

        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
            tvUbuf_tensor,
            fm * BLOCK_SIZE_T / FLOAT_VECTOR_SIZE_T, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
            tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
            fm * BLOCK_SIZE_T / FLOAT_VECTOR_SIZE_T, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);

        if (vmPingpongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingpongFlag = 0;
        }

        for (int32_t vmulIdx = 0; vmulIdx < (fk / BLOCK_SIZE_T); ++vmulIdx) {
            mul_v<ArchType::ASCEND_V200, float>(
                goUbuf_tensor[vmulIdx * fm * BLOCK_SIZE_T],
                goUbuf_tensor[vmulIdx * fm * BLOCK_SIZE_T],
                tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
                fm * BLOCK_SIZE_T / FLOAT_VECTOR_SIZE_T, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
        }
        for (int32_t vaddIdx = 0; vaddIdx < 2; ++vaddIdx) {
            add_v<ArchType::ASCEND_V200, float>(
                goUbuf_tensor[vaddIdx * oSize / 2],
                goUbuf_tensor[vaddIdx * oSize / 2],
                loUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE + vaddIdx * oSize / 2],
                oSize / 2 / FLOAT_VECTOR_SIZE_T, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
        }
    } else {
        ub_to_ub<ArchType::ASCEND_V200, float>(
            glUbuf_tensor,
            llUbuf_tensor[Pingflag * UB_FLOAT_LINE_SIZE_T],
            0, 1, fm / 8, 0, 0);
        PIPE_BARRIER(V);
        if (vmPingpongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingpongFlag = 0;
        }
        ub_to_ub<ArchType::ASCEND_V200, float>(
            goUbuf_tensor,
            loUbuf_tensor[Pingflag * LOCAL_STORAGE_BUFFER_SIZE],
            0, 1, oSize / 8, 0, 0);
        PIPE_BARRIER(V);
    }
    PIPE_BARRIER(V);
    initGgO = 0;
    // 7. ################ Update Ping Ends #######################
    if (n1 != -1) {
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbuf_tensor.ReinterpretCast<float>(),
            dmUbuf_tensor[Pongflag * UB_HALF_LINE_SIZE_T],
            mD64, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbuf_tensor.ReinterpretCast<float>(),
            tvUbuf_tensor.ReinterpretCast<float>(),
            mD64, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        mul_v<ArchType::ASCEND_V200, float>(
            glUbuf_tensor,
            tvUbuf_tensor.ReinterpretCast<float>(),
            glUbuf_tensor,
            mD64, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            glUbuf_tensor,
            glUbuf_tensor,
            llUbuf_tensor[Pongflag * UB_FLOAT_LINE_SIZE_T],
            mD64, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        ExpandToBlockHalf(tvUbuf_tensor, dmUbuf_tensor[Pongflag * UB_HALF_LINE_SIZE_T], fm);
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
            tvUbuf_tensor,
            fm * BLOCK_SIZE_T / FLOAT_VECTOR_SIZE_T, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
            tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
            fm * BLOCK_SIZE_T / FLOAT_VECTOR_SIZE_T, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        if (vmPingpongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingpongFlag = 0;
        }

        for (int32_t vmulIdx = 0; vmulIdx < (fk / BLOCK_SIZE_T); ++vmulIdx) {
            mul_v<ArchType::ASCEND_V200, float>(
                goUbuf_tensor[vmulIdx * fm * BLOCK_SIZE_T],
                goUbuf_tensor[vmulIdx * fm * BLOCK_SIZE_T],
                tvUbuf_tensor.ReinterpretCast<float>()[fm * BLOCK_SIZE_T / 2],
                fm * BLOCK_SIZE_T / FLOAT_VECTOR_SIZE_T, 1, 1, 1, 8, 8, 8);

            PIPE_BARRIER(V);
        }
        for (int32_t vaddIdx = 0; vaddIdx < 2; ++vaddIdx) { // update Oj
            add_v<ArchType::ASCEND_V200, float>(
                goUbuf_tensor[vaddIdx * oSize / 2],
                goUbuf_tensor[vaddIdx * oSize / 2],
                loUbuf_tensor[Pongflag * LOCAL_STORAGE_BUFFER_SIZE + vaddIdx * oSize / 2],
                oSize / 2 / FLOAT_VECTOR_SIZE_T, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
        }
        SET_FLAG(V, M, Pongflag + 2);
    }
    SET_FLAG(V, M, Pingflag + 2);
    // 8. ################ Update Pong Ends #######################
    // 9. ################ Line Output Starts #####################
    if (wrapO == 1) {
        conv_v<ArchType::ASCEND_V200, float, half>(
            glUbuf_tensor.ReinterpretCast<half>(),
            glUbuf_tensor,
            mD64, 1, 1, uint16_t(4), uint16_t(8));
        PIPE_BARRIER(V);
        conv_v<ArchType::ASCEND_V200, float, half>(
            goUbuf_tensor.ReinterpretCast<half>(),
            goUbuf_tensor,
            oSize / FLOAT_VECTOR_SIZE_T, 1, 1, uint16_t(4), uint16_t(8));
        PIPE_BARRIER(V);

        ExpandToBlockHalf(tvUbuf_tensor, glUbuf_tensor.ReinterpretCast<half>(), fm);

        for (int32_t vdivIdx = 0; vdivIdx < (fk / BLOCK_SIZE_T); ++vdivIdx) {
            div_v<ArchType::ASCEND_V200, half>(
                goUbuf_tensor.ReinterpretCast<half>()[vdivIdx * fm * BLOCK_SIZE_T],
                goUbuf_tensor.ReinterpretCast<half>()[vdivIdx * fm * BLOCK_SIZE_T],
                tvUbuf_tensor,
                m0 * BLOCK_SIZE_T / VECTOR_SIZE_T, 1, 1, 1, 8, 8, 8);

            PIPE_BARRIER(V);
        }
        int32_t blockV = VECTOR_SIZE_T / BLOCK_SIZE_T;
        if (m0 % blockV != 0) {
            __set_mask(m0 * BLOCK_SIZE_T % VECTOR_SIZE_T);
            div_v<ArchType::ASCEND_V200, half>(
                goUbuf_tensor.ReinterpretCast<half>()[m0 * BLOCK_SIZE_T / VECTOR_SIZE_T * VECTOR_SIZE_T],
                goUbuf_tensor.ReinterpretCast<half>()[m0 * BLOCK_SIZE_T / VECTOR_SIZE_T * VECTOR_SIZE_T],
                tvUbuf_tensor[m0 / blockV * blockV * BLOCK_SIZE_T],
                fk / BLOCK_SIZE_T, 1, 1, 1, fm, fm, 0);
            SetVectorMask<int8_t>(-1, -1);
        }
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, EVENT_ID2);
        WAIT_FLAG(V, MTE3, EVENT_ID2);

        ub_to_gm<ArchType::ASCEND_V200, half>(
            gmDsto_tensor[(int64_t)dstoOffset],
            goUbuf_tensor.ReinterpretCast<half>(),
            0, fk / BLOCK_SIZE_T, m0, fm - m0, ntokensQ - m0);
        if (vmPingpongFlag == 0) {
            SET_FLAG(MTE3, V, EVENT_ID2);
            vmPingpongFlag = 1;
        }
    }
    // 9. ################ Line Output Ends #####################
}

template <typename IFAT> class UnpadFlashAttentionDecoder {
public:
    __aicore__ inline UnpadFlashAttentionDecoder(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace, const typename IFAT::TilingType *__restrict tiling);
    __aicore__ inline void Process();

    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;

protected:
    const typename IFAT::TilingType *__restrict tilingData = nullptr;
    uint32_t embeddingSize = 0;
    half tor = 0;
    uint32_t batchSize = 0;
    uint32_t scaleType = 0;
    uint32_t headSplit = 0;
    uint32_t maskType = 0;
    uint32_t maskStride = 0;
    uint32_t runMode = 0;
    int64_t headMaskStride = 0;
    int64_t batchMaskStride = 0;
    uint32_t qTokens = 0;
    uint32_t kvHead = 0;
    int64_t heads = 0;
    uint32_t maxSeqlen = 0;
    uint32_t windowLen = 0;
    uint32_t cacheType = 0;
    uint32_t maxKvSeqlen = 0;
    bool batchContinuous = true;

    uint32_t startBlk = 0;
    uint32_t endBlk = 0;
    uint32_t startBatch = 0;
    uint32_t endBatch = 0;

    uint32_t addrKHigh32 = 0;
    uint32_t addrKLow32 = 0;
    int64_t addrKScalar = 0;
    uint32_t addrVHigh32 = 0;
    uint32_t addrVLow32 = 0;
    int64_t addrVScalar = 0;

    __gm__ uint8_t *query;
    __gm__ uint8_t *key;
    __gm__ uint8_t *value;
    __gm__ uint8_t *attenMask;
    __gm__ uint8_t *attentionOut;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }
    __aicore__ inline void InitTilingData();
};

template <typename IFAT>
__aicore__ inline void UnpadFlashAttentionDecoder<IFAT>::Init(__gm__ uint8_t *_query, __gm__ uint8_t *_key,
                                __gm__ uint8_t *_value, __gm__ uint8_t *_attenMask, __gm__ uint8_t *_attentionOut,
                                __gm__ uint8_t *workspace, const typename IFAT::TilingType *__restrict tiling)
{
    tilingData = tiling;

    InitTilingData();
    this->query = _query;
    this->key = _key;
    this->value = _value;
    this->attenMask = _attenMask;
    this->attentionOut = _attentionOut;
}

template <typename IFAT>
__aicore__ inline void UnpadFlashAttentionDecoder<IFAT>::Process()
{
    SetMaskNorm();
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    SetLoadDataPaddingValue<int8_t>(uint16_t(0));
    SetAtomicNone();

    UnpadFlashAttentionCommon<float, half, Enums::PrecType::BMM1_FP16_EXP_FP32, Enums::PrecType::BMM1_FP16_EXP_FP32>
        flashAttentionDecoder(query, key, value, attenMask, nullptr, nullptr, nullptr, attentionOut);

    int32_t kv_real_heads = kvHead > 0 ? kvHead : heads;
    int32_t group_num = heads / kv_real_heads;
    int32_t kv_copy_stride = qTokens;
    uint64_t stride_batch_kv = batchSize * maxKvSeqlen * kv_real_heads * embeddingSize * 2;
    uint64_t stride_qo = qTokens * embeddingSize;
    uint64_t stride_kv = maxKvSeqlen * embeddingSize;
    flashAttentionDecoder.gmSrck = key;
    flashAttentionDecoder.gmSrcv = value;
    headMaskStride = headMaskStride * maxSeqlen;
    batchMaskStride = batchMaskStride * maxSeqlen;

    flashAttentionDecoder.SetParams(tor, maxKvSeqlen);

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

    int32_t cur_batch = startBatch;
    int64_t cur_bms = batchMaskStride * startBatch;
    for (uint32_t curr_q_blk = startBlk; curr_q_blk < endBlk; curr_q_blk++) {
        int32_t q_seqlen_aligned = tilingData->tilingBatch.qSeqLenAligned[cur_batch];
        int32_t kv_seqlen_aligned = tilingData->tilingBatch.kvSeqLenAligned[cur_batch];
        uint32_t q_seqlen_real = tilingData->tilingBatch.qSeqLen[cur_batch];
        uint32_t kv_seqlen_real = tilingData->tilingBatch.kvSeqLen[cur_batch];
        int32_t pp_m_scalar = tilingData->tilingBatch.qBlock[cur_batch];
        int32_t pp_n_scalar = tilingData->tilingBatch.kvBlock[cur_batch];
        int64_t addr_q_scalar = tilingData->tilingBatch.addrQSeqOffset[cur_batch];
        int64_t addr_o_scalar = tilingData->tilingBatch.addrOSeqOffset[cur_batch];
        int64_t addrKScalar = tilingData->tilingBatch.addrKSeqOffset[cur_batch];
        int64_t addrVScalar = tilingData->tilingBatch.addrVSeqOffset[cur_batch];
        uint32_t cur_total_qblk = tilingData->tilingBatch.curQBlk[cur_batch];
        uint32_t cur_proc_num = tilingData->tilingBatch.batchProc[cur_batch];
        half local_tor = tor;

        uint32_t swa_mode = (maskType == Enums::AttentonMaskType::MASK_TYPE_SWA_NORM
                            && kv_seqlen_real > windowLen) ? 1 : 0;
        uint32_t window_offset = 0;
        if (swa_mode) {
            window_offset = (cacheType == 0) ? (kv_seqlen_real - windowLen) : 0;
            kv_seqlen_real = windowLen;
            kv_seqlen_aligned = (windowLen + BLOCK_SIZE_T - 1) / BLOCK_SIZE_T * BLOCK_SIZE_T;
        }
        uint32_t cur_q_blk_id = curr_q_blk - (cur_total_qblk - cur_proc_num);
        int32_t m_loop = (q_seqlen_aligned + pp_m_scalar - 1) / pp_m_scalar;
        int32_t n_loop = (kv_seqlen_aligned + pp_n_scalar - 1) / pp_n_scalar;

        int32_t start = cur_q_blk_id * n_loop;
        int32_t end = start + n_loop;

        for (int32_t loop_idx = start; loop_idx < end; loop_idx += 2) {
            int32_t head_idx0 = loop_idx / (m_loop * n_loop);
            int32_t m_idx0 = loop_idx % (m_loop * n_loop) / n_loop;
            int32_t n_idx0 = loop_idx % (m_loop * n_loop) % n_loop;
            int64_t q_offset = addr_q_scalar + head_idx0 * stride_qo + m_idx0 * pp_m_scalar * BLOCK_SIZE_T;
            int64_t k_offset = (batchContinuous ? addrKScalar : 0) + head_idx0 / group_num * stride_kv +
                               window_offset * BLOCK_SIZE_T + n_idx0 * pp_n_scalar * BLOCK_SIZE_T;
            int64_t v_offset = (batchContinuous ? addrVScalar : 0) + head_idx0 / group_num * stride_kv +
                               window_offset * BLOCK_SIZE_T + n_idx0 * pp_n_scalar * BLOCK_SIZE_T;
            int64_t o_offset = addr_o_scalar + head_idx0 * stride_qo + m_idx0 * pp_m_scalar * BLOCK_SIZE_T;
            int64_t mask_offset = cur_bms + headMaskStride * head_idx0 + m_idx0 * pp_m_scalar * BLOCK_SIZE_T +
                                  n_idx0 * maskStride * pp_n_scalar;
            int32_t last_n_loop = (n_idx0 == (n_loop - 1) || (n_idx0 + 1) == (n_loop - 1)) ? 1 : 0;
            int32_t warp_o = last_n_loop;
            int32_t init_g = (n_idx0 == 0) ? 1 : 0;
            int32_t m0 = (m_idx0 == (m_loop - 1)) ? (q_seqlen_real - m_idx0 * pp_m_scalar) : pp_m_scalar;
            int32_t n0 = (n_idx0 == (n_loop - 1)) ? (kv_seqlen_real - n_idx0 * pp_n_scalar) : pp_n_scalar;
            int32_t n1 = ((n_idx0 + 1) == (n_loop - 1)) ? (kv_seqlen_real - (n_idx0 + 1) * pp_n_scalar) : pp_n_scalar;
            int32_t k0 = embeddingSize;
            int32_t round_m0 = (m0 + 15) / BLOCK_SIZE_T * BLOCK_SIZE_T;
            int32_t round_n0 = (n0 + 15) / BLOCK_SIZE_T * BLOCK_SIZE_T;
            int32_t round_k0 = (k0 + 15) / BLOCK_SIZE_T * BLOCK_SIZE_T;
            int32_t round_n1 = (n1 + 15) / BLOCK_SIZE_T * BLOCK_SIZE_T;

            if ((n_idx0 + 1) == (n_loop)) {
                n1 = -1;
            }
            flashAttentionDecoder.InitDec(round_m0, round_n0, round_k0, q_offset, k_offset, v_offset, mask_offset,
                                          o_offset, init_g, warp_o, qTokens, maskStride);
            flashAttentionDecoder.FlashAttentionNzDecoderCompute(round_m0, round_n0, round_k0, round_n1, m0, n0,
                                                                 n1, pp_n_scalar, local_tor, scaleType);
        }
        if (cur_q_blk_id == cur_proc_num - 1) {
            cur_batch++;
            cur_bms += batchMaskStride;
        }
    }
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
    PIPE_BARRIER(ALL);
}

template <typename IFAT>
__aicore__ inline void UnpadFlashAttentionDecoder<IFAT>::InitTilingData()
{
    uint32_t tmpBlockIdx = GetBlockIdx();
    
    batchSize = tilingData->tilingBase.batchSize;
    embeddingSize = tilingData->tilingBase.headSize;
    tor = tilingData->tilingBase.scaleValue;
    maskType = tilingData->tilingBase.attenMaskFlag;
    maskStride = BLOCK_SIZE;
    headMaskStride = tilingData->tilingBase.maskHeadStride;
    batchMaskStride = tilingData->tilingBase.maskBatchStride;

    startBlk = tilingData->tilingPerCore.startBlk[tmpBlockIdx];
    endBlk = tilingData->tilingPerCore.endBlk[tmpBlockIdx];
    startBatch = tilingData->tilingPerCore.startBatch[tmpBlockIdx];
    endBatch = tilingData->tilingPerCore.endBatch[tmpBlockIdx];

    qTokens = tilingData->tilingBase.qTokens;
    kvHead = tilingData->tilingBase.kvHeadNum;
    heads = tilingData->tilingBase.qHeadNum;
    maxSeqlen = tilingData->tilingBase.maxSeqlen;
    cacheType = 0;
    maxKvSeqlen = tilingData->tilingBase.maxKvSeqlen;
}

#endif