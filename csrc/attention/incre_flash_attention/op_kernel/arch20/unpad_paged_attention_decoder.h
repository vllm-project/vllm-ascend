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
 * \file unpad_paged_attention_decoder.h
 * \brief
 */

#ifndef UNPAD_PAGED_ATTENTION_DECODER_H
#define UNPAD_PAGED_ATTENTION_DECODER_H

#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "iterator.h"
#include "mma.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../ifa_public_define.h"
#include "gm_to_l1_iterator.h"
#include "gm_to_ub_iterator.h"
#include "l0c_to_ub_iterator.h"
#include "l1_to_bt_iterator.h"
#include "l1_to_fb_iterator.h"
#include "l1_to_l0_iterator.h"
#include "l1_to_ub_iterator.h"

constexpr int32_t LOCAL_STORAGE_BUFFER_SIZE = 4096;

template <CalcMode DECODE_MODE = CalcMode::CALC_MODE_DEFAULT>
class PagedAttentionDecoder {
public:
    __aicore__ inline PagedAttentionDecoder(__gm__ uint8_t *__restrict__ gmSrcQ, __gm__ uint8_t *__restrict__ gmSrcK,
                                            __gm__ uint8_t *__restrict__ gmSrcV, __gm__ uint8_t *__restrict__ gmSrcM,
                                            __gm__ uint8_t *__restrict__ gmDstO, half tor, uint32_t ntokensQ,
                                            uint32_t blockSize, uint32_t maskStride)
        : gmSrcQ(gmSrcQ), gmSrcK(gmSrcK), gmSrcV(gmSrcV), gmSrcM(gmSrcM), gmDstO(gmDstO), tor(tor),
        ntokensQ(ntokensQ), blockSize(blockSize), maskStride(maskStride)
    {
        gmSrcQTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcQ));
        gmSrcKTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcK));
        gmSrcVTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcV));
        gmSrcMTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcM));
        gmDstOTensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmDstO));

        switch (DECODE_MODE) {
            case (CalcMode::CALC_MODE_PREFILL):{
                InitOffsetPrefill();
                break;
            }
            case (CalcMode::CALC_MODE_DEFAULT):{
            }
            default: {
                InitOffsetDefault();
            }
        }

        lsUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lsUbufOffset);
        lpUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lpUbufOffset);
        ls32UbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32UbufOffset);
        loUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbufOffset);
        lmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lmUbufOffset);
        hmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hmUbufOffset);
        gmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gmUbufOffset);
        dmUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dmUbufOffset);
        llUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(llUbufOffset);
        glUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(glUbufOffset);
        tvUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tvUbufOffset);
        goUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbufOffset);
        maskUbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(maskUbufOffset);
    }

    __aicore__ inline void SetMask(int32_t len)
    {
        if (len >= VECTOR_SIZE_I) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - static_cast<int32_t>(MAX_LEN_64_BYTES) > 0 ? len - MAX_LEN_64_BYTES : 0;
        int32_t lowMask = len -static_cast<int32_t>(MAX_LEN_64_BYTES) >= 0 ? MAX_LEN_64_BYTES : len;
        if (len < MAX_LEN_64_BYTES) {
            SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
            return;
        }
        SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
    }

    __aicore__ inline void SetVcgMask(int32_t len)
    {
        if (len > BLOCK_SIZE_I) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        SetVectorMask<int8_t>(maskValue, maskValue);
    }

    __aicore__ inline void ExpandToBlockHalf(AscendC::LocalTensor<half> dstTensor,
                                            AscendC::LocalTensor<half> srcTensor, int32_t len)
    {
        const uint32_t BLOCK_TWO = 2;
        const uint32_t BLOCK_NUM = 8;
        // (len,) -> len / 16 个 (16, 16)
        for (int32_t vaddsIdx = 0; vaddsIdx < BLOCK_TWO; ++vaddsIdx) {
            adds_v<ArchType::ASCEND_V200, half>(
                dstTensor[vaddsIdx * BLOCK_NUM * BLOCK_SIZE_I],
                srcTensor,
                0.0, len / BLOCK_SIZE_I, 1, 0, BLOCK_TWO * BLOCK_NUM, 1);
        }
        PIPE_BARRIER(V);
        // (16, len) -> (len, 16)
        for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE_I); ++vtransIdx) {
            tranpose_v<ArchType::ASCEND_V200, half>(
                dstTensor[vtransIdx * CUBE_MATRIX_SIZE_I],
                dstTensor[vtransIdx * CUBE_MATRIX_SIZE_I]);
        }
        PIPE_BARRIER(V);
    }

    __aicore__ inline void InitOffsetPrefill()
    {
        uint32_t ls32_pre_block_index = 2;
        uint32_t lm_pre_block_index = 4;
        uint32_t tv_pre_block_index = 5;
        uint32_t go_pre_block_index = 6;

        uint32_t hm_pre_line_index = 1;
        uint32_t gm_pre_line_index = 2;
        uint32_t dm_pre_line_index = 3;
        uint32_t ll_pre_line_index = 5;
        uint32_t gl_pre_line_index = 7;

        lsUbufOffset = 0; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        lpUbufOffset = 0; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        ls32UbufOffset = ls32_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        loUbufOffset = ls32_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为两个UB_UINT8_BLOCK_SIZE的偏移量
        lmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        hmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + hm_pre_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        gmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + gm_pre_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        dmUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + dm_pre_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        llUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + ll_pre_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        glUbufOffset = lm_pre_block_index * UB_UINT8_BLOCK_SIZE_I + gl_pre_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为二十五个UB_UINT8_LINE_SIZE的偏移量
        tvUbufOffset = tv_pre_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        goUbufOffset = go_pre_block_index * UB_UINT8_BLOCK_SIZE_I; 
    }

    __aicore__ inline void InitOffsetDefault()
    {
        uint32_t lp_dec_block_index = 2;
        uint32_t ls_dec_block_index = 4;
        uint32_t mask_dec_block_index = 8;
        uint32_t lo_dec_block_index = 10;

        uint32_t lm_block_index = 4;
        uint32_t tv_block_index = 5;
        uint32_t go_block_index = 6;

        uint32_t hm_line_index = 1;
        uint32_t dm_line_index = 2;
        uint32_t ll_line_index = 4;
        uint32_t gm_line_index = 6;
        uint32_t gl_line_index = 16;

        lsUbufOffset = 0; // 初始化为两个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        lpUbufOffset = lp_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE; // 初始化为两个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        ls32UbufOffset = ls_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE; // 初始化为四个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        maskUbufOffset = mask_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE; // 初始化为两个DEC_UB_UINT8_BLOCK_SIZE的偏移量
        loUbufOffset = lo_dec_block_index * DEC_UB_UINT8_BLOCK_SIZE; 
        lmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        hmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + hm_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为一个UB_UINT8_LINE_SIZE的偏移量
        dmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + dm_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        llUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + ll_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为两个UB_UINT8_LINE_SIZE的偏移量
        gmUbufOffset = lm_block_index * UB_UINT8_BLOCK_SIZE_I + gm_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为二十六个UB_UINT8_LINE_SIZE的偏移量
        tvUbufOffset = tv_block_index * UB_UINT8_BLOCK_SIZE_I; // 初始化为十六个UB_UINT8_LINE_SIZE的偏移量
        glUbufOffset = tv_block_index * UB_UINT8_BLOCK_SIZE_I + gl_line_index * UB_UINT8_LINE_SIZE_I; // 初始化为十六个UB_UINT8_LINE_SIZE的偏移量
        goUbufOffset = go_block_index * UB_UINT8_BLOCK_SIZE_I;
    }

    __aicore__ inline void Init(uint64_t srcqOffsetReal, uint64_t srckOffsetReal, uint64_t srcvOffsetReal,
                                uint64_t srckOffsetReal1, uint64_t srcvOffsetReal1, uint64_t srcmOffsetReal,
                                uint64_t dstoOffsetReal, uint32_t initGReal, uint32_t wrapOReal)
    {
        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srckOffset1 = srckOffsetReal1;
        srcvOffset1 = srcvOffsetReal1;
        srcmOffset = srcmOffsetReal;
        dstoOffset = dstoOffsetReal;
        initG = initGReal;
        wrapO = wrapOReal;
    }

public:
    __aicore__ inline void Decode(const uint32_t fm, const uint32_t fn, const uint32_t fk, const uint32_t bn,
                                const uint32_t mActual, const uint32_t n0Actual, const uint32_t n1Actual,
                                const uint32_t maskType, const uint32_t initKVE, const uint32_t headOffset = 0,
                                const uint32_t initKV = 1, half localTor = 1, const uint32_t scaleType = 0);

private:
    const uint32_t PingFlag = 0;
    const uint32_t PongFlag = 1;
    uint32_t vmPingPongFlag = 1;

    __gm__ uint8_t *__restrict__ gmSrcQ;
    __gm__ uint8_t *__restrict__ gmSrcK;
    __gm__ uint8_t *__restrict__ gmSrcV;
    __gm__ uint8_t *__restrict__ gmSrcM;
    __gm__ uint8_t *__restrict__ gmDstO;
    AscendC::GlobalTensor<half> gmSrcQTensor;
    AscendC::GlobalTensor<half> gmSrcKTensor;
    AscendC::GlobalTensor<half> gmSrcVTensor;
    AscendC::GlobalTensor<half> gmSrcMTensor;
    AscendC::GlobalTensor<half> gmDstOTensor;

    AsdopsBuffer<ArchType::ASCEND_V200> buf;

    uint32_t l1qBufAddrOffset = 0;
    uint32_t l1kBufAddrOffset = 2 * UB_UINT8_BLOCK_SIZE_I;
    uint32_t l1pBufAddrOffset = 2 * L1_UINT8_BLOCK_SIZE;
    uint32_t l1vBufAddrOffset = 2 * L1_UINT8_BLOCK_SIZE + 2 * UB_UINT8_BLOCK_SIZE_I;
    uint32_t l1maskBufAddrOffset = 4 * L1_UINT8_BLOCK_SIZE;

    uint32_t l0aBufOffset = 0;
    uint32_t l0bBufOffset = 0;
    uint32_t l0cBufOffset = 0;

    uint32_t lsUbufOffset = 0;
    uint32_t lpUbufOffset = 0;
    uint32_t ls32UbufOffset = 0;
    uint32_t maskUbufOffset = 0;
    uint32_t loUbufOffset = 0;
    uint32_t lmUbufOffset = 0;
    uint32_t hmUbufOffset = 0;
    uint32_t gmUbufOffset = 0;
    uint32_t dmUbufOffset = 0;
    uint32_t llUbufOffset = 0;
    uint32_t glUbufOffset = 0;
    uint32_t tvUbufOffset = 0;
    uint32_t goUbufOffset = 0;

    AscendC::LocalTensor<half> l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1qBufAddrOffset);
    AscendC::LocalTensor<half> l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1kBufAddrOffset);
    AscendC::LocalTensor<half> l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1pBufAddrOffset);
    AscendC::LocalTensor<half> l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1vBufAddrOffset);
    AscendC::LocalTensor<half> l1maskBufAddr_tensor =
        buf.GetBuffer<BufferType::ASCEND_CB, half>(l1maskBufAddrOffset);

    AscendC::LocalTensor<half> l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, half>(l0aBufOffset);
    AscendC::LocalTensor<half> l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, half>(l0bBufOffset);
    AscendC::LocalTensor<float> l0cBufTensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(l0cBufOffset);

    AscendC::LocalTensor<half> lsUbufTensor;
    AscendC::LocalTensor<half> lpUbufTensor;
    AscendC::LocalTensor<half> lmUbufTensor;
    AscendC::LocalTensor<half> hmUbufTensor;
    AscendC::LocalTensor<half> gmUbufTensor;
    AscendC::LocalTensor<half> dmUbufTensor;
    AscendC::LocalTensor<half> tvUbufTensor;
    AscendC::LocalTensor<half> maskUbufTensor;
    AscendC::LocalTensor<float> ls32UbufTensor;
    AscendC::LocalTensor<float> loUbufTensor;
    AscendC::LocalTensor<float> llUbufTensor;
    AscendC::LocalTensor<float> glUbufTensor;
    AscendC::LocalTensor<float> goUbufTensor;

    half tor = 1.0;
    uint32_t ntokensQ = 0;
    uint32_t blockSize = VECTOR_SIZE_I;

    uint64_t srcqOffset = 0;
    uint64_t srckOffset = 0;
    uint64_t srcvOffset = 0;
    uint64_t srckOffset1 = 0;
    uint64_t srcvOffset1 = 0;
    uint64_t dstoOffset = 0;
    uint64_t srcmOffset = 0;
    uint64_t maskStride = 0;

    uint32_t initG = 0;
    uint32_t wrapO = 0;
};


template<>
__aicore__ inline void PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT>::Decode(const uint32_t fm, const uint32_t fn,
                                                                                const uint32_t fk, const uint32_t bn,
                                                                                const uint32_t mActual,
                                                                                const uint32_t n0Actual,
                                                                                const uint32_t n1Actual,
                                                                                const uint32_t maskType,
                                                                                const uint32_t initKVE,
                                                                                const uint32_t headOffset,
                                                                                const uint32_t initKV,
                                                                                half localTor,
                                                                                const uint32_t scaleType)
{
    if (scaleType == 0) {
        localTor = tor;
    }

    AscendC::LocalTensor<half> l1kPingBufTensor =
        l1kBufAddrTensor.ReinterpretCast<uint8_t>()[PingFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1kPongBufTensor =
        l1kBufAddrTensor.ReinterpretCast<uint8_t>()[PongFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1vPingBufTensor =
        l1vBufAddrTensor.ReinterpretCast<uint8_t>()[PingFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1vPongBufTensor =
        l1vBufAddrTensor.ReinterpretCast<uint8_t>()[PongFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1pPingBufTensor =
        l1pBufAddrTensor.ReinterpretCast<uint8_t>()[PingFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();
    AscendC::LocalTensor<half> l1pPongBufTensor =
        l1pBufAddrTensor.ReinterpretCast<uint8_t>()[PongFlag * 4 * L1_UINT8_BLOCK_SIZE].ReinterpretCast<half>();

    uint32_t oSize = fm * fk;
    uint32_t mAlignFloat = (fm + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;
    uint32_t kAlignFloat = (fk + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;
    uint32_t n0AlignFloat = (fn + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;
    uint32_t n1AlignFloat = (bn + FLOAT_VECTOR_SIZE_I - 1) / FLOAT_VECTOR_SIZE_I;
    uint32_t mAlignVector = (fm + VECTOR_SIZE_I - 1) / VECTOR_SIZE_I;
    uint32_t kAlignVector = (fk + VECTOR_SIZE_I - 1) / VECTOR_SIZE_I;
    uint32_t n0AlignVector = (fn + VECTOR_SIZE_I - 1) / VECTOR_SIZE_I;
    uint32_t n1AlignVector = (bn + VECTOR_SIZE_I - 1) / VECTOR_SIZE_I;
    uint32_t initGgDm = (initG == 1) ? 1 : 0;
    uint32_t initGgO = (initG == 1) ? 1 : 0;
    uint32_t initKVG = (initG && initKV) ? 1 : 0; // synchronization for head loop start
    uint32_t initGgMask = (initG == 1) ? 1 : 0;
    uint32_t qOffset = headOffset * FLOAT_VECTOR_SIZE_I * kAlignFloat;
    uint32_t gmUOffset = headOffset * BLOCK_SIZE_I;
    uint32_t glUOffset = gmUOffset;
    uint32_t goUOffset = qOffset;

    uint32_t pSize = fm * fn;
    uint32_t pSizeB = fm * bn;

    // 1. ################ Bmm1 Ping Start #######################
    // 1.1 ################ QK Ping LOAD ################
    if (initGgO != 0) {
        if (initKVG) {
            WAIT_FLAG(MTE1, MTE2, PingFlag);
            WAIT_FLAG(MTE1, MTE2, PongFlag);
        }
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1qBufAddrTensor[qOffset],
            gmSrcQTensor[srcqOffset],
            1, ntokensQ, 1, fk, fk, fk);
        SET_FLAG(MTE2, MTE1, PingFlag);
        if (n1Actual != 0) {
            SET_FLAG(MTE2, MTE1, PongFlag);
        }
    }

    if (maskType != 0) {
        WAIT_FLAG(MTE1, MTE2, PingFlag + 2);
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1maskBufAddr_tensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            gmSrcMTensor[srcmOffset],
            1, maskStride, 1, fn, fn, fn);
        SET_FLAG(MTE2, MTE1, PingFlag + 2);
        WAIT_FLAG(MTE2, MTE1, PingFlag + 2);
        WAIT_FLAG(V, MTE1, PingFlag);
        l1_to_ub<ArchType::ASCEND_V200, half>(
            maskUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            l1maskBufAddr_tensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            1, fn / BLOCK_SIZE_I, 0, 0);
        SET_FLAG(MTE1, V, PingFlag);
        SET_FLAG(MTE1, MTE2, PingFlag + 2);
        if (n1Actual != 0) {
            WAIT_FLAG(MTE1, MTE2, PongFlag + 2);
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1maskBufAddr_tensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
                gmSrcMTensor[srcmOffset + maskStride * blockSize],
                1, maskStride, 1, bn, bn, bn);
            SET_FLAG(MTE2, MTE1, PongFlag + 2);
            WAIT_FLAG(MTE2, MTE1, PongFlag + 2);
            WAIT_FLAG(V, MTE1, PongFlag);
            l1_to_ub<ArchType::ASCEND_V200, half>(
                maskUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                l1maskBufAddr_tensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
                1, bn / BLOCK_SIZE_I, 0, 0);
            SET_FLAG(MTE1, V, PongFlag);
            SET_FLAG(MTE1, MTE2, PongFlag + 2);
        }
    }

    WAIT_FLAG(M, MTE1, PingFlag);
    if (initGgO == 1) {
        WAIT_FLAG(MTE2, MTE1, PingFlag);
    }
    l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l1qBufAddrTensor[qOffset].ReinterpretCast<half>(),
        0, 1, 0, 1, 0, 0);
    SET_FLAG(MTE1, M, PingFlag);
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    WAIT_FLAG(MTE1, MTE2, PingFlag + 4);
#endif
    if (initKV) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1kPingBufTensor,
            gmSrcKTensor[srckOffset],
            fn, blockSize, fn, fk, fk, fk);
    }
    SET_FLAG(MTE2, MTE1, PingFlag);
    WAIT_FLAG(MTE2, MTE1, PingFlag);
    WAIT_FLAG(M, MTE1, PingFlag + 2);
    l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l1kPingBufTensor,
        0, fk * fn / CUBE_MATRIX_SIZE_I, 0, 1, 0, 0);
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    SET_FLAG(MTE1, MTE2, PingFlag + 4);
#endif
    SET_FLAG(MTE1, M, PingFlag + 2);
    // 2. ################ Bmm1 Pong Starts #######################
    // 2.1 ################ QK Pong PRELOAD ################
    if (n1Actual != 0) {
        WAIT_FLAG(M, MTE1, PongFlag);
        if (initGgO == 1) {
            WAIT_FLAG(MTE2, MTE1, PongFlag);
        }
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1qBufAddrTensor[qOffset],
            0, 1, 0, 1, 0, 0);
        SET_FLAG(MTE1, M, PongFlag);
#if __CCE_AICORE__ == 100
        WAIT_FLAG(MTE1, MTE2, PongFlag + 2);
#else
        WAIT_FLAG(MTE1, MTE2, PongFlag + 4);
#endif
        if (initKV) {
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1kPongBufTensor,
                gmSrcKTensor[srckOffset1],
                bn, blockSize, bn, fk, fk, fk);
        }
        SET_FLAG(MTE2, MTE1, PongFlag);
        WAIT_FLAG(MTE2, MTE1, PongFlag);
        WAIT_FLAG(M, MTE1, PongFlag + 2);
        l1_to_l0_b<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1kPongBufTensor,
            0, fk * bn / CUBE_MATRIX_SIZE_I, 0, 1, 0, 0);
        SET_FLAG(MTE1, M, PongFlag + 2);
#if __CCE_AICORE__ == 100
        SET_FLAG(MTE1, MTE2, PongFlag + 2);
#else
        SET_FLAG(MTE1, MTE2, PongFlag + 4);
#endif
    }
    // 1.2 ################ Bmm1 Ping + V PRELOAD ################
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    WAIT_FLAG(MTE1, MTE2, PingFlag + 6);
#endif
    if (initKV) {
        gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
            l1vPingBufTensor,
            gmSrcVTensor[srcvOffset],
            fn, blockSize, fn, fk, fk, fk);
    }
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE2, MTE1, PingFlag + 2);
#else
    SET_FLAG(MTE2, MTE1, PingFlag + 4);
#endif
    WAIT_FLAG(MTE1, M, PingFlag + 2);
    WAIT_FLAG(MTE1, M, PingFlag);
    WAIT_FLAG(V, M, PingFlag);

    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        mActual, n0Actual, fk, 1);
    
    SET_FLAG(M, V, PingFlag);
    SET_FLAG(M, MTE1, PingFlag);
    SET_FLAG(M, MTE1, PingFlag + 2);
#if __CCE_AICORE__ == 100
    WAIT_FLAG(MTE2, MTE1, PingFlag + 2);
#else
    WAIT_FLAG(MTE2, MTE1, PingFlag + 4);
#endif
    WAIT_FLAG(M, MTE1, PingFlag + 2);
    if (fk == 16) {
        l1_to_l0_b<ArchType::ASCEND_V200, half, 1, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
            l1vPingBufTensor,
            0, fn / BLOCK_SIZE_I, 0, 1, 0, 0);
    } else {
        for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (fn / BLOCK_SIZE_I); ++l0bLoadIdx) {
            l1_to_l0_b<ArchType::ASCEND_V200, half, 1, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I + l0bLoadIdx * fk * BLOCK_SIZE_I],
                l1vPingBufTensor[l0bLoadIdx * CUBE_MATRIX_SIZE_I],
                0, fk / BLOCK_SIZE_I, 0, fn / BLOCK_SIZE_I, 0, 0);
        }
    }
#if __CCE_AICORE__ == 100
    SET_FLAG(MTE1, MTE2, PingFlag + 2);
#else
    SET_FLAG(MTE1, MTE2, PingFlag + 6);
#endif
    SET_FLAG(MTE1, M, PingFlag + 2);
    // 1. ################ Bmm1 Ping Ends #######################
    // 2.2 ################ Bmm1 Pong + V PRELOAD ################
    if (n1Actual != 0) {
#if __CCE_AICORE__ == 100
        WAIT_FLAG(MTE1, MTE2, PongFlag + 2);
#else
        WAIT_FLAG(MTE1, MTE2, PongFlag + 6);
#endif
        if (initKV) {
            gm_to_l1<ArchType::ASCEND_V200, half, DataFormatT::NZ, DataFormatT::NZ>(
                l1vPongBufTensor,
                gmSrcVTensor[srcvOffset1],
                bn, blockSize, bn, fk, fk, fk);
        }
#if __CCE_AICORE__ == 100
        SET_FLAG(MTE2, MTE1, PongFlag + 2);
#else
        SET_FLAG(MTE2, MTE1, PongFlag + 4);
#endif
        WAIT_FLAG(MTE1, M, PongFlag + 2);
        WAIT_FLAG(MTE1, M, PongFlag);
        WAIT_FLAG(V, M, PongFlag);

        mmad<ArchType::ASCEND_V200, half, half, float, false>(
            l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            mActual, n1Actual, fk, 1);
        
        SET_FLAG(M, V, PongFlag);
        SET_FLAG(M, MTE1, PongFlag);
        SET_FLAG(M, MTE1, PongFlag + 2);
#if __CCE_AICORE__ == 100
        WAIT_FLAG(MTE2, MTE1, PongFlag + 2);
#else
        WAIT_FLAG(MTE2, MTE1, PongFlag + 4);
#endif
        WAIT_FLAG(M, MTE1, PongFlag + 2);
        if (fk == 16) {
            l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
                l1vPongBufTensor,
                0, bn / BLOCK_SIZE_I, 0, 1, 0, 0);
        } else {
            for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (bn / BLOCK_SIZE_I); ++l0bLoadIdx) { // Nz -> nZ
                l1_to_l0_b<ArchType::ASCEND_V200, half, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                    l0bBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I + l0bLoadIdx * fk * BLOCK_SIZE_I],
                    l1vPongBufTensor[l0bLoadIdx * CUBE_MATRIX_SIZE_I],
                    0, fk / BLOCK_SIZE_I, 0, bn / BLOCK_SIZE_I, 0, 0);
            }
        }
#if __CCE_AICORE__ == 100
        SET_FLAG(MTE1, MTE2, PongFlag + 2);
#else
        SET_FLAG(MTE1, MTE2, PongFlag + 6);
#endif
        SET_FLAG(MTE1, M, PongFlag + 2);
    }
    // 2. ################ Bmm1 Pong Ends #######################
    // 3. ################ Softmax Ping Starts #######################
    WAIT_FLAG(M, V, PingFlag);
    l0c_to_ub<ArchType::ASCEND_V200, float, half>(
        lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        1, pSize / CUBE_MATRIX_SIZE_I, 0, 0);
    PIPE_BARRIER(V);
    SET_FLAG(V, M, PingFlag);
    muls_v<ArchType::ASCEND_V200, half>(
        lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        localTor, n0AlignVector, 1, fm, 8, fm * 8);
    PIPE_BARRIER(V);
    
    if (maskType != 0) {
        WAIT_FLAG(MTE1, V, PingFlag);
        add_v<ArchType::ASCEND_V200, half>(lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           maskUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                           n0Actual / VECTOR_SIZE_I, 
                                           1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        if (n0Actual % VECTOR_SIZE_I != 0) {
            SetMask(n0Actual % VECTOR_SIZE_I);
            add_v<ArchType::ASCEND_V200, half>(lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + (n0Actual / VECTOR_SIZE_I) * VECTOR_SIZE_I],
                                               lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + (n0Actual / VECTOR_SIZE_I) * VECTOR_SIZE_I],
                                               maskUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + (n0Actual / VECTOR_SIZE_I) * VECTOR_SIZE_I],
                                               1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        }
        SET_FLAG(V, MTE1, PingFlag);
    }

    if (n0Actual <= VECTOR_SIZE_I) {
        if (n0Actual != VECTOR_SIZE_I) {
            SetMask(n0Actual % VECTOR_SIZE_I);
        }
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbufTensor,
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            1, 1, 1, 8);
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, half>(
            tvUbufTensor,
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            0, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        if (n0Actual % VECTOR_SIZE_I != 0) {
            SetMask(n0Actual % VECTOR_SIZE_I);
        }
        max_v<ArchType::ASCEND_V200, half>(
            tvUbufTensor,
            tvUbufTensor,
            lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + VECTOR_SIZE_I],
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
            lmUbufTensor,
            tvUbufTensor,
            1, 1, 1, 8);
        PIPE_BARRIER(V);
    }
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    PIPE_BARRIER(V);

    if (initGgDm == 0) {
        max_v<ArchType::ASCEND_V200, half>(
            hmUbufTensor,
            lmUbufTensor,
            gmUbufTensor[gmUOffset],
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(
            dmUbufTensor[PingFlag * UB_HALF_LINE_SIZE_I],
            gmUbufTensor[gmUOffset],
            hmUbufTensor,
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbufTensor[gmUOffset],
            hmUbufTensor,
            0, 1, 1, 0, 0);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbufTensor, hmUbufTensor, fm);
    } else {
        initGgDm = 0;
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbufTensor[gmUOffset],
            lmUbufTensor,
            0, 1, 1, 0, 0);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbufTensor, gmUbufTensor[gmUOffset], fm);
    }
    sub_v<ArchType::ASCEND_V200, half>(lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                       lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                                       tvUbufTensor,
                                       n0AlignVector, 1, 1, 0, 8, 8, 0);
    PIPE_BARRIER(V);
    conv_v<ArchType::ASCEND_V200, half, float>(
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        lsUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        n0AlignFloat, 1, 1, 8, 4);
    PIPE_BARRIER(V);
    exp_v<ArchType::ASCEND_V200, float>(
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        n0AlignFloat, 1, 1, 8, 8);
    PIPE_BARRIER(V);
    WAIT_FLAG(MTE3, V, PingFlag);
    conv_v<ArchType::ASCEND_V200, float, half>(
        lpUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        n0AlignFloat, 1, 1, 4, 8);
    PIPE_BARRIER(V);
    SET_FLAG(V, MTE3, PingFlag);
    SetMaskNorm();

    if (n0Actual < FLOAT_VECTOR_SIZE_I) {
        if (n0Actual != FLOAT_VECTOR_SIZE_I) {
            SetVectorMask<int8_t>(0x0, ((long)1 << n0Actual) - 1);
        }
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            1,
            1,
            1,
            2);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    } else {
        for (int64_t vcalcIdx = 1; vcalcIdx < n0Actual / FLOAT_VECTOR_SIZE_I; vcalcIdx++) {
            add_v<ArchType::ASCEND_V200, float>(
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + vcalcIdx * FLOAT_VECTOR_SIZE_I],
                1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
        }
        if (n0Actual % FLOAT_VECTOR_SIZE_I != 0) {
            SetMask(n0Actual % FLOAT_VECTOR_SIZE_I);
            add_v<ArchType::ASCEND_V200, float>(
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
                ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE + (n0Actual / FLOAT_VECTOR_SIZE_I) * FLOAT_VECTOR_SIZE_I],
                1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        }
        cadd_v<ArchType::ASCEND_V200, float>(
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            ls32UbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            1, 1, 1, 2);
    }

    PIPE_BARRIER(V);
    SetMaskNorm();
    SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
    WAIT_FLAG(MTE1, MTE3, PingFlag);
    WAIT_FLAG(V, MTE3, PingFlag);
    ub_to_l1<ArchType::ASCEND_V200, half>(
        l1pPingBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        lpUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
        1, fn / BLOCK_SIZE_I, 0, 0);
    SET_FLAG(MTE3, V, PingFlag);
    SET_FLAG(MTE3, MTE1, PingFlag);
    // 3. ################ Softmax Ping Ends #######################
    // 4. ################ Softmax Pong Starts #######################
    if (n1Actual != 0) {
        WAIT_FLAG(M, V, PongFlag);
        l0c_to_ub<ArchType::ASCEND_V200, float, half>(
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            1, pSize / CUBE_MATRIX_SIZE_I, 0, 0);
        PIPE_BARRIER(V);
        SET_FLAG(V, M, PongFlag);

        muls_v<ArchType::ASCEND_V200, half>(
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            localTor,
            n1AlignVector, 1, fm, 8, fm * 8);
        PIPE_BARRIER(V);
        if (maskType != 0) {
            WAIT_FLAG(MTE1, V, PongFlag);
            add_v<ArchType::ASCEND_V200, half>(
                lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                maskUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                n1Actual / VECTOR_SIZE_I, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            if (n1Actual % VECTOR_SIZE_I != 0) {
                SetMask(n1Actual % VECTOR_SIZE_I);
                add_v<ArchType::ASCEND_V200, half>(
                    lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + n1Actual / VECTOR_SIZE_I * VECTOR_SIZE_I],
                    lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + n1Actual / VECTOR_SIZE_I * VECTOR_SIZE_I],
                    maskUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + n1Actual / VECTOR_SIZE_I * VECTOR_SIZE_I],
                    1, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);
                SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
            }
            SET_FLAG(V, MTE1, PongFlag);
        }

        if (n1Actual <= VECTOR_SIZE_I) {
            if (n1Actual != VECTOR_SIZE_I) {
                SetMask(n1Actual % VECTOR_SIZE_I);
            }
            cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
                lmUbufTensor,
                lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                1, 1, 1, 8);
            PIPE_BARRIER(V);
        } else {
            ub_to_ub<ArchType::ASCEND_V200, half>(
                tvUbufTensor,
                lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                0, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            if (n1Actual % VECTOR_SIZE_I != 0) {
                SetMask(n1Actual % VECTOR_SIZE_I);
            }
            max_v<ArchType::ASCEND_V200, half>(
                tvUbufTensor,
                tvUbufTensor,
                lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + VECTOR_SIZE_I],
                1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
            cmax_v<ArchType::ASCEND_V200, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(
                lmUbufTensor,
                tvUbufTensor,
                1, 1, 1, 8);
            PIPE_BARRIER(V);
        }
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        
        PIPE_BARRIER(V);
        max_v<ArchType::ASCEND_V200, half>(
            hmUbufTensor,
            lmUbufTensor,
            gmUbufTensor[gmUOffset],
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(
            dmUbufTensor[PongFlag * UB_HALF_LINE_SIZE_I],
            gmUbufTensor[gmUOffset],
            hmUbufTensor,
            1, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbufTensor, hmUbufTensor, fm);
        ub_to_ub<ArchType::ASCEND_V200, half>(
            gmUbufTensor[gmUOffset],
            hmUbufTensor,
            0, 1, 1, 0, 0);
        PIPE_BARRIER(V);
        sub_v<ArchType::ASCEND_V200, half>(
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            tvUbufTensor,
            n1AlignVector, 1, 1, 0, 8, 8, 0);
        PIPE_BARRIER(V);
        conv_v<ArchType::ASCEND_V200, half, float>(
            ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            lsUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            n1AlignFloat, 1, 1, 8, 4);
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            n1AlignFloat, 1, 1, 8, 8);
        PIPE_BARRIER(V);
        WAIT_FLAG(MTE3, V, PongFlag);
        conv_v<ArchType::ASCEND_V200, float, half>(
            lpUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            n1AlignFloat, 1, 1, 4, 8);
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, PongFlag);
        SetMaskNorm();

        if (n1Actual < FLOAT_VECTOR_SIZE_I) {
            if (n1Actual != FLOAT_VECTOR_SIZE_I) {
                SetVectorMask<int8_t>(0x0, ((long)1 << n1Actual) - 1);
            }
            cadd_v<ArchType::ASCEND_V200, float>(
                llUbufTensor[PongFlag * UB_FLOAT_LINE_SIZE_I],
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                1, 1, 1, 2);
            SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        } else {
            for (int64_t vcalcIdx = 1; vcalcIdx < n1Actual / FLOAT_VECTOR_SIZE_I; vcalcIdx++) {
                add_v<ArchType::ASCEND_V200, float>(
                    ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + vcalcIdx * FLOAT_VECTOR_SIZE_I],
                    1, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);
            }
            if (n1Actual % FLOAT_VECTOR_SIZE_I != 0) {
                SetVectorMask<int8_t>(0x0, ((long)1 << (n1Actual % FLOAT_VECTOR_SIZE_I)) - 1);
                add_v<ArchType::ASCEND_V200, float>(
                    ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                    ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE + (n1Actual / FLOAT_VECTOR_SIZE_I) * FLOAT_VECTOR_SIZE_I],
                    1, 1, 1, 1, 8, 8, 8);
                PIPE_BARRIER(V);
                SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
            }
            cadd_v<ArchType::ASCEND_V200, float>(
                llUbufTensor[PongFlag * UB_FLOAT_LINE_SIZE_I],
                ls32UbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
                1, 1, 1, 2);
        }

        PIPE_BARRIER(V);
        SetMaskNorm();
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        WAIT_FLAG(MTE1, MTE3, PongFlag);
        WAIT_FLAG(V, MTE3, PongFlag);
        ub_to_l1<ArchType::ASCEND_V200, half>(
            l1pPongBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            lpUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            1, bn / BLOCK_SIZE_I, 0, 0);
        SET_FLAG(MTE3, V, PongFlag);
        SET_FLAG(MTE3, MTE1, PongFlag);
    }
    // 4. ################ Softmax Pong Ends #######################
    // 5. ################ Bmm2 Ping Starts #######################
    WAIT_FLAG(MTE3, MTE1, PingFlag);
    WAIT_FLAG(M, MTE1, PingFlag);
    l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
        l0aBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l1pPingBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        0, 1, 0, 1, 0, 0);
    SET_FLAG(MTE1, MTE3, PingFlag);
    SET_FLAG(MTE1, M, PingFlag);
    WAIT_FLAG(MTE1, M, PingFlag);
    WAIT_FLAG(MTE1, M, PingFlag + 2);
    WAIT_FLAG(V, M, PingFlag);

    mmad<ArchType::ASCEND_V200, half, half, float, false>(
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0aBufTensor.ReinterpretCast<half>()[PingFlag * L0AB_HALF_BUF_SIZE_I],
        l0bBufTensor.ReinterpretCast<half>()[PingFlag * L0AB_HALF_BUF_SIZE_I],
        mActual, fk, n0Actual, 1);
    
    SET_FLAG(M, MTE1, PingFlag);
    SET_FLAG(M, MTE1, PingFlag + 2);
    SET_FLAG(M, V, PingFlag);
    if (initKVE) {
        SET_FLAG(MTE1, MTE2, PingFlag);
        if (n1Actual == 0) {
            SET_FLAG(MTE1, MTE2, PongFlag);
        }
    }
    // 5. ################ Bmm2 Ping Ends #######################
    // 6. ################ Bmm2 Pong Starts #######################
    if (n1Actual != 0) {
        WAIT_FLAG(MTE3, MTE1, PongFlag);
        WAIT_FLAG(M, MTE1, PongFlag);
        l1_to_l0_a<ArchType::ASCEND_V200, half, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
            l0aBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l1pPongBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            0, 1, 0, 1, 0, 0);
        SET_FLAG(MTE1, MTE3, PongFlag);
        SET_FLAG(MTE1, M, PongFlag);
        WAIT_FLAG(MTE1, M, PongFlag);
        WAIT_FLAG(MTE1, M, PongFlag + 2);
        WAIT_FLAG(V, M, PongFlag);

        mmad<ArchType::ASCEND_V200, half, half, float, false>(
            l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l0aBufTensor.ReinterpretCast<half>()[PongFlag * L0AB_HALF_BUF_SIZE_I],
            l0bBufTensor.ReinterpretCast<half>()[PongFlag * L0AB_HALF_BUF_SIZE_I],
            mActual, fk, n1Actual, 1);
        
        SET_FLAG(M, MTE1, PongFlag);
        SET_FLAG(M, MTE1, PongFlag + 2);
        SET_FLAG(M, V, PongFlag);
        if (initKVE) {
            SET_FLAG(MTE1, MTE2, PongFlag);
        }
    }
    // 6. ################ Bmm2 Pong Ends #######################
    // 7. ################ Update Ping Starts #######################
    WAIT_FLAG(M, V, PingFlag);
    l0c_to_ub<ArchType::ASCEND_V200, float, float, false>(
        loUbufTensor,
        l0cBufTensor[PingFlag * L0AB_HALF_BUF_SIZE_I],
        fk / BLOCK_SIZE_I, 1, 0, 0);
    PIPE_BARRIER(V);
    // 8. ################ Update Pong Starts #######################
    if (n1Actual != 0) {
        WAIT_FLAG(M, V, PongFlag);
        l0c_to_ub<ArchType::ASCEND_V200, float, float, false>(
            loUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            l0cBufTensor[PongFlag * L0AB_HALF_BUF_SIZE_I],
            fk / BLOCK_SIZE_I, 1, 0, 0);
        PIPE_BARRIER(V);
    }
    if (initGgO == 0) {
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            dmUbufTensor[PingFlag * UB_HALF_LINE_SIZE_I],
            mAlignFloat, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            tvUbufTensor.ReinterpretCast<float>(),
            mAlignFloat, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0x0, ((long)1 << (16)) - 1);
        mul_v<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            tvUbufTensor.ReinterpretCast<float>(),
            glUbufTensor[glUOffset],
            mAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            glUbufTensor[glUOffset],
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            mAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        ExpandToBlockHalf(tvUbufTensor, dmUbufTensor[PingFlag * UB_HALF_LINE_SIZE_I], fm);

        adds_v<ArchType::ASCEND_V200, half>(
            tvUbufTensor[BLOCK_SIZE_I],
            tvUbufTensor,
            0.0, kAlignFloat, 1, 0, 8, 0);
        PIPE_BARRIER(V);
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbufTensor.ReinterpretCast<float>()[fk],
            tvUbufTensor,
            kAlignFloat, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbufTensor.ReinterpretCast<float>()[fk],
            tvUbufTensor.ReinterpretCast<float>()[fk],
            kAlignFloat, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        if (vmPingPongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingPongFlag = 0;
        }
        mul_v<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            goUbufTensor[goUOffset],
            tvUbufTensor.ReinterpretCast<float>()[fk],
            kAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            goUbufTensor[goUOffset],
            loUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            kAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
    } else {
        ub_to_ub<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            llUbufTensor[PingFlag * UB_FLOAT_LINE_SIZE_I],
            0, 1, 1, 0, 0);
        PIPE_BARRIER(V);
        if (vmPingPongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingPongFlag = 0;
        }
        ub_to_ub<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            loUbufTensor[PingFlag * LOCAL_STORAGE_BUFFER_SIZE],
            0, 1, fk / 8, 0, 0);
        PIPE_BARRIER(V);
    }
    PIPE_BARRIER(V);
    initGgO = 0;

    // 7. ################ Update Ping Ends #######################
    if (n1Actual != 0) {
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            dmUbufTensor[PongFlag * UB_HALF_LINE_SIZE_I],
            mAlignFloat, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbufTensor.ReinterpretCast<float>(),
            tvUbufTensor.ReinterpretCast<float>(),
            mAlignFloat, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0x0, ((long)1 << (16)) - 1);
        mul_v<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            tvUbufTensor.ReinterpretCast<float>(),
            glUbufTensor[glUOffset],
            mAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            glUbufTensor[glUOffset],
            glUbufTensor[glUOffset],
            llUbufTensor[PongFlag * UB_FLOAT_LINE_SIZE_I],
            mAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        ExpandToBlockHalf(tvUbufTensor, dmUbufTensor[PongFlag * UB_HALF_LINE_SIZE_I], fm);

        adds_v<ArchType::ASCEND_V200, half>(
            tvUbufTensor[BLOCK_SIZE_I],
            tvUbufTensor,
            0.0, kAlignFloat, 1, 0, 8, 0);
        PIPE_BARRIER(V);
        conv_v<ArchType::ASCEND_V200, half, float>(
            tvUbufTensor.ReinterpretCast<float>()[fk],
            tvUbufTensor,
            kAlignFloat, 1, 1, uint16_t(8), uint16_t(4));
        PIPE_BARRIER(V);
        exp_v<ArchType::ASCEND_V200, float>(
            tvUbufTensor.ReinterpretCast<float>()[fk],
            tvUbufTensor.ReinterpretCast<float>()[fk],
            kAlignFloat, 1, 1, uint16_t(8), uint16_t(8));
        PIPE_BARRIER(V);

        if (vmPingPongFlag == 1) {
            WAIT_FLAG(MTE3, V, EVENT_ID2);
            vmPingPongFlag = 0;
        }

        mul_v<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            goUbufTensor[goUOffset],
            tvUbufTensor.ReinterpretCast<float>()[fk],
            kAlignFloat, 1, 1, 1, 8, 8, 8);

        PIPE_BARRIER(V);
        add_v<ArchType::ASCEND_V200, float>(
            goUbufTensor[goUOffset],
            goUbufTensor[goUOffset],
            loUbufTensor[PongFlag * LOCAL_STORAGE_BUFFER_SIZE],
            kAlignFloat, 1, 1, 1, 8, 8, 8);
        PIPE_BARRIER(V);
        SET_FLAG(V, M, PongFlag);
    }
    SET_FLAG(V, M, PingFlag);
    // 8. ################ Update Pong Ends #######################
    // 9. ################ Line Output Starts #####################
    if (wrapO == 1) {
        SetVectorMask<int8_t>(0x0, ((long)1 << (16)) - 1);
        conv_v<ArchType::ASCEND_V200, float, half>(
            glUbufTensor[glUOffset].ReinterpretCast<half>(),
            glUbufTensor[glUOffset],
            mAlignFloat, 1, 1, uint16_t(4), uint16_t(8));
        PIPE_BARRIER(V);
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        conv_v<ArchType::ASCEND_V200, float, half>(
            goUbufTensor[goUOffset].ReinterpretCast<half>(),
            goUbufTensor[goUOffset],
            kAlignFloat, 1, 1, uint16_t(4), uint16_t(8));
        PIPE_BARRIER(V);
        ExpandToBlockHalf(tvUbufTensor, glUbufTensor[glUOffset].ReinterpretCast<half>(), fm);

        SetVectorMask<int8_t>(0x0, ((long)1 << (16)) - 1);
        for (int32_t vdivIdx = 0; vdivIdx < (fk / BLOCK_SIZE_I); ++vdivIdx) { // Oi / li
            div_v<ArchType::ASCEND_V200, half>(
                goUbufTensor[goUOffset].ReinterpretCast<half>()[vdivIdx * BLOCK_SIZE_I],
                goUbufTensor[goUOffset].ReinterpretCast<half>()[vdivIdx * BLOCK_SIZE_I],
                tvUbufTensor,
                1, 1, 1, 1, 8, 8, 8);
            PIPE_BARRIER(V);
        }
        SetVectorMask<int8_t>(0xffffffffffffffff, 0xffffffffffffffff);
        PIPE_BARRIER(V);
        SET_FLAG(V, MTE3, EVENT_ID2);
        WAIT_FLAG(V, MTE3, EVENT_ID2);
        ub_to_gm<ArchType::ASCEND_V200, half>(
            gmDstOTensor[(int64_t)dstoOffset],
            goUbufTensor[goUOffset].ReinterpretCast<half>(),
            0, fk / BLOCK_SIZE_I, mActual, 0, ntokensQ - mActual);

        if (vmPingPongFlag == 0) {
            SET_FLAG(MTE3, V, EVENT_ID2);
            vmPingPongFlag = 1;
        }
    }
}

template <typename IFAT>
class PagedAttentionDecoderMask {
public:
    __aicore__ inline PagedAttentionDecoderMask(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable,  __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace, const typename IFAT::TilingType *__restrict tiling);
    __aicore__ inline void Process();

    using T = float;
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;

protected:
    const typename IFAT::TilingType *__restrict tilingData = nullptr;

    GlobalTensor<int64_t> contextLensGmTensor;
    GlobalTensor<int32_t> blockTablesGmTensor;
    uint32_t numTokens = 0;
    uint32_t numTokensPad = 0;
    uint32_t numHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t maxNumBlocksPerQuery = 0;
    half tor = 0;
    uint32_t kvHeads = 0;
    uint32_t groupNum = 0;
    uint32_t scaleType = 0;
    uint32_t headSplit = 0;
    uint32_t maskType = 0;
    uint32_t maskStride = 0;
    uint32_t runMode = 0;
    int64_t headMaskStride = 0;
    int64_t batchMaskStride = 0;

    uint32_t startBlk = 0;
    uint32_t endBlk = 0;
    uint32_t startBatch = 0;
    uint32_t endBatch = 0;
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
    __aicore__ inline void RunDefault(uint32_t startBlk, uint32_t endBlk, uint32_t numHeads, uint32_t startBatch,
                                        uint32_t endBatch, uint32_t kvHeads, uint64_t batchMaskStride, uint64_t headMaskStride,
                                        uint64_t strideKV, uint32_t groupNum, uint32_t blockSize, uint64_t strideQO,
                                        uint32_t maxNumBlocksPerQuery, uint32_t embeddingSize, uint32_t maskStride,
                                        uint32_t maskType, uint32_t headSplit,
                                        AscendC::GlobalTensor<int64_t> contextLensGmTensor,
                                        AscendC::GlobalTensor<int32_t> blockTablesGmTensor,
                                        PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT> &pa,
                                        PagedAttentionDecoder<CalcMode::CALC_MODE_PREFILL> &paPrefill,
                                        uint32_t scaleType, half tor);
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
    #if __CCE_AICORE__ == 100
    #else
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(MTE1, MTE2, EVENT_ID6);
        SET_FLAG(MTE1, MTE2, EVENT_ID7);
    #endif
    }

    __aicore__ inline void SyncEnd()
    {
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    #if __CCE_AICORE__ == 100
    #else
        WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
    #endif
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
};


template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                                                __gm__ uint8_t *blockTable, __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                                                const typename IFAT::TilingType *__restrict tiling)
{
    contextLensGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(actualSeqLengths));
    blockTablesGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
    ListTensorDesc keyListTensorDesc((__gm__ void *)key);
    ListTensorDesc valueListTensorDesc((__gm__ void *)value);
    __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    this->tilingData = tiling;
    this->query = query;
    this->key = key_;
    this->value = value_;
    this->attenMask = attenMask;
    this->attentionOut = attentionOut;

    InitTilingData();
}

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::Process()
{
    SetMaskNorm();
    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    SetLoadDataPaddingValue<uint64_t>(uint16_t(0));
    SetAtomicNone();

    SET_FLAG(MTE2, S, EVENT_ID0);
    WAIT_FLAG(MTE2, S, EVENT_ID0);

    uint64_t strideQO = numTokensPad * embeddingSize;
    uint64_t strideKV = blockSize * embeddingSize;

    PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT> pa(query, key, value, attenMask, attentionOut, tor, numTokensPad, blockSize, maskStride);
    PagedAttentionDecoder<CalcMode::CALC_MODE_PREFILL> paPrefill(query, key, value, attenMask, attentionOut, tor, numTokensPad, blockSize, maskStride);

    SET_FLAG(S, MTE2, EVENT_ID0);
    WAIT_FLAG(S, MTE2, EVENT_ID0);

    SyncStart();
    RunDefault(startBlk, endBlk, numHeads, startBatch, endBatch, kvHeads,
                batchMaskStride, headMaskStride, strideKV, groupNum, blockSize,
                strideQO, maxNumBlocksPerQuery, embeddingSize, maskStride, maskType, headSplit,
                contextLensGmTensor, blockTablesGmTensor, pa, paPrefill, scaleType, tor);
    SyncEnd();
}

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::InitTilingData()
{
    uint32_t tmp_block_idx = GetBlockIdx();
    startBlk = tilingData->tilingPerCore.startBlk[tmp_block_idx];
    endBlk = tilingData->tilingPerCore.endBlk[tmp_block_idx];
    startBatch = tilingData->tilingPerCore.startBatch[tmp_block_idx];
    endBatch = tilingData->tilingPerCore.endBatch[tmp_block_idx];
    numTokens = tilingData->tilingBase.batchSize;
    numHeads = tilingData->tilingBase.qHeadNum;
    embeddingSize = tilingData->tilingBase.headSize;
    numBlocks = tilingData->tilingBase.totalBlockNum;
    blockSize = tilingData->tilingBase.blockSize;
    maxNumBlocksPerQuery = tilingData->tilingBase.maxBlockNumPerBatch;
    tor = tilingData->tilingBase.scaleValue;
    kvHeads = tilingData->tilingBase.kvHeadNum;
    headSplit = tilingData->tilingPerCore.headSplit;
    maskType = tilingData->tilingBase.attenMaskFlag;
    headMaskStride = tilingData->tilingPerCore.maskHeadStride;
    batchMaskStride = tilingData->tilingPerCore.maskBatchStride;
    numTokensPad = 1;
    groupNum = numHeads / kvHeads;
    maskStride = BLOCK_SIZE_I;
    runMode = 0;
}

template <typename IFAT>
__aicore__ inline void PagedAttentionDecoderMask<IFAT>::RunDefault(uint32_t startBlk, uint32_t endBlk, uint32_t numHeads, uint32_t startBatch,
                                                                   uint32_t endBatch, uint32_t kvHeads, uint64_t batchMaskStride, uint64_t headMaskStride,
                                                                   uint64_t strideKV, uint32_t groupNum, uint32_t blockSize, uint64_t strideQO,
                                                                   uint32_t maxNumBlocksPerQuery, uint32_t embeddingSize, uint32_t maskStride,
                                                                   uint32_t maskType, uint32_t headSplit,
                                                                   AscendC::GlobalTensor<int64_t> contextLensGmTensor,
                                                                   AscendC::GlobalTensor<int32_t> blockTablesGmTensor,
                                                                   PagedAttentionDecoder<CalcMode::CALC_MODE_DEFAULT> &pa,
                                                                   PagedAttentionDecoder<CalcMode::CALC_MODE_PREFILL> &paPrefill,
                                                                   uint32_t scaleType, half tor)
{
    const uint32_t contextLenUbOffset = 4 * UB_UINT8_BLOCK_SIZE_I + 10 * UB_UINT8_LINE_SIZE_I;
    const uint32_t blockTablesUbOffset = 4 * UB_UINT8_BLOCK_SIZE_I + 12 * UB_UINT8_LINE_SIZE_I;
    const uint32_t contextLenUbLen = 2 * UB_UINT8_LINE_SIZE_I;
    const uint32_t blockTablesUbLen = 20 * UB_UINT8_LINE_SIZE_I;
    AscendC::LocalTensor<int64_t> contextLenUbTensor;
    AscendC::LocalTensor<int32_t> blockTablesUbTensor;
    contextLenUbTensor.InitBuffer((const uint32_t)contextLenUbOffset, (const uint32_t)contextLenUbLen);
    blockTablesUbTensor.InitBuffer((const uint32_t)blockTablesUbOffset, (const uint32_t)blockTablesUbLen);
    gm_to_ub<ArchType::ASCEND_V200, int64_t>(
        contextLenUbTensor,
        contextLensGmTensor[startBatch],
        0, 1, Align<uint32_t>((endBatch - startBatch + 1) * 4, BUFFER_SIZE_BYTE_32B) / BUFFER_SIZE_BYTE_32B + 1, 0, 0);
    for (uint32_t taskId = startBlk; taskId < endBlk; taskId++) {
        uint32_t curBatch = taskId / numHeads;
        uint32_t headId = taskId % numHeads;
        uint32_t repeatLen = min(headSplit, min(endBlk - taskId, groupNum - headId % groupNum));
        gm_to_ub<ArchType::ASCEND_V200, int32_t>(
            blockTablesUbTensor,
            blockTablesGmTensor[curBatch * maxNumBlocksPerQuery],
            0, 1, Align<uint32_t>(maxNumBlocksPerQuery * 4, BUFFER_SIZE_BYTE_32B) / BUFFER_SIZE_BYTE_32B + 1, 0, 0);
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        uint32_t contextLen = (uint32_t)(*((__ubuf__ int64_t *)contextLenUbTensor.GetPhyAddr() + curBatch - startBatch));
        uint32_t nLoop = (contextLen + blockSize - 1) / blockSize;
        uint32_t tail = contextLen % blockSize == 0 ? blockSize : contextLen % blockSize;
        uint32_t mActual = 1;
        uint32_t roundM = Align<uint32_t>(mActual, 16);
        uint32_t __k = embeddingSize;
        uint32_t roundK = Align<uint32_t>(__k, 16);
        uint64_t kvHeadOffset = (headId / groupNum) * strideKV;
        half localTor = 0;
        for (uint32_t nIdx = 0; nIdx < nLoop; nIdx += 2) {
            uint64_t numBlocksId0 = (uint64_t)(*((__ubuf__ int32_t *)blockTablesUbTensor.GetPhyAddr() + nIdx));
            uint64_t kvOffset0 = numBlocksId0 * blockSize * kvHeads * embeddingSize + kvHeadOffset;
            uint64_t numBlocksId1 = 0;
            uint64_t kvOffset1 = 0;
            if ((nIdx + 1) != nLoop) {
                numBlocksId1 = (uint64_t)(*((__ubuf__ int32_t *)blockTablesUbTensor.GetPhyAddr() + (nIdx + 1)));
                kvOffset1 = numBlocksId1 * blockSize * kvHeads * embeddingSize + kvHeadOffset;
            }
            uint32_t warpO = (nIdx == (nLoop - 1) || (nIdx + 1) == (nLoop - 1)) ? 1 : 0;
            uint32_t initG = (nIdx == 0) ? 1 : 0;
            uint32_t n0Actual = (nIdx == nLoop - 1) ? tail : blockSize;
            uint32_t n1Actual = ((nIdx + 1) == nLoop - 1) ? tail : blockSize;
            uint32_t roundN0 = Align<uint32_t>(n0Actual, 16);
            uint32_t roundN1 = Align<uint32_t>(n1Actual, 16);
            if ((nIdx + 1) == (nLoop)) {
                n1Actual = 0;
            }
            uint64_t qOffset = curBatch * embeddingSize * numHeads + headId * strideQO;
            uint64_t maskOffset = curBatch * batchMaskStride + nIdx * blockSize * 16 + headId * headMaskStride;
            for (uint32_t headOffset = 0; headOffset < repeatLen; ++headOffset) {
                uint32_t initKV = (headOffset == 0) ? 1 : 0; // kv move to L1 flags
                uint32_t initKVE = (warpO && headOffset == repeatLen - 1) ? 1 : 0; // synchronization for head loop end
                pa.Init(qOffset, kvOffset0, kvOffset0, kvOffset1, kvOffset1, maskOffset, qOffset, initG, warpO);
                pa.Decode(roundM, roundN0, roundK, roundN1, mActual, n0Actual, n1Actual, maskType, initKVE,
                            headOffset, initKV, localTor, scaleType);
                qOffset += strideQO;
                maskOffset += headMaskStride;
            }
        }
        taskId += repeatLen - 1;
    }
}
#endif // UNPAD_PAGED_ATTENTION_DECODER_H