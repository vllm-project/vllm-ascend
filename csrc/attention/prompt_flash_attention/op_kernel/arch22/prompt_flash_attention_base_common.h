/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef __DAV_C220_VEC__
constexpr int32_t ROW_OP_SPEC_MASK_32 = 32;
constexpr int32_t ROW_OP_SPEC_MASK_8 = 8;
constexpr int32_t ROW_OP_SPEC_MASK_4 = 4;
constexpr int32_t REDUCE_UB_SIZE = 1024;
constexpr int32_t S_DB_SIZE = 8192;

constexpr int32_t TILE_256 = 256;
constexpr int32_t TILE_512 = 512;

constexpr int32_t TWO_COMMON = 2;

enum class RowCalcTile {
    TAIL_TILE = 0,
    SPEC_TILE_256,
    SPEC_TILE_512
};

__aicore__ __attribute__((always_inline)) inline void SetVecMask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % FLOAT_VECTOR_SIZE;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }

    if (len == VECTOR_SIZE || len == 0) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (len >= FLOAT_VECTOR_SIZE) {
        AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
    } else {
        AscendC::SetVectorMask<int8_t>(0x0, mask);
    }
}

template<typename T>
__aicore__ __attribute__((always_inline)) inline void SetBlockReduceMask(int32_t len);

template<typename T, RowCalcTile TILE_MODE>
struct Rowsum {
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<T> &srcUb,
        const AscendC::LocalTensor<T> &rowsumUb,
        const AscendC::LocalTensor<T> &tmpUb,
        uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned);
};

template<typename T, RowCalcTile TILE_MODE>
struct Rowmax {
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<T> &srcUb,
        const AscendC::LocalTensor<T> &rowmaxUb,
        const AscendC::LocalTensor<T> &tmpUb,
        uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned);
};

template<typename S_DTYPE, typename EXP_DTYPE, typename P_DTYPE, typename MASK_DTYPE, MaskType MASK_TYPE>
struct OnlineSoftmaxStage1 {
    __aicore__ __attribute__((always_inline)) inline OnlineSoftmaxStage1(
        const AscendC::LocalTensor<S_DTYPE> &sUb,
        const AscendC::LocalTensor<MASK_DTYPE> &maskOrigUb,
        const AscendC::LocalTensor<S_DTYPE> &maskProcessedUb,
        const AscendC::LocalTensor<S_DTYPE> &localRowmaxUb,
        const AscendC::LocalTensor<S_DTYPE> &hatRowmaxUb,
        const AscendC::LocalTensor<S_DTYPE> &globalRowmaxUb,
        const AscendC::LocalTensor<S_DTYPE> &diffRowmaxUb,
        const AscendC::LocalTensor<EXP_DTYPE> &sExpUb,
        const AscendC::LocalTensor<EXP_DTYPE> &localRowsumUb,
        const AscendC::LocalTensor<EXP_DTYPE> &globalRowsumUb,
        const AscendC::LocalTensor<P_DTYPE> &pUb,
        const AscendC::LocalTensor<EXP_DTYPE> &tmpUb,
        const AscendC::GlobalTensor<S_DTYPE> &sGm,
        const AscendC::GlobalTensor<P_DTYPE> &pGm,
        bool firstNIter, S_DTYPE tor,
        uint32_t m, uint32_t nReal, uint32_t nStride, uint32_t pingpongFlag);
};

template<>
__aicore__ __attribute__((always_inline)) inline void SetBlockReduceMask<float>(int32_t len)
{
    if (len > 8 || len < 1) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t) 1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask +
                            (subMask << 56) + (subMask << 40) + (subMask << 24) + (subMask << 8);
    AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
}

template<>
__aicore__ __attribute__((always_inline)) inline void SetBlockReduceMask<half>(int32_t len)
{
    if (len > 16 || len < 1) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t) 1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
    AscendC::SetVectorMask<int8_t>(maskValue, maskValue);
}

template<>
struct Rowsum<float, RowCalcTile::SPEC_TILE_512>{
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<float> &srcUb,
        const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tmpUb,
        uint32_t numRowsRound , uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(
            tmpUb,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            tmpUb[REDUCE_UB_SIZE],
            tmpUb,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tmpUb[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template<>
struct Rowsum<float, RowCalcTile::SPEC_TILE_256>{
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<float> &srcUb,
        const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tmpUb,
        uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceSum<float, false>(
            tmpUb,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OP_SPEC_MASK_32);
        AscendC::BlockReduceSum<float, false>(
            tmpUb[REDUCE_UB_SIZE],
            tmpUb,
            numRowsRound,
            0,
            1,
            1,
            STRIDE_4
        );
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask<float>(ROW_OP_SPEC_MASK_4);
        AscendC::BlockReduceSum<float, false>(
            rowsumUb,
            tmpUb[REDUCE_UB_SIZE],
            (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
};

template<>
struct Rowsum<float, RowCalcTile::TAIL_TILE>{
    __aicore__ __attribute__((always_inline)) inline Rowsum(
        const AscendC::LocalTensor<float> &srcUb,
        const AscendC::LocalTensor<float> &rowsumUb,
        const AscendC::LocalTensor<float> &tmpUb,
        uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceSum<float, false>(
                tmpUb,
                srcUb,
                numRowsRound,
                0,
                1,
                1,
                numElemsAligned / FLOAT_BLOCK_SIZE
            );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceSum<float, false>(
                rowsumUb,
                tmpUb,
                (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                0,
                1,
                1,
                STRIDE_8
            );
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowsum_idx = 1; rowsum_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowsum_idx) {
                AscendC::BlockReduceSum<float, false>(
                    tmpUb,
                    srcUb[rowsum_idx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0,
                    1,
                    1,
                    numElemsAligned / FLOAT_BLOCK_SIZE
                );
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceSum<float, false>(
                    tmpUb[REDUCE_UB_SIZE],
                    tmpUb,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0,
                    1,
                    1,
                    STRIDE_8
                );
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tmpUb[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                );
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceSum<float, false>(
                tmpUb,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0,
                1,
                1,
                numElemsAligned / FLOAT_BLOCK_SIZE
            );
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask<float>((numElems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceSum<float, false>(
                    rowsumUb,
                    tmpUb,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0,
                    1,
                    1,
                    STRIDE_8
                );
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceSum<float, false>(
                    tmpUb[REDUCE_UB_SIZE],
                    tmpUb,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0,
                    1,
                    1,
                    STRIDE_8
                );
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Add<float, false>(
                    rowsumUb,
                    rowsumUb,
                    tmpUb[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                );
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
};

template<>
struct Rowmax<float, RowCalcTile::SPEC_TILE_512> {
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<float> &srcUb,
        const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tmpUb,
        uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(
            tmpUb,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            tmpUb[REDUCE_UB_SIZE],
            tmpUb,
            numRowsRound * numElemsAligned / FLOAT_BLOCK_SIZE / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tmpUb[REDUCE_UB_SIZE],
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
    }
};

template<>
struct Rowmax<float, RowCalcTile::SPEC_TILE_256>{
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<float> &srcUb,
        const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tmpUb,
        uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned)
    {
        AscendC::BlockReduceMax<float, false>(
            tmpUb,
            srcUb,
            numRowsRound * numElemsAligned / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        SetVecMask(ROW_OP_SPEC_MASK_32);
        AscendC::BlockReduceMax<float, false>(
            tmpUb[REDUCE_UB_SIZE],
            tmpUb,
            numRowsRound,
            0,
            1,
            1,
            STRIDE_4
        );
        AscendC::PipeBarrier<PIPE_V>();
        SetBlockReduceMask<float>(ROW_OP_SPEC_MASK_4);
        AscendC::BlockReduceMax<float, false>(
            rowmaxUb,
            tmpUb[REDUCE_UB_SIZE],
            (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
            0,
            1,
            1,
            STRIDE_8
        );
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    }
};

template<>
struct Rowmax<float, RowCalcTile::TAIL_TILE>{
    __aicore__ __attribute__((always_inline)) inline Rowmax(
        const AscendC::LocalTensor<float> &srcUb,
        const AscendC::LocalTensor<float> &rowmaxUb,
        const AscendC::LocalTensor<float> &tmpUb,
        uint32_t numRowsRound, uint32_t numElems, uint32_t numElemsAligned)
    {
        if (numElems >= FLOAT_VECTOR_SIZE) {
            AscendC::BlockReduceMax<float, false>(
                tmpUb,
                srcUb,
                numRowsRound,
                0,
                1,
                1,
                numElemsAligned / FLOAT_BLOCK_SIZE
            );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BlockReduceMax<float, false>(
                rowmaxUb,
                tmpUb,
                (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                0,
                1,
                1,
                STRIDE_8
            );
            AscendC::PipeBarrier<PIPE_V>();
            for (uint64_t rowmax_idx = 1; rowmax_idx < (uint64_t)numElems / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                AscendC::BlockReduceMax<float, false>(
                    tmpUb,
                    srcUb[rowmax_idx * FLOAT_VECTOR_SIZE],
                    numRowsRound,
                    0,
                    1,
                    1,
                    numElemsAligned / FLOAT_BLOCK_SIZE
                );
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::BlockReduceMax<float, false>(
                    tmpUb[REDUCE_UB_SIZE],
                    tmpUb,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0,
                    1,
                    1,
                    STRIDE_8
                );
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(
                    rowmaxUb,
                    rowmaxUb,
                    tmpUb[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                );
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (numElems % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(numElems % FLOAT_VECTOR_SIZE);
            AscendC::BlockReduceMax<float, false>(
                tmpUb,
                srcUb[numElems / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                numRowsRound,
                0,
                1,
                1,
                numElemsAligned / FLOAT_BLOCK_SIZE
            );
            AscendC::PipeBarrier<PIPE_V>();
            SetBlockReduceMask<float>((numElems % FLOAT_VECTOR_SIZE + FLOAT_BLOCK_SIZE - 1)/ FLOAT_BLOCK_SIZE);
            if (numElems < FLOAT_VECTOR_SIZE) {
                AscendC::BlockReduceMax<float, false>(
                    rowmaxUb,
                    tmpUb,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0,
                    1,
                    1,
                    STRIDE_8
                );
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::BlockReduceMax<float, false>(
                    tmpUb[REDUCE_UB_SIZE],
                    tmpUb,
                    (numRowsRound * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    0,
                    1,
                    1,
                    STRIDE_8
                );
                AscendC::PipeBarrier<PIPE_V>();
                SetVecMask(numRowsRound);
                AscendC::Max<float, false>(
                    rowmaxUb,
                    rowmaxUb,
                    tmpUb[REDUCE_UB_SIZE],
                    (uint64_t)0,
                    1,
                    AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
                );
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
};

template<typename P_DTYPE, typename MASK_DTYPE>
struct OnlineSoftmaxStage1<float, float, P_DTYPE, MASK_DTYPE, MaskType::MASK_TYPE_NONE> {
    __aicore__ __attribute__((always_inline)) inline OnlineSoftmaxStage1(
        const AscendC::LocalTensor<float> &sUb,
        const AscendC::LocalTensor<MASK_DTYPE> &maskOrigUb,
        const AscendC::LocalTensor<float> &maskProcessedUb,
        const AscendC::LocalTensor<float> &localRowmaxUb,
        const AscendC::LocalTensor<float> &hatRowmaxUb,
        const AscendC::LocalTensor<float> &globalRowmaxUb,
        const AscendC::LocalTensor<float> &diffRowmaxUb,
        const AscendC::LocalTensor<float> &sExpUb,
        const AscendC::LocalTensor<float> &localRowsumUb,
        const AscendC::LocalTensor<float> &globalRowsumUb,
        const AscendC::LocalTensor<P_DTYPE> &pUb,
        const AscendC::LocalTensor<float> &tmpUb,
        const AscendC::GlobalTensor<float> &sGm,
        const AscendC::GlobalTensor<P_DTYPE> &pGm,
        bool firstNIter, float tor,
        uint32_t m, uint32_t nReal, uint32_t nStride, uint32_t pingpongFlag)
    {
        uint32_t roundM = (m + FLOAT_BLOCK_SIZE - 1) / FLOAT_BLOCK_SIZE * FLOAT_BLOCK_SIZE;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((pingpongFlag));
        // input QK
        AscendC::DataCopy(
            sUb,
            sGm,
            AscendC::DataCopyParams(m, nStride / FLOAT_BLOCK_SIZE, 0, 0)
        );
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>((pingpongFlag));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>((pingpongFlag));
        // *** ls = tor * ls
        AscendC::Muls<float, false>(
            sUb,
            sUb,
            tor,
            (uint64_t)0,
            (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
        );
        AscendC::PipeBarrier<PIPE_V>();
        if (nReal == TILE_512) {
            Rowmax<float, RowCalcTile::SPEC_TILE_512>(
                sUb,
                localRowmaxUb,
                tmpUb,
                roundM, nReal, nStride
            );
        } else if (nReal == TILE_256) {
            Rowmax<float, RowCalcTile::SPEC_TILE_256>(
                sUb,
                localRowmaxUb,
                tmpUb,
                roundM, nReal, nStride
            );
        } else {
            Rowmax<float, RowCalcTile::TAIL_TILE>(
                sUb,
                localRowmaxUb,
                tmpUb,
                roundM, nReal, nStride
            );
        }

        if (firstNIter) {
            // *** hm = lm
            AscendC::DataCopy(
                hatRowmaxUb,
                localRowmaxUb,
                AscendC::DataCopyParams(1, roundM / FLOAT_BLOCK_SIZE, 0, 0)
            );
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(m);
            // *** hm = MAX(lm, gm)
            AscendC::Max<float, false>(
                hatRowmaxUb,
                localRowmaxUb,
                globalRowmaxUb,
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = gm - hm
            AscendC::Sub<float, false>(
                diffRowmaxUb,
                globalRowmaxUb,
                hatRowmaxUb,
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            // *** dm = exp(dm)
            AscendC::Exp<float, false>(
                diffRowmaxUb,
                diffRowmaxUb,
                (uint64_t)0,
                1,
                AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
            );
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::PipeBarrier<PIPE_V>();
        // *** gm = hm
        AscendC::DataCopy(
            globalRowmaxUb,
            hatRowmaxUb,
            AscendC::DataCopyParams(1, roundM / FLOAT_BLOCK_SIZE, 0, 0)
        );
        AscendC::PipeBarrier<PIPE_V>();
        // *** hm_block = expand_to_block(hm), 存放于 tv
        AscendC::Brcb(
            tmpUb.template ReinterpretCast<uint32_t>(),
            hatRowmaxUb.template ReinterpretCast<uint32_t>(),
            roundM / FLOAT_BLOCK_SIZE,
            AscendC::BrcbRepeatParams(1, STRIDE_8)
        );
        AscendC::PipeBarrier<PIPE_V>();
        // *** ls = ls - hm_block
        for (uint32_t vsubIdx = 0; vsubIdx < nReal / FLOAT_VECTOR_SIZE; ++vsubIdx) {
            AscendC::Sub<float, false>(
                sUb[vsubIdx * FLOAT_VECTOR_SIZE],
                sUb[vsubIdx * FLOAT_VECTOR_SIZE],
                tmpUb,
                (uint64_t)0,
                m,
                AscendC::BinaryRepeatParams(1, 1, 0, nStride / FLOAT_BLOCK_SIZE, nStride / FLOAT_BLOCK_SIZE, 1)
            );
        }
        if (nReal % FLOAT_VECTOR_SIZE > 0) {
            SetVecMask(nReal % FLOAT_VECTOR_SIZE);
            AscendC::Sub<float, false>(
                sUb[nReal / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                sUb[nReal / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                tmpUb,
                (uint64_t)0,
                m,
                AscendC::BinaryRepeatParams(1, 1, 0, nStride / FLOAT_BLOCK_SIZE, nStride / FLOAT_BLOCK_SIZE, 1)
            );
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // *** ls = exp(ls)
        AscendC::Exp<float, false>(
            sExpUb,
            sUb,
            (uint64_t)0,
            (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, STRIDE_8, STRIDE_8)
        );
        AscendC::PipeBarrier<PIPE_V>();
        // *** ll = rowsum(ls32)
        if (nReal == TILE_512) {
            Rowsum<float, RowCalcTile::SPEC_TILE_512>(
                sExpUb,
                localRowsumUb,
                tmpUb,
                roundM, nReal, nStride
            );
        } else if (nReal == TILE_256) {
            Rowsum<float, RowCalcTile::SPEC_TILE_256>(
                sExpUb,
                localRowsumUb,
                tmpUb,
                roundM, nReal, nStride
            );
        } else {
            Rowsum<float, RowCalcTile::TAIL_TILE>(
                sExpUb,
                localRowsumUb,
                tmpUb,
                roundM, nReal, nStride
            );
        }

        // *** lp = castfp32to16(ls)
        if constexpr (std::is_same<P_DTYPE, __bf16>::value) {
            AscendC::Cast<P_DTYPE, float, false>(
                pUb,
                sExpUb,
                AscendC::RoundMode::CAST_RINT,
                (uint64_t)0,
                (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8));
        } else {
            AscendC::Cast<P_DTYPE, float, false>(
                pUb,
                sExpUb,
                AscendC::RoundMode::CAST_NONE,
                (uint64_t)0,
                (m * nStride + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::UnaryRepeatParams(1, 1, STRIDE_4, STRIDE_8));
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>((pingpongFlag));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>((pingpongFlag));
        AscendC::DataCopy(
            pGm,
            pUb,
            AscendC::DataCopyParams(m, nStride * TWO_COMMON / BlockSize<int8_t>(), 0, 0)
        );
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((pingpongFlag));
        if (firstNIter) {
            // *** gl = ll
            AscendC::DataCopy(
                globalRowsumUb,
                localRowsumUb,
                AscendC::DataCopyParams(1, roundM / FLOAT_BLOCK_SIZE, 0, 0)
            );
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            SetVecMask(m);
            // *** gl = dm * gl
            AscendC::Mul<float, false>(
                globalRowsumUb,
                diffRowmaxUb,
                globalRowsumUb,
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            // *** gl = ll + gl
            AscendC::Add<float, false>(
                globalRowsumUb,
                globalRowsumUb,
                localRowsumUb,
                (uint64_t)0,
                1,
                AscendC::BinaryRepeatParams(1, 1, 1, STRIDE_8, STRIDE_8, STRIDE_8)
            );
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        }
    }
};

#endif