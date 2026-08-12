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
 * \file mc2_matmul_compute.h
 * \brief
 */

#ifndef MC2_MATMUL_COMPUTE_H
#define MC2_MATMUL_COMPUTE_H

#include "mc2_matmul_block_l2cache.h"

namespace AscendC {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
class MatmulCompute {
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

public:
    __aicore__ inline MatmulCompute()
    {
    }
    __aicore__ inline void Init(Mc2Tiling::RCSTiling &cfg, TCubeTiling &tiling, Mc2Tiling::TileL2Tiling &l2Tiling,
                                uint32_t rankID);
    __aicore__ inline void InitGlobalTensor(GM_ADDR bGM, GM_ADDR biasGM);
    __aicore__ inline void InitGlobalTensor(GM_ADDR aGM, uint64_t aSize, GM_ADDR cGM, uint64_t cSize);
    __aicore__ inline void Compute(uint32_t index = 0, uint32_t offset = 0);
    __aicore__ inline void ComputeWithoutIndex();
    __aicore__ inline void ComputeWithL2Cache(uint32_t index = 0);
    __aicore__ inline void End();

private:
    __aicore__ inline void SetOrgShapeAlign();
    __aicore__ inline void ComputeL2Tile(int32_t mTileIndex, int32_t nTileIndex);

private:
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> mm_;
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;
    typename BlockType<T>::PARAMS block_;
    TCubeTiling tiling_;
    Mc2Tiling::TileL2Tiling l2Tiling_;
    Mc2Tiling::RCSTiling cfg_;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::InitGlobalTensor(GM_ADDR bGM,
                                                                                             GM_ADDR biasGM)
{
    // MC2的计算流中默认B矩阵不变，GM地址无需偏移
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling_.Kb * tiling_.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), tiling_.N);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void
MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::InitGlobalTensor(GM_ADDR aGM, uint64_t aSize, GM_ADDR cGM,
                                                                      uint64_t cSize)
{
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), aSize);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), cSize);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void
MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::Init(Mc2Tiling::RCSTiling &cfg, TCubeTiling &tiling,
                                                          Mc2Tiling::TileL2Tiling &l2Tiling, uint32_t rankID)
{
    // MatmulImpl初始化
    mm_.SetSubBlockIdx(0);
    PRELOAD(4);
    mm_.Init(&tiling, GetTPipePtr());
    tiling_ = tiling;
    l2Tiling_ = l2Tiling;
    cfg_ = cfg;
    block_.Init(cfg, tiling, l2Tiling, rankID);
    SetOrgShapeAlign();
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::SetOrgShapeAlign()
{
    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        auto alignKa = AlignUp(tiling_.Ka, C0_SIZE);
        auto alignKb = AlignUp(tiling_.Kb, C0_SIZE);
        auto alignM = AlignUp(tiling_.M, C0_SIZE);
        auto alignN = AlignUp(tiling_.N, C0_SIZE);
        mm_.SetOrgShape(alignM, alignN, alignKa, alignKb, cfg_.rankN);
    } else if (A_TYPE::format == CubeFormat::NZ) {
        auto alignKa = AlignUp(tiling_.Ka, C0_SIZE);
        auto alignM = AlignUp(tiling_.M, C0_SIZE);
        mm_.SetOrgShape(alignM, tiling_.N, alignKa, tiling_.Kb, cfg_.rankN);
    } else if (B_TYPE::format == CubeFormat::NZ) {
        auto alignKb = AlignUp(tiling_.Kb, C0_SIZE);
        auto alignN = AlignUp(tiling_.N, C0_SIZE);
        mm_.SetOrgShape(tiling_.M, alignN, tiling_.Ka, alignKb, cfg_.rankN);
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::Compute(uint32_t index, uint32_t offset)
{
    if (cfg_.rankN == 0) {
        return;
    }
    // 每次block循环开始前需要计算初始blockIndex
    block_.InitBlockIndex(index);
    for (uint32_t i = 0; i < block_.args_.blockCnt; i++) {
        // calculate blockCurrIndex
        block_.UpdateBlockIndex(i + offset);
        if (block_.args_.blockCurrIdx < block_.args_.totalBlockCnt) {
            block_.UpdateBlockParams();
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            mm_.SetSingleShape(block_.args_.singleCoreM, block_.args_.singleCoreN, tiling_.singleCoreK);
            mm_.SetTensorA(aGlobal[block_.offset_.offsetA], block_.args_.isTransA);
            mm_.SetTensorB(bGlobal[block_.offset_.offsetB], block_.args_.isTransB);
            if (tiling_.isBias) {
                mm_.SetBias(biasGlobal[block_.offset_.offsetBias]);
            }
            mm_.Iterate();
            mm_.GetTensorC(cGlobal[block_.offset_.offsetC]);
            // 增加M等FIX同步
            event_t eventIDFixToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
            SetFlag<HardEvent::FIX_M>(eventIDFixToM);
            WaitFlag<HardEvent::FIX_M>(eventIDFixToM);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::ComputeWithoutIndex()
{
    if (cfg_.rankN == 0) {
        return;
    }
    // 每次block循环开始前需要计算初始blockIndex
    block_.InitBlockWithoutIndex();
    for (uint32_t i = 0; i < block_.args_.blockCnt; i++) {
        // calculate blockCurrIndex
        block_.UpdateBlockIndex(i);
        if (block_.args_.blockCurrIdx < block_.args_.totalBlockCnt) {
            block_.UpdateBlockParams();
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            mm_.SetSingleShape(block_.args_.singleCoreM, block_.args_.singleCoreN, tiling_.singleCoreK);
            mm_.SetTensorA(aGlobal[block_.offset_.offsetA], block_.args_.isTransA);
            mm_.SetTensorB(bGlobal[block_.offset_.offsetB], block_.args_.isTransB);
            if (tiling_.isBias) {
                mm_.SetBias(biasGlobal[block_.offset_.offsetBias]);
            }
            mm_.Iterate();
            mm_.GetTensorC(cGlobal[block_.offset_.offsetC]);
            // 增加M等FIX同步
            event_t eventIDFixToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
            SetFlag<HardEvent::FIX_M>(eventIDFixToM);
            WaitFlag<HardEvent::FIX_M>(eventIDFixToM);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::ComputeL2Tile(int32_t mTileIndex,
                                                                                          int32_t nTileIndex)
{
    if (cfg_.rankN == 0) {
        return;
    }
    // 每次block循环开始前需要更新m&n方向分核数
    block_.UpdateBlockCnt(mTileIndex, nTileIndex);
    block_.InitBlockIndex();
    for (uint32_t i = 0; i < block_.args_.blockCnt; i++) {
        // calculate blockCurrIndex
        block_.UpdateBlockIndex(i);
        if (block_.args_.blockCurrIdx < block_.args_.totalBlockCnt) {
            block_.UpdateBlockParams(mTileIndex, nTileIndex);
            block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
            // 调用MatmulImpl完成一次Block计算
            mm_.SetSingleShape(block_.args_.singleCoreM, block_.args_.singleCoreN, tiling_.singleCoreK);
            mm_.SetTensorB(bGlobal[block_.offset_.offsetB], block_.args_.isTransB);
            mm_.SetTensorA(aGlobal[block_.offset_.offsetA], block_.args_.isTransA);
            if (tiling_.isBias) {
                mm_.SetBias(biasGlobal[block_.offset_.offsetBias]);
            }
            mm_.Iterate();
            mm_.GetTensorC(cGlobal[block_.offset_.offsetC]);
            // 增加M等FIX同步
            event_t eventIDFixToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
            SetFlag<HardEvent::FIX_M>(eventIDFixToM);
            WaitFlag<HardEvent::FIX_M>(eventIDFixToM);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::ComputeWithL2Cache(uint32_t index)
{
    // 进入切片计算之前先更新多余尾块分配核ID的起始位置
    if (index > 0) {
        auto mBlockCnt = DivCeil(tiling_.M, tiling_.baseM) * l2Tiling_.rankTileNum;
        auto preCoreNum = (DivCeil(tiling_.N, tiling_.baseN) * mBlockCnt) % tiling_.usedCoreNum;
        block_.args_.preCoreStartIdx = (index * preCoreNum) % tiling_.usedCoreNum;
    }
    bool reverse = false;
    for (int32_t mTileIndex = 0; mTileIndex < l2Tiling_.mL2TileCnt; mTileIndex++) {
        for (int32_t nTileIndexTmp = 0; nTileIndexTmp < l2Tiling_.nL2TileCnt; nTileIndexTmp++) {
            int32_t nTileIndex = reverse ? (l2Tiling_.nL2TileCnt - nTileIndexTmp - 1) : nTileIndexTmp;
            // L2切分后的单片矩阵计算
            ComputeL2Tile(mTileIndex, nTileIndex);
        }
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, SplitType T>
__aicore__ inline void MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, T>::End()
{
    mm_.End();
}
} // namespace AscendC
#endif // MC2_MATMUL_COMPUTE_H
