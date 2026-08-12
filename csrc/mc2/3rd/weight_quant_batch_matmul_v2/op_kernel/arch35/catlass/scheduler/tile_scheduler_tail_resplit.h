/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_SCHEDULER_TILE_SCHEDULER_TAIL_RESPLIT_H
#define ARCH35_CATLASS_SCHEDULER_TILE_SCHEDULER_TAIL_RESPLIT_H

#include "../iterator/continuous_iterator.h"
#include "../iterator/tail_resplit_iterator.h"
#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#define BLOCK_N 32
#define BLOCK_M 1

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
// SubBlockNum实际使用的核数
template <int32_t SubBlockNum>
class TileSchedulerTailResplit {
public:
    template <typename ProblemShape, typename Params,
              AscendC::Std::enable_if_t<!AscendC::Std::is_tuple_v<Params>, bool> = true>
    DEVICE TileSchedulerTailResplit(ProblemShape const &problemShape, Params const &params)
        : nIter(params.mainBlockCount, params.firstTailBlockCount, params.secondTailBlockCount, params.mainBlockSize,
                params.firstTailBlockSize, params.secondTailBlockSize, params.cubeBlockDimN,
                static_cast<uint64_t>(get<1>(problemShape)))
    {
        auto mSize = get<0>(problemShape);
        decltype(AscendC::GetBlockIdx()) blockIdx;
        if ASCEND_IS_AIC {
            blockIdx = AscendC::GetBlockIdx();
        } else {
            // 硬件核数
            blockIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        }

        // 连续访问
        auto singleCoreM = (mSize + params.cubeBlockDimM - 1) / params.cubeBlockDimM;
        mIter.start = blockIdx / params.cubeBlockDimN * singleCoreM;
        mIter.curr = mIter.start;
        mIter.stop = Min(mIter.start + singleCoreM, mSize);
        mIter.step = params.matmulTiling.baseM * params.matmulTiling.stepM;
        mIter.UpdateSize();
        fullloadM = singleCoreM <= mIter.step;

        if ASCEND_IS_AIC {
            nIter.Config(blockIdx, false, 0);
        } else {
            nIter.Config(blockIdx, true, AscendC::GetSubBlockIdx());
        }
    }

    template <typename ProblemShape, typename TilingShape,
              AscendC::Std::enable_if_t<AscendC::Std::is_tuple_v<TilingShape>, bool> = true>
    DEVICE TileSchedulerTailResplit(ProblemShape const &problemShape, TilingShape const &tiling)
    {
    }

    DEVICE bool IsValid()
    {
        return !mIter.End() && !nIter.End();
    }

    DEVICE void FetchNextWork()
    {
        if (fullloadM) {
            ++nIter;
        } else {
            ++nIter;
            if (nIter.End()) {
                nIter.Reset();
                ++mIter;
            }
        }
    }

    DEVICE ContinuousIterator<uint64_t> MIter()
    {
        return mIter;
    }

    DEVICE const ContinuousIterator<uint64_t> MIter() const
    {
        return mIter;
    }

    DEVICE ContinuousIterator<uint64_t> &MIterRef()
    {
        return mIter;
    }

    DEVICE const ContinuousIterator<uint64_t> &MIterRef() const
    {
        return mIter;
    }

    DEVICE TailResplitIterator<uint64_t> NIter()
    {
        return nIter;
    }

    DEVICE const TailResplitIterator<uint64_t> NIter() const
    {
        return nIter;
    }

    DEVICE TailResplitIterator<uint64_t> &NIterRef()
    {
        return nIter;
    }

    DEVICE const TailResplitIterator<uint64_t> &NIterRef() const
    {
        return nIter;
    }

    DEVICE void Print() const
    {
#if defined(__CCE_KT_TEST__)
        mIter.Print(0);
        nIter.Print(1);
#endif
    }

private:
    ContinuousIterator<uint64_t> mIter;
    TailResplitIterator<uint64_t> nIter;
    bool fullloadM;
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
