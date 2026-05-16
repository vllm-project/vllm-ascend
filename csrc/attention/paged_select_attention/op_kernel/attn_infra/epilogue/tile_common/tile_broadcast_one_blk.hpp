/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_TILE_TILE_BROADCAST_ONE_BLK_HPP
#define EPILOGUE_TILE_TILE_BROADCAST_ONE_BLK_HPP

#include "../../../attn_infra/base_defs.hpp"

namespace NpuArch::Epilogue::Tile
{

template <
    class ArchTag_,
    class ComputeType_,
    uint32_t COMPUTE_LENGTH_
>
struct TileBroadcastOneBlk {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    static constexpr uint32_t COMPUTE_LENGTH = COMPUTE_LENGTH_;

    __aicore__ inline
    TileBroadcastOneBlk() {}

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn
    )
    {
        constexpr uint32_t maxRepeatNum = 255;
        constexpr uint32_t eleNumPerBlk = static_cast<uint32_t>(BYTE_PER_BLK) / static_cast<uint32_t>(sizeof(ElementCompute));

        AscendC::BrcbRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.dstRepStride = BLK_NUM_PER_VECTOR_FRACTAL;

        constexpr uint32_t eleNumPerCompute =
            NpuArch::Detail::Alignment::RoundDown<eleNumPerBlk>(maxRepeatNum * BLK_NUM_PER_VECTOR_FRACTAL);
        for (uint32_t offset = 0; offset < COMPUTE_LENGTH; offset += eleNumPerCompute) {
            uint32_t residueM = COMPUTE_LENGTH - offset;
            uint32_t computeM = (residueM > eleNumPerCompute) ? eleNumPerCompute : residueM;
            uint8_t repeatTimes =
                static_cast<uint8_t>(NpuArch::Detail::Alignment::CeilDiv<BLK_NUM_PER_VECTOR_FRACTAL>(computeM));
            AscendC::Brcb(
                ubOut[offset * eleNumPerBlk], ubIn[offset],
                repeatTimes, repeatParams
            );
        }
    }
};

} // namespace NpuArch::Epilogue::Tile

#endif
