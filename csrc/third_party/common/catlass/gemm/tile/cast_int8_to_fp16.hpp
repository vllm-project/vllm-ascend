/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_CAST_INT8_TO_FP16_HPP
#define CATLASS_GEMM_TILE_CAST_INT8_TO_FP16_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class SrcType_,
    class DstType_,
    // Length of the compute elements
    uint32_t COMPUTE_LEN_,
    uint32_t STAGES = 2
>
struct TileCastInt8ToFp16Dequant {
    using ElementSrc = typename SrcType_::Element;
    using ElementDst = typename DstType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using LayoutDst = typename DstType_::Layout;

    static_assert(std::is_same_v<LayoutSrc, layout::RowMajor> || std::is_same_v<LayoutSrc, layout::ColumnMajor>,
        "Unsupported layout, only can be RowMajor and ColumnMajor");
    static_assert(std::is_same_v<LayoutDst, LayoutSrc>, "layoutDst and layoutSrc must be the same");

    static constexpr uint32_t ELE_NUM_PER_BLK_INT8 = BYTE_PER_BLK / sizeof(ElementSrc);
    static constexpr uint32_t ELE_NUM_PER_BLK_HALF = BYTE_PER_BLK / sizeof(ElementDst);

    static constexpr uint32_t COMPUTE_LEN = COMPUTE_LEN_;

    static_assert(COMPUTE_LEN <= 32 * 1024, "COMPUTE_LEN cannot exceed 32 * 1024");

    struct Params {
        half deqScalar;
        half deqZeroPoint;

        CATLASS_HOST_DEVICE
        Params() = default;

        CATLASS_HOST_DEVICE
        Params(half deqScalar_, half deqZeroPoint_)
        {
            deqScalar = deqScalar_;
            deqZeroPoint = deqZeroPoint_;
        }
    };

    /// Construct
    CATLASS_DEVICE
    TileCastInt8ToFp16Dequant(Arch::Resource<ArchTag> const &resource, Params const &params_) : params(params_)
    {
        if constexpr (g_coreType == AscendC::AIV) {
            uint32_t ubOffset = 0;
            uint32_t ubInSize = COMPUTE_LEN * sizeof(ElementSrc);
            uint32_t ubOutSize = COMPUTE_LEN * sizeof(ElementDst);
            // Init buffers
            for (uint32_t i = 0; i < STAGES; i++) {
                ubInTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementSrc>(ubOffset);
                ubOffset += ubInSize;
                ubOutTensorList[i] = resource.ubBuf.template GetBufferByByte<ElementDst>(ubOffset);
                ubOffset += ubOutSize;

                ubEventList[i] = i;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

    /// Destructor
    CATLASS_DEVICE
    ~TileCastInt8ToFp16Dequant() {
        if constexpr (g_coreType == AscendC::AIV) {
            for (uint32_t i = 0; i < STAGES; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[i]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[i]);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const &gmDst, LayoutDst const &layoutDst,
        AscendC::GlobalTensor<ElementSrc> const &gmSrc, LayoutSrc const &layoutSrc
    )
    {
        uint32_t tilesNum = layoutSrc.shape(0);
        uint32_t tileLen = layoutSrc.shape(1);
        uint32_t tileLenRoundInt8 = RoundUp(layoutSrc.shape(1), ELE_NUM_PER_BLK_INT8);
        uint64_t tileStrideSrc = layoutSrc.stride(0);
        uint64_t tileStrideDst = layoutDst.stride(0);
        if constexpr (std::is_same_v<LayoutSrc, layout::ColumnMajor>) {
            tilesNum = layoutSrc.shape(1);
            tileLen = layoutSrc.shape(0);
            tileLenRoundInt8 = RoundUp(layoutSrc.shape(0), ELE_NUM_PER_BLK_INT8);
            tileStrideSrc = layoutSrc.stride(1);
            tileStrideDst = layoutDst.stride(1);
        }
        uint32_t tilesPerAiv = tilesNum / AscendC::GetSubBlockNum();
        if (AscendC::GetSubBlockIdx() < (tilesNum % AscendC::GetSubBlockNum())) {
            tilesPerAiv++;
        }
        uint64_t taskOffsetSrc = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideSrc;
        uint64_t taskOffsetDst = AscendC::GetSubBlockIdx() * tilesPerAiv * tileStrideDst;
        if (AscendC::GetSubBlockIdx() >= (tilesNum % AscendC::GetSubBlockNum())) {
            taskOffsetSrc += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideSrc;
            taskOffsetDst += (tilesNum % AscendC::GetSubBlockNum()) * tileStrideDst;
        }
        uint32_t tilesPerLoop = COMPUTE_LEN / tileLenRoundInt8;
        uint32_t loops = CeilDiv(tilesPerAiv, tilesPerLoop);
        uint32_t pingpong = 0;
        for (uint32_t loopIdx = 0; loopIdx < loops; loopIdx++) {
            uint32_t actualTiles = tilesPerLoop;
            if (loopIdx == loops - 1) {
                actualTiles = tilesPerAiv - loopIdx * tilesPerLoop;
            }
            uint64_t tileOffsetSrc = loopIdx * tilesPerLoop * tileStrideSrc;
            AscendC::DataCopyExtParams dataCopyParamsIn(
                actualTiles, tileLen * sizeof(ElementSrc),
                (tileStrideSrc - tileLen) * sizeof(ElementSrc),
                (tileLenRoundInt8 - tileLen) / ELE_NUM_PER_BLK_INT8, 0
            );
            AscendC::DataCopyPadExtParams<ElementSrc> padParams(false, 0, 0, 0);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);
            AscendC::DataCopyPad(ubInTensorList[pingpong],
                gmSrc[taskOffsetSrc + tileOffsetSrc], dataCopyParamsIn, padParams);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ubEventList[pingpong]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            AscendC::Cast(ubOutTensorList[pingpong], ubInTensorList[pingpong],
                AscendC::RoundMode::CAST_NONE, actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(ubEventList[pingpong]);

            AscendC::Adds(ubOutTensorList[pingpong], ubOutTensorList[pingpong],
                params.deqZeroPoint, actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(ubOutTensorList[pingpong], ubOutTensorList[pingpong],
                params.deqScalar, actualTiles * tileLenRoundInt8);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubEventList[pingpong]);

            uint64_t tileOffsetDst = loopIdx * tilesPerLoop * tileStrideDst;
            AscendC::DataCopyExtParams dataCopyParamsOut(
                actualTiles, tileLen * sizeof(ElementDst),
                (tileLenRoundInt8 - tileLen) / ELE_NUM_PER_BLK_HALF,
                (tileStrideDst - tileLen) * sizeof(ElementDst), 0
            );
            AscendC::DataCopyPad(gmDst[taskOffsetDst + tileOffsetDst],
                ubOutTensorList[pingpong], dataCopyParamsOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubEventList[pingpong]);

            pingpong = (pingpong + 1) % STAGES;
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementSrc> ubInTensorList[STAGES];
    AscendC::LocalTensor<ElementDst> ubOutTensorList[STAGES];
    int32_t ubEventList[STAGES];

    Params params;
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_CAST_INT8_TO_FP16_HPP