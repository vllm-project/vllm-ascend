/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_TILE_ASCEND950_COPY_L0C_TO_UB_950_HPP
#define COMMON_TILE_ASCEND950_COPY_L0C_TO_UB_950_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510
#include "catlass/gemm/tile/ascend950/copy_l0c_to_dst.hpp"
#endif
#include "tla/tensor.hpp"
#include "catlass/detail/tag_to_layout.hpp"

namespace Common::Tile {

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    Catlass::Gemm::Tile::CopyL0CToUBMode CopyMode = Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
    Catlass::Gemm::Tile::ScaleGranularity DEQUANT_GRANULARITY = Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
    bool ReluEnable = false,
    class Enable = void
>
struct CopyL0CToUBTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to ub, can not find the specialization.");
};

#endif

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA_,
    class ElementB_,
    class LayoutTagB_,
    class ElementC_,
    class LayoutTagC_,
    class ElementBias = void,
    bool ReluEnable_ = false,
    Catlass::Gemm::Tile::ScaleGranularity DEQUANT_GRANULARITY_ = Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
    class L0CCopyMode = Catlass::Gemm::Tile::CopyToGM>
struct PackedTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using LayoutTagA = LayoutTagA_;
    using LayoutTagB = LayoutTagB_;
    using LayoutTagC = LayoutTagC_;
    using ElementAccumulator = typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    static constexpr bool ReluEnable = ReluEnable_;
    static constexpr Catlass::Gemm::Tile::ScaleGranularity DEQUANT_GRANULARITY = DEQUANT_GRANULARITY_;

    static constexpr bool HAS_BIAS = !std::is_void_v<ElementBias>;
    static constexpr bool HAS_QUANT_TENSOR = (DEQUANT_GRANULARITY == Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL);

    using LayoutTagL1A = typename Catlass::Gemm::helper::L1ATypeSelector<Catlass::Gemm::GemmType<ElementA, LayoutTagA>>::L1AType::Layout;
    using LayoutTagL1B = typename Catlass::Gemm::helper::L1BTypeSelector<Catlass::Gemm::GemmType<ElementB, LayoutTagB>>::L1BType::Layout;
    using LayoutTagL0A = typename Catlass::Gemm::helper::L0ALayoutSelector<ArchTag>::Layout;
    using LayoutTagL0B = Catlass::layout::nZ;
    using LayoutTagL0C = Catlass::layout::L0C;

    using LayoutA = Catlass::detail::TagToLayout_t<ElementA, LayoutTagA>;
    using LayoutB = Catlass::detail::TagToLayout_t<ElementB, LayoutTagB>;
    using LayoutC = Catlass::detail::TagToLayout_t<ElementC_, LayoutTagC>;

    using LayoutL1A = Catlass::detail::TagToLayout_t<ElementA, LayoutTagL1A>;
    using LayoutL1B = Catlass::detail::TagToLayout_t<ElementB, LayoutTagL1B>;
    using LayoutL0A = Catlass::detail::TagToLayout_t<ElementA, LayoutTagL0A>;
    using LayoutL0B = Catlass::detail::TagToLayout_t<ElementB, LayoutTagL0B>;
    using LayoutL0C = typename Catlass::detail::LayoutL0C;

    using TensorL1AVectorLayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0>, AscendC::TPosition::A1>;
    using TensorL1ALayout =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL1A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;

    using TensorL1A =
        std::conditional_t<tla::detail::isVector<LayoutTagA>::value, TensorL1AVectorLayout, TensorL1ALayout>;
    using TensorL1B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL1B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A1>;
    using TensorL0A =
        tla::Tensor<AscendC::LocalTensor<ElementA>, LayoutL0A, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::A2>;
    using TensorL0B =
        tla::Tensor<AscendC::LocalTensor<ElementB>, LayoutL0B, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::B2>;
    using TensorL0C = tla::
        Tensor<AscendC::LocalTensor<ElementAccumulator>, LayoutL0C, tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::CO1>;
    using TensorL1Bias = std::conditional_t<
        HAS_BIAS,
        tla::Tensor<
            AscendC::LocalTensor<ElementBias>,
            Catlass::detail::TagToLayout_t<ElementBias, Catlass::layout::VectorLayout>,
            tla::Coord<tla::_0>,
            AscendC::TPosition::A1>,
        Catlass::EmptyClass>;
    using TensorL0Bias = tla::Tensor<
        AscendC::LocalTensor<ElementAccumulator>,
        Catlass::detail::TagToLayout_t<ElementAccumulator, Catlass::layout::VectorLayout>,
        tla::Coord<tla::_0>,
        AscendC::TPosition::C2>;
    using TensorL1Quant = std::conditional_t<
        HAS_QUANT_TENSOR,
        tla::Tensor<AscendC::LocalTensor<uint64_t>,
            Catlass::detail::TagToLayout_t<uint64_t, Catlass::layout::VectorLayout>,
            tla::Coord<tla::_0>,
            AscendC::TPosition::A1>,
        Catlass::EmptyClass>;

    using L1AAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementA, LayoutTagA>;
    using L1BAlignHelper = Catlass::Gemm::helper::L1AlignHelper<ElementB, LayoutTagB>;

    template <class TensorA>
    using CopyGmToL1A = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorA, TensorL1A>;

    template <class TensorB>
    using CopyGmToL1B = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorB, TensorL1B>;

    template <class TensorBias>
    using CopyGmToL1Bias =
        std::conditional_t<HAS_BIAS, Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorBias, TensorL1Bias>, Catlass::EmptyClass>;

    template <class TensorQuant>
    using CopyGmToL1Scale =
        std::conditional_t<HAS_QUANT_TENSOR, Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorQuant, TensorL1Quant>, Catlass::EmptyClass>;

    using CopyL1ToL0A = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorL1A, TensorL0A>;
    using CopyL1ToL0B = Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorL1B, TensorL0B>;
    using CopyL1ToBT =
        std::conditional_t<HAS_BIAS, Catlass::Gemm::Tile::TileCopyTla<ArchTag, TensorL1Bias, TensorL0Bias>, Catlass::EmptyClass>;

#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 3510)
    template <class TensorC>
    using CopyL0CToDst = Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
#endif
#if (defined(CATLASS_ARCH) && CATLASS_ARCH == 2201)
    template <class TensorC>
    using CopyL0CToGm = Catlass::Gemm::Tile::CopyL0CToGmTla<ArchTag, TensorL0C, TensorC, DEQUANT_GRANULARITY, ReluEnable>;
#endif
};

#if defined(CATLASS_ARCH) && CATLASS_ARCH == 3510

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
    Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        Catlass::Gemm::Tile::CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = tla::get<0>(dstTensor.originShape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t ubRowNum,
        uint8_t beginSubBlockIdx, uint8_t sendVecNum,uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        uint32_t dstTensorRowNum = tla::get<0>(dstTensor.originShape());
        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = min(dstTensorRowNum, ubRowNum);
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = beginSubBlockIdx;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        if (sendVecNum > 1) {
            if (dstTensorRowNum > ubRowNum) {
                intriParams.mSize = dstTensorRowNum - ubRowNum;
                intriParams.subBlockId = (beginSubBlockIdx + 1) % 2;
                srcOffset += ubRowNum * tla::get<0, 0>(srcTensor.stride());
                AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
                    dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
            }
        }
    }

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor,
        uint8_t subBlockIdx, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.shape());
        intriParams.mSize = tla::get<0>(dstTensor.shape());
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 0;
        intriParams.subBlockId = subBlockIdx;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M,
    Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        Catlass::Gemm::Tile::CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = tla::get<1>(dstTensor.originShape());
        intriParams.mSize = RoundUp(tla::get<0>(dstTensor.originShape()), 2); // m must be even when spilt m
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 1;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

template <class TensorSrc_, class ElementDst_, class LayoutDst_, class CoordDst_, bool ReluEnable_>
struct CopyL0CToUBTla<
    Catlass::Arch::Ascend950,
    TensorSrc_,
    tla::Tensor<AscendC::LocalTensor<ElementDst_>, LayoutDst_, CoordDst_, AscendC::TPosition::VECCALC>,
    Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_N,
    Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT,
    ReluEnable_,
    std::enable_if_t<tla::detail::isRowMajor<LayoutDst_>::value>> {
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementDst = ElementDst_;
    using ElementSrc = typename TensorSrc_::Element;
    static constexpr auto quantPre =
        Catlass::Gemm::Tile::CopyL0CToDstQuantMode<ArchTag, ElementSrc, ElementDst, Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint8_t unitFlag = 0)
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorDst::Layout>::value && TensorSrc::position == AscendC::TPosition::CO1
                && TensorDst::position == AscendC::TPosition::VECCALC,
            "The input parameters do not match. TensorSrc must be L0C, while TensorDst must be UB and RowMajor"
        );

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> intriParams;

        // Fixpipe layout information
        intriParams.nSize = RoundUp(tla::get<1>(dstTensor.originShape()), 32);
        intriParams.mSize = tla::get<0>(dstTensor.originShape()); // m must be even when spilt m
        intriParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / tla::get<0, 0>(srcTensor.stride());
        intriParams.dstStride = tla::get<0>(dstTensor.stride());

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        intriParams.dualDstCtl = 2;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, CFG_ROW_MAJOR_UB>(
            dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

#endif

template <
    /// Tag indicating architecture
    class ArchTag,
    class ElementA_,
    class LayoutTagA,
    class ElementB_,
    class LayoutTagB,
    class ElementC_,
    class LayoutTagC,
    class ElementBias = void,
    Catlass::Gemm::Tile::CopyL0CToUBMode CopyMode_ = Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT,
    bool ReluEnable = false,
    Catlass::Gemm::Tile::ScaleGranularity DEQUANT_GRANULARITY = Catlass::Gemm::Tile::ScaleGranularity::NO_QUANT>
struct PackedTileCopyTlaToUB
    : public PackedTileCopyTla<ArchTag, ElementA_, LayoutTagA, ElementB_, LayoutTagB, ElementC_, LayoutTagC, ElementBias> {
    static constexpr Catlass::Gemm::Tile::CopyL0CToUBMode CopyMode = CopyMode_;
    // 重写 CopyL0CToDst
    using TensorL0C = typename PackedTileCopyTla<
        ArchTag,
        ElementA_,
        LayoutTagA,
        ElementB_,
        LayoutTagB,
        ElementC_,
        LayoutTagC,
        ElementBias>::TensorL0C;

    template <class TensorC>
    using CopyL0CToDst =
        CopyL0CToUBTla<ArchTag, TensorL0C, TensorC, CopyMode, DEQUANT_GRANULARITY, ReluEnable>;
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Common::Tile

#endif // CATLASS_GEMM_TILE_ASCEND950_COPY_L0C_TO_UB_950_HPP
