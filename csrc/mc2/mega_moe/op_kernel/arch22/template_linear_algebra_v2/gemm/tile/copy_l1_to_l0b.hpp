/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP
#define CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP

#include "../../../template_linear_algebra_v2/catlass.hpp"
#include "../../../template_linear_algebra_v2/numeric_size.hpp"
#include "../../../template_linear_algebra_v2/arch/arch.hpp"
#include "../../../template_linear_algebra_v2/layout/layout.hpp"
#include "../../../template_linear_algebra_v2/gemm/gemm_type.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////////////
/// new add gemm
template<class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>>{
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for(uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++){  // K N
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template<class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<float, layout::zZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<float, layout::nZ, AscendC::TPosition::B2>>{
    using Element = float;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1))) - 1;
        for(uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++){ // K N
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};


template<class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B2>>{
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0B(){}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B2>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};
/////////////////////////////////////////////

////////////////////////////////////////////
/// new add gemv
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(3); i++)
        {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(3)], srcTensor[i * layoutSrc.stride(3)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<Element, layout::nN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>>
{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = layoutDst.shape(1) * layoutDst.shape(3);
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
    };
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<float, layout::nN, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<float, layout::zN, AscendC::TPosition::B2>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nN;
    using Element = float;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)) - 1;

        for (uint32_t i = 0; i < CeilDiv<2 * ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
        {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3) * 2],
                srcTensor[i * layoutSrc.stride(3)],
                loadDataParams);
        }
    };
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::B1>, Catlass::Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::B2>>{
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::nZ;
    using Element = int8_t;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)); i++)
        {
            AscendC::LoadDataWithTranspose(
                dstTensor[i * layoutDst.stride(3)],
                srcTensor[i * layoutSrc.stride(3) * 2],
                loadDataParams);
        }
    }
};
////////////////////////////////////////////

/// Partial specialization for int8_t, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<int8_t, layout::zN, AscendC::TPosition::A1>> {
    using Element = int8_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 2;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

/// Partial specialization for float, zN in and nZ out.
template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<float, layout::zN, AscendC::TPosition::A1>> {
    using Element = float;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        constexpr uint8_t PAD_LIST[4] = {0, 0, 0, 0};
        uint16_t l1K = layoutSrc.shape(0) * layoutSrc.shape(1);
        uint16_t l1N = layoutSrc.shape(2) * layoutSrc.shape(3);
        uint16_t l0K = layoutDst.shape(0) * layoutDst.shape(1);
        uint16_t l0N = layoutDst.shape(2) * layoutDst.shape(3);
        // K, N need to be 16 aligned for f32
        uint16_t l1KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1K);
        uint16_t l1NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l1N);
        uint16_t l0KAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0K);
        uint16_t l0NAlign = RoundUp<C0_NUM_PER_FRACTAL>(l0N);
        AscendC::SetFmatrix(1, l1KAlign, PAD_LIST, AscendC::FmatrixMode::FMATRIX_RIGHT);
        static constexpr AscendC::IsResetLoad3dConfig config = {false, false};
        AscendC::LoadData3DParamsV2<Element> loadDataParams;
        loadDataParams.kExtension = l0NAlign;
        loadDataParams.mExtension = l0KAlign;
        loadDataParams.channelSize = l1NAlign;
        loadDataParams.fMatrixCtrl = true;

        AscendC::LoadData<Element, config>(dstTensor, srcTensor, loadDataParams);
    }
};

/// Partial specialization for zN in and nZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<AscendC::int4b_t, layout::zN, AscendC::TPosition::A1>> {
    using Element = AscendC::int4b_t;
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL / 4;
        loadDataParams.dstGap = 3;
        loadDataParams.dstFracGap = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)],
                                           srcTensor[i * layoutSrc.stride(1) * 4],
                                           loadDataParams);
        }
    }
};

/// Partial specialization for nZ in and nZ out. (Transpose B)
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0B() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        if (layoutSrc.shape(3) == layoutDst.shape(3)) {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1) * layoutDst.shape(3));
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        } else {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
            loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;

            for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }

    }
};


} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L1_TO_L0B_HPP