/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_L1_TO_L0A_HPP
#define CATLASS_GEMM_TILE_COPY_L1_TO_L0A_HPP

#include "../../catlass.hpp"
#include "../../layout/layout.hpp"
#include "../../gemm/gemm_type.hpp"


namespace Catlass::Gemm::Tile {

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

////////////////////////////////
/// new add gemm
template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>,
                   Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRCATLASSAL = BytesToBits(BYTE_PER_FRCATLASSAL) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {
    }

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor,
                    LayoutDst layoutDst, LayoutSrc layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRCATLASSAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRCATLASSAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Catlass::Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>,
                   Catlass::Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::A2>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {
    }

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor,
                    LayoutDst layoutDst, LayoutSrc layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRCATLASSAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)));
        ;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Catlass::Gemm::GemmType<float, layout::nN, AscendC::TPosition::A1>,
                   Catlass::Gemm::GemmType<float, layout::zZ, AscendC::TPosition::A2>> {
    using Element = float;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {
    }

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor,
                    LayoutDst layoutDst, LayoutSrc layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRCATLASSAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRCATLASSAL>(layoutSrc.orgShape(0)));
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;
        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRCATLASSAL>(layoutSrc.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1) * 2],
                                           loadDataParams);
        }
    }
};

template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Catlass::Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>,
                   Catlass::Gemm::GemmType<int8_t, layout::zZ, AscendC::TPosition::A2>> {
    using Element = int8_t;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A()
    {
    }

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> dstTensor, AscendC::LocalTensor<Element> srcTensor,
                    LayoutDst layoutDst, LayoutSrc layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2], srcTensor[i * layoutSrc.stride(1)],
                                           loadDataParams);
        }
    }
};
//////////////////////////////////////////

/// Partial specialization for zN in and zZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRCATLASSAL = BytesToBits(BYTE_PER_FRCATLASSAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::LocalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRCATLASSAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRCATLASSAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRCATLASSAL = BytesToBits(BYTE_PER_FRCATLASSAL) / SizeOfBits<Element>::value;

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::LocalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRCATLASSAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRCATLASSAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRCATLASSAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

/// Partial specialization for int8_t, nZ in and zZ out. (Transpose A)
template <class ArchTag>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<int8_t, layout::nZ, AscendC::TPosition::A1>> {
    using Element = int8_t;
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
    static constexpr uint32_t ELE_NUM_PER_FRCATLASSAL = BytesToBits(BYTE_PER_FRCATLASSAL) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyL1ToL0A() {};

    CATLASS_DEVICE
    void operator()(AscendC::LocalTensor<Element> const &dstTensor, AscendC::LocalTensor<Element> const &srcTensor,
                    LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2dTransposeParams loadDataParams;

        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)) - 1;

        for (uint32_t i = 0; i < CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadDataWithTranspose(dstTensor[i * layoutDst.stride(1) * 2], srcTensor[i * layoutSrc.stride(1)],
                                           loadDataParams);
        }
    }
};
} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_L1_TO_L0A_HPP
