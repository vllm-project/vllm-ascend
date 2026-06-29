/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_L1_HPP
#define CATLASS_GEMM_TILE_COPY_GM_TO_L1_HPP

#include "../../../template_linear_algebra_v2/catlass.hpp"
#include "../../../template_linear_algebra_v2/numeric_size.hpp"
#include "../../../template_linear_algebra_v2/arch/arch.hpp"
#include "../../../template_linear_algebra_v2/layout/layout.hpp"
#include "../../../template_linear_algebra_v2/gemm/gemm_type.hpp"

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1IntervalDataCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1GMMPTD {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1DynamicOptimized {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1DynamicOptimized() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.shape(0) <= 16) {
            // If the number of matrix rows is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            for (int i = 0; i < layoutSrc.shape(0); ++i) {
                AscendC::DataCopyParams dataCopyParams(
                    CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                    layoutDst.shape(2) / ELE_NUM_PER_C0,
                    0,
                    (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0
                );
                AscendC::DataCopy(
                    dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            if (layoutSrc.stride(0) < STRIDE_LIMIT) {
                if (layoutSrc.shape(1) != ELE_NUM_PER_C0 || layoutSrc.stride(0) != ELE_NUM_PER_C0) {
                    intriParams.nValue = layoutSrc.shape(0);
                    intriParams.srcDValue = layoutSrc.stride(0);
                    intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                } else {
                    // If the matrix has ELE_NUM_PER_C0, columns and a stride of ELE_NUM_PER_C0, it follows a row-major
                    // layout in L1, allowing the use of standard contiguous DataCopy interface for more efficient
                    // transfers.
                    AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.shape(0) * layoutSrc.shape(1));
                }
            } else {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
                }
            }
        }
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1DynamicOptimized() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.shape(1) <= 16) {
            // If the number of matrix cols is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            for (int i = 0; i < layoutSrc.shape(1); ++i) {
                AscendC::DataCopyParams dataCopyParams(
                    CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)),
                    layoutDst.shape(0) / ELE_NUM_PER_C0,
                    0,
                    (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0
                );
                AscendC::DataCopy(
                    dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
            }
        } else {
            if (layoutSrc.stride(1) < STRIDE_LIMIT) {
                if (layoutSrc.shape(0) != ELE_NUM_PER_C0 || layoutSrc.stride(1) != ELE_NUM_PER_C0) {
                    intriParams.nValue = layoutSrc.shape(1);
                    intriParams.srcDValue = layoutSrc.stride(1);
                    intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                } else {
                    // If the matrix has ELE_NUM_PER_C0, rows and a stride of ELE_NUM_PER_C0, it follows a col-major
                    // layout in L1, allowing the use of standard contiguous DataCopy interface for more efficient
                    // transfers.
                    AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.shape(0) * layoutSrc.shape(1));
                }
            } else {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
                }
            }
        }
    }
};

/// Partial specialization for AtlasA2, zN in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::zN>> : 
    public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::zN>> {};

/// Partial specialization for AtlasA2, nZ in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::nZ>> : 
    public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::nZ>> {};

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingRowMajor>> : 
    public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingRowMajor>> {};

/// Partial specialization for AtlasA2, PaddingColumnMajor in and nZ out.
template <class Element>
struct CopyGmToL1DynamicOptimized<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingColumnMajor>> : 
    public CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {};

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1GMMPTD<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1GMMPTD() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.shape(0) == 1) {
            // If the number of matrix rows is 1, the regular interval-based DataCopy interface can be used instead of
            // the ND2NZ DataCopy interface, resulting in higher transfer efficiency.
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0);
            AscendC::DataCopy(dstTensor, srcTensor, dataCopyParams);
        } else {
            if (layoutSrc.stride(0) < STRIDE_LIMIT) {
                if (layoutSrc.shape(1) != ELE_NUM_PER_C0 || layoutSrc.stride(0) != ELE_NUM_PER_C0) {
                    intriParams.nValue = layoutSrc.shape(0);
                    intriParams.srcDValue = layoutSrc.stride(0);
                    intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                    AscendC::DataCopy(dstTensor, srcTensor, intriParams);
                } else {
                    // If the matrix has ELE_NUM_PER_C0 columns and a stride of ELE_NUM_PER_C0, it follows a row-major
                    // layout in L1, allowing the use of the standard contiguous DataCopy interface for more efficient
                    // transfers.
                    AscendC::DataCopy(dstTensor, srcTensor, layoutSrc.shape(0) * layoutSrc.shape(1));
                }
            } else {
                intriParams.nValue = 1;
                intriParams.srcDValue = 0;
                intriParams.dstNzNStride = 0;
                for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                    AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
                }
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};

////////////////////////////////////////
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
/// Partial specialization for AtlasA2, half, RowMajor in and zN out.
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, PaddingRowMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.orgShape(1), layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, ColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)),
                layoutDst.shape(0) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, PaddingColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1IntervalDataCopy() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                CeilDiv(layoutSrc.orgShape(0), layoutDst.shape(0)),
                layoutDst.shape(0) / ELE_NUM_PER_C0,
                0,
                (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(2)], dataCopyParams);
        }
    }
};

/// new add gemm
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(0);
        uint32_t ndNum = layoutSrc.shape(0) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(0) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(1);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;  //`

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(1)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(1) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(0);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nN;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(3);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};
////////////////////////////////////////

///////////////////////////////////////
/// new add gemv, VectorLayout -> zN
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::VectorLayout>, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;
        intriParams.nValue = 1;
        intriParams.srcDValue = layoutSrc.shape(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};


template <class ArchTag, class Element> 
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>, 
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> { 
    using LayoutDst = layout::VectorLayout; 
    using LayoutSrc = layout::VectorLayout; 


    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value; 


    // Mehtods 


    CATLASS_DEVICE 
    CopyGmToL1() {}; 


    CATLASS_DEVICE 
    void operator()( 
        AscendC::LocalTensor<Element> const &dstTensor, 
        AscendC::GlobalTensor<Element> const &srcTensor, 
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc) 
    { 
        AscendC::DataCopyParams intriParams; 
        intriParams.blockCount = 1; 
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C0; 
        intriParams.srcStride = 0; 
        intriParams.dstStride = 0; 
        AscendC::DataCopy(dstTensor, srcTensor, intriParams); 
    } 
};


template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::NDC1HWC0, AscendC::TPosition::GM>> {
    using LayoutDst = layout::NDC1HWC0;
    using LayoutSrc = layout::NDC1HWC0;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        const static uint64_t MAX_UINT16 = 65535;

        uint32_t cin1LoadL1 = layoutDst.orgShape(2);
        uint32_t hiLoadL1 = layoutDst.orgShape(3);

        uint32_t dilationD = layoutSrc.orgShape(1);
        uint32_t OriC1 = layoutSrc.orgShape(2);
        uint32_t OriH = layoutSrc.orgShape(3);
        uint32_t OriW = layoutSrc.orgShape(4);
        uint32_t OriK0 = layoutSrc.orgShape(5);

        uint64_t dataCopyLoop = CeilDiv(cin1LoadL1, OriC1);
        uint64_t dataCopySubLoop = 0;
        uint64_t blockCount = OriC1;
        bool srcStrideBeyondMaxU16 = false;
        if (OriH * OriW - hiLoadL1 * OriW > MAX_UINT16) {
            dataCopySubLoop = dataCopyLoop > 0 ? cin1LoadL1 / dataCopyLoop : 0;
            blockCount = 1;
            srcStrideBeyondMaxU16 = true;
        }

        uint64_t aL1GmOffset = 0;
        uint64_t aL1Offset = 0;
        if (cin1LoadL1 > OriC1 || srcStrideBeyondMaxU16) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = hiLoadL1 * OriW;
            repeatParams.srcStride = OriW * OriH - repeatParams.blockLen;
            repeatParams.dstStride = 0;
            for (uint64_t i = 0; i < dataCopyLoop; i++) {
                if (srcStrideBeyondMaxU16) {
                    uint64_t aL1GmSubOffset = aL1GmOffset;
                    uint64_t aL1SubOffset = aL1Offset;
                    for (uint64_t j = 0; j < dataCopySubLoop; j++) {
                        AscendC::DataCopy<Element>(dstTensor[aL1SubOffset], srcTensor[aL1GmSubOffset], repeatParams);
                        aL1GmSubOffset += OriH * OriW * OriK0;
                        aL1SubOffset += hiLoadL1 * OriW * OriK0;
                    }
                } else {
                    AscendC::DataCopy<Element>(dstTensor[aL1Offset], srcTensor[aL1GmOffset], repeatParams);
                }
                aL1GmOffset += dilationD * OriC1 * OriH * OriW * OriK0;
                aL1Offset += OriC1 * hiLoadL1 * OriW * OriK0;
            }
        } else {
            repeatParams.blockCount = cin1LoadL1;
            repeatParams.blockLen = hiLoadL1 * OriW;
            repeatParams.srcStride = OriW * OriH - repeatParams.blockLen;
            repeatParams.dstStride = 0;
            AscendC::DataCopy<Element>(dstTensor[aL1Offset], srcTensor[aL1GmOffset], repeatParams);
            aL1Offset += cin1LoadL1 * hiLoadL1 * OriW * OriK0;
        }
    }
private:
    uint64_t aL1GmOffset = 0;
    AscendC::DataCopyParams repeatParams;
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::KDC1KHKWN1N0C0, AscendC::TPosition::GM>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::KDC1KHKWN1N0C0;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t currentNBL1 = layoutDst.orgShape(1);
        uint32_t currentKBL1 = layoutDst.orgShape(0);

        uint32_t N1 = layoutSrc.shape(3);
        uint32_t N0 = layoutSrc.shape(2);
        uint32_t C0 = layoutSrc.shape(0);
        uint32_t OriCoAlign = N1 * N0;

        const static uint32_t LOAD2D_MAX_REPEAT_TIMES = 255;

        if (currentNBL1 >= OriCoAlign) {
            uint32_t repeatTimes = (currentKBL1 * currentNBL1) / (N0 * C0);
            if(repeatTimes > LOAD2D_MAX_REPEAT_TIMES) {
                repeatParams.blockCount = 1;
                repeatParams.srcStride = 0;
                repeatParams.blockLen = CeilDiv(currentKBL1 * currentNBL1, C0);
                AscendC::DataCopy<Element>(dstTensor, srcTensor, repeatParams);
            } else {
                AscendC::LoadData2DParams loadData2dParams;
                loadData2dParams.srcStride = 1;
                loadData2dParams.repeatTimes = repeatTimes;
                AscendC::LoadData<Element>(dstTensor, srcTensor, loadData2dParams);
            }
        } else {
            repeatParams.blockCount = currentKBL1 / C0;
            repeatParams.blockLen = currentNBL1;
            repeatParams.srcStride = N1 * N0 - currentNBL1;
            AscendC::DataCopy<Element>(dstTensor, srcTensor, repeatParams);
        }
    }
private:
    AscendC::DataCopyParams repeatParams;
};

///////////////////////////////////////
/// new add gemv, ColumnMajor -> nN
template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::ColumnMajor>, Gemm::GemmType<Element, layout::nN, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nN;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;

                intriParams.dstNzMatrixStride = layoutDst.stride(3);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::RowMajor>, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Methods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};
/////////////////////////////////

/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        // There is an issue with AscenC in the data migration of nd2nz of type int4b_t: dValue needs to be passed
        // with the length of int8, and it is uncertain whether this will be fixed in subsequent versions.
        if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
                intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
            }
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }

    // layoutSrc must be the layout of one of the src matrices
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};
/////////////////////////////////

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <
    class Element
>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        // There is an issue with AscenC in the data migration of nd2nz of type int4b_t: dValue needs to be passed
        // with the length of int8, and it is uncertain whether this will be fixed in subsequent versions.
        if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            if constexpr (std::is_same_v<Element, AscendC::int4b_t>) {
                intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
            }
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

/// Partial specialization for zN in and zN out.
template <
    class ArchTag,
    class Element
>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::zN>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(1));
        uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(0));

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(3) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(3) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(3);
                uint64_t srcOffset = i * layoutSrc.stride(3);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};


/// Partial specialization for nZ in and nZ out.
template <
    class ArchTag,
    class Element
>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::nZ>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t blockCount = CeilDiv<ELE_NUM_PER_C0>(layoutSrc.orgShape(0));
        uint32_t blockLen = RoundUp<C0_NUM_PER_FRACTAL>(layoutSrc.orgShape(1));

        AscendC::DataCopyParams repeatParams;

        if (layoutSrc.stride(1) / ELE_NUM_PER_C0 < STRIDE_LIMIT) {
            repeatParams.blockCount = blockCount;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_C0 - blockLen;
            repeatParams.dstStride = layoutDst.stride(1) / ELE_NUM_PER_C0 - blockLen;
            AscendC::DataCopy(dstTensor, srcTensor, repeatParams);
        } else {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = blockLen;
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
            for (uint32_t i = 0; i < blockCount; i++) {
                uint64_t dstOffset = i * layoutDst.stride(1);
                uint64_t srcOffset = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], repeatParams);
            }
        }
    }
};

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(0);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and nZ out.
template <
    class Element
>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.orgShape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(1);
        intriParams.srcDValue = layoutSrc.stride(2);
        intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::RowMajor, AscendC::TPosition::A1>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;

    // Mehtods

    CATLASS_DEVICE
    CopyGmToL1() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t rows = layoutSrc.shape(0);
        uint32_t cols = layoutSrc.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(
                    rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride
                );
                DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                         srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    }
};

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_GM_TO_L1_HPP
