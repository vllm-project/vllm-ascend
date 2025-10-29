/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP
#define CATLASS_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {

template <
    class ArchTag,
    class GmType
>

struct MatrixCopyGmToUB
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to UB, can not find the specialization.");
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
/// Matrix A confirm
template <class Element>
struct MatrixCopyGmToUB<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>>
{
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element); 

    // Mehtods

    CATLASS_DEVICE
    MatrixCopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t m_catlassual = layoutSrc.shape(0);
        uint32_t n_catlassual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t stride = layoutSrc.stride(0);

        AscendC::DataCopyParams params;
        if ((n_catlassual % ELE_NUM_PER_C0 == 0) && (stride % ELE_NUM_PER_C0 == 0) && (stride < STRIDE_LIMIT))
        {
            params.blockCount = m_catlassual;
            params.blockLen = CeilDiv(n_catlassual, ELE_NUM_PER_C0);
            params.srcStride = (stride - n_catlassual) / ELE_NUM_PER_C0;
            params.dstStride = (n_round - n_catlassual) / ELE_NUM_PER_C0;
            AscendC::DataCopy(
                dstTensor,
                srcTensor,
                params);
        }
        else if ((n_catlassual % ELE_NUM_PER_C0 == 0) && (stride * ELE_NUM_PER_C0 < STRIDE_LIMIT))
        {
            uint32_t counts = m_catlassual / ELE_NUM_PER_C0;
            uint32_t remain = m_catlassual % ELE_NUM_PER_C0;
            if (counts > 0)
            {
                params.blockCount = counts;
                params.blockLen = CeilDiv(n_catlassual, ELE_NUM_PER_C0);
                params.srcStride = (ELE_NUM_PER_C0 * stride - n_catlassual) / ELE_NUM_PER_C0;
                params.dstStride = (ELE_NUM_PER_C0 * n_round - n_catlassual) / ELE_NUM_PER_C0;
                for (uint32_t i = 0; i < ELE_NUM_PER_C0; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[i * n_round],
                        srcTensor[i * stride],
                        params);
                }
            }
            if (remain > 0)
            {
                params.blockCount = 1;
                params.blockLen = CeilDiv(n_catlassual, ELE_NUM_PER_C0);
                params.srcStride = 0;
                params.dstStride = 0;
                for (uint32_t i = 0; i < remain; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[counts * n_round * ELE_NUM_PER_C0 + i * n_round],
                        srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride],
                        params);
                }
            }
        }
        else
        {
            params.blockCount = 1;
            params.blockLen = CeilDiv(n_catlassual, ELE_NUM_PER_C0);
            params.srcStride = 0;
            params.dstStride = 0;
            for (uint32_t i = 0; i < m_catlassual; i++)
            {
                AscendC::DataCopy(
                    dstTensor[i * n_round],
                    srcTensor[i * stride],
                    params);
            }
        }
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and ColumnMajor out.
/// Matrix A confirm
template <class Element>
struct MatrixCopyGmToUB<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>>
{
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    MatrixCopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t m_catlassual = layoutSrc.shape(0);
        uint32_t n_catlassual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t stride = layoutSrc.stride(1); 

        AscendC::DataCopyParams params;
        if ((m_catlassual % ELE_NUM_PER_C0 == 0) && (stride % ELE_NUM_PER_C0 == 0) && (stride < STRIDE_LIMIT))
        {
            params.blockCount = n_catlassual;
            params.blockLen = CeilDiv(m_catlassual, ELE_NUM_PER_C0);
            params.srcStride = (stride - m_catlassual) / ELE_NUM_PER_C0;
            params.dstStride = (m_round - m_catlassual) / ELE_NUM_PER_C0;
            AscendC::DataCopy(
                dstTensor,
                srcTensor,
                params);
        }
        else if ((m_catlassual % ELE_NUM_PER_C0 == 0) && (stride * ELE_NUM_PER_C0 < STRIDE_LIMIT))
        {
            uint32_t counts = n_catlassual / ELE_NUM_PER_C0;
            uint32_t remain = n_catlassual % ELE_NUM_PER_C0;
            if (counts > 0)
            {
                params.blockCount = counts;
                params.blockLen = CeilDiv(m_catlassual, ELE_NUM_PER_C0);
                params.srcStride = (ELE_NUM_PER_C0 * stride - m_catlassual) / ELE_NUM_PER_C0;
                params.dstStride = (ELE_NUM_PER_C0 * m_round - m_catlassual) / ELE_NUM_PER_C0;
                for (uint32_t i = 0; i < ELE_NUM_PER_C0; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[i * m_round],
                        srcTensor[i * stride],
                        params);
                }
            }
            if (remain > 0)
            {
                params.blockCount = 1;
                params.blockLen = CeilDiv(m_catlassual, ELE_NUM_PER_C0);
                params.srcStride = 0;
                params.dstStride = 0;
                for (uint32_t i = 0; i < remain; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[counts * m_round * ELE_NUM_PER_C0 + i * m_round],
                        srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride],
                        params);
                }
            }
        }
        else
        {
            params.blockCount = 1;
            params.blockLen = CeilDiv(m_catlassual, ELE_NUM_PER_C0);
            params.srcStride = 0;
            params.dstStride = 0;
            for (uint32_t i = 0; i < n_catlassual; i++)
            {
                AscendC::DataCopy(
                    dstTensor[i * m_round],
                    srcTensor[i * stride],
                    params);
            }
        }
    }
};

} // namespace Catlass::Gemv::Tile

#endif // CATLASS_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP
