/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_VEC_COPY_UB_TO_GM_HPP
#define CATLASS_GEMV_TILE_VEC_COPY_UB_TO_GM_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace Catlass::Gemv::Tile {

template <
    class ArchTag,
    class GmType,
    bool is_atoadd = false
>
struct VecCopyUBToGm
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy UB to gm, can not find the specialization.");
};

template <class Element>
struct VecCopyUBToGm<Arch::AtlasA2, Gemm::GemmType<Element, layout::VectorLayout>>
{
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element); 

    // Mehtods

    CATLASS_DEVICE
    VecCopyUBToGm() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        layout::VectorLayout const &layoutDst,
        layout::VectorLayout const &layoutSrc
    )
    {
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = layoutDst.shape(0) * sizeof(Element);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        AscendC::DataCopyPad(
            dstTensor,
            srcTensor,
            params
        );
    }
};

template <class Element>
struct VecCopyUBToGm<Arch::AtlasA2, Gemm::GemmType<Element, layout::VectorLayout>,true>
{
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element); 

    // Mehtods

    CATLASS_DEVICE
    VecCopyUBToGm() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        layout::VectorLayout const &layoutDst,
        layout::VectorLayout const &layoutSrc
    )
    {
        AscendC::SetAtomicAdd<Element>();
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = layoutDst.shape(0) * sizeof(Element);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        AscendC::DataCopyPad(
            dstTensor,
            srcTensor,
            params
        );
        AscendC::SetAtomicNone();
    }
};

} // namespace Catlass::Gemv::Tile

#endif // CATLASS_GEMV_TILE_VEC_COPY_UB_TO_GM_HPP
