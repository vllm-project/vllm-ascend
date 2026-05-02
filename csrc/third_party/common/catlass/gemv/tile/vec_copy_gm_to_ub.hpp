/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
#define CATLASS_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

constexpr uint32_t STRIDE_LIMIT = 65536;

namespace Catlass::Gemv::Tile {

template <
    class ArchTag_,
    class VType_
>
struct VecCopyGmToUB {
    using Element = typename VType_::Element;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    VecCopyGmToUB() {};

    CATLASS_DEVICE
    void operator()(
    AscendC::LocalTensor<Element> dstTensor,
    AscendC::GlobalTensor<Element> srcTensor,
    uint32_t len
) {
    AscendC::DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = CeilDiv(len, ELE_NUM_PER_C0);
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::DataCopy(
        dstTensor,
        srcTensor,
        params);
}
};
} // namespace Catlass::Gemv::Tile

#endif // CATLASS_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
