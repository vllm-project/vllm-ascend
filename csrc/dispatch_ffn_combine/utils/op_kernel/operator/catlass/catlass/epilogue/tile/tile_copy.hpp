/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_TILE_TILE_COPY_HPP
#define CATLASS_EPILOGUE_TILE_TILE_COPY_HPP

#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"

namespace Catlass::Epilogue::Tile {

template <
    /// Tag indicating architecture
    class ArchTag,
    class... Args
>
struct TileCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported tile copy, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for C matrix operand
    class CType,
    /// GemmType for X matrix operand
    class XType,
    /// GemmType for D matrix operand
    class DType
>
struct TileCopy<ArchTag, CType, XType, DType> {
    using ElementC = typename CType::Element;
    using ElementX = typename XType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbX = CopyGm2Ub<ArchTag, XType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};

template <
    class ArchTag,
    class CType,
    class XType,
    class YType,
    class DType
>
struct TileCopy<ArchTag, CType, XType, YType, DType> {
    using ElementC = typename CType::Element;
    using ElementX = typename XType::Element;
    using ElementY = typename YType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbX = CopyGm2Ub<ArchTag, XType>;
    using CopyGmToUbY = CopyGm2Ub<ArchTag, YType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};

template <
    class ArchTag,
    class CType,
    class XType,
    class YType,
    class DType
>
struct TileCopyBf16 {
    using ElementC = typename CType::Element;
    using ElementX = bfloat16_t;
    using ElementY = bfloat16_t;
    using ElementD = bfloat16_t;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbX = CopyGm2Ub<ArchTag, Gemm::GemmType<bfloat16_t, typename XType::Layout>>;
    using CopyGmToUbY = CopyGm2Ub<ArchTag, Gemm::GemmType<bfloat16_t, typename YType::Layout>>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, Gemm::GemmType<bfloat16_t, typename DType::Layout>>;
};

template <
    class ArchTag,
    class CType,
    class ScaleType,
    class PerTokenScaleType,
    class DType
>
struct TileCopyPerTokenDequant {
    using ElementC = typename CType::Element;
    using ElementScale = typename ScaleType::Element;
    using ElementPerTokenScale = typename PerTokenScaleType::Element;
    using ElementD = typename DType::Element;

    using CopyGmToUbC = CopyGm2Ub<ArchTag, CType>;
    using CopyGmToUbScale = CopyGm2Ub<ArchTag, ScaleType>;
    using CopyGmToUbPerTokenScale = CopyPerTokenScale2Ub<ArchTag, PerTokenScaleType>;
    using CopyUbToGmD = CopyUb2Gm<ArchTag, DType>;
};
} // namespace Catlass::Epilogue::Tile

#endif  // CATLASS_EPILOGUE_TILE_TILE_COPY_HPP
