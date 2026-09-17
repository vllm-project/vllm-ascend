/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILE_COPY_TLA_HPP
#define TILE_COPY_TLA_HPP

#include "../../../attn_infra/base_defs.hpp"

namespace NpuArch::Gemm::Tile {
template <
    class ArchTag,
    class TensorSrc,
    class TensorDst,
    class Enable = void
>
struct TileCopyTla {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileCopyTla, can not find the specialization.");
};
} // namespace NpuArch::Gemm::Tile

#endif // GEMM_TILE_TILE_COPY_TLA_HPP