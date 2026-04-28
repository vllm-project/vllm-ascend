/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile_copy_policy.h
 * \brief
 */
#ifndef MATMUL_TILE_TILE_COPY_POLICY_H
#define MATMUL_TILE_TILE_COPY_POLICY_H

#include "../utils/arch.h"

namespace Act {
namespace Gemm {
namespace Tile {
struct CopyWithParams {};                      ///< Copy policy with additional parameters
struct CopyOutSplitMWithParams {};             ///< Copy policy for splitting output along the M dimension with parameters
struct CopyOutSplitNWithParams {};             ///< Copy policy for splitting output along the N dimension with parameters
struct CopyWithLayout {};                      ///< Copy policy with specific layout considerations
struct CopyEnUnitFlagWithLayout {};            ///< Copy policy with layout and unit flag considerations
struct CopySparseWithLayout {};                ///< Copy policy for sparse data with layout considerations
struct CopyNoGmIn {};                          ///< Copy policy excluding global memory input
struct CopyBasedBaseK {};                      ///< Copy policy based on base K dimension
struct CopyInAndCopyOutSplitMWithParams {};    ///< Copy policy for splitting input and output along the M dimension with parameters

/**
 * @struct Copy
 * @brief Primary template class for copy policies
 * @param [in] ArchTag: architecture tag
 * @param [in] DispatchPolicy: dispatch policy
 * @param [in] DataType: data type
 * @param [in] DstTrait: trait of destination tensor
 * @param [in] SrcTrait: trait of source tensor
 * @param [in] Enable: additional template parameter, default is void
 * @param [in] cfg: configuration settings, default is CFG_NORM
 */
template <class ArchTag, class DispatchPolicy, class DataType, class DstTrait, class SrcTrait, typename Enable = void,
          const auto& cfg = CFG_NORM>
struct Copy {};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
