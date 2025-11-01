/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP

#include "catlass/catlass.hpp"
namespace Catlass::Gemv::Block {

template <
    class DispatchPolicy,
    class... Args
>
struct BlockGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockGemv is not implemented for this DispatchPolicy");
};
}  // namespace Catlass::Gemv::Block

#include "catlass/gemv/block/block_gemv_aiv.hpp"
#include "catlass/gemv/block/block_gemv_aic.hpp"

#endif
