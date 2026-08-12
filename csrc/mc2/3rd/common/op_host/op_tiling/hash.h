/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hash.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_HASH_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_HASH_H_

#include <cstdint>

namespace Ops {
namespace Transformer {
constexpr uint32_t kHashSeed = 271828;
uint32_t MurmurHash(const void *src, uint32_t len, uint32_t seed = kHashSeed);
} // namespace Transformer
} // namespace Ops
#endif // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_HASH_H_
