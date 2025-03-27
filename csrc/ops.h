/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>
#include "kernels/types.h"

namespace vllm_ascend {
  extern void rotary_embedding_impl(AscendType type, bool isNeox, void *stream, int64_t *positions, void *queryDst,
    void *keyDst, void *query, void *key, void *cosSinCache, const int rotDim,
    const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride,
    const int64_t dstKeyStride, const int numHeads, const int numKvHeads,
    const int headSize, const int64_t numTokens, const uint32_t loopCnt,
    uint32_t aivNum);
}