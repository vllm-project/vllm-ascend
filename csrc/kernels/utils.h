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
#include "kernel_type.h"
namespace vllm_ascend {

template <typename scalar_t> struct AccType;

template <> struct AccType<bfloat16_t> {
    using type = float;
};

template <> struct AccType<half> {
    using type = half;
};

template <> struct AccType<float> {
    using type = float;
};

template <> struct AccType<int8_t> {
    using type = int;
};

template <typename scalar_t>
__aicore__ inline void local_mem_copy(AscendC::LocalTensor<scalar_t> dst, AscendC::LocalTensor<scalar_t> src, int size)
{
    constexpr int loadSize = 256 / sizeof(scalar_t);
    int loopCnt = size / loadSize;
    int tailSize = size % loadSize;
    if (loopCnt)
        AscendC::Copy(dst, src, loadSize, loopCnt, {1, 1, 8, 8});
    AscendC::Copy(dst[loopCnt * loadSize], src[loopCnt * loadSize], tailSize, 1, {1, 1, 8, 8});
}
} // namespace vllm_ascend