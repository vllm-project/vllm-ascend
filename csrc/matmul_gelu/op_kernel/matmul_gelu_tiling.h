/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef ASCENDC_OP_KERNEL_MATMUL_GELU_TILING_H
#define ASCENDC_OP_KERNEL_MATMUL_GELU_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"


struct MatmulGeluTilingData
{
        uint32_t m;
        uint32_t n;
        uint32_t k;
        bool transB;
};


#endif