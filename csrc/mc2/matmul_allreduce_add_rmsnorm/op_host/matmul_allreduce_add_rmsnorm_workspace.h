/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#ifndef MATMUL_ALLREDUCE_ADD_RMSNORM_WORKSPACE_H
#define MATMUL_ALLREDUCE_ADD_RMSNORM_WORKSPACE_H

#include <cstdint>

#pragma once
const constexpr uint32_t ALIGN_BYTES = 512;
const constexpr int32_t INT8_ELE_SIZE = 1;
const constexpr int32_t FP_BF_16_ELE_SIZE = 2;

enum CoCDataTypeDesc : int {
    COC_DATA_TYPE_UNDEFINED = -1,
    FP16FP16_FP32_FP16 = 0,
    BF16BF16_FP32_BF16 = 1,
    INT8INT8_INT32_FP16 = 2,
    INT8INT8_INT32_BF16 = 3,
    FP16INT8_INT32_FP16 = 4,
    BF16INT8_INT32_BF16 = 5,
    FP16INT8_FP32_FP16 = 6,
    BF16INT8_FP32_BF16 = 7,
    FP16INT4_FP32_FP16 = 8,
    BF16INT4_FP32_BF16 = 9,
    COC_DATA_TYPE_DESC_MAX = 10,
};

const std::map<CoCDataTypeDesc, int32_t> COC_TYPE2ELE_SIZE = {
    {FP16FP16_FP32_FP16, FP_BF_16_ELE_SIZE},
    {BF16BF16_FP32_BF16, FP_BF_16_ELE_SIZE},
    {INT8INT8_INT32_FP16, INT8_ELE_SIZE},
    {INT8INT8_INT32_BF16, INT8_ELE_SIZE},
    {FP16INT8_INT32_FP16, INT8_ELE_SIZE},
    {BF16INT8_INT32_BF16, INT8_ELE_SIZE},
    {FP16INT8_FP32_FP16, FP_BF_16_ELE_SIZE},
    {BF16INT8_FP32_BF16, FP_BF_16_ELE_SIZE},
    {FP16INT4_FP32_FP16, FP_BF_16_ELE_SIZE},
    {BF16INT4_FP32_BF16, FP_BF_16_ELE_SIZE}
};

struct MatMulInfo {
    int64_t batchSize = 1;
    int64_t m = -1;
    int64_t n = -1;
    int64_t k = -1;
    bool transA = false;
    bool transB = false;
    bool withBias = false;
    bool isInt8 = false;
    bool weightNz = false;
};

struct WorkspaceDetail {
    int64_t matrixActivationSize{0};
    int64_t matrixWeightSize{0};
    int64_t matrixIntermediateSize{0};
    int64_t formatDequantParamSize{0};

    int64_t GetSize() const
    {
        return matrixActivationSize + matrixWeightSize + matrixIntermediateSize + formatDequantParamSize;
    }
};

#endif