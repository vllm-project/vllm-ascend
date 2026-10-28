/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_MLA_CHECKER_CONTEXT_H
#define SPARSE_MLA_CHECKER_CONTEXT_H

#include <cstdint>
#include <vector>
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
namespace sparse_mla_checker {

enum class OperatorVariant : uint32_t {
    SPARSE = 0,
    MIXED_QUANT = 1,
    QUANT = 2,
};

enum class Layout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2,
};

struct TensorParam {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::Shape *shape = nullptr;
    bool present = false;
};

struct CheckContext {
    const char *opName = nullptr;
    OperatorVariant variant = OperatorVariant::SPARSE;

    TensorParam q;
    TensorParam oriKv;
    TensorParam cmpKv;
    TensorParam qDescale;
    TensorParam oriKvDescale;
    TensorParam cmpKvDescale;
    TensorParam oriSparseIndices;
    TensorParam cmpSparseIndices;
    TensorParam oriBlockTable;
    TensorParam cmpBlockTable;
    TensorParam cuSeqlensQ;
    TensorParam cuSeqlensOriKv;
    TensorParam cuSeqlensCmpKv;
    TensorParam sequsedQ;
    TensorParam sequsedOriKv;
    TensorParam sequsedCmpKv;
    TensorParam cmpResidualKv;
    TensorParam oriTopkLength;
    TensorParam cmpTopkLength;
    TensorParam sinks;
    TensorParam metadata;
    TensorParam attentionOut;
    TensorParam softmaxLse;

    Layout qLayout = Layout::BSND;
    Layout kvLayout = Layout::BSND;
    int64_t bSize = 0;
    int64_t qNumHeads = 0;
    int64_t kvNumHeads = 0;
    int64_t qSeqSize = 0;
    int64_t qTotalSize = 0;
    int64_t qHeadDim = 0;
    int64_t oriKvHeadDim = 0;
    int64_t cmpKvHeadDim = 0;
    int64_t oriBlockSize = 0;
    int64_t cmpBlockSize = 0;
    int64_t oriTopk = 0;
    int64_t cmpTopk = 0;

    int64_t quantMode = 0;
    int64_t ropeHeadDim = 0;
    float softmaxScale = 1.0F;
    int64_t cmpRatio = 0;
    int64_t oriMaskMode = 0;
    int64_t cmpMaskMode = 0;
    int64_t oriWinLeft = -1;
    int64_t oriWinRight = -1;
    int64_t topkValueMode = 1;
    bool returnSoftmaxLse = false;

    std::vector<int64_t> oriKvStrides;
    std::vector<int64_t> cmpKvStrides;
};

} // namespace sparse_mla_checker
} // namespace optiling

#endif // SPARSE_MLA_CHECKER_CONTEXT_H
