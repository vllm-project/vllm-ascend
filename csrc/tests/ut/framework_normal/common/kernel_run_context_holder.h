/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRANSFORMER_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_HOLDER_H_
#define TRANSFORMER_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_HOLDER_H_

#include "op_tiling_parse_context_builder.h"
#include "op_tiling_context_builder.h"
#include "op_infer_shape_context_builder.h"
#include "op_infer_shape_range_context_builder.h"
#include "op_infer_datatype_context_builder.h"

namespace gert {
class KernelRunContextHolder {
public:
    KernelRunContextHolder() = default;
    ~KernelRunContextHolder() = default;

    KernelRunContextHolder &operator=(KernelRunContextHolder &&holder)
    {
        context_ = holder.context_;
        opType_ = std::move(holder.opType_);
        outputShapes_ = std::move(holder.outputShapes_);
        inputTensors_ = std::move(holder.inputTensors_);
        outputTensors_ = std::move(holder.outputTensors_);
        inputMinTensors_ = std::move(holder.inputMinTensors_);
        inputMaxTensors_ = std::move(holder.inputMaxTensors_);
        inputTensorRanges_ = std::move(holder.inputTensorRanges_);

        tilingContextHolder_ = std::move(holder.tilingContextHolder_);
        tilingParseContextHolder_ = std::move(holder.tilingParseContextHolder_);
        inferShapeContextHolder_ = std::move(holder.inferShapeContextHolder_);
        inferShapeRangeContextHolder_ = std::move(holder.inferShapeRangeContextHolder_);
        inferDataTypeContextHolder_ = std::move(holder.inferDataTypeContextHolder_);
        return *this;
    }
    KernelRunContextHolder(KernelRunContextHolder &&holder)
    {
        KernelRunContextHolder::operator=(std::move(holder));
    }

    void SetContext(void *context)
    {
        context_ = context;
    }

    template <typename T>
    T *GetContext()
    {
        return static_cast<T *>(context_);
    }

protected:
    // 需要在holder里存储部分数据，避免这些数据在Context析构前就被释放
    void *context_ = nullptr;
    std::string opType_;
    std::vector<StorageShape> outputShapes_;
    std::vector<Tensor> inputTensors_;
    std::vector<Tensor> outputTensors_;
    std::vector<Tensor> inputMinTensors_;
    std::vector<Tensor> inputMaxTensors_;
    std::vector<Range<Tensor>> inputTensorRanges_;

    // 这不是很好的实现，GE没有提供构造holder的接口，直接取private成员变量耦合更强，只能多存几份了
    ContextHolder<TilingContext> tilingContextHolder_;
    ContextHolder<TilingParseContext> tilingParseContextHolder_;
    ContextHolder<InferShapeContext> inferShapeContextHolder_;
    ContextHolder<InferShapeRangeContext> inferShapeRangeContextHolder_;
    ContextHolder<InferDataTypeContext> inferDataTypeContextHolder_;
};
} // namespace gert
#endif // TRANSFORMER_TESTS_UT_COMMON_KERNEL_RUN_CONTEXT_HOLDER_H_
