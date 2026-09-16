/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPERANGE_CONTEXT_FAKER_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPERANGE_CONTEXT_FAKER_H

#include <vector>
#include <string>

#include "kernel_run_context_holder.h"
#include "any_value.h"

namespace gert {
class InferShapeRangeContextFaker : public OpInferShapeRangeContextBuilder, public KernelRunContextHolder {
public:
    InferShapeRangeContextFaker() = default;
    InferShapeRangeContextFaker &operator=(InferShapeRangeContextFaker &&);
    InferShapeRangeContextFaker(InferShapeRangeContextFaker &&);

    InferShapeRangeContextFaker &SetOpType(const std::string opType);

    InferShapeRangeContextFaker &IrInputNum(size_t inputNum);

    InferShapeRangeContextFaker &NodeIoNum(size_t inputNum, size_t outputNum);

    InferShapeRangeContextFaker &IrInstanceNum(const std::vector<uint32_t> &instanceNum);

    InferShapeRangeContextFaker &NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                             ge::Format storageFormat, const gert::StorageShape &shape = {});

    InferShapeRangeContextFaker &NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                              ge::Format storageFormat);

    template <typename T>
    InferShapeRangeContextFaker &Attr(const std::string &attrName, T attr)
    {
    OpInferShapeRangeContextBuilder:
        AppendAttr(attr);
        return *this;
    }

    InferShapeRangeContextFaker &Attr(const std::string &attrName, const std::string &attr)
    {
        OpInferShapeRangeContextBuilder::AppendAttr(ge::AscendString(attr.c_str()));
        return *this;
    }

    InferShapeRangeContextFaker &InputTensors(const std::vector<Range<Tensor> *> &inputTensors);

    InferShapeRangeContextFaker &InputShapeRanges(const std::vector<Range<Shape> *> &inputShapeRanges);

    InferShapeRangeContextFaker &OutputShapeRanges(const std::vector<Range<Shape> *> &outputShapeRanges);

    InferShapeRangeContextFaker &NodeAttrs(
        const std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> &attrs);

    KernelRunContextHolder Build();
};
} // namespace gert
#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPERANGE_CONTEXT_FAKER_H
