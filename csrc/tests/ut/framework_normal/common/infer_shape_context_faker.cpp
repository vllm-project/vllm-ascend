/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_shape_context_faker.h"

namespace gert {

InferShapeContextFaker &InferShapeContextFaker::SetOpType(const std::string opType)
{
    OpInferShapeContextBuilder::OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

InferShapeContextFaker &InferShapeContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    OpInferShapeContextBuilder::IONum(inputNum, outputNum);
    return *this;
}

InferShapeContextFaker &InferShapeContextFaker::IrInstanceNum(const std::vector<uint32_t> &inputInstanceNum,
                                                              const std::vector<uint32_t> &outputInstanceNum)
{
    OpInferShapeContextBuilder::IOInstanceNum(inputInstanceNum, outputInstanceNum);
    return *this;
}

InferShapeContextFaker &InferShapeContextFaker::NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                                            ge::Format storageFormat)
{
    return *this;
}

InferShapeContextFaker &InferShapeContextFaker::NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                                             ge::Format storageFormat)
{
    OpInferShapeContextBuilder::OutputTensorDesc(index, dtype, originFormat, storageFormat);
    return *this;
}

InferShapeContextFaker &InferShapeContextFaker::InputTensors(const std::vector<Tensor *> &inputTensors)
{
    OpInferShapeContextBuilder::InputTensors(inputTensors);
    return *this;
}

InferShapeContextFaker &InferShapeContextFaker::OutputShapes(const std::vector<StorageShape *> &outputShapes)
{
    return *this;
}

ContextHolder<InferShapeContext> InferShapeContextFaker::Build()
{
    return OpInferShapeContextBuilder::Build();
}

} // namespace gert
