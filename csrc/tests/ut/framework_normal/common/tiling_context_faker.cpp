/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_context_faker.h"

namespace gert {

TilingContextFaker &TilingContextFaker::SetOpType(const std::string opType)
{
    OpTilingContextBuilder::OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

TilingContextFaker &TilingContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    OpTilingContextBuilder::IONum(inputNum, outputNum);
    return *this;
}

TilingContextFaker &TilingContextFaker::IrInstanceNum(const std::vector<uint32_t> &inputInstanceNum,
                                                      const std::vector<uint32_t> &outputInstanceNum)
{
    OpTilingContextBuilder::IOInstanceNum(inputInstanceNum, outputInstanceNum);
    return *this;
}

TilingContextFaker &TilingContextFaker::NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                                    ge::Format storageFormat)
{
    return *this;
}

TilingContextFaker &TilingContextFaker::NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                                     ge::Format storageFormat)
{
    return *this;
}

TilingContextFaker &TilingContextFaker::InputTensors(const std::vector<Tensor *> &inputTensors)
{
    OpTilingContextBuilder::InputTensors(inputTensors);
    return *this;
}

TilingContextFaker &TilingContextFaker::OutputTensors(const std::vector<Tensor *> &outputTensors)
{
    OpTilingContextBuilder::OutputTensors(outputTensors);
    return *this;
}

TilingContextFaker &TilingContextFaker::CompileInfo(const void *compileInfo)
{
    OpTilingContextBuilder::CompileInfo(compileInfo);
    return *this;
}

TilingContextFaker &TilingContextFaker::PlatformInfo(const void *platformInfo)
{
    OpTilingContextBuilder::PlatformInfo(platformInfo);
    return *this;
}

TilingContextFaker &TilingContextFaker::DeterministicInfo(int32_t deterministicInfo)
{
    OpTilingContextBuilder::Deterministic(deterministicInfo);
    return *this;
}

TilingContextFaker &TilingContextFaker::TilingData(const void *tilingData)
{
    OpTilingContextBuilder::TilingData(static_cast<const gert::TilingData *>(tilingData));
    return *this;
}

TilingContextFaker &TilingContextFaker::Workspace(const ContinuousVector *workspace)
{
    OpTilingContextBuilder::Workspace(workspace);
    return *this;
}

ContextHolder<TilingContext> TilingContextFaker::Build()
{
    return OpTilingContextBuilder::Build();
}

} // namespace gert
