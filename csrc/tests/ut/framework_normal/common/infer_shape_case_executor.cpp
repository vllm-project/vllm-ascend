/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_shape_case_executor.h"
#include <gtest/gtest.h>
#include "base/registry/op_impl_space_registry_v2.h"

#define DO_INFERSHAPE(infershapeContextPara)                                                                           \
    auto contextFaker = gert::InferShapeContextFaker();                                                                \
    /* 1. input/output information */                                                                                  \
    size_t inputNum = infershapeContextPara.inputTensorDesc_.size();                                                   \
    size_t outputNum = infershapeContextPara.outputTensorDesc_.size();                                                 \
    if (infershapeContextPara.inputInstanceNum_.size() != 0 || infershapeContextPara.outputInstanceNum_.size() != 0) { \
        contextFaker.IrInstanceNum(infershapeContextPara.inputInstanceNum_, infershapeContextPara.outputInstanceNum_); \
    } else {                                                                                                           \
        contextFaker.NodeIoNum(inputNum, outputNum);                                                                   \
    }                                                                                                                  \
    std::vector<gert::Tensor *> inputTensors = {};                                                                     \
    std::vector<std::unique_ptr<gert::Tensor>> inputTensorsKeepAlive = {};                                             \
    for (size_t index = 0; index < inputNum; index++) {                                                                \
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(                                      \
            infershapeContextPara.inputTensorDesc_[index].shape_,                                                      \
            gert::StorageFormat(infershapeContextPara.inputTensorDesc_[index].format_,                                 \
                                infershapeContextPara.inputTensorDesc_[index].format_, gert::ExpandDimsType()),        \
            gert::TensorPlacement::kOnHost, infershapeContextPara.inputTensorDesc_[index].dtype_,                      \
            infershapeContextPara.inputTensorDesc_[index].isConst_ ?                                                   \
                infershapeContextPara.inputTensorDesc_[index].constValue_ :                                            \
                nullptr);                                                                                              \
        inputTensors.push_back(curTensor.get());                                                                       \
        inputTensorsKeepAlive.push_back(std::move(curTensor));                                                         \
    }                                                                                                                  \
    for (size_t index = 0; index < outputNum; index++) {                                                               \
        contextFaker.NodeOutputTd(index, infershapeContextPara.outputTensorDesc_[index].dtype_,                        \
                                  infershapeContextPara.outputTensorDesc_[index].format_,                              \
                                  infershapeContextPara.outputTensorDesc_[index].format_);                             \
    }                                                                                                                  \
    contextFaker.InputTensors(inputTensors);                                                                           \
    for (auto &attrInfo : infershapeContextPara.attrs_) {                                                              \
        switch (attrInfo.attr_.type_) {                                                                                \
            case Ops::Transformer::AnyValue::ValueType::VT_BOOL:                                                       \
                {                                                                                                      \
                    contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool *>(attrInfo.attr_.valuePtr_.get()));  \
                    break;                                                                                             \
                }                                                                                                      \
            case Ops::Transformer::AnyValue::ValueType::VT_INT:                                                        \
                {                                                                                                      \
                    contextFaker.Attr(attrInfo.attrName_,                                                              \
                                      *reinterpret_cast<int64_t *>(attrInfo.attr_.valuePtr_.get()));                   \
                    break;                                                                                             \
                }                                                                                                      \
            case Ops::Transformer::AnyValue::ValueType::VT_FLOAT:                                                      \
                {                                                                                                      \
                    contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float *>(attrInfo.attr_.valuePtr_.get())); \
                    break;                                                                                             \
                }                                                                                                      \
            case Ops::Transformer::AnyValue::ValueType::VT_STRING:                                                     \
                {                                                                                                      \
                    contextFaker.Attr(                                                                                 \
                        attrInfo.attrName_,                                                                            \
                        ge::AscendString(reinterpret_cast<std::string *>(attrInfo.attr_.valuePtr_.get())->c_str()));   \
                    break;                                                                                             \
                }                                                                                                      \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_BOOL:                                                  \
                {                                                                                                      \
                    contextFaker.Attr(attrInfo.attrName_,                                                              \
                                      *reinterpret_cast<std::vector<bool> *>(attrInfo.attr_.valuePtr_.get()));         \
                    break;                                                                                             \
                }                                                                                                      \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_INT:                                                   \
                {                                                                                                      \
                    contextFaker.Attr(attrInfo.attrName_,                                                              \
                                      *reinterpret_cast<std::vector<int64_t> *>(attrInfo.attr_.valuePtr_.get()));      \
                    break;                                                                                             \
                }                                                                                                      \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_LIST_INT:                                              \
                {                                                                                                      \
                    contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<std::vector<int64_t>> *>(      \
                                                              attrInfo.attr_.valuePtr_.get()));                        \
                    break;                                                                                             \
                }                                                                                                      \
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_FLOAT:                                                 \
                {                                                                                                      \
                    contextFaker.Attr(attrInfo.attrName_,                                                              \
                                      *reinterpret_cast<std::vector<float> *>(attrInfo.attr_.valuePtr_.get()));        \
                    break;                                                                                             \
                }                                                                                                      \
            default:                                                                                                   \
                std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << "The ValueType " << attrInfo.attr_.type_      \
                          << "is not supported!" << std::endl;                                                         \
        }                                                                                                              \
    }                                                                                                                  \
    auto contextHolder = contextFaker.SetOpType(infershapeContextPara.opName_.c_str()).Build();                        \
    /* 2. get infershape func */                                                                                       \
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();                         \
    auto infershapeFunc = spaceRegistry->GetOpImpl(infershapeContextPara.opName_.c_str())->infer_shape;                \
    /* 3. check infershape func */                                                                                     \
    auto infershapeRet = infershapeFunc(contextHolder.GetContext());

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

void ExecuteTestCase(gert::InfershapeContextPara &infershapeContextPara, ge::graphStatus expectResult,
                     const std::vector<std::vector<int64_t>> &expectOutputShape)
{
    DO_INFERSHAPE(infershapeContextPara);

    // check infershape func
    ASSERT_EQ(infershapeRet, expectResult);
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    // check output shape
    for (int i = 0; i < expectOutputShape.size(); i++) {
        EXPECT_EQ(ToVector(*contextHolder.GetContext()->GetOutputShape(i)), expectOutputShape[i]);
    }
}
