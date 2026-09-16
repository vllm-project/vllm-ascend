/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H

#include "op_infer_shape_context_builder.h"
#include "any_value.h"

namespace gert {

class InfershapeContextPara {
public:
    class TensorDescription {
    public:
        TensorDescription(const gert::StorageShape &shape, ge::DataType dtype, ge::Format format, bool isConst = false,
                          void *constValue = nullptr)
            : shape_(shape),
              dtype_(dtype),
              format_(format),
              isConst_(isConst),
              constValue_(constValue)
        {}

    public:
        gert::StorageShape shape_;
        ge::DataType dtype_ = ge::DT_FLOAT;
        ge::Format format_ = ge::FORMAT_ND;
        bool isConst_ = false;
        void *constValue_ = nullptr;
    };

    class OpAttr {
    public:
        OpAttr(const std::string &attrName, const Ops::Transformer::AnyValue &attr)
            : attrName_(attrName),
              attr_(attr)
        {}

    public:
        std::string attrName_;
        Ops::Transformer::AnyValue attr_;
    };

public:
    InfershapeContextPara(const std::string &opName, const std::vector<TensorDescription> &inputTensorDesc,
                          const std::vector<TensorDescription> &outputTensorDesc, const std::vector<OpAttr> &attrs)
        : opName_(opName),
          inputTensorDesc_(inputTensorDesc),
          outputTensorDesc_(outputTensorDesc),
          attrs_(attrs)
    {}

    InfershapeContextPara(const std::string &opName, const std::vector<TensorDescription> &inputTensorDesc,
                          const std::vector<TensorDescription> &outputTensorDesc)
        : opName_(opName),
          inputTensorDesc_(inputTensorDesc),
          outputTensorDesc_(outputTensorDesc)
    {}

    InfershapeContextPara(const std::string &opName, const std::vector<TensorDescription> &inputTensorDesc,
                          const std::vector<TensorDescription> &outputTensorDesc, const std::vector<OpAttr> &attrs,
                          const std::vector<uint32_t> &inputInstanceNum, const std::vector<uint32_t> &outputInstanceNum)
        : opName_(opName),
          inputTensorDesc_(inputTensorDesc),
          outputTensorDesc_(outputTensorDesc),
          attrs_(attrs),
          inputInstanceNum_(inputInstanceNum),
          outputInstanceNum_(outputInstanceNum)
    {}

    InfershapeContextPara(const std::string &opName, const std::vector<TensorDescription> &inputTensorDesc,
                          const std::vector<TensorDescription> &outputTensorDesc,
                          const std::vector<uint32_t> &inputInstanceNum, const std::vector<uint32_t> &outputInstanceNum)
        : opName_(opName),
          inputTensorDesc_(inputTensorDesc),
          outputTensorDesc_(outputTensorDesc),
          inputInstanceNum_(inputInstanceNum),
          outputInstanceNum_(outputInstanceNum)
    {}

public:
    std::string opName_;
    std::vector<uint32_t> inputInstanceNum_;
    std::vector<uint32_t> outputInstanceNum_;
    std::vector<TensorDescription> inputTensorDesc_;
    std::vector<TensorDescription> outputTensorDesc_;
    std::vector<OpAttr> attrs_;
};

class InferShapeContextFaker : public OpInferShapeContextBuilder {
public:
    InferShapeContextFaker &SetOpType(const std::string opType);

    /* only one can be choosed from IrInstanceNum */
    InferShapeContextFaker &NodeIoNum(size_t inputNum, size_t outputNum);

    /* can be used for dynamic inputs/outputs
     * only one can be choosed from NodeIoNum */
    InferShapeContextFaker &IrInstanceNum(const std::vector<uint32_t> &inputInstanceNum,
                                          const std::vector<uint32_t> &outputInstanceNum);

    InferShapeContextFaker &NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                        ge::Format storageFormat);

    InferShapeContextFaker &NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                         ge::Format storageFormat);

    InferShapeContextFaker &Attr(const std::string &attrName, bool attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, int64_t attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, float attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, const ge::AscendString &attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, const std::vector<bool> &attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, const std::vector<int64_t> &attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, const std::vector<float> &attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, const std::vector<ge::AscendString> &attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }
    InferShapeContextFaker &Attr(const std::string &attrName, const std::vector<std::vector<int64_t>> &attr)
    {
        OpInferShapeContextBuilder::AppendAttr(attr);
        return *this;
    }

    InferShapeContextFaker &InputTensors(const std::vector<Tensor *> &inputTensors);

    InferShapeContextFaker &OutputShapes(const std::vector<StorageShape *> &outputShapes);

    ContextHolder<InferShapeContext> Build();
};

} // namespace gert
#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_INFER_SHAPE_CONTEXT_FAKER_H
