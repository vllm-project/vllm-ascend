/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_ANY_VALUE_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_ANY_VALUE_H

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace Ops {
namespace Transformer {
class AnyValue {
public:
    enum ValueType {
        VT_STRING = 1,
        VT_FLOAT = 2,
        VT_BOOL = 3,
        VT_INT = 4,
        VT_LIST_LIST_INT = 10,
        VT_LIST_BASE = 1000,

        VT_LIST_FLOAT = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_FLOAT),
        VT_LIST_BOOL = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_BOOL),
        VT_LIST_INT = static_cast<int32_t>(VT_LIST_BASE) + static_cast<int32_t>(VT_INT),
    };

    AnyValue(ValueType type, const std::shared_ptr<void> &valuePtr)
        : type_(type),
          valuePtr_(valuePtr)
    {}
    ~AnyValue() = default;
    AnyValue(const AnyValue &anyValue)
        : type_(anyValue.type_),
          valuePtr_(anyValue.valuePtr_)
    {}

    template <typename T>
    static inline AnyValue CreateFrom(const T &value);

    template <typename T>
    void SetAttr(const std::string &item, T &faker) const
    {
        switch (type_) {
            case ValueType::VT_BOOL:
                faker.Attr(item, *reinterpret_cast<bool *>(valuePtr_.get()));
                break;
            case ValueType::VT_INT:
                faker.Attr(item, *reinterpret_cast<int64_t *>(valuePtr_.get()));
                break;
            case ValueType::VT_FLOAT:
                faker.Attr(item, *reinterpret_cast<float *>(valuePtr_.get()));
                break;
            case ValueType::VT_STRING:
                faker.Attr(item, ge::AscendString(reinterpret_cast<std::string *>(valuePtr_.get())->c_str()));
                break;
            case ValueType::VT_LIST_BOOL:
                faker.Attr(item, *reinterpret_cast<std::vector<bool> *>(valuePtr_.get()));
                break;
            case ValueType::VT_LIST_INT:
                faker.Attr(item, *reinterpret_cast<std::vector<int64_t> *>(valuePtr_.get()));
                break;
            case ValueType::VT_LIST_LIST_INT:
                faker.Attr(item, *reinterpret_cast<std::vector<std::vector<int64_t>> *>(valuePtr_.get()));
                break;
            case ValueType::VT_LIST_FLOAT:
                faker.Attr(item, *reinterpret_cast<std::vector<float> *>(valuePtr_.get()));
                break;
            default:
                std::cout << "[ERROR]" << __FILE__ << ":" << __LINE__ << "The ValueType is not supported!" << std::endl;
        }
    }

    ValueType type_;
    std::shared_ptr<void> valuePtr_;
};

template <>
inline AnyValue AnyValue::CreateFrom<std::string>(const std::string &value)
{
    auto valuePtr = new std::string;
    *valuePtr = value;
    return AnyValue(VT_STRING, std::shared_ptr<void>(valuePtr));
}

template <>
inline AnyValue AnyValue::CreateFrom<float>(const float &value)
{
    auto valuePtr = new float;
    *valuePtr = value;
    return AnyValue(VT_FLOAT, std::shared_ptr<void>(valuePtr));
}

template <>
inline AnyValue AnyValue::CreateFrom<bool>(const bool &value)
{
    auto valuePtr = new bool;
    *valuePtr = value;
    return AnyValue(VT_BOOL, std::shared_ptr<void>(valuePtr));
}

template <>
inline AnyValue AnyValue::CreateFrom<int64_t>(const int64_t &value)
{
    auto valuePtr = new int64_t;
    *valuePtr = value;
    return AnyValue(VT_INT, std::shared_ptr<void>(valuePtr));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<float>>(const std::vector<float> &value)
{
    auto valuePtr = new std::vector<float>;
    *valuePtr = value;
    return AnyValue(VT_LIST_FLOAT, std::shared_ptr<void>(valuePtr));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<bool>>(const std::vector<bool> &value)
{
    auto valuePtr = new std::vector<bool>;
    *valuePtr = value;
    return AnyValue(VT_LIST_BOOL, std::shared_ptr<void>(valuePtr));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<int64_t>>(const std::vector<int64_t> &value)
{
    auto valuePtr = new std::vector<int64_t>;
    *valuePtr = value;
    return AnyValue(VT_LIST_INT, std::shared_ptr<void>(valuePtr));
}

template <>
inline AnyValue AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(const std::vector<std::vector<int64_t>> &value)
{
    auto valuePtr = new std::vector<std::vector<int64_t>>;
    *valuePtr = value;
    return AnyValue(VT_LIST_LIST_INT, std::shared_ptr<void>(valuePtr));
}
} // namespace Transformer
} // namespace Ops

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_ANY_VALUE_H
