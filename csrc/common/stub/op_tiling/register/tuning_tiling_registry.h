/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_REGISTER_TUNING_TILING_REGISTRY_HEADER__
#define __INC_REGISTER_TUNING_TILING_REGISTRY_HEADER__
#include <vector>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include "graph/ascend_string.h"
#include "register/tuning_tiling_reflection_utils.h"
namespace tuningtiling {
struct TilingItem {
    ge::AscendString dtype_;
    ge::AscendString name_;
};

class TuningTilingDef
{
public:
    virtual void FromJson(const nlohmann::json& j) = 0;
    virtual void ToJson(nlohmann::json& j) = 0;
    ge::AscendString GetClassName() const;
    virtual std::vector<TilingItem> GetItemInfo() const = 0;

protected:
    TuningTilingDef() = default;
    virtual ~TuningTilingDef() = default;
    // dtype , name
    std::vector<TilingItem> field_info_;
    ge::AscendString class_name_;
};

#define BEGIN_TUNING_TILING_DEF(class_name)                                                                  \
    class class_name : public TuningTilingDef                                                                \
    {                                                                                                        \
    public:                                                                                                  \
        virtual void FromJson(const nlohmann::json& j)                                                       \
        {                                                                                                    \
            FromJsonImpl(*this, "", j);                                                                      \
        }                                                                                                    \
                                                                                                             \
        virtual void ToJson(nlohmann::json& j)                                                               \
        {                                                                                                    \
            DumpObj(*this, "", j);                                                                           \
        }                                                                                                    \
                                                                                                             \
        std::vector<TilingItem> GetItemInfo() const                                                          \
        {                                                                                                    \
            return field_info_;                                                                              \
        }                                                                                                    \
                                                                                                             \
        class FieldHandler                                                                                   \
        {                                                                                                    \
        public:                                                                                              \
            FieldHandler(class_name* pinstance, const ge::AscendString& dtype, const ge::AscendString& name) \
            {                                                                                                \
                pinstance->field_info_.push_back({dtype, name});                                             \
            }                                                                                                \
        };                                                                                                   \
        friend class FieldHandler;                                                                           \
                                                                                                             \
    public:                                                                                                  \
        class_name()                                                                                         \
        {                                                                                                    \
            class_name_ = #class_name;                                                                       \
        };

#define TUNING_TILING_DATA_FIELD_DEF(data_type, field_name) \
public:                                                     \
    data_type field_name;                                   \
    FieldHandler field_name##_handler_ = FieldHandler(this, #data_type, #field_name);

#define END_TUNING_TILING_DEF \
    }                         \
    ;

using TuningTilingDefConstructor = std::shared_ptr<TuningTilingDef> (*)();
class TuningTilingClassFactory
{
public:
    static std::map<ge::AscendString, TuningTilingDefConstructor>& RegisterInfo();
    static void RegisterTilingData(const ge::AscendString& optype, TuningTilingDefConstructor const constructor);
    static std::shared_ptr<TuningTilingDef> CreateTilingDataInstance(const ge::AscendString& optype);
};

#define REGISTER_TUNING_TILING_CLASS(optype, class_name)                                                     \
    class optype##Helper                                                                                     \
    {                                                                                                        \
    public:                                                                                                  \
        optype##Helper()                                                                                     \
        {                                                                                                    \
            TuningTilingClassFactory::RegisterTilingData(#optype, optype##Helper::CreateTilingDataInstance); \
        }                                                                                                    \
        static std::shared_ptr<TuningTilingDef> CreateTilingDataInstance()                                   \
        {                                                                                                    \
            return std::make_shared<class_name>();                                                           \
        }                                                                                                    \
    };                                                                                                       \
    optype##Helper g_tuning_tiling_##optype##Helper;
using TuningTilingDefPtr = std::shared_ptr<TuningTilingDef>;
} // namespace tuningtiling

#endif