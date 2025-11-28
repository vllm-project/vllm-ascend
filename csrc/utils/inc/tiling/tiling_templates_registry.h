/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_templates_registry.h
 * \brief
 */

#pragma once

#include <map>
#include <string>
#include <memory>
#include <exe_graph/runtime/tiling_context.h>
#include "tiling/tiling_base.h"
#include "log/ops_log.h"
#include "error/ops_error.h"

namespace optiling {

template <typename T> std::unique_ptr<TilingBaseClass> TILING_CLASS(gert::TilingContext *context)
{
    return std::unique_ptr<T>(new (std::nothrow) T(context));
}

using TilingClassCase = std::unique_ptr<TilingBaseClass> (*)(gert::TilingContext *);

class TilingCases {
public:
    explicit TilingCases(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> void AddTiling(int32_t priority)
    {
        OPS_ERR_IF(cases_.find(priority) != cases_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(op_type_, "There are duplicate registrations."), return);
        cases_[priority] = TILING_CLASS<T>;
        OPS_ERR_IF(
            cases_[priority] == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(op_type_, "Register op tiling func failed, please check the class name."),
            return);
    }

    const std::map<int32_t, TilingClassCase> &GetTilingCases()
    {
        return cases_;
    }

private:
    std::map<int32_t, TilingClassCase> cases_;
    const std::string op_type_;
};

class TilingRegistry {
public:
    TilingRegistry() = default;

#ifdef ASCENDC_OP_TEST
    static TilingRegistry &GetInstance();
#else
    static TilingRegistry &GetInstance()
    {
        static TilingRegistry registry_impl_;
        return registry_impl_;
    }
#endif

    std::shared_ptr<TilingCases> RegisterOp(const std::string &op_type)
    {
        if (registry_map_.find(op_type) == registry_map_.end()) {
            registry_map_[op_type] = std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
        }
        OPS_ERR_IF(registry_map_[op_type] == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(op_type, "Register tiling func failed, please check the class name."),
                   return nullptr);
        return registry_map_[op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context)
    {
        const char *op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OPS_LOG_D(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                }
                OPS_LOG_D(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context, const std::vector<int32_t> &priorities)
    {
        const char *op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto priorityId : priorities) {
            auto templateFunc = tilingTemplateRegistryMap[priorityId](context);
            if (templateFunc != nullptr) {
                ge::graphStatus status = templateFunc->DoTiling();
                if (status == ge::GRAPH_SUCCESS) {
                    OPS_LOG_D(context, "Do general op tiling success priority=%d", priorityId);
                    return status;
                }
                OPS_LOG_D(context, "Ignore general op tiling priority=%d", priorityId);
            }
        }
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, TilingClassCase> &GetTilingTemplates(const std::string &op_type)
    {
        OPS_ERR_IF(registry_map_.find(op_type) == registry_map_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(op_type, "Get op tiling func failed, please check the op name."),
                   return empty_tiling_case_);
        return registry_map_[op_type]->GetTilingCases();
    }

private:
    std::map<std::string, std::shared_ptr<TilingCases>> registry_map_;
    const std::map<int32_t, TilingClassCase> empty_tiling_case_ {};
};

class Register {
public:
    explicit Register(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> Register &tiling(int32_t priority)
    {
        auto tilingCases = TilingRegistry::GetInstance().RegisterOp(op_type_);
        OPS_ERR_IF(tilingCases == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(op_type_, "Register op tiling failed, please the op name."),
                   return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

private:
    const std::string op_type_;
};

// op_type: 算子名称， class_name: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
#define REGISTER_TILING_TEMPLATE(op_type, class_name, priority)                                                        \
    static Register VAR_UNUSED##op_type_##class_name##priority_register = Register(op_type).tiling<class_name>(priority)

} // namespace optiling
