/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_templates_registry.h
 * \brief
 */

#pragma once

#include <map>
#include <string>
#include <memory>
#include <exe_graph/runtime/tiling_context.h>
#include "fia_tiling_base.h"
#include "err/ops_err.h"
#include "platform/soc_spec.h"

namespace optiling {

template <typename T> 
std::unique_ptr<FiaTilingBase> TILING_CLASS(gert::TilingContext *context)
{
    return std::unique_ptr<T>(new (std::nothrow) T(context));
}

using FiaTilingClassCase = std::unique_ptr<FiaTilingBase> (*)(gert::TilingContext *);

class FiaTilingCases {
public:
    explicit FiaTilingCases(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> 
    void AddTiling(int32_t priority)
    {
        OP_CHECK_IF(cases_.find(priority) != cases_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(op_type_, "There are duplicate registrations."), return);
        cases_[priority] = TILING_CLASS<T>;
        OP_CHECK_IF(
            cases_[priority] == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(op_type_, "Register op tiling func failed, please check the class name."),
            return);
    }

    const std::map<int32_t, FiaTilingClassCase> &GetTilingCases()
    {
        return cases_;
    }

private:
    std::map<int32_t, FiaTilingClassCase> cases_;
    const std::string op_type_;
};

// --------------------------------Interfacce with soc version --------------------------------
class FiaTilingRegistry {
public:
    FiaTilingRegistry() = default;

    static FiaTilingRegistry &GetInstance()
    {
        static FiaTilingRegistry registry_impl_;
        return registry_impl_;
    }

    std::shared_ptr<FiaTilingCases> RegisterOp(const std::string &op_type,
                                            int32_t soc_version)
    {
        auto soc_iter = registry_map_.find(soc_version);
        if (soc_iter == registry_map_.end()) {
            std::map<std::string, std::shared_ptr<FiaTilingCases>> op_type_map;
            op_type_map[op_type] = std::shared_ptr<FiaTilingCases>(new (std::nothrow) FiaTilingCases(op_type));
            registry_map_[soc_version] = op_type_map;
        } else {
            if (soc_iter->second.find(op_type) == soc_iter->second.end()) {
                soc_iter->second[op_type] =
                    std::shared_ptr<FiaTilingCases>(new (std::nothrow) FiaTilingCases(op_type));
            }
        }

        OP_CHECK_IF(registry_map_[soc_version][op_type] == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(op_type, "Register tiling func failed, please check the class name."),
                    return nullptr);
        return registry_map_[soc_version][op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context, TilingInfo *tilingInfo)
    {
        int32_t npuArch = static_cast<int32_t>(NpuArch::DAV_RESV);
        const char *op_type = context->GetNodeType();
        fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, platformInfoPtr is nullptr.");
            return ge::GRAPH_FAILED;
        }
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        npuArch = static_cast<int32_t>(ascendcPlatform.GetCurNpuArch());
        OP_LOGI(context, "NpuArch is %d", npuArch);
        if (npuArch == static_cast<int32_t>(NpuArch::DAV_RESV)) {
            OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, cannot find npuArch.");
            return ge::GRAPH_FAILED;
        }
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type, npuArch);
        OP_LOGI(context, "op_type: %s", op_type);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling(tilingInfo);
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OP_LOGI(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                }
                OP_LOGI(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        // arch35各模板tiling切换至注册方式中，报error会导致图模式下沉场景执行失败
        OP_LOGW(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, FiaTilingClassCase> &GetTilingTemplates(const std::string &op_type,
                                                                    int32_t soc_version)
    {
        auto soc_iter = registry_map_.find(soc_version);
        OP_CHECK_IF(soc_iter == registry_map_.end(),
                    OPS_REPORT_VECTOR_INNER_ERR(op_type, "Get op tiling func failed, please check the soc version %d",
                                                soc_version),
                    return empty_tiling_case_);
        auto op_iter = soc_iter->second.find(op_type);
        OP_CHECK_IF(op_iter == soc_iter->second.end(),
                    OPS_REPORT_VECTOR_INNER_ERR(op_type, "Get op tiling func failed, please check the op name."),
                    return empty_tiling_case_);
        return op_iter->second->GetTilingCases();
    }

private:
    std::map<int32_t, std::map<std::string, std::shared_ptr<FiaTilingCases>>> registry_map_; // key is socversion
    const std::map<int32_t, FiaTilingClassCase> empty_tiling_case_{};
};

class FiaRegister {
public:
    explicit FiaRegister(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> 
    FiaRegister &tiling(int32_t priority, int32_t soc_version)
    {
        auto tilingCases = FiaTilingRegistry::GetInstance().RegisterOp(op_type_, soc_version);
        OP_CHECK_IF(tilingCases == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(op_type_, "Register op tiling failed, please the op name."),
                    return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

    template <typename T> 
    FiaRegister &tiling(int32_t priority, const std::vector<int32_t>& soc_versions)
    {
        for (int32_t soc_version : soc_versions) {
            auto tilingCases = FiaTilingRegistry::GetInstance().RegisterOp(op_type_, soc_version);
            OP_CHECK_IF(tilingCases == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(op_type_, "Register op tiling failed, please the op name."),
                    return *this);
            tilingCases->AddTiling<T>(priority);
        }
        return *this;
    }

private:
    const std::string op_type_;
};

// op_type: 算子名称， class_name: 注册的 tiling 类, soc_version：芯片版本号
// priority: tiling 类的优先级, 越小表示优先级越高, 即会优先选择这个tiling类
#define REGISTER_TILING_TEMPLATE_FIA(op_type, class_name, soc_versions, priority) \
        [[maybe_unused]] uint32_t op_impl_register_template_##op_type##_##class_name##priority;          \
        static FiaRegister VAR_UNUSED##op_type##class_name##priority_register = \
                FiaRegister(#op_type).tiling<class_name>(priority, soc_versions)
} // namespace optiling
