/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "exe_graph/runtime/tiling_context.h"
#include "tiling_base.h"
#include "error_log.h"

namespace Ops {
namespace Transformer {
namespace OpTiling {

template <typename T>
std::unique_ptr<TilingBaseClass> TILING_CLASS(gert::TilingContext* context)
{
    return std::unique_ptr<T>(new (std::nothrow) T(context));
}

using TilingClassCase = std::unique_ptr<TilingBaseClass> (*)(gert::TilingContext*);

class TilingCases {
public:
    explicit TilingCases(std::string op_type) : op_type_(std::move(op_type))
    {}

    template <typename T>
    void AddTiling(int32_t priority)
    {
        OP_CHECK_IF(
            cases_.find(priority) != cases_.end(), OP_LOGE(op_type_, "There are duplicate registrations."), return);
        cases_[priority] = TILING_CLASS<T>;
        OP_CHECK_IF(
            cases_[priority] == nullptr,
            OP_LOGE(op_type_, "Register op tiling func failed, please check the class name."), return);
    }

    const std::map<int32_t, TilingClassCase>& GetTilingCases()
    {
        return cases_;
    }

private:
    std::map<int32_t, TilingClassCase> cases_;
    const std::string op_type_;
};

// --------------------------------Interfacce with soc version --------------------------------
class TilingRegistryNew {
public:
    TilingRegistryNew() = default;

#ifdef ASCENDC_OP_TEST
    static TilingRegistryNew& GetInstance();
#else
    static TilingRegistryNew& GetInstance()
    {
        static TilingRegistryNew registry_impl_;
        return registry_impl_;
    }
#endif

    std::shared_ptr<TilingCases> RegisterOp(const std::string& op_type, int32_t soc_version)
    {
        auto soc_iter = registry_map_.find(soc_version);
        if (soc_iter == registry_map_.end()) {
            std::map<std::string, std::shared_ptr<TilingCases>> op_type_map;
            op_type_map[op_type] = std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
            registry_map_[soc_version] = op_type_map;
        } else {
            if (soc_iter->second.find(op_type) == soc_iter->second.end()) {
                soc_iter->second[op_type] = std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
            }
        }

        OP_CHECK_IF(
            registry_map_[soc_version][op_type] == nullptr,
            OP_LOGE(op_type, "Register tiling func failed, please check the class name."), return nullptr);
        return registry_map_[soc_version][op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext* context)
    {
        int32_t soc_version = (int32_t)platform_ascendc::SocVersion::RESERVED_VERSION;
        const char* op_type = context->GetNodeType();
        fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr = static_cast<const CompileInfoCommon*>(context->GetCompileInfo());
            OP_CHECK_IF(
                compileInfoPtr == nullptr, OP_LOGE(op_type, "compileInfoPtr is null."), return ge::GRAPH_FAILED);
            soc_version = compileInfoPtr->socVersion;
            OP_LOGD(context, "soc version in compileInfo is %d", soc_version);
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            soc_version = static_cast<int32_t>(ascendcPlatform.GetSocVersion());
            OP_LOGD(context, "soc version is %d", soc_version);
            if (soc_version == (int32_t)platform_ascendc::SocVersion::RESERVED_VERSION) {
                OP_LOGE(op_type, "Do op tiling failed, cannot find soc version.");
                return ge::GRAPH_FAILED;
            }
        }
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type, soc_version);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OP_LOGD(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                }
                OP_LOGD(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        OP_LOGE(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext* context, const std::vector<int32_t>& priorities)
    {
        int32_t soc_version;
        const char* op_type = context->GetNodeType();
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr = reinterpret_cast<const CompileInfoCommon*>(context->GetCompileInfo());
            OP_CHECK_IF(
                compileInfoPtr == nullptr, OP_LOGE(op_type, "compileInfoPtr is null."), return ge::GRAPH_FAILED);
            soc_version = compileInfoPtr->socVersion;
            OP_LOGD(context, "soc version in compileInfo is %d", soc_version);
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            soc_version = static_cast<int32_t>(ascendcPlatform.GetSocVersion());
            OP_LOGD(context, "soc version is %d", soc_version);
        }

        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type, soc_version);
        for (auto priority_id : priorities) {
            auto tilingCaseIter = tilingTemplateRegistryMap.find(priority_id);
            if (tilingCaseIter != tilingTemplateRegistryMap.end()) {
                auto templateFunc = tilingCaseIter->second(context);
                if (templateFunc != nullptr) {
                    ge::graphStatus status = templateFunc->DoTiling();
                    if (status == ge::GRAPH_SUCCESS) {
                        OP_LOGD(context, "Do general op tiling success priority=%d", priority_id);
                        return status;
                    }
                    OP_LOGD(context, "Ignore general op tiling priority=%d", priority_id);
                }
            }
        }
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, TilingClassCase>& GetTilingTemplates(const std::string& op_type, int32_t soc_version)
    {
        auto soc_iter = registry_map_.find(soc_version);
        OP_CHECK_IF(
            soc_iter == registry_map_.end(),
            OP_LOGE(op_type, "Get op tiling func failed, please check the soc version %d", soc_version),
            return empty_tiling_case_);
        auto op_iter = soc_iter->second.find(op_type);
        OP_CHECK_IF(
            op_iter == soc_iter->second.end(), OP_LOGE(op_type, "Get op tiling func failed, please check the op name."),
            return empty_tiling_case_);
        return op_iter->second->GetTilingCases();
    }

private:
    std::map<int32_t, std::map<std::string, std::shared_ptr<TilingCases>>> registry_map_; // key is socversion
    const std::map<int32_t, TilingClassCase> empty_tiling_case_{};
};

class RegisterNew {
public:
    explicit RegisterNew(std::string op_type) : op_type_(std::move(op_type))
    {}

    template <typename T>
    RegisterNew& tiling(int32_t priority, int32_t soc_version)
    {
        auto tilingCases = TilingRegistryNew::GetInstance().RegisterOp(op_type_, soc_version);
        OP_CHECK_IF(
            tilingCases == nullptr, OP_LOGE(op_type_, "Register op tiling failed, please the op name."), return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

    template <typename T>
    RegisterNew& tiling(int32_t priority, const std::vector<int32_t>& soc_versions)
    {
        for (int32_t soc_version : soc_versions) {
            auto tilingCases = TilingRegistryNew::GetInstance().RegisterOp(op_type_, soc_version);
            OP_CHECK_IF(
                tilingCases == nullptr, OP_LOGE(op_type_, "Register op tiling failed, please the op name."),
                return *this);
            tilingCases->AddTiling<T>(priority);
        }
        return *this;
    }

private:
    const std::string op_type_;
};

// --------------------------------Interfacce without soc version --------------------------------
class TilingRegistry {
public:
    TilingRegistry() = default;

#ifdef ASCENDC_OP_TEST
    static TilingRegistry& GetInstance();
#else
    static TilingRegistry& GetInstance()
    {
        static TilingRegistry registry_impl_;
        return registry_impl_;
    }
#endif

    std::shared_ptr<TilingCases> RegisterOp(const std::string& op_type)
    {
        if (registry_map_.find(op_type) == registry_map_.end()) {
            registry_map_[op_type] = std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
        }
        OP_CHECK_IF(
            registry_map_[op_type] == nullptr,
            OP_LOGE(op_type, "Register tiling func failed, please check the class name."), return nullptr);
        return registry_map_[op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext* context)
    {
        const char* op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OP_LOGD(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                }
                OP_LOGD(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        OP_LOGE(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext* context, const std::vector<int32_t>& priorities)
    {
        const char* op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto priorityId : priorities) {
            auto templateFunc = tilingTemplateRegistryMap[priorityId](context);
            if (templateFunc != nullptr) {
                ge::graphStatus status = templateFunc->DoTiling();
                if (status == ge::GRAPH_SUCCESS) {
                    OP_LOGD(context, "Do general op tiling success priority=%d", priorityId);
                    return status;
                }
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OP_LOGD(context, "Do op tiling failed");
                    return status;
                }
                OP_LOGD(context, "Ignore general op tiling priority=%d", priorityId);
            }
        }
        OP_LOGE(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, TilingClassCase>& GetTilingTemplates(const std::string& op_type)
    {
        OP_CHECK_IF(
            registry_map_.find(op_type) == registry_map_.end(),
            OP_LOGE(op_type, "Get op tiling func failed, please check the op name."), return empty_tiling_case_);
        return registry_map_[op_type]->GetTilingCases();
    }

private:
    std::map<std::string, std::shared_ptr<TilingCases>> registry_map_;
    const std::map<int32_t, TilingClassCase> empty_tiling_case_;
};

class Register {
public:
    explicit Register(std::string op_type) : op_type_(std::move(op_type))
    {}

    template <typename T>
    Register& tiling(int32_t priority)
    {
        auto tilingCases = TilingRegistry::GetInstance().RegisterOp(op_type_);
        OP_CHECK_IF(
            tilingCases == nullptr, OP_LOGE(op_type_, "Register op tiling failed, please the op name."), return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

private:
    const std::string op_type_;
};
} // namespace OpTiling
} // namespace Transformer
} // namespace Ops

// op_type: operator name, class_name: registered tiling class, soc_version: chip version number
// priority: priority of tiling class, smaller value means higher priority, i.e., this tiling class will be selected first
#define REGISTER_TILING_TEMPLATE_WITH_SOCVERSION(op_type, class_name, soc_versions, priority)    \
    [[maybe_unused]] uint32_t op_impl_register_template_##op_type##_##class_name##priority;      \
    static Ops::Transformer::OpTiling::RegisterNew VAR_UNUSED##op_type##class_name##priority_register = \
        Ops::Transformer::OpTiling::RegisterNew(#op_type).tiling<class_name>(priority, soc_versions)

// op_type: operator name, class_name: registered tiling class
// priority: priority of tiling class, smaller value means higher priority, i.e., higher probability of being selected
#define REGISTER_TILING_TEMPLATE(op_type, class_name, priority)                                \
    [[maybe_unused]] uint32_t op_impl_register_template_##op_type##_##class_name##priority;      \
    static Ops::Transformer::OpTiling::Register VAR_UNUSED##op_type_##class_name##priority_register = \
        Ops::Transformer::OpTiling::Register(op_type).tiling<class_name>(priority)

// op_type: operator name, class_name: registered tiling class
// soc_version: SOC version, used to distinguish different SOCs
// priority: priority of tiling class, smaller value means higher priority, i.e., this tiling class will be selected first
#define REGISTER_TILING_TEMPLATE_NEW(op_type, class_name, soc_version, priority)                 \
    [[maybe_unused]] uint32_t op_impl_register_template_##op_type##_##class_name##priority;      \
    static Ops::Transformer::OpTiling::RegisterNew VAR_UNUSED##op_type##class_name##priority_register = \
        Ops::Transformer::OpTiling::RegisterNew(#op_type).tiling<class_name>(priority, soc_version)

// op_type: operator name, class_name: registered tiling class
// priority: priority of tiling class, smaller value means higher priority, i.e., higher probability of being selected
// Replaces REGISTER_TILING_TEMPLATE, if op_type is a string constant, remove the quotes
#define REGISTER_OPS_TILING_TEMPLATE(op_type, class_name, priority)                       \
    [[maybe_unused]] uint32_t op_impl_register_template_##op_type##_##class_name##priority;      \
    static Ops::Transformer::OpTiling::Register                                                  \
        __attribute__((unused)) tiling_##op_type##_##class_name##_##priority##_register = \
            Ops::Transformer::OpTiling::Register(#op_type).tiling<class_name>(priority)
