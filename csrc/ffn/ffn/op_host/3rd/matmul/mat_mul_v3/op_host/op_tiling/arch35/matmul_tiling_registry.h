/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_tiling_registry.h
 * \brief
 */

#pragma once

#include <map>
#include <string>
#include <memory>
#include <functional>

#include "op_host/static_register_symbol.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_base.h"
#include "error_util.h"

#include "matmul_base_tiling.h"
#include "matmul_tiling_cfg.h"

namespace optiling {
struct MMRegisterCfg {
    const char* opType{nullptr};
    NpuArch npuArch{NpuArch::DAV_RESV};
    std::vector<int32_t> priorities{}; // 0 base
};

template <typename T>
std::unique_ptr<MatMulBaseTiling> MM_TILING_CLASS(gert::TilingContext* context, MatMulTilingCfg& cfg)
{
    return std::unique_ptr<T>(new (std::nothrow) T(context, cfg));
}

using MMTilingClassCase = std::unique_ptr<MatMulBaseTiling> (*)(gert::TilingContext*, MatMulTilingCfg&);

class MMTilingCases {
public:
    explicit MMTilingCases(std::string opType) : opType_(std::move(opType)) {}

    template <typename T>
    void AddTiling(int32_t priority)
    {
        OPS_ERR_IF(cases_.find(priority) != cases_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(opType_, "There are duplicate registrations."), return );
        cases_[priority] = MM_TILING_CLASS<T>;
        OPS_ERR_IF(cases_[priority] == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opType_, "Register op tiling func failed, please check the class name."),
                   return );
    }

    const std::map<int32_t, MMTilingClassCase>& GetTilingCases() { return cases_; }

private:
    std::map<int32_t, MMTilingClassCase> cases_;
    const std::string opType_;
};

class MMTilingRegistry {
public:
    MMTilingRegistry() = default;

#ifdef ASCENDC_OP_TEST
    static MMTilingRegistry& GetInstance();
#else
    static MMTilingRegistry& GetInstance()
    {
        static MMTilingRegistry registryImpl_;
        return registryImpl_;
    }
#endif

    std::shared_ptr<MMTilingCases> RegisterOp(const std::string& opType, NpuArch npuArch)
    {
        auto socIter = registryMap_.find(npuArch);
        if (socIter == registryMap_.end()) {
            std::map<std::string, std::shared_ptr<MMTilingCases>> opTypeMap;
            opTypeMap[opType] = std::shared_ptr<MMTilingCases>(new (std::nothrow) MMTilingCases(opType));
            registryMap_[npuArch] = opTypeMap;
        } else {
            if (socIter->second.find(opType) == socIter->second.end()) {
                socIter->second[opType] = std::shared_ptr<MMTilingCases>(new (std::nothrow) MMTilingCases(opType));
            }
        }

        OPS_ERR_IF(registryMap_[npuArch][opType] == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opType, "Register tiling func failed, please check the class name."),
                   return nullptr);
        return registryMap_[npuArch][opType];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext* context, MatMulTilingCfg& tilingCfg,
                                 const MMRegisterCfg& registerCfg)
    {
        if (context == nullptr || tilingCfg.compileInfo == nullptr || tilingCfg.args == nullptr) {
            OPS_LOG_E(context, "DoTilingImpl failed, context or tilingCfg or args is null.");
            return ge::GRAPH_FAILED;
        }
        const char* opType = registerCfg.opType == nullptr ? context->GetNodeType() : registerCfg.opType;
        auto tilingTemplateRegistryMap = GetTilingTemplates(opType, registerCfg.npuArch);
        OPS_LOG_D(context, "registry map find by opType %s, npu arch %d", opType,
                  static_cast<int32_t>(registerCfg.npuArch));
        if (tilingTemplateRegistryMap.empty()) {
            OPS_LOG_E(context, "no registry map find by opType %s, npu arch %d", opType,
                      static_cast<int32_t>(registerCfg.npuArch));
            return ge::GRAPH_FAILED;
        }
        std::vector<int32_t> priorities{registerCfg.priorities};
        if (priorities.empty()) {
            for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
                priorities.push_back(it->first);
            }
        }
        for (auto priorityId : priorities) {
            if (tilingTemplateRegistryMap.find(priorityId) == tilingTemplateRegistryMap.end()) {
                OPS_LOG_E(context, "no registry map find by priority %d", priorityId);
                return ge::GRAPH_FAILED;
            }
            auto templateFunc = tilingTemplateRegistryMap[priorityId](context, tilingCfg);
            if (templateFunc != nullptr) {
                ge::graphStatus status = templateFunc->DoTiling();
                if (status == ge::GRAPH_SUCCESS) {
                    OPS_LOG_D(context, "Do general op tiling success priority=%d", priorityId);
                    return status;
                }
                OPS_LOG_D(context, "Ignore general op tiling priority=%d", priorityId);
            }
        }
        OPS_LOG_E(context, "no general op tiling.");
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, MMTilingClassCase>& GetTilingTemplates(const std::string& opType, NpuArch npuArch)
    {
        auto socIter = registryMap_.find(npuArch);
        OPS_ERR_IF(socIter == registryMap_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(opType, "Get op tiling func failed, please check the npu arch %d",
                                               static_cast<int32_t>(npuArch)),
                   return emptyTilingCase_);
        auto opIter = socIter->second.find(opType);
        OPS_ERR_IF(opIter == socIter->second.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(opType, "Get op tiling func failed, please check the op name."),
                   return emptyTilingCase_);
        return opIter->second->GetTilingCases();
    }

private:
    std::map<NpuArch, std::map<std::string, std::shared_ptr<MMTilingCases>>> registryMap_; // key is socversion
    const std::map<int32_t, MMTilingClassCase> emptyTilingCase_{};
};

class MMRegister {
public:
    explicit MMRegister(std::string opType) : opType_(std::move(opType)) {}

    template <typename T>
    MMRegister& tiling(int32_t priority, NpuArch npuArch)
    {
        auto tilingCases = MMTilingRegistry::GetInstance().RegisterOp(opType_, npuArch);
        OPS_ERR_IF(tilingCases == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opType_, "Register op tiling failed, please check the op name."),
                   return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

private:
    const std::string opType_;
};

// opType: 算子名称， className: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
// 取代 MM_REGISTER_TILING_TEMPLATE , 传入的op_type如果是字符串常量，需要去掉引号
#define MM_REGISTER_TILING_TEMPLATE(opType, className, npuArch, priority)                                   \
    GLOBAL_REGISTER_SYMBOL(opType, className, priority, __COUNTER__, __LINE__);                             \
    static MMRegister __attribute__((unused))                                                               \
    mm_register_##opType##_##className##_##npuArch##_##priority##_ = MMRegister(#opType).tiling<className>( \
        static_cast<int32_t>(priority), NpuArch::npuArch)
} // namespace optiling
