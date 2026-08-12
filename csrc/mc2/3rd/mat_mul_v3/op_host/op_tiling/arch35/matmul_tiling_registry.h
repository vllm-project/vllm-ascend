/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#ifndef __OP_HOST_MATMUL_TILING_REGISTRY_H__
#define __OP_HOST_MATMUL_TILING_REGISTRY_H__

#include <map>
#include <string>
#include <memory>
#include <functional>

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "mc2_log.h"

#include "matmul_base_tiling.h"
#include "matmul_tiling_cfg.h"
#include "platform/soc_spec.h"

namespace optiling {
struct Mc2MMRegisterCfg {
    const char *opType{nullptr};
    NpuArch npuArch{NpuArch::DAV_RESV};
    std::vector<int32_t> priorities{}; // 0 base
};

template <typename T>
std::unique_ptr<Mc2MatMulBaseTiling> MM_TILING_CLASS(gert::TilingContext *context, Mc2MatMulTilingCfg &cfg)
{
    return std::unique_ptr<T>(new (std::nothrow) T(context, cfg));
}

using Mc2MMTilingClassCase = std::unique_ptr<Mc2MatMulBaseTiling> (*)(gert::TilingContext *, Mc2MatMulTilingCfg &);

class Mc2MMTilingCases {
public:
    explicit Mc2MMTilingCases(std::string opType) : opType_(std::move(opType))
    {
    }

    template <typename T>
    void AddTiling(int32_t priority)
    {
        OPS_ERR_IF(cases_.find(priority) != cases_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(opType_, "There are duplicate registrations."), return);
        cases_[priority] = MM_TILING_CLASS<T>;
        OPS_ERR_IF(cases_[priority] == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opType_, "Register op tiling func failed, please check the class name."),
                   return);
    }

    const std::map<int32_t, Mc2MMTilingClassCase> &GetTilingCases()
    {
        return cases_;
    }

private:
    std::map<int32_t, Mc2MMTilingClassCase> cases_;
    const std::string opType_;
};

class Mc2MMTilingRegistry {
public:
    Mc2MMTilingRegistry() = default;

#ifdef ASCENDC_OP_TEST
    static Mc2MMTilingRegistry &GetInstance();
#else
    static Mc2MMTilingRegistry &GetInstance()
    {
        static Mc2MMTilingRegistry registryImpl_;
        return registryImpl_;
    }
#endif

    std::shared_ptr<Mc2MMTilingCases> RegisterOp(const std::string &opType, NpuArch npuArch)
    {
        auto socIter = registryMap_.find(npuArch);
        if (socIter == registryMap_.end()) {
            std::map<std::string, std::shared_ptr<Mc2MMTilingCases>> opTypeMap;
            opTypeMap[opType] = std::shared_ptr<Mc2MMTilingCases>(new (std::nothrow) Mc2MMTilingCases(opType));
            registryMap_[npuArch] = opTypeMap;
        } else {
            if (socIter->second.find(opType) == socIter->second.end()) {
                socIter->second[opType] =
                    std::shared_ptr<Mc2MMTilingCases>(new (std::nothrow) Mc2MMTilingCases(opType));
            }
        }

        OPS_ERR_IF(registryMap_[npuArch][opType] == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opType, "Register tiling func failed, please check the class name."),
                   return nullptr);
        return registryMap_[npuArch][opType];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context, Mc2MatMulTilingCfg &tilingCfg,
                                 const Mc2MMRegisterCfg &registerCfg)
    {
        if (context == nullptr || tilingCfg.compileInfo == nullptr || tilingCfg.args == nullptr) {
            OPS_LOG_E(context, "DoTilingImpl failed, context or tilingCfg or args is null.");
            return ge::GRAPH_FAILED;
        }
        const char *opType = registerCfg.opType == nullptr ? context->GetNodeType() : registerCfg.opType;
        auto tilingTemplateRegistryMap = GetTilingTemplates(opType, registerCfg.npuArch);
        OPS_LOG_D(context, "registry map find by opType %s, soc version %d", opType,
                  static_cast<int32_t>(registerCfg.npuArch));
        if (tilingTemplateRegistryMap.empty()) {
            OPS_LOG_E(context, "no registry map find by opType %s, soc version %d", opType,
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

    const std::map<int32_t, Mc2MMTilingClassCase> &GetTilingTemplates(const std::string &opType, NpuArch npuArch)
    {
        auto socIter = registryMap_.find(npuArch);
        OPS_ERR_IF(socIter == registryMap_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(opType, "Get op tiling func failed, please check the soc version %d",
                                               static_cast<int32_t>(npuArch)),
                   return emptyTilingCase_);
        auto opIter = socIter->second.find(opType);
        OPS_ERR_IF(opIter == socIter->second.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(opType, "Get op tiling func failed, please check the op name."),
                   return emptyTilingCase_);
        return opIter->second->GetTilingCases();
    }

private:
    std::map<NpuArch, std::map<std::string, std::shared_ptr<Mc2MMTilingCases>>> registryMap_; // key is npuArch
    const std::map<int32_t, Mc2MMTilingClassCase> emptyTilingCase_{};
};

class Mc2MMRegister {
public:
    explicit Mc2MMRegister(std::string opType) : opType_(std::move(opType))
    {
    }

    template <typename T>
    Mc2MMRegister &tiling(int32_t priority, NpuArch npuArch)
    {
        auto tilingCases = Mc2MMTilingRegistry::GetInstance().RegisterOp(opType_, npuArch);
        OPS_ERR_IF(tilingCases == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(opType_, "Register op tiling failed, please the op name."),
                   return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

private:
    const std::string opType_;
};

// opType: 算子名称， className: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
// 取代 MC2_MM_REGISTER_TILING_TEMPLATE , 传入的op_type如果是字符串常量，需要去掉引号
#define MC2_MM_REGISTER_TILING_TEMPLATE(opType, className, npuArch, priority)                                          \
    [[maybe_unused]] uint32_t op_impl_register_template_##opType##_##className##_##npuArch##_##priority;               \
    static Mc2MMRegister __attribute__((unused)) mc2_mm_register_##opType##_##className##_##npuArch##_##priority##_ =  \
        Mc2MMRegister(#opType).tiling<className>(static_cast<int32_t>(priority), NpuArch::npuArch)
} // namespace optiling

#endif // __OP_HOST_MATMUL_TILING_REGISTRY_H__
