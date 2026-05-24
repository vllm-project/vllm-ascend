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
 * \file static_space.h
 * \brief
 */
#ifndef CANN_OPS_STATIC_SPACE_H_
#define CANN_OPS_STATIC_SPACE_H_
#include "base/registry/op_impl_space_registry_v2.h"

class StaticSpaceInitializer {
public:
    static StaticSpaceInitializer& GetInstance() {
        static StaticSpaceInitializer instance;
        return instance;
    }
private:
    StaticSpaceInitializer () {
        auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        if (space_registry == nullptr) {
            space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
            gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
        }
    }
};
#endif