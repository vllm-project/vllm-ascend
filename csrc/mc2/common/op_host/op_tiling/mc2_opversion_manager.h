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
 * \file mc2_opversion_manager.h
 * \brief
 */

#ifndef __MC2_OPVERSION_MANAGER_H__
#define __MC2_OPVERSION_MANAGER_H__

#pragma once
#include <cstdint>

class OpVersionManager {
public:
    OpVersionManager(const OpVersionManager &) = delete;
    OpVersionManager &operator=(const OpVersionManager &) = delete;

    static void SetVersion(uint32_t version)
    {
        GetInstance().opVersion = version;
    }

    static OpVersionManager &GetInstance()
    {
        static OpVersionManager instance(0);
        return instance;
    }

    uint32_t GetVersion() const
    {
        return opVersion;
    }

private:
    uint32_t opVersion;
    explicit OpVersionManager(uint32_t version) : opVersion(version)
    {
    }
};

#endif // __MC2_OPVERSION_MANAGER_H__
