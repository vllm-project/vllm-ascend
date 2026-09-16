/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_HCOM_TOPOLOGY_MOCKER_H
#define MC2_HCOM_TOPOLOGY_MOCKER_H

#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

namespace Mc2Hcom {
using MockValues = std::vector<std::pair<const char *, uint64_t>>;

class MC2HcomTopologyMocker {
public:
    static MC2HcomTopologyMocker &GetInstance();
    void SetValue(const char *key, uint64_t value);
    void SetValues(const MockValues &values);
    uint64_t GetValue(const char *key, uint64_t defaultValue) const;
    void Reset();

private:
    MC2HcomTopologyMocker() = default;
    std::unordered_map<std::string, uint64_t> mockValue_;
};

} // namespace Mc2Hcom

#endif // MC2_HCOM_TOPOLOGY_MOCKER_H
