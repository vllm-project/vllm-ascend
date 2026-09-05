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
 * \file matmul_v3_tiling_key.h
 * \brief
 */
#pragma once

#include <sstream>
#include "../../../op_kernel/arch35/mat_mul_v3_tiling_key_public.h"

namespace optiling {
namespace matmul_v3_advanced {
constexpr uint64_t DECIMAL_DIVISOR = 10UL;

class MMTilingKey {
public:
    virtual uint64_t GetTilingKey() const = 0;
    virtual ~MMTilingKey() = default;
};
class MatMulV3TilingKey : public MMTilingKey {
public:
    MatMulV3TilingKey& SetTrans(bool aTrans, bool bTrans)
    {
        if (aTrans) {
            atrans_ = MatMulV3ATrans::A_TRANS;
        }
        if (bTrans) {
            btrans_ = MatMulV3BTrans::B_TRANS;
        }
        return *this;
    }

    MatMulV3TilingKey& SetModel(MatMulV3Model model)
    {
        model_ = model;
        return *this;
    }

    MatMulV3TilingKey& SetBatchModel(MatMulV3BatchModel batchModel)
    {
        batchModel_ = batchModel;
        return *this;
    }

    static uint64_t GetHexTilingKey(const uint64_t tilingkey)
    {
        std::stringstream ss;
        ss << std::hex << std::uppercase << tilingkey;
        std::string tilingkey_str = ss.str();
        uint64_t tilingkey_hex = std::stoull(tilingkey_str);
        return tilingkey_hex;
    }

    static MatMulV3Model GetModel(const uint64_t tilingkey)
    {
        uint64_t tilingkey_hex = GetHexTilingKey(tilingkey);
        constexpr uint64_t ModelDigit = 3;
        uint64_t divisor = 1;
        for (uint64_t i = 0; i < ModelDigit; i++) {
            divisor *= DECIMAL_DIVISOR; // Obtain digit of one decimal number
        }
        return static_cast<MatMulV3Model>((tilingkey_hex / divisor) % DECIMAL_DIVISOR);
    }

    static MatMulV3BatchModel GetBatchModel(const uint64_t tilingkey)
    {
        uint64_t tilingkey_hex = GetHexTilingKey(tilingkey);
        constexpr uint64_t BachModelDigit = 2;
        uint64_t divisor = 1;
        for (uint64_t i = 0; i < BachModelDigit; i++) {
            divisor *= DECIMAL_DIVISOR; // Obtain digit of one decimal number
        }
        return static_cast<MatMulV3BatchModel>((tilingkey_hex / divisor) % DECIMAL_DIVISOR);
    }

    MatMulV3TilingKey& SetApiLevel(MatMulV3ApiLevel apiLevel)
    {
        apiLevel_ = apiLevel;
        return *this;
    }

    static MatMulV3ApiLevel GetApiLevel(const uint64_t tilingkey)
    {
        uint64_t tilingkey_hex = GetHexTilingKey(tilingkey);
        return static_cast<MatMulV3ApiLevel>(tilingkey_hex % DECIMAL_DIVISOR);
    }

    MatMulV3TilingKey& SetFullLoad(MatMulV3FullLoad fullLoad)
    {
        fullLoad_ = fullLoad;
        return *this;
    }

    MatMulV3TilingKey& SetL0C2Out(MatMulV3L0C2Out out)
    {
        out_ = out;
        return *this;
    }

    uint64_t GetTilingKey() const override;

protected:
    MatMulV3ATrans atrans_ = MatMulV3ATrans::A_NO_TRANS;
    MatMulV3BTrans btrans_ = MatMulV3BTrans::B_NO_TRANS;
    MatMulV3Model model_ = MatMulV3Model::BASIC;
    MatMulV3BatchModel batchModel_ = MatMulV3BatchModel::BATCH_MODEL;
    MatMulV3ApiLevel apiLevel_ = MatMulV3ApiLevel::HIGH_LEVEL;
    MatMulV3FullLoad fullLoad_ = MatMulV3FullLoad::NONE_FULL_LOAD;
    MatMulV3L0C2Out out_ = MatMulV3L0C2Out::ON_THE_FLY;
};
} // namespace matmul_v3_advanced
} // namespace optiling
