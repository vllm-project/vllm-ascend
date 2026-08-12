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
 * \file matmul_v3_tiling_key.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_TILING_KEY_H__
#define __OP_HOST_MATMUL_V3_TILING_KEY_H__

#include "op_host/tiling_key.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
using Ops::Transformer::OpTiling::GET_TILINGKEY;
constexpr uint64_t DECIMAL_DIVISOR = 10UL;
enum class Mc2MatMulV3Trans : std::uint8_t {
    NO_TRANS = 0,
    A_TRANS = 1,
    B_TRANS = 2,
    AB_TRANS = 3
};

enum class Mc2MatMulV3ApiLevel : std::uint8_t {
    HIGH_LEVEL = 0,
    BASIC_LEVEL = 1
};

enum class Mc2MatMulV3Model : std::uint8_t {
    BASIC = 0,
    ITER_BATCH_BATCH_BIAS = 1,
    STREAM_K = 2,
    ITER_BATCH_SINGLE_BIAS = 3,
    DOUBLE_ASWT = 4
};

enum class Mc2MatMulV3FullLoad : std::uint8_t {
    NONE_FULL_LOAD = 0,
    A_FULL_LOAD = 1,
    B_FULL_LOAD = 2,
    AB_FULL_LOAD = 3
};

enum class Mc2MatMulV3GM2L1 : std::uint8_t {
    ON_THE_FLY = 0,
    MOVE_ALIGN = 1,
    VNCHW_CONV = 2
};

enum class Mc2MatMulV3L0C2Out : std::uint8_t {
    ON_THE_FLY = 0,
    ND_FIXPIPE_1_1 = 1,
    ND_FIXPIPE_1_2 = 2,
    NZ_TRANSDATA = 3
};

class Mc2MatMulV3TilingKey {
public:
    Mc2MatMulV3TilingKey &SetTrans(bool aTrans, bool bTrans)
    {
        trans_ = Mc2MatMulV3Trans::NO_TRANS;
        if (aTrans && bTrans) {
            trans_ = Mc2MatMulV3Trans::AB_TRANS;
        } else if (aTrans) {
            trans_ = Mc2MatMulV3Trans::A_TRANS;
        } else if (bTrans) {
            trans_ = Mc2MatMulV3Trans::B_TRANS;
        }
        return *this;
    }

    Mc2MatMulV3TilingKey &SetModel(Mc2MatMulV3Model model)
    {
        model_ = model;
        return *this;
    }

    Mc2MatMulV3Model GetModel(const uint64_t tilingkey) const
    {
        constexpr uint64_t modelDigit = 5;
        uint64_t divisor = 1;
        for (uint64_t i = 0; i < modelDigit; i++) {
            divisor *= DECIMAL_DIVISOR; // Obtain digit of one decimal number
        }
        return static_cast<Mc2MatMulV3Model>((tilingkey / divisor) % DECIMAL_DIVISOR);
    }

    Mc2MatMulV3TilingKey &SetApiLevel(Mc2MatMulV3ApiLevel apiLevel)
    {
        apiLevel_ = apiLevel;
        return *this;
    }

    Mc2MatMulV3ApiLevel GetApiLevel(const uint64_t tilingkey) const
    {
        constexpr uint64_t apiLevelDigit = 15;
        uint64_t divisor = 1;
        for (uint64_t i = 0; i < apiLevelDigit; i++) {
            divisor *= DECIMAL_DIVISOR; // Obtain digit of one decimal number
        }
        return static_cast<Mc2MatMulV3ApiLevel>((tilingkey / divisor) % DECIMAL_DIVISOR);
    }

    Mc2MatMulV3TilingKey &SetFullLoad(Mc2MatMulV3FullLoad fullLoad)
    {
        fullLoad_ = fullLoad;
        return *this;
    }

    Mc2MatMulV3TilingKey &SetAGM2L1(Mc2MatMulV3GM2L1 aGm2L1)
    {
        aGm2L1_ = aGm2L1;
        return *this;
    }

    Mc2MatMulV3TilingKey &SetBGM2L1(Mc2MatMulV3GM2L1 bGm2L1)
    {
        bGm2L1_ = bGm2L1;
        return *this;
    }

    Mc2MatMulV3TilingKey &SetL0C2Out(Mc2MatMulV3L0C2Out out)
    {
        out_ = out;
        return *this;
    }

    uint64_t GetTilingKey() const
    {
        return GET_TILINGKEY(trans_, 0, 0, 0, 9,           // 9: delimiter
                             model_, fullLoad_, 0, 0, 9,   // 9: delimiter
                             aGm2L1_, bGm2L1_, out_, 0, 9, // 9: delimiter
                             apiLevel_);
    }

private:
    Mc2MatMulV3Trans trans_ = Mc2MatMulV3Trans::NO_TRANS;
    Mc2MatMulV3Model model_ = Mc2MatMulV3Model::BASIC;
    Mc2MatMulV3ApiLevel apiLevel_ = Mc2MatMulV3ApiLevel::HIGH_LEVEL;
    Mc2MatMulV3FullLoad fullLoad_ = Mc2MatMulV3FullLoad::NONE_FULL_LOAD;
    Mc2MatMulV3GM2L1 aGm2L1_ = Mc2MatMulV3GM2L1::ON_THE_FLY;
    Mc2MatMulV3GM2L1 bGm2L1_ = Mc2MatMulV3GM2L1::ON_THE_FLY;
    Mc2MatMulV3L0C2Out out_ = Mc2MatMulV3L0C2Out::ON_THE_FLY;
};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling

#endif // __OP_HOST_MATMUL_V3_STREAM_K_TILING_H__
