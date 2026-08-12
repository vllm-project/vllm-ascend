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
 * \file matmul_v3_common.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_COMMON_H__
#define __OP_HOST_MATMUL_V3_COMMON_H__

#include <cstdint>
#include <map>
#include "graph/types.h"
#include "util/math_util.h"
#include "mc2_log_compat.h"

namespace optiling {
namespace mc2_matmul_v3 {

constexpr uint64_t BASIC_ALIGN_8 = 8;
constexpr uint64_t BASIC_ALIGN_16 = 16;
constexpr uint64_t BASIC_ALIGN_32 = 32;
constexpr uint64_t BASIC_ALIGN_256 = 256;
constexpr uint64_t BASIC_ALIGN_512 = 512;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t KB_SIZE = 1024;
constexpr uint64_t DB_SIZE = 2;
constexpr uint64_t DB_OFF_SIZE = 1;
constexpr uint64_t LOC_DATA_SIZE = 4;
constexpr uint64_t L2_TILE_LENGTH = 3072;
constexpr uint64_t L2_TILE_LENGTH_L2_128 = 1536;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16;
constexpr uint64_t BASIC_BLOCK_K_256_BYTE = 256;
constexpr uint64_t BASIC_BLOCK_K_128_BYTE = 128;
constexpr uint64_t BIAS_TABLE_NUM = 256;
constexpr uint64_t BASIC_ALIGN_BLK = 256;
constexpr uint64_t DATA_SIZE_FP32 = 4;
constexpr uint64_t DATA_SIZE_FP16 = 2;
constexpr uint64_t L0C_SIZE_256_KB = 262144;
constexpr uint64_t NUM_HALF = 2;
constexpr uint64_t BASE_STEP = 1;
constexpr uint64_t ITER_COL_FIRST = 0;
constexpr uint64_t ITER_ROW_FIRST = 1;
constexpr uint64_t N_FIRST_COEFF = 10;
constexpr uint64_t SMALL_SHAPE_THRES = 256;
constexpr uint64_t IF_MULTIK_THRES = 8;
constexpr uint64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint64_t FP32_HF32_DTYPE_SIZE = 4;
constexpr uint64_t SPLIT_K_THRES = 27392;
constexpr uint64_t ND2NZ_ON_THE_FLY_LIMIT = 65535;
constexpr uint64_t DETER_THRES_OUT_SIZE = 4;
constexpr uint64_t ALIGN_128 = 128;
constexpr uint64_t ALIGN_OUTER = 32;
constexpr uint64_t ALIGN_INNER = 256;
constexpr uint64_t RPC_WORKSIZE = 20;
constexpr size_t BIAS_IDX = 2;
constexpr uint64_t NMK_N_THERS = 64;
constexpr uint64_t NMK_M_THERS = 1920;

enum class Mc2CalcType : int32_t {
    M_BY_BASE_NK,
    MN_BY_BASE_K
};

enum class Mc2MatmulV3Trans : int32_t {
    NO_TRANS = 0,
    A_TRANS = 1,
    B_TRANS = 2,
    AB_TRANS = 3
};

enum class Mc2MixNd2NzType : int32_t {
    V_HEAD_ND2NZ = 0,    // 所有核在头部做nd2nz
    NO_ND2NZ = 1,        // 不做nd2nz
    V_PARALELL_ND2NZ = 2 // vect和cube并行做nd2nz
};

enum class Mc2TilingCalcSelect : int32_t // 选择不同的计算Tiling的方法
{
    ALL = 0,
    BASE = 1,
    SINGLE_CORE_SPLIT_K = 2,
    DETERMINISTIC_SPLIT_K = 3
};

enum class Mc2TilingEnableSplitCore : int32_t // 互斥flag, 对应不同切K模板选择
{
    BASE = 0,
    SINGLE_CORE_SPLIT_K = 2,
    DETERMINISTIC_SPLIT_K = 3,
    MULTI_CORE_SPLIT_K = 4,
    SINGLE_CORE_NKM_SPLIT_K = 5,
    SINGLE_CORE_SPLIT_K_GM_TO_L1 = 6,
    MAX = 10 // 模板类别不能超过10个
};

enum class Mc2TilingEnableFullLoad : int32_t // 互斥flag, 对应不同全载模板选择
{
    BASE = 0,
    AL1_FULL_LOAD = 1,
    BL1_FULL_LOAD = 2,
    MAX = 10 // 模板类别不能超过10个
};

enum class Mc2TilingEnableFixOpti : int32_t // 互斥flag, 对应不同输出优化模板选择
{
    BASE = 0,
    BASE_ENABLE_ALIGNOUT = 1,
    VEC_NZ2ND_UNALIGNOUT = 2,
    MAX = 10 // 模板类别不能超过10个
};

struct Mc2TilingEnable {
    Mc2TilingEnableSplitCore tilingEnableSplitCore = Mc2TilingEnableSplitCore::BASE; // aoetilingenable的个位
    Mc2TilingEnableFullLoad tilingEnableFullLoad = Mc2TilingEnableFullLoad::BASE;    // aoetilingenable的十位
    Mc2TilingEnableFixOpti tilingEnableFixOpti = Mc2TilingEnableFixOpti::BASE;       // aoetilingenable的千位
};

struct Mc2MatmulV3Args {
    const char *opName = nullptr;
    bool isATrans = false;
    bool isBTrans = false;
    bool isHf32 = false;
    bool hasBias = false;
    bool nd2nzA = false;
    bool nd2nzB = false;
    bool isNzA = false;
    bool isNzB = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::Format aFormat = ge::FORMAT_ND;
    ge::Format bFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_ND;
    uint8_t unAlignProcessType = 0;
    uint64_t mValue = 0L;
    uint64_t mOriValue = 0L;
    uint64_t nOriValue = 0L;
    uint64_t kValue = 0L;
    uint64_t nValue = 0L;
    double l2Ratio = 0;
};

struct Mc2MatmulV3L2RunInfo {
    uint64_t mTile = 1;
    uint64_t nTile = 1;
    uint64_t mTileBlock = 0;
    uint64_t nTileBlock = 0;
    uint64_t calOrder = 0;
};

struct Mc2MatmulV3RunInfo {
    bool needUpdate = false;
    uint64_t usedCoreNum = 1;
    uint64_t singleCoreM = 1;
    uint64_t singleCoreN = 1;
    uint64_t singleCoreK = 1;
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t stepKa = 1;
    uint64_t stepKb = 1;
    uint64_t depthA1 = 1;
    uint64_t depthB1 = 1;
    uint64_t stepM = 1;
    uint64_t stepN = 1;
    uint64_t iterateOrder = 0;
    uint64_t dbL0c = 0;
    uint64_t baseAN = 0;
    uint64_t baseAD = 0;
    uint64_t baseBN = 0;
    uint64_t baseBD = 0;
    Mc2MatmulV3L2RunInfo l2Info;
};

struct Mc2MatmulV3L2SplitParams {
    uint64_t outBase = 0;
    uint64_t innerBase = 0;
    uint64_t outValue = 0;
    uint64_t innerValue = 0;
    uint64_t outDtypeSize = 0;
    uint64_t innerDtypeSize = 0;
    uint64_t maxConflictDim = 0;
    uint64_t minConflictDim = 0;
    uint64_t outTailCnt = 0;
    uint64_t innerTailCnt = 0;
};

const std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP = {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
};

template <typename T>
inline bool Is256BAlign(T base, uint64_t dTypeSize)
{
    if (base * dTypeSize % 256 == 0) { // 256: align byte size
        return true;
    }
    return false;
};

template <typename T>
inline bool Is32BAlign(T base, uint64_t dTypeSize)
{
    if (base * dTypeSize % 32 == 0) { // 32: align byte size
        return true;
    }
    return false;
};

template <typename T>
inline bool Is16Align(T base)
{
    if (base % BASIC_ALIGN_16 == 0) {
        return true;
    }
    return false;
};

inline uint64_t GetSizeC0(const uint64_t &dataTypeSize)
{
    uint64_t c0Size = BASIC_ALIGN_16;
    if (dataTypeSize == sizeof(float)) {
        c0Size = BASIC_ALIGN_8;
    } else if (dataTypeSize == sizeof(int8_t)) {
        c0Size = BASIC_ALIGN_32;
    }
    return c0Size;
}

template <typename T>
inline bool IsNumAlign(T base, uint64_t size)
{
    if (size == 0UL) {
        return false;
    }

    if (base % size == 0UL) {
        return true;
    }
    return false;
};

template <typename T>
inline bool CheckNumberIsValid(const T &number, const std::string &opName, const std::string &description)
{
    if (number > static_cast<uint64_t>(UINT32_MAX)) {
        OP_LOGW(opName, "%s size is greater than UINT32_MAX or less than 0, number: %s", description.c_str(),
                std::to_string(number).c_str());
        return true;
    }
    return false;
};

// 需要对齐16的参数需要判断是否大于floorAlign(uint32_max, 16)
template <typename T>
inline bool CheckNumberIsValid2(const T &number, const std::string &opName, const std::string &description)
{
    if (number > Ops::Base::FloorAlign(static_cast<uint64_t>(UINT32_MAX), BASIC_ALIGN_16)) {
        OP_LOGW(opName, "%s size is greater than floorAlign(UINT32_MAX, 16) or less than 0, number: %s",
                description.c_str(), std::to_string(number).c_str());
        return true;
    }
    return false;
};
} // namespace mc2_matmul_v3
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_COMMON_H__
