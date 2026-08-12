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
 * \file matmul_formulaic_tiling.h
 * \brief
 */
#ifndef __MATMUL_FORMULAIC_TILING_H__
#define __MATMUL_FORMULAIC_TILING_H__

#pragma once
#include "graph/ge_error_codes.h"
#include "../../op_kernel/mc2_tiling_struct.h"
#include "register/tilingdata_base.h"
#include "mc2_tiling_struct.h"
#include "tiling/tiling_api.h"

namespace mc2tiling {
constexpr uint32_t AC_MAX_AIV = 64;
constexpr uint32_t AC_MSG_CNT = 64;
constexpr uint32_t KB_SIZE = 1024;
constexpr uint32_t MB_SIZE = KB_SIZE * KB_SIZE;
constexpr uint32_t WORK_SPACE_OFFSET = 16 * MB_SIZE;
constexpr uint32_t NOTIFY_WRITE_CNT = 1;
constexpr uint64_t SHAPE_ALIGN_SIZE = 256;
constexpr uint32_t GATHER_FUNC_ID = 3;
constexpr uint32_t ALL_REDUCE_FUNC_ID = 2;
constexpr uint32_t REDUCE_SCATTER_FUNC_ID = 1;
constexpr uint32_t C0_SIZE = 16;
constexpr uint32_t BASE_BLOCK_M = 128;
constexpr uint32_t BASE_BLOCK_N = 256;
constexpr uint32_t BASE_BLOCK_K = 64;
constexpr uint32_t DEPTH_A1 = 8;
constexpr uint32_t DEPTH_B1 = 8;
constexpr uint32_t MIN_DEPTH = 2; // double buffer on
constexpr uint32_t DB_ON = 2;
constexpr uint32_t TOTAL_THRESHOLD = 300;
constexpr uint32_t INPUT_THRESHOLD = 128;
constexpr uint32_t L1_SIZE = 512 * KB_SIZE - 32; // reserve 32Bit for allreduce mix
constexpr uint32_t TOTAL_L1_SIZE = 512 * KB_SIZE;
constexpr uint32_t L0_SIZE_DB_ON = 32 * KB_SIZE;
constexpr uint32_t L0C_SIZE_DB_ON = 128 * KB_SIZE;
constexpr uint32_t L2_SIZE_FULL = 192 * MB_SIZE;
constexpr uint32_t INT8_SIZE = 1;
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t FP16_CUBE_CAL_POWER = 4096;
constexpr uint32_t BASE_K_ALIGN_SIZE = 64;
constexpr float FOMUL_AIC_NUM_THRESHOLD = 0.9f;

constexpr uint32_t BASE_BLOCK_M_L2CACHE = 256;
constexpr uint32_t BASE_BLOCK_N_L2CACHE = 256;
constexpr uint32_t BASE_BLOCK_K_L2CACHE = 64;
constexpr uint32_t DEPTH_A1_L2CACHE = 6;
constexpr uint32_t DEPTH_B1_L2CACHE = 6;

constexpr uint32_t BASE_BLOCK_M_L2CACHE_NZ = 128;
constexpr uint32_t BASE_BLOCK_N_L2CACHE_NZ = 256;
constexpr uint32_t BASE_BLOCK_K_L2CACHE_NZ = 64;
constexpr uint32_t DEPTH_A1_L2CACHE_NZ = 8;
constexpr uint32_t DEPTH_B1_L2CACHE_NZ = 8;

constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t ALIGN_64 = 64;
constexpr uint32_t ALIGN_128 = 128;
constexpr uint32_t ALIGN_256 = 256;
constexpr uint32_t ALIGN_512 = 512;
constexpr uint32_t TIMES_4 = 4;
constexpr uint32_t L0A_DTYPE_SIZE = 2;
constexpr uint32_t L0C_DTYPE_SIZE = 4;
// biasTable只有1024， 256 * 4B(跟L0C f32保持同类型占4B) = 1024B
constexpr uint32_t MAX_BIAS_BASE_BLOCK_N = 256;

enum class AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_HALFALLTOALLV = 20,
    HCCL_CMD_MAX
};

enum class HcclDataType {
    HCCL_DATA_TYPE_INT8 = 0,     /* *< int8 */
    HCCL_DATA_TYPE_INT16 = 1,    /* *< int16 */
    HCCL_DATA_TYPE_INT32 = 2,    /* *< int32 */
    HCCL_DATA_TYPE_FP16 = 3,     /* *< fp16 */
    HCCL_DATA_TYPE_FP32 = 4,     /* *< fp32 */
    HCCL_DATA_TYPE_INT64 = 5,    /* *< int64 */
    HCCL_DATA_TYPE_UINT64 = 6,   /* *< uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,    /* *< uint8 */
    HCCL_DATA_TYPE_UINT16 = 8,   /* *< uint16 */
    HCCL_DATA_TYPE_UINT32 = 9,   /* *< uint32 */
    HCCL_DATA_TYPE_FP64 = 10,    /* *< fp64 */
    HCCL_DATA_TYPE_BFP16 = 11,   /* *< bfp16 */
    HCCL_DATA_TYPE_HIF8 = 14,    /* *<hif8 */
    HCCL_DATA_TYPE_FP8E4M3 = 15, /* *<fp8e4m3 */
    HCCL_DATA_TYPE_FP8E5M2 = 16, /* *<fp8e5m2 */
    HCCL_DATA_TYPE_RESERVED      /* *< reserved */
};

enum class MC2_BUFFER_TYPE {
    MC2_BUFFER_TYPE_DEFAULT = 0,
    MC2_BUFFER_TYPE_OUTPUT,
    MC2_BUFFER_TYPE_WINDOW_IN,
    MC2_BUFFER_TYPE_WINDOW_OUT,
    MC2_BUFFER_TYPE_WORKSPACE,
    MC2_BUFFER_TYPE_INPUT,
    MC2_BUFFER_TYPE_COMMOUT,
    MC2_BUFFER_TYPE_END
};

enum class HcclReduceOp {
    HCCL_REDUCE_SUM = 0,  /* *< sum */
    HCCL_REDUCE_PROD = 1, /* *< prod */
    HCCL_REDUCE_MAX = 2,  /* *< max */
    HCCL_REDUCE_MIN = 3,  /* *< min */
    HCCL_REDUCE_RESERVED  /* *< reserved */
};

enum class KfcTaskType {
    KFC_TASK_HCC_RES_INIT = 1,     // 集合通信资源下发&校验
    KFC_TASK_HCC_START_SERVER = 2, // 只启动KFC server, 从msg queue接收任务
    KFC_TASK_HCC_TASK_PREPARE = 3, // 从参数获取通信任务, 完成准备, 待AIC通知消息再激活
    KFC_TASK_HCC_TASK_DELIVER = 4, // 从参数获取通信任务, 直接下发. AIC自己发Record激活
    KFC_TASK_COMMON_CMD_EXE = 5,   // 预留, 通用任务执行, 具体任务信息可以在参数中描述
    KFC_TASK_TYPE_END
};

enum class TilingKey {
    NO_ATRANS_NO_BTRANS_ALIGN = 1100,
    NO_ATRANS_NO_BTRANS_NO_ALIGN = 1000,
    ATRANS_NO_BTRANS_ALIGN = 1101,
    ATRANS_NO_BTRANS_NO_ALIGN = 1001,
    NO_ATRANS_BTRANS_ALIGN = 1102,
    NO_ATRANS_BTRANS_NO_ALIGN = 1002,
    ATRANS_BTRANS_ALIGN = 1103,
    ATRANS_BTRANS_NO_ALIGN = 1003,
};

struct TilingArgs {
    AicpuComType cmdType;
    uint32_t rankDim;
    uint32_t usedCoreNum;
    uint64_t orgMValue;
    uint64_t orgNValue;
    uint64_t orgKValue;
    uint64_t batchValue;
    uint64_t mValue;
    uint64_t kValue;
    uint64_t nValue;
    uint64_t baseMLimit; // 新增字段，归一方案里，用于多卡多片数据一把传给mmv3做tiling时，表示单卡单片数据的大小
    uint64_t inputDtypeSize;
    uint64_t outputDtypeSize;
    uint64_t aicCoreNum;
    uint64_t commTurn;
    uint64_t rankTileNum;
    uint8_t commAlg;
    bool isATrans;
    bool isBTrans;
    bool enablePad;
    bool enableSplitK;
    bool isBias;
    bool isStorageGather;
    bool isLocal = false;
    ge::DataType geAType;
    ge::DataType geBType;
    ge::DataType geCType;
    ge::DataType geBiasType;
    ge::DataType antiquantscaleDType;
    matmul_tiling::DataType aType;
    matmul_tiling::DataType bType;
    matmul_tiling::DataType cType;
    matmul_tiling::DataType biasType;
};

struct ChipArgs {
    float kilo;
    float bwUsage; // 带宽预计使用理论值的0.8
    float frequency;
    float hbmBWSingle;
    float hbmBWPeak;
    float l2BWSingle;
    float l2BWPeak;
    uint32_t l0ASize;
    uint32_t l0CSize;
};

struct BestArgs {
    uint32_t bestCore = 0;
    uint32_t bestRoundCnt = 0;
    uint32_t bestShort = 0;
    uint32_t bestLong = 0;
    uint32_t bestK = 0;
};

struct MatmulL2RunInfo {
    uint32_t mL2TileCnt = 1;
    uint32_t nL2TileCnt = 1;
    uint32_t mTileBlocks = 0;
    uint32_t nTileBlocks = 0;
    uint32_t mTailBlocks = 0;
    uint32_t nTailBlocks = 0;
    uint32_t calOrder = 0;
};

struct MatmulRunInfo {
    bool needUpdate = false;
    uint32_t usedCoreNum = 1;
    uint32_t singleCoreM = 1;
    uint32_t singleCoreN = 1;
    uint32_t singleCoreK = 1;
    uint32_t baseM = BASE_BLOCK_M;
    uint32_t baseN = BASE_BLOCK_N;
    uint32_t baseK = BASE_BLOCK_K;
    uint32_t stepKa = DEPTH_A1 / 2;
    uint32_t stepKb = DEPTH_B1 / 2;
    uint32_t depthA1 = DEPTH_A1;
    uint32_t depthB1 = DEPTH_B1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t iterateOrder = 0;
    uint32_t dbL0c = 0;
    MatmulL2RunInfo l2Info;
};

struct MatmulArguments {
    bool isATrans = false;
    bool isBTrans = false;
    bool isBias = false;
    uint32_t mValue = 0;
    uint32_t kValue = 0;
    uint32_t nValue = 0;
    uint32_t rankDim = 0;
    uint32_t rankM = 0;
    uint32_t rankTileM = 0;
    uint32_t rankTileNum = 0;
    uint32_t aicCoreNum = 20; // 20: default core num
    uint8_t aDtypeSize = 2;   // 2: default fp16
    uint8_t bDtypeSize = 2;   // 2: default fp16
    uint8_t cDtypeSize = 2;   // 2: default fp16
};

struct SoCInfo {
    uint64_t l1Size = L1_SIZE;
    uint64_t l2Size = L2_SIZE_FULL;
    uint64_t l0CSize = L0C_SIZE_DB_ON;
    uint64_t l0ASize = L0_SIZE_DB_ON;
    uint64_t l0BSize = L0_SIZE_DB_ON;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

class MatmulFormulaicTiling {
public:
    // Constructor
    explicit MatmulFormulaicTiling(std::string opName) : opName_(opName) {};
    static uint32_t GetRankSize(const char *group);
    void SetWeightFormat(const matmul_tiling::CubeFormat weightFormat);
    void SetSocVersion(const platform_ascendc::SocVersion &version)
    {
        socInfo_.socVersion = version;
    }
    ge::graphStatus GetCubeTiling(TilingArgs &args, AscendC::tiling::TCubeTiling &cubeTiling,
                                  Mc2Tiling::TileL2Tiling &tileL2Tiling);
    ge::graphStatus GetCubeTiling(TilingArgs &args, AscendC::tiling::TCubeTiling &cubeTiling);
    static void GetBaseBlockParm(const platform_ascendc::SocVersion &version, uint64_t &blockBaseM,
                                 uint64_t &blockBaseN, uint64_t &blockBaseK, uint64_t &blockDepthA1,
                                 uint64_t &blockDepthB1);

private:
    MatmulRunInfo runInfo_;
    MatmulArguments args_;
    SoCInfo socInfo_;
    std::string opName_;
    matmul_tiling::CubeFormat weightFormat_ = matmul_tiling::CubeFormat::ND;
    void CalcBaseBlockTiling();
    void UpdateDepth();
    bool DoL2CacheTiling();
    void InitBaseBlockTiling();
    void InitTilingArgs(TilingArgs &args);
    std::string opTypeStr_;
    platform_ascendc::SocVersion socVer_ = platform_ascendc::SocVersion::ASCEND910B;
    bool IsShapeAlign(uint64_t aValue, uint64_t bValue) const;
    uint32_t GetTilingKey(TilingArgs &args, bool &aligned) const;
    void CalcBaseBlockTiling(TilingArgs &args, uint64_t &baseM, uint64_t &baseN, uint64_t &baseK) const;
    void UpdateDepth(TilingArgs &args, uint64_t &depthA1, uint64_t &depthB1, uint64_t &baseM, uint64_t &baseN,
                     uint64_t &baseK) const;
};

template <typename T>
inline T MathCeil(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
inline T MathFloor(T num1, T num2)
{
    if (num2 == 0) {
        return num1;
    }
    return num1 / num2;
}

template <typename T>
inline T AlignUp(T num1, T num2)
{
    return MathCeil(num1, num2) * num2;
}

template <typename T>
inline T AlignDown(T num1, T num2)
{
    return MathFloor(num1, num2) * num2;
}
} // namespace mc2tiling

#endif // __MATMUL_FORMULAIC_TILING_H__
