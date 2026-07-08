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
 * \file mc2_tiling_struct.h
 * \brief
 */
#ifndef MC2_TILING_STRUCT_H
#define MC2_TILING_STRUCT_H

#include "kernel_tiling/kernel_tiling.h"

namespace Mc2Tiling {
constexpr uint8_t COMM_ALG_DEFAULT = 0;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_ALG_DOUBLE_RING = 2;
constexpr uint8_t COMM_ALG_SWITCH_WING = 3;
constexpr uint32_t DOUBLE_RING_FACTOR = 2;
constexpr int64_t KVALUE_MIN = 256;
constexpr int64_t KVALUE_MAX = 65535;

struct MC2ServerCfg {          // server 通用参数
    uint32_t verseion;        // tiling结构体版本号
    uint8_t debugMode;        // 调测模式
    uint8_t sendArgIndex;     // 发送数据参数索引，对应算子原型的参数顺序
    uint8_t recvArgIndex;     // 接收数据参数索引，对应算子原型的参数顺序
    uint8_t commOutArgIndex;  // 通信输出参数索引，对应算子原型的参数顺序
    uint8_t reserved[8];      // 保留字段
}; // 16 bytes

struct MC2HcommCfg {
    uint8_t skipLocalRankCopy;     // 跳过本卡拷贝，在通信结果只需要给MC2内部计算使用或者本卡拷贝由aicore完成时，
                                   // 可以跳过本卡数据send-recv搬运
    uint8_t skipBufferWindowCopy;  // 跳过hbm到window间搬运 0 不跳过， 1 跳过snd-window， 2 跳过 window-rcv
    uint8_t stepSize;              // 通信步长，粗粒度融合时填0
                                   // 细粒度融合时连续计算stepsize块数据再commit或wait通信
    char reserved[13];             // 保留字段,前面是3字节，为字节对齐，预留13字节
    char groupName[128];           // groupName， 128是通信域名称限制
    char algConfig[128];           // 算法配置， 128是算法名称限制
    uint32_t opType;               // tiling结构体版本号
    uint32_t reduceType;           // reduce类型
    uint32_t srcDataType;          // 输入数据类型
    uint32_t dstDataType;          // 输出数据类型
};// 280 bytes

struct Mc2Msg {                  // 同aicpu_hccl_def KFCTilingData
    uint32_t preparePosition;   // 任务准备位置：0表示在host完成所有通信任务的准备，1表示在kernel侧完成
    uint64_t sendOff;           // 发送数据地址偏移，count * dataTypeSize
    uint64_t recvOff;           // 接收数据地址偏移, count * dataTypeSize
    uint64_t tailSendOff;       // 尾块发送数据地址偏移，count * dataTypeSize
    uint64_t tailRecvOff;       // 尾块接收数据地址偏移, count * dataTypeSize
    uint64_t sendCnt;           // 整块发送数据个数
    uint64_t recvCnt;           // 整块接收数据个数
    uint64_t tailSendCnt;       // 尾块发送数据个数
    uint64_t tailRecvCnt;       // 尾块接收数据个数
    uint64_t totalCnt;          // 总数据个数
    uint32_t turnNum;           // 总轮次
    uint32_t tailNum;           // 总轮次
    uint32_t stride;            // 跳写间隔
    uint32_t workspaceOff;      // 使用workspace作为recvbuf时的workspace偏移
    uint32_t notifyOff;         // device notify write/read value偏移

    uint16_t notifyBeginCnt;    // notift write value的使用个数
    uint16_t notifyEndCnt;      // notift write value的使用个数
    uint8_t useBufferType;      // recvBuf类型
    uint8_t funID;              // funtion ID
    uint8_t dataType;           // hccl 数据类型
    uint8_t groupNum;           // groupNum

    uint8_t reuseMode;          // 不复用填turnNum，内存优化选择复用的内存块个数
    uint8_t commType;           // 通信类型
    uint8_t reduceOp;           // reduce op type
    uint8_t commOrder;          // 通信顺序，0表示通信在前，1表示通信在后
    uint8_t waitPolicy;         // 等待任务启动的阻塞策略，2、首轮等待，1、每轮等待。
                                // KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
    uint8_t rspPolicy;          // 任务执行结束时的响应策略， 2、最后通知一次，
                                // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
    uint8_t exitPolicy;         // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
                                // 等待AIC通知退出(可以多次执行任务)。
    uint8_t commAlg;            // 用于指定具体通信算法。
                                // 0：defualt, 1：fullmesh, 2：doublering, 3：switchwing
    uint8_t taskType;           // 从参数获取通信任务，直接下发。AIC自己发Record激活
    uint8_t debugMode;          // 调测模式
                                // 1:单独执行CUBE
                                // 2:单独执行Vector
                                // 4:单独执行AICPU KFC算子
                                // 8:KFC等待通信结束
                                // 16:KFC统计各阶段耗时
    uint8_t stepSize;           // 用于指定通算频率步长
    uint8_t sendArgIndex;       // 发送数据参数索引，对应算子原型的参数顺序
    uint8_t recvArgIndex;       // 接收数据参数索引，对应算子原型的参数顺序
    uint8_t commOutArgIndex;    // 通信输出参数索引，对应算子原型的参数顺序
    uint8_t hasCommOut;         // 是否有通信输出
    uint8_t reserve;
    uint32_t reserve2;
};

struct RCSTiling {
    uint32_t rankDim;
    uint32_t rankID;
    uint32_t commtype;
    uint32_t subtype;
    uint32_t tileCnt;
    uint32_t tailM;
    uint32_t tailCnt;
    uint32_t biasLen;
    uint32_t isAdd;
    uint32_t rankM;
    uint32_t rankN;
    uint32_t rankK;
    uint32_t gatherIndex;
    uint32_t isTransposeA;
    uint32_t isTransposeB;
    uint32_t storageGather;
    uint64_t nd2NzWorkLen;
    uint64_t cToFloatLen;
    uint64_t gatherLen;
    uint32_t workspaceAddr4;
    uint32_t aicCoreNum;
    uint32_t needUbBuffer;
    uint32_t addX3UbCnt;
    uint32_t commWorkSpaceSize;
    uint32_t isInputCommQuantScale;
    uint32_t dataType;
    uint32_t commInt8WorkSpace;
    uint32_t dynamicQuantTempBuffSize;
};

struct TileL2Tiling {
    uint32_t mL2TileCnt;
    uint32_t nL2TileCnt;
    uint32_t mTileBlocks;
    uint32_t nTileBlocks;
    uint32_t mTailBlocks;
    uint32_t nTailBlocks;
    uint32_t rankTileNum;
    uint32_t calcOrder;
    uint32_t enableL2Tile;
};

struct TileInfo {
    uint32_t mTailCnt;
    uint32_t nTailCnt;
    uint32_t kTailCnt;
    uint32_t isHf32;
};

struct MC2MatmulV3TilingData {
    TCubeTiling matmulTiling;
    uint32_t mTailCnt;
    uint32_t nTailCnt;
    uint32_t kTailCnt;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
    uint32_t isHf32;
    uint32_t aswWindowLen;
};
}
#endif // MC2_TILING_STRUCT_H
