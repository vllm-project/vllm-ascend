/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// mc2注释
//
//
//
//
// mc2注释
/*!
 * \file mc2_tiling_struct.h
 * \brief
 */

#ifndef __MC2_TILING_STRUCT_H__
#define __MC2_TILING_STRUCT_H__

#include "register/tilingdata_base.h"
#include "../../../3rd/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_tiling.h"

namespace optiling {
constexpr uint8_t COMM_ALG_DEFAULT = 0;
constexpr uint8_t COMM_ALG_FULL_MESH = 1;
constexpr uint8_t COMM_ALG_DOUBLE_RING = 2;
constexpr uint8_t COMM_ALG_SWITCH_WING = 3;
constexpr uint32_t DOUBLE_RING_FACTOR = 2;
constexpr int64_t KVALUE_MIN = 256;
constexpr int64_t KVALUE_MAX = 65535;

/**
 * @deprecated MC2ServerCfg 结构体在后续版本中将不再支持，请不要使用
 */

BEGIN_TILING_DATA_DEF(MC2ServerCfg)              // server 通用参数
TILING_DATA_FIELD_DEF(uint32_t, verseion);       // tiling结构体版本号
TILING_DATA_FIELD_DEF(uint8_t, debugMode);       // 调测模式
TILING_DATA_FIELD_DEF(uint8_t, sendArgIndex);    // 发送数据参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF(uint8_t, recvArgIndex);    // 接收数据参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF(uint8_t, commOutArgIndex); // 通信输出参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF_ARR(uint8_t, 8, reserved); // 保留字段
END_TILING_DATA_DEF;                             // 16 bytes
REGISTER_TILING_DATA_CLASS(MC2ServerCfgOp, MC2ServerCfg)


/**
 * @deprecated MC2HcommCfg 结构体在后续版本中将不再支持，请不要使用
 */

BEGIN_TILING_DATA_DEF(MC2HcommCfg)
TILING_DATA_FIELD_DEF(uint8_t,
                      skipLocalRankCopy); // 跳过本卡拷贝，在通信结果只需要给MC2内部计算使用或者本卡拷贝由aicore完成时，
                                          // 可以跳过本卡数据send-recv搬运
TILING_DATA_FIELD_DEF(uint8_t,
                      skipBufferWindowCopy); // 跳过hbm到window间搬运 0 不跳过， 1 跳过snd-window， 2 跳过 window-rcv
TILING_DATA_FIELD_DEF(uint8_t, stepSize);      // 通信步长，粗粒度融合时填0
                                               // 细粒度融合时连续计算stepsize块数据再commit或wait通信
TILING_DATA_FIELD_DEF_ARR(char, 13, reserved); // 保留字段
TILING_DATA_FIELD_DEF_ARR(char, 128, groupName); // groupName
TILING_DATA_FIELD_DEF_ARR(char, 128, algConfig); // 算法配置
TILING_DATA_FIELD_DEF(uint32_t, opType);         // tiling结构体版本号
TILING_DATA_FIELD_DEF(uint32_t, reduceType);     // reduce类型
TILING_DATA_FIELD_DEF(uint32_t, srcDataType);    // 输入数据类型
TILING_DATA_FIELD_DEF(uint32_t, dstDataType);    // 输出数据类型
END_TILING_DATA_DEF;                             // 280 bytes
REGISTER_TILING_DATA_CLASS(MC2HcommCfgOp, MC2HcommCfg)

/**
 * @deprecated RCSTiling 结构体在后续版本中将不再支持，请不要使用
 */

BEGIN_TILING_DATA_DEF(RCSTiling)
TILING_DATA_FIELD_DEF(uint32_t, rankDim);
TILING_DATA_FIELD_DEF(uint32_t, rankID);
TILING_DATA_FIELD_DEF(uint32_t, commtype);
TILING_DATA_FIELD_DEF(uint32_t, subtype);

TILING_DATA_FIELD_DEF(uint32_t, tileCnt); // 整块的个数
TILING_DATA_FIELD_DEF(uint32_t, tailM);
TILING_DATA_FIELD_DEF(uint32_t, tailCnt);
TILING_DATA_FIELD_DEF(uint32_t, biasLen);
TILING_DATA_FIELD_DEF(uint32_t, isAdd);

TILING_DATA_FIELD_DEF(uint32_t, rankM); // 存放用户原始输入的mValue
TILING_DATA_FIELD_DEF(uint32_t, rankN); // 存放用户原始输入的mValue
TILING_DATA_FIELD_DEF(uint32_t, rankK);
TILING_DATA_FIELD_DEF(uint32_t, gatherIndex);
TILING_DATA_FIELD_DEF(uint32_t, isTransposeA);
TILING_DATA_FIELD_DEF(uint32_t, isTransposeB);

TILING_DATA_FIELD_DEF(uint32_t, storageGather);
TILING_DATA_FIELD_DEF(uint64_t, nd2NzWorkLen);
TILING_DATA_FIELD_DEF(uint64_t, cToFloatLen);
TILING_DATA_FIELD_DEF(uint64_t, gatherLen);
TILING_DATA_FIELD_DEF(uint32_t, workspaceAddr4);
TILING_DATA_FIELD_DEF(uint32_t, aicCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, needUbBuffer);
TILING_DATA_FIELD_DEF(uint32_t, addX3UbCnt);
TILING_DATA_FIELD_DEF(uint32_t, commWorkSpaceSize);     // 没有 int8 通信时总共 workspace 的开销
TILING_DATA_FIELD_DEF(uint32_t, isInputCommQuantScale); // 是否传入CommQuantScale
TILING_DATA_FIELD_DEF(uint32_t, dataType);
TILING_DATA_FIELD_DEF(uint32_t, commInt8WorkSpace); // int8 通信时用于存放reduceScatter输入 workspace 的开销
TILING_DATA_FIELD_DEF(uint32_t, dynamicQuantTempBuffSize); // fp8通信时用于存放BroadCast接口需要的临时空间开销
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RCSTilingOp, RCSTiling)

/**
 * @deprecated Mc2Msg 结构体在后续版本中将不再支持，请不要使用
 */

BEGIN_TILING_DATA_DEF(Mc2Msg) // 同aicpu_hccl_def KFCTilingData
TILING_DATA_FIELD_DEF(uint32_t,
                      preparePosition); // 任务准备位置：0表示在host完成所有通信任务的准备，1表示在kernel侧完成
TILING_DATA_FIELD_DEF(uint64_t, sendOff);      // 发送数据地址偏移，count * dataTypeSize
TILING_DATA_FIELD_DEF(uint64_t, recvOff);      // 接收数据地址偏移, count * dataTypeSize
TILING_DATA_FIELD_DEF(uint64_t, tailSendOff);  // 尾块发送数据地址偏移，count * dataTypeSize
TILING_DATA_FIELD_DEF(uint64_t, tailRecvOff);  // 尾块接收数据地址偏移, count * dataTypeSize
TILING_DATA_FIELD_DEF(uint64_t, sendCnt);      // 整块发送数据个数
TILING_DATA_FIELD_DEF(uint64_t, recvCnt);      // 尾块接收数据个数
TILING_DATA_FIELD_DEF(uint64_t, tailSendCnt);  // 尾块发送数据个数
TILING_DATA_FIELD_DEF(uint64_t, tailRecvCnt);  // 尾块接收数据个数
TILING_DATA_FIELD_DEF(uint64_t, totalCnt);     // 总数据个数
TILING_DATA_FIELD_DEF(uint32_t, turnNum);      // 总轮次
TILING_DATA_FIELD_DEF(uint32_t, tailNum);      // 尾块的轮次
TILING_DATA_FIELD_DEF(uint32_t, stride);       // 跳写间隔
TILING_DATA_FIELD_DEF(uint32_t, workspaceOff); // 使用workspace作为recvbuf时的workspace偏移
TILING_DATA_FIELD_DEF(uint32_t, notifyOff);    // device notify write/read value偏移

TILING_DATA_FIELD_DEF(uint16_t, notifyBeginCnt); // notify write value的使用个数
TILING_DATA_FIELD_DEF(uint16_t, notifyEndCnt);   // notify write value的使用个数
TILING_DATA_FIELD_DEF(uint8_t, useBufferType);   // recvBuf类型
TILING_DATA_FIELD_DEF(uint8_t, funID);           // function ID
TILING_DATA_FIELD_DEF(uint8_t, dataType);        // hccl 数据类型
TILING_DATA_FIELD_DEF(uint8_t, groupNum);        // groupNum

TILING_DATA_FIELD_DEF(uint8_t, reuseMode);  // 不复用填turnNum，内存优化选择复用的内存块个数
TILING_DATA_FIELD_DEF(uint8_t, commType);   // 通信类型
TILING_DATA_FIELD_DEF(uint8_t, reduceOp);   // reduce op type
TILING_DATA_FIELD_DEF(uint8_t, commOrder);  // 通信顺序，0表示通信在前，1表示通信在后
TILING_DATA_FIELD_DEF(uint8_t, waitPolicy); // 等待任务启动的阻塞策略，2、首轮等待，1、每轮等待。
                                            // KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
TILING_DATA_FIELD_DEF(uint8_t, rspPolicy); // 任务执行结束时的响应策略， 2、最后通知一次，
                                           // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
TILING_DATA_FIELD_DEF(uint8_t, exitPolicy); // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
                                            // 等待AIC通知退出(可以多次执行任务)。
TILING_DATA_FIELD_DEF(uint8_t, commAlg);      // 用于指定具体通信算法。
                                              // 0：default, 1：fullmesh, 2：doublering, 3：switchwing
TILING_DATA_FIELD_DEF(uint8_t, taskType);     // 从参数获取通信任务，直接下发。AIC自己发Record激活
TILING_DATA_FIELD_DEF(uint8_t, debugMode);    // 调测模式
                                              // 1:单独执行CUBE
                                              // 2:单独执行Vector
                                              // 4:单独执行AICPU KFC算子
                                              // 8:KFC等待通信结束
                                              // 16:KFC统计各阶段耗时
TILING_DATA_FIELD_DEF(uint8_t, stepSize);     // 用于指定通算频率步长
TILING_DATA_FIELD_DEF(uint8_t, sendArgIndex); // 发送数据参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF(uint8_t, recvArgIndex); // 接收数据参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF(uint8_t, commOutArgIndex); // 通信输出参数索引，对应算子原型的参数顺序
TILING_DATA_FIELD_DEF(uint8_t, hasCommOut);      // 是否有通信输出
TILING_DATA_FIELD_DEF(uint8_t, reserve);
TILING_DATA_FIELD_DEF(uint32_t, reserve2);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Mc2MsgOp, Mc2Msg)

BEGIN_TILING_DATA_DEF(TileL2Tiling)
TILING_DATA_FIELD_DEF(uint32_t, mL2TileCnt);
TILING_DATA_FIELD_DEF(uint32_t, nL2TileCnt);
TILING_DATA_FIELD_DEF(uint32_t, mTileBlocks);
TILING_DATA_FIELD_DEF(uint32_t, nTileBlocks);
TILING_DATA_FIELD_DEF(uint32_t, mTailBlocks);
TILING_DATA_FIELD_DEF(uint32_t, nTailBlocks);
TILING_DATA_FIELD_DEF(uint32_t, rankTileNum);
TILING_DATA_FIELD_DEF(uint32_t, calcOrder);
TILING_DATA_FIELD_DEF(uint32_t, enableL2Tile);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TileL2TilingOp, TileL2Tiling);

// mc2注释
//
//
//
//
//
//
//
// mc2注释

BEGIN_TILING_DATA_DEF(TileInfo)
TILING_DATA_FIELD_DEF(uint64_t, tileCnt);
TILING_DATA_FIELD_DEF(uint64_t, tileLen);
TILING_DATA_FIELD_DEF(uint64_t, tailCnt);
TILING_DATA_FIELD_DEF(uint64_t, tailLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TileInfoOp, TileInfo)

BEGIN_TILING_DATA_DEF(MC2MatmulV3TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
TILING_DATA_FIELD_DEF(uint32_t, mTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, nTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, kTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, mBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, nBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, mTailMain);
TILING_DATA_FIELD_DEF(uint32_t, nTailMain);
TILING_DATA_FIELD_DEF(uint32_t, isHf32);
TILING_DATA_FIELD_DEF(uint32_t, aswWindowLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Mc2MMV3TilingDataOp, MC2MatmulV3TilingData);

} // namespace optiling

#endif
