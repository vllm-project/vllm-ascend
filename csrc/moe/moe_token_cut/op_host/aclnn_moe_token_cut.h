/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef MOE_TOKEN_CUT_TILING_H
#define MOE_TOKEN_CUT_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

/**
 * 每个 AIV 共用的静态 tiling 参数。
 *
 * shape、通信拓扑和内存布局相关的值必须在 host 侧确定；会逐轮变化的
 * local_cut_count、batch_size、enable_cut 作为 Tensor 运行时读取。
 */
struct MoeTokenCutInfo {
    uint32_t batchSize;            // 实际输入行数，范围 [1, 64]。
    uint32_t topK;                 // 每个 token 路由到的专家数。
    uint32_t rankId;               // 当前 EP 通信组内的 rank。
    uint32_t layerIdx;             // 当前 MoE 层编号；通信 magic 为 layerIdx + 1。
    uint32_t rankNum;              // EP 通信组的 rank 数。
    uint32_t expertNum;            // 全局物理专家总数。
    uint32_t expertPerRank;        // 每个 rank 连续持有的专家数。
    uint32_t levelLimit;           // 外部指定的目标档位上限。
    uint32_t cutCountThreshold;    // 累计截断数达到该值后停止截断。
    uint32_t smallExpertTokenThreshold; // 可删除专家的最大 token 数。
    uint32_t aivNum;               // 本 kernel 实际启动的 AIV 数。
    uint32_t sendAivNum;           // 前若干 AIV 专门负责远端写。
    uint32_t recvAivNum;           // 随后的 AIV 专门负责轮询接收 magic。
    uint32_t hasTopkWeights;       // 兼容字段；必选 topk_weights 下恒为 1。
    uint32_t weightTypeSize;       // 权重 dtype 字节数，仅用于校验和调试。
    uint32_t hiddenDim;            // hidden / residual 的最后一维，restore 拷贝用。
    uint64_t windowBytes;          // 每张卡 HCCL input window 的可用字节数。
    uint64_t userWorkspaceBytes;   // 扣除框架保留前缀后的 workspace 字节数。
};

/**
 * 两份 MC2 tiling 用于初始化直接访问 HCCL window 的通信上下文，
 * info 保存本算子的固定布局参数。
 */
struct MoeTokenCutTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeTokenCutInfo info;
};

#endif  // MOE_TOKEN_CUT_TILING_H
