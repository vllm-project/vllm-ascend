/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_STRUCT_H
#define CHUNK_KDA_FWD_PREPARE_STRUCT_H

#include "kernel_operator.h"

namespace KdaPrepare {

// 与 host 侧 TILING_DATA_FIELD_DEF 的字段顺序和类型严格一致。
// 该结构只描述 GM 中的序列化布局，运行时计算使用下方的精简副本。
struct ChunkKdaFwdPrepareTilingData {
    uint32_t batch;
    uint32_t seqNum;
    uint32_t seqLen;
    uint32_t qkHeadNum;
    uint32_t valueHeadNum;
    uint32_t totalChunks;
    uint32_t usedCoreNum;
    uint32_t headsPerPartition;
    float epsilon;
    float lowerBound;
    float scale;
    bool isVarLen;
    bool inputSequenceMajor;
    bool hasDtBias;
};

// 入口从框架生成的 tiling 数据中一次性复制这些字段，Stage 不再读取 GM tiling。
struct PrepareRuntimeTiling {
    uint32_t batch = 0;
    uint32_t seqNum = 0;
    uint32_t seqLen = 0;
    uint32_t qkHeadNum = 0;
    uint32_t valueHeadNum = 0;
    uint32_t totalChunks = 0;
    uint32_t usedCoreNum = 0;
    uint32_t headsPerPartition = 0;
    float epsilon = 1.0e-6F;
    float lowerBound = -5.0F;
    float scale = 1.0F;
    bool isVarLen = false;
    bool inputSequenceMajor = false;
    bool hasDtBias = false;
};

// 一个实际 chunk 的直接索引。Stage 只接收这些标量，不再传递通用 StageArgs。
struct ChunkRange {
    // dense: batchIndex=sequence，tokenBegin 为序列内位置；
    // varlen: batchIndex=0，tokenBegin 为压平后的全局 token 位置。
    uint32_t batchIndex = 0;
    uint32_t sequence = 0;
    uint32_t globalChunk = 0;
    uint32_t tokenBegin = 0;
    uint32_t validRows = 0;
};

// 仅保存真实 GM 参数和运行时 tiling；本结构不承担 buffer、公式或同步记录功能。
struct PrepareKernelArgs {
    GM_ADDR q = nullptr;
    GM_ADDR k = nullptr;
    GM_ADDR v = nullptr;
    GM_ADDR rawGate = nullptr;
    GM_ADDR beta = nullptr;
    GM_ADDR aLog = nullptr;
    GM_ADDR dtBias = nullptr;
    GM_ADDR cuSeqlens = nullptr;
    GM_ADDR chunkIndices = nullptr;

    GM_ADDR gk = nullptr;
    GM_ADDR aqk = nullptr;
    GM_ADDR akk = nullptr;
    GM_ADDR w = nullptr;
    GM_ADDR u = nullptr;
    GM_ADDR qg = nullptr;
    GM_ADDR kg = nullptr;
    GM_ADDR qgScaled = nullptr;
    GM_ADDR qHat = nullptr;
    GM_ADDR kHat = nullptr;
    GM_ADDR qRstd = nullptr;
    GM_ADDR kRstd = nullptr;
    GM_ADDR betaEff = nullptr;
    GM_ADDR workspace = nullptr;

    PrepareRuntimeTiling tiling{};
};

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_STRUCT_H
