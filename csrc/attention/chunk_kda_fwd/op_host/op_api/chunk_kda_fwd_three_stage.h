/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OP_API_INC_LEVEL0_CHUNK_KDA_FWD_THREE_STAGE_H
#define OP_API_INC_LEVEL0_CHUNK_KDA_FWD_THREE_STAGE_H

#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

// aclnnChunkKdaFwd 的内部组合参数。所有张量 shape/dtype/连续性契约由 L2
// 入口（aclnn_chunk_kda_fwd.cpp）统一校验，这里只承载编排所需的信息。
struct KdaFwdThreeStageArgs {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aLog = nullptr;
    const aclTensor *dtBias = nullptr;
    const aclTensor *initialState = nullptr;
    const aclIntArray *cuSeqlens = nullptr;
    const aclIntArray *chunkIndices = nullptr;
    // layout 描述 q/k/v/g/beta 输入；attnLayout 是公开 attnOut 的落盘布局，
    // 本算子固定为 sequence-major（rank-4 用 BSND，rank-3 用 TND）。
    const char *layout = "BSND";
    const char *attnLayout = "BSND";
    double scale = 1.0;
    double epsilon = 1.0e-6;
    double lowerBound = -5.0;
    int64_t chunkSize = 64;
    bool safeGate = false;
    bool useGateInKernel = false;
    bool stateVFirst = false;
    // 由 L2 传入的 gate/L2 norm 开关；默认值与融合实现语义一致。
    bool useQkL2normInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool useExp2 = true;
    // 归一化后的 head-major 维度信息。
    int64_t batch = 0;
    int64_t qkHeads = 0;
    int64_t valueHeads = 0;
    int64_t seqLen = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    int64_t seqNum = 0;
    int64_t totalChunks = 0;
    // NTD/TND：Prepare/Finalize 使用 rank-3 head-major 张量。
    bool packed = false;
    const aclTensor *attnOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
    const aclTensor *gkOut = nullptr;
    const aclTensor *aqkOut = nullptr;
    const aclTensor *akkOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *hOut = nullptr;
};

// 由公开输出指针组合推导 ChunkKdaFwdPrepare 的编译期 outputMode。
// Aqk/Akk 是公开必选输出，但 qHat/kHat/qRstd/kRstd/betaEff 只服务反向重计算，
// 因此 Akk 存在时使用只多搬 Akk 的 forward 档；只有同时需要 w/u/qg/kg/v_new
// 时才升级到 save 档。
int64_t KdaFwdThreeStageOutputMode(const KdaFwdThreeStageArgs &args);

// 依次提交 ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize，
// 并在需要时把内部 head-major/chunk-major 结果回写到公开输出。
aclnnStatus KdaFwdThreeStage(const KdaFwdThreeStageArgs &args, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_CHUNK_KDA_FWD_THREE_STAGE_H
