/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef OP_API_INC_ACLNN_CHUNK_KDA_FWD_V2_H
#define OP_API_INC_ACLNN_CHUNK_KDA_FWD_V2_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ChunkKdaFwd V2 入口：在同一个 executor 内按
 *   ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize
 * 组合三个已交付算子完成前向。
 *
 * 与 aclnnChunkKdaFwdGetWorkspaceSize 的关系：
 *   - aclnnChunkKdaFwdGetWorkspaceSize 的签名与 ABI 保持不变，仍使用私有 L0
 *     融合实现；
 *   - 本接口接受额外的归一化 / gate 开关，并新增 epsilon 入参；
 *   - 场景选择由上层（fla_npu.ops.ascendc.chunk_kda_fwd）完成：满足组合场景时
 *     优先调用本接口，否则回落到 aclnnChunkKdaFwd。
 *
 * 支持范围：q/k/v 为 BF16、K=V=128、chunk_size=64、公开输出连续、cu_seqlens
 * 严格递增。不满足时返回 ACLNN_ERR_PARAM_INVALID。
 *
 * 开关语义：
 *   epsilon                = 1e-6 默认；仅 useQkL2normInKernel=true 时参与 rsqrt
 *   useQkL2normInKernel    = false：q/k 由调用方预先归一化
 *   useBetaSigmoidInKernel = false：beta 由调用方预先 sigmoid
 *   allowNegEigval         = false；true 时必须同时 useBetaSigmoidInKernel=true
 *   useExp2                = true：门控走 exp2
 *
 * 输出必选性：attnOut/aqkOut 必传；akkOut 与 op def 一致为可选，传 nullptr 时
 * 组合入口使用 Prepare 的 none 档（不搬出 Akk）。
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaFwdV2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    double epsilon,
    bool useQkL2normInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool useExp2,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    const aclTensor *qHatOut,
    const aclTensor *kHatOut,
    const aclTensor *qRstdOut,
    const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnChunkKdaFwdV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CHUNK_KDA_FWD_V2_H
