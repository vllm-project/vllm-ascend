/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef OP_API_INC_LEVEL0_CHUNK_KDA_FWD_PREPARE_H
#define OP_API_INC_LEVEL0_CHUNK_KDA_FWD_PREPARE_H

#include <array>
#include <cstdint>

#include "opdev/op_executor.h"

namespace l0op {

using ChunkKdaFwdPrepareOutputs = std::array<const aclTensor *, 13>;

ChunkKdaFwdPrepareOutputs ChunkKdaFwdPrepare(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool useExp2,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *qgScaledOut,
    const aclTensor *qHatOut,
    const aclTensor *kHatOut,
    const aclTensor *qRstdOut,
    const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    int64_t outputMode,
    aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_CHUNK_KDA_FWD_PREPARE_H
