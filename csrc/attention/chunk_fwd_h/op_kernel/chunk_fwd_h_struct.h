/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_h_struct.h
 * \brief Host-side plain tiling data used by the framework-independent size calculator.
 *
 * Kernel code uses the framework-generated ChunkFwdHTilingData. Keeping
 * a differently named host structure avoids shadowing that generated type.
 */

#ifndef CHUNK_FWD_H_STRUCT_H
#define CHUNK_FWD_H_STRUCT_H

#include <cstdint>

struct ChunkFwdHPlainTilingData {
    int64_t batch;
    int64_t seqlen;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t kHeadDim;
    int64_t vHeadDim;
    int64_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    int64_t isVariedLen;
    int64_t shapeBatch;
    int64_t tokenBatch;
    int64_t vWorkspaceOffset;
    int64_t vUpdateWorkspaceOffset;
    int64_t kDecayWorkspaceOffset;
    int64_t hWorkspaceOffset;
};

#endif // CHUNK_FWD_H_STRUCT_H
