// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host-direct Dsv41MlaProlog entry points (implemented in the ASC-compiled
 * direct shared library and linked into the PyTorch extension).
 */

#ifndef DSV41_MLA_PROLOG_DIRECT_API_H_
#define DSV41_MLA_PROLOG_DIRECT_API_H_

#include <cstdint>

struct Dsv41MlaPrologDirectArgs {
    const uint16_t *x;
    const uint16_t *wQa;
    const uint16_t *gammaQa;
    const int8_t *wQb;
    const uint16_t *wQbScale;
    const uint16_t *wKv;
    const uint16_t *gammaKv;
    const float *ropeCos;
    const float *ropeSin;
    const int64_t *slots;
    uint16_t *kvCache;
    uint16_t *kvOutput; // Dense [T, 512] KV output in place of paged cache writes.
    uint64_t cacheStride0;
    uint64_t cacheStride1;
    uint64_t cachePages;
    uint32_t numHeads;
    uint16_t *qOutput;
    uint16_t *qr;
    void *workspace;
    uint64_t workspaceSize;
    int32_t t;
    float epsilon;
};

namespace vllm_ascend {

uint64_t dsv41_mla_prolog_workspace_size(int64_t t, uint32_t numHeads);

// Returns 0 on success, otherwise the aclError code.
int dsv41_mla_prolog_direct(
    const Dsv41MlaPrologDirectArgs &args, void *stream);

}  // namespace vllm_ascend

#endif  // DSV41_MLA_PROLOG_DIRECT_API_H_
