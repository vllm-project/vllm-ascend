/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Host-direct Dsv41MlaProlog entry points (implemented in the ASC-compiled
 * csrc/kernels/dsv41_mla_prolog_direct.cpp and linked via vllm_ascend_kernels).
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
    const int32_t *slots;
    uint16_t *kvCache;
    uint64_t cacheStride0;
    uint64_t cacheStride1;
    uint32_t cacheBlockSize;
    uint16_t *qOutput;
    uint16_t *qr;
    void *workspace;
    uint64_t workspaceSize;
    int32_t t;
    float epsilon;
};

namespace vllm_ascend {

uint64_t dsv41_mla_prolog_workspace_size(int64_t t);

// Returns 0 on success, otherwise the aclError code.
int dsv41_mla_prolog_direct(
    const Dsv41MlaPrologDirectArgs &args, void *stream);

}  // namespace vllm_ascend

#endif  // DSV41_MLA_PROLOG_DIRECT_API_H_
