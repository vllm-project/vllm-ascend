/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_PREPARE_KERNEL_H
#define CHUNK_KDA_FWD_PREPARE_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_prepare_policy.h"
#include "chunk_kda_fwd_prepare_struct.h"

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_prepare_cube.h"
#include "arch35/chunk_kda_fwd_prepare_vec.h"
#else
#include "arch22/chunk_kda_fwd_prepare_cube.h"
#include "arch22/chunk_kda_fwd_prepare_vec.h"
#endif

namespace KdaPrepare {

template <int DTYPE>
struct PrepareStorageType;

template <>
struct PrepareStorageType<CHUNK_KDA_FWD_PREPARE_TPL_BF16> {
    using type = bfloat16_t;
};

template <>
struct PrepareStorageType<CHUNK_KDA_FWD_PREPARE_TPL_FP32> {
    using type = float;
};

template <typename GateT, typename BetaT, typename CompilePolicy>
__aicore__ inline void RunPrepare(const PrepareKernelArgs &args)
{
    if ASCEND_IS_AIV {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch35::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
        vec.Init(args);
#else
        AscendC::TPipe pipe;
        Arch22::ChunkKdaFwdPrepareVec<GateT, BetaT, CompilePolicy> vec;
        vec.Init(args, &pipe);
#endif
        vec.Process();
    }

    if ASCEND_IS_AIC {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        Arch35::ChunkKdaFwdPrepareCube<CompilePolicy> cube;
        cube.Init(args);
#else
        AscendC::TPipe pipe;
        Arch22::ChunkKdaFwdPrepareCube<CompilePolicy> cube;
        cube.Init(args, &pipe);
#endif
        cube.Process();
    }
}

} // namespace KdaPrepare

#endif // CHUNK_KDA_FWD_PREPARE_KERNEL_H
