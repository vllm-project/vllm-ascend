/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef CHUNK_KDA_FWD_FINALIZE_KERNEL_H
#define CHUNK_KDA_FWD_FINALIZE_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_kda_fwd_finalize_cube.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/chunk_kda_fwd_finalize_vec.h"
#endif

namespace KdaFinalize {

template <bool UseAivInputMover, bool StateVFirst, bool OutputSequenceMajor>
__aicore__ inline void RunFinalize(const FinalizeArgs &args)
{
    if ASCEND_IS_AIC {
        FinalizeCube<StateVFirst, OutputSequenceMajor, UseAivInputMover> cube;
        cube.Init(args);
        cube.Process();
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    if constexpr (UseAivInputMover) {
        if ASCEND_IS_AIV {
            Arch35::FinalizeInputMover<StateVFirst> inputMover;
            inputMover.Init(args);
            inputMover.Process();
        }
    }
#endif
}

template <bool UseAivInputMover>
__aicore__ inline void DispatchFinalize(const FinalizeArgs &args)
{
    if (args.tiling.stateVFirst) {
        if (args.tiling.outputSequenceMajor) {
            RunFinalize<UseAivInputMover, true, true>(args);
        } else {
            RunFinalize<UseAivInputMover, true, false>(args);
        }
    } else if (args.tiling.outputSequenceMajor) {
        RunFinalize<UseAivInputMover, false, true>(args);
    } else {
        RunFinalize<UseAivInputMover, false, false>(args);
    }
}

} // namespace KdaFinalize

#endif // CHUNK_KDA_FWD_FINALIZE_KERNEL_H
