/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"
#include "hccl/hccl.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void win_probe(GM_ADDR context, GM_ADDR addrs, GM_ADDR workspace, GM_ADDR tiling)
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    // `context` is the CANN CommContext struct:
    //   uint32 epRankId; uint32 rankSizePerServer; uint64 kfcContextAddr;
    //   uint64 epHcclBuffer_[rank]; ...
    // epHcclBuffer_[rank] holds the local (rank == self) / remote IPC buffer
    // addresses of the registered HCCL CCL buffers.
    __gm__ uint64_t* ctx = reinterpret_cast<__gm__ uint64_t*>(context);

    AscendC::GlobalTensor<int64_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(addrs));
    constexpr uint32_t kWorld = 2;
    for (uint32_t i = 0; i < kWorld; ++i) {
        out.SetValue(i, static_cast<int64_t>(ctx[2 + i]));
    }
    AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                     AscendC::DcciDst::CACHELINE_OUT>(out);
}
