/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "compressor_metadata.h"

extern "C" __global__ __aicore__ void compressor_metadata(
    GM_ADDR ropeCos,
    GM_ADDR ropeSin,
    GM_ADDR cuSeqlens,
    GM_ADDR startPos,
    GM_ADDR kvBlockTable,
    GM_ADDR compressCos,
    GM_ADDR compressSin,
    GM_ADDR slotMapping,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CompressorMetadata::CompressorMetadataTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;

    if (TILING_KEY_IS(1)) {
        CompressorMetadata::CompressorMetadataKernel<float> op;
        op.Init(&tilingData, &pipe);
        op.Process(ropeCos, ropeSin, cuSeqlens, startPos, kvBlockTable, compressCos, compressSin, slotMapping, workspace);
    } else if (TILING_KEY_IS(2)) {
        CompressorMetadata::CompressorMetadataKernel<half> op;
        op.Init(&tilingData, &pipe);
        op.Process(ropeCos, ropeSin, cuSeqlens, startPos, kvBlockTable, compressCos, compressSin, slotMapping, workspace);
    } else if (TILING_KEY_IS(3)) {
        CompressorMetadata::CompressorMetadataKernel<bfloat16_t> op;
        op.Init(&tilingData, &pipe);
        op.Process(ropeCos, ropeSin, cuSeqlens, startPos, kvBlockTable, compressCos, compressSin, slotMapping, workspace);
    }
}
