/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "categorical_sample.h"

#if !defined(ENABLE_ASSERT) || !defined(ENABLE_ASSERT_DUMP_SIZE)
// CANN still detects these precompile feature markers and uses them to reserve
// and initialize the assert dump workspace, but some A2, A3, and A5 AscendC
// headers do not expose the legacy ENABLE_ASSERT* helper macros.
// Keep the marker names identical to the compiler's kernel-info inference
// contract so AssertPrint remains available in both eager and graph launches.
#if defined(__CHECK_FEATURE_AT_PRECOMPILE)
#define CATEGORICAL_SAMPLE_ENABLE_ASSERT()                   \
    auto __enable_feature_for_compile_assert = 1;           \
    auto __enable_feature_for_compile_assertBufSize = 1024
#else
#define CATEGORICAL_SAMPLE_ENABLE_ASSERT()
#endif
#else
#define CATEGORICAL_SAMPLE_ENABLE_ASSERT() \
    ENABLE_ASSERT();                       \
    ENABLE_ASSERT_DUMP_SIZE()
#endif

extern "C" __global__ __aicore__ void categorical_sample(
    GM_ADDR processedLogits,
    GM_ADDR expandedIdxMapping,
    GM_ADDR temperature,
    GM_ADDR seed,
    GM_ADDR pos,
    GM_ADDR outputProcessedLogits,
    GM_ADDR outputProcessedLogitsCol,
    GM_ADDR sampledTokenIds,
    GM_ADDR lse,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CategoricalSample::CategoricalSampleTilingData);
    CATEGORICAL_SAMPLE_ENABLE_ASSERT();
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;

    if (TILING_KEY_IS(1)) {
        CategoricalSample::CategoricalSampleKernel<float> op;
        op.Init(processedLogits, expandedIdxMapping, temperature, seed, pos, outputProcessedLogits,
                outputProcessedLogitsCol, sampledTokenIds, lse, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        CategoricalSample::CategoricalSampleKernel<half> op;
        op.Init(processedLogits, expandedIdxMapping, temperature, seed, pos, outputProcessedLogits,
                outputProcessedLogitsCol, sampledTokenIds, lse, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        CategoricalSample::CategoricalSampleKernel<bfloat16_t> op;
        op.Init(processedLogits, expandedIdxMapping, temperature, seed, pos, outputProcessedLogits,
                outputProcessedLogitsCol, sampledTokenIds, lse, &tilingData, &pipe);
        op.Process();
    }
}
