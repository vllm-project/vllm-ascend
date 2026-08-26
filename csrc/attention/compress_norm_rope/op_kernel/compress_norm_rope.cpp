/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file compress_norm_rope.cpp
 * \brief Compressor 的压缩计算：GEMM（x @ wkv / x @ wgate）由外部 MatMulV3 完成，
 *        本算子只做 ape + softmax gate + 加权压缩 + state 递归 + rms_norm + rope。
 *        输入 mm_kv / mm_score 为 [tokenSize, coff*headDim] 的 bf16/fp16 GEMM 结果。
 *
 * A2/A3 两阶段实现：
 *   - C4（coff=2, cmpRatio=4）：组流式 + 双缓冲流水，UB 内融合 RmsNorm/RoPE/cast（无 SyncAll）
 *   - C128（coff=1, cmpRatio=128）：d 分块压缩 → GM workspace → SyncAll
 *     → 完整行高精度 RmsNorm/RoPE/cast
 * X_T/NORM_T/ROPE_T、coff 和空输入模式均由 ASCENDC_TPL template key 编译期分发。
 */

#include <type_traits>

#if (__CCE_AICORE__ == 220)
#include "arch32/compress_norm_rope_template_tiling_key.h"
#include "arch32/compress_norm_rope_kernel_c4.h"
#include "arch32/compress_norm_rope_kernel_c128.h"
#else
#error "compress_norm_rope currently only supports A2/A3"
#endif

using namespace CompressNormRope;

template <int X_T, int NORM_T, int ROPE_T, uint8_t Coff, uint8_t EmptyX>
__global__ __aicore__ void compress_norm_rope(
    __gm__ uint8_t *mmKv,
    __gm__ uint8_t *mmScore,
    __gm__ uint8_t *stateCache,
    __gm__ uint8_t *ape,
    __gm__ uint8_t *normWeight,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *stateBlockTable,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *seqUsed,
    __gm__ uint8_t *startPos,
    __gm__ uint8_t *cmpKvOut,
    __gm__ uint8_t *stateCacheOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
    REGISTER_TILING_DEFAULT(optiling::CompressNormRopeTilingData);
    if constexpr (EmptyX != 0) {
        return;
    }
    // C4 无 SyncAll，可使用 AIV_ONLY；C128 两阶段包含 SyncAll。
    // SyncAll 需 MIX_AIV_1_0（全核同调度）。本 CANN 下 KERNEL_TASK_TYPE(key,..) 无法按 ASCENDC_TPL
    // 编码 key 区分，统一用 MIX_AIV_1_0 默认（与 v2 旧代码一致，AIV 执行，taskRation 0:1）。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(optiling::CompressNormRopeTilingData, tilingDataIn, tiling);
    const optiling::CompressNormRopeTilingData *__restrict tilingData = &tilingDataIn;
    TPipe pipe;
    using XType = std::conditional_t<X_T == CNR_TPL_BF16, bfloat16_t, half>;
    using NormType = std::conditional_t<NORM_T == CNR_TPL_BF16, bfloat16_t, half>;
    using RopeType = std::conditional_t<ROPE_T == CNR_TPL_FP32, float,
                                        std::conditional_t<ROPE_T == CNR_TPL_BF16, bfloat16_t, half>>;
    if constexpr (Coff == 2) {
        CompressNormRopeKernelC4<XType, NormType, RopeType> op;
        op.Init(&pipe, tilingData, mmKv, mmScore, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,
                cuSeqlens, seqUsed, startPos, cmpKvOut);
        op.Process();
    } else {
        __gm__ uint8_t *userWs = GetUserWorkspace(workspace);
        CompressNormRopeKernelC128<XType, NormType, RopeType> op;
        op.Init(&pipe, tilingData, mmKv, mmScore, stateCache, ape, normWeight, ropeSin, ropeCos, stateBlockTable,
                cuSeqlens, seqUsed, startPos, cmpKvOut, userWs);
        op.Process();
    }
}
