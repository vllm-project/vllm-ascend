/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GATED_ACTIVATION_HPP
#define MEGA_MOE_GATED_ACTIVATION_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "../template_linear_algebra_v2/catlass.hpp"

namespace Catlass::Epilogue {

enum class GatedActivationMode : uint32_t {
    SWIGLU = 0,
    SITU = 2,
    SWIGLU_STEP = 3,
    SWIGLU_OAI = 4,
};

// All gated activations used by MegaMoe follow this interface. A new
// activation only needs to implement Compute; data movement, tiling and
// quantization remain shared by the epilogues.
struct SwigluActivation {
    // 3072 is the largest common tile that leaves at least 32 KiB of
    // temporary UB for every BF16, INT8 and W4A8 double-buffered path.
    static constexpr uint32_t PREFERRED_TILE_LENGTH = 3072;
    static constexpr uint32_t MIN_SHARED_TMP_BYTES = 32 * 1024;

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float clampLimit)
    {
        if (clampLimit > 0.0f) {
            // Preserve MegaMoe's existing clamp semantics: the gate branch is
            // upper-bounded, while the up branch is symmetrically clamped.
            AscendC::ClampMax(gate, gate, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMax(up, up, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMin(up, up, sharedTmpBuffer, -clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // SiLU(gate) * up = gate / (1 + exp(-gate)) * up.
        AscendC::Muls(output, gate, -1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(output, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(output, output, 1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(output, gate, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(output, output, up, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

// STEP 3.5 SwiGLU keeps the up-branch clamp used by MegaMoe, but applies
// the gate upper bound to SiLU(gate), rather than to the raw gate input:
//     min(SiLU(gate), clampLimit) * clip(up, -clampLimit, clampLimit).
struct SwigluStepActivation {
    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float clampLimit)
    {
        if (clampLimit > 0.0f) {
            AscendC::ClampMax(up, up, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMin(up, up, sharedTmpBuffer, -clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Muls(output, gate, -1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(output, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(output, output, 1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(output, gate, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        if (clampLimit > 0.0f) {
            AscendC::ClampMax(output, output, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(output, output, up, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

// MiniMax M3 SwiGLU-OAI:
//     (up + beta) * gate * sigmoid(alpha * gate).
struct SwigluOaiActivation {
    static constexpr float DEFAULT_ALPHA = 1.702f;
    static constexpr float DEFAULT_BETA = 1.0f;

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float clampLimit, float alpha = DEFAULT_ALPHA, float beta = DEFAULT_BETA)
    {
        if (clampLimit > 0.0f) {
            AscendC::ClampMax(gate, gate, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMax(up, up, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMin(up, up, sharedTmpBuffer, -clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Adds(up, up, beta, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(output, gate, -alpha, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(output, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(output, output, 1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(output, gate, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(output, output, up, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

// Kimi K3 SiTUGLU:
//     beta * tanh(gate / beta) * sigmoid(gate) * up
// When linearBeta is enabled, up becomes:
//     linearBeta * tanh(up / linearBeta)
// gate/up use the regular contiguous half split.
struct SituActivation {
    static constexpr float DEFAULT_BETA = 1.0f;
    // Zero is not a valid scale because the formula divides by linearBeta,
    // so it is also the unambiguous internal representation of None.
    static constexpr float LINEAR_BETA_DISABLED = 0.0f;

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float beta = DEFAULT_BETA, float linearBeta = LINEAR_BETA_DISABLED)
    {
        bool enableLinearBeta = linearBeta != LINEAR_BETA_DISABLED;
        if (enableLinearBeta) {
            // output = linearBeta * tanh(up / linearBeta). The original up
            // tile can then be reused for the sigmoid denominator.
            AscendC::Muls(up, up, 1.0f / linearBeta, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Tanh(output, up, sharedTmpBuffer, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(output, output, linearBeta, dataLength);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(up, gate, -1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(up, up, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(up, up, 1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(output, output, up, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // output = sigmoid(gate) * up.
            AscendC::Muls(output, gate, -1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(output, output, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(output, output, 1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(output, up, output, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Tanh requires non-overlapping dst/src/tmp tensors. Reusing up as
        // dst satisfies that constraint without allocating another UB tile.
        if (beta != DEFAULT_BETA) {
            AscendC::Muls(gate, gate, 1.0f / beta, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Tanh(up, gate, sharedTmpBuffer, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        if (beta != DEFAULT_BETA) {
            AscendC::Muls(up, up, beta, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(output, output, up, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

// Runtime dispatcher shared by all A2/A3 epilogues. New gated activations can
// be added here without duplicating data movement, tiling or quantization.
struct GatedActivation {
    static constexpr uint32_t PREFERRED_TILE_LENGTH = SwigluActivation::PREFERRED_TILE_LENGTH;
    static constexpr uint32_t MIN_SHARED_TMP_BYTES = SwigluActivation::MIN_SHARED_TMP_BYTES;

    template <class Element>
    CATLASS_DEVICE static void PrepareBranches(AscendC::LocalTensor<float> const &gate,
                                               AscendC::LocalTensor<float> const &up,
                                               AscendC::LocalTensor<Element> const &packedInput,
                                               uint32_t branchStride, uint32_t dataLength)
    {
        AscendC::Cast(gate, packedInput, AscendC::RoundMode::CAST_NONE, dataLength);
        AscendC::Cast(up, packedInput[branchStride], AscendC::RoundMode::CAST_NONE, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float clampLimit, uint32_t activationCode,
                        float activationParams1 = SwigluOaiActivation::DEFAULT_ALPHA,
                        float activationParams2 = SituActivation::DEFAULT_BETA)
    {
        switch (static_cast<GatedActivationMode>(activationCode)) {
            case GatedActivationMode::SWIGLU:
                SwigluActivation::Compute(output, gate, up, sharedTmpBuffer, dataLength, clampLimit);
                break;
            case GatedActivationMode::SWIGLU_OAI:
                SwigluOaiActivation::Compute(output, gate, up, sharedTmpBuffer, dataLength, clampLimit,
                                             activationParams1, activationParams2);
                break;
            case GatedActivationMode::SITU:
                SituActivation::Compute(output, gate, up, sharedTmpBuffer, dataLength, activationParams1,
                                        activationParams2);
                break;
            case GatedActivationMode::SWIGLU_STEP:
                SwigluStepActivation::Compute(output, gate, up, sharedTmpBuffer, dataLength, clampLimit);
                break;
            default:
                break;
        }
    }
};

} // namespace Catlass::Epilogue

#endif // MEGA_MOE_GATED_ACTIVATION_HPP
