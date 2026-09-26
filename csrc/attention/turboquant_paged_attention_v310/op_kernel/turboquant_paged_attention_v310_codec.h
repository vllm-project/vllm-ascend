/* SYNCED COPY -- edit csrc/attention/turboquant_common/turboquant_codec.h instead.
 * The op build copies each op_kernel/ into an isolated tree, so a cross-op
 * relative include cannot resolve. The build also copies ONLY files matching the
 * op-name prefix, so this copy must be named <op_name>_codec.h to be picked up.
 */
/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * TurboQuant (arXiv 2504.19874) device-side codec primitives for Ascend 310P.
 *
 * Shared by turboquant_reshape_and_cache_v310 (write) and
 * turboquant_paged_attention_v310 (read).
 *
 * PARAMETERISATION
 *   BITS        compile-time template arg (2/3/4). Must be compile-time: the
 *               pack/unpack shift amounts have to be constants for the loops to
 *               unroll, and LEVELS sizes the codebook UB buffer.
 *               -> selected by tiling key (200 + BITS).
 *   variant     RUNTIME field (TQ_VARIANT_MSE | TQ_VARIANT_PROD). QJL adds a
 *               separate accumulation term; branching once per vector is free.
 *   codebook    RUNTIME field (TQ_CB_UNIFORM | TQ_CB_LUT). Uniform dequant is an
 *               affine multiply, LUT is a gather; the branch hoists out of the
 *               inner loop.
 *   Rationale: templating all three would give 12 kernel instantiations for 12
 *   A/B scenarios. This gives 3, with no measurable inner-loop cost.
 *
 * KEY ALGEBRAIC PROPERTY (why Pi never touches the cache)
 *   Pi = D*H*D is symmetric AND self-inverse, so
 *       <q, n*Pi@yhat>          == n * <Pi@q, yhat>
 *       sum_i p_i*n_i*Pi@yhat_i == Pi @ (sum_i p_i*n_i*yhat_i)
 *   Attention therefore runs entirely in the ROTATED basis: rotate the query
 *   once and the output once, O(1) per request instead of O(context).
 *   Verified on device at b=2/3/4: rotated == original, cosine 1.000000.
 *
 * 310P CONSTRAINTS OBSERVED HERE
 *   - no DataCopyPad  -> callers use the safe-data-copy-pad block pattern
 *   - no Sort32 / Fixpipe
 *   - packed plane is stored in an fp16-typed FRACTAL_NZ cache (an int8-typed
 *     cache silently writes nothing on this SoC), so pack() emits bytes that the
 *     caller reinterprets as half.
 *
 * TESTING NOTE -- do not use structured signals as test input.
 *   The codebook covers +-3.5 sigma of a Gaussian. TurboQuant's distortion
 *   guarantee assumes the rotation makes coordinates near-Gaussian, which holds
 *   for real activations but NOT for inputs the Hadamard transform happens to
 *   sparsify. A pure sinusoid x[i] = sin(0.11*i) rotates to a 5.5-sigma peak
 *   with 13/256 coords clipping, giving relL2 0.4665 instead of 0.1926 -- which
 *   looks exactly like a broken quantizer. Random input reproduces the expected
 *   0.197. Verified in CPU simulation on Ascend310P3.
 */
#ifndef TURBOQUANT_PAGED_ATTENTION_V310_CODEC_H
#define TURBOQUANT_PAGED_ATTENTION_V310_CODEC_H

#include "kernel_operator.h"

namespace TurboQuant {

using namespace AscendC;

// runtime selectors (tiling data fields)
constexpr uint32_t TQ_VARIANT_MSE = 0;   // paper Alg 1
constexpr uint32_t TQ_VARIANT_PROD = 1;  // paper Alg 2 (MSE at b-1 + 1-bit QJL residual)
constexpr uint32_t TQ_CB_UNIFORM = 0;    // affine: scale * (idx - offset)
constexpr uint32_t TQ_CB_LUT = 1;        // Lloyd-Max gather

// MSE-optimal uniform step for a unit-variance Gaussian, by bit-width.
// After rotation each coordinate is ~N(0, 1/d), so the reconstruction level is
// (step * (idx - (LEVELS-1)/2)) / sqrt(d). Measured cost vs Lloyd-Max at b=3:
// +0.01 equivalent bits -- effectively free, and it turns dequant into a multiply.
__aicore__ inline float UniformStep(int bits)
{
    // 2 -> 0.995686, 3 -> 0.586020, 4 -> 0.335201
    return (bits == 2) ? 0.995686f : ((bits == 3) ? 0.586020f : 0.335201f);
}

template <int BITS>
struct TQTraits {
    static constexpr int kBits = BITS;
    static constexpr int kLevels = 1 << BITS;
    static constexpr int kCodesPerWord = 8;              // 8 codes pack into BITS bytes exactly
    static constexpr int kBytesPerWord = BITS;
    static constexpr uint32_t kMask = (1u << BITS) - 1u;

    // packed bytes for a head_dim vector; must be even (stored via fp16 view)
    __aicore__ static constexpr int PackedBytes(int headDim) { return headDim * BITS / 8; }
    __aicore__ static constexpr int PackedHalves(int headDim) { return headDim * BITS / 16; }
};

/*
 * Fast Walsh-Hadamard transform, in place, on a UB tensor of length `len`
 * (power of two). `len` stages of pairwise add/sub, then a 1/sqrt(len) scale.
 *
 * Pi = D*H*D is applied as: sign-flip (D), FWHT (H), sign-flip (D). Because D is
 * a diagonal of +-1 and H is symmetric, the composite is symmetric and squares
 * to the identity -- one matrix serves forward and inverse, so the read path
 * uses the SAME routine to rotate the query and the output.
 *
 * O(len log len) with no matmul, so this stays on the vector pipe and leaves
 * Cube free for the attention GEMMs.
 */
__aicore__ inline void FwhtInPlace(const LocalTensor<float> &x, const LocalTensor<float> &tmp, int len)
{
    /*
     * Scalar butterfly.
     *
     * The obvious vectorisation -- Add(x[base], x[base], x[base+stride], stride)
     * per 2*stride block -- is WRONG on this hardware for the early stages:
     * AscendC vector instructions operate on 32-byte blocks (8 floats), so a
     * count of 1/2/4 is not a valid vector op. In CPU simulation that produced
     * an all-zero result (||Pi x|| = 0), and on device it would have been
     * silent misbehaviour rather than a clean failure.
     *
     * Correctness first (kBlockFloats-aligned stages can be vectorised later:
     * stages with stride >= 8 operate on contiguous runs and map to Add/Sub
     * directly, which covers 5 of the 8 stages at len=256).
     */
    for (int stride = 1; stride < len; stride <<= 1) {
        for (int base = 0; base < len; base += (stride << 1)) {
            for (int i = 0; i < stride; ++i) {
                const int lo = base + i;
                const int hi = lo + stride;
                const float a = x.GetValue(lo);
                const float b = x.GetValue(hi);
                x.SetValue(lo, a + b);
                x.SetValue(hi, a - b);
            }
        }
    }
    (void)tmp;  // scratch no longer needed; kept for API stability
}

// Apply the random sign vector D (stored as +-1.0f in UB).
__aicore__ inline void ApplySigns(const LocalTensor<float> &x, const LocalTensor<float> &signs, int len)
{
    Mul(x, x, signs, len);
    PipeBarrier<PIPE_V>();
}

/*
 * Full rotation y = Pi @ x = D * H * D * x, in place.
 * Self-inverse: calling twice returns the input (up to fp rounding).
 */
__aicore__ inline void RotatePi(const LocalTensor<float> &x, const LocalTensor<float> &signs,
                                const LocalTensor<float> &tmp, int len, float invSqrtLen)
{
    ApplySigns(x, signs, len);
    FwhtInPlace(x, tmp, len);
    Muls(x, x, invSqrtLen, len);      // H normalisation
    PipeBarrier<PIPE_V>();
    ApplySigns(x, signs, len);
}

/*
 * Quantize a rotated, unit-norm vector to BITS-bit codes.
 *
 * Uniform codebook: code = clamp(round(v * sqrtLen / step + centre), 0, LEVELS-1).
 * LUT codebook: nearest centroid by threshold comparison against LEVELS-1
 * midpoints (kept in UB); equivalent to bucketize.
 *
 * `v` is consumed, `codes` receives values in [0, LEVELS).
 */
template <int BITS>
__aicore__ inline void QuantizeVec(const LocalTensor<float> &v, const LocalTensor<int32_t> &codes,
                                   const LocalTensor<float> &bounds, const LocalTensor<float> &scratch,
                                   int len, float sqrtLen, uint32_t codebookMode)
{
    using T = TQTraits<BITS>;
    if (codebookMode == TQ_CB_UNIFORM) {
        const float centre = static_cast<float>(T::kLevels - 1) * 0.5f;
        const float inv = sqrtLen / UniformStep(BITS);
        Muls(scratch, v, inv, len);
        Adds(scratch, scratch, centre, len);
        PipeBarrier<PIPE_V>();
        // round-to-nearest then clamp into range
        Maxs(scratch, scratch, 0.0f, len);
        Mins(scratch, scratch, static_cast<float>(T::kLevels - 1), len);
        PipeBarrier<PIPE_V>();
        Cast(codes, scratch, RoundMode::CAST_RINT, len);
    } else {
        // threshold compare against LEVELS-1 midpoints: code = #bounds below v
        Duplicate(codes, 0, len);
        PipeBarrier<PIPE_V>();
        for (int k = 0; k < T::kLevels - 1; ++k) {
            // scratch = (v >= bounds[k]) ? 1 : 0, accumulated into codes
            Adds(scratch, v, -bounds.GetValue(k), len);
            PipeBarrier<PIPE_V>();
            Maxs(scratch, scratch, 0.0f, len);
            Mins(scratch, scratch, 1.0f, len);
            PipeBarrier<PIPE_V>();
            Ceil(scratch, scratch, len);
            PipeBarrier<PIPE_V>();
            Cast(scratch.ReinterpretCast<int32_t>(), scratch, RoundMode::CAST_RINT, len);
            Add(codes, codes, scratch.ReinterpretCast<int32_t>(), len);
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_V>();
}

/*
 * Dequantize codes back to rotated-basis values.
 * Uniform: affine multiply. LUT: gather from the centroid table.
 * Result is NOT scaled by the stored norm -- callers fold that in (see the
 * pertoken-mul pattern), because the attention path folds ||k|| into the score
 * and ||v|| into the probability weight.
 */
template <int BITS>
__aicore__ inline void DequantizeVec(const LocalTensor<int32_t> &codes, const LocalTensor<float> &out,
                                     const LocalTensor<float> &centroids, int len, float invSqrtLen,
                                     uint32_t codebookMode)
{
    using T = TQTraits<BITS>;
    if (codebookMode == TQ_CB_UNIFORM) {
        const float centre = static_cast<float>(T::kLevels - 1) * 0.5f;
        const float scale = UniformStep(BITS) * invSqrtLen;
        Cast(out, codes, RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
        Adds(out, out, -centre, len);
        PipeBarrier<PIPE_V>();
        Muls(out, out, scale, len);
    } else {
        Gather(out, centroids, codes.ReinterpretCast<uint32_t>(), 0, len);
    }
    PipeBarrier<PIPE_V>();
}


/*
 * PLANAR PACK/UNPACK (vectorised) -- for BITS that divide 8 evenly (2 and 4).
 *
 * msprof on the read kernel: scalar pipe 100.2% utilised, vector 12.8%, MTE2
 * 14.4%. The scalar GetValue/SetValue unpack was the entire bottleneck
 * (256 codes x2 for K/V x16 query heads ~= 8192 scalar ops per token).
 *
 * The fix is a LAYOUT change, not just an instruction change. Interleaved
 * packing (codes 0..7 -> bytes 0..2) cannot be unpacked without a per-lane
 * shuffle. Planar packing puts each code's plane in its own contiguous run:
 *
 *   BITS=4, len=256:  byte[i] = c[i] | (c[i+128] << 4)          i in [0,128)
 *     unpack: c[0:128] = byte & 0x0F ;  c[128:256] = byte >> 4
 *
 *   BITS=2, len=256:  byte[i] = c[i] | c[i+64]<<2 | c[i+128]<<4 | c[i+192]<<6
 *     unpack: four (shift, mask) pairs
 *
 * Byte count is IDENTICAL to the interleaved form, so the FRACTAL_NZ cache
 * layout, packedHalves and every tiling check are unaffected -- only the order
 * of codes within the bytes changes, and pack/unpack agree by construction.
 * Code i still means coordinate i, so the quantiser and dequantiser are
 * untouched.
 */
template <int BITS>
struct TQPlanar {
    static constexpr bool kSupported = (8 % BITS == 0);
    static constexpr int kPlanes = 8 / BITS;              // codes per byte
    __aicore__ static constexpr int Lanes(int len) { return len / kPlanes; }
};

// bytes -> codes, vectorised, using FLOAT ARITHMETIC only.
//
// ShiftRight/ShiftLeft on int16 silently returned 0 on this target (plane 0 came
// back as the raw byte 255 because the mask degenerated to `x - 0`, plane 1 came
// back as 0). And/Or are int16-only and `Ands` is not visible for this chip, so
// the bit slicing is done in float, where every op is bread-and-butter.
//
// Planes are peeled from the TOP down with a running remainder, so only two
// scratch buffers are needed and no masking step is required:
//     for pl = kPlanes-1 .. 0:
//         plane = floor(rem / 2^(BITS*pl));  rem -= plane * 2^(BITS*pl)
/*
 * FUSED unpack + dequantise, uniform codebook only.
 *
 * The split form peels a plane into `pln` (a float already holding an exact
 * integer after Floor), Casts it to int32, and DequantizeVec immediately Casts
 * it straight back to float to compute (code - centre) * step. The int32 round
 * trip buys nothing when the codebook is uniform.
 *
 * Fusing writes (pln - centre) * step directly into `out`, keeping `pln` intact
 * for the remainder chain. Per unpack call at kPlanes=2 that is 13 -> 12 vector
 * ops AND ~16*lanes -> ~12*lanes of element-work; the element saving is the
 * point, because the vector pipe is now the bound (V 2509us vs S 2415us).
 *
 * Arithmetically identical to the split form: Floor leaves an exact integer in
 * pln, so Cast-to-int32-and-back is the identity on it, and the subsequent
 * Adds/Muls are the same operands in the same order. Expect BIT-IDENTICAL gates.
 *
 * The int32 `codes` path is still required for TQ_CB_LUT (Gather indexes by
 * code) and for the code-dump debug variants, so this is a separate entry point
 * rather than a replacement.
 */
template <int BITS>
__aicore__ inline void UnpackDequantVec(const LocalTensor<uint8_t> &bytes,
                                        const LocalTensor<float> &out,
                                        const LocalTensor<half> &tmpH,
                                        const LocalTensor<float> &rem,
                                        const LocalTensor<float> &pln, int len,
                                        float invSqrtLen)
{
    using P = TQPlanar<BITS>;
    using T = TQTraits<BITS>;
    const int lanes = P::Lanes(len);
    const float centre = static_cast<float>(T::kLevels - 1) * 0.5f;
    const float step = UniformStep(BITS) * invSqrtLen;

    Cast(tmpH, bytes, RoundMode::CAST_NONE, lanes);          // u8 -> half
    PipeBarrier<PIPE_V>();
    Cast(rem, tmpH, RoundMode::CAST_NONE, lanes);            // half -> float (0..255)
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (int pl = P::kPlanes - 1; pl >= 0; --pl) {
        float scale = 1.0f;                                   // 2^(BITS*pl)
        for (int k = 0; k < pl; ++k) {
            scale *= static_cast<float>(1 << BITS);
        }
        Muls(pln, rem, 1.0f / scale, lanes);
        PipeBarrier<PIPE_V>();
        Floor(pln, pln, lanes);                               // exact integer code
        PipeBarrier<PIPE_V>();
        // dequantise straight out of pln; pln itself must survive for the peel
        Adds(out[pl * lanes], pln, -centre, lanes);
        PipeBarrier<PIPE_V>();
        Muls(out[pl * lanes], out[pl * lanes], step, lanes);
        PipeBarrier<PIPE_V>();
        if (pl > 0) {
            Muls(pln, pln, scale, lanes);
            PipeBarrier<PIPE_V>();
            Sub(rem, rem, pln, lanes);                        // peel this plane off
            PipeBarrier<PIPE_V>();
        }
    }
}

template <int BITS>
__aicore__ inline void UnpackCodesVec(const LocalTensor<uint8_t> &bytes,
                                      const LocalTensor<int32_t> &codes,
                                      const LocalTensor<half> &tmpH,
                                      const LocalTensor<float> &rem,
                                      const LocalTensor<float> &pln, int len)
{
    using P = TQPlanar<BITS>;
    const int lanes = P::Lanes(len);
    Cast(tmpH, bytes, RoundMode::CAST_NONE, lanes);          // u8 -> half
    PipeBarrier<PIPE_V>();
    Cast(rem, tmpH, RoundMode::CAST_NONE, lanes);            // half -> float (0..255)
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (int pl = P::kPlanes - 1; pl >= 0; --pl) {
        float scale = 1.0f;                                   // 2^(BITS*pl)
        for (int k = 0; k < pl; ++k) {
            scale *= static_cast<float>(1 << BITS);
        }
        Muls(pln, rem, 1.0f / scale, lanes);
        PipeBarrier<PIPE_V>();
        Floor(pln, pln, lanes);
        PipeBarrier<PIPE_V>();
        Cast(codes[pl * lanes], pln, RoundMode::CAST_RINT, lanes);
        PipeBarrier<PIPE_V>();
        if (pl > 0) {
            Muls(pln, pln, scale, lanes);
            PipeBarrier<PIPE_V>();
            Sub(rem, rem, pln, lanes);                        // peel this plane off
            PipeBarrier<PIPE_V>();
        }
    }
}

// codes -> bytes, vectorised. Planes do not overlap, so OR is just ADD:
//     byte = sum_p code[p] * 2^(BITS*p)
template <int BITS>
__aicore__ inline void PackCodesVec(const LocalTensor<int32_t> &codes,
                                    const LocalTensor<uint8_t> &bytes,
                                    const LocalTensor<half> &tmpH,
                                    const LocalTensor<float> &f0,
                                    const LocalTensor<float> &f1, int len)
{
    using P = TQPlanar<BITS>;
    const int lanes = P::Lanes(len);
    constexpr float kUnit = static_cast<float>(1 << BITS);
    Cast(f0, codes[0], RoundMode::CAST_NONE, lanes);         // plane 0
    PipeBarrier<PIPE_V>();
#pragma unroll
    for (int pl = 1; pl < P::kPlanes; ++pl) {
        float mul = 1.0f;
        for (int k = 0; k < pl; ++k) {
            mul *= kUnit;
        }
        Cast(f1, codes[pl * lanes], RoundMode::CAST_NONE, lanes);
        PipeBarrier<PIPE_V>();
        Muls(f1, f1, mul, lanes);
        PipeBarrier<PIPE_V>();
        Add(f0, f0, f1, lanes);
        PipeBarrier<PIPE_V>();
    }
    Cast(tmpH, f0, RoundMode::CAST_NONE, lanes);
    PipeBarrier<PIPE_V>();
    Cast(bytes, tmpH, RoundMode::CAST_NONE, lanes);
    PipeBarrier<PIPE_V>();
}

/*
 * Pack `len` codes into bytes. 8 codes -> BITS bytes, exactly, for BITS in 2/3/4.
 * Emitted bytes are reinterpreted as half by the caller: the 310P paged cache
 * must be fp16-typed (an int8-typed cache reports success and writes nothing).
 */
template <int BITS>
__aicore__ inline void PackCodes(const LocalTensor<int32_t> &codes, const LocalTensor<uint8_t> &bytes, int len)
{
    using T = TQTraits<BITS>;
    const int words = len / T::kCodesPerWord;
    for (int w = 0; w < words; ++w) {
        uint32_t acc = 0;
        const int c0 = w * T::kCodesPerWord;
#pragma unroll
        for (int i = 0; i < T::kCodesPerWord; ++i) {
            acc |= (static_cast<uint32_t>(codes.GetValue(c0 + i)) & T::kMask) << (BITS * i);
        }
        const int b0 = w * T::kBytesPerWord;
#pragma unroll
        for (int j = 0; j < T::kBytesPerWord; ++j) {
            bytes.SetValue(b0 + j, static_cast<uint8_t>((acc >> (8 * j)) & 0xFFu));
        }
    }
}

// Inverse of PackCodes.
template <int BITS>
__aicore__ inline void UnpackCodes(const LocalTensor<uint8_t> &bytes, const LocalTensor<int32_t> &codes, int len)
{
    using T = TQTraits<BITS>;
    const int words = len / T::kCodesPerWord;
    for (int w = 0; w < words; ++w) {
        uint32_t acc = 0;
        const int b0 = w * T::kBytesPerWord;
#pragma unroll
        for (int j = 0; j < T::kBytesPerWord; ++j) {
            acc |= static_cast<uint32_t>(bytes.GetValue(b0 + j)) << (8 * j);
        }
        const int c0 = w * T::kCodesPerWord;
#pragma unroll
        for (int i = 0; i < T::kCodesPerWord; ++i) {
            codes.SetValue(c0 + i, static_cast<int32_t>((acc >> (BITS * i)) & T::kMask));
        }
    }
}

}  // namespace TurboQuant

#endif  // TURBOQUANT_PAGED_ATTENTION_V310_CODEC_H
