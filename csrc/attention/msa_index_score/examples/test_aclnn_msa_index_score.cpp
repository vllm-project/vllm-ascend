/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_msa_index_score.cpp
 * \brief aclnnMsaIndexScore 调用示例，内置 CPU golden 做端到端精度自验证。
 *
 * 用例矩阵覆盖：Prefill 多 M-tile、prefix 非 128 对齐的边界 block、varlen 多 batch、
 * Decode(q_len=1)、投机解码(q_len>1)、长序列多 S-tile 轮转、block_table 乱序、
 * 无效尾填充、bf16 / fp16 双 dtype、int8 key 前融合反量化、PA BNBD、TND packed key。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_msa_index_score.h"

namespace {

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t NUM_KV_HEADS = 1; // MSA index cache 为单头共享，P0 仅支持 1
constexpr int64_t SCORE_STRIDE_ALIGN = 16;
constexpr float kNegInf = -3.4028234663852886e+38F;
constexpr float kAtol = 1e-3F;
constexpr float kRtol = 1e-3F;

enum class KeyLayout {
    BBND = 0,
    BNBD = 1,
    TND = 2,
};

const char *LayoutKeyName(KeyLayout layout)
{
    switch (layout) {
        case KeyLayout::TND:
            return "TND";
        case KeyLayout::BNBD:
            return "BNBD";
        case KeyLayout::BBND:
        default:
            return "BBND";
    }
}

struct TestCase {
    const char *name;
    int64_t numQHeads;
    int64_t headDim;
    int64_t numPages;
    std::vector<int32_t> qLen;     // 每个请求的 query 长度
    std::vector<int32_t> kvLen;    // 每个请求可见的 kv 长度
    std::vector<int32_t> startLoc; // 当前 query 所在逻辑 block 索引（local_mask）
    bool useBf16;
    bool useInt8Key;        // true: key=int8，scale=[NP,N_kv,P] 或 TND [T2,N2]
    int64_t sparseMode = 3; // 0 / 3
    KeyLayout keyLayout = KeyLayout::BBND;
};

constexpr float kLocalScoreInit = 1.0e30F;
constexpr float kLocalScoreLocal = 1.0e29F;
constexpr int64_t kInitBlocks = 0;
constexpr int64_t kLocalBlocks = 1;
constexpr int64_t kAttenMaskSize = 2048;

int64_t CeilDivI64(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t RoundUpI64(int64_t a, int64_t b) { return CeilDivI64(a, b) * b; }

// 简单可复现的伪随机源，取值落在 [-1, 1)。
float PseudoRandom(uint32_t seed)
{
    seed = seed * 1664525U + 1013904223U;
    seed ^= seed >> 16;
    seed = seed * 2246822519U;
    seed ^= seed >> 13;
    return static_cast<float>(seed % 20000U) / 10000.0F - 1.0F;
}

// bf16 <-> fp32：截断低 16 位尾数（round-to-nearest-even 对本用例不必要）。
uint16_t FloatToBf16(float v)
{
    uint32_t bits = 0;
    (void)memcpy(&bits, &v, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t v)
{
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out = 0.0F;
    (void)memcpy(&out, &bits, sizeof(out));
    return out;
}

class DeviceBuffer {
public:
    ~DeviceBuffer()
    {
        for (auto *t : tensors_) {
            if (t != nullptr) {
                (void)aclDestroyTensor(t);
            }
        }
        for (auto *p : addrs_) {
            if (p != nullptr) {
                (void)aclrtFree(p);
            }
        }
    }

    template <typename T>
    aclTensor *Create(const std::vector<T> &host, const std::vector<int64_t> &shape, aclDataType dtype,
                      void **addrOut = nullptr)
    {
        void *devAddr = nullptr;
        const size_t bytes = host.size() * sizeof(T);
        if (aclrtMalloc(&devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        addrs_.push_back(devAddr);
        if (aclrtMemcpy(devAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            return nullptr;
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        aclTensor *t = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), devAddr);
        tensors_.push_back(t);
        if (addrOut != nullptr) {
            *addrOut = devAddr;
        }
        return t;
    }

private:
    std::vector<void *> addrs_;
    std::vector<aclTensor *> tensors_;
};

/// CPU 参考：score = Maxpool[(scale·)Q@Kᵀ + atten_mask] + local_mask
void ComputeGolden(const TestCase &tc, const std::vector<int32_t> &actualSeqQlen,
                   const std::vector<int32_t> &actualSeqKlen, const std::vector<int32_t> &blockTable, int64_t maxBlocks,
                   int64_t scoreStride, int64_t totalQ, const std::vector<float> &queryF,
                   const std::vector<float> &keyF, const std::vector<float> &deqScale, std::vector<float> &golden)
{
    const int64_t batch = static_cast<int64_t>(tc.qLen.size());
    golden.assign(static_cast<size_t>(tc.numQHeads * totalQ * scoreStride), kNegInf);

    auto keyAt = [&](int64_t pageOrTok, int64_t n, int64_t d) -> float {
        if (tc.keyLayout == KeyLayout::TND) {
            return keyF[static_cast<size_t>(((pageOrTok + n) * NUM_KV_HEADS) * tc.headDim + d)];
        }
        // BBND [NP,P,1,D] 与 BNBD [NP,1,P,D] 在 N2=1 时同址
        return keyF[static_cast<size_t>(((pageOrTok * BLOCK_SIZE) + n) * tc.headDim + d)];
    };
    auto scaleAt = [&](int64_t pageOrTok, int64_t n) -> float {
        if (!tc.useInt8Key) {
            return 1.0F;
        }
        if (tc.keyLayout == KeyLayout::TND) {
            return deqScale[static_cast<size_t>(pageOrTok + n)];
        }
        return deqScale[static_cast<size_t>(pageOrTok * BLOCK_SIZE + n)];
    };

    for (int64_t b = 0; b < batch; ++b) {
        const int32_t qBegin = actualSeqQlen[b];
        const int32_t qEnd = actualSeqQlen[b + 1];
        const int32_t qLen = qEnd - qBegin;
        const int32_t kvLen = tc.kvLen[b];
        const int64_t numBlocks = CeilDivI64(kvLen, BLOCK_SIZE);
        const int32_t qBlock = tc.startLoc[b];
        const int32_t localStart =
            (qBlock + 1 > static_cast<int32_t>(kLocalBlocks)) ? (qBlock + 1 - static_cast<int32_t>(kLocalBlocks)) : 0;
        const int32_t cuK = (tc.keyLayout == KeyLayout::TND) ? actualSeqKlen[b] : 0;

        for (int32_t t = qBegin; t < qEnd; ++t) {
            const int32_t tOff = t - qBegin;
            int32_t visibleKeyEnd = kvLen;
            if (tc.sparseMode == 3) {
                visibleKeyEnd = kvLen - qLen + tOff + 1;
                if (visibleKeyEnd < 0) {
                    visibleKeyEnd = 0;
                }
                if (visibleKeyEnd > kvLen) {
                    visibleKeyEnd = kvLen;
                }
            }
            for (int64_t h = 0; h < tc.numQHeads; ++h) {
                for (int64_t blk = 0; blk < numBlocks; ++blk) {
                    const int32_t pageOrTok = (tc.keyLayout == KeyLayout::TND) ?
                                                  (cuK + static_cast<int32_t>(blk * BLOCK_SIZE)) :
                                                  blockTable[b * maxBlocks + blk];
                    float best = 0.0F;
                    bool any = false;
                    for (int64_t n = 0; n < BLOCK_SIZE; ++n) {
                        if (blk * BLOCK_SIZE + n >= visibleKeyEnd) {
                            break;
                        }
                        float acc = 0.0F;
                        const float s = scaleAt(pageOrTok, n);
                        for (int64_t d = 0; d < tc.headDim; ++d) {
                            acc += queryF[((t * tc.numQHeads) + h) * tc.headDim + d] * (keyAt(pageOrTok, n, d) * s);
                        }
                        if (!any || acc > best) {
                            best = acc;
                            any = true;
                        }
                    }
                    if (!any) {
                        continue;
                    }
                    golden[(h * totalQ + t) * scoreStride + blk] = best;
                }
                for (int64_t blk = 0; blk < numBlocks; ++blk) {
                    float boost = 0.0F;
                    if (blk < kInitBlocks) {
                        boost = kLocalScoreInit;
                    }
                    if (blk >= localStart && blk <= qBlock) {
                        boost = kLocalScoreLocal;
                    }
                    if (boost != 0.0F) {
                        golden[(h * totalQ + t) * scoreStride + blk] = boost;
                    }
                }
            }
        }
    }
}

void PrintVec(const char *tag, const float *data, int64_t n)
{
    (void)printf("  %s[", tag);
    for (int64_t i = 0; i < n; ++i) {
        (void)printf("%s%.6g", i == 0 ? "" : ", ", static_cast<double>(data[i]));
    }
    (void)printf("]\n");
}

/// 小尺寸用例的 host 侧逐步拆解：Q/K -> DOT/DEQUANT -> MASK -> MAX。
void PrintTracePipeline(const TestCase &tc, const std::vector<int32_t> &actualSeqQlen,
                        const std::vector<int32_t> &blockTable, int64_t maxBlocks, int64_t scoreStride, int64_t totalQ,
                        const std::vector<float> &queryF, const std::vector<float> &keyF,
                        const std::vector<float> &deqScale, const std::vector<float> &actual)
{
    (void)printf("\n======== HOST TRACE: %s (int8=%d) ========\n", tc.name, tc.useInt8Key ? 1 : 0);
    (void)printf("shape: Hq=%ld D=%ld qLen=%d kvLen=%d startLoc=%d blockSize=%ld maxBlocks=%ld scoreStride=%ld\n",
                 tc.numQHeads, tc.headDim, tc.qLen[0], tc.kvLen[0], tc.startLoc[0], BLOCK_SIZE, maxBlocks, scoreStride);
    const int32_t page = blockTable[0];
    for (int64_t t = 0; t < tc.qLen[0]; ++t) {
        for (int64_t h = 0; h < tc.numQHeads; ++h) {
            const int64_t flat = t * tc.numQHeads + h;
            const float *q = &queryF[static_cast<size_t>(flat * tc.headDim)];
            (void)printf("\n-- row flat=%ld (token=%ld head=%ld) --\n", flat, t, h);
            PrintVec("Q", q, tc.headDim);
            const int32_t tOff = static_cast<int32_t>(t);
            int32_t visibleKeyEnd = tc.kvLen[0];
            if (tc.sparseMode == 3) {
                visibleKeyEnd = tc.kvLen[0] - tc.qLen[0] + tOff + 1;
                if (visibleKeyEnd < 0) {
                    visibleKeyEnd = 0;
                }
                if (visibleKeyEnd > tc.kvLen[0]) {
                    visibleKeyEnd = tc.kvLen[0];
                }
            }
            float best = kNegInf;
            bool any = false;
            for (int64_t n = 0; n < visibleKeyEnd + 2 && n < BLOCK_SIZE; ++n) {
                const float *k = &keyF[static_cast<size_t>((page * BLOCK_SIZE + n) * tc.headDim)];
                const float ds = tc.useInt8Key ? deqScale[static_cast<size_t>(page) * BLOCK_SIZE + n] : 1.0F;
                float acc = 0.0F;
                for (int64_t d = 0; d < tc.headDim; ++d) {
                    acc += q[d] * (k[d] * ds);
                }
                const bool visible = (n < visibleKeyEnd);
                (void)printf("    S[%ld]=%.6g deqScale=%.6g %s\n", n, static_cast<double>(acc), static_cast<double>(ds),
                             visible ? "KEEP" : "MASK");
                if (visible && (!any || acc > best)) {
                    best = acc;
                    any = true;
                }
            }
            const float deviceOut = actual[static_cast<size_t>((h * totalQ + (actualSeqQlen[0] + t)) * scoreStride)];
            (void)printf("  [MAX]=%.6g [OUT]=%.6g\n", static_cast<double>(any ? best : kNegInf),
                         static_cast<double>(deviceOut));
        }
    }
    (void)printf("======== END HOST TRACE ========\n\n");
    (void)maxBlocks;
}

bool Compare(const std::string &name, const std::vector<float> &actual, const std::vector<float> &golden)
{
    size_t badCount = 0;
    size_t infBad = 0;
    size_t total = 0;
    float maxAbsDiff = 0.0F;
    size_t firstBad = 0;
    bool hasBad = false;

    for (size_t i = 0; i < golden.size(); ++i) {
        const float g = golden[i];
        const float a = actual[i];
        if (g <= kNegInf * 0.5F) { // 填充位：要求实测同样是极小值
            if (!(a <= kNegInf * 0.5F)) {
                if (!hasBad) {
                    firstBad = i;
                    hasBad = true;
                }
                ++infBad;
            }
            continue;
        }
        if (g >= 1.0e28F) { // local_mask 强制高分
            if (!(a >= 1.0e28F)) {
                if (!hasBad) {
                    firstBad = i;
                    hasBad = true;
                }
                ++badCount;
            }
            continue;
        }
        ++total;
        const float diff = std::fabs(a - g);
        maxAbsDiff = diff > maxAbsDiff ? diff : maxAbsDiff;
        if (diff > kAtol + kRtol * std::fabs(g)) {
            if (!hasBad) {
                firstBad = i;
                hasBad = true;
            }
            ++badCount;
        }
    }

    const bool pass = (badCount == 0) && (infBad == 0);
    (void)printf("  [%s] valid=%zu mismatch=%zu fill_mismatch=%zu max_abs_diff=%.6g -> %s\n", name.c_str(), total,
                 badCount, infBad, static_cast<double>(maxAbsDiff), pass ? "PASS" : "FAIL");
    if (!pass) {
        (void)printf("    first mismatch at %zu: actual=%g golden=%g\n", firstBad,
                     static_cast<double>(actual[firstBad]), static_cast<double>(golden[firstBad]));
    }
    return pass;
}

bool RunCase(const TestCase &tc, aclrtStream stream)
{
    const int64_t batch = static_cast<int64_t>(tc.qLen.size());
    std::vector<int32_t> actualSeqQlen(batch + 1, 0);
    for (int64_t b = 0; b < batch; ++b) {
        actualSeqQlen[b + 1] = actualSeqQlen[b] + tc.qLen[b];
    }
    const int64_t totalQ = actualSeqQlen[batch];

    int64_t maxBlocks = 1;
    for (int64_t b = 0; b < batch; ++b) {
        maxBlocks = std::max<int64_t>(maxBlocks, CeilDivI64(tc.kvLen[b], BLOCK_SIZE));
    }
    const int64_t scoreStride = RoundUpI64(maxBlocks, SCORE_STRIDE_ALIGN);

    // block_table 故意打乱，验证 paged 间接寻址。
    std::vector<int32_t> blockTable(static_cast<size_t>(batch * maxBlocks), 0);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t k = 0; k < maxBlocks; ++k) {
            blockTable[b * maxBlocks + k] = static_cast<int32_t>((b * 7 + k * 3 + 1) % tc.numPages);
        }
    }

    // fp32 参考数据；再按 dtype 转成低精度输入，golden 用转换后的值以对齐数值路径。
    const bool isTnd = (tc.keyLayout == KeyLayout::TND);
    int64_t totalK = 0;
    std::vector<int32_t> actualSeqKlenPrefix(static_cast<size_t>(batch + 1), 0);
    if (isTnd) {
        for (int64_t b = 0; b < batch; ++b) {
            actualSeqKlenPrefix[b + 1] = actualSeqKlenPrefix[b] + tc.kvLen[b];
        }
        totalK = actualSeqKlenPrefix[batch];
    }
    const int64_t keyTokens = isTnd ? totalK : (tc.numPages * BLOCK_SIZE);
    std::vector<float> queryF(static_cast<size_t>(totalQ * tc.numQHeads * tc.headDim));
    std::vector<float> keyF(static_cast<size_t>(keyTokens * tc.headDim));
    std::vector<uint16_t> queryBf(queryF.size());
    std::vector<uint16_t> keyBf(keyF.size());
    std::vector<aclFloat16> queryHf(queryF.size());
    std::vector<aclFloat16> keyHf(keyF.size());

    for (size_t i = 0; i < queryF.size(); ++i) {
        const float v = PseudoRandom(static_cast<uint32_t>(i) + 1U) * 0.5F;
        if (tc.useBf16) {
            queryBf[i] = FloatToBf16(v);
            queryF[i] = Bf16ToFloat(queryBf[i]);
        } else {
            queryHf[i] = aclFloatToFloat16(v);
            queryF[i] = aclFloat16ToFloat(queryHf[i]);
        }
    }
    for (size_t i = 0; i < keyF.size(); ++i) {
        if (tc.useInt8Key) {
            // int8 量化值：落在 [-64, 63]，golden 用同一整数值的 fp32。
            const int8_t qv =
                static_cast<int8_t>(static_cast<int>(PseudoRandom(static_cast<uint32_t>(i) + 999983U) * 64.0F));
            keyF[i] = static_cast<float>(qv);
        } else {
            const float v = PseudoRandom(static_cast<uint32_t>(i) + 999983U) * 0.5F;
            if (tc.useBf16) {
                keyBf[i] = FloatToBf16(v);
                keyF[i] = Bf16ToFloat(keyBf[i]);
            } else {
                keyHf[i] = aclFloatToFloat16(v);
                keyF[i] = aclFloat16ToFloat(keyHf[i]);
            }
        }
    }
    // 反量化 scale：PA [NP, N_kv=1, P]；TND [T2, N2=1]。
    std::vector<float> deqScale(static_cast<size_t>(keyTokens), 1.0F);
    std::vector<int8_t> keyI8;
    if (tc.useInt8Key) {
        keyI8.resize(keyF.size());
        for (size_t i = 0; i < keyF.size(); ++i) {
            keyI8[i] = static_cast<int8_t>(keyF[i]);
        }
        for (size_t i = 0; i < deqScale.size(); ++i) {
            deqScale[i] = 0.01F + 0.02F * (PseudoRandom(static_cast<uint32_t>(i) + 424242U) + 1.0F);
        }
    }

    DeviceBuffer buf;
    const std::vector<int64_t> queryShape = {totalQ, tc.numQHeads, tc.headDim};
    std::vector<int64_t> keyShape;
    if (isTnd) {
        keyShape = {totalK, NUM_KV_HEADS, tc.headDim};
    } else if (tc.keyLayout == KeyLayout::BNBD) {
        keyShape = {tc.numPages, NUM_KV_HEADS, BLOCK_SIZE, tc.headDim};
    } else {
        keyShape = {tc.numPages, BLOCK_SIZE, NUM_KV_HEADS, tc.headDim};
    }
    const std::vector<int64_t> scoreShape = {tc.numQHeads, totalQ, scoreStride};

    aclTensor *queryT =
        tc.useBf16 ? buf.Create(queryBf, queryShape, ACL_BF16) : buf.Create(queryHf, queryShape, ACL_FLOAT16);
    aclTensor *keyT = nullptr;
    if (tc.useInt8Key) {
        keyT = buf.Create(keyI8, keyShape, ACL_INT8);
    } else {
        keyT = tc.useBf16 ? buf.Create(keyBf, keyShape, ACL_BF16) : buf.Create(keyHf, keyShape, ACL_FLOAT16);
    }
    aclTensor *blockTableT = nullptr;
    if (!isTnd) {
        blockTableT = buf.Create(blockTable, {batch, maxBlocks}, ACL_INT32);
    }
    aclTensor *scaleT = nullptr;
    if (tc.useInt8Key) {
        if (isTnd) {
            scaleT = buf.Create(deqScale, {totalK, NUM_KV_HEADS}, ACL_FLOAT);
        } else {
            scaleT = buf.Create(deqScale, {tc.numPages, NUM_KV_HEADS, BLOCK_SIZE}, ACL_FLOAT);
        }
    }
    aclTensor *actualSeqQlenT = buf.Create(actualSeqQlen, {batch + 1}, ACL_INT32);
    aclTensor *actualSeqKlenT =
        isTnd ? buf.Create(actualSeqKlenPrefix, {batch + 1}, ACL_INT32) : buf.Create(tc.kvLen, {batch}, ACL_INT32);
    aclTensor *startLocT = buf.Create(tc.startLoc, {batch}, ACL_INT32);
    std::vector<int8_t> attenMaskHost(static_cast<size_t>(kAttenMaskSize * kAttenMaskSize), 0);
    aclTensor *attenMaskT = nullptr;
    if (tc.sparseMode == 3) {
        attenMaskT = buf.Create(attenMaskHost, {kAttenMaskSize, kAttenMaskSize}, ACL_INT8);
    }

    std::vector<float> scoreInit(static_cast<size_t>(tc.numQHeads * totalQ * scoreStride), 0.0F);
    void *scoreDev = nullptr;
    aclTensor *scoreT = buf.Create(scoreInit, scoreShape, ACL_FLOAT, &scoreDev);

    if (queryT == nullptr || keyT == nullptr || actualSeqQlenT == nullptr || actualSeqKlenT == nullptr ||
        startLocT == nullptr || scoreT == nullptr || (!isTnd && blockTableT == nullptr) ||
        (tc.sparseMode == 3 && attenMaskT == nullptr)) {
        (void)printf("  [%s] create tensor failed -> FAIL\n", tc.name);
        return false;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    int ret =
        aclnnMsaIndexScoreGetWorkspaceSize(queryT, keyT, blockTableT, scaleT, attenMaskT, actualSeqQlenT,
                                           actualSeqKlenT, startLocT, const_cast<char *>(LayoutKeyName(tc.keyLayout)),
                                           tc.sparseMode, kInitBlocks, kLocalBlocks, scoreT, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        (void)printf("  [%s] GetWorkspaceSize failed, ERROR %d -> FAIL\n", tc.name, ret);
        return false;
    }
    if (workspaceSize > 0ULL && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        (void)printf("  [%s] malloc workspace failed -> FAIL\n", tc.name);
        return false;
    }
    ret = aclnnMsaIndexScore(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        (void)printf("  [%s] aclnnMsaIndexScore failed, ERROR %d -> FAIL\n", tc.name, ret);
        (void)aclrtFree(workspaceAddr);
        return false;
    }
    (void)aclrtSynchronizeStream(stream);

    std::vector<float> actual(scoreInit.size(), 0.0F);
    (void)aclrtMemcpy(actual.data(), actual.size() * sizeof(float), scoreDev, actual.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }

    std::vector<float> golden;
    ComputeGolden(tc, actualSeqQlen, isTnd ? actualSeqKlenPrefix : tc.kvLen, blockTable, maxBlocks, scoreStride, totalQ,
                  queryF, keyF, deqScale, golden);
    if (std::strstr(tc.name, "debug-trace") != nullptr) {
        PrintTracePipeline(tc, actualSeqQlen, blockTable, maxBlocks, scoreStride, totalQ, queryF, keyF, deqScale,
                           actual);
    }
    return Compare(tc.name, actual, golden);
}

} // namespace

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS ||
        aclrtCreateStream(&stream) != ACL_SUCCESS) {
        (void)printf("[FAIL] init acl failed\n");
        return -1;
    }

    // startLoc 为逻辑 block 索引；因果由 sparseMode=3（rightDownCausal）承担。
    const std::vector<TestCase> cases = {
        // name                       Hq   D    pages  qLen           kvLen         startLoc(block) bf16  int8
        {"L0-debug-trace", 2, 16, 2, {2}, {5}, {16}, false, false},
        {"L0-int8-dequant-trace", 2, 16, 2, {2}, {5}, {16}, false, true},
        {"L0-prefill-aligned", 8, 128, 8, {32}, {256}, {1}, false, false},
        {"L1-prefill-unaligned", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, false, false},
        {"L1-prefill-multi-mtile", 8, 128, 16, {64, 48}, {700, 520}, {4, 3}, false, false},
        {"L1-decode-lq1",
         8,
         128,
         16,
         {1, 1, 1, 1, 1, 1},
         {900, 512, 128, 129, 1, 4096},
         {7, 3, 0, 1, 0, 31},
         false,
         false},
        {"L1-decode-speculative", 8, 128, 16, {4, 2}, {1024, 260}, {7, 2}, false, false},
        {"L1-long-seq-multi-stile", 8, 128, 40, {8}, {4096}, {31}, false, false},
        {"L1-bf16", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, true, false},
        {"L1-head-dim-64", 8, 64, 8, {32, 17}, {300, 130}, {2, 0}, false, false},
        {"L1-heads-16", 16, 128, 8, {13}, {300}, {2}, false, false},
        {"L1-int8-dequant", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, false, true},
        {"L2-tiny-kv", 8, 128, 4, {1, 1}, {1, 3}, {0, 0}, false, false},
        {"L1-bnbd", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, false, false, 3, KeyLayout::BNBD},
        {"L1-bnbd-int8", 8, 128, 8, {32, 17}, {300, 130}, {2, 0}, false, true, 3, KeyLayout::BNBD},
        {"L1-tnd-unaligned", 8, 128, 0, {32, 17}, {300, 130}, {2, 0}, false, false, 3, KeyLayout::TND},
        {"L1-tnd-int8", 8, 128, 0, {32, 17}, {300, 130}, {2, 0}, false, true, 3, KeyLayout::TND},
        {"L0-tnd-tiny", 2, 16, 0, {2}, {5}, {16}, false, false, 3, KeyLayout::TND},
    };

    size_t passed = 0;
    (void)printf("running %zu MsaIndexScore cases\n", cases.size());
    for (const auto &tc : cases) {
        if (RunCase(tc, stream)) {
            ++passed;
        }
    }
    (void)printf("%s: %zu/%zu cases passed\n", passed == cases.size() ? "[PASS]" : "[FAIL]", passed, cases.size());

    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
    return passed == cases.size() ? 0 : -1;
}
