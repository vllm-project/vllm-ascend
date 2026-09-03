/**
 * A5/arch35 tiling helpers（A2 多核流水亦复用）。
 *
 * 放在 op_host/arch35/：ascend950 构建时 ARCH_DIRECTORY 含 arch35，
 * cmake 会自动 GLOB 本目录下 *_tiling*.cpp（见 obj_func.cmake）。
 */
#ifndef K2Q_CSR_TILING_ARCH35_H
#define K2Q_CSR_TILING_ARCH35_H

#include <cstdint>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace k2q_csr_arch35 {

constexpr uint64_t kUbThresholdBytes = 180ULL * 1024ULL;
constexpr int64_t kRowMapUbMax = 32768;

inline bool IsArch35Ub(uint64_t ubSize)
{
    return ubSize >= kUbThresholdBytes;
}

/** 真正启用 A5 算法字段：仅 950 / 910_95（与 opFile=k2q_csr_apt 对齐） */
inline bool IsArch35Soc(platform_ascendc::SocVersion soc)
{
    return soc == platform_ascendc::SocVersion::ASCEND950;
}

/** q 维多核划分：G = min(aivNum, T)，每核一段连续 q（MC 路径） */
inline void FillMultiCoreGroups(int64_t T, int64_t aivNum, int64_t &numGroups, int64_t &qPerGroup)
{
    if (T <= 0) {
        numGroups = 1;
        qPerGroup = 1;
        return;
    }
    int64_t cores = aivNum > 0 ? aivNum : 1;
    numGroups = T < cores ? T : cores;
    if (numGroups < 1) {
        numGroups = 1;
    }
    qPerGroup = (T + numGroups - 1) / numGroups;
    if (qPerGroup < 1) {
        qPerGroup = 1;
    }
}

/**
 * 对齐 CUDA build_k2q_csr.cu CTA 策略（SIMT 路径）：
 *   target_g ≈ aiv * 2..3（单 wave），且每 CTA 至少 kMinQPerCta 个 q。
 *   NPU AIV-only：G ≤ aiv（blockDim 上限）；关键差异相对 MC 是加大 q_per_cta。
 *   q_per_cta = ceil(T/G)，再回写 G = ceil(T/q_per_cta)。
 */
inline void FillCudaLikeGroups(int64_t T, int64_t aivNum, int64_t &numGroups, int64_t &qPerGroup)
{
    constexpr int64_t kMinQPerCta = 256;
    if (T <= 0) {
        numGroups = 1;
        qPerGroup = 1;
        return;
    }
    int64_t aiv = aivNum > 0 ? aivNum : 1;
    // CUDA: num_sms * min(max_ctas_per_sm, 3)；NPU 用 aiv*2 再夹到 aiv
    int64_t targetG = aiv * 2;
    if (targetG > aiv * 3) {
        targetG = aiv * 3;
    }
    if (targetG > aiv) {
        targetG = aiv;
    }
    int64_t maxGForQ = (T + kMinQPerCta - 1) / kMinQPerCta;
    if (maxGForQ < 1) {
        maxGForQ = 1;
    }
    int64_t G = targetG;
    if (G > maxGForQ) {
        G = maxGForQ;
    }
    if (G > T) {
        G = T;
    }
    if (G < 1) {
        G = 1;
    }
    int64_t qpc = (T + G - 1) / G;
    if (qpc < 1) {
        qpc = 1;
    }
    G = (T + qpc - 1) / qpc;
    if (G < 1) {
        G = 1;
    }
    if (G > aiv) {
        G = aiv;
        qpc = (T + G - 1) / G;
        if (qpc < 1) {
            qpc = 1;
        }
    }
    numGroups = G;
    qPerGroup = qpc;
}

// 兼容旧名
inline void FillArch35Groups(int64_t T, int64_t aivNum, int64_t &numGroups, int64_t &qPerGroup)
{
    FillMultiCoreGroups(T, aivNum, numGroups, qPerGroup);
}

/** Stage M：row_map[B,max_kv] + token_batch[T] */
inline int64_t MetaWorkspaceBytes(int64_t B, int64_t maxKv, int64_t T)
{
    return B * maxKv * 4 + T * 4;
}

/** Hist workspace：tile_counts[G,H,R] + abs_base[G,H,R] + row_counts[H,R] + soft-barrier */
inline int64_t WorkspaceBytes(int64_t H, int64_t totalRows, int64_t numGroups)
{
    int64_t g = numGroups > 0 ? numGroups : 1;
    int64_t h = H > 0 ? H : 1;
    int64_t r = totalRows > 0 ? totalRows : 1;
    return g * h * r * 4 * 2 + h * r * 4 + 16 * 4;
}

} // namespace k2q_csr_arch35
} // namespace optiling

#endif
