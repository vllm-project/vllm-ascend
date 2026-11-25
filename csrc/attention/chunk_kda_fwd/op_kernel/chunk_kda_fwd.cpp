#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "chunk_kda_fwd_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310 && \
    (!defined(TILING_KEY_VAR) || TILING_KEY_VAR == 2UL)
#define KDA_COMPILE_ARCH35_FAST_PATH 1
#include "arch35/chunk_kda_fwd_impl.h"
#else
#define KDA_COMPILE_ARCH35_FAST_PATH 0
#endif

namespace KdaForward {

constexpr int64_t KDA_STAGE_FULL = -1;
constexpr int64_t KDA_STAGE_GATE_PREPARE = 0;
constexpr int64_t KDA_STAGE_POST_WU = 1;
constexpr int64_t KDA_STAGE_FWD_H = 2;
constexpr int64_t KDA_STAGE_FINALIZE = 3;

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchStage(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    auto addresses = ResolveAddresses(
        finalState, gk, w, u, qg, kg, vNew, h, userWorkspace, tiling);
    addresses.qgScaled = qgScaled;
    if (tiling.stage == KDA_STAGE_GATE_PREPARE) {
        RunGateCumsum(g, aLog, dtBias, cuSeqlens, addresses.gk, tiling);
        if (!tiling.computeGateInPrepare) {
            SyncAll<false>();
        }
        TPipe pipe;
        KdaPrepare::RunChunkKdaPrepare<SAFE_GATE, T, float, BETA_T,
            TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, addresses.gk, g, aLog, dtBias, beta, initialState,
            cuSeqlens, chunkIndices, aqk, akk, addresses.qg,
            addresses.qgScaled, addresses.w, uSeed, addresses.kg,
            userWorkspace, tiling, pipe, tiling.storeQG);
    } else if (tiling.stage == KDA_STAGE_POST_WU) {
        TPipe pipe;
        KdaPostWu::RunChunkKdaPostWu<T, float, BETA_T>(
            q, k, v, addresses.gk, beta, initialState, cuSeqlens,
            chunkIndices, addresses.w, akk, uSeed, addresses.w,
            addresses.u, addresses.kg, addresses.vNew, userWorkspace,
            tiling, pipe);
    } else if (tiling.stage == KDA_STAGE_FWD_H) {
        if (tiling.vHeadDim > 128) {
            RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes256>(
                initialState, cuSeqlens, chunkIndices, addresses,
                userWorkspace, tiling);
        } else {
            RunFwdH<T, Catlass::Gemm::Kernel::GDNFwdHTileShapes128>(
                initialState, cuSeqlens, chunkIndices, addresses,
                userWorkspace, tiling);
        }
    } else if (tiling.stage == KDA_STAGE_FINALIZE) {
        TPipe pipe;
        KdaFinalize::RunChunkKdaOutput<T, float, BETA_T>(
            q, k, v, addresses.gk, beta, initialState, cuSeqlens,
            chunkIndices, addresses.qgScaled, aqk, addresses.vNew,
            addresses.h, attnOut, userWorkspace, tiling, pipe);
    }
}

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchStageSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR qgScaled, GM_ADDR uSeed, GM_ADDR userWorkspace,
    const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchStage<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, qgScaled, uSeed, userWorkspace, tiling);
    } else {
        DispatchStage<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, qgScaled, uSeed, userWorkspace, tiling);
    }
}

template <bool SAFE_GATE, typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGeneric(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    RunGeneric<SAFE_GATE, T, BETA_T, TilingData,
        COMPILE_BT, COMPILE_K, COMPILE_V>(
        q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
        chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg, kg,
        vNew, h, userWorkspace, tiling);
}

#if KDA_COMPILE_ARCH35_FAST_PATH
template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    AscendC::TPipe pipe;
    if (tiling.safeGate) {
        arch35::Run<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling, pipe);
    } else {
        arch35::Run<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling, pipe);
    }
}
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchArch35SafeGate(
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
    GM_ADDR,
    const TilingData &)
{
}
#endif

template <typename T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT, uint32_t COMPILE_K, uint32_t COMPILE_V>
__aicore__ inline void DispatchGenericSafeGate(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR attnOut,
    GM_ADDR finalState, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
    GM_ADDR userWorkspace, const TilingData &tiling)
{
    if (tiling.safeGate) {
        DispatchGeneric<true, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    } else {
        DispatchGeneric<false, T, BETA_T, TilingData,
            COMPILE_BT, COMPILE_K, COMPILE_V>(
            q, k, v, g, beta, aLog, dtBias, initialState, cuSeqlens,
            chunkIndices, attnOut, finalState, gk, aqk, akk, w, u, qg,
            kg, vNew, h, userWorkspace, tiling);
    }
}

} // namespace KdaForward

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR initial_state,
    GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR attn_out,
    GM_ADDR final_state, GM_ADDR gk, GM_ADDR aqk, GM_ADDR akk,
    GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR v_new, GM_ADDR h,
    GM_ADDR qg_scaled, GM_ADDR u_seed, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(ChunkKdaFwdTilingData, tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
            KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
                ChunkKdaFwdTilingData, 0, 0, 0>(
                q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
                chunk_indices, attn_out, final_state, gk, aqk, akk, w, u,
                qg, kg, v_new, h, qg_scaled, u_seed, userWorkspace,
                tilingData);
            return;
        }
        KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 0, 0, 0>(
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, tilingData);
    } else if (TILING_KEY_IS(2)) {
        if (tilingData.stage != KdaForward::KDA_STAGE_FULL) {
            KdaForward::DispatchStageSafeGate<DTYPE_Q, DTYPE_BETA,
                ChunkKdaFwdTilingData, 64, 128, 128>(
                q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
                chunk_indices, attn_out, final_state, gk, aqk, akk, w, u,
                qg, kg, v_new, h, qg_scaled, u_seed, userWorkspace,
                tilingData);
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        KdaForward::DispatchArch35SafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
#else
        KdaForward::DispatchGenericSafeGate<DTYPE_Q, DTYPE_BETA,
            ChunkKdaFwdTilingData, 64, 128, 128>(
#endif
            q, k, v, g, beta, a_log, dt_bias, initial_state, cu_seqlens,
            chunk_indices, attn_out, final_state, gk, aqk, akk, w, u, qg,
            kg, v_new, h, userWorkspace, tilingData);
    }
}

#undef KDA_COMPILE_ARCH35_FAST_PATH
