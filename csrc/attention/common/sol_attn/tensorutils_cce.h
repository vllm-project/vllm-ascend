// CCE intrinsic wrappers used by the Sol-Attn device kernels.
// The cube and vector compile units use the target-specific intrinsic forms
// selected by the Ascend compiler macros. GM and on-chip views carry the
// addresses supplied by each kernel.
#pragma once

#include <stdint.h>

// Inside a CANN custom-op build the CANN headers are on the include path: take their definitions
// of the entry-compatibility block (guards below yield to them) and of CacheLine / DcciDst. In a
// bare bisheng compile they are absent and this header provides the same declarations itself.
#if defined(__has_include)
#if __has_include("basic_api/kernel_common.h")
#include "basic_api/kernel_common.h"
#define ASCRIP_HAVE_CANN_HEADERS 1
#endif
#endif

#if defined(__CCE_AICORE__) && !defined(__DAV_C310__) && __CCE_AICORE__ != 220
#error "tensorutils_cce.h supports the a5/dav-c310 (-D__DAV_C310__) and a2/dav-c220 compile flows"
#endif

// ===========================================================================================
// Entry-point compatibility: what the CANN custom-op build's auto-generated wrapper expects the
// kernel source to provide (copied from CANN asc/impl/basic_api/utils/kernel_utils_macros.h,
// guards kept so the CANN originals win when both are visible).
// ===========================================================================================
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#include "scalar_math.h"
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

#ifndef ASCENDC_MODULE_UTILS_MACROS_H
#ifndef __PLUGIN__KERNEL_META_TYPE_ENUME_DEFINED__
#define __PLUGIN__KERNEL_META_TYPE_ENUME_DEFINED__
enum KernelMetaType : uint8_t {
    KERNEL_TYPE_AIV_ONLY,
    KERNEL_TYPE_AIC_ONLY,
    KERNEL_TYPE_MIX_AIV_1_0,
    KERNEL_TYPE_MIX_AIC_1_0,
    KERNEL_TYPE_MIX_AIC_1_1,
    KERNEL_TYPE_MIX_AIC_1_2,
    KERNEL_TYPE_AICORE,
    KERNEL_TYPE_VECTORCORE,
    KERNEL_TYPE_MIX_AICORE,
    KERNEL_TYPE_MIX_VECTOR_CORE,
    KERNEL_TYPE_MAX,
};
#endif
#ifndef ENABLE_FEATURE_FOR_COMPILE
#ifdef __CHECK_FEATURE_AT_PRECOMPILE
#define ENABLE_FEATURE_FOR_COMPILE(f, val) auto __enable_feature_for_compile_##f = val
#else
#define ENABLE_FEATURE_FOR_COMPILE(f, val)
#endif
#endif
#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(value) ENABLE_FEATURE_FOR_COMPILE(default, value)
#endif
#ifndef KERNEL_TASK_TYPE
#define KERNEL_TASK_TYPE(key, value) ENABLE_FEATURE_FOR_COMPILE(key, value)
#endif
enum KernelType {
    K_TYPE_AICORE = 1,
    K_TYPE_AIC = 2,
    K_TYPE_AIV = 3,
    K_TYPE_MIX_AIC_MAIN = 4,
    K_TYPE_MIX_AIV_MAIN = 5,
    K_TYPE_AIC_ROLLBACK = 6,
    K_TYPE_AIV_ROLLBACK = 7,
    K_TYPE_MAX
};
enum FuncMetaType {
    F_TYPE_KTYPE = 1,
    F_TYPE_CROSS_CORE_SYNC = 2,
    F_TYPE_MIX_TASK_RATION = 3,
    F_TYPE_L0_EXCEPTION_DFX = 4,
    F_TYPE_L0_EXCEPTION_DFX_ARGSINFO = 5,
    F_TYPE_L0_EXCEPTION_DFX_IS_TIK = 6,
    F_TYPE_DETERMINISTIC_INFO = 13,
    F_TYPE_FUNCTION_ENTRY_INFO = 14,
    F_TYPE_BLOCK_NUM_INFO = 15,
    F_TYPE_MAX
};
struct BaseTlv {
    unsigned short type;
    unsigned short len;
};
struct FunMetaKType {
    BaseTlv head;
    unsigned int ktype;
};
struct FunMetaCrossCoreType {
    BaseTlv head;
    unsigned int usedCrossCoreSync;
};
struct FunMetaMixCoreType {
    BaseTlv head;
    unsigned short taskRation0;
    unsigned short taskRation1;
};
struct FunLevelKType {
    struct FunMetaKType ktypeMeta;
};
struct FunLevelCrossCoreType {
    struct FunMetaKType ktypeMeta;
    struct FunMetaCrossCoreType crossCoreType;
};
struct FunLevelMixCoreType {
    struct FunMetaKType ktypeMeta;
    struct FunMetaMixCoreType mixCoreType;
};
#endif  // ASCENDC_MODULE_UTILS_MACROS_H

#ifndef IMPL_UTILS_SYS_CONSTANTS_H
namespace AscendC {
constexpr int32_t MIX = 0;
constexpr int32_t AIC = 1;
constexpr int32_t AIV = 2;
}  // namespace AscendC
#endif
#ifndef ASCEND_IS_AIV
// The SDK owns g_coreType; its include guards differ across CANN releases.
// Bare compiler invocations use a private fallback when the SDK macros are absent.
#if defined(__DAV_CUBE__)
constexpr int32_t ascrip_cce_core_type = AscendC::AIC;
#elif defined(__DAV_VEC__)
constexpr int32_t ascrip_cce_core_type = AscendC::AIV;
#else
constexpr int32_t ascrip_cce_core_type = AscendC::MIX;
#endif
#define ASCEND_IS_AIV constexpr(ascrip_cce_core_type == AscendC::AIV)
#define ASCEND_IS_AIC constexpr(ascrip_cce_core_type == AscendC::AIC)
#define ASCEND_IS_NOT_AIV constexpr(ascrip_cce_core_type != AscendC::AIV)
#define ASCEND_IS_NOT_AIC constexpr(ascrip_cce_core_type != AscendC::AIC)
#endif

// Workspace. The CANN custom-op build wraps the printed entry in an auto-generated kernel that
// calls AscendC::SetSysWorkspaceForce(workspace) and then passes the *user* region
// (workspace + AscendC::RESERVED_WORKSPACE, 16 MB on a5) as the entry's `workspace` argument —
// so a printed `mem.workspace` window is `workspace + offset`, nothing is added here. The two
// functions the wrapper needs live in CANN's kernel_operator_common_impl.h, which
// basic_api/kernel_common.h does not pull in: provide them (same bodies) when it is not visible.
#if defined(ASCRIP_HAVE_CANN_HEADERS) && !defined(ASCENDC_MODULE_OPERATOR_COMMON_IMPL_H)
namespace AscendC {
__aicore__ inline void SetSysWorkspaceForce(GM_ADDR workspace)
{
#if (WORKSPACE_PARAM_OFFSET == 0xffffffff)
#if defined(__NPU_DEVICE__)
    __set_kfc_workspace_addr(workspace);
#else
    g_sysWorkspaceReserved = workspace;
#endif
#endif
}

__aicore__ inline GM_ADDR GetUserWorkspace(GM_ADDR workspace)
{
    (void)(workspace);
    return GetSysWorkSpacePtr() + RESERVED_WORKSPACE;
}
}  // namespace AscendC
#endif

// The 910B custom-op wrapper prepends `matmul::clearWorkspace(workspace)` to every MIX kernel
// (tbe/tikcpp/compile_op.py, v220 path) — the AscendC matmul API's KFC message-area reset plus a
// handful of SPR defaults. The Sol-Attn kernels use no KFC server or flag 15, so the shim
// keeps only the guarantee our own stores depend on: a prior kernel's atomic mode must not leak
// into this launch's fixpipe / MTE3 stores.
#if defined(ASCRIP_HAVE_CANN_HEADERS) && defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
namespace matmul {
__aicore__ inline void clearWorkspace(__gm__ uint8_t* workspace)
{
    (void)workspace;
    set_atomic_none();
}
}  // namespace matmul
#endif

// CANN's CacheLine / DcciDst enums (kernel_reg.h) when reachable, a local copy otherwise; the
// dcci intrinsic takes their values as raw uint64, so both must agree (asserted below).
#if !defined(ASCRIP_HAVE_CANN_HEADERS)
namespace AscendC {
enum class CacheLine : uint64_t { SINGLE_CACHE_LINE = 0, ENTIRE_DATA_CACHE };
enum class DcciDst : uint64_t { CACHELINE_ALL = 0, CACHELINE_UB, CACHELINE_OUT, CACHELINE_ATOMIC };
}  // namespace AscendC
#endif
static_assert(static_cast<uint64_t>(AscendC::CacheLine::ENTIRE_DATA_CACHE) == 1, "CacheLine values must match CANN's kernel_reg.h");
static_assert(static_cast<uint64_t>(AscendC::DcciDst::CACHELINE_ATOMIC) == 3, "DcciDst values must match CANN's kernel_reg.h");

// Narrow-float spellings the printer uses (CANN's AscendC names over the compiler's own types).
// The c220 compiler has no fp8 / fp4 register types: a5-only.
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)
typedef float8_e4m3_t fp8_e4m3fn_t;
typedef float8_e5m2_t fp8_e5m2_t;
typedef float8_e8m0_t fp8_e8m0_t;
typedef float4_e2m1x2_t fp4x2_e2m1_t;
typedef float4_e1m2x2_t fp4x2_e1m2_t;
#endif

namespace ascrip {

// ===========================================================================================
// Scalar helpers and core queries.
// ===========================================================================================
__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b) { return b == 0 ? 0 : (a + b - 1) / b; }
__aicore__ inline int32_t AlignUp(int32_t a, int32_t n) { return (a + n - 1) / n * n; }
template <typename A, typename B>
__aicore__ inline A Min(A a, B b) { return (a < b) ? a : (A)b; }
template <typename A, typename B>
__aicore__ inline A Max(A a, B b) { return (a < b) ? (A)b : a; }

__aicore__ inline int32_t GetCubeNum() { return (int32_t)get_block_num(); }
__aicore__ inline int32_t GetCubeIdx() { return (int32_t)get_block_idx(); }
__aicore__ inline int32_t GetVecNum()
{
    if ASCEND_IS_AIV {
        return (int32_t)(get_block_num() * get_subblockdim());
    } else {
        return (int32_t)get_block_num();
    }
}
__aicore__ inline int32_t GetVecIdx()
{
    if ASCEND_IS_AIV {
        return (int32_t)(get_block_idx() * get_subblockdim() + get_subblockid());
    } else {
        return (int32_t)get_block_idx();
    }
}
__aicore__ inline int32_t GetSubBlockIdx()
{
    if ASCEND_IS_AIV {
        return (int32_t)get_subblockid();
    } else {
        return 0;
    }
}

template <typename A, typename B>
struct is_same { static constexpr bool value = false; };
template <typename A>
struct is_same<A, A> { static constexpr bool value = true; };
template <typename T>
struct is_integral {
    static constexpr bool value = is_same<T, int8_t>::value || is_same<T, uint8_t>::value || is_same<T, int16_t>::value ||
                                  is_same<T, uint16_t>::value || is_same<T, int32_t>::value || is_same<T, uint32_t>::value ||
                                  is_same<T, int64_t>::value || is_same<T, uint64_t>::value;
};
template <typename T>
struct is_signed_int {
    static constexpr bool value = is_same<T, int8_t>::value || is_same<T, int16_t>::value || is_same<T, int32_t>::value ||
                                  is_same<T, int64_t>::value;
};
template <typename T>
struct is_fp4 {
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)
    static constexpr bool value = is_same<T, fp4x2_e2m1_t>::value || is_same<T, fp4x2_e1m2_t>::value;
#else
    static constexpr bool value = false;  // the c220 compiler has no fp4 types
#endif
};

// ===========================================================================================
// Windows: GMTensor<T> (typed __gm__ pointer), Tensor<T, P> (absolute on-chip byte address plus the
// compile-time position), Buff<T, P, N> (the slot buffer of mem.alloc buf<...>). `w[i]` is the window i elements further on, `w.as<U>()` the
// same bytes as U, `w.ptr()` the typed pointer of the window's position. Element offsets count as
// the IR does (an fp4x2 element is one carrier byte).
// ===========================================================================================
// Position (the DSL's Position.L1): not `Pos`, which the compiler's vector intrinsics define at global scope
// (enum class Pos { LOWEST, HIGHEST }) and which would be ambiguous under `using namespace ascrip`.
enum class Position : int { UB, L1, L0A, L0B, L0C, BT };

template <typename T>
struct GMTensor {
    __gm__ T* p = nullptr;
    __aicore__ inline GMTensor() {}
    __aicore__ inline explicit GMTensor(__gm__ T* ptr) : p(ptr) {}
    __aicore__ inline __gm__ T* ptr() const { return p; }
    // the window `elems` elements further on (an fp4x2 element is one carrier byte)
    __aicore__ inline GMTensor<T> operator[](int64_t elems) const { return GMTensor<T>(p + elems); }
    template <typename U>
    __aicore__ inline GMTensor<U> as() const { return GMTensor<U>((__gm__ U*)p); }
    __aicore__ inline T load(int64_t i) const { return p[i]; }
    __aicore__ inline void store(int64_t i, T v) const { p[i] = v; }
};

// A gmlist parameter: AscendC's ListTensorDesc in its DYNAMIC layout, read with plain scalar loads.
// head[0] is the byte offset of the pointer array; from head[1] every member has a header whose low half is the rank
// (the high half is a framework word: 1 on the board and cannsim, not the index) followed by its rank dims; the
// pointer array holds one GM address per member. count() is what the layout implies — the pointer array offset
// divided by the per-member descriptor size (CANN's own consistency rule) — and so does not depend on that word.
template <typename T>
struct GMList {
    __gm__ uint64_t* head;
    __aicore__ inline GMList(GM_ADDR p) : head((__gm__ uint64_t*)p) {}
    __aicore__ inline uint32_t rank() const { return (uint32_t)(head[1] & 0xffffffffu); }
    __aicore__ inline int32_t count() const
    {
        uint32_t desc = rank() ? 1 + rank() : 2;
        return (int32_t)((head[0] - 8) / (desc * 8));
    }
    __aicore__ inline __gm__ T* ptr(int32_t i) const { return (__gm__ T*)head[head[0] / 8 + i]; }
    __aicore__ inline int64_t dim(int32_t i, int32_t d) const { return (int64_t)head[1 + i * (1 + rank()) + 1 + d]; }
};

template <typename T>
__aicore__ inline constexpr uint64_t elem_bytes_x2()
{
    // bytes of two elements: 1 for the packed 4-bit carriers, 2 * sizeof(T) otherwise
    return is_fp4<T>::value ? 2 : 2 * (uint64_t)sizeof(T);
}

template <typename T, Position P>
struct Tensor {
    uint64_t addr = 0;  // absolute byte address inside the position's memory
    __aicore__ inline Tensor() {}
    __aicore__ inline explicit Tensor(uint64_t a) : addr(a) {}
    // The typed pointer of the window's position. The address-space attributes cannot name a typedef,
    // so the type is deduced from the one branch the position keeps; the bias table has no address
    // space of its own and hands the intrinsics a number.
    __aicore__ inline auto ptr() const
    {
        if constexpr (P == Position::UB) {
            return (__ubuf__ T*)addr;
        } else if constexpr (P == Position::L1) {
            return (__cbuf__ T*)addr;
        } else if constexpr (P == Position::L0A) {
            return (__ca__ T*)addr;
        } else if constexpr (P == Position::L0B) {
            return (__cb__ T*)addr;
        } else if constexpr (P == Position::L0C) {
            return (__cc__ T*)addr;
        } else {
            return addr;
        }
    }
    // the window `elems` elements further on (an fp4x2 element is one carrier byte)
    __aicore__ inline Tensor<T, P> operator[](int64_t elems) const { return Tensor<T, P>(addr + (uint64_t)elems * sizeof(T)); }
    template <typename U>
    __aicore__ inline Tensor<U, P> as() const { return Tensor<U, P>(addr); }
    __aicore__ inline T load(int64_t i) const { static_assert(P == Position::UB, "scalar access is UB-only"); return *((__ubuf__ T*)addr + i); }
    __aicore__ inline void store(int64_t i, T v) const { static_assert(P == Position::UB, "scalar access is UB-only"); *((__ubuf__ T*)addr + i) = v; }
};

template <typename T, Position P, int N>
struct Buff {
    class Slots {
        static constexpr bool cached = N >= 2 && N <= 4;
        uint64_t addresses[cached ? N : 2];
        template <int I = 0>
        __aicore__ inline uint64_t select_cached(int64_t i) const
        {
            if constexpr (I + 1 < N) {
                return i == I ? addresses[I] : select_cached<I + 1>(i);
            } else {
                return addresses[I];
            }
        }
    public:
        __aicore__ inline Slots(uint64_t b, uint64_t pitch)
        {
            if constexpr (cached) {
                for (int i = 0; i < N; ++i) {
                    addresses[i] = b + (uint64_t)i * pitch;
                }
            } else {
                addresses[0] = b;
                addresses[1] = pitch;
            }
        }
        // Direct selection returns a value; only a proven in-range index reads the cache.
        __aicore__ inline Tensor<T, P> operator[](int64_t i) const
        {
            if constexpr (cached) {
                if (i >= 0 && i < N) {
                    if constexpr (N == 2) {
                        return Tensor<T, P>(select_cached(i));
                    } else {
                        return Tensor<T, P>(addresses[i]);
                    }
                }
                // Unsigned subtraction recovers the original pitch even after address wrap.
                return Tensor<T, P>(addresses[0] + (uint64_t)i * (addresses[1] - addresses[0]));
            } else {
                return Tensor<T, P>(addresses[0] + (uint64_t)i * addresses[1]);
            }
        }
    };
    Slots slot;
    __aicore__ inline Buff(uint64_t base, uint64_t slot_bytes) : slot(base, slot_bytes) {}
    template <typename Index>
    __aicore__ inline Tensor<T, P> get(Index i) const {
        auto index = i % N;
        if constexpr (std::is_signed<decltype(index)>::value) {
            return slot[index < 0 ? index + N : index];
        } else {
            return slot[index];
        }
    }
};
// ===========================================================================================
// Same-side events with static flag ids (events pass): depth = number of ids, the tokens rotate
// through them; PRESET tokens are set in the constructor and drained in the destructor, the old
// SEvent/DEvent/TEvent/QEvent protocol with any depth (those names are the depth-1..4 aliases).
// ===========================================================================================
template <pipe_t SET, pipe_t WAIT, int... IDS>
struct Flags {
    static constexpr int depth = sizeof...(IDS);
    template <int ID, int... REST>
    __aicore__ static inline void set_nth(int k)
    {
        if (k == 0) {
            set_flag(SET, WAIT, (event_t)ID);
        } else if constexpr (sizeof...(REST) > 0) {
            set_nth<REST...>(k - 1);
        }
    }
    template <int ID, int... REST>
    __aicore__ static inline void wait_nth(int k)
    {
        if (k == 0) {
            wait_flag(SET, WAIT, (event_t)ID);
        } else if constexpr (sizeof...(REST) > 0) {
            wait_nth<REST...>(k - 1);
        }
    }
    __aicore__ static inline void set(int k) { set_nth<IDS...>(k); }
    __aicore__ static inline void wait(int k) { wait_nth<IDS...>(k); }
};

template <pipe_t SET, pipe_t WAIT, int PRESET, int... IDS>
class Event {
public:
    static constexpr int depth = sizeof...(IDS);
    __aicore__ inline Event()
    {
        for (int i = 0; i < PRESET; ++i) {
            set();
        }
    }
    __aicore__ inline ~Event() { release(); }
    __aicore__ inline void set()
    {
        Flags<SET, WAIT, IDS...>::set(set_cnt % depth);
        set_cnt += 1;
    }
    __aicore__ inline void wait()
    {
        Flags<SET, WAIT, IDS...>::wait(wait_cnt % depth);
        wait_cnt += 1;
    }
    __aicore__ inline void set_all()
    {
        for (int i = 0; i < depth; ++i) {
            set();
        }
    }
    __aicore__ inline void release()
    {
        for (int i = wait_cnt; i < set_cnt; ++i) {
            wait();
        }
    }

private:
    int set_cnt = 0;
    int wait_cnt = 0;
};
// The old event names by depth (ids appended, as the events pass assigns them).
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1>
using SEvent = Event<SET, WAIT, PRESET, ID1>;
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1, int ID2>
using DEvent = Event<SET, WAIT, PRESET, ID1, ID2>;
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1, int ID2, int ID3>
using TEvent = Event<SET, WAIT, PRESET, ID1, ID2, ID3>;
template <pipe_t SET, pipe_t WAIT, int PRESET, int ID1, int ID2, int ID3, int ID4>
using QEvent = Event<SET, WAIT, PRESET, ID1, ID2, ID3, ID4>;

// Explicit flags and barriers (sync.set_flag / sync.wait_flag / sync.barrier).
template <pipe_t SET, pipe_t WAIT>
__aicore__ inline void SetFlag(int id) { set_flag(SET, WAIT, (event_t)id); }
template <pipe_t SET, pipe_t WAIT>
__aicore__ inline void WaitFlag(int id) { wait_flag(SET, WAIT, (event_t)id); }
template <pipe_t P>
__aicore__ inline void PipeBarrier() { pipe_barrier(P); }

// ===========================================================================================
// Cross-core synchronisation (sync.crosscore.*). a5/C310: cube <-> vector point-to-point rides
// mode 0x4 (intra-block); the AIC side sets / waits BOTH flag N and N+16 because AIV1's N is
// remapped to N+16, the AIV side handles the single N. The ALL* / INTRACORE groups stay on the
// FFTS path (mode 0x0 / 0x1); message packing from dav_3510 kernel_operator_sync_impl.h.
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // cross-core: the c310 point-to-point modes; the c220 ffts forms land with the mutex pass
template <int MODE, pipe_t P>
__aicore__ inline void CrossCoreSetFlag(uint16_t flag_id)
{
    if constexpr (MODE == 0x4) {
        set_intra_block(P, flag_id);
    } else {
        ffts_cross_core_sync(P, 0x1ull | ((uint64_t)(MODE & 0x3) << 4) | ((uint64_t)(flag_id & 0xf) << 8));
    }
}
template <int MODE, pipe_t P>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flag_id)
{
    if constexpr (MODE == 0x4) {
        wait_intra_block(P, flag_id);
    } else {
        wait_flag_dev(P, flag_id);
    }
}
template <pipe_t P>
__aicore__ inline void CUBE_READY(int id)
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<0x4, P>((uint16_t)id);
        CrossCoreSetFlag<0x4, P>((uint16_t)(id + 16));
    }
}
template <pipe_t P>
__aicore__ inline void WAIT_VEC(int id)
{
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<0x4, P>((uint16_t)id);
        CrossCoreWaitFlag<0x4, P>((uint16_t)(id + 16));
    }
}
template <pipe_t P>
__aicore__ inline void VEC_READY(int id)
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<0x4, P>((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void WAIT_CUBE(int id)
{
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<0x4, P>((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void ALLCUBE_READY(int id) { CrossCoreSetFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLCUBE_WAIT(int id) { CrossCoreWaitFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLVEC_READY(int id) { CrossCoreSetFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLVEC_WAIT(int id) { CrossCoreWaitFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void INTRACORE_ALLVEC_READY(int id) { CrossCoreSetFlag<0x1, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void INTRACORE_ALLVEC_WAIT(int id) { CrossCoreWaitFlag<0x1, P>((uint16_t)id); }

#else  // c220: pair sync rides FFTS mode 0x2, the all-core groups 0x0 / 0x1; every wait blocks
// the scalar pipe (wait_flag_dev, no pipe / mode operand). Flags 11-14 are AscendC SyncAll's.
template <int MODE, pipe_t P>
__aicore__ inline void CrossCoreSetFlag(uint16_t flag_id)
{
    ffts_cross_core_sync(P, 0x1ull | ((uint64_t)(MODE & 0x3) << 4) | ((uint64_t)(flag_id & 0xf) << 8));
}
template <pipe_t P>
__aicore__ inline void CUBE_READY(int id)
{
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<0x2, P>((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void WAIT_VEC(int id)
{
    if ASCEND_IS_AIC {
        wait_flag_dev((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void VEC_READY(int id)
{
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<0x2, P>((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void WAIT_CUBE(int id)
{
    if ASCEND_IS_AIV {
        wait_flag_dev((uint16_t)id);
    }
}
template <pipe_t P>
__aicore__ inline void ALLCUBE_READY(int id) { CrossCoreSetFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLCUBE_WAIT(int id) { wait_flag_dev((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLVEC_READY(int id) { CrossCoreSetFlag<0x0, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void ALLVEC_WAIT(int id) { wait_flag_dev((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void INTRACORE_ALLVEC_READY(int id) { CrossCoreSetFlag<0x1, P>((uint16_t)id); }
template <pipe_t P>
__aicore__ inline void INTRACORE_ALLVEC_WAIT(int id) { wait_flag_dev((uint16_t)id); }
#endif  // cross-core
// ===========================================================================================
// Atomic GM stores (atomic.*): dav_3510 picks the dtype SPR, then the op SPR.
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // dav_3510: add also applies to AIC fixpipe stores
template <typename T>
__aicore__ inline void atomic_dtype()
{
    if constexpr (is_same<T, float>::value) {
        set_atomic_f32();
    } else if constexpr (is_same<T, half>::value) {
        set_atomic_f16();
    } else if constexpr (is_same<T, bfloat16_t>::value) {
        set_atomic_bf16();
    } else if constexpr (is_same<T, int16_t>::value) {
        set_atomic_s16();
    } else if constexpr (is_same<T, int32_t>::value) {
        set_atomic_s32();
    } else if constexpr (is_same<T, int8_t>::value) {
        set_atomic_s8();
    } else {
        static_assert(is_same<T, float>::value, "unsupported atomic dtype");
    }
}
template <typename T>
__aicore__ inline void SetAtomicAdd()
{
    atomic_dtype<T>();
    set_atomic_add();
}
template <typename T>
__aicore__ inline void SetAtomicMax()
{
    if ASCEND_IS_AIV {
        set_atomic_max();
        atomic_dtype<T>();
    }
}
template <typename T>
__aicore__ inline void SetAtomicMin()
{
    if ASCEND_IS_AIV {
        set_atomic_min();
        atomic_dtype<T>();
    }
}
template <typename T>
__aicore__ inline void SetAtomicType()
{
    atomic_dtype<T>();
}
__aicore__ inline void SetAtomicNone()
{
    set_atomic_none();
}

// atomic.begin {op}: the operation SPR alone (atomic.set_type sets the dtype).
__aicore__ inline void SetAtomicOpAdd()
{
    set_atomic_add();
}
__aicore__ inline void SetAtomicOpMax()
{
    if ASCEND_IS_AIV {
        set_atomic_max();
    }
}
__aicore__ inline void SetAtomicOpMin()
{
    if ASCEND_IS_AIV {
        set_atomic_min();
    }
}

#elif __CCE_AICORE__ == 220
// c220: the raw SPR spellings are identical to dav_3510's, but the units compile separately
// (dav-c220-vec / dav-c220-cube), so no ASCEND_IS_AIV guard: the SPR applies on whichever core
// issues the store (AIV for UB->GM, AIC for fixpipe stores). Order follows the CANN c220 impl
// (kernel_operator_set_atomic_impl.h): the op SPR first, then the dtype SPR.
template <typename T>
__aicore__ inline void atomic_dtype()
{
    if constexpr (is_same<T, float>::value) {
        set_atomic_f32();
    } else if constexpr (is_same<T, half>::value) {
        set_atomic_f16();
    } else if constexpr (is_same<T, bfloat16_t>::value) {
        set_atomic_bf16();
    } else if constexpr (is_same<T, int16_t>::value) {
        set_atomic_s16();
    } else if constexpr (is_same<T, int32_t>::value) {
        set_atomic_s32();
    } else if constexpr (is_same<T, int8_t>::value) {
        set_atomic_s8();
    } else {
        static_assert(is_same<T, float>::value, "unsupported atomic dtype");
    }
}
template <typename T>
__aicore__ inline void SetAtomicAdd()
{
    set_atomic_add();
    atomic_dtype<T>();
}
template <typename T>
__aicore__ inline void SetAtomicMax()
{
    set_atomic_max();
    atomic_dtype<T>();
}
template <typename T>
__aicore__ inline void SetAtomicMin()
{
    set_atomic_min();
    atomic_dtype<T>();
}
template <typename T>
__aicore__ inline void SetAtomicType()
{
    atomic_dtype<T>();
}
__aicore__ inline void SetAtomicNone()
{
    set_atomic_none();
}

// atomic.begin {op}: the operation SPR alone (atomic.set_type sets the dtype).
__aicore__ inline void SetAtomicOpAdd()
{
    set_atomic_add();
}
__aicore__ inline void SetAtomicOpMax()
{
    set_atomic_max();
}
__aicore__ inline void SetAtomicOpMin()
{
    set_atomic_min();
}
#endif  // atomics
// ===========================================================================================
// Vector-mask SPR (vec.set_mask & co.), HF32 mode, data-cache maintenance.
// ===========================================================================================
__aicore__ inline void SetVectorMask(uint64_t high, uint64_t low)
{
    if ASCEND_IS_NOT_AIC {
        set_vector_mask(high, low);
    }
}
__aicore__ inline void ResetMask()
{
    if ASCEND_IS_NOT_AIC {
        set_vector_mask(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull);
    }
}
// Counter mode of the mask SPR: the c220 backend's D-062 bracket uses it around every counted op.
// c310 never reaches this -- the printer refuses vec.set_mask_count there, because c310 has no
// operation the mode would govern (D-217) -- so the wrapper needs no arch guard of its own. It used
// to carry a static_assert template on the belief that the c310 compiler declares no usable builtin;
// it does declare one, and that was never the reason.
__aicore__ inline void SetMaskCount()
{
    if ASCEND_IS_NOT_AIC {
        set_mask_count();
    }
}
__aicore__ inline void SetMaskNorm()
{
    if ASCEND_IS_NOT_AIC {
        set_mask_norm();
    }
}
// The counter-mode count (the old one-argument SetVectorMask).
__aicore__ inline void SetVectorMask(int32_t count)
{
    if ASCEND_IS_NOT_AIC {
        set_vector_mask(0, (uint64_t)count);
    }
}
// The old SetVectorMaskByCount ladder: a <= 128-lane mask (b16 granularity) from a count.
__aicore__ inline void SetVectorMaskByCount(int32_t count)
{
    uint64_t high = 0;
    uint64_t low = 0;
    if (count <= 0) {
        high = 0;
        low = 0;
    } else if (count < 64) {
        low = (1ULL << count) - 1;
    } else if (count == 64) {
        low = (uint64_t)-1;
    } else if (count < 128) {
        low = (uint64_t)-1;
        high = (1ULL << (count - 64)) - 1;
    } else {
        low = (uint64_t)-1;
        high = (uint64_t)-1;
    }
    SetVectorMask(high, low);
}


// Raw CTRL bits: float=48, float8=50, int=53, cast=59, global=60.
// global=0 selects each cast's RS operand; global=1 selects cast (0 saturates, 1 truncates).
__aicore__ inline void SetSatFlag(int32_t bit, bool on)
{
    if (on) {
        set_ctrl(sbitset1(get_ctrl(), bit));
    } else {
        set_ctrl(sbitset0(get_ctrl(), bit));
    }
}
__aicore__ inline int32_t GetSatFlag(int32_t bit) { return (int32_t)((get_ctrl() >> bit) & 1ULL); }

__aicore__ inline void SetHF32Mode(bool on)
{
    constexpr int32_t HF32_MODE_BIT = 46;
    if (on) {
        set_ctrl(sbitset1(get_ctrl(), HF32_MODE_BIT));
    } else {
        set_ctrl(sbitset0(get_ctrl(), HF32_MODE_BIT));
    }
}
__aicore__ inline void SetHF32TransMode(bool nearest_away)
{
    constexpr int32_t HF32_TRANS_MODE_BIT = 47;
    if (nearest_away) {
        set_ctrl(sbitset1(get_ctrl(), HF32_TRANS_MODE_BIT));
    } else {
        set_ctrl(sbitset0(get_ctrl(), HF32_TRANS_MODE_BIT));
    }
}
// The A5 compiler takes dcci's cache line and destination only as integer constant expressions;
// a function parameter fails the build (I037). They are template arguments, as the pipes of the
// flag helpers are, and the macro keeps the printed DataCacheCleanAndInvalid(window, line, dst).
template <uint64_t ENTIRE, uint64_t DCCI_DST, typename T>
__aicore__ inline void DataCacheCleanAndInvalidAt(GMTensor<T> dst)
{
    dcci((__gm__ void*)dst.ptr(), ENTIRE, DCCI_DST);
}
#define DataCacheCleanAndInvalid(dst, entire, dcci_dst) DataCacheCleanAndInvalidAt<(entire), (dcci_dst)>(dst)

// ===========================================================================================
// UB <-> GM / UB / L1 DMA (the vector core's MTE2 / MTE3 / V copies). Byte-unit DataCopyPad
// semantics on the align_v2 intrinsics, stride310 formulas from dav_3510
// kernel_operator_data_copy_impl.h:  GM side = stride_bytes + burst_bytes (u64);
// UB side = align32(stride_blocks * 32 + burst_bytes) (u32).
// ===========================================================================================
// `is_pad` fills the destination's 32-byte alignment tail with `pad_bits` instead of leaving it
// as the transfer found it. The SPR and the order are AscendC's own (dav_3510 / dav_c220
// DataCopyPadGm2UBImpl: `if (padParams.isPad) set_mov_pad_val(GetScalarBitcodeValue(...))` before
// the copy), and `pad_bits` is the value's BIT PATTERN -- the printer bitcasts it, so a float pad
// arrives as its IEEE encoding. c310 also takes an `isPad` flag on the intrinsic; c220 has no such
// argument and the SPR alone governs.
template <typename T>
__aicore__ inline void gm_to_ub_pad(Tensor<T, Position::UB> dst, GMTensor<T> src, int32_t n_burst, int32_t burst_len_byte,
                                    int32_t src_stride_byte, int32_t dst_stride, bool is_pad = false, uint64_t pad_bits = 0, uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIV {
        if (is_pad) {
            set_mov_pad_val(pad_bits);
        }
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)
        uint32_t burst = (uint32_t)burst_len_byte;
        uint64_t src310 = (uint64_t)(uint32_t)src_stride_byte + burst;
        uint32_t dst310 = ((uint32_t)dst_stride * 32u + burst + 31u) & ~31u;
        if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
            copy_gm_to_ubuf_align_v2((__ubuf__ uint32_t*)dst.ptr(), (__gm__ uint32_t*)src.ptr(), 0, (uint16_t)n_burst, burst, 0, 0, is_pad, l2_cache_mode,
                                     src310, dst310);
        } else if constexpr (sizeof(T) == 2) {
            copy_gm_to_ubuf_align_v2((__ubuf__ uint16_t*)dst.ptr(), (__gm__ uint16_t*)src.ptr(), 0, (uint16_t)n_burst, burst, 0, 0, is_pad, l2_cache_mode,
                                     src310, dst310);
        } else {
            copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)dst.ptr(), (__gm__ uint8_t*)src.ptr(), 0, (uint16_t)n_burst, burst, 0, 0, is_pad, l2_cache_mode,
                                     src310, dst310);
        }
#else
        // c220 (dav_c220 DataCopyPadGm2UBImpl): burst / src gap in bytes, dst gap in 32-B blocks;
        // 8-byte types ride the b32 form (the paddings, always zero here, would double).
        if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
            copy_gm_to_ubuf_align_b32((__ubuf__ uint32_t*)dst.ptr(), (__gm__ uint32_t*)src.ptr(), 0, (uint16_t)n_burst,
                                      (uint32_t)burst_len_byte, (uint8_t)0, (uint8_t)0, (uint32_t)src_stride_byte, (uint32_t)dst_stride);
        } else if constexpr (sizeof(T) == 2) {
            copy_gm_to_ubuf_align_b16((__ubuf__ uint16_t*)dst.ptr(), (__gm__ uint16_t*)src.ptr(), 0, (uint16_t)n_burst,
                                      (uint32_t)burst_len_byte, (uint8_t)0, (uint8_t)0, (uint32_t)src_stride_byte, (uint32_t)dst_stride);
        } else {
            copy_gm_to_ubuf_align_b8((__ubuf__ uint8_t*)dst.ptr(), (__gm__ uint8_t*)src.ptr(), 0, (uint16_t)n_burst,
                                     (uint32_t)burst_len_byte, (uint8_t)0, (uint8_t)0, (uint32_t)src_stride_byte, (uint32_t)dst_stride);
        }
#endif
    }
}

template <typename T>
__aicore__ inline void ub_to_gm_pad(GMTensor<T> dst, Tensor<T, Position::UB> src, int32_t n_burst, int32_t burst_len_byte,
                                    int32_t src_stride, int32_t dst_stride_byte, uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIV {
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)
        uint32_t burst = (uint32_t)burst_len_byte;
        uint64_t dst310 = (uint64_t)(uint32_t)dst_stride_byte + burst;
        uint32_t src310 = ((uint32_t)src_stride * 32u + burst + 31u) & ~31u;
        copy_ubuf_to_gm_align_v2((__gm__ void*)dst.ptr(), (__ubuf__ void*)src.ptr(), 0, (uint16_t)n_burst, burst, l2_cache_mode, dst310, src310);
#else
        // c220 (DataCopyPadUB2GMImpl): burst / dst gap in bytes, src gap in 32-B blocks.
        if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
            copy_ubuf_to_gm_align_b32((__gm__ uint32_t*)dst.ptr(), (__ubuf__ uint32_t*)src.ptr(), 0, (uint16_t)n_burst,
                                      (uint32_t)burst_len_byte, (uint8_t)0, (uint8_t)0, (uint32_t)src_stride, (uint32_t)dst_stride_byte);
        } else if constexpr (sizeof(T) == 2) {
            copy_ubuf_to_gm_align_b16((__gm__ uint16_t*)dst.ptr(), (__ubuf__ uint16_t*)src.ptr(), 0, (uint16_t)n_burst,
                                      (uint32_t)burst_len_byte, (uint8_t)0, (uint8_t)0, (uint32_t)src_stride, (uint32_t)dst_stride_byte);
        } else {
            copy_ubuf_to_gm_align_b8((__gm__ uint8_t*)dst.ptr(), (__ubuf__ uint8_t*)src.ptr(), 0, (uint16_t)n_burst,
                                     (uint32_t)burst_len_byte, (uint8_t)0, (uint8_t)0, (uint32_t)src_stride, (uint32_t)dst_stride_byte);
        }
#endif
    }
}

// Classic 32-byte-block forms: burst_len and both strides in 32-byte blocks.
template <typename T>
__aicore__ inline void ub_to_ub(Tensor<T, Position::UB> dst, Tensor<T, Position::UB> src, int32_t n_burst, int32_t burst_len, int32_t src_stride,
                                int32_t dst_stride)
{
    if ASCEND_IS_AIV {
        copy_ubuf_to_ubuf((__ubuf__ void*)dst.ptr(), (__ubuf__ void*)src.ptr(), 0, (uint16_t)n_burst, (uint16_t)burst_len, (uint16_t)src_stride,
                          (uint16_t)dst_stride);
    }
}

template <typename T>
__aicore__ inline void ub_to_l1(Tensor<T, Position::L1> dst, Tensor<T, Position::UB> src, int32_t n_burst, int32_t burst_len, int32_t src_stride,
                                int32_t dst_stride)
{
    if ASCEND_IS_AIV {
        copy_ubuf_to_cbuf((__cbuf__ void*)dst.ptr(), (__ubuf__ void*)src.ptr(), 0, (uint16_t)n_burst, (uint16_t)burst_len, (uint16_t)src_stride,
                          (uint16_t)dst_stride);
    }
}

// UB ND [m_src, N_src-strided] -> L1 NZ tile of m_dst rows: one datablock per burst, one call per
// fractal column. dst / src are the tile origins (the printer folds dst_row0 / dst_col0 in).
template <typename T>
__aicore__ inline void ub_to_l1_nd2nz(Tensor<T, Position::L1> dst, Tensor<T, Position::UB> src, int32_t m_src, int32_t n_src, int32_t m_dst,
                                      int32_t n_dst, int32_t N_src)
{
    if ASCEND_IS_AIV {
        (void)n_dst;
        const int32_t C0 = 32 / (int32_t)sizeof(T);
        uint16_t block_count = (uint16_t)m_src;
        uint16_t block_len = 1;
        uint16_t src_stride = (uint16_t)((N_src + C0 - 1) / C0 - 1);
        uint16_t dst_stride = 0;
        int32_t loops = (n_src + C0 - 1) / C0;
        int32_t dst_frac_stride = C0 * ((m_dst + 15) / 16 * 16);
        for (int32_t i = 0; i < loops; ++i) {
            copy_ubuf_to_cbuf((__cbuf__ void*)dst[i * dst_frac_stride].ptr(), (__ubuf__ void*)src[i * C0].ptr(), 0, block_count, block_len,
                              src_stride, dst_stride);
        }
    }
}

// UB NZ fractals -> L1 NZ tile: block_len / strides in NZ rows (one datablock each); M_src is the
// source tile's fractal-column height, m_src the rows moved, m_dst the destination tile's rows.
template <typename T>
__aicore__ inline void ub_to_l1_nz(Tensor<T, Position::L1> dst, Tensor<T, Position::UB> src, int32_t m_src, int32_t n_src, int32_t m_dst,
                                   int32_t n_dst, int32_t M_src)
{
    if ASCEND_IS_AIV {
        (void)n_dst;
        const int32_t C0 = 32 / (int32_t)sizeof(T);
        uint16_t block_count = (uint16_t)((n_src + C0 - 1) / C0);
        uint16_t block_len = (uint16_t)m_src;
        uint16_t src_stride = (uint16_t)(M_src - m_src);
        uint16_t dst_stride = (uint16_t)((m_dst + 15) / 16 * 16 - m_src);
        copy_ubuf_to_cbuf((__cbuf__ void*)dst.ptr(), (__ubuf__ void*)src.ptr(), 0, block_count, block_len, src_stride, dst_stride);
    }
}

// ===========================================================================================
// GM -> L1 (the cube core's MTE2). 32-byte-block and byte-unit forms on the align_v2 intrinsic
// (the L1 flavour passes isPad = true unconditionally, as dav_3510 does); ND2NZ / DN2NZ ride the
// MTE2_NZ_PARA SPR ([15:0] ND or DN count, [31:16] dstNzNStride, [47:32] dstNzC0Stride,
// [63:48] dstNzMatrixStride * sizeof(T) / 32) and byte-unit source strides.
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)
template <typename T>
__aicore__ inline void gm_to_l1_raw(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t n_burst, uint32_t burst_bytes, uint64_t src310, uint32_t dst310,
                                   uint8_t l2_cache_mode = 0)
{
    if constexpr (sizeof(T) == 8 || sizeof(T) == 4) {
        copy_gm_to_cbuf_align_v2((__cbuf__ uint32_t*)dst.ptr(), (__gm__ uint32_t*)src.ptr(), 0, (uint16_t)n_burst, burst_bytes, 0, 0, true, l2_cache_mode,
                                 src310, dst310);
    } else if constexpr (sizeof(T) == 2) {
        copy_gm_to_cbuf_align_v2((__cbuf__ uint16_t*)dst.ptr(), (__gm__ uint16_t*)src.ptr(), 0, (uint16_t)n_burst, burst_bytes, 0, 0, true, l2_cache_mode,
                                 src310, dst310);
    } else {
        copy_gm_to_cbuf_align_v2((__cbuf__ uint8_t*)dst.ptr(), (__gm__ uint8_t*)src.ptr(), 0, (uint16_t)n_burst, burst_bytes, 0, 0, true, l2_cache_mode,
                                 src310, dst310);
    }
}

template <typename T>
__aicore__ inline void gm_to_l1(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t n_burst, int32_t burst_len, int32_t src_stride, int32_t dst_stride,
                               uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        uint32_t burst = (uint32_t)burst_len * 32u;
        uint64_t src310 = (uint64_t)(uint32_t)src_stride * 32u + burst;
        uint32_t dst310 = ((uint32_t)dst_stride * 32u + burst + 31u) & ~31u;
        gm_to_l1_raw<T>(dst, src, n_burst, burst, src310, dst310, l2_cache_mode);
    }
}
#else
// c220: the classic 32-B block copy (dav_c220 DataCopyGM2L1Impl -> copy_gm_to_cbuf, pad mode off).
template <typename T>
__aicore__ inline void gm_to_l1(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t n_burst, int32_t burst_len, int32_t src_stride, int32_t dst_stride)
{
    if ASCEND_IS_AIC {
        copy_gm_to_cbuf((__cbuf__ void*)dst.ptr(), (__gm__ void*)src.ptr(), (int8_t)0, (uint16_t)n_burst, (uint16_t)burst_len,
                        (uint16_t)src_stride, (uint16_t)dst_stride, (pad_t)0);
    }
}
#endif

#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // byte-unit / ND2NZ / DN2NZ / MX forms: c310; the c220 nd2nz lands with the cube pass
template <typename T>
__aicore__ inline void gm_to_l1_pad(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t n_burst, int32_t burst_len_byte, int32_t src_stride_byte,
                                    int32_t dst_stride, uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        uint32_t burst = (uint32_t)burst_len_byte;
        uint64_t src310 = (uint64_t)(uint32_t)src_stride_byte + burst;
        uint32_t dst310 = ((uint32_t)dst_stride * 32u + burst + 31u) & ~31u;
        gm_to_l1_raw<T>(dst, src, n_burst, burst, src310, dst310, l2_cache_mode);
    }
}

template <typename T>
__aicore__ inline void gm_to_l1_nd2nz_raw(__cbuf__ T* dst, __gm__ T* src, uint16_t nd_num, uint16_t n_value, uint16_t d_value,
                                          int32_t src_nd_matrix_stride, int32_t src_d_value, uint16_t dst_nz_c0_stride,
                                          uint16_t dst_nz_n_stride, int32_t dst_nz_matrix_stride, uint8_t l2_cache_mode = 0)
{
    uint64_t para = (uint64_t)(uint16_t)(dst_nz_matrix_stride * (int32_t)sizeof(T) / 32) << 48;
    para |= (uint64_t)dst_nz_c0_stride << 32;
    para |= (uint64_t)dst_nz_n_stride << 16;
    para |= (uint64_t)nd_num;
    set_mte2_nz_para(para);
    uint64_t loop1_src_stride = (uint64_t)src_d_value * sizeof(T);
    uint64_t loop4_src_stride = (uint64_t)src_nd_matrix_stride * sizeof(T);
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ int8_t*)dst, (__gm__ int8_t*)src, 0, loop1_src_stride, l2_cache_mode, n_value, d_value, loop4_src_stride, false);
    } else if constexpr (sizeof(T) == 2) {
        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ half*)dst, (__gm__ half*)src, 0, loop1_src_stride, l2_cache_mode, n_value, d_value, loop4_src_stride, false);
    } else {
        copy_gm_to_cbuf_multi_nd2nz((__cbuf__ float*)dst, (__gm__ float*)src, 0, loop1_src_stride, l2_cache_mode, n_value, d_value, loop4_src_stride, false);
    }
}

// GM ND window [M, N] with row stride N_src -> L1 NZ tile whose fractal columns hold M_dst rows.
template <typename T>
__aicore__ inline void gm_to_l1_nd2nz(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t M, int32_t N, int32_t M_dst, int32_t N_src,
                                     uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        gm_to_l1_nd2nz_raw<T>(dst.ptr(), src.ptr(), 1, (uint16_t)M, (uint16_t)N, 0, N_src, (uint16_t)((M_dst + 15) / 16 * 16), 1, 0, l2_cache_mode);
    }
}

// GM holds the transposed matrix: GM [N, M] with row stride N_src -> L1 NZ [M, N].
template <typename T>
__aicore__ inline void gm_to_l1_dn2nz(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t M, int32_t N, int32_t M_dst, int32_t N_src,
                                     uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        uint64_t para = (uint64_t)0 << 48;
        para |= (uint64_t)(uint16_t)((M_dst + 15) / 16 * 16) << 32;
        para |= (uint64_t)1 << 16;
        para |= (uint64_t)1;
        set_mte2_nz_para(para);
        uint64_t loop1_src_stride = (uint64_t)N_src * sizeof(T);
        copy_gm_to_cbuf_multi_dn2nz(dst.ptr(), src.ptr(), 0, loop1_src_stride, l2_cache_mode, (uint16_t)M, (uint16_t)N, 0, false);
    }
}

// Logical e8m0 scale [rows, k_groups] (row-major, src_k_groups per row) -> the packed stream the
// L0 MX scale planes expect (two e8m0 bytes ride one half lane).
template <typename T>
__aicore__ inline void gm_to_l1_mx_scale_nd2nz(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t rows, int32_t k_groups, int32_t src_k_groups,
                                              uint8_t l2_cache_mode = 0)
{
    static_assert(sizeof(T) == 1, "gm_to_l1_mx_scale_nd2nz expects a 1-byte scale dtype");
    if ASCEND_IS_AIC {
        int32_t dst_half_cols = CeilDiv(k_groups, 2);
        int32_t src_half_cols = CeilDiv(src_k_groups, 2);
        uint64_t para = (uint64_t)0 << 48;
        para |= (uint64_t)(uint16_t)dst_half_cols << 32;
        para |= (uint64_t)1 << 16;
        para |= (uint64_t)1;
        set_mte2_nz_para(para);
        uint64_t loop1_src_stride = (uint64_t)src_half_cols * sizeof(half);
        copy_gm_to_cbuf_multi_dn2nz((__cbuf__ half*)dst.ptr(), (__gm__ half*)src.ptr(), 0, loop1_src_stride, l2_cache_mode, (uint16_t)dst_half_cols,
                                    (uint16_t)rows, 0, false);
    }
}

#else  // c220
// GM ND window [M, N] with row stride N_src -> L1 tile: b8 / b16 / b32 integers are the native
// ND2NZ; **float** feeds the c220 cube's ZZ layout instead — 16-row bands, each its own ND matrix
// (the old GM2L1_ND2ZZ, field-proven; srcNdMatrixStride is 16 bits, hence the N_src >= 4096 loop).
//
// The test is the dtype, not its width. It used to be `sizeof(T) == 4`, which sent int32 down the
// ZZ route as well: an int32 tile was then written as ZZ (align16(N) * M * 4 bytes, so a narrow
// tile spilled past its logical end) and read back as ZZ by `l1_to_l0`, while `addr_alloc` and the
// model both had it as NZ. The old framework tests `dst.dtype is Datatype.float` for the same
// reason. int4 carriers are int32 tiles, which is where this surfaced.
template <typename T>
__aicore__ inline void gm_to_l1_nd2nz(Tensor<T, Position::L1> dst, GMTensor<T> src, int32_t M, int32_t N, int32_t M_dst, int32_t N_src)
{
    if ASCEND_IS_AIC {
        if constexpr (is_same<T, float>::value) {
            const int32_t wa = (N + 15) / 16 * 16;
            if (N_src < 4096) {
                copy_gm_to_cbuf_multi_nd2nz_b32s(dst.ptr(), src.ptr(), 0, (uint16_t)(M / 16), (uint16_t)16, (uint16_t)N,
                                                 (uint16_t)(N_src * 16), (uint16_t)N_src, (uint16_t)16, (uint16_t)1,
                                                 (uint16_t)(wa * 16));
            } else {
                for (int32_t i = 0; i < M / 16; ++i) {
                    copy_gm_to_cbuf_multi_nd2nz_b32s(dst[i * 16 * wa].ptr(), src[i * 16 * N_src].ptr(), 0, (uint16_t)1, (uint16_t)16,
                                                     (uint16_t)N, (uint16_t)0, (uint16_t)N_src, (uint16_t)16, (uint16_t)1, (uint16_t)0);
                }
            }
            if (M % 16 != 0) {
                const int32_t tail = M / 16 * 16;
                copy_gm_to_cbuf_multi_nd2nz_b32s(dst[tail * wa].ptr(), src[tail * N_src].ptr(), 0, (uint16_t)1, (uint16_t)(M % 16),
                                                 (uint16_t)N, (uint16_t)0, (uint16_t)N_src, (uint16_t)16, (uint16_t)1, (uint16_t)0);
            }
        } else if constexpr (sizeof(T) == 4) {
            // A 32-bit integer tile is NZ like every other non-float dtype: one ND matrix, the
            // ordinary dstNzC0Stride. Only the intrinsic's width suffix follows sizeof(T).
            copy_gm_to_cbuf_multi_nd2nz_b32s(dst.ptr(), src.ptr(), 0, (uint16_t)1, (uint16_t)M, (uint16_t)N, (uint16_t)0,
                                             (uint16_t)N_src, (uint16_t)((M_dst + 15) / 16 * 16), (uint16_t)1, (uint16_t)0);
        } else if constexpr (sizeof(T) == 2) {
            copy_gm_to_cbuf_multi_nd2nz_b16(dst.ptr(), src.ptr(), 0, (uint16_t)1, (uint16_t)M, (uint16_t)N, (uint16_t)0,
                                            (uint16_t)N_src, (uint16_t)((M_dst + 15) / 16 * 16), (uint16_t)1, (uint16_t)0);
        } else {
            copy_gm_to_cbuf_multi_nd2nz_b8(dst.ptr(), src.ptr(), 0, (uint16_t)1, (uint16_t)M, (uint16_t)N, (uint16_t)0,
                                           (uint16_t)N_src, (uint16_t)((M_dst + 15) / 16 * 16), (uint16_t)1, (uint16_t)0);
        }
    }
}
#endif  // gm->l1 arch forms
// L1 constant fill through the MTE2 matrix-init intrinsic. The repeat word is
// blockNum << 16 | dstGap << 32 | repeatTimes, and the value lane and the pointer type are routed
// by dtype exactly as InitL1BufferCal does on BOTH arches (dav_3510 and dav_c220's
// kernel_operator_mm_impl.h are byte-identical here), so this routing is shared (the intrinsic
// exists for the bf16 / half / uint32_t pointer forms only; narrower dtypes replicate their bits
// into a half). `byte_addr` is absolute, as Tensor::addr is.
template <typename T>
__aicore__ inline void _create_cbuf_matrix(uint64_t byte_addr, T val, int64_t repeat_bit)
{
    if constexpr (is_same<T, bfloat16_t>::value) {
        create_cbuf_matrix_bf16((__cbuf__ bfloat16_t*)byte_addr, repeat_bit, val);
    } else if constexpr (is_same<T, half>::value) {
        create_cbuf_matrix((__cbuf__ half*)byte_addr, repeat_bit, val);
    } else if constexpr (is_same<T, uint32_t>::value) {
        create_cbuf_matrix((__cbuf__ uint32_t*)byte_addr, repeat_bit, val);
    } else if constexpr (sizeof(T) == 2) {
        half hv;
        __builtin_memcpy(&hv, &val, 2);
        create_cbuf_matrix((__cbuf__ half*)byte_addr, repeat_bit, hv);
    } else if constexpr (sizeof(T) == 4) {
        uint32_t raw;
        __builtin_memcpy(&raw, &val, 4);
        create_cbuf_matrix((__cbuf__ uint32_t*)byte_addr, repeat_bit, raw);
    } else if constexpr (sizeof(T) == 1) {
        uint8_t b;
        __builtin_memcpy(&b, &val, 1);
        uint16_t raw16 = (uint16_t)b | ((uint16_t)b << 8);
        half hv;
        __builtin_memcpy(&hv, &raw16, 2);
        create_cbuf_matrix((__cbuf__ half*)byte_addr, repeat_bit, hv);
    } else {
        static_assert(sizeof(T) == 0, "create_cbuf_matrix: unsupported dtype");
    }
}

// n_blocks 32-byte blocks of val, as one repeat.
template <typename T>
__aicore__ inline void set_constant_to_l1(Tensor<T, Position::L1> dst, T val, int32_t n_blocks)
{
    if ASCEND_IS_AIC {
        _create_cbuf_matrix<T>(dst.addr, val, ((uint64_t)(uint16_t)n_blocks << 16) | ((uint64_t)0 << 32) | (uint64_t)1);
    }
}

// ===========================================================================================
// Cube-side loads (MTE1): L1 NZ -> L0A / L0B, the MX variants, img2col, and the bias table.
// The 2-D load's transpose flag must be a compile-time constant ("Bool Const Expr"), so it is a
// template parameter.
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // cube loads: c310 forms (and MX); the c220 load_cbuf_to_ca/cb land with the cube pass
template <bool TRANS, typename T, Position P>
__aicore__ inline void l0_load_2d(Tensor<T, P> dst, Tensor<T, Position::L1> src, int32_t m_start, int32_t k_start, int32_t m_step, int32_t k_step,
                                  int32_t src_stride, int32_t dst_stride)
{
    static_assert(P == Position::L0A || P == Position::L0B, "L0 load destination must be L0A or L0B");
    if ASCEND_IS_AIC {
        if constexpr (P == Position::L0A) {
            load_cbuf_to_ca(dst.ptr(), src.ptr(), m_start, k_start, m_step, k_step, src_stride, dst_stride, TRANS);
        } else {
            load_cbuf_to_cb(dst.ptr(), src.ptr(), m_start, k_start, m_step, k_step, src_stride, dst_stride, TRANS);
        }
    }
}

// dma.l1_to_l0: the [m_dst, n_dst] window of an L1 NZ tile of m_src rows -> L0 NZ (TRANS = the
// transposing load, dst fractal-column height n_dst). src is the window origin (the printer
// folds src_row0 / src_col0 into it).
template <bool TRANS, typename T, Position P>
__aicore__ inline void l1_to_l0(Tensor<T, P> dst, Tensor<T, Position::L1> src, int32_t m_src, int32_t n_src, int32_t m_dst, int32_t n_dst)
{
    (void)n_src;
    int32_t C0 = 32 / (int32_t)sizeof(T);
    if constexpr (TRANS) {
        l0_load_2d<true>(dst, src, 0, 0, (m_dst + 15) / 16, (n_dst + C0 - 1) / C0, (m_src + 15) / 16, (n_dst + 15) / 16);
    } else {
        l0_load_2d<false>(dst, src, 0, 0, (m_dst + 15) / 16, (n_dst + C0 - 1) / C0, (m_src + 15) / 16, (m_dst + 15) / 16);
    }
}

// MX loads: the data fractal through the plain 2-D load (fp4 via the _s4 form), then the e8m0
// scale plane through the *_mx intrinsic at dataAddr / 16 (the fixed L0 <-> L0MX mapping).
template <bool TRANS, typename T, typename TMx, Position P>
__aicore__ inline void l0_load_mx(Tensor<T, P> dst, Tensor<T, Position::L1> src, Tensor<TMx, Position::L1> src_mx, int32_t m_step, int32_t k_step,
                                  int32_t src_stride, int32_t dst_stride, int32_t mx_x_step, int32_t mx_y_step, int32_t mx_src_stride,
                                  int32_t mx_dst_stride)
{
    static_assert(P == Position::L0A || P == Position::L0B, "MX L0 load destination must be L0A or L0B");
    if ASCEND_IS_AIC {
        if constexpr (P == Position::L0A) {
            if constexpr (is_fp4<T>::value) {
                load_cbuf_to_ca_s4(dst.ptr(), src.ptr(), 0, 0, m_step, k_step, src_stride, dst_stride, TRANS);
            } else {
                load_cbuf_to_ca(dst.ptr(), src.ptr(), 0, 0, m_step, k_step, src_stride, dst_stride, TRANS);
            }
            load_cbuf_to_ca_mx(dst.addr / 16, (__cbuf__ void*)((__cbuf__ fp8_e8m0_t*)src_mx.addr), 0, 0, mx_x_step, mx_y_step, mx_src_stride,
                               mx_dst_stride);
        } else {
            if constexpr (is_fp4<T>::value) {
                load_cbuf_to_cb_s4(dst.ptr(), src.ptr(), 0, 0, m_step, k_step, src_stride, dst_stride, TRANS);
            } else {
                load_cbuf_to_cb(dst.ptr(), src.ptr(), 0, 0, m_step, k_step, src_stride, dst_stride, TRANS);
            }
            load_cbuf_to_cb_mx(dst.addr / 16, (__cbuf__ void*)((__cbuf__ fp8_e8m0_t*)src_mx.addr), 0, 0, mx_x_step, mx_y_step, mx_src_stride,
                               mx_dst_stride);
        }
    }
}

// dma.l1_to_l0.mx: derived steps; the scale plane packs two C0 groups per e8m0 column.
template <bool TRANS, typename T, typename TMx, Position P>
__aicore__ inline void l1_to_l0_mx(Tensor<T, P> dst, Tensor<T, Position::L1> src, Tensor<TMx, Position::L1> src_mx, int32_t m_src, int32_t n_src,
                                   int32_t m_dst, int32_t n_dst)
{
    int32_t C0 = 32 / (int32_t)sizeof(T);
    int32_t m_step = CeilDiv(m_dst, 16);
    int32_t k_step = CeilDiv(n_dst, C0);
    int32_t data_src_stride = CeilDiv(m_src, 16);
    if constexpr (TRANS) {
        int32_t data_dst_stride = CeilDiv(n_dst, 16);
        int32_t mx_x_step = CeilDiv(n_dst, 16);
        int32_t mx_k_step = CeilDiv(m_dst, C0 * 2);
        int32_t mx_src_stride = CeilDiv(m_src, C0 * 2);
        l0_load_mx<true>(dst, src, src_mx, m_step, k_step, data_src_stride, data_dst_stride, mx_x_step, mx_k_step, mx_src_stride, mx_k_step);
    } else {
        int32_t data_dst_stride = CeilDiv(m_dst, 16);
        int32_t mx_k_step = CeilDiv(n_dst, C0 * 2);
        int32_t mx_src_stride = CeilDiv(n_src, C0 * 2);
        l0_load_mx<false>(dst, src, src_mx, m_step, k_step, data_src_stride, data_dst_stride, m_step, mx_k_step, mx_src_stride, mx_k_step);
    }
}

// dma.l1_to_l0.img2col: L1 NC1HWC0 feature map -> L0A im2col window. FMATRIX (h / w / pads),
// PADDING (0) and the load3d repeat SPR describe the same M pitch for each K fractal.
template <typename T>
__aicore__ inline void l1_to_l0_img2col(Tensor<T, Position::L0A> dst, Tensor<T, Position::L1> src, int32_t h, int32_t w, int32_t c, int32_t kh, int32_t kw,
                                        int32_t pad_l, int32_t pad_r, int32_t pad_t, int32_t pad_b, int32_t stride_h, int32_t stride_w,
                                        int32_t dil_h, int32_t dil_w, int32_t k0, int32_t m0, int32_t k_ext, int32_t m_ext)
{
    if ASCEND_IS_AIC {
        uint64_t fmatrix = 0;
        fmatrix |= (uint64_t)(w & 0xFFFF);
        fmatrix |= (uint64_t)(h & 0xFFFF) << 16;
        fmatrix |= (uint64_t)(pad_l & 0xFF) << 32;
        fmatrix |= (uint64_t)(pad_r & 0xFF) << 40;
        fmatrix |= (uint64_t)(pad_t & 0xFF) << 48;
        fmatrix |= (uint64_t)(pad_b & 0xFF) << 56;
        set_fmatrix(fmatrix);
        set_padding(0);
        constexpr int32_t C0 = 32 / (int32_t)sizeof(T);
        uint64_t rpt = ((uint64_t)1 << 16) | ((uint64_t)((m_ext + 15) / 16) << 32);
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            // The native b16 window can advance along both fractal axes in one instruction.
            if (m_ext > 0 && k_ext > C0 &&
                ((m0 | m_ext) & 15) == 0 && ((k0 | k_ext) & (C0 - 1)) == 0 &&
                (uint32_t)m0 <= 0xffff && (uint32_t)k0 <= 0xffff &&
                m_ext <= 0xffff && k_ext <= 0xffff && k0 + k_ext <= 0x10000) {
                set_l3d_rpt(rpt);
                img2colv2_cbuf_to_ca(dst.ptr(), src.ptr(), (uint16_t)k_ext, (uint16_t)m_ext,
                                     (uint16_t)k0, (uint16_t)m0, (uint8_t)stride_w, (uint8_t)stride_h, (uint8_t)kw, (uint8_t)kh,
                                     (uint8_t)dil_w, (uint8_t)dil_h, false, false, false, false, (uint16_t)c);
                return;
            }
        }
        int32_t dst_k_frac_stride = ((m_ext + 15) / 16 * 16) * C0;
        int32_t n_chunk = (k_ext + C0 - 1) / C0;
        for (int32_t i = 0; i < n_chunk; i++) {
            set_l3d_rpt(rpt);
            img2colv2_cbuf_to_ca((__ca__ T*)(dst.addr + (uint64_t)(i * dst_k_frac_stride) * sizeof(T)), src.ptr(), (uint16_t)C0, (uint16_t)m_ext,
                                 (uint16_t)(k0 + i * C0), (uint16_t)m0, (uint8_t)stride_w, (uint8_t)stride_h, (uint8_t)kw, (uint8_t)kh,
                                 (uint8_t)dil_w, (uint8_t)dil_h, false, false, false, false, (uint16_t)c);
        }
    }
}

// dma.l1_to_bt: n bias values, L1 -> the C2 bias table. Same-width entries occupy two 16-bit C2
// slots (even burst count), narrow sources one; convControl = half -> float only.
template <typename TDst, typename TSrc>
__aicore__ inline void l1_to_bt(Tensor<TDst, Position::BT> dst, Tensor<TSrc, Position::L1> src, int32_t n)
{
    static_assert(is_same<TDst, TSrc>::value || (is_same<TSrc, bfloat16_t>::value && is_same<TDst, float>::value) ||
                      (is_same<TSrc, half>::value && is_same<TDst, float>::value),
                  "l1_to_bt: dav-3510 supports same-dtype, bf16->float and half->float only");
    if ASCEND_IS_AIC {
        constexpr int32_t one_data_len = (sizeof(TDst) == sizeof(TSrc)) ? 2 : 1;
        uint16_t len_burst = (uint16_t)(((n * one_data_len * 2) + 31) / 32);
        if (sizeof(TDst) == sizeof(TSrc)) {
            len_burst = (uint16_t)(((len_burst + 1) / 2) * 2);
        }
        constexpr bool conv_control = is_same<TSrc, half>::value && is_same<TDst, float>::value;
        copy_cbuf_to_bt(dst.addr, src.ptr(), conv_control, (uint16_t)1, len_burst, (uint16_t)0, (uint16_t)0);
    }
}

#endif  // cube loads
// ===========================================================================================
// cube.mmad: C = (init ? 0 : C) + A @ B, M rounded up to 16 (whole fractal rows); the bias form
// carries the BT address in Xd[63:32] next to the L0C address (dav_3510 MmadCal) with
// cmatrixSource = 1 and init forced off.
// ===========================================================================================
template <typename TC, typename TA, typename TB>
__aicore__ inline void mmad(Tensor<TC, Position::L0C> dst, Tensor<TA, Position::L0A> a, Tensor<TB, Position::L0B> b, int32_t M, int32_t N, int32_t K, bool init, uint8_t unit_flag = 0)
{
    if ASCEND_IS_AIC {
        mad(dst.ptr(), a.ptr(), b.ptr(), (uint16_t)((M + 15) / 16 * 16), (uint16_t)K, (uint16_t)N, unit_flag, false, false, init);
    }
}
#if __CCE_AICORE__ == 220  // signed-int4 mmad: a dedicated c220 intrinsic (void* operands, K in logical int4 elements)
#if !defined(ASCRIP_HAVE_CANN_HEADERS)  // the CANN kernel_operator headers define int4b_t themselves
struct int4b_t {};  // marker element type: the data lives packed in the int32 carriers; only mad_s4 reads it as s4
#endif
template <typename TC>
__aicore__ inline void mmad(Tensor<TC, Position::L0C> dst, Tensor<int4b_t, Position::L0A> a, Tensor<int4b_t, Position::L0B> b, int32_t M, int32_t N, int32_t K,
                            bool init, uint8_t unit_flag = 0)
{
    static_assert(is_same<TC, int32_t>::value, "an int4 mmad accumulates in int32");
    if ASCEND_IS_AIC {
        mad_s4(dst.ptr(), (__ca__ void*)a.ptr(), (__cb__ void*)b.ptr(), (uint16_t)((M + 15) / 16 * 16), (uint16_t)K, (uint16_t)N, unit_flag, false, false,
               init);
    }
}
#endif  // c220 int4 mmad
template <typename TC, typename TA, typename TB, typename TBias>
__aicore__ inline void mmad_bias(Tensor<TC, Position::L0C> dst, Tensor<TA, Position::L0A> a, Tensor<TB, Position::L0B> b, Tensor<TBias, Position::BT> bias,
                                 int32_t M, int32_t N, int32_t K, uint8_t unit_flag = 0)
{
    if ASCEND_IS_AIC {
        uint64_t xd = (dst.addr & 0xffffffffULL) | ((bias.addr & 0xffffffffULL) << 32);
        mad((__cc__ TC*)xd, a.ptr(), b.ptr(), (uint16_t)((M + 15) / 16 * 16), (uint16_t)K, (uint16_t)N, unit_flag, false, true, false);
    }
}
#if __CCE_AICORE__ == 220  // signed-int4 mmad_bias: `mad_s4` takes the bias-table address itself
template <typename TC, typename TBias>
__aicore__ inline void mmad_bias(Tensor<TC, Position::L0C> dst, Tensor<int4b_t, Position::L0A> a, Tensor<int4b_t, Position::L0B> b,
                                 Tensor<TBias, Position::BT> bias, int32_t M, int32_t N, int32_t K, uint8_t unit_flag = 0)
{
    static_assert(is_same<TC, int32_t>::value, "an int4 mmad accumulates in int32");
    if ASCEND_IS_AIC {
        // The generic form packs dst and the BT address into one pointer because `mad` takes
        // one; the c220 `mad_s4` builtin takes the BT address as its own fourth argument, after
        // both operands. The trailing controls match the generic: C is read from the bias table
        // (ctrlMatrixC) and not initialised from L0C (initMatrixC).
        mad_s4(dst.ptr(), (__ca__ void*)a.ptr(), (__cb__ void*)b.ptr(), (uint64_t)(bias.addr & 0xffffffffULL),
               (uint16_t)((M + 15) / 16 * 16), (uint16_t)K, (uint16_t)N, unit_flag, false, true, false);
    }
}
#endif  // c220 int4 mmad_bias
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // the MX forms: c310 only (mad_mx)
template <typename TC, typename TA, typename TB>
__aicore__ inline void mmad_mx(Tensor<TC, Position::L0C> dst, Tensor<TA, Position::L0A> a, Tensor<TB, Position::L0B> b, int32_t M, int32_t N, int32_t K,
                               bool init)
{
    static_assert(is_same<TC, float>::value, "mmad_mx requires a float L0C destination");
    if ASCEND_IS_AIC {
        mad_mx(dst.ptr(), a.ptr(), b.ptr(), (uint16_t)((M + 15) / 16 * 16), (uint16_t)K, (uint16_t)N, (uint8_t)0, false, false, init);
    }
}
template <typename TC, typename TA, typename TB, typename TBias>
__aicore__ inline void mmad_mx_bias(Tensor<TC, Position::L0C> dst, Tensor<TA, Position::L0A> a, Tensor<TB, Position::L0B> b, Tensor<TBias, Position::BT> bias,
                                    int32_t M, int32_t N, int32_t K)
{
    static_assert(is_same<TC, float>::value, "mmad_mx_bias requires a float L0C destination");
    if ASCEND_IS_AIC {
        uint64_t xd = (dst.addr & 0xffffffffULL) | ((bias.addr & 0xffffffffULL) << 32);
        mad_mx((__cc__ TC*)xd, a.ptr(), b.ptr(), (uint16_t)((M + 15) / 16 * 16), (uint16_t)K, (uint16_t)N, (uint8_t)0, false, true, false);
    }
}

#endif  // mmad
// ===========================================================================================
// L0C stores (fixpipe, FIX). QuantMode_t and its enumerators come from the compiler's
// cce_aicore_intrinsics.h. Scalar quant: deqScalar = [31:13] fp32 scale top bits, [45:37] int9
// offset, [46] signed saturation; the mode comes from the (dst, L0C) dtype pair; LOOP3 / CHANNEL
// SPRs carry the ND / DN layouts; a FIX barrier precedes every copy_matrix (dav_3510 order).
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // fixpipe: c310 QuantMode / forms; c220 copy_matrix_cc_to_gm lands with the cube pass
__aicore__ inline uint64_t pack_deq_scalar(float scale, int32_t offset)
{
    union {
        float f;
        uint32_t u;
    } cvt;
    cvt.f = scale;
    uint64_t deq = (uint64_t)(cvt.u & 0xFFFFE000u);
    deq |= ((uint64_t)((uint32_t)offset & 0x1FFu)) << 37;
    return deq;
}

__aicore__ inline bool is_scalar_quant(QuantMode_t q)
{
    return (q == DEQF16 || q == QF322B8_PRE || q == REQ8 || q == QS322BF16_PRE || q == QF322F16_PRE || q == QF322BF16_PRE ||
            q == QF322FP8_PRE || q == QF322HIF8_PRE || q == QF322HIF8_PRE_HYBRID || q == QF322F32_PRE);
}

template <typename T, typename T2>
__aicore__ inline void fixpipe_quant(QuantMode_t& q, uint64_t& deq, float scale, int32_t offset, bool scaled, bool hif8_hybrid)
{
    deq = 0;
    if constexpr (is_same<T, float>::value && is_same<T2, float>::value) {
        if (scaled) {
            q = QF322F32_PRE;
            deq = pack_deq_scalar(scale, 0);
        } else {
            q = NoQuant;
        }
    } else if constexpr (is_same<T, half>::value && is_same<T2, float>::value) {
        if (scaled) {
            q = QF322F16_PRE;
            deq = pack_deq_scalar(scale, 0);
        } else {
            q = F322F16;
        }
    } else if constexpr (is_same<T, bfloat16_t>::value && is_same<T2, float>::value) {
        if (scaled) {
            q = QF322BF16_PRE;
            deq = pack_deq_scalar(scale, 0);
        } else {
            q = F322BF16;
        }
    } else if constexpr (is_same<T, bfloat16_t>::value && is_integral<T2>::value) {
        q = QS322BF16_PRE;
        deq = pack_deq_scalar(scale, 0);
    } else if constexpr (is_same<T, half>::value && is_integral<T2>::value) {
        q = DEQF16;
        deq = pack_deq_scalar(scale, 0);
    } else if constexpr (sizeof(T) == 1 && is_integral<T>::value && is_same<T2, float>::value) {
        q = QF322B8_PRE;
        deq = pack_deq_scalar(scale, offset);
        if constexpr (is_signed_int<T>::value) {
            deq |= (1ULL << 46);
        }
    } else if constexpr (sizeof(T) == 1 && is_integral<T>::value && is_integral<T2>::value) {
        q = REQ8;
        deq = pack_deq_scalar(scale, offset);
        if constexpr (is_signed_int<T>::value) {
            deq |= (1ULL << 46);
        }
    } else if constexpr (is_same<T, fp8_e4m3fn_t>::value && is_same<T2, float>::value) {
        q = QF322FP8_PRE;
        deq = pack_deq_scalar(scale, 0);
    } else if constexpr (is_same<T, hifloat8_t>::value && is_same<T2, float>::value) {
        q = hif8_hybrid ? QF322HIF8_PRE_HYBRID : QF322HIF8_PRE;
        deq = pack_deq_scalar(scale, 0);
    } else {
        q = NoQuant;
    }
}

// dma.l0c_to_gm.nz2nd: L0C [M, N] (fractal-column height M_src) -> GM rows of stride N_dst.
template <typename T, typename T2>
__aicore__ inline void l0c_to_gm_nz2nd(GMTensor<T> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t N_dst, int32_t M_src, bool relu,
                                       float scale, int32_t offset, bool scaled, bool hif8_hybrid, uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        QuantMode_t q;
        uint64_t deq;
        fixpipe_quant<T, T2>(q, deq, scale, offset, scaled, hif8_hybrid);
        set_loop3_para((uint64_t)1);
        if (is_scalar_quant(q)) {
            set_quant_pre(deq);
        }
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)N_dst, (uint16_t)((M_src + 15) / 16 * 16), l2_cache_mode, 0, 0,
                             (uint64_t)q, (uint8_t)relu, false, true, (uint64_t)NoConv, 0, false, false, 0, false, false, false, false, false,
                             false);
    }
}

constexpr int32_t fixpipe_quant_table_columns = 256;

__aicore__ inline void load_fixpipe_quant_table(__cbuf__ uint64_t* table, uint16_t width)
{
    if ASCEND_IS_AIC {
        // Wait for the preceding FIX store to release FBUF before loading the next panel.
        pipe_barrier(PIPE_FIX);
        copy_cbuf_to_fbuf((__fbuf__ uint64_t*)0, table, 1, (uint16_t)(width / 8), 0, 0);
        set_fpc(0);
        pipe_barrier(PIPE_FIX);
    }
}

__aicore__ inline void l0c_to_gm_nz2nd_quant_table(GMTensor<half> dst, Tensor<float, Position::L0C> src,
                                                 Tensor<uint64_t, Position::L1> quant_table,
                                                 int32_t M, int32_t N, int32_t N_dst, int32_t M_src,
                                                 uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        if (M == 0) {
            return;
        }
        const uint16_t pitch = (uint16_t)((M_src + 15) / 16 * 16);
        set_loop3_para((uint64_t)1);
        for (int32_t column = 0; column < N; column += fixpipe_quant_table_columns) {
            const uint16_t width = (uint16_t)Min(N - column, fixpipe_quant_table_columns);
            load_fixpipe_quant_table(quant_table.ptr() + column, width);
            copy_matrix_cc_to_gm(dst.ptr() + column, src.ptr() + (int64_t)column * pitch,
                                 0, width, (uint16_t)M, (uint32_t)N_dst, pitch, l2_cache_mode, 0, 0,
                                 (uint64_t)VQF322F16_PRE, 0, false, true, (uint64_t)NoConv, 0, false,
                                 false, 0, false, false, false, false, false, false);
        }
    }
}

__aicore__ inline void l0c_to_l1_quant_table(Tensor<half, Position::L1> dst, Tensor<float, Position::L0C> src,
                                            Tensor<uint64_t, Position::L1> quant_table,
                                            int32_t M, int32_t N, int32_t M_dst, int32_t M_src)
{
    if ASCEND_IS_AIC {
        if (M == 0) {
            return;
        }
        const uint16_t src_pitch = (uint16_t)((M_src + 15) / 16 * 16);
        const uint32_t dst_pitch = (uint32_t)((M_dst + 15) / 16 * 16);
        for (int32_t column = 0; column < N; column += fixpipe_quant_table_columns) {
            const uint16_t width = (uint16_t)Min(N - column, fixpipe_quant_table_columns);
            load_fixpipe_quant_table(quant_table.ptr() + column, width);
            copy_matrix_cc_to_cbuf(dst.ptr() + (int64_t)column * dst_pitch, src.ptr() + (int64_t)column * src_pitch,
                                   0, width, (uint16_t)M, dst_pitch * 16, src_pitch, 0, 0, 0,
                                   (uint64_t)VQF322F16_PRE, 0, false, false, (uint64_t)NoConv, 0, false,
                                   false, 0, false, false, false, false, false, false);
        }
    }
}

__aicore__ inline void l0c_to_ub_quant_table(Tensor<half, Position::UB> dst, Tensor<float, Position::L0C> src,
                                            Tensor<uint64_t, Position::L1> quant_table,
                                            int32_t M, int32_t N, int32_t N_dst, int32_t M_src, bool sub_block_id)
{
    if ASCEND_IS_AIC {
        if (M == 0) {
            return;
        }
        const uint16_t pitch = (uint16_t)((M_src + 15) / 16 * 16);
        set_loop3_para((uint64_t)1);
        for (int32_t column = 0; column < N; column += fixpipe_quant_table_columns) {
            const uint16_t width = (uint16_t)Min(N - column, fixpipe_quant_table_columns);
            load_fixpipe_quant_table(quant_table.ptr() + column, width);
            copy_matrix_cc_to_ub(dst.ptr() + column, src.ptr() + (int64_t)column * pitch,
                                 0, width, (uint16_t)M, (uint32_t)N_dst, pitch, 0, sub_block_id, 0, 0,
                                 (uint64_t)VQF322F16_PRE, 0, false, true, (uint64_t)NoConv, 0, false,
                                 false, 0, false, false, false, false, false, false);
        }
    }
}

// dma.l0c_to_gm.nz2nz: the GM stays NZ, strips of M_pad rows (dstStride in elements on C310). A column block holds
// C0 = 32 elements for a 1-byte destination and 16 otherwise, as the IR and PTO's doubled stride place them (I039).
template <typename T, typename T2>
__aicore__ inline void l0c_to_gm_nz2nz(GMTensor<T> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t M_pad, int32_t M_src, bool relu,
                                       float scale, int32_t offset, bool scaled, bool hif8_hybrid, uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        QuantMode_t q;
        uint64_t deq;
        fixpipe_quant<T, T2>(q, deq, scale, offset, scaled, hif8_hybrid);
        if (is_scalar_quant(q)) {
            set_quant_pre(deq);
        }
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)(M_pad * (sizeof(T) == 1 ? 32 : 16)), (uint16_t)((M_src + 15) / 16 * 16), l2_cache_mode, 0,
                             0, (uint64_t)q, (uint8_t)relu, false, false, (uint64_t)NoConv, 0, false, false, 0, false, false, false, false,
                             false, false);
    }
}

// dma.l0c_to_gm.nz2dn: the transposed [N, M] plane of row stride M_dst (Nz2DnParams {1, 0, 0, 1}).
template <typename T, typename T2>
__aicore__ inline void l0c_to_gm_nz2dn(GMTensor<T> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t M_dst, int32_t M_src, bool relu,
                                       float scale, int32_t offset, bool scaled, bool hif8_hybrid, uint8_t unit_flag = 0, uint8_t l2_cache_mode = 0)
{
    if ASCEND_IS_AIC {
        QuantMode_t q;
        uint64_t deq;
        fixpipe_quant<T, T2>(q, deq, scale, offset, scaled, hif8_hybrid);
        set_loop3_para((uint64_t)1);
        set_channel_para((uint64_t)1 << 48);
        if (is_scalar_quant(q)) {
            set_quant_pre(deq);
        }
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)M_dst, (uint16_t)((M_src + 15) / 16 * 16), l2_cache_mode, 0, unit_flag,
                             (uint64_t)q, (uint8_t)relu, false, false, (uint64_t)NoConv, 0, false, false, 0, false, false, false, false, false,
                             true);
    }
}

// dma.l0c_to_l1: fp32 -> fp32 takes the channel-split leg (dstStride in C0 = 8 rows), every other
// pair a plain cast with dstStride in 16-element rows; the instruction also receives relu.
template <typename T, typename T2>
__aicore__ inline void l0c_to_l1(Tensor<T, Position::L1> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t M_dst, int32_t M_src, bool relu)
{
    if ASCEND_IS_AIC {
        if constexpr (is_same<T, float>::value && is_same<T2, float>::value) {
            copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)((M_dst + 15) / 16 * 16 * 8),
                                   (uint16_t)((M_src + 15) / 16 * 16), 0, 0, 0, NoQuant, (uint8_t)relu, true, false, 0, 0, false, false, 0, false,
                                   false, false, false, false, false);
        } else {
            QuantMode_t q;
            if constexpr (is_same<T, half>::value && is_same<T2, float>::value) {
                q = F322F16;
            } else if constexpr (is_same<T, bfloat16_t>::value && is_same<T2, float>::value) {
                q = F322BF16;
            } else {
                q = NoQuant;
            }
            copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)((M_dst + 15) / 16 * 16 * 16),
                                   (uint16_t)((M_src + 15) / 16 * 16), 0, 0, 0, q, (uint8_t)relu, false, false, 0, 0, false, false, 0, false,
                                   false, false, false, false, false);
        }
    }
}

// dma.l0c_to_ub: L0C [M, N] -> vector UB rows of stride N_dst; dual_mode 0 = SINGLE (one sub-block,
// sub_block_id), 1 = SPLITM, 2 = SPLITN.
template <typename T, typename T2>
__aicore__ inline void l0c_to_ub(Tensor<T, Position::UB> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t N_dst, int32_t M_src,
                                 int32_t dual_mode, bool sub_block_id, bool relu, float scale, int32_t offset, bool scaled, bool hif8_hybrid)
{
    if ASCEND_IS_AIC {
        QuantMode_t q;
        uint64_t deq;
        fixpipe_quant<T, T2>(q, deq, scale, offset, scaled, hif8_hybrid);
        set_loop3_para((uint64_t)1);
        if (is_scalar_quant(q)) {
            set_quant_pre(deq);
        }
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_ub((__ubuf__ T*)dst.addr, src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)N_dst, (uint16_t)((M_src + 15) / 16 * 16),
                             (uint8_t)dual_mode, sub_block_id, 0, 0, (uint64_t)q, (uint8_t)relu, false, true, (uint64_t)NoConv, 0, false, false,
                             0, false, false, false, false, false, false);
    }
}

#endif  // fixpipe

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
// ===========================================================================================
// c220 cube path. L1 -> L0 realises the four fractal layouts the c220 mmad wants:
// L0A takes ZZ (or NN transposed), L0B takes NZ (or ZN transposed); fp32 arrives in L1 as ZZ
// (gm_to_l1_nd2nz's float route), everything else as NZ. The loop math is the old framework's
// field-proven tensorutils (L0NZ2* / L0ZZ2* / LOADL0), spelled on the raw dav-c220 intrinsics.
// Fixpipe: set_nd_para / set_quant_pre + the 12-argument copy_matrix_cc_to_gm.
// ===========================================================================================
template <typename T>
struct LoadCarrier {
    using type = T;
};
template <>
struct LoadCarrier<int16_t> {
    using type = half;
};
template <>
struct LoadCarrier<uint16_t> {
    using type = half;
};

template <bool TP, typename T, Position P>
__aicore__ inline void l0_2d(Tensor<T, P> dst, Tensor<T, Position::L1> src, int32_t repeat, int32_t src_stride)
{
    // int16 / uint16 ride the half overload (CANN's own Conditional); the argument types and the
    // literal transpose flag mirror dav_c220 kernel_operator_mm_impl.h exactly (the builtin is
    // polymorphic and resolves by the argument pattern).
    using U = typename LoadCarrier<T>::type;
    const uint16_t si = 0;
    const uint8_t rep = (uint8_t)repeat;
    const uint16_t ss = (uint16_t)src_stride;
    const uint16_t dg = 0;
    const uint8_t sid = 0;
    if constexpr (P == Position::L0A) {
        if constexpr (TP) {
            load_cbuf_to_ca((__ca__ U*)dst.ptr(), (__cbuf__ U*)src.ptr(), si, rep, ss, dg, sid, 1, inc);
        } else {
            load_cbuf_to_ca((__ca__ U*)dst.ptr(), (__cbuf__ U*)src.ptr(), si, rep, ss, dg, sid, 0, inc);
        }
    } else {
        if constexpr (TP) {
            load_cbuf_to_cb((__cb__ U*)dst.ptr(), (__cbuf__ U*)src.ptr(), si, rep, ss, dg, sid, 1, inc);
        } else {
            load_cbuf_to_cb((__cb__ U*)dst.ptr(), (__cbuf__ U*)src.ptr(), si, rep, ss, dg, sid, 0, inc);
        }
    }
}

template <typename T, Position P>
__aicore__ inline void l0_2d_transpose(Tensor<T, P> dst, Tensor<T, Position::L1> src, int32_t repeat, int32_t src_stride, int32_t dst_gap,
                                       int32_t dst_frac_gap)
{
    const uint16_t si = 0;
    const uint8_t rep = (uint8_t)repeat;
    const uint16_t ss = (uint16_t)src_stride;
    const uint16_t dg = (uint16_t)dst_gap;
    const uint16_t dfg = (uint16_t)dst_frac_gap;
    if constexpr (P == Position::L0A) {
        load_cbuf_to_ca_transpose(dst.ptr(), src.ptr(), si, rep, ss, dg, inc, dfg);
    } else {
        load_cbuf_to_cb_transpose(dst.ptr(), src.ptr(), si, rep, ss, dg, inc, dfg);
    }
}

template <bool TRANS, typename T, Position P>
__aicore__ inline void l1_to_l0(Tensor<T, P> dst, Tensor<T, Position::L1> src, int32_t m_src, int32_t n_src, int32_t m_dst, int32_t n_dst)
{
    static_assert(P == Position::L0A || P == Position::L0B, "L0 load destination must be L0A or L0B");
    if ASCEND_IS_AIC {
        const int32_t C0 = 32 / (int32_t)sizeof(T);
        // The same predicate `gm_to_l1_nd2nz` writes with: only **float** L1 tiles are ZZ here.
        // Testing sizeof(T) instead sent int32 down these four branches while the tile had been
        // written as NZ, so the read walked a layout the data was not in.
        if constexpr (is_same<T, float>::value) {
            // float: the L1 tile is ZZ
            if constexpr (P == Position::L0A && !TRANS) {  // ZZ -> ZZ
                if (n_dst == n_src && n_dst % 16 > 8) {
                    l0_2d<false, T, P>(dst, src, AlignUp(m_dst, 16) * AlignUp(n_dst, 8) * (int32_t)sizeof(T) / 32 / 16, 1);
                    return;
                }
                for (int32_t i = 0; i < CeilDiv(m_dst, 16); ++i) {
                    l0_2d<false, T, P>(dst[i * 16 * AlignUp(n_dst, 8)], src[i * 16 * AlignUp(n_src, 16)], CeilDiv(n_dst, 8), 1);
                }
            } else if constexpr (P == Position::L0A) {  // ZZ -> NN (transposing load)
                for (int32_t i = 0; i < (n_dst + 15) / 16; ++i) {
                    l0_2d_transpose<T, P>(dst[i * 16 * AlignUp(m_dst, 16)], src[i * 16 * 16], (m_dst + 15) / 16, (n_src + 15) / 16, 1, 0);
                }
            } else if constexpr (!TRANS) {  // ZZ -> NZ
                for (int32_t i = 0; i < CeilDiv(n_dst, 8); ++i) {
                    l0_2d<false, T, P>(dst[i * 8 * AlignUp(m_dst, 16)], src[i * 8 * 16], CeilDiv(m_dst, 16), AlignUp(n_src, 16) / 8);
                }
            } else {  // ZZ -> ZN
                for (int32_t i = 0; i < (m_dst + 15) / 16; ++i) {
                    l0_2d_transpose<T, P>(dst[i * 16 * AlignUp(n_dst, 16)], src[i * 16 * AlignUp(n_src, 16)], CeilDiv(n_dst, 16), 1, 0,
                                          CeilDiv(n_dst, 16) - 1);
                }
            }
        } else if constexpr (P == Position::L0A && !TRANS) {  // NZ -> ZZ
            for (int32_t i = 0; i < (m_dst + 15) / 16; ++i) {
                l0_2d<false, T, P>(dst[16 * i * AlignUp(n_dst, C0)], src[i * 16 * C0], (n_dst + C0 - 1) / C0, (m_src + 15) / 16);
            }
        } else if constexpr (P == Position::L0A) {  // NZ -> NN
            if constexpr (sizeof(T) == 1) {
                for (int32_t i = 0; i < (n_dst + 31) / 32; ++i) {
                    l0_2d_transpose<T, P>(dst[i * 32 * AlignUp(m_dst, 32)], src[i * 32 * AlignUp(m_src, 32)], (m_dst + 31) / 32, 1, 0,
                                          (m_dst + 31) / 32 - 1);
                }
            } else {
                for (int32_t i = 0; i < (n_dst + C0 - 1) / C0; ++i) {
                    l0_2d<true, T, P>(dst[C0 * i * AlignUp(m_dst, 16)], src[C0 * i * AlignUp(m_src, 16)], (m_dst + 15) / 16, 1);
                }
            }
        } else if constexpr (!TRANS) {  // NZ -> NZ
            if (m_dst == m_src) {
                l0_2d<false, T, P>(dst, src, AlignUp(m_dst, 16) * AlignUp(n_dst, C0) * (int32_t)sizeof(T) / 32 / 16, 1);
            } else {
                for (int32_t i = 0; i < (n_dst + C0 - 1) / C0; ++i) {
                    l0_2d<false, T, P>(dst[C0 * i * AlignUp(m_dst, 16)], src[C0 * i * AlignUp(m_src, 16)], (m_dst + 15) / 16, 1);
                }
            }
        } else {  // NZ -> ZN
            if constexpr (sizeof(T) == 1) {
                for (int32_t i = 0; i < (m_dst + 31) / 32; ++i) {
                    l0_2d_transpose<T, P>(dst[i * 32 * AlignUp(n_dst, 32)], src[i * 32 * 32], (n_dst + 31) / 32, (m_src + 31) / 32, 1, 0);
                }
            } else {
                for (int32_t i = 0; i < (m_dst + 15) / 16; ++i) {
                    l0_2d<true, T, P>(dst[16 * i * AlignUp(n_dst, C0)], src[i * 16 * C0], (n_dst + C0 - 1) / C0, (m_src + 15) / 16);
                }
            }
        }
    }
}

// L1 -> the bias table: 64-B bursts; loading half into a float BT converts on the fly.
template <typename TDst, typename TSrc>
__aicore__ inline void l1_to_bt(Tensor<TDst, Position::BT> dst, Tensor<TSrc, Position::L1> src, int32_t n)
{
    if ASCEND_IS_AIC {
        constexpr bool conv = sizeof(TDst) != sizeof(TSrc);
        const int32_t one = conv ? 1 : 2;
        int32_t len = (n * one * 2 + 31) / 32;
        if (!conv && (len & 1)) {
            ++len;  // same-width bursts round to an even 64-B count
        }
        copy_cbuf_to_bt((uint64_t)dst.addr, (__cbuf__ void*)src.ptr(), (uint8_t)(conv ? 1 : 0), (uint16_t)1, (uint16_t)len,
                        (uint16_t)0, (uint16_t)0);
    }
}

// The c220 fixpipe scalar-quant modes (the old ResolveFixpipeQuant, c220 arms): the mode follows the
// (dst, L0C) dtype pair, 8-bit signedness follows the dst dtype (bit 46 of the deq scalar).
__aicore__ inline uint64_t pack_deq_scalar(float scale, int32_t offset)
{
    union { float f; uint32_t u; } cvt;
    cvt.f = scale;
    uint64_t deq = (uint64_t)(cvt.u & 0xFFFFE000u);
    deq |= ((uint64_t)((uint32_t)offset & 0x1FFu)) << 37;
    return deq;
}

template <typename T, typename T2>
__aicore__ inline void fixpipe_quant(QuantMode_t& q, uint64_t& deq, float scale, int32_t offset, bool scaled, bool hif8_hybrid)
{
    (void)hif8_hybrid;
    (void)scaled;
    deq = 0;
    if constexpr (is_same<T, float>::value && is_same<T2, float>::value) {
        q = QuantMode_t::NoQuant;
    } else if constexpr (is_same<T, half>::value && is_same<T2, float>::value) {
        q = QuantMode_t::F322F16;
    } else if constexpr (is_same<T, bfloat16_t>::value && is_same<T2, float>::value) {
        q = QuantMode_t::F322BF16;
    } else if constexpr (is_same<T, half>::value && is_same<T2, int32_t>::value) {
        q = QuantMode_t::DEQF16;
        deq = pack_deq_scalar(scale, 0);
    } else if constexpr (sizeof(T) == 1 && is_same<T2, float>::value) {
        q = QuantMode_t::QF322B8_PRE;
        deq = pack_deq_scalar(scale, offset);
        if constexpr (is_signed_int<T>::value) {
            deq |= (1ULL << 46);
        }
    } else if constexpr (sizeof(T) == 1 && is_same<T2, int32_t>::value) {
        q = QuantMode_t::REQ8;
        deq = pack_deq_scalar(scale, offset);
        if constexpr (is_signed_int<T>::value) {
            deq |= (1ULL << 46);
        }
    } else {
        q = QuantMode_t::NoQuant;
    }
}

__aicore__ inline bool is_scalar_quant(QuantMode_t q)
{
    return q == QuantMode_t::DEQF16 || q == QuantMode_t::QF322B8_PRE || q == QuantMode_t::REQ8;
}

// L0C NZ -> GM ND (ROW_MAJOR): N_dst = the destination row pitch in elements.
template <typename T, typename T2>
__aicore__ inline void l0c_to_gm_nz2nd(GMTensor<T> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t N_dst, int32_t M_src, bool relu,
                                       float scale, int32_t offset, bool scaled, bool hif8_hybrid)
{
    if ASCEND_IS_AIC {
        QuantMode_t q;
        uint64_t deq;
        fixpipe_quant<T, T2>(q, deq, scale, offset, scaled, hif8_hybrid);
        set_nd_para((1ULL << 32) | (1ULL << 16) | 1ULL);
        if (is_scalar_quant(q)) {
            set_quant_pre(deq);
        }
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)N_dst, (uint16_t)((M_src + 15) / 16 * 16),
                             (uint8_t)0, q, (uint8_t)(relu ? 1 : 0), false, true);
    }
}

// L0C NZ -> GM NZ: strips of M_pad rows; dstStride between fractal columns in 32-B units.
template <typename T, typename T2>
__aicore__ inline void l0c_to_gm_nz2nz(GMTensor<T> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t M_pad, int32_t M_src, bool relu,
                                       float scale, int32_t offset, bool scaled, bool hif8_hybrid)
{
    if ASCEND_IS_AIC {
        QuantMode_t q;
        uint64_t deq;
        fixpipe_quant<T, T2>(q, deq, scale, offset, scaled, hif8_hybrid);
        if (is_scalar_quant(q)) {
            set_quant_pre(deq);
        }
        pipe_barrier(PIPE_FIX);
        copy_matrix_cc_to_gm(dst.ptr(), src.ptr(), 0, (uint16_t)N, (uint16_t)M, (uint32_t)(M_pad * (sizeof(T) == 1 ? 32 : 16) * (int32_t)sizeof(T) / 32),
                             (uint16_t)((M_src + 15) / 16 * 16), (uint8_t)0, q, (uint8_t)(relu ? 1 : 0), false, false);
    }
}

// dma.l0c_to_l1 (mapping B.12): L0C NZ -> L1 NZ via copy_matrix_cc_to_cbuf. dstStride is in 32-B
// units between fractal columns (M_dst rows of a 16-element row). fp32 -> fp32 has NO c220 form:
// the builtin's dst overloads carry no float, and the old framework's C220 float->float L0C2L1
// path is disabled in its field code ("until support is clarified") — the printer gaps it.
template <typename T, typename T2>
__aicore__ inline void l0c_to_l1(Tensor<T, Position::L1> dst, Tensor<T2, Position::L0C> src, int32_t M, int32_t N, int32_t M_dst, int32_t M_src, bool relu)
{
    static_assert(!(is_same<T, float>::value && is_same<T2, float>::value),
                  "l0c_to_l1: fp32 -> fp32 has no c220 spelling (the old framework disabled it too)");
    if ASCEND_IS_AIC {
        QuantMode_t q;
        if constexpr (is_same<T, half>::value && is_same<T2, float>::value) {
            q = QuantMode_t::F322F16;
        } else if constexpr (is_same<T, bfloat16_t>::value && is_same<T2, float>::value) {
            q = QuantMode_t::F322BF16;
        } else {
            q = QuantMode_t::NoQuant;
        }
        copy_matrix_cc_to_cbuf(dst.ptr(), src.ptr(), (uint8_t)0, (uint16_t)N, (uint16_t)M,
                               (uint32_t)((M_dst + 15) / 16 * 16 * 16 * (int32_t)sizeof(T) / 32),
                               (uint16_t)((M_src + 15) / 16 * 16), (uint8_t)0, q, (uint8_t)(relu ? 1 : 0), false, false);
    }
}

// dma.l1_to_l0.img2col (mapping B.11): L1 NC1HWC0 feature map -> L0A ZZ im2col window. The c220
// load3dv2 materialises the whole [m_ext, k_ext] window in one call (no c310 repeat-SPR / per-C0
// chunk dance): FMATRIX (w | h<<16 | pads at 32..56) and PADDING (0) first, then img2colv2.
// Argument types mirror the CANN dav_c220 impl (kernel_operator_mm_impl.h::LoadData3DV2L12L0ACal)
// exactly -- the intrinsic is a polymorphic builtin resolved by argument pattern. bf16 has no
// overload of its own and goes through the half spelling, as in CANN.
template <typename T>
__aicore__ inline void l1_to_l0_img2col(Tensor<T, Position::L0A> dst, Tensor<T, Position::L1> src, int32_t h, int32_t w, int32_t c, int32_t kh, int32_t kw,
                                        int32_t pad_l, int32_t pad_r, int32_t pad_t, int32_t pad_b, int32_t stride_h, int32_t stride_w,
                                        int32_t dil_h, int32_t dil_w, int32_t k0, int32_t m0, int32_t k_ext, int32_t m_ext)
{
    if ASCEND_IS_AIC {
        uint64_t fmatrix = 0;
        fmatrix |= (uint64_t)(w & 0xFFFF);
        fmatrix |= (uint64_t)(h & 0xFFFF) << 16;
        fmatrix |= (uint64_t)(pad_l & 0xFF) << 32;
        fmatrix |= (uint64_t)(pad_r & 0xFF) << 40;
        fmatrix |= (uint64_t)(pad_t & 0xFF) << 48;
        fmatrix |= (uint64_t)(pad_b & 0xFF) << 56;
        set_fmatrix(fmatrix);
        set_padding(0);
        const uint16_t kx = (uint16_t)k_ext;
        const uint16_t mx = (uint16_t)m_ext;
        const uint16_t ks = (uint16_t)k0;
        const uint16_t ms = (uint16_t)m0;
        const uint8_t sw = (uint8_t)stride_w;
        const uint8_t sh = (uint8_t)stride_h;
        const uint8_t fw = (uint8_t)kw;
        const uint8_t fh = (uint8_t)kh;
        const uint8_t dw = (uint8_t)dil_w;
        const uint8_t dh = (uint8_t)dil_h;
        const uint16_t cs = (uint16_t)c;
        if constexpr (is_same<T, bfloat16_t>::value) {
            img2colv2_cbuf_to_ca((__ca__ half*)dst.addr, (__cbuf__ half*)src.addr, kx, mx, ks, ms,
                                 sw, sh, fw, fh, dw, dh, false, false, false, false, cs);
        } else {
            img2colv2_cbuf_to_ca(dst.ptr(), src.ptr(), kx, mx, ks, ms,
                                 sw, sh, fw, fh, dw, dh, false, false, false, false, cs);
        }
    }
}
#endif  // c220 cube path
// ===========================================================================================
// Vector-core special copies: TransDataTo5HD (vnchwconv through the VA address registers), the
// ND DMA (GM -> UB multi-dim strided copy with padding) and the sort family.
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // special copies: c310 vnchwconv / ND DMA / sort; the c220 forms land with the vec-sort pass
template <typename T>
__aicore__ inline void transdata5hd(Tensor<T, Position::UB> dst, Tensor<T, Position::UB> src, int32_t repeat, int32_t src_row_stride, int32_t dst_row_stride,
                                    int32_t src_rep_stride, int32_t dst_rep_stride)
{
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "transdata5hd supports b16 / b32 element types");
    if ASCEND_IS_AIV {
        uint64_t dst_list[16];
        uint64_t src_list[16];
        for (int i = 0; i < 16; ++i) {
            // b32 produces eight 16-element rows, each split across two address entries.
            const int32_t dst_offset = sizeof(T) == 4 ? (i / 2) * dst_row_stride + (i % 2) * 8 : i * dst_row_stride;
            dst_list[i] = dst.addr + (uint64_t)dst_offset * sizeof(T);
            src_list[i] = src.addr + (uint64_t)(i * src_row_stride) * sizeof(T);
        }
        uint16_t dst_rep = (repeat == 1) ? 0 : (uint16_t)dst_rep_stride;
        uint16_t src_rep = (repeat == 1) ? 0 : (uint16_t)src_rep_stride;
        set_va_reg_sb(VA0, dst_list);
        set_va_reg_sb(VA1, dst_list + 8);
        set_va_reg_sb(VA2, src_list);
        set_va_reg_sb(VA3, src_list + 8);
        if constexpr (sizeof(T) == 2) {
            scatter_vnchwconv_b16(VA0, VA2, (uint8_t)repeat, dst_rep, src_rep);
        } else {
            scatter_vnchwconv_b32(VA0, VA2, (uint8_t)repeat, dst_rep, src_rep);
        }
    }
}

// ND DMA from GM to UB (dim <= 5). Loop encodings from dav_3510: pad counts 16 bits per axis
// (lp at bit 16*i - 16, rp at 16*i - 8), stride words = src_stride << 20 | dst_stride & 0xfffff.
template <int DIM>
struct NdLoops {
    uint64_t src_stride[DIM];
    uint32_t dst_stride[DIM];
    uint32_t size[DIM];
    uint8_t left_pad[DIM];
    uint8_t right_pad[DIM];
};

template <typename T, int DIM>
__aicore__ inline void gm_to_ub_nd(Tensor<T, Position::UB> dst, GMTensor<T> src, const NdLoops<DIM>& lp, T constant_value, bool nearest_value_mode)
{
    static_assert(DIM >= 1 && DIM <= 5, "ND DMA supports dim 1..5");
    static_assert(sizeof(T) <= 4, "ND DMA requires elements of at most 32 bits");
    if ASCEND_IS_AIV {
        nd_dma_dci();
        uint32_t size[5] = {lp.size[0], 1, 1, 1, 1};
        uint64_t stride[5] = {0, 0, 0, 0, 0};
        uint64_t pad_count = 0;
        stride[0] = (lp.src_stride[0] << 20) | (uint64_t)(lp.dst_stride[0] & 0xfffff);
        for (int i = 1; i < DIM; ++i) {
            pad_count |= (uint64_t)(lp.left_pad[i] & 0xff) << (16 * i - 16);
            pad_count |= (uint64_t)(lp.right_pad[i] & 0xff) << (16 * i - 8);
            size[i] = lp.size[i];
            stride[i] = (lp.src_stride[i] << 20) | (uint64_t)(lp.dst_stride[i] & 0xfffff);
        }
        set_pad_cnt_nddma(pad_count);
        set_loop0_stride_nddma(stride[0]);
        set_loop1_stride_nddma(stride[1]);
        set_loop2_stride_nddma(stride[2]);
        set_loop3_stride_nddma(stride[3]);
        set_loop4_stride_nddma(stride[4]);
        set_pcie_rd_ctrl(0);
        if constexpr (sizeof(T) == 1) {
            uint8_t pad;
            __builtin_memcpy(&pad, &constant_value, 1);
            set_pad_val_nddma(pad);
            nddma_out_to_ub_b8((__ubuf__ T*)dst.ptr(), (__gm__ T*)src.ptr(), 0, size[0], size[1], size[2], size[3], size[4], lp.left_pad[0],
                               lp.right_pad[0], !nearest_value_mode, 0);
        } else if constexpr (sizeof(T) == 2) {
            uint16_t pad;
            __builtin_memcpy(&pad, &constant_value, 2);
            set_pad_val_nddma(pad);
            nddma_out_to_ub_b16((__ubuf__ T*)dst.ptr(), (__gm__ T*)src.ptr(), 0, size[0], size[1], size[2], size[3], size[4], lp.left_pad[0],
                                lp.right_pad[0], !nearest_value_mode, 0);
        } else {
            uint32_t pad;
            __builtin_memcpy(&pad, &constant_value, 4);
            set_pad_val_nddma(pad);
            nddma_out_to_ub_b32((__ubuf__ T*)dst.ptr(), (__gm__ T*)src.ptr(), 0, size[0], size[1], size[2], size[3], size[4], lp.left_pad[0],
                                lp.right_pad[0], !nearest_value_mode, 0);
        }
    }
}

template <typename T>
__aicore__ inline void sort32(Tensor<T, Position::UB> dst, Tensor<T, Position::UB> src, Tensor<uint32_t, Position::UB> idx, int32_t repeat)
{
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "sort32 supports half / float");
    if ASCEND_IS_AIV {
        uint64_t config = (uint64_t)(uint8_t)repeat << 56;
        vbs(dst.ptr(), src.ptr(), idx.ptr(), config);
    }
}

template <typename T>
__aicore__ inline void vmrgsort4_raw(__ubuf__ T* dst, __ubuf__ T* a0, __ubuf__ T* a1, __ubuf__ T* a2, __ubuf__ T* a3, uint16_t len0, uint16_t len1,
                                     uint16_t len2, uint16_t len3, uint16_t valid_bit, uint16_t repeat)
{
    uint64_t config = 0;
    config |= (uint64_t)(repeat & 0xFF);
    config |= (uint64_t)(valid_bit & 0xF) << 8;
    uint64_t src1 = 0;
    src1 |= (uint64_t)(len0 & 0xFFFF);
    src1 |= (uint64_t)(len1 & 0xFFFF) << 16;
    src1 |= (uint64_t)(len2 & 0xFFFF) << 32;
    src1 |= (uint64_t)(len3 & 0xFFFF) << 48;
    __ubuf__ T* addr[4] = {a0, a1, a2, a3};
    vmrgsort4(dst, addr, src1, config);
}

// Four sorted lists of len entries each (8-byte score + index pairs) -> one; repeat groups.
template <typename T>
__aicore__ inline void mergesort4(Tensor<T, Position::UB> dst, Tensor<T, Position::UB> src, int32_t len, int32_t repeat)
{
    if ASCEND_IS_AIV {
        const int32_t step = len * 2 * 4 / (int32_t)sizeof(T);
        vmrgsort4_raw<T>(dst.ptr(), src.ptr(), src[step].ptr(), src[2 * step].ptr(), src[3 * step].ptr(), (uint16_t)len, (uint16_t)len,
                         (uint16_t)len, (uint16_t)len, 0b1111, (uint16_t)repeat);
    }
}

template <typename T>
__aicore__ inline void mergesort_2seq(Tensor<T, Position::UB> dst, Tensor<T, Position::UB> src1, Tensor<T, Position::UB> src2, int32_t size1, int32_t size2)
{
    if ASCEND_IS_AIV {
        vmrgsort4_raw<T>(dst.ptr(), src1.ptr(), src2.ptr(), src2.ptr(), src2.ptr(), (uint16_t)size1, (uint16_t)size2, 0, 0, 3, 1);
    }
}

#else  // c220 special copies: the VA-register 16x16 transpose and the (score, index) sort family
// TransDataTo5HD: 16 row addresses per side through VA0..VA3, one vnchwconv per repeat batch
// (dav_c220 kernel_operator_vec_transpose_impl.h; with repeat == 1 both rep strides print 0).
template <typename T>
__aicore__ inline void transdata5hd(Tensor<T, Position::UB> dst, Tensor<T, Position::UB> src, int32_t repeat, int32_t src_row_stride,
                                    int32_t dst_row_stride, int32_t src_rep_stride, int32_t dst_rep_stride)
{
    static_assert(sizeof(T) == 2, "transdata5hd rides the b16 vnchwconv on c220");
    if ASCEND_IS_AIV {
        uint64_t dst_list[16];
        uint64_t src_list[16];
        for (int32_t i = 0; i < 16; ++i) {
            dst_list[i] = dst[i * dst_row_stride].addr;
            src_list[i] = src[i * src_row_stride].addr;
        }
        set_va_reg_sb(VA0, dst_list);
        set_va_reg_sb(VA1, dst_list + 8);
        set_va_reg_sb(VA2, src_list);
        set_va_reg_sb(VA3, src_list + 8);
        scatter_vnchwconv_b16(VA0, VA2, (uint8_t)repeat, (uint16_t)(repeat == 1 ? 0 : dst_rep_stride),
                              (uint16_t)(repeat == 1 ? 0 : src_rep_stride));
    }
}

// The sort family: fp32 (score, uint32 index) 8-byte records, descending (D-047's record shape).
__aicore__ inline void sort32(Tensor<float, Position::UB> dst, Tensor<float, Position::UB> src, Tensor<uint32_t, Position::UB> idx,
                              int32_t repeat)
{
    if ASCEND_IS_AIV {
        vbitsort(dst.ptr(), src.ptr(), (__ubuf__ unsigned int*)idx.ptr(), (uint8_t)repeat);
    }
}

__aicore__ inline uint64_t vmrgsort4_config(int32_t repeat, int32_t valid_bit)
{
    return ((uint64_t)repeat & 0xFF) | (((uint64_t)valid_bit & 0xF) << 8);
}

// Four contiguous sorted lists of length_per_seq records each -> one list, `repeat` batches.
__aicore__ inline void mergesort4(Tensor<float, Position::UB> dst, Tensor<float, Position::UB> src, int32_t length_per_seq, int32_t repeat)
{
    if ASCEND_IS_AIV {
        __ubuf__ float* addr[4] = {src.ptr(), src[2 * length_per_seq].ptr(), src[4 * length_per_seq].ptr(),
                                   src[6 * length_per_seq].ptr()};
        uint64_t lens = (uint64_t)length_per_seq | ((uint64_t)length_per_seq << 16) | ((uint64_t)length_per_seq << 32) |
                        ((uint64_t)length_per_seq << 48);
        vmrgsort4(dst.ptr(), addr, lens, vmrgsort4_config(repeat, 0xF));
    }
}

__aicore__ inline void mergesort_2seq(Tensor<float, Position::UB> dst, Tensor<float, Position::UB> src1, Tensor<float, Position::UB> src2,
                                      int32_t size1, int32_t size2)
{
    if ASCEND_IS_AIV {
        __ubuf__ float* addr[4] = {src1.ptr(), src2.ptr(), src2.ptr(), src2.ptr()};
        uint64_t lens = (uint64_t)size1 | ((uint64_t)size2 << 16);
        vmrgsort4(dst.ptr(), addr, lens, vmrgsort4_config(1, 0x3));
    }
}
#endif  // special copies
// ===========================================================================================
// SIMT shim: the launch and the few helpers a printed @simt function calls, on the compiler's own
// SIMT layer (__clang_cce_simt*.h: threadIdx / blockDim / blockIdx / gridDim, the atomic family,
// __sync_workitems, cce::dim3 / cce::async_invoke). Semantics transcribed from CANN 9.2
// asc/impl/simt_api/cpp/dav_3510 (fire-and-forget plus readback where the hardware op returns
// nothing).
// ===========================================================================================
#if !defined(__CCE_AICORE__) || defined(__DAV_C310__)  // simt shim: rides c310's own SIMT layer
namespace simt {
template <typename A, typename B>
__simt_callee__ inline A Min(A a, B b) { return (a < b) ? a : (A)b; }
template <typename A, typename B>
__simt_callee__ inline A Max(A a, B b) { return (a < b) ? (A)b : a; }

template <auto FN, typename... Args>
__aicore__ inline void launch(int threads, Args&&... args)
{
#if defined(__DAV_VEC__)
    cce::async_invoke<FN>(cce::dim3{(unsigned)threads, 1u, 1u}, args...);
#else
    (void)threads;
#endif
}
__simt_callee__ inline uint32_t thread_id() { return threadIdx.x; }
__simt_callee__ inline uint32_t thread_num() { return blockDim.x; }
__simt_callee__ inline void barrier() { __sync_workitems(); }

template <typename T>
__simt_callee__ inline T atomic_add(__ubuf__ T* address, T val) { return atomicAdd(address, val); }
template <typename T>
__simt_callee__ inline T atomic_add(__gm__ T* address, T val) { return atomicAdd(address, val); }
template <typename T>
__simt_callee__ inline T atomic_sub(__ubuf__ T* address, T val) { return atomicSub(address, val); }
template <typename T>
__simt_callee__ inline T atomic_sub(__gm__ T* address, T val) { return atomicSub(address, val); }
template <typename T>
__simt_callee__ inline T atomic_max(__ubuf__ T* address, T val) { return atomicMax(address, val); }
template <typename T>
__simt_callee__ inline T atomic_max(__gm__ T* address, T val) { return atomicMax(address, val); }
template <typename T>
__simt_callee__ inline T atomic_min(__ubuf__ T* address, T val) { return atomicMin(address, val); }
template <typename T>
__simt_callee__ inline T atomic_min(__gm__ T* address, T val) { return atomicMin(address, val); }
template <typename T>
__simt_callee__ inline T atomic_exch(__ubuf__ T* address, T val) { return atomicExch(address, val); }
template <typename T>
__simt_callee__ inline T atomic_exch(__gm__ T* address, T val) { return atomicExch(address, val); }
template <typename T>
__simt_callee__ inline T atomic_and(__ubuf__ T* address, T val) { return atomicAnd(address, val); }
template <typename T>
__simt_callee__ inline T atomic_and(__gm__ T* address, T val) { return atomicAnd(address, val); }
template <typename T>
__simt_callee__ inline T atomic_or(__ubuf__ T* address, T val) { return atomicOr(address, val); }
template <typename T>
__simt_callee__ inline T atomic_or(__gm__ T* address, T val) { return atomicOr(address, val); }
template <typename T>
__simt_callee__ inline T atomic_xor(__ubuf__ T* address, T val) { return atomicXOr(address, val); }
template <typename T>
__simt_callee__ inline T atomic_xor(__gm__ T* address, T val) { return atomicXOr(address, val); }
template <typename T>
__simt_callee__ inline T atomic_cas(__ubuf__ T* address, T compare, T val) { return atomicCAS(address, compare, val); }
template <typename T>
__simt_callee__ inline T atomic_cas(__gm__ T* address, T compare, T val) { return atomicCAS(address, compare, val); }

// ring inc/dec (CUDA semantics: inc wraps past `limit` to 0, dec wraps 0 / >limit to `limit`)
__simt_callee__ inline uint32_t atomic_inc(__ubuf__ uint32_t* address, uint32_t limit) { return atomicInc(address, limit); }
__simt_callee__ inline uint32_t atomic_inc(__gm__ uint32_t* address, uint32_t limit) { return atomicInc(address, limit); }
__simt_callee__ inline uint32_t atomic_dec(__ubuf__ uint32_t* address, uint32_t limit) { return atomicDec(address, limit); }
__simt_callee__ inline uint32_t atomic_dec(__gm__ uint32_t* address, uint32_t limit) { return atomicDec(address, limit); }

__simt_callee__ inline void threadfence() { __threadfence(); }
__simt_callee__ inline void threadfence_block() { __threadfence_block(); }

// ---- scalar math on the SIMT layer. The dav-c310 compiler builtins are the same layer the
// PyPTO Pro CCE codegen prints (__expf / __logf / __sqrtf / __fma / ...); everything else is
// synthesised here the way its backend and the CANN dav_c310 SIMT impl do.
__simt_callee__ inline float exp(float x) { return __expf(x); }
__simt_callee__ inline float log(float x) { return __logf(x); }
__simt_callee__ inline float exp2(float x) { return __expf(x * 0.6931471805599453f); }
__simt_callee__ inline float log2(float x) { return __logf(x) * 1.4426950408889634f; }
__simt_callee__ inline float log1p(float x) { return __logf(1.0f + x); }
__simt_callee__ inline float rsqrt(float x) { return 1.0f / __sqrtf(x); }
__simt_callee__ inline float tanh(float x) { return 1.0f - (2.0f / (__expf(2.0f * x) + 1.0f)); }
// A5 native integral rounding can canonicalize -0 to +0 (M10-054). Adapt that
// target detail here, using integer fields so signed-zero/fast-math rewrites
// cannot turn the zero-magnitude test or restored sign into floating arithmetic.
__simt_callee__ inline float integral_zero_sign_(float x, float result)
{
    union { float f; uint32_t u; } source, rounded;
    source.f = x;
    rounded.f = result;
    if ((rounded.u & 0x7FFFFFFFu) == 0u)
        rounded.u = source.u & 0x80000000u;
    return rounded.f;  // Every nonzero result, including NaN payloads, is unchanged.
}
__simt_callee__ inline float trunc_native_(float x) { return x >= 0.0f ? __floorf(x) : __ceilf(x); }
__simt_callee__ inline float rint(float x) { return integral_zero_sign_(x, __rintf(x)); }
__simt_callee__ inline float round(float x) { return integral_zero_sign_(x, __roundf(x)); }
__simt_callee__ inline float floor(float x) { return integral_zero_sign_(x, __floorf(x)); }
__simt_callee__ inline float ceil(float x) { return integral_zero_sign_(x, __ceilf(x)); }
__simt_callee__ inline float trunc(float x) { return integral_zero_sign_(x, trunc_native_(x)); }
// Exact FP32 remainder, as PyPTO Pro prints it: long division of the
// integer significand fields never rounds, keeps subnormals and gives a zero result the
// dividend's sign. A NaN operand, infinite dividend or zero divisor is quiet NaN 0x7FC00000.
__simt_callee__ inline float fmod(float x, float y)
{
    union { float f; uint32_t u; } dividend, divisor;
    dividend.f = x;
    divisor.f = y;
    uint32_t sign = dividend.u & 0x80000000u;
    uint32_t ax = dividend.u & 0x7FFFFFFFu, ay = divisor.u & 0x7FFFFFFFu;
    if (ay == 0u || ax >= 0x7F800000u || ay > 0x7F800000u) {
        dividend.u = 0x7FC00000u;
    } else if (ax >= ay) {  // Otherwise |x| < |y| and x is the remainder.
        int32_t ex = (int32_t)(ax >> 23), ey = (int32_t)(ay >> 23);
        uint32_t mx = ax & 0x007FFFFFu, my = ay & 0x007FFFFFu;
        if (ex == 0) {
            ex = 1;
            while (mx < 0x00800000u) { mx <<= 1; --ex; }
        } else {
            mx |= 0x00800000u;
        }
        if (ey == 0) {
            ey = 1;
            while (my < 0x00800000u) { my <<= 1; --ey; }
        } else {
            my |= 0x00800000u;
        }
        while (ex > ey) {
            if (mx >= my) mx -= my;
            mx <<= 1;
            --ex;
        }
        if (mx >= my) mx -= my;
        if (mx == 0u) {
            dividend.u = sign;
        } else {
            while (mx < 0x00800000u) { mx <<= 1; --ex; }
            dividend.u = sign | (ex > 0 ? (mx & 0x007FFFFFu) | ((uint32_t)ex << 23) : mx >> (1 - ex));
        }
    }
    return dividend.f;
}
__simt_callee__ inline int32_t ffs(uint32_t x)  // 1-based index of the lowest set bit; 0 when empty
{
    return x ? (int32_t)__popc((x & (0u - x)) - 1u) + 1 : 0;
}
__simt_callee__ inline int32_t ffs(uint64_t x)
{
    return __ffs(static_cast<long long>(x));
}

// sin / cos: the CANN dav_c310 SIMT reduction + polynomials (Cody-Waite for small angles,
// Payne-Hanek for |x| > 71476.0625f), the same pair the PyPTO Pro codegen inlines.
__simt_callee__ inline float sincos_reduce_ph_(float x, int32_t* quadrant)
{
    uint32_t bits = reinterpret_cast<uint32_t&>(x);
    int32_t exponent = (int32_t)((bits & 0x7F800000u) >> 23) - 127;
    uint32_t ei = (uint32_t)exponent >> 5;
    const uint32_t tbl[] = {0x517cc1b7u, 0x27220a94u, 0xfe13abe8u, 0xfa9a6ee0u, 0x6db14accu, 0x9e21c820u};
    uint32_t hi = ei ? tbl[ei - 1] : 0u;
    uint32_t mid = tbl[ei];
    uint32_t lo = tbl[ei + 1];
    uint32_t last = tbl[ei + 2];
    int32_t rem = (int32_t)((uint32_t)exponent & 0x1Fu);
    if (rem != 0) {
        hi = (hi << rem) | (mid >> (32 - rem));
        mid = (mid << rem) | (lo >> (32 - rem));
        lo = (lo << rem) | (last >> (32 - rem));
    }
    uint32_t mant = (bits & 0x007FFFFFu) | 0x4F000000u;
    uint32_t nm = (uint32_t)reinterpret_cast<float&>(mant);
    uint64_t prod = (uint64_t)nm * lo;
    prod = (uint64_t)nm * mid + (prod >> 32);
    prod = ((uint64_t)(nm * hi) << 32) + prod;
    int32_t q = (int32_t)(prod >> 62);
    prod &= 0x3FFFFFFFFFFFFFFFull;
    if (prod & 0x2000000000000000ull) {
        prod -= 0x4000000000000000ull;
        q += 1;
    }
    int64_t p64 = (int64_t)prod;
    float hf = (float)p64;
    p64 -= (int64_t)hf;
    float lf = (float)p64;
    float reduced = (hf + lf) * 3.4061215800865545e-19f;  // pi/2 * 2^-62
    if (x < 0.0f) {
        reduced = -reduced;
        q = -q;
    }
    *quadrant = q;
    return reduced;
}
__simt_callee__ inline float sincos_reduce_(float x, int32_t* quadrant)
{
    x = __fma(x, 0.0f, x);
    if (__fabsf(x) > 71476.0625f) {
        return sincos_reduce_ph_(x, quadrant);
    }
    float y = __fma(x, 0.636619747f, 12582912.0f);  // 2/pi; 1.5*2^23 truncates the mantissa
    *quadrant = reinterpret_cast<int32_t&>(y);
    y = y - 12582912.0f;
    x = __fma(y, -1.57079601e+00f, x);
    x = __fma(y, -3.13916473e-07f, x);
    return __fma(y, -5.39030253e-15f, x);
}
__simt_callee__ inline float sincos_cospoly_(float x)
{
    x = x * x;
    float y = __fma(x, 2.44677067e-5f, -1.38877297e-3f);
    y = __fma(x, y, 4.16666567e-2f);
    y = __fma(x, y, -5.00000000e-1f);
    return __fma(x, y, 1.00000000e+0f);
}
__simt_callee__ inline float sincos_sinpoly_(float x)
{
    float y = x * x;
    float m = __fma(x, y, 0.0f);
    float z = __fma(y, 2.86567956e-6f, -1.98559923e-4f);
    z = __fma(y, z, 8.33338592e-3f);
    z = __fma(y, z, -1.66666672e-1f);
    return __fma(z, m, x);
}
__simt_callee__ inline float sin(float x)
{
    int32_t q;
    float y = sincos_reduce_(x, &q);
    float c = sincos_cospoly_(y);
    float s = sincos_sinpoly_(y);
    if (q & 2) { s = -s; c = -c; }
    if (q & 1) { s = c; }
    return s;
}
__simt_callee__ inline float cos(float x)
{
    int32_t q;
    float y = sincos_reduce_(x, &q);
    float c = sincos_cospoly_(y);
    float s = sincos_sinpoly_(y);
    if (q & 2) { s = -s; c = -c; }
    if (q & 1) { c = -s; }
    return c;
}
}  // namespace simt
#endif  // simt

}  // namespace ascrip
