#ifndef LIM_SIMT_CONFIG_H
#define LIM_SIMT_CONFIG_H

// build.sh substitutes only its generated copy, never the checked-in source.
#ifndef LIM_SIMT_POSTPROCESS
#define LIM_SIMT_POSTPROCESS 1
#endif
#if LIM_SIMT_POSTPROCESS != 0 && LIM_SIMT_POSTPROCESS != 1
#error "LIM_SIMT_POSTPROCESS must be 0 or 1"
#endif

// Accepted e705800 baseline: victim unpack stays enabled independently.
#ifndef LIM_SIMT_VICTIM_UNPACK
#define LIM_SIMT_VICTIM_UNPACK 1
#endif
#if LIM_SIMT_VICTIM_UNPACK != 0 && LIM_SIMT_VICTIM_UNPACK != 1
#error "LIM_SIMT_VICTIM_UNPACK must be 0 or 1"
#endif
#if LIM_SIMT_VICTIM_UNPACK && !LIM_SIMT_POSTPROCESS
#error "LIM_SIMT_VICTIM_UNPACK requires LIM_SIMT_POSTPROCESS=1 (SIMT UB reservation)"
#endif

// One miss-slot repair VF per request, instead of one per nonempty query.
// With REQUEST_META_CACHE=0, set to 0 with POSTPROCESS=1, VICTIM_UNPACK=1
// for the e705800 A/B baseline.
#ifndef LIM_SIMT_BATCHED_REPAIR
#define LIM_SIMT_BATCHED_REPAIR LIM_SIMT_POSTPROCESS
#endif
#if LIM_SIMT_BATCHED_REPAIR != 0 && LIM_SIMT_BATCHED_REPAIR != 1
#error "LIM_SIMT_BATCHED_REPAIR must be 0 or 1"
#endif
#if LIM_SIMT_BATCHED_REPAIR && !LIM_SIMT_POSTPROCESS
#error "LIM_SIMT_BATCHED_REPAIR requires LIM_SIMT_POSTPROCESS=1 (SIMT UB reservation)"
#endif

// Reuse validated metadata within one steady request's management stage.
// Independent of SIMT: no new UB reservation or persistent device state.
#ifndef LIM_SIMT_REQUEST_META_CACHE
#define LIM_SIMT_REQUEST_META_CACHE 1
#endif
#if LIM_SIMT_REQUEST_META_CACHE != 0 && LIM_SIMT_REQUEST_META_CACHE != 1
#error "LIM_SIMT_REQUEST_META_CACHE must be 0 or 1"
#endif

#define LIM_SIMT_THREADS 256
// 256 KiB UB = 216 KiB user + 8 KiB runtime/stack + 32 KiB SIMT DCache.
#define LIM_SIMT_USER_UB_BYTES (216U * 1024U)

#endif
