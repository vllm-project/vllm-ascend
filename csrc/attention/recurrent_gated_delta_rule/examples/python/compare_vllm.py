"""Compare ops-transformer RGDR (ctypes) vs vLLM RGDR (torch.ops._C_ascend) vs golden CPU ref.

Usage:
    # on 310P device with both ops-transformer and vllm-ascend installed:
    python3 compare_vllm.py [--dtype fp16|fp32] [--qwen]
"""
import os
import sys

_CANN = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
_VLLM_VENDOR = os.environ.get(
    "VLLM_ASCEND_VENDOR",
    "/usr/local/python3.11.10/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend")
_OT_VENDOR = f"{_CANN}/opp/vendors/custom_transformer"
_cur = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{_OT_VENDOR}:{_VLLM_VENDOR}:{_cur}" if _cur else f"{_OT_VENDOR}:{_VLLM_VENDOR}"

import torch
import torch_npu
import ctypes
import argparse

torch.manual_seed(42)

# ---- ctypes wrapper for ops-transformer custom op ----

def _find_lib(name, paths):
    for p in paths:
        full = os.path.join(p, name)
        if os.path.exists(full):
            return full
    return name

_CUST = f"{_CANN}/opp/vendors/custom_transformer/op_api/lib"
_LIB_PATHS = [_CUST, f"{_CANN}/lib64", f"{_CANN}/aarch64-linux/lib64"]

_acl = ctypes.CDLL(_find_lib("libnnopbase.so", _LIB_PATHS))
_opapi = ctypes.CDLL(_find_lib("libcust_opapi.so", _LIB_PATHS))

_TORCH_TO_ACL_DTYPE = {
    torch.float16: 1, torch.float32: 0, torch.int32: 3, torch.bfloat16: 27,
}

_acl.aclCreateTensor.restype = ctypes.c_void_p
_acl.aclCreateTensor.argtypes = [
    ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
    ctypes.c_void_p,
]
_acl.aclDestroyTensor.argtypes = [ctypes.c_void_p]


def _make_acl_tensor(t):
    if t is None:
        return None
    shape = list(t.shape)
    strides = list(t.stride())
    ndim = len(shape)
    c_shape = (ctypes.c_int64 * ndim)(*shape)
    c_strides = (ctypes.c_int64 * ndim)(*strides)
    c_storage = (ctypes.c_int64 * ndim)(*shape)
    return ctypes.c_void_p(_acl.aclCreateTensor(
        c_shape, ndim, _TORCH_TO_ACL_DTYPE[t.dtype], c_strides, ctypes.c_int64(0),
        2, c_storage, ndim, ctypes.c_void_p(t.data_ptr())))


def call_ops_transformer(query, key, value, beta, state, actual_seq_lengths,
                         ssm_state_indices, g, num_accepted_tokens, scale):
    out = torch.empty_like(value)
    tensors = []
    def mk(t):
        p = _make_acl_tensor(t)
        if p is not None:
            tensors.append(p)
        return p

    q_t, k_t, v_t = mk(query), mk(key), mk(value)
    b_t, s_t = mk(beta), mk(state)
    sl_t, ss_t = mk(actual_seq_lengths), mk(ssm_state_indices)
    g_t, gk_t = mk(g), None
    nat_t = mk(num_accepted_tokens)
    o_t = mk(out)

    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p(0)
    ret = _opapi.aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
        q_t, k_t, v_t, b_t, s_t, sl_t, ss_t, g_t, gk_t, nat_t,
        ctypes.c_float(scale), o_t,
        ctypes.byref(ws_size), ctypes.byref(executor))
    assert ret == 0, f"GetWorkspaceSize failed: {ret}"

    ws_ptr = ctypes.c_void_p(0)
    if ws_size.value > 0:
        workspace = torch.empty(ws_size.value, dtype=torch.uint8, device=query.device)
        ws_ptr = ctypes.c_void_p(workspace.data_ptr())

    stream = torch.npu.current_stream().npu_stream
    ret = _opapi.aclnnRecurrentGatedDeltaRule(ws_ptr, ws_size, executor, ctypes.c_void_p(stream))
    assert ret == 0, f"aclnnRecurrentGatedDeltaRule failed: {ret}"
    torch.npu.synchronize()

    for t in tensors:
        _acl.aclDestroyTensor(t)
    return out


# ---- vLLM wrapper ----

ctypes.CDLL(_find_lib("libopapi.so", _LIB_PATHS), mode=ctypes.RTLD_GLOBAL)
_vllm_opapi = ctypes.CDLL(f"{_VLLM_VENDOR}/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)


def call_vllm(query, key, value, beta, state, actual_seq_lengths,
              ssm_state_indices, g, num_accepted_tokens, scale):
    """Call aclnnRecurrentGatedDeltaRuleV310 via ctypes (same pattern as ops-transformer)."""
    out = torch.empty_like(value)
    tensors = []
    def mk(t):
        p = _make_acl_tensor(t)
        if p is not None:
            tensors.append(p)
        return p

    q_t, k_t, v_t = mk(query.contiguous()), mk(key.contiguous()), mk(value.contiguous())
    b_t, s_t = mk(beta.contiguous()), mk(state)
    sl_t, ss_t = mk(actual_seq_lengths), mk(ssm_state_indices)
    g_t = mk(g.contiguous()) if g is not None else None
    gk_t = None
    nat_t = mk(num_accepted_tokens)
    o_t = mk(out)

    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p(0)
    ret = _vllm_opapi.aclnnRecurrentGatedDeltaRuleV310GetWorkspaceSize(
        q_t, k_t, v_t, b_t, s_t, sl_t, ss_t, g_t, gk_t, nat_t,
        ctypes.c_float(scale), o_t,
        ctypes.byref(ws_size), ctypes.byref(executor))
    assert ret == 0, f"vllm GetWorkspaceSize failed: {ret}"

    ws_ptr = ctypes.c_void_p(0)
    if ws_size.value > 0:
        workspace = torch.empty(ws_size.value, dtype=torch.uint8, device=query.device)
        ws_ptr = ctypes.c_void_p(workspace.data_ptr())

    stream = torch.npu.current_stream().npu_stream
    ret = _vllm_opapi.aclnnRecurrentGatedDeltaRuleV310(ws_ptr, ws_size, executor, ctypes.c_void_p(stream))
    assert ret == 0, f"aclnnRecurrentGatedDeltaRuleV310 failed: {ret}"
    torch.npu.synchronize()

    for t in tensors:
        _acl.aclDestroyTensor(t)
    return out


# ---- golden CPU reference ----

def golden_rgdr(query, key, value, state, beta, scale, actual_seq_lengths,
                ssm_state_indices, g, num_accepted_tokens):
    k = key.to(torch.float32)
    q = query.to(torch.float32)
    v = value.to(torch.float32)
    initial_state = state.clone().to(torch.float32)
    T, n_heads_v, Dv = v.shape
    n_heads_k = q.shape[-2]
    g_f = torch.ones(T, n_heads_v, dtype=torch.float32) if g is None else g.to(torch.float32).exp()
    beta_f = beta.to(torch.float32)
    o = torch.empty_like(v, dtype=torch.float32)
    q = q * scale

    seq_start = 0
    for i in range(len(actual_seq_lengths)):
        if num_accepted_tokens is None:
            init_state = initial_state[ssm_state_indices[seq_start]]
        else:
            init_state = initial_state[ssm_state_indices[seq_start + num_accepted_tokens[i] - 1]]

        for head_id in range(n_heads_v):
            S = init_state[head_id]
            for slot_id in range(seq_start, seq_start + actual_seq_lengths[i]):
                q_i = q[slot_id][head_id // (n_heads_v // n_heads_k)]
                k_i = k[slot_id][head_id // (n_heads_v // n_heads_k)]
                v_i = v[slot_id][head_id]
                alpha_i = g_f[slot_id][head_id]
                beta_i = beta_f[slot_id][head_id]
                S = S * alpha_i
                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i
                S = S + y[:, None] * k_i[None, :]
                initial_state[ssm_state_indices[slot_id]][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]
    return o, initial_state


# ---- comparison ----

def compare(label, dtype, rtol, b, mtp, nk, nv, dk, dv, num_cache_slots):
    scale = dk ** -0.5

    actual_seq_lengths = torch.ones(b, dtype=torch.int32) * mtp
    T = int(actual_seq_lengths.sum())

    # state is a cache: [num_cache_slots, nv, dv, dk], indexed by ssm_state_indices
    state = torch.rand(num_cache_slots, nv, dv, dk).to(dtype)
    # each token maps to a unique cache slot
    ssm_state_indices = torch.randperm(num_cache_slots, dtype=torch.int32)[:T]
    num_accepted_tokens = torch.ones(b, dtype=torch.int32)  # decode: 1 accepted per seq

    query = torch.nn.functional.normalize(torch.randn(T, nk, dk), dim=-1).to(dtype)
    key   = torch.nn.functional.normalize(torch.randn(T, nk, dk), dim=-1).to(dtype)
    value = torch.randn(T, nv, dv).to(dtype)
    beta  = torch.rand(T, nv).to(dtype)
    g     = torch.rand(T, nv, dtype=torch.float32)

    print(f"\n=== {label} ({dtype}) ===")
    print(f"  q={list(query.shape)} k={list(key.shape)} v={list(value.shape)} "
          f"state={list(state.shape)} b={b} mtp={mtp}")

    # golden (CPU fp32)
    out_gold, state_gold = golden_rgdr(
        query, key, value, state, beta, scale,
        actual_seq_lengths, ssm_state_indices, g, num_accepted_tokens)

    # move inputs to NPU once
    q_npu = query.npu()
    k_npu = key.npu()
    v_npu = value.npu()
    b_npu = beta.npu()
    sl_npu = actual_seq_lengths.npu()
    ss_npu = ssm_state_indices.npu()
    g_npu = g.npu()
    nat_npu = num_accepted_tokens.npu()

    # ops-transformer (our custom op)
    state_ot = state.clone().npu()
    out_ot = call_ops_transformer(q_npu, k_npu, v_npu, b_npu, state_ot,
                                  sl_npu, ss_npu, g_npu, nat_npu, scale)
    out_ot = out_ot.float().cpu()
    state_ot = state_ot.float().cpu()

    # vllm
    state_vllm = state.clone().npu()
    out_vllm = call_vllm(q_npu, k_npu, v_npu, b_npu, state_vllm,
                          sl_npu, ss_npu, g_npu, nat_npu, scale)
    out_vllm = out_vllm.float().cpu()
    state_vllm = state_vllm.float().cpu()

    # benchmark
    warmup, iters = 10, 100

    state_bench = state.clone().npu()
    for _ in range(warmup):
        call_ops_transformer(q_npu, k_npu, v_npu, b_npu, state_bench,
                             sl_npu, ss_npu, g_npu, nat_npu, scale)
    torch.npu.synchronize()
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        call_ops_transformer(q_npu, k_npu, v_npu, b_npu, state_bench,
                             sl_npu, ss_npu, g_npu, nat_npu, scale)
    t1.record()
    torch.npu.synchronize()
    ot_us = t0.elapsed_time(t1) * 1000.0 / iters

    state_bench = state.clone().npu()
    for _ in range(warmup):
        call_vllm(q_npu, k_npu, v_npu, b_npu, state_bench,
                  sl_npu, ss_npu, g_npu, nat_npu, scale)
    torch.npu.synchronize()
    t0 = torch.npu.Event(enable_timing=True)
    t1 = torch.npu.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        call_vllm(q_npu, k_npu, v_npu, b_npu, state_bench,
                  sl_npu, ss_npu, g_npu, nat_npu, scale)
    t1.record()
    torch.npu.synchronize()
    vllm_us = t0.elapsed_time(t1) * 1000.0 / iters

    print(f"  ours: {ot_us:.1f} us   vllm: {vllm_us:.1f} us   speedup: {vllm_us/ot_us:.2f}x")

    def report(name_a, a, name_b, b_tensor):
        diff = (a - b_tensor).abs()
        maxe = diff.max().item()
        meane = diff.mean().item()
        ok = "PASS" if maxe < rtol else "FAIL"
        print(f"  {name_a} vs {name_b}: max={maxe:.6e} mean={meane:.6e} [{ok}]")
        return maxe < rtol

    all_pass = True
    all_pass &= report("ours  out  ", out_ot,   "golden out", out_gold.float())
    all_pass &= report("vllm  out  ", out_vllm,  "golden out", out_gold.float())
    all_pass &= report("ours  out  ", out_ot,   "vllm   out", out_vllm)
    touched = ssm_state_indices.long()
    all_pass &= report("ours  state", state_ot[touched],   "golden state", state_gold[touched].float())
    all_pass &= report("vllm  state", state_vllm[touched], "golden state", state_gold[touched].float())
    all_pass &= report("ours  state", state_ot[touched],   "vllm   state", state_vllm[touched])
    return all_pass


def profile_ours(dtype, b, mtp, nk, nv, dk, dv, num_cache_slots):
    """Single ops-transformer call for msprof capture."""
    scale = dk ** -0.5
    actual_seq_lengths = torch.ones(b, dtype=torch.int32) * mtp
    T = int(actual_seq_lengths.sum())
    state = torch.rand(num_cache_slots, nv, dv, dk).to(dtype).npu()
    query = torch.nn.functional.normalize(torch.randn(T, nk, dk), dim=-1).to(dtype).npu()
    key   = torch.nn.functional.normalize(torch.randn(T, nk, dk), dim=-1).to(dtype).npu()
    value = torch.randn(T, nv, dv).to(dtype).npu()
    beta  = torch.rand(T, nv).to(dtype).npu()
    g     = torch.rand(T, nv, dtype=torch.float32).npu()
    ssm_state_indices = torch.randperm(num_cache_slots, dtype=torch.int32)[:T].npu()
    num_accepted_tokens = torch.ones(b, dtype=torch.int32).npu()

    call_ops_transformer(query, key, value, beta, state,
                         actual_seq_lengths.npu(), ssm_state_indices,
                         g, num_accepted_tokens, scale)
    torch.npu.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--qwen", action="store_true", help="use Qwen3.5 shapes")
    parser.add_argument("--profile", action="store_true",
                        help="single call to our op (for msprof)")
    args = parser.parse_args()

    torch.npu.set_device(0)
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    rtol_map  = {"fp16": 2e-3, "fp32": 1e-5}

    if args.profile:
        if args.qwen:
            profile_ours(dtype_map[args.dtype], b=1, mtp=1, nk=8, nv=16,
                         dk=128, dv=128, num_cache_slots=444)
        else:
            profile_ours(dtype_map[args.dtype], b=4, mtp=2, nk=4, nv=4,
                         dk=64, dv=64, num_cache_slots=32)
        sys.exit(0)

    ok = True
    if args.qwen:
        ok &= compare("Qwen3.5 decode", dtype_map[args.dtype], rtol_map[args.dtype],
                       b=1, mtp=1, nk=8, nv=16, dk=128, dv=128, num_cache_slots=444)
        ok &= compare("Qwen3.5 mtp=2", dtype_map[args.dtype], rtol_map[args.dtype],
                       b=2, mtp=2, nk=8, nv=16, dk=128, dv=128, num_cache_slots=444)
    else:
        ok &= compare("small", dtype_map[args.dtype], rtol_map[args.dtype],
                       b=4, mtp=2, nk=4, nv=4, dk=64, dv=64, num_cache_slots=32)

    print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)
