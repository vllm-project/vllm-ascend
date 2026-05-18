import torch
import torch_npu
import ctypes
import numpy as np


# ---- aclnn ctypes wrapper (standalone testing without integrated torch_npu build) ----

def _find_lib(name, paths):
    import os
    for p in paths:
        full = os.path.join(p,\ name)
        if os.path.exists(full):
            return full
    return name  # fallback to LD_LIBRARY_PATH

import os
_CANN = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
_CUST = f"{_CANN}/opp/vendors/custom_math/op_api/lib"
_LIB_PATHS = [_CUST, f"{_CANN}/lib64", f"{_CANN}/aarch64-linux/lib64"]

_acl = ctypes.CDLL(_find_lib("libnnopbase.so", _LIB_PATHS))
_opapi = ctypes.CDLL(_find_lib("libcust_opapi.so", _LIB_PATHS))

_TORCH_TO_ACL_DTYPE = {
    torch.float16: 1,   # ACL_FLOAT16
    torch.float32: 0,   # ACL_FLOAT
    torch.int32: 3,     # ACL_INT32
    torch.bfloat16: 27, # ACL_BF16
}
_ACL_FORMAT_ND = 2

_acl.aclCreateTensor.restype = ctypes.c_void_p
_acl.aclCreateTensor.argtypes = [
    ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,  # shape, ndim
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,  # dtype, strides, offset
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,  # format, storageshape, storageNdim
    ctypes.c_void_p,  # data ptr
]
_acl.aclDestroyTensor.argtypes = [ctypes.c_void_p]

def _make_acl_tensor(t):
    """Create an aclTensor* from a torch npu tensor."""
    if t is None:
        return None
    shape = list(t.shape)
    strides = list(t.stride())
    ndim = len(shape)
    c_shape = (ctypes.c_int64 * ndim)(*shape)
    c_strides = (ctypes.c_int64 * ndim)(*strides)
    c_storage = (ctypes.c_int64 * ndim)(*shape)
    acl_dtype = _TORCH_TO_ACL_DTYPE[t.dtype]
    ptr = _acl.aclCreateTensor(
        c_shape, ndim, acl_dtype, c_strides, ctypes.c_int64(0),
        _ACL_FORMAT_ND, c_storage, ndim, ctypes.c_void_p(t.data_ptr()))
    return ctypes.c_void_p(ptr)


def _call_rgdr(query, key, value, beta, state, actual_seq_lengths, ssm_state_indices,
               g, gk, num_accepted_tokens, scale, out):
    """Call aclnnRecurrentGatedDeltaRule via ctypes two-stage API."""
    tensors = []
    def mk(t):
        p = _make_acl_tensor(t)
        if p is not None:
            tensors.append(p)
        return p

    q_t, k_t, v_t = mk(query), mk(key), mk(value)
    b_t, s_t = mk(beta), mk(state)
    sl_t, ss_t = mk(actual_seq_lengths), mk(ssm_state_indices)
    g_t, gk_t = mk(g), mk(gk)
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


def npu_rgdr(query, key, value, beta, state, actual_seq_lengths, ssm_state_indices,
             g=None, gk=None, num_accepted_tokens=None, scale=1.0):
    """Call RecurrentGatedDeltaRule — tries integrated torch_npu first, falls back to ctypes."""
    out = torch.empty_like(value)
    # try integrated build first
    if hasattr(torch_npu, "npu_recurrent_gated_delta_rule"):
        return torch_npu.npu_recurrent_gated_delta_rule(
            query, key, value, state, beta=beta, scale=scale,
            actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
            g=g, num_accepted_tokens=num_accepted_tokens)
    # ctypes fallback
    return _call_rgdr(query, key, value, beta, state, actual_seq_lengths, ssm_state_indices,
                      g, gk, num_accepted_tokens, scale, out)


# ---- golden reference ----

def golden_rgdr(query, key, value, state, beta, scale, actual_seq_lengths, ssm_state_indices, g, num_accepted_tokens):
    k = key.to(torch.float32)
    q = query.to(torch.float32)
    v = value.to(torch.float32)
    initial_state = state.clone().to(torch.float32)
    T, n_heads_v, Dv = v.shape
    n_heads_k = q.shape[-2]
    g = torch.ones(T, n_heads_v).to(torch.float32) if g is None else g.to(torch.float32).exp()

    beta = torch.ones(T, n_heads_v).to(torch.float32) if beta is None else beta.to(torch.float32)
    o = torch.empty_like(v).to(torch.float32)
    if scale is None:
        scale = k.shape[-1]**-0.5
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
                alpha_i = g[slot_id][head_id]
                beta_i = beta[slot_id][head_id]
                S = S * alpha_i
                x = (S * k_i.unsqueeze(-2)).sum(dim=-1)
                y = (v_i - x) * beta_i
                S_ = y[:, None] * k_i[None, :]
                S = S + S_
                initial_state[ssm_state_indices[slot_id]][head_id] = S
                o[slot_id][head_id] = (S * q_i.unsqueeze(-2)).sum(dim=-1)
        seq_start += actual_seq_lengths[i]

    return o.to(query.dtype), initial_state.to(query.dtype)


# ---- test ----

def run_test(dtype, rtol):
    (b, mtp, nk, nv, dk, dv) = (4, 2, 4, 4, 64, 64)

    actual_seq_lengths = (torch.ones(b, dtype=torch.int32) * mtp)
    T = int(torch.sum(actual_seq_lengths))
    state = torch.rand((T, nv, dv, dk)).to(dtype)
    query = torch.nn.functional.normalize(torch.rand((T, nk, dk)), p=2, dim=-1).to(dtype)
    key = torch.nn.functional.normalize(torch.rand((T, nk, dk)), p=2, dim=-1).to(dtype)
    value = torch.rand((T, nv, dv)).to(dtype)
    g = torch.rand((T, nv), dtype=torch.float32)
    beta = torch.rand((T, nv)).to(dtype)
    ssm_state_indices = torch.arange(T, dtype=torch.int32)
    num_accepted_tokens = torch.randint(1, mtp + 1, (b,), dtype=torch.int32)
    scale = 0.5

    out_golden, state_golden = golden_rgdr(query, key, value, state, beta, scale,
                                           actual_seq_lengths, ssm_state_indices, g, num_accepted_tokens)
    out_golden = out_golden.to(torch.float32)
    state_golden = state_golden.to(torch.float32)

    state_npu = state.npu()
    out = torch.empty_like(value).npu()
    _call_rgdr(query.npu(), key.npu(), value.npu(), beta.npu(), state_npu,
               actual_seq_lengths.npu(), ssm_state_indices.npu(),
               g.npu(), None, num_accepted_tokens.npu(), scale, out)
    out = out.to(torch.float32).cpu()

    max_err = (out - out_golden).abs().max().item()
    import sys
    msg = f"  out max_err={max_err:.6f} {'PASS' if max_err < rtol else 'FAIL'}"
    print(msg); sys.stderr.write(msg + "\n"); sys.stderr.flush()
    assert max_err < rtol, f"out max_err {max_err} > {rtol}"


if __name__ == "__main__":
    # ensure device is initialized before ctypes calls
    torch.npu.set_device(0)

    print("=== FP16 (310P) ===")
    run_test(torch.float16, 0.002)
    print("=== FP32 (310P) ===")
    run_test(torch.float32, 1e-5)
    print("ALL PASSED")
