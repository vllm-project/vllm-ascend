"""Test V310 kernel via ctypes API against golden CPU reference."""

import ctypes
import os

import pytest
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

_CANN = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
_CUST = f"{_CANN}/opp/vendors/custom_transformer/op_api/lib"
_LIB_PATHS = [_CUST, f"{_CANN}/lib64", f"{_CANN}/aarch64-linux/lib64"]


def _find_lib(name, paths):
    for p in paths:
        full = os.path.join(p, name)
        if os.path.exists(full):
            return full
    return name


_acl = ctypes.CDLL(_find_lib("libnnopbase.so", _LIB_PATHS))
_opapi = ctypes.CDLL(_find_lib("libcust_opapi.so", _LIB_PATHS))

_acl.aclCreateTensor.restype = ctypes.c_void_p
_acl.aclCreateTensor.argtypes = [
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_uint64,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int64,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_uint64,
    ctypes.c_void_p,
]
_acl.aclDestroyTensor.argtypes = [ctypes.c_void_p]

_DTYPE_MAP = {torch.float16: 1, torch.float32: 0, torch.int32: 3}


def mk(t):
    if t is None:
        return None
    shape, strides, ndim = list(t.shape), list(t.stride()), len(t.shape)
    return ctypes.c_void_p(
        _acl.aclCreateTensor(
            (ctypes.c_int64 * ndim)(*shape),
            ndim,
            _DTYPE_MAP[t.dtype],
            (ctypes.c_int64 * ndim)(*strides),
            ctypes.c_int64(0),
            2,
            (ctypes.c_int64 * ndim)(*shape),
            ndim,
            ctypes.c_void_p(t.data_ptr()),
        )
    )


def call_v310(query, key, value, beta, state, seq_lens, indices, g, nat, scale):
    out = torch.empty_like(value)
    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p(0)
    ret = _opapi.aclnnRecurrentGatedDeltaRuleV310GetWorkspaceSize(
        mk(query),
        mk(key),
        mk(value),
        mk(beta),
        mk(state),
        mk(seq_lens),
        mk(indices),
        mk(g),
        None,
        mk(nat),
        ctypes.c_float(scale),
        mk(out),
        ctypes.byref(ws_size),
        ctypes.byref(executor),
    )
    assert ret == 0, f"GetWorkspaceSize failed: {ret}"
    ws_ptr = ctypes.c_void_p(0)
    if ws_size.value > 0:
        ws = torch.empty(ws_size.value, dtype=torch.uint8, device=query.device)
        ws_ptr = ctypes.c_void_p(ws.data_ptr())
    stream = torch.npu.current_stream().npu_stream
    ret = _opapi.aclnnRecurrentGatedDeltaRuleV310(ws_ptr, ws_size, executor, ctypes.c_void_p(stream))
    assert ret == 0, f"Execute failed: {ret}"
    torch.npu.synchronize()
    return out


def golden(query, key, value, state, beta, scale, seq_lens, indices, g, nat):
    k = key.float()
    q = query.float()
    v = value.float()
    S = state.clone().float()
    T, nv, Dv = v.shape
    nk = q.shape[1]
    g_f = torch.ones(T, nv) if g is None else g.float().exp()
    beta_f = beta.float()
    o = torch.empty_like(v, dtype=torch.float32)
    q = q * scale
    seq_start = 0
    for i in range(len(seq_lens)):
        init_idx = indices[seq_start + nat[i] - 1] if nat is not None else indices[seq_start]
        for head in range(nv):
            s = S[init_idx][head].clone()
            for t in range(seq_start, seq_start + seq_lens[i]):
                qi = q[t][head // (nv // nk)]
                ki = k[t][head // (nv // nk)]
                vi = v[t][head]
                s = s * g_f[t][head]
                x = (s * ki.unsqueeze(-2)).sum(dim=-1)
                y = (vi - x) * beta_f[t][head]
                s = s + y[:, None] * ki[None, :]
                S[indices[t]][head] = s
                o[t][head] = (s * qi.unsqueeze(-2)).sum(dim=-1)
        seq_start += seq_lens[i]
    return o, S


@pytest.mark.parametrize(
    "batch_size,mtp,nk,nv,dk,dv,num_slots",
    [
        (1, 1, 8, 16, 128, 128, 444),
        (2, 2, 8, 16, 128, 128, 444),
        (4, 2, 4, 4, 64, 64, 32),
    ],
)
def test_recurrent_gated_delta_rule_v310(batch_size, mtp, nk, nv, dk, dv, num_slots):
    torch.manual_seed(42)
    scale = dk**-0.5
    seq_lens = torch.ones(batch_size, dtype=torch.int32) * mtp
    T = int(seq_lens.sum())
    state = torch.rand(num_slots, nv, dv, dk, dtype=torch.float16)
    indices = torch.randperm(num_slots, dtype=torch.int32)[:T]
    nat = torch.ones(batch_size, dtype=torch.int32)
    query = torch.nn.functional.normalize(torch.randn(T, nk, dk), dim=-1).to(torch.float16)
    key = torch.nn.functional.normalize(torch.randn(T, nk, dk), dim=-1).to(torch.float16)
    value = torch.randn(T, nv, dv, dtype=torch.float16)
    beta = torch.rand(T, nv, dtype=torch.float16)
    g = torch.rand(T, nv, dtype=torch.float32)

    out_gold, state_gold = golden(query, key, value, state, beta, scale, seq_lens, indices, g, nat)

    state_npu = state.clone().npu()
    out_npu = call_v310(
        query.npu(),
        key.npu(),
        value.npu(),
        beta.npu(),
        state_npu,
        seq_lens.npu(),
        indices.npu(),
        g.npu(),
        nat.npu(),
        scale,
    )

    touched = indices.long()
    torch.testing.assert_close(
        out_npu.float().cpu(),
        out_gold,
        rtol=3e-3,
        atol=2e-3,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.float().cpu()[touched],
        state_gold.float()[touched],
        rtol=3e-3,
        atol=2e-3,
        equal_nan=True,
    )
