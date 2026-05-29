"""
Pytest for chunk_gated_delta_rule_fwd_h on 310P via aclnn ctypes.
"""

import ctypes
import glob
import os

import pytest
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

CHUNK_SIZE = 64


def _find_lib(name):
    search = (
        os.environ.get("LD_LIBRARY_PATH", "").split(":")
        + [
            "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
        ]
        + glob.glob("/usr/local/Ascend/cann-*/opp/vendors/custom_transformer/op_api/lib/")
    )
    for d in search:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name


_acl = ctypes.CDLL(_find_lib("libascendcl.so"))
_nnop = ctypes.CDLL(_find_lib("libnnopbase.so"))
_opapi = ctypes.CDLL(_find_lib("libcust_opapi.so"))
_c = ctypes

_nnop.aclCreateTensor.restype = _c.c_void_p
_nnop.aclCreateTensor.argtypes = [
    _c.POINTER(_c.c_int64),
    _c.c_uint64,
    _c.c_int,
    _c.POINTER(_c.c_int64),
    _c.c_int64,
    _c.c_int,
    _c.POINTER(_c.c_int64),
    _c.c_uint64,
    _c.c_void_p,
]
_nnop.aclDestroyTensor.restype = _c.c_int
_nnop.aclDestroyTensor.argtypes = [_c.c_void_p]
_acl.aclrtSynchronizeStream.restype = _c.c_int
_acl.aclrtSynchronizeStream.argtypes = [_c.c_void_p]
_acl.aclrtMalloc.restype = _c.c_int
_acl.aclrtMalloc.argtypes = [_c.POINTER(_c.c_void_p), _c.c_uint64, _c.c_int]
_acl.aclrtFree.restype = _c.c_int
_acl.aclrtFree.argtypes = [_c.c_void_p]

_getws = _opapi.aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize
_exec = _opapi.aclnnChunkGatedDeltaRuleFwdH
_getws.restype = _c.c_int
_getws.argtypes = [
    _c.c_void_p,
    _c.c_void_p,
    _c.c_void_p,
    _c.c_void_p,
    _c.c_void_p,
    _c.c_void_p,
    _c.c_bool,
    _c.c_int64,
    _c.c_bool,
    _c.c_void_p,
    _c.c_void_p,
    _c.c_bool,
    _c.c_bool,
    _c.c_void_p,
    _c.c_void_p,
    _c.c_void_p,
    _c.POINTER(_c.c_uint64),
    _c.POINTER(_c.c_void_p),
]
_exec.restype = _c.c_int
_exec.argtypes = [_c.c_void_p, _c.c_uint64, _c.c_void_p, _c.c_void_p]

_DTYPE_MAP = {
    torch.float16: 1,
    torch.float32: 0,
    torch.int64: 9,
    torch.bfloat16: 27,
}


def _make_acl_tensor(t):
    if t is None:
        return None
    shape = (_c.c_int64 * len(t.shape))(*t.shape)
    strides = (_c.c_int64 * len(t.stride()))(*t.stride())
    sd = _c.c_int64(t.untyped_storage().nbytes() // t.element_size())
    sb = _c.c_void_p(t.untyped_storage().data_ptr())
    return _nnop.aclCreateTensor(
        shape,
        len(t.shape),
        _DTYPE_MAP[t.dtype],
        strides,
        t.storage_offset(),
        2,
        _c.byref(sd),
        1,
        sb,
    )


def npu_chunk_gdr_fwd_h(k, w, u, g, initial_state=None, chunk_size=64):
    B, Hg, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    NT = (T + chunk_size - 1) // chunk_size

    h_elems = B * HV * NT * K * V
    h_bytes = h_elems * k.element_size()
    h_pad = ((h_bytes + 1023) // 512) * 512
    vn_elems = B * HV * T * V
    vn_bytes = vn_elems * k.element_size()
    vn_pad = ((vn_bytes + 1023) // 512) * 512
    total = h_pad // k.element_size() + vn_pad // k.element_size() + 256
    buf = torch.empty(total, dtype=k.dtype, device=k.device)

    h_out = buf[:h_elems].view(B, HV, NT, K, V)
    vn_out = buf[h_pad // k.element_size() :][:vn_elems].view(B, HV, T, V)
    fs_out = buf[h_pad // k.element_size() + vn_pad // k.element_size() :][:1]

    stream = _c.c_void_p(torch_npu.npu.current_stream().npu_stream)
    _acl.aclrtSynchronizeStream(stream)

    ws_size, executor = _c.c_uint64(0), _c.c_void_p(None)
    ret = _getws(
        _make_acl_tensor(k),
        _make_acl_tensor(w),
        _make_acl_tensor(u),
        _make_acl_tensor(g),
        None,
        _make_acl_tensor(initial_state),
        False,
        chunk_size,
        True,
        None,
        None,
        False,
        False,
        _make_acl_tensor(h_out),
        _make_acl_tensor(vn_out),
        _make_acl_tensor(fs_out),
        _c.byref(ws_size),
        _c.byref(executor),
    )
    assert ret == 0, f"GetWorkspaceSize failed: {ret}"

    ws_ptr = _c.c_void_p(None)
    if ws_size.value > 0:
        _acl.aclrtMalloc(_c.byref(ws_ptr), ws_size, 0)

    ret = _exec(ws_ptr, ws_size, executor, stream)
    assert ret == 0, f"Execute failed: {ret}"
    _acl.aclrtSynchronizeStream(stream)

    if ws_ptr.value:
        _acl.aclrtFree(ws_ptr)

    return h_out, vn_out


def cpu_reference(k, w, u, g, initial_state=None, chunk_size=64):
    k, w, u, g = k.float(), w.float(), u.float(), g.float()
    B, Hg, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    NT = T // chunk_size
    h = initial_state.float().clone() if initial_state is not None else torch.zeros(B, HV, K, V)
    h_chunks = [h.clone()]
    v_new = torch.zeros_like(u)

    for c in range(NT):
        t0 = c * chunk_size
        W_chunk = w[:, :, t0 : t0 + chunk_size, :]
        ws = torch.einsum("bhik,bhkv->bhiv", W_chunk, h)
        g_chunk = g[:, :, t0 : t0 + chunk_size]
        v_update = torch.zeros(B, HV, chunk_size, V)
        for i in range(chunk_size):
            gi_cum = g_chunk[:, :, -1] - g_chunk[:, :, i]
            vn = u[:, :, t0 + i, :] - ws[:, :, i, :]
            v_new[:, :, t0 + i, :] = vn
            v_update[:, :, i, :] = gi_cum.unsqueeze(-1).exp() * vn
        K_chunk = k[:, :, t0 : t0 + chunk_size, :]
        h_work = torch.einsum("bhik,bhiv->bhkv", K_chunk, v_update)
        h = h * g_chunk[:, :, -1:].unsqueeze(-1).exp() + h_work
        h_chunks.append(h.clone())
    return h_chunks, v_new


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


class TestChunkGatedDeltaRuleFwdH310:
    """chunk_gated_delta_rule_fwd_h kernel correctness on Ascend 310P."""

    @pytest.mark.parametrize(
        "B,Hg,HV,T,K,V",
        [
            (1, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
        ],
    )
    def test_h_state_correctness(self, B, Hg, HV, T, K, V):
        torch.manual_seed(42)
        DTYPE = torch.float16
        k = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        w = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        u = torch.randn(B, HV, T, V, dtype=DTYPE) * 0.1
        g = (-torch.rand(B, HV, T) * 0.1).float()
        init = torch.randn(B, HV, K, V, dtype=DTYPE) * 0.01

        h_ref, _ = cpu_reference(k, w, u, g, init, CHUNK_SIZE)
        h_out, _ = npu_chunk_gdr_fwd_h(
            k.npu(),
            w.npu(),
            u.npu(),
            g.npu(),
            initial_state=init.npu(),
            chunk_size=CHUNK_SIZE,
        )
        h_npu = h_out.cpu().float()
        NT = T // CHUNK_SIZE

        for c in range(min(NT + 1, h_npu.shape[2])):
            ref = h_ref[c].flatten()
            npu = h_npu[0, :, c].flatten()
            cos = cosine(npu, ref)
            assert cos >= 0.99, f"h[{c}] cos={cos:.6f} too low"

    @pytest.mark.parametrize(
        "B,Hg,HV,T,K,V",
        [
            (1, 1, 1, 128, 128, 128),
            (1, 2, 2, 128, 128, 128),
        ],
    )
    def test_v_new_correctness(self, B, Hg, HV, T, K, V):
        torch.manual_seed(42)
        DTYPE = torch.float16
        k = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        w = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
        u = torch.randn(B, HV, T, V, dtype=DTYPE) * 0.1
        g = (-torch.rand(B, HV, T) * 0.1).float()
        init = torch.randn(B, HV, K, V, dtype=DTYPE) * 0.01

        _, vn_ref = cpu_reference(k, w, u, g, init, CHUNK_SIZE)
        _, vn_out = npu_chunk_gdr_fwd_h(
            k.npu(),
            w.npu(),
            u.npu(),
            g.npu(),
            initial_state=init.npu(),
            chunk_size=CHUNK_SIZE,
        )
        vn_npu = vn_out.cpu().float()
        NT = T // CHUNK_SIZE

        for c in range(NT):
            t0, t1 = c * CHUNK_SIZE, (c + 1) * CHUNK_SIZE
            ref = vn_ref[:, :, t0:t1].flatten()
            npu = vn_npu[:, :, t0:t1].flatten()
            cos = cosine(npu, ref)
            assert cos >= 0.99, f"v_new chunk {c} cos={cos:.6f} too low"

    def test_no_nan(self):
        torch.manual_seed(42)
        B, Hg, HV, T, K, V = 1, 1, 1, 128, 128, 128
        DTYPE = torch.float16
        k = torch.randn(B, Hg, T, K, dtype=DTYPE).npu() * 0.1
        w = torch.randn(B, Hg, T, K, dtype=DTYPE).npu() * 0.1
        u = torch.randn(B, HV, T, V, dtype=DTYPE).npu() * 0.1
        g = (-torch.rand(B, HV, T).float()).npu() * 0.1
        init = torch.randn(B, HV, K, V, dtype=DTYPE).npu() * 0.01

        h_out, vn_out = npu_chunk_gdr_fwd_h(k, w, u, g, initial_state=init, chunk_size=CHUNK_SIZE)

        assert torch.isnan(h_out.cpu()).sum() == 0, "h_out has NaN"
        assert torch.isnan(vn_out.cpu()).sum() == 0, "vn_out has NaN"
        assert torch.isinf(h_out.cpu()).sum() == 0, "h_out has Inf"
        assert torch.isinf(vn_out.cpu()).sum() == 0, "vn_out has Inf"
