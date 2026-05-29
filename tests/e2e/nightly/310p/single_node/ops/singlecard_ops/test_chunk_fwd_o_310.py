import ctypes
import glob
import os

import pytest
import torch
import torch_npu

torch_npu.npu.set_compile_mode(jit_compile=False)

CHUNK_SIZE = 64


def _find_lib(name):
    search = os.environ.get("LD_LIBRARY_PATH", "").split(":") + \
        glob.glob("/usr/local/Ascend/cann-*/opp/vendors/custom_transformer/op_api/lib/")
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
    _c.POINTER(_c.c_int64), _c.c_uint64, _c.c_int,
    _c.POINTER(_c.c_int64), _c.c_int64, _c.c_int,
    _c.POINTER(_c.c_int64), _c.c_uint64, _c.c_void_p]
_acl.aclrtSynchronizeStream.restype = _c.c_int
_acl.aclrtSynchronizeStream.argtypes = [_c.c_void_p]
_acl.aclrtMalloc.restype = _c.c_int
_acl.aclrtMalloc.argtypes = [_c.POINTER(_c.c_void_p), _c.c_uint64, _c.c_int]
_acl.aclrtFree.restype = _c.c_int
_acl.aclrtFree.argtypes = [_c.c_void_p]
_getws = _opapi.aclnnChunkFwdOGetWorkspaceSize
_exec = _opapi.aclnnChunkFwdO
_getws.restype = _c.c_int
_getws.argtypes = [_c.c_void_p] * 5 + [
    _c.c_void_p, _c.c_void_p, _c.c_double, _c.c_int64, _c.c_void_p,
    _c.POINTER(_c.c_uint64), _c.POINTER(_c.c_void_p)]
_exec.restype = _c.c_int
_exec.argtypes = [_c.c_void_p, _c.c_uint64, _c.c_void_p, _c.c_void_p]
_DTYPE_MAP = {torch.float16: 1, torch.float32: 0}


def _make_acl_tensor(t):
    shape = (_c.c_int64 * len(t.shape))(*t.shape)
    strides = (_c.c_int64 * len(t.stride()))(*t.stride())
    sd = _c.c_int64(t.untyped_storage().nbytes() // t.element_size())
    sb = _c.c_void_p(t.untyped_storage().data_ptr())
    return _nnop.aclCreateTensor(
        shape, len(t.shape), _DTYPE_MAP[t.dtype],
        strides, t.storage_offset(), 2,
        _c.byref(sd), 1, sb)


def npu_chunk_fwd_o(q, k, v, h, g, scale):
    o = torch.zeros_like(v)
    stream = _c.c_void_p(torch_npu.npu.current_stream().npu_stream)
    _acl.aclrtSynchronizeStream(stream)
    ws_size, executor = _c.c_uint64(0), _c.c_void_p(None)
    ret = _getws(
        _make_acl_tensor(q), _make_acl_tensor(k), _make_acl_tensor(v),
        _make_acl_tensor(h), _make_acl_tensor(g), None, None,
        _c.c_double(scale), _c.c_int64(CHUNK_SIZE), _make_acl_tensor(o),
        _c.byref(ws_size), _c.byref(executor))
    assert ret == 0, f"GetWorkspaceSize failed: {ret}"
    ws_ptr = _c.c_void_p(None)
    if ws_size.value > 0:
        _acl.aclrtMalloc(_c.byref(ws_ptr), ws_size, 0)
    ret = _exec(ws_ptr, ws_size, executor, stream)
    assert ret == 0, f"Execute failed: {ret}"
    _acl.aclrtSynchronizeStream(stream)
    if ws_ptr.value:
        _acl.aclrtFree(ws_ptr)
    return o


def golden_chunk_fwd_o(q, k, v, h_state, g, scale):
    """CPU fp32 reference.

    Per chunk c (CS tokens starting at t0):
      attn = q[c] @ k[c].T                             [CS, CS]
      gate[i,j] = exp(min(0, g[j] - g[i])) * (j<=i)   [CS, CS]
      attn_masked = attn * gate
      h_work = q[c] @ h_state[c]                       [CS, Dv]
      v_work = attn_masked @ v[c]                       [CS, Dv]
      o[c] = scale * (v_work + exp(g[c]) * h_work)
    """
    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    h_state = h_state.float()
    B, H_k, L, D_k = q.shape
    H_v, D_v = v.shape[1], v.shape[3]
    CS = CHUNK_SIZE
    NT = L // CS
    head_groups = H_v // H_k
    o = torch.zeros(B, H_v, L, D_v)
    for b in range(B):
        for hv in range(H_v):
            hk = hv // head_groups
            for c in range(NT):
                t0 = c * CS
                q_c = q[b, hk, t0:t0 + CS]
                k_c = k[b, hk, t0:t0 + CS]
                v_c = v[b, hv, t0:t0 + CS]
                g_c = g[b, hv, t0:t0 + CS]
                h_c = h_state[b, hv, c * D_k:(c + 1) * D_k]
                attn = q_c @ k_c.T
                g_row = g_c.unsqueeze(1)
                g_col = g_c.unsqueeze(0)
                gate = torch.exp(torch.clamp(g_col - g_row, max=0.0))
                causal = torch.tril(torch.ones(CS, CS))
                attn_masked = attn * gate * causal
                h_work = q_c @ h_c
                v_work = attn_masked @ v_c
                g_exp = torch.exp(g_c).unsqueeze(1)
                o[b, hv, t0:t0 + CS] = scale * (v_work + g_exp * h_work)
    return o


class TestChunkFwdO310:
    """chunk_fwd_o kernel correctness on Ascend 310P."""

    @pytest.mark.parametrize("B,Hk,Hv,L,Dk,Dv", [
        (1, 2, 2, 128, 128, 128),
        (1, 4, 4, 256, 128, 128),
    ])
    def test_constant_inputs(self, B, Hk, Hv, L, Dk, Dv):
        """Constant q=k=v, h=0, g=0 => analytically verifiable output."""
        scale = 1.0 / (Dk ** 0.5)
        NC = L // CHUNK_SIZE
        c = 0.01
        q = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        k = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        v = torch.full((B, Hv, L, Dv), c, dtype=torch.float16).npu()
        h = torch.zeros(B, Hv, NC * Dk, Dv, dtype=torch.float16).npu()
        g = torch.zeros(B, Hv, L, dtype=torch.float32).npu()

        o = npu_chunk_fwd_o(q, k, v, h, g, scale)
        oc = o.cpu().float()

        assert torch.isnan(oc).sum() == 0, "output has NaN"
        assert torch.isinf(oc).sum() == 0, "output has Inf"

        # With constant c, g=0, h=0:
        # attn[i,j] = c^2 * Dk for all i,j
        # gate = 1 (g=0), causal mask => attn_masked[i,j] = c^2*Dk if j<=i
        # v_work[i,:] = sum_{j<=i} c^2*Dk * c = (i+1) * c^3 * Dk
        # h_work = 0, o[i,:] = scale * v_work[i,:]
        attn_val = c * c * Dk
        for i in range(min(CHUNK_SIZE, 8)):
            expected = scale * (i + 1) * attn_val * c
            actual = oc[0, 0, i, 0].item()
            rel_err = abs(actual - expected) / max(abs(expected), 1e-10)
            assert rel_err < 0.10, \
                f"row {i}: actual={actual:.8f} expected={expected:.8f} rel_err={rel_err:.2f}"

    @pytest.mark.parametrize("B,Hk,Hv,L,Dk,Dv", [
        (1, 2, 2, 128, 128, 128),
        (1, 4, 4, 256, 128, 128),
    ])
    def test_random_inputs_no_nan(self, B, Hk, Hv, L, Dk, Dv):
        """Random small inputs: no NaN/Inf in output."""
        torch.manual_seed(42)
        scale = 1.0 / (Dk ** 0.5)
        NC = L // CHUNK_SIZE
        q = (torch.randn(B, Hk, L, Dk) * 0.01).half().npu()
        k = (torch.randn(B, Hk, L, Dk) * 0.01).half().npu()
        v = (torch.randn(B, Hv, L, Dv) * 0.01).half().npu()
        h = (torch.randn(B, Hv, NC * Dk, Dv) * 0.01).half().npu()
        g = torch.randn(B, Hv, L, dtype=torch.float32).npu() * 0.001

        o = npu_chunk_fwd_o(q, k, v, h, g, scale)
        oc = o.cpu().float()

        assert torch.isnan(oc).sum() == 0, "output has NaN"
        assert torch.isinf(oc).sum() == 0, "output has Inf"
        assert oc.abs().max() > 0, "output is all zeros"

    def test_g_zero_reduces_to_standard_attention(self):
        """g=0 => gate=1, so kernel = scale*(causal_attn@v + q@h)."""
        torch.manual_seed(123)
        B, Hk, Hv, L, Dk, Dv = 1, 2, 2, 128, 128, 128
        scale = 1.0 / (Dk ** 0.5)
        NC = L // CHUNK_SIZE
        q = (torch.randn(B, Hk, L, Dk) * 0.01).half()
        k = (torch.randn(B, Hk, L, Dk) * 0.01).half()
        v = (torch.randn(B, Hv, L, Dv) * 0.01).half()
        h = torch.zeros(B, Hv, NC * Dk, Dv, dtype=torch.float16)
        g = torch.zeros(B, Hv, L, dtype=torch.float32)

        o_npu = npu_chunk_fwd_o(q.npu(), k.npu(), v.npu(), h.npu(), g.npu(), scale)
        o_ref = golden_chunk_fwd_o(q, k, v, h, g, scale)

        cos = torch.nn.functional.cosine_similarity(
            o_npu.cpu().float().flatten(),
            o_ref.flatten(), dim=0).item()
        assert cos > 0.999, f"cosine {cos:.4f} too low for g=0 h=0 case"

    def test_chunk_boundary_independence(self):
        """Each chunk should produce the same output for identical data."""
        B, Hk, Hv, L, Dk, Dv = 1, 2, 2, 128, 128, 128
        scale = 1.0 / (Dk ** 0.5)
        NC = L // CHUNK_SIZE
        c = 0.02
        q = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        k = torch.full((B, Hk, L, Dk), c, dtype=torch.float16).npu()
        v = torch.full((B, Hv, L, Dv), c, dtype=torch.float16).npu()
        h = torch.zeros(B, Hv, NC * Dk, Dv, dtype=torch.float16).npu()
        g = torch.zeros(B, Hv, L, dtype=torch.float32).npu()

        o = npu_chunk_fwd_o(q, k, v, h, g, scale).cpu().float()

        chunk0 = o[0, 0, :CHUNK_SIZE, :]
        chunk1 = o[0, 0, CHUNK_SIZE:, :]
        cos = torch.nn.functional.cosine_similarity(
            chunk0.flatten(), chunk1.flatten(), dim=0).item()
        assert cos > 0.999, f"chunks differ: cosine={cos:.6f}"
