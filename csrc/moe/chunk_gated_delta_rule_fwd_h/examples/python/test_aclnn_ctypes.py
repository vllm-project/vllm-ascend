"""
Direct aclnn kernel test via ctypes — no C++ binding needed.
Calls aclnnChunkGatedDeltaRuleFwdH through libcust_opapi.so.

Usage:
  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
  export LD_LIBRARY_PATH=/.../opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH
  python test_aclnn_ctypes.py
"""

import ctypes
import os
import sys

import torch
import torch_npu


# ── Load CANN libs ──────────────────────────────────────────────
def _find_lib(name):
    """Search LD_LIBRARY_PATH + common locations."""
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") + [
        "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
        "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64",
    ]:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return name  # let ctypes try default search


_acl = ctypes.CDLL(_find_lib("libascendcl.so"))
_nnop = ctypes.CDLL(_find_lib("libnnopbase.so"))
_opapi = ctypes.CDLL(_find_lib("libcust_opapi.so"))

# ── ACL type aliases ────────────────────────────────────────────
c_void_p = ctypes.c_void_p
c_int64 = ctypes.c_int64
c_uint64 = ctypes.c_uint64
c_int = ctypes.c_int
c_bool = ctypes.c_bool

# aclCreateTensor is in libnnopbase, not libascendcl
_nnop.aclCreateTensor.restype = c_void_p
_nnop.aclCreateTensor.argtypes = [
    ctypes.POINTER(c_int64),
    c_uint64,
    c_int,
    ctypes.POINTER(c_int64),
    c_int64,
    c_int,
    ctypes.POINTER(c_int64),
    c_uint64,
    c_void_p,
]

# aclDestroyTensor
_nnop.aclDestroyTensor.restype = c_int
_nnop.aclDestroyTensor.argtypes = [c_void_p]

# aclrtSynchronizeStream
_acl.aclrtSynchronizeStream.restype = c_int
_acl.aclrtSynchronizeStream.argtypes = [c_void_p]

# aclrtMalloc / aclrtFree for workspace
_acl.aclrtMalloc.restype = c_int
_acl.aclrtMalloc.argtypes = [ctypes.POINTER(c_void_p), c_uint64, c_int]
_acl.aclrtFree.restype = c_int
_acl.aclrtFree.argtypes = [c_void_p]

# Our kernel
_getws = getattr(_opapi, "aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize", None)
_exec = getattr(_opapi, "aclnnChunkGatedDeltaRuleFwdH", None)
if not _getws or not _exec:
    print("ERROR: aclnn kernel symbols not found. Install the .run package first.", file=sys.stderr)
    sys.exit(1)

# GetWorkspaceSize(k,w,u, g,gk,init, outFS,chunkSz,saveNV, cuSeq,chunkIdx, exp2,transpose, h,vn,fs, wsSize,executor)
_getws.restype = c_int
_getws.argtypes = [
    c_void_p,
    c_void_p,
    c_void_p,  # k, w, u
    c_void_p,
    c_void_p,
    c_void_p,  # g, gk, initial_state
    c_bool,
    c_int64,
    c_bool,  # output_final_state, chunk_size, save_new_value
    c_void_p,
    c_void_p,  # cu_seqlens, chunk_indices
    c_bool,
    c_bool,  # use_exp2, transpose_state_layout
    c_void_p,
    c_void_p,
    c_void_p,  # h_out, v_new_out, final_state_out
    ctypes.POINTER(c_uint64),
    ctypes.POINTER(c_void_p),  # ws_size, executor
]

# Execute(workspace, ws_size, executor, stream)
_exec.restype = c_int
_exec.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]

# ── ACL dtype map ───────────────────────────────────────────────
_DTYPE_MAP = {
    torch.float16: 1,  # ACL_FLOAT16
    torch.float32: 0,  # ACL_FLOAT
    torch.int64: 9,  # ACL_INT64
    torch.int32: 3,  # ACL_INT32
    torch.bfloat16: 27,  # ACL_BF16
}
ACL_FORMAT_ND = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0


def _make_acl_tensor(t):
    """Create an aclTensor* from a PyTorch NPU tensor."""
    if t is None:
        return None
    shape = (c_int64 * len(t.shape))(*t.shape)
    strides = (c_int64 * len(t.stride()))(*t.stride())
    storage_dim = c_int64(t.storage().nbytes() // t.element_size())
    storage_base = c_void_p(t.untyped_storage().data_ptr())
    return _nnop.aclCreateTensor(
        shape,
        len(t.shape),
        _DTYPE_MAP[t.dtype],
        strides,
        t.storage_offset(),
        ACL_FORMAT_ND,
        ctypes.byref(storage_dim),
        1,
        storage_base,
    )


def _get_stream():
    """Get the current NPU stream as a void*."""
    return c_void_p(torch_npu.npu.current_stream().npu_stream)


# ── Kernel wrapper ──────────────────────────────────────────────
def call_kernel(k, w, u, g, initial_state=None, chunk_size=64):
    """Call the aclnn kernel and return (h_out, v_new_out)."""
    B, Hg, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    NT = (T + chunk_size - 1) // chunk_size

    # Contiguous output buffer (310P packs outputs contiguously)
    h_elems = B * HV * NT * K * V
    h_bytes = h_elems * k.element_size()
    h_pad = ((h_bytes + 1023) // 512) * 512  # CANN 310P adds 512B gap between outputs
    vn_elems = B * HV * T * V
    vn_bytes = vn_elems * k.element_size()
    vn_pad = ((vn_bytes + 1023) // 512) * 512
    total = h_pad // k.element_size() + vn_pad // k.element_size() + 256
    buf = torch.empty(total, dtype=k.dtype, device=k.device)

    h_out = buf[:h_elems].view(B, HV, NT, K, V)
    vn_out = buf[h_pad // k.element_size() :][:vn_elems].view(B, HV, T, V)
    fs_out = buf[h_pad // k.element_size() + vn_pad // k.element_size() :][:1]

    # Create aclTensors
    acl_k = _make_acl_tensor(k)
    acl_w = _make_acl_tensor(w)
    acl_u = _make_acl_tensor(u)
    acl_g = _make_acl_tensor(g)
    acl_init = _make_acl_tensor(initial_state)
    acl_h = _make_acl_tensor(h_out)
    acl_vn = _make_acl_tensor(vn_out)
    acl_fs = _make_acl_tensor(fs_out)

    stream = _get_stream()
    _acl.aclrtSynchronizeStream(stream)

    # GetWorkspaceSize
    ws_size = c_uint64(0)
    executor = c_void_p(None)
    ret = _getws(
        acl_k,
        acl_w,
        acl_u,
        acl_g,
        None,
        acl_init,
        False,
        chunk_size,
        True,
        None,
        None,
        False,
        False,
        acl_h,
        acl_vn,
        acl_fs,
        ctypes.byref(ws_size),
        ctypes.byref(executor),
    )
    assert ret == 0, f"GetWorkspaceSize failed: {ret}"

    # Allocate workspace
    ws_ptr = c_void_p(None)
    if ws_size.value > 0:
        ret = _acl.aclrtMalloc(ctypes.byref(ws_ptr), ws_size, ACL_MEM_MALLOC_HUGE_FIRST)
        assert ret == 0, f"aclrtMalloc failed: {ret}"

    # Execute
    ret = _exec(ws_ptr, ws_size, executor, stream)
    assert ret == 0, f"Execute failed: {ret}"
    _acl.aclrtSynchronizeStream(stream)

    # Cleanup
    if ws_ptr.value:
        _acl.aclrtFree(ws_ptr)
    for t in [acl_k, acl_w, acl_u, acl_g, acl_init, acl_h, acl_vn, acl_fs]:
        if t:
            _nnop.aclDestroyTensor(t)

    return h_out, vn_out


# ── CPU reference (matches kernel semantics) ────────────────────
def cpu_reference(k, w, u, g, initial_state=None, chunk_size=64):
    """Kernel does: cube1=w@h, vec1=u-ws, cube2=k^T@v_update, vec2=h*exp(g)+h_work."""
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


# ── Test ────────────────────────────────────────────────────────
def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    torch.manual_seed(42)
    torch_npu.npu.set_device(0)

    B, T, Hg, HV, K, V, CHUNK = 1, 128, 1, 1, 128, 128, 64
    DTYPE = torch.float16

    k = torch.randn(B, Hg, T, K, dtype=DTYPE, device="cpu") * 0.1
    w = torch.randn(B, Hg, T, K, dtype=DTYPE, device="cpu") * 0.1
    u = torch.randn(B, HV, T, V, dtype=DTYPE, device="cpu") * 0.1
    g = (-torch.rand(B, HV, T, device="cpu") * 0.1).float()
    init = torch.randn(B, HV, K, V, dtype=DTYPE, device="cpu") * 0.01

    print("CPU reference...")
    h_ref, vn_ref = cpu_reference(k, w, u, g, init, CHUNK)

    print("NPU kernel (aclnn via ctypes)...")
    h_out, vn_out = call_kernel(
        k.npu(),
        w.npu(),
        u.npu(),
        g.npu(),
        initial_state=init.npu(),
        chunk_size=CHUNK,
    )
    torch.npu.synchronize()
    h_npu = h_out.cpu().float()
    vn_npu = vn_out.cpu().float()

    NT = T // CHUNK
    ok = True
    print(f"\nB={B} T={T} K={K} V={V} chunk={CHUNK}")

    print("\nh (state):")
    for c in range(min(NT + 1, h_npu.shape[2])):
        ref = h_ref[c].flatten()
        npu = h_npu[0, :, c].flatten()
        c_val = cosine(npu, ref)
        mae = (npu - ref).abs().mean().item()
        tag = "init" if c == 0 else f"after chunk {c - 1}"
        passed = c_val >= 0.999
        if not passed:
            ok = False
        print(f"  h[{c}] ({tag}): cos={c_val:.6f} mae={mae:.6f} [{'OK' if passed else 'FAIL'}]")

    print("\nv_new:")
    for c in range(NT):
        t0, t1 = c * CHUNK, (c + 1) * CHUNK
        ref = vn_ref[:, :, t0:t1].flatten()
        npu = vn_npu[:, :, t0:t1].flatten()
        c_val = cosine(npu, ref)
        mae = (npu - ref).abs().mean().item()
        passed = c_val >= 0.999
        if not passed:
            ok = False
        print(f"  chunk {c}: cos={c_val:.6f} mae={mae:.6f} [{'OK' if passed else 'FAIL'}]")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
