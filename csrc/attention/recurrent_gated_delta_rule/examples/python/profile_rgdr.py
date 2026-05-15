"""Single RGDR call for msprof profiling."""
import os, sys, ctypes, torch, torch_npu

torch.npu.set_device(0)

def _find_lib(name, paths):
    for p in paths:
        full = os.path.join(p, name)
        if os.path.exists(full):
            return full
    return name

_CANN = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
_CUST = f"{_CANN}/opp/vendors/custom_transformer/op_api/lib"
_LIB_PATHS = [_CUST, f"{_CANN}/lib64", f"{_CANN}/aarch64-linux/lib64"]

_acl = ctypes.CDLL(_find_lib("libnnopbase.so", _LIB_PATHS))
_opapi = ctypes.CDLL(_find_lib("libcust_opapi.so", _LIB_PATHS))

_acl.aclCreateTensor.restype = ctypes.c_void_p
_acl.aclCreateTensor.argtypes = [
    ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
    ctypes.c_void_p,
]
_acl.aclDestroyTensor.argtypes = [ctypes.c_void_p]

_DTYPE_MAP = {torch.float16: 1, torch.float32: 0, torch.int32: 3}

def mk(t):
    if t is None:
        return None
    shape, strides, ndim = list(t.shape), list(t.stride()), len(t.shape)
    return ctypes.c_void_p(_acl.aclCreateTensor(
        (ctypes.c_int64 * ndim)(*shape), ndim, _DTYPE_MAP[t.dtype],
        (ctypes.c_int64 * ndim)(*strides), ctypes.c_int64(0),
        2, (ctypes.c_int64 * ndim)(*shape), ndim, ctypes.c_void_p(t.data_ptr())))

# Qwen3.5 shapes
b, nk, nv, dk, dv = 1, 8, 16, 128, 128
T = 1
scale = dk ** -0.5

query = torch.randn(T, nk, dk, dtype=torch.float16).npu()
key   = torch.randn(T, nk, dk, dtype=torch.float16).npu()
value = torch.randn(T, nv, dv, dtype=torch.float16).npu()
beta  = torch.rand(T, nv, dtype=torch.float16).npu()
state = torch.rand(444, nv, dv, dk, dtype=torch.float16).npu()
g     = torch.rand(T, nv, dtype=torch.float32).npu()
seq_lens = torch.ones(b, dtype=torch.int32).npu()
indices  = torch.zeros(T, dtype=torch.int32).npu()
nat      = torch.ones(b, dtype=torch.int32).npu()
out      = torch.empty_like(value)

ws_size = ctypes.c_uint64(0)
executor = ctypes.c_void_p(0)
ret = _opapi.aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    mk(query), mk(key), mk(value), mk(beta), mk(state),
    mk(seq_lens), mk(indices), mk(g), None, mk(nat),
    ctypes.c_float(scale), mk(out),
    ctypes.byref(ws_size), ctypes.byref(executor))
assert ret == 0, f"GetWorkspaceSize failed: {ret}"

ws_ptr = ctypes.c_void_p(0)
if ws_size.value > 0:
    ws = torch.empty(ws_size.value, dtype=torch.uint8, device="npu")
    ws_ptr = ctypes.c_void_p(ws.data_ptr())

stream = torch.npu.current_stream().npu_stream
ret = _opapi.aclnnRecurrentGatedDeltaRule(ws_ptr, ws_size, executor, ctypes.c_void_p(stream))
assert ret == 0, f"Execute failed: {ret}"
torch.npu.synchronize()
