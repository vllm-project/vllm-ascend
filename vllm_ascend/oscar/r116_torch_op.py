"""R116 Phase-4: formal torch.library mutating-schema wrapper for the native
aclnn OscarInt2CacheUpdate (F1).

Contract (R116 task §Phase4.6):
  - torch.ops.oscar.int2_cache_update_v2 with EXPLICIT mutation of the six
    persistent cache tensors (mutates_args) — no hidden aliasing;
  - Meta/Fake implementation registered (compile/fx-safe);
  - fail-closed: extension/vendor must be PRELOADED before any ACLGraph
    capture; the wrapper never JIT-builds inside a capture region;
  - the underlying binding is the R115-proven two-phase aclnn (stream(false),
    cached ws, immediate aclDestroyTensor, QUEUE_WRAP fail-closed OFF).

Default OFF at the overlay call site (R116_TORCHOP=1 opts in).
"""

import os

import torch

_MOD = None
_PRELOADED = False

# both artifacts are resolved through env vars / the workdir layout — no
# machine-specific absolute fallbacks
_R116_KNOWN_CPP = os.environ.get("R116_BINDING_FALLBACK_SRC", "")
_R116_KNOWN_VENDOR = os.environ.get("R116_BINDING_FALLBACK_VENDOR", "")


def _binding_source():
    W = os.environ.get("R116_WORKDIR", os.environ.get("R116_FALLBACK_WORKDIR", ""))
    cands = [
        os.environ.get("R116_BINDING_SRC", ""),
        W + "/TASK_OWNED/r116_cu_aclnn.cpp",
        _R116_KNOWN_CPP,
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "r116 binding source not found: set R116_BINDING_SRC or R116_WORKDIR (layout TASK_OWNED/r116_cu_aclnn.cpp)"
    )


def _vendor_dir():
    W = os.environ.get("R116_WORKDIR", os.environ.get("R116_FALLBACK_WORKDIR", ""))
    cands = [
        os.environ.get("R116_VENDOR_DIR", ""),
        W + "/TASK_OWNED/custom_opp_r116v2/vendors/oscar_r115_min",
        _R116_KNOWN_VENDOR,
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "op_api/lib/libcust_opapi.so")):
            return c
    raise FileNotFoundError(
        "r116 vendor dir not found: set R116_VENDOR_DIR or R116_WORKDIR "
        "(layout TASK_OWNED/custom_opp_r116v2/vendors/oscar_r115_min)"
    )


def preload(device=None):
    """Build+load the pybind extension NOW (engine-load phase / warmup).
    Idempotent; MUST be called before ACLGraph capture begins."""
    global _MOD, _PRELOADED
    if _MOD is not None:
        _PRELOADED = True
        return _MOD
    import ctypes
    import importlib

    V = _vendor_dir()
    # resolve the CANN root from the standard env (falls back to the toolkit
    # symlink); the versioned hard-code broke on other CANN installs
    cann = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
    ctypes.CDLL(os.path.join(cann, "lib64", "libopapi.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(V + "/op_api/lib/libcust_opapi.so", mode=ctypes.RTLD_GLOBAL)
    import torch  # noqa

    _tn = os.path.dirname(importlib.import_module("torch_npu").__file__ or "")
    # Serialize the (idempotent-per-process) JIT build across TP ranks: two
    # ranks compiling the same extension concurrently race on the ninja
    # build directory and can corrupt or fail the build.
    from filelock import FileLock
    from torch.utils import cpp_extension as _cext

    _lock = FileLock(os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp"), "r116_cu_aclnn.build.lock"))
    with _lock:
        _MOD = _cext.load(
            name="r116_cu_aclnn",
            sources=[os.environ.get("R116_BINDING_SRC") or _binding_source()],
            extra_include_paths=[
                os.path.join(_tn, "include"),
                os.path.join(cann, "aarch64-linux/include"),
                V + "/op_api/include",
            ],
            extra_ldflags=[
                "-L" + V + "/op_api/lib",
                "-lcust_opapi",
                "-L" + os.path.join(cann, "lib64"),
                "-lascendcl",
                "-L" + os.path.join(_tn, "lib"),
                "-ltorch_npu",
                "-Wl,-rpath," + V + "/op_api/lib",
            ],
            verbose=False,
        )
    _PRELOADED = True
    return _MOD


def _impl(
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ks_cache: torch.Tensor,
    kz_cache: torch.Tensor,
    vs_cache: torch.Tensor,
    vz_cache: torch.Tensor,
) -> None:
    if _MOD is None:
        # fail-closed: no lazy JIT here — capture regions must never build.
        raise RuntimeError(
            "[R116] oscar.int2_cache_update_v2 used before preload(); "
            "call oscar_preload() during engine load (fail-closed)"
        )
    rc, _ws, lr, _c, _h, _ge, _re = _MOD.cache_update_aclnn(
        k, v, slots, k_cache, v_cache, ks_cache, kz_cache, vs_cache, vz_cache
    )
    if rc != 0 or lr != 0:
        raise RuntimeError(f"[R116] int2_cache_update_v2 rc={rc} lr={lr}")


def _meta(
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    ks_cache: torch.Tensor,
    kz_cache: torch.Tensor,
    vs_cache: torch.Tensor,
    vz_cache: torch.Tensor,
) -> None:
    return None


def register():
    import torch

    if hasattr(torch.ops.oscar, "int2_cache_update_v2"):
        return torch.ops.oscar.int2_cache_update_v2
    try:
        op = torch.library.custom_op(
            "oscar::int2_cache_update_v2",
            mutates_args=("k_cache", "v_cache", "ks_cache", "kz_cache", "vs_cache", "vz_cache"),
            device_types="npu",
        )(_impl)
        op.register_fake(_meta)
    except Exception:
        # fallback registration style for torch builds without custom_op kw support
        torch.library.define(
            "oscar::int2_cache_update_v2",
            "(Tensor k, Tensor v, Tensor slots, Tensor(a!) k_cache, Tensor(b!) v_cache, "
            "Tensor(c!) ks_cache, Tensor(d!) kz_cache, Tensor(e!) vs_cache, Tensor(f!) vz_cache) -> ()",
        )
        torch.library.impl("oscar::int2_cache_update_v2", "npu", _impl, with_keys=True)
        torch.library.impl("oscar::int2_cache_update_v2", "Meta", _meta, with_keys=True)
    return torch.ops.oscar.int2_cache_update_v2
