import math
import os
import subprocess
from functools import cache
from pathlib import Path

import torch
from vllm.utils.torch_utils import direct_register_custom_op

DEFAULT_BLOCK_DIM = 20
HEAD_DIM = 128
LOG2_HEAD_DIM = 7


def _get_device_block_dim() -> int:
    try:
        from vllm_ascend.platform import NPUPlatform

        return NPUPlatform.num_compute_units(torch.npu.current_device())
    except Exception:
        return DEFAULT_BLOCK_DIM


def _get_toolkit_home():
    toolkit = (
        os.environ.get("PTO_LIB_PATH") or os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    )
    if not toolkit:
        raise RuntimeError("PTO_LIB_PATH, ASCEND_HOME_PATH, or ASCEND_TOOLKIT_HOME must be set")
    return toolkit


def _get_cce_soc_version() -> str:
    soc_version = os.environ.get("SOC_VERSION")
    if not soc_version:
        return "Ascend910B4"

    normalized = soc_version.lower()
    normalized = {
        "910b": "ascend910b1",
        "910c": "ascend910_9392",
        "310p": "ascend310p1",
    }.get(normalized, normalized)
    if normalized.startswith("ascend910b"):
        suffix = normalized.removeprefix("ascend910b").split("-", 1)[0]
        return f"Ascend910B{suffix or '1'}"
    if normalized.startswith("ascend910_"):
        return "Ascend" + normalized.removeprefix("ascend")
    if normalized.startswith("ascend310p"):
        suffix = normalized.removeprefix("ascend310p")
        return f"Ascend310P{suffix or '1'}"
    return soc_version


def _is_cached_lib_fresh(kernel_cpp: str, lib_path: str) -> bool:
    src = Path(kernel_cpp)
    lib = Path(lib_path)
    if not lib.is_file():
        return False
    try:
        return lib.stat().st_mtime >= src.stat().st_mtime
    except FileNotFoundError:
        return False


def compile_cpp(
    kernel_cpp: str,
    verbose: bool = False,
    timeout: int = 120,
    output_name: str = "fast_hadamard_dynamic_quant_int8_jit.so",
) -> str:
    lib_path = os.path.join(os.path.dirname(kernel_cpp), output_name)
    if _is_cached_lib_fresh(kernel_cpp, lib_path):
        if verbose:
            print(f"reusing cached {lib_path}")
        return lib_path

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        f"--cce-soc-version={_get_cce_soc_version()}",
        "--cce-soc-core-type=VecCore",
        f"-I{_get_toolkit_home()}/include",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:", " ".join(command))
    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e
    return lib_path


def fast_hadamard_pto_ref_inplace(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.clone()
    if x.shape[-1] != HEAD_DIM:
        raise ValueError("fast_hadamard_pto_ref_inplace expects last dim 128")

    out = x.clone()
    for _ in range(LOG2_HEAD_DIM):
        even = out[..., 0::2].clone()
        odd = out[..., 1::2].clone()
        out[..., :64] = even + odd
        out[..., 64:] = even - odd
    return out


def _fast_hadamard_dynamic_quant_int8_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 1:
        raise ValueError("fast_hadamard_dynamic_quant_last_dim expects rank >= 1")
    if x.shape[-1] != HEAD_DIM:
        raise ValueError("fast_hadamard_dynamic_quant_last_dim expects last dim 128")

    rows = x.reshape(-1, HEAD_DIM).to(torch.float32)
    transformed = fast_hadamard_pto_ref_inplace(rows).to(torch.float32) / math.sqrt(float(HEAD_DIM))
    max_abs = transformed.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 127.0, min=1e-6)
    quant = torch.round(transformed / scale).clamp(-128, 127).to(torch.int8)
    return quant.reshape(*x.shape[:-1], HEAD_DIM), scale.to(torch.float32).reshape(*x.shape[:-1], 1)


def _load_dynamic_quant_int8_lib(lib_path: str):
    import ctypes

    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)
    lib.call_dynamic_quant_int8_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.call_dynamic_quant_int8_kernel.restype = None

    def fused_kernel_func(x, quant_out, row_scales, batch, block_dim=None, stream_ptr=None):
        if block_dim is None:
            block_dim = _get_device_block_dim()
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream().npu_stream
        lib.call_dynamic_quant_int8_kernel(
            block_dim,
            stream_ptr,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(quant_out.data_ptr()),
            ctypes.c_void_p(row_scales.data_ptr()),
            batch,
        )

    return fused_kernel_func


def _fast_hadamard_dynamic_quant_int8_custom_op_impl(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x.shape[0]
    quant_out = torch.empty((batch, HEAD_DIM), dtype=torch.int8, device=x.device)
    scale_out = torch.empty((batch + 7,), dtype=torch.float32, device=x.device)
    _ensure_fast_hadamard_dynamic_quant_int8_ready_for_dispatch()
    fused_func = _get_fast_hadamard_dynamic_quant_int8_jit_func()
    fused_func(x, quant_out, scale_out, batch)
    return quant_out, scale_out[:batch]


def _fast_hadamard_dynamic_quant_int8_custom_op_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch = x.shape[0]
    return (
        torch.empty((batch, HEAD_DIM), dtype=torch.int8, device=x.device),
        torch.empty((batch,), dtype=torch.float32, device=x.device),
    )


direct_register_custom_op(
    op_name="fast_hadamard_dynamic_quant_int8",
    op_func=_fast_hadamard_dynamic_quant_int8_custom_op_impl,
    fake_impl=_fast_hadamard_dynamic_quant_int8_custom_op_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def ensure_fast_hadamard_dynamic_quant_int8_shared_object() -> str:
    kernel_cpp = os.path.join(os.path.dirname(__file__), "fast_hadamard_dynamic_quant_int8_pto-isa.cpp")
    lib_path = compile_cpp(
        kernel_cpp,
        verbose=False,
        timeout=120,
        output_name="fast_hadamard_dynamic_quant_int8_jit.so",
    )
    return lib_path


def _is_torch_compiling() -> bool:
    is_compiling = getattr(torch.compiler, "is_compiling", None)
    return bool(is_compiling()) if callable(is_compiling) else False


def _ensure_fast_hadamard_dynamic_quant_int8_ready_for_dispatch() -> None:
    if _get_fast_hadamard_dynamic_quant_int8_jit_func.cache_info().currsize > 0:
        return
    if _is_torch_compiling():
        raise RuntimeError("fast_hadamard_dynamic_quant_int8 must be compiled and loaded before torch.compile dispatch")


@cache
def _get_fast_hadamard_dynamic_quant_int8_jit_func():
    return _load_dynamic_quant_int8_lib(ensure_fast_hadamard_dynamic_quant_int8_shared_object())


def fast_hadamard_dynamic_quant_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 1:
        raise ValueError("fast_hadamard_dynamic_quant_last_dim expects rank >= 1")
    if x.shape[-1] != HEAD_DIM:
        raise ValueError("fast_hadamard_dynamic_quant_last_dim expects last dim 128")
    if x.numel() == 0:
        return (
            torch.empty(*x.shape[:-1], HEAD_DIM, dtype=torch.int8, device=x.device),
            torch.empty(*x.shape[:-1], 1, dtype=torch.float32, device=x.device),
        )

    rows = x.reshape(-1, HEAD_DIM).contiguous()
    if x.device.type != "npu":
        return _fast_hadamard_dynamic_quant_int8_ref(x)

    if rows.dtype == torch.bfloat16:
        rows = rows.to(torch.float16)
    elif rows.dtype != torch.float16:
        raise TypeError("fast_hadamard_dynamic_quant_last_dim expects float16 or bfloat16 input")

    quant_out, scale_out = torch.ops.vllm.fast_hadamard_dynamic_quant_int8(rows)
    return quant_out.reshape(*x.shape[:-1], HEAD_DIM), scale_out.to(torch.float32).reshape(*x.shape[:-1], 1)
