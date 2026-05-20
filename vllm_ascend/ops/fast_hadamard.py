import math
import os
import subprocess
from pathlib import Path

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.utils import enable_custom_op

BLOCK_DIM = 20  # hard-coded to 910B4 vector core number
DYNAMIC_QUANT_BLOCK_DIM = int(os.environ.get("VLLM_ASCEND_FHT_DYNAMIC_QUANT_BLOCK_DIM", "20"))
ELEMENTS_PER_TILE = (32 * 1024) // 2
# The PTO kernel now has compile-time batched dispatch for N>=64 and
# leverages sub-block parallelism internally (blockDim * 2).  The
# MIN_JIT_N guard remains as a safety valve for very small shapes that
# may still trigger UB alignment issues on some toolkit builds.
MIN_JIT_N = int(os.environ.get("VLLM_ASCEND_FHT_MIN_JIT_N", "64"))
_FAST_HADAMARD_JIT_FUNC = None
_FAST_HADAMARD_DYNAMIC_QUANT_JIT_FUNC = None
FAST_HADAMARD_SO_ENV = "VLLM_ASCEND_FAST_HADAMARD_SO"
FAST_HADAMARD_DYNAMIC_QUANT_SO_ENV = "VLLM_ASCEND_FAST_HADAMARD_DYNAMIC_QUANT_SO"


def _get_current_npu_stream_ptr():
    return torch.npu.current_stream().npu_stream


def _torch_to_ctypes(tensor: torch.Tensor):
    import ctypes

    return ctypes.c_void_p(tensor.data_ptr())


def _check_power_of_two(n: int, name: str):
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"{name} must be power of two")


def fast_hadamard_pto_ref_inplace(x: torch.Tensor) -> torch.Tensor:
    """Reference FHT matching the PTO fast-hadamard layout.

    Mirrors pto-kernels/examples/jit_cpp/fast-hadamard/run_hadamard.py.
    Keeps dtype/device and returns a new tensor (same dtype as input).
    """
    if x.numel() == 0:
        return x.clone()

    n = x.shape[-1]
    _check_power_of_two(n, "last dim")
    log2_n = int(math.log2(n))

    x = x.clone()
    n_half = n // 2
    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def fast_hadamard_pto(x: torch.Tensor, batch: int, n: int, log2_n: int, block_dim: int | None = None) -> None:
    """In-place FHT with PTO-style signature.

    The input tensor x is modified in-place to match run_hadamard.py behavior.
    """
    _ = block_dim
    if x.numel() == 0:
        return
    if x.shape[-1] != n:
        raise ValueError("fast_hadamard_pto: last dim mismatch")
    if x.shape[0] != batch:
        raise ValueError("fast_hadamard_pto: batch mismatch")
    _check_power_of_two(n, "n")
    if 2**log2_n != n:
        raise ValueError("fast_hadamard_pto: log2_n mismatch")

    n_half = n // 2
    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd


def _get_toolkit_home():
    toolkit = os.environ.get("ASCEND_TOOLKIT_HOME")
    if not toolkit:
        raise RuntimeError("ASCEND_TOOLKIT_HOME is not set")
    return toolkit


def _get_pto_lib_path():
    toolkit = _get_toolkit_home()
    return os.environ.get("PTO_LIB_PATH", toolkit)


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
    output_name: str = "fast_hadamard_jit.so",
) -> str:
    lib_path = os.path.join(os.path.dirname(kernel_cpp), output_name)
    if _is_cached_lib_fresh(kernel_cpp, lib_path):
        if verbose:
            print(f"reusing cached {lib_path}")
        return lib_path
    pto_lib_path = _get_pto_lib_path()

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "--cce-soc-version=Ascend910B4",
        "--cce-soc-core-type=VecCore",
        f"-I{pto_lib_path}/include",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def load_lib(lib_path):
    import ctypes

    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    # call_kernel(blockDim, stream, x, batch, n, log2_n)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x (in-place)
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n
        ctypes.c_uint32,  # log2_n
    ]
    lib.call_kernel.restype = None

    default_block_dim = BLOCK_DIM

    def hadamard_func(x, batch, n, log2_n, block_dim=default_block_dim, stream_ptr=None):
        # Runtime guard: fall back to Python reference for shapes known to be
        # fragile on vector-core execution.
        if n < MIN_JIT_N:
            fast_hadamard_pto(x, batch, n, log2_n, block_dim=block_dim)
            return

        if stream_ptr is None:
            stream_ptr = _get_current_npu_stream_ptr()
        lib.call_kernel(
            block_dim,
            stream_ptr,
            _torch_to_ctypes(x),
            batch,
            n,
            log2_n,
        )

    return hadamard_func


def jit_compile(src_path, verbose=True, clean_up=False):
    lib_path = compile_cpp(src_path, verbose=verbose)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func


def ensure_fast_hadamard_shared_object() -> str:
    kernel_cpp = os.path.join(os.path.dirname(__file__), "fast_hadamard_pto-isa.cpp")
    lib_path = compile_cpp(kernel_cpp, verbose=False, timeout=120, output_name="fast_hadamard_jit.so")
    os.environ[FAST_HADAMARD_SO_ENV] = lib_path
    return lib_path


def _get_fast_hadamard_jit_func():
    global _FAST_HADAMARD_JIT_FUNC
    if _FAST_HADAMARD_JIT_FUNC is not None:
        return _FAST_HADAMARD_JIT_FUNC
    kernel_cpp = os.path.join(os.path.dirname(__file__), "fast_hadamard_pto-isa.cpp")
    _FAST_HADAMARD_JIT_FUNC = jit_compile(kernel_cpp, verbose=False, clean_up=False)
    return _FAST_HADAMARD_JIT_FUNC


def _pack_signed_int4_nibbles(values: torch.Tensor) -> torch.Tensor:
    q = values.to(torch.int32)
    low = torch.bitwise_and(q[..., 0::2], 0xF)
    high = torch.bitwise_and(q[..., 1::2], 0xF)
    packed = torch.bitwise_or(low, torch.bitwise_left_shift(high, 4))
    packed = packed.to(torch.int16)
    packed = torch.where(packed >= 128, packed - 256, packed)
    return packed.to(torch.int8)


def _pack_signed_int4_words(values: torch.Tensor) -> torch.Tensor:
    packed_bytes = _pack_signed_int4_nibbles(values).contiguous()
    if packed_bytes.shape[-1] % 4 != 0:
        raise ValueError("last dim must be divisible by 8 to pack int4 activations into int32 words")
    return packed_bytes.view(torch.int32)


def _fast_hadamard_dynamic_quant_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 1:
        raise ValueError("fast_hadamard_dynamic_quant_last_dim expects rank >= 1")
    n = x.shape[-1]
    _check_power_of_two(n, "last dim")
    if n % 8 != 0:
        raise ValueError("fast_hadamard_dynamic_quant_last_dim expects last dim divisible by 8")
    rows = x.reshape(-1, n).to(torch.float32)
    transformed = fast_hadamard_pto_ref_inplace(rows).to(torch.float32) / math.sqrt(float(n))
    max_abs = transformed.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 7.0, min=1e-6)
    quant = torch.round(transformed / scale).clamp(-8, 7).to(torch.int8)
    packed = _pack_signed_int4_words(quant)
    return packed.reshape(*x.shape[:-1], n // 8), scale.to(torch.float32).reshape(*x.shape[:-1], 1)


def _fast_hadamard_dynamic_quant_blockwise_ref(
    x: torch.Tensor,
    hadamard_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 1:
        raise ValueError("fast_hadamard_dynamic_quant_blockwise_last_dim expects rank >= 1")
    full_n = x.shape[-1]
    _check_power_of_two(hadamard_n, "hadamard_n")
    if full_n % hadamard_n != 0:
        raise ValueError("last dim must be divisible by hadamard_n")
    if full_n % 8 != 0:
        raise ValueError("fast_hadamard_dynamic_quant_blockwise_last_dim expects last dim divisible by 8")
    if full_n == hadamard_n:
        return _fast_hadamard_dynamic_quant_ref(x)

    num_blocks = full_n // hadamard_n
    rows = x.reshape(-1, num_blocks, hadamard_n).to(torch.float32)
    transformed = fast_hadamard_pto_ref_inplace(rows.reshape(-1, hadamard_n))
    transformed = transformed.to(torch.float32) / math.sqrt(float(hadamard_n))
    transformed = transformed.reshape(-1, full_n)
    max_abs = transformed.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 7.0, min=1e-6)
    quant = torch.round(transformed / scale).clamp(-8, 7).to(torch.int8)
    packed = _pack_signed_int4_words(quant)
    return packed.reshape(*x.shape[:-1], full_n // 8), scale.to(torch.float32).reshape(*x.shape[:-1], 1)


def _load_dynamic_quant_lib(lib_path: str):
    import ctypes

    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)
    lib.call_dynamic_quant_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x scratch (fp16, in-place hadamard)
        ctypes.c_void_p,  # y packed int4 output
        ctypes.c_void_p,  # row_scales (fp32 output)
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # full_n
        ctypes.c_uint32,  # hadamard_n
        ctypes.c_uint32,  # log2_hadamard_n
        ctypes.c_float,  # inv_sqrt_hadamard_n
    ]
    lib.call_dynamic_quant_kernel.restype = None

    def fused_kernel_func(
        x,
        packed,
        row_scales,
        batch,
        full_n,
        hadamard_n,
        log2_hadamard_n,
        block_dim=DYNAMIC_QUANT_BLOCK_DIM,
        stream_ptr=None,
    ):
        if stream_ptr is None:
            stream_ptr = _get_current_npu_stream_ptr()
        lib.call_dynamic_quant_kernel(
            block_dim,
            stream_ptr,
            _torch_to_ctypes(x),
            _torch_to_ctypes(packed),
            _torch_to_ctypes(row_scales),
            batch,
            full_n,
            hadamard_n,
            log2_hadamard_n,
            1.0 / math.sqrt(float(hadamard_n)),
        )

    return fused_kernel_func


def _fast_hadamard_dynamic_quant_custom_op_impl(
    x: torch.Tensor,
    hadamard_n: int,
    log2_hadamard_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_n = x.shape[-1]
    batch = x.shape[0]
    packed = torch.empty((batch, full_n // 2), dtype=torch.int8, device=x.device)
    scale_out = torch.empty((batch,), dtype=torch.float32, device=x.device)
    _ensure_fast_hadamard_dynamic_quant_ready_for_dispatch()
    fused_func = _get_fast_hadamard_dynamic_quant_jit_func()
    fused_func(x, packed, scale_out, batch, full_n, hadamard_n, log2_hadamard_n)
    return packed, scale_out


def _fast_hadamard_dynamic_quant_custom_op_fake(
    x: torch.Tensor,
    hadamard_n: int,
    log2_hadamard_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = hadamard_n, log2_hadamard_n
    batch = x.shape[0]
    full_n = x.shape[-1]
    packed = torch.empty((batch, full_n // 2), dtype=torch.int8, device=x.device)
    scale_out = torch.empty((batch,), dtype=torch.float32, device=x.device)
    return packed, scale_out


direct_register_custom_op(
    op_name="fast_hadamard_dynamic_quant",
    op_func=_fast_hadamard_dynamic_quant_custom_op_impl,
    fake_impl=_fast_hadamard_dynamic_quant_custom_op_fake,
    mutates_args=["x"],
    dispatch_key="PrivateUse1",
)


def ensure_fast_hadamard_dynamic_quant_shared_object() -> str:
    kernel_cpp = os.path.join(os.path.dirname(__file__), "fast_hadamard_dynamic_quant_pto-isa.cpp")
    lib_path = compile_cpp(
        kernel_cpp,
        verbose=False,
        timeout=120,
        output_name="fast_hadamard_dynamic_quant_jit.so",
    )
    os.environ[FAST_HADAMARD_DYNAMIC_QUANT_SO_ENV] = lib_path
    return lib_path


def _ensure_fast_hadamard_dynamic_quant_ready_for_dispatch() -> None:
    if _FAST_HADAMARD_DYNAMIC_QUANT_JIT_FUNC is not None:
        return
    if os.environ.get(FAST_HADAMARD_DYNAMIC_QUANT_SO_ENV):
        return
    if _is_torch_compiling():
        raise RuntimeError(
            f"{FAST_HADAMARD_DYNAMIC_QUANT_SO_ENV} must be preloaded before "
            "torch.compile dispatch for fast_hadamard_dynamic_quant"
        )
    ensure_fast_hadamard_dynamic_quant_shared_object()


def _get_fast_hadamard_dynamic_quant_jit_func():
    global _FAST_HADAMARD_DYNAMIC_QUANT_JIT_FUNC
    if _FAST_HADAMARD_DYNAMIC_QUANT_JIT_FUNC is not None:
        return _FAST_HADAMARD_DYNAMIC_QUANT_JIT_FUNC
    lib_path = os.environ.get(FAST_HADAMARD_DYNAMIC_QUANT_SO_ENV)
    if not lib_path:
        kernel_cpp = os.path.join(os.path.dirname(__file__), "fast_hadamard_dynamic_quant_pto-isa.cpp")
        lib_path = compile_cpp(
            kernel_cpp,
            verbose=False,
            timeout=120,
            output_name="fast_hadamard_dynamic_quant_jit.so",
        )
    _FAST_HADAMARD_DYNAMIC_QUANT_JIT_FUNC = _load_dynamic_quant_lib(lib_path)
    return _FAST_HADAMARD_DYNAMIC_QUANT_JIT_FUNC


def _fast_hadamard_last_dim_impl(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.clone()
    n = x.shape[-1]
    _check_power_of_two(n, "last dim")
    rows = x.reshape(-1, n)
    log2_n = int(math.log2(n))
    if x.device.type != "npu":
        return fast_hadamard_pto_ref_inplace(rows).reshape_as(x) / math.sqrt(float(n))

    pto_rows = rows
    needs_cast_back = False
    if rows.dtype == torch.bfloat16:
        pto_rows = rows.to(torch.float16)
        needs_cast_back = True
    output = pto_rows.clone()
    ensure_fast_hadamard_shared_object()
    hadamard_func = _get_fast_hadamard_jit_func()
    hadamard_func(output, output.shape[0], n, log2_n)
    output = output / math.sqrt(float(n))
    if needs_cast_back:
        output = output.to(rows.dtype)
    return output.reshape_as(x)


def _is_torch_compiling() -> bool:
    is_compiling = getattr(torch.compiler, "is_compiling", None)
    return bool(is_compiling()) if callable(is_compiling) else False


def fast_hadamard_last_dim_custom_op(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "npu":
        return _fast_hadamard_last_dim_impl(x)
    if x.shape[-1] < MIN_JIT_N:
        if _is_torch_compiling():
            n = x.shape[-1]
            return fast_hadamard_pto_ref_inplace(x.reshape(-1, n)).reshape_as(x) / math.sqrt(float(n))
        return _fast_hadamard_last_dim_impl(x)
    if not os.environ.get(FAST_HADAMARD_SO_ENV):
        if _is_torch_compiling():
            raise RuntimeError(
                f"{FAST_HADAMARD_SO_ENV} must be preloaded before torch.compile dispatch for fast_hadamard_last_dim"
            )
        ensure_fast_hadamard_shared_object()
    enable_custom_op()
    if x.dtype == torch.bfloat16:
        out = torch.ops._C_ascend.fast_hadamard_last_dim(x.to(torch.float16))
        return out.to(torch.bfloat16)
    return torch.ops._C_ascend.fast_hadamard_last_dim(x)


def fast_hadamard_dynamic_quant_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.numel() == 0:
        empty_quant = torch.empty(*x.shape[:-1], x.shape[-1] // 8, dtype=torch.int32, device=x.device)
        empty_scale = torch.empty(*x.shape[:-1], 1, dtype=torch.float32, device=x.device)
        return empty_quant, empty_scale
    if x.ndim < 1:
        raise ValueError("fast_hadamard_dynamic_quant_last_dim expects rank >= 1")

    n = x.shape[-1]
    _check_power_of_two(n, "last dim")

    rows = x.reshape(-1, n).contiguous()
    if x.device.type != "npu" or n < MIN_JIT_N or n > ELEMENTS_PER_TILE:
        return _fast_hadamard_dynamic_quant_ref(x)

    kernel_input = rows
    if rows.dtype == torch.bfloat16:
        kernel_input = rows.to(torch.float16)
    elif rows.dtype != torch.float16:
        raise TypeError("fast_hadamard_dynamic_quant_last_dim expects float16 or bfloat16 input")

    scratch = kernel_input.clone()
    packed, scale_out = torch.ops.vllm.fast_hadamard_dynamic_quant(
        scratch,
        n,
        int(math.log2(n)),
    )
    packed = packed.contiguous().view(torch.int32)
    return packed.reshape(*x.shape[:-1], n // 8), scale_out.to(torch.float32).reshape(*x.shape[:-1], 1)


def fast_hadamard_dynamic_quant_blockwise_last_dim(
    x: torch.Tensor,
    hadamard_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.numel() == 0:
        empty_quant = torch.empty(*x.shape[:-1], x.shape[-1] // 8, dtype=torch.int32, device=x.device)
        empty_scale = torch.empty(*x.shape[:-1], 1, dtype=torch.float32, device=x.device)
        return empty_quant, empty_scale
    if x.ndim < 1:
        raise ValueError("fast_hadamard_dynamic_quant_blockwise_last_dim expects rank >= 1")

    full_n = x.shape[-1]
    _check_power_of_two(hadamard_n, "hadamard_n")
    if full_n % hadamard_n != 0:
        raise ValueError("last dim must be divisible by hadamard_n")
    if full_n % 8 != 0:
        raise ValueError("fast_hadamard_dynamic_quant_blockwise_last_dim expects last dim divisible by 8")
    if full_n == hadamard_n:
        return fast_hadamard_dynamic_quant_last_dim(x)

    rows = x.reshape(-1, full_n).contiguous()
    if x.device.type != "npu" or hadamard_n < MIN_JIT_N or hadamard_n > ELEMENTS_PER_TILE:
        return _fast_hadamard_dynamic_quant_blockwise_ref(x, hadamard_n)

    kernel_input = rows
    if rows.dtype == torch.bfloat16:
        kernel_input = rows.to(torch.float16)
    elif rows.dtype != torch.float16:
        raise TypeError("fast_hadamard_dynamic_quant_blockwise_last_dim expects float16 or bfloat16 input")

    scratch = kernel_input.clone()
    packed, scale_out = torch.ops.vllm.fast_hadamard_dynamic_quant(
        scratch,
        hadamard_n,
        int(math.log2(hadamard_n)),
    )
    packed = packed.contiguous().view(torch.int32)
    return packed.reshape(*x.shape[:-1], full_n // 8), scale_out.reshape(*x.shape[:-1], 1)
