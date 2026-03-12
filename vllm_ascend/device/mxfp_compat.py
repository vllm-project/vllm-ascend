import torch
import torch_npu

# TODO(linfeng): Temporary compatibility shim for MXFP4/MXFP8 because current torch_npu
# releases do not expose the required dtype attributes yet. Simplify or remove this
# file after the torch_npu release in March 2026 includes those dtype symbols.
FLOAT8_E8M0FNU_DTYPE = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
FLOAT4_E2M1FN_X2_DTYPE = getattr(torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None))
HIFLOAT8_DTYPE = getattr(torch_npu, "hifloat8", None)


def _get_missing_symbols(symbols: tuple[str, ...]) -> list[str]:
    return [symbol for symbol in symbols if not hasattr(torch_npu, symbol)]


def _ensure_symbols_available(feature: str, symbols: tuple[str, ...]) -> None:
    missing_symbols = _get_missing_symbols(symbols)
    if not missing_symbols:
        return
    missing_symbols_str = ", ".join(missing_symbols)
    raise RuntimeError(
        f"{feature} requires a newer torch_npu runtime. Missing symbols: {missing_symbols_str}. "
        "Please upgrade torch_npu or disable MXFP quantization."
    )


def ensure_mxfp8_scale_dtype_available(feature: str) -> None:
    _ensure_symbols_available(feature, ("float8_e8m0fnu",))


def ensure_mxfp4_dtype_available(feature: str) -> None:
    _ensure_symbols_available(feature, ("float4_e2m1fn_x2", "float8_e8m0fnu"))


def ensure_mxfp8_linear_available(feature: str) -> None:
    _ensure_symbols_available(feature, ("float8_e8m0fnu", "npu_dynamic_mx_quant", "npu_quant_matmul"))


def ensure_mxfp8_moe_available(feature: str) -> None:
    _ensure_symbols_available(
        feature,
        ("float8_e8m0fnu", "npu_dynamic_mx_quant", "npu_grouped_matmul_swiglu_quant_v2"),
    )
