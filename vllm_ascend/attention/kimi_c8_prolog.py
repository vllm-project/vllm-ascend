from __future__ import annotations

from functools import partial
from importlib import import_module

import torch


def load_quant_mla_prolog():
    """Bind the installed DSL 0.8 arena-net C8 execution and precheck APIs."""
    try:
        api = import_module("ops.quant_mla_prolog")
        modes = dict(
            w_qr_quant_mode=api.WQrQuantMode.A8W8_MXFP8,
            qkv_quant_mode=api.QkvQuantMode.A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR,
            cache_mode=api.CacheMode.PA_NZ,
            rope_mode=api.RopeMode.NO_ROPE,
            return_qr=False,
        )
        return partial(api.quant_mla_prolog, **modes), partial(api.quant_mla_prolog_precheck, **modes)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Kimi C8 requires the installed arena-net DSL 0.8 API at ops.quant_mla_prolog") from exc


def _to_nz(value):
    import torch_npu

    dtype = torch_npu.float8_e8m0fnu if value.dtype == torch.float8_e8m0fnu else torch_npu.float8_e4m3fn
    return torch_npu.npu_format_cast(value, 29, customize_dtype=dtype)


def copy_prepared_weight(target, source):
    # A5 aclnn copy_ rejects internal formats. DLPack exposes the same physical
    # bytes with an ND descriptor; all Kimi matrix/MX dimensions are aligned.
    if target.device.type in ("npu", "privateuseone"):
        import torch_npu
        from torch.utils.dlpack import from_dlpack, to_dlpack

        if torch_npu.get_npu_format(target) == 29:
            dst = from_dlpack(to_dlpack(target.view(torch.uint8)))
            src = from_dlpack(to_dlpack(source.view(torch.uint8)))
            if dst.data_ptr() != target.data_ptr() or src.data_ptr() != source.data_ptr():
                raise RuntimeError("C8 NZ reload requires zero-copy byte views")
            dst.copy_(src)
            return
    target.copy_(source)


def prepare_quant_mla_prolog_weights(weights):
    """Reuse native [K,O] NZ matrices and prepare allocation-root MX scales."""
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

    if weights.get("quant_type") is not AscendW8A8MXFP8DynamicLinearMethod:
        raise ValueError("C8 Prolog requires W8A8_MXFP8 projections")
    # DSL provider leases require base Tensors, while real model weights
    # can remain Parameters after contiguous(). Detach keeps the same storage.
    wkb = weights["weight_uk"].detach()
    if (
        wkb.ndim != 3
        or not 1 <= wkb.shape[0] <= 128
        or tuple(wkb.shape[1:]) != (128, 512)
        or wkb.dtype != torch.bfloat16
    ):
        raise ValueError("C8 Prolog requires BF16 wkb [1..128,128,512]")
    prepared = {"wkb": wkb.contiguous()}
    for target, source, output, hidden in (
        ("wqa", "dq", 1536, 7168),
        ("wqb", "uq_qr", wkb.shape[0] * 192, 1536),
        ("wkva", "dkv_kr", 576, 7168),
    ):
        weight = weights[f"weight_{source}"].detach()
        scale = weights[f"dequant_scale_w_{source}"].detach()
        if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (hidden, output):
            raise ValueError(f"{target} must be native E4M3 [K,O]=[{hidden},{output}]")
        if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu) or scale.numel() != output * hidden // 32:
            raise ValueError(f"descale_{target} must contain E8M0 bytes for [{output},{hidden // 32}]")
        prepared[target] = _to_nz(weight)
        # Reorder bytes, not FP8 values: E8M0 transpose is not supported by all CANN builds.
        logical = scale.view(torch.uint8).reshape(output, hidden // 64, 2)
        prepared[f"descale_{target}"] = _to_nz(logical.transpose(0, 1).contiguous().view(torch.float8_e8m0fnu))
    return prepared
