import importlib

import vllm.model_executor.layers.mamba.ops.causal_conv1d
import vllm.v1.worker.gpu.sample.gumbel

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_recurrent_gated_delta_rule_fwd_kernel
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample as ascend_gumbel_sample


def _try_import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return None
        raise


def _patch_module_attrs(module, attrs: dict[str, object]) -> None:
    if module is None:
        return
    for attr_name, attr_value in attrs.items():
        if hasattr(module, attr_name):
            setattr(module, attr_name, attr_value)


def _patch_kda_ops() -> None:
    kda_ops_module = _try_import_module("vllm.model_executor.layers.fla.ops.kda")
    kda_layer_module = _try_import_module("vllm.model_executor.layers.kda")
    if kda_ops_module is None and kda_layer_module is None:
        return

    from vllm_ascend.ops.triton.fla.kda import (
        FusedRMSNormGated,
        chunk_kda,
        fused_kda_gate,
        fused_recurrent_kda,
        rms_norm_gated,
    )

    kda_attrs = {
        "FusedRMSNormGated": FusedRMSNormGated,
        "chunk_kda": chunk_kda,
        "fused_kda_gate": fused_kda_gate,
        "fused_recurrent_kda": fused_recurrent_kda,
        "rms_norm_gated": rms_norm_gated,
    }
    _patch_module_attrs(kda_ops_module, kda_attrs)
    _patch_module_attrs(kda_layer_module, kda_attrs)


vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = (
    fused_recurrent_gated_delta_rule_fwd_kernel
)
vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn
vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule
vllm.v1.worker.gpu.sample.gumbel.gumbel_sample = ascend_gumbel_sample
_patch_kda_ops()
