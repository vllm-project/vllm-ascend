#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Ascend operator rebinds that do not involve Triton.

These swap upstream CUDA kernels for AscendC operators or torch
implementations: the AscendC ``causal_conv1d`` entry points, ``torch_npu``
RMSNorm for the fused Q/K RMSNorm, and the mHC pre/post torch fallbacks. They
used to live in ``patch_triton.py`` and were split out because nothing here is
Triton.

Import order matters: the ``_cc1d.causal_conv1d_update`` wrapper installed here
is deliberately overwritten by the NPU Triton kernel in ``patch_triton.py``, so
this module must be imported *before* ``patch_triton`` (see
``vllm_ascend/patch/worker/__init__.py``).
"""

import vllm.model_executor.layers.mamba.ops.causal_conv1d as _cc1d

from vllm_ascend.ops.causal_conv1d import (
    causal_conv1d_fn as _npu_causal_conv1d_fn_impl,
)
from vllm_ascend.ops.causal_conv1d import causal_conv1d_update as _npu_causal_conv1d_update_impl


def _npu_causal_conv1d_update(*args, **kwargs):
    for key in (
        "null_block_id",
        "block_idx_first_scheduled_token",
        "block_idx_last_scheduled_token",
        "initial_state_idx",
        "num_computed_tokens",
        "block_size_to_align",
        "validate_data",
        "max_query_len",
        "out",
        "metadata",
    ):
        kwargs.pop(key, None)
    return _npu_causal_conv1d_update_impl(*args, **kwargs)


# npu_kda_causal_conv1d_bind: CUDA causal_conv1d kernels use tl.extra.cuda.gdc_wait
# which Ascend Triton does not provide. Swap the NPU update kernel and a PyTorch
# prefill fn. Models that live in vllm-ascend import the Ascend entry points
# directly and do not rely on this rebind.
_CC1D_FN_DROP = (
    "null_block_id",
    "block_idx_first_scheduled_token",
    "block_idx_last_scheduled_token",
    "initial_state_idx",
    "num_computed_tokens",
    "block_size_to_align",
    "validate_data",
)


def _npu_causal_conv1d_fn(*args, metadata=None, **kwargs):
    for key in _CC1D_FN_DROP:
        kwargs.pop(key, None)
    return _npu_causal_conv1d_fn_impl(*args, **kwargs)


_cc1d.causal_conv1d_update = _npu_causal_conv1d_update
_cc1d.causal_conv1d_fn = _npu_causal_conv1d_fn

# GLM-5.3-Flash vision tower (and any other NVIDIA fused Q/K RMSNorm) launches a
# CUDA Triton kernel that references tl.extra.cuda.gdc_wait. Ascend Triton has
# no such symbol; the AST visitor raises even when launch_pdl is False.
try:
    import torch_npu
    import vllm.models.common.ops as _common_ops
    import vllm.models.common.ops.fused_qk_rmsnorm as _fused_qk_mod

    def _npu_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
        # Same math as the CUDA kernel: RMS in fp32, one cast at store.
        q_out, _ = torch_npu.npu_rms_norm(qr, q_weight, eps)
        kv_out, _ = torch_npu.npu_rms_norm(kv, kv_weight, eps)
        return q_out, kv_out

    _fused_qk_mod.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm
    _common_ops.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm
    # from_import_sweep: the GLM vision tower is imported during model
    # registration, before this patch runs, so its from-import already
    # holds a direct reference to the CUDA kernel. Rebinding the defining
    # module is not enough; rebind every module that captured the name.
    import sys as _sys

    for _m in list(_sys.modules.values()):
        try:
            if getattr(_m, "fused_q_kv_rmsnorm", None) not in (None, _npu_fused_q_kv_rmsnorm):
                _m.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm  # type: ignore[attr-defined]
        except Exception:
            # Lazy loaders, frozen modules and C extensions may refuse the
            # attribute read or the assignment. Skipping them keeps the sweep
            # going for the modules that do hold the CUDA kernel.
            continue
except ImportError:
    pass


# npu_mhc_fused_rmsnorm: the tilelang mHC pre kernel fuses the layer's input
# RMSNorm into layer_input, but mhc_pre_torch (which forward_native/forward_oot
# use on NPU) ignores norm_weight/norm_eps entirely, so every layer would run
# without its input norm. Reapply it here to match the CUDA semantics:
#   layer_input = layer_input * rsqrt(mean(layer_input^2) + norm_eps) * norm_weight
try:
    import torch as _t
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_post_torch as _mhc_post_torch,
    )
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_pre_torch as _mhc_pre_torch,
    )
    from vllm.model_executor.layers import mhc as _mhc_mod

    def _mhc_rms_norm(x, weight, eps):
        xf = x.float()
        var = xf.square().mean(dim=-1, keepdim=True)
        return (xf * _t.rsqrt(var + eps) * weight.float()).to(x.dtype)

    def _mhc_pre_npu(
        self,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        post_mix, comb_mix, layer_input = _mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            layer_input = _mhc_rms_norm(layer_input, norm_weight, norm_eps)
        return post_mix, comb_mix, layer_input

    def _mhc_fused_post_pre_npu(
        self,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        tile_n=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        residual_cur = _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        post_mix_cur, comb_mix_cur, layer_input_cur = _mhc_pre_torch(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            layer_input_cur = _mhc_rms_norm(layer_input_cur, norm_weight, norm_eps)
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    _mhc_mod.MHCPreOp.forward_oot = _mhc_pre_npu
    _mhc_mod.MHCPreOp.forward_native = _mhc_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_oot = _mhc_fused_post_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_native = _mhc_fused_post_pre_npu
except ImportError:
    pass
