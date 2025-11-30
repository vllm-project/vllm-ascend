#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from typing import Optional, Tuple, Union, cast, Dict, Any

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.triton_utils import tl, triton
from functools import cache


@cache
def get_device_properties() -> Tuple[int, int]:
    device = torch.npu.current_device()
    device_properties: Dict[str, Any] = (
        triton.runtime.driver.active.utils.get_device_properties(device)
    )

    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None
})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def rms_norm_fwd_kernel(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        Z,  # pointer to the residual
        Z_Out,  # pointer to the residual output
        stride_x_row,  # how much to increase the pointer when moving by 1 row
        stride_y_row,
        stride_z_row,
        stride_z_out_row,
        n_rows,  # number of rows in X_base
        n_cols,  # number of columns in X_base
        eps,  # epsilon to avoid division by zero
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_Z: tl.constexpr,
):
    # Map the program id to the row of X_base and Y_base it should compute.
    # Each program computes a row of X_base and store to Y_base
    row_start = tl.program_id(0)
    for row_idx in tl.range(row_start, n_rows, tl.num_programs(0)):
        start_x = X + row_idx * stride_x_row
        start_y = Y + row_idx * stride_y_row
        if HAS_Z:
            start_z = Z + row_idx * stride_z_row
            start_z_out = Z_Out + row_idx * stride_z_out_row
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < n_cols
        x = tl.load(start_x + offsets, mask=mask, other=0.0)
        original_dtype = x.dtype
        x = x.to(tl.float32)
        if HAS_Z:
            z = tl.load(start_z + offsets, mask=mask, other=0.0).to(tl.float32)
            x = x + z
            tl.store(start_z_out + offsets, x, mask=mask)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = 1 / tl.sqrt(var + eps)
        w = tl.load(W + offsets, mask=mask).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(B + offsets, mask=mask).to(tl.float32)

        x_hat = x * rstd
        # Cast normalized x back to original data type to preserve precision contract
        x_hat = x_hat.to(original_dtype)
        y = x_hat * w
        if HAS_BIAS:
            y = y + bias
        tl.store(start_y + offsets, y, mask=mask)


def _rms_norm_fwd_triton(
        x,
        weight,
        eps,
        residual=None,
        bias=None,
        out=None,
        residual_out=None,
):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    # logger.info(f"bias is {bias}")
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if residual is not None:
        assert residual.shape == x.shape
        assert residual.stride(-1) == 1
        if residual_out is None:
            residual_out = torch.empty_like(x)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This rms norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # _, num_vectorcore = get_device_properties()
    num_vectorcore = 40
    grid = (M if M < num_vectorcore else num_vectorcore,)
    # with torch.npu.device(x.device.index):
    rms_norm_fwd_kernel[grid](
        x,
        out,
        weight,
        bias,
        residual,
        residual_out,
        x.stride(0),
        out.stride(0),
        residual.stride(0) if residual is not None else None,
        residual_out.stride(0) if residual is not None else None,
        M,
        N,
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        # multibuffer=True,
    )
    return out, residual_out

def _addrmsnorm_forward_oot(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    layer: Optional[torch.nn.Module] = None,
    bias: Optional[torch.nn.Parameter] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    import torch_npu

    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    if layer is not None and get_ascend_device_type(
    ) != AscendDeviceType._310P:
        layer_cls_name = layer.__class__.__name__
        try:
            weight_prefetch_method = get_forward_context(
            ).weight_prefetch_method
        except AssertionError:
            weight_prefetch_method = None

        # prefetch qkvo_proj.weight preprocess
        if weight_prefetch_method:
            weight_prefetch_method.maybe_prefetch_attn_weight_preprocess(
                layer_cls_name=layer_cls_name,
                weight=layer.weight,
                start_flag=x,
            )
        # add_rms_norm_quant
        x, _, residual = torch_npu.npu_add_rms_norm_quant(
            x,
            residual,
            self.weight,
            layer.aclnn_input_scale,
            layer.aclnn_input_offset,
            beta=bias,
            epsilon=self.variance_epsilon)

        # prefetch qkvo_proj.weight postprocess
        if weight_prefetch_method:
            weight_prefetch_method.maybe_prefetch_attn_weight_postprocess(
                layer_cls_name=layer_cls_name,
                stop_flag=x,
            )

    else:
        if get_ascend_device_type() == AscendDeviceType._310P:
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                          self.variance_epsilon)
        else:
            x, residual = _rms_norm_fwd_triton(x, self.weight, self.variance_epsilon, residual, bias)
    torch.ops.vllm.maybe_wait_prefetch_done(x)
    return x, residual


class AscendRMSNorm(RMSNorm):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        # quantization with anti_method m4 will generate none-zero norm bias
        if vllm_config.quant_config is not None and \
                any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                           requires_grad=False)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        if residual is not None:
            assert x.size(0) == residual.size(0)
            x, residual = _addrmsnorm_forward_oot(
                self, x, residual, self.next_need_quant_fusion_linear,
                self.bias)
            return x, residual
        if self.bias is not None:
            x, _ = _rms_norm_fwd_triton(x, self.weight, self.variance_epsilon, None, self.bias)
        else:
            x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                          self.variance_epsilon)
        return x

    @property
    def next_need_quant_fusion_linear(self):
        try:
            forward_context = get_forward_context()
            if not forward_context.addrmsnorm_quant_fusion_enabled or \
                forward_context.layer_idx == forward_context.num_hidden_layers:
                return None
        except AssertionError:
            return None

        next_linear = None
        model_instance = forward_context.model_instance
        layer_idx = forward_context.layer_idx
        fusion_linear = forward_context.fusion_linear
        next_linear = None
        if fusion_linear == "qkv_dense":
            next_linear = model_instance.model.layers[
                layer_idx].self_attn.qkv_proj
            forward_context.fusion_linear = "gate_up_dense"
        elif fusion_linear == "gate_up_dense":
            next_linear = model_instance.model.layers[
                layer_idx].mlp.gate_up_proj
            forward_context.fusion_linear = "qkv_dense"
            # if prefetch_mlp_weight enabled, following accumulation operation
            # does not need to be repeated
            if not forward_context.prefetch_mlp_enabled:
                forward_context.layer_idx += 1
        elif fusion_linear == "qkv_moe":
            next_linear = model_instance.model.layers[
                layer_idx].self_attn.qkv_proj
            forward_context.fusion_linear = "gate_moe"
        elif fusion_linear == "gate_moe":
            forward_context.fusion_linear = "qkv_moe"
            forward_context.layer_idx += 1
        from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
        if next_linear is not None and \
            not isinstance(next_linear.quant_method.quant_method, AscendW8A8LinearMethod):
            next_linear = None
        return next_linear


class AscendQuantRMSNorm(AscendRMSNorm):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                       requires_grad=False)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x, residual = super().forward_oot(x, residual)
            return x.add_(self.bias), residual
        return cast(torch.Tensor, super().forward_oot(x)).add_(self.bias)


class AscendGemmaRMSNorm(GemmaRMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        if residual is not None:
            if get_ascend_device_type() == AscendDeviceType._310P:
                orig_dtype = residual.dtype
                x = x + residual.to(x.dtype)
                residual = x.to(orig_dtype)
                x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight,
                                              self.variance_epsilon)
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(
                    x, residual, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight,
                                      self.variance_epsilon)
        return x
