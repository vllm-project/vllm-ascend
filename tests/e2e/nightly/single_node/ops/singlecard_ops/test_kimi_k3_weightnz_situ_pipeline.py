#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Real-checkpoint A3 smoke for Kimi K3 routed W4A8 SiTU."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu
from safetensors import safe_open
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.utils import enable_custom_op, maybe_trans_nz

K3_WEIGHT_PATH = Path(os.environ.get("K3_WEIGHT_PATH", "/mnt/weight/k3/Kimi-K3-w4a8-int-multicards"))


def _load_checkpoint_tensor(weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(K3_WEIGHT_PATH / weight_map[name], framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(name)


def _load_checkpoint_slice(weight_map: dict[str, str], name: str, slices: tuple[slice, ...]) -> torch.Tensor:
    with safe_open(K3_WEIGHT_PATH / weight_map[name], framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_slice(name)[slices]


@pytest.mark.skip_global_cleanup
@pytest.mark.skipif(not K3_WEIGHT_PATH.exists(), reason="requires real Kimi K3 W4A8 checkpoint")
@torch.inference_mode()
def test_real_kimi_k3_routed_weightnz_gmm_to_situ_quant():
    """Exercise real packed weights, encoded scales, bias, GMM and SiTU quant."""
    index_path = K3_WEIGHT_PATH / "quant_model_weights.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    torch.npu.config.allow_internal_format = True
    init_ascend_config(VllmConfig(additional_config={"weight_nz_mode": 1}))
    torch.npu.set_device(0)
    torch.manual_seed(7)

    base = "language_model.model.layers.1.block_sparse_moe.experts"
    weights, scales, biases = [], [], []
    for expert in (0, 1):
        prefix = f"{base}.{expert}"
        weights.append(
            torch.cat(
                (
                    _load_checkpoint_tensor(weight_map, f"{prefix}.w1.weight"),
                    _load_checkpoint_tensor(weight_map, f"{prefix}.w3.weight"),
                ),
                dim=0,
            )
        )
        scales.append(
            torch.cat(
                (
                    _load_checkpoint_tensor(weight_map, f"{prefix}.w1.weight_scale"),
                    _load_checkpoint_tensor(weight_map, f"{prefix}.w3.weight_scale"),
                ),
                dim=0,
            )
        )
        biases.append(
            torch.cat(
                (
                    _load_checkpoint_tensor(weight_map, f"{prefix}.w1.scale_bias"),
                    _load_checkpoint_tensor(weight_map, f"{prefix}.w3.scale_bias"),
                ),
                dim=0,
            )
        )

    weight_nd = torch.stack(weights).transpose(1, 2).contiguous().npu()
    weight_nz = maybe_trans_nz(weight_nd)
    packed_weight = weight_nz.view(torch.int32).contiguous()
    assert torch_npu.get_npu_format(weight_nz) == torch_npu.Format.FRACTAL_NZ
    assert tuple(packed_weight.shape) == (2, 3584, 768)

    scale = torch.stack(scales).transpose(1, 2).contiguous()
    scale_numpy = scale.numpy()
    scale_numpy.dtype = np.uint32
    encoded_scale = torch.from_numpy(scale_numpy.astype(np.int64)).npu()
    stored_scale = encoded_scale.squeeze(1)
    gmm_scale = stored_scale.unsqueeze(1)
    bias = torch.stack(biases).transpose(1, 2).contiguous().sum(1).npu()

    x = torch.randn(2, 3584, dtype=torch.bfloat16, device="npu")
    quantized_x, per_token_scale = torch_npu.npu_dynamic_quant(x)
    group_counts = torch.tensor([1, 1], dtype=torch.int64, device="npu")
    gmm_output = torch_npu.npu_grouped_matmul(
        x=[quantized_x],
        weight=[packed_weight],
        scale=[gmm_scale],
        bias=[bias],
        per_token_scale=[per_token_scale],
        group_list=group_counts,
        split_item=2,
        group_list_type=1,
        group_type=0,
        output_dtype=torch.bfloat16,
    )[0]
    torch.npu.synchronize()
    assert tuple(gmm_output.shape) == (2, 6144)
    assert gmm_output.dtype == torch.bfloat16
    assert torch.isfinite(gmm_output).all()

    assert enable_custom_op() and hasattr(torch.ops._C_ascend, "dequant_situ_quant")
    actual_y, actual_scale = torch.ops._C_ascend.dequant_situ_quant(
        x=gmm_output,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        beta=4.0,
        linear_beta=25.0,
        activate_left=True,
        quant_mode="dynamic",
    )

    gate, up = gmm_output.cpu().float().chunk(2, dim=-1)
    situ = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))
    expected_scale = situ.abs().amax(dim=-1) / 127.0
    expected_scale = torch.where(expected_scale == 0, torch.ones_like(expected_scale), expected_scale)
    expected_y = torch.round(situ / expected_scale[:, None]).clamp(-128, 127).to(torch.int8)

    torch.testing.assert_close(actual_y.cpu(), expected_y, rtol=0, atol=1)
    torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=5e-3, atol=1e-5)


@pytest.mark.skip_global_cleanup
@pytest.mark.skipif(not K3_WEIGHT_PATH.exists(), reason="requires real Kimi K3 W4A8 checkpoint")
@torch.inference_mode()
def test_real_kimi_k3_tp16_shared_w8a8_situ_pipeline():
    """Confirm shared experts retain their valid W8A8 INT32 accumulator path."""
    assert enable_custom_op() and hasattr(torch.ops._C_ascend, "dequant_situ_quant")
    weight_map = json.loads(
        (K3_WEIGHT_PATH / "quant_model_weights.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    description = json.loads((K3_WEIGHT_PATH / "quant_model_description.json").read_text(encoding="utf-8"))
    torch.npu.config.allow_internal_format = True
    init_ascend_config(VllmConfig(additional_config={"weight_nz_mode": 1}))
    torch.npu.set_device(0)

    base = "language_model.model.layers.1.block_sparse_moe.shared_experts"
    for projection in ("gate_proj", "up_proj", "down_proj"):
        assert description[f"{base}.{projection}.weight"] == "W8A8_DYNAMIC"

    local_intermediate = 6144 // 16
    gate = _load_checkpoint_slice(weight_map, f"{base}.gate_proj.weight", (slice(0, local_intermediate), slice(None)))
    up = _load_checkpoint_slice(weight_map, f"{base}.up_proj.weight", (slice(0, local_intermediate), slice(None)))
    gate_scale = _load_checkpoint_slice(
        weight_map, f"{base}.gate_proj.weight_scale", (slice(0, local_intermediate), slice(None))
    )
    up_scale = _load_checkpoint_slice(
        weight_map, f"{base}.up_proj.weight_scale", (slice(0, local_intermediate), slice(None))
    )
    raw_gate_up_weight = torch.cat((gate, up), dim=0).contiguous()
    gate_up_weight = maybe_trans_nz(raw_gate_up_weight.t().contiguous().npu())
    gate_up_scale = torch.cat((gate_scale, up_scale), dim=0).flatten().to(torch.bfloat16).npu()

    x = torch.randn(1, 7168, dtype=torch.bfloat16, device="npu")
    quantized_x, per_token_scale = torch_npu.npu_dynamic_quant(x)
    accumulator = torch_npu.npu_quant_matmul(
        quantized_x,
        gate_up_weight,
        gate_up_scale,
        pertoken_scale=None,
        bias=None,
        output_dtype=torch.int32,
    )
    torch.npu.synchronize()
    assert tuple(accumulator.shape) == (1, 768)
    assert accumulator.dtype == torch.int32
    reference_accumulator = quantized_x.cpu().int() @ raw_gate_up_weight[:64].t().int()
    torch.testing.assert_close(accumulator[:, :64].cpu(), reference_accumulator, rtol=0, atol=0)

    quantized_situ, situ_scale = torch.ops._C_ascend.dequant_situ_quant(
        x=accumulator,
        weight_scale=gate_up_scale.float(),
        activation_scale=per_token_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        beta=4.0,
        linear_beta=25.0,
        activate_left=True,
        quant_mode="dynamic",
    )
    assert tuple(quantized_situ.shape) == (1, local_intermediate)
    assert quantized_situ.dtype == torch.int8

    down_weight = _load_checkpoint_slice(
        weight_map, f"{base}.down_proj.weight", (slice(None), slice(0, local_intermediate))
    ).contiguous()
    down_scale = _load_checkpoint_slice(
        weight_map, f"{base}.down_proj.weight_scale", (slice(None), slice(None))
    ).flatten()
    down_weight_nz = maybe_trans_nz(down_weight.t().contiguous().npu())
    partial_output = torch_npu.npu_quant_matmul(
        quantized_situ,
        down_weight_nz,
        down_scale.to(torch.bfloat16).npu(),
        pertoken_scale=situ_scale,
        bias=None,
        output_dtype=torch.bfloat16,
    )
    torch.npu.synchronize()
    assert tuple(partial_output.shape) == (1, 7168)
    assert partial_output.dtype == torch.bfloat16
    assert torch.isfinite(partial_output).all()
