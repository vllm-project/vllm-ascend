# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone output and input-validation contracts for the A5 metadata op."""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.ops.helpers.c_ascend_loader import ensure_c_ascend_loaded
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile


@pytest.fixture
def metadata_op():
    if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
        pytest.skip("SparseFlashMlaMetadata requires A5")
    ensure_c_ascend_loaded(required_op="npu_sparse_flash_mla_metadata")
    return torch.ops._C_ascend.npu_sparse_flash_mla_metadata


def _inputs(selected_count):
    return dict(
        num_heads_q=4,
        num_heads_kv=1,
        head_dim=512,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        seqused_ori_kv=torch.tensor([selected_count], dtype=torch.int32, device="npu"),
        ori_topk_length=torch.tensor([[selected_count]], dtype=torch.int32, device="npu"),
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_ori_kv=selected_count,
        max_seqlen_cmp_kv=0,
        ori_topk=2051,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=3,
        cmp_mask_mode=3,
        ori_win_left=0,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device="npu",
    )


@pytest.mark.parametrize("selected_count", [1, 128, 2048])
@torch.inference_mode()
def test_metadata_output_contract(metadata_op, selected_count):
    output = metadata_op(**_inputs(selected_count))
    torch.npu.synchronize()
    assert output.shape == (1024,)
    assert output.dtype == torch.int32
    assert output.device.type == "npu"


@torch.inference_mode()
def test_metadata_rejects_unsupported_head_dimension(metadata_op):
    inputs = _inputs(128)
    inputs["head_dim"] = 511
    with pytest.raises(RuntimeError):
        metadata_op(**inputs)
        torch.npu.synchronize()
