# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


def test_empty_preparation_has_explicit_head_shape():
    from vllm_ascend.ops.triton.fla.fused_gdn_prepare import fused_gdn_prepare

    qkv, z, b, a, out = fused_gdn_prepare(torch.empty(0, 768), torch.empty(0, 4), 512, 2, 128)
    assert qkv.shape == (0, 512)
    assert z.shape == out.shape == (0, 2, 128)
    assert b.shape == a.shape == (0, 2)


@pytest.mark.parametrize(
    "bad_input",
    [
        "qkv_column_stride",
        "ba_column_stride",
        "token_count",
        "ba_width",
        "qkv_width",
        "core_shape",
        "core_stride",
        "head_count",
    ],
)
def test_invalid_layout_rejected_before_kernel_launch(bad_input):
    from vllm_ascend.ops.triton.fla.fused_gdn_prepare import fused_gdn_prepare

    qkvz, ba = torch.empty(4, 768), torch.empty(4, 4)
    output = None
    heads = 2
    if bad_input == "qkv_column_stride":
        qkvz = torch.empty(4, 1536)[:, ::2]
    elif bad_input == "ba_column_stride":
        ba = torch.empty(4, 8)[:, ::2]
    elif bad_input == "token_count":
        ba = torch.empty(3, 4)
    elif bad_input == "ba_width":
        ba = torch.empty(4, 8)
    elif bad_input == "qkv_width":
        qkvz = torch.empty(4, 767)
    elif bad_input == "core_shape":
        output = torch.empty(4, 2, 127)
    elif bad_input == "core_stride":
        output = torch.empty(4, 2, 256)[:, :, ::2]
    else:
        heads = 0
    with pytest.raises(ValueError):
        fused_gdn_prepare(qkvz, ba, 512, heads, 128, output)
