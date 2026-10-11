# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op


@pytest.mark.skipif(
    "950" not in torch.npu.get_device_name(0),
    reason="Regression for the A5 SFA kernel",
)
@pytest.mark.parametrize("selected_count", [0, 1, 127, 128, 129, 257])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout_kv", ["TND", "PA_BSND"])
def test_sparse_flash_attention_padded_indices_lse(selected_count, dtype, layout_kv):
    _check_sparse_flash_attention_lse([selected_count], dtype, layout_kv)


@pytest.mark.skipif(
    "950" not in torch.npu.get_device_name(0),
    reason="Regression for the A5 SFA kernel",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout_kv", ["TND", "PA_BSND"])
def test_sparse_flash_attention_mixed_queries_lse(dtype, layout_kv):
    _check_sparse_flash_attention_lse([0, 1, 129, 257], dtype, layout_kv)


def _check_sparse_flash_attention_lse(selected_counts, dtype, layout_kv):
    # Preserve an explicitly selected OPP package for isolated kernel validation.
    if not os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        bootstrap_custom_op_env()
    enable_custom_op()
    # A5 uses the ACLNN adapter even when runtime custom Python ops are disabled.
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    torch.manual_seed(928)
    heads, kv_length, capacity = 64, 512, 2048
    tokens = len(selected_counts)
    query = torch.randn(tokens, heads, 512, dtype=dtype)
    query_rope = torch.randn(tokens, heads, 64, dtype=dtype)
    key = torch.randn(kv_length, 512, dtype=dtype)
    key_rope = torch.randn(kv_length, 64, dtype=dtype)
    indices = torch.full((tokens, 1, capacity), -1, dtype=torch.int32)
    selections = []
    for token, selected_count in enumerate(selected_counts):
        selected = torch.randperm(kv_length)[:selected_count].sort().values
        selections.append(selected)
        indices[token, 0, :selected_count] = selected.int()
    if layout_kv == "PA_BSND":
        pages = key.reshape(4, 128, 1, 512).npu()
        rope_pages = key_rope.reshape(4, 128, 1, 64).npu()
        block_table = torch.arange(4, dtype=torch.int32).reshape(1, -1).npu()
    else:
        pages = key.reshape(kv_length, 1, 512).npu()
        rope_pages = key_rope.reshape(kv_length, 1, 64).npu()
        block_table = None
    output, maximum, total = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query.npu(),
        key=pages,
        value=pages,
        sparse_indices=indices.npu(),
        scale_value=1 / 24,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=torch.tensor([tokens], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([kv_length], dtype=torch.int32).npu(),
        query_rope=query_rope.npu(),
        key_rope=rope_pages,
        layout_query="TND",
        layout_kv=layout_kv,
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    output = output.cpu().float().reshape(tokens, heads, 512)
    lse = (maximum.cpu().float() + total.cpu().float().log()).reshape(tokens, heads)
    for token, selected in enumerate(selections):
        if selected.numel() == 0:
            assert torch.count_nonzero(output[token]) == 0
            assert torch.isneginf(lse[token]).all()
            continue
        logits = (
            query[token].float() @ key[selected].float().T + query_rope[token].float() @ key_rope[selected].float().T
        ) / 24
        expected = logits.softmax(-1) @ key[selected].float()
        torch.testing.assert_close(output[token], expected, atol=0.03 if dtype == torch.bfloat16 else 0.005, rtol=0.01)
        torch.testing.assert_close(lse[token], logits.logsumexp(-1), atol=0.005, rtol=0.001)
