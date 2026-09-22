#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.kv_cache_interface import CrossAttentionSpec

from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    _select_seq_lens,
    filter_chunked_req_indices,
    get_or_register_attention_buffer,
    is_freetime_model_type,
)

NUM_REQS = 2
# Distinct values so the selected source (CPU mirror vs NPU tensor) is identifiable.
CPU_SEQ_LENS = torch.tensor([10, 20, 30], dtype=torch.int32)
NPU_SEQ_LENS = torch.tensor([99, 99, 99], dtype=torch.int32)


def _common_attn_metadata() -> AscendCommonAttentionMetadata:
    return AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 5]),
        query_start_loc_cpu=torch.tensor([0, 2, 5]),
        seq_lens=NPU_SEQ_LENS,
        _seq_lens_cpu=CPU_SEQ_LENS,
        num_reqs=NUM_REQS,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=30,
        block_table_tensor=torch.zeros((NUM_REQS, 4), dtype=torch.int32),
        slot_mapping=torch.arange(5, dtype=torch.int32),
    )


def _spec_config(method: str, parallel_drafting: bool) -> MagicMock:
    spec_config = MagicMock()
    spec_config.parallel_drafting = parallel_drafting
    spec_config.use_dspark.return_value = method == "dspark"
    return spec_config


def _vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
    )


def test_select_seq_lens_defaults_to_cpu_mirror() -> None:
    seq_lens = _select_seq_lens(_common_attn_metadata(), None, None, _vllm_config("glm5_next"))

    torch.testing.assert_close(seq_lens, CPU_SEQ_LENS[:NUM_REQS])


@pytest.mark.parametrize(
    ("method", "parallel_drafting", "model_type", "expected"),
    [
        # DSpark on the GLM5.2 family keeps the CPU seq_lens mirror.
        ("dspark", True, "glm5_next", CPU_SEQ_LENS[:NUM_REQS]),
        ("dspark", True, "glm5_next_text", CPU_SEQ_LENS[:NUM_REQS]),
        # DSpark on other models and other parallel-drafting methods (DFlash,
        # PARD draft_model) keep the NPU seq_lens.
        ("dspark", True, "qwen3", NPU_SEQ_LENS),
        ("dflash", True, "glm5_next", NPU_SEQ_LENS),
        ("draft_model", True, "glm5_next", NPU_SEQ_LENS),
        # Non-parallel spec decode keeps the CPU mirror.
        ("mtp", False, "glm5_next", CPU_SEQ_LENS[:NUM_REQS]),
    ],
)
def test_select_seq_lens_spec_decode(
    method: str,
    parallel_drafting: bool,
    model_type: str,
    expected: torch.Tensor,
) -> None:
    seq_lens = _select_seq_lens(
        _common_attn_metadata(),
        None,
        _spec_config(method, parallel_drafting),
        _vllm_config(model_type),
    )

    torch.testing.assert_close(seq_lens, expected)


def test_select_seq_lens_cross_attention_uses_npu_seq_lens() -> None:
    kv_cache_spec = CrossAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16)

    # Cross-attention wins even under dspark + GLM5.2.
    seq_lens = _select_seq_lens(
        _common_attn_metadata(), kv_cache_spec, _spec_config("dspark", True), _vllm_config("glm5_next")
    )

    assert seq_lens is NPU_SEQ_LENS


def test_is_freetime_model_type_matches_nested_text_config() -> None:
    # Multimodal checkpoints expose the family via hf_config.text_config.
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="top", text_config=SimpleNamespace(model_type="glm5_next_text")),
        )
    )

    assert is_freetime_model_type(vllm_config)


def test_get_or_register_attention_buffer() -> None:
    module_a = torch.nn.Module()
    module_b = torch.nn.Module()
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer.a": module_a,
                "layer.b": module_b,
            }
        )
    )
    factory_call_count = 0

    def factory() -> torch.Tensor:
        nonlocal factory_call_count
        factory_call_count += 1
        return torch.tensor([1, 2, 3])

    buffer = get_or_register_attention_buffer(
        vllm_config,
        ["layer.a", "layer.b"],
        "_test_buffer",
        factory,
    )

    assert factory_call_count == 1
    assert module_a._buffers["_test_buffer"] is buffer
    assert module_b._buffers["_test_buffer"] is buffer
    assert "_test_buffer" not in module_a.state_dict()
    assert "_test_buffer" not in module_b.state_dict()


def test_filter_chunked_req_indices_empty_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [False, False, False],
    )

    torch.testing.assert_close(indices, torch.empty(0, dtype=torch.long))


def test_filter_chunked_req_indices_mixed_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [True, False, True],
    )

    torch.testing.assert_close(indices, torch.tensor([0, 1, 3, 4, 5]))
