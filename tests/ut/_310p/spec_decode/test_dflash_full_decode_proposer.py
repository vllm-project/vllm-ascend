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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend._310p.ops import rotary_embedding as rotary_embedding_310
from vllm_ascend._310p.spec_decode.dflash_proposer_310 import (
    wrap_dummy_run_with_draft_flag,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import (
    AscendSpecDecodeBaseProposer310,
)
from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)


def _config(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dflash"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
        ),
        additional_config={
            "ascend_compilation_config": {
                "dflash_full_and_piecewise_capture_config": {
                    "piecewise_capture_size": 64,
                    "full_capture_size": 160,
                },
            },
        },
    )


def test_310_patch_registers_hybrid_draft_slot_mapping_override():
    # Production proposers are instances of the public base class. 310P uses
    # patch_idex_310 to attach the private methods; testing only a directly
    # constructed subclass would miss an unregistered production override.
    import vllm_ascend.patch.worker.patch_idex_310  # noqa: F401

    assert (
        AscendSpecDecodeBaseProposer._compute_draft_step_slot_mapping
        is AscendSpecDecodeBaseProposer310._compute_draft_step_slot_mapping
    )


def test_patched_public_proposer_keeps_non_hybrid_slot_mapping():
    import vllm_ascend.patch.worker.patch_idex_310  # noqa: F401

    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_DECODE_ONLY)
    proposer.uses_mrope = False
    table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    positions = torch.tensor([0, 129], dtype=torch.int32)

    slots = proposer._compute_draft_step_slot_mapping(
        table,
        positions,
        block_size=128,
    )

    assert slots.tolist() == [1280, 2689]


def test_hybrid_draft_step_slot_mapping_uses_persistent_gather_without_add():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    proposer.uses_mrope = False
    capacity = 8
    proposer.slot_mapping_group = [torch.empty(capacity, dtype=torch.int32)]
    table = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
    positions = torch.tensor([0, 129, 255], dtype=torch.int32)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch.object(torch, "add", side_effect=AssertionError("private Add is forbidden")),
    ):
        proposer._initialize_hybrid_draft_slot_mapping_310()
        first = proposer._compute_draft_step_slot_mapping(
            table,
            positions,
            block_size=128,
        )
        first_ptr = first.data_ptr()
        second = proposer._compute_draft_step_slot_mapping(
            table[:2],
            positions[:2],
            block_size=128,
        )

    assert first.tolist() == [1280, 2689, 4095]
    assert second.tolist() == [1280, 2689]
    assert second.data_ptr() == first_ptr


def test_hybrid_draft_slot_mapping_normalizes_int64_inputs_to_int32_math():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    proposer.uses_mrope = False
    proposer.slot_mapping_group = [torch.empty(8, dtype=torch.int32)]
    table = torch.tensor([[15, 16], [350, 351]], dtype=torch.int64)
    positions = torch.tensor([36, 179], dtype=torch.int64)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "is_310p_dflash_full_and_piecewise",
            return_value=True,
        ),
        patch.object(torch, "add", side_effect=AssertionError("int64 Add is forbidden")),
    ):
        proposer._initialize_hybrid_draft_slot_mapping_310()
        slots = proposer._compute_draft_step_slot_mapping(
            table,
            positions,
            block_size=128,
        )

    assert slots.dtype == torch.int32
    assert slots.tolist() == [1956, 44979]


def _piecewise_proposer(*, max_query_tokens=6):
    config = _config(CUDAGraphMode.PIECEWISE)
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = config
    proposer.method = "dflash"
    proposer.max_query_tokens = max_query_tokens
    proposer.runner = SimpleNamespace(
        max_num_tokens=6,
        vllm_config=config,
    )
    proposer._query_positions_buffer = torch.tensor(
        [1, 2, 99, 99, 99, 99][:max_query_tokens],
        dtype=torch.int32,
    )
    proposer._get_positions = MagicMock(
        side_effect=lambda num_tokens: proposer._query_positions_buffer[:num_tokens],
    )
    proposer._context_positions_buffer = torch.tensor(
        [7, 8, 99, 99, 99, 99],
        dtype=torch.int32,
    )
    proposer._dflash_num_context = 2
    proposer._full_decode_draft_rotary_310 = SimpleNamespace(
        cos_sin_cache=torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8),
    )
    return proposer


def test_piecewise_capture_precomputes_rope_outside_graph():
    proposer = _piecewise_proposer()
    proposer._query_positions_buffer.copy_(torch.arange(6, dtype=torch.int32))
    observed = []

    def capture(self, num_tokens, *, aclgraph_runtime_mode):
        del self, num_tokens, aclgraph_runtime_mode
        observed.append(rotary_embedding_310._full_decode_rope_precomputed)
        return "captured"

    wrapped = wrap_dummy_run_with_draft_flag(capture)
    rotary_embedding_310.clear_full_decode_draft_rope_310()
    try:
        with patch(
            "vllm_ascend._310p.dflash_piecewise.is_310p",
            return_value=True,
        ):
            result = wrapped(
                proposer,
                6,
                aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            )
    finally:
        rotary_embedding_310.clear_full_decode_draft_rope_310()

    assert result == "captured"
    assert observed == [True]
    assert rotary_embedding_310._full_decode_rope_precomputed is False


def test_piecewise_warmup_and_runtime_refresh_persistent_rope_buffers():
    proposer = _piecewise_proposer()
    rotary_embedding_310.clear_full_decode_draft_rope_310()
    try:
        with patch(
            "vllm_ascend._310p.dflash_piecewise.is_310p",
            return_value=True,
        ):
            warmup_prepared = proposer._prepare_full_decode_draft_rope(
                query_positions=torch.full((2,), 31, dtype=torch.int32),
                query_actual_tokens=2,
                descriptor_tokens=6,
                runtime_mode=CUDAGraphMode.NONE,
            )
            warmup_buffers = rotary_embedding_310.get_full_decode_draft_rope_buffers_310()
            warmup_query_cos = warmup_buffers[0]
            warmup_context_cos = warmup_buffers[2]
            assert warmup_query_cos is not None
            assert warmup_context_cos is not None
            warmup_query_values = warmup_query_cos[:, :2].clone()
            warmup_context_values = warmup_context_cos[:, :2].clone()
            warmup_query_ptr = warmup_query_cos.data_ptr()
            warmup_context_ptr = warmup_context_cos.data_ptr()
            proposer._finish_full_decode_draft_rope(warmup_prepared)

            proposer._query_positions_buffer.copy_(
                torch.tensor([3, 4, 99, 99, 99, 99], dtype=torch.int32),
            )
            proposer._context_positions_buffer.copy_(
                torch.tensor([9, 10, 99, 99, 99, 99], dtype=torch.int32),
            )
            runtime_prepared = proposer._prepare_full_decode_draft_rope(
                query_positions=torch.full((2,), 30, dtype=torch.int32),
                query_actual_tokens=2,
                descriptor_tokens=6,
                runtime_mode=CUDAGraphMode.PIECEWISE,
            )
            runtime_buffers = rotary_embedding_310.get_full_decode_draft_rope_buffers_310()

        assert warmup_prepared is True
        assert runtime_prepared is True
        assert proposer._query_positions_buffer.tolist() == [3, 4, 0, 0, 0, 0]
        assert proposer._context_positions_buffer.tolist() == [9, 10, 0, 0, 0, 0]
        assert runtime_buffers[0] is not None
        assert runtime_buffers[2] is not None
        assert runtime_buffers[0].data_ptr() == warmup_query_ptr
        assert runtime_buffers[2].data_ptr() == warmup_context_ptr
        assert runtime_buffers[0].data_ptr() != runtime_buffers[2].data_ptr()
        assert not torch.equal(runtime_buffers[0][:, :2], warmup_query_values)
        assert not torch.equal(runtime_buffers[2][:, :2], warmup_context_values)
    finally:
        proposer._finish_full_decode_draft_rope(
            locals().get("runtime_prepared", False),
        )


def test_eager_dflash_does_not_enable_graph_external_rope_precompute():
    proposer = _piecewise_proposer()
    proposer.vllm_config = _config(CUDAGraphMode.NONE)
    proposer.runner.vllm_config = proposer.vllm_config
    original_query_positions = proposer._query_positions_buffer.clone()
    original_context_positions = proposer._context_positions_buffer.clone()

    rotary_embedding_310.clear_full_decode_draft_rope_310()
    with patch(
        "vllm_ascend._310p.dflash_piecewise.is_310p",
        return_value=True,
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=torch.arange(2, dtype=torch.int32),
            query_actual_tokens=2,
            descriptor_tokens=6,
            runtime_mode=CUDAGraphMode.NONE,
        )

    assert prepared is False
    assert torch.equal(proposer._query_positions_buffer, original_query_positions)
    assert torch.equal(proposer._context_positions_buffer, original_context_positions)
    assert rotary_embedding_310._full_decode_rope_precomputed is False


def test_full_decode_draft_uses_persistent_target_positions_view():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = _config()
    original = torch.arange(16, dtype=torch.int64)
    persistent = torch.arange(64, dtype=torch.int64)[:16]
    proposer._get_positions = MagicMock(return_value=persistent)

    with patch(
        "vllm_ascend._310p.dflash_full_decode_only.is_310p",
        return_value=True,
    ):
        selected = proposer._select_full_decode_target_positions(
            target_positions=original,
            num_input_tokens=16,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert selected is persistent
    proposer._get_positions.assert_called_once_with(16)


def test_non_full_draft_keeps_original_target_positions():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = _config()
    original = torch.arange(16, dtype=torch.int64)
    proposer._get_positions = MagicMock()

    with patch(
        "vllm_ascend._310p.dflash_full_decode_only.is_310p",
        return_value=True,
    ):
        selected = proposer._select_full_decode_target_positions(
            target_positions=original,
            num_input_tokens=16,
            runtime_mode=CUDAGraphMode.NONE,
        )

    assert selected is original
    proposer._get_positions.assert_not_called()


def test_hybrid_effective_full_draft_uses_persistent_target_positions_view():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    original = torch.arange(16, dtype=torch.int64)
    persistent = torch.arange(64, dtype=torch.int64)[:16]
    proposer._get_positions = MagicMock(return_value=persistent)

    with (
        patch(
            "vllm_ascend._310p.dflash_full_decode_only.is_310p",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
    ):
        selected = proposer._select_full_decode_target_positions(
            target_positions=original,
            num_input_tokens=16,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert selected is persistent
    proposer._get_positions.assert_called_once_with(16)


def test_hybrid_piecewise_draft_keeps_runtime_target_positions():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    original = torch.arange(16, dtype=torch.int64)
    proposer._get_positions = MagicMock()

    with (
        patch(
            "vllm_ascend._310p.dflash_full_decode_only.is_310p",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
    ):
        selected = proposer._select_full_decode_target_positions(
            target_positions=original,
            num_input_tokens=16,
            runtime_mode=CUDAGraphMode.PIECEWISE,
        )

    assert selected is original
    proposer._get_positions.assert_not_called()


def test_pure_full_draft_keeps_runtime_target_positions():
    proposer = object.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = _config(CUDAGraphMode.FULL)
    original = torch.arange(16, dtype=torch.int64)
    proposer._get_positions = MagicMock()

    with (
        patch(
            "vllm_ascend._310p.dflash_full_decode_only.is_310p",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
    ):
        selected = proposer._select_full_decode_target_positions(
            target_positions=original,
            num_input_tokens=16,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert selected is original
    proposer._get_positions.assert_not_called()


def test_full_decode_draft_rope_prepares_distinct_query_and_context_sources():
    assert hasattr(
        AscendSpecDecodeBaseProposer310,
        "_prepare_full_decode_draft_rope",
    ), "310P FULL draft RoPE lifecycle hook is missing"

    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config()
    proposer.method = "dflash"
    proposer.runner = SimpleNamespace(max_num_tokens=1280)
    proposer._context_positions_buffer = torch.arange(1280, dtype=torch.int32)
    proposer._dflash_num_context = 6
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(32, 8))
    proposer.model = SimpleNamespace(modules=lambda: [rotary])
    query_positions = torch.arange(160, dtype=torch.int32)
    query_cos = torch.empty(1, 1280, 1, 8)
    query_sin = torch.empty_like(query_cos)
    context_cos = torch.empty_like(query_cos)
    context_sin = torch.empty_like(query_cos)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.is_310p_dflash_full_decode_only",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.AscendRotaryEmbedding310",
            type(rotary),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.prepare_full_decode_draft_rope_310") as prepare,
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.get_full_decode_draft_rope_buffers_310",
            return_value=(query_cos, query_sin, context_cos, context_sin),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.clear_full_decode_draft_rope_310") as clear,
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=query_positions,
            query_actual_tokens=96,
            descriptor_tokens=160,
            runtime_mode=CUDAGraphMode.FULL,
        )
        proposer._finish_full_decode_draft_rope(prepared)

    prepare.assert_called_once()
    prepare_args, prepare_kwargs = prepare.call_args
    assert len(prepare_args) == 1
    assert prepare_args[0] is rotary.cos_sin_cache
    assert prepare_kwargs["query_positions"] is query_positions
    assert prepare_kwargs["query_actual_tokens"] == 96
    assert prepare_kwargs["context_positions"].data_ptr() == (proposer._context_positions_buffer.data_ptr())
    assert prepare_kwargs["context_positions"].shape == (160,)
    assert prepare_kwargs["context_actual_tokens"] == 6
    assert prepare_kwargs["capacity_tokens"] == 1280
    assert proposer._full_decode_draft_query_rope_cos_310 is query_cos
    assert proposer._full_decode_draft_query_rope_sin_310 is query_sin
    assert proposer._full_decode_draft_context_rope_cos_310 is context_cos
    assert proposer._full_decode_draft_context_rope_sin_310 is context_sin
    clear.assert_called_once_with()


def test_hybrid_effective_full_prepares_persistent_draft_rope():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    proposer.method = "dflash"
    proposer.runner = SimpleNamespace(
        max_num_tokens=1280,
        vllm_config=proposer.vllm_config,
    )
    proposer._context_positions_buffer = torch.arange(1280, dtype=torch.int32)
    proposer._dflash_num_context = 6
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(32, 8))
    proposer.model = SimpleNamespace(modules=lambda: [rotary])
    query_positions = torch.arange(64, dtype=torch.int32)
    query_cos = torch.empty(1, 1280, 1, 8)
    query_sin = torch.empty_like(query_cos)
    context_cos = torch.empty_like(query_cos)
    context_sin = torch.empty_like(query_cos)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.AscendRotaryEmbedding310",
            type(rotary),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.prepare_full_decode_draft_rope_310") as prepare,
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.get_full_decode_draft_rope_buffers_310",
            return_value=(query_cos, query_sin, context_cos, context_sin),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.clear_full_decode_draft_rope_310") as clear,
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=query_positions,
            query_actual_tokens=16,
            descriptor_tokens=64,
            runtime_mode=CUDAGraphMode.FULL,
        )
        proposer._finish_full_decode_draft_rope(prepared)

    assert prepared is True
    prepare.assert_called_once()
    clear.assert_called_once_with()


def test_hybrid_full_rope_separates_target_descriptor_from_draft_capacity():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    proposer.method = "dflash"
    proposer.max_query_tokens = 32
    proposer.runner = SimpleNamespace(
        max_num_tokens=1280,
        vllm_config=proposer.vllm_config,
    )
    proposer._context_positions_buffer = torch.arange(1280, dtype=torch.int32)
    proposer._dflash_num_context = 2
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(64, 8))
    proposer.model = SimpleNamespace(modules=lambda: [rotary])
    query_positions = torch.arange(32, dtype=torch.int32)
    query_cos = torch.empty(1, 1280, 1, 8)
    query_sin = torch.empty_like(query_cos)
    context_cos = torch.empty_like(query_cos)
    context_sin = torch.empty_like(query_cos)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "AscendRotaryEmbedding310",
            type(rotary),
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "prepare_full_decode_draft_rope_310",
        ) as prepare,
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310."
            "get_full_decode_draft_rope_buffers_310",
            return_value=(query_cos, query_sin, context_cos, context_sin),
        ),
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=query_positions,
            query_actual_tokens=2,
            descriptor_tokens=64,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert prepared is True
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["query_positions"] is query_positions
    assert prepare.call_args.kwargs["query_actual_tokens"] == 2
    assert prepare.call_args.kwargs["context_positions"].shape == (64,)


def test_hybrid_effective_piecewise_refreshes_capture_stable_draft_rope():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config(CUDAGraphMode.FULL_AND_PIECEWISE)
    proposer.method = "dflash"
    proposer.runner = SimpleNamespace(
        max_num_tokens=1280,
        vllm_config=proposer.vllm_config,
    )
    proposer._context_positions_buffer = torch.arange(1280, dtype=torch.int32)
    proposer._dflash_num_context = 52
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(128, 8))
    proposer.model = SimpleNamespace(modules=lambda: [rotary])
    runtime_positions = torch.arange(64, dtype=torch.int32)
    proposer._get_positions = MagicMock(return_value=runtime_positions)
    query_cos = torch.empty(1, 1280, 1, 8)
    query_sin = torch.empty_like(query_cos)
    context_cos = torch.empty_like(query_cos)
    context_sin = torch.empty_like(query_cos)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.AscendRotaryEmbedding310",
            type(rotary),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.prepare_full_decode_draft_rope_310") as prepare,
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.get_full_decode_draft_rope_buffers_310",
            return_value=(query_cos, query_sin, context_cos, context_sin),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.clear_full_decode_draft_rope_310") as clear,
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=torch.arange(52, dtype=torch.int32),
            query_actual_tokens=16,
            descriptor_tokens=64,
            runtime_mode=CUDAGraphMode.PIECEWISE,
        )
        proposer._finish_full_decode_draft_rope(prepared)

    assert prepared is True
    proposer._get_positions.assert_called_once_with(64)
    assert prepare.call_args.kwargs["query_positions"] is runtime_positions
    assert prepare.call_args.kwargs["query_actual_tokens"] == 16
    assert prepare.call_args.kwargs["context_actual_tokens"] == 52
    clear.assert_called_once_with()


def test_pure_full_draft_does_not_enable_hybrid_rope_lifecycle():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config(CUDAGraphMode.FULL)
    proposer.method = "dflash"
    proposer.runner = SimpleNamespace(vllm_config=proposer.vllm_config)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch(
            "vllm_ascend._310p.dflash_full_and_piecewise.is_310p",
            return_value=True,
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.prepare_full_decode_draft_rope_310") as prepare,
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=torch.arange(16, dtype=torch.int32),
            query_actual_tokens=16,
            descriptor_tokens=16,
            runtime_mode=CUDAGraphMode.FULL,
        )

    assert prepared is False
    prepare.assert_not_called()


def test_fdo_non_full_draft_refreshes_full_context_beyond_query_descriptor():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config()
    proposer.method = "dflash"
    proposer.runner = SimpleNamespace(max_num_tokens=1280)
    proposer._context_positions_buffer = torch.arange(1280, dtype=torch.int32)
    proposer._dflash_num_context = 64
    rotary = SimpleNamespace(cos_sin_cache=torch.zeros(32, 8))
    proposer.model = SimpleNamespace(modules=lambda: [rotary])
    target_positions = torch.arange(3 * 34, dtype=torch.int32).reshape(3, 34)
    query_positions = torch.arange(16, dtype=torch.int32) + 34
    proposer._get_positions = MagicMock(return_value=query_positions)
    query_cos = torch.empty(1, 1280, 1, 8)
    query_sin = torch.empty_like(query_cos)
    context_cos = torch.empty_like(query_cos)
    context_sin = torch.empty_like(query_cos)

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.is_310p_dflash_full_decode_only",
            return_value=True,
        ),
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.AscendRotaryEmbedding310",
            type(rotary),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.prepare_full_decode_draft_rope_310") as prepare,
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.get_full_decode_draft_rope_buffers_310",
            return_value=(query_cos, query_sin, context_cos, context_sin),
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.clear_full_decode_draft_rope_310") as clear,
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=target_positions,
            query_actual_tokens=16,
            descriptor_tokens=16,
            runtime_mode=CUDAGraphMode.NONE,
        )
        proposer._finish_full_decode_draft_rope(prepared)

    assert prepared is True
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["query_positions"] is query_positions
    assert prepare.call_args.kwargs["query_actual_tokens"] == 16
    assert prepare.call_args.kwargs["context_positions"].shape == (64,)
    assert prepare.call_args.kwargs["context_actual_tokens"] == 64
    proposer._get_positions.assert_called_once_with(16)
    clear.assert_called_once_with()


def test_non_fdo_draft_does_not_touch_full_rope_selection():
    proposer = object.__new__(AscendSpecDecodeBaseProposer310)
    proposer.vllm_config = _config()
    proposer.method = "dflash"

    with (
        patch(
            "vllm_ascend._310p.spec_decode.llm_base_proposer_310.is_310p_dflash_full_decode_only",
            return_value=False,
        ),
        patch("vllm_ascend._310p.spec_decode.llm_base_proposer_310.clear_full_decode_draft_rope_310") as clear,
    ):
        prepared = proposer._prepare_full_decode_draft_rope(
            query_positions=torch.arange(16, dtype=torch.int32),
            query_actual_tokens=16,
            descriptor_tokens=16,
            runtime_mode=CUDAGraphMode.NONE,
        )

    assert prepared is False
    clear.assert_not_called()
