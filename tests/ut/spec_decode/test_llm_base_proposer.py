#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


@dataclass(frozen=True)
class _CacheConfig:
    cache_dtype: str
    kv_cache_dtype_skip_layers: list[str] | None


@dataclass(frozen=True)
class _VllmConfig:
    cache_config: _CacheConfig
    model_config: object


class TestMultimodalImageTokenIndex:
    @pytest.mark.parametrize(
        "model_name",
        [
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "Step3p7ForConditionalGeneration",
            "Gemma4ForConditionalGeneration",
            "Gemma4UnifiedForConditionalGeneration",
        ],
    )
    def test_models_using_image_token_id(self, model_name: str):
        config = SimpleNamespace(image_token_id=123, image_token_index=456)

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(model_name, config)

        assert image_token_index == 123

    def test_pixtral_uses_vision_config_image_token_id(self):
        config = SimpleNamespace(
            image_token_id=123,
            image_token_index=456,
            vision_config=SimpleNamespace(image_token_id=789),
        )

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(
            "PixtralForConditionalGeneration", config
        )

        assert image_token_index == 789

    @pytest.mark.parametrize(
        "model_name",
        [
            "KimiK25ForConditionalGeneration",
            "KimiK3ForConditionalGeneration",
            "AscendKimiK3ForConditionalGeneration",
        ],
    )
    def test_kimi_uses_media_placeholder_token_id(self, model_name: str):
        config = SimpleNamespace(
            image_token_id=123,
            image_token_index=456,
            media_placeholder_token_id=789,
        )

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(model_name, config)

        assert image_token_index == 789

    def test_default_uses_image_token_index(self):
        config = SimpleNamespace(image_token_id=123, image_token_index=456)

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(
            "OtherForConditionalGeneration", config
        )

        assert image_token_index == 456


class TestMtpSharesTheTargetLmHead:
    """``_maybe_share_lm_head`` for the MTP branch.

    Weight equality only distinguishes the two heads when the draft loaded one.
    GLM-5.3 and friends ship no ``shared_head.head``, so its buffer holds
    whatever the allocation left behind; comparing that against the target head
    says "different" and would leave the draft predicting from garbage.
    """

    @staticmethod
    def _build(draft_head_weight, has_own_lm_head=None):
        target = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones(4, 2)))
        mtp_layer = SimpleNamespace(shared_head=SimpleNamespace(head=SimpleNamespace(weight=draft_head_weight)))

        draft = MagicMock()
        draft.model.layers = {"78": mtp_layer}
        if has_own_lm_head is None:
            del draft.has_own_lm_head
        else:
            draft.has_own_lm_head = has_own_lm_head

        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.model = draft
        proposer.use_cuda_graph = False
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )
        return proposer, target, mtp_layer

    def test_shares_when_the_checkpoint_has_no_mtp_head(self):
        proposer, target, mtp_layer = self._build(torch.randn(4, 2), has_own_lm_head=False)
        proposer._maybe_share_lm_head(target)
        assert mtp_layer.shared_head.head is target.lm_head

    def test_keeps_a_head_the_draft_owns(self):
        proposer, target, mtp_layer = self._build(torch.randn(4, 2), has_own_lm_head=True)
        own_head = mtp_layer.shared_head.head
        proposer._maybe_share_lm_head(target)
        assert mtp_layer.shared_head.head is own_head

    def test_still_deduplicates_an_identical_head(self):
        # DeepSeek ships shared_head.head equal to lm_head; sharing it saves a
        # copy of the vocabulary projection.
        proposer, target, mtp_layer = self._build(torch.ones(4, 2), has_own_lm_head=True)
        proposer._maybe_share_lm_head(target)
        assert mtp_layer.shared_head.head is target.lm_head


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize(
    "model_config",
    [
        SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="minimax_m3"),
        ),
        SimpleNamespace(
            hf_text_config=SimpleNamespace(),
            hf_config=SimpleNamespace(model_type="minimax_m3_vl"),
        ),
    ],
    ids=["text-config", "vl-top-level-config"],
)
def test_minimax_m3_eagle3_draft_uses_model_dtype_with_mixed_kv_cache(
    cache_dtype: str,
    model_config: SimpleNamespace,
):
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    base_config = _VllmConfig(
        cache_config=_CacheConfig(
            cache_dtype=cache_dtype,
            kv_cache_dtype_skip_layers=["0", "1", "2"],
        ),
        model_config=model_config,
    )
    proposer.vllm_config = base_config
    proposer.speculative_config = SimpleNamespace(kv_cache_dtype=None)
    proposer.method = "eagle3"

    with patch.object(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        return_value=base_config,
    ):
        draft_config = proposer._create_draft_vllm_config()

    assert draft_config.cache_config.cache_dtype == "auto"
    assert draft_config.cache_config.kv_cache_dtype_skip_layers == ["0", "1", "2"]
    assert base_config.cache_config.cache_dtype == cache_dtype


def test_minimax_m3_eagle3_respects_explicit_draft_kv_cache_dtype():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    base_config = _VllmConfig(
        cache_config=_CacheConfig(
            cache_dtype="fp8",
            kv_cache_dtype_skip_layers=["0", "1", "2"],
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="minimax_m3"),
        ),
    )
    proposer.vllm_config = base_config
    proposer.speculative_config = SimpleNamespace(kv_cache_dtype="fp8")
    proposer.method = "eagle3"

    with patch.object(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        return_value=base_config,
    ):
        draft_config = proposer._create_draft_vllm_config()

    assert draft_config is base_config


def test_load_model_reads_validated_draft_window_size():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace(additional_config={"draft_window_size": 64})
    proposer.maybe_eager_context = nullcontext()
    draft_model = MagicMock()
    proposer._get_model = MagicMock(return_value=draft_model)
    proposer.method = "eagle3"
    proposer.num_speculative_tokens = 4
    proposer.runner = SimpleNamespace(max_num_reqs=8)
    proposer.device = "cpu"
    proposer.parallel_drafting = False
    proposer.supports_mm_inputs = False
    proposer._maybe_share_embeddings = MagicMock()
    proposer._maybe_share_topk_indices = MagicMock()
    proposer._maybe_share_lm_head = MagicMock()

    draft_layer = MagicMock()
    draft_layer.get_kv_cache_spec.return_value = object()
    draft_layer.get_attn_backend.return_value.get_supported_kernel_block_sizes.return_value = [16]

    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_pp_group") as mock_pp_group,
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.get_layers_from_vllm_config",
            side_effect=[{}, {"draft": draft_layer}, {}, {"draft": draft_layer}],
        ),
        patch("vllm_ascend.ascend_config.get_ascend_config") as mock_get_ascend_config,
        patch("vllm_ascend.spec_decode.llm_base_proposer.SlidingWindowAdapter") as mock_adapter,
        patch("vllm_ascend.spec_decode.llm_base_proposer.supports_multimodal", return_value=False),
    ):
        mock_pp_group.return_value.is_last_rank = True
        mock_get_ascend_config.return_value.draft_window_size = 4096

        proposer.load_model(MagicMock())

    assert proposer.draft_window_size == 4096
    mock_adapter.assert_called_once_with(4096, 16, 8, 4, "cpu")


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()
