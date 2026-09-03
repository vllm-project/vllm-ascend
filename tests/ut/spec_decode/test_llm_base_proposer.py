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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
    _draft_embed_accepts_mm,
)

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
            "Glm5NextForConditionalGeneration",
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


class TestMtpTargetLmHeadLookup:
    """Where the MTP branch of ``_maybe_share_lm_head`` finds the target head.

    Multimodal wrappers such as ``Glm5NextForConditionalGeneration`` own no
    ``lm_head``; it lives on the nested language model, so reading it off the
    wrapper raises ``AttributeError``.
    """

    @staticmethod
    def _share(target) -> SimpleNamespace:
        """Return the resulting draft head after sharing runs."""
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )
        proposer.use_cuda_graph = False
        layers = {"0": SimpleNamespace(shared_head=SimpleNamespace(head=SimpleNamespace(weight=torch.zeros(4, 3))))}
        # The checkpoint ships no MTP head, so the draft head is shared
        # whenever the target head is reachable at all.
        proposer.model = SimpleNamespace(model=SimpleNamespace(layers=layers), has_own_lm_head=False)

        proposer._maybe_share_lm_head(target)

        return layers["0"].shared_head.head

    def test_resolves_lm_head_through_get_language_model(self):
        target_lm_head = SimpleNamespace(weight=torch.ones(4, 3))
        target = SimpleNamespace(get_language_model=lambda: SimpleNamespace(lm_head=target_lm_head))

        assert self._share(target) is target_lm_head

    def test_resolves_lm_head_through_the_language_model_attribute(self):
        target_lm_head = SimpleNamespace(weight=torch.ones(4, 3))
        target = SimpleNamespace(language_model=SimpleNamespace(lm_head=target_lm_head))

        assert self._share(target) is target_lm_head

    def test_unreachable_lm_head_keeps_the_draft_head(self):
        draft_head = self._share(SimpleNamespace())

        assert torch.equal(draft_head.weight, torch.zeros(4, 3))


class TestDraftEmbedMmSupport:
    """Text-only MTP heads such as Glm5NextMTP expose ``embed_input_ids``
    without multimodal parameters, so forwarding a multimodal target model's
    multimodal kwargs to them raises ``TypeError``.
    """

    def test_head_taking_multimodal_embeddings_accepts_mm(self):
        def embed_input_ids(input_ids, multimodal_embeddings=None):
            return input_ids

        assert _draft_embed_accepts_mm(embed_input_ids) is True

    def test_text_only_head_does_not_accept_mm(self):
        def embed_input_ids(input_ids):
            return input_ids

        assert _draft_embed_accepts_mm(embed_input_ids) is False

    @pytest.mark.parametrize("error", [TypeError("C-bound callable"), ValueError("no signature found")])
    def test_uninspectable_callable_is_treated_as_text_only(self, error: Exception):
        """Falling back to text-only cannot raise, whereas assuming multimodal
        support and forwarding the kwargs to a head that rejects them would.
        """

        def embed_input_ids(input_ids, multimodal_embeddings=None):
            return input_ids

        with patch("vllm_ascend.spec_decode.llm_base_proposer._inspect") as fake_inspect:
            fake_inspect.signature.side_effect = error

            assert _draft_embed_accepts_mm(embed_input_ids) is False
