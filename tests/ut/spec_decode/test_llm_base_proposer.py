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
from vllm.model_executor.models.bailing_moe_v3_mtp import (
    BailingMoeV3MTPModel,
    BailingMoeV3MTPSharedHead,
    BailingMoeV3MultiTokenPredictor,
)

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


def test_query_start_loc_arange_expands_to_required_capacity():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.max_batch_size = 2
    proposer.max_num_tokens = 4
    proposer.arange = torch.arange(4, dtype=torch.int64)

    proposer._ensure_query_start_loc_arange_capacity()

    assert proposer.arange.dtype == torch.int64
    assert torch.equal(proposer.arange, torch.arange(5, dtype=torch.int64))


def test_query_start_loc_arange_keeps_sufficient_buffer():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.max_batch_size = 4
    proposer.max_num_tokens = 2
    arange = torch.arange(5, dtype=torch.int32)
    proposer.arange = arange

    proposer._ensure_query_start_loc_arange_capacity()

    assert proposer.arange is arange


@pytest.mark.parametrize(
    "target_use_mla,draft_use_mla,dcp_size,expected_rows",
    [
        pytest.param(True, False, 1, 1, id="mla-target-gqa-draft"),
        pytest.param(False, False, 1, 1, id="gqa-target-gqa-draft"),
        pytest.param(True, True, 1, 4, id="mla-target-mla-draft"),
        pytest.param(False, True, 1, 4, id="gqa-target-mla-draft"),
        pytest.param(True, False, 2, 4, id="mla-target-dcp"),
        pytest.param(False, False, 2, 4, id="gqa-target-dcp"),
    ],
)
def test_eager_propose_aligns_block_table_to_draft_batch(target_use_mla, draft_use_mla, dcp_size, expected_rows):
    """Target graph padding must not add requests to an eager GQA draft."""
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "dflash"
    proposer.model = SimpleNamespace(combine_hidden_states=lambda states: states)
    proposer.hidden_size = 4
    proposer.use_cuda_graph = False
    proposer.parallel_drafting = True
    proposer.dcp_size = dcp_size
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(use_mla=target_use_mla))
    proposer.draft_model_config = SimpleNamespace(use_mla=draft_use_mla)
    proposer.draft_window_size = None
    proposer.supports_mm_inputs = False
    proposer.slot_mapping_group = [torch.zeros(6, dtype=torch.int32)]
    proposer.seq_lens_group = [torch.zeros(1, dtype=torch.int32)]
    proposer.query_start_loc_group = [torch.zeros(2, dtype=torch.int32)]
    proposer._pad_draft_buffers = MagicMock()
    proposer.runner = SimpleNamespace(
        dcp_manager=None,
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        _sync_metadata_across_dp=lambda num_tokens, **kwargs: (num_tokens, None, None),
    )
    block_table = torch.arange(1, 13, dtype=torch.int32).reshape(4, 3)
    query_start_loc = torch.tensor([0, 6], dtype=torch.int32)
    metadata = SimpleNamespace(
        batch_size=lambda: 1,
        num_reqs=1,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([16], dtype=torch.int32),
        block_table_tensor=block_table,
        slot_mapping=torch.arange(6, dtype=torch.int32),
    )
    sample_indices = torch.arange(1, 6, dtype=torch.int32)
    proposer.set_inputs_first_pass = MagicMock(return_value=(6, sample_indices, metadata, None))

    class MetadataChecked(Exception):
        pass

    def check_metadata(common_metadata, num_input_tokens, num_actual_tokens):
        assert num_input_tokens == num_actual_tokens == 6
        assert common_metadata.num_reqs == 1
        assert torch.equal(common_metadata.query_start_loc, query_start_loc)
        assert common_metadata.seq_lens.tolist() == [16]
        assert torch.equal(common_metadata.block_table_tensor, block_table[:expected_rows])
        raise MetadataChecked

    proposer.build_draft_attn_metadata = check_metadata
    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer._HIDDEN_STATE_DRAFTER_TYPES", (object,)),
        pytest.raises(MetadataChecked),
    ):
        proposer._propose(
            5,
            target_token_ids=torch.ones(1, dtype=torch.int64),
            target_positions=torch.zeros(1, dtype=torch.int32),
            target_hidden_states=torch.ones((1, 4)),
            next_token_ids=torch.ones(1, dtype=torch.int64),
            token_indices_to_sample=sample_indices,
            common_attn_metadata=metadata,
            target_model_batch_desc=SimpleNamespace(uniform=True),
            sampling_metadata=MagicMock(),
        )


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

    def test_model_with_multiple_image_sentinels_needs_no_single_index(self):
        config = SimpleNamespace()

        image_token_index = AscendSpecDecodeBaseProposer._get_multimodal_image_token_index(
            "AscendDeepseekV4ForConditionalGeneration", config
        )

        assert image_token_index is None


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
        del draft.share_lm_head
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

    def test_bailing_draft_uses_share_lm_head_hook_without_traversing_layers(self):
        class BailingDraft:
            def __init__(self):
                self.lm_head = None
                self.model = SimpleNamespace(layers=MagicMock())
                self.model.layers.items.side_effect = AssertionError("Bailing hook must bypass shared_head traversal")

            def share_lm_head(self, lm_head):
                self.lm_head = lm_head

        target_lm_head = SimpleNamespace(weight=torch.ones(4, 2))
        target = SimpleNamespace(lm_head=target_lm_head)
        draft = BailingDraft()
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.model = draft
        proposer.use_cuda_graph = False
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )

        proposer._maybe_share_lm_head(target)

        assert draft.lm_head is target_lm_head
        draft.model.layers.items.assert_not_called()

    @pytest.mark.parametrize("spec_step_idx", [0, 1, 2])
    def test_bailing_compute_logits_uses_target_head_for_every_mtp_step(self, spec_step_idx):
        # Keep the real logits methods, without constructing attention or MoE.
        predictor = BailingMoeV3MultiTokenPredictor.__new__(BailingMoeV3MultiTokenPredictor)
        torch.nn.Module.__init__(predictor)
        predictor.mtp_start_layer_idx = 7
        predictor.num_mtp_layers = 2
        predictor.layers = torch.nn.ModuleDict()
        for layer_idx in (7, 8):
            shared_head = BailingMoeV3MTPSharedHead.__new__(BailingMoeV3MTPSharedHead)
            torch.nn.Module.__init__(shared_head)
            shared_head.head = torch.nn.Linear(3, 5, bias=False)
            torch.nn.init.zeros_(shared_head.head.weight)
            layer = torch.nn.Module()
            layer.shared_head = shared_head
            predictor.layers[str(layer_idx)] = layer
        predictor.logits_processor = MagicMock(side_effect=lambda head, states: head(states))

        draft = BailingMoeV3MTPModel.__new__(BailingMoeV3MTPModel)
        torch.nn.Module.__init__(draft)
        draft.model = predictor
        draft.lm_head = None
        target_head = torch.nn.Linear(3, 5, bias=False)
        torch.nn.init.ones_(target_head.weight)
        hidden_states = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(draft.compute_logits(hidden_states, spec_step_idx), torch.zeros(1, 5))

        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "mtp"
        proposer.model = draft
        proposer.use_cuda_graph = False
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(is_deepseek_mla=True),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        )
        proposer._maybe_share_lm_head(SimpleNamespace(lm_head=target_head))

        torch.testing.assert_close(draft.compute_logits(hidden_states, spec_step_idx), target_head(hidden_states))
        assert predictor.logits_processor.call_args.args[0] is target_head


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
            side_effect=[{}, {"draft": draft_layer}, {"draft": draft_layer}],
        ),
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_ascend_config") as mock_get_ascend_config,
        patch("vllm_ascend.spec_decode.llm_base_proposer.SlidingWindowAdapter") as mock_adapter,
        patch("vllm_ascend.spec_decode.llm_base_proposer.supports_multimodal", return_value=False),
    ):
        mock_pp_group.return_value.is_last_rank = True
        mock_get_ascend_config.return_value.draft_window_size = 4096

        proposer.load_model(MagicMock())

    assert proposer.draft_window_size == 4096
    mock_adapter.assert_called_once_with(4096, 16, 8, 4, "cpu")


@pytest.mark.parametrize(
    "method,has_post_process",
    [("dspark", True), ("dspark", False), ("dflash", True), ("dflash", False), ("eagle3", True), ("mtp", True)],
)
def test_load_model_aligns_draft_after_sharing_before_precomputing_hidden_states(method, has_post_process):
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.vllm_config = SimpleNamespace(quant_config=object())
    proposer.maybe_eager_context = nullcontext()
    proposer.method = method
    proposer.supports_mm_inputs = False
    proposer.parallel_drafting = True
    proposer.pass_hidden_states_to_model = True
    proposer.eagle3_use_aux_hidden_state = True
    proposer.hidden_size = 2
    proposer.parallel_drafting_hidden_state_tensor = torch.empty(2)
    events = []
    draft = SimpleNamespace(mask_hidden=torch.ones(6))

    def post_process(config):
        assert config is proposer.vllm_config
        events.append("post_process")

    def combine_hidden_states(hidden):
        events.append("combine")
        assert torch.equal(hidden, torch.ones(6))
        return torch.full((2,), 2.0 if "post_process" in events else 1.0)

    if has_post_process:
        draft.post_process = post_process
    draft.combine_hidden_states = combine_hidden_states
    proposer._get_model = MagicMock(return_value=draft)
    proposer._maybe_share_embeddings = MagicMock(side_effect=lambda _: events.append("embeddings"))
    proposer._maybe_share_topk_indices = MagicMock(side_effect=lambda _: events.append("indices"))
    proposer._maybe_share_lm_head = MagicMock(side_effect=lambda _: events.append("lm_head"))
    draft_layer = MagicMock()
    draft_layer.get_kv_cache_spec.return_value = object()
    draft_layer.get_attn_backend.return_value.get_supported_kernel_block_sizes.return_value = [16]
    module = "vllm_ascend.spec_decode.llm_base_proposer"
    with (
        patch(f"{module}.get_pp_group", return_value=SimpleNamespace(is_last_rank=True)),
        patch(
            f"{module}.get_layers_from_vllm_config",
            side_effect=[{}, {"draft": draft_layer}, {"draft": draft_layer}],
        ),
        patch("vllm_ascend.ascend_config.get_ascend_config", return_value=SimpleNamespace(draft_window_size=None)),
        patch(f"{module}.get_ascend_config", return_value=SimpleNamespace(draft_window_size=None)),
        patch(f"{module}.supports_multimodal", return_value=False),
    ):
        proposer.load_model(MagicMock())

    should_process = method in ("dspark", "dflash") and has_post_process
    expected = ["embeddings", "indices", "lm_head"]
    if should_process:
        expected.append("post_process")
    assert events == [*expected, "combine"]
    assert torch.equal(proposer.parallel_drafting_hidden_state_tensor, torch.full((2,), 2.0 if should_process else 1.0))


def test_draft_vllm_config_only_propagates_draft_runner_type():
    draft_model_config = SimpleNamespace(
        runner_type="draft",
        architecture="draft-architecture",
        num_experts=0,
    )
    base_model_config = SimpleNamespace(
        runner_type="generate",
        architecture="target-architecture",
        num_experts=256,
    )
    base_vllm_config = SimpleNamespace(model_config=base_model_config)
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.speculative_config = SimpleNamespace(
        draft_model_config=draft_model_config,
    )

    with (
        patch(
            "vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer._create_draft_vllm_config",
            return_value=base_vllm_config,
        ),
    ):
        draft_vllm_config = proposer._create_draft_vllm_config()

    assert draft_vllm_config is not base_vllm_config
    assert draft_vllm_config.model_config is not base_model_config
    assert draft_vllm_config.model_config is not draft_model_config
    assert draft_vllm_config.model_config.runner_type == "draft"
    assert draft_vllm_config.model_config.architecture == "target-architecture"
    assert draft_vllm_config.model_config.num_experts == 256
    assert base_model_config.runner_type == "generate"


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


@pytest.mark.parametrize(
    "dcp_size,draft_index,parallel_drafting,normalize",
    [
        (1, 1, False, False),
        (2, 0, False, False),
        (2, 1, False, True),
        (2, 2, False, True),
        (2, 1, True, False),
    ],
)
def test_dcp_draft_capture_query_layout(dcp_size, draft_index, parallel_drafting, normalize):
    from vllm_ascend.attention.context_parallel.common_cp import use_history_current_split_decode

    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.dcp_size = dcp_size
    proposer.parallel_drafting = parallel_drafting
    proposer.draft_model_config = SimpleNamespace(use_mla=False)
    proposer.query_start_loc_group = [torch.zeros(9, dtype=torch.int32) for _ in range(3)]
    proposer.seq_lens_group = [torch.tensor([32, 48, 0, 0, 0, 0, 0, 0], dtype=torch.int32) for _ in range(3)]
    device_buffer = proposer.query_start_loc_group[draft_index][:3]
    device_buffer.copy_(torch.tensor([0, 4, 8], dtype=torch.int32))
    cpu_buffer = device_buffer.clone()
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        query_start_loc_cpu=cpu_buffer,
        query_start_loc=device_buffer,
        seq_lens_cpu=torch.tensor([32, 48]),
        _seq_lens_cpu=torch.tensor([32, 48]),
    )
    proposer._prepare_dcp_draft_dummy_metadata(common, draft_index)

    assert common.query_start_loc.data_ptr() == device_buffer.data_ptr()
    assert cpu_buffer.tolist() == [0, 4, 8]
    expected = list(range(9)) if normalize else [0, 4, 8]
    assert common.num_reqs == (8 if normalize else 2)
    if normalize:
        assert common.seq_lens.tolist() == [32, 48, 0, 0, 0, 0, 0, 0]
        assert common.seq_lens_cpu.tolist() == [32, 48, 0, 0, 0, 0, 0, 0]
        assert common._seq_lens_cpu.tolist() == [32, 48, 0, 0, 0, 0, 0, 0]
    assert common.query_start_loc.tolist() == expected
    assert common.query_start_loc_cpu.tolist() == expected
    assert common.num_actual_tokens == (2 if normalize else 8)
    assert common.num_input_tokens == 8
    assert common.max_query_len == (1 if normalize else 4)
    if normalize:
        proposer._prepare_dcp_draft_dummy_metadata(common, 2)
        assert common.num_reqs == 8
        assert common.num_actual_tokens == 2
        assert common.query_start_loc.tolist() == list(range(9))
        assert common.seq_lens_cpu.tolist() == [32, 48, 0, 0, 0, 0, 0, 0]
    metadata = SimpleNamespace(causal=True, decode=SimpleNamespace(actual_seq_lengths_q=expected[1:]))
    assert use_history_current_split_decode(metadata, is_draft_model=True) == (not normalize)


@pytest.mark.parametrize("draft_index", [1, 2])
def test_mla_dcp_draft_dummy_metadata_preserves_existing_layout(draft_index):
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.dcp_size = 2
    proposer.parallel_drafting = False
    proposer.draft_model_config = SimpleNamespace(use_mla=True)
    query_start_loc = torch.tensor([0, 4, 8], dtype=torch.int32)
    seq_lens = torch.tensor([32, 48], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=8,
        num_input_tokens=8,
        max_query_len=4,
        query_start_loc_cpu=query_start_loc.clone(),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        _seq_lens_cpu=seq_lens.clone(),
        seq_lens_cpu_upper_bound=seq_lens.clone(),
        num_computed_tokens_cpu=torch.tensor([28, 44], dtype=torch.int32),
    )
    before = vars(common).copy()
    tensor_values = {key: value.clone() for key, value in before.items() if isinstance(value, torch.Tensor)}
    proposer._prepare_dcp_draft_dummy_metadata(common, draft_index)
    for key, value in before.items():
        assert getattr(common, key) is value
    for key, value in tensor_values.items():
        torch.testing.assert_close(getattr(common, key), value)
