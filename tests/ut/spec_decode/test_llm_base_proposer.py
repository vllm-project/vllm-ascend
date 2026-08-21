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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

from vllm_ascend.spec_decode.llm_base_proposer import (
    _HIDDEN_STATE_DRAFTER_TYPES,
    AscendSpecDecodeBaseProposer,
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


def test_kimi_k3_dspark_is_supported_as_hidden_state_drafter():
    assert K3DSparkForCausalLM in _HIDDEN_STATE_DRAFTER_TYPES


class TestQuaRotDraftBoundaries:
    @staticmethod
    def _make_proposer() -> AscendSpecDecodeBaseProposer:
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "dspark"
        proposer.device = torch.device("cpu")
        proposer.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(model="target"),
            quant_config=SimpleNamespace(),
        )
        return proposer

    @pytest.mark.parametrize(
        "draft_class_name",
        ["K3DSparkForCausalLM", "Qwen3DSparkForCausalLM"],
    )
    def test_loads_rotation_for_shared_dspark_boundaries(self, monkeypatch, draft_class_name):
        class FakeDSpark:
            pass

        proposer = self._make_proposer()
        rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        proposer.model = FakeDSpark()
        monkeypatch.setattr(
            f"vllm_ascend.spec_decode.llm_base_proposer.{draft_class_name}",
            FakeDSpark,
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.llm_base_proposer.get_rotation_path",
            lambda _: "rotation.safetensors",
        )
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.llm_base_proposer.get_rotation_matrix",
            lambda _: rotation,
        )

        proposer._maybe_load_quarot_rotation()

        torch.testing.assert_close(proposer._quarot_rotation, rotation)

    def test_does_not_apply_dspark_boundary_rotation_to_other_drafts(self, monkeypatch):
        proposer = self._make_proposer()
        proposer.model = SimpleNamespace()
        calls = []
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.llm_base_proposer.get_rotation_path",
            lambda config: calls.append(config),
        )

        proposer._maybe_load_quarot_rotation()

        assert calls == []

    def test_materializes_unrotated_shared_layer(self):
        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        target = nn.Linear(2, 3, bias=False)
        target_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target.weight.data.copy_(target_weight)

        prepared = proposer._prepare_unrotated_shared_layer(
            None,
            target,
            "draft embed_tokens.weight",
        )

        assert prepared is not None
        assert prepared is not target
        torch.testing.assert_close(
            prepared.weight,
            target_weight @ proposer._quarot_rotation.T,
        )
        torch.testing.assert_close(target.weight, target_weight)

    def test_materialized_layer_reuses_noncopyable_comm_group(self):
        class NonCopyableCommGroup:
            def __deepcopy__(self, memo):
                del memo
                raise TypeError("cannot pickle ProcessGroup")

        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.eye(2)
        target = nn.Linear(2, 3, bias=False)
        target.comm_group = NonCopyableCommGroup()

        prepared = proposer._prepare_unrotated_shared_layer(
            None,
            target,
            "draft embed_tokens.weight",
        )

        assert prepared is not None
        assert prepared.comm_group is target.comm_group
        assert prepared.weight.data_ptr() != target.weight.data_ptr()

    def test_incompatible_shared_layer_fails_instead_of_aliasing(self):
        proposer = self._make_proposer()
        proposer._quarot_rotation = torch.eye(2)
        target = nn.Linear(2, 3, bias=False)
        draft = nn.Linear(3, 3, bias=False)

        with pytest.raises(RuntimeError):
            proposer._prepare_unrotated_shared_layer(
                draft,
                target,
                "draft lm_head.weight",
            )


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


class TestParallelDraftSeqLens:
    @staticmethod
    def _metadata():
        return SimpleNamespace(
            _seq_lens_cpu=torch.tensor([100, 200], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_cpu_upper_bound=None,
            parallel_draft_seq_lens_cpu=None,
            parallel_draft_num_reject_cpu=None,
            parallel_draft_num_reject_event=None,
            parallel_draft_num_reject_num_reqs=0,
        )

    @staticmethod
    def _prepare(proposer, metadata, has_rejected_tokens):
        AscendSpecDecodeBaseProposer._prepare_parallel_draft_seq_lens_cpu(
            proposer,
            metadata,
            batch_size=2,
            has_rejected_tokens=has_rejected_tokens,
        )

    def test_async_rejected_tokens_publish_deferred_finalize(self):
        reject_event = object()
        reject_cpu = torch.tensor([0, 2], dtype=torch.int32)
        proposer = SimpleNamespace(
            method="dspark",
            parallel_drafting=True,
            num_query_per_req=8,
            runner=SimpleNamespace(
                num_rejected_tokens_event=reject_event,
                num_rejected_tokens_cpu=reject_cpu,
            ),
        )
        metadata = self._metadata()

        self._prepare(proposer, metadata, has_rejected_tokens=True)

        torch.testing.assert_close(
            metadata.parallel_draft_seq_lens_cpu,
            torch.tensor([108, 208], dtype=torch.int32),
        )
        assert metadata.parallel_draft_num_reject_event is reject_event
        assert metadata.parallel_draft_num_reject_cpu is reject_cpu
        assert metadata.parallel_draft_num_reject_num_reqs == 2

    def test_non_async_rejected_tokens_keep_device_fallback(self):
        proposer = SimpleNamespace(
            method="dspark",
            parallel_drafting=True,
            num_query_per_req=8,
            runner=SimpleNamespace(
                num_rejected_tokens_event=None,
                num_rejected_tokens_cpu=None,
            ),
        )
        metadata = self._metadata()

        self._prepare(proposer, metadata, has_rejected_tokens=True)

        assert metadata.parallel_draft_seq_lens_cpu is None

    def test_first_parallel_draft_pass_extends_host_lengths(self):
        proposer = SimpleNamespace(
            method="dspark",
            parallel_drafting=True,
            num_query_per_req=8,
            runner=SimpleNamespace(
                num_rejected_tokens_event=None,
                num_rejected_tokens_cpu=None,
            ),
        )
        metadata = self._metadata()

        self._prepare(proposer, metadata, has_rejected_tokens=False)

        torch.testing.assert_close(
            metadata.parallel_draft_seq_lens_cpu,
            torch.tensor([108, 208], dtype=torch.int32),
        )
