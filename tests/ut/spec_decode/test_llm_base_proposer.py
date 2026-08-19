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
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from vllm.config import CUDAGraphMode
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM

from vllm_ascend.spec_decode.llm_base_proposer import (
    _HIDDEN_STATE_DRAFTER_TYPES,
    AscendSpecDecodeBaseProposer,
)
from vllm_ascend.spec_decode.utils import _disable_flash_comm_v1_context

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


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_parallel_draft_multimodal_embeddings_enter_sequence_parallelism(method):
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = method
    full_embeddings = torch.zeros(112, 8)
    sharded_embeddings = torch.zeros(7, 8)
    positions = torch.arange(112)
    sharded_positions = torch.arange(7)
    calls = []

    def fake_split_inputs_tp_to_sp(tensor, output):
        calls.append((tensor, output))
        return sharded_embeddings if tensor is full_embeddings else sharded_positions

    with (
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer._EXTRA_CTX",
            SimpleNamespace(flash_comm_v1_enabled=True),
        ),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.split_inputs_tp_to_sp",
            side_effect=fake_split_inputs_tp_to_sp,
        ),
    ):
        output_embeddings, output_positions = proposer._maybe_shard_parallel_draft_embeddings(
            full_embeddings,
            positions,
        )

    assert output_embeddings is sharded_embeddings
    assert output_positions is sharded_positions
    assert calls == [
        (full_embeddings, full_embeddings),
        (positions, positions),
    ]


def test_parallel_draft_text_ids_do_not_require_embedding_sharding():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    positions = torch.arange(7)

    embeddings, returned_positions = proposer._maybe_shard_parallel_draft_embeddings(None, positions)

    assert embeddings is None
    assert returned_positions is positions


def test_parallel_draft_mrope_positions_shard_token_axis():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    full_embeddings = torch.arange(24).reshape(6, 4)
    full_positions = torch.arange(18).reshape(3, 6)
    calls = []

    def fake_split_inputs_tp_to_sp(value, out):
        calls.append((value, out))
        return value[:3]

    with (
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer._EXTRA_CTX",
            SimpleNamespace(flash_comm_v1_enabled=True),
        ),
        patch(
            "vllm_ascend.spec_decode.llm_base_proposer.split_inputs_tp_to_sp",
            side_effect=fake_split_inputs_tp_to_sp,
        ),
    ):
        output_embeddings, output_positions = proposer._maybe_shard_parallel_draft_embeddings(
            full_embeddings,
            full_positions,
        )

    torch.testing.assert_close(output_embeddings, full_embeddings[:3])
    torch.testing.assert_close(output_positions, full_positions[:, :3])
    assert calls[1][0].shape == (6, 3)
    assert calls[1][1].shape == (6, 3)


class TestQuaRotDraftBoundaries:
    @staticmethod
    def _make_proposer() -> AscendSpecDecodeBaseProposer:
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.method = "dspark"
        proposer.device = torch.device("cpu")
        proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="target"))
        return proposer

    def test_anti_rotates_k3_context_projection(self, monkeypatch):
        proposer = self._make_proposer()
        rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        context_proj = nn.Linear(10, 2, bias=False)
        initial_weight = torch.arange(1.0, 21.0).reshape(2, 10)
        context_proj.weight.data.copy_(initial_weight)
        proposer.model = SimpleNamespace(model=SimpleNamespace(context_proj=context_proj))
        monkeypatch.setattr(
            proposer,
            "_load_quarot_rotation",
            lambda _: rotation,
        )

        proposer._maybe_anti_rotate_draft_projection()

        expected = torch.matmul(
            initial_weight.reshape(2, 5, 2),
            rotation,
        ).reshape(2, 10)
        torch.testing.assert_close(context_proj.weight, expected)
        torch.testing.assert_close(proposer._quarot_rotation, rotation)

    def test_incompatible_context_projection_fails_closed(self, monkeypatch):
        proposer = self._make_proposer()
        proposer.model = SimpleNamespace(model=SimpleNamespace(context_proj=nn.Linear(3, 2, bias=False)))
        monkeypatch.setattr(
            proposer,
            "_load_quarot_rotation",
            lambda _: torch.eye(2),
        )

        with pytest.raises(ValueError, match="incompatible"):
            proposer._maybe_anti_rotate_draft_projection()

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

        with pytest.raises(ValueError, match="refusing to alias"):
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


class TestDisableFlashCommV1Context:
    """``_disable_flash_comm_v1_context`` temporarily clears
    ``forward_context.flash_comm_v1_enabled`` while MarkovHead runs -- MarkovHead
    operates in the all-gathered full space, so SP's reduce-scatter must not
    split ``markov_emb`` -- then restores the original value on exit, including
    on exception. See commit c62ef687b ([BugFix] Fix `sp` in dspark).
    """

    @staticmethod
    def _patch_forward_context(monkeypatch, flash_comm_v1_enabled: bool):
        ctx = SimpleNamespace(flash_comm_v1_enabled=flash_comm_v1_enabled)
        monkeypatch.setattr(
            "vllm_ascend.spec_decode.utils.get_forward_context",
            lambda: ctx,
        )
        return ctx

    def test_clears_while_inside_when_sp_on(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with _disable_flash_comm_v1_context():
            assert ctx.flash_comm_v1_enabled is False

    def test_restores_true_on_exit(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with _disable_flash_comm_v1_context():
            pass
        assert ctx.flash_comm_v1_enabled is True

    def test_restores_false_on_exit(self, monkeypatch):
        """SP already off -> clearing is a no-op, original False preserved."""
        ctx = self._patch_forward_context(monkeypatch, False)
        with _disable_flash_comm_v1_context():
            assert ctx.flash_comm_v1_enabled is False
        assert ctx.flash_comm_v1_enabled is False

    def test_restores_on_exception(self, monkeypatch):
        ctx = self._patch_forward_context(monkeypatch, True)
        with pytest.raises(RuntimeError, match="boom"), _disable_flash_comm_v1_context():
            raise RuntimeError("boom")
        assert ctx.flash_comm_v1_enabled is True


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
