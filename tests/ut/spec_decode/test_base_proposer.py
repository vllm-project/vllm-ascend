# ruff: noqa: E501
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend.spec_decode.eagle_proposer import AscendSpecDecodeBaseProposer

_LMHEAD_TARGET = "vllm_ascend.spec_decode.llm_base_proposer.lmhead_tp_enable"
_EXTRA_CTX_TARGET = "vllm_ascend.spec_decode.llm_base_proposer._EXTRA_CTX"
_FORWARD_CTX_TARGET = "vllm_ascend.spec_decode.llm_base_proposer.get_forward_context"
_PCP_GROUP_TARGET = "vllm_ascend.spec_decode.llm_base_proposer.get_pcp_group"


def _make_proposer_mock(**attrs):
    """Create a MagicMock with the right spec and extra attributes for testing."""
    mock = MagicMock(spec=AscendSpecDecodeBaseProposer)
    for k, v in attrs.items():
        setattr(mock, k, v)
    return mock


class TestBaseProposerUsesDraftVocabRemapping:
    def test_no_remapping_returns_false(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(model=MagicMock(spec=object))
        del instance.model.draft_id_to_target_id

        result = AscendSpecDecodeBaseProposer._uses_draft_vocab_remapping(instance)
        assert result is False

    def test_remapping_none_returns_false(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(model=MagicMock(draft_id_to_target_id=None))

        result = AscendSpecDecodeBaseProposer._uses_draft_vocab_remapping(instance)
        assert result is False

    def test_remapping_exists_returns_true(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(model=MagicMock(draft_id_to_target_id={0: 0, 1: 1}))

        result = AscendSpecDecodeBaseProposer._uses_draft_vocab_remapping(instance)
        assert result is True


class TestBaseProposerCanUseLocalArgmaxReduction:
    def test_all_conditions_met(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(
            use_local_argmax_reduction=True,
        )
        instance._uses_draft_vocab_remapping.return_value = False

        with patch(_LMHEAD_TARGET, return_value=False):
            result = AscendSpecDecodeBaseProposer._can_use_local_argmax_reduction(instance)
            assert result is True

    def test_lmhead_tp_disables(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(
            use_local_argmax_reduction=True,
        )

        with patch(_LMHEAD_TARGET, return_value=True):
            result = AscendSpecDecodeBaseProposer._can_use_local_argmax_reduction(instance)
            assert result is False

    def test_disabled_by_config(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(
            use_local_argmax_reduction=False,
        )

        with patch(_LMHEAD_TARGET, return_value=False):
            result = AscendSpecDecodeBaseProposer._can_use_local_argmax_reduction(instance)
            assert result is False

    def test_vocab_remapping_disables(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(
            use_local_argmax_reduction=True,
        )
        instance._uses_draft_vocab_remapping.return_value = True

        with patch(_LMHEAD_TARGET, return_value=False):
            result = AscendSpecDecodeBaseProposer._can_use_local_argmax_reduction(instance)
            assert result is False


class TestBaseProposerDraftArgmax:
    def test_local_argmax_when_applicable(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        expected = torch.tensor([1, 2, 3])
        instance = _make_proposer_mock(
            model=MagicMock(get_top_tokens=MagicMock(return_value=expected)),
        )
        instance._can_use_local_argmax_reduction.return_value = True

        hidden_states = torch.randn(3, 64)
        result = AscendSpecDecodeBaseProposer._draft_argmax(instance, hidden_states, num_indices=3)

        instance.model.get_top_tokens.assert_called_once_with(hidden_states)
        assert torch.equal(result, expected)

    def test_fallback_to_compute_logits(self):
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        logits = torch.randn(3, 100)
        instance = _make_proposer_mock(
            model=MagicMock(compute_logits=MagicMock(return_value=logits)),
        )
        instance._can_use_local_argmax_reduction.return_value = False

        hidden_states = torch.randn(3, 64)
        with patch(_LMHEAD_TARGET, return_value=False):
            result = AscendSpecDecodeBaseProposer._draft_argmax(instance, hidden_states, num_indices=3)

        instance.model.compute_logits.assert_called_once_with(hidden_states)
        assert result.shape == (3,)

    def test_fallback_with_lmhead_tp_truncates_logits(self):
        """When lmhead_tp_enable and num_indices < logits.shape[0], logits are truncated."""
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        logits = torch.randn(5, 100)
        instance = _make_proposer_mock(
            model=MagicMock(compute_logits=MagicMock(return_value=logits)),
        )
        instance._can_use_local_argmax_reduction.return_value = False

        hidden_states = torch.randn(5, 64)
        with patch(_LMHEAD_TARGET, return_value=True):
            result = AscendSpecDecodeBaseProposer._draft_argmax(instance, hidden_states, num_indices=3)

        instance.model.compute_logits.assert_called_once_with(hidden_states)
        assert result.shape == (3,)

    def test_fallback_with_lmhead_tp_no_truncate(self):
        """When lmhead_tp_enable but num_indices >= logits.shape[0], no truncation."""
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        logits = torch.randn(3, 100)
        instance = _make_proposer_mock(
            model=MagicMock(compute_logits=MagicMock(return_value=logits)),
        )
        instance._can_use_local_argmax_reduction.return_value = False

        hidden_states = torch.randn(3, 64)
        with patch(_LMHEAD_TARGET, return_value=True):
            result = AscendSpecDecodeBaseProposer._draft_argmax(instance, hidden_states, num_indices=3)

        instance.model.compute_logits.assert_called_once_with(hidden_states)
        assert result.shape == (3,)

    def test_local_argmax_raises_when_model_lacks_get_top_tokens(self):
        """When local argmax is enabled but model has no get_top_tokens, raise ValueError."""
        from vllm_ascend.spec_decode.eagle_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        instance = _make_proposer_mock(
            model=MagicMock(spec=object),
        )
        # Remove get_top_tokens from the model mock
        del instance.model.get_top_tokens
        instance._can_use_local_argmax_reduction.return_value = True

        hidden_states = torch.randn(3, 64)
        with pytest.raises(ValueError, match="does not implement get_top_tokens"):
            AscendSpecDecodeBaseProposer._draft_argmax(instance, hidden_states, num_indices=3)


class TestBaseProposerMergeDraft(TestBase):
    def _make_instance(self, **overrides):
        """Create a minimal mock instance for _run_merged_draft testing."""
        attrs = dict(
            input_ids=torch.arange(256, device="cpu"),
            hidden_states=torch.randn(256, 64),
            model=MagicMock(),
            method="mtp",
            pass_hidden_states_to_model=False,
            num_speculative_tokens=1,
            parallel_drafting=False,
            pcp_size=1,
            dcp_size=1,
            use_cuda_graph=False,
            uses_mrope=False,
            device="cpu",
            supports_mm_inputs=False,
            _draft_attn_layer_names=[],
            vllm_config=MagicMock(
                model_config=MagicMock(max_model_len=1024),
            ),
        )
        attrs.update(overrides)

        instance = _make_proposer_mock(**attrs)

        instance.model_returns_tuple = MagicMock(return_value=False)
        instance._get_positions = MagicMock(return_value=torch.arange(10))
        instance._draft_argmax = MagicMock(return_value=torch.tensor([0, 1, 2]))
        instance.maybe_all_gather_and_unpad = MagicMock(
            return_value=(
                torch.randn(10, 64),
                torch.arange(10),
                torch.randn(10, 64),
            )
        )
        instance.maybe_pad_and_reduce = MagicMock(
            return_value=(
                torch.randn(10, 64),
                torch.arange(10),
            )
        )
        instance._set_positions = MagicMock()
        return instance

    def _call_run_merged_draft(self, instance, token_indices=None, **kwargs):
        """Call _run_merged_draft with patched lmhead_tp_enable and standard args."""
        if token_indices is None:
            token_indices = torch.tensor([0, 1, 2])

        defaults = dict(
            num_input_tokens=5,
            batch_size=3,
            token_indices_to_sample=token_indices,
            target_positions=None,
            inputs_embeds=None,
            multi_steps_attn_metadata=None,
            num_tokens=5,
        )
        defaults.update(kwargs)

        with patch(_LMHEAD_TARGET, return_value=False), patch(_EXTRA_CTX_TARGET, MagicMock()):
            return AscendSpecDecodeBaseProposer._run_merged_draft(instance, **defaults)

    def test_early_exit_single_token(self):
        instance = self._make_instance(num_speculative_tokens=1)
        result = self._call_run_merged_draft(instance)
        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 1)

    def test_parallel_drafting_early_exit(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            parallel_drafting=True,
        )
        result = self._call_run_merged_draft(instance)
        instance._draft_argmax.assert_called_once()
        assert result.shape == (1, 3)

    def test_lmhead_tp_path(self):
        instance = self._make_instance(
            num_speculative_tokens=1,
            vllm_config=MagicMock(scheduler_config=MagicMock(max_num_seqs=32)),
            runner=MagicMock(uniform_decode_query_len=4),
        )

        token_indices = torch.tensor([0, 1, 2])
        with patch(_LMHEAD_TARGET, return_value=True):
            result = AscendSpecDecodeBaseProposer._run_merged_draft(
                instance,
                num_input_tokens=5,
                batch_size=3,
                token_indices_to_sample=token_indices,
                target_positions=None,
                inputs_embeds=None,
                multi_steps_attn_metadata=None,
                num_tokens=5,
            )

        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 1)

    def test_dflash_method(self):
        instance = self._make_instance(
            method="dflash",
            num_speculative_tokens=1,
        )
        instance.build_model_inputs_first_pass = MagicMock(
            return_value={
                "input_ids": instance.input_ids[:5],
            }
        )

        result = self._call_run_merged_draft(instance)
        instance.build_model_inputs_first_pass.assert_called_once_with(5)
        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 1)

    def test_pass_hidden_states_to_model(self):
        instance = self._make_instance(
            num_speculative_tokens=1,
            pass_hidden_states_to_model=True,
        )
        result = self._call_run_merged_draft(instance)
        instance.maybe_pad_and_reduce.assert_called()
        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 1)

    def test_model_returns_tuple(self):
        last_hs = torch.randn(5, 64)
        hs = torch.randn(5, 64)
        instance = self._make_instance(num_speculative_tokens=1)
        instance.model_returns_tuple = MagicMock(return_value=True)
        instance.model.return_value = (last_hs, hs)

        result = self._call_run_merged_draft(instance)
        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 1)

    def test_multi_step_generation(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            arange=torch.arange(5),
            positions=torch.arange(5),
        )
        instance.maybe_all_gather_and_unpad = MagicMock(
            return_value=(
                torch.randn(3, 64),
                torch.arange(3),
                torch.randn(3, 64),
            )
        )
        instance._draft_argmax = MagicMock(return_value=torch.tensor([0, 1, 2]))

        with patch(_FORWARD_CTX_TARGET, return_value=MagicMock()):
            result = self._call_run_merged_draft(instance)

        assert instance._draft_argmax.call_count == 3
        assert result.shape == (3, 3)

    def test_multi_step_with_lmhead_tp(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            method="mtp",
            arange=torch.arange(128),
            positions=torch.arange(128),
            vllm_config=MagicMock(
                scheduler_config=MagicMock(max_num_seqs=32),
                model_config=MagicMock(max_model_len=1024),
            ),
            runner=MagicMock(uniform_decode_query_len=4),
        )
        instance.maybe_all_gather_and_unpad = MagicMock(
            return_value=(
                torch.randn(128, 64),
                torch.arange(128),
                torch.randn(128, 64),
            )
        )
        instance._draft_argmax = MagicMock(return_value=torch.arange(3))

        token_indices = torch.tensor([0, 1, 2])
        with (
            patch(_LMHEAD_TARGET, return_value=True),
            patch(_EXTRA_CTX_TARGET, MagicMock()),
            patch(_FORWARD_CTX_TARGET, return_value=MagicMock()),
        ):
            result = AscendSpecDecodeBaseProposer._run_merged_draft(
                instance,
                num_input_tokens=5,
                batch_size=3,
                token_indices_to_sample=token_indices,
                target_positions=None,
                inputs_embeds=None,
                multi_steps_attn_metadata=None,
                num_tokens=5,
            )

        assert instance._draft_argmax.call_count == 3
        assert result.shape == (3, 3)

    def test_supports_mm_inputs(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            supports_mm_inputs=True,
            arange=torch.arange(5),
            positions=torch.arange(5),
            inputs_embeds=torch.randn(10, 64),
        )
        instance.maybe_all_gather_and_unpad = MagicMock(
            return_value=(
                torch.randn(3, 64),
                torch.arange(3),
                torch.randn(3, 64),
            )
        )
        instance.model.embed_input_ids = MagicMock(return_value=torch.randn(3, 64))
        instance._draft_argmax = MagicMock(return_value=torch.tensor([0, 1, 2]))

        with patch(_FORWARD_CTX_TARGET, return_value=MagicMock()):
            result = self._call_run_merged_draft(instance)

        instance.model.embed_input_ids.assert_called()
        assert result.shape == (3, 3)

    def test_uses_mrope_path(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            uses_mrope=True,
            mrope_positions=torch.arange(30).view(3, 10),
            arange=torch.arange(5),
            positions=torch.arange(5),
        )
        instance.maybe_all_gather_and_unpad = MagicMock(
            return_value=(
                torch.randn(3, 64),
                torch.arange(3),
                torch.randn(3, 64),
            )
        )
        instance._draft_argmax = MagicMock(return_value=torch.tensor([0, 1, 2]))

        with patch(_FORWARD_CTX_TARGET, return_value=MagicMock()):
            result = self._call_run_merged_draft(instance)

        assert instance._draft_argmax.call_count == 3
        assert result.shape == (3, 3)

    def test_multi_step_with_pass_hidden_states(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            pass_hidden_states_to_model=True,
            arange=torch.arange(5),
            positions=torch.arange(5),
        )
        instance.maybe_all_gather_and_unpad = MagicMock(
            return_value=(
                torch.randn(3, 64),
                torch.arange(3),
                torch.randn(3, 64),
            )
        )
        instance._draft_argmax = MagicMock(return_value=torch.tensor([0, 1, 2]))

        with patch(_FORWARD_CTX_TARGET, return_value=MagicMock()):
            result = self._call_run_merged_draft(instance)

        assert instance._draft_argmax.call_count == 3
        assert result.shape == (3, 3)

    def test_multi_step_model_returns_tuple(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            arange=torch.arange(5),
            positions=torch.arange(5),
        )
        instance.model_returns_tuple = MagicMock(return_value=True)
        instance.model.return_value = (torch.randn(5, 64), torch.randn(5, 64))
        instance.maybe_all_gather_and_unpad = MagicMock(
            return_value=(
                torch.randn(3, 64),
                torch.arange(3),
                torch.randn(3, 64),
            )
        )
        instance._draft_argmax = MagicMock(return_value=torch.tensor([0, 1, 2]))

        with patch(_FORWARD_CTX_TARGET, return_value=MagicMock()):
            result = self._call_run_merged_draft(instance)

        assert instance._draft_argmax.call_count == 3
        assert result.shape == (3, 3)

    def test_pcp_size_gt_one_path(self):
        instance = self._make_instance(
            num_speculative_tokens=1,
            pcp_size=2,
            runner=MagicMock(
                pcp_manager=MagicMock(
                    pcp_allgather_restore_idx=MagicMock(gpu=torch.arange(20)),
                ),
            ),
        )
        mock_pcp_group = MagicMock()
        mock_pcp_group.all_gather.return_value = torch.randn(20, 64)

        token_indices = torch.tensor([0, 1, 2])
        with patch(_LMHEAD_TARGET, return_value=False), patch(_PCP_GROUP_TARGET, return_value=mock_pcp_group):
            result = AscendSpecDecodeBaseProposer._run_merged_draft(
                instance,
                num_input_tokens=5,
                batch_size=3,
                token_indices_to_sample=token_indices,
                target_positions=None,
                inputs_embeds=None,
                multi_steps_attn_metadata=None,
                num_tokens=5,
            )

        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 1)

    def test_pcp_size_gt_one_eagle_path(self):
        instance = self._make_instance(
            num_speculative_tokens=1,
            pcp_size=2,
            method="eagle",
            runner=MagicMock(
                pcp_manager=MagicMock(
                    pcp_allgather_restore_idx=MagicMock(gpu=torch.arange(20)),
                ),
            ),
        )
        mock_pcp_group = MagicMock()
        mock_pcp_group.all_gather.return_value = torch.randn(20, 64)

        token_indices = torch.tensor([0, 1, 2])
        with patch(_LMHEAD_TARGET, return_value=False), patch(_PCP_GROUP_TARGET, return_value=mock_pcp_group):
            result = AscendSpecDecodeBaseProposer._run_merged_draft(
                instance,
                num_input_tokens=5,
                batch_size=3,
                token_indices_to_sample=token_indices,
                target_positions=None,
                inputs_embeds=None,
                multi_steps_attn_metadata=None,
                num_tokens=5,
            )

        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 1)

    def test_pcp_and_dcp_gt_one_prefill_path(self):
        instance = self._make_instance(
            num_speculative_tokens=3,
            pcp_size=1,
            dcp_size=4,
            parallel_drafting=False,
        )
        instance._draft_argmax.return_value = torch.tensor([0, 1, 2])

        token_indices = torch.tensor([0, 1, 2])
        with patch(_LMHEAD_TARGET, return_value=False):
            result = AscendSpecDecodeBaseProposer._run_merged_draft(
                instance,
                num_input_tokens=5,
                batch_size=3,
                token_indices_to_sample=token_indices,
                target_positions=None,
                inputs_embeds=None,
                multi_steps_attn_metadata=None,
                num_tokens=5,
                is_prefill=True,
            )

        instance._draft_argmax.assert_called_once()
        assert result.shape == (3, 3)


if __name__ == "__main__":
    unittest.main()
