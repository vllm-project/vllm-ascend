from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase

PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = 0.0


def mock_pin_memory(original_func):
    def func_wo_pin_memory(*args, **kwargs):
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return original_func(*args, **kwargs)

    return func_wo_pin_memory


def _mock_ascend_config():
    mock_config = MagicMock()
    mock_config.enable_reduce_sample = False
    return mock_config


def _make_sampling_metadata(
    batch_size: int,
    all_greedy: bool = True,
    temperature: float | None = None,
    top_p: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
    generators: dict | None = None,
    no_penalties: bool = True,
    prompt_token_ids: torch.Tensor | None = None,
    output_token_ids: list[list[int]] | None = None,
    frequency_penalties: torch.Tensor | None = None,
    presence_penalties: torch.Tensor | None = None,
    repetition_penalties: torch.Tensor | None = None,
):
    from dataclasses import fields

    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    if temperature is None:
        temperature_val = GREEDY_TEMPERATURE if all_greedy else 1.0
        temperature = torch.full((batch_size,), temperature_val, dtype=torch.float32)
    if generators is None:
        generators = {}
    if output_token_ids is None:
        output_token_ids = [[] for _ in range(batch_size)]
    if frequency_penalties is None:
        frequency_penalties = torch.zeros(batch_size, dtype=torch.float32)
    if presence_penalties is None:
        presence_penalties = torch.zeros(batch_size, dtype=torch.float32)
    if repetition_penalties is None:
        repetition_penalties = torch.ones(batch_size, dtype=torch.float32)

    kwargs = {
        "temperature": temperature,
        "all_greedy": all_greedy,
        "all_random": not all_greedy,
        "top_p": top_p,
        "top_k": top_k,
        "generators": generators,
        "max_num_logprobs": None,
        "no_penalties": no_penalties,
        "prompt_token_ids": prompt_token_ids,
        "frequency_penalties": frequency_penalties,
        "presence_penalties": presence_penalties,
        "repetition_penalties": repetition_penalties,
        "output_token_ids": output_token_ids,
        "allowed_token_ids_mask": None,
        "bad_words_token_ids": {},
        "logitsprocs": LogitsProcessors([]),
        "logprob_token_ids": None,
        "spec_token_ids": None,
    }

    field_names = {f.name for f in fields(SamplingMetadata)}
    if "thinking_budget_state_holder" in field_names:
        kwargs["thinking_budget_state_holder"] = None

    return SamplingMetadata(**kwargs)


def _make_spec_decode_metadata(
    num_draft_tokens: list[int],
    draft_token_ids: torch.Tensor,
):
    import numpy as np
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

    batch_size = len(num_draft_tokens)
    num_tokens = len(draft_token_ids)
    num_sampled = [n + 1 for n in num_draft_tokens]

    cu_num_draft = np.cumsum(num_draft_tokens, dtype=np.int32)
    cu_num_draft_tensor = torch.from_numpy(cu_num_draft)
    cu_num_sampled = np.cumsum(num_sampled, dtype=np.int32)
    cu_num_sampled_tensor = torch.from_numpy(cu_num_sampled)

    target_logits_indices = torch.zeros(num_tokens, dtype=torch.int32)
    bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32)
    logits_indices = torch.zeros(num_tokens + batch_size, dtype=torch.int32)

    return SpecDecodeMetadata(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=num_draft_tokens,
        cu_num_draft_tokens=cu_num_draft_tensor,
        cu_num_sampled_tokens=cu_num_sampled_tensor,
        target_logits_indices=target_logits_indices,
        bonus_logits_indices=bonus_logits_indices,
        logits_indices=logits_indices,
    )


class TestRejectionSamplerDPSharding(TestBase):
    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    @patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False)
    @patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", side_effect=_mock_ascend_config)
    @patch("vllm_ascend.sample.rejection_sampler.get_tp_group")
    def test_dp_sharding_empty_local_batch(self, mock_get_tp_group, mock_get_ascend_config):
        from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler

        mock_sampler = MagicMock()
        rejection_sampler = AscendRejectionSampler(mock_sampler)
        rejection_sampler.synthetic_mode = False
        rejection_sampler.synthetic_conditional_rates = None
        rejection_sampler.is_processed_logprobs_mode = False

        mock_tp_group = MagicMock()
        mock_tp_group.world_size = 4
        mock_tp_group.rank_in_group = 3

        def all_gather_impl(tensor, dim=0):
            return tensor.repeat(4, *([1] * (tensor.ndim - 1)))

        mock_tp_group.all_gather.side_effect = all_gather_impl
        mock_get_tp_group.return_value = mock_tp_group

        num_draft_tokens = [2, 3]
        batch_size = len(num_draft_tokens)
        max_spec_len = 3
        chunk_size = (batch_size + 4 - 1) // 4

        draft_token_ids = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
        metadata = _make_spec_decode_metadata(num_draft_tokens, draft_token_ids)

        vocab_size = 100
        num_tokens_total = len(draft_token_ids)
        target_logits = torch.randn(num_tokens_total, vocab_size, dtype=torch.float32)
        bonus_token_ids = torch.tensor([[99], [98]], dtype=torch.int32)
        sampling_metadata = _make_sampling_metadata(batch_size, all_greedy=True)

        start_batch = 3 * chunk_size
        end_batch = min(start_batch + chunk_size, batch_size)

        result = rejection_sampler._rejection_sample_with_dp_sharding(
            metadata=metadata,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
            start_batch=start_batch,
            end_batch=end_batch,
            chunk_size=chunk_size,
            skip_target_logits_sampling=False,
        )

        assert result.shape == (batch_size, max_spec_len + 1)
        assert (result == PLACEHOLDER_TOKEN_ID).all()

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    @patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False)
    @patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", side_effect=_mock_ascend_config)
    @patch("vllm_ascend.sample.sampler.get_ascend_config", side_effect=_mock_ascend_config)
    @patch("vllm_ascend.sample.rejection_sampler.get_tp_group")
    def test_dp_sharding_vs_non_dp_greedy(self, mock_get_tp_group, mock_sampler_config, mock_rejection_config):
        from vllm_ascend.sample.rejection_sampler import (
            AscendRejectionSampler,
            apply_sampling_constraints,
            rejection_sample,
        )

        torch.manual_seed(42)

        mock_sampler = MagicMock()
        rejection_sampler = AscendRejectionSampler(mock_sampler)
        rejection_sampler.synthetic_mode = False
        rejection_sampler.synthetic_conditional_rates = None
        rejection_sampler.is_processed_logprobs_mode = False

        mock_tp_group = MagicMock()
        mock_tp_group.world_size = 2
        mock_tp_group.rank_in_group = 0

        def all_gather_impl(tensor, dim=0):
            return tensor.repeat(2, *([1] * (tensor.ndim - 1)))

        mock_tp_group.all_gather.side_effect = all_gather_impl
        mock_get_tp_group.return_value = mock_tp_group

        num_draft_tokens = [2, 3, 1, 2]
        batch_size = len(num_draft_tokens)
        vocab_size = 100

        draft_token_ids = torch.randint(0, vocab_size, (sum(num_draft_tokens),), dtype=torch.int32)
        metadata = _make_spec_decode_metadata(num_draft_tokens, draft_token_ids)

        target_logits = torch.randn(sum(num_draft_tokens), vocab_size, dtype=torch.float32)
        bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.int32)
        sampling_metadata = _make_sampling_metadata(batch_size, all_greedy=True)

        chunk_size = (batch_size + 2 - 1) // 2

        result_dp_rank0 = rejection_sampler._rejection_sample_with_dp_sharding(
            metadata=metadata,
            target_logits=target_logits.clone(),
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
            start_batch=0,
            end_batch=chunk_size,
            chunk_size=chunk_size,
            skip_target_logits_sampling=False,
        )

        mock_tp_group.rank_in_group = 1
        result_dp_rank1 = rejection_sampler._rejection_sample_with_dp_sharding(
            metadata=metadata,
            target_logits=target_logits.clone(),
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
            start_batch=chunk_size,
            end_batch=batch_size,
            chunk_size=chunk_size,
            skip_target_logits_sampling=False,
        )

        result_dp = torch.cat([result_dp_rank0[:chunk_size], result_dp_rank1[: batch_size - chunk_size]], dim=0)

        processed_logits = rejection_sampler.apply_logits_processors(target_logits.clone(), sampling_metadata, metadata)
        processed_logits = apply_sampling_constraints(
            processed_logits, metadata.cu_num_draft_tokens, sampling_metadata, rejection_sampler.top_k
        )
        result_non_dp = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            None,
            processed_logits,
            bonus_token_ids,
            sampling_metadata,
        )

        assert result_dp.shape == result_non_dp.shape
        assert torch.equal(result_dp, result_non_dp)

    @patch("torch.arange", new=mock_pin_memory(torch.arange))
    @patch("torch.ones", new=mock_pin_memory(torch.ones))
    @patch("torch.full", new=mock_pin_memory(torch.full))
    @patch("torch.tensor", new=mock_pin_memory(torch.tensor))
    @patch("vllm_ascend.sample.rejection_sampler.HAS_TRITON", False)
    @patch("vllm_ascend.sample.rejection_sampler.get_ascend_config", side_effect=_mock_ascend_config)
    @patch("vllm_ascend.sample.sampler.get_ascend_config", side_effect=_mock_ascend_config)
    @patch("vllm_ascend.sample.rejection_sampler.get_tp_group")
    def test_dp_sharding_vs_non_dp_penalties(self, mock_get_tp_group, mock_sampler_config, mock_rejection_config):
        from vllm_ascend.sample.rejection_sampler import (
            AscendRejectionSampler,
            apply_sampling_constraints,
            rejection_sample,
        )

        torch.manual_seed(42)

        mock_sampler = MagicMock()
        rejection_sampler = AscendRejectionSampler(mock_sampler)
        rejection_sampler.synthetic_mode = False
        rejection_sampler.synthetic_conditional_rates = None
        rejection_sampler.is_processed_logprobs_mode = False

        mock_tp_group = MagicMock()
        mock_tp_group.world_size = 2
        mock_tp_group.rank_in_group = 0

        def all_gather_impl(tensor, dim=0):
            return tensor.repeat(2, *([1] * (tensor.ndim - 1)))

        mock_tp_group.all_gather.side_effect = all_gather_impl
        mock_get_tp_group.return_value = mock_tp_group

        num_draft_tokens = [2, 3, 1, 2]
        batch_size = len(num_draft_tokens)
        vocab_size = 100

        draft_token_ids = torch.randint(0, vocab_size, (sum(num_draft_tokens),), dtype=torch.int32)
        metadata = _make_spec_decode_metadata(num_draft_tokens, draft_token_ids)

        target_logits = torch.randn(sum(num_draft_tokens), vocab_size, dtype=torch.float32)
        bonus_token_ids = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.int32)

        prompt_token_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        output_token_ids = [[10, 20], [30], [40, 50, 60], [70]]

        generators = {i: torch.Generator().manual_seed(42 + i) for i in range(batch_size)}
        sampling_metadata = _make_sampling_metadata(
            batch_size,
            all_greedy=True,
            no_penalties=False,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
            frequency_penalties=torch.tensor([0.1, 0.2, 0.0, 0.3]),
            presence_penalties=torch.tensor([0.0, 0.1, 0.2, 0.0]),
            repetition_penalties=torch.tensor([1.0, 1.1, 1.2, 1.0]),
            generators=generators,
        )

        chunk_size = (batch_size + 2 - 1) // 2

        result_dp_rank0 = rejection_sampler._rejection_sample_with_dp_sharding(
            metadata=metadata,
            target_logits=target_logits.clone(),
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
            start_batch=0,
            end_batch=chunk_size,
            chunk_size=chunk_size,
            skip_target_logits_sampling=False,
        )

        mock_tp_group.rank_in_group = 1
        result_dp_rank1 = rejection_sampler._rejection_sample_with_dp_sharding(
            metadata=metadata,
            target_logits=target_logits.clone(),
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
            start_batch=chunk_size,
            end_batch=batch_size,
            chunk_size=chunk_size,
            skip_target_logits_sampling=False,
        )

        result_dp = torch.cat([result_dp_rank0[:chunk_size], result_dp_rank1[: batch_size - chunk_size]], dim=0)

        processed_logits = rejection_sampler.apply_logits_processors(target_logits.clone(), sampling_metadata, metadata)
        processed_logits = apply_sampling_constraints(
            processed_logits, metadata.cu_num_draft_tokens, sampling_metadata, rejection_sampler.top_k
        )
        result_non_dp = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            None,
            processed_logits,
            bonus_token_ids,
            sampling_metadata,
        )

        assert result_dp.shape == result_non_dp.shape
        assert torch.equal(result_dp, result_non_dp)
