# SPDX-License-Identifier: Apache-2.0
"""Test DSpark local-argmax selection and the MRV1 sampling path."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.spec_decode.llm_base_proposer as proposer_module
from vllm_ascend.models.deepseek_v4.dspark import DSparkDeepseekV4ForCausalLM
from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("deepseek", [False, True])
@pytest.mark.parametrize("parallel_mode", list(VocabParallelMode))
def test_local_argmax_selection(enabled, deepseek, parallel_mode):
    model = Mock(spec=DSparkDeepseekV4ForCausalLM) if deepseek else Mock()
    model.lm_head = SimpleNamespace(parallel_mode=parallel_mode)
    proposer = SimpleNamespace(model=model, use_local_argmax_reduction=enabled)
    expected = enabled and deepseek and parallel_mode is VocabParallelMode.STANDARD
    assert AscendSpecDecodeBaseProposer._can_use_dspark_local_argmax(proposer) is expected


@pytest.mark.parametrize("soft_cap,scale", [(None, 1.0), (2.0, 1.0), (None, 0.5), (2.0, 0.5)])
def test_local_projection_keeps_normalization_and_logits_transforms(soft_cap, scale):
    hidden = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, 0.5]])
    norm = Mock(side_effect=lambda values: values / 2)
    apply_head = Mock(side_effect=lambda head, values, bias: values @ head.T)
    model = SimpleNamespace(
        lm_head=weight,
        model=SimpleNamespace(norm=norm),
        logits_processor=SimpleNamespace(_apply_head=apply_head, soft_cap=soft_cap, scale=scale),
    )
    actual = DSparkDeepseekV4ForCausalLM.compute_local_draft_logits(model, hidden)
    expected = (hidden / 2) @ weight.T
    if soft_cap is not None:
        expected = torch.tanh(expected / soft_cap) * soft_cap
    expected *= scale
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    norm.assert_called_once()
    apply_head.assert_called_once()


@pytest.mark.parametrize("steps", [5, 7])
@pytest.mark.parametrize("mode", ["local", "disabled", "probabilistic", "mapping", "other_model", "pcp_sharded"])
def test_sampling_preserves_buffers_fallbacks_and_dynamic_update(steps, mode, monkeypatch):
    batch, vocab = 2, 11
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(batch * steps, vocab, generator=generator)
    seeds = torch.tensor([3, 7])
    hidden = torch.zeros(batch * steps, 4)
    model = (
        Mock(return_value=hidden)
        if mode in ("mapping", "other_model")
        else Mock(spec=DSparkDeepseekV4ForCausalLM, return_value=hidden)
    )
    model.config = SimpleNamespace(vocab_size=vocab)
    model.lm_head = SimpleNamespace(
        shard_indices=SimpleNamespace(org_vocab_start_index=0),
        parallel_mode=VocabParallelMode.PCP_X_TP if mode == "pcp_sharded" else VocabParallelMode.STANDARD,
    )
    model.draft_id_to_target_id = torch.arange(vocab) if mode == "mapping" else None
    model.compute_local_draft_logits = Mock(side_effect=lambda _: logits.clone())
    model.compute_draft_logits = Mock(side_effect=lambda _: logits.clone())
    model.markov_embed = lambda tokens: tokens

    def bias(tokens):
        values = torch.zeros(batch, vocab)
        values.scatter_(1, ((tokens + 1) % vocab).unsqueeze(-1), 4)
        return values

    model.markov_bias = bias
    if mode == "mapping":
        model.map_draft_to_target = lambda tokens: (tokens + 2) % vocab
    proposer = SimpleNamespace(
        model=model,
        method="dspark",
        runner=None,
        num_speculative_tokens=steps,
        parallel_drafting=True,
        use_local_argmax_reduction=mode not in ("disabled", "probabilistic"),
        _enable_probabilistic_draft_probs=mode == "probabilistic",
        _dspark_seed_buffer=seeds,
        _dspark_draft_buffer=torch.full((batch + 1, steps + 1), -1, dtype=torch.int64),
        _share_mtp_indices=False,
        _context_slot_mapping_buffers={},
        input_ids=torch.zeros(batch * steps, dtype=torch.int64),
        _get_positions=lambda count: torch.arange(count),
        build_model_inputs_first_pass=Mock(),
        dynamic_spec=SimpleNamespace(update=Mock()),
        _sample_draft_from_logits=lambda values, metadata: (values.argmax(-1), values.softmax(-1)),
    )
    proposer._can_use_dspark_local_argmax = MethodType(
        AscendSpecDecodeBaseProposer._can_use_dspark_local_argmax, proposer
    )
    monkeypatch.setattr(proposer_module, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(proposer_module, "get_ascend_config", lambda: SimpleNamespace(enable_reduce_sample=False))
    monkeypatch.setattr(
        proposer_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, all_gather=Mock(side_effect=AssertionError)),
    )
    expected = torch.empty(batch, steps + 1, dtype=torch.int64)
    expected[:, 0] = seeds
    for step in range(steps):
        tokens = (logits.view(batch, steps, vocab)[:, step] + bias(expected[:, step])).argmax(-1)
        expected[:, step + 1] = model.map_draft_to_target(tokens) if mode == "mapping" else tokens
    actual = AscendSpecDecodeBaseProposer._run_merged_draft(
        proposer,
        num_input_tokens=batch * steps,
        batch_size=batch,
        token_indices_to_sample=torch.arange(batch * steps),
        target_positions=None,
        inputs_embeds=None,
        multi_steps_attn_metadata=None,
        num_tokens=batch * steps,
    )
    torch.testing.assert_close(actual, expected[:, 1:])
    torch.testing.assert_close(proposer._dspark_draft_buffer[:batch], expected)
    assert actual.data_ptr() == proposer._dspark_draft_buffer[:, 1:].data_ptr()
    assert torch.all(proposer._dspark_draft_buffer[batch] == -1)
    assert model.compute_local_draft_logits.call_count == (mode == "local")
    assert model.compute_draft_logits.call_count == (mode != "local")
    proposer.dynamic_spec.update.assert_called_once()
    torch.testing.assert_close(proposer.dynamic_spec.update.call_args.kwargs["draft_token_ids"], expected)
    assert proposer.dynamic_spec.update.call_args.kwargs["num_reqs"] == batch
    if mode == "probabilistic":
        assert proposer._last_draft_probs.shape == (batch, steps, vocab)
    else:
        assert proposer._last_draft_probs is None
