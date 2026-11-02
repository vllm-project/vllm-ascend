# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.compilation import decorators
from vllm.config.compilation import DynamicShapesType
from vllm.forward_context import DPMetadata, ForwardContext, get_forward_context, override_forward_context

from vllm_ascend.models.deepseek_v41.decoder_replay_layers import AscendDecoderReplayLayers
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM, DeepseekV41Model


@pytest.mark.parametrize("num_tokens", [384, 1024])
def test_backbone_marks_embedding_token_dimension_dynamic(monkeypatch, num_tokens):
    """Profile and prefill must not specialize the embedding row count."""
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.do_not_compile = False
    model.compiled = False
    model.compilation_config = SimpleNamespace(dynamic_shapes_config=SimpleNamespace(type=DynamicShapesType.BACKED))
    monkeypatch.setattr(decorators.envs, "VLLM_USE_AOT_COMPILE", False)
    monkeypatch.setattr(decorators, "is_forward_context_available", lambda: False)

    class ReadyToCompile(Exception):
        pass

    def stop_before_compilation():
        # The real decorator has marked inputs at this point. Avoid requiring
        # model weights or a device compiler for this input-contract test.
        raise ReadyToCompile

    monkeypatch.setattr(model, "original_code_object", stop_before_compilation)
    ids = torch.zeros(num_tokens, dtype=torch.long)
    positions = torch.arange(num_tokens)
    embeddings = torch.zeros(num_tokens, 8)
    with pytest.raises(ReadyToCompile):
        model(ids, positions, None, inputs_embeds=embeddings)

    for tensor in (ids, positions, embeddings):
        assert tensor._dynamo_dynamic_indices == {0}


@pytest.mark.parametrize("replay_enabled", [False, True])
@pytest.mark.parametrize("captured_engram", [False, True])
def test_replay_uses_runtime_rows_and_preserves_compiled_entry(replay_enabled, captured_engram):
    """Replay must use runtime rows and DP context without changing later compiled calls."""
    seen_rows = []
    seen_dp_tokens = []

    def run_layers(hidden, *unused):
        seen_rows.append(hidden.shape[0])
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        seen_dp_tokens.append(dp_metadata.num_tokens_across_dp_cpu.tolist())
        return (hidden + 1,)

    replay = AscendDecoderReplayLayers(2, run_layers, [])
    ids = torch.arange(4)
    hidden = ids.float().view(-1, 1)

    def run_python(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        return replay(inputs_embeds, None, positions)[0]

    def run_compiled(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        # Model the cached full-batch program, which no longer reads replay.rows.
        return run_layers(inputs_embeds)[0]

    backbone = MagicMock(side_effect=run_compiled)
    backbone.forward.side_effect = run_python
    backbone.decoder_replay_layers = replay if replay_enabled else None
    backbone.captured_engram_inputs.return_value = nullcontext({})
    model = AscendDeepseekV41LLMForCausalLM.__new__(AscendDeepseekV41LLMForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = backbone
    kwargs = {"engram_query_start_loc": torch.tensor([0, 4])} if captured_engram else {}

    def forward():
        return model.forward(ids, ids, inputs_embeds=hidden, **kwargs)

    outer_context = ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        dp_metadata=DPMetadata(torch.tensor([4, 4], dtype=torch.int32)),
    )
    with override_forward_context(outer_context):
        assert forward()[:, 0].tolist() == [1, 2, 3, 4]
        for indices, expected, dp_tokens in (
            (None, [1, 2, 3, 4], [2, 4]),  # Only the peer DP replica trims.
            ([2, 3], [0, 0, 3, 4], [4, 2]),
            ([], [0, 0, 0, 0], [4, 0]),
        ):
            replay.rows = None if indices is None else torch.tensor(indices, dtype=torch.int64)
            replay.num_actual_rows = dp_tokens[1]
            decoder_context = replace(outer_context, dp_metadata=DPMetadata(torch.tensor(dp_tokens, dtype=torch.int32)))
            replay.context_factory = partial(override_forward_context, decoder_context)
            output = forward()
            assert output[:, 0].tolist() == (expected if replay_enabled else [1, 2, 3, 4])
            assert seen_dp_tokens[-1] == (dp_tokens if replay_enabled else [4, 4])
            assert get_forward_context() is outer_context

        replay.rows = None
        replay.context_factory = None
        assert forward()[:, 0].tolist() == [1, 2, 3, 4]
        assert seen_dp_tokens[-1] == [4, 4]
    assert seen_rows == ([4, 4, 2, 0, 4] if replay_enabled else [4, 4, 4, 4, 4])
    assert backbone.call_count == (2 if replay_enabled else 5)
    assert backbone.forward.call_count == (3 if replay_enabled else 0)
