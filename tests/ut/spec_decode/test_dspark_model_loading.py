# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.vllm import set_current_vllm_config
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

import vllm_ascend.ops.vocab_parallel_embedding as vocab_module
from vllm_ascend.models import quarot_utils, qwen3_dflash2
from vllm_ascend.models.kimi_k3.kimi_k3_dspark import AscendK3DSparkForCausalLM
from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as shared


@pytest.mark.parametrize("fail", [False, True])
def test_post_process_receives_target_config_after_loading(monkeypatch, fail):
    events: list[tuple[str, object]] = []
    config = SimpleNamespace(
        quant_config=object(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    spec = shared.AscendDSparkSpeculator.__new__(shared.AscendDSparkSpeculator)
    spec.vllm_config = config
    target = object()
    draft = SimpleNamespace(
        post_process=lambda received: events.append(("post_process", received)),
        configure_target_aux_hidden_capture=lambda received: events.append(("capture", received)),
    )

    def load(self, received, names):
        assert received is target
        assert self.vllm_config is config
        events.append(("load", config))
        if fail:
            raise ValueError("load failed")
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", load)
    with pytest.raises(ValueError, match="load failed") if fail else nullcontext():
        assert spec.load_draft_model(target, set()) is draft
    expected: list[tuple[str, object]] = [("load", config)]
    if not fail:
        expected.extend([("post_process", config), ("capture", target)])
    assert events == expected


@pytest.mark.parametrize(
    "model_cls,projection_name,num_aux_layers",
    [
        (AscendQwen3DSparkForCausalLM, "fc", 2),
        (AscendK3DSparkForCausalLM, "context_proj", 2),
        (DFlash2Qwen3ForCausalLM, "fc", 6),
    ],
)
@pytest.mark.parametrize("own_embed,own_head", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("lmhead_tp", [False, True])
def test_post_process_aligns_weights_without_modifying_target(
    monkeypatch, model_cls, projection_name, num_aux_layers, own_embed, own_head, rotated, pp_size, lmhead_tp
):
    monkeypatch.setattr(quarot_utils, "lmhead_tp_enable", lambda: lmhead_tp)
    target_embed = torch.nn.Embedding(4, 2, dtype=torch.float64)
    target_head = torch.nn.Linear(2, 4, bias=False, dtype=torch.float64)
    target_embed_before = target_embed.weight.detach().clone()
    target_head_before = target_head.weight.detach().clone()
    draft = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    projection = torch.nn.Linear(2 * num_aux_layers, 2, bias=False, dtype=torch.float64)
    # Distinct, exactly representable weights keep this basis check independent
    # of the helper's intentional float32 conversion and random cancellation.
    with torch.no_grad():
        projection.weight.copy_(torch.arange(projection.weight.numel()).view_as(projection.weight) / 8)
    setattr(draft.model, projection_name, projection)
    draft.model.embed_tokens = torch.nn.Embedding(4, 2, dtype=torch.float64) if own_embed else target_embed
    draft.lm_head = torch.nn.Linear(2, 4, bias=False, dtype=torch.float64) if own_head else target_head
    draft.has_own_embed_tokens, draft.has_own_lm_head = own_embed, own_head
    original_embed, original_head = draft.model.embed_tokens, draft.lm_head
    original_embed_weight = original_embed.weight.detach().clone()
    original_head_weight = original_head.weight.detach().clone()
    original_parameter = projection.weight
    original_projection = projection.weight.detach().clone()
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="/target", hf_text_config=SimpleNamespace(vocab_size=4, hidden_size=2))
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(quarot_utils, "get_rotation_path", lambda received: "/rotation" if rotated else None)
    monkeypatch.setattr(qwen3_dflash2, "get_rotation_path", lambda received: "/rotation" if rotated else None)
    monkeypatch.setattr(quarot_utils, "get_pp_group", lambda: SimpleNamespace(world_size=pp_size))
    loader = MagicMock(return_value=rotation)
    monkeypatch.setattr(quarot_utils, "get_rotation_matrix", loader)
    created = []

    def make_layer(vocab, hidden, params_dtype, *, prefix):
        layer = torch.nn.Embedding(vocab, hidden, dtype=params_dtype)
        layer.prefix = prefix
        layer.quant_method = SimpleNamespace(process_weights_after_loading=MagicMock())
        created.append(layer)
        return layer

    def load_layer(layer, path, names, matrix, label):
        source = target_embed_before if "embed_tokens" in label else target_head_before
        layer.weight.copy_(source @ matrix.T if matrix is not None else source)

    monkeypatch.setattr(quarot_utils, "VocabParallelEmbedding", make_layer)
    monkeypatch.setattr(quarot_utils, "ParallelLMHead", make_layer)
    monkeypatch.setattr(quarot_utils, "load_quarot_target_layer", load_layer)
    draft.post_process(config)
    assert projection.weight is original_parameter

    torch.testing.assert_close(target_embed.weight, target_embed_before)
    torch.testing.assert_close(target_head.weight, target_head_before)
    if rotated:
        loader.assert_called_once_with("/rotation")
        inputs = torch.arange(1, 2 * num_aux_layers + 1, dtype=torch.float64).view(1, -1)
        rotated_inputs = (inputs.view(1, num_aux_layers, 2) @ rotation).view_as(inputs)
        torch.testing.assert_close(rotated_inputs @ projection.weight.T, inputs @ original_projection.T)
    else:
        loader.assert_not_called()
        torch.testing.assert_close(projection.weight, original_projection)
    # DFlash2 only invokes weight alignment for rotated targets.
    reload_embed = rotated or (pp_size > 1 and model_cls is not DFlash2Qwen3ForCausalLM)
    assert len(created) == int(not own_embed and reload_embed) + int(not own_head and rotated)
    for layer in created:
        layer.quant_method.process_weights_after_loading.assert_called_once_with(layer)
    for actual, original, before, target_weight, owns, is_embed in (
        (draft.model.embed_tokens, original_embed, original_embed_weight, target_embed_before, own_embed, True),
        (draft.lm_head, original_head, original_head_weight, target_head_before, own_head, False),
    ):
        if not owns and (reload_embed if is_embed else rotated):
            assert actual is not original
            assert actual.weight.data_ptr() != original.weight.data_ptr()
            expected_prefix = ""
            if not is_embed and lmhead_tp:
                expected_prefix = "lm_head"
            assert actual.prefix == expected_prefix
            torch.testing.assert_close(actual.weight, target_weight @ rotation.T if rotated else target_weight)
        else:
            assert actual is original
            torch.testing.assert_close(actual.weight, before)


@pytest.mark.parametrize("rotated", [False, True])
def test_dflash2_without_aux_hidden_states_requires_unrotated_target(monkeypatch, rotated):
    draft = DFlash2Qwen3ForCausalLM.__new__(DFlash2Qwen3ForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.use_aux_hidden_state = False
    config = SimpleNamespace()
    monkeypatch.setattr(qwen3_dflash2, "get_rotation_path", lambda received: "/rotation" if rotated else None)
    align = MagicMock()
    monkeypatch.setattr(qwen3_dflash2, "align_draft_weights", align)

    expected_error = "DFlash2 with a QuaRot target requires auxiliary hidden states."
    with pytest.raises(ValueError, match=expected_error) if rotated else nullcontext():
        draft.post_process(config)
    align.assert_not_called()


@pytest.mark.parametrize("rank", range(8))
def test_dspark_dense_logits_keep_dp_rows_with_shared_lmhead_exchange(monkeypatch, rank):
    """Exercise the DSpark model and worker wrapper through real logits processing."""
    world_size, capacity, hidden_size, shard_size = 8, 6, 4, 5
    counts = [0, 1, 6, 2, 4, 0, 3, 5]
    generator = torch.Generator().manual_seed(43)
    hidden = torch.randn(world_size, capacity, hidden_size, generator=generator)
    for dp, count in enumerate(counts):
        hidden[dp, count:] = 0
    weight = torch.randn(world_size * shard_size, hidden_size, generator=generator)
    all_hidden = hidden.flatten(0, 1)
    full_logits = all_hidden @ weight.T
    calls = []

    def gather(rows, dim):
        assert dim == 0
        torch.testing.assert_close(rows, hidden[rank])
        calls.append("gather")
        return all_hidden

    group = SimpleNamespace(world_size=world_size, device_group=object(), all_gather=gather)

    def all_to_all(output, input_, *, group):
        local_logits = full_logits[:, rank * shard_size : (rank + 1) * shard_size]
        torch.testing.assert_close(input_, local_logits.reshape(world_size, capacity, shard_size))
        for source in range(world_size):
            output[source].copy_(
                full_logits[rank * capacity : (rank + 1) * capacity, source * shard_size : (source + 1) * shard_size]
            )
        calls.append("exchange")

    monkeypatch.setattr(vocab_module, "get_lmhead_tp_group", lambda: group)
    monkeypatch.setattr(vocab_module.dist, "all_to_all_single", all_to_all)
    monkeypatch.setattr(shared, "lmhead_tp_enable", lambda: True)
    config = MagicMock()
    config.compilation_config.custom_ops = ["all"]
    config.model_config = None
    # The final row is padding and must not reach DSpark's sampler.
    vocab_size = world_size * shard_size - 1
    with set_current_vllm_config(config):
        processor = vocab_module.AscendLogitsProcessor(vocab_size=vocab_size, scale=1.25, soft_cap=2.0)
    draft = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.lm_head = torch.nn.Linear(hidden_size, shard_size, bias=False)
    draft.lm_head.weight = torch.nn.Parameter(weight[rank * shard_size : (rank + 1) * shard_size])
    draft.lm_head.quant_method = UnquantizedEmbeddingMethod()
    draft.lm_head.parallel_mode = vocab_module.VocabParallelMode.FINE_GRAINED
    draft.logits_processor = processor
    spec = shared.AscendDSparkSpeculator.__new__(shared.AscendDSparkSpeculator)
    spec.max_num_reqs, spec.num_speculative_steps = 2, 3
    spec._lmhead_tp_wrap_draft_logits(draft)

    logits = draft.compute_draft_logits(hidden[rank, : counts[rank]])
    expected = full_logits[rank * capacity : rank * capacity + counts[rank], :vocab_size]
    torch.testing.assert_close(logits, torch.tanh(expected / 2.0) * 2.0 * 1.25)
    assert logits.shape == (counts[rank], vocab_size)
    assert calls == ["gather", "exchange"]
    # Local-logit callers must still bypass all communication. Exercise the
    # hook directly: older upstream DSpark models do not expose a local method.
    local = processor._get_logits(hidden[rank, : counts[rank]], draft.lm_head, skip_gather=True)
    expected_local = hidden[rank, : counts[rank]] @ draft.lm_head.weight.T
    torch.testing.assert_close(local, expected_local)
    assert calls == ["gather", "exchange"]
