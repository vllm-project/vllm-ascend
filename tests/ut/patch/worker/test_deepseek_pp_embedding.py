# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from vllm.distributed import parallel_state
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.models import qwen3_dflash2, qwen3_dspark
from vllm_ascend.ops import vocab_parallel_embedding as ascend_vocab
from vllm_ascend.patch.worker import patch_deepseek_v2


def _make_target(monkeypatch, pp_size, pp_rank, method, *, existing=False, quant_config=None):
    group = SimpleNamespace(world_size=pp_size, is_first_rank=pp_rank == 0, is_last_rank=pp_rank == pp_size - 1)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(eagle_utils, "get_pp_group", lambda: group)
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        vocab_size=65,
        hidden_size=2,
        num_hidden_layers=78,
        num_nextn_predict_layers=0,
        n_routed_experts=0,
        n_shared_experts=0,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=quant_config,
        speculative_config=None if method is None else SimpleNamespace(method=method),
        use_v2_model_runner=True,
    )
    initial = torch.nn.Embedding(65, 2) if group.is_first_rank or existing else PPMissingLayer()

    def original_init(model, *, vllm_config, prefix):
        torch.nn.Module.__init__(model)
        model.config = vllm_config.model_config.hf_config
        model.hidden_size = model.config.hidden_size
        model.embed_tokens = initial
        model.layers = torch.nn.ModuleList()
        model.make_empty_intermediate_tensors = Mock()
        model.use_mha = False
        model.fused_shared_expert_layers = set()
        model.is_fused_shared_expert_enabled = False
        model.num_redundant_experts = 0

    monkeypatch.setattr(patch_deepseek_v2, "_original_deepseek_v2_model_init", original_init)
    monkeypatch.setattr(patch_deepseek_v2, "configure_pp_topk_transport", lambda *args: None)
    monkeypatch.setattr(
        patch_deepseek_v2.pp_transport, "make_empty_intermediate_tensors", lambda model, factory, _: factory
    )
    model = deepseek_v2.DeepseekV2Model.__new__(deepseek_v2.DeepseekV2Model)
    patch_deepseek_v2._patched_deepseek_v2_model_init(model, vllm_config=config, prefix="language_model.model")
    return model, initial


@pytest.mark.parametrize(
    "pp_size,pp_rank,method,existing,create_embedding",
    [
        pytest.param(2, 1, "dflash", False, True, id="create-missing-last-stage-embedding"),
        pytest.param(2, 1, "dflash", True, False, id="preserve-existing-embedding"),
        pytest.param(3, 1, "dflash", False, False, id="skip-middle-stage"),
        pytest.param(2, 1, None, False, False, id="skip-without-speculative-decoding"),
    ],
)
def test_target_embedding_provisioning_scope(monkeypatch, pp_size, pp_rank, method, existing, create_embedding):
    quant_config = object()
    replacement = torch.nn.Embedding(65, 2)
    constructor = Mock(return_value=replacement)
    monkeypatch.setattr(patch_deepseek_v2, "VocabParallelEmbedding", constructor)
    model, initial = _make_target(monkeypatch, pp_size, pp_rank, method, existing=existing, quant_config=quant_config)
    if create_embedding:
        constructor.assert_called_once_with(
            65, 2, quant_config=quant_config, prefix="language_model.model.embed_tokens"
        )
        assert model.embed_tokens is replacement
    else:
        constructor.assert_not_called()
        assert model.embed_tokens is initial


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("rotated", [False, True])
def test_pp_target_embedding_uses_normal_weight_loader_before_dflash_sharing(monkeypatch, tmp_path, tp_rank, rotated):
    # Exercise the real embedding factory, TP shard loader, target load_weights
    # and upstream sharing check. Only distributed setup and unrelated MoE /
    # transformer-layer construction are replaced for this CPU regression.
    monkeypatch.setattr(vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(deepseek_v2, "fused_moe_make_expert_params_mapping", lambda *args, **kwargs: [])
    # Model the TP-only runtime groups without initializing distributed NPUs.
    monkeypatch.setattr(ascend_vocab, "enable_pcp_embedding_lmhead_weight_sharding", lambda: False)
    monkeypatch.setattr(ascend_vocab, "embedding_tp_enable", lambda: False)
    monkeypatch.setattr(ascend_vocab, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(ascend_vocab, "get_tp_group", lambda: SimpleNamespace(rank_in_group=tp_rank, world_size=2))
    target, _ = _make_target(monkeypatch, 2, 1, "dflash")
    assert isinstance(target.embed_tokens, vocab_parallel_embedding.VocabParallelEmbedding)
    checkpoint_weight = torch.arange(130, dtype=torch.float32).reshape(65, 2)
    loaded = target.load_weights([("embed_tokens.weight", checkpoint_weight)])
    assert "embed_tokens.weight" in loaded
    indices = target.embed_tokens.shard_indices
    expected = torch.zeros_like(target.embed_tokens.weight)
    shard = checkpoint_weight[indices.org_vocab_start_index : indices.org_vocab_end_index]
    expected[: len(shard)] = shard
    torch.testing.assert_close(target.embed_tokens.weight, expected)

    draft = qwen3_dflash2.DFlash2Qwen3ForCausalLM.__new__(qwen3_dflash2.DFlash2Qwen3ForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.model = torch.nn.Module()
    draft.model.embed_tokens = torch.nn.Embedding(65, 2)
    draft.model.fc = torch.nn.Linear(4, 2, bias=False)
    draft.has_own_embed_tokens = draft.has_own_lm_head = False
    eagle_utils.maybe_share_target_embed(draft, draft.model, target)
    assert draft.model.embed_tokens is target.embed_tokens
    torch.testing.assert_close(draft.model.embed_tokens.weight, expected)

    target_head = torch.nn.Linear(2, len(expected), bias=False)
    head_weight = checkpoint_weight + 10
    expected_head = torch.zeros_like(target_head.weight)
    expected_head[: len(shard)] = head_weight[indices.org_vocab_start_index : indices.org_vocab_end_index]
    with torch.no_grad():
        target_head.weight.copy_(expected_head)
    draft.lm_head = target_head
    original_fc = draft.model.fc.weight.detach().clone()
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    save_file(
        {"model.embed_tokens.weight": checkpoint_weight, "lm_head.weight": head_weight}, tmp_path / "model.safetensors"
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path), hf_text_config=SimpleNamespace(vocab_size=65, hidden_size=2))
    )
    monkeypatch.setattr(qwen3_dflash2, "get_rotation_path", lambda _: "/rotation" if rotated else None)
    monkeypatch.setattr(qwen3_dspark, "get_rotation_path", lambda _: "/rotation" if rotated else None)
    monkeypatch.setattr(qwen3_dspark, "get_rotation_matrix", lambda _: rotation)
    # Device-specific post-load packing is outside this CPU test.
    process_weights = Mock()
    monkeypatch.setattr(
        vocab_parallel_embedding.UnquantizedEmbeddingMethod, "process_weights_after_loading", process_weights
    )
    draft.post_process(config)
    torch.testing.assert_close(target.embed_tokens.weight, expected)
    torch.testing.assert_close(target_head.weight, expected_head)
    if rotated:
        assert draft.model.embed_tokens is not target.embed_tokens
        assert draft.lm_head is not target_head
        assert draft.has_own_embed_tokens and draft.has_own_lm_head
        torch.testing.assert_close(draft.model.embed_tokens.weight, expected @ rotation.T)
        torch.testing.assert_close(draft.lm_head.weight, expected_head @ rotation.T)
        torch.testing.assert_close(draft.model.fc.weight, (original_fc.reshape(2, 2, 2) @ rotation).reshape(2, 4))
        assert process_weights.call_count == 2
    else:
        assert draft.model.embed_tokens is target.embed_tokens
        assert draft.lm_head is target_head
        torch.testing.assert_close(draft.model.fc.weight, original_fc)
        process_weights.assert_not_called()
