# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Kimi K3 target provisions the shared draft embedding on the last PP stage.

RFC #21537 PR 1: a DSpark draft without a checkpoint-owned embedding shares
the target input embedding. Under PP the draft runs on the last stage while
the target embedding normally lives on the first, so the target must create
the missing vocab shard there and let the normal target checkpoint loader
fill it. The Kimi K3 MLA DSpark draft ships a frozen checkpoint-owned
embedding and must not trigger a second target allocation.
"""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.distributed import parallel_state
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

from vllm_ascend.models.kimi_k3 import kimi_k3
from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM
from vllm_ascend.ops import vocab_parallel_embedding as ascend_vocab
from vllm_ascend.patch.worker import patch_dspark

VOCAB_SIZE = 16
HIDDEN_SIZE = 8


def _vllm_config(pp_size, method, draft_architectures=None, draft_vocab_size=None, uses_target_hidden_states=True):
    hf_config = SimpleNamespace(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=4,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        attn_res_block_size=None,
    )
    speculative_config = None
    if method is not None:
        speculative_config = SimpleNamespace(
            method=method,
            uses_target_hidden_states=uses_target_hidden_states,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3", architectures=draft_architectures or []),
                get_vocab_size=lambda: draft_vocab_size or VOCAB_SIZE,
            ),
        )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config, get_vocab_size=lambda: VOCAB_SIZE),
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        use_v2_model_runner=True,
        quant_config=None,
    )


def _make_target(
    monkeypatch,
    pp_size,
    pp_rank,
    method,
    draft_architectures=None,
    *,
    real_embedding=False,
    draft_vocab_size=None,
    uses_target_hidden_states=True,
    rotation_path=None,
):
    group = SimpleNamespace(
        world_size=pp_size,
        is_first_rank=pp_rank == 0,
        is_last_rank=pp_rank == pp_size - 1,
    )
    # The constructor, the provisioning predicate, the share check, and the
    # PP-safe wrapper each read get_pp_group/get_rotation_path from their own
    # module namespace.
    monkeypatch.setattr(kimi_k3, "get_pp_group", lambda: group)
    monkeypatch.setattr(kimi_k3, "get_rotation_path", lambda _: rotation_path)
    monkeypatch.setattr(eagle_utils, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(patch_dspark, "get_pp_group", lambda: group)
    monkeypatch.setattr(kimi_k3, "get_tensor_model_parallel_world_size", lambda: 1)
    # Decoder-layer and EPLB construction are irrelevant to embedding
    # provisioning; keep a trivial single layer so the last-stage branch runs.
    monkeypatch.setattr(
        kimi_k3,
        "make_layers",
        lambda num_layers, get_layer, prefix: (0, min(1, num_layers), nn.ModuleList()),
    )
    # The last-stage final norm is an Ascend CustomOp whose forward dispatch
    # probes the process-wide vLLM config at construction time; unit tests
    # never install one, and the norm is irrelevant to embedding provisioning
    # (same treatment as the namespace fakes in test_kimi_k3_pp_sp.py).
    monkeypatch.setattr(kimi_k3, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", lambda *args, **kwargs: nn.Identity())
    constructor = Mock()
    if not real_embedding:
        monkeypatch.setattr(kimi_k3, "VocabParallelEmbedding", constructor)
    model = kimi_k3.AscendKimiLinearModel(
        vllm_config=_vllm_config(pp_size, method, draft_architectures, draft_vocab_size, uses_target_hidden_states),
        prefix="model",
    )
    return model, constructor


class _GqaDraftStub(nn.Module):
    """Same has_own_embed_tokens contract as the real GQA draft class: a
    class-level declaration set to False, with the instance flag flipped only
    when the checkpoint owns the embedding. ``model`` mirrors the real draft's
    inner structure (the loader passes ``draft.model`` as ``draft_inner``);
    ``post_process`` is annotation-only so ``hasattr`` stays False until a
    test opts into the fallback-drafts shape."""

    has_own_embed_tokens = False
    has_own_lm_head = False
    model: Any
    post_process: Callable[[object], None]

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(embed_tokens=None, fc=None)


def _make_draft(own_embed: bool = False):
    draft = _GqaDraftStub()
    if own_embed:
        draft.has_own_embed_tokens = True
    draft_inner = nn.Module()
    draft_inner.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    return draft, draft_inner


@pytest.mark.parametrize(
    "pp_size,pp_rank,method,draft_architectures,create_embedding",
    [
        pytest.param(2, 1, "dspark", ["Qwen3DSparkModel"], True, id="gqa-draft-needs-shared-embed"),
        pytest.param(2, 1, "dspark", ["DSparkDraftModel"], True, id="deepseek-v4-draft-needs-shared-embed"),
        pytest.param(2, 1, "dspark", ["K3DSparkModel"], False, id="mla-draft-owns-checkpoint-embed"),
        pytest.param(3, 1, "dspark", ["Qwen3DSparkModel"], False, id="skip-middle-stage"),
        pytest.param(2, 1, "dspark", [], True, id="provision-unknown-dspark-draft"),
        pytest.param(1, 0, "dspark", ["Qwen3DSparkModel"], True, id="single-stage-first-rank-embed"),
        pytest.param(2, 1, None, None, False, id="skip-without-speculative-decoding"),
        pytest.param(2, 1, "mtp", ["KimiK3MTPModel"], False, id="skip-mtp-draft"),
    ],
)
def test_target_embedding_provisioning_scope(
    monkeypatch, pp_size, pp_rank, method, draft_architectures, create_embedding
):
    model, constructor = _make_target(monkeypatch, pp_size, pp_rank, method, draft_architectures, real_embedding=False)
    if create_embedding:
        constructor.assert_called_once_with(VOCAB_SIZE, HIDDEN_SIZE, prefix="model.embed_tokens")
    else:
        constructor.assert_not_called()
        assert isinstance(model.embed_tokens, PPMissingLayer)


def _expected_shard(embed_tokens, checkpoint_weight):
    indices = embed_tokens.shard_indices
    expected = torch.zeros_like(embed_tokens.weight)
    shard = checkpoint_weight[indices.org_vocab_start_index : indices.org_vocab_end_index]
    expected[: len(shard)] = shard
    return expected


def _stub_vocab_parallel_groups(monkeypatch, tp_rank):
    # Model the TP-only runtime groups without initializing distributed NPUs,
    # so the real VocabParallelEmbedding factory builds on CPU.
    monkeypatch.setattr(vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(ascend_vocab, "enable_pcp_embedding_lmhead_weight_sharding", lambda: False)
    monkeypatch.setattr(ascend_vocab, "embedding_tp_enable", lambda: False)
    monkeypatch.setattr(ascend_vocab, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(ascend_vocab, "get_tp_group", lambda: SimpleNamespace(rank_in_group=tp_rank, world_size=2))


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_last_stage_embedding_uses_normal_loader_and_draft_shares_it(monkeypatch, tp_rank):
    # Exercise the real embedding factory, the TP shard loader, and the
    # upstream sharing check. Only distributed setup and decoder-layer
    # construction are replaced for this CPU regression.
    _stub_vocab_parallel_groups(monkeypatch, tp_rank)

    target, _ = _make_target(monkeypatch, 2, 1, "dspark", ["Qwen3DSparkModel"], real_embedding=True)
    assert isinstance(target.embed_tokens, vocab_parallel_embedding.VocabParallelEmbedding)

    checkpoint_weight = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(VOCAB_SIZE, HIDDEN_SIZE)
    # The normal target loader dispatches to the embedding's own weight_loader
    # with the full checkpoint tensor; each TP rank keeps its vocab shard.
    target.embed_tokens.weight_loader(target.embed_tokens.weight, checkpoint_weight)
    expected = _expected_shard(target.embed_tokens, checkpoint_weight)
    torch.testing.assert_close(target.embed_tokens.weight, expected)

    draft, draft_inner = _make_draft()
    # The class-level declaration passes the PP>1 sharing gate while the
    # share has not happened yet, so the draft post_process would still
    # reload its own copy (the fallback this PR wires around).
    assert draft.has_own_embed_tokens is False
    patch_dspark._pp_safe_maybe_share_target_embed(draft, draft_inner, target)
    assert draft_inner.embed_tokens is target.embed_tokens
    # The wrapper records the share on the draft so align_draft_weights
    # keeps the shared shard instead of reloading a private copy over it.
    assert draft.has_own_embed_tokens is True
    torch.testing.assert_close(draft_inner.embed_tokens.weight, expected)


def test_qwen3_gqa_draft_declares_sharing_gate_attribute():
    # SupportsEagleBase already provides this attribute through the Qwen3
    # MRO, so this declaration is a convention alignment that also guards
    # against upstream MRO changes; it must not be dropped silently.
    assert AscendQwen3DSparkForCausalLM.has_own_embed_tokens is False


def test_skip_provisioning_when_draft_vocab_exceeds_target(monkeypatch):
    # load_dspark_model only runs the share check when the draft vocab fits
    # the target's; provisioning there would leave the shard unconsumed.
    model, constructor = _make_target(
        monkeypatch, 2, 1, "dspark", ["Qwen3DSparkModel"], real_embedding=False, draft_vocab_size=VOCAB_SIZE * 2
    )
    constructor.assert_not_called()
    assert isinstance(model.embed_tokens, PPMissingLayer)


def test_skip_provisioning_when_draft_ignores_target_hidden_states(monkeypatch):
    # A dspark draft that consumes its own hidden states never aliases the
    # target embedding, so the last stage keeps the missing-layer marker.
    model, constructor = _make_target(
        monkeypatch, 2, 1, "dspark", ["Qwen3DSparkModel"], real_embedding=False, uses_target_hidden_states=False
    )
    constructor.assert_not_called()
    assert isinstance(model.embed_tokens, PPMissingLayer)


def test_skip_provisioning_for_rotated_target(monkeypatch):
    # QuaRot drafts rebuild a rotated private copy in post_process, so a
    # provisioned shard would sit unconsumed; keep the main-only behavior.
    model, constructor = _make_target(
        monkeypatch, 2, 1, "dspark", ["Qwen3DSparkModel"], real_embedding=False, rotation_path="/rotation"
    )
    constructor.assert_not_called()
    assert isinstance(model.embed_tokens, PPMissingLayer)


def test_mla_draft_keeps_own_embedding_without_target_provisioning(monkeypatch):
    # The Kimi K3 MLA draft allocates a frozen checkpoint-owned embedding on
    # every PP stage past the first; the target must not allocate a second
    # copy, and the sharing check must leave the draft's own module alone.
    target, constructor = _make_target(monkeypatch, 2, 1, "dspark", ["K3DSparkModel"], real_embedding=False)
    constructor.assert_not_called()
    assert isinstance(target.embed_tokens, PPMissingLayer)

    draft, draft_inner = _make_draft(own_embed=True)
    draft.post_process = lambda vllm_config: None  # mirror the real MLA draft's fallback-drafts shape
    own_weight = draft_inner.embed_tokens.weight.detach().clone()
    patch_dspark._pp_safe_maybe_share_target_embed(draft, draft_inner, target)
    assert draft_inner.embed_tokens is not target.embed_tokens
    torch.testing.assert_close(draft_inner.embed_tokens.weight, own_weight)


def test_missing_shared_embedding_raises_without_provisioning(monkeypatch):
    # Without a provisioning trigger the last stage keeps PPMissingLayer; a
    # draft that needs the shared embedding must fail the upstream check
    # instead of silently skipping it.
    target, _ = _make_target(monkeypatch, 2, 1, None, real_embedding=False)
    assert isinstance(target.embed_tokens, PPMissingLayer)

    draft, draft_inner = _make_draft()
    with pytest.raises(RuntimeError, match="needs the target input embedding"):
        eagle_utils.maybe_share_target_embed(draft, draft_inner, target)


def test_pp_safe_wrapper_defers_to_fallback_without_provisioned_shard(monkeypatch):
    # No provisioning: the last stage keeps PPMissingLayer and the wrapper
    # must keep deferring to the draft post_process fallback instead of
    # letting the upstream check raise (fallback removal is follow-up PR 2).
    target, _ = _make_target(monkeypatch, 2, 1, None, real_embedding=False)
    assert isinstance(target.embed_tokens, PPMissingLayer)

    draft, draft_inner = _make_draft()
    draft.post_process = lambda vllm_config: None
    own_embed = draft_inner.embed_tokens
    patch_dspark._pp_safe_maybe_share_target_embed(draft, draft_inner, target)
    assert draft_inner.embed_tokens is own_embed
    assert draft.has_own_embed_tokens is False


@pytest.mark.parametrize("rotated", [False, True])
def test_post_process_keeps_shared_shard_unless_rotated(monkeypatch, rotated):
    # End to end: provisioned shard -> wrapper share -> draft post_process.
    # Unrotated keeps the shared module; QuaRot rebuilds the rotated copy
    # over it (align_draft_weights owns the final embedding basis).
    from vllm_ascend.models import qwen3_dspark

    _stub_vocab_parallel_groups(monkeypatch, 0)
    target, _ = _make_target(monkeypatch, 2, 1, "dspark", ["Qwen3DSparkModel"], real_embedding=True)
    checkpoint_weight = torch.arange(VOCAB_SIZE * HIDDEN_SIZE, dtype=torch.float32).reshape(VOCAB_SIZE, HIDDEN_SIZE)
    # The provisioned module registers a real parameter filled by the normal
    # target loader path (weight_loader stands in for checkpoint loading
    # here). The unit test instantiates the text model directly, so the
    # registered name lacks the "model." prefix that the ForCausalLM wrapper
    # adds in production (whose checkpoint key is model.embed_tokens.weight).
    target.embed_tokens.weight_loader(target.embed_tokens.weight, checkpoint_weight)
    params = dict(target.named_parameters())
    assert params["embed_tokens.weight"] is target.embed_tokens.weight

    draft, draft_inner = _make_draft()
    draft.model = draft_inner  # the real draft loader passes draft.model as draft_inner
    draft_inner.fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    draft.has_own_lm_head = True  # isolate the embedding; lm_head is align's own contract
    # The load wrapper builds the wrapper with mark_shared=False for QuaRot.
    share = patch_dspark._make_pp_safe_maybe_share_target_embed(mark_shared=not rotated)
    share(draft, draft_inner, target)

    rotation = torch.eye(HIDDEN_SIZE).roll(1, dims=1)
    monkeypatch.setattr(qwen3_dspark, "get_rotation_path", lambda _: "/rotation" if rotated else None)
    monkeypatch.setattr(qwen3_dspark, "get_rotation_matrix", lambda _: rotation)
    monkeypatch.setattr(qwen3_dspark, "get_pp_group", lambda: SimpleNamespace(world_size=2))

    def make_layer(vocab, hidden, params_dtype):
        layer = nn.Embedding(vocab, hidden, dtype=params_dtype)
        layer.quant_method = SimpleNamespace(process_weights_after_loading=lambda layer: None)
        return layer

    def load_layer(layer, path, names, matrix, label):
        layer.weight.copy_(checkpoint_weight @ matrix.T if matrix is not None else checkpoint_weight)

    monkeypatch.setattr(qwen3_dspark, "VocabParallelEmbedding", make_layer)
    monkeypatch.setattr(qwen3_dspark, "ParallelLMHead", make_layer)
    monkeypatch.setattr(qwen3_dspark, "load_quarot_target_layer", load_layer)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE), model="target"
        )
    )
    qwen3_dspark.align_draft_weights(draft, draft.model.fc, cfg)
    if rotated:
        assert draft.model.embed_tokens is not target.embed_tokens
        torch.testing.assert_close(draft.model.embed_tokens.weight, checkpoint_weight @ rotation.T)
    else:
        assert draft.model.embed_tokens is target.embed_tokens


@pytest.mark.parametrize("rotated", [False, True])
def test_load_wrapper_marks_shared_only_when_unrotated(monkeypatch, rotated):
    # The load wrapper installs the share check with mark_shared derived from
    # the target rotation state; this wiring is what keeps the QuaRot
    # post_process rebuild from being skipped.
    installed = []

    def fake_load(target_model, vllm_config):
        installed.append(patch_dspark.eagle_utils.maybe_share_target_embed)
        return "draft-model"

    monkeypatch.setattr(patch_dspark, "_original_load_dspark_model", fake_load)
    monkeypatch.setattr(patch_dspark, "resolve_spec_pp_support", lambda _: None)
    monkeypatch.setattr(patch_dspark, "get_rotation_path", lambda _: "/rotation" if rotated else None)
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(model="target")),
        model_config=SimpleNamespace(model="target"),
        quant_config=None,
    )
    assert patch_dspark._load_dspark_model_with_target_quant(object(), cfg) == "draft-model"
    (share_check,) = installed
    assert share_check is not patch_dspark._original_maybe_share_target_embed

    group = SimpleNamespace(world_size=2, is_first_rank=False, is_last_rank=True)
    monkeypatch.setattr(patch_dspark, "get_pp_group", lambda: group)
    monkeypatch.setattr(eagle_utils, "get_pp_group", lambda: group)
    draft, draft_inner = _make_draft()
    target_like = SimpleNamespace(embed_tokens=nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE))
    share_check(draft, draft_inner, target_like)
    assert draft_inner.embed_tokens is target_like.embed_tokens
    assert draft.has_own_embed_tokens is (not rotated)
    # The wrapper is restored for the next load.
    assert patch_dspark.eagle_utils.maybe_share_target_embed is patch_dspark._original_maybe_share_target_embed
