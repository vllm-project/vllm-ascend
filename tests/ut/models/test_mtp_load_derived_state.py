# SPDX-License-Identifier: Apache-2.0
"""Weight ownership must survive a loader that never runs load_weights.

RFork copies tensor bytes, so ``load_weights`` never runs on a receiver. Each
model that decides something while loading has to re-derive it, and the restored
value has to match what loading would have recorded.
"""

import json
from types import SimpleNamespace

import pytest
from torch import nn

from vllm_ascend.models.common.checkpoint import (
    checkpoint_contains_any_weight,
    checkpoint_contains_weight,
)
from vllm_ascend.models.deepseek_mtp import AscendDeepSeekMTP
from vllm_ascend.models.glm5next.mtp import Glm5NextMTP
from vllm_ascend.models.qwen3_dspark import (
    TARGET_EMBED_WEIGHT_NAMES,
    TARGET_LM_HEAD_WEIGHT_NAMES,
    AscendQwen3DSparkForCausalLM,
)

MTP_LAYER_IDX = 45
OWN_HEAD_WEIGHT = f"model.layers.{MTP_LAYER_IDX}.shared_head.head.weight"


def _write_indexed_checkpoint(root, weight_names):
    shard_name = "model-00001-of-00001.safetensors"
    index = {"weight_map": {name: shard_name for name in weight_names}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    # The helper reads names only; an empty shard is never opened when the index resolves.
    (root / shard_name).write_bytes(b"")


def _make_deepseek_mtp():
    model = AscendDeepSeekMTP.__new__(AscendDeepSeekMTP)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = MTP_LAYER_IDX
    shared_head = nn.Module()
    shared_head.head = nn.Linear(4, 4, bias=False)
    mtp_layer = nn.Module()
    mtp_layer.shared_head = shared_head
    model.model.layers = nn.ModuleDict({str(MTP_LAYER_IDX): mtp_layer})
    return model


def _make_glm5next_mtp():
    model = Glm5NextMTP.__new__(Glm5NextMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=MTP_LAYER_IDX)
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = MTP_LAYER_IDX
    return model


@pytest.mark.parametrize("make_model", [_make_deepseek_mtp, _make_glm5next_mtp])
@pytest.mark.parametrize("ships_head", [True, False])
def test_restore_load_derived_state_matches_load_weights(tmp_path, make_model, ships_head):
    """The restored flag must equal what load_weights would have recorded."""
    checkpoint_names = [f"model.layers.{MTP_LAYER_IDX}.enorm.weight"]
    if ships_head:
        checkpoint_names.append(OWN_HEAD_WEIGHT)
    _write_indexed_checkpoint(tmp_path, checkpoint_names)

    restored = make_model()
    restored.restore_load_derived_state(str(tmp_path))

    loaded = make_model()
    loaded._maybe_set_own_lm_head(set(checkpoint_names))

    assert restored.has_own_lm_head is ships_head
    assert restored.has_own_lm_head == loaded.has_own_lm_head


def test_restore_load_derived_state_rebuilds_deepseek_lm_head_alias(tmp_path):
    """DeepSeek exposes the checkpoint head through ``lm_head``; the alias must come back."""
    _write_indexed_checkpoint(tmp_path, [OWN_HEAD_WEIGHT])
    model = _make_deepseek_mtp()

    model.restore_load_derived_state(str(tmp_path))

    assert model.lm_head is model.model.layers[str(MTP_LAYER_IDX)].shared_head.head


def test_restore_load_derived_state_leaves_no_alias_without_a_checkpoint_head(tmp_path):
    _write_indexed_checkpoint(tmp_path, [f"model.layers.{MTP_LAYER_IDX}.enorm.weight"])
    model = _make_deepseek_mtp()

    model.restore_load_derived_state(str(tmp_path))

    assert model.has_own_lm_head is False
    # No alias is installed, so the proposer falls back to sharing the target head.
    assert getattr(model, "lm_head", None) is None


@pytest.mark.parametrize("make_model", [_make_deepseek_mtp, _make_glm5next_mtp])
def test_restore_load_derived_state_raises_when_the_checkpoint_is_unreadable(tmp_path, make_model):
    """An uninspectable checkpoint must raise, not silently report a missing head.

    Reporting False here would make the proposer share the target head for a
    draft that owns one, so the loader needs the failure to reach its fallback.
    """
    model = make_model()

    with pytest.raises(ValueError, match="cannot inspect it"):
        model.restore_load_derived_state(str(tmp_path / "does-not-exist"))


def _make_dspark_draft():
    model = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    nn.Module.__init__(model)
    return model


@pytest.mark.parametrize("ships_embed", [True, False])
@pytest.mark.parametrize("ships_head", [True, False])
def test_dspark_restore_sets_both_ownership_flags(tmp_path, ships_embed, ships_head):
    """align_draft_weights reads these flags to decide whether to rebuild a module.

    It treats a missing attribute as "no own weight", so a receiver that skipped
    load_weights would have its transferred embed_tokens/lm_head overwritten from
    the target checkpoint.
    """
    checkpoint_names = ["model.fc.weight"]
    if ships_embed:
        checkpoint_names.append(TARGET_EMBED_WEIGHT_NAMES[-1])
    if ships_head:
        checkpoint_names.append(TARGET_LM_HEAD_WEIGHT_NAMES[-1])
    _write_indexed_checkpoint(tmp_path, checkpoint_names)

    model = _make_dspark_draft()
    model.restore_load_derived_state(str(tmp_path))

    assert model.has_own_embed_tokens is ships_embed
    assert model.has_own_lm_head is ships_head


def test_dspark_restore_accepts_either_checkpoint_naming(tmp_path):
    """Multimodal checkpoints prefix these weights with ``language_model``."""
    _write_indexed_checkpoint(tmp_path, ["language_model.model.embed_tokens.weight", "language_model.lm_head.weight"])

    model = _make_dspark_draft()
    model.restore_load_derived_state(str(tmp_path))

    assert model.has_own_embed_tokens is True
    assert model.has_own_lm_head is True


def test_dspark_restore_raises_when_the_checkpoint_is_unreadable(tmp_path):
    model = _make_dspark_draft()

    with pytest.raises(ValueError, match="cannot inspect it"):
        model.restore_load_derived_state(str(tmp_path / "does-not-exist"))


def test_checkpoint_contains_any_weight_matches_in_one_pass(tmp_path):
    _write_indexed_checkpoint(tmp_path, ["b.weight"])

    assert checkpoint_contains_any_weight(tmp_path, ("a.weight", "b.weight")) is True
    assert checkpoint_contains_any_weight(tmp_path, ("a.weight", "c.weight")) is False
    with pytest.raises(ValueError, match="must not be empty"):
        checkpoint_contains_any_weight(tmp_path, ())


def test_checkpoint_contains_weight_reads_every_index(tmp_path):
    """Quantized checkpoints ship their own index next to the base one."""
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"other.weight": "a.safetensors"}}), encoding="utf-8"
    )
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {OWN_HEAD_WEIGHT: "b.safetensors"}}), encoding="utf-8"
    )

    assert checkpoint_contains_weight(tmp_path, OWN_HEAD_WEIGHT) is True
    assert checkpoint_contains_weight(tmp_path, "absent.weight") is False


def test_checkpoint_contains_weight_rejects_a_directory_without_safetensors(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"")

    with pytest.raises(ValueError, match="No safetensors checkpoint"):
        checkpoint_contains_weight(tmp_path, OWN_HEAD_WEIGHT)
