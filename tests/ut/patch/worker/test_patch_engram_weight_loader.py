# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.config import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from vllm_ascend.patch.worker import patch_engram_weight_loader

CODES = "layers.1.engram.embed.weight"
SCALES = "layers.1.engram.embed.scale"
PROJECTION = "layers.1.self_attn.q_weight"
LOCAL_EXPERT = "layers.1.mlp.experts.0.gate_proj.weight"
REMOTE_EXPERT = "layers.1.mlp.experts.1.gate_proj.weight"


def _save_weights(path, *names):
    save_file({name: torch.ones(2) for name in names}, path)


@pytest.mark.parametrize(("load_format", "strategy"), [("auto", None), ("auto", "lazy"), ("safetensors", "lazy")])
def test_mixed_shard_filters_engram_before_get_tensor(tmp_path, monkeypatch, load_format, strategy):
    assert DefaultModelLoader.get_all_weights is patch_engram_weight_loader._get_all_weights_with_node_shared_engram
    _save_weights(tmp_path / "primary.safetensors", CODES, SCALES, PROJECTION, LOCAL_EXPERT, REMOTE_EXPERT)
    secondary = tmp_path / "secondary"
    secondary.mkdir()
    _save_weights(secondary / "draft.safetensors", CODES)
    reads = []

    @contextmanager
    def record_safe_open(*args, **kwargs):
        with safe_open(*args, **kwargs) as reader:

            def get_tensor(name):
                reads.append(name)
                return reader.get_tensor(name)

            yield SimpleNamespace(keys=reader.keys, get_tensor=get_tensor)

    monkeypatch.setattr(patch_engram_weight_loader, "safe_open", record_safe_open)
    config = SimpleNamespace(model=str(tmp_path), revision=None)
    loader = DefaultModelLoader(
        LoadConfig(load_format=load_format, safetensors_load_strategy=strategy, use_tqdm_on_load=False)
    )
    loader.local_expert_ids = {0}
    model = SimpleNamespace(
        requires_lazy_checkpoint_loading=True,
        is_unused_checkpoint_weight={CODES, SCALES}.__contains__,
        secondary_weights=(DefaultModelLoader.Source(str(secondary), None, prefix="draft."),),
    )

    weights = dict(loader.get_all_weights(config, model))

    assert set(reads) == {PROJECTION, LOCAL_EXPERT}
    assert set(weights) == {PROJECTION, LOCAL_EXPERT, f"draft.{CODES}"}


def test_unshared_model_uses_original_loader(tmp_path):
    _save_weights(tmp_path / "primary.safetensors", CODES, SCALES, PROJECTION)
    config = SimpleNamespace(model=str(tmp_path), revision=None)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors", use_tqdm_on_load=False))

    weights = dict(loader.get_all_weights(config, SimpleNamespace()))

    assert set(weights) == {CODES, SCALES, PROJECTION}


@pytest.mark.parametrize(
    "load_options",
    [
        {"safetensors_load_strategy": "eager"},
        {"safetensors_load_strategy": "prefetch"},
        {"model_loader_extra_config": {"enable_multithread_load": True}},
    ],
)
def test_node_shared_engram_rejects_nonlazy_loaders(tmp_path, load_options):
    config = SimpleNamespace(model=str(tmp_path), revision=None)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors", **load_options))
    model = SimpleNamespace(
        requires_lazy_checkpoint_loading=True,
        is_unused_checkpoint_weight={CODES, SCALES}.__contains__,
    )

    with pytest.raises(ValueError, match="default lazy safetensors"):
        list(loader.get_all_weights(config, model))
