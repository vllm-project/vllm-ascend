#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/tests/v1/worker/test_gpu_worker.py
#
"""Spec-decode frontend: aux-layer resolution and xlite KV registration."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("xlite")

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    HiddenStateCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (  # noqa: E402
    get_eagle3_aux_layers_from_config,
)

from vllm_ascend.utils import is_hidden_state_cache_spec  # noqa: E402
from vllm_ascend.xlite.xlite import StandardXliteModel  # noqa: E402
from vllm_ascend.xlite.xlite_model_runner import (  # noqa: E402
    XliteModelRunnerMixin,
    _hidden_state_cache_layers,
)

SPEC_LAYERS = [3, 17, 29]  # aux layers from the speculative config
DEFAULT_LAYERS = [2, 24, 45]  # aux layers from the loaded model


def _adapter(spec_config, model) -> SimpleNamespace:
    """A `StandardXliteModel` stand-in carrying only what `_resolve_aux_layer_ids` reads."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(speculative_config=spec_config),
        npu_runner=SimpleNamespace(use_aux_hidden_state_outputs=True, model=model),
    )


def _spec_config(layers: list[int] | None) -> SimpleNamespace:
    hf_config = SimpleNamespace()
    if layers is not None:
        hf_config.eagle_aux_hidden_state_layer_ids = layers
    return SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf_config))


def _model_with_default() -> SimpleNamespace:
    return SimpleNamespace(get_eagle3_default_aux_hidden_state_layers=lambda: list(DEFAULT_LAYERS))


def test_extract_hidden_states_aux_layers_match_native_resolution():
    """The adapter's aux layers must equal the native path's resolution, in order."""
    # Config-provided layers win, exactly as the native pre-swap resolves them.
    spec = _spec_config(SPEC_LAYERS)
    adapter = _adapter(spec, _model_with_default())
    resolved = StandardXliteModel._resolve_aux_layer_ids(adapter)  # type: ignore[arg-type]
    assert resolved == SPEC_LAYERS
    assert resolved == list(get_eagle3_aux_layers_from_config(spec))

    # Without config layers, the loaded model's default applies with its order intact.
    spec = _spec_config(None)
    adapter = _adapter(spec, _model_with_default())
    assert StandardXliteModel._resolve_aux_layer_ids(adapter) == DEFAULT_LAYERS  # type: ignore[arg-type]

    # Non-ascending/duplicate ids normalize like the native set_aux_hidden_state_layers.
    spec = _spec_config([29, 3, 17, 17])
    adapter = _adapter(spec, _model_with_default())
    assert StandardXliteModel._resolve_aux_layer_ids(adapter) == sorted(set(SPEC_LAYERS))  # type: ignore[arg-type]


def test_missing_model_default_aux_layers_raises():
    spec = _spec_config(None)
    adapter = _adapter(spec, model=SimpleNamespace())
    with pytest.raises(RuntimeError, match="aux hidden-state layers"):
        StandardXliteModel._resolve_aux_layer_ids(adapter)  # type: ignore[arg-type]


def test_hidden_state_layers_excluded_from_xlite_registration():
    """Hidden-state-only caches are dropped from xlite registration by spec, not shape."""
    hidden_spec = HiddenStateCacheSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.float16)
    assert is_hidden_state_cache_spec(hidden_spec)

    runner = XliteModelRunnerMixin.__new__(XliteModelRunnerMixin)
    k0, v0, k1, v1, aux = (torch.zeros(2, 8, 128, dtype=torch.float16) for _ in range(5))
    runner.kv_caches_by_layer = {"model.layers.0": [k0, v0], "model.layers.1": [k1, v1], "aux": aux}
    runner.kv_caches = [k0, v0, k1, v1, aux]
    runner.device = torch.device("cpu")

    flat = runner._attention_kv_caches(frozenset({"aux"}))
    assert [t is c for t, c in zip(flat, (k0, v0, k1, v1))] == [True] * 4
    assert all(t is not aux for t in flat)


def test_hidden_state_layers_found_through_uniform_type_group():
    """A CacheOnly layer merged into a UniformTypeKVCacheSpecs group is still found.

    The ehs cache registers one hidden-state layer (num_kv_heads=3, head_size=hidden) that
    merges with attention layers into a uniform-type group; the group's own spec is the
    wrapper, so the hidden layer must be resolved through the per-layer specs.
    """
    attn_spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.float16)
    hidden_spec = HiddenStateCacheSpec(block_size=128, num_kv_heads=3, head_size=2048, dtype=torch.float16)
    layer_specs = {"model.layers.0": attn_spec, "aux": hidden_spec}
    group = SimpleNamespace(
        layer_names=list(layer_specs),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=128, kv_cache_specs=layer_specs),
    )
    assert _hidden_state_cache_layers(SimpleNamespace(kv_cache_groups=[group])) == frozenset({"aux"})
