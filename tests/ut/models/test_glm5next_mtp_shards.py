# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.model_executor.model_loader.weight_utils import filter_safetensors_files_by_weight_name

from vllm_ascend.models.glm5next.mtp import Glm5NextMTP


@pytest.mark.parametrize("prefix", ["model.", "model.language_model."])
@pytest.mark.parametrize("use_rotation", [False, True])
def test_select_mtp_checkpoint_shards(tmp_path, prefix, use_rotation):
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1),
        is_rot_weight_used=use_rotation,
    )
    shard_keys = {
        "target": [f"{prefix}layers.44.enorm.weight", f"{prefix}layers.46.enorm.weight"],
        "embedding": [f"{prefix}embed_tokens.weight"],
        "mixed": [f"{prefix}layers.44.enorm.weight", f"{prefix}layers.45.enorm.weight"],
        "draft_scale": [f"{prefix}layers.45.mlp.experts.0.down_proj.weight_scale"],
        "rotary_buffer": [f"{prefix}layers.45.rotary_emb.inv_freq"],
        "rotation": ["rot.weight"],
    }
    paths = {name: str(tmp_path / f"{name}.safetensors") for name in shard_keys}
    for name, keys in shard_keys.items():
        save_file({key: torch.ones(1) for key in keys}, paths[name])

    expected = [paths["embedding"], paths["mixed"], paths["draft_scale"]]
    if use_rotation:
        expected.append(paths["rotation"])
    is_unused = partial(Glm5NextMTP.is_unused_checkpoint_weight, model)
    assert filter_safetensors_files_by_weight_name(list(paths.values()), is_unused) == expected
