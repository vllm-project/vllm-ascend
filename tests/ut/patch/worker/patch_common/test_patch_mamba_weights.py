# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.patch.worker.patch_mamba_weights import _normalize_mamba_weight_name


def test_normalize_mamba_weight_name_strips_model_prefix():
    assert _normalize_mamba_weight_name("model.backbone.layers.0.mixer.A") == "backbone.layers.0.mixer.A"


def test_normalize_mamba_weight_name_keeps_regular_name():
    assert _normalize_mamba_weight_name("backbone.layers.0.mixer.A") == "backbone.layers.0.mixer.A"


def test_normalize_mamba_weight_name_fixes_embedding():
    assert _normalize_mamba_weight_name("embedding.weight") == "embeddings.weight"


def test_normalize_mamba_weight_name_fixes_backbone_embedding():
    assert _normalize_mamba_weight_name("backbone.embedding.weight") == "backbone.embeddings.weight"


def test_normalize_mamba_weight_name_keeps_embeddings_with_s():
    assert _normalize_mamba_weight_name("embeddings.weight") == "embeddings.weight"
