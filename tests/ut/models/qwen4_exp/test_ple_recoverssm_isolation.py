# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadataBuilder

from vllm_ascend.models.qwen4_exp.ple import AscendPLEBackend, AscendPLELayer, AscendPLEMetadataBuilder


@pytest.mark.parametrize(("recover", "kda"), [(False, False), (True, False), (True, True)])
def test_ple_builder_isolates_recovery_flags_without_changing_target_or_cache(monkeypatch, recover, kda):
    original_cache = NS(use_replayssm=recover, use_kda_recoverssm=kda, block_size=1024)
    original = NS(cache_config=original_cache, compilation_config=object())
    spec, names, device = object(), ["model.layers.2.ple"], torch.device("cpu")
    calls = []

    def initialize(self, actual_spec, actual_names, actual_config, actual_device):
        calls.append((actual_spec, actual_names, actual_config, actual_device))

    monkeypatch.setattr(PleShortConvAttentionMetadataBuilder, "__init__", initialize)
    AscendPLEMetadataBuilder(spec, names, original, device)
    assert len(calls) == 1
    actual_spec, actual_names, actual, actual_device = calls[0]
    assert actual_spec is spec and actual_names is names and actual_device is device
    assert original.cache_config is original_cache
    assert (original_cache.use_replayssm, original_cache.use_kda_recoverssm) == (recover, kda)
    assert actual.compilation_config is original.compilation_config
    assert actual.cache_config.block_size == 1024
    assert actual.cache_config.use_replayssm is False
    assert actual.cache_config.use_kda_recoverssm is False
    assert (actual is original) == (not recover)
    assert (actual.cache_config is original_cache) == (not recover)


def test_ple_backend_uses_the_isolated_builder_and_original_metadata_consumer():
    assert AscendPLELayer.get_attn_backend(None) is AscendPLEBackend
    assert AscendPLEBackend.get_builder_cls() is AscendPLEMetadataBuilder
    assert AscendPLEMetadataBuilder.build is PleShortConvAttentionMetadataBuilder.build
