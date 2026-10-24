"""Shared configuration inputs and logging capture for AscendStore v1 tests."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.logger import logger
from vllm.v1.kv_cache_interface import FullAttentionSpec


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.LogCaptureFixture]:
    """Capture v1 logs even when vLLM does not propagate them to the root logger."""
    # Capture at the emitting logger; disable propagation to avoid duplicate records.
    monkeypatch.setattr(logger, "propagate", False)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


@pytest.fixture
def configuration_case():
    model = SimpleNamespace(
        model="model",
        max_model_len=64,
        use_mla=False,
        hf_text_config=SimpleNamespace(num_hidden_layers=4),
        get_total_num_kv_heads=Mock(return_value=4),
        get_total_num_hidden_layers=lambda: 4,
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            world_size=1,
        ),
        model_config=model,
        cache_config=SimpleNamespace(block_size=4, enable_prefix_caching=True, prefix_match_unit=None),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    spec = FullAttentionSpec(block_size=4, num_kv_heads=4, head_size=1, dtype=torch.float32)
    cache = SimpleNamespace(
        num_blocks=8,
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["model.layers.0"], is_eagle_group=False)],
        prefix_cache_retention_interval=None,
    )
    return config, cache
