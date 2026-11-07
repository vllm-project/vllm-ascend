# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_config import PushConfig

from .helpers import make_full_spec


def configuration():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        speculative_config=None,
        model_config=SimpleNamespace(use_mla=False, enforce_eager=True),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=lambda _key, default: default),
    )


def validate(config):
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=make_full_spec())])
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_config.get_ascend_config",
        return_value=SimpleNamespace(kvpp_config=SimpleNamespace(size=1)),
    ):
        return PushConfig.from_config(config, cache)


def test_accept_initial_configuration():
    assert validate(configuration()).max_pending == 128


@pytest.mark.parametrize(
    "group,name,value",
    [
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("cache_config", "enable_prefix_caching", True),
        ("scheduler_config", "async_scheduling", True),
        ("model_config", "enforce_eager", False),
        ("model_config", "use_mla", True),
    ],
)
def test_reject_unsupported_mode(group, name, value):
    config = configuration()
    setattr(getattr(config, group), name, value)
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_reject_invalid_deadline(value):
    config = configuration()
    config.kv_transfer_config.get_from_extra_config = lambda key, default: value if key == "push_timeout" else default
    with pytest.raises(ValueError):
        validate(config)
