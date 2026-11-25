# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.ops.weight_prefetch_config import Qwen35WeightPrefetchConfig


def test_config_defaults_to_disabled():
    config = Qwen35WeightPrefetchConfig.parse({})

    assert not config.enabled
    assert config.targets == ("next_input",)
    assert config.next_input_join == "immediate"


def test_config_accepts_supported_options():
    config = Qwen35WeightPrefetchConfig.parse(
        {
            "enabled": True,
            "mode": "control",
            "targets": ["next_input", "gdn_out"],
            "next_input_bytes": "full",
            "layers": [0, 39],
            "next_input_join": "deferred",
            "min_decode_concurrency": 8,
        }
    )

    assert config.enabled
    assert config.mode == "control"
    assert config.targets == ("next_input", "gdn_out")
    assert config.next_input_bytes == "full"
    assert config.layers == (0, 39)
    assert config.next_input_join == "deferred"


@pytest.mark.parametrize(
    "options",
    [
        None,
        [],
        {"unknown": True},
        {"enabled": 1},
        {"mode": "invalid"},
        {"targets": []},
        {"targets": ["unknown"]},
        {"targets": ["next_input", "next_input"]},
        {"next_input_bytes": 0},
        {"next_input_bytes": 33},
        {"next_input_bytes": True},
        {"next_input_bytes": "all"},
        {"routed_w2_bytes": -32},
        {"gdn_out_bytes": 31},
        {"layers": [-1]},
        {"layers": [40]},
        {"layers": [True]},
        {"next_layer_types": ["unknown"]},
        {"routed_start": "before_dispatch"},
        {"next_input_join": "late"},
        {"min_decode_concurrency": 0},
        {"min_decode_concurrency": True},
        {"vector_cores": 3},
    ],
)
def test_config_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        Qwen35WeightPrefetchConfig.parse(options)
