# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_ascend.ascend_config import NovaKVConfig


def test_capture_configuration():
    assert NovaKVConfig(capture_path="/capture").artifact_path is None


def test_compression_configuration():
    assert NovaKVConfig(artifact_path="/artifacts").capture_path is None


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"capture_path": "/capture", "artifact_path": "/artifacts"},
        {"capture_path": "/capture", "recent_tokens": 0},
        {"capture_path": "/capture", "sink_tokens": -1},
        {"capture_path": "/capture", "codec_batch_size": 0},
        {"capture_path": "/capture", "sample_tokens": 0},
        {"capture_path": "/capture", "unknown_option": True},
        {"artifact_path": "/artifacts", "backend": "tilelang"},
        {"artifact_path": "/artifacts", "backend": "unknown"},
        {"artifact_path": "/artifacts", "backend": "direct"},
        {"artifact_path": "/artifacts", "backend": "cube"},
        {"artifact_path": "/artifacts", "direct_library_root": "/kernel"},
        {"artifact_path": "/artifacts", "direct_split_tokens": 512},
        {"artifact_path": "/artifacts", "prefill_query_tile": 8192},
        {"artifact_path": "/artifacts", "prefill_kv_tile": 32768},
    ],
)
def test_invalid_or_removed_configuration(options):
    with pytest.raises(ValueError):
        NovaKVConfig(**options)
