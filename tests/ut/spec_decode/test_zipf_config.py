# SPDX-License-Identifier: Apache-2.0

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.config import get_ascend_spec_decode_method, get_zipf_config
from vllm_ascend.spec_decode.zipf_proposer import AscendZipfDecodingProposer


def make_vllm_config(additional_config=None):
    vllm_config = MagicMock()
    vllm_config.speculative_config = MagicMock()
    vllm_config.speculative_config.method = "suffix"
    vllm_config.speculative_config.num_speculative_tokens = 3
    vllm_config.model_config.max_model_len = 1024
    vllm_config.additional_config = additional_config
    return vllm_config


def test_ascend_spec_decode_method_defaults_to_speculative_config_method():
    vllm_config = make_vllm_config()

    assert get_ascend_spec_decode_method(vllm_config) == "suffix"


def test_ascend_spec_decode_method_uses_additional_config_override():
    vllm_config = make_vllm_config({"ascend_spec_decode_method": "zipf"})

    assert get_ascend_spec_decode_method(vllm_config) == "zipf"


def test_zipf_config_reads_additional_config_values():
    vllm_config = make_vllm_config(
        {
            "zipf_config": {
                "zipf_ngram_size": 4,
                "zipf_min_window": 1,
                "zipf_initial_speculative_tokens": 2,
                "zipf_skip_shared": False,
                "zipf_generalized_before_shared": True,
            }
        }
    )

    assert get_zipf_config(vllm_config) == {
        "zipf_ngram_size": 4,
        "zipf_min_window": 1,
        "zipf_initial_speculative_tokens": 2,
        "zipf_skip_shared": False,
        "zipf_generalized_before_shared": True,
    }


@pytest.mark.parametrize(
    ("zipf_config", "error"),
    [
        ({"zipf_initial_speculative_tokens": 4}, "less than or equal"),
        ({"zipf_min_window": 0}, "greater than 0"),
        ({"zipf_ngram_size": 9}, "less than or equal to 8"),
        ({"zipf_min_window": 5, "zipf_ngram_size": 4}, "greater than or equal"),
        ({"zipf_skip_shared": 1}, "zipf_skip_shared must be a boolean"),
    ],
)
def test_zipf_config_validates_values(zipf_config, error):
    vllm_config = make_vllm_config({"zipf_config": zipf_config})

    with pytest.raises(ValueError, match=error):
        get_zipf_config(vllm_config)


def test_get_spec_decode_method_creates_zipf_proposer_from_override():
    vllm_config = make_vllm_config(
        {
            "ascend_spec_decode_method": "zipf",
            "zipf_config": {
                "zipf_ngram_size": 4,
                "zipf_min_window": 1,
                "zipf_initial_speculative_tokens": 2,
                "zipf_skip_shared": True,
                "zipf_generalized_before_shared": False,
            },
        }
    )
    runner = MagicMock()

    zipf_cache_module = ModuleType("vllm_ascend.spec_decode.zipf_cache")
    mock_zipf_cache = MagicMock()
    zipf_cache_module.ZipfCache = mock_zipf_cache
    with patch.dict("sys.modules", {"vllm_ascend.spec_decode.zipf_cache": zipf_cache_module}):
        proposer = get_spec_decode_method(
            get_ascend_spec_decode_method(vllm_config),
            vllm_config,
            device=None,
            runner=runner,
        )

    assert isinstance(proposer, AscendZipfDecodingProposer)
    assert proposer.num_speculative_tokens == 2
    assert proposer.max_speculative_tokens == 3
    assert proposer.min_window == 1
    assert proposer.max_window == 4
    assert proposer.skip_shared is True
    mock_zipf_cache.assert_called_once_with(
        min_window=1,
        max_window=4,
        skip_shared=True,
        generalized_before_shared=False,
    )
