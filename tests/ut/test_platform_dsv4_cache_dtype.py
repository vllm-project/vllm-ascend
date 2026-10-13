# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.device.hardware_profile import HardwareCapability
from vllm_ascend.platform import _reject_unsupported_kv_cache_dtype


@contextmanager
def _hardware_profile(*, supports_compressed_cache):
    profile = SimpleNamespace(
        supports=lambda capability: (
            supports_compressed_cache and capability == HardwareCapability.DSV4_COMPRESSED_CACHE
        ),
    )
    with patch(
        "vllm_ascend.attention.dsa_v1.get_current_hardware_profile",
        return_value=profile,
    ):
        yield


def _selector_config(kv_cache_dtype):
    return SimpleNamespace(kv_cache_dtype=kv_cache_dtype)


class TestAscendDSASupportsKvCacheDtype:
    @pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2", None])
    def test_a5_accepts_the_dtypes_the_operator_takes(self, kv_cache_dtype):
        with _hardware_profile(supports_compressed_cache=True):
            assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is True

    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "float16"])
    def test_a5_rejects_the_dtypes_without_an_implementation(self, kv_cache_dtype):
        """``auto`` lands here too: it resolves to the model dtype, bf16."""
        with _hardware_profile(supports_compressed_cache=True):
            assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is False

    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "float16", "fp8", "fp8_e4m3"])
    def test_other_hardware_keeps_the_wider_set(self, kv_cache_dtype):
        with _hardware_profile(supports_compressed_cache=False):
            assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is True


class TestRejectUnsupportedKvCacheDtype:
    def test_rejects_with_the_backend_selection_wording(self):
        with (
            _hardware_profile(supports_compressed_cache=True),
            pytest.raises(ValueError, match="kv_cache_dtype not supported"),
        ):
            _reject_unsupported_kv_cache_dtype("npu", AscendDSABackend, _selector_config("auto"))

    def test_error_lists_the_dtypes_that_work(self):
        with (
            _hardware_profile(supports_compressed_cache=True),
            pytest.raises(
                ValueError,
                match=r"Supported kv_cache_dtype values: fp8, fp8_e4m3, fp8_e5m2",
            ),
        ):
            _reject_unsupported_kv_cache_dtype("npu", AscendDSABackend, _selector_config("bfloat16"))

    def test_accepts_a_dtype_the_backend_takes(self):
        with _hardware_profile(supports_compressed_cache=True):
            _reject_unsupported_kv_cache_dtype("npu", AscendDSABackend, _selector_config("fp8"))

    def test_other_hardware_is_untouched(self):
        with _hardware_profile(supports_compressed_cache=False):
            _reject_unsupported_kv_cache_dtype("npu", AscendDSABackend, _selector_config("auto"))
