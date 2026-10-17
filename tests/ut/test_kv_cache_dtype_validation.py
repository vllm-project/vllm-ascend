# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.device.hardware_profile import HardwareCapability
from vllm_ascend.platform import (
    _check_selector_kv_cache_dtype,
    reject_unsupported_kv_cache_dtype,
)

_BACKEND_PATH = "vllm_ascend.attention.dsa_v1.AscendDSABackend"


@contextmanager
def _hardware_profile(*, compressed_cache):
    profile = SimpleNamespace(
        supports=lambda capability: compressed_cache and capability == HardwareCapability.DSV4_COMPRESSED_CACHE,
    )
    with patch(
        "vllm_ascend.attention.dsa_v1.get_current_hardware_profile",
        return_value=profile,
    ):
        yield


@contextmanager
def _selected_backend(backend_cls):
    """Stand in for the qualname lookup the validation resolves the backend with."""
    with patch("vllm.utils.import_utils.resolve_obj_by_qualname", return_value=backend_cls):
        yield


def _selector_config(kv_cache_dtype):
    return SimpleNamespace(
        head_size=576,
        dtype=None,
        kv_cache_dtype=kv_cache_dtype,
        block_size=32,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        use_mm_prefix=False,
    )


def _combination_reason(kv_cache_dtype):
    """Ask the DeepSeek backend upstream's device-specific question."""
    return AscendDSABackend.supports_combination(
        576,
        None,
        kv_cache_dtype,
        32,
        True,
        False,
        True,
        False,
        None,
    )


class _Backend:
    """The parts of ``AttentionBackend`` the validation touches, minus the opt-in."""

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        return False

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ):
        return None


class _UndeclaredBackend(_Backend):
    """Declares nothing, the way a backend inheriting the upstream default does."""


class _DeclaredBackend(_Backend):
    """A backend that opts in to the check."""

    supported_kv_cache_dtypes = ["bfloat16", "fp8"]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes


class _InheritingBackend(_DeclaredBackend):
    """Declares nothing itself, the way the DeepSeek sparse subclasses do not."""


class _DeviceNarrowedBackend(_DeclaredBackend):
    """Declares a wide set and narrows it per device, the way the DeepSeek one does."""

    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8"]

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ):
        if kv_cache_dtype in ("bfloat16", "fp8"):
            return None
        return "kv_cache_dtype not supported"


class TestAscendDSADeclaredSet:
    """The declaration states the backend's capability, whatever the device."""

    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "fp8", None])
    def test_declared_dtypes_are_supported(self, kv_cache_dtype):
        assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is True

    @pytest.mark.parametrize("kv_cache_dtype", ["fp8_e4m3", "fp8_e5m2", "int8", "mxfp8", "nvfp4"])
    def test_undeclared_dtypes_are_not(self, kv_cache_dtype):
        assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is False


class TestAscendDSADeviceNarrowing:
    """``supports_combination`` is where the device decides, as it is upstream."""

    @pytest.mark.parametrize("kv_cache_dtype", [None, "bfloat16", "fp8"])
    def test_a5_accepts_the_dtypes_it_implements(self, kv_cache_dtype):
        with _hardware_profile(compressed_cache=True):
            assert _combination_reason(kv_cache_dtype) is None

    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "float16", "int8"])
    def test_a5_rejects_the_dtypes_without_an_implementation(self, kv_cache_dtype):
        """``auto`` lands here too: it resolves to the model dtype, bf16."""
        with _hardware_profile(compressed_cache=True):
            assert _combination_reason(kv_cache_dtype) == "kv_cache_dtype not supported"

    @pytest.mark.parametrize("kv_cache_dtype", [None, "bfloat16", "auto"])
    def test_a3_leaves_the_choice_to_the_config(self, kv_cache_dtype):
        with _hardware_profile(compressed_cache=False):
            assert _combination_reason(kv_cache_dtype) is None

    @pytest.mark.parametrize("kv_cache_dtype", ["fp8", "int8"])
    def test_a3_rejects_what_it_does_not_implement(self, kv_cache_dtype):
        with _hardware_profile(compressed_cache=False):
            assert _combination_reason(kv_cache_dtype) == "kv_cache_dtype not supported"


class TestRaiseForRejectedKvCacheDtype:
    def test_rejects_a_dtype_the_declaring_backend_does_not_accept(self):
        with (
            _hardware_profile(compressed_cache=True),
            _selected_backend(AscendDSABackend),
            pytest.raises(ValueError, match="kv_cache_dtype not supported"),
        ):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("float16"))

    def test_rejects_a_dtype_the_device_does_not_accept(self):
        """The declared set is wide here; the device is what refuses."""
        with (
            _hardware_profile(compressed_cache=True),
            _selected_backend(_DeviceNarrowedBackend),
            pytest.raises(ValueError, match="kv_cache_dtype not supported"),
        ):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("auto"))

    def test_error_lists_the_dtypes_that_would_have_worked(self):
        with (
            _hardware_profile(compressed_cache=True),
            _selected_backend(AscendDSABackend),
            pytest.raises(ValueError) as excinfo,
        ):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("auto"))

        message = str(excinfo.value)
        assert "Supported kv_cache_dtype values: bfloat16, fp8." in message
        # The declared set is wider than what A5 accepts; listing a dtype the
        # device just refused would contradict the message.
        assert "auto, bfloat16" not in message

    def test_accepts_a_dtype_the_backend_takes(self):
        with _hardware_profile(compressed_cache=True), _selected_backend(AscendDSABackend):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("fp8"))

    def test_skips_a_backend_that_does_not_declare(self):
        """The inherited upstream default is not enforced."""
        with _hardware_profile(compressed_cache=True), _selected_backend(_UndeclaredBackend):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("auto"))

    def test_does_not_read_the_config_for_an_undeclared_backend(self):
        """A backend that never opted in must not care about the config shape.

        Several existing tests drive ``get_attn_backend_cls`` with a bare
        ``SimpleNamespace`` carrying only the fields the selection table reads, so
        reaching for ``kv_cache_dtype`` first broke the SFA paths.
        """
        with _selected_backend(_UndeclaredBackend):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, SimpleNamespace())

    def test_skips_a_backend_whose_module_cannot_be_imported(self):
        """The FA3 backend imports flash_attn_npu_v3, which a CPU box has not got."""
        with (
            _hardware_profile(compressed_cache=True),
            patch("vllm.utils.import_utils.resolve_obj_by_qualname", side_effect=ImportError("flash_attn_npu_v3")),
        ):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("auto"))

    def test_subclasses_inherit_the_opt_in(self):
        """The DeepSeek sparse layers select subclasses, not the declaring base."""
        with (
            _hardware_profile(compressed_cache=True),
            _selected_backend(_InheritingBackend),
            pytest.raises(ValueError, match="_InheritingBackend"),
        ):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("auto"))

    def test_other_hardware_is_untouched(self):
        with _hardware_profile(compressed_cache=False), _selected_backend(AscendDSABackend):
            _check_selector_kv_cache_dtype(_BACKEND_PATH, _selector_config("bfloat16"))


def _deepseek_combination(dtype):
    """What the DeepSeek sparse layer binds when it picks its own backend."""
    return AscendDSABackend.supports_combination(576, None, dtype, 32, True, False, True, False, None)


class TestModelSideEntry:
    """The DeepSeek sparse layers never reach the platform selector."""

    def test_rejects_a_dtype_the_device_does_not_run(self):
        with _hardware_profile(compressed_cache=True), pytest.raises(ValueError) as excinfo:
            reject_unsupported_kv_cache_dtype(AscendDSABackend, "auto", _deepseek_combination)

        assert "Supported kv_cache_dtype values: bfloat16, fp8." in str(excinfo.value)

    def test_accepts_a_dtype_the_device_runs(self):
        with _hardware_profile(compressed_cache=True):
            reject_unsupported_kv_cache_dtype(AscendDSABackend, "fp8", _deepseek_combination)

    def test_skips_a_backend_that_does_not_declare(self):
        def never_asked(_dtype):
            raise AssertionError("the question must not be asked")

        reject_unsupported_kv_cache_dtype(_UndeclaredBackend, "auto", never_asked)
