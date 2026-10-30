# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.selector import AttentionSelectorConfig

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.device.hardware_profile import HardwareCapability
from vllm_ascend.models.layer.attention.layer import DSAAttention
from vllm_ascend.platform import NPUPlatform

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
    """A real selector config: ``validate_configuration`` consumes ``_asdict()``."""
    return AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        block_size=32,
        use_mla=True,
        use_sparse=True,
    )


class _UndeclaredBackend(AttentionBackend):
    """Inherits the upstream default set, the way a non-adopting backend does."""


class _DeclaredBackend(AttentionBackend):
    """A backend that opts in to the check."""

    supported_kv_cache_dtypes = ["bfloat16", "fp8"]


class _InheritingBackend(_DeclaredBackend):
    """Declares nothing itself, the way the DeepSeek sparse subclasses do not."""


class _DeviceNarrowedBackend(_DeclaredBackend):
    """Declares a wide set and narrows it per device, the way the DeepSeek one does."""

    supported_kv_cache_dtypes = ["auto", "bfloat16", "fp8"]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        return kv_cache_dtype is None or kv_cache_dtype in ("bfloat16", "fp8")


class TestAscendDSADeclaredSet:
    """The class attribute is the union; the device narrows it in the override."""

    @pytest.mark.parametrize("kv_cache_dtype", ["auto", "bfloat16", "fp8", "int8", "turboquant_4bit_nc"])
    def test_the_declared_set_names_both_parts(self, kv_cache_dtype):
        assert kv_cache_dtype in AscendDSABackend.supported_kv_cache_dtypes

    @pytest.mark.parametrize("kv_cache_dtype", ["fp8_e4m3", "fp8_e5m2", "mxfp8", "nvfp4"])
    def test_undeclared_dtypes_are_not(self, kv_cache_dtype):
        assert kv_cache_dtype not in AscendDSABackend.supported_kv_cache_dtypes

    @pytest.mark.parametrize("compressed_cache", [True, False])
    def test_an_unset_dtype_is_accepted_on_every_device(self, compressed_cache):
        with _hardware_profile(compressed_cache=compressed_cache):
            assert AscendDSABackend.supports_kv_cache_dtype(None) is True


class TestAscendDSADeviceNarrowing:
    """``supports_kv_cache_dtype`` is where the device decides."""

    @pytest.mark.parametrize("kv_cache_dtype", [None, "auto", "bfloat16", "fp8"])
    def test_a5_accepts_the_dtypes_it_implements(self, kv_cache_dtype):
        with _hardware_profile(compressed_cache=True):
            assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is True

    @pytest.mark.parametrize("kv_cache_dtype", ["float16", "int8"])
    def test_a5_rejects_the_dtypes_without_an_implementation(self, kv_cache_dtype):
        """int8 belongs to A3; float16 is not a KV cache dtype here."""
        with _hardware_profile(compressed_cache=True):
            assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is False

    @pytest.mark.parametrize("kv_cache_dtype", [None, "auto", "bfloat16", "int8", "turboquant_4bit_nc"])
    def test_a3_accepts_the_dtypes_it_implements(self, kv_cache_dtype):
        with _hardware_profile(compressed_cache=False):
            assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is True

    @pytest.mark.parametrize("kv_cache_dtype", ["fp8"])
    def test_a3_rejects_what_it_does_not_implement(self, kv_cache_dtype):
        with _hardware_profile(compressed_cache=False):
            assert AscendDSABackend.supports_kv_cache_dtype(kv_cache_dtype) is False


def _get_backend_cls(kv_cache_dtype=None, *, backend_cls=AscendDSABackend, selector=None):
    """Drive the real entry point; only the lookup table's answer is stubbed."""
    config = selector if selector is not None else _selector_config(kv_cache_dtype)
    with (
        _selected_backend(backend_cls),
        patch.object(NPUPlatform, "_select_attn_backend_cls", return_value=_BACKEND_PATH),
    ):
        return NPUPlatform.get_attn_backend_cls(None, config)


class TestRejectedKvCacheDtype:
    def test_rejects_a_dtype_the_declaring_backend_does_not_accept(self):
        with _hardware_profile(compressed_cache=True), pytest.raises(ValueError, match="kv_cache_dtype not supported"):
            _get_backend_cls("float16")

    def test_rejects_a_dtype_the_device_does_not_accept(self):
        """The declared set is wide here; the device is what refuses."""
        with _hardware_profile(compressed_cache=True), pytest.raises(ValueError, match="kv_cache_dtype not supported"):
            _get_backend_cls("auto", backend_cls=_DeviceNarrowedBackend)

    def test_error_lists_the_dtypes_that_would_have_worked(self):
        with _hardware_profile(compressed_cache=True), pytest.raises(ValueError) as excinfo:
            _get_backend_cls("int8")

        message = str(excinfo.value)
        assert "Supported kv_cache_dtype values: auto, bfloat16, fp8." in message
        # int8 is A3-only, so the hint must not offer it even though the rejected
        # value themselves appears in the sentence before.
        hint = message.split("Supported kv_cache_dtype values:")[-1]
        assert "int8" not in hint

    def test_accepts_a_dtype_the_backend_takes(self):
        with _hardware_profile(compressed_cache=True):
            assert _get_backend_cls("fp8") == _BACKEND_PATH

    def test_skips_a_backend_that_does_not_declare(self):
        """The inherited upstream default is not enforced."""
        with _hardware_profile(compressed_cache=True):
            assert _get_backend_cls("auto", backend_cls=_UndeclaredBackend) == _BACKEND_PATH

    def test_does_not_read_the_config_for_an_undeclared_backend(self):
        """A backend that never opted in must not care about the config shape.

        Several existing tests drive ``get_attn_backend_cls`` with a bare
        ``SimpleNamespace`` carrying only the fields the selection table reads, so
        reaching for ``kv_cache_dtype`` first broke the SFA paths.
        """
        assert _get_backend_cls(backend_cls=_UndeclaredBackend, selector=SimpleNamespace()) == _BACKEND_PATH

    def test_accepts_a_selector_config_that_is_not_a_named_tuple(self):
        """``use_compress`` is not a field of ``AttentionSelectorConfig``.

        Callers that set it build a ``SimpleNamespace`` instead (see
        ``tests/ut/attention/test_flash_attn.py``), so nothing here may assume
        ``_asdict()`` exists.
        """
        selector = SimpleNamespace(**{**_selector_config("int8")._asdict(), "use_compress": True})
        with _hardware_profile(compressed_cache=True), pytest.raises(ValueError, match="kv_cache_dtype not supported"):
            _get_backend_cls(backend_cls=AscendDSABackend, selector=selector)

    def test_skips_a_backend_whose_module_cannot_be_imported(self):
        """The FA3 backend imports flash_attn_npu_v3, which a CPU box has not got."""
        with (
            _hardware_profile(compressed_cache=True),
            patch("vllm.utils.import_utils.resolve_obj_by_qualname", side_effect=ImportError("flash_attn_npu_v3")),
            patch.object(NPUPlatform, "_select_attn_backend_cls", return_value=_BACKEND_PATH),
        ):
            assert NPUPlatform.get_attn_backend_cls(None, _selector_config("auto")) == _BACKEND_PATH

    def test_a_subclass_is_not_checked_here(self):
        """Only the class that states the set is asked.

        The DeepSeek sparse layers are the only Ascend subclasses that inherit the
        declaration, and they pick their backend in the layer and ask it there, so
        the platform selector leaves them alone.
        """
        with _hardware_profile(compressed_cache=True):
            assert _get_backend_cls("auto", backend_cls=_InheritingBackend) == _BACKEND_PATH

    def test_other_hardware_is_untouched(self):
        with _hardware_profile(compressed_cache=False):
            assert _get_backend_cls("bfloat16") == _BACKEND_PATH


class TestQuantPathsThatStateNothing:
    """The C8 and TurboQuant dtypes reach backends that state no set.

    ``Attention.__init__`` calls ``get_attn_backend(kv_cache_dtype=...)`` before the
    quant method swaps the backend in ``create_weights`` (``kv_cache/kv_c8.py``,
    ``kv_cache/mxfp_c8.py``), so those dtypes do reach this selector. What keeps them
    working is the guard, not the absence of a call: stating a set on these backends
    is the change that would start refusing them, which is why none of them does yet.
    """

    @pytest.mark.parametrize(
        ("backend_cls", "kv_cache_dtype"),
        [
            (AscendAttentionBackend, "int8"),  # C8 INT8 upgrades this one's impl
            (AscendAttentionBackend, "mxfp8"),  # C8 MXFP8 swaps the backend after it
            (AscendSFABackend, "turboquant_4bit_nc"),  # TurboQuant on SFA
        ],
    )
    def test_the_selector_leaves_them_alone(self, backend_cls, kv_cache_dtype):
        for compressed_cache in (True, False):
            with _hardware_profile(compressed_cache=compressed_cache):
                assert _get_backend_cls(kv_cache_dtype, backend_cls=backend_cls) == _BACKEND_PATH


class TestSparseLayerEntry:
    """The path that fires on A5: the layer picks its own backend."""

    def _layer_kwargs(self, kv_cache_dtype):
        return dict(
            dim=7168,
            n_heads=128,
            scale=0.1,
            n_local_heads=128,
            q_lora_rank=1536,
            o_lora_rank=1536,
            head_dim=576,
            rope_head_dim=64,
            nope_head_dim=512,
            n_groups=16,
            n_local_groups=16,
            window_size=128,
            compress_ratio=4,
            cache_config=SimpleNamespace(cache_dtype=kv_cache_dtype, block_size=32, enable_prefix_caching=False),
            prefix="test.layer",
            swa_cache_layer=object(),
        )

    def test_rejects_before_the_layer_stores_a_dtype(self):
        """int8 is A3-only, so A5 must refuse it where the backend is picked."""
        with (
            _hardware_profile(compressed_cache=True),
            patch("vllm_ascend.models.layer.attention.layer._init_kv_cache_quant"),
            pytest.raises(ValueError, match="Supported kv_cache_dtype values: auto, bfloat16, fp8."),
        ):
            DSAAttention(**self._layer_kwargs("int8"))

    def test_the_hint_names_what_this_device_accepts(self):
        """A3 stores int8 and turboquant; naming fp8 there would be wrong either way."""
        with (
            _hardware_profile(compressed_cache=False),
            patch("vllm_ascend.models.layer.attention.layer._init_kv_cache_quant"),
            pytest.raises(
                ValueError,
                match="Supported kv_cache_dtype values: auto, bfloat16, int8, turboquant_4bit_nc.",
            ),
        ):
            DSAAttention(**self._layer_kwargs("fp8"))
