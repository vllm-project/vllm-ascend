# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
from transformers import AutoImageProcessor

import vllm_ascend.patch.platform.patch_hunyuan_vl_processor as compat


def test_import_skips_legacy_image_processor_registration(monkeypatch):
    hunyuan_vision = object()
    original_descriptor = AutoImageProcessor.__dict__["register"]

    def import_hunyuan_vision(name: str) -> object:
        assert name == "vllm.model_executor.models.hunyuan_vision"
        AutoImageProcessor.register("HunYuanVLImageProcessor", object)
        return hunyuan_vision

    monkeypatch.setattr(compat.importlib, "import_module", import_hunyuan_vision)

    assert compat._import_v025_hunyuan_vision() is hunyuan_vision
    assert AutoImageProcessor.__dict__["register"] is original_descriptor


def test_import_restores_registration_after_error(monkeypatch):
    original_descriptor = AutoImageProcessor.__dict__["register"]

    def fail_import(_name: str) -> object:
        raise ImportError("expected test failure")

    monkeypatch.setattr(compat.importlib, "import_module", fail_import)

    with pytest.raises(ImportError, match="expected test failure"):
        compat._import_v025_hunyuan_vision()

    assert AutoImageProcessor.__dict__["register"] is original_descriptor


def test_registers_missing_hunyuan_tokenizer_schema_once():
    class FakeTokenizer:
        pad_token = compat._HUNYUAN_VL_SPECIAL_TOKENS["pad_token"]
        pad_token_id = compat._HUNYUAN_VL_SPECIAL_TOKEN_IDS["pad_token"]

        def __init__(self) -> None:
            self.registrations: list[dict[str, str]] = []

        def _set_model_specific_special_tokens(self, special_tokens: dict[str, str]) -> None:
            self.registrations.append(special_tokens)
            for name, token in special_tokens.items():
                setattr(self, name, token)
                setattr(self, f"{name}_id", compat._HUNYUAN_VL_SPECIAL_TOKEN_IDS[name])

    tokenizer = FakeTokenizer()

    compat._register_hunyuan_tokenizer_special_tokens(tokenizer)
    compat._register_hunyuan_tokenizer_special_tokens(tokenizer)

    assert tokenizer.registrations == [compat._HUNYUAN_VL_EXTRA_SPECIAL_TOKENS]


def test_preserves_existing_hunyuan_tokenizer_schema():
    tokenizer = SimpleNamespace(
        **compat._HUNYUAN_VL_SPECIAL_TOKENS,
        **{f"{name}_id": token_id for name, token_id in compat._HUNYUAN_VL_SPECIAL_TOKEN_IDS.items()},
        _set_model_specific_special_tokens=lambda **_kwargs: pytest.fail("unexpected registration"),
    )

    compat._register_hunyuan_tokenizer_special_tokens(tokenizer)


def test_rejects_hunyuan_token_id_mismatch():
    tokenizer = SimpleNamespace(
        **compat._HUNYUAN_VL_SPECIAL_TOKENS,
        **{f"{name}_id": token_id for name, token_id in compat._HUNYUAN_VL_SPECIAL_TOKEN_IDS.items()},
    )
    tokenizer.image_token_id = 1

    with pytest.raises(ValueError, match="does not match the model vocabulary"):
        compat._register_hunyuan_tokenizer_special_tokens(tokenizer)


def test_compat_processor_registers_schema_before_native_init(monkeypatch):
    tokenizer = object()
    calls: list[tuple[Any, ...]] = []

    monkeypatch.setattr(
        compat,
        "_register_hunyuan_tokenizer_special_tokens",
        lambda value: calls.append(("register", value)),
    )

    def native_init(
        self: Any,
        image_processor: Any = None,
        tokenizer: Any = None,
        chat_template: Any = None,
        cat_extra_token: bool = True,
        **kwargs: Any,
    ) -> None:
        calls.append(("native", tokenizer, cat_extra_token, kwargs))

    monkeypatch.setattr(compat.HunYuanVLProcessor, "__init__", native_init)

    compat._HunYuanVLProcessorCompat(
        image_processor=object(),
        tokenizer=tokenizer,
        cat_extra_token=False,
        custom=True,
    )

    assert calls == [
        ("register", tokenizer),
        ("native", tokenizer, False, {"custom": True}),
    ]


def test_installer_patches_main_loader_only(monkeypatch):
    class FakeProcessingInfo:
        pass

    class FakeMultiModalProcessor:
        def _call_hf_processor(self, *args: Any) -> str:
            return "native"

    hunyuan_vision = SimpleNamespace(
        HunYuanVLProcessingInfo=FakeProcessingInfo,
        HunYuanVLMultiModalProcessor=FakeMultiModalProcessor,
    )
    native_call_hf_processor = FakeMultiModalProcessor._call_hf_processor
    monkeypatch.setattr(compat, "vllm_version_is", lambda _version: False)
    monkeypatch.setattr(compat.importlib, "import_module", lambda _name: hunyuan_vision)
    compat.install_hunyuan_vl_processor_compat()
    calls: list[tuple[Any, dict[str, Any]]] = []

    def get_hf_processor(processor_class: Any, **kwargs: Any) -> str:
        calls.append((processor_class, kwargs))
        return "processor"

    processing_info = SimpleNamespace(
        ctx=SimpleNamespace(get_hf_processor=get_hf_processor),
    )
    patched_get_hf_processor = FakeProcessingInfo.get_hf_processor  # type: ignore[attr-defined]

    result = patched_get_hf_processor(
        processing_info,
        use_fast=True,
        min_pixels=128,
    )

    assert result == "processor"
    assert calls == [
        (
            compat._HunYuanVLProcessorCompat,
            {"min_pixels": 128, "backend": "pil"},
        )
    ]
    assert FakeMultiModalProcessor._call_hf_processor is native_call_hf_processor


def test_installer_backports_v025_image_token_wrapping(monkeypatch):
    class FakeProcessingInfo:
        pass

    class FakeMultiModalProcessor:
        info: Any

        def _call_hf_processor(self, *args: Any) -> str:
            return "native"

    hunyuan_vision = SimpleNamespace(
        HunYuanVLProcessingInfo=FakeProcessingInfo,
        HunYuanVLMultiModalProcessor=FakeMultiModalProcessor,
    )
    native_call_hf_processor = FakeMultiModalProcessor._call_hf_processor
    monkeypatch.setattr(compat, "vllm_version_is", lambda version: version == "0.25.1")
    monkeypatch.setattr(compat, "_import_v025_hunyuan_vision", lambda: hunyuan_vision)
    compat.install_hunyuan_vl_processor_compat()
    assert FakeMultiModalProcessor._call_hf_processor is not native_call_hf_processor
    calls: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    hf_processor = SimpleNamespace(
        image_token="<image>",
        image_start_token="<image_start>",
        image_end_token="<image_end>",
    )

    def call_hf_processor(
        processor: Any,
        inputs: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> str:
        calls.append((processor, inputs, kwargs))
        return "processed"

    processor = FakeMultiModalProcessor()
    processor.info = SimpleNamespace(
        get_hf_processor=lambda **_kwargs: hf_processor,
        ctx=SimpleNamespace(call_hf_processor=call_hf_processor),
    )

    call_processor = FakeMultiModalProcessor._call_hf_processor  # type: ignore[attr-defined]
    assert (
        call_processor(
            processor,
            "before<image>after",
            {"images": [object()]},
            {},
            {},
        )
        == "processed"
    )
    assert calls[0][1]["text"] == "before<image_start><image><image_end>after"

    wrapped_prompt = "before<image_start><image><image_end>after"
    assert (
        call_processor(
            processor,
            wrapped_prompt,
            {"images": [object()]},
            {},
            {},
        )
        == "processed"
    )
    assert calls[1][1]["text"] == wrapped_prompt
