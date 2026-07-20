# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest

import vllm_ascend.patch.hunyuan_vl_processor_compat as compat


def test_installer_runs_release_backports_in_order(monkeypatch):
    import vllm.model_executor.models as vllm_models

    hunyuan_vision = SimpleNamespace(
        HunYuanVLProcessingInfo=SimpleNamespace(),
    )
    calls: list[Any] = []

    def clean_registry() -> bool:
        calls.append("registry")
        return True

    def patch_loader(module: Any) -> None:
        calls.append(("loader", module))

    def patch_wrapping(module: Any) -> None:
        calls.append(("wrapping", module))

    monkeypatch.setattr(
        compat,
        "_remove_stale_registry_entries",
        clean_registry,
    )
    monkeypatch.setattr(vllm_models, "hunyuan_vision", hunyuan_vision, raising=False)
    monkeypatch.setattr(
        compat,
        "_patch_hunyuan_processor_loader",
        patch_loader,
    )
    monkeypatch.setattr(
        compat,
        "_patch_image_token_wrapping",
        patch_wrapping,
    )
    # On v0.25.0 the image-token wrapping backport runs after the loader patch.
    monkeypatch.setattr(compat, "vllm_version_is", lambda version: version == "0.25.0")
    compat.install_hunyuan_vl_processor_compat()

    assert calls == [
        "registry",
        ("loader", hunyuan_vision),
        ("wrapping", hunyuan_vision),
    ]


def test_installer_skips_wrapping_backport_on_main(monkeypatch):
    import vllm.model_executor.models as vllm_models

    hunyuan_vision = SimpleNamespace(
        HunYuanVLProcessingInfo=SimpleNamespace(),
    )
    calls: list[Any] = []

    def clean_registry() -> bool:
        calls.append("registry")
        return True

    def patch_loader(module: Any) -> None:
        calls.append(("loader", module))

    def patch_wrapping(module: Any) -> None:
        calls.append(("wrapping", module))

    monkeypatch.setattr(
        compat,
        "_remove_stale_registry_entries",
        clean_registry,
    )
    monkeypatch.setattr(vllm_models, "hunyuan_vision", hunyuan_vision, raising=False)
    monkeypatch.setattr(
        compat,
        "_patch_hunyuan_processor_loader",
        patch_loader,
    )
    monkeypatch.setattr(
        compat,
        "_patch_image_token_wrapping",
        patch_wrapping,
    )
    # On vllm main the wrapping is native, so the backport must not run.
    monkeypatch.setattr(compat, "vllm_version_is", lambda version: False)
    compat.install_hunyuan_vl_processor_compat()

    assert calls == [
        "registry",
        ("loader", hunyuan_vision),
    ]


def test_installer_cleans_main_registry_before_model_patch(monkeypatch):
    import vllm.model_executor.models as vllm_models

    def native_get_prompt_updates(*_args: Any, **_kwargs: Any) -> str:
        return "native"

    class FakeMultiModalProcessor:
        _get_prompt_updates = native_get_prompt_updates

    hunyuan_vision = SimpleNamespace(
        HunYuanVLMultiModalProcessor=FakeMultiModalProcessor,
    )
    calls: list[Any] = []

    def clean_registry() -> bool:
        calls.append("registry")
        return True

    def patch_loader(module: Any) -> None:
        calls.append(("loader", module))

    monkeypatch.setattr(
        compat,
        "_remove_stale_registry_entries",
        clean_registry,
    )
    monkeypatch.setattr(vllm_models, "hunyuan_vision", hunyuan_vision, raising=False)
    monkeypatch.setattr(
        compat,
        "_patch_hunyuan_processor_loader",
        patch_loader,
    )
    monkeypatch.setattr(compat, "_patch_image_token_wrapping", lambda _module: None)
    monkeypatch.setattr(compat, "vllm_version_is", lambda version: False)
    compat.install_hunyuan_vl_processor_compat()

    assert calls == [
        "registry",
        ("loader", hunyuan_vision),
    ]
    assert FakeMultiModalProcessor._get_prompt_updates is native_get_prompt_updates


def test_registers_hunyuan_tokenizer_schema_without_changing_ids():
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


def test_main_removes_only_stale_registry_entries(monkeypatch):
    import vllm.transformers_utils.processors as vllm_processors

    registry = {
        **compat._STALE_PROCESSOR_MODULES,
        "OtherProcessor": "vllm.transformers_utils.processors.other",
    }
    exported_names = [*registry]
    monkeypatch.setattr(vllm_processors, "_CLASS_TO_MODULE", registry)
    monkeypatch.setattr(vllm_processors, "__all__", exported_names)

    assert compat._remove_stale_registry_entries()
    assert registry == {
        "OtherProcessor": "vllm.transformers_utils.processors.other",
    }
    assert exported_names == ["OtherProcessor"]
    assert not compat._remove_stale_registry_entries()


def test_main_rejects_unexpected_registry_replacement(monkeypatch):
    import vllm.transformers_utils.processors as vllm_processors

    registry = {
        "HunYuanVLProcessor": "future.hunyuan_vl",
    }
    monkeypatch.setattr(vllm_processors, "_CLASS_TO_MODULE", registry)
    monkeypatch.setattr(vllm_processors, "__all__", ["HunYuanVLProcessor"])

    with pytest.raises(RuntimeError, match="Unexpected vLLM processor registry entry"):
        compat._remove_stale_registry_entries()

    assert registry == {"HunYuanVLProcessor": "future.hunyuan_vl"}


def test_installer_preserves_native_prompt_update_protocol(monkeypatch):
    def native_get_prompt_updates(*_args: Any, **_kwargs: Any) -> str:
        return "native"

    class FakeMultiModalProcessor:
        _get_prompt_updates = native_get_prompt_updates

    monkeypatch.setattr(compat, "_remove_stale_registry_entries", lambda: True)
    monkeypatch.setattr(compat, "_patch_hunyuan_processor_loader", lambda _module: None)
    monkeypatch.setattr(compat, "_patch_image_token_wrapping", lambda _module: None)

    compat.install_hunyuan_vl_processor_compat()

    assert FakeMultiModalProcessor._get_prompt_updates is native_get_prompt_updates
