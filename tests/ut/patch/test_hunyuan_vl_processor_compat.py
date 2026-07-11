# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_ascend.patch.hunyuan_vl_processor_compat as compat


def test_v023_imports_native_processors_without_persistent_aliases(monkeypatch):
    import transformers.models.hunyuan_vl.image_processing_hunyuan_vl as native_image
    import vllm.transformers_utils.processors as vllm_processors

    class FakeProcessor:
        pass

    class FakeImageProcessor:
        pass

    def fake_smart_resize(*_args: Any, **_kwargs: Any) -> tuple[int, int]:
        return (1, 1)

    processor_module_name = compat._STALE_PROCESSOR_MODULES["HunYuanVLProcessor"]
    image_module_name = compat._STALE_PROCESSOR_MODULES["HunYuanVLImageProcessor"]
    previous_modules = {
        processor_module_name: sys.modules.get(processor_module_name),
        image_module_name: sys.modules.get(image_module_name),
    }
    previous_attributes = {
        "hunyuan_vl": vars(vllm_processors).get("hunyuan_vl"),
        "hunyuan_vl_image": vars(vllm_processors).get("hunyuan_vl_image"),
    }
    hunyuan_vision = ModuleType("vllm.model_executor.models.hunyuan_vision")

    def import_hunyuan_vision(name: str) -> ModuleType:
        assert name == hunyuan_vision.__name__
        assert sys.modules[processor_module_name].HunYuanVLProcessor is FakeProcessor
        assert sys.modules[image_module_name].HunYuanVLImageProcessor is FakeImageProcessor
        assert sys.modules[image_module_name].smart_resize is fake_smart_resize
        vars(vllm_processors)["hunyuan_vl"] = sys.modules[processor_module_name]
        vars(vllm_processors)["hunyuan_vl_image"] = sys.modules[image_module_name]
        return hunyuan_vision

    monkeypatch.setattr(compat, "HunYuanVLProcessor", FakeProcessor)
    monkeypatch.setattr(native_image, "HunYuanVLImageProcessor", FakeImageProcessor)
    monkeypatch.setattr(native_image, "smart_resize", fake_smart_resize)
    monkeypatch.setattr(compat.importlib, "import_module", import_hunyuan_vision)

    assert compat._import_v023_hunyuan_vision() is hunyuan_vision

    for module_name, previous_module in previous_modules.items():
        assert sys.modules.get(module_name) is previous_module
    for attribute_name, previous_attribute in previous_attributes.items():
        assert vars(vllm_processors).get(attribute_name) is previous_attribute


def test_v023_restores_aliases_after_import_error(monkeypatch):
    import transformers.models.hunyuan_vl.image_processing_hunyuan_vl as native_image

    class FakeProcessor:
        pass

    class FakeImageProcessor:
        pass

    processor_module_name = compat._STALE_PROCESSOR_MODULES["HunYuanVLProcessor"]
    image_module_name = compat._STALE_PROCESSOR_MODULES["HunYuanVLImageProcessor"]
    previous_modules = {
        processor_module_name: sys.modules.get(processor_module_name),
        image_module_name: sys.modules.get(image_module_name),
    }

    monkeypatch.setattr(compat, "HunYuanVLProcessor", FakeProcessor)
    monkeypatch.setattr(native_image, "HunYuanVLImageProcessor", FakeImageProcessor)

    def fail_import(_name: str) -> ModuleType:
        raise ImportError("expected test failure")

    monkeypatch.setattr(compat.importlib, "import_module", fail_import)

    with pytest.raises(ImportError, match="expected test failure"):
        compat._import_v023_hunyuan_vision()

    for module_name, previous_module in previous_modules.items():
        assert sys.modules.get(module_name) is previous_module


def test_installer_runs_v023_backports_in_order(monkeypatch):
    hunyuan_vision = object()
    calls: list[Any] = []

    def import_hunyuan_vision() -> object:
        calls.append("import")
        return hunyuan_vision

    def clean_registry() -> bool:
        calls.append("registry")
        return True

    def patch_processor(module: Any) -> None:
        calls.append(("processor", module))

    def patch_loader(module: Any) -> None:
        calls.append(("loader", module))

    def patch_prompt(module: Any) -> None:
        calls.append(("prompt", module))

    monkeypatch.setattr(compat, "vllm_version_is", lambda version: version == "0.23.0")
    monkeypatch.setattr(
        compat,
        "_import_v023_hunyuan_vision",
        import_hunyuan_vision,
    )
    monkeypatch.setattr(
        compat,
        "_remove_stale_registry_entries",
        clean_registry,
    )
    monkeypatch.setattr(
        compat,
        "_patch_hunyuan_processor_loader",
        patch_loader,
    )
    monkeypatch.setattr(
        compat,
        "_patch_v023_processor_methods",
        patch_processor,
    )
    monkeypatch.setattr(
        compat,
        "_patch_prompt_updates",
        patch_prompt,
    )

    compat.install_hunyuan_vl_processor_compat()

    assert calls == [
        "import",
        "registry",
        ("loader", hunyuan_vision),
        ("processor", hunyuan_vision),
        ("prompt", hunyuan_vision),
    ]


def test_installer_cleans_main_registry_before_model_patch(monkeypatch):
    import vllm.model_executor.models as vllm_models

    hunyuan_vision = object()
    calls: list[Any] = []

    def clean_registry() -> bool:
        calls.append("registry")
        return True

    def patch_prompt(module: Any) -> None:
        calls.append(("prompt", module))

    def patch_loader(module: Any) -> None:
        calls.append(("loader", module))

    monkeypatch.setattr(compat, "vllm_version_is", lambda _version: False)
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
        "_patch_prompt_updates",
        patch_prompt,
    )

    compat.install_hunyuan_vl_processor_compat()

    assert calls == [
        "registry",
        ("loader", hunyuan_vision),
        ("prompt", hunyuan_vision),
    ]


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


def test_v023_backports_native_processor_call_protocol(monkeypatch):
    class FakeProcessingInfo:
        pass

    class FakeMultiModalProcessor:
        pass

    hunyuan_vision = SimpleNamespace(
        HunYuanVLProcessingInfo=FakeProcessingInfo,
        HunYuanVLMultiModalProcessor=FakeMultiModalProcessor,
    )
    compat._patch_hunyuan_processor_loader(hunyuan_vision)
    compat._patch_v023_processor_methods(hunyuan_vision)

    processor_args: list[tuple[Any, dict[str, Any]]] = []

    def get_processor(processor_class: Any, **kwargs: Any) -> object:
        processor_args.append((processor_class, kwargs))
        return object()

    processing_info = SimpleNamespace(
        ctx=SimpleNamespace(get_hf_processor=get_processor),
    )
    get_hf_processor = vars(FakeProcessingInfo)["get_hf_processor"]

    get_hf_processor(processing_info, use_fast=True, min_pixels=128)

    assert processor_args == [
        (
            compat._HunYuanVLProcessorCompat,
            {"min_pixels": 128, "backend": "pil"},
        )
    ]

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

    processor = SimpleNamespace(
        info=SimpleNamespace(
            get_hf_processor=lambda **_kwargs: hf_processor,
            ctx=SimpleNamespace(call_hf_processor=call_hf_processor),
        )
    )
    call_processor = vars(FakeMultiModalProcessor)["_call_hf_processor"]

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
    assert calls[0][1]["text"] == ("before<image_start><image><image_end>after")


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


def test_main_prompt_update_preserves_wrappers_and_embedding_mask():
    class FakeMultiModalProcessor:
        pass

    hunyuan_vision = SimpleNamespace(
        HunYuanVLMultiModalProcessor=FakeMultiModalProcessor,
    )
    compat._patch_prompt_updates(hunyuan_vision)

    hf_processor = SimpleNamespace(
        image_token_id=11,
        image_start_token_id=12,
        image_end_token_id=13,
    )
    image_processor = SimpleNamespace(merge_size=2)
    info = SimpleNamespace(
        get_hf_processor=lambda **_kwargs: hf_processor,
        get_image_processor=lambda **_kwargs: image_processor,
    )
    processor = SimpleNamespace(info=info)
    out_mm_kwargs = {
        "image": [
            {
                "image_grid_thw": SimpleNamespace(
                    data=torch.tensor([1, 4, 6]),
                )
            }
        ]
    }

    get_prompt_updates = vars(FakeMultiModalProcessor)["_get_prompt_updates"]
    updates = get_prompt_updates(processor, None, {}, out_mm_kwargs)
    details = updates[0].replacement(0)

    assert details.full == [12] + [11] * 10 + [13]
    assert details.is_embed is not None
    assert details.is_embed(None, details.full).tolist() == [False] + [True] * 10 + [False]
