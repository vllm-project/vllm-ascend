# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType

import vllm_ascend.patch.hunyuan_vl_processor_compat as compat


def test_tag_does_not_install_hunyuan_compat(monkeypatch):
    monkeypatch.setattr(compat, "vllm_version_is", lambda version: version == "0.23.0")

    assert not compat.install_hunyuan_vl_processor_compat()


def test_main_installs_hunyuan_compat(monkeypatch):
    from vllm_ascend.patch import transformers_compat

    monkeypatch.setattr(compat, "vllm_version_is", lambda _version: False)

    transformers = ModuleType("transformers")
    transformers.__path__ = []  # type: ignore[attr-defined]
    transformers_models = ModuleType("transformers.models")
    transformers_models.__path__ = []  # type: ignore[attr-defined]
    transformers.models = transformers_models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", transformers_models)

    parent_name = "transformers.models.hunyuan_vl"
    image_module_name = f"{parent_name}.image_processing_hunyuan_vl"
    parent_module = ModuleType(parent_name)
    parent_module.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, parent_name, parent_module)
    monkeypatch.setitem(sys.modules, image_module_name, ModuleType(image_module_name))

    assert compat.install_hunyuan_vl_processor_compat()

    from transformers import HunYuanVLProcessor
    from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import (
        HunYuanVLImageProcessor,
        smart_resize,
    )

    assert HunYuanVLProcessor is transformers_compat.HunYuanVLProcessor
    assert HunYuanVLImageProcessor is transformers_compat.HunYuanVLImageProcessor
    assert smart_resize is transformers_compat.smart_resize


def test_compat_processor_preserves_0706_call_protocol(monkeypatch):
    from transformers.processing_utils import ProcessorMixin

    from vllm_ascend.patch.transformers_compat import HunYuanVLProcessor

    captured_kwargs = {}

    def fake_from_pretrained(cls, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return cls.__new__(cls)

    monkeypatch.setattr(
        ProcessorMixin,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    HunYuanVLProcessor.from_pretrained("model", backend="pil")

    assert "backend" not in captured_kwargs
    assert captured_kwargs["use_fast"] is True


def test_compat_processor_disables_main_image_wrapping(monkeypatch):
    from transformers.processing_utils import ProcessorMixin

    from vllm_ascend.patch.transformers_compat import HunYuanVLProcessor

    class FakeTokenizer:
        vocab_size = 120121

        @staticmethod
        def convert_ids_to_tokens(token_id):
            return f"token-{token_id}"

    monkeypatch.setattr(ProcessorMixin, "__init__", lambda *_args, **_kwargs: None)

    processor = HunYuanVLProcessor(
        image_processor=object(),
        tokenizer=FakeTokenizer(),
    )

    assert processor.image_start_token == ""
    assert processor.image_end_token == ""
