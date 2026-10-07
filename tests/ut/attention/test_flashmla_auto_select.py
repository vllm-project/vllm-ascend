from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.ops  # noqa: F401  # Register ops before importing attention/device modules.
from vllm_ascend.attention import flashmla, mla_v1
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.hardware_profile import HardwareCapability


def test_flashmla_api_detection_distinguishes_absence_from_broken_import(monkeypatch):
    monkeypatch.setattr(flashmla, "_has_flashmla_module", lambda: True)

    def missing_package(_name):
        raise ModuleNotFoundError("No module named 'cann_ops_transformer'", name="cann_ops_transformer")

    monkeypatch.setattr(flashmla, "import_module", missing_package)
    assert flashmla.get_flashmla_ops() is None

    monkeypatch.setattr(flashmla, "import_module", lambda _name: SimpleNamespace(flash_mla_with_kvcache=lambda: None))
    assert flashmla.get_flashmla_ops() is None

    def attention_op():
        return None

    def metadata_op():
        return None

    monkeypatch.setattr(
        flashmla,
        "import_module",
        lambda _name: SimpleNamespace(
            flash_mla_with_kvcache=attention_op,
            flash_mla_with_kvcache_metadata=metadata_op,
        ),
    )
    assert flashmla.get_flashmla_ops() == (attention_op, metadata_op)

    def broken_dependency(_name):
        raise ModuleNotFoundError("No module named 'installed_package_dependency'", name="installed_package_dependency")

    monkeypatch.setattr(flashmla, "import_module", broken_dependency)
    with pytest.raises(ModuleNotFoundError, match="installed_package_dependency"):
        flashmla.get_flashmla_ops()


def test_flashmla_module_precheck_skips_eager_cann_import(monkeypatch, tmp_path):
    package = tmp_path / "cann_ops_transformer"
    package.mkdir()
    monkeypatch.setattr(
        flashmla,
        "find_spec",
        lambda _name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    monkeypatch.setattr(
        flashmla,
        "import_module",
        lambda _name: pytest.fail("absent FlashMLA must not import the CANN ops package"),
    )
    assert not flashmla._has_flashmla_module()
    assert flashmla.get_flashmla_ops() is None

    module = package / "ops/attention/flash_mla_with_kvcache"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    assert flashmla._has_flashmla_module()


def test_flashmla_selection_requires_operator_and_supported_mla_config(monkeypatch):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    impl = SimpleNamespace(
        vllm_config=config,
        num_heads=8,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=False,
    )
    monkeypatch.setattr(
        mla_v1,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda capability: capability is HardwareCapability.MLA_FLASH),
    )
    monkeypatch.setattr(mla_v1, "supports_component_major_mla_pd", lambda _config: True)
    monkeypatch.setattr(mla_v1, "enable_sfa", lambda _config: False)
    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (lambda: None, lambda: None))

    assert AscendMLAImpl._can_use_flashmla(impl)

    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: None)
    assert not AscendMLAImpl._can_use_flashmla(impl)

    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (lambda: None, lambda: None))
    config.use_v2_model_runner = False
    assert not AscendMLAImpl._can_use_flashmla(impl)

    config.use_v2_model_runner = True
    impl.num_heads = 48
    assert not AscendMLAImpl._can_use_flashmla(impl)

    impl.num_heads = 8
    config.parallel_config.decode_context_parallel_size = 2
    assert not AscendMLAImpl._can_use_flashmla(impl)
