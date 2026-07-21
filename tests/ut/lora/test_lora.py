from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.lora.layers.base import BaseLayerWithLoRA

from vllm_ascend.lora.fused_moe import AscendFusedMoEWithLoRA
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.patch.worker.patch_lora_vlm_prefix import (
    _detect_prefix,
    _enable_wrapped_language_model_expand_slice,
)


def test_detects_vlm_wrapper_prefix() -> None:
    lora_keys = ["model.layers.0.mlp.down_proj"]
    model_modules = ["language_model.model.layers.0.mlp.down_proj"]

    assert _detect_prefix(lora_keys, model_modules) == "language_model."


def test_aligned_lora_modules_need_no_prefix() -> None:
    lora_keys = ["model.layers.0.mlp.down_proj"]

    assert _detect_prefix(lora_keys, lora_keys) == ""


@pytest.mark.parametrize("shared_experts", [None, object()])
def test_shared_experts_select_compatible_expand_slice(shared_experts) -> None:
    base_layer = SimpleNamespace(
        _shared_experts=shared_experts,
        set_lora_context=Mock(),
    )
    wrapper = SimpleNamespace(
        base_layer=base_layer,
        _build_lora_context=Mock(return_value="context"),
    )
    punica_wrapper = Mock()

    with patch.object(BaseLayerWithLoRA, "set_mapping"):
        AscendFusedMoEWithLoRA.set_mapping(wrapper, punica_wrapper)

    if shared_experts is None:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_not_called()
    else:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_called_once_with()
    base_layer.set_lora_context.assert_called_once_with("context")


@pytest.mark.parametrize("language_prefix", ["", "language_model"])
def test_wrapped_language_model_selects_compatible_expand_slice(language_prefix: str) -> None:
    punica_wrapper = Mock()
    manager = SimpleNamespace(
        supports_mm=True,
        mm_mapping=SimpleNamespace(language_model=[language_prefix]),
        punica_wrapper_mapping={language_prefix: punica_wrapper},
    )

    _enable_wrapped_language_model_expand_slice(manager)

    if language_prefix:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_called_once_with()
    else:
        punica_wrapper.enable_compatible_lora_bmm_expand_slice.assert_not_called()


@pytest.mark.parametrize(
    ("force_compatible_path", "rank", "slice_size", "expect_compatible_path"),
    [
        (False, 4, 8, False),
        (False, 16, 8, True),
        (True, 4, 8, True),
    ],
)
def test_expand_slice_path_follows_model_structure_and_tensor_shape(
    force_compatible_path: bool,
    rank: int,
    slice_size: int,
    expect_compatible_path: bool,
) -> None:
    wrapper = SimpleNamespace(
        no_lora=False,
        _force_lora_bmm_expand_slice=force_compatible_path,
        _bmm_expand_slice=Mock(),
        sgmv_expand_slice=Mock(),
        prefill_metadata=("batches", "tokens", "indices"),
    )
    wrapper._requires_bmm_expand_slice = MethodType(PunicaWrapperNPU._requires_bmm_expand_slice, wrapper)
    x = SimpleNamespace(shape=(2, rank))

    PunicaWrapperNPU._expand_slice_prefill(
        wrapper,
        "y",
        x,
        "weights",
        4,
        slice_size,
        True,
    )

    if expect_compatible_path:
        wrapper._bmm_expand_slice.assert_called_once_with("y", x, "weights", 4, slice_size, True)
        wrapper.sgmv_expand_slice.assert_not_called()
    else:
        wrapper._bmm_expand_slice.assert_not_called()
        wrapper.sgmv_expand_slice.assert_called_once_with(
            x,
            "weights",
            "y",
            "batches",
            "tokens",
            "indices",
            4,
            slice_size,
            True,
        )
