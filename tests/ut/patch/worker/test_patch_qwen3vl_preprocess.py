# SPDX-License-Identifier: Apache-2.0
"""Equivalence guards for the device-side Qwen3-VL rescale/normalize patch.

The device path only replaces work the HF processor used to do on the host, so
these tests pin it against a reference implementation of that host math. The
packed-patch layout is the fragile part: the HF image/video processors flatten
each patch as ``[channel][temporal][ph][pw]``, and reading it back with the
channel axis in the wrong place silently normalizes the wrong channels. That
mistake is invisible for checkpoints whose ``image_mean``/``image_std`` are
equal across channels, so the tests below deliberately use per-channel values.
"""

import importlib
import inspect

import pytest
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

from vllm_ascend.patch.worker.patch_qwen3vl import _PREPROCESS_ATTR, _apply_rescale_normalize, _VLPreprocessConfig

CHANNEL = 3
TEMPORAL_PATCH_SIZE = 2
PATCH_SIZE = 4
GRID_H = GRID_W = 2

# Per-channel values (the transformers class defaults) rather than the equal
# values some checkpoints ship, so a channel mixup shows up as a mismatch.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
RESCALE_FACTOR = 1 / 255.0


class _FakeVisual:
    dtype = torch.float32


class _FakeModel:
    """Stands in for the model the patch binds ``_apply_rescale_normalize`` to."""

    def __init__(self, cfg):
        self.visual = _FakeVisual()
        setattr(self, _PREPROCESS_ATTR, cfg)  # noqa: B010


def _make_config(do_rescale=True, do_normalize=True):
    return _VLPreprocessConfig(
        channel=CHANNEL,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        do_rescale=do_rescale,
        rescale_factor=RESCALE_FACTOR,
        do_normalize=do_normalize,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )


def _flatten_patches(image: torch.Tensor) -> torch.Tensor:
    """Pack ``(C, T, H, W)`` the way the HF processors emit ``pixel_values``."""
    packed = image.reshape(CHANNEL, TEMPORAL_PATCH_SIZE, GRID_H, PATCH_SIZE, GRID_W, PATCH_SIZE)
    packed = packed.permute(2, 4, 0, 1, 3, 5)
    return packed.reshape(GRID_H * GRID_W, CHANNEL * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)


def _host_reference(image: torch.Tensor, do_rescale: bool, do_normalize: bool) -> torch.Tensor:
    """What the HF processor would have produced before this patch."""
    out = image
    if do_rescale:
        out = out * RESCALE_FACTOR
    if do_normalize:
        mean = torch.tensor(IMAGE_MEAN).view(CHANNEL, 1, 1, 1)
        std = torch.tensor(IMAGE_STD).view(CHANNEL, 1, 1, 1)
        out = (out - mean) / std
    return _flatten_patches(out)


def _random_image() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    shape = (CHANNEL, TEMPORAL_PATCH_SIZE, GRID_H * PATCH_SIZE, GRID_W * PATCH_SIZE)
    return torch.randint(0, 256, shape, generator=generator).float()


def test_device_preprocess_matches_host_fused_rescale_normalize():
    image = _random_image()
    model = _FakeModel(_make_config())

    actual = _apply_rescale_normalize(model, _flatten_patches(image))

    torch.testing.assert_close(actual, _host_reference(image, True, True), rtol=1e-5, atol=1e-5)


def test_device_preprocess_matches_host_normalize_only():
    image = _random_image()
    model = _FakeModel(_make_config(do_rescale=False))

    actual = _apply_rescale_normalize(model, _flatten_patches(image))

    torch.testing.assert_close(actual, _host_reference(image, False, True), rtol=1e-5, atol=1e-5)


def test_device_preprocess_matches_host_rescale_only():
    image = _random_image()
    model = _FakeModel(_make_config(do_normalize=False))

    actual = _apply_rescale_normalize(model, _flatten_patches(image))

    torch.testing.assert_close(actual, _host_reference(image, True, False), rtol=1e-5, atol=1e-5)


def test_device_preprocess_is_a_noop_when_host_still_preprocesses():
    """Both flags off means the platform patch did not run and HF already did the work."""
    image = _random_image()
    flat = _flatten_patches(image)
    model = _FakeModel(_make_config(do_rescale=False, do_normalize=False))

    actual = _apply_rescale_normalize(model, flat)

    torch.testing.assert_close(actual, flat)


def test_device_preprocess_preserves_packed_patch_shape():
    image = _random_image()
    flat = _flatten_patches(image)
    model = _FakeModel(_make_config())

    assert _apply_rescale_normalize(model, flat).shape == flat.shape


def _hooked_model_classes():
    classes = [Qwen3VLForConditionalGeneration]
    for module_name, class_names in (
        ("vllm.model_executor.models.qwen3_vl_moe", ("Qwen3VLMoeForConditionalGeneration",)),
        (
            "vllm.model_executor.models.qwen3_5",
            ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"),
        ),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        classes.extend(cls for cls in (getattr(module, name, None) for name in class_names) if cls is not None)
    return classes


@pytest.mark.parametrize("model_cls", _hooked_model_classes(), ids=lambda cls: cls.__name__)
def test_init_hook_keeps_vllm_config_visible_in_signature(model_cls):
    """vLLM picks the construction path by looking for these two parameter names.

    A wrapper that only takes ``*args, **kwargs`` hides them, and vLLM then
    falls back to its old-style path and constructs the model without
    ``vllm_config``.
    """
    params = inspect.signature(model_cls.__init__).parameters

    assert "vllm_config" in params
    assert "prefix" in params
