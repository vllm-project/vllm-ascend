# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Ascend Qwen3.5-VL NPU preprocessing offload.

Covers the review-critical configurations:
- fallback to the stock CPU pipeline when mm_device_do_normalize is disabled
  or non-default processor settings (do_resize/resample) are requested;
- encoder graph path: raw payloads are sliced BEFORE preprocessing so each
  item is preprocessed exactly once, and nested re-selection (DP re-packing)
  never normalizes twice (images and videos).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

try:
    from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm  # noqa: F401

    from vllm_ascend.models.qwen3_5_vl import (
        _DEVICE_PREPROCESS_DONE,
        AscendQwen2VLImageProcessor,
        AscendQwen3_5VLProcessor,
        _AscendVLPreprocessMixin,
    )

    _MM_INPUT_NORM_AVAILABLE = True
except ImportError:
    # vllm release wheels do not ship mm_input_norm and the model module
    # imports it unconditionally; the release line is covered by the
    # releases/v0.30.0 branch instead.
    _AscendVLPreprocessMixin = object
    _MM_INPUT_NORM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MM_INPUT_NORM_AVAILABLE,
    reason="requires vllm.model_executor.layers.fusion.mm_input_norm (vllm main)",
)

C, PS, TPS, MS = 3, 14, 2, 2
PATCH_DIM = C * TPS * PS * PS  # 1176

# With mm_device_do_normalize enabled vLLM merges do_rescale=False /
# do_normalize=False into the processor kwargs, so the raw-pixel path is
# exercised with those flags.
RAW_KWARGS = {"do_rescale": False, "do_normalize": False}


# ---------------------------------------------------------------------------
# Image processor: raw path vs stock fallback
# ---------------------------------------------------------------------------


def _make_image_processor(**overrides):
    kwargs = dict(
        do_resize=True,
        size={"shortest_edge": 4 * 28 * 28, "longest_edge": 16384 * 28 * 28},
        resample=Image.Resampling.BICUBIC,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        patch_size=PS,
        temporal_patch_size=TPS,
        merge_size=MS,
    )
    kwargs.update(overrides)
    return AscendQwen2VLImageProcessor(**kwargs)


def _pil_image(h, w, seed=0):
    torch.manual_seed(seed)
    arr = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8).numpy()
    return Image.fromarray(arr)


def test_raw_preprocess_emits_flat_payload():
    """Default resize settings + device normalize (do_rescale/do_normalize
    disabled): flat raw values + original (H, W) + smart_resize grid."""
    proc = _make_image_processor()
    out = proc(images=[_pil_image(56, 56)], return_tensors="pt", **RAW_KWARGS)
    assert "image_hw" in out
    assert out["pixel_values"].ndim == 1
    assert out["pixel_values"].numel() == C * 56 * 56
    assert float(out["pixel_values"].max()) <= 255.0
    torch.testing.assert_close(out["image_hw"], torch.tensor([[56, 56]]))
    torch.testing.assert_close(out["image_grid_thw"], torch.tensor([[1, 4, 4]]))


def test_do_resize_false_keeps_stock_grid():
    """do_resize=False must keep the stock grid (review example: 224x224 with
    max_pixels=56*56 keeps [1, 16, 16] instead of shrinking to [1, 4, 4])."""
    proc = _make_image_processor(size={"shortest_edge": 56 * 56, "longest_edge": 56 * 56})
    out = proc(
        images=[_pil_image(224, 224)],
        return_tensors="pt",
        do_resize=False,
        **RAW_KWARGS,
    )
    # Stock CPU pipeline: no raw marker, patchified 2-D output.
    assert "image_hw" not in out
    assert out["pixel_values"].ndim == 2
    assert out["pixel_values"].shape == (16 * 16, PATCH_DIM)
    torch.testing.assert_close(out["image_grid_thw"], torch.tensor([[1, 16, 16]]))


def test_non_bicubic_resample_falls_back_to_stock():
    """A non-bicubic resample filter must keep the stock CPU pipeline."""
    proc = _make_image_processor()
    out = proc(
        images=[_pil_image(56, 56)],
        return_tensors="pt",
        resample=Image.Resampling.BILINEAR,
        **RAW_KWARGS,
    )
    assert "image_hw" not in out
    assert out["pixel_values"].ndim == 2
    assert out["pixel_values"].shape == (16, PATCH_DIM)


def test_device_normalize_disabled_keeps_stock_pipeline():
    """mm_device_do_normalize disabled leaves do_rescale/do_normalize enabled:
    the CPU must run the complete stock pipeline (no raw [0, 255] payload)."""
    proc = _make_image_processor()
    out = proc(images=[_pil_image(56, 56)], return_tensors="pt")
    assert "image_hw" not in out
    assert out["pixel_values"].ndim == 2
    assert out["pixel_values"].shape == (16, PATCH_DIM)
    # Stock output is rescaled+normalized, i.e. far away from [0, 255].
    assert float(out["pixel_values"].abs().max()) < 10.0


def test_float32_input_keeps_stock_pipeline():
    """Floating-point pixels must not be quantized by the raw device path
    (review example: a constant 128.4 image must survive, not round to 128)."""
    proc = _make_image_processor()
    img = np.full((56, 56, 3), 128.4, dtype=np.float32)
    out = proc(images=[img], return_tensors="pt", **RAW_KWARGS)
    assert "image_hw" not in out
    assert out["pixel_values"].shape == (16, PATCH_DIM)
    torch.testing.assert_close(out["pixel_values"], torch.full((16, PATCH_DIM), 128.4))


# ---------------------------------------------------------------------------
# Field config gating
# ---------------------------------------------------------------------------


def test_field_config_follows_emitted_keys(monkeypatch):
    """pixel_values/image_hw overrides must only apply to raw outputs."""
    base = {"pixel_values": object()}
    monkeypatch.setattr(
        Qwen3VLMultiModalProcessor,
        "_get_mm_fields_config",
        lambda self, hf_inputs, hf_processor_mm_kwargs: base,
    )
    proc = object.__new__(AscendQwen3_5VLProcessor)

    # Stock output (no image_hw): base mapping returned untouched.
    stock_cfg = proc._get_mm_fields_config({"pixel_values": torch.zeros(16, PATCH_DIM)}, {})
    assert stock_cfg["pixel_values"] is base["pixel_values"]
    assert "image_hw" not in stock_cfg

    # Raw output: pixel_values is remapped and image_hw is batched on CPU.
    proc.info = SimpleNamespace(get_hf_config=lambda: SimpleNamespace(vision_config=SimpleNamespace(in_channels=C)))
    raw_cfg = proc._get_mm_fields_config({"image_hw": torch.tensor([[56, 56], [28, 56]])}, {})
    assert raw_cfg["pixel_values"] is not base["pixel_values"]
    assert "image_hw" in raw_cfg


# ---------------------------------------------------------------------------
# Encoder graph path: slice first, preprocess exactly once
# ---------------------------------------------------------------------------


def _affine_norm(x, dtype):
    """Plain-torch stand-in for the fused input norm (same (x / rescale -
    mean) / std semantics, uniform per channel). The tests verify the model's
    orchestration (exactly-once slicing/normalization); the real normalizer
    is upstream code exercised by the e2e tests. Constructing the real
    CustomOp here would also require a vllm config context."""
    return ((x / 255.0 - 0.5) / 0.25).to(dtype)


class _CountingNorm:
    def __init__(self):
        self.inner = _affine_norm
        self.calls = 0

    def __call__(self, x, dtype):
        self.calls += 1
        return self.inner(x, dtype)


class _StubVLModel(_AscendVLPreprocessMixin):
    """CPU stub exposing only the machinery the mixin methods rely on."""

    def __init__(self, device_do_normalize=True):
        self._c, self._ps, self._tps, self._ms = C, PS, TPS, MS
        self.model_config = SimpleNamespace(
            multimodal_config=SimpleNamespace(mm_device_do_normalize=device_do_normalize)
        )
        self.visual = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
        self.use_data_parallel = False
        self.counting_norm = _CountingNorm()
        self.input_norm = self.counting_norm

    def get_input_modality(self, mm_kwargs):
        if "image_grid_thw" in mm_kwargs:
            return "image"
        if "video_grid_thw" in mm_kwargs:
            return "video"
        raise AssertionError("This line should be unreachable.")

    def _get_grid_thw_by_modality(self, mm_kwargs):
        grid_thw = mm_kwargs[f"{self.get_input_modality(mm_kwargs)}_grid_thw"]
        if not isinstance(grid_thw, list):
            grid_thw = grid_thw.tolist()
        return grid_thw


def _raw_image_mm_kwargs():
    """Two raw images: 56x56 (grid [1,4,4]) and 28x56 (grid [1,2,4])."""
    torch.manual_seed(0)
    hw = [(56, 56), (28, 56)]
    flat = torch.randint(0, 256, (sum(C * h * w for h, w in hw),), dtype=torch.uint8)
    return {
        "pixel_values": flat,
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
        "image_hw": torch.tensor(hw),
    }


def _instrument_preprocess(stub):
    calls = []
    orig = stub._resize_and_patchify

    def wrapped(image_input, grid_thw):
        calls.append(image_input["pixel_values"].numel())
        return orig(image_input, grid_thw)

    stub._resize_and_patchify = wrapped
    return calls


def test_select_slices_raw_payload_before_preprocess():
    """Each sub-batch selection must preprocess only its own items, not the
    whole batch (two images split into two sub-batches -> two resizes)."""
    stub = _StubVLModel()
    seen = _instrument_preprocess(stub)
    mm_kwargs = _raw_image_mm_kwargs()

    first = stub.select_encoder_cudagraph_items(mm_kwargs, [0])
    assert seen == [C * 56 * 56]  # only image 0
    assert first["pixel_values"].shape == (4 * 4, PATCH_DIM)
    assert first[_DEVICE_PREPROCESS_DONE]

    second = stub.select_encoder_cudagraph_items(mm_kwargs, [1])
    assert seen == [C * 56 * 56, C * 28 * 56]  # only image 1
    assert second["pixel_values"].shape == (2 * 4, PATCH_DIM)


def test_select_raw_empty_indices():
    stub = _StubVLModel()
    seen = _instrument_preprocess(stub)
    out = stub.select_encoder_cudagraph_items(_raw_image_mm_kwargs(), [])
    assert seen == []
    assert out["pixel_values"].shape == (0, PATCH_DIM)
    assert out["image_grid_thw"] == []


def test_select_repack_never_renormalizes():
    """The DP path re-selects from an already-selected dict: the raw branch is
    not retried and normalization is applied exactly once."""
    stub = _StubVLModel()
    mm_kwargs = _raw_image_mm_kwargs()

    selected = stub.select_encoder_cudagraph_items(mm_kwargs, [0, 1])
    assert stub.counting_norm.calls == 1

    # Repack the already-processed sub-batch (image-format dict, marker set).
    repacked = stub.select_encoder_cudagraph_items(selected, [1])
    assert stub.counting_norm.calls == 1  # no second normalization
    torch.testing.assert_close(
        repacked["pixel_values"],
        selected["pixel_values"][16:],  # image 1 occupies patches 16..23
    )


def test_select_stock_images_normalized_exactly_once():
    """Stock patchified inputs (e.g. do_resize=False fallback) arrive
    unnormalized when mm_device_do_normalize is enabled; the selector
    normalizes once and re-packing does not normalize again."""
    torch.manual_seed(0)
    stub = _StubVLModel(device_do_normalize=True)
    raw = torch.randint(0, 256, (24, PATCH_DIM), dtype=torch.float32)
    mm_kwargs = {
        "pixel_values": raw,
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
    }
    selected = stub.select_encoder_cudagraph_items(mm_kwargs, [0])
    assert stub.counting_norm.calls == 1
    expected = stub.counting_norm.inner(raw[:16], torch.float32)
    torch.testing.assert_close(selected["pixel_values"], expected)
    assert selected[_DEVICE_PREPROCESS_DONE]

    repacked = stub.select_encoder_cudagraph_items(selected, [0])
    assert stub.counting_norm.calls == 1
    torch.testing.assert_close(repacked["pixel_values"], selected["pixel_values"])


def test_select_stock_images_flag_off_skips_normalization():
    """mm_device_do_normalize disabled: CPU already normalized; the selector
    must pass the patches through untouched."""
    torch.manual_seed(0)
    stub = _StubVLModel(device_do_normalize=False)
    raw = torch.randn(24, PATCH_DIM)
    mm_kwargs = {
        "pixel_values": raw,
        "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
    }
    selected = stub.select_encoder_cudagraph_items(mm_kwargs, [1])
    assert stub.counting_norm.calls == 0
    torch.testing.assert_close(selected["pixel_values"], raw[16:])


def test_select_videos_normalized_exactly_once():
    """Video patches are normalized exactly once across selection and
    re-packing (graph replay and eager fallback reuse the selected dict)."""
    torch.manual_seed(0)
    stub = _StubVLModel(device_do_normalize=True)
    raw = torch.randint(0, 256, (24, PATCH_DIM), dtype=torch.float32)
    mm_kwargs = {
        "pixel_values_videos": raw,
        "video_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
    }
    selected = stub.select_encoder_cudagraph_items(mm_kwargs, [0, 1])
    assert stub.counting_norm.calls == 1
    expected = stub.counting_norm.inner(raw, torch.float32)
    torch.testing.assert_close(selected["pixel_values_videos"], expected)

    # Nested selection (replay buffers / eager fallback / DP re-pack).
    repacked = stub.select_encoder_cudagraph_items(selected, [1])
    assert stub.counting_norm.calls == 1
    torch.testing.assert_close(repacked["pixel_values_videos"], selected["pixel_values_videos"][16:])


def test_select_videos_flag_off_skips_normalization():
    torch.manual_seed(0)
    stub = _StubVLModel(device_do_normalize=False)
    raw = torch.randn(24, PATCH_DIM)
    mm_kwargs = {
        "pixel_values_videos": raw,
        "video_grid_thw": torch.tensor([[1, 2, 4]]),
    }
    selected = stub.select_encoder_cudagraph_items(mm_kwargs, [0])
    assert stub.counting_norm.calls == 0
    torch.testing.assert_close(selected["pixel_values_videos"], raw[: 2 * 4])


# ---------------------------------------------------------------------------
# Eager path: normalize exactly once in every configuration
# ---------------------------------------------------------------------------


def test_process_image_input_raw_path_normalizes_once():
    stub = _StubVLModel()
    stub._run_visual = lambda pixel_values, grid_thw: pixel_values
    seen = _instrument_preprocess(stub)
    out = stub._process_image_input(
        {
            "type": "pixel_values",
            "pixel_values": torch.randint(0, 256, (C * 56 * 56,), dtype=torch.uint8),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "image_hw": torch.tensor([[56, 56]]),
        }
    )
    assert seen == [C * 56 * 56]
    assert stub.counting_norm.calls == 1
    assert out.shape == (16, PATCH_DIM)


def test_process_image_input_stock_path_normalizes_once():
    """Stock patches (device normalize on -> CPU skipped rescale/normalize):
    the model-side normalizer must run exactly once."""
    stub = _StubVLModel(device_do_normalize=True)
    stub._run_visual = lambda pixel_values, grid_thw: pixel_values
    raw = torch.randint(0, 256, (16, PATCH_DIM), dtype=torch.float32)
    out = stub._process_image_input(
        {
            "type": "pixel_values",
            "pixel_values": raw,
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "image_hw": None,
        }
    )
    assert stub.counting_norm.calls == 1
    expected = stub.counting_norm.inner(raw, torch.float32)
    torch.testing.assert_close(out, expected)


def test_process_video_input_normalizes_once():
    stub = _StubVLModel()
    stub._run_visual = lambda pixel_values, grid_thw: pixel_values
    raw = torch.randint(0, 256, (8, PATCH_DIM), dtype=torch.float32)
    stub._process_video_input(
        {
            "type": "pixel_values_videos",
            "pixel_values_videos": raw,
            "video_grid_thw": torch.tensor([[1, 2, 4]]),
        }
    )
    assert stub.counting_norm.calls == 1


def test_resize_and_patchify_empty_input():
    stub = _StubVLModel()
    out = stub._resize_and_patchify(
        {
            "pixel_values": torch.zeros(0, dtype=torch.uint8),
            "image_hw": torch.zeros((0, 2), dtype=torch.long),
        },
        torch.zeros((0, 3), dtype=torch.long),
    )
    assert out.shape == (0, PATCH_DIM)
