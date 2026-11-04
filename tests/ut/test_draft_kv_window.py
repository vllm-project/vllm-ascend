# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#
# This file is a part of the vllm-ascend project.
#
"""CPU-only unit tests for draft KV windowing (``draft_kv_window``).

Covers the parts that need no NPU:

* ``AscendConfig._validate_draft_kv_window`` — config entry validation.
* ``apply_draft_kv_window_to_spec`` — FullAttentionSpec -> SlidingWindowSpec
  conversion for draft attention layers.
* ``_maybe_window_draft_spec`` — the gating hook inside ``get_kv_cache_spec``.
* ``_get_kimi_k3_dspark_mixed_kv_cache_groups`` — explicit rejection of the
  DSpark/Kimi-K3 mixed target+draft grouping when draft layers are windowed.

Shared full/sliding KV allocation is covered in
``tests/ut/worker/test_full_sliding_kv_cache.py``.
"""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch
from vllm.config import DeviceConfig
from vllm.config import VllmConfig as _VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.core.kv_cache_interface import declared_kwarg
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_kimi_k3_dspark_mixed_kv_cache_groups,
)
from vllm_ascend.worker.v2.attn_utils import (
    _maybe_window_draft_spec,
    apply_draft_kv_window_to_spec,
)

_K3_TARGET = "language_model.model.layers.0.self_attn.attn"
_K3_DRAFT = "model.layers.0.self_attn.attn"
_K3_MAMBA = "language_model.model.layers.1.linear_attn"


def _full_spec(**overrides) -> FullAttentionSpec:
    kwargs = {"block_size": 16, "num_kv_heads": 1, "head_size": 16, "dtype": torch.bfloat16}
    kwargs.update(overrides)
    return FullAttentionSpec(**kwargs)


def _make_ascend_config(draft_kv_window) -> AscendConfig:
    return AscendConfig(
        draft_kv_window=draft_kv_window,
        sparse_kv_offload_config=SparseKVOffloadConfig(),
    )


def _make_vllm_config(
    *,
    method: str | None = "eagle3",
    num_speculative_tokens: int = 3,
    dcp_size: int = 1,
) -> _VllmConfig:
    config = _VllmConfig(device_config=DeviceConfig(device="cpu"))
    # Mirror tests.ut.test_ascend_config: derive_and_validate touches
    # model_config, which a bare VllmConfig leaves unset.
    if config.model_config is None:
        config.model_config = SimpleNamespace(
            is_moe=False,
            is_deepseek_mla=False,
            use_mla=False,
            enforce_eager=True,
            architectures=[],
            hf_text_config=SimpleNamespace(),
            get_total_num_kv_heads=lambda: 0,
            get_num_experts=lambda: 0,
            get_hidden_size=lambda: 0,
        )
    config.parallel_config.decode_context_parallel_size = dcp_size
    if method is not None:
        config.speculative_config = SimpleNamespace(
            method=method,
            num_speculative_tokens=num_speculative_tokens,
            use_step3p5_mtp=lambda: False,
            use_multi_module_mtp=lambda: False,
            use_gemma4_mtp=lambda: False,
        )
    return config


class TestDraftKvWindowConfigValidation(TestBase):
    """AscendConfig._validate_draft_kv_window (run through derive_and_validate)."""

    def _derive(self, ascend_config: AscendConfig, vllm_config: _VllmConfig, use_v2: bool = True):
        with (
            patch("vllm_ascend.ascend_config.is_310p", return_value=False),
            patch.object(
                _VllmConfig,
                "use_v2_model_runner",
                new_callable=PropertyMock,
                return_value=use_v2,
            ),
        ):
            return ascend_config.derive_and_validate(vllm_config)

    def test_default_none_is_noop(self):
        # No speculative config at all: the default None must be a no-op.
        config = _make_vllm_config(method=None)
        self._derive(_make_ascend_config(None), config)

    def test_non_positive_window_rejected(self):
        config = _make_vllm_config()
        for window in (0, -128):
            with self.assertRaisesRegex(ValueError, "draft_kv_window must be a positive integer"):
                self._derive(_make_ascend_config(window), config)

    def test_requires_speculative_decoding(self):
        config = _make_vllm_config(method=None)
        with self.assertRaisesRegex(ValueError, "requires speculative decoding"):
            self._derive(_make_ascend_config(4096), config)

    def test_unsupported_methods_rejected(self):
        for method in ("dspark", "dflash", "ngram", "draft_model"):
            config = _make_vllm_config(method=method)
            with self.assertRaisesRegex(ValueError, "only supported for EAGLE/EAGLE3/MTP"):
                self._derive(_make_ascend_config(4096), config)

    def test_multi_module_mtp_rejected(self):
        config = _make_vllm_config(method="mtp")
        config.speculative_config.use_multi_module_mtp = lambda: True
        with self.assertRaisesRegex(ValueError, "use_multi_module_mtp"):
            self._derive(_make_ascend_config(4096), config)

    def test_window_smaller_than_speculative_step_rejected(self):
        config = _make_vllm_config(num_speculative_tokens=8)
        with self.assertRaisesRegex(ValueError, "must be >= num_speculative_tokens"):
            self._derive(_make_ascend_config(4), config)

    def test_window_equal_to_speculative_step_allowed(self):
        config = _make_vllm_config(num_speculative_tokens=4)
        self._derive(_make_ascend_config(4), config)

    def test_v1_runner_rejected(self):
        config = _make_vllm_config()
        with self.assertRaisesRegex(ValueError, "requires the V2 model runner"):
            self._derive(_make_ascend_config(4096), config, use_v2=False)

    def test_dcp_rejected(self):
        config = _make_vllm_config(dcp_size=2)
        with self.assertRaisesRegex(ValueError, "decode context parallelism"):
            self._derive(_make_ascend_config(4096), config)

    def test_happy_path_eagle_and_mtp(self):
        for method in ("eagle", "eagle3", "mtp"):
            config = _make_vllm_config(method=method)
            derived = self._derive(_make_ascend_config(4096), config)
            self.assertEqual(derived.draft_kv_window, 4096)

    def test_310p_rejected(self):
        config = _make_vllm_config()
        with (
            patch("vllm_ascend.ascend_config.is_310p", return_value=True),
            patch.object(_VllmConfig, "use_v2_model_runner", new_callable=PropertyMock, return_value=True),
            self.assertRaisesRegex(ValueError, "310P"),
        ):
            _make_ascend_config(4096).derive_and_validate(config)


class TestApplyDraftKvWindowToSpec(TestBase):
    """FullAttentionSpec -> SlidingWindowSpec conversion for draft layers."""

    def test_full_spec_converted_with_fields_preserved(self):
        spec = _full_spec(**declared_kwarg(FullAttentionSpec, "is_draft", True))
        windowed = apply_draft_kv_window_to_spec(spec, window=4096, extra_retained_tokens=3, layer_name="draft.0")

        self.assertIsInstance(windowed, SlidingWindowSpec)
        self.assertEqual(windowed.sliding_window, 4096)
        self.assertEqual(windowed.block_size, spec.block_size)
        self.assertEqual(windowed.num_kv_heads, spec.num_kv_heads)
        self.assertEqual(windowed.head_size, spec.head_size)
        self.assertEqual(windowed.dtype, spec.dtype)
        self.assertEqual(windowed.page_size_bytes, spec.page_size_bytes)
        # The is_draft tag (vLLM main lane) must survive the conversion so
        # EAGLE group annotation still finds the draft layers.
        if declared_kwarg(SlidingWindowSpec, "is_draft", True):
            self.assertTrue(windowed.is_draft)
        # extra_retained_tokens keeps one speculative step below the window
        # alive for the draft re-prefill (upstream SlidingWindowSpec field).
        if declared_kwarg(SlidingWindowSpec, "extra_retained_tokens", 0):
            self.assertEqual(windowed.extra_retained_tokens, 3)

    def test_existing_sliding_window_draft_is_kept(self):
        spec = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.bfloat16,
            sliding_window=1024,
        )
        self.assertIs(apply_draft_kv_window_to_spec(spec, 4096, 3), spec)

    def test_mla_draft_rejected(self):
        spec = MLAAttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "MLA draft attention"):
            apply_draft_kv_window_to_spec(spec, 4096, 3, layer_name="draft.mla.0")

    def test_non_attention_spec_rejected(self):
        spec = MambaSpec(block_size=16, shapes=((16, 32),), dtypes=(torch.bfloat16,))
        with self.assertRaisesRegex(ValueError, "full-attention"):
            apply_draft_kv_window_to_spec(spec, 4096, 3, layer_name="draft.mamba.0")


class TestMaybeWindowDraftSpec(TestBase):
    """The gating hook inside get_kv_cache_spec."""

    def test_non_draft_layer_untouched(self):
        spec = _full_spec()
        module = SimpleNamespace(is_draft_layer=False)
        out = _maybe_window_draft_spec(spec, module, SimpleNamespace(speculative_config=None), "target.0")
        self.assertIs(out, spec)

    def test_window_none_untouched(self):
        spec = _full_spec()
        module = SimpleNamespace(is_draft_layer=True)
        with patch(
            "vllm_ascend.worker.v2.attn_utils.get_ascend_config",
            return_value=SimpleNamespace(draft_kv_window=None),
        ):
            out = _maybe_window_draft_spec(spec, module, SimpleNamespace(speculative_config=None), "draft.0")
        self.assertIs(out, spec)

    def test_draft_layer_windowed(self):
        spec = _full_spec()
        module = SimpleNamespace(is_draft_layer=True)
        vllm_config = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=3))
        with patch(
            "vllm_ascend.worker.v2.attn_utils.get_ascend_config",
            return_value=SimpleNamespace(draft_kv_window=4096),
        ):
            out = _maybe_window_draft_spec(spec, module, vllm_config, "draft.0")
        self.assertIsInstance(out, SlidingWindowSpec)
        self.assertEqual(out.sliding_window, 4096)


class TestKimiK3DSparkGroupingRejection(TestBase):
    """Windowed draft layers must not silently fall out of the K3/DSpark
    mixed target+draft grouping."""

    def _specs(self, draft_spec) -> dict:
        # One target attention layer, one draft attention layer and one mamba
        # layer, all with identical block/page sizes so the mixed grouping
        # matches when the draft is not windowed.
        return {
            _K3_TARGET: _full_spec(),
            _K3_DRAFT: draft_spec,
            _K3_MAMBA: MambaSpec(block_size=16, shapes=((16, 32),), dtypes=(torch.bfloat16,)),
        }

    def test_full_attention_draft_groups_normally(self):
        groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(self._specs(_full_spec()))
        assert groups is not None
        # One mixed target+draft attention group, one mamba group.
        self.assertEqual(len(groups), 2)
        self.assertIn(_K3_TARGET, groups[0].layer_names)
        self.assertIn(_K3_DRAFT, groups[0].layer_names)

    def test_windowed_draft_rejected(self):
        windowed_draft = SlidingWindowSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.bfloat16,
            sliding_window=4096,
        )
        with self.assertRaisesRegex(ValueError, "draft_kv_window is not supported"):
            _get_kimi_k3_dspark_mixed_kv_cache_groups(self._specs(windowed_draft))
