#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for CANN MegaMoe helper functions in moe_comm_method.

The helpers exercised here are pure functions (no NPU / collective state)
so they run on any host. They cover the four "fast death" gates that
keep the FUSED_MC2 mode-2 path from misrouting work to MegaMoe:

* ``_pick_mega_moe_bias`` — resolves which per-expert bias to forward
  (W4A8 scale_bias vs. W8A8 placeholder vs. raw bias).
* ``_get_cann_mega_moe_quant_settings`` — maps QuantType enums to CANN
  ACL dtype + quant-mode tuples.
* ``_normalize_cann_activation`` — whitelists silu/swiglu and raises
  on anything else.
* ``_cann_megamoe_supported_by_config`` — the hidden_size / quant gate
  in select_moe_comm_method's A2/A3 mode-2 branches.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ascend_forward_context import _cann_megamoe_supported_by_config
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    _CANN_ACL_INT8,
    _CANN_MEGA_MOE_QUANT_MODE_INT8,
    _CANN_MEGA_MOE_QUANT_MODE_MX,
    _CANN_TORCH_FLOAT8_E4M3FN,
    _get_cann_mega_moe_quant_settings,
    _normalize_cann_activation,
    _pick_mega_moe_bias,
)
from vllm_ascend.quantization.quant_type import QuantType

# ---------------------------------------------------------------------------
# _pick_mega_moe_bias
# ---------------------------------------------------------------------------


class TestPickMegaMoeBias:
    """Cover the bias resolution helper that bridges W4A8 / W8A8 inputs."""

    def test_real_scale_bias_wins(self):
        """A non-empty per-expert scale_bias is forwarded as-is (W4A8 path)."""
        scale_bias = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        fallback = [torch.tensor([99.0])]

        out = _pick_mega_moe_bias(scale_bias, fallback)

        assert out is not None
        assert len(out) == 2
        assert torch.equal(out[0], scale_bias[0])
        assert torch.equal(out[1], scale_bias[1])

    def test_empty_scale_bias_placeholder_falls_through(self):
        """W8A8 mode-1 stuffs in `[empty_tensor]` — must be skipped, not forwarded."""
        scale_bias = [torch.empty(0, dtype=torch.float32)]
        out = _pick_mega_moe_bias(scale_bias, fallback_bias=None)
        assert out is None

    def test_all_empty_tensors_filter_to_none(self):
        """Mixed list of all-empty placeholders also resolves to None."""
        scale_bias = [
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
        ]
        out = _pick_mega_moe_bias(scale_bias, fallback_bias=None)
        assert out is None

    def test_partial_empty_keeps_list(self):
        """If at least one tensor has data the whole list is returned.

        We do NOT silently drop empty entries — that would break the
        expected per-expert ordering required by the CANN op.
        """
        scale_bias = [
            torch.empty(0, dtype=torch.float32),
            torch.tensor([0.5, 0.5]),
        ]
        out = _pick_mega_moe_bias(scale_bias, fallback_bias=None)
        assert out is not None
        assert len(out) == 2
        assert out[0].numel() == 0
        assert out[1].numel() == 2

    def test_scale_bias_single_tensor_wrapped(self):
        """A bare tensor is normalized to a one-element list."""
        scale_bias = torch.tensor([1.0, 2.0])
        out = _pick_mega_moe_bias(scale_bias, fallback_bias=None)
        assert isinstance(out, list)
        assert len(out) == 1
        assert torch.equal(out[0], scale_bias)

    def test_fallback_bias_used_when_scale_bias_none(self):
        """W8A8 mode-2 leaves scale_bias=None — fall back to the model bias."""
        fallback = [torch.tensor([7.0])]
        out = _pick_mega_moe_bias(scale_bias=None, fallback_bias=fallback)
        assert out is not None
        assert len(out) == 1
        assert torch.equal(out[0], fallback[0])

    def test_both_none(self):
        """No bias of any kind → None (typical bf16 weights-only case)."""
        assert _pick_mega_moe_bias(None, None) is None


# ---------------------------------------------------------------------------
# _get_cann_mega_moe_quant_settings
# ---------------------------------------------------------------------------


class TestGetCANNMegaMoeQuantSettings:
    """The QuantType → (mode, dispatch_dtype) mapping table.

    weight_type is intentionally NOT returned: weight1_type/weight2_type are
    reserved params in the mega_moe doc and are never passed to the op.
    """

    def test_w8a8(self):
        mode, dispatch_dtype = _get_cann_mega_moe_quant_settings(QuantType.W8A8)
        assert mode == _CANN_MEGA_MOE_QUANT_MODE_INT8
        assert dispatch_dtype == _CANN_ACL_INT8

    def test_w4a8(self):
        """W4A8 dispatches int8 across rank; the weight tile is int4 (inferred by the op)."""
        mode, dispatch_dtype = _get_cann_mega_moe_quant_settings(QuantType.W4A8)
        assert mode == _CANN_MEGA_MOE_QUANT_MODE_INT8
        assert dispatch_dtype == _CANN_ACL_INT8

    def test_mxfp8(self):
        mode, dispatch_dtype = _get_cann_mega_moe_quant_settings(QuantType.MXFP8)
        assert mode == _CANN_MEGA_MOE_QUANT_MODE_MX
        assert dispatch_dtype == _CANN_TORCH_FLOAT8_E4M3FN

    def test_w4a8mxfp(self):
        mode, dispatch_dtype = _get_cann_mega_moe_quant_settings(QuantType.W4A8MXFP)
        assert mode == _CANN_MEGA_MOE_QUANT_MODE_MX
        assert dispatch_dtype == _CANN_TORCH_FLOAT8_E4M3FN

    def test_unsupported_raises(self):
        """Unsupported QuantType must fail loud, not silently mismap."""
        with pytest.raises(RuntimeError, match="Unsupported quant type"):
            _get_cann_mega_moe_quant_settings(QuantType.W4A16)


# ---------------------------------------------------------------------------
# _normalize_cann_activation
# ---------------------------------------------------------------------------


class TestNormalizeCANNActivation:
    """Whitelist guard for activations forwarded to CANN MegaMoe."""

    @pytest.mark.parametrize("inp", ["silu", "swiglu", "SiLU", "SWIGLU"])
    def test_whitelisted(self, inp):
        """Both casings of silu/swiglu collapse to 'swiglu' (CANN canonical name)."""
        assert _normalize_cann_activation(inp) == "swiglu"

    def test_none_defaults_to_swiglu(self):
        """When the model does not specify an activation we default to swiglu."""
        assert _normalize_cann_activation(None) == "swiglu"

    def test_enum_like_value_attribute(self):
        """Supports enum-style activation objects with a .value attribute."""
        obj = SimpleNamespace(value="silu")
        assert _normalize_cann_activation(obj) == "swiglu"

    @pytest.mark.parametrize("bad", ["relu", "gelu", "tanh", "geglu"])
    def test_unsupported_raises(self, bad):
        """Anything outside silu/swiglu raises a clear error.

        This is the difference vs. the original implementation, which
        silently transported e.g. ``"relu"`` into the CANN op where it
        produced an opaque kernel error.
        """
        with pytest.raises(ValueError, match="CANN MegaMoe only supports"):
            _normalize_cann_activation(bad)


# ---------------------------------------------------------------------------
# _cann_megamoe_supported_by_config (the select_moe_comm_method gate)
# ---------------------------------------------------------------------------


def _make_vllm_config(hidden_size):
    """Build the minimal stub that ``_cann_megamoe_supported_by_config`` reads."""
    hf_text = SimpleNamespace(hidden_size=hidden_size)
    model_config = MagicMock()
    model_config.hf_text_config = hf_text
    # Force the function down the ``hasattr is True`` branch by stubbing the
    # method too — but the hf_text_config.hidden_size getattr wins first.
    model_config.get_hidden_size = MagicMock(return_value=hidden_size)
    return SimpleNamespace(model_config=model_config)


class TestCANNMegamoeSupportedByConfig:
    """Hidden-size + quant gate used inside select_moe_comm_method."""

    @pytest.mark.parametrize("hidden", [1024, 1536, 4096, 7168, 8192])
    def test_within_bounds_supported_quant(self, hidden):
        """Common production hidden sizes (e.g. DSv4=7168) pass with W8A8."""
        cfg = _make_vllm_config(hidden_size=hidden)
        assert _cann_megamoe_supported_by_config(cfg, "w8a8_dynamic") is True

    @pytest.mark.parametrize("hidden", [512, 896, 1023])
    def test_below_minimum_rejected(self, hidden):
        """Hidden < 1024 (CANN cube tile minimum) → rejected."""
        cfg = _make_vllm_config(hidden_size=hidden)
        assert _cann_megamoe_supported_by_config(cfg, "w8a8_dynamic") is False

    @pytest.mark.parametrize("hidden", [8704, 9216, 12288])
    def test_above_maximum_rejected(self, hidden):
        """Hidden > 8192 (CANN cube tile maximum) → rejected."""
        cfg = _make_vllm_config(hidden_size=hidden)
        assert _cann_megamoe_supported_by_config(cfg, "w8a8_dynamic") is False

    @pytest.mark.parametrize("hidden", [1025, 1500, 2049, 7000])
    def test_not_multiple_of_512_rejected(self, hidden):
        """Hidden must be a multiple of the cube K-step (512)."""
        cfg = _make_vllm_config(hidden_size=hidden)
        assert _cann_megamoe_supported_by_config(cfg, "w8a8_dynamic") is False

    def test_missing_hidden_size_rejected(self):
        """Models without a hidden_size attribute are conservatively rejected."""
        hf_text = SimpleNamespace()
        model_config = MagicMock(spec=[])
        model_config.hf_text_config = hf_text
        cfg = SimpleNamespace(model_config=model_config)
        # Belt-and-suspenders: also ensure no get_hidden_size on the config.
        assert _cann_megamoe_supported_by_config(cfg, "w8a8_dynamic") is False

    def test_unknown_quant_name_rejected(self):
        """A valid hidden size cannot rescue an unsupported quant name."""
        cfg = _make_vllm_config(hidden_size=4096)
        assert _cann_megamoe_supported_by_config(cfg, "fp16") is False

    @pytest.mark.parametrize("name", ["w8a8", "w4a8", "w8a8_dynamic", "w4a8_dynamic"])
    def test_supported_quant_names(self, name):
        """The full set the helper claims to recognize."""
        cfg = _make_vllm_config(hidden_size=4096)
        assert _cann_megamoe_supported_by_config(cfg, name) is True

    def test_none_quant_passes_through(self):
        """quant_type=None means "let the dispatch layer decide" — return True."""
        cfg = _make_vllm_config(hidden_size=4096)
        assert _cann_megamoe_supported_by_config(cfg, None) is True

    def test_enum_like_quant_type(self):
        """Enum-style quant_type (with .name) is accepted via lowercase match."""
        cfg = _make_vllm_config(hidden_size=4096)
        quant = SimpleNamespace(name="W8A8")
        assert _cann_megamoe_supported_by_config(cfg, quant) is True
