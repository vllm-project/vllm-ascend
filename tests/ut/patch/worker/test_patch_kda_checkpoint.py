# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, eager_break_during_capture
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend import ascend_config as ascend_config_mod
from vllm_ascend.patch.worker import patch_kda_checkpoint as mod


@pytest.fixture(autouse=True)
def _reset_ascend_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ascend_config_mod, "_ASCEND_CONFIG", None)


def _legacy_modules(forward_fn: Any):
    def _align(backend: str) -> int | None:
        return 16 if backend == "flashkda" else None

    class _BaseMamba:
        def get_prefill_checkpoint_alignment(self) -> int | None:
            return None

    class _LegacyKimiGDN(_BaseMamba):
        def __init__(self, prefix: str, vllm_config: Any) -> None:
            self.prefix, self.vllm_config = prefix, vllm_config
            self.head_dim, self.conv_size, self.local_num_heads = 16, 4, 2
            self.kv_cache = (
                torch.zeros((2, 6, 3), dtype=torch.bfloat16),
                torch.zeros((2, 2, 16, 16), dtype=torch.bfloat16),
            )

        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)

        def get_kv_cache_spec(self, vllm_config: Any):
            return "base-spec"

        _forward = forward_fn

    def _base_kda(q, k, v, raw_g, raw_beta, A_log, g_bias, **kw):
        return q, v

    kda_ckpt = types.ModuleType("vllm.model_executor.layers.mamba.kda_checkpoint")
    kda_ckpt.kda_prefill_checkpoint_alignment = _align  # type: ignore[attr-defined]
    nv_kda = types.ModuleType("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
    nv_kda.chunk_kda_with_fused_gate = _base_kda  # type: ignore[attr-defined]
    kimi_gdn = types.ModuleType("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
    kimi_gdn.KimiGatedDeltaNetAttention = _LegacyKimiGDN  # type: ignore[attr-defined]
    kimi_gdn.chunk_kda_with_fused_gate = nv_kda.chunk_kda_with_fused_gate  # type: ignore[attr-defined]
    return {
        "vllm.model_executor.layers.mamba.kda_checkpoint": kda_ckpt,
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": nv_kda,
        "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn": kimi_gdn,
    }


def _align_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, use_ubatching=False),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        kv_transfer_config=None,
        speculative_config=None,
        num_hash_lookahead_tokens=0,
        additional_config={"kda_internal_prefill_checkpoints": True},
    )


def test_patch_kda_checkpoint_mixed_zero_pad_and_contextvar_reset() -> None:
    should_raise = False

    def _fwd(self: Any, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any):
        if should_raise:
            raise RuntimeError("boom")
        return mods["vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"].chunk_kda_with_fused_gate(
            q=core_attn_out,
            k=core_attn_out,
            v=core_attn_out,
            raw_g=core_attn_out,
            raw_beta=torch.zeros((1, core_attn_out.shape[1], 2), dtype=torch.bfloat16),
            A_log=torch.zeros((2,)),
            g_bias=None,
            initial_state=torch.zeros((g1.numel() - 1, 2, 16, 16)),
            cu_seqlens=g1,
        )

    mods = _legacy_modules(_fwd)
    orig_align = mods["vllm.model_executor.layers.mamba.kda_checkpoint"].kda_prefill_checkpoint_alignment
    with patch("importlib.import_module", side_effect=lambda n: mods[n]):
        mod.apply_patch()
        mod.apply_patch()

    assert mods["vllm.model_executor.layers.mamba.kda_checkpoint"].kda_prefill_checkpoint_alignment is orig_align
    assert orig_align("triton") is None
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        layer_cls = mods["vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"].KimiGatedDeltaNetAttention
        layer = layer_cls("l0", _align_cfg())
        opt_out_cfg = _align_cfg()
        opt_out_cfg.additional_config = {}
        opt_out_layer = layer_cls("l1", opt_out_cfg)
    assert layer.get_prefill_checkpoint_alignment() == 64
    assert layer.get_kv_cache_checkpoint_alignment(_align_cfg()) == 64
    assert opt_out_layer.get_prefill_checkpoint_alignment() is None
    assert opt_out_layer.get_kv_cache_checkpoint_alignment(opt_out_cfg) is None

    cu_mixed = torch.tensor([0, 1, 2, 66, 66, 194], dtype=torch.int32)
    meta = SimpleNamespace(
        num_prefills=2,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=194,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        non_spec_query_start_loc_cpu=cu_mixed,
        checkpoint=SimpleNamespace(
            checkpoint_offsets=torch.tensor([64, 128], dtype=torch.int32),
            state_indices=torch.tensor([5, 9], dtype=torch.int64),
            offsets=[64, 128],
            query_lens=[64, 128],
        ),
    )
    calls: list[dict[str, Any]] = []

    def _record_kda(**kw: Any) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(kw)
        return kw["q"], torch.ones((5, 2, 16, 16))

    mixed_qkv = torch.zeros((194, 32), dtype=torch.bfloat16)
    with (
        patch("vllm.forward_context.get_forward_context", return_value=SimpleNamespace(attn_metadata={"l0": meta})),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(mod, "ascend_chunk_kda_with_fused_gate", side_effect=_record_kda),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export,
    ):
        layer._forward(mixed_qkv, cu_mixed, None, None, mixed_qkv.view(1, 194, 2, 16))
        assert calls[0]["checkpoint_offsets"].tolist() == [0, 0, 64, 0, 128]
        expected_indices = [NULL_BLOCK_ID, NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 9]
        assert mock_export.call_args.args[1].state_indices.tolist() == expected_indices
        should_raise = True
        with pytest.raises(RuntimeError, match="boom"):
            layer._forward(mixed_qkv, cu_mixed, None, None, mixed_qkv.view(1, 194, 2, 16))
        kda_fn = mods["vllm.models.kimi_k3.nvidia.ops.third_party.kda"].chunk_kda_with_fused_gate
        assert getattr(kda_fn, mod._KDA_CKPT_HOOK_VAR_ATTR).get() is None


def test_patch_kda_checkpoint_keeps_eager_break_outermost() -> None:
    fwd_ctx: list[Any] = []

    class _FakeGraph:
        def capture_begin(self, pool: Any = None) -> None: ...
        def capture_end(self) -> None: ...
        def replay(self) -> None: ...

    @eager_break_during_capture
    def _eager_fwd(self: Any, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any) -> None:
        m = fwd_ctx[0].attn_metadata[self.prefix]
        if m.num_prefills <= 0:
            return
        q = mixed_qkv[2:66].view(1, 64, 2, 16)
        mods["vllm.models.kimi_k3.nvidia.ops.third_party.kda"].chunk_kda_with_fused_gate(
            q=q,
            k=q,
            v=q,
            raw_g=q,
            raw_beta=torch.zeros((1, 64, 2), dtype=torch.bfloat16),
            A_log=torch.zeros((2,)),
            g_bias=None,
            initial_state=torch.zeros((1, 2, 16, 16)),
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32),
        )

    vllm_cfg = VllmConfig()
    vllm_cfg.additional_config = {"kda_internal_prefill_checkpoints": True}
    with patch("vllm_ascend.platform.NPUPlatform.check_and_update_config"):
        ascend_config_mod.init_ascend_config(vllm_cfg)

    mods = _legacy_modules(_eager_fwd)
    with patch("importlib.import_module", side_effect=lambda n: mods[n]):
        mod.apply_patch()
    cls = mods["vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"].KimiGatedDeltaNetAttention
    assert "BreakableCUDAGraphCapture" in cls._forward.__code__.co_names
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        layer = cls("l0", _align_cfg())

    qkv = torch.ones((66, 32), dtype=torch.bfloat16)
    decode_meta = SimpleNamespace(
        num_prefills=0, num_decodes=2, num_decode_tokens=2, num_actual_tokens=66, checkpoint=None
    )
    mixed_meta = SimpleNamespace(
        num_prefills=1,
        num_decodes=2,
        num_decode_tokens=2,
        num_actual_tokens=66,
        spec_sequence_masks=None,
        non_spec_token_indx=None,
        checkpoint=SimpleNamespace(
            checkpoint_offsets=torch.tensor([0, 0, 64], dtype=torch.int32),
            state_indices=torch.tensor([-1, -1, 5]),
        ),
    )
    with (
        patch("torch.cuda.CUDAGraph", _FakeGraph),
        patch("vllm.compilation.breakable_cudagraph.weak_ref_tensor", side_effect=lambda x: x),
        patch("vllm.compilation.breakable_cudagraph.is_forward_context_available", return_value=True),
        patch("vllm.compilation.breakable_cudagraph.get_forward_context", side_effect=lambda: fwd_ctx[0]),
        patch("vllm.forward_context.get_forward_context", side_effect=lambda: fwd_ctx[0]),
        patch("vllm.model_executor.layers.mamba.mamba_utils.is_conv_state_dim_first", return_value=False),
        patch.object(
            mod,
            "ascend_chunk_kda_with_fused_gate",
            side_effect=lambda **kw: (kw["q"], torch.ones((1, 2, 16, 16))),
        ),
        patch.object(mod, "export_kda_prefill_checkpoint") as mock_export,
    ):
        fwd_ctx.append(
            SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, attn_metadata={"l0": decode_meta})
        )
        cap = BreakableCUDAGraphCapture()
        with cap:
            layer._forward(qkv, None, None, None, qkv)
        assert cap.num_eager_breaks == 1 and mock_export.call_count == 0
        fwd_ctx[0] = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, attn_metadata={"l0": mixed_meta})
        cap.replay()
        assert mock_export.call_count == 1


def test_patch_kda_checkpoint_native_noop_and_failsafe() -> None:
    def _native_fwd(self: Any, *a: Any, **kw: Any):
        return kw.get("checkpoint_state")

    mods = _legacy_modules(_native_fwd)
    cls = mods["vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"].KimiGatedDeltaNetAttention
    cls.get_kv_cache_checkpoint_alignment = lambda self, cfg: 64
    nv_mod = mods["vllm.models.kimi_k3.nvidia.ops.third_party.kda"]
    nv_mod.chunk_kda_with_fused_gate = lambda *a, checkpoint_state=None, **kw: a[0]  # type: ignore[attr-defined]
    orig_kda = nv_mod.chunk_kda_with_fused_gate
    with patch("importlib.import_module", side_effect=lambda n: mods[n]):
        mod.apply_patch()
    assert nv_mod.chunk_kda_with_fused_gate is orig_kda
    assert cls._forward is _native_fwd
    assert mod._non_bool_int(True) == 0 and mod._non_bool_int(3) == 3

    uninspectable = object()
    fail_mods = _legacy_modules(lambda *a, **kw: None)
    fail_nv = fail_mods["vllm.models.kimi_k3.nvidia.ops.third_party.kda"]
    fail_gdn = fail_mods["vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn"]
    fail_nv.chunk_kda_with_fused_gate = uninspectable  # type: ignore[attr-defined]
    fail_gdn.chunk_kda_with_fused_gate = uninspectable  # type: ignore[attr-defined]
    with patch("importlib.import_module", side_effect=lambda n: fail_mods[n]):
        mod.apply_patch()
    assert fail_nv.chunk_kda_with_fused_gate is uninspectable


class _FakeDevTensor(torch.Tensor):
    _fake_dev: Any = None
    _forbid_tolist: bool = False

    @property
    def device(self) -> Any:  # type: ignore[override]
        return self._fake_dev if self._fake_dev is not None else super().device

    def to(self, *a: Any, **kw: Any) -> Any:  # type: ignore[override]
        if a and getattr(a[0], "type", None) == "npu":
            return self
        return super().to(*a, **kw)

    def tolist(self) -> Any:
        if self._forbid_tolist:
            raise AssertionError("unexpected D2H .tolist() sync")
        return super().tolist()


def test_align_checkpoint_rows_device_only_zero_pad() -> None:
    cu_dev = torch.tensor([0, 1, 2, 66, 66, 194], dtype=torch.int32).as_subclass(_FakeDevTensor)
    cu_dev._fake_dev = SimpleNamespace(type="npu", index=0)
    cu_dev._forbid_tolist = True
    ckpt = SimpleNamespace(
        checkpoint_offsets=torch.tensor([64, 128], dtype=torch.int32),
        state_indices=torch.tensor([5, 9], dtype=torch.int64),
        offsets=[64, 128],
        query_lens=[64, 128],
    )
    aligned = mod._align_checkpoint_rows(ckpt, 5, 2, False, cu_seqlens=cu_dev, cu_seqlens_cpu=None)
    assert aligned.checkpoint_offsets.tolist() == [0, 0, 64, 0, 128]
    assert aligned.state_indices.tolist() == [NULL_BLOCK_ID, NULL_BLOCK_ID, 5, NULL_BLOCK_ID, 9]
    assert aligned.offsets is None and aligned.query_lens is None
