# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata, GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend import ascend_config as ascend_config_mod
from vllm_ascend.attention.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
from vllm_ascend.ops import kda_checkpoint
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention
from vllm_ascend.ops.triton.kda import chunk_delta_h
from vllm_ascend.ops.triton.kda import kda as kda_triton
from vllm_ascend.ops.triton.kda.utils import prepare_chunk_indices


@pytest.fixture(autouse=True)
def _reset_ascend_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ascend_config_mod, "_ASCEND_CONFIG", None)


def _vllm_cfg(
    *,
    mode: str = "align",
    pcp: int = 1,
    ubatch: bool = False,
    kv_xfer: Any = None,
    opt_in: Any = True,
    use_replayssm: bool = False,
    use_kda_recoverssm: bool = False,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(
            mamba_cache_mode=mode,
            block_size=256,
            use_replayssm=use_replayssm,
            use_kda_recoverssm=use_kda_recoverssm,
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, use_ubatching=ubatch),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=512),
        kv_transfer_config=kv_xfer,
        speculative_config=None,
        num_hash_lookahead_tokens=0,
        additional_config={} if opt_in is None else {"kda_internal_prefill_checkpoints": opt_in},
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def test_kda_checkpoint_opt_in_and_spec_cleanup() -> None:
    assert kda_checkpoint.supports_kda_prefill_checkpoint(_vllm_cfg(), head_dim=128, conv_size=4, local_num_heads=4)
    assert kda_checkpoint.supports_kda_prefill_checkpoint(
        _vllm_cfg(opt_in=" TRUE "), head_dim=128, conv_size=4, local_num_heads=4
    )
    no_parallel = _vllm_cfg()
    no_parallel.parallel_config = None
    for bad in (
        _vllm_cfg(opt_in=None),
        _vllm_cfg(opt_in="0"),
        _vllm_cfg(mode="none"),
        _vllm_cfg(pcp=2),
        _vllm_cfg(kv_xfer=object()),
        _vllm_cfg(use_replayssm=True),
        _vllm_cfg(use_kda_recoverssm=True),
        no_parallel,
    ):
        assert not kda_checkpoint.supports_kda_prefill_checkpoint(bad, head_dim=128, conv_size=4, local_num_heads=4)

    layer: Any = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim, layer.conv_size, layer.local_num_heads = 128, 4, 4
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.bfloat16)

    base = MambaSpec(
        block_size=256,
        shapes=((8, 3), (4, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=0,
        prefill_checkpoint_alignment=None,
    )
    with patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False):
        assert kda_checkpoint.initialize_kda_checkpoints(layer, _vllm_cfg()) is True
    upgraded = kda_checkpoint.build_kda_kv_cache_spec(layer, _vllm_cfg(), base)
    assert (upgraded.num_prefill_checkpoint_blocks, upgraded.prefill_checkpoint_alignment) == (1, 64)

    disabled_cfg = _vllm_cfg(opt_in=False)
    assert kda_checkpoint.initialize_kda_checkpoints(layer, disabled_cfg) is False
    stale = replace(base, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=64)
    with patch("vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention.get_kv_cache_spec", return_value=stale):
        cleared = layer.get_kv_cache_spec(disabled_cfg)
    assert (cleared.num_prefill_checkpoint_blocks, cleared.prefill_checkpoint_alignment) == (0, None)

    def _base_init(s: Any, spec: Any, names: Any, cfg: Any, dev: Any) -> None:
        s.vllm_config = cfg
        s.decode_cudagraph_max_bs = 0

    disabled_cfg.compilation_config.static_forward_context = {"layer.0": layer}
    with patch.object(GDNAttentionMetadataBuilder, "__init__", _base_init):
        with pytest.raises(ValueError, match="Ascend-owned checkpoint-capable layers"):
            AscendGDNAttentionMetadataBuilder(stale, ["layer.0"], disabled_cfg, torch.device("cpu"))
        builder = AscendGDNAttentionMetadataBuilder(cleared, ["layer.0"], disabled_cfg, torch.device("cpu"))
    assert builder._ascend_gdn_checkpoints is False


@dataclass
class _DummyCkpt:
    checkpoint_offsets: torch.Tensor
    state_indices: torch.Tensor
    offsets: list[int]
    query_lens: tuple[int, ...]


class _NoSyncTensor(torch.Tensor):
    _fake_dev: Any = None
    _forbid_tolist: bool = False

    @property
    def device(self) -> Any:  # type: ignore[override]
        return self._fake_dev if self._fake_dev is not None else super().device

    def contiguous(self, *a: Any, **kw: Any) -> Any:  # type: ignore[override]
        out = super().contiguous(*a, **kw)
        if self._fake_dev is not None and out is not self:
            out = out.as_subclass(_NoSyncTensor)
            out._fake_dev = self._fake_dev
        return out

    def to(self, *a: Any, **kw: Any) -> Any:  # type: ignore[override]
        return self if self._fake_dev is not None else super().to(*a, **kw)

    def tolist(self) -> Any:
        if self._forbid_tolist:
            raise AssertionError("unexpected D2H .tolist() sync")
        return super().tolist()


def _fake_npu(t: torch.Tensor, idx: int = 0) -> _NoSyncTensor:
    w = t.as_subclass(_NoSyncTensor)
    w._fake_dev = SimpleNamespace(type="npu", index=idx)
    return w


def test_kda_checkpoint_compaction_no_sync_and_export() -> None:
    si, hi = torch.tensor([1, 2, 3]), torch.tensor([True, False, True])
    cu = torch.tensor([0, 64, 64, 192], dtype=torch.int32)
    ckpt = _DummyCkpt(
        torch.tensor([64, 0, 128], dtype=torch.int32), torch.tensor([11, 22, 33]), [64, 0, 128], (64, 0, 128)
    )
    keep = _fake_npu(torch.tensor([True, False, True], dtype=torch.bool))
    keep._forbid_tolist = True
    meta = SimpleNamespace(keep_meta=keep, cu_seqlens_host=(0, 64, 64, 192), cu_seqlens_kern=(0, 64, 192))
    c_si, c_hi, c_cu, c_ckpt, c_pre = kda_checkpoint.prepare_kda_checkpoint_prefill_inputs(
        state_indices=si, has_initial_state=hi, prefill_query_start_loc=cu, chunk_meta=meta, checkpoint=ckpt
    )
    assert c_pre is None and c_si.tolist() == [1, 3] and c_hi.tolist() == [True, True] and c_cu.tolist() == [0, 64, 192]
    assert c_ckpt is not ckpt and c_ckpt.offsets == [64, 128] and ckpt.offsets == [64, 0, 128]

    exporter = MagicMock()
    q = torch.ones((1, 128, 2, 16), dtype=torch.bfloat16)
    with (
        patch("vllm.v1.worker.workspace.is_workspace_manager_initialized", return_value=False),
        patch.object(kda_checkpoint, "chunk_kda_with_fused_gate", return_value=(q, torch.ones((1, 2, 16, 16)))) as mk,
    ):
        kda_checkpoint.run_kda_checkpoint_prefill(
            exporter=exporter,
            checkpoint_spec=((4, 2, 16, 16), torch.bfloat16),
            checkpoint=c_ckpt,
            q=q,
            k=q,
            v=q,
            raw_gate=q,
            raw_beta=torch.zeros((1, 128, 2), dtype=torch.bfloat16),
            a_log=torch.zeros((2,)),
            dt_bias=None,
            initial_state=torch.zeros((1, 2, 16, 16)),
            cu_seqlens=c_cu,
            raw_qkv=torch.zeros((128, 96), dtype=torch.bfloat16),
            conv_state=torch.zeros((4, 3, 96), dtype=torch.bfloat16),
            conv_state_is_dim_first=False,
            recurrent_state=torch.zeros((4, 2, 16, 16), dtype=torch.bfloat16),
        )
    assert tuple(mk.call_args.kwargs["checkpoint_state"].shape) == (1, 2, 16, 16)
    assert tuple(exporter.export.call_args.kwargs["conv_state"].shape) == (4, 96, 3)


def test_kda_checkpoint_triton_guards() -> None:
    k_npu = _fake_npu(torch.zeros((1, 128, 2, 64), dtype=torch.float32))
    cu_bad = _fake_npu(torch.tensor([0.0, 128.0], dtype=torch.float32))
    with pytest.raises(ValueError, match="int32 or int64"):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(k=k_npu, w=k_npu, u=k_npu, cu_seqlens=cu_bad)

    non_contig = _fake_npu(torch.tensor([0, 99, 64, 99, 128], dtype=torch.int64)[::2])
    fake_kernel = MagicMock()
    with patch.object(chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_kda", fake_kernel):
        chunk_delta_h.chunk_gated_delta_rule_fwd_h_kda(
            k=k_npu,
            w=k_npu,
            u=k_npu,
            cu_seqlens=non_contig,
            chunk_indices=torch.zeros((2, 2), dtype=torch.int32),
            chunk_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
            checkpoint_state=_fake_npu(torch.zeros((2, 2, 64, 64), dtype=torch.bfloat16)),
            checkpoint_offsets=_fake_npu(torch.tensor([64, 99, 0], dtype=torch.int64)[::2]),
        )
    assert fake_kernel.__getitem__.return_value.call_args.kwargs["checkpoint_offsets"].is_contiguous()

    with pytest.raises(ValueError, match="provided together"):
        kda_triton.chunk_kda(
            q=k_npu,
            k=k_npu,
            v=k_npu,
            g=k_npu,
            beta=torch.zeros((1, 128, 2)),
            checkpoint_state=k_npu,
        )

    q_small = torch.zeros((1, 8, 2, 16), dtype=torch.bfloat16)
    with (
        patch.object(kda_triton, "apply_kda_gate", return_value=q_small) as mk_gate,
        patch.object(kda_triton, "chunk_kda", return_value=(q_small, None)),
    ):
        kda_triton.chunk_kda_with_fused_gate(
            q=q_small, k=q_small, v=q_small, raw_g=q_small, raw_beta=torch.zeros((1, 8, 2)), A_log=torch.zeros((2,))
        )
        assert mk_gate.call_args.kwargs["safe_gate"] is False
        kda_triton.chunk_kda_with_fused_gate(
            q=q_small,
            k=q_small,
            v=q_small,
            raw_g=q_small,
            raw_beta=torch.zeros((1, 8, 2)),
            A_log=torch.zeros((2,)),
            lower_bound=-5.0,
        )
        assert mk_gate.call_args.kwargs["safe_gate"] is True


def test_kda_checkpoint_layer_prefill_paths() -> None:
    import vllm_ascend.models.glm5next.kda as glm5next_kda_mod

    layer: Any = glm5next_kda_mod.Glm5NextLinearAttention.__new__(glm5next_kda_mod.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix, layer.kda_lower_bound, layer.local_num_heads, layer.head_dim = "l0", -5.0, 2, 16
    layer.local_projection_size, layer._conv_state_dim_first = 32, False
    layer.kv_cache = (torch.zeros((4, 3, 96)), torch.zeros((4, 2, 16, 16)))
    layer._merged_conv_weight, layer.A_log, layer.dt_bias = torch.zeros((96, 4)), torch.zeros((2,)), None
    layer._ascend_internal_checkpoint_enabled, layer._checkpoint_exporter = True, MagicMock()
    layer._triton_checkpoint_spec = ((4, 2, 16, 16), torch.float32)

    meta: Any = GDNAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=64,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=64,
        non_spec_state_indices_tensor=torch.tensor([0, 1], dtype=torch.int32),
    )
    meta.prefill_query_start_loc = torch.tensor([0, 64, 64], dtype=torch.int32)
    meta.prefill_state_indices, meta.prefill_has_initial_state = torch.tensor([2, 3]), torch.tensor([True, False])
    meta.non_spec_prefill_metadata = SimpleNamespace(
        causal_conv1d=SimpleNamespace(query_start_loc=None, cache_indices=None, initial_state_mode=None),
        chunk=SimpleNamespace(keep_meta=torch.tensor([True, False]), cu_seqlens_kern=(0, 64)),
    )
    meta.checkpoint = _DummyCkpt(torch.tensor([64, 0], dtype=torch.int32), torch.tensor([2, 3]), [64, 0], (64, 0))
    fake_out = torch.ones((1, 64, 2, 16))
    with (
        patch.object(glm5next_kda_mod, "get_forward_context", return_value=SimpleNamespace(attn_metadata={"l0": meta})),
        patch.object(glm5next_kda_mod, "wait_for_kv_layer_from_connector"),
        patch.object(glm5next_kda_mod, "record_attention_compute_start"),
        patch.object(glm5next_kda_mod, "maybe_save_kv_layer_to_connector"),
        patch.object(glm5next_kda_mod, "causal_conv1d", side_effect=lambda x, *a, **kw: x),
        patch.object(glm5next_kda_mod, "gather_initial_states", return_value=torch.zeros((1, 2, 16, 16))),
        patch.object(glm5next_kda_mod, "scatter_states"),
        patch.object(
            glm5next_kda_mod, "run_kda_checkpoint_prefill", return_value=(fake_out, torch.ones((1, 2, 16, 16)))
        ) as mk_run,
    ):
        layer._forward_core(
            torch.ones((64, 96)), torch.zeros((1, 64, 2, 16)), torch.zeros((1, 64, 2)), torch.zeros((1, 64, 2, 16))
        )
    assert (
        mk_run.call_args.kwargs["cu_seqlens"].tolist() == [0, 64] and mk_run.call_args.kwargs["prebuilt_meta"] is None
    )

    kimi_layer: Any = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(kimi_layer)
    kimi_layer.A_log, kimi_layer.dt_bias, kimi_layer.gate_lower_bound = torch.zeros((2,)), None, None
    kimi_layer._checkpoint_exporter, kimi_layer._triton_checkpoint_spec = MagicMock(), ((4, 2, 16, 16), torch.float32)
    kimi_layer._kda_state_copy_ready = True
    kimi_layer._ascend_kda_state_copy = MagicMock(gather=MagicMock(return_value=torch.zeros((1, 2, 16, 16))))
    with (
        patch("vllm_ascend.ops.kimi_kda.is_conv_state_dim_first", return_value=True),
        patch(
            "vllm_ascend.ops.kimi_kda.run_kda_checkpoint_prefill",
            return_value=(fake_out, torch.ones((1, 2, 16, 16))),
        ) as mk_kimi,
    ):
        kimi_layer._run_prefill_with_checkpoint(
            q=fake_out,
            k=fake_out,
            v=fake_out,
            raw_gate=fake_out,
            beta=torch.zeros((1, 64, 2)),
            raw_qkv=torch.zeros((64, 96)),
            conv_state=torch.zeros((4, 96, 3)),
            recurrent_state=torch.zeros((4, 2, 16, 16)),
            state_indices=torch.tensor([2, 3]),
            has_initial_state=torch.tensor([True, False]),
            prefill_query_start_loc=None,
            prebuilt_metadata=SimpleNamespace(keep_meta=torch.tensor([True, False]), cu_seqlens_kern=(0, 64)),
            checkpoint=meta.checkpoint,
        )
    assert mk_kimi.call_args.kwargs["cu_seqlens"].tolist() == [0, 64] and mk_kimi.call_args.kwargs["safe_gate"] is False


def test_prepare_chunk_indices_with_zero_length_segments() -> None:
    cu_mixed = torch.tensor([0, 1, 2, 66, 66, 194], dtype=torch.int32)
    idx_mixed = prepare_chunk_indices(cu_mixed, 64)
    assert idx_mixed.dtype == torch.int32 and idx_mixed.tolist() == [[0, 0], [1, 0], [2, 0], [4, 0], [4, 1]]

    cu_leading = torch.tensor([0, 0, 64], dtype=torch.int64)
    idx_leading = prepare_chunk_indices(cu_leading, 64)
    assert idx_leading.dtype == torch.int64 and idx_leading.tolist() == [[1, 0]]

    cu_empty = torch.tensor([0, 0, 0], dtype=torch.int32)
    idx_empty = prepare_chunk_indices(cu_empty, 64)
    assert idx_empty.dtype == torch.int32 and tuple(idx_empty.shape) == (0, 2)
