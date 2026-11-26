# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the INT8 CSA packed-cache ABI in ``mixed_quant_attention``.

The record is read positionally by the packaged operator, so the layout and the
quantization grid are pure contracts: any drift silently corrupts attention. The
Triton scatter and the CANN operators are stubbed, so the module stays CPU-only.
"""

import sys
import types

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import mixed_quant_attention as mqa


RECORD = mqa.INT8_KV_RECORD_BYTES
ROPE = mqa.INT8_ROPE_HEAD_DIM
NOPE = mqa.INT8_NOPE_HEAD_DIM
GROUPS = mqa.INT8_NOPE_GROUPS
LATENT = NOPE + ROPE
ROPE_END = ROPE * 2
NOPE_END = ROPE_END + NOPE
SCALE_END = NOPE_END + GROUPS * 2


class TestPackedRecordLayout:
    def test_every_byte_of_the_record_is_covered(self):
        assert (ROPE_END, NOPE_END, SCALE_END) == (128, 576, 590)
        assert RECORD - SCALE_END == 18

    def test_rope_half_is_copied_verbatim(self):
        values = torch.randn(3, LATENT)
        packed = mqa.pack_int8_kv(values)
        actual = packed.view(torch.bfloat16)[:, :ROPE]
        torch.testing.assert_close(actual, values[:, NOPE:].to(torch.bfloat16), rtol=0, atol=0)

    def test_nope_is_quantized_into_its_own_slot(self):
        values = torch.randn(4, LATENT) * 50.0
        packed = mqa.pack_int8_kv(values)
        body = packed.view(torch.uint8)[:, ROPE_END:NOPE_END].view(torch.int8)
        scale = packed.view(torch.bfloat16)[:, ROPE + NOPE // 2 : SCALE_END // 2].float()

        restored = body.reshape(4, GROUPS, 64).float() * scale.unsqueeze(-1)
        expected = values[:, :NOPE].float().view(4, GROUPS, 64)
        # Symmetric int8 on a 127-step grid bounds the error by half a step.
        assert torch.all((restored - expected).abs() <= scale.unsqueeze(-1) * 0.5 + 1e-6)

    def test_scales_use_the_symmetric_127_grid(self):
        # 128 would under-scale by one step and clip the largest magnitude.
        values = torch.full((1, LATENT), 127.0)
        scale = mqa.pack_int8_kv(values).view(torch.bfloat16)[:, ROPE + NOPE // 2 : SCALE_END // 2]
        assert scale.float().max() == pytest.approx(1.0, rel=1e-2)

    def test_quantization_rounds_to_nearest(self):
        # truncate/floor would bias every value down by up to one step.
        values = torch.zeros(1, LATENT)
        values[0, :NOPE] = torch.linspace(-1.0, 1.0, NOPE)
        body = mqa.pack_int8_kv(values).view(torch.uint8)[:, ROPE_END:NOPE_END].view(torch.int8)
        assert int(body.max()) == 127
        assert int(body.min()) == -127
        # A floor-based grid would leave the positive peak one step short.
        assert int((body == 127).sum()) >= 1


class TestAttentionCacheWriter:
    @pytest.mark.parametrize("kind", ["win", "cmp"])
    def test_supported_kinds_pack_and_forward(self, recorded_writes, kind):
        cache = packed_cache()
        values = torch.randn(3, LATENT)
        slots = torch.tensor([0, 130, 260], dtype=torch.int64)

        mqa.int8_write_attention_cache(cache, slots, values, kind=kind)

        call = recorded_writes[0]
        assert call["cache"] is cache
        assert call["updates"].shape == (3, RECORD)
        assert torch.equal(call["slots"], slots)


class TestInt8QsmlaGuards:
    def test_missing_compressed_cache_raises(self, cann_ops):
        # A ratio-0 layer entering this path would otherwise reach the operator
        # with cmp_kv=None and fail inside CANN.
        with pytest.raises(ValueError, match="requires a compressed KV cache"):
            mqa.int8_qsmla(
                torch.zeros(1, 2, 128),
                torch.zeros(1, 128),
                None,
                metadata(smla_metadata=torch.zeros(RECORD, dtype=torch.int32)),
                torch.zeros(1, 4, dtype=torch.int32),
                window_size=128,
                sinks=None,
                softmax_scale=0.1,
            )

    def test_missing_metadata_raises(self, cann_ops):
        # Without this guard the operator is launched with smla_metadata=None
        # and fails deep inside CANN instead of at the call site.
        with pytest.raises(RuntimeError, match="metadata was not built"):
            mqa.int8_qsmla(
                torch.zeros(1, 2, 128),
                torch.zeros(1, 128),
                torch.zeros(1, 128),
                metadata(smla_metadata=None),
                torch.zeros(1, 4, dtype=torch.int32),
                window_size=128,
                sinks=None,
                softmax_scale=0.1,
            )

    def test_ratio_one_drops_the_residual_hint(self, cann_ops):
        # ratio 1 has no residual slot; forwarding one would read stale memory.
        mqa.int8_qsmla(
            torch.zeros(2, 2, 128),
            torch.zeros(2, 128),
            torch.zeros(2, 128),
            metadata(
                smla_metadata=torch.zeros(RECORD, dtype=torch.int32),
                cmp_residual=torch.ones(2, dtype=torch.int32),
            ),
            torch.zeros(2, 4, dtype=torch.int32),
            window_size=128,
            sinks=None,
            softmax_scale=0.5,
            compress_ratio=1,
            index_topk=4,
        )
        assert cann_ops.attention_calls[-1]["cmp_residual_kv"] is None


class TestInt8MetadataBuilder:
    def test_forwards_the_packed_layout(self, cann_ops):
        # A wrong layout makes the operator parse the packed record wrongly.
        mqa.int8_build_smla_metadata(**metadata_kwargs(ori_mask_mode=4, ori_win_left=3, ori_win_right=0))
        kwargs = cann_ops.metadata_calls[-1]
        assert kwargs["rope_head_dim"] == ROPE
        assert kwargs["cmp_mask_mode"] == 3
        assert (kwargs["layout_q"], kwargs["layout_kv"]) == ("TND", "PA_BBND")
        assert (kwargs["ori_win_left"], kwargs["ori_win_right"]) == (3, 0)


class TestBackendMarkers:
    def test_backends_declare_mutually_exclusive_abis(self):
        # _packed_cache_ready tests both markers as parallel branches, so a
        # missing attribute would silently fall through to the scatter path.
        assert mqa.MixedQuantPackedCacheOps.uses_int8_packed_cache is False
        assert mqa.Int8MixedQuantPackedCacheOps.uses_a5_packed_cache is False
        assert mqa.Int8MixedQuantPackedCacheOps.uses_int8_packed_cache is True


def packed_cache(pages: int = 4, page_size: int = 128) -> torch.Tensor:
    return torch.zeros(pages, page_size, 1, RECORD, dtype=torch.int8)


def metadata_kwargs(**overrides) -> dict:
    kwargs = dict(
        num_heads_q=2,
        head_dim=128,
        num_reqs=1,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        cmp_seq_lens=torch.tensor([5], dtype=torch.int32),
        cmp_residual=torch.zeros(1, dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_ori_kv=10,
        max_seqlen_cmp_kv=5,
        index_topk=4,
        compress_ratio=2,
        ori_mask_mode=3,
        ori_win_left=0,
        ori_win_right=0,
    )
    kwargs.update(overrides)
    return kwargs


def metadata(smla_metadata, cmp_residual=None):
    return types.SimpleNamespace(
        swa=types.SimpleNamespace(
            smla_metadata=smla_metadata,
            num_reqs=1,
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            seq_lens=torch.tensor([8], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            replay_block_table=None,
            replay_cache_blocks=None,
            replay_cache_offsets=None,
            replay_seq_lens=None,
            storage_block_size=128,
            ori_mask_mode=3,
            ori_win_left=-1,
            ori_win_right=-1,
        ),
        attention=types.SimpleNamespace(
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            cache_seq_lens=torch.tensor([4], dtype=torch.int32),
            cmp_residual=cmp_residual,
        ),
    )


class _RecordingCannOps:
    def __init__(self):
        self.attention_calls = []
        self.metadata_calls = []

    def mixed_quant_sparse_flash_mla(self, q, *args, **kwargs):
        self.attention_calls.append(kwargs)
        return torch.zeros(q.shape[0], 128), None

    def mixed_quant_sparse_flash_mla_metadata(self, *args, **kwargs):
        self.metadata_calls.append(kwargs)
        return torch.zeros(mqa.INT8_METADATA_SIZE, dtype=torch.int32)


@pytest.fixture
def recorded_writes(monkeypatch):
    calls = []

    def record(cache, updates, slots):
        calls.append({"cache": cache, "updates": updates.clone(), "slots": slots.clone()})

    monkeypatch.setattr(mqa, "_int8_write_packed_cache_tensor", record)
    return calls


@pytest.fixture
def cann_ops(monkeypatch):
    recorder = _RecordingCannOps()
    namespace = types.SimpleNamespace(
        mixed_quant_sparse_flash_mla=recorder.mixed_quant_sparse_flash_mla,
        mixed_quant_sparse_flash_mla_metadata=recorder.mixed_quant_sparse_flash_mla_metadata,
    )
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", types.ModuleType("cann_ops_transformer"))
    monkeypatch.setattr(torch, "ops", types.SimpleNamespace(cann_ops_transformer=namespace))
    monkeypatch.setattr(mqa, "wait_for_device_metadata", lambda *a, **k: None)
    return recorder
