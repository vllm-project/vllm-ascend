# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import vllm_ascend.patch.worker.patch_gdn_attn as patch_gdn_attn
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.fla.ops import index as _fla_index
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec
from vllm_ascend.ops.gdn import (
    get_non_spec_causal_conv1d_host_args,
    get_non_spec_chunked_prefill_meta,
    to_int64_tuple,
)
from vllm_ascend.ops.triton.fla import utils as fla_utils
from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices as runtime_prepare_chunk_indices,
    prepare_chunk_offsets as runtime_prepare_chunk_offsets,
    prepare_final_chunk_indices as runtime_prepare_final_chunk_indices,
    prepare_update_chunk_offsets as runtime_prepare_update_chunk_offsets,
)


@pytest.fixture(autouse=True)
def _patch_triton_cdiv(monkeypatch):
    if not hasattr(_fla_index.triton, "cdiv"):
        monkeypatch.setattr(
            _fla_index.triton,
            "cdiv",
            lambda a, b: (a + b - 1) // b,
            raising=False,
        )


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]
    name: str = "unnamed"

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens,
        dtype=torch.int32,
        device=device,
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = sum(batch_spec.query_lens)

    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [
        batch_spec.seq_lens[i] - batch_spec.query_lens[i]
        for i in range(batch_spec.batch_size)
    ]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def _make_vllm_config(
    *,
    max_model_len: int = 8192,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_heads: int = 32,
    num_speculative_tokens: int = 0,
):
    speculative_config = None
    if num_speculative_tokens > 0:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
        )

    model_config = SimpleNamespace(max_model_len=max_model_len)
    model_config.get_num_attention_heads = lambda parallel_config: num_heads

    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        model_config=model_config,
    )


def _make_builder(*, device: torch.device, num_heads: int, num_speculative_tokens: int):
    vllm_config = _make_vllm_config(
        num_heads=num_heads,
        num_speculative_tokens=num_speculative_tokens,
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    return GDNAttentionMetadataBuilder(spec, ["layer0"], vllm_config, device)


def _build_attn_metadata(
    batch_spec: BatchSpec,
    *,
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
):
    device = torch.device("cpu")
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=device,
    )
    builder = _make_builder(
        device=device,
        num_heads=32,
        num_speculative_tokens=num_speculative_tokens,
    )
    num_accepted_tokens = None
    if num_decode_draft_tokens_cpu is not None:
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size,
            dtype=torch.int32,
            device=device,
        )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )
    return builder, common_attn_metadata, attn_metadata


def _assert_chunk_meta_matches_runtime(builder, chunk_meta, cu_seqlens: torch.Tensor) -> None:
    assert torch.equal(
        chunk_meta.chunk_indices_chunk64,
        runtime_prepare_chunk_indices(cu_seqlens, patch_gdn_attn._GDN_CHUNK_SIZE),
    )
    assert torch.equal(
        chunk_meta.chunk_offsets_chunk64,
        runtime_prepare_chunk_offsets(cu_seqlens, patch_gdn_attn._GDN_CHUNK_SIZE),
    )
    assert torch.equal(
        chunk_meta.update_chunk_offsets_chunk64,
        runtime_prepare_update_chunk_offsets(
            cu_seqlens,
            patch_gdn_attn._GDN_CHUNK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.final_chunk_indices_chunk64,
        runtime_prepare_final_chunk_indices(
            cu_seqlens,
            patch_gdn_attn._GDN_CHUNK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.chunk_indices_large_block,
        runtime_prepare_chunk_indices(
            cu_seqlens,
            patch_gdn_attn._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.block_indices_cumsum,
        runtime_prepare_chunk_indices(
            cu_seqlens,
            builder._ascend_gdn_cumsum_block_size,
        ),
    )


def _patch_missing_runtime_cdiv(monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(fla_utils.triton, "cdiv"):
        return
    monkeypatch.setattr(
        fla_utils.triton,
        "cdiv",
        lambda x, y: (x + y - 1) // y,
        raising=False,
    )


def _expected_conv1d_host_args(attn_metadata) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        to_int64_tuple(attn_metadata.non_spec_query_start_loc),
        to_int64_tuple(attn_metadata.non_spec_state_indices_tensor),
        to_int64_tuple(attn_metadata.has_initial_state),
    )


@pytest.mark.parametrize(
    ("batch_spec", "num_speculative_tokens", "num_decode_draft_tokens_cpu"),
    [
        (
            BatchSpec(
                seq_lens=[8, 12],
                query_lens=[4, 8],
                name="pure_non_spec_prefill",
            ),
            0,
            None,
        ),
        (
            BatchSpec(
                seq_lens=[8, 4, 0, 12],
                query_lens=[4, 4, 0, 8],
                name="mixed_spec_non_spec_with_padding",
            ),
            3,
            torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
        ),
        (
            BatchSpec(
                seq_lens=[5, 12, 0, 9],
                query_lens=[1, 8, 0, 1],
                name="mixed_prefill_decode_without_spec",
            ),
            0,
            None,
        ),
    ],
    ids=lambda case: case.name if isinstance(case, BatchSpec) else None,
)
def test_non_spec_prefill_fallback_meta_matches_original_inputs_and_runtime_helpers(
    batch_spec: BatchSpec,
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    builder, _, attn_metadata = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=num_speculative_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    fallback_meta = getattr(attn_metadata, "non_spec_prefill_fallback_meta", None)
    assert fallback_meta is not None
    assert fallback_meta.causal_conv1d is not None
    assert fallback_meta.chunk is not None

    assert get_non_spec_causal_conv1d_host_args(attn_metadata) == _expected_conv1d_host_args(attn_metadata)

    _assert_chunk_meta_matches_runtime(
        builder,
        fallback_meta.chunk,
        attn_metadata.non_spec_query_start_loc,
    )

def test_build_non_spec_causal_conv1d_host_meta_avoids_seq_lens_cpu_fallback():
    class GuardSeqLens:
        def to(self, *args, **kwargs):
            raise AssertionError("seq_lens.to('cpu') should not be reached")

    builder = SimpleNamespace(use_spec_decode=False)
    attn_metadata = SimpleNamespace(
        num_prefills=2,
        non_spec_state_indices_tensor=torch.tensor([3, 9], dtype=torch.int32),
        has_initial_state=torch.tensor([True, False]),
    )

    host_meta = patch_gdn_attn._build_non_spec_causal_conv1d_host_meta(
        builder,
        attn_metadata,
        non_spec_query_start_loc_cpu=torch.tensor([0, 4, 12], dtype=torch.int32),
    )

    assert host_meta is not None
    assert torch.equal(
        host_meta.has_initial_state_cpu,
        torch.tensor([True, False]),
    )


def test_build_non_spec_causal_conv1d_host_meta_requires_has_initial_state():
    builder = SimpleNamespace(use_spec_decode=False)
    attn_metadata = SimpleNamespace(
        num_prefills=2,
        non_spec_state_indices_tensor=torch.tensor([3, 9], dtype=torch.int32),
        has_initial_state=None,
    )
    with pytest.raises(RuntimeError, match="has_initial_state"):
        patch_gdn_attn._build_non_spec_causal_conv1d_host_meta(
            builder,
            attn_metadata,
            non_spec_query_start_loc_cpu=torch.tensor([0, 4, 12], dtype=torch.int32),
        )


def test_get_non_spec_causal_conv1d_host_args_requires_prefill_fallback_meta():
    attn_metadata = SimpleNamespace(
        non_spec_prefill_fallback_meta=None,
        non_spec_causal_conv1d_meta=SimpleNamespace(
            query_start_loc_opt=(0, 4, 12),
            cache_indices_opt=(3, 9),
            initial_state_mode_opt=(1, 0),
        ),
    )

    with pytest.raises(RuntimeError, match="non_spec_prefill_fallback_meta\\.causal_conv1d"):
        get_non_spec_causal_conv1d_host_args(
            attn_metadata,
        )


def test_get_non_spec_chunked_prefill_meta_requires_prefill_fallback_meta():
    attn_metadata = SimpleNamespace(
        non_spec_prefill_fallback_meta=None,
        non_spec_chunked_prefill_meta=SimpleNamespace(chunk_offsets_chunk64=torch.tensor([0, 1])),
    )

    with pytest.raises(RuntimeError, match="non_spec_prefill_fallback_meta\\.chunk"):
        get_non_spec_chunked_prefill_meta(attn_metadata)


def test_builder_uses_device_chunk_builder_with_non_spec_query_start_loc(monkeypatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[8, 4, 0, 12],
        query_lens=[4, 4, 0, 8],
        name="mixed_spec_non_spec_with_padding",
    )
    builder, common_attn_metadata, _ = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=3,
        num_decode_draft_tokens_cpu=torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
    )

    builder._ascend_gdn_chunk_meta_initialized = True
    builder._ascend_gdn_chunk_meta_device = SimpleNamespace(type="npu")
    builder._ascend_gdn_chunk_size = patch_gdn_attn._GDN_CHUNK_SIZE
    builder._ascend_gdn_large_block_size = patch_gdn_attn._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE
    builder._ascend_gdn_cumsum_block_size = 256
    builder._ascend_gdn_chunked_prefill_pool_idx = -1
    builder._ascend_gdn_chunked_prefill_pool = [
        SimpleNamespace(
            chunk_indices_chunk64=torch.zeros((8, 2), dtype=torch.int32),
            chunk_offsets_chunk64=torch.zeros((8,), dtype=torch.int32),
            update_chunk_offsets_chunk64=torch.zeros((8,), dtype=torch.int32),
            final_chunk_indices_chunk64=torch.zeros((8,), dtype=torch.int32),
            chunk_indices_large_block=torch.zeros((8, 2), dtype=torch.int32),
            block_indices_cumsum=torch.zeros((8, 2), dtype=torch.int32),
        )
    ]
    builder._ascend_gdn_causal_conv1d_host_meta_initialized = True
    builder._ascend_gdn_causal_conv1d_host_pool = []
    builder._ascend_gdn_causal_conv1d_host_pool_idx = -1

    helper_calls: dict[int, dict[str, object]] = {}

    def fake_build_chunk_meta_device(**kwargs):
        helper_calls[kwargs["chunk_size"]] = kwargs

    monkeypatch.setattr(
        patch_gdn_attn,
        "build_chunk_meta_device",
        fake_build_chunk_meta_device,
        raising=False,
    )
    monkeypatch.setattr(
        patch_gdn_attn,
        "_prepare_chunk_counts_cpu",
        lambda *args, **kwargs: pytest.fail("_prepare_chunk_counts_cpu should not be used on the device path"),
    )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=torch.ones(batch_spec.batch_size, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
    )

    expected_chunk_indices = runtime_prepare_chunk_indices(
        attn_metadata.non_spec_query_start_loc,
        patch_gdn_attn._GDN_CHUNK_SIZE,
    )
    expected_chunk_offsets = runtime_prepare_chunk_offsets(
        attn_metadata.non_spec_query_start_loc,
        patch_gdn_attn._GDN_CHUNK_SIZE,
    )
    expected_update_chunk_offsets = runtime_prepare_update_chunk_offsets(
        attn_metadata.non_spec_query_start_loc,
        patch_gdn_attn._GDN_CHUNK_SIZE,
    )
    expected_final_chunk_indices = runtime_prepare_final_chunk_indices(
        attn_metadata.non_spec_query_start_loc,
        patch_gdn_attn._GDN_CHUNK_SIZE,
    )

    chunk64_call = helper_calls[patch_gdn_attn._GDN_CHUNK_SIZE]
    out_chunk_indices = cast(torch.Tensor, chunk64_call["out_chunk_indices"])
    out_chunk_offsets = cast(torch.Tensor, chunk64_call["out_chunk_offsets"])
    out_update_chunk_offsets = cast(
        torch.Tensor,
        chunk64_call["out_update_chunk_offsets"],
    )
    out_final_chunk_indices = cast(
        torch.Tensor,
        chunk64_call["out_final_chunk_indices"],
    )
    assert chunk64_call["cu_seqlens"] is attn_metadata.non_spec_query_start_loc
    assert out_chunk_indices.shape == expected_chunk_indices.shape
    assert out_chunk_indices.dtype == expected_chunk_indices.dtype
    assert out_chunk_offsets.shape == expected_chunk_offsets.shape
    assert out_chunk_offsets.dtype == torch.int32
    assert out_update_chunk_offsets.shape == expected_update_chunk_offsets.shape
    assert out_update_chunk_offsets.dtype == torch.int32
    assert out_final_chunk_indices.shape == expected_final_chunk_indices.shape
    assert out_final_chunk_indices.dtype == torch.int32


@pytest.mark.parametrize(
    "batch_spec",
    [
        BatchSpec(seq_lens=[1, 1, 1], query_lens=[1, 1, 1], name="decode_only"),
        BatchSpec(seq_lens=[4, 4], query_lens=[4, 4], name="spec_only"),
    ],
)
def test_builder_skips_prebuilt_meta_without_non_spec_prefill(batch_spec: BatchSpec):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3 if batch_spec.name == "spec_only" else 0,
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )

    num_accepted_tokens = None
    num_decode_draft_tokens_cpu = None
    if batch_spec.name == "spec_only":
        num_accepted_tokens = torch.ones(
            batch_spec.batch_size,
            dtype=torch.int32,
            device="cpu",
        )
        num_decode_draft_tokens_cpu = torch.full(
            (batch_spec.batch_size,),
            3,
            dtype=torch.int32,
        )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    assert getattr(attn_metadata, "non_spec_prefill_fallback_meta", None) is None
