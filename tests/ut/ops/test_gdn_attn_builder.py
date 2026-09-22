# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.fla.ops import index as _fla_index
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops import gdn as gdn_module
from vllm_ascend.ops import gdn_attn_builder as ascend_gdn_attn_builder
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
)
from vllm_ascend.ops.triton.fla import utils as fla_utils
from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices as runtime_prepare_chunk_indices,
)
from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_offsets as runtime_prepare_chunk_offsets,
)
from vllm_ascend.ops.triton.fla.utils import (
    prepare_final_chunk_indices as runtime_prepare_final_chunk_indices,
)
from vllm_ascend.ops.triton.fla.utils import (
    prepare_update_chunk_offsets as runtime_prepare_update_chunk_offsets,
)
from vllm_ascend.utils import vllm_version_is


@pytest.fixture(autouse=True)
def _patch_triton_cdiv(monkeypatch):
    if not hasattr(_fla_index.triton, "cdiv"):
        monkeypatch.setattr(
            _fla_index.triton,
            "cdiv",
            lambda a, b: (a + b - 1) // b,
            raising=False,
        )


@pytest.fixture(autouse=True)
def _no_pin_memory():
    # compute_causal_conv1d_metadata uses np_to_pinned_tensor which reads
    # PIN_MEMORY.  Without physical NPU, t.pin_memory() raises
    # "Please register PrivateUse1HooksInterface first".
    with patch("vllm.utils.torch_utils.PIN_MEMORY", False):
        if vllm_version_is("0.23.0"):
            yield
        else:
            with patch("vllm.v1.attention.backends.utils.PIN_MEMORY", False):
                yield


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]
    name: str = "unnamed"

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)


class _DispatchCalls(TypedDict):
    custom_op_modes: list[bool]
    fused: int
    stock_core: int
    norm: int


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    query_lens_cpu = torch.tensor(batch_spec.query_lens, dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(
        batch_spec.batch_size + 1,
        dtype=torch.int32,
    )
    query_start_loc_cpu[1:] = query_lens_cpu.cumsum(0)
    query_start_loc = query_start_loc_cpu.to(device=device)
    num_tokens = sum(batch_spec.query_lens)

    seq_lens_cpu = torch.tensor(batch_spec.seq_lens, dtype=torch.int32)
    seq_lens = seq_lens_cpu.to(device=device)
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i] for i in range(batch_spec.batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    # Mirror model_runner: is_prefilling = num_computed < num_prompt_tokens.
    # Chunked prefills still have prompt tokens beyond num_computed; decodes do not.
    num_prompt_tokens_cpu = torch.tensor(
        [
            context_lens[i] + batch_spec.query_lens[i] if batch_spec.query_lens[i] > 1 else context_lens[i]
            for i in range(batch_spec.batch_size)
        ],
        dtype=torch.int32,
    )
    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks,
        dtype=torch.int32,
        device=device,
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
        is_prefilling=is_prefilling,
    )


def _make_vllm_config(
    *,
    max_model_len: int = 8192,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    num_heads: int = 32,
    num_speculative_tokens: int = 0,
    mamba_cache_mode: str = "none",
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    prefill_context_parallel_size: int = 1,
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
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=prefill_context_parallel_size,
            tensor_parallel_size=1,
        ),
        model_config=model_config,
        additional_config=None,
    )


def _make_builder(
    *,
    device: torch.device,
    num_heads: int,
    num_speculative_tokens: int,
    mamba_cache_mode: str = "none",
    block_size: int = 16,
    num_speculative_blocks: int = 0,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    prefill_context_parallel_size: int = 1,
):
    vllm_config = _make_vllm_config(
        num_heads=num_heads,
        num_speculative_tokens=num_speculative_tokens,
        mamba_cache_mode=mamba_cache_mode,
        cudagraph_mode=cudagraph_mode,
        prefill_context_parallel_size=prefill_context_parallel_size,
    )
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    return AscendGDNAttentionMetadataBuilder(spec, ["layer0"], vllm_config, device)


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
        )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )
    return builder, common_attn_metadata, attn_metadata


def _assert_chunk_meta_matches_runtime(builder, chunk_meta, cu_seqlens: torch.Tensor) -> None:
    hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
    if hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
        gdn_num_heads = (
            hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
        )
    else:
        gdn_num_heads = builder.vllm_config.model_config.get_num_attention_heads(builder.vllm_config.parallel_config)
    cumsum_chunks = max(
        1,
        ascend_gdn_attn_builder._GDN_CUMSUM_WORKING_SET // (gdn_num_heads * ascend_gdn_attn_builder._GDN_CHUNK_SIZE),
    )
    cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()
    sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    assert chunk_meta.num_decodes == (sequence_lengths == 1).sum().item()
    assert torch.equal(
        chunk_meta.chunk_indices_chunk64,
        runtime_prepare_chunk_indices(cu_seqlens, ascend_gdn_attn_builder._GDN_CHUNK_SIZE),
    )
    assert torch.equal(
        chunk_meta.chunk_offsets_chunk64,
        runtime_prepare_chunk_offsets(cu_seqlens, ascend_gdn_attn_builder._GDN_CHUNK_SIZE),
    )
    assert torch.equal(
        chunk_meta.update_chunk_offsets_chunk64,
        runtime_prepare_update_chunk_offsets(
            cu_seqlens,
            ascend_gdn_attn_builder._GDN_CHUNK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.final_chunk_indices_chunk64,
        runtime_prepare_final_chunk_indices(
            cu_seqlens,
            ascend_gdn_attn_builder._GDN_CHUNK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.chunk_indices_large_block,
        runtime_prepare_chunk_indices(
            cu_seqlens,
            ascend_gdn_attn_builder._GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        ),
    )
    assert torch.equal(
        chunk_meta.block_indices_cumsum,
        runtime_prepare_chunk_indices(
            cu_seqlens,
            cumsum_chunk_size,
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


def test_ascend_gdn_attention_uses_ascend_backend():
    assert AscendGatedDeltaNetAttention.get_attn_backend(object()) is AscendGDNAttentionBackend
    assert AscendGDNAttentionBackend.get_builder_cls() is AscendGDNAttentionMetadataBuilder


def test_rocm_named_upstream_hook_delegates_to_ascend_implementation():
    tensors = [torch.empty(0) for _ in range(4)]
    captured = []
    attention = SimpleNamespace(_forward_core_ascend=lambda *args: captured.extend(args))

    AscendGatedDeltaNetAttention._forward_core_rocm(attention, *tensors)

    assert captured == tensors


@pytest.mark.parametrize(
    (
        "gqa_interleaved_layout",
        "request_kind",
        "expected_fused_calls",
        "expected_stock_core_calls",
        "expected_norm_calls",
    ),
    [
        # A compatible packed-layout decode must use the fused kernel.
        pytest.param(False, "decode", 1, 0, 0, id="packed-layout-decode"),
        # Prefill passes the static layout gate but must fall back at runtime.
        pytest.param(False, "prefill", 0, 1, 1, id="packed-layout-prefill"),
        # An incompatible projection layout must stay on the stock path.
        pytest.param(True, "decode", 0, 1, 1, id="interleaved-layout-decode"),
    ],
)
def test_qwen35_fused_dispatch_requires_packed_layout_and_decode(
    monkeypatch: pytest.MonkeyPatch,
    gqa_interleaved_layout: bool,
    request_kind: str,
    expected_fused_calls: int,
    expected_stock_core_calls: int,
    expected_norm_calls: int,
):
    """Only a compatible packed-layout decode may invoke the fused kernel."""
    is_prefill = request_kind == "prefill"
    query_len = 4 if is_prefill else 1
    seq_len = query_len if is_prefill else 5
    _, _, layer_metadata = _build_attn_metadata(
        BatchSpec(
            seq_lens=[seq_len],
            query_lens=[query_len],
            name=f"qwen35_{request_kind}_layout_{gqa_interleaved_layout}",
        ),
        num_speculative_tokens=0,
        num_decode_draft_tokens_cpu=None,
    )
    assert layer_metadata.num_prefills == int(is_prefill)
    assert layer_metadata.num_decodes == int(not is_prefill)
    monkeypatch.setattr(
        gdn_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer0": layer_metadata}),
    )
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_GDN_DECODE_TILE_PIPELINE", "1")
    monkeypatch.setattr(
        gdn_module,
        "maybe_save_kv_layer_to_connector",
        lambda *args, **kwargs: None,
    )

    # Count calls at the fused and stock boundaries to verify dispatch only;
    # the underlying NPU kernel numerics are covered by operator tests.
    calls: _DispatchCalls = {
        "custom_op_modes": [],
        "fused": 0,
        "stock_core": 0,
        "norm": 0,
    }
    num_tokens = layer_metadata.num_actual_tokens
    projected_qkvz = torch.zeros((num_tokens, 12288), dtype=torch.bfloat16)
    projected_qkvz[:, 8192:] = 2
    projected_ba = torch.zeros((num_tokens, 64), dtype=torch.bfloat16)
    mixed_qkv = projected_qkvz[:, :8192]
    z = projected_qkvz[:, 8192:].reshape(num_tokens, 32, 128)
    b, a = projected_ba.chunk(2, dim=-1)

    class RecordingNorm:
        def __init__(self):
            self.weight = torch.empty(0)
            self.eps = 1e-6

        def __call__(self, core_attn_out, gate):
            calls["norm"] += 1
            return core_attn_out + gate

    def fake_fused_decode(**kwargs):
        calls["fused"] += 1
        return torch.full(
            (kwargs["projected_qkvz"].shape[0], kwargs["num_v_heads"], kwargs["head_dim"]),
            6,
            dtype=torch.bfloat16,
        )

    def record_stock_core(_mixed_qkv, _b, _a, core_attn_out):
        # This is a routing spy; stock-kernel numerics are covered separately.
        calls["stock_core"] += 1
        core_attn_out.fill_(3)

    def supports_packed_dispatch(hidden_states):
        return AscendGatedDeltaNetAttention._supports_qwen35_decode_tile_pipeline(attention, hidden_states)

    def try_fused_decode(qkvz, ba):
        return AscendGatedDeltaNetAttention._try_qwen35_decode_tile(attention, qkvz, ba)

    def fake_attention_core(*args, **kwargs):
        # Mirror the upstream custom-op callback selection on CPU.
        use_packed_callback = kwargs.get("use_aiter", args[5] if len(args) > 5 else False)
        calls["custom_op_modes"].append(use_packed_callback)
        if use_packed_callback:
            AscendGatedDeltaNetAttention._forward_core_ascend(attention, *args[:4])
        else:
            attention._forward_core(*args[:4])

    monkeypatch.setattr(gdn_module, "gdn_decode_tile", fake_fused_decode)
    monkeypatch.setattr(
        gdn_module,
        "fused_qkvzba_split_reshape_cat",
        lambda *_args, **_kwargs: (mixed_qkv, z, b, a),
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen_gdn_attention_core",
        fake_attention_core,
    )

    attention = SimpleNamespace(
        _supports_qwen35_decode_tile_pipeline=supports_packed_dispatch,
        _try_qwen35_decode_tile=try_fused_decode,
        _split_ba_for_tp=lambda ba: ba.chunk(2, dim=-1),
        _forward_core=record_stock_core,
        in_proj_qkvz=lambda hidden_states: (projected_qkvz, None),
        in_proj_ba=lambda hidden_states: (projected_ba, None),
        out_proj=lambda hidden_states: (hidden_states, None),
        norm=RecordingNorm(),
        conv1d=SimpleNamespace(weight=torch.zeros((8192, 1, 4), dtype=torch.bfloat16)),
        kv_cache=(torch.empty(0), torch.empty(0)),
        A_log=torch.empty(0),
        dt_bias=torch.empty(0),
        key_dim=2048,
        value_dim=4096,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        activation="silu",
        tp_size=1,
        gqa_interleaved_layout=gqa_interleaved_layout,
        prefix="layer0",
    )
    hidden_states = torch.zeros((num_tokens, 4096), dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)

    is_packed_dispatch_supported = AscendGatedDeltaNetAttention._supports_qwen35_decode_tile_pipeline(
        attention, hidden_states
    )
    assert is_packed_dispatch_supported == (not gqa_interleaved_layout)

    AscendGatedDeltaNetAttention.forward(attention, hidden_states, output)

    assert calls["custom_op_modes"] == [not gqa_interleaved_layout]
    assert calls["fused"] == expected_fused_calls
    assert calls["stock_core"] == expected_stock_core_calls
    assert calls["norm"] == expected_norm_calls
    expected_output = 6 if expected_fused_calls == 1 else 5
    assert torch.equal(output, torch.full_like(output, expected_output))


def test_qwen35_piecewise_decode_uses_real_metadata_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    """PIECEWISE pads projections but not per-request GDN metadata."""
    _, _, layer_metadata = _build_attn_metadata(
        BatchSpec(
            seq_lens=[5, 9],
            query_lens=[1, 1],
            name="piecewise_projection_padding",
        ),
        num_speculative_tokens=0,
        num_decode_draft_tokens_cpu=None,
    )
    monkeypatch.setattr(
        gdn_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer0": layer_metadata}),
    )

    captured = {}

    def fake_decode_tile(**kwargs):
        captured.update(kwargs)
        return torch.zeros(
            (kwargs["projected_qkvz"].shape[0], 32, 128),
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(gdn_module, "gdn_decode_tile", fake_decode_tile)
    attention = SimpleNamespace(
        key_dim=2048,
        value_dim=4096,
        num_v_heads=32,
        num_k_heads=16,
        head_k_dim=128,
        tp_size=1,
        conv1d=SimpleNamespace(weight=torch.zeros((8192, 1, 4), dtype=torch.bfloat16)),
        kv_cache=(torch.empty(0), torch.empty(0)),
        A_log=torch.empty(0),
        dt_bias=torch.empty(0),
        norm=SimpleNamespace(weight=torch.empty(0), eps=1e-6),
        prefix="layer0",
    )
    projected_qkvz = torch.zeros((8, 12288), dtype=torch.bfloat16)
    projected_ba = torch.zeros((8, 64), dtype=torch.bfloat16)

    result = AscendGatedDeltaNetAttention._try_qwen35_decode_tile(
        attention,
        projected_qkvz,
        projected_ba,
    )

    assert result is not None
    assert result.shape == (2, 32, 128)
    assert captured["projected_qkvz"].shape == (2, 12288)
    assert captured["projected_ba"].shape == (2, 64)
    assert captured["cache_indices"].numel() == 2
    assert captured["query_start_loc"].numel() == 3


def test_qwen35_decode_pipeline_accepts_larger_batch_and_divisible_tp(
    monkeypatch: pytest.MonkeyPatch,
):
    batch = 32
    _, _, layer_metadata = _build_attn_metadata(
        BatchSpec(
            seq_lens=[5] * batch,
            query_lens=[1] * batch,
            name="decode_batch_32",
        ),
        num_speculative_tokens=0,
        num_decode_draft_tokens_cpu=None,
    )
    monkeypatch.setattr(
        gdn_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer0": layer_metadata}),
    )

    captured = {}

    def fake_decode_tile(**kwargs):
        captured.update(kwargs)
        return torch.zeros(
            (
                kwargs["projected_qkvz"].shape[0],
                kwargs["num_v_heads"],
                kwargs["head_dim"],
            ),
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(gdn_module, "gdn_decode_tile", fake_decode_tile)
    attention = SimpleNamespace(
        key_dim=2048,
        value_dim=4096,
        num_v_heads=32,
        num_k_heads=16,
        head_k_dim=128,
        tp_size=4,
        conv1d=SimpleNamespace(weight=torch.zeros((8192, 1, 4), dtype=torch.bfloat16)),
        kv_cache=(torch.empty(0), torch.empty(0)),
        A_log=torch.empty(0),
        dt_bias=torch.empty(0),
        norm=SimpleNamespace(weight=torch.empty(0), eps=1e-6),
        prefix="layer0",
    )

    result = AscendGatedDeltaNetAttention._try_qwen35_decode_tile(
        attention,
        torch.zeros((batch, 3072), dtype=torch.bfloat16),
        torch.zeros((batch, 16), dtype=torch.bfloat16),
    )

    assert result is not None
    assert result.shape == (batch, 8, 128)
    assert captured["cache_indices"].numel() == batch
    assert captured["num_qk_heads"] == 4
    assert captured["num_v_heads"] == 8


def test_qwen35_decode_pipeline_accepts_supported_dtypes_and_divisible_tp(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_GDN_DECODE_TILE_PIPELINE", "1")
    attention = SimpleNamespace(
        tp_size=1,
        in_proj_qkvz=object(),
        gqa_interleaved_layout=False,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        activation="silu",
    )

    for dtype in (torch.bfloat16, torch.float16):
        assert AscendGatedDeltaNetAttention._supports_qwen35_decode_tile_pipeline(
            attention,
            torch.empty((2, 4096), dtype=dtype),
        )
    assert not AscendGatedDeltaNetAttention._supports_qwen35_decode_tile_pipeline(
        attention,
        torch.empty((2, 4096), dtype=torch.float32),
    )
    attention.tp_size = 4
    assert AscendGatedDeltaNetAttention._supports_qwen35_decode_tile_pipeline(
        attention,
        torch.empty((32, 4096), dtype=torch.bfloat16),
    )
    attention.tp_size = 3
    assert not AscendGatedDeltaNetAttention._supports_qwen35_decode_tile_pipeline(
        attention,
        torch.empty((2, 4096), dtype=torch.bfloat16),
    )


def test_sequence_index_buffers_cover_spec_decode_when_cudagraph_disabled():
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
    )
    assert builder.spec_sequence_indices_cpu.numel() >= builder.vllm_config.scheduler_config.max_num_seqs

    spec_indices, non_spec_indices = builder._copy_sequence_indices_to_device(
        torch.tensor([True], dtype=torch.bool),
        num_spec_decodes=1,
    )

    assert torch.equal(spec_indices, torch.tensor([0]))
    assert non_spec_indices.numel() == 0


def _cache_index_first_column(cache_indices: torch.Tensor) -> torch.Tensor:
    if cache_indices.dim() == 1:
        return cache_indices
    return cache_indices[:, 0]


def _assert_non_spec_conv1d_args_match_metadata(attn_metadata) -> None:
    conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
    assert torch.equal(conv1d_meta.query_start_loc, attn_metadata.non_spec_query_start_loc)
    assert torch.equal(
        _cache_index_first_column(conv1d_meta.cache_indices),
        attn_metadata.non_spec_state_indices_tensor,
    )
    assert torch.equal(conv1d_meta.initial_state_mode, attn_metadata.has_initial_state)


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
def test_non_spec_prefill_metadata_matches_original_inputs_and_runtime_helpers(
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

    prefill_metadata = getattr(attn_metadata, "non_spec_prefill_metadata", None)
    assert prefill_metadata is not None
    assert prefill_metadata.causal_conv1d is not None
    assert prefill_metadata.chunk is not None

    _assert_non_spec_conv1d_args_match_metadata(attn_metadata)

    _assert_chunk_meta_matches_runtime(
        builder,
        prefill_metadata.chunk,
        attn_metadata.prefill_query_start_loc,
    )


def test_non_spec_prefill_metadata_uses_prefill_tail_for_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[5, 12, 9],
        query_lens=[1, 8, 4],
        name="decode_prefill_without_spec",
    )
    builder, _, attn_metadata = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=0,
        num_decode_draft_tokens_cpu=None,
    )

    assert attn_metadata.num_decodes == 1
    assert attn_metadata.num_prefills == 2
    assert torch.equal(
        attn_metadata.non_spec_query_start_loc,
        torch.tensor([0, 1, 9, 13], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.prefill_query_start_loc,
        torch.tensor([0, 8, 12], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.prefill_state_indices,
        torch.tensor([1, 2], dtype=torch.int32),
    )

    prefill_metadata = getattr(attn_metadata, "non_spec_prefill_metadata", None)
    assert prefill_metadata is not None
    decode_metadata = getattr(attn_metadata, "non_spec_decode_metadata", None)
    assert decode_metadata is not None
    assert torch.equal(
        decode_metadata.actual_seq_lengths,
        torch.tensor([0, 1], dtype=torch.int32),
    )
    conv1d_meta = prefill_metadata.causal_conv1d
    assert torch.equal(conv1d_meta.query_start_loc, torch.tensor([0, 1, 9, 13], dtype=torch.int32))
    assert torch.equal(_cache_index_first_column(conv1d_meta.cache_indices), torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(conv1d_meta.initial_state_mode, torch.tensor([True, True, True]))
    assert prefill_metadata.chunk.num_decodes == 0
    _assert_chunk_meta_matches_runtime(
        builder,
        prefill_metadata.chunk,
        attn_metadata.prefill_query_start_loc,
    )


def test_mixed_spec_prefill_chunk_metadata_preserves_single_token_count(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[1, 4, 8],
        query_lens=[1, 4, 8],
        name="mixed_spec_prefill_with_single_token_non_spec",
    )
    builder, _, attn_metadata = _build_attn_metadata(
        batch_spec,
        num_speculative_tokens=3,
        num_decode_draft_tokens_cpu=torch.tensor([-1, 3, -1], dtype=torch.int32),
    )

    assert attn_metadata.num_decodes == 0
    assert attn_metadata.num_prefills == 2
    assert torch.equal(
        attn_metadata.prefill_query_start_loc,
        torch.tensor([0, 1, 9], dtype=torch.int32),
    )
    chunk_metadata = attn_metadata.non_spec_prefill_metadata.chunk
    assert chunk_metadata.num_decodes == 1
    _assert_chunk_meta_matches_runtime(
        builder,
        chunk_metadata,
        attn_metadata.prefill_query_start_loc,
    )


def test_spec_conv1d_args_use_device_cache_and_accepted_tokens():
    batch_spec = BatchSpec(
        seq_lens=[4, 4],
        query_lens=[4, 4],
        name="spec_only_device_args",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
    )
    num_accepted_tokens = torch.tensor([2, 4], dtype=torch.int32)

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )

    spec_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
    query_start_loc = spec_conv1d_meta.query_start_loc
    assert torch.equal(query_start_loc, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert torch.equal(
        spec_conv1d_meta.cache_indices,
        torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32),
    )
    assert torch.equal(spec_conv1d_meta.num_accepted_tokens, num_accepted_tokens)
    assert torch.equal(
        attn_metadata.spec_decode_metadata.actual_seq_lengths,
        torch.tensor([0, 4, 4], dtype=torch.int32),
    )


def test_full_graph_spec_conv1d_args_keep_request_granularity():
    batch_spec = BatchSpec(
        seq_lens=[4, 4, 4],
        query_lens=[4, 4, 4],
        name="full_graph_spec_only_device_args",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    num_accepted_tokens = torch.tensor([2, 4, 3], dtype=torch.int32)

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=torch.tensor([3, 3, 3], dtype=torch.int32),
    )

    spec_conv1d_meta = attn_metadata.spec_decode_metadata.spec_causal_conv1d
    query_start_loc = spec_conv1d_meta.query_start_loc
    assert torch.equal(query_start_loc, torch.tensor([0, 4, 8, 12], dtype=torch.int32))
    assert query_start_loc.numel() == batch_spec.batch_size + 1
    assert spec_conv1d_meta.cache_indices.shape == (batch_spec.batch_size, 4)
    assert torch.equal(spec_conv1d_meta.cache_indices[:, 0], torch.tensor([10, 20, 30], dtype=torch.int32))
    assert torch.equal(spec_conv1d_meta.num_accepted_tokens, num_accepted_tokens)
    assert torch.equal(
        attn_metadata.spec_decode_metadata.actual_seq_lengths,
        torch.tensor([0, 4, 4, 4], dtype=torch.int32),
    )


def test_full_graph_spec_actual_seq_lengths_use_padded_builder_buffer():
    batch_spec = BatchSpec(
        seq_lens=[4, 4],
        query_lens=[4, 4],
        name="full_graph_padded_spec_actual_seq_lengths",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.num_reqs = 4
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=torch.tensor([2, 4], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )

    assert torch.equal(
        attn_metadata.spec_query_start_loc,
        torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32),
    )
    assert (
        attn_metadata.spec_decode_metadata.actual_seq_lengths.data_ptr() == builder.spec_actual_seq_lengths.data_ptr()
    )
    assert torch.equal(
        attn_metadata.spec_decode_metadata.actual_seq_lengths,
        torch.tensor([0, 4, 4, 0, 0], dtype=torch.int32),
    )


def test_full_graph_without_runtime_spec_resets_captured_spec_inputs():
    capture_batch = BatchSpec(
        seq_lens=[4, 4],
        query_lens=[4, 4],
        name="full_graph_spec_capture",
    )
    capture_common_metadata = create_common_attn_metadata(
        batch_spec=capture_batch,
        block_size=16,
        device=torch.device("cpu"),
    )
    capture_common_metadata.num_reqs = 4
    capture_common_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    captured_metadata = builder.build(
        0,
        capture_common_metadata,
        num_accepted_tokens=torch.tensor([2, 4], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )
    captured_spec_metadata = captured_metadata.spec_decode_metadata
    captured_conv1d_metadata = captured_spec_metadata.spec_causal_conv1d

    assert torch.count_nonzero(captured_conv1d_metadata.query_start_loc) > 0
    assert torch.count_nonzero(captured_spec_metadata.actual_seq_lengths) > 0

    replay_batch = BatchSpec(
        seq_lens=[1, 1, 0, 0],
        query_lens=[1, 1, 0, 0],
        name="full_graph_replay_without_spec",
    )
    replay_common_metadata = create_common_attn_metadata(
        batch_spec=replay_batch,
        block_size=16,
        device=torch.device("cpu"),
    )
    replay_metadata = builder.build(
        0,
        replay_common_metadata,
        num_accepted_tokens=torch.ones(4, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.full((4,), -1, dtype=torch.int32),
    )

    assert replay_metadata.spec_sequence_masks is None
    assert replay_metadata.spec_decode_metadata is None
    assert torch.equal(
        captured_conv1d_metadata.cache_indices,
        torch.full((4, 4), PAD_SLOT_ID, dtype=torch.int32),
    )
    assert torch.count_nonzero(captured_conv1d_metadata.query_start_loc) == 0
    assert torch.count_nonzero(captured_conv1d_metadata.num_accepted_tokens) == 0
    assert torch.count_nonzero(captured_spec_metadata.actual_seq_lengths) == 0


@pytest.mark.parametrize(
    ("num_speculative_tokens", "num_decode_draft_tokens_cpu"),
    [
        pytest.param(0, None, id="without_mtp"),
        pytest.param(
            3,
            torch.full((4,), -1, dtype=torch.int32),
            id="mtp_without_spec_requests",
        ),
    ],
)
def test_full_graph_non_spec_metadata_nulls_padded_state_indices(
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
):
    batch_spec = BatchSpec(
        seq_lens=[1, 1, 0, 0],
        query_lens=[1, 1, 0, 0],
        name="full_graph_padded_non_spec_actual_seq_lengths",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    # PCP leaves padded block-table rows untouched. Model the stale valid
    # state slots that can remain there after the preceding decode batch.
    common_attn_metadata.block_table_tensor[:, 0] = torch.tensor([10, 11, 98, 99])
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=num_speculative_tokens,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    builder.non_spec_state_indices_tensor.fill_(77)
    builder.non_spec_query_start_loc.fill_(77)
    builder.non_spec_actual_seq_lengths.fill_(77)

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )

    assert attn_metadata.num_decodes == 4
    assert attn_metadata.num_decode_tokens == 2
    assert torch.equal(
        attn_metadata.non_spec_query_start_loc,
        torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor([10, 11, 0, 0], dtype=torch.int32),
    )
    decode_metadata = attn_metadata.non_spec_decode_metadata
    conv1d_metadata = decode_metadata.causal_conv1d
    assert conv1d_metadata.query_start_loc.data_ptr() == attn_metadata.non_spec_query_start_loc.data_ptr()
    assert conv1d_metadata.cache_indices.data_ptr() == attn_metadata.non_spec_state_indices_tensor.data_ptr()
    assert decode_metadata.actual_seq_lengths.data_ptr() == builder.non_spec_actual_seq_lengths.data_ptr()
    assert torch.equal(
        decode_metadata.actual_seq_lengths,
        torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32),
    )


def test_causal_conv1d_cache_indices_use_device_block_table(monkeypatch: pytest.MonkeyPatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[4, 4],
        query_lens=[4, 4],
        name="device_block_table_source",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[40], [41]],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
    )

    attn_metadata = builder.build(0, common_attn_metadata)

    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor([40, 41], dtype=torch.int32),
    )
    conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
    assert torch.equal(conv1d_meta.query_start_loc, torch.tensor([0, 4, 8], dtype=torch.int32))
    assert torch.equal(_cache_index_first_column(conv1d_meta.cache_indices), torch.tensor([40, 41], dtype=torch.int32))
    assert torch.equal(conv1d_meta.initial_state_mode, torch.tensor([False, False]))


def test_pcp_prefill_initial_state_mode_is_built_in_metadata(monkeypatch: pytest.MonkeyPatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[1, 4],
        query_lens=[1, 4],
        name="pcp_decode_prefill",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        prefill_context_parallel_size=2,
    )

    with patch(
        "vllm_ascend.ops.gdn_attn_builder.get_pcp_group",
        return_value=SimpleNamespace(world_size=2, rank_in_group=1),
    ):
        attn_metadata = builder.build(0, common_attn_metadata)

    conv1d_meta = attn_metadata.non_spec_prefill_metadata.causal_conv1d
    assert torch.equal(
        conv1d_meta.initial_state_mode,
        torch.tensor([False, True]),
    )


def test_mamba_align_cache_indices_follow_device_seq_lens(monkeypatch: pytest.MonkeyPatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[1, 9],
        query_lens=[1, 1],
        name="align_device_seq_lens",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=4,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.arange(20, dtype=torch.int32).view(2, 10)
    common_attn_metadata._seq_lens_cpu = torch.tensor([5, 13], dtype=torch.int32)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        mamba_cache_mode="align",
        block_size=4,
        num_speculative_blocks=2,
    )

    attn_metadata = builder.build(0, common_attn_metadata)

    conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
    assert torch.equal(
        _cache_index_first_column(conv1d_meta.cache_indices),
        torch.tensor([0, 12], dtype=torch.int32),
    )


def test_builder_builds_prebuilt_chunk_metadata_with_prefill_query_start_loc(monkeypatch):
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

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=torch.ones(batch_spec.batch_size, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([-1, 3, -1, -1], dtype=torch.int32),
    )

    chunk_meta = attn_metadata.non_spec_prefill_metadata.chunk
    assert chunk_meta.chunk_indices_chunk64 is attn_metadata.chunk_indices
    assert chunk_meta.chunk_offsets_chunk64 is attn_metadata.chunk_offsets
    _assert_chunk_meta_matches_runtime(
        builder,
        chunk_meta,
        attn_metadata.prefill_query_start_loc,
    )
    assert chunk_meta.cu_seqlens_host == tuple(attn_metadata.prefill_query_start_loc.to(torch.int64).tolist())
    expected_chunk_indices = runtime_prepare_chunk_indices(
        attn_metadata.prefill_query_start_loc,
        ascend_gdn_attn_builder._GDN_CHUNK_SIZE,
    )
    assert chunk_meta.chunk_indices_chunk64_host == tuple(expected_chunk_indices.to(torch.int64).reshape(-1).tolist())


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

    assert getattr(attn_metadata, "non_spec_prefill_metadata", None) is None
    if batch_spec.name == "decode_only":
        decode_metadata = getattr(attn_metadata, "non_spec_decode_metadata", None)
        assert decode_metadata is not None
        assert torch.equal(
            decode_metadata.actual_seq_lengths,
            torch.tensor([0, 1, 1, 1], dtype=torch.int32),
        )
    else:
        spec_decode_metadata = getattr(attn_metadata, "spec_decode_metadata", None)
        assert spec_decode_metadata is not None
        assert torch.equal(
            spec_decode_metadata.actual_seq_lengths,
            torch.tensor([0, 4, 4], dtype=torch.int32),
        )
