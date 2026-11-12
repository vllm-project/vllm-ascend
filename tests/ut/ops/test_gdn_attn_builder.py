# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.third_party.flash_linear_attention.ops import index as _fla_index
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend, GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.spec_decode import adaptive_verification
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
    AscendGDNAttentionBackend310,
    GDNAttentionMetadataBuilder310,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops import gdn as ascend_gdn
from vllm_ascend.ops import gdn_attn_builder as ascend_gdn_attn_builder
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
    AscendGDNFusedAttentionBackend,
    AscendGDNHostMetadataBackend,
    AscendGDNHostMetadataBuilder,
)
from vllm_ascend.ops.kda_pcp import KimiKDAHostMetadataBackend
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


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
@pytest.mark.parametrize("pcp_size,pcp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("has_initial", [[False], [True], [False, False], [True, False, True]])
def test_prefill_preserves_request_convolution_history(monkeypatch, builder_cls, pcp_size, pcp_rank, has_initial):
    builder = builder_cls.__new__(builder_cls)
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp_size))
    monkeypatch.setattr(ascend_gdn_attn_builder, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=pcp_rank))
    num_decodes = int(len(has_initial) == 3)
    starts = torch.arange(len(has_initial) + 1, dtype=torch.int32) * 4
    initial = torch.tensor(has_initial)
    metadata = SimpleNamespace(
        num_prefills=len(has_initial) - num_decodes,
        non_spec_query_start_loc=starts,
        prefill_query_start_loc=starts[num_decodes:] - starts[num_decodes],
        has_initial_state=initial,
    )

    builder._attach_non_spec_prefill_metadata(metadata, object(), torch.arange(len(has_initial)))

    # Fresh requests ignore reused cache blocks on every rank. Continuations
    # and mixed decode rows must retain their actual request history.
    conv = metadata.non_spec_prefill_metadata.causal_conv1d
    torch.testing.assert_close(conv.initial_state_mode, torch.tensor(has_initial))
    torch.testing.assert_close(initial, torch.tensor(has_initial))


@pytest.mark.parametrize(
    "values",
    [[], [False], [True], [False] * 8, [True] * 8, [True, False, True, False, False, True]],
)
@pytest.mark.parametrize("strided", [False, True])
def test_stable_argsort_boolean_partition(values, strided):
    mask = torch.tensor(values, dtype=torch.bool)
    if strided:
        mask = torch.stack((mask, mask), dim=1)[:, 0]
    expected = sorted(range(len(values)), key=values.__getitem__)

    with patch.object(ascend_gdn_attn_builder.torch, "argsort", wraps=torch.argsort) as argsort:
        indices = ascend_gdn_attn_builder._stable_argsort_for_npu(mask)

    assert indices.tolist() == expected
    assert indices.dtype == torch.int64
    # Stable ordering alone also passes with the old AiCPU fallback. Check the
    # dispatch dtype to prevent reintroducing integer sorting for boolean masks.
    assert argsort.call_args.args[0].dtype == torch.float32
    assert argsort.call_args.kwargs["stable"] is True


def test_stable_argsort_preserves_integer_precision():
    # These distinct integers collapse to the same float32 value. Only boolean
    # masks may be cast; other callers must retain their original precision.
    values = torch.tensor([2**40 + 1, 2**40, 2**40 + 1], dtype=torch.int64)
    indices = ascend_gdn_attn_builder._stable_argsort_for_npu(values)
    assert indices.tolist() == [1, 0, 2]


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
    enable_adaptive_verification: bool = False,
    mamba_cache_mode: str = "none",
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
):
    speculative_config = None
    if num_speculative_tokens > 0:
        speculative_config = SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens,
            parallel_drafting=False,
            enable_adaptive_verification=enable_adaptive_verification,
            use_eagle_block_drop=lambda: False,
        )

    model_config = SimpleNamespace(max_model_len=max_model_len, hf_text_config=SimpleNamespace())
    model_config.get_num_attention_heads = lambda parallel_config: num_heads

    return SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
        compilation_config=SimpleNamespace(
            cudagraph_mode=cudagraph_mode,
            max_cudagraph_capture_size=None,
        ),
        speculative_config=speculative_config,
        num_hash_lookahead_tokens=0,
        use_v2_model_runner=False,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
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
    enable_adaptive_verification: bool = False,
    builder_cls=AscendGDNAttentionMetadataBuilder,
    mamba_cache_mode: str = "none",
    block_size: int = 16,
    num_speculative_blocks: int = 0,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
):
    vllm_config = _make_vllm_config(
        num_heads=num_heads,
        num_speculative_tokens=num_speculative_tokens,
        enable_adaptive_verification=enable_adaptive_verification,
        mamba_cache_mode=mamba_cache_mode,
        cudagraph_mode=cudagraph_mode,
    )
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    return builder_cls(spec, ["layer0"], vllm_config, device)


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
    linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
    if isinstance(linear_attn_config, dict) and linear_attn_config.get("num_heads") is not None:
        gdn_num_heads = linear_attn_config["num_heads"] // builder.vllm_config.parallel_config.tensor_parallel_size
    elif hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
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


def test_kimi_chunk_metadata_uses_linear_attention_head_count() -> None:
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=128,
        num_speculative_tokens=0,
    )
    builder.vllm_config.model_config.hf_text_config = SimpleNamespace(
        linear_attn_config={"num_heads": 32},
    )
    cu_seqlens = torch.tensor([0, 130], dtype=torch.int32)

    chunk_meta = ascend_gdn_attn_builder._build_non_spec_chunked_prefill_metadata(
        builder,
        cu_seqlens,
        torch.device("cpu"),
    )

    _assert_chunk_meta_matches_runtime(builder, chunk_meta, cu_seqlens)


def test_ascend_gdn_attention_uses_ascend_backend():
    from vllm_ascend.models.glm5next.kda import Glm5NextGDNHostMetadataBackend, Glm5NextLinearAttention
    from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

    fused_backend = AscendGatedDeltaNetAttention.get_attn_backend(object())  # type: ignore[arg-type]
    kimi_backend = AscendKimiK3DeltaAttention.get_attn_backend(object())  # type: ignore[arg-type]
    glm_backend = Glm5NextLinearAttention.get_attn_backend(object())  # type: ignore[arg-type]
    assert fused_backend is AscendGDNFusedAttentionBackend
    assert kimi_backend is KimiKDAHostMetadataBackend
    assert issubclass(kimi_backend, AscendGDNHostMetadataBackend)
    assert glm_backend is Glm5NextGDNHostMetadataBackend
    assert Glm5NextLinearAttention.use_contiguous_conv_indices
    assert AscendGDNAttentionBackend.get_builder_cls() is AscendGDNAttentionMetadataBuilder


@pytest.mark.parametrize(
    "model_type,enabled,recover",
    [
        ("glm5_next", True, True),
        ("glm5_next_text", True, True),
        ("glm5_next_text", False, False),
        ("kimi_k3", True, False),
    ],
)
def test_host_backend_selects_recoverssm_only_for_opted_in_glm(model_type, enabled, recover):
    from vllm_ascend.models.glm5next.recoverssm_metadata import AscendKDARecoverSSMMetadataBuilder

    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        cache_config=SimpleNamespace(use_replayssm=enabled, use_kda_recoverssm=False),
    )
    expected = AscendKDARecoverSSMMetadataBuilder if recover else AscendGDNHostMetadataBuilder
    with set_current_vllm_config(config):
        assert AscendGDNHostMetadataBackend.get_builder_cls() is expected
        assert not config.cache_config.use_kda_recoverssm


def test_fused_backend_selects_host_metadata_only_when_supported(monkeypatch):
    monkeypatch.setattr(
        AscendGatedDeltaNetAttention,
        "_supports_host_metadata_prefill",
        lambda: True,
    )
    monkeypatch.setattr(
        ascend_gdn_attn_builder,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    assert AscendGDNFusedAttentionBackend.get_builder_cls() is AscendGDNHostMetadataBuilder

    monkeypatch.setattr(
        ascend_gdn_attn_builder,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    assert AscendGDNFusedAttentionBackend.get_builder_cls() is AscendGDNAttentionMetadataBuilder
    assert AscendGDNHostMetadataBackend.get_builder_cls() is AscendGDNHostMetadataBuilder

    monkeypatch.setattr(
        AscendGatedDeltaNetAttention,
        "_supports_host_metadata_prefill",
        lambda: False,
    )
    monkeypatch.setattr(
        ascend_gdn_attn_builder,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    assert AscendGDNFusedAttentionBackend.get_builder_cls() is AscendGDNAttentionMetadataBuilder


def test_fla_prefill_supports_host_metadata(monkeypatch):
    monkeypatch.setattr(
        ascend_gdn,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda _capability: True),
    )
    monkeypatch.setattr(ascend_gdn, "_get_fla_gdn_prefill_op", lambda: object())
    monkeypatch.setattr(AscendGatedDeltaNetAttention, "_probe_fused_chunk", lambda: False)

    assert AscendGatedDeltaNetAttention._supports_host_metadata_prefill()


def test_host_metadata_skips_unused_chunk_helpers(monkeypatch):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        builder_cls=AscendGDNHostMetadataBuilder,
    )
    cu_seqlens = torch.tensor([0, 130], dtype=torch.int32)
    for helper_name in (
        "prepare_chunk_offsets",
        "prepare_update_chunk_offsets",
        "prepare_final_chunk_indices",
    ):
        monkeypatch.setattr(
            ascend_gdn_attn_builder,
            helper_name,
            lambda *args, _name=helper_name, **kwargs: pytest.fail(f"unexpected {_name} call"),
        )
    original_prepare_indices = ascend_gdn_attn_builder.prepare_chunk_indices

    def prepare_indices_once(cu, chunk_size):
        assert chunk_size == ascend_gdn_attn_builder._GDN_CHUNK_SIZE
        return original_prepare_indices(cu, chunk_size)

    monkeypatch.setattr(ascend_gdn_attn_builder, "prepare_chunk_indices", prepare_indices_once)
    chunk_meta = ascend_gdn_attn_builder._build_non_spec_chunked_prefill_metadata(
        builder,
        cu_seqlens,
        torch.device("cpu"),
        build_device_metadata=False,
    )

    assert chunk_meta.chunk_indices_chunk64 is None
    assert chunk_meta.chunk_offsets_chunk64 is None
    assert chunk_meta.update_chunk_offsets_chunk64 is None
    assert chunk_meta.final_chunk_indices_chunk64 is None
    assert chunk_meta.chunk_indices_large_block is None
    assert chunk_meta.block_indices_cumsum is None
    expected = original_prepare_indices(cu_seqlens, ascend_gdn_attn_builder._GDN_CHUNK_SIZE)
    assert chunk_meta.chunk_indices_chunk64_host == tuple(expected.to(torch.int64).flatten().tolist())


@pytest.mark.parametrize("has_pcp_context", [False, True])
def test_host_builder_exposes_host_descriptors_without_device_chunks(has_pcp_context):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        builder_cls=AscendGDNHostMetadataBuilder,
    )
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[130], query_lens=[130]),
        block_size=16,
        device=torch.device("cpu"),
    )
    context = object() if has_pcp_context else None
    common_attn_metadata.pcp_context = context
    attn_metadata = builder.build(0, common_attn_metadata)
    assert attn_metadata.pcp_context is context
    chunk_meta = attn_metadata.non_spec_prefill_metadata.chunk
    assert chunk_meta.chunk_indices_chunk64 is None
    assert chunk_meta.chunk_offsets_chunk64 is None
    assert chunk_meta.chunk_indices_chunk64_host
    assert attn_metadata.chunk_indices is None
    assert attn_metadata.chunk_offsets is None


@pytest.mark.parametrize("creates_checkpoint", [False, True])
def test_host_builder_materializes_device_chunks_for_checkpoint_enabled_prefill(monkeypatch, creates_checkpoint):
    _patch_missing_runtime_cdiv(monkeypatch)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        builder_cls=AscendGDNHostMetadataBuilder,
    )
    builder._ascend_gdn_checkpoints = True
    checkpoint = (
        SimpleNamespace(checkpoint_offsets=torch.tensor([64]), state_indices=torch.tensor([NULL_BLOCK_ID]))
        if creates_checkpoint
        else None
    )
    builder.checkpoint_builder = SimpleNamespace(build=Mock(return_value=checkpoint))
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[130], query_lens=[130]),
        block_size=16,
        device=torch.device("cpu"),
    )
    metadata = builder.build(0, common, num_actual_reqs=1)
    chunk = metadata.non_spec_prefill_metadata.chunk
    for field in (
        "chunk_indices_chunk64",
        "chunk_offsets_chunk64",
        "chunk_indices_large_block",
        "block_indices_cumsum",
    ):
        assert getattr(chunk, field) is not None
    assert metadata.checkpoint is checkpoint
    if checkpoint is not None:
        assert checkpoint.checkpoint_offsets.tolist() == [0]
    builder.checkpoint_builder.build.assert_called_once_with(common, [0])


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


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_dspark_target_reorder_threshold_includes_base_token_regardless_of_anchor(
    sample_from_anchor: bool,
):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
    )
    builder.vllm_config.speculative_config.method = "dspark"
    builder.vllm_config.speculative_config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(sample_from_anchor=sample_from_anchor),
    )

    builder._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    assert builder.reorder_batch_threshold == 8


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
    _assert_chunk_meta_matches_runtime(
        builder,
        prefill_metadata.chunk,
        attn_metadata.prefill_query_start_loc,
    )


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY])
@pytest.mark.parametrize("cache_mode,expected_slots", [("none", [12, 20]), ("align", [15, 23])])
@pytest.mark.parametrize("draft_count", [0, 3])
def test_spec_rows_preserve_previous_accepted_state(builder_cls, graph_mode, cache_mode, expected_slots, draft_count):
    query_len = draft_count + 1
    common = create_common_attn_metadata(BatchSpec([64, 64], [query_len, query_len]), 16, torch.device("cpu"))
    common.block_table_tensor = torch.tensor([list(range(10, 18)), list(range(20, 28))], dtype=torch.int32)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        mamba_cache_mode=cache_mode,
        enable_adaptive_verification=True,
        num_speculative_blocks=3,
        cudagraph_mode=graph_mode,
        builder_cls=builder_cls,
    )
    # Include one-token capture rows, before any draft tokens are admitted.
    captured = builder.build_for_cudagraph_capture(common)
    assert AscendGDNAttentionBackend.supports_device_cpu_query_lens_mismatch()
    assert builder.get_varlen_cudagraph_max_query_len(builder.vllm_config, builder.kv_cache_spec) == 4
    assert captured.num_spec_decodes == 2
    assert captured.num_decodes == 0

    accepted = torch.tensor([3, 1], dtype=torch.int32)
    with patch.object(
        ascend_gdn_attn_builder,
        "_build_actual_seq_lengths",
        side_effect=AssertionError("adaptive recurrent consumes device query boundaries directly"),
    ):
        metadata = builder.build(0, common, accepted, torch.full((2,), draft_count, dtype=torch.int32))
    assert metadata.num_spec_decodes == 2
    assert metadata.num_decodes == 0
    assert metadata.non_spec_state_indices_tensor is None
    assert metadata.num_accepted_tokens.tolist() == [3, 1]
    slots = metadata.spec_state_indices_tensor
    assert slots[torch.arange(2), accepted - 1].tolist() == expected_slots
    assert metadata.spec_decode_metadata.actual_seq_lengths is None
    assert metadata.spec_decode_metadata.use_upstream_recurrent
    assert metadata.spec_decode_metadata.use_upstream_conv
    assert metadata.spec_decode_metadata.spec_causal_conv1d.query_start_loc.tolist() == [0, query_len, 2 * query_len]
    if graph_mode == CUDAGraphMode.NONE:
        # Pure speculative rows reuse views rather than materializing a gather.
        assert metadata.num_accepted_tokens.data_ptr() == accepted.data_ptr()
        if cache_mode == "none":
            assert slots.data_ptr() == common.block_table_tensor.data_ptr()


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
@pytest.mark.parametrize(
    "draft_count,host_lens,device_queries,expected_queries,expected_lens",
    [
        (9, [5, 5, 2], [0, 1, 10, 12], [0, 1, 10, 10], [0, 1, 9, 0]),
        (0, [1, 1, 2], [0, 1, 2, 4], [0, 1, 2, 2], [0, 1, 1, 0]),
    ],
)
def test_adaptive_gdn_padding_preserves_device_query_boundaries(
    builder_cls, draft_count, host_lens, device_queries, expected_queries, expected_lens
):
    common = create_common_attn_metadata(BatchSpec([64, 64, 0], host_lens), 16, torch.device("cpu"))
    common.query_start_loc = torch.tensor(device_queries, dtype=torch.int32)
    common.block_table_tensor = torch.arange(30, dtype=torch.int32).reshape(3, 10)
    original_queries = common.query_start_loc.clone()
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=9,
        enable_adaptive_verification=True,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        builder_cls=builder_cls,
    )
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=torch.tensor([2, 3, 1], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([draft_count, draft_count, -1], dtype=torch.int32),
        num_actual_reqs=2,
    )
    # The host's [5, 5] split must not replace the admitted [1, 9] split.
    assert metadata.spec_query_start_loc.tolist() == expected_queries
    assert metadata.spec_decode_metadata.actual_seq_lengths is None
    assert ascend_gdn_attn_builder._build_actual_seq_lengths(metadata.spec_query_start_loc, 3).tolist() == expected_lens
    assert metadata.num_accepted_tokens.tolist() == [2, 3, 1]
    assert metadata.num_spec_decode_tokens == expected_queries[-1]
    assert torch.equal(common.query_start_loc, original_queries)


@pytest.mark.parametrize("num_actual_reqs,shares_device_boundaries", [(2, True), (None, False)])
def test_adaptive_gdn_padding_reuses_device_boundaries_when_suffix_is_unpadded(
    num_actual_reqs, shares_device_boundaries
):
    common = create_common_attn_metadata(BatchSpec([64, 64, 0], [5, 5, 2]), 16, torch.device("cpu"))
    common.query_start_loc = torch.tensor([0, 1, 10, 12], dtype=torch.int32)
    original = common.query_start_loc.clone()

    result = ascend_gdn_attn_builder._remove_spec_graph_padding_queries(
        common,
        torch.tensor([9, 9, -1], dtype=torch.int32),
        num_actual_reqs=num_actual_reqs,
    )

    assert (result.query_start_loc.data_ptr() == common.query_start_loc.data_ptr()) is shares_device_boundaries
    assert result.query_start_loc_cpu[-1] == 10
    torch.testing.assert_close(common.query_start_loc, original)


def test_310p_does_not_advertise_adaptive_recurrent_graphs():
    assert not AscendGDNAttentionBackend310.supports_device_cpu_query_lens_mismatch()
    assert GDNAttentionMetadataBuilder310.get_varlen_cudagraph_max_query_len(None, None) is None


@pytest.mark.parametrize(
    "backend_cls", [AscendGDNAttentionBackend, AscendGDNFusedAttentionBackend, AscendGDNAttentionBackend310]
)
@pytest.mark.parametrize("host_metadata", [False, True])
@pytest.mark.parametrize("num_steps", [3, 9])
def test_adaptive_gdn_real_initialization_gate(monkeypatch, backend_cls, host_metadata, num_steps):
    """Exercise both upstream capability gates, even before vLLM #60229."""
    # New vLLM versions supply these hooks on the parent classes. Simulate
    # the PR's pinned revision so inheritance cannot conceal a missing hook.
    monkeypatch.setattr(GDNAttentionBackend, "supports_device_cpu_query_lens_mismatch", classmethod(lambda cls: False))
    monkeypatch.setattr(
        GDNAttentionMetadataBuilder, "get_varlen_cudagraph_max_query_len", classmethod(lambda cls, config, spec: None)
    )
    monkeypatch.setattr(ascend_gdn_attn_builder, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(AscendGatedDeltaNetAttention, "_supports_host_metadata_prefill", lambda: host_metadata)
    builder_cls = backend_cls.get_builder_cls()
    config = _make_vllm_config(num_speculative_tokens=num_steps, enable_adaptive_verification=True)
    config.use_v2_model_runner = True
    group = AttentionGroup(
        backend=backend_cls,
        layer_names=["layer0"],
        kv_cache_spec=Mock(spec=MambaSpec),
        kv_cache_group_id=0,
        metadata_builders=[builder_cls.__new__(builder_cls)],
    )
    kwargs = dict(
        enable_adaptive_verification=True,
        attn_groups=[[group]],
        req_states=SimpleNamespace(num_speculative_steps=num_steps),
        query_start_loc=torch.zeros(3, dtype=torch.int32),
        num_bonus_tokens=1,
        max_total_logits=32,
        vllm_config=config,
    )
    # Only manager construction is stubbed; the initializer and both capability
    # checks run unchanged against the real Ascend backend and builder.
    with patch.object(adaptive_verification, "AdaptiveVerificationManager") as manager:
        if backend_cls is AscendGDNAttentionBackend310:
            with pytest.raises(ValueError, match="trims verification requests on device"):
                adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs)
            manager.assert_not_called()
            return
        assert adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs) is manager.return_value
        manager.assert_called_once()
        assert builder_cls.get_varlen_cudagraph_max_query_len(config, group.kv_cache_spec) == num_steps + 1
        kwargs["req_states"].num_speculative_steps += 1
        with pytest.raises(ValueError, match="per-request query lengths vary"):
            adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs)
        assert manager.call_count == 1


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
def test_varlen_graph_capability_requires_adaptive_recurrent_path(builder_cls, adaptive):
    config = _make_vllm_config(num_speculative_tokens=3, enable_adaptive_verification=adaptive)
    # The supported bound follows Ascend's state-table width, including when
    # a caller asks before MRV2 configuration validation has completed.
    config.speculative_config.parallel_drafting = True
    assert builder_cls.get_varlen_cudagraph_max_query_len(config, None) == (4 if adaptive else None)


def test_adaptive_mixed_batch_rejects_cpu_fused_layout():
    builder = AscendGDNHostMetadataBuilder.__new__(AscendGDNHostMetadataBuilder)
    builder._spec_conv_indices = torch.empty(3, dtype=torch.int32)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    builder.kv_cache_spec = SimpleNamespace(num_speculative_blocks=3)
    builder.num_spec = 3
    builder.speculative_config = SimpleNamespace(enable_adaptive_verification=True)
    with patch.object(ascend_gdn_attn_builder, "prepare_gdn", side_effect=AssertionError("CPU layout is unsafe")):
        assert builder._build_fused_metadata(SimpleNamespace(num_reqs=3), None, None, 2, 3, ()) is None


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
@pytest.mark.parametrize("query_len", [1, 4])
def test_static_capture_keeps_native_recurrent_route(builder_cls, query_len):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        builder_cls=builder_cls,
    )
    common = create_common_attn_metadata(BatchSpec([64, 64], [query_len, query_len]), 16, torch.device("cpu"))
    metadata = builder.build_for_cudagraph_capture(common)
    if query_len == 1:
        assert metadata.num_spec_decodes == 0
        assert metadata.num_decodes == 2
        assert metadata.spec_decode_metadata is None
    else:
        assert metadata.num_spec_decodes == 2
        assert metadata.num_decodes == 0
        assert not metadata.spec_decode_metadata.use_upstream_recurrent


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
    # Mirror the runner: every common request-level array has the graph size,
    # even though only the first two rows describe real requests.
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(seq_lens=batch_spec.seq_lens + [0, 0], query_lens=batch_spec.query_lens + [0, 0]),
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [NULL_BLOCK_ID] * 4, [NULL_BLOCK_ID] * 4],
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
        num_accepted_tokens=torch.tensor([2, 4, 99, 99], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3, -1, -1], dtype=torch.int32),
        num_actual_reqs=batch_spec.batch_size,
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
    assert torch.equal(
        attn_metadata.spec_state_indices_tensor[2:],
        torch.full((2, 4), NULL_BLOCK_ID, dtype=torch.int32),
    )
    assert torch.equal(
        attn_metadata.num_accepted_tokens,
        torch.tensor([2, 4, 1, 1], dtype=torch.int32),
    )


def test_full_graph_non_spec_actual_seq_lengths_use_padded_builder_buffer():
    batch_spec = BatchSpec(
        seq_lens=[1, 1],
        query_lens=[1, 1],
        name="full_graph_padded_non_spec_actual_seq_lengths",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.num_reqs = 4
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    builder.spec_state_indices_tensor.fill_(77)

    attn_metadata = builder.build(
        0,
        common_attn_metadata,
        num_accepted_tokens=torch.ones(4, dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.full((4,), -1, dtype=torch.int32),
        num_actual_reqs=batch_spec.batch_size,
    )

    assert torch.equal(
        attn_metadata.non_spec_query_start_loc,
        torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32),
    )
    assert (
        attn_metadata.non_spec_decode_metadata.actual_seq_lengths.data_ptr()
        == builder.non_spec_actual_seq_lengths.data_ptr()
    )
    assert torch.equal(
        attn_metadata.non_spec_decode_metadata.actual_seq_lengths,
        torch.tensor([0, 1, 1, 0, 0], dtype=torch.int32),
    )
    assert attn_metadata.num_actual_tokens == 2
    assert attn_metadata.num_decodes == batch_spec.batch_size
    assert torch.equal(
        attn_metadata.non_spec_state_indices_tensor,
        torch.tensor([0, 1, NULL_BLOCK_ID, NULL_BLOCK_ID], dtype=torch.int32),
    )
    assert torch.all(builder.spec_state_indices_tensor[:4] == PAD_SLOT_ID)


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


def test_mamba_align_with_padded_block_table_rows(monkeypatch: pytest.MonkeyPatch):
    """vLLM #58737 derives num_reqs from the block table and asserts it equals
    seq_lens. Ascend's ``unpadded()`` keeps the graph-padded block-table rows, so
    the GDN builder must hand the helper only the real request rows."""
    _patch_missing_runtime_cdiv(monkeypatch)
    batch_spec = BatchSpec(
        seq_lens=[16, 16, 16, 16],
        query_lens=[1, 1, 0, 0],
        name="align_padded_block_table",
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=4,
        device=torch.device("cpu"),
    )
    common_attn_metadata.is_prefilling.zero_()
    # Graph-sized block table (4 rows) with only 2 real requests.
    common_attn_metadata.block_table_tensor = torch.arange(32, dtype=torch.int32).view(4, 8)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        mamba_cache_mode="align",
        block_size=4,
        num_speculative_blocks=2,
    )

    attn_metadata = builder.build(0, common_attn_metadata, num_actual_reqs=2)

    conv1d_meta = attn_metadata.non_spec_decode_metadata.causal_conv1d
    assert conv1d_meta.cache_indices.shape[0] == 2


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
    _assert_chunk_meta_matches_runtime(
        builder,
        chunk_metadata,
        attn_metadata.prefill_query_start_loc,
    )


@pytest.mark.parametrize(
    ("seq_len", "expected_decodes", "expected_prefills"),
    [
        pytest.param(1, 0, 1, id="first_token_stays_prefill"),
        pytest.param(17, 1, 0, id="block-size-plus-one-becomes-decode"),
    ],
)
def test_one_token_prefill_selection_respects_recurrent_state(
    monkeypatch: pytest.MonkeyPatch,
    seq_len: int,
    expected_decodes: int,
    expected_prefills: int,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[seq_len], query_lens=[1]),
        block_size=16,
        device=torch.device("cpu"),
    )
    # Model a prompt chunk explicitly. The helper normally classifies a
    # one-token row as decode when synthesizing test metadata.
    common_attn_metadata.is_prefilling = torch.tensor([True])
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
    )

    attn_metadata = builder.build(0, common_attn_metadata)

    assert common_attn_metadata.is_prefilling.tolist() == [True]
    assert attn_metadata.num_decodes == expected_decodes
    assert attn_metadata.num_prefills == expected_prefills


@pytest.mark.parametrize(
    "dcp_size,num_spec,context_len,mixed_spec,graph_mode",
    [
        (1, 3, 0, False, CUDAGraphMode.NONE),
        (1, 3, 384, False, CUDAGraphMode.NONE),
        (1, 5, 384, True, CUDAGraphMode.FULL_DECODE_ONLY),
        (16, 5, 0, False, CUDAGraphMode.NONE),
        (16, 5, 384, False, CUDAGraphMode.NONE),
        (16, 5, 384, False, CUDAGraphMode.FULL_DECODE_ONLY),
        (16, 5, 384, True, CUDAGraphMode.NONE),
        (16, 5, 384, True, CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, GDNAttentionMetadataBuilder310])
def test_spec_width_prompt_chunk_retains_prefill_metadata(
    monkeypatch: pytest.MonkeyPatch,
    builder_cls,
    dcp_size: int,
    num_spec: int,
    context_len: int,
    mixed_spec: bool,
    graph_mode: CUDAGraphMode,
):
    _patch_missing_runtime_cdiv(monkeypatch)
    width = num_spec + 1
    query_lens = [width, width] if mixed_spec else [width]
    seq_lens = [768 + width, context_len + width] if mixed_spec else [context_len + width]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=384,
        device=torch.device("cpu"),
    )
    common.is_prefilling = torch.tensor([False, True] if mixed_spec else [True])
    common.block_table_tensor = torch.arange(len(query_lens) * 10, dtype=torch.int32).view(-1, 10)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=num_spec,
        builder_cls=builder_cls,
        mamba_cache_mode="align",
        block_size=384,
        num_speculative_blocks=num_spec,
        cudagraph_mode=graph_mode,
    )
    builder.vllm_config.parallel_config.decode_context_parallel_size = dcp_size
    accepted = torch.tensor([2, 1] if mixed_spec else [1], dtype=torch.int32)
    metadata = builder.build(
        0,
        common,
        num_accepted_tokens=accepted,
        num_decode_draft_tokens_cpu=torch.tensor([num_spec, -1] if mixed_spec else [-1], dtype=torch.int32),
    )

    assert accepted.tolist() == ([2, 1] if mixed_spec else [1])
    # Prompt chunks stay prefills on every device/graph configuration.
    assert metadata.num_prefills == 1
    assert metadata.num_prefill_tokens == width
    assert metadata.num_spec_decodes == int(mixed_spec)
    assert metadata.prefill_has_initial_state.tolist() == [context_len > 0]
    expected_slot = (10 if mixed_spec else 0) + (seq_lens[-1] - 1) // 384
    assert metadata.prefill_state_indices.tolist() == [expected_slot]
    if mixed_spec:
        assert metadata.spec_sequence_masks.tolist() == [True, False]
        assert metadata.num_accepted_tokens.tolist() == [2]
    else:
        assert metadata.spec_sequence_masks is None
        assert metadata.num_accepted_tokens is None


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
def test_full_graph_without_runtime_spec_resets_captured_spec_inputs(builder_cls):
    capture_common_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[4, 4],
            query_lens=[4, 4],
            name="full_graph_spec_capture",
        ),
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
        builder_cls=builder_cls,
    )
    if builder_cls is AscendGDNHostMetadataBuilder:
        builder.vllm_config.model_config.dtype = torch.bfloat16
        builder.compilation_config.static_forward_context = {
            "layer0": SimpleNamespace(
                use_contiguous_conv_indices=True,
                use_shared_conv_output=True,
                local_projection_size=8,
            ),
        }
    captured_metadata = builder.build(
        0,
        capture_common_metadata,
        num_accepted_tokens=torch.tensor([2, 4], dtype=torch.int32),
        num_decode_draft_tokens_cpu=torch.tensor([3, 3], dtype=torch.int32),
    )
    captured_spec_metadata = captured_metadata.spec_decode_metadata
    captured_conv1d_metadata = captured_spec_metadata.spec_causal_conv1d

    if builder_cls is AscendGDNHostMetadataBuilder:
        assert captured_conv1d_metadata.cache_indices.is_contiguous()
        assert captured_conv1d_metadata.cache_indices.tolist() == [10, 20, NULL_BLOCK_ID, NULL_BLOCK_ID]
        captured_output = captured_conv1d_metadata.output_buffer
        assert captured_output is not None
        captured_output.fill_(float("nan"))
    assert torch.count_nonzero(captured_conv1d_metadata.query_start_loc) > 0
    assert torch.count_nonzero(captured_spec_metadata.actual_seq_lengths) > 0

    replay_common_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[1, 1, 0, 0],
            query_lens=[1, 1, 0, 0],
            name="full_graph_replay_without_spec",
        ),
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
    expected_shape = (4,) if builder_cls is AscendGDNHostMetadataBuilder else (4, 4)
    assert torch.equal(
        captured_conv1d_metadata.cache_indices,
        torch.full(expected_shape, PAD_SLOT_ID, dtype=torch.int32),
    )
    assert torch.count_nonzero(captured_conv1d_metadata.query_start_loc) == 0
    assert torch.count_nonzero(captured_conv1d_metadata.num_accepted_tokens) == 0
    assert torch.count_nonzero(captured_spec_metadata.actual_seq_lengths) == 0
    if builder_cls is AscendGDNHostMetadataBuilder:
        assert torch.count_nonzero(captured_output) == 0
        assert replay_metadata.non_spec_decode_metadata.causal_conv1d.output_buffer is captured_output


@pytest.mark.parametrize("num_actual_reqs", [None, 0])
@pytest.mark.parametrize("adaptive", [False, True])
def test_full_graph_idle_dummy_uses_zero_length_recurrent_metadata(num_actual_reqs, adaptive):
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[8, 8, 8, 8],
            query_lens=[0, 0, 0, 0],
            name="full_graph_idle_dummy",
        ),
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata.block_table_tensor[:, 0] = torch.tensor(
        [10, 11, 98, 99],
        dtype=torch.int32,
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        enable_adaptive_verification=adaptive,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    builder.spec_state_indices_tensor.fill_(77)
    builder.spec_query_start_loc.fill_(77)
    builder.non_spec_state_indices_tensor.fill_(77)
    builder.non_spec_query_start_loc.fill_(77)

    attn_metadata = builder.build(0, common_attn_metadata, num_actual_reqs=num_actual_reqs)

    assert attn_metadata.num_actual_tokens == 0
    assert attn_metadata.num_decode_tokens == 0
    assert torch.count_nonzero(attn_metadata.non_spec_query_start_loc) == 0
    assert torch.all(attn_metadata.non_spec_state_indices_tensor == NULL_BLOCK_ID)
    assert torch.count_nonzero(builder.spec_query_start_loc[:5]) == 0
    assert torch.all(builder.spec_state_indices_tensor[:4] == PAD_SLOT_ID)


@pytest.mark.parametrize(
    ("num_speculative_tokens", "num_decode_draft_tokens_cpu"),
    [
        pytest.param(0, None, id="without_spec_decode"),
        pytest.param(
            3,
            torch.full((4,), -1, dtype=torch.int32),
            id="spec_decode_without_runtime_spec_requests",
        ),
    ],
)
def test_full_graph_non_spec_metadata_nulls_padded_state_indices(
    num_speculative_tokens: int,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
):
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=BatchSpec(
            seq_lens=[1, 1, 0, 0],
            query_lens=[1, 1, 0, 0],
            name="full_graph_padded_non_spec_actual_seq_lengths",
        ),
        block_size=16,
        device=torch.device("cpu"),
    )
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
        torch.tensor(
            [10, 11, NULL_BLOCK_ID, NULL_BLOCK_ID],
            dtype=torch.int32,
        ),
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


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, GDNAttentionMetadataBuilder310])
@pytest.mark.parametrize("spec", [False, True])
def test_graph_attaches_once_after_padding_and_reuses_final_buffers(builder_cls, spec):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        builder_cls=builder_cls,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    is_common = builder_cls is ascend_gdn_attn_builder.AscendGDNAttentionMetadataBuilder
    assert (builder.spec_actual_seq_lengths is not None) == is_common
    assert (builder.non_spec_actual_seq_lengths is not None) == is_common
    pointers = []
    # Grow, shrink, and grow the logical batch while retaining the graph size.
    for count in (2, 1, 3, 2):
        width = 4 if spec else 1
        common = create_common_attn_metadata(
            BatchSpec(seq_lens=[16] * 4, query_lens=[width] * count + [0] * (4 - count)),
            block_size=16,
            device=torch.device("cpu"),
        )
        common.is_prefilling.zero_()
        common.block_table_tensor = torch.arange(16, dtype=torch.int32).view(4, 4) + 10
        with (
            patch.object(
                builder, "_attach_spec_decode_metadata", wraps=builder._attach_spec_decode_metadata
            ) as attach_spec,
            patch.object(
                builder, "_attach_non_spec_decode_metadata", wraps=builder._attach_non_spec_decode_metadata
            ) as attach_decode,
            patch.object(
                ascend_gdn_attn_builder,
                "_build_actual_seq_lengths",
                wraps=ascend_gdn_attn_builder._build_actual_seq_lengths,
            ) as lengths,
            patch.object(
                CommonAttentionMetadata,
                "compute_num_computed_tokens",
                side_effect=AssertionError("decode must not compute prefill context"),
            ),
        ):
            metadata = builder.build(
                0,
                common,
                num_actual_reqs=count,
                num_accepted_tokens=torch.tensor([2] * count + [99] * (4 - count), dtype=torch.int32),
                num_decode_draft_tokens_cpu=torch.tensor(
                    [3 if spec else -1] * count + [99] * (4 - count), dtype=torch.int32
                ),
            )
        assert attach_spec.call_count == attach_decode.call_count == 1
        assert lengths.call_count == int(is_common)
        if spec:
            conv = metadata.spec_decode_metadata.spec_causal_conv1d
            state_buffer = builder.spec_state_indices_tensor
            query_buffer = builder.spec_query_start_loc
            assert conv.num_accepted_tokens.tolist() == [2] * count + [1] * (4 - count)
            assert conv.num_accepted_tokens.data_ptr() == builder.num_accepted_tokens.data_ptr()
            actual_lengths = metadata.spec_decode_metadata.actual_seq_lengths
        else:
            state_buffer = builder.non_spec_state_indices_tensor
            query_buffer = builder.non_spec_query_start_loc
            if not is_common:
                assert metadata.non_spec_decode_metadata is None
                assert metadata.non_spec_query_start_loc.data_ptr() == query_buffer.data_ptr()
                assert metadata.non_spec_state_indices_tensor.data_ptr() == state_buffer.data_ptr()
                continue
            conv = metadata.non_spec_decode_metadata.causal_conv1d
            actual_lengths = metadata.non_spec_decode_metadata.actual_seq_lengths
        assert conv.cache_indices.data_ptr() == state_buffer.data_ptr()
        assert conv.query_start_loc.data_ptr() == query_buffer.data_ptr()
        assert conv.query_start_loc.tolist() == [i * width for i in range(count + 1)] + [count * width] * (4 - count)
        pointers.append((conv.cache_indices.data_ptr(), conv.query_start_loc.data_ptr()))
        if is_common:
            assert actual_lengths.tolist() == [0] + [width] * count + [0] * (4 - count)
        else:
            assert actual_lengths is None
    assert len(set(pointers)) <= 1


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, GDNAttentionMetadataBuilder310])
@pytest.mark.parametrize("mixed_spec", [False, True])
def test_prefill_builds_only_consumed_kernel_metadata(builder_cls, mixed_spec):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3,
        builder_cls=builder_cls,
    )
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[12, 3, 9] if mixed_spec else [3, 9], query_lens=[4, 3, 5] if mixed_spec else [3, 5]),
        block_size=16,
        device=torch.device("cpu"),
    )
    common.block_table_tensor = torch.arange(common.num_reqs * 4, dtype=torch.int32).view(-1, 4)
    common.is_prefilling = torch.tensor([False, True, True] if mixed_spec else [True, True])
    with (
        patch.object(
            ascend_gdn_attn_builder,
            "_build_non_spec_chunked_prefill_metadata",
            wraps=ascend_gdn_attn_builder._build_non_spec_chunked_prefill_metadata,
        ) as chunks,
        patch.object(
            ascend_gdn_attn_builder,
            "_build_actual_seq_lengths",
            wraps=ascend_gdn_attn_builder._build_actual_seq_lengths,
        ) as lengths,
    ):
        metadata = builder.build(
            0,
            common,
            num_accepted_tokens=torch.tensor([2, 1, 1]) if mixed_spec else None,
            num_decode_draft_tokens_cpu=torch.tensor([3, -1, -1]) if mixed_spec else None,
        )
    assert metadata.has_initial_state.tolist() == [False, True]
    assert metadata.prefill_has_initial_state.tolist() == [False, True]
    assert metadata.nums_dict is None
    assert metadata.batch_ptr is None
    assert metadata.token_chunk_offset_ptr is None
    assert metadata.num_decodes == 0
    assert metadata.non_spec_decode_metadata is None
    assert lengths.call_count == int(
        mixed_spec and builder_cls is ascend_gdn_attn_builder.AscendGDNAttentionMetadataBuilder
    )
    if builder_cls is GDNAttentionMetadataBuilder310:
        assert chunks.call_count == 0
        assert metadata.chunk_indices is None and metadata.chunk_offsets is None
        assert metadata.non_spec_prefill_metadata is None
        assert metadata.non_spec_decode_metadata is None
        if mixed_spec:
            assert metadata.spec_decode_metadata.spec_causal_conv1d.num_accepted_tokens.tolist() == [2]
            assert metadata.spec_decode_metadata.actual_seq_lengths is None
    else:
        assert chunks.call_count == 1
        assert metadata.non_spec_prefill_metadata.chunk is not None


@pytest.mark.parametrize("live_requests", [1, 2])
@pytest.mark.parametrize("explicit_request_count", [False, True])
def test_spec_graph_fia_padding_refreshes_captured_buffers(live_requests, explicit_request_count):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    capture = create_common_attn_metadata(BatchSpec([8, 8], [8, 8]), 16, torch.device("cpu"))
    capture.block_table_tensor = torch.arange(16, dtype=torch.int32).view(2, 8) + 10
    captured = builder.build(0, capture, torch.ones(2, dtype=torch.int32), torch.tensor([7, 7]))
    stable = captured.spec_decode_metadata.spec_causal_conv1d
    pointers = [
        stable.query_start_loc.data_ptr(),
        stable.cache_indices.data_ptr(),
        stable.num_accepted_tokens.data_ptr(),
    ]

    lengths = [71, 34] if live_requests == 2 else [71, 0]
    replay = create_common_attn_metadata(BatchSpec(lengths, [8, 8]), 16, torch.device("cpu"))
    replay.block_table_tensor = torch.arange(16, dtype=torch.int32).view(2, 8) + 30
    drafts = torch.tensor([7, 7 if live_requests == 2 else -1])
    runtime = builder.build(
        0,
        replay,
        torch.tensor([1, 1], dtype=torch.int32),
        drafts,
        num_actual_reqs=live_requests if explicit_request_count else None,
    )

    assert runtime.num_prefills == 0
    assert runtime.num_spec_decodes == live_requests
    assert runtime.num_actual_tokens == 8 * live_requests
    actual = runtime.spec_decode_metadata.spec_causal_conv1d
    assert pointers == [
        actual.query_start_loc.data_ptr(),
        actual.cache_indices.data_ptr(),
        actual.num_accepted_tokens.data_ptr(),
    ]
    assert stable.query_start_loc.tolist() == [0, 8, 8 * live_requests]
    assert stable.cache_indices[0].tolist() == list(range(30, 38))
    if live_requests == 1:
        assert torch.all(stable.cache_indices[1] == NULL_BLOCK_ID)
    assert replay.query_start_loc_cpu.tolist() == [0, 8, 16]
    assert replay.num_actual_tokens == 16


def test_spec_graph_real_prefill_is_not_treated_as_padding():
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    common = create_common_attn_metadata(BatchSpec([71, 8], [8, 8]), 16, torch.device("cpu"))
    common.block_table_tensor = torch.arange(16, dtype=torch.int32).view(2, 8) + 10

    runtime = builder.build(0, common, torch.ones(2, dtype=torch.int32), torch.tensor([7, -1]))

    assert runtime.num_spec_decodes == 1
    assert runtime.num_prefills == 1


def test_host_conv_indices_are_shared_and_refreshed(monkeypatch):
    _patch_missing_runtime_cdiv(monkeypatch)
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=0,
        builder_cls=AscendGDNHostMetadataBuilder,
    )
    common = create_common_attn_metadata(BatchSpec([5, 12], [1, 8]), 16, torch.device("cpu"))
    common.block_table_tensor = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    unmarked = builder.build(0, common).non_spec_prefill_metadata.causal_conv1d.cache_indices
    assert unmarked.data_ptr() == common.block_table_tensor.data_ptr()
    builder.compilation_config.static_forward_context = {
        "layer0": SimpleNamespace(use_contiguous_conv_indices=True),
    }
    metadata = builder.build(0, common)
    indices = metadata.non_spec_prefill_metadata.causal_conv1d.cache_indices
    assert indices.is_contiguous()
    assert indices.tolist() == [10, 20]
    assert indices.data_ptr() == metadata.non_spec_decode_metadata.causal_conv1d.cache_indices.data_ptr()
    common.block_table_tensor.add_(30)
    refreshed = builder.build(0, common).non_spec_prefill_metadata.causal_conv1d.cache_indices
    assert refreshed.data_ptr() == indices.data_ptr()
    assert indices.tolist() == [40, 50]


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("num_speculative_tokens", [7, 8, 9])
def test_long_fixed_drafts_use_strided_recurrent_route(monkeypatch, builder_cls, device_type, num_speculative_tokens):
    monkeypatch.setattr(
        ascend_gdn_attn_builder, "get_current_hardware_profile", lambda: get_hardware_profile(device_type)
    )
    expected_fused = device_type == AscendDeviceType.A2 and num_speculative_tokens >= 8
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=num_speculative_tokens,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        builder_cls=builder_cls,
    )
    query_len = num_speculative_tokens + 1
    common = create_common_attn_metadata(BatchSpec([64, 64], [query_len, query_len]), 16, torch.device("cpu"))
    # The generic fixture has only ceil(64 / 16) == 4 cache columns.
    # Speculative GDN needs one state slot per verification token; retain
    # a wider physical row to exercise the strided state-table view too.
    state_slots = torch.arange(1, 2 * (query_len + 2) + 1, dtype=torch.int32).view(2, query_len + 2)
    common.block_table_tensor = state_slots
    metadata = builder.build_for_cudagraph_capture(common)
    assert metadata.num_spec_decodes == 2
    torch.testing.assert_close(metadata.spec_state_indices_tensor, state_slots[:, :query_len])
    assert metadata.spec_decode_metadata.use_upstream_recurrent is expected_fused
    assert not metadata.spec_decode_metadata.use_upstream_conv
    assert (metadata.spec_decode_metadata.actual_seq_lengths is None) is expected_fused


@pytest.mark.parametrize("builder_cls", [AscendGDNAttentionMetadataBuilder, AscendGDNHostMetadataBuilder])
@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("query_len", [1, 9])
def test_adaptive_recurrent_route_is_hardware_independent(monkeypatch, builder_cls, device_type, query_len):
    monkeypatch.setattr(
        ascend_gdn_attn_builder, "get_current_hardware_profile", lambda: get_hardware_profile(device_type)
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=8,
        enable_adaptive_verification=True,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        builder_cls=builder_cls,
    )
    common = create_common_attn_metadata(BatchSpec([64, 64], [query_len, query_len]), 16, torch.device("cpu"))
    common.block_table_tensor = torch.arange(1, 23, dtype=torch.int32).view(2, 11)
    metadata = builder.build(
        0,
        common,
        torch.tensor([9, 9], dtype=torch.int32),
        torch.full((2,), query_len - 1, dtype=torch.int32),
    )
    spec = metadata.spec_decode_metadata
    assert spec.use_upstream_recurrent and spec.use_upstream_conv
    assert spec.actual_seq_lengths is None
    assert spec.spec_causal_conv1d.num_accepted_tokens.tolist() == [9, 9]
    torch.testing.assert_close(metadata.spec_state_indices_tensor, common.block_table_tensor[:, :9])


@pytest.mark.parametrize("adaptive", [False, True])
def test_310p_never_selects_common_strided_recurrent_workaround(monkeypatch, adaptive):
    monkeypatch.setattr(
        ascend_gdn_attn_builder,
        "get_current_hardware_profile",
        Mock(side_effect=AssertionError("310P must not query the common recurrent workaround")),
    )
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=8,
        enable_adaptive_verification=adaptive,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        builder_cls=GDNAttentionMetadataBuilder310,
    )
    common = create_common_attn_metadata(BatchSpec([64, 64], [9, 9]), 16, torch.device("cpu"))
    common.block_table_tensor = torch.arange(1, 23, dtype=torch.int32).view(2, 11)
    spec = builder.build_for_cudagraph_capture(common).spec_decode_metadata
    assert not spec.use_upstream_recurrent
    assert not spec.use_upstream_conv


def test_adaptive_fused_descriptors_keep_prepared_conv_indices():
    """The target branch's fused builder must select the migrated kernels too."""
    builder = SimpleNamespace(
        _USE_COMMON_KERNEL_METADATA=True,
        speculative_config=SimpleNamespace(enable_adaptive_verification=True),
        num_spec=8,
        use_full_cuda_graph=True,
        spec_actual_seq_lengths=torch.empty(3, dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        spec_sequence_masks=torch.tensor([True, False]),
        spec_query_start_loc=torch.tensor([0, 9, 9], dtype=torch.int32),
        spec_state_indices_tensor=torch.arange(18, dtype=torch.int32).reshape(2, 9),
        num_accepted_tokens=torch.tensor([9, 1], dtype=torch.int32),
        num_spec_decodes=1,
        num_prefills=0,
        num_decodes=0,
    )
    prepared_conv_indices = torch.tensor([1, 0], dtype=torch.int32)
    native_lengths = torch.tensor([0, 9, 0], dtype=torch.int32)

    AscendGDNAttentionMetadataBuilder._attach_spec_decode_metadata(
        builder, metadata, native_lengths, prepared_conv_indices
    )

    spec = metadata.spec_decode_metadata
    assert spec.use_upstream_conv and spec.use_upstream_recurrent
    assert spec.actual_seq_lengths is None
    assert spec.spec_causal_conv1d.cache_indices.data_ptr() == prepared_conv_indices.data_ptr()
    assert spec.spec_causal_conv1d.query_start_loc is metadata.spec_query_start_loc
    assert spec.spec_causal_conv1d.num_accepted_tokens.tolist() == [9, 1]


@pytest.mark.parametrize("creates_checkpoint", [False, True])
def test_forward_core_uses_checkpoint_recurrence_for_cold_and_cache_hit_prefills(monkeypatch, creates_checkpoint):
    from vllm_ascend.ops import gdn_checkpoint

    builder = _make_builder(device=torch.device("cpu"), num_heads=1, num_speculative_tokens=0)
    common = create_common_attn_metadata(BatchSpec(seq_lens=[4], query_lens=[4]), 16, torch.device("cpu"))
    metadata = builder.build(0, common, num_actual_reqs=1)
    metadata.checkpoint = SimpleNamespace(checkpoint_offsets=torch.tensor([0])) if creates_checkpoint else None
    qkv = torch.randn(4, 384, dtype=torch.bfloat16)
    q = torch.randn(1, 4, 1, 128, dtype=torch.bfloat16)
    initial = torch.zeros(1, 1, 128, 128)
    expected_output = torch.ones_like(q)
    expected_state = torch.ones_like(initial)
    exporter = Mock()
    layer = SimpleNamespace(
        prefix="gdn",
        _ascend_internal_checkpoint_enabled=True,
        _exports_checkpoints=True,
        head_k_dim=128,
        head_v_dim=128,
        num_k_heads=1,
        num_v_heads=1,
        tp_size=1,
        activation=True,
        conv1d=SimpleNamespace(bias=None),
        kv_cache=(torch.empty(2, 3, 384), torch.zeros(2, 1, 128, 128, dtype=torch.bfloat16)),
        _ascend_checkpoint_workspace=torch.empty_like(initial),
        _checkpoint_exporter=exporter,
        A_log=torch.empty(1),
        dt_bias=torch.empty(1),
    )
    monkeypatch.setattr(ascend_gdn, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"gdn": metadata}))
    monkeypatch.setattr(ascend_gdn, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    for name in (
        "ensure_gdn_checkpoint_workspace",
        "wait_for_kv_layer_from_connector",
        "record_attention_compute_start",
        "maybe_save_kv_layer_to_connector",
    ):
        monkeypatch.setattr(ascend_gdn, name, Mock())
    monkeypatch.setattr(ascend_gdn, "_get_packed_conv_weights", lambda _: torch.empty(4, 384))
    monkeypatch.setattr(ascend_gdn, "causal_conv1d_fn", lambda x, *args, **kwargs: x.clone())
    monkeypatch.setattr(ascend_gdn, "_split_head_first_conv_output", lambda *args: (q, q, q))
    monkeypatch.setattr(
        ascend_gdn.DeviceOperator, "fused_gdn_gating", lambda *args: (torch.zeros(1, 4, 1), torch.ones(1, 4, 1))
    )
    monkeypatch.setattr(ascend_gdn, "gather_ssm_states", lambda *args: initial)
    scatter = Mock()
    monkeypatch.setattr(ascend_gdn, "scatter_ssm_states_", scatter)
    backend = Mock(return_value=(expected_output, expected_state))
    monkeypatch.setattr(gdn_checkpoint, "chunk_gated_delta_rule_checkpoint", backend)
    alternative = Mock(side_effect=AssertionError("checkpoint-enabled prefill selected another recurrence"))
    monkeypatch.setattr(ascend_gdn, "chunk_gated_delta_rule", alternative)
    monkeypatch.setattr(ascend_gdn, "_get_fla_gdn_prefill_op", alternative)
    output = torch.empty(4, 1, 128, dtype=torch.bfloat16)

    AscendGatedDeltaNetAttention._forward_core(layer, qkv, torch.empty(4, 1), torch.empty(4, 1), output)

    backend.assert_called_once()
    assert backend.call_args.args[6] is metadata.prefill_query_start_loc
    assert backend.call_args.kwargs["checkpoint_state_dtype"] == torch.bfloat16
    assert (backend.call_args.args[8] is None) == (not creates_checkpoint)
    assert exporter.export.call_count == int(creates_checkpoint)
    if creates_checkpoint:
        raw = exporter.export.call_args.kwargs["raw_qkv"]
        assert raw.data_ptr() == qkv.data_ptr()
        torch.testing.assert_close(raw, qkv)
    scatter.assert_called_once()
    alternative.assert_not_called()
    torch.testing.assert_close(output, expected_output.squeeze(0))


@pytest.mark.parametrize("speculative", [False, True])
def test_checkpoint_rows_match_non_spec_prefill_inputs(speculative):
    builder = _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=3 if speculative else 0,
    )
    builder._ascend_gdn_checkpoints = True
    builder.checkpoint_builder = SimpleNamespace(build=Mock(return_value=None))
    lengths = [1, 4, 130] if speculative else [1, 130]
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=lengths, query_lens=lengths),
        block_size=16,
        device=torch.device("cpu"),
    )
    if speculative:
        metadata = builder.build(
            0,
            common,
            torch.ones(3, dtype=torch.int32),
            torch.tensor([-1, 3, -1], dtype=torch.int32),
            num_actual_reqs=3,
        )
    else:
        metadata = builder.build(0, common, num_actual_reqs=2)
    # With speculative requests, ordinary one-token rows also execute through
    # the non-spec prefill branch and must retain their planner row.
    expected_rows = [0, 2] if speculative else [1]
    expected_starts = [0, 1, 131] if speculative else [0, 130]
    builder.checkpoint_builder.build.assert_called_once_with(common, expected_rows)
    assert metadata.num_decodes == (0 if speculative else 1)
    assert metadata.num_prefills == len(expected_rows)
    assert metadata.prefill_state_indices.numel() == len(expected_rows)
    assert metadata.prefill_query_start_loc.tolist() == expected_starts
