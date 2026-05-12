from functools import partial
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.attention.utils import (
    BatchSpec,
    create_and_prepopulate_kv_cache,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from tests.ut.conftest import npu_test
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    mock_config.parallel_config = MagicMock()
    mock_config.parallel_config.prefill_context_parallel_size = 1
    mock_config.parallel_config.decode_context_parallel_size = 1
    mock_config.parallel_config.tensor_parallel_size = 1
    mock_config.model_config = MagicMock()
    mock_config.model_config.dtype = torch.float16
    mock_config.speculative_config = None
    mock_config.cache_config = MagicMock()
    mock_config.cache_config.block_size = 128
    mock_config.kv_transfer_config = None
    mock_config.additional_config = None
    mock_config.quant_config = None

    with set_current_vllm_config(mock_config):
        yield mock_config


BATCH_SPECS = {
    "small_decode": BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill": BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small": BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "medium_decode": BatchSpec(
        seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
        query_lens=[1, 1, 1, 1, 1, 1, 1, 1],
    ),
    "medium_prefill": BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]),
    "mixed_medium": BatchSpec(seq_lens=[512, 1024, 2048, 512, 1024, 2048], query_lens=[1, 1, 1, 7, 7, 7]),
    "large_decode": BatchSpec(seq_lens=[2048] * 32, query_lens=[1] * 32),
    "large_prefill": BatchSpec(seq_lens=[4096] * 8, query_lens=[32] * 8),
    "mixed_large": BatchSpec(seq_lens=[1024, 2048, 4096, 1024, 2048, 4096], query_lens=[1, 1, 1, 32, 32, 32]),
    "single_decode": BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill": BatchSpec(seq_lens=[1024], query_lens=[64]),
    "small_encoder_prefill": BatchSpec(seq_lens=[32, 64, 128, 256], query_lens=[32, 64, 128, 256]),
    "medium_encoder_prefill": BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[256, 512, 1024, 2048]),
}


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self.layer_name = "model.layers.0"


def run_attention_backend(
    kv_cache_spec: FullAttentionSpec,
    layer_names: list[str],
    vllm_config,
    device: torch.device,
    common_attn_metadata: AscendCommonAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    dtype: torch.dtype,
    attn_type: AttentionType = AttentionType.DECODER,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    num_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)

    backend = get_attn_backend(head_size, dtype, None, use_mla=False, use_sparse=False, use_mm_prefix=False)
    impl_cls = backend.get_impl_cls()
    builder_cls = backend.get_builder_cls()

    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        attn_type=attn_type.value if hasattr(attn_type, "value") else attn_type,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        kv_sharing_target_layer_name=None,
    )

    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)
    output = impl.forward(mock_layer, query, key, value, kv_cache, attn_metadata, output=output)

    return output


def compute_sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_contexts: list[torch.Tensor],
    v_contexts: list[torch.Tensor],
    batch_spec: BatchSpec,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    device: torch.device,
    attn_type: AttentionType = AttentionType.DECODER,
) -> torch.Tensor:
    """Compute reference attention output using SDPA as golden baseline."""

    all_sdpa_outputs = []
    query_offset = 0
    kv_offset = 0

    for i in range(batch_spec.batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q_i = query[query_offset : query_offset + q_len]
        k_new_i = key[kv_offset : kv_offset + q_len]
        v_new_i = value[kv_offset : kv_offset + q_len]

        k_full_i = torch.cat([k_contexts[i], k_new_i], dim=0)
        v_full_i = torch.cat([v_contexts[i], v_new_i], dim=0)

        q_sdpa_in = q_i.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full_i.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full_i.unsqueeze(0).transpose(1, 2)

        if attn_type == AttentionType.ENCODER_ONLY:
            attn_mask = None
        else:
            attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=device)
            causal_mask = torch.tril(torch.ones(q_len, q_len, device=device))
            attn_mask[:, context_len:] = causal_mask

        sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in,
            k_sdpa_in,
            v_sdpa_in,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=(num_q_heads != num_kv_heads),
            scale=scale,
        )

        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))
        query_offset += q_len
        kv_offset += q_len

    return torch.cat(all_sdpa_outputs, dim=0)


def _test_npu_attention_correctness(
    batch_spec: BatchSpec,
    model: str,
    *,
    attn_type: AttentionType = AttentionType.DECODER,
    block_size: int = 128,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    tensor_parallel_size: int = 1,
):
    """Test attention backend correctness with SDPA as reference."""

    set_random_seed(42)

    hf_config_override = None
    if tensor_parallel_size > 1:
        from vllm.config import ModelConfig

        temp_config = ModelConfig(model=model, max_model_len=1)
        original_num_heads = temp_config.hf_text_config.num_attention_heads
        original_num_kv_heads = getattr(temp_config.hf_text_config, "num_key_value_heads", None)
        hf_config_override = {
            "num_attention_heads": original_num_heads // tensor_parallel_size,
        }
        if original_num_kv_heads is not None:
            hf_config_override["num_key_value_heads"] = max(1, original_num_kv_heads // tensor_parallel_size)

    vllm_config = create_vllm_config(
        model_name=model,
        tensor_parallel_size=1,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
        hf_config_override=hf_config_override,
    )
    device = torch.device("npu")

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    batch_size = batch_spec.batch_size
    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    sliding_window = vllm_config.model_config.get_sliding_window()
    dtype = torch.bfloat16
    scale = 1.0 / (head_size**0.5)

    k_contexts, v_contexts = [], []
    all_q, all_k, all_v = [], [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        q = torch.randn(q_len, num_q_heads, head_size, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)

        all_q.append(q)
        all_k.append(k_full[context_len:])
        all_v.append(v_full[context_len:])

        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query = torch.cat(all_q, dim=0)
    key = torch.cat(all_k, dim=0)
    value = torch.cat(all_v, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)
    if attn_type == AttentionType.ENCODER_ONLY:
        common_attn_metadata.causal = False

    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=8192,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
    )

    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        forward_ctx.num_tokens = query.shape[0]
        forward_ctx.is_draft_model = False
        forward_ctx.is_draft_model_prefill = False
        forward_ctx.capturing = False
        forward_ctx.flash_comm_v1_enabled = False
        forward_ctx.flashcomm_v2_enabled = False

        backend_output = run_attention_backend(
            kv_cache_spec,
            ["placeholder"],
            vllm_config,
            device,
            common_attn_metadata,
            query,
            key,
            value,
            kv_cache,
            dtype,
            sliding_window=sliding_window,
            attn_type=attn_type,
        )

    sdpa_output = compute_sdpa_reference(
        query,
        key,
        value,
        k_contexts,
        v_contexts,
        batch_spec,
        num_q_heads,
        num_kv_heads,
        head_size,
        scale,
        device,
        attn_type=attn_type,
    )

    name = "GQA"
    assert backend_output.shape == sdpa_output.shape, (
        f"[{name}] shape {backend_output.shape} != SDPA shape {sdpa_output.shape}"
    )
    assert backend_output.dtype == sdpa_output.dtype, (
        f"[{name}] dtype {backend_output.dtype} != SDPA dtype {sdpa_output.dtype}"
    )

    assert torch.isfinite(backend_output).all(), f"[{name}] produced non-finite values"

    # Calculate and print differences for debugging
    diff = torch.abs(backend_output - sdpa_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"\n[{name}] Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")
    print(f"[{name}] Backend output range: [{backend_output.min().item():.6f}, {backend_output.max().item():.6f}]")
    print(f"[{name}] SDPA output range: [{sdpa_output.min().item():.6f}, {sdpa_output.max().item():.6f}]")

    def error_msg(msg: str, backend_name: str):
        return f"[{backend_name}] output differs from SDPA baseline. {msg}"

    torch.testing.assert_close(
        backend_output,
        sdpa_output,
        rtol=rtol,
        atol=atol,
        msg=partial(error_msg, backend_name="GQA"),
    )


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_decode",
        "small_prefill",
        "mixed_small",
        "medium_decode",
        "medium_prefill",
        "mixed_medium",
        "large_decode",
        "large_prefill",
        "single_decode",
        "single_prefill",
    ],
)
@pytest.mark.parametrize("model", ["/home/weights/Qwen3-8B"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
@npu_test(num_npus=1, npu_type="a2")
def test_causal_backend_correctness(default_vllm_config, batch_spec_name: str, model: str, tensor_parallel_size: int):
    """Test backend's correctness with causal attention."""
    batch_spec = BATCH_SPECS[batch_spec_name]

    _test_npu_attention_correctness(
        batch_spec,
        model,
        tensor_parallel_size=tensor_parallel_size,
    )


@pytest.mark.parametrize(
    "batch_spec_name",
    [
        "small_encoder_prefill",
        "medium_encoder_prefill",
    ],
)
@pytest.mark.parametrize("model", ["/home/weights/Qwen3-8B"])
@npu_test(num_npus=1, npu_type="a2")
def test_encoder_only_backend_correctness(default_vllm_config, batch_spec_name: str, model: str):
    """Test backend's correctness with encoder-only attention."""
    batch_spec = BATCH_SPECS[batch_spec_name]

    _test_npu_attention_correctness(
        batch_spec,
        model,
        attn_type=AttentionType.ENCODER_ONLY,
    )
