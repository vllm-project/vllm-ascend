from functools import partial

import pytest
import torch
from vllm.forward_context import set_forward_context
from vllm.utils.torch_utils import (
    set_random_seed,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import get_attn_backend  # type: ignore
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.attention.utils import (
    BatchSpec,
    create_and_prepopulate_kv_cache,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

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
    # encoder-only
    "small_encoder_prefill": BatchSpec(seq_lens=[32, 64, 128, 256], query_lens=[32, 64, 128, 256]),
    "medium_encoder_prefill": BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[256, 512, 1024, 2048]),
}


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


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
    dtype: torch.bfloat16,
    attn_type: AttentionType = AttentionType.DECODER,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    backend = get_attn_backend(0, dtype, None, 128, use_mla=False, use_sparse=False, use_mm_prefix=False)
    impl_cls = backend.get_impl_cls()
    builder_cls = backend.get_builder_cls()
    # Build metadata
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Instantiate implementation
    num_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        attn_type=attn_type,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        kv_sharing_target_layer_name=None,
    )

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)
    output = impl.forward(mock_layer, query, key, value, kv_cache, attn_metadata, output=output)

    return output


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
    """
    Test that all backends produce similar outputs to a reference implementation
    using torch.nn.functional.scaled_dot_product_attention.

    This test works by:
    1. Generating a batch of sequences with specified context and query lengths.
    2. Computing a ground-truth attention output using torch.sdpa on
       contiguous Q, K, and V tensors.
    3. Simulating vLLM's paged KV cache: It takes the context portion of the
       K/V tensors and manually places them into a paged buffer according to
       the test's (randomly generated) block table.
    4. Running each vLLM attention backend with the new queries and the
       simulated paged KV cache.
    5. Comparing the vLLM backend's output to the ground-truth SDPA output.

    Note: When tensor_parallel_size > 1, we simulate the head partitioning
    by overriding the model config to use fewer heads, without requiring
    multiple GPUs. This tests that backends work correctly with different
    head counts.
    """
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
        tensor_parallel_size=1,  # Always use TP=1 to avoid multi-GPU requirements
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
        hf_config_override=hf_config_override,
    )
    device = torch.device("npu")

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    # 1. Setup
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    num_q_heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    sliding_window = vllm_config.model_config.get_sliding_window()
    dtype = torch.bfloat16
    block_size = vllm_config.cache_config.block_size
    scale = 1.0 / (head_size**0.5)

    # 2. Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    all_sdpa_outputs = []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence to be used in SDPA
        q = torch.randn(q_len, num_q_heads, head_size, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_size, dtype=dtype, device=device)
        # SDPA expects (N, H, L, D), so unsqueeze batch and permute
        q_sdpa_in = q.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0, (
                f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        # Create causal mask: query token i attends to positions 0 to
        #  (context_len + i)
        attn_mask = torch.ones(q_len, s_len, dtype=torch.bool, device=device)
        # Apply causal mask only to the query portion (context_len onwards)
        causal_mask = torch.tril(torch.ones(q_len, q_len, device=device))
        attn_mask[:, context_len:] = causal_mask
        sdpa_out_i = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa_in, k_sdpa_in, v_sdpa_in, attn_mask=attn_mask, is_causal=False, enable_gqa=True, scale=scale
        )

        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))

        # Inputs for vLLM backends are just the new tokens
        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)
    if attn_type == AttentionType.ENCODER_ONLY:
        # For encoder-only, all tokens are prefill tokens
        common_attn_metadata.causal = False

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
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
    # 4. Run vLLM backends and compare
    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        backend_output = run_attention_backend(
            kv_cache_spec,
            ["placeholder"],
            vllm_config,
            device,
            common_attn_metadata,
            query_vllm,
            key_vllm,
            value_vllm,
            kv_cache,
            dtype,
            sliding_window=sliding_window,
            attn_type=attn_type,
        )
    name = "GQA"
    # Check shape and dtype consistency
    assert backend_output.shape == sdpa_output.shape, (
        f"[{name}] shape {backend_output.shape} != SDPA shape {sdpa_output.shape}"
    )
    assert backend_output.dtype == sdpa_output.dtype, (
        f"[{name}] dtype {backend_output.dtype} != SDPA dtype {sdpa_output.dtype}"
    )

    assert torch.isfinite(backend_output).all(), f"[{name}] produced non-finite values"

    # Check numerical similarity
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
@pytest.mark.parametrize("model", ["Qwen/Qwen3-8B"])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_causal_backend_correctness(default_vllm_config, batch_spec_name: str, model: str, tensor_parallel_size: int):
    """Test backend's correctness with causal attention."""
    batch_spec = BATCH_SPECS[batch_spec_name]

    _test_npu_attention_correctness(
        batch_spec,
        model,
        tensor_parallel_size=tensor_parallel_size,
    )
