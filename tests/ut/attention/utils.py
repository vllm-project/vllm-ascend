from functools import wraps
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


from vllm.distributed.parallel_state import all_gather_fake
import torch
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.model import ModelDType
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

def patch_distributed_groups(dcp_size=1,
                             dcp_rank=0,
                             pcp_size=1,
                             pcp_rank=0,
                             needs_mocks=True):
    """
    Decorator to patch common distributed group mocks with configuration
    
    Args:
        dcp_size: DCP world size (default: 1)
        dcp_rank: DCP rank (default: 0)
        pcp_size: PCP world size (default: 1)
        pcp_rank: PCP rank (default: 0)
        needs_mocks: Whether to pass mock objects as the first arguments 
             after 'self' to the decorated function. 
             If True, the decorated function receives: 
                 func(self, mock_all_to_all_single, mock_dcp, mock_pcp, *args, **kwargs)
             If False, mocks are not passed and function receives: 
                 func(self, *args, **kwargs)
             (default: True)
    """

    def decorator(func):

        @wraps(func)
        @patch('torch.distributed.all_to_all_single')
        @patch('vllm.distributed.parallel_state._PCP')
        @patch('vllm.distributed.parallel_state._DCP')
        def wrapper(self, mock_dcp, mock_pcp, mock_all_to_all_single, *args,
                    **kwargs):
            mock_dcp.rank_in_group = dcp_rank
            mock_dcp.world_size = dcp_size
            mock_dcp.device_group = MagicMock()

            mock_dcp.all_gather = MagicMock()
            mock_dcp.all_gather.side_effect = lambda input_, dim: all_gather_fake(
                input_, dim, mock_dcp.world_size, "mock_dcp_group")

            mock_pcp.rank_in_group = pcp_rank
            mock_pcp.world_size = pcp_size
            mock_pcp.device_group = MagicMock()

            mock_pcp.all_gather = MagicMock()
            mock_pcp.all_gather.side_effect = lambda input_, dim: all_gather_fake(
                input_, dim, mock_pcp.world_size, "mock_pcp_group")

            mock_all_to_all_single.side_effect = lambda output, input, *a, **kw: output.copy_(
                input)

            if needs_mocks:
                return func(self, mock_all_to_all_single, mock_dcp, mock_pcp,
                            *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


@dataclass
class BatchSpec:
    """Specification for a batch configuration (workload shape only)."""

    seq_lens: list[int]
    query_lens: list[int]

    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)

    def __post_init__(self):
        assert len(self.seq_lens) == len(self.query_lens)

    def compute_num_tokens(self):
        return sum(self.query_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
    max_block_idx: int = 1000,
    arange_block_indices: bool = True,
) -> AscendCommonAttentionMetadata:
    """Create CommonAttentionMetadata from a BatchSpec and ModelParams."""
    # Create query start locations
    query_start_loc = torch.zeros(batch_spec.batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(batch_spec.query_lens, dtype=torch.int32, device=device).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.compute_num_tokens()
    # Create sequence lengths
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())

    # Create computed tokens (context length for each sequence)
    context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i] for i in range(batch_spec.batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)

    # Create block table and slot mapping
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    num_blocks = batch_spec.batch_size * max_blocks
    block_table_tensor = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        batch_spec.batch_size, max_blocks
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device).view(num_tokens)

    # Calculate max query length
    max_query_len = max(batch_spec.query_lens)

    # Create positions tensor
    positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
        positions=positions,
    )


def create_vllm_config(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    tensor_parallel_size: int = 1,
    max_model_len: int = 1024,
    dtype: ModelDType | torch.dtype = "auto",
    num_gpu_blocks: int = 1000,
    block_size: int = 16,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    add_mock_model_methods: bool = True,
    hf_config_override: dict | None = None,
) -> VllmConfig:
    """Create a VllmConfig for testing with reasonable defaults."""

    model_config = ModelConfig(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=False,
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
    )

    cache_config = CacheConfig(
        block_size=block_size,
        cache_dtype="auto",
    )
    # Set cache blocks for testing
    #   (these may be set during initialization normally)
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0

    parallel_config = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
    )

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    device_config = DeviceConfig()
    load_config = LoadConfig()
    compilation_config = CompilationConfig()

    if add_mock_model_methods:
        # Add mock methods to satisfy backends that need them
        # This is a workaround because tests don't build full, real models,
        # but some backends expect to query the model for layer-specific
        # parameters
        import types

        model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
        model_config.get_sliding_window_for_layer = types.MethodType(lambda self, i: None, model_config)
        model_config.get_logits_soft_cap_for_layer = types.MethodType(lambda self, i: 0.0, model_config)
        model_config.get_sm_scale_for_layer = types.MethodType(
            lambda self, i: 1.0 / model_config.get_head_size() ** 0.5, model_config
        )

    if hf_config_override:
        model_config.hf_config.update(hf_config_override)

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        compilation_config=compilation_config,
    )


def create_standard_kv_cache_spec(vllm_config: VllmConfig) -> FullAttentionSpec:
    """Create a FullAttentionSpec from ModelParams only."""
    return FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
        sliding_window=vllm_config.model_config.get_sliding_window(),
    )


def create_and_prepopulate_kv_cache(
    k_contexts: list[torch.Tensor],
    v_contexts: list[torch.Tensor],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_blocks: int,
    common_attn_metadata: AscendCommonAttentionMetadata,
    randomize_blocks: bool = True,
) -> torch.Tensor:
    """Create and prepopulate a KV cache with context data.

    Args:
        k_contexts: List of key context tensors for each sequence
        v_contexts: List of value context tensors for each sequence
        seq_lens: List of sequence lengths
        block_size: Size of each block
        num_kv_heads: Number of KV heads
        head_size: Size of each head
        dtype: Data type for the cache
        device: Device to create the cache on
        num_blocks: Total number of blocks in the cache
        block_table: Block table tensor to populate
        randomize_blocks: Whether to randomly permute blocks
                          or use sequential order

    Returns:
        Tuple of (kv_cache, updated_block_table)
    """
    batch_size = len(k_contexts)
    seq_lens = common_attn_metadata.seq_lens.cpu()
    query_lens = common_attn_metadata.query_start_loc_cpu[1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = seq_lens - query_lens
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping
    # Create KV cache
    kv_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    kv_cache_flat = kv_cache.view(2, -1, num_kv_heads, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=0
    start_block_idx = 0
    for i in range(batch_size):
        k_context, v_context = k_contexts[i], v_contexts[i]
        start = start_block_idx * block_size
        end = start + k_context.shape[0]
        kv_cache_flat[0, start:end, ...] = k_context
        kv_cache_flat[1, start:end, ...] = v_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks
    if randomize_blocks:
        # Random permutation starting from block 0
        perm = torch.randperm(blocks_end)
    else:
        # Sequential order starting from block 0
        perm = torch.arange(blocks_end)
    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    inv_perm = torch.argsort(perm)
    kv_cache[:, :blocks_end, ...] = kv_cache[:, perm, ...]
    # Construct the right block table
    # Start from block_id=0
    start_block_idx = 0
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        start_block_idx += num_blocks_for_seq

        # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[i, block_indices] * block_size + token_inter_block_offsets.to(device).to(
            torch.int32
        )
    return kv_cache