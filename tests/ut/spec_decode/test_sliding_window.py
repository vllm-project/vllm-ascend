"""Unit tests for ``_apply_sliding_window`` in ``AscendSpecDecodeBaseProposer``.

Tests verify the core sliding-window invariants:
  - ``block_table_tensor`` is cropped to the window region [start_block, start_block + needed)
  - ``seq_lens`` is clamped to ``min(L, W)``
  - ``_sliding_window_full_block_table`` preserves the original (uncropped) table
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig, CompilationMode, VllmConfig, set_current_vllm_config

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import vllm_version_is

_CPU_GPU_BUFFER_TARGET = (
    "vllm.v1.spec_decode.eagle.CpuGpuBuffer"
    if vllm_version_is("0.19.1")
    else "vllm.v1.spec_decode.llm_base_proposer.CpuGpuBuffer"
)

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_deps():
    with (
        patch(_CPU_GPU_BUFFER_TARGET),
        patch("vllm.multimodal.registry.MultiModalRegistry.supports_multimodal_inputs", return_value=False),
    ):
        yield
    clear_ascend_config()
    set_current_vllm_config(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_vllm_config(
    *,
    block_size: int = BLOCK_SIZE,
    num_speculative_tokens: int = 2,
    draft_window_size: int | None = None,
):
    """Create a mock ``VllmConfig`` with sliding-window parameters."""
    from vllm.config import SpeculativeConfig

    vllm_config = MagicMock(spec=VllmConfig)
    sc = vllm_config.speculative_config
    sc.method = "eagle3"
    sc.num_speculative_tokens = num_speculative_tokens
    sc.parallel_drafting = False
    sc.draft_tensor_parallel_size = 1
    sc.draft_model_config = MagicMock()
    sc.draft_model_config.uses_xdrope_dim = 0
    sc.draft_model_config.uses_mrope = False
    sc.draft_model_config.get_hidden_size.return_value = 2048
    sc.draft_model_config.get_inputs_embeds_size.return_value = 2048
    sc.disable_padded_drafter_batch = False
    sc.draft_window_size = draft_window_size
    if "speculative_token_tree" in SpeculativeConfig.__dataclass_fields__:
        sc.speculative_token_tree = str([(i + 1) * (0,) for i in range(num_speculative_tokens)])

    vllm_config.cache_config = MagicMock(spec=CacheConfig)
    vllm_config.cache_config.block_size = block_size
    vllm_config.scheduler_config.max_num_batched_tokens = 1024
    vllm_config.scheduler_config.max_num_seqs = 32
    vllm_config.model_config.dtype = torch.float16
    vllm_config.model_config.max_model_len = 2048
    vllm_config.model_config.uses_mrope = False
    vllm_config.model_config.uses_xdrope_dim = 0
    vllm_config.model_config.use_mla = False
    vllm_config.model_config.enforce_eager = True
    vllm_config.model_config.is_deepseek_mla = False
    vllm_config.model_config.hf_text_config = MagicMock(spec=[])
    vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.enable_expert_parallel = False
    vllm_config.pipeline_parallel_size = 1
    vllm_config.kv_transfer_config = None
    vllm_config.compilation_config = MagicMock()
    vllm_config.compilation_config.mode = CompilationMode.NONE
    vllm_config.compilation_config.pass_config = MagicMock()
    vllm_config.compilation_config.pass_config.enable_sp = False
    vllm_config.additional_config = (
        {"draft_window_size": draft_window_size} if draft_window_size is not None else None
    )

    init_ascend_config(vllm_config)
    return vllm_config


def _create_proposer(
    *,
    block_size: int = BLOCK_SIZE,
    num_speculative_tokens: int = 2,
    draft_window_size: int | None = None,
) -> AscendEagleProposer:
    """Create an ``AscendEagleProposer`` with the given sliding-window config."""
    vllm_config = _create_vllm_config(
        block_size=block_size,
        num_speculative_tokens=num_speculative_tokens,
        draft_window_size=draft_window_size,
    )
    runner = MagicMock()
    runner.block_size = block_size
    runner.pin_memory = False
    runner.pcp_size = 1
    runner.dcp_size = 1
    runner.max_num_tokens = 1024
    runner.max_num_reqs = 32

    with set_current_vllm_config(vllm_config):
        proposer = AscendEagleProposer(vllm_config=vllm_config, device=DEVICE, runner=runner)

    # Fallback: if __init__ didn't pick up draft_window_size, set it manually.
    if draft_window_size is not None and getattr(proposer, "draft_window_size", None) is None:
        proposer.draft_window_size = draft_window_size
        proposer.block_size = block_size
        proposer.window_blocks = (draft_window_size + block_size - 1) // block_size
        proposer.max_window_blocks = proposer.window_blocks + 1
        proposer._sliding_window_full_block_table = None
        proposer._sliding_window_start_block_indices = None

    return proposer


def _build_metadata(
    seq_lens: list[int],
    block_size: int,
) -> AscendCommonAttentionMetadata:
    """Build a minimal ``AscendCommonAttentionMetadata`` with deterministic
    block indices (``arange``) for easy assertion."""
    bs = len(seq_lens)
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(bs * max_blocks, dtype=torch.int32).view(bs, max_blocks)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    return AscendCommonAttentionMetadata(
        query_start_loc=torch.zeros(bs + 1, dtype=torch.int32),
        query_start_loc_cpu=torch.zeros(bs + 1, dtype=torch.int32),
        seq_lens=seq_lens_t,
        seq_lens_cpu=seq_lens_t.cpu(),
        num_computed_tokens_cpu=torch.zeros(bs, dtype=torch.int32),
        num_reqs=bs,
        num_actual_tokens=bs,
        max_query_len=1,
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.arange(bs, dtype=torch.int64),
        causal=True,
        positions=torch.zeros(bs, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "W, B, K, L",
    [
        (64,  16, 2,   50),  # L+K=52  < W  — full coverage
        (64,  16, 2,   62),  # L+K=64  == W — exact boundary
        (64,  16, 2,  100),  # L+K=102 > W  — window crops
        (512, 128, 4, 800),  # B=128, K=4
        (32,  16, 8,  100),  # K=8, tiny W
        (1,   16, 2,  100),  # W=1 extreme
    ],
)
def test_block_table_and_seq_lens(W, B, K, L):
    """Verify block-table cropping and seq_lens clamping for various (W, B, K, L)."""
    proposer = _create_proposer(draft_window_size=W, num_speculative_tokens=K, block_size=B)
    meta = _build_metadata([L], B)
    orig_bt = meta.block_table_tensor.clone()
    full_seq_lens = meta.seq_lens.clone()

    proposer._apply_sliding_window(meta, full_seq_lens)

    # --- expected values ---
    final_len = L + K
    start_token = max(0, final_len - W)
    start_block = (start_token // B) * B
    start_block_idx = start_block // B
    tokens_to_cover = final_len - start_block
    needed_blocks = (tokens_to_cover + B - 1) // B
    max_blocks = proposer.max_window_blocks

    # --- seq_lens clamped to min(L, W) ---
    assert meta.seq_lens[0].item() == min(L, W)

    # --- block_table content matches original[start_block_idx : start_block_idx + needed] ---
    actual_needed = min(needed_blocks, max_blocks)
    end_idx = start_block_idx + actual_needed
    if end_idx <= orig_bt.shape[1]:
        assert torch.equal(meta.block_table_tensor[0, :actual_needed], orig_bt[0, start_block_idx:end_idx])
    # trailing positions are zero
    assert torch.all(meta.block_table_tensor[0, actual_needed:] == 0)

    # --- output shape is fixed at max_window_blocks (graph compat) ---
    assert meta.block_table_tensor.shape[1] == max_blocks

    # --- original full block table is preserved for slot-mapping ---
    assert torch.equal(proposer._sliding_window_full_block_table, orig_bt)


def test_multi_request_batch():
    """Each request in a batch gets an independently cropped block table."""
    W, B, K = 128, BLOCK_SIZE, 2
    proposer = _create_proposer(draft_window_size=W, num_speculative_tokens=K, block_size=B)
    # L = [50, 200, 130]  →  L+K = [52, 202, 132]
    meta = _build_metadata([50, 200, 130], B)
    orig_bt = meta.block_table_tensor.clone()

    proposer._apply_sliding_window(meta, meta.seq_lens.clone())

    # seq_lens clamped independently
    assert torch.equal(meta.seq_lens, torch.tensor([50, 128, 128], dtype=torch.int32))

    # block table content for each request
    # req0: start=0, needed=ceil(52/16)=4  → blocks [0:4]
    assert torch.equal(meta.block_table_tensor[0, :4], orig_bt[0, :4])
    # req1: start_token=74, start_block=64, idx=4, needed=ceil(138/16)=9 → blocks [4:13]
    assert torch.equal(meta.block_table_tensor[1, :9], orig_bt[1, 4:13])

    # full block table preserved
    assert torch.equal(proposer._sliding_window_full_block_table, orig_bt)


def test_block_table_underflow_pads_zeros():
    """When the full block table has fewer columns than the window needs,
    valid blocks are copied and missing positions are filled with zeros."""
    W, B, K = 256, BLOCK_SIZE, 2
    proposer = _create_proposer(draft_window_size=W, num_speculative_tokens=K, block_size=B)
    # L=300 → needs 17 blocks, but table only has 10
    meta = _build_metadata([300], B)
    meta.block_table_tensor = meta.block_table_tensor[:, :10]

    proposer._apply_sliding_window(meta, meta.seq_lens.clone())

    # blocks 2..9 present, rest zero
    assert torch.equal(meta.block_table_tensor[0, :8], torch.arange(2, 10, dtype=torch.int32))
    assert torch.all(meta.block_table_tensor[0, 8:] == 0)
